#!/usr/bin/env python3
"""Run the resumable Cairngorms airborne-LiDAR/TESSERA evaluation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import yaml
from geotessera.registry import download_file_to_temp
from pyproj import Transformer
from rasterio.transform import rowcol
from rasterio.windows import Window
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_tessera_alignment as tessera  # noqa: E402


CONFIG_PATH = ROOT / "configs/scotland_lidar.yaml"
PROTOCOL_PATH = ROOT / "metadata/project_config_phase14_scotland_lidar_protocol_freeze.yaml"
INVENTORY_PATH = ROOT / "metadata/phase14_scotland_lidar_inventory.json"
SAMPLE_PATH = ROOT / "data/processed/phase14_scotland_lidar_sample.parquet"
SAMPLE_QC_PATH = ROOT / "metadata/phase14_scotland_lidar_sample_qc.json"
TESSERA_CHUNK_DIR = ROOT / "data/interim/phase14_scotland_tessera_chunks"
TESSERA_STREAM_DIR = ROOT / "data/interim/phase14_scotland_tessera_stream"
TESSERA_PATH = ROOT / "data/processed/phase14_scotland_tessera.parquet"
TESSERA_MANIFEST_PATH = ROOT / "metadata/phase14_scotland_tessera_manifest.json"
METRICS_PATH = ROOT / "outputs/tables/phase14_scotland_spatial_metrics.csv"
PRIMARY_PREDICTIONS_PATH = (
    ROOT / "data/processed/phase14_scotland_primary_oof_predictions.parquet"
)
CORRELATION_PATH = ROOT / "outputs/tables/phase14_scotland_lidar_spearman.csv"
HEIGHT_ASSOCIATION_PATH = ROOT / "outputs/tables/phase14_scotland_height_associations.csv"
BOOTSTRAP_PATH = ROOT / "outputs/tables/phase14_scotland_primary_bootstrap.csv"
MORAN_PATH = ROOT / "outputs/tables/phase14_scotland_primary_moran.csv"
FIGURE_PATH = ROOT / "outputs/figures/phase14_scotland_lidar_evaluation.png"
REPORT_PATH = ROOT / "outputs/reports/phase14_scotland_lidar_results.md"
RESULT_FREEZE_PATH = ROOT / "metadata/phase14_scotland_lidar_result_freeze.json"

EMBEDDING_FEATURES = [f"tessera_{index:03d}" for index in range(128)]
METRIC_COLUMNS = ["rmse", "mae", "r2", "pearson_r", "spearman_r", "bias"]
STAGES = ["inventory", "sample", "tessera", "evaluate", "report"]


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "phase14_scotland_lidar"
    ]


def source_path(config: dict[str, Any]) -> Path:
    return ROOT / config["source"]["path"]


def band_lookup(dataset: rasterio.DatasetReader) -> dict[str, int]:
    descriptions = list(dataset.descriptions)
    if any(description is None for description in descriptions):
        raise RuntimeError("The Scotland LiDAR raster has an unnamed band")
    if len(set(descriptions)) != len(descriptions):
        raise RuntimeError("The Scotland LiDAR raster has duplicate band names")
    return {str(name): index for index, name in enumerate(descriptions, start=1)}


def quality_mask(core: dict[str, np.ndarray], rules: dict[str, Any]) -> np.ndarray:
    names = [
        "lidar_maxH",
        "lidar_meanH",
        "lidar_p_95",
        "lidar_p_999",
        "lidar_Cov",
        "lidar_canopy_shannon",
    ]
    valid = np.logical_and.reduce([np.isfinite(core[name]) for name in names])
    valid &= core["lidar_meanH"] >= 1.3
    valid &= core["lidar_meanH"] <= core["lidar_maxH"]
    valid &= core["lidar_p_95"] >= float(rules["minimum_p95_height_m"])
    valid &= core["lidar_p_95"] <= core["lidar_p_999"]
    valid &= core["lidar_p_999"] <= float(rules["maximum_p999_height_m"])
    valid &= core["lidar_maxH"] <= float(rules["maximum_return_height_m"])
    valid &= core["lidar_Cov"] >= float(rules["minimum_canopy_cover"])
    valid &= core["lidar_Cov"] <= float(rules["maximum_canopy_cover"])
    valid &= core["lidar_canopy_shannon"] >= 0
    valid &= core["lidar_canopy_shannon"] <= float(
        rules["maximum_canopy_shannon"]
    )
    return valid


def run_inventory(config: dict[str, Any]) -> None:
    path = source_path(config)
    if not path.exists():
        raise RuntimeError(f"Missing Scotland LiDAR raster: {path}")
    if INVENTORY_PATH.exists():
        existing = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
        if (
            existing["source_sha256"] == sha256(path)
            and existing["config_sha256"] == sha256(CONFIG_PATH)
            and existing["protocol_sha256"] == sha256(PROTOCOL_PATH)
        ):
            print("inventory: validated existing checkpoint", flush=True)
            return
        raise RuntimeError(
            "The LiDAR source or frozen design changed after inventory was recorded"
        )

    expected = config["source"]
    with rasterio.open(path) as dataset:
        if dataset.count != int(expected["expected_band_count"]):
            raise RuntimeError("Unexpected Scotland LiDAR band count")
        if tuple(dataset.res) != (
            float(expected["expected_resolution_m"]),
            float(expected["expected_resolution_m"]),
        ):
            raise RuntimeError("Unexpected Scotland LiDAR resolution")
        if dataset.crs is not None:
            raise RuntimeError(
                "The source now contains a CRS; review the recorded override before running"
            )
        observed_bounds = np.asarray(dataset.bounds, dtype=np.float64)
        expected_bounds = np.asarray(expected["expected_bounds_bng"], dtype=np.float64)
        if not np.allclose(observed_bounds, expected_bounds, atol=0.01):
            raise RuntimeError("LiDAR bounds do not match the frozen Cairngorms source")
        lookup = band_lookup(dataset)
        profile = {
            "driver": dataset.driver,
            "width": dataset.width,
            "height": dataset.height,
            "band_count": dataset.count,
            "bands": list(lookup),
            "resolution_m": list(dataset.res),
            "bounds_source_coordinates": list(dataset.bounds),
            "source_crs_tag": None,
            "source_crs_override": expected["source_crs_override"],
            "compression": dataset.profile.get("compress"),
            "nodata": [
                None if value is None else str(value) for value in dataset.nodatavals
            ],
        }
    transformer = Transformer.from_crs(
        expected["source_crs_override"], "EPSG:4326", always_xy=True
    )
    west, south = transformer.transform(expected_bounds[0], expected_bounds[1])
    east, north = transformer.transform(expected_bounds[2], expected_bounds[3])
    inventory = {
        "created_utc": utc_now(),
        "source_path": str(path.relative_to(ROOT)),
        "source_size_bytes": path.stat().st_size,
        "source_sha256": sha256(path),
        "survey_year": expected["survey_year"],
        "profile": profile,
        "bounds_wgs84_from_explicit_override": [west, south, east, north],
        "config_sha256": sha256(CONFIG_PATH),
        "protocol_sha256": sha256(PROTOCOL_PATH),
    }
    write_json(INVENTORY_PATH, inventory)
    print(
        f"inventory: {dataset.width:,} x {dataset.height:,}, "
        f"{dataset.count} bands, WGS84 bounds {west:.4f}, {south:.4f}, "
        f"{east:.4f}, {north:.4f}",
        flush=True,
    )


def assign_spatial_regions(
    frame: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    settings = config["spatial_evaluation"]
    block_size = int(settings["block_size_m"])
    frame = frame.copy()
    frame["block_x"] = np.floor(frame["bng_x"] / block_size).astype(np.int32)
    frame["block_y"] = np.floor(frame["bng_y"] / block_size).astype(np.int32)
    frame["spatial_block"] = (
        frame["block_x"].astype(str) + "_" + frame["block_y"].astype(str)
    )
    blocks = (
        frame.groupby(["block_x", "block_y"], as_index=False)
        .size()
        .rename(columns={"size": "sample_rows"})
    )
    block_centres = np.column_stack(
        [
            (blocks["block_x"].to_numpy() + 0.5) * block_size,
            (blocks["block_y"].to_numpy() + 0.5) * block_size,
        ]
    )
    centred = block_centres - np.average(
        block_centres,
        axis=0,
        weights=blocks["sample_rows"].to_numpy(dtype=np.float64),
    )
    covariance = np.cov(
        centred.T,
        aweights=blocks["sample_rows"].to_numpy(dtype=np.float64),
    )
    _, eigenvectors = np.linalg.eigh(covariance)
    principal_axis = eigenvectors[:, -1]
    if principal_axis[0] < 0:
        principal_axis *= -1
    score = centred @ principal_axis
    weighted_score = np.repeat(
        score, blocks["sample_rows"].to_numpy(dtype=np.int64)
    )
    region_count = int(settings["region_count"])
    boundaries = np.quantile(
        weighted_score, np.arange(1, region_count) / region_count
    )
    blocks["spatial_region"] = np.searchsorted(
        boundaries, score, side="right"
    ).astype(np.int16)
    return frame.merge(
        blocks[["block_x", "block_y", "spatial_region"]],
        on=["block_x", "block_y"],
        how="left",
        validate="many_to_one",
    )


def run_sample(config: dict[str, Any]) -> None:
    if SAMPLE_PATH.exists() and SAMPLE_QC_PATH.exists():
        print("sample: resumed existing checkpoint", flush=True)
        return
    path = source_path(config)
    sample_config = config["sampling"]
    chunk_rows = int(sample_config["raster_chunk_rows"])
    with rasterio.open(path) as dataset:
        lookup = band_lookup(dataset)
        core_names = [
            "lidar_maxH",
            "lidar_meanH",
            "lidar_p_95",
            "lidar_p_999",
            "lidar_Cov",
            "lidar_canopy_shannon",
        ]
        missing = sorted(set(core_names) - set(lookup))
        if missing:
            raise RuntimeError(f"Missing LiDAR quality bands: {missing}")
        eligible_indices: list[np.ndarray] = []
        scanned = 0
        for row_start in range(0, dataset.height, chunk_rows):
            height = min(chunk_rows, dataset.height - row_start)
            window = Window(0, row_start, dataset.width, height)
            values = dataset.read([lookup[name] for name in core_names], window=window)
            core = {name: values[index] for index, name in enumerate(core_names)}
            mask = quality_mask(core, sample_config["forest_mask"])
            rows, columns = np.nonzero(mask)
            flat = (rows + row_start) * dataset.width + columns
            eligible_indices.append(flat.astype(np.int64))
            scanned += height * dataset.width
            print(
                f"sample scan: {min(row_start + height, dataset.height):,}/"
                f"{dataset.height:,} rows; {sum(len(x) for x in eligible_indices):,} "
                "eligible forest cells",
                flush=True,
            )
        eligible = np.concatenate(eligible_indices)
        if len(eligible) < int(config["spatial_evaluation"]["minimum_training_rows_per_target"]):
            raise RuntimeError("Too few quality-controlled forest cells")
        eligible_count = len(eligible)
        maximum = int(sample_config["maximum_rows"])
        rng = np.random.default_rng(int(sample_config["seed"]))
        selected = (
            np.sort(rng.choice(eligible, size=maximum, replace=False))
            if len(eligible) > maximum
            else np.sort(eligible)
        )
        del eligible_indices, eligible
        gc.collect()

        frames: list[pd.DataFrame] = []
        descriptions = [str(name) for name in dataset.descriptions]
        for row_start in range(0, dataset.height, chunk_rows):
            height = min(chunk_rows, dataset.height - row_start)
            lower = row_start * dataset.width
            upper = (row_start + height) * dataset.width
            left = int(np.searchsorted(selected, lower, side="left"))
            right = int(np.searchsorted(selected, upper, side="left"))
            if right <= left:
                continue
            scoped = selected[left:right]
            local_rows = scoped // dataset.width - row_start
            columns = scoped % dataset.width
            values = dataset.read(
                window=Window(0, row_start, dataset.width, height)
            )[:, local_rows, columns].T
            cell_rows = scoped // dataset.width
            xs, ys = rasterio.transform.xy(
                dataset.transform,
                cell_rows.astype(int),
                columns.astype(int),
                offset="center",
            )
            chunk = pd.DataFrame(values, columns=descriptions)
            chunk.insert(0, "raster_flat_index", scoped)
            chunk["raster_row"] = cell_rows.astype(np.int32)
            chunk["raster_col"] = columns.astype(np.int32)
            chunk["bng_x"] = np.asarray(xs, dtype=np.float64)
            chunk["bng_y"] = np.asarray(ys, dtype=np.float64)
            frames.append(chunk)
        frame = pd.concat(frames, ignore_index=True)

    transformer = Transformer.from_crs(
        config["source"]["source_crs_override"], "EPSG:4326", always_xy=True
    )
    longitude, latitude = transformer.transform(
        frame["bng_x"].to_numpy(), frame["bng_y"].to_numpy()
    )
    frame["longitude"] = longitude
    frame["latitude"] = latitude
    frame["tessera_tile"] = [
        tessera.tile_name(tessera.tile_from_world(float(lon), float(lat)))
        for lon, lat in zip(longitude, latitude, strict=True)
    ]
    frame = assign_spatial_regions(frame, config)
    frame.insert(0, "row_id", np.arange(len(frame), dtype=np.int64))
    float_columns = [
        column
        for column in descriptions
        if frame[column].dtype.kind in {"f", "i", "u"}
    ]
    frame[float_columns] = frame[float_columns].astype(np.float32)
    write_parquet(frame, SAMPLE_PATH)
    qc = {
        "created_utc": utc_now(),
        "source_sha256": sha256(path),
        "eligible_forest_cells_before_sampling": eligible_count,
        "sample_rows": len(frame),
        "sample_seed": sample_config["seed"],
        "sample_is_complete_population": eligible_count <= maximum,
        "spatial_blocks": int(frame["spatial_block"].nunique()),
        "spatial_regions": {
            str(region): int(count)
            for region, count in frame.groupby("spatial_region").size().items()
        },
        "tessera_tiles": sorted(frame["tessera_tile"].unique().tolist()),
        "sample_sha256": sha256(SAMPLE_PATH),
    }
    write_json(SAMPLE_QC_PATH, qc)
    print(
        f"sample: wrote {len(frame):,} cells across "
        f"{frame['spatial_block'].nunique()} 1 km blocks and "
        f"{frame['tessera_tile'].nunique()} TESSERA tiles",
        flush=True,
    )


def download_checked(
    url: str,
    destination: Path,
    expected_size: int,
    attempts: int,
    delay_seconds: int,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == expected_size:
        return
    if destination.exists():
        destination.unlink()
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            download_file_to_temp(url, cache_path=destination)
            if destination.stat().st_size != expected_size:
                raise RuntimeError(
                    f"wrong size for {destination.name}: "
                    f"{destination.stat().st_size} != {expected_size}"
                )
            return
        except Exception as error:  # the public object store can fail in several ways
            last_error = error
            destination.unlink(missing_ok=True)
            if attempt < attempts:
                print(
                    f"download retry {attempt + 1}/{attempts} for "
                    f"{destination.name} in {delay_seconds}s: {error}",
                    flush=True,
                )
                time.sleep(delay_seconds)
    raise RuntimeError(f"Failed to download {url}") from last_error


def parse_tile_name(name: str) -> tuple[float, float]:
    prefix, lon, lat = name.split("_")
    if prefix != "grid":
        raise ValueError(f"Unexpected TESSERA tile name: {name}")
    return float(lon), float(lat)


def run_tessera(config: dict[str, Any]) -> None:
    if TESSERA_PATH.exists() and TESSERA_MANIFEST_PATH.exists():
        print("tessera: resumed complete alignment checkpoint", flush=True)
        return
    sample = pd.read_parquet(
        SAMPLE_PATH,
        columns=[
            "row_id",
            "bng_x",
            "bng_y",
            "longitude",
            "latitude",
            "tessera_tile",
        ],
    )
    setting = config["tessera"]
    version = str(setting["dataset_version"])
    variant = str(setting["dataset_variant"])
    year = int(setting["embedding_year"])
    tile_names = sorted(sample["tessera_tile"].unique())
    tiles = [parse_tile_name(name) for name in tile_names]
    manifest_path, landmask_path = tessera.ensure_registry(version)
    rows = tessera.lookup_embedding_rows(manifest_path, tiles, year, version, variant)
    masks = tessera.lookup_landmask_rows(landmask_path, tiles)
    tessera.assert_complete_inventory(rows, masks, tiles)
    rows_by_tile = {
        tessera.tile_name((float(row.lon), float(row.lat))): row
        for row in rows.itertuples(index=False)
    }
    masks_by_tile = {
        tessera.tile_name((float(row.lon), float(row.lat))): row
        for row in masks.itertuples(index=False)
    }
    TESSERA_CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    attempts = int(setting["download_attempts"])
    delay = int(setting["retry_delay_seconds"])
    records: list[dict[str, Any]] = []

    for position, name in enumerate(tile_names, start=1):
        tile_sample = sample[sample["tessera_tile"].eq(name)].copy()
        chunk_path = TESSERA_CHUNK_DIR / f"{name}.parquet"
        if chunk_path.exists():
            chunk = pd.read_parquet(chunk_path, columns=["row_id", "tessera_valid"])
            if len(chunk) != len(tile_sample) or set(chunk["row_id"]) != set(
                tile_sample["row_id"]
            ):
                raise RuntimeError(f"Stale TESSERA chunk: {name}")
            records.append(
                {
                    "tile": name,
                    "sample_rows": len(chunk),
                    "valid_rows": int(chunk["tessera_valid"].sum()),
                    "chunk_sha256": sha256(chunk_path),
                    "resumed": True,
                }
            )
            print(
                f"tessera [{position}/{len(tile_names)}]: resumed {name}",
                flush=True,
            )
            continue

        row = rows_by_tile[name]
        mask_row = masks_by_tile[name]
        tile = parse_tile_name(name)
        paths = tessera.local_tile_paths(TESSERA_STREAM_DIR, year, tile)
        sources = {
            "embedding": str(row.grid_path),
            "scales": str(row.scales_path),
            "landmask": str(mask_row.key),
        }
        sizes = {
            "embedding": int(row.grid_size),
            "scales": int(row.scales_size),
            "landmask": int(mask_row.file_size),
        }
        required = sum(sizes.values()) + 512 * 1024 * 1024
        if shutil.disk_usage(ROOT).free < required:
            raise RuntimeError(
                f"Insufficient free space for one streamed TESSERA tile: "
                f"{shutil.disk_usage(ROOT).free / 1e9:.2f} GB free, "
                f"{required / 1e9:.2f} GB required"
            )
        print(
            f"tessera [{position}/{len(tile_names)}]: acquire {name} "
            f"for {len(tile_sample):,} cells",
            flush=True,
        )
        try:
            for kind in ["embedding", "scales", "landmask"]:
                download_checked(
                    tessera.s3_url(sources[kind]),
                    paths[kind],
                    sizes[kind],
                    attempts,
                    delay,
                )
            quantized = np.load(paths["embedding"], mmap_mode="r")
            scales = np.load(paths["scales"], mmap_mode="r")
            with rasterio.open(paths["landmask"]) as dataset:
                landmask = dataset.read(1)
                destination_crs = dataset.crs
                transform = dataset.transform
            transformer = Transformer.from_crs(
                config["source"]["source_crs_override"],
                destination_crs,
                always_xy=True,
            )
            xs, ys = transformer.transform(
                tile_sample["bng_x"].to_numpy(),
                tile_sample["bng_y"].to_numpy(),
            )
            pixel_rows, pixel_columns = rowcol(transform, xs, ys)
            pixel_rows = np.asarray(pixel_rows, dtype=np.int64)
            pixel_columns = np.asarray(pixel_columns, dtype=np.int64)
            inside = (
                (pixel_rows >= 0)
                & (pixel_columns >= 0)
                & (pixel_rows < quantized.shape[0])
                & (pixel_columns < quantized.shape[1])
            )
            valid = np.zeros(len(tile_sample), dtype=bool)
            embeddings = np.full(
                (len(tile_sample), 128), np.nan, dtype=np.float32
            )
            indices = np.flatnonzero(inside)
            if len(indices):
                local_scales = scales[
                    pixel_rows[indices], pixel_columns[indices]
                ].astype(np.float32)
                local_land = landmask[
                    pixel_rows[indices], pixel_columns[indices]
                ]
                usable = (
                    np.isfinite(local_scales)
                    & (local_scales > 0)
                    & (local_land > 0)
                )
                valid_indices = indices[usable]
                embeddings[valid_indices] = (
                    quantized[
                        pixel_rows[valid_indices], pixel_columns[valid_indices]
                    ].astype(np.float32)
                    * scales[
                        pixel_rows[valid_indices], pixel_columns[valid_indices]
                    ].astype(np.float32)[:, None]
                )
                valid[valid_indices] = np.isfinite(
                    embeddings[valid_indices]
                ).all(axis=1)
            output = pd.DataFrame(
                embeddings, columns=EMBEDDING_FEATURES, dtype=np.float32
            )
            output.insert(0, "tessera_valid", valid)
            output.insert(
                0, "row_id", tile_sample["row_id"].to_numpy(dtype=np.int64)
            )
            write_parquet(output, chunk_path)
            records.append(
                {
                    "tile": name,
                    "sample_rows": len(output),
                    "valid_rows": int(valid.sum()),
                    "embedding_source": sources["embedding"],
                    "scales_source": sources["scales"],
                    "landmask_source": sources["landmask"],
                    "source_sizes": sizes,
                    "chunk_sha256": sha256(chunk_path),
                    "resumed": False,
                }
            )
        finally:
            for path in paths.values():
                path.unlink(missing_ok=True)
            gc.collect()

    chunks = [pd.read_parquet(TESSERA_CHUNK_DIR / f"{name}.parquet") for name in tile_names]
    aligned = pd.concat(chunks, ignore_index=True).sort_values("row_id")
    if len(aligned) != len(sample) or aligned["row_id"].duplicated().any():
        raise RuntimeError("TESSERA alignment does not match the LiDAR sample")
    valid_fraction = float(aligned["tessera_valid"].mean())
    if valid_fraction < float(setting["minimum_valid_alignment_fraction"]):
        raise RuntimeError(
            f"TESSERA valid fraction {valid_fraction:.3f} is below the frozen threshold"
        )
    write_parquet(aligned, TESSERA_PATH)
    manifest = {
        "created_utc": utc_now(),
        "sample_sha256": sha256(SAMPLE_PATH),
        "registry_manifest_sha256": sha256(manifest_path),
        "landmask_manifest_sha256": sha256(landmask_path),
        "dataset_version": version,
        "dataset_variant": variant,
        "embedding_year": year,
        "tile_count": len(tile_names),
        "sample_rows": len(aligned),
        "valid_rows": int(aligned["tessera_valid"].sum()),
        "valid_fraction": valid_fraction,
        "streamed_source_tiles_retained": False,
        "chunks": records,
        "output_sha256": sha256(TESSERA_PATH),
    }
    write_json(TESSERA_MANIFEST_PATH, manifest)
    if TESSERA_STREAM_DIR.exists():
        shutil.rmtree(TESSERA_STREAM_DIR)
    print(
        f"tessera: aligned {manifest['valid_rows']:,}/{len(aligned):,} "
        f"cells across {len(tile_names)} tiles",
        flush=True,
    )


def valid_target_mask(
    frame: pd.DataFrame, target: str, config: dict[str, Any]
) -> np.ndarray:
    values = frame[target].to_numpy(dtype=np.float64)
    valid = np.isfinite(values)
    bounds = config["targets"]["valid_ranges"].get(target)
    if bounds is not None:
        valid &= values >= float(bounds[0])
        valid &= values <= float(bounds[1])
    return valid


def spatial_weights(blocks: pd.Series) -> np.ndarray:
    counts = blocks.value_counts()
    weights = blocks.map(1.0 / counts).to_numpy(dtype=np.float64)
    return weights * len(weights) / weights.sum()


def metric_values(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    valid = np.isfinite(observed) & np.isfinite(predicted)
    observed = observed[valid]
    predicted = predicted[valid]
    if len(observed) < 3:
        return {name: float("nan") for name in METRIC_COLUMNS}
    prediction_is_constant = bool(np.ptp(predicted) <= 1e-14)
    observation_is_constant = bool(np.ptp(observed) <= 1e-14)
    if prediction_is_constant or observation_is_constant:
        pearson = float("nan")
        spearman = float("nan")
    else:
        pearson = pearsonr(observed, predicted).statistic
        spearman = spearmanr(observed, predicted).statistic
    return {
        "rmse": float(np.sqrt(mean_squared_error(observed, predicted))),
        "mae": float(mean_absolute_error(observed, predicted)),
        "r2": float(r2_score(observed, predicted)),
        "pearson_r": float(pearson),
        "spearman_r": float(spearman),
        "bias": float(np.mean(predicted - observed)),
    }


def fit_ridge(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    alpha: float,
) -> Any:
    scaler = StandardScaler().fit(x, sample_weight=weights)
    model = Ridge(alpha=alpha).fit(
        scaler.transform(x), y, sample_weight=weights
    )
    return scaler, model


def fit_histogram_model(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    config: dict[str, Any],
) -> HistGradientBoostingRegressor:
    settings = config["models"]["nonlinear"]
    model = HistGradientBoostingRegressor(
        learning_rate=float(settings["learning_rate"]),
        max_iter=int(settings["max_iter"]),
        max_leaf_nodes=int(settings["max_leaf_nodes"]),
        min_samples_leaf=int(settings["min_samples_leaf"]),
        l2_regularization=float(settings["l2_regularization"]),
        random_state=int(settings["random_state"]),
        early_stopping=False,
    )
    return model.fit(x, y, sample_weight=weights)


def buffered_fold_masks(
    frame: pd.DataFrame, fold: int, config: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    test = frame["spatial_region"].to_numpy() == fold
    candidate = ~test
    tree = cKDTree(frame.loc[test, ["bng_x", "bng_y"]].to_numpy(dtype=np.float64))
    distance = tree.query(
        frame.loc[candidate, ["bng_x", "bng_y"]].to_numpy(dtype=np.float64),
        workers=-1,
    )[0]
    retained_candidate = distance >= float(
        config["spatial_evaluation"]["exclusion_buffer_m"]
    )
    train = np.zeros(len(frame), dtype=bool)
    train[np.flatnonzero(candidate)[retained_candidate]] = True
    return train, test, {
        "candidate_training_rows": int(candidate.sum()),
        "buffered_training_rows": int((~retained_candidate).sum()),
        "retained_training_rows": int(train.sum()),
        "test_rows": int(test.sum()),
        "minimum_retained_distance_m": float(distance[retained_candidate].min()),
    }


def append_metric(
    records: list[dict[str, Any]],
    target: str,
    model: str,
    fold: int,
    observed: np.ndarray,
    predicted: np.ndarray,
    train_rows: int,
) -> None:
    records.append(
        {
            "target": target,
            "model": model,
            "fold": fold,
            "aggregation": "pixel",
            "train_rows": train_rows,
            "test_rows": len(observed),
            **metric_values(observed, predicted),
        }
    )


def run_target_models(
    frame: pd.DataFrame,
    target: str,
    fold: int,
    base_train: np.ndarray,
    base_test: np.ndarray,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    target_valid = valid_target_mask(frame, target, config)
    train = base_train & target_valid
    test = base_test & target_valid
    minimum_train = int(
        config["spatial_evaluation"]["minimum_training_rows_per_target"]
    )
    minimum_test = int(
        config["spatial_evaluation"]["minimum_test_rows_per_target"]
    )
    if train.sum() < minimum_train or test.sum() < minimum_test:
        raise RuntimeError(
            f"{target} fold {fold} has too few rows: "
            f"{train.sum()} train, {test.sum()} test"
        )
    y_train = frame.loc[train, target].to_numpy(dtype=np.float64)
    y_test = frame.loc[test, target].to_numpy(dtype=np.float64)
    weights = spatial_weights(frame.loc[train, "spatial_block"])
    metric_records: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    predictions: dict[str, np.ndarray] = {}
    predictions["training_region_mean"] = np.full(
        len(y_test), np.average(y_train, weights=weights)
    )

    coordinate_pipeline = make_pipeline(
        PolynomialFeatures(
            degree=int(config["models"]["coordinate_polynomial_degree"]),
            include_bias=False,
        ),
        StandardScaler(),
        Ridge(alpha=float(config["models"]["ridge_alpha"])),
    )
    coordinate_pipeline.fit(
        frame.loc[train, ["bng_x", "bng_y"]].to_numpy(dtype=np.float64),
        y_train,
        ridge__sample_weight=weights,
    )
    predictions["spatial_coordinate_ridge"] = coordinate_pipeline.predict(
        frame.loc[test, ["bng_x", "bng_y"]].to_numpy(dtype=np.float64)
    )

    x_train = frame.loc[train, EMBEDDING_FEATURES].to_numpy(dtype=np.float32)
    x_test = frame.loc[test, EMBEDDING_FEATURES].to_numpy(dtype=np.float32)
    scaler, ridge = fit_ridge(
        x_train, y_train, weights, float(config["models"]["ridge_alpha"])
    )
    predictions["tessera_ridge"] = ridge.predict(scaler.transform(x_test))
    if target in config["targets"]["nonlinear"]:
        nonlinear = fit_histogram_model(x_train, y_train, weights, config)
        predictions["tessera_hist_gradient_boosting"] = nonlinear.predict(x_test)

    for model, prediction in predictions.items():
        append_metric(
            metric_records,
            target,
            model,
            fold,
            y_test,
            prediction,
            int(train.sum()),
        )
        if target == config["targets"]["primary"]:
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "row_id": frame.loc[test, "row_id"].to_numpy(dtype=np.int64),
                        "target": target,
                        "fold": fold,
                        "model": model,
                        "observed": y_test,
                        "predicted": prediction,
                        "spatial_block": frame.loc[
                            test, "spatial_block"
                        ].to_numpy(),
                        "block_x": frame.loc[test, "block_x"].to_numpy(),
                        "block_y": frame.loc[test, "block_y"].to_numpy(),
                    }
                )
            )
    return metric_records, prediction_frames


def run_height_adjusted(
    frame: pd.DataFrame,
    fold: int,
    base_train: np.ndarray,
    base_test: np.ndarray,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    target = config["targets"]["primary"]
    height_features = config["targets"]["height_adjustment_predictors"]
    valid = valid_target_mask(frame, target, config)
    valid &= np.isfinite(
        frame[height_features].to_numpy(dtype=np.float64)
    ).all(axis=1)
    train = base_train & valid
    test = base_test & valid
    weights = spatial_weights(frame.loc[train, "spatial_block"])
    y_train = frame.loc[train, target].to_numpy(dtype=np.float64)
    y_test = frame.loc[test, target].to_numpy(dtype=np.float64)
    height_model = fit_histogram_model(
        frame.loc[train, height_features].to_numpy(dtype=np.float64),
        y_train,
        weights,
        config,
    )
    height_train = height_model.predict(
        frame.loc[train, height_features].to_numpy(dtype=np.float64)
    )
    height_test = height_model.predict(
        frame.loc[test, height_features].to_numpy(dtype=np.float64)
    )
    residual_train = y_train - height_train
    residual_test = y_test - height_test
    adjusted_target = f"{target}_height_adjusted"
    records: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []

    append_metric(
        records,
        target,
        "lidar_height_diagnostic",
        fold,
        y_test,
        height_test,
        int(train.sum()),
    )
    model_predictions = {
        "height_adjusted_training_mean": np.full(
            len(residual_test), np.average(residual_train, weights=weights)
        )
    }
    x_train = frame.loc[train, EMBEDDING_FEATURES].to_numpy(dtype=np.float32)
    x_test = frame.loc[test, EMBEDDING_FEATURES].to_numpy(dtype=np.float32)
    scaler, ridge = fit_ridge(
        x_train,
        residual_train,
        weights,
        float(config["models"]["ridge_alpha"]),
    )
    model_predictions["height_adjusted_tessera_ridge"] = ridge.predict(
        scaler.transform(x_test)
    )
    nonlinear = fit_histogram_model(
        x_train, residual_train, weights, config
    )
    model_predictions["height_adjusted_tessera_hist_gradient_boosting"] = (
        nonlinear.predict(x_test)
    )
    for model, prediction in model_predictions.items():
        append_metric(
            records,
            adjusted_target,
            model,
            fold,
            residual_test,
            prediction,
            int(train.sum()),
        )
        predictions.append(
            pd.DataFrame(
                {
                    "row_id": frame.loc[test, "row_id"].to_numpy(dtype=np.int64),
                    "target": adjusted_target,
                    "fold": fold,
                    "model": model,
                    "observed": residual_test,
                    "predicted": prediction,
                    "spatial_block": frame.loc[test, "spatial_block"].to_numpy(),
                    "block_x": frame.loc[test, "block_x"].to_numpy(),
                    "block_y": frame.loc[test, "block_y"].to_numpy(),
                }
            )
        )
    return records, predictions


def correlation_outputs(frame: pd.DataFrame) -> None:
    lidar_columns = [column for column in frame.columns if column.startswith("lidar_")]
    finite_counts = {
        column: int(np.isfinite(frame[column].to_numpy(dtype=np.float64)).sum())
        for column in lidar_columns
    }
    usable = [column for column in lidar_columns if finite_counts[column] >= 1000]
    correlation_sample = frame[usable].replace([np.inf, -np.inf], np.nan)
    if len(correlation_sample) > 50000:
        correlation_sample = correlation_sample.sample(
            n=50000, random_state=20260727
        )
    correlations = correlation_sample.corr(method="spearman", min_periods=500)
    correlations.index.name = "metric"
    CORRELATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = CORRELATION_PATH.with_suffix(".tmp.csv")
    correlations.to_csv(temporary)
    temporary.replace(CORRELATION_PATH)
    records = []
    for metric in usable:
        metric_values_array = frame[metric].to_numpy(dtype=np.float64)
        mean_height = frame["lidar_meanH"].to_numpy(dtype=np.float64)
        max_height = frame["lidar_maxH"].to_numpy(dtype=np.float64)
        mean_valid = np.isfinite(metric_values_array) & np.isfinite(mean_height)
        max_valid = np.isfinite(metric_values_array) & np.isfinite(max_height)
        records.append(
            {
                "metric": metric,
                "rows_with_mean_height": int(mean_valid.sum()),
                "rows_with_max_height": int(max_valid.sum()),
                "spearman_with_mean_height": float(
                    spearmanr(
                        metric_values_array[mean_valid], mean_height[mean_valid]
                    ).statistic
                ),
                "spearman_with_max_height": float(
                    spearmanr(
                        metric_values_array[max_valid], max_height[max_valid]
                    ).statistic
                ),
            }
        )
    write_csv(pd.DataFrame(records), HEIGHT_ASSOCIATION_PATH)


def moran_i(blocks: pd.DataFrame, residual_column: str = "residual") -> float:
    values = {
        (int(row.block_x), int(row.block_y)): float(getattr(row, residual_column))
        for row in blocks.itertuples(index=False)
    }
    if len(values) < 3:
        return float("nan")
    mean = float(np.mean(list(values.values())))
    denominator = sum((value - mean) ** 2 for value in values.values())
    numerator = 0.0
    weight_sum = 0
    for (x, y), value in values.items():
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                neighbour = values.get((x + dx, y + dy))
                if neighbour is None:
                    continue
                numerator += (value - mean) * (neighbour - mean)
                weight_sum += 1
    if denominator <= 0 or weight_sum == 0:
        return float("nan")
    return len(values) / weight_sum * numerator / denominator


def primary_diagnostics(predictions: pd.DataFrame, config: dict[str, Any]) -> None:
    primary = config["targets"]["primary"]
    raw = predictions[predictions["target"].eq(primary)].copy()
    moran_records = []
    for model, scoped in raw.groupby("model"):
        scoped["residual"] = scoped["observed"] - scoped["predicted"]
        blocks = (
            scoped.groupby(["block_x", "block_y"], as_index=False)
            .agg(residual=("residual", "mean"), rows=("residual", "size"))
        )
        moran_records.append(
            {
                "target": primary,
                "model": model,
                "blocks": len(blocks),
                "moran_i_queen_1km_block_residual": moran_i(blocks),
            }
        )
    write_csv(pd.DataFrame(moran_records), MORAN_PATH)

    wide = raw.pivot_table(
        index=["row_id", "spatial_block"],
        columns="model",
        values=["observed", "predicted"],
        aggfunc="first",
    )
    observed_columns = [
        column for column in wide.columns if column[0] == "observed"
    ]
    observed = wide[observed_columns[0]].to_numpy(dtype=np.float64)
    blocks = wide.index.get_level_values("spatial_block").to_numpy()
    comparisons = [
        ("tessera_ridge", "training_region_mean"),
        ("tessera_hist_gradient_boosting", "training_region_mean"),
        ("tessera_hist_gradient_boosting", "spatial_coordinate_ridge"),
        ("tessera_hist_gradient_boosting", "tessera_ridge"),
    ]
    rng = np.random.default_rng(int(config["uncertainty"]["seed"]))
    replicates = int(config["uncertainty"]["block_bootstrap_replicates"])
    unique_blocks = np.unique(blocks)
    records = []
    for candidate, baseline in comparisons:
        candidate_column = ("predicted", candidate)
        baseline_column = ("predicted", baseline)
        if candidate_column not in wide.columns or baseline_column not in wide.columns:
            continue
        candidate_error = (
            wide[candidate_column].to_numpy(dtype=np.float64) - observed
        ) ** 2
        baseline_error = (
            wide[baseline_column].to_numpy(dtype=np.float64) - observed
        ) ** 2
        aggregates = pd.DataFrame(
            {
                "block": blocks,
                "candidate_sse": candidate_error,
                "baseline_sse": baseline_error,
            }
        ).groupby("block", as_index=False).agg(
            candidate_sse=("candidate_sse", "sum"),
            baseline_sse=("baseline_sse", "sum"),
            rows=("candidate_sse", "size"),
        )
        delta = np.empty(replicates, dtype=np.float64)
        for index in range(replicates):
            sampled = rng.integers(0, len(aggregates), size=len(aggregates))
            scoped = aggregates.iloc[sampled]
            candidate_rmse = math.sqrt(
                scoped["candidate_sse"].sum() / scoped["rows"].sum()
            )
            baseline_rmse = math.sqrt(
                scoped["baseline_sse"].sum() / scoped["rows"].sum()
            )
            delta[index] = candidate_rmse - baseline_rmse
        records.append(
            {
                "target": primary,
                "contrast": f"{candidate}_minus_{baseline}",
                "spatial_blocks": len(unique_blocks),
                "replicates": replicates,
                "delta_rmse_mean": float(delta.mean()),
                "delta_rmse_ci_low": float(np.quantile(delta, 0.025)),
                "delta_rmse_ci_high": float(np.quantile(delta, 0.975)),
                "probability_delta_below_zero": float(np.mean(delta < 0)),
            }
        )
    write_csv(pd.DataFrame(records), BOOTSTRAP_PATH)


def run_evaluation(config: dict[str, Any]) -> None:
    protected = [
        METRICS_PATH,
        PRIMARY_PREDICTIONS_PATH,
        CORRELATION_PATH,
        HEIGHT_ASSOCIATION_PATH,
        BOOTSTRAP_PATH,
        MORAN_PATH,
    ]
    if all(path.exists() for path in protected):
        print("evaluate: resumed complete checkpoint", flush=True)
        return
    sample = pd.read_parquet(SAMPLE_PATH)
    aligned = pd.read_parquet(TESSERA_PATH)
    frame = sample.merge(
        aligned, on="row_id", how="inner", validate="one_to_one"
    )
    frame = frame[frame["tessera_valid"]].reset_index(drop=True)
    if not np.isfinite(
        frame[EMBEDDING_FEATURES].to_numpy(dtype=np.float32)
    ).all():
        raise RuntimeError("Valid TESSERA rows contain non-finite embeddings")
    correlation_outputs(frame)
    records: list[dict[str, Any]] = []
    primary_predictions: list[pd.DataFrame] = []
    folds = range(int(config["spatial_evaluation"]["region_count"]))
    for fold in folds:
        base_train, base_test, fold_summary = buffered_fold_masks(
            frame, fold, config
        )
        print(
            f"evaluate fold {fold + 1}/{config['spatial_evaluation']['region_count']}: "
            f"{fold_summary['retained_training_rows']:,} train, "
            f"{fold_summary['test_rows']:,} test, "
            f"{fold_summary['buffered_training_rows']:,} buffer exclusions",
            flush=True,
        )
        for target in config["targets"]["representative"]:
            target_records, target_predictions = run_target_models(
                frame, target, fold, base_train, base_test, config
            )
            records.extend(target_records)
            primary_predictions.extend(target_predictions)
        adjusted_records, adjusted_predictions = run_height_adjusted(
            frame, fold, base_train, base_test, config
        )
        records.extend(adjusted_records)
        primary_predictions.extend(adjusted_predictions)

    metrics = pd.DataFrame(records)
    macro = (
        metrics.groupby(["target", "model", "aggregation"], as_index=False)
        .agg(
            train_rows=("train_rows", "mean"),
            test_rows=("test_rows", "sum"),
            **{column: (column, "mean") for column in METRIC_COLUMNS},
        )
    )
    macro["fold"] = "macro_equal_region"
    metrics["fold"] = metrics["fold"].astype(str)
    metrics = pd.concat([metrics, macro], ignore_index=True)
    write_csv(metrics, METRICS_PATH)
    predictions = pd.concat(primary_predictions, ignore_index=True)
    write_parquet(predictions, PRIMARY_PREDICTIONS_PATH)
    primary_diagnostics(predictions, config)
    print(
        f"evaluate: wrote {len(metrics):,} metric rows and "
        f"{len(predictions):,} primary prediction rows",
        flush=True,
    )


def make_figure(config: dict[str, Any]) -> None:
    metrics = pd.read_csv(METRICS_PATH)
    predictions = pd.read_parquet(PRIMARY_PREDICTIONS_PATH)
    sample = pd.read_parquet(
        SAMPLE_PATH,
        columns=[
            "row_id",
            "bng_x",
            "bng_y",
            config["targets"]["primary"],
        ],
    )
    primary = config["targets"]["primary"]
    macro = metrics[
        metrics["fold"].eq("macro_equal_region")
        & metrics["target"].isin(
            [
                primary,
                "lidar_vci_2m",
                "lidar_Cov",
                "lidar_meanH",
                f"{primary}_height_adjusted",
            ]
        )
    ].copy()
    model_order = [
        "training_region_mean",
        "spatial_coordinate_ridge",
        "tessera_ridge",
        "tessera_hist_gradient_boosting",
        "lidar_height_diagnostic",
        "height_adjusted_training_mean",
        "height_adjusted_tessera_ridge",
        "height_adjusted_tessera_hist_gradient_boosting",
    ]
    colours = {
        "training_region_mean": "#7A8288",
        "spatial_coordinate_ridge": "#E69F00",
        "tessera_ridge": "#168AAD",
        "tessera_hist_gradient_boosting": "#007F5F",
        "lidar_height_diagnostic": "#6A4C93",
        "height_adjusted_training_mean": "#7A8288",
        "height_adjusted_tessera_ridge": "#168AAD",
        "height_adjusted_tessera_hist_gradient_boosting": "#007F5F",
    }

    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), constrained_layout=True)
    raw = predictions[predictions["target"].eq(primary)]
    best_name = (
        "tessera_hist_gradient_boosting"
        if "tessera_hist_gradient_boosting" in set(raw["model"])
        else "tessera_ridge"
    )
    scoped = raw[raw["model"].eq(best_name)].merge(
        sample, on="row_id", validate="many_to_one"
    )
    point_sample = scoped.sample(
        n=min(50000, len(scoped)), random_state=20260727
    )
    hexbin = axes[0, 0].hexbin(
        point_sample["observed"],
        point_sample["predicted"],
        gridsize=55,
        mincnt=1,
        cmap="viridis",
    )
    limits = [
        min(point_sample["observed"].min(), point_sample["predicted"].min()),
        max(point_sample["observed"].max(), point_sample["predicted"].max()),
    ]
    axes[0, 0].plot(limits, limits, "--", color="#333333", linewidth=1)
    axes[0, 0].set(
        xlabel="Airborne-LiDAR canopy entropy",
        ylabel="Out-of-region TESSERA prediction",
        title="Held-out-region canopy entropy",
    )
    figure.colorbar(hexbin, ax=axes[0, 0], label="Sampled cells per hexagon")

    model_rows = macro[
        macro["target"].eq(primary)
        & macro["model"].isin(model_order[:5])
    ].set_index("model").reindex([name for name in model_order[:5] if name in set(macro["model"])])
    axes[0, 1].bar(
        np.arange(len(model_rows)),
        model_rows["r2"],
        color=[colours[name] for name in model_rows.index],
    )
    axes[0, 1].axhline(0, color="#333333", linewidth=0.8)
    axes[0, 1].set_xticks(
        np.arange(len(model_rows)),
        [
            "Mean",
            "Coordinates",
            "TESSERA\nlinear",
            "TESSERA\nnonlinear",
            "LiDAR height\ndiagnostic",
        ][: len(model_rows)],
        rotation=20,
        ha="right",
    )
    axes[0, 1].set(
        ylabel="Mean held-out-region $R^2$",
        title="Primary-target comparison",
    )

    mapped = sample.merge(
        scoped[["row_id", "predicted"]], on="row_id", how="inner"
    )
    map_sample = mapped.sample(
        n=min(80000, len(mapped)), random_state=20260727
    )
    scatter = axes[1, 0].scatter(
        map_sample["bng_x"] / 1000,
        map_sample["bng_y"] / 1000,
        c=map_sample[primary],
        s=2,
        cmap="viridis",
        linewidths=0,
    )
    axes[1, 0].set(
        xlabel="British National Grid easting (km)",
        ylabel="British National Grid northing (km)",
        title="Airborne-LiDAR canopy entropy",
        aspect="equal",
    )
    figure.colorbar(scatter, ax=axes[1, 0], label="Canopy entropy")

    adjusted = macro[
        macro["target"].eq(f"{primary}_height_adjusted")
    ].copy()
    adjusted_order = [
        name
        for name in model_order[5:]
        if name in set(adjusted["model"])
    ]
    adjusted = adjusted.set_index("model").reindex(adjusted_order)
    axes[1, 1].bar(
        np.arange(len(adjusted)),
        adjusted["r2"],
        color=[colours[name] for name in adjusted.index],
    )
    axes[1, 1].axhline(0, color="#333333", linewidth=0.8)
    axes[1, 1].set_xticks(
        np.arange(len(adjusted)),
        ["Mean", "TESSERA\nlinear", "TESSERA\nnonlinear"][: len(adjusted)],
        rotation=20,
        ha="right",
    )
    axes[1, 1].set(
        ylabel="Mean held-out-region $R^2$",
        title="Canopy entropy after removing LiDAR height",
    )
    for label, axis in zip(["a", "b", "c", "d"], axes.ravel(), strict=True):
        axis.text(
            -0.12,
            1.04,
            label,
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=12,
        )
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=220, bbox_inches="tight")
    plt.close(figure)


def result_sentence(row: pd.Series) -> str:
    rank = (
        "undefined for a constant prediction"
        if not np.isfinite(row.spearman_r)
        else f"{row.spearman_r:.3f}"
    )
    return (
        f"RMSE {row.rmse:.3f}, R2 {row.r2:.3f}, "
        f"Spearman r {rank}"
    )


def run_report(config: dict[str, Any]) -> None:
    if REPORT_PATH.exists() and RESULT_FREEZE_PATH.exists() and FIGURE_PATH.exists():
        print("report: resumed complete checkpoint", flush=True)
        return
    make_figure(config)
    metrics = pd.read_csv(METRICS_PATH)
    macro = metrics[metrics["fold"].eq("macro_equal_region")].copy()
    primary = config["targets"]["primary"]
    primary_rows = macro[macro["target"].eq(primary)].set_index("model")
    adjusted_rows = macro[
        macro["target"].eq(f"{primary}_height_adjusted")
    ].set_index("model")
    height_row = primary_rows.loc["lidar_height_diagnostic"]
    bootstrap = pd.read_csv(BOOTSTRAP_PATH)
    associations = pd.read_csv(HEIGHT_ASSOCIATION_PATH)
    primary_height = associations[
        associations["metric"].eq(primary)
    ].iloc[0]
    lines = [
        "# Phase 14 Scotland Airborne-LiDAR Results",
        "",
        "## Design",
        "",
        (
            "The analysis used a deterministic sample from the complete 2023 "
            "Cairngorms airborne-LiDAR metric raster and 2023 TESSERA embeddings. "
            "Five contiguous spatial regions were withheld in turn, with a 1 km "
            "training exclusion buffer."
        ),
        "",
        "## Primary canopy-entropy result",
        "",
    ]
    for model in [
        "training_region_mean",
        "spatial_coordinate_ridge",
        "tessera_ridge",
        "tessera_hist_gradient_boosting",
        "lidar_height_diagnostic",
    ]:
        if model in primary_rows.index:
            lines.append(f"- `{model}`: {result_sentence(primary_rows.loc[model])}.")
    lines.extend(
        [
            "",
            (
                "The LiDAR canopy-entropy target had a Spearman association of "
                f"{primary_height.spearman_with_mean_height:.3f} with mean canopy "
                "height. A model using LiDAR height metrics alone obtained "
                f"{result_sentence(height_row)}."
            ),
            "",
            "## Height-adjusted result",
            "",
        ]
    )
    for model in [
        "height_adjusted_training_mean",
        "height_adjusted_tessera_ridge",
        "height_adjusted_tessera_hist_gradient_boosting",
    ]:
        if model in adjusted_rows.index:
            lines.append(f"- `{model}`: {result_sentence(adjusted_rows.loc[model])}.")
    lines.extend(
        [
            "",
            "## Paired spatial-block uncertainty",
            "",
        ]
    )
    for row in bootstrap.itertuples(index=False):
        lines.append(
            f"- `{row.contrast}`: delta RMSE {row.delta_rmse_mean:.4f} "
            f"[{row.delta_rmse_ci_low:.4f}, {row.delta_rmse_ci_high:.4f}]."
        )
    lines.extend(
        [
            "",
            "## Scope",
            "",
            (
                "These results measure spatial transfer across one Scottish landscape. "
                "They do not establish geographic transfer to another landscape, and "
                "the airborne-LiDAR metrics are structural reference measurements rather "
                "than direct biodiversity observations."
            ),
            "",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = REPORT_PATH.with_suffix(".tmp.md")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(REPORT_PATH)
    freeze_basis = {
        "analysis_script_sha256": sha256(Path(__file__).resolve()),
        "config_sha256": sha256(CONFIG_PATH),
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "inventory_sha256": sha256(INVENTORY_PATH),
        "sample_sha256": sha256(SAMPLE_PATH),
        "tessera_sha256": sha256(TESSERA_PATH),
        "metrics_sha256": sha256(METRICS_PATH),
        "predictions_sha256": sha256(PRIMARY_PREDICTIONS_PATH),
        "correlation_sha256": sha256(CORRELATION_PATH),
        "bootstrap_sha256": sha256(BOOTSTRAP_PATH),
        "moran_sha256": sha256(MORAN_PATH),
        "figure_sha256": sha256(FIGURE_PATH),
        "report_sha256": sha256(REPORT_PATH),
    }
    freeze = {
        "freeze_id": f"phase14-scotland-lidar-{canonical_hash(freeze_basis)[:12]}",
        "created_utc": utc_now(),
        "status": "complete",
        "freeze_basis": freeze_basis,
    }
    write_json(RESULT_FREEZE_PATH, freeze)
    print(f"report: {freeze['freeze_id']}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--through",
        choices=STAGES,
        default="report",
        help="Run resumable stages from inventory through this stage.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config()
    runners = {
        "inventory": run_inventory,
        "sample": run_sample,
        "tessera": run_tessera,
        "evaluate": run_evaluation,
        "report": run_report,
    }
    for stage in STAGES:
        print(f"\n=== phase14 {stage} ===", flush=True)
        runners[stage](config)
        if stage == args.through:
            break
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1)
