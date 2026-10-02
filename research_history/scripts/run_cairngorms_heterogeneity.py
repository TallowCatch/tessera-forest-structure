#!/usr/bin/env python3
"""Run the resumable Cairngorms forest-heterogeneity experiment."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import shutil
import sys
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
from pyproj import Transformer
from rasterio.transform import rowcol
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_scotland_lidar as shared  # noqa: E402


CONFIG_PATH = ROOT / "configs/cairngorms_heterogeneity.yaml"
PROTOCOL_PATH = (
    ROOT
    / "metadata/project_config_phase17_cairngorms_heterogeneity_protocol_freeze.yaml"
)
INVENTORY_PATH = ROOT / "metadata/phase17_cairngorms_inventory.json"
TARGET_PATH = ROOT / "data/processed/phase17_cairngorms_50m_targets.parquet"
TARGET_QC_PATH = ROOT / "metadata/phase17_cairngorms_target_qc.json"
TESSERA_CHUNK_DIR = ROOT / "data/interim/phase17_cairngorms_tessera_chunks"
TESSERA_STREAM_DIR = ROOT / "data/interim/phase17_cairngorms_tessera_stream"
TESSERA_PATH = (
    ROOT / "data/processed/phase17_cairngorms_50m_tessera_context.parquet"
)
TESSERA_MANIFEST_PATH = (
    ROOT / "metadata/phase17_cairngorms_tessera_manifest.json"
)
FOLD_DIR = ROOT / "data/interim/phase17_cairngorms_evaluation_folds"
METRICS_PATH = ROOT / "outputs/tables/phase17_cairngorms_metrics.csv"
POOLED_METRICS_PATH = (
    ROOT / "outputs/tables/phase17_cairngorms_pooled_metrics.csv"
)
PREDICTIONS_PATH = (
    ROOT / "data/processed/phase17_cairngorms_oof_predictions.parquet"
)
BOOTSTRAP_PATH = (
    ROOT / "outputs/tables/phase17_cairngorms_primary_bootstrap.csv"
)
FIGURE_PATH = (
    ROOT / "outputs/figures/phase17_cairngorms_heterogeneity.png"
)
REPORT_PATH = (
    ROOT / "outputs/reports/phase17_cairngorms_heterogeneity_results.md"
)
RESULT_FREEZE_PATH = (
    ROOT / "metadata/phase17_cairngorms_result_freeze.json"
)

EMBEDDING_DIMENSIONS = 128
STAGES = ["inventory", "targets", "tessera", "evaluate", "report"]
METRIC_NAMES = ["rmse", "mae", "r2", "pearson_r", "spearman_r", "bias"]


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


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "phase17_cairngorms_heterogeneity"
    ]


def source_path(config: dict[str, Any]) -> Path:
    return ROOT / config["source"]["lidar_metrics_path"]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def metric_values(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    valid = np.isfinite(observed) & np.isfinite(predicted)
    observed = observed[valid]
    predicted = predicted[valid]
    if len(observed) < 3:
        return {name: float("nan") for name in METRIC_NAMES}
    constant = np.ptp(observed) <= 1e-14 or np.ptp(predicted) <= 1e-14
    return {
        "rmse": float(np.sqrt(mean_squared_error(observed, predicted))),
        "mae": float(mean_absolute_error(observed, predicted)),
        "r2": float(r2_score(observed, predicted)),
        "pearson_r": (
            float("nan")
            if constant
            else float(pearsonr(observed, predicted).statistic)
        ),
        "spearman_r": (
            float("nan")
            if constant
            else float(spearmanr(observed, predicted).statistic)
        ),
        "bias": float(np.mean(predicted - observed)),
    }


def run_inventory(config: dict[str, Any]) -> None:
    source = source_path(config)
    if not source.exists():
        raise RuntimeError(f"Missing Cairngorms LiDAR metrics: {source}")
    if not PROTOCOL_PATH.exists():
        PROTOCOL_PATH.parent.mkdir(parents=True, exist_ok=True)
        PROTOCOL_PATH.write_text(
            CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8"
        )
    if CONFIG_PATH.read_bytes() != PROTOCOL_PATH.read_bytes():
        raise RuntimeError(
            "The phase17 configuration differs from its frozen protocol"
        )

    source_hash = sha256(source)
    expected_hash = str(config["source"]["expected_source_sha256"])
    if source_hash != expected_hash:
        raise RuntimeError("The Cairngorms LiDAR source hash changed")

    with rasterio.open(source) as dataset:
        if dataset.count != int(config["source"]["expected_band_count"]):
            raise RuntimeError("Unexpected LiDAR band count")
        if not np.allclose(
            dataset.res,
            [float(config["source"]["expected_resolution_m"])] * 2,
        ):
            raise RuntimeError("Unexpected LiDAR raster resolution")
        if not np.allclose(
            dataset.bounds,
            config["source"]["expected_bounds_bng"],
            atol=0.01,
        ):
            raise RuntimeError("Unexpected LiDAR raster bounds")
        descriptions = list(dataset.descriptions)
        required = {
            "lidar_meanH",
            "lidar_p_95",
            "lidar_Cov",
            "lidar_canopy_shannon",
            "lidar_gapFrac",
            "lidar_stdH",
        }
        missing = sorted(required - set(descriptions))
        if missing:
            raise RuntimeError(f"LiDAR raster lacks required bands: {missing}")
        profile = {
            "width": dataset.width,
            "height": dataset.height,
            "count": dataset.count,
            "resolution_m": list(dataset.res),
            "bounds_bng": list(dataset.bounds),
            "bands": descriptions,
        }

    record = {
        "created_utc": utc_now(),
        "source_path": str(source.relative_to(ROOT)),
        "source_sha256": source_hash,
        "source_size_bytes": source.stat().st_size,
        "config_sha256": sha256(CONFIG_PATH),
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "profile": profile,
    }
    if INVENTORY_PATH.exists():
        existing = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
        for key in ["source_sha256", "config_sha256", "protocol_sha256"]:
            if existing[key] != record[key]:
                raise RuntimeError("The phase17 inventory checkpoint is stale")
        print("inventory: validated existing checkpoint", flush=True)
        return
    atomic_json(INVENTORY_PATH, record)
    print(
        f"inventory: {profile['width']:,} x {profile['height']:,}, "
        f"{profile['count']} LiDAR metrics",
        flush=True,
    )


def block_mean(values: np.ndarray, cells: int) -> tuple[np.ndarray, np.ndarray]:
    height = values.shape[0] // cells * cells
    width = values.shape[1] // cells * cells
    shaped = values[:height, :width].reshape(
        height // cells, cells, width // cells, cells
    )
    valid = np.isfinite(shaped)
    count = valid.sum(axis=(1, 3)).astype(np.int16)
    total = np.where(valid, shaped, 0.0).sum(axis=(1, 3), dtype=np.float64)
    mean = np.full(count.shape, np.nan, dtype=np.float32)
    usable = count > 0
    mean[usable] = (total[usable] / count[usable]).astype(np.float32)
    return mean, count


def block_std(values: np.ndarray, cells: int) -> np.ndarray:
    height = values.shape[0] // cells * cells
    width = values.shape[1] // cells * cells
    shaped = values[:height, :width].reshape(
        height // cells, cells, width // cells, cells
    )
    valid = np.isfinite(shaped)
    count = valid.sum(axis=(1, 3))
    total = np.where(valid, shaped, 0.0).sum(axis=(1, 3), dtype=np.float64)
    total_sq = np.where(valid, shaped * shaped, 0.0).sum(
        axis=(1, 3), dtype=np.float64
    )
    result = np.full(count.shape, np.nan, dtype=np.float32)
    usable = count > 1
    variance = np.maximum(
        total_sq[usable] / count[usable]
        - np.square(total[usable] / count[usable]),
        0.0,
    )
    result[usable] = np.sqrt(variance).astype(np.float32)
    return result


def aggregate_band(
    dataset: rasterio.DatasetReader,
    lookup: dict[str, int],
    name: str,
    cells: int,
    operation: str = "mean",
) -> tuple[np.ndarray, np.ndarray | None]:
    values = np.array(
        dataset.read(lookup[name]), dtype=np.float32, copy=True
    )
    values[~np.isfinite(values)] = np.nan
    if operation == "mean":
        result, count = block_mean(values, cells)
        return result, count
    if operation == "std":
        return block_std(values, cells), None
    raise ValueError(f"Unsupported block operation: {operation}")


def target_profile(values: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return {
        "count": int(len(finite)),
        "mean": float(np.mean(finite)),
        "standard_deviation": float(np.std(finite)),
        "quantiles": {
            str(q): float(v)
            for q, v in zip(
                [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0],
                np.quantile(finite, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]),
            )
        },
    }


def run_targets(config: dict[str, Any]) -> None:
    if TARGET_PATH.exists() and TARGET_QC_PATH.exists():
        qc = json.loads(TARGET_QC_PATH.read_text(encoding="utf-8"))
        if (
            qc["source_sha256"] == sha256(source_path(config))
            and qc["config_sha256"] == sha256(CONFIG_PATH)
            and qc["output_sha256"] == sha256(TARGET_PATH)
        ):
            print("targets: validated existing checkpoint", flush=True)
            return
        raise RuntimeError("The phase17 target checkpoint is stale")

    settings = config["population"]
    cells = int(settings["patch_cells"])
    with rasterio.open(source_path(config)) as dataset:
        lookup = {
            str(name): index
            for index, name in enumerate(dataset.descriptions, start=1)
        }
        mean_height, _ = aggregate_band(
            dataset, lookup, "lidar_meanH", cells
        )
        between_height_sd, _ = aggregate_band(
            dataset, lookup, "lidar_meanH", cells, "std"
        )
        mean_p95, _ = aggregate_band(
            dataset, lookup, "lidar_p_95", cells
        )
        mean_cover, _ = aggregate_band(
            dataset, lookup, "lidar_Cov", cells
        )
        mean_shannon, valid_count = aggregate_band(
            dataset, lookup, "lidar_canopy_shannon", cells
        )
        mean_gap, _ = aggregate_band(
            dataset, lookup, "lidar_gapFrac", cells
        )
        mean_within_height_sd, _ = aggregate_band(
            dataset, lookup, "lidar_stdH", cells
        )
        transform = dataset.transform

    assert valid_count is not None
    eligible = (
        (valid_count >= int(settings["minimum_valid_cells"]))
        & np.isfinite(mean_height)
        & np.isfinite(mean_p95)
        & np.isfinite(mean_cover)
        & np.isfinite(mean_shannon)
        & (mean_height >= float(settings["minimum_mean_canopy_height_m"]))
        & (mean_height <= float(settings["maximum_mean_canopy_height_m"]))
        & (mean_p95 <= float(settings["maximum_p95_height_m"]))
        & (mean_cover >= float(settings["minimum_mean_canopy_cover"]))
    )
    patch_rows, patch_columns = np.nonzero(eligible)
    centre_offset = (cells - 1) / 2
    raster_rows = patch_rows * cells + centre_offset
    raster_columns = patch_columns * cells + centre_offset
    bng_x = (
        transform.c
        + (raster_columns + 0.5) * transform.a
        + (raster_rows + 0.5) * transform.b
    )
    bng_y = (
        transform.f
        + (raster_columns + 0.5) * transform.d
        + (raster_rows + 0.5) * transform.e
    )
    transformer = Transformer.from_crs(
        config["study"]["analysis_crs"], "EPSG:4326", always_xy=True
    )
    longitude, latitude = transformer.transform(bng_x, bng_y)
    frame = pd.DataFrame(
        {
            "raster_patch_row": patch_rows.astype(np.int32),
            "raster_patch_column": patch_columns.astype(np.int32),
            "bng_x": np.asarray(bng_x, dtype=np.float64),
            "bng_y": np.asarray(bng_y, dtype=np.float64),
            "longitude": np.asarray(longitude, dtype=np.float64),
            "latitude": np.asarray(latitude, dtype=np.float64),
            "lidar_valid_cell_count": valid_count[eligible].astype(np.int16),
            "mean_canopy_shannon_50m": mean_shannon[eligible],
            "between_cell_height_sd_50m": between_height_sd[eligible],
            "mean_gap_fraction_50m": mean_gap[eligible],
            "mean_height_50m": mean_height[eligible],
            "mean_p95_height_50m": mean_p95[eligible],
            "mean_cover_50m": mean_cover[eligible],
            "mean_within_cell_height_sd_50m": mean_within_height_sd[eligible],
        }
    )
    frame.insert(0, "row_id", np.arange(len(frame), dtype=np.int64))
    frame = shared.assign_spatial_regions(frame, config)
    atomic_parquet(frame, TARGET_PATH)

    target_names = list(config["targets"]["representative"])
    qc = {
        "created_utc": utc_now(),
        "source_sha256": sha256(source_path(config)),
        "config_sha256": sha256(CONFIG_PATH),
        "patch_size_m": int(settings["patch_size_m"]),
        "all_raster_patches": int(eligible.size),
        "eligible_patches": int(len(frame)),
        "spatial_region_counts": {
            str(key): int(value)
            for key, value in frame["spatial_region"].value_counts().sort_index().items()
        },
        "targets": {
            name: target_profile(frame[name].to_numpy()) for name in target_names
        },
        "output_sha256": sha256(TARGET_PATH),
    }
    atomic_json(TARGET_QC_PATH, qc)
    print(
        f"targets: retained {len(frame):,}/{eligible.size:,} non-overlapping "
        f"50 m mature-forest units",
        flush=True,
    )


def patch_sample_points(
    targets: pd.DataFrame,
    transform: rasterio.Affine,
    cells: int,
) -> dict[str, np.ndarray]:
    offsets_row = np.repeat(np.arange(cells), cells)
    offsets_column = np.tile(np.arange(cells), cells)
    patch_count = len(targets)
    patch_ids = np.repeat(
        targets["row_id"].to_numpy(dtype=np.int64), cells * cells
    )
    local_rows = np.tile(offsets_row, patch_count)
    local_columns = np.tile(offsets_column, patch_count)
    raster_rows = (
        np.repeat(
            targets["raster_patch_row"].to_numpy(dtype=np.int64), cells * cells
        )
        * cells
        + local_rows
    )
    raster_columns = (
        np.repeat(
            targets["raster_patch_column"].to_numpy(dtype=np.int64),
            cells * cells,
        )
        * cells
        + local_columns
    )
    x = (
        transform.c
        + (raster_columns + 0.5) * transform.a
        + (raster_rows + 0.5) * transform.b
    )
    y = (
        transform.f
        + (raster_columns + 0.5) * transform.d
        + (raster_rows + 0.5) * transform.e
    )
    centre = (cells - 1) / 2
    return {
        "patch_id": patch_ids,
        "bng_x": np.asarray(x, dtype=np.float64),
        "bng_y": np.asarray(y, dtype=np.float64),
        "offset_x": (local_columns - centre).astype(np.float32),
        "offset_y": (centre - local_rows).astype(np.float32),
    }


def reduce_tile_embeddings(
    patch_id: np.ndarray,
    offset_x: np.ndarray,
    offset_y: np.ndarray,
    embeddings: np.ndarray,
) -> dict[str, np.ndarray]:
    order = np.argsort(patch_id, kind="stable")
    patch_id = patch_id[order]
    offset_x = offset_x[order].astype(np.float32)
    offset_y = offset_y[order].astype(np.float32)
    embeddings = embeddings[order].astype(np.float32)
    unique, starts, count = np.unique(
        patch_id, return_index=True, return_counts=True
    )

    def grouped_sum(values: np.ndarray) -> np.ndarray:
        return np.add.reduceat(values, starts, axis=0)

    centre_mask = (offset_x == 0) & (offset_y == 0)
    centre = np.full(
        (len(unique), EMBEDDING_DIMENSIONS), np.nan, dtype=np.float32
    )
    centre_valid = np.zeros(len(unique), dtype=bool)
    if centre_mask.any():
        positions = np.searchsorted(unique, patch_id[centre_mask])
        centre[positions] = embeddings[centre_mask]
        centre_valid[positions] = True

    return {
        "patch_id": unique.astype(np.int64),
        "count": count.astype(np.int16),
        "sum_x": grouped_sum(offset_x).astype(np.float32),
        "sum_y": grouped_sum(offset_y).astype(np.float32),
        "sum_x2": grouped_sum(offset_x * offset_x).astype(np.float32),
        "sum_y2": grouped_sum(offset_y * offset_y).astype(np.float32),
        "sum_embedding": grouped_sum(embeddings).astype(np.float32),
        "sum_embedding_sq": grouped_sum(embeddings * embeddings).astype(
            np.float32
        ),
        "sum_x_embedding": grouped_sum(
            embeddings * offset_x[:, None]
        ).astype(np.float32),
        "sum_y_embedding": grouped_sum(
            embeddings * offset_y[:, None]
        ).astype(np.float32),
        "centre_valid": centre_valid,
        "centre_embedding": centre,
    }


def atomic_npz(path: Path, values: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **values)
    temporary.replace(path)


def parse_tile_pair(pair: np.ndarray) -> tuple[float, float]:
    return round(float(pair[0]) / 100.0, 2), round(float(pair[1]) / 100.0, 2)


def run_tessera(config: dict[str, Any]) -> None:
    if TESSERA_PATH.exists() and TESSERA_MANIFEST_PATH.exists():
        manifest = json.loads(
            TESSERA_MANIFEST_PATH.read_text(encoding="utf-8")
        )
        if (
            manifest["target_sha256"] == sha256(TARGET_PATH)
            and manifest["config_sha256"] == sha256(CONFIG_PATH)
            and manifest["output_sha256"] == sha256(TESSERA_PATH)
        ):
            print("tessera: validated complete context checkpoint", flush=True)
            return
        raise RuntimeError("The phase17 TESSERA checkpoint is stale")

    targets = pd.read_parquet(TARGET_PATH)
    settings = config["tessera"]
    cells = int(config["population"]["patch_cells"])
    with rasterio.open(source_path(config)) as dataset:
        points = patch_sample_points(targets, dataset.transform, cells)

    to_wgs84 = Transformer.from_crs(
        config["study"]["analysis_crs"], "EPSG:4326", always_xy=True
    )
    longitude, latitude = to_wgs84.transform(
        points["bng_x"], points["bng_y"]
    )
    tile_lon = np.round(np.floor(np.asarray(longitude) * 10) / 10 + 0.05, 2)
    tile_lat = np.round(np.floor(np.asarray(latitude) * 10) / 10 + 0.05, 2)
    tile_pairs, tile_code = np.unique(
        np.column_stack(
            [
                np.rint(tile_lon * 100).astype(np.int16),
                np.rint(tile_lat * 100).astype(np.int16),
            ]
        ),
        axis=0,
        return_inverse=True,
    )
    tiles = [parse_tile_pair(pair) for pair in tile_pairs]
    version = str(settings["dataset_version"])
    variant = str(settings["dataset_variant"])
    year = int(settings["embedding_year"])
    registry_path, landmask_path = shared.tessera.ensure_registry(version)
    rows = shared.tessera.lookup_embedding_rows(
        registry_path, tiles, year, version, variant
    )
    masks = shared.tessera.lookup_landmask_rows(landmask_path, tiles)
    shared.tessera.assert_complete_inventory(rows, masks, tiles)
    rows_by_name = {
        shared.tessera.tile_name((float(row.lon), float(row.lat))): row
        for row in rows.itertuples(index=False)
    }
    masks_by_name = {
        shared.tessera.tile_name((float(row.lon), float(row.lat))): row
        for row in masks.itertuples(index=False)
    }

    TESSERA_CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    attempts = int(settings["download_attempts"])
    delay = int(settings["retry_delay_seconds"])
    config_hash = sha256(CONFIG_PATH)
    records: list[dict[str, Any]] = []
    for code, tile in enumerate(tiles):
        name = shared.tessera.tile_name(tile)
        indices = np.flatnonzero(tile_code == code)
        chunk_path = TESSERA_CHUNK_DIR / f"{name}.npz"
        if chunk_path.exists():
            with np.load(chunk_path) as chunk:
                stored_hash = str(chunk["config_sha256"][0])
                stored_points = int(chunk["source_point_count"][0])
            if stored_hash != config_hash or stored_points != len(indices):
                raise RuntimeError(f"Stale phase17 TESSERA chunk: {name}")
            records.append(
                {
                    "tile": name,
                    "source_points": len(indices),
                    "chunk_sha256": sha256(chunk_path),
                    "resumed": True,
                }
            )
            print(
                f"tessera [{code + 1}/{len(tiles)}]: resumed {name}",
                flush=True,
            )
            continue

        row = rows_by_name[name]
        mask_row = masks_by_name[name]
        paths = shared.tessera.local_tile_paths(TESSERA_STREAM_DIR, year, tile)
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
        required = sum(sizes.values()) + int(
            settings["minimum_free_space_after_download_bytes"]
        )
        free = shutil.disk_usage(ROOT).free
        if free < required:
            raise RuntimeError(
                f"Insufficient space for streamed tile {name}: "
                f"{free / 1e9:.2f} GB free, {required / 1e9:.2f} GB required"
            )
        print(
            f"tessera [{code + 1}/{len(tiles)}]: acquire {name} for "
            f"{len(indices):,} patch pixels",
            flush=True,
        )
        try:
            for kind in ["embedding", "scales", "landmask"]:
                shared.download_checked(
                    shared.tessera.s3_url(sources[kind]),
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
                destination_transform = dataset.transform
            transformer = Transformer.from_crs(
                config["study"]["analysis_crs"],
                destination_crs,
                always_xy=True,
            )
            x, y = transformer.transform(
                points["bng_x"][indices], points["bng_y"][indices]
            )
            rows_index, columns_index = rowcol(destination_transform, x, y)
            rows_index = np.asarray(rows_index, dtype=np.int64)
            columns_index = np.asarray(columns_index, dtype=np.int64)
            inside = (
                (rows_index >= 0)
                & (columns_index >= 0)
                & (rows_index < quantized.shape[0])
                & (columns_index < quantized.shape[1])
            )
            local_positions = np.flatnonzero(inside)
            local_scales = scales[
                rows_index[local_positions], columns_index[local_positions]
            ].astype(np.float32)
            local_landmask = landmask[
                rows_index[local_positions], columns_index[local_positions]
            ]
            usable = (
                np.isfinite(local_scales)
                & (local_scales > 0)
                & (local_landmask > 0)
            )
            valid_positions = local_positions[usable]
            embeddings = (
                quantized[
                    rows_index[valid_positions],
                    columns_index[valid_positions],
                ].astype(np.float32)
                * scales[
                    rows_index[valid_positions],
                    columns_index[valid_positions],
                ].astype(np.float32)[:, None]
            )
            finite = np.isfinite(embeddings).all(axis=1)
            valid_positions = valid_positions[finite]
            embeddings = embeddings[finite]
            source_indices = indices[valid_positions]
            partial = reduce_tile_embeddings(
                points["patch_id"][source_indices],
                points["offset_x"][source_indices],
                points["offset_y"][source_indices],
                embeddings,
            )
            partial["config_sha256"] = np.asarray([config_hash])
            partial["source_point_count"] = np.asarray(
                [len(indices)], dtype=np.int64
            )
            partial["valid_point_count"] = np.asarray(
                [len(source_indices)], dtype=np.int64
            )
            atomic_npz(chunk_path, partial)
            records.append(
                {
                    "tile": name,
                    "source_points": len(indices),
                    "valid_points": len(source_indices),
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

    patch_count = len(targets)
    shape = (patch_count, EMBEDDING_DIMENSIONS)
    count = np.zeros(patch_count, dtype=np.int16)
    sum_x = np.zeros(patch_count, dtype=np.float32)
    sum_y = np.zeros(patch_count, dtype=np.float32)
    sum_x2 = np.zeros(patch_count, dtype=np.float32)
    sum_y2 = np.zeros(patch_count, dtype=np.float32)
    sum_embedding = np.zeros(shape, dtype=np.float32)
    sum_embedding_sq = np.zeros(shape, dtype=np.float32)
    sum_x_embedding = np.zeros(shape, dtype=np.float32)
    sum_y_embedding = np.zeros(shape, dtype=np.float32)
    centre_valid = np.zeros(patch_count, dtype=bool)
    centre_embedding = np.full(shape, np.nan, dtype=np.float32)

    for tile in tiles:
        name = shared.tessera.tile_name(tile)
        with np.load(TESSERA_CHUNK_DIR / f"{name}.npz") as chunk:
            patch_id = chunk["patch_id"].astype(np.int64)
            count[patch_id] += chunk["count"]
            sum_x[patch_id] += chunk["sum_x"]
            sum_y[patch_id] += chunk["sum_y"]
            sum_x2[patch_id] += chunk["sum_x2"]
            sum_y2[patch_id] += chunk["sum_y2"]
            sum_embedding[patch_id] += chunk["sum_embedding"]
            sum_embedding_sq[patch_id] += chunk["sum_embedding_sq"]
            sum_x_embedding[patch_id] += chunk["sum_x_embedding"]
            sum_y_embedding[patch_id] += chunk["sum_y_embedding"]
            local_centre = chunk["centre_valid"].astype(bool)
            if local_centre.any():
                selected = patch_id[local_centre]
                if centre_valid[selected].any():
                    raise RuntimeError("A patch centre was sampled from two tiles")
                centre_valid[selected] = True
                centre_embedding[selected] = chunk["centre_embedding"][
                    local_centre
                ]

    usable = count > 0
    mean = np.full(shape, np.nan, dtype=np.float32)
    standard_deviation = np.full(shape, np.nan, dtype=np.float32)
    gradient_x = np.full(shape, np.nan, dtype=np.float32)
    gradient_y = np.full(shape, np.nan, dtype=np.float32)
    mean[usable] = sum_embedding[usable] / count[usable, None]
    variance = np.maximum(
        sum_embedding_sq[usable] / count[usable, None]
        - mean[usable] * mean[usable],
        0.0,
    )
    standard_deviation[usable] = np.sqrt(variance)
    denominator_x = np.zeros(patch_count, dtype=np.float32)
    denominator_y = np.zeros(patch_count, dtype=np.float32)
    denominator_x[usable] = (
        sum_x2[usable] - sum_x[usable] * sum_x[usable] / count[usable]
    )
    denominator_y[usable] = (
        sum_y2[usable] - sum_y[usable] * sum_y[usable] / count[usable]
    )
    valid_x = denominator_x > 0
    valid_y = denominator_y > 0
    gradient_x[valid_x] = (
        sum_x_embedding[valid_x]
        - sum_x[valid_x, None]
        * sum_embedding[valid_x]
        / count[valid_x, None]
    ) / denominator_x[valid_x, None]
    gradient_y[valid_y] = (
        sum_y_embedding[valid_y]
        - sum_y[valid_y, None]
        * sum_embedding[valid_y]
        / count[valid_y, None]
    ) / denominator_y[valid_y, None]

    identifiers = pd.DataFrame(
        {
            "row_id": targets["row_id"].to_numpy(dtype=np.int64),
            "tessera_valid_pixel_count": count,
            "tessera_centre_valid": centre_valid,
            "tessera_context_valid": count
            >= int(settings["minimum_valid_pixels_per_patch"]),
        }
    )
    arrays = {
        "centre": centre_embedding,
        "mean": mean,
        "std": standard_deviation,
        "dx": gradient_x,
        "dy": gradient_y,
    }
    feature_names: list[str] = []
    feature_arrays: list[np.ndarray] = []
    for prefix, values in arrays.items():
        feature_names.extend(
            [
                f"tessera_{prefix}_{index:03d}"
                for index in range(EMBEDDING_DIMENSIONS)
            ]
        )
        feature_arrays.append(values)
    feature_frame = pd.DataFrame(
        np.concatenate(feature_arrays, axis=1),
        columns=feature_names,
        dtype=np.float32,
    )
    output = pd.concat([identifiers, feature_frame], axis=1)
    atomic_parquet(output, TESSERA_PATH)
    valid_fraction = float(output["tessera_context_valid"].mean())
    manifest = {
        "created_utc": utc_now(),
        "target_sha256": sha256(TARGET_PATH),
        "config_sha256": config_hash,
        "registry_sha256": sha256(registry_path),
        "landmask_registry_sha256": sha256(landmask_path),
        "embedding_year": year,
        "dataset_version": version,
        "dataset_variant": variant,
        "tile_count": len(tiles),
        "patch_count": patch_count,
        "valid_context_patches": int(output["tessera_context_valid"].sum()),
        "valid_context_fraction": valid_fraction,
        "tile_records": records,
        "output_sha256": sha256(TESSERA_PATH),
    }
    atomic_json(TESSERA_MANIFEST_PATH, manifest)
    shutil.rmtree(TESSERA_CHUNK_DIR, ignore_errors=True)
    shutil.rmtree(TESSERA_STREAM_DIR, ignore_errors=True)
    print(
        f"tessera: summarized 25-pixel context for "
        f"{manifest['valid_context_patches']:,}/{patch_count:,} patches "
        f"across {len(tiles)} tiles",
        flush=True,
    )


def feature_columns(frame: pd.DataFrame, prefix: str) -> list[str]:
    expected_prefix = f"tessera_{prefix}_"
    columns = sorted(
        column
        for column in frame
        if column.startswith(expected_prefix)
        and len(column.removeprefix(expected_prefix)) == 3
        and column.removeprefix(expected_prefix).isdigit()
    )
    if len(columns) != EMBEDDING_DIMENSIONS:
        raise RuntimeError(
            f"Expected {EMBEDDING_DIMENSIONS} {prefix} features, found "
            f"{len(columns)}"
        )
    return columns


def spatial_weights(blocks: pd.Series) -> np.ndarray:
    counts = blocks.value_counts()
    weights = blocks.map(1.0 / counts).to_numpy(dtype=np.float64)
    return weights * len(weights) / weights.sum()


def fit_ridge(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray, alpha: float
) -> tuple[StandardScaler, Ridge]:
    scaler = StandardScaler().fit(x, sample_weight=weights)
    model = Ridge(alpha=alpha).fit(
        scaler.transform(x), y, sample_weight=weights
    )
    return scaler, model


def fit_nonlinear(
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
        early_stopping=False,
        random_state=int(settings["random_state"]),
    )
    return model.fit(x, y, sample_weight=weights)


def fit_mlp(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    config: dict[str, Any],
) -> tuple[StandardScaler, MLPRegressor]:
    settings = config["models"]["mlp"]
    scaler = StandardScaler().fit(x, sample_weight=weights)
    model = MLPRegressor(
        hidden_layer_sizes=tuple(settings["hidden_layer_sizes"]),
        activation="relu",
        solver="adam",
        alpha=float(settings["alpha"]),
        batch_size=int(settings["batch_size"]),
        learning_rate_init=float(settings["learning_rate_init"]),
        max_iter=int(settings["max_iter"]),
        early_stopping=bool(settings["early_stopping"]),
        validation_fraction=float(settings["validation_fraction"]),
        n_iter_no_change=int(settings["n_iter_no_change"]),
        random_state=int(settings["random_state"]),
    )
    model.fit(scaler.transform(x), y, sample_weight=weights)
    return scaler, model


def add_prediction(
    records: list[pd.DataFrame],
    metrics: list[dict[str, Any]],
    frame: pd.DataFrame,
    test: np.ndarray,
    target: str,
    model: str,
    observed: np.ndarray,
    predicted: np.ndarray,
    fold: int,
    train_rows: int,
) -> None:
    metrics.append(
        {
            "target": target,
            "model": model,
            "fold": fold,
            "train_rows": train_rows,
            "test_rows": len(observed),
            **metric_values(observed, predicted),
        }
    )
    records.append(
        pd.DataFrame(
            {
                "row_id": frame.loc[test, "row_id"].to_numpy(dtype=np.int64),
                "target": target,
                "model": model,
                "fold": fold,
                "observed": observed,
                "predicted": predicted,
                "spatial_block": frame.loc[test, "spatial_block"].to_numpy(),
                "block_x": frame.loc[test, "block_x"].to_numpy(),
                "block_y": frame.loc[test, "block_y"].to_numpy(),
            }
        )
    )


def run_fold(
    frame: pd.DataFrame,
    fold: int,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_train, base_test, diagnostics = shared.buffered_fold_masks(
        frame, fold, config
    )
    centre_features = feature_columns(frame, "centre")
    context_features = (
        feature_columns(frame, "mean")
        + feature_columns(frame, "std")
        + feature_columns(frame, "dx")
        + feature_columns(frame, "dy")
    )
    required_features = centre_features + context_features
    predictor_valid = (
        frame["tessera_context_valid"].to_numpy(dtype=bool)
        & frame["tessera_centre_valid"].to_numpy(dtype=bool)
        & np.isfinite(
            frame[required_features].to_numpy(dtype=np.float32)
        ).all(axis=1)
    )
    metrics: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    primary = str(config["targets"]["primary"])
    nonlinear_targets = set(config["targets"]["nonlinear"])
    minimum_train = int(
        config["spatial_evaluation"]["minimum_training_rows_per_target"]
    )
    minimum_test = int(
        config["spatial_evaluation"]["minimum_test_rows_per_target"]
    )

    for target in config["targets"]["representative"]:
        values = frame[target].to_numpy(dtype=np.float64)
        bounds = config["targets"]["valid_ranges"][target]
        target_valid = (
            np.isfinite(values)
            & (values >= float(bounds[0]))
            & (values <= float(bounds[1]))
        )
        train = base_train & predictor_valid & target_valid
        test = base_test & predictor_valid & target_valid
        if train.sum() < minimum_train or test.sum() < minimum_test:
            raise RuntimeError(
                f"{target} fold {fold} has {train.sum()} training and "
                f"{test.sum()} test rows"
            )
        y_train = values[train]
        y_test = values[test]
        weights = spatial_weights(frame.loc[train, "spatial_block"])
        train_rows = int(train.sum())

        model_predictions: dict[str, np.ndarray] = {
            "training_region_mean": np.full(
                len(y_test), np.average(y_train, weights=weights)
            )
        }
        coordinate_model = make_pipeline(
            PolynomialFeatures(
                degree=int(config["models"]["coordinate_polynomial_degree"]),
                include_bias=False,
            ),
            StandardScaler(),
            Ridge(alpha=float(config["models"]["ridge_alpha"])),
        )
        coordinate_model.fit(
            frame.loc[train, ["bng_x", "bng_y"]].to_numpy(dtype=np.float64),
            y_train,
            ridge__sample_weight=weights,
        )
        model_predictions["spatial_coordinate_ridge"] = (
            coordinate_model.predict(
                frame.loc[test, ["bng_x", "bng_y"]].to_numpy(
                    dtype=np.float64
                )
            )
        )

        x_centre_train = frame.loc[train, centre_features].to_numpy(
            dtype=np.float32
        )
        x_centre_test = frame.loc[test, centre_features].to_numpy(
            dtype=np.float32
        )
        x_context_train = frame.loc[train, context_features].to_numpy(
            dtype=np.float32
        )
        x_context_test = frame.loc[test, context_features].to_numpy(
            dtype=np.float32
        )
        scaler, ridge = fit_ridge(
            x_centre_train,
            y_train,
            weights,
            float(config["models"]["ridge_alpha"]),
        )
        model_predictions["tessera_centre_ridge"] = ridge.predict(
            scaler.transform(x_centre_test)
        )
        scaler, ridge = fit_ridge(
            x_context_train,
            y_train,
            weights,
            float(config["models"]["ridge_alpha"]),
        )
        model_predictions["tessera_context_ridge"] = ridge.predict(
            scaler.transform(x_context_test)
        )
        if target in nonlinear_targets:
            model_predictions["tessera_centre_hist_gradient_boosting"] = (
                fit_nonlinear(
                    x_centre_train, y_train, weights, config
                ).predict(x_centre_test)
            )
            model_predictions["tessera_context_hist_gradient_boosting"] = (
                fit_nonlinear(
                    x_context_train, y_train, weights, config
                ).predict(x_context_test)
            )
        if (
            target == primary
            and bool(config["models"]["mlp"]["enabled"])
        ):
            scaler, mlp = fit_mlp(
                x_context_train, y_train, weights, config
            )
            model_predictions["tessera_context_mlp"] = mlp.predict(
                scaler.transform(x_context_test)
            )

        for model, predicted in model_predictions.items():
            add_prediction(
                predictions,
                metrics,
                frame,
                test,
                target,
                model,
                y_test,
                predicted,
                fold,
                train_rows,
            )

    height_features = list(
        config["targets"]["height_adjustment_predictors"]
    )
    values = frame[primary].to_numpy(dtype=np.float64)
    adjusted_valid = predictor_valid & np.isfinite(
        frame[height_features].to_numpy(dtype=np.float64)
    ).all(axis=1)
    train = base_train & adjusted_valid
    test = base_test & adjusted_valid
    weights = spatial_weights(frame.loc[train, "spatial_block"])
    height_model = fit_nonlinear(
        frame.loc[train, height_features].to_numpy(dtype=np.float32),
        values[train],
        weights,
        config,
    )
    residual_train = values[train] - height_model.predict(
        frame.loc[train, height_features].to_numpy(dtype=np.float32)
    )
    residual_test = values[test] - height_model.predict(
        frame.loc[test, height_features].to_numpy(dtype=np.float32)
    )
    adjusted_target = f"{primary}_height_adjusted"
    centre_train = frame.loc[train, centre_features].to_numpy(dtype=np.float32)
    centre_test = frame.loc[test, centre_features].to_numpy(dtype=np.float32)
    context_train = frame.loc[train, context_features].to_numpy(
        dtype=np.float32
    )
    context_test = frame.loc[test, context_features].to_numpy(
        dtype=np.float32
    )
    adjusted_predictions: dict[str, np.ndarray] = {
        "height_adjusted_training_mean": np.full(
            len(residual_test), np.average(residual_train, weights=weights)
        )
    }
    scaler, ridge = fit_ridge(
        centre_train,
        residual_train,
        weights,
        float(config["models"]["ridge_alpha"]),
    )
    adjusted_predictions["height_adjusted_tessera_centre_ridge"] = (
        ridge.predict(scaler.transform(centre_test))
    )
    scaler, ridge = fit_ridge(
        context_train,
        residual_train,
        weights,
        float(config["models"]["ridge_alpha"]),
    )
    adjusted_predictions["height_adjusted_tessera_context_ridge"] = (
        ridge.predict(scaler.transform(context_test))
    )
    adjusted_predictions[
        "height_adjusted_tessera_context_hist_gradient_boosting"
    ] = fit_nonlinear(
        context_train, residual_train, weights, config
    ).predict(context_test)
    for model, predicted in adjusted_predictions.items():
        add_prediction(
            predictions,
            metrics,
            frame,
            test,
            adjusted_target,
            model,
            residual_test,
            predicted,
            fold,
            int(train.sum()),
        )

    for record in metrics:
        record.update(
            {
                f"fold_{key}": value
                for key, value in diagnostics.items()
            }
        )
    return pd.DataFrame(metrics), pd.concat(predictions, ignore_index=True)


def bootstrap_primary(
    predictions: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    primary = str(config["targets"]["primary"])
    subset = predictions[predictions["target"].eq(primary)].copy()
    replicates = int(config["uncertainty"]["block_bootstrap_replicates"])
    rng = np.random.default_rng(int(config["uncertainty"]["seed"]))
    records: list[dict[str, Any]] = []
    for model, model_frame in subset.groupby("model"):
        blocks = sorted(model_frame["spatial_block"].unique())
        grouped = {
            block: group[["observed", "predicted"]].to_numpy(dtype=np.float64)
            for block, group in model_frame.groupby("spatial_block")
        }
        draws = np.empty((replicates, 2), dtype=np.float64)
        for replicate in range(replicates):
            selected = rng.choice(blocks, size=len(blocks), replace=True)
            values = np.concatenate([grouped[block] for block in selected])
            metrics = metric_values(values[:, 0], values[:, 1])
            draws[replicate] = [metrics["rmse"], metrics["r2"]]
        records.append(
            {
                "target": primary,
                "model": model,
                "block_count": len(blocks),
                "replicates": replicates,
                "rmse_ci_low": float(np.quantile(draws[:, 0], 0.025)),
                "rmse_ci_high": float(np.quantile(draws[:, 0], 0.975)),
                "r2_ci_low": float(np.quantile(draws[:, 1], 0.025)),
                "r2_ci_high": float(np.quantile(draws[:, 1], 0.975)),
            }
        )
    return pd.DataFrame(records)


def run_evaluation(config: dict[str, Any]) -> None:
    if all(
        path.exists()
        for path in [
            METRICS_PATH,
            POOLED_METRICS_PATH,
            PREDICTIONS_PATH,
            BOOTSTRAP_PATH,
        ]
    ):
        print("evaluate: resumed complete evaluation checkpoint", flush=True)
        return
    targets = pd.read_parquet(TARGET_PATH)
    tessera = pd.read_parquet(TESSERA_PATH)
    frame = targets.merge(tessera, on="row_id", validate="one_to_one")
    FOLD_DIR.mkdir(parents=True, exist_ok=True)
    input_hashes = {
        "config_sha256": sha256(CONFIG_PATH),
        "targets_sha256": sha256(TARGET_PATH),
        "tessera_sha256": sha256(TESSERA_PATH),
    }
    fold_metrics: list[pd.DataFrame] = []
    fold_predictions: list[pd.DataFrame] = []
    fold_count = int(config["spatial_evaluation"]["region_count"])
    for fold in range(fold_count):
        metric_path = FOLD_DIR / f"fold_{fold}_metrics.csv"
        prediction_path = FOLD_DIR / f"fold_{fold}_predictions.parquet"
        manifest_path = FOLD_DIR / f"fold_{fold}_manifest.json"
        if metric_path.exists() and prediction_path.exists() and manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest["inputs"] != input_hashes
                or manifest["metrics_sha256"] != sha256(metric_path)
                or manifest["predictions_sha256"] != sha256(prediction_path)
            ):
                raise RuntimeError(f"Stale phase17 evaluation fold {fold}")
            metrics = pd.read_csv(metric_path)
            predictions = pd.read_parquet(prediction_path)
            print(f"evaluate [{fold + 1}/{fold_count}]: resumed fold {fold}", flush=True)
        else:
            print(f"evaluate [{fold + 1}/{fold_count}]: fit fold {fold}", flush=True)
            metrics, predictions = run_fold(frame, fold, config)
            atomic_csv(metrics, metric_path)
            atomic_parquet(predictions, prediction_path)
            atomic_json(
                manifest_path,
                {
                    "created_utc": utc_now(),
                    "fold": fold,
                    "inputs": input_hashes,
                    "metrics_sha256": sha256(metric_path),
                    "predictions_sha256": sha256(prediction_path),
                },
            )
        fold_metrics.append(metrics)
        fold_predictions.append(predictions)

    metrics = pd.concat(fold_metrics, ignore_index=True)
    predictions = pd.concat(fold_predictions, ignore_index=True)
    atomic_csv(metrics, METRICS_PATH)
    atomic_parquet(predictions, PREDICTIONS_PATH)
    pooled_records: list[dict[str, Any]] = []
    for (target, model), group in predictions.groupby(["target", "model"]):
        fold_rows = metrics[
            metrics["target"].eq(target) & metrics["model"].eq(model)
        ]
        pooled_records.append(
            {
                "target": target,
                "model": model,
                "test_rows": len(group),
                **metric_values(
                    group["observed"].to_numpy(),
                    group["predicted"].to_numpy(),
                ),
                **{
                    f"mean_fold_{name}": float(fold_rows[name].mean())
                    for name in METRIC_NAMES
                },
                "positive_r2_folds": int((fold_rows["r2"] > 0).sum()),
            }
        )
    pooled = pd.DataFrame(pooled_records).sort_values(
        ["target", "rmse", "model"]
    )
    atomic_csv(pooled, POOLED_METRICS_PATH)
    bootstrap = bootstrap_primary(predictions, config)
    atomic_csv(bootstrap, BOOTSTRAP_PATH)
    print(
        f"evaluate: wrote {len(predictions):,} out-of-fold predictions "
        f"for {pooled['target'].nunique()} targets",
        flush=True,
    )


def make_figure(config: dict[str, Any]) -> None:
    pooled = pd.read_csv(POOLED_METRICS_PATH)
    primary = str(config["targets"]["primary"])
    primary_rows = pooled[pooled["target"].eq(primary)].sort_values("rmse")
    context_model = "tessera_context_hist_gradient_boosting"
    target_rows = pooled[
        pooled["model"].eq(context_model)
        & pooled["target"].isin(config["targets"]["representative"])
    ].copy()
    labels = {
        "training_region_mean": "Training mean",
        "spatial_coordinate_ridge": "Coordinates",
        "tessera_centre_ridge": "Centre ridge",
        "tessera_context_ridge": "Context ridge",
        "tessera_centre_hist_gradient_boosting": "Centre nonlinear",
        "tessera_context_hist_gradient_boosting": "Context nonlinear",
        "tessera_context_mlp": "Context MLP",
    }
    target_labels = {
        "mean_canopy_shannon_50m": "Vertical entropy",
        "between_cell_height_sd_50m": "Height variation",
        "mean_gap_fraction_50m": "Gap fraction",
        "mean_within_cell_height_sd_50m": "Within-cell height variation",
    }
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    x = np.arange(len(primary_rows))
    axes[0].bar(
        x,
        primary_rows["r2"],
        color=[
            "#178F8D" if "context" in model else "#7A858C"
            for model in primary_rows["model"]
        ],
    )
    axes[0].axhline(0, color="#333333", linewidth=0.8)
    axes[0].axhline(
        float(config["success"]["primary_r2_target"]),
        color="#D95F02",
        linewidth=1,
        linestyle="--",
    )
    axes[0].set_xticks(
        x,
        [labels.get(model, model) for model in primary_rows["model"]],
        rotation=45,
        ha="right",
    )
    axes[0].set_ylabel(r"Pooled spatially withheld $R^2$")
    axes[0].set_title("a  Primary vertical-entropy target", loc="left")

    x = np.arange(len(target_rows))
    axes[1].bar(x, target_rows["r2"], color="#178F8D")
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[1].set_xticks(
        x,
        [target_labels.get(target, target) for target in target_rows["target"]],
        rotation=35,
        ha="right",
    )
    axes[1].set_ylabel(r"Pooled spatially withheld $R^2$")
    axes[1].set_title("b  Structural targets", loc="left")
    for axis in axes:
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        axis.set_axisbelow(True)
    figure.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=220, bbox_inches="tight")
    plt.close(figure)


def run_report(config: dict[str, Any]) -> None:
    if REPORT_PATH.exists() and RESULT_FREEZE_PATH.exists():
        freeze = json.loads(RESULT_FREEZE_PATH.read_text(encoding="utf-8"))
        if freeze["report_sha256"] == sha256(REPORT_PATH):
            print("report: validated complete result checkpoint", flush=True)
            return
    pooled = pd.read_csv(POOLED_METRICS_PATH)
    bootstrap = pd.read_csv(BOOTSTRAP_PATH)
    primary = str(config["targets"]["primary"])
    fixed_model = "tessera_context_hist_gradient_boosting"
    central_model = "tessera_centre_hist_gradient_boosting"
    fixed = pooled[
        pooled["target"].eq(primary) & pooled["model"].eq(fixed_model)
    ].iloc[0]
    central = pooled[
        pooled["target"].eq(primary) & pooled["model"].eq(central_model)
    ].iloc[0]
    interval = bootstrap[
        bootstrap["target"].eq(primary)
        & bootstrap["model"].eq(fixed_model)
    ].iloc[0]
    improvement = (float(central.rmse) - float(fixed.rmse)) / float(
        central.rmse
    )
    r2_gate = float(config["success"]["primary_r2_target"])
    success = (
        float(fixed.r2) >= r2_gate
        and int(fixed.positive_r2_folds)
        >= int(config["success"]["minimum_positive_fold_count"])
        and improvement
        >= float(
            config["success"][
                "minimum_rmse_reduction_over_central_fraction"
            ]
        )
    )
    lines = [
        "# Cairngorms forest structural heterogeneity results",
        "",
        "## Design",
        "",
        (
            "The experiment used non-overlapping 50 m mature-forest units from "
            "the 2023 Cairngorms airborne-LiDAR survey. Five complete spatial "
            "regions were withheld in turn, with a 1 km exclusion buffer."
        ),
        "",
        "## Primary result",
        "",
        (
            f"The predeclared contextual nonlinear TESSERA model obtained "
            f"RMSE {fixed.rmse:.3f}, R2 {fixed.r2:.3f}, and Spearman "
            f"r {fixed.spearman_r:.3f}. Its block-bootstrap R2 interval was "
            f"{interval.r2_ci_low:.3f} to {interval.r2_ci_high:.3f}."
        ),
        (
            f"The corresponding central-pixel model obtained RMSE "
            f"{central.rmse:.3f} and R2 {central.r2:.3f}. Spatial context "
            f"changed RMSE by {improvement * 100:.1f}%."
        ),
        "",
        "## Predeclared usefulness gate",
        "",
        f"Overall gate passed: **{'yes' if success else 'no'}**.",
        "",
        "## Scope",
        "",
        (
            "These results test prediction across withheld parts of one "
            "Scottish landscape. The LiDAR metrics are structural reference "
            "measurements and are not direct biodiversity observations."
        ),
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    make_figure(config)
    inputs = {
        str(path.relative_to(ROOT)): sha256(path)
        for path in [
            CONFIG_PATH,
            PROTOCOL_PATH,
            TARGET_PATH,
            TESSERA_PATH,
            METRICS_PATH,
            POOLED_METRICS_PATH,
            PREDICTIONS_PATH,
            BOOTSTRAP_PATH,
            FIGURE_PATH,
        ]
    }
    freeze = {
        "created_utc": utc_now(),
        "status": "complete",
        "study": config["study"]["name"],
        "primary_model": fixed_model,
        "primary_r2": float(fixed.r2),
        "success_gate_passed": bool(success),
        "inputs_and_outputs": inputs,
        "report_sha256": sha256(REPORT_PATH),
    }
    atomic_json(RESULT_FREEZE_PATH, freeze)
    shutil.rmtree(FOLD_DIR, ignore_errors=True)
    print(
        f"report: primary R2={fixed.r2:.3f}; "
        f"success gate={'passed' if success else 'not passed'}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--through",
        choices=STAGES,
        default="report",
        help="Run resumable stages through the selected stage.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config()
    runners = {
        "inventory": run_inventory,
        "targets": run_targets,
        "tessera": run_tessera,
        "evaluate": run_evaluation,
        "report": run_report,
    }
    for stage in STAGES:
        print(f"\n=== phase17 {stage} ===", flush=True)
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
