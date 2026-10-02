#!/usr/bin/env python3
"""Build the resumable dense Cairngorms LiDAR/TESSERA analysis arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import yaml
from pyproj import Transformer
from rasterio.transform import rowcol
from scipy.ndimage import binary_dilation, convolve, distance_transform_edt


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/cairngorms_dense_10m.yaml"
PROTOCOL_PATH = ROOT / "metadata/cairngorms_dense_10m_protocol_freeze.yaml"
STATUS_PATH = ROOT / "metadata/cairngorms_dense_10m_preparation_status.json"
INPUT_FREEZE_PATH = ROOT / "metadata/cairngorms_dense_10m_input_freeze.json"
S3_HTTPS_ROOT = "https://s3.us-west-2.amazonaws.com/tessera-embeddings"


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
        for chunk in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "cairngorms_dense_10m"
    ]


def dense_dir(config: dict[str, Any]) -> Path:
    return ROOT / str(config["outputs"]["dense_directory"])


def output_paths(config: dict[str, Any]) -> dict[str, Path]:
    directory = dense_dir(config)
    return {
        "targets": directory / "targets.npy",
        "lidar_eligible": directory / "lidar_eligible.npy",
        "embedding": directory / "embedding_int8.npy",
        "scale": directory / "embedding_scale.npy",
        "tessera_valid": directory / "tessera_valid.npy",
        "row": directory / "row.npy",
        "column": directory / "column.npy",
        "region": directory / "spatial_region.npy",
        "block": directory / "spatial_block.npy",
        "target_rows": directory / "target_rows.npy",
        "index_grid": directory / "index_grid.npy",
        "manifest": directory / "manifest.json",
        "tile_markers": directory / "tile_markers",
        "tile_stream": directory / "tile_stream",
        "folds": directory / "folds",
    }


def update_status(stage: str, message: str, **values: Any) -> None:
    existing: dict[str, Any] = {}
    if STATUS_PATH.exists():
        existing = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    existing.update(
        {
            "updated_utc": utc_now(),
            "stage": stage,
            "message": message,
            **values,
        }
    )
    atomic_json(STATUS_PATH, existing)


def freeze_protocol(config: dict[str, Any]) -> None:
    PROTOCOL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not PROTOCOL_PATH.exists():
        PROTOCOL_PATH.write_bytes(CONFIG_PATH.read_bytes())
    if PROTOCOL_PATH.read_bytes() != CONFIG_PATH.read_bytes():
        raise RuntimeError("Dense configuration differs from its frozen protocol")

    source = ROOT / str(config["source"]["lidar_metrics_path"])
    if not source.exists():
        raise RuntimeError(f"Missing LiDAR raster: {source}")
    observed_hash = sha256(source)
    if observed_hash != str(config["source"]["expected_source_sha256"]):
        raise RuntimeError("The Cairngorms LiDAR source hash changed")

    with rasterio.open(source) as dataset:
        expected = config["source"]
        if dataset.count != int(expected["expected_band_count"]):
            raise RuntimeError("Unexpected LiDAR band count")
        if list(dataset.shape) != [int(v) for v in expected["expected_shape"]]:
            raise RuntimeError("Unexpected LiDAR raster shape")
        if not np.allclose(dataset.res, [float(expected["expected_resolution_m"])] * 2):
            raise RuntimeError("Unexpected LiDAR resolution")
        if not np.allclose(dataset.bounds, expected["expected_bounds_bng"], atol=0.01):
            raise RuntimeError("Unexpected LiDAR bounds")
        required = {
            "lidar_meanH",
            "lidar_p_95",
            "lidar_Cov",
            "lidar_canopy_shannon",
            "lidar_gapFrac",
            *config["targets"]["profile_source_bands"],
        }
        missing = sorted(required - set(dataset.descriptions))
        if missing:
            raise RuntimeError(f"LiDAR raster lacks bands: {', '.join(missing)}")

    freeze = {
        "created_utc": utc_now(),
        "config_sha256": sha256(CONFIG_PATH),
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "lidar_path": str(source.relative_to(ROOT)),
        "lidar_sha256": observed_hash,
        "lidar_size_bytes": source.stat().st_size,
        "tessera_manifest_path": str(config["source"]["tessera_tile_manifest"]),
        "tessera_manifest_sha256": sha256(
            ROOT / str(config["source"]["tessera_tile_manifest"])
        ),
    }
    atomic_json(INPUT_FREEZE_PATH, freeze)
    update_status("preflight", "Input hashes and raster geometry validated")


def moving_mean_and_count(values: np.ndarray, cells: int) -> tuple[np.ndarray, np.ndarray]:
    kernel = np.ones((cells, cells), dtype=np.float32)
    valid = np.isfinite(values)
    count = convolve(valid.astype(np.int16), kernel, mode="constant", cval=0)
    total = convolve(np.where(valid, values, 0.0), kernel, mode="constant", cval=0.0)
    mean = np.full(values.shape, np.nan, dtype=np.float32)
    usable = count > 0
    mean[usable] = total[usable] / count[usable]
    return mean, count.astype(np.int16)


def moving_std(values: np.ndarray, cells: int) -> np.ndarray:
    kernel = np.ones((cells, cells), dtype=np.float32)
    valid = np.isfinite(values)
    count = convolve(valid.astype(np.int16), kernel, mode="constant", cval=0)
    total = convolve(np.where(valid, values, 0.0), kernel, mode="constant", cval=0.0)
    total_square = convolve(
        np.where(valid, values * values, 0.0), kernel, mode="constant", cval=0.0
    )
    result = np.full(values.shape, np.nan, dtype=np.float32)
    usable = count > 1
    variance = np.maximum(
        total_square[usable] / count[usable]
        - np.square(total[usable] / count[usable]),
        0.0,
    )
    result[usable] = np.sqrt(variance)
    return result


def read_band(dataset: rasterio.DatasetReader, lookup: dict[str, int], name: str) -> np.ndarray:
    values = dataset.read(lookup[name]).astype(np.float32)
    values[~np.isfinite(values)] = np.nan
    return values


def prepare_lidar_targets(config: dict[str, Any]) -> None:
    paths = output_paths(config)
    if paths["targets"].exists() and paths["lidar_eligible"].exists():
        update_status("lidar", "Resumed completed dense LiDAR targets")
        print("lidar: resumed completed targets", flush=True)
        return

    directory = dense_dir(config)
    directory.mkdir(parents=True, exist_ok=True)
    source = ROOT / str(config["source"]["lidar_metrics_path"])
    cells = int(config["population"]["structural_window_cells"])
    target_count = len(config["targets"]["names"])

    with rasterio.open(source) as dataset:
        shape = dataset.shape
        lookup = {str(name): i for i, name in enumerate(dataset.descriptions, 1)}
        targets = np.lib.format.open_memmap(
            paths["targets"], mode="w+", dtype=np.float32, shape=(target_count, *shape)
        )
        targets[:] = np.nan
        eligible = np.ones(shape, dtype=bool)

        shannon = read_band(dataset, lookup, "lidar_canopy_shannon")
        shannon_mean, shannon_count = moving_mean_and_count(shannon, cells)
        targets[0] = shannon_mean
        eligible &= shannon_count >= int(config["population"]["minimum_valid_lidar_cells"])
        del shannon, shannon_mean, shannon_count

        mean_height = read_band(dataset, lookup, "lidar_meanH")
        rolling_height, _ = moving_mean_and_count(mean_height, cells)
        targets[1] = moving_std(mean_height, cells)
        eligible &= np.isfinite(rolling_height)
        eligible &= rolling_height >= float(config["population"]["minimum_mean_canopy_height_m"])
        eligible &= rolling_height <= float(config["population"]["maximum_mean_canopy_height_m"])
        del mean_height, rolling_height

        p95 = read_band(dataset, lookup, "lidar_p_95")
        rolling_p95, _ = moving_mean_and_count(p95, cells)
        eligible &= np.isfinite(rolling_p95)
        eligible &= rolling_p95 <= float(config["population"]["maximum_mean_p95_height_m"])
        del p95, rolling_p95

        cover = read_band(dataset, lookup, "lidar_Cov")
        rolling_cover, _ = moving_mean_and_count(cover, cells)
        eligible &= np.isfinite(rolling_cover)
        eligible &= rolling_cover >= float(config["population"]["minimum_mean_canopy_cover"])
        del cover, rolling_cover

        gap = read_band(dataset, lookup, "lidar_gapFrac")
        rolling_gap, _ = moving_mean_and_count(gap, cells)
        targets[2] = rolling_gap
        eligible &= np.isfinite(rolling_gap)
        del gap, rolling_gap

        volume_means: list[np.ndarray] = []
        for name in config["targets"]["profile_source_bands"]:
            values = read_band(dataset, lookup, str(name))
            mean, _ = moving_mean_and_count(values, cells)
            volume_means.append(mean)
            del values
        volumes = np.stack(volume_means, axis=0)
        profile_valid = np.isfinite(volumes).all(axis=0) & (volumes >= 0).all(axis=0)
        total = np.where(profile_valid, volumes.sum(axis=0), np.nan)
        profile_valid &= total > float(config["targets"]["profile_minimum_total_volume"])
        proportions = np.divide(
            volumes,
            total[None, :, :],
            out=np.zeros_like(volumes),
            where=np.isfinite(total)[None, :, :] & (total[None, :, :] > 0),
        )
        entropy = -np.where(
            proportions > 0,
            proportions * np.log(np.maximum(proportions, 1e-12)),
            0.0,
        ).sum(axis=0)
        entropy[~profile_valid] = np.nan
        targets[3] = entropy.astype(np.float32)
        eligible &= profile_valid
        del volumes, volume_means, total, proportions, entropy, profile_valid

        half = cells // 2
        eligible[:half] = False
        eligible[-half:] = False
        eligible[:, :half] = False
        eligible[:, -half:] = False
        for index, name in enumerate(config["targets"]["names"]):
            lower, upper = config["targets"]["valid_ranges"][name]
            values = np.asarray(targets[index])
            eligible &= np.isfinite(values) & (values >= lower) & (values <= upper)
        targets.flush()
        del targets

    eligible_array = np.lib.format.open_memmap(
        paths["lidar_eligible"], mode="w+", dtype=np.uint8, shape=eligible.shape
    )
    eligible_array[:] = eligible.astype(np.uint8)
    eligible_array.flush()
    count = int(eligible.sum())
    update_status("lidar", f"Prepared {count:,} LiDAR-eligible sliding windows", lidar_eligible=count)
    print(f"lidar: retained {count:,} sliding 50 m windows at 10 m stride", flush=True)


def s3_url(source: str) -> str:
    prefix = "s3://tessera-embeddings/"
    if not source.startswith(prefix):
        raise RuntimeError(f"Unexpected TESSERA source: {source}")
    return f"{S3_HTTPS_ROOT}/{source[len(prefix):]}"


def download_checked(url: str, path: Path, expected_size: int, attempts: int, delay: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size == expected_size:
        return
    path.unlink(missing_ok=True)
    for attempt in range(1, attempts + 1):
        temporary = path.with_suffix(path.suffix + ".part")
        temporary.unlink(missing_ok=True)
        try:
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
            if temporary.stat().st_size != expected_size:
                raise RuntimeError(
                    f"Downloaded {temporary.stat().st_size} bytes; expected {expected_size}"
                )
            temporary.replace(path)
            return
        except Exception as error:
            temporary.unlink(missing_ok=True)
            if attempt == attempts:
                raise RuntimeError(f"Failed to download {url}") from error
            print(f"download retry {attempt + 1}/{attempts} for {path.name}: {error}", flush=True)
            time.sleep(delay)


def tile_name_from_lonlat(longitude: np.ndarray, latitude: np.ndarray) -> np.ndarray:
    tile_lon = np.round(np.floor(longitude * 10.0) / 10.0 + 0.05, 2)
    tile_lat = np.round(np.floor(latitude * 10.0) / 10.0 + 0.05, 2)
    return np.asarray(
        [f"grid_{lon:.2f}_{lat:.2f}" for lon, lat in zip(tile_lon, tile_lat)],
        dtype="U32",
    )


def align_tessera(config: dict[str, Any]) -> None:
    paths = output_paths(config)
    completion = dense_dir(config) / "tessera_alignment_complete.json"
    if completion.exists():
        update_status("tessera", "Resumed completed TESSERA alignment")
        print("tessera: resumed completed alignment", flush=True)
        return

    lidar_eligible = np.load(paths["lidar_eligible"], mmap_mode="r").astype(bool)
    cells = int(config["tessera"]["patch_cells"])
    needed = binary_dilation(lidar_eligible, structure=np.ones((cells, cells), dtype=bool))
    needed_rows, needed_columns = np.nonzero(needed)
    source_path = ROOT / str(config["source"]["lidar_metrics_path"])
    with rasterio.open(source_path) as dataset:
        transform = dataset.transform
        shape = dataset.shape

    bng_x = transform.c + (needed_columns + 0.5) * transform.a
    bng_y = transform.f + (needed_rows + 0.5) * transform.e
    to_wgs84 = Transformer.from_crs(config["study"]["analysis_crs"], "EPSG:4326", always_xy=True)
    longitude, latitude = to_wgs84.transform(bng_x, bng_y)
    names = tile_name_from_lonlat(np.asarray(longitude), np.asarray(latitude))

    manifest_path = ROOT / str(config["source"]["tessera_tile_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest["embedding_year"]) != int(config["tessera"]["embedding_year"]):
        raise RuntimeError("TESSERA year differs from the frozen tile manifest")
    if str(manifest["dataset_version"]) != str(config["tessera"]["dataset_version"]):
        raise RuntimeError("TESSERA version differs from the frozen tile manifest")
    if str(manifest["dataset_variant"]) != str(config["tessera"]["dataset_variant"]):
        raise RuntimeError("TESSERA variant differs from the frozen tile manifest")
    records = {str(record["tile"]): record for record in manifest["tile_records"]}
    required_tiles = sorted(np.unique(names).tolist())
    missing = sorted(set(required_tiles) - set(records))
    if missing:
        raise RuntimeError(f"Frozen manifest lacks dense-grid tiles: {', '.join(missing)}")

    dimensions = int(config["tessera"]["embedding_dimensions"])
    if not paths["embedding"].exists():
        embedding = np.lib.format.open_memmap(
            paths["embedding"], mode="w+", dtype=np.int8, shape=(*shape, dimensions)
        )
        embedding[:] = 0
        embedding.flush()
        del embedding
    if not paths["scale"].exists():
        scale = np.lib.format.open_memmap(paths["scale"], mode="w+", dtype=np.float16, shape=shape)
        scale[:] = 0
        scale.flush()
        del scale
    if not paths["tessera_valid"].exists():
        valid = np.lib.format.open_memmap(
            paths["tessera_valid"], mode="w+", dtype=np.uint8, shape=shape
        )
        valid[:] = 0
        valid.flush()
        del valid

    embedding_output = np.load(paths["embedding"], mmap_mode="r+")
    scale_output = np.load(paths["scale"], mmap_mode="r+")
    valid_output = np.load(paths["tessera_valid"], mmap_mode="r+")
    markers = paths["tile_markers"]
    markers.mkdir(parents=True, exist_ok=True)
    stream = paths["tile_stream"]
    stream.mkdir(parents=True, exist_ok=True)
    attempts = int(config["tessera"]["download_attempts"])
    delay = int(config["tessera"]["retry_delay_seconds"])

    for tile_index, name in enumerate(required_tiles, start=1):
        marker = markers / f"{name}.json"
        positions = np.flatnonzero(names == name)
        if marker.exists():
            saved = json.loads(marker.read_text(encoding="utf-8"))
            if int(saved["source_points"]) != len(positions):
                raise RuntimeError(f"Stale marker for {name}")
            print(f"tessera [{tile_index}/{len(required_tiles)}]: resumed {name}", flush=True)
            continue

        record = records[name]
        sources = {
            "embedding": str(record["embedding_source"]),
            "scales": str(record["scales_source"]),
            "landmask": str(record["landmask_source"]),
        }
        sizes = {key: int(record["source_sizes"][key]) for key in sources}
        tile_dir = stream / name
        local = {
            "embedding": tile_dir / f"{name}.npy",
            "scales": tile_dir / f"{name}_scales.npy",
            "landmask": tile_dir / f"{name}.tiff",
        }
        print(
            f"tessera [{tile_index}/{len(required_tiles)}]: acquire {name} for {len(positions):,} grid cells",
            flush=True,
        )
        for kind in ["embedding", "scales", "landmask"]:
            download_checked(s3_url(sources[kind]), local[kind], sizes[kind], attempts, delay)

        quantized = np.load(local["embedding"], mmap_mode="r")
        scales = np.load(local["scales"], mmap_mode="r")
        with rasterio.open(local["landmask"]) as mask_dataset:
            landmask = mask_dataset.read(1)
            transformer = Transformer.from_crs(
                config["study"]["analysis_crs"], mask_dataset.crs, always_xy=True
            )
            x, y = transformer.transform(bng_x[positions], bng_y[positions])
            source_rows, source_columns = rowcol(mask_dataset.transform, x, y)
        source_rows = np.asarray(source_rows, dtype=np.int64)
        source_columns = np.asarray(source_columns, dtype=np.int64)
        inside = (
            (source_rows >= 0)
            & (source_columns >= 0)
            & (source_rows < quantized.shape[0])
            & (source_columns < quantized.shape[1])
        )
        local_positions = np.flatnonzero(inside)
        source_scale = scales[source_rows[local_positions], source_columns[local_positions]].astype(np.float32)
        source_land = landmask[source_rows[local_positions], source_columns[local_positions]]
        usable = np.isfinite(source_scale) & (source_scale > 0) & (source_land > 0)
        local_positions = local_positions[usable]
        destination_positions = positions[local_positions]
        destination_rows = needed_rows[destination_positions]
        destination_columns = needed_columns[destination_positions]
        q = quantized[
            source_rows[local_positions], source_columns[local_positions]
        ].astype(np.int8)
        finite = np.isfinite(q).all(axis=1)
        destination_rows = destination_rows[finite]
        destination_columns = destination_columns[finite]
        local_positions = local_positions[finite]
        embedding_output[destination_rows, destination_columns] = q[finite]
        scale_output[destination_rows, destination_columns] = scales[
            source_rows[local_positions], source_columns[local_positions]
        ].astype(np.float16)
        valid_output[destination_rows, destination_columns] = 1
        embedding_output.flush()
        scale_output.flush()
        valid_output.flush()
        atomic_json(
            marker,
            {
                "created_utc": utc_now(),
                "tile": name,
                "source_points": len(positions),
                "valid_points": int(len(destination_rows)),
                "sources": sources,
                "source_sizes": sizes,
            },
        )
        if bool(config["tessera"]["delete_source_tiles_after_alignment"]):
            shutil.rmtree(tile_dir, ignore_errors=True)
        update_status(
            "tessera",
            f"Aligned tile {tile_index}/{len(required_tiles)}",
            completed_tiles=tile_index,
            total_tiles=len(required_tiles),
        )

    embedding_output.flush()
    scale_output.flush()
    valid_output.flush()
    valid_count = convolve(
        np.asarray(valid_output, dtype=np.int16),
        np.ones((cells, cells), dtype=np.int16),
        mode="constant",
        cval=0,
    )
    final_eligible = lidar_eligible & (
        valid_count >= int(config["population"]["minimum_valid_tessera_pixels"])
    )
    final_eligible &= np.asarray(valid_output, dtype=bool)
    half = cells // 2
    final_eligible[:half] = False
    final_eligible[-half:] = False
    final_eligible[:, :half] = False
    final_eligible[:, -half:] = False
    build_indices_and_folds(config, final_eligible, transform)
    del embedding_output, scale_output, valid_output
    shutil.rmtree(stream, ignore_errors=True)
    atomic_json(
        completion,
        {
            "created_utc": utc_now(),
            "config_sha256": sha256(CONFIG_PATH),
            "required_tiles": required_tiles,
            "lidar_eligible": int(lidar_eligible.sum()),
            "final_eligible": int(final_eligible.sum()),
        },
    )
    update_status(
        "complete",
        f"Dense arrays ready for {int(final_eligible.sum()):,} locations",
        lidar_eligible=int(lidar_eligible.sum()),
        final_eligible=int(final_eligible.sum()),
        completed_tiles=len(required_tiles),
        total_tiles=len(required_tiles),
    )
    print(f"tessera: retained {int(final_eligible.sum()):,} final locations", flush=True)


def weighted_quantiles(values: np.ndarray, weights: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ordered_values = values[order]
    ordered_weights = weights[order]
    cumulative = np.cumsum(ordered_weights) - 0.5 * ordered_weights
    cumulative /= ordered_weights.sum()
    return np.interp(probabilities, cumulative, ordered_values)


def training_statistics(
    paths: dict[str, Path],
    rows: np.ndarray,
    columns: np.ndarray,
    training_indices: np.ndarray,
    target_rows: np.ndarray,
    cells: int,
    batch_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    embedding = np.load(paths["embedding"], mmap_mode="r")
    scale = np.load(paths["scale"], mmap_mode="r")
    valid = np.load(paths["tessera_valid"], mmap_mode="r")
    dimensions = embedding.shape[2]
    total = np.zeros(dimensions, dtype=np.float64)
    total_square = np.zeros(dimensions, dtype=np.float64)
    count = 0
    half = cells // 2
    for start in range(0, len(training_indices), batch_size):
        selected = training_indices[start : start + batch_size]
        selected_rows = rows[selected]
        selected_columns = columns[selected]
        for row_offset in range(-half, half + 1):
            for column_offset in range(-half, half + 1):
                local_valid = valid[
                    selected_rows + row_offset,
                    selected_columns + column_offset,
                ].astype(bool)
                if not local_valid.any():
                    continue
                values = embedding[
                    selected_rows[local_valid] + row_offset,
                    selected_columns[local_valid] + column_offset,
                ].astype(np.float32)
                values *= scale[
                    selected_rows[local_valid] + row_offset,
                    selected_columns[local_valid] + column_offset,
                ].astype(np.float32)[:, None]
                total += values.sum(axis=0, dtype=np.float64)
                total_square += np.square(values).sum(axis=0, dtype=np.float64)
                count += len(values)
    if count == 0:
        raise RuntimeError("No valid training embeddings were available")
    input_mean = total / count
    input_variance = np.maximum(total_square / count - np.square(input_mean), 1e-12)
    selected_targets = target_rows[training_indices].astype(np.float64)
    target_mean = selected_targets.mean(axis=0)
    target_sd = selected_targets.std(axis=0)
    target_sd[target_sd < 1e-6] = 1.0
    return (
        input_mean.astype(np.float32),
        np.sqrt(input_variance).astype(np.float32),
        target_mean.astype(np.float32),
        target_sd.astype(np.float32),
    )


def build_indices_and_folds(
    config: dict[str, Any], eligible: np.ndarray, transform: rasterio.Affine
) -> None:
    paths = output_paths(config)
    rows, columns = np.nonzero(eligible)
    x = transform.c + (columns + 0.5) * transform.a
    y = transform.f + (rows + 0.5) * transform.e
    block_size = int(config["spatial_evaluation"]["block_size_m"])
    block_x = np.floor(x / block_size).astype(np.int32)
    block_y = np.floor(y / block_size).astype(np.int32)
    block_pairs = np.column_stack([block_x, block_y])
    unique_blocks, block_inverse, block_counts = np.unique(
        block_pairs, axis=0, return_inverse=True, return_counts=True
    )
    centres = (unique_blocks.astype(np.float64) + 0.5) * block_size
    centred = centres - np.average(centres, axis=0, weights=block_counts)
    covariance = np.cov(centred.T, aweights=block_counts)
    _, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, -1]
    if axis[0] < 0:
        axis *= -1
    score = centred @ axis
    region_count = int(config["spatial_evaluation"]["region_count"])
    boundaries = weighted_quantiles(
        score, block_counts.astype(np.float64), np.arange(1, region_count) / region_count
    )
    block_region = np.searchsorted(boundaries, score, side="right").astype(np.int8)
    region = block_region[block_inverse]
    targets = np.load(paths["targets"], mmap_mode="r")
    target_rows = np.column_stack([targets[i, rows, columns] for i in range(targets.shape[0])])

    arrays = {
        "row": rows.astype(np.int32),
        "column": columns.astype(np.int32),
        "region": region,
        "block": block_inverse.astype(np.int32),
        "target_rows": target_rows.astype(np.float32),
    }
    for key, values in arrays.items():
        path = paths[key]
        temporary = path.with_suffix(".tmp.npy")
        np.save(temporary, values)
        temporary.replace(path)

    index_grid = np.lib.format.open_memmap(
        paths["index_grid"], mode="w+", dtype=np.int32, shape=eligible.shape
    )
    index_grid[:] = -1
    index_grid[rows, columns] = np.arange(len(rows), dtype=np.int32)
    index_grid.flush()
    del index_grid

    fold_dir = paths["folds"]
    fold_dir.mkdir(parents=True, exist_ok=True)
    full_grid = np.zeros(eligible.shape, dtype=np.int8)
    full_grid[rows, columns] = region + 1
    validation_fraction = float(config["spatial_evaluation"]["validation_block_fraction"])
    buffer_m = float(config["spatial_evaluation"]["exclusion_buffer_m"])
    resolution = float(config["population"]["output_stride_m"])
    seed = int(config["spatial_evaluation"]["validation_seed"])
    diagnostics: list[dict[str, Any]] = []
    all_indices = np.arange(len(rows), dtype=np.int64)
    for fold in range(region_count):
        test = region == fold
        test_grid = full_grid == fold + 1
        distance = distance_transform_edt(~test_grid, sampling=resolution)
        train = (~test) & (distance[rows, columns] >= buffer_m)
        train_blocks = np.unique(block_inverse[train])
        rng = np.random.default_rng(seed + fold)
        validation_count = max(1, int(round(len(train_blocks) * validation_fraction)))
        validation_blocks = rng.choice(train_blocks, validation_count, replace=False)
        validation = train & np.isin(block_inverse, validation_blocks)
        subtrain = train & ~validation
        input_mean, input_sd, target_mean, target_sd = training_statistics(
            paths,
            rows,
            columns,
            all_indices[subtrain],
            target_rows,
            int(config["tessera"]["patch_cells"]),
        )
        path = fold_dir / f"fold_{fold}.npz"
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            train_indices=all_indices[train],
            subtrain_indices=all_indices[subtrain],
            validation_indices=all_indices[validation],
            test_indices=all_indices[test],
            input_mean=input_mean,
            input_sd=input_sd,
            target_mean=target_mean,
            target_sd=target_sd,
        )
        temporary.replace(path)
        diagnostics.append(
            {
                "fold": fold,
                "train_rows": int(train.sum()),
                "subtrain_rows": int(subtrain.sum()),
                "validation_rows": int(validation.sum()),
                "test_rows": int(test.sum()),
                "minimum_training_distance_m": float(distance[rows[train], columns[train]].min()),
            }
        )
    atomic_json(
        paths["manifest"],
        {
            "created_utc": utc_now(),
            "config_sha256": sha256(CONFIG_PATH),
            "shape": list(eligible.shape),
            "target_names": config["targets"]["names"],
            "eligible_rows": len(rows),
            "spatial_blocks": len(unique_blocks),
            "spatial_regions": {
                str(i): int((region == i).sum()) for i in range(region_count)
            },
            "folds": diagnostics,
            "embedding_shape": [*eligible.shape, int(config["tessera"]["embedding_dimensions"])],
            "target_shape": [len(config["targets"]["names"]), *eligible.shape],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=["all", "preflight", "lidar", "tessera"], default="all"
    )
    args = parser.parse_args()
    config = load_config()
    freeze_protocol(config)
    if args.stage == "preflight":
        return
    if args.stage in {"all", "lidar"}:
        prepare_lidar_targets(config)
    if args.stage in {"all", "tessera"}:
        if not output_paths(config)["lidar_eligible"].exists():
            raise RuntimeError("Run the LiDAR stage before TESSERA alignment")
        align_tessera(config)


if __name__ == "__main__":
    main()
