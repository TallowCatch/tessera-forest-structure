#!/usr/bin/env python3
"""Run the frozen, resumable Cairngorms representation diagnostics."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import shutil
import subprocess
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
from sklearn.ensemble import HistGradientBoostingRegressor


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_conventional_predictors as conventional  # noqa: E402
import run_cairngorms_heterogeneity as phase17  # noqa: E402
import run_cairngorms_spatial_heads as phase18  # noqa: E402


CONFIG_PATH = ROOT / "configs/cairngorms_diagnostics.yaml"
PROTOCOL_PATH = (
    ROOT
    / "metadata/project_config_phase19_cairngorms_diagnostics_protocol_freeze.yaml"
)
PREFLIGHT_PATH = ROOT / "metadata/phase19_cairngorms_preflight.json"
PROFILE_PATH = (
    ROOT / "data/processed/phase19_cairngorms_vertical_profiles.parquet"
)
PROFILE_MANIFEST_PATH = (
    ROOT / "metadata/phase19_cairngorms_vertical_profile_manifest.json"
)
PATCH_DIR = ROOT / "data/interim/phase19_cairngorms_patch9"
PATCH_TILE_DIR = ROOT / "data/interim/phase19_cairngorms_patch9_tiles"
PATCH_STREAM_DIR = ROOT / "data/interim/phase19_cairngorms_patch9_stream"
PATCH_QUANTIZED_PATH = PATCH_DIR / "patch_quantized.npy"
PATCH_SCALES_PATH = PATCH_DIR / "patch_scales.npy"
PATCH_VALID_PATH = PATCH_DIR / "patch_valid.npy"
PATCH_MANIFEST_PATH = ROOT / "metadata/phase19_cairngorms_patch9_manifest.json"
SUMMARY_DIR = ROOT / "data/interim/phase19_cairngorms_context_summaries"
SUMMARY_MANIFEST_PATH = (
    ROOT / "metadata/phase19_cairngorms_context_manifest.json"
)
CONVENTIONAL_CHUNK_DIR = (
    ROOT / "data/interim/phase19_cairngorms_conventional_chunks"
)
CONVENTIONAL_PATH = (
    ROOT / "data/processed/phase19_cairngorms_conventional_context.parquet"
)
CONVENTIONAL_MANIFEST_PATH = (
    ROOT / "metadata/phase19_cairngorms_conventional_manifest.json"
)
CONVENTIONAL_INVENTORY_PATH = (
    ROOT / "metadata/phase19_cairngorms_conventional_inventory.csv"
)
ARRAY_DIR = ROOT / "data/interim/phase19_cairngorms_arrays"
ARRAY_MANIFEST_PATH = ARRAY_DIR / "manifest.json"
FOLD_DIR = ROOT / "data/interim/phase19_cairngorms_evaluation_folds"
NEURAL_RESULT_DIR = (
    ROOT / "data/interim/phase19_cairngorms_neural_results"
)
NEURAL_WORKER_PATH = ROOT / "scripts/cairngorms_diagnostics_neural_worker.py"
METRICS_PATH = ROOT / "outputs/tables/phase19_cairngorms_metrics.csv"
POOLED_METRICS_PATH = (
    ROOT / "outputs/tables/phase19_cairngorms_pooled_metrics.csv"
)
COMPARISON_PATH = (
    ROOT / "outputs/tables/phase19_cairngorms_paired_comparisons.csv"
)
PREDICTIONS_PATH = (
    ROOT / "data/processed/phase19_cairngorms_oof_predictions.parquet"
)
FIGURE_PATH = (
    ROOT / "outputs/figures/phase19_cairngorms_diagnostics.png"
)
REPORT_PATH = (
    ROOT / "outputs/reports/phase19_cairngorms_diagnostics_results.md"
)
RESULT_FREEZE_PATH = (
    ROOT / "metadata/phase19_cairngorms_result_freeze.json"
)

STAGES = [
    "preflight",
    "profiles",
    "patches",
    "summaries",
    "conventional",
    "prepare",
    "evaluate",
    "report",
]
METRIC_NAMES = ["rmse", "mae", "r2", "pearson_r", "spearman_r", "bias"]


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "phase19_cairngorms_diagnostics"
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def input_path(config: dict[str, Any], key: str) -> Path:
    return ROOT / str(config["frozen_inputs"][key])


def raw_target_names(config: dict[str, Any]) -> list[str]:
    return [str(value) for value in config["targets"]["raw"]]


def target_names(config: dict[str, Any]) -> list[str]:
    return raw_target_names(config) + [str(config["targets"]["adjusted_name"])]


def context_points(
    targets: pd.DataFrame, cells: int, resolution_m: float = 10.0
) -> dict[str, np.ndarray]:
    if cells % 2 != 1:
        raise ValueError("Context size must be odd")
    half = cells // 2
    offsets_row = np.repeat(np.arange(-half, half + 1), cells)
    offsets_column = np.tile(np.arange(-half, half + 1), cells)
    count = cells * cells
    return {
        "row_position": np.repeat(
            np.arange(len(targets), dtype=np.int64), count
        ),
        "row_id": np.repeat(
            targets["row_id"].to_numpy(dtype=np.int64), count
        ),
        "bng_x": (
            np.repeat(targets["bng_x"].to_numpy(dtype=np.float64), count)
            + np.tile(offsets_column, len(targets)) * resolution_m
        ),
        "bng_y": (
            np.repeat(targets["bng_y"].to_numpy(dtype=np.float64), count)
            - np.tile(offsets_row, len(targets)) * resolution_m
        ),
        "local_row": np.tile(
            offsets_row + half, len(targets)
        ).astype(np.int16),
        "local_column": np.tile(
            offsets_column + half, len(targets)
        ).astype(np.int16),
        "offset_x": np.tile(offsets_column, len(targets)).astype(np.float32),
        "offset_y": np.tile(-offsets_row, len(targets)).astype(np.float32),
    }


def parse_tile_pair(pair: np.ndarray) -> tuple[float, float]:
    return round(float(pair[0]) / 100.0, 2), round(float(pair[1]) / 100.0, 2)


def model_spec(config: dict[str, Any], name: str) -> dict[str, str]:
    matches = [
        value for value in config["model_specs"] if value["name"] == name
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Unknown or duplicated model specification: {name}")
    return {str(key): str(value) for key, value in matches[0].items()}


def discard_completed_patch_arrays() -> int:
    removed = 0
    for path in [
        PATCH_QUANTIZED_PATH,
        PATCH_SCALES_PATH,
        PATCH_VALID_PATH,
    ]:
        if path.exists():
            removed += path.stat().st_size
            path.unlink()
    try:
        PATCH_DIR.rmdir()
    except OSError:
        pass
    return removed


def run_preflight(config: dict[str, Any]) -> None:
    if not PROTOCOL_PATH.exists():
        PROTOCOL_PATH.parent.mkdir(parents=True, exist_ok=True)
        PROTOCOL_PATH.write_text(
            CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8"
        )
    if CONFIG_PATH.read_bytes() != PROTOCOL_PATH.read_bytes():
        raise RuntimeError(
            "The Phase 19 configuration differs from its frozen protocol"
        )

    input_records: dict[str, dict[str, Any]] = {}
    frozen = config["frozen_inputs"]
    for key, value in frozen.items():
        if not key.endswith("_path"):
            continue
        path = ROOT / str(value)
        hash_key = key.replace("_path", "_sha256")
        if not path.exists():
            raise RuntimeError(f"Missing frozen Phase 19 input: {path}")
        observed = sha256(path)
        expected = str(frozen[hash_key])
        if observed != expected:
            raise RuntimeError(f"Frozen Phase 19 input changed: {path}")
        input_records[key] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": observed,
            "size_bytes": path.stat().st_size,
        }

    with rasterio.open(input_path(config, "lidar_metrics_path")) as dataset:
        descriptions = set(str(value) for value in dataset.descriptions)
        missing = sorted(
            set(config["lidar_profile"]["source_bands"]) - descriptions
        )
        if missing:
            raise RuntimeError(
                f"LiDAR profile bands are missing: {', '.join(missing)}"
            )

    probe = subprocess.run(
        [
            str(config["neural"]["python"]),
            "-c",
            (
                "import torch; print(torch.__version__); "
                "print(torch.backends.mps.is_available())"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    targets = pd.read_parquet(input_path(config, "targets_path"))
    free_bytes = shutil.disk_usage(ROOT).free
    minimum_free = 1_500_000_000
    if free_bytes < minimum_free:
        raise RuntimeError(
            f"Phase 19 needs at least {minimum_free / 1e9:.2f} GB free; "
            f"{free_bytes / 1e9:.2f} GB is available"
        )
    record = {
        "created_utc": utc_now(),
        "config_sha256": sha256(CONFIG_PATH),
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "rows": len(targets),
        "input_records": input_records,
        "neural_environment_probe": probe.stdout.strip().splitlines(),
        "free_space_bytes": free_bytes,
        "model_specs": config["model_specs"],
    }
    if PREFLIGHT_PATH.exists():
        existing = json.loads(PREFLIGHT_PATH.read_text(encoding="utf-8"))
        for key in [
            "config_sha256",
            "protocol_sha256",
            "rows",
            "input_records",
            "model_specs",
        ]:
            if existing[key] != record[key]:
                raise RuntimeError("The Phase 19 preflight checkpoint is stale")
        print("preflight: validated frozen protocol and inputs", flush=True)
        return
    atomic_json(PREFLIGHT_PATH, record)
    print(
        f"preflight: {len(targets):,} rows, "
        f"{len(config['model_specs'])} neural specifications, "
        f"{free_bytes / 1e9:.2f} GB free",
        flush=True,
    )


def run_profiles(config: dict[str, Any]) -> None:
    if PROFILE_PATH.exists() and PROFILE_MANIFEST_PATH.exists():
        manifest = json.loads(
            PROFILE_MANIFEST_PATH.read_text(encoding="utf-8")
        )
        if (
            manifest["config_sha256"] == sha256(CONFIG_PATH)
            and manifest["output_sha256"] == sha256(PROFILE_PATH)
        ):
            print("profiles: validated three-layer checkpoint", flush=True)
            return
        raise RuntimeError("The Phase 19 profile checkpoint is stale")

    targets = pd.read_parquet(input_path(config, "targets_path"))
    profile_config = config["lidar_profile"]
    source_bands = [str(value) for value in profile_config["source_bands"]]
    labels = [str(value) for value in profile_config["layer_labels"]]
    if len(source_bands) != len(labels):
        raise RuntimeError("LiDAR profile bands and labels differ in length")
    with rasterio.open(input_path(config, "lidar_metrics_path")) as dataset:
        lookup = {
            str(name): index
            for index, name in enumerate(dataset.descriptions, start=1)
        }
        aggregate: list[np.ndarray] = []
        for name in source_bands:
            values, _ = phase17.aggregate_band(
                dataset, lookup, name, cells=5
            )
            aggregate.append(
                values[
                    targets["raster_patch_row"].to_numpy(dtype=np.int64),
                    targets["raster_patch_column"].to_numpy(dtype=np.int64),
                ]
            )
    volumes = np.column_stack(aggregate).astype(np.float64)
    finite = np.isfinite(volumes).all(axis=1) & (volumes >= 0).all(axis=1)
    total = np.where(finite, volumes.sum(axis=1), np.nan)
    valid = finite & (
        total > float(profile_config["minimum_total_volume"])
    )
    proportions = np.full(volumes.shape, np.nan, dtype=np.float64)
    proportions[valid] = volumes[valid] / total[valid, None]
    entropy = np.full(len(targets), np.nan, dtype=np.float64)
    positive = proportions > 0
    entropy[valid] = -np.where(
        positive[valid],
        proportions[valid] * np.log(proportions[valid]),
        0.0,
    ).sum(axis=1)
    output = targets[["row_id"]].copy()
    for index, label in enumerate(labels):
        output[f"profile_volume_{label}"] = volumes[:, index].astype(
            np.float32
        )
        output[f"profile_proportion_{label}"] = proportions[:, index].astype(
            np.float32
        )
    entropy_name = str(profile_config["entropy_name"])
    output[entropy_name] = entropy.astype(np.float32)
    output[f"{entropy_name}_normalized"] = (
        entropy / math.log(len(labels))
    ).astype(np.float32)
    output["profile_valid"] = valid
    atomic_parquet(output, PROFILE_PATH)
    manifest = {
        "created_utc": utc_now(),
        "config_sha256": sha256(CONFIG_PATH),
        "source_sha256": sha256(input_path(config, "lidar_metrics_path")),
        "rows": len(output),
        "valid_rows": int(valid.sum()),
        "source_bands": source_bands,
        "layer_labels": labels,
        "documented_lidarshm_commit": str(
            profile_config["documented_lidarshm_commit"]
        ),
        "interpretation": str(profile_config["note"]),
        "output_sha256": sha256(PROFILE_PATH),
    }
    atomic_json(PROFILE_MANIFEST_PATH, manifest)
    print(
        f"profiles: retained documented three-layer proportions for "
        f"{int(valid.sum()):,}/{len(output):,} units",
        flush=True,
    )


def run_patches(config: dict[str, Any]) -> None:
    paths = [PATCH_QUANTIZED_PATH, PATCH_SCALES_PATH, PATCH_VALID_PATH]
    if SUMMARY_MANIFEST_PATH.exists() and not any(
        path.exists() for path in paths
    ):
        summary_manifest = json.loads(
            SUMMARY_MANIFEST_PATH.read_text(encoding="utf-8")
        )
        if (
            summary_manifest["config_sha256"] == sha256(CONFIG_PATH)
            and summary_manifest["phase19_patch_manifest_sha256"]
            == sha256(PATCH_MANIFEST_PATH)
        ):
            print(
                "patches: compact summaries are frozen; raw patch was "
                "discarded",
                flush=True,
            )
            return
        raise RuntimeError("The compact context checkpoint is stale")
    if PATCH_MANIFEST_PATH.exists() and all(path.exists() for path in paths):
        manifest = json.loads(
            PATCH_MANIFEST_PATH.read_text(encoding="utf-8")
        )
        if (
            manifest["config_sha256"] == sha256(CONFIG_PATH)
            and manifest["target_sha256"]
            == sha256(input_path(config, "targets_path"))
            and all(
                manifest["output_sha256"][path.name] == sha256(path)
                for path in paths
            )
        ):
            print("patches: validated ordered 9 by 9 checkpoint", flush=True)
            return
        raise RuntimeError("The Phase 19 patch checkpoint is stale")

    targets = pd.read_parquet(input_path(config, "targets_path"))
    settings = config["tessera"]
    cells = 9
    dimensions = int(settings["embedding_dimensions"])
    points = context_points(targets, cells)
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
    registry_path, landmask_path = phase17.shared.tessera.ensure_registry(
        version
    )
    rows = phase17.shared.tessera.lookup_embedding_rows(
        registry_path, tiles, year, version, variant
    )
    masks = phase17.shared.tessera.lookup_landmask_rows(landmask_path, tiles)
    phase17.shared.tessera.assert_complete_inventory(rows, masks, tiles)
    rows_by_name = {
        phase17.shared.tessera.tile_name((float(row.lon), float(row.lat))): row
        for row in rows.itertuples(index=False)
    }
    masks_by_name = {
        phase17.shared.tessera.tile_name((float(row.lon), float(row.lat))): row
        for row in masks.itertuples(index=False)
    }

    PATCH_DIR.mkdir(parents=True, exist_ok=True)
    shape = (len(targets), dimensions, cells, cells)
    if not all(path.exists() for path in paths):
        if any(path.exists() for path in paths):
            raise RuntimeError("Only part of the Phase 19 patch array exists")
        quantized_output = np.lib.format.open_memmap(
            PATCH_QUANTIZED_PATH, mode="w+", dtype=np.int8, shape=shape
        )
        scales_output = np.lib.format.open_memmap(
            PATCH_SCALES_PATH,
            mode="w+",
            dtype=np.float16,
            shape=(len(targets), cells, cells),
        )
        valid_output = np.lib.format.open_memmap(
            PATCH_VALID_PATH,
            mode="w+",
            dtype=np.uint8,
            shape=(len(targets), cells, cells),
        )
        quantized_output[:] = 0
        scales_output[:] = 0
        valid_output[:] = 0
        quantized_output.flush()
        scales_output.flush()
        valid_output.flush()
    else:
        quantized_output = np.load(PATCH_QUANTIZED_PATH, mmap_mode="r+")
        scales_output = np.load(PATCH_SCALES_PATH, mmap_mode="r+")
        valid_output = np.load(PATCH_VALID_PATH, mmap_mode="r+")
        if quantized_output.shape != shape:
            raise RuntimeError("The resumable Phase 19 patch has wrong shape")

    PATCH_TILE_DIR.mkdir(parents=True, exist_ok=True)
    PATCH_STREAM_DIR.mkdir(parents=True, exist_ok=True)
    attempts = int(settings["download_attempts"])
    delay = int(settings["retry_delay_seconds"])
    config_hash = sha256(CONFIG_PATH)
    tile_records: list[dict[str, Any]] = []
    for code, tile in enumerate(tiles):
        name = phase17.shared.tessera.tile_name(tile)
        indices = np.flatnonzero(tile_code == code)
        marker_path = PATCH_TILE_DIR / f"{name}.json"
        if marker_path.exists():
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if (
                marker["config_sha256"] != config_hash
                or marker["source_points"] != len(indices)
            ):
                raise RuntimeError(f"Stale Phase 19 tile marker: {name}")
            tile_records.append(marker)
            print(
                f"patches [{code + 1}/{len(tiles)}]: resumed {name}",
                flush=True,
            )
            continue

        row = rows_by_name[name]
        mask_row = masks_by_name[name]
        local_paths = phase17.shared.tessera.local_tile_paths(
            PATCH_STREAM_DIR, year, tile
        )
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
                f"Insufficient space for {name}: {free / 1e9:.2f} GB "
                f"free, {required / 1e9:.2f} GB required"
            )
        print(
            f"patches [{code + 1}/{len(tiles)}]: acquire {name} for "
            f"{len(indices):,} ordered pixels",
            flush=True,
        )
        try:
            for kind in ["embedding", "scales", "landmask"]:
                phase17.shared.download_checked(
                    phase17.shared.tessera.s3_url(sources[kind]),
                    local_paths[kind],
                    sizes[kind],
                    attempts,
                    delay,
                )
            quantized = np.load(local_paths["embedding"], mmap_mode="r")
            scales = np.load(local_paths["scales"], mmap_mode="r")
            with rasterio.open(local_paths["landmask"]) as dataset:
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
            local_scale = scales[
                rows_index[local_positions], columns_index[local_positions]
            ].astype(np.float32)
            local_landmask = landmask[
                rows_index[local_positions], columns_index[local_positions]
            ]
            usable = (
                np.isfinite(local_scale)
                & (local_scale > 0)
                & (local_landmask > 0)
            )
            valid_positions = local_positions[usable]
            source_indices = indices[valid_positions]
            row_position = points["row_position"][source_indices].astype(
                np.int64
            )
            local_row = points["local_row"][source_indices].astype(np.int64)
            local_column = points["local_column"][source_indices].astype(
                np.int64
            )
            values = quantized[
                rows_index[valid_positions], columns_index[valid_positions]
            ].astype(np.int8)
            quantized_output[
                row_position, :, local_row, local_column
            ] = values
            scales_output[row_position, local_row, local_column] = scales[
                rows_index[valid_positions], columns_index[valid_positions]
            ].astype(np.float16)
            valid_output[row_position, local_row, local_column] = 1
            quantized_output.flush()
            scales_output.flush()
            valid_output.flush()
            marker = {
                "created_utc": utc_now(),
                "tile": name,
                "config_sha256": config_hash,
                "source_points": len(indices),
                "valid_points": len(source_indices),
                "source_sizes": sizes,
                "embedding_source": sources["embedding"],
                "scales_source": sources["scales"],
                "landmask_source": sources["landmask"],
            }
            atomic_json(marker_path, marker)
            tile_records.append(marker)
        finally:
            for path in local_paths.values():
                path.unlink(missing_ok=True)
            gc.collect()

    valid_count = np.asarray(valid_output).sum(axis=(1, 2))
    minimum = int(settings["minimum_valid_pixels"]["9"])
    centre = cells // 2
    centre_valid = np.asarray(valid_output[:, centre, centre], dtype=bool)
    if int((valid_count >= minimum).sum()) < int(0.90 * len(targets)):
        raise RuntimeError("Too few 9 by 9 TESSERA contexts passed")
    if int(centre_valid.sum()) < int(0.95 * len(targets)):
        raise RuntimeError("Too few 9 by 9 contexts have valid centres")
    del quantized_output, scales_output, valid_output
    gc.collect()
    manifest = {
        "created_utc": utc_now(),
        "config_sha256": config_hash,
        "target_sha256": sha256(input_path(config, "targets_path")),
        "embedding_year": year,
        "dataset_version": version,
        "dataset_variant": variant,
        "shape": list(shape),
        "tile_count": len(tiles),
        "valid_patch_count": int((valid_count >= minimum).sum()),
        "minimum_valid_pixels": minimum,
        "tile_records": tile_records,
        "output_sha256": {path.name: sha256(path) for path in paths},
    }
    atomic_json(PATCH_MANIFEST_PATH, manifest)
    shutil.rmtree(PATCH_STREAM_DIR, ignore_errors=True)
    print(
        f"patches: retained 90 m context for "
        f"{manifest['valid_patch_count']:,}/{len(targets):,} units",
        flush=True,
    )


def summarise_patch_batch(
    quantized: np.ndarray,
    scales: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    values = quantized.astype(np.float32)
    values *= scales.astype(np.float32)[:, None, :, :]
    mask = valid.astype(bool)
    cells = values.shape[2]
    centre = cells // 2
    centre_values = values[:, :, centre, centre]
    if cells == 1:
        return centre_values.astype(np.float32)

    weights = mask[:, None, :, :]
    count = mask.sum(axis=(1, 2)).astype(np.float64)
    safe_count = np.maximum(count, 1.0)
    total = np.where(weights, values, 0.0).sum(
        axis=(2, 3), dtype=np.float64
    )
    total_square = np.where(weights, values * values, 0.0).sum(
        axis=(2, 3), dtype=np.float64
    )
    mean = total / safe_count[:, None]
    variance = np.maximum(
        total_square / safe_count[:, None] - np.square(mean), 0.0
    )
    standard_deviation = np.sqrt(variance)
    offset = np.arange(cells, dtype=np.float64) - centre
    x_grid = np.broadcast_to(offset[None, :], (cells, cells))
    y_grid = np.broadcast_to(-offset[:, None], (cells, cells))
    sum_x = np.where(mask, x_grid[None, :, :], 0.0).sum(axis=(1, 2))
    sum_y = np.where(mask, y_grid[None, :, :], 0.0).sum(axis=(1, 2))
    sum_x2 = np.where(mask, x_grid[None, :, :] ** 2, 0.0).sum(
        axis=(1, 2)
    )
    sum_y2 = np.where(mask, y_grid[None, :, :] ** 2, 0.0).sum(
        axis=(1, 2)
    )
    sum_x_values = np.where(
        weights, values * x_grid[None, None, :, :], 0.0
    ).sum(axis=(2, 3), dtype=np.float64)
    sum_y_values = np.where(
        weights, values * y_grid[None, None, :, :], 0.0
    ).sum(axis=(2, 3), dtype=np.float64)
    denominator_x = sum_x2 - np.square(sum_x) / safe_count
    denominator_y = sum_y2 - np.square(sum_y) / safe_count
    gradient_x = np.zeros_like(mean)
    gradient_y = np.zeros_like(mean)
    usable_x = denominator_x > 0
    usable_y = denominator_y > 0
    gradient_x[usable_x] = (
        sum_x_values[usable_x]
        - sum_x[usable_x, None]
        * total[usable_x]
        / safe_count[usable_x, None]
    ) / denominator_x[usable_x, None]
    gradient_y[usable_y] = (
        sum_y_values[usable_y]
        - sum_y[usable_y, None]
        * total[usable_y]
        / safe_count[usable_y, None]
    ) / denominator_y[usable_y, None]
    return np.concatenate(
        [
            centre_values,
            mean.astype(np.float32),
            standard_deviation.astype(np.float32),
            gradient_x.astype(np.float32),
            gradient_y.astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)


def write_context_summary(
    cells: int,
    source_quantized: Path,
    source_scales: Path,
    source_valid: Path,
    destination: Path,
    valid_destination: Path,
    minimum_valid: int,
) -> None:
    quantized = np.load(source_quantized, mmap_mode="r")
    scales = np.load(source_scales, mmap_mode="r")
    valid = np.load(source_valid, mmap_mode="r")
    source_cells = quantized.shape[2]
    if cells > source_cells or cells % 2 != 1:
        raise RuntimeError("Invalid context subset")
    start = (source_cells - cells) // 2
    end = start + cells
    dimensions = 128 if cells == 1 else 640
    output = np.lib.format.open_memmap(
        destination,
        mode="w+",
        dtype=np.float32,
        shape=(quantized.shape[0], dimensions),
    )
    valid_output = np.lib.format.open_memmap(
        valid_destination,
        mode="w+",
        dtype=np.uint8,
        shape=(quantized.shape[0],),
    )
    batch_size = 256
    for batch_start in range(0, quantized.shape[0], batch_size):
        batch_end = min(batch_start + batch_size, quantized.shape[0])
        local_valid = np.asarray(
            valid[batch_start:batch_end, start:end, start:end],
            dtype=np.uint8,
        )
        output[batch_start:batch_end] = summarise_patch_batch(
            np.asarray(
                quantized[
                    batch_start:batch_end, :, start:end, start:end
                ],
                dtype=np.int8,
            ),
            np.asarray(
                scales[batch_start:batch_end, start:end, start:end],
                dtype=np.float16,
            ),
            local_valid,
        )
        valid_output[batch_start:batch_end] = (
            (local_valid.sum(axis=(1, 2)) >= minimum_valid)
            & (local_valid[:, cells // 2, cells // 2] > 0)
        ).astype(np.uint8)
        if batch_start % (batch_size * 20) == 0:
            print(
                f"summaries {cells} by {cells}: "
                f"{batch_end:,}/{quantized.shape[0]:,}",
                flush=True,
            )
    output.flush()
    valid_output.flush()


def run_summaries(config: dict[str, Any]) -> None:
    expected_paths = [
        SUMMARY_DIR / "tessera_1.npy",
        SUMMARY_DIR / "tessera_1_valid.npy",
        SUMMARY_DIR / "tessera_3.npy",
        SUMMARY_DIR / "tessera_3_valid.npy",
        SUMMARY_DIR / "tessera_5_valid.npy",
        SUMMARY_DIR / "tessera_9.npy",
        SUMMARY_DIR / "tessera_9_valid.npy",
    ]
    if SUMMARY_MANIFEST_PATH.exists() and all(
        path.exists() for path in expected_paths
    ):
        manifest = json.loads(
            SUMMARY_MANIFEST_PATH.read_text(encoding="utf-8")
        )
        if (
            manifest["config_sha256"] == sha256(CONFIG_PATH)
            and all(
                manifest["output_sha256"][path.name] == sha256(path)
                for path in expected_paths
            )
        ):
            removed = discard_completed_patch_arrays()
            print("summaries: validated all context checkpoints", flush=True)
            if removed:
                print(
                    f"summaries: discarded {removed / 1e6:.1f} MB of "
                    "redundant raw patches",
                    flush=True,
                )
            return
        raise RuntimeError("The Phase 19 context checkpoint is stale")

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    phase18_quantized = input_path(
        config, "phase18_patch_quantized_path"
    )
    phase18_scales = input_path(config, "phase18_patch_scales_path")
    phase18_valid = input_path(config, "phase18_patch_valid_path")
    minimum = config["tessera"]["minimum_valid_pixels"]
    for cells in [1, 3]:
        destination = SUMMARY_DIR / f"tessera_{cells}.npy"
        valid_destination = SUMMARY_DIR / f"tessera_{cells}_valid.npy"
        if destination.exists() != valid_destination.exists():
            raise RuntimeError(f"Partial {cells} by {cells} summary exists")
        if not destination.exists():
            write_context_summary(
                cells,
                phase18_quantized,
                phase18_scales,
                phase18_valid,
                destination,
                valid_destination,
                int(minimum[str(cells)]),
            )
    five_valid_destination = SUMMARY_DIR / "tessera_5_valid.npy"
    if not five_valid_destination.exists():
        valid = np.load(phase18_valid, mmap_mode="r")
        five_valid = (
            (np.asarray(valid).sum(axis=(1, 2)) >= int(minimum["5"]))
            & (np.asarray(valid[:, 2, 2]) > 0)
        ).astype(np.uint8)
        np.save(five_valid_destination, five_valid)
    nine_destination = SUMMARY_DIR / "tessera_9.npy"
    nine_valid_destination = SUMMARY_DIR / "tessera_9_valid.npy"
    if nine_destination.exists() != nine_valid_destination.exists():
        raise RuntimeError("Partial 9 by 9 summary exists")
    if not nine_destination.exists():
        write_context_summary(
            9,
            PATCH_QUANTIZED_PATH,
            PATCH_SCALES_PATH,
            PATCH_VALID_PATH,
            nine_destination,
            nine_valid_destination,
            int(minimum["9"]),
        )
    manifest = {
        "created_utc": utc_now(),
        "config_sha256": sha256(CONFIG_PATH),
        "phase18_patch_sha256": {
            "quantized": sha256(phase18_quantized),
            "scales": sha256(phase18_scales),
            "valid": sha256(phase18_valid),
        },
        "phase19_patch_manifest_sha256": sha256(PATCH_MANIFEST_PATH),
        "external_tessera_5_summary": {
            "path": "data/interim/phase18_cairngorms_neural/summary.npy",
            "sha256": sha256(
                ROOT / "data/interim/phase18_cairngorms_neural/summary.npy"
            ),
        },
        "output_sha256": {
            path.name: sha256(path) for path in expected_paths
        },
    }
    atomic_json(SUMMARY_MANIFEST_PATH, manifest)
    removed = discard_completed_patch_arrays()
    print("summaries: froze 10 m, 30 m, 50 m and 90 m inputs", flush=True)
    print(
        f"summaries: discarded {removed / 1e6:.1f} MB of redundant "
        "raw patches",
        flush=True,
    )


def exact_catalog_items(
    catalog: Any, collection: str, item_ids: list[str]
) -> list[Any]:
    items = list(
        catalog.search(collections=[collection], ids=item_ids).items()
    )
    by_id = {item.id: item for item in items}
    missing = sorted(set(item_ids) - set(by_id))
    if missing:
        raise RuntimeError(
            f"{collection} did not return {len(missing)} frozen items"
        )
    return [by_id[item_id] for item_id in item_ids]


def conventional_predictor_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "row_id",
        "conventional_context_valid",
        "conventional_valid_pixel_count",
    }
    return [
        column
        for column in frame.columns
        if column not in excluded and "observation_count" not in column
    ]


def conventional_context_columns(features: list[str]) -> list[str]:
    statistics = ["centre", "mean", "std", "dx", "dy"]
    return [
        f"{feature}__{statistic}"
        for statistic in statistics
        for feature in features
    ]


def summarise_conventional_context(
    frame: pd.DataFrame,
    rows: int,
    cells: int,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    features = conventional_predictor_columns(frame)
    values = frame[features].to_numpy(dtype=np.float32).reshape(
        rows, cells, cells, len(features)
    )
    s2_valid = (
        frame["s2_valid_observation_count"].to_numpy()
        >= int(
            config["conventional"]["sentinel_2"][
                "minimum_valid_observations_per_point"
            ]
        )
    )
    s1_count_columns = [
        column
        for column in frame
        if column.startswith("s1_")
        and column.endswith("observation_count")
    ]
    if not s1_count_columns:
        raise RuntimeError("No Sentinel-1 count columns were extracted")
    pixel_valid = s2_valid
    for column in s1_count_columns:
        pixel_valid &= (
            frame[column].to_numpy()
            >= int(
                config["conventional"]["sentinel_1"][
                    "minimum_valid_observations_per_point_per_state"
                ]
            )
        )
    pixel_valid &= np.isfinite(values.reshape(-1, len(features))).all(axis=1)
    valid_grid = pixel_valid.reshape(rows, cells, cells)
    feature_grid = values.transpose(0, 3, 1, 2)
    summary = summarise_patch_batch(feature_grid, np.ones(
        (rows, cells, cells), dtype=np.float32
    ), valid_grid)
    minimum = int(config["conventional"]["minimum_valid_pixels"])
    context_valid = (
        (valid_grid.sum(axis=(1, 2)) >= minimum)
        & valid_grid[:, cells // 2, cells // 2]
        & np.isfinite(summary).all(axis=1)
    )
    return (
        summary.astype(np.float32),
        context_valid,
        valid_grid.sum(axis=(1, 2)).astype(np.int16),
        conventional_context_columns(features),
    )


def run_conventional(config: dict[str, Any]) -> None:
    if CONVENTIONAL_PATH.exists() and CONVENTIONAL_MANIFEST_PATH.exists():
        manifest = json.loads(
            CONVENTIONAL_MANIFEST_PATH.read_text(encoding="utf-8")
        )
        if (
            manifest["config_sha256"] == sha256(CONFIG_PATH)
            and manifest["output_sha256"] == sha256(CONVENTIONAL_PATH)
            and manifest["target_sha256"]
            == sha256(input_path(config, "targets_path"))
        ):
            shutil.rmtree(CONVENTIONAL_CHUNK_DIR, ignore_errors=True)
            print("conventional: validated 50 m context checkpoint", flush=True)
            return
        raise RuntimeError("The Phase 19 conventional checkpoint is stale")

    targets = pd.read_parquet(
        input_path(config, "targets_path"),
        columns=["row_id", "bng_x", "bng_y", "longitude", "latitude"],
    )
    phase18_manifest = json.loads(
        input_path(
            config, "phase18_conventional_manifest_path"
        ).read_text(encoding="utf-8")
    )
    settings = config["conventional"]
    catalog = conventional.open_catalog(str(settings["stac_api"]))
    s2_items = exact_catalog_items(
        catalog,
        str(settings["sentinel_2"]["collection"]),
        [str(value) for value in phase18_manifest["sentinel_2_item_ids"]],
    )
    s1_items = exact_catalog_items(
        catalog,
        str(settings["sentinel_1"]["collection"]),
        [str(value) for value in phase18_manifest["sentinel_1_item_ids"]],
    )
    dem_items = exact_catalog_items(
        catalog,
        str(settings["topography"]["collection"]),
        [str(value) for value in phase18_manifest["terrain_item_ids"]],
    )
    print(
        f"conventional: reusing {len(s2_items)} Sentinel-2, "
        f"{len(s1_items)} Sentinel-1 and {len(dem_items)} DEM items",
        flush=True,
    )

    CONVENTIONAL_CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    cells = int(settings["spatial_context_cells"])
    chunk_rows = int(settings["chunk_rows"])
    to_wgs84 = Transformer.from_crs(
        config["study"]["analysis_crs"], "EPSG:4326", always_xy=True
    )
    expected_columns: list[str] | None = None
    chunk_paths: list[Path] = []
    for start in range(0, len(targets), chunk_rows):
        end = min(start + chunk_rows, len(targets))
        chunk_path = CONVENTIONAL_CHUNK_DIR / (
            f"rows_{start:06d}_{end:06d}.parquet"
        )
        chunk_paths.append(chunk_path)
        if chunk_path.exists():
            existing = pd.read_parquet(chunk_path)
            if len(existing) != end - start:
                raise RuntimeError(f"Stale conventional chunk: {chunk_path}")
            if expected_columns is None:
                expected_columns = list(existing.columns)
            elif list(existing.columns) != expected_columns:
                raise RuntimeError("Conventional chunk columns differ")
            print(
                f"conventional [{end:,}/{len(targets):,}]: resumed",
                flush=True,
            )
            continue

        target_chunk = targets.iloc[start:end].reset_index(drop=True)
        points = context_points(target_chunk, cells)
        longitude, latitude = to_wgs84.transform(
            points["bng_x"], points["bng_y"]
        )
        longitude = np.asarray(longitude, dtype=np.float64)
        latitude = np.asarray(latitude, dtype=np.float64)
        bbox = [
            float(longitude.min()),
            float(latitude.min()),
            float(longitude.max()),
            float(latitude.max()),
        ]
        print(
            f"conventional [{end:,}/{len(targets):,}]: "
            f"extract {len(longitude):,} pixels",
            flush=True,
        )
        s2, _ = conventional.extract_sentinel2(
            s2_items, longitude, latitude, settings["sentinel_2"]
        )
        s1, _ = conventional.extract_sentinel1(
            s1_items, longitude, latitude, settings["sentinel_1"]
        )
        terrain_frame = pd.DataFrame(
            {
                "x_epsg5070": points["bng_x"],
                "y_epsg5070": points["bng_y"],
            }
        )
        terrain, _ = conventional.extract_topography(
            dem_items, terrain_frame, bbox, settings["topography"]
        )
        pixel_frame = pd.DataFrame(
            {
                name: values
                for name, values in sorted({**terrain, **s2, **s1}.items())
            }
        )
        summary, valid, valid_pixel_count, columns = (
            summarise_conventional_context(
            pixel_frame, len(target_chunk), cells, config
            )
        )
        output = pd.DataFrame(summary, columns=columns)
        output.insert(
            0, "row_id", target_chunk["row_id"].to_numpy(dtype=np.int64)
        )
        output["conventional_valid_pixel_count"] = valid_pixel_count
        output["conventional_context_valid"] = valid
        if expected_columns is None:
            expected_columns = list(output.columns)
        elif list(output.columns) != expected_columns:
            raise RuntimeError("Conventional feature order changed by chunk")
        atomic_parquet(output, chunk_path)
        gc.collect()

    output = pd.concat(
        [pd.read_parquet(path) for path in chunk_paths], ignore_index=True
    )
    if not targets[["row_id"]].equals(output[["row_id"]]):
        raise RuntimeError("Conventional context rows are misaligned")
    atomic_parquet(output, CONVENTIONAL_PATH)
    source_inventory = (
        ROOT / "metadata/phase18_cairngorms_conventional_inventory.csv"
    )
    if sha256(source_inventory) != phase18_manifest["inventory_sha256"]:
        raise RuntimeError("The Phase 18 conventional inventory changed")
    shutil.copyfile(source_inventory, CONVENTIONAL_INVENTORY_PATH)
    manifest = {
        "created_utc": utc_now(),
        "config_sha256": sha256(CONFIG_PATH),
        "target_sha256": sha256(input_path(config, "targets_path")),
        "phase18_manifest_sha256": sha256(
            input_path(config, "phase18_conventional_manifest_path")
        ),
        "spatial_context_cells": cells,
        "spatial_support_m": cells * 10,
        "predictor_count": len(
            [
                column
                for column in output
                if "__" in column
            ]
        ),
        "valid_rows": int(output["conventional_context_valid"].sum()),
        "sentinel_2_item_ids": [item.id for item in s2_items],
        "sentinel_1_item_ids": [item.id for item in s1_items],
        "terrain_item_ids": [item.id for item in dem_items],
        "inventory_sha256": sha256(CONVENTIONAL_INVENTORY_PATH),
        "output_sha256": sha256(CONVENTIONAL_PATH),
    }
    atomic_json(CONVENTIONAL_MANIFEST_PATH, manifest)
    shutil.rmtree(CONVENTIONAL_CHUNK_DIR, ignore_errors=True)
    print(
        f"conventional: valid 50 m predictors for "
        f"{manifest['valid_rows']:,}/{len(output):,} units",
        flush=True,
    )


def atomic_npy(values: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npy")
    np.save(temporary, values)
    temporary.replace(path)


def raw_target_valid(
    targets: pd.DataFrame, config: dict[str, Any]
) -> np.ndarray:
    values = targets[raw_target_names(config)].to_numpy(dtype=np.float64)
    return np.isfinite(values).all(axis=1)


def run_prepare(config: dict[str, Any]) -> None:
    required = [
        PROFILE_PATH,
        SUMMARY_MANIFEST_PATH,
        CONVENTIONAL_PATH,
        input_path(config, "phase18_conventional_path"),
    ]
    if not all(path.exists() for path in required):
        raise RuntimeError("Phase 19 preparation inputs are incomplete")
    local_outputs = [
        ARRAY_DIR / "conventional_5.npy",
        ARRAY_DIR / "profiles.npy",
        ARRAY_DIR / "common_valid.npy",
        ARRAY_DIR / "row_id.npy",
    ]
    if ARRAY_MANIFEST_PATH.exists() and all(
        path.exists() for path in local_outputs
    ):
        manifest = json.loads(
            ARRAY_MANIFEST_PATH.read_text(encoding="utf-8")
        )
        if (
            manifest["config_sha256"] == sha256(CONFIG_PATH)
            and all(
                manifest["local_array_sha256"][path.name] == sha256(path)
                for path in local_outputs
            )
        ):
            print("prepare: validated common-cohort arrays", flush=True)
            return
        raise RuntimeError("The Phase 19 prepared arrays are stale")

    targets = pd.read_parquet(input_path(config, "targets_path"))
    profiles = pd.read_parquet(PROFILE_PATH)
    conventional_context = pd.read_parquet(CONVENTIONAL_PATH)
    conventional_centre = pd.read_parquet(
        input_path(config, "phase18_conventional_path")
    )
    for frame, label in [
        (profiles, "profile"),
        (conventional_context, "conventional context"),
        (conventional_centre, "conventional centre"),
    ]:
        if not targets[["row_id"]].equals(frame[["row_id"]]):
            raise RuntimeError(f"{label} rows are misaligned")

    summary_paths = {
        1: SUMMARY_DIR / "tessera_1.npy",
        3: SUMMARY_DIR / "tessera_3.npy",
        5: input_path(config, "phase18_patch_quantized_path").parent
        / "summary.npy",
        9: SUMMARY_DIR / "tessera_9.npy",
    }
    summary_valid_paths = {
        cells: SUMMARY_DIR / f"tessera_{cells}_valid.npy"
        for cells in [1, 3, 5, 9]
    }
    common = raw_target_valid(targets, config)
    common &= profiles["profile_valid"].to_numpy(dtype=bool)
    common &= conventional_context[
        "conventional_context_valid"
    ].to_numpy(dtype=bool)
    common &= conventional_centre[
        "conventional_valid"
    ].to_numpy(dtype=bool)
    for cells, path in summary_paths.items():
        values = np.load(path, mmap_mode="r")
        valid = np.load(summary_valid_paths[cells], mmap_mode="r")
        if len(values) != len(targets):
            raise RuntimeError(f"TESSERA {cells} context rows differ")
        common &= np.asarray(valid, dtype=bool)
        common &= np.isfinite(np.asarray(values)).all(axis=1)

    context_columns = [
        column for column in conventional_context if "__" in column
    ]
    conventional_values = conventional_context[context_columns].to_numpy(
        dtype=np.float32
    )
    common &= np.isfinite(conventional_values).all(axis=1)
    profile_columns = [
        column
        for column in profiles
        if column.startswith("profile_proportion_")
    ]
    profile_values = profiles[profile_columns].to_numpy(dtype=np.float32)
    if len(profile_columns) != 3:
        raise RuntimeError("Expected three normalized vertical-profile layers")
    common &= np.isfinite(profile_values).all(axis=1)
    if int(common.sum()) < int(
        config["spatial_evaluation"]["minimum_training_rows"]
    ):
        raise RuntimeError("Too few rows survive the shared Phase 19 cohort")

    ARRAY_DIR.mkdir(parents=True, exist_ok=True)
    atomic_npy(conventional_values, ARRAY_DIR / "conventional_5.npy")
    atomic_npy(profile_values, ARRAY_DIR / "profiles.npy")
    atomic_npy(common.astype(np.uint8), ARRAY_DIR / "common_valid.npy")
    atomic_npy(
        targets["row_id"].to_numpy(dtype=np.int64),
        ARRAY_DIR / "row_id.npy",
    )
    external_arrays = {
        "tessera_1.npy": summary_paths[1],
        "tessera_3.npy": summary_paths[3],
        "tessera_5.npy": summary_paths[5],
        "tessera_9.npy": summary_paths[9],
        "conventional_centre.npy": (
            input_path(config, "phase18_patch_quantized_path").parent
            / "conventional.npy"
        ),
    }
    manifest = {
        "created_utc": utc_now(),
        "config_sha256": sha256(CONFIG_PATH),
        "rows": len(targets),
        "common_valid_rows": int(common.sum()),
        "target_names": target_names(config),
        "profile_columns": profile_columns,
        "conventional_context_columns": context_columns,
        "local_array_sha256": {
            path.name: sha256(path) for path in local_outputs
        },
        "external_arrays": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
            }
            for name, path in external_arrays.items()
        },
    }
    atomic_json(ARRAY_MANIFEST_PATH, manifest)
    print(
        f"prepare: fixed one paired cohort of "
        f"{manifest['common_valid_rows']:,}/{len(targets):,} units",
        flush=True,
    )


def spatial_weights(blocks: pd.Series) -> np.ndarray:
    counts = blocks.value_counts()
    weights = blocks.map(1.0 / counts).to_numpy(dtype=np.float64)
    return weights * len(weights) / weights.sum()


def validation_split(
    frame: pd.DataFrame,
    train: np.ndarray,
    fold: int,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    blocks = np.asarray(sorted(frame.loc[train, "spatial_block"].unique()))
    rng = np.random.default_rng(
        int(config["spatial_evaluation"]["seed"]) + fold
    )
    count = max(
        1,
        int(
            round(
                len(blocks)
                * float(
                    config["spatial_evaluation"][
                        "validation_block_fraction"
                    ]
                )
            )
        ),
    )
    validation_blocks = set(
        rng.choice(blocks, size=count, replace=False).tolist()
    )
    validation = train & frame["spatial_block"].isin(
        validation_blocks
    ).to_numpy()
    return train & ~validation, validation


def fold_target_matrix(
    targets: pd.DataFrame,
    train: np.ndarray,
    fold: int,
    weights: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    raw_names = raw_target_names(config)
    matrix = np.full(
        (len(targets), len(raw_names) + 1), np.nan, dtype=np.float32
    )
    matrix[:, : len(raw_names)] = targets[raw_names].to_numpy(
        dtype=np.float32
    )
    adjustment = [str(value) for value in config["targets"][
        "adjustment_predictors"
    ]]
    settings = config["gradient_boosting"]
    model = HistGradientBoostingRegressor(
        learning_rate=float(settings["learning_rate"]),
        max_iter=int(settings["max_iter"]),
        max_leaf_nodes=int(settings["max_leaf_nodes"]),
        min_samples_leaf=int(settings["min_samples_leaf"]),
        l2_regularization=float(settings["l2_regularization"]),
        early_stopping=False,
        random_state=int(settings["random_state"]) + fold,
    )
    model.fit(
        targets.loc[train, adjustment].to_numpy(dtype=np.float32),
        targets.loc[train, raw_names[0]].to_numpy(dtype=np.float32),
        sample_weight=weights[train],
    )
    expected = model.predict(
        targets[adjustment].to_numpy(dtype=np.float32)
    )
    matrix[:, -1] = (
        targets[raw_names[0]].to_numpy(dtype=np.float32) - expected
    )
    return matrix


def save_fold_arrays(
    fold: int,
    target_matrix: np.ndarray,
    profiles: np.ndarray,
    weights: np.ndarray,
    train: np.ndarray,
    subtrain: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
) -> Path:
    path = ARRAY_DIR / f"fold_{fold}.npz"
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        targets=target_matrix.astype(np.float32),
        profiles=profiles.astype(np.float32),
        weights=weights.astype(np.float32),
        train_indices=np.flatnonzero(train).astype(np.int64),
        subtrain_indices=np.flatnonzero(subtrain).astype(np.int64),
        validation_indices=np.flatnonzero(validation).astype(np.int64),
        test_indices=np.flatnonzero(test).astype(np.int64),
    )
    temporary.replace(path)
    return path


def neural_predictions(
    fold: int,
    model_name: str,
    test_indices: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray | None]:
    regression: list[np.ndarray] = []
    profiles: list[np.ndarray] = []
    for seed in config["neural"]["seeds"]:
        subprocess.run(
            [
                str(config["neural"]["python"]),
                str(NEURAL_WORKER_PATH),
                "--model",
                model_name,
                "--fold",
                str(fold),
                "--seed",
                str(seed),
            ],
            cwd=ROOT,
            check=True,
        )
        result_path = (
            NEURAL_RESULT_DIR
            / f"fold_{fold}_{model_name}_seed_{seed}.npz"
        )
        with np.load(result_path) as result:
            if not np.array_equal(result["test_indices"], test_indices):
                raise RuntimeError(
                    f"{model_name} test rows changed in fold {fold}"
                )
            regression.append(
                result["regression_predictions"].astype(np.float64)
            )
            if "profile_predictions" in result:
                profiles.append(
                    result["profile_predictions"].astype(np.float64)
                )
    regression_mean = np.mean(regression, axis=0)
    profile_mean = np.mean(profiles, axis=0) if profiles else None
    return regression_mean, profile_mean


def fit_gradient_boosting(
    inputs: np.ndarray,
    target: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    weights: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    settings = config["gradient_boosting"]
    model = HistGradientBoostingRegressor(
        learning_rate=float(settings["learning_rate"]),
        max_iter=int(settings["max_iter"]),
        max_leaf_nodes=int(settings["max_leaf_nodes"]),
        min_samples_leaf=int(settings["min_samples_leaf"]),
        l2_regularization=float(settings["l2_regularization"]),
        early_stopping=False,
        random_state=int(settings["random_state"]),
    )
    model.fit(
        np.asarray(inputs[train], dtype=np.float32),
        target[train],
        sample_weight=weights[train],
    )
    return model.predict(
        np.asarray(inputs[test], dtype=np.float32)
    ).astype(np.float64)


def append_prediction_records(
    records: list[pd.DataFrame],
    metrics: list[dict[str, Any]],
    targets: pd.DataFrame,
    test_indices: np.ndarray,
    observed: np.ndarray,
    predicted: np.ndarray,
    names: list[str],
    model: str,
    fold: int,
    train_rows: int,
) -> None:
    observed = np.atleast_2d(observed)
    predicted = np.atleast_2d(predicted)
    if observed.shape[0] != len(test_indices):
        observed = observed.T
    if predicted.shape[0] != len(test_indices):
        predicted = predicted.T
    if observed.shape != predicted.shape or observed.shape[1] != len(names):
        raise RuntimeError(f"Prediction shape mismatch for {model}")
    location = targets.iloc[test_indices]
    for index, name in enumerate(names):
        values = phase17.metric_values(
            observed[:, index], predicted[:, index]
        )
        metrics.append(
            {
                "target": name,
                "model": model,
                "fold": fold,
                "train_rows": train_rows,
                "test_rows": len(test_indices),
                **values,
            }
        )
        records.append(
            pd.DataFrame(
                {
                    "row_id": location["row_id"].to_numpy(dtype=np.int64),
                    "target": name,
                    "model": model,
                    "fold": fold,
                    "observed": observed[:, index],
                    "predicted": predicted[:, index],
                    "spatial_block": location[
                        "spatial_block"
                    ].to_numpy(),
                    "block_x": location["block_x"].to_numpy(),
                    "block_y": location["block_y"].to_numpy(),
                }
            )
        )


def evaluate_fold(
    fold: int,
    targets: pd.DataFrame,
    inputs: dict[str, np.ndarray],
    profiles: np.ndarray,
    common: np.ndarray,
    weights: np.ndarray,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_train, base_test, diagnostics = (
        phase17.shared.buffered_fold_masks(targets, fold, config)
    )
    train = base_train & common
    test = base_test & common
    minimum_train = int(
        config["spatial_evaluation"]["minimum_training_rows"]
    )
    minimum_test = int(
        config["spatial_evaluation"]["minimum_test_rows"]
    )
    if train.sum() < minimum_train or test.sum() < minimum_test:
        raise RuntimeError(
            f"Fold {fold} has {train.sum()} train and {test.sum()} test rows"
        )
    subtrain, validation = validation_split(targets, train, fold, config)
    target_matrix = fold_target_matrix(
        targets, train, fold, weights, config
    )
    test_indices = np.flatnonzero(test)
    fold_path = save_fold_arrays(
        fold,
        target_matrix,
        profiles,
        weights,
        train,
        subtrain,
        validation,
        test,
    )
    print(
        f"evaluate fold {fold}: train={train.sum():,}; "
        f"validation={validation.sum():,}; test={test.sum():,}; "
        f"buffered={diagnostics['buffered_training_rows']:,}",
        flush=True,
    )
    metric_records: list[dict[str, Any]] = []
    prediction_records: list[pd.DataFrame] = []
    names = target_names(config)
    for position, spec_value in enumerate(
        config["model_specs"], start=1
    ):
        model_name = str(spec_value["name"])
        objective = str(spec_value["objective"])
        print(
            f"evaluate fold {fold} neural "
            f"[{position}/{len(config['model_specs'])}]: {model_name}",
            flush=True,
        )
        regression, profile_prediction = neural_predictions(
            fold, model_name, test_indices, config
        )
        selected_names = names if objective == "multi" else names[:1]
        observed = target_matrix[
            np.ix_(test_indices, np.arange(len(selected_names)))
        ]
        append_prediction_records(
            prediction_records,
            metric_records,
            targets,
            test_indices,
            observed,
            regression,
            selected_names,
            model_name,
            fold,
            int(train.sum()),
        )
        if profile_prediction is not None:
            profile_names = [
                "profile_proportion_0_to_1m",
                "profile_proportion_1_to_10m",
                "profile_proportion_above_10m",
            ]
            profile_model = f"{model_name}__vertical_profile"
            append_prediction_records(
                prediction_records,
                metric_records,
                targets,
                test_indices,
                profiles[test_indices],
                profile_prediction,
                profile_names,
                profile_model,
                fold,
                int(train.sum()),
            )
            observed_entropy = -np.where(
                profiles[test_indices] > 0,
                profiles[test_indices]
                * np.log(np.maximum(profiles[test_indices], 1e-12)),
                0.0,
            ).sum(axis=1)
            predicted_entropy = -np.where(
                profile_prediction > 0,
                profile_prediction
                * np.log(np.maximum(profile_prediction, 1e-12)),
                0.0,
            ).sum(axis=1)
            append_prediction_records(
                prediction_records,
                metric_records,
                targets,
                test_indices,
                observed_entropy[:, None],
                predicted_entropy[:, None],
                ["three_layer_volume_entropy"],
                profile_model,
                fold,
                int(train.sum()),
            )

    primary = target_matrix[:, 0]
    for model_name, input_name in [
        ("tessera_5_hgb", "tessera_5"),
        ("conventional_5_hgb", "conventional_5"),
    ]:
        prediction = fit_gradient_boosting(
            inputs[input_name],
            primary,
            train,
            test,
            weights,
            config,
        )
        append_prediction_records(
            prediction_records,
            metric_records,
            targets,
            test_indices,
            primary[test_indices, None],
            prediction[:, None],
            names[:1],
            model_name,
            fold,
            int(train.sum()),
        )

    fold_metrics = pd.DataFrame(metric_records)
    fold_predictions = pd.concat(prediction_records, ignore_index=True)
    checkpoint_metrics = FOLD_DIR / f"fold_{fold}_metrics.csv"
    checkpoint_predictions = FOLD_DIR / f"fold_{fold}_predictions.parquet"
    checkpoint_manifest = FOLD_DIR / f"fold_{fold}_manifest.json"
    atomic_csv(fold_metrics, checkpoint_metrics)
    atomic_parquet(fold_predictions, checkpoint_predictions)
    atomic_json(
        checkpoint_manifest,
        {
            "created_utc": utc_now(),
            "config_sha256": sha256(CONFIG_PATH),
            "array_manifest_sha256": sha256(ARRAY_MANIFEST_PATH),
            "fold_array_sha256": sha256(fold_path),
            "metrics_sha256": sha256(checkpoint_metrics),
            "predictions_sha256": sha256(checkpoint_predictions),
        },
    )
    return fold_metrics, fold_predictions


def pooled_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (target, model), group in predictions.groupby(
        ["target", "model"], sort=False
    ):
        records.append(
            {
                "target": target,
                "model": model,
                "test_rows": len(group),
                **phase17.metric_values(
                    group["observed"].to_numpy(),
                    group["predicted"].to_numpy(),
                ),
            }
        )
    return pd.DataFrame(records)


def paired_block_comparison(
    predictions: pd.DataFrame,
    candidate: str,
    reference: str,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    candidate_frame = predictions.loc[
        predictions["model"].eq(candidate),
        ["row_id", "observed", "predicted", "spatial_block"],
    ].rename(columns={"predicted": "candidate"})
    reference_frame = predictions.loc[
        predictions["model"].eq(reference),
        ["row_id", "observed", "predicted"],
    ].rename(
        columns={"observed": "reference_observed", "predicted": "reference"}
    )
    paired = candidate_frame.merge(
        reference_frame, on="row_id", how="inner", validate="one_to_one"
    )
    if len(paired) != len(candidate_frame) or len(paired) != len(reference_frame):
        raise RuntimeError(
            f"Paired rows differ for {candidate} and {reference}"
        )
    if not np.allclose(
        paired["observed"], paired["reference_observed"], equal_nan=True
    ):
        raise RuntimeError("Paired models have different observations")
    observed = paired["observed"].to_numpy(dtype=np.float64)
    candidate_values = paired["candidate"].to_numpy(dtype=np.float64)
    reference_values = paired["reference"].to_numpy(dtype=np.float64)
    candidate_metric = phase17.metric_values(observed, candidate_values)
    reference_metric = phase17.metric_values(observed, reference_values)
    blocks = paired["spatial_block"].to_numpy()
    unique_blocks = np.unique(blocks)
    positions = {
        block: np.flatnonzero(blocks == block) for block in unique_blocks
    }
    rng = np.random.default_rng(seed)
    rmse_delta = np.empty(replicates, dtype=np.float64)
    r2_delta = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled = rng.choice(
            unique_blocks, size=len(unique_blocks), replace=True
        )
        index = np.concatenate([positions[block] for block in sampled])
        values_candidate = phase17.metric_values(
            observed[index], candidate_values[index]
        )
        values_reference = phase17.metric_values(
            observed[index], reference_values[index]
        )
        rmse_delta[replicate] = (
            values_candidate["rmse"] - values_reference["rmse"]
        )
        r2_delta[replicate] = (
            values_candidate["r2"] - values_reference["r2"]
        )
    return {
        "rows": len(paired),
        "blocks": len(unique_blocks),
        "candidate_rmse": candidate_metric["rmse"],
        "reference_rmse": reference_metric["rmse"],
        "rmse_delta": candidate_metric["rmse"] - reference_metric["rmse"],
        "rmse_delta_ci_lower": float(np.quantile(rmse_delta, 0.025)),
        "rmse_delta_ci_upper": float(np.quantile(rmse_delta, 0.975)),
        "candidate_r2": candidate_metric["r2"],
        "reference_r2": reference_metric["r2"],
        "r2_delta": candidate_metric["r2"] - reference_metric["r2"],
        "r2_delta_ci_lower": float(np.quantile(r2_delta, 0.025)),
        "r2_delta_ci_upper": float(np.quantile(r2_delta, 0.975)),
        "candidate_spearman_r": candidate_metric["spearman_r"],
        "reference_spearman_r": reference_metric["spearman_r"],
        "spearman_delta": (
            candidate_metric["spearman_r"]
            - reference_metric["spearman_r"]
        ),
    }


def run_evaluate(config: dict[str, Any]) -> None:
    expected = [
        METRICS_PATH,
        POOLED_METRICS_PATH,
        COMPARISON_PATH,
        PREDICTIONS_PATH,
    ]
    if all(path.exists() for path in expected):
        print("evaluate: final comparison tables already exist", flush=True)
        return
    if not ARRAY_MANIFEST_PATH.exists():
        raise RuntimeError("Phase 19 arrays have not been prepared")

    targets = pd.read_parquet(input_path(config, "targets_path"))
    profiles = np.load(ARRAY_DIR / "profiles.npy", mmap_mode="r")
    common = np.load(
        ARRAY_DIR / "common_valid.npy", mmap_mode="r"
    ).astype(bool)
    weights = spatial_weights(targets["spatial_block"])
    inputs = {
        "tessera_5": np.load(
            input_path(config, "phase18_patch_quantized_path").parent
            / "summary.npy",
            mmap_mode="r",
        ),
        "conventional_5": np.load(
            ARRAY_DIR / "conventional_5.npy", mmap_mode="r"
        ),
    }
    FOLD_DIR.mkdir(parents=True, exist_ok=True)
    all_metrics: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []
    for fold in range(int(config["spatial_evaluation"]["region_count"])):
        checkpoint_metrics = FOLD_DIR / f"fold_{fold}_metrics.csv"
        checkpoint_predictions = FOLD_DIR / f"fold_{fold}_predictions.parquet"
        checkpoint_manifest = FOLD_DIR / f"fold_{fold}_manifest.json"
        if (
            checkpoint_metrics.exists()
            and checkpoint_predictions.exists()
            and checkpoint_manifest.exists()
        ):
            manifest = json.loads(
                checkpoint_manifest.read_text(encoding="utf-8")
            )
            if (
                manifest["config_sha256"] != sha256(CONFIG_PATH)
                or manifest["array_manifest_sha256"]
                != sha256(ARRAY_MANIFEST_PATH)
                or manifest["metrics_sha256"]
                != sha256(checkpoint_metrics)
                or manifest["predictions_sha256"]
                != sha256(checkpoint_predictions)
            ):
                raise RuntimeError(f"Stale Phase 19 fold {fold} checkpoint")
            print(f"evaluate fold {fold}: resumed checkpoint", flush=True)
            fold_metrics = pd.read_csv(checkpoint_metrics)
            fold_predictions = pd.read_parquet(checkpoint_predictions)
        else:
            fold_metrics, fold_predictions = evaluate_fold(
                fold,
                targets,
                inputs,
                np.asarray(profiles),
                np.asarray(common),
                weights,
                config,
            )
        all_metrics.append(fold_metrics)
        all_predictions.append(fold_predictions)

    metrics = pd.concat(all_metrics, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    pooled = pooled_metrics(predictions)
    primary = str(config["interpretation_gates"]["primary_target"])
    primary_predictions = predictions.loc[
        predictions["target"].eq(primary)
    ].copy()
    comparisons: list[dict[str, Any]] = []
    replicates = int(config["uncertainty"]["block_bootstrap_replicates"])
    seed = int(config["uncertainty"]["seed"])
    for position, comparison in enumerate(config["comparisons"]):
        candidate = str(comparison["candidate"])
        reference = str(comparison["reference"])
        values = paired_block_comparison(
            primary_predictions,
            candidate,
            reference,
            replicates,
            seed + position,
        )
        reduction = (
            -values["rmse_delta"] / values["reference_rmse"]
            if values["reference_rmse"] > 0
            else float("nan")
        )
        comparisons.append(
            {
                "comparison": str(comparison["name"]),
                "candidate": candidate,
                "reference": reference,
                **values,
                "rmse_reduction_fraction": reduction,
                "material_improvement": bool(
                    reduction
                    >= float(
                        config["interpretation_gates"][
                            "context_material_rmse_reduction_fraction"
                        ]
                    )
                    and values["rmse_delta_ci_upper"] < 0
                ),
            }
        )
    atomic_csv(metrics, METRICS_PATH)
    atomic_csv(pooled, POOLED_METRICS_PATH)
    atomic_csv(pd.DataFrame(comparisons), COMPARISON_PATH)
    atomic_parquet(predictions, PREDICTIONS_PATH)
    print(
        f"evaluate: wrote {len(pooled)} pooled model-target results and "
        f"{len(comparisons)} paired comparisons",
        flush=True,
    )


def model_label(name: str) -> str:
    labels = {
        "tessera_1_single": "TESSERA MLP, 10 m centre",
        "tessera_3_single": "TESSERA MLP, 30 m context",
        "tessera_5_single": "TESSERA MLP, 50 m context",
        "tessera_9_single": "TESSERA MLP, 90 m context",
        "conventional_centre_single": (
            "Sentinel-1/2 + terrain MLP, 10 m centre"
        ),
        "conventional_5_single": (
            "Sentinel-1/2 + terrain MLP, 50 m context"
        ),
        "tessera_5_hgb": "TESSERA gradient boosting, 50 m context",
        "conventional_5_hgb": (
            "Sentinel-1/2 + terrain gradient boosting, 50 m context"
        ),
        "tessera_5_multi": "TESSERA multi-target MLP, 50 m context",
        "tessera_9_multi": "TESSERA multi-target MLP, 90 m context",
        "tessera_5_profile_aux": (
            "TESSERA profile-assisted MLP, 50 m context"
        ),
        "tessera_9_profile_aux": (
            "TESSERA profile-assisted MLP, 90 m context"
        ),
        "conventional_5_profile_aux": (
            "Sentinel-1/2 + terrain profile-assisted MLP"
        ),
    }
    return labels.get(name, name.replace("_", " "))


def comparison_sentence(row: pd.Series) -> str:
    direction = "lower" if float(row.rmse_delta) < 0 else "higher"
    supported = (
        float(row.rmse_delta_ci_upper) < 0
        or float(row.rmse_delta_ci_lower) > 0
    )
    return (
        f"{model_label(str(row.candidate))} had {abs(float(row.rmse_delta)):.3f} "
        f"{direction} RMSE than {model_label(str(row.reference))} "
        f"(paired block-bootstrap 95% interval "
        f"{float(row.rmse_delta_ci_lower):.3f} to "
        f"{float(row.rmse_delta_ci_upper):.3f}; "
        f"{'supported' if supported else 'not clearly separated'})."
    )


def make_figure(config: dict[str, Any]) -> None:
    pooled = pd.read_csv(POOLED_METRICS_PATH)
    comparisons = pd.read_csv(COMPARISON_PATH)
    predictions = pd.read_parquet(PREDICTIONS_PATH)
    primary = str(config["interpretation_gates"]["primary_target"])
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))

    panel = axes[0, 0]
    models = [
        "conventional_centre_single",
        "tessera_1_single",
        "conventional_5_single",
        "tessera_5_single",
    ]
    values = (
        pooled.loc[
            pooled["target"].eq(primary) & pooled["model"].isin(models)
        ]
        .set_index("model")
        .reindex(models)
    )
    colors = ["#3b82f6", "#0f766e", "#60a5fa", "#14b8a6"]
    panel.bar(np.arange(len(models)), values["r2"], color=colors)
    panel.axhline(0, color="#444444", linewidth=0.8)
    panel.set_xticks(
        np.arange(len(models)),
        ["Conventional\n10 m", "TESSERA\n10 m", "Conventional\n50 m", "TESSERA\n50 m"],
    )
    panel.set_ylabel(r"Out-of-fold $R^2$")
    panel.set_title("a  Fair spatial-support comparison", loc="left")

    panel = axes[0, 1]
    context_models = [
        "tessera_1_single",
        "tessera_3_single",
        "tessera_5_single",
        "tessera_9_single",
    ]
    context = (
        pooled.loc[
            pooled["target"].eq(primary)
            & pooled["model"].isin(context_models)
        ]
        .set_index("model")
        .reindex(context_models)
    )
    panel.plot(
        [10, 30, 50, 90],
        context["rmse"],
        marker="o",
        linewidth=2,
        color="#0f766e",
    )
    panel.set_xlabel("TESSERA spatial context (m)")
    panel.set_ylabel("Out-of-fold RMSE")
    panel.set_title("b  Context-size ablation", loc="left")
    panel.grid(axis="y", alpha=0.25)

    panel = axes[1, 0]
    selected_names = [
        "context_30m_vs_10m",
        "context_50m_vs_10m",
        "context_90m_vs_10m",
        "multitask_effect_50m",
        "profile_aux_effect_50m",
        "mlp_vs_hgb_tessera_50m",
    ]
    selected = (
        comparisons.set_index("comparison").reindex(selected_names).dropna()
    )
    y = np.arange(len(selected))
    point = selected["rmse_delta"].to_numpy()
    lower = point - selected["rmse_delta_ci_lower"].to_numpy()
    upper = selected["rmse_delta_ci_upper"].to_numpy() - point
    panel.errorbar(
        point,
        y,
        xerr=np.vstack([lower, upper]),
        fmt="o",
        color="#334155",
        ecolor="#64748b",
        capsize=3,
    )
    panel.axvline(0, color="#b91c1c", linewidth=0.9, linestyle="--")
    panel.set_yticks(
        y,
        [
            "30 vs 10 m",
            "50 vs 10 m",
            "90 vs 10 m",
            "Multi-target vs single",
            "Profile-assisted vs single",
            "MLP vs gradient boosting",
        ][: len(selected)],
    )
    panel.set_xlabel("Paired RMSE change (candidate - reference)")
    panel.set_title("c  Complete-block paired comparisons", loc="left")

    panel = axes[1, 1]
    entropy = predictions.loc[
        predictions["target"].eq("three_layer_volume_entropy")
        & predictions["model"].eq(
            "tessera_5_profile_aux__vertical_profile"
        )
    ]
    if not entropy.empty:
        panel.hexbin(
            entropy["observed"],
            entropy["predicted"],
            gridsize=36,
            mincnt=1,
            cmap="viridis",
        )
        bounds = [
            float(
                min(entropy["observed"].min(), entropy["predicted"].min())
            ),
            float(
                max(entropy["observed"].max(), entropy["predicted"].max())
            ),
        ]
        panel.plot(bounds, bounds, color="#444444", linestyle="--")
    panel.set_xlabel("Observed three-layer entropy")
    panel.set_ylabel("Predicted three-layer entropy")
    panel.set_title("d  Vertical-profile auxiliary task", loc="left")

    figure.suptitle(
        "Cairngorms TESSERA representation diagnostics",
        fontsize=14,
        y=0.995,
    )
    figure.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=220, bbox_inches="tight")
    plt.close(figure)


def run_report(config: dict[str, Any]) -> None:
    required = [
        METRICS_PATH,
        POOLED_METRICS_PATH,
        COMPARISON_PATH,
        PREDICTIONS_PATH,
    ]
    if not all(path.exists() for path in required):
        raise RuntimeError("Phase 19 evaluation outputs are incomplete")
    pooled = pd.read_csv(POOLED_METRICS_PATH)
    comparisons = pd.read_csv(COMPARISON_PATH).set_index("comparison")
    primary = str(config["interpretation_gates"]["primary_target"])
    primary_rows = pooled.loc[pooled["target"].eq(primary)].set_index("model")

    def result(model: str) -> str:
        row = primary_rows.loc[model]
        return (
            f"{model_label(model)}: RMSE {float(row.rmse):.3f}, "
            f"R2 {float(row.r2):.3f}, "
            f"Spearman r {float(row.spearman_r):.3f}"
        )

    context_rows = primary_rows.reindex(
        [
            "tessera_1_single",
            "tessera_3_single",
            "tessera_5_single",
            "tessera_9_single",
        ]
    )
    best_context = str(context_rows["rmse"].idxmin())
    profile_rows = pooled.loc[
        pooled["target"].eq("three_layer_volume_entropy")
    ].sort_values("rmse")
    lines = [
        "# Cairngorms representation and target diagnostics",
        "",
        "## What was tested",
        "",
        (
            "The experiment used the same mature-forest units, five complete "
            "spatial holdouts, 1 km training buffers and valid-row cohort for "
            "every comparison. It separates four questions: whether TESSERA "
            "beats conventional annual remote-sensing predictors at equal "
            "spatial support; whether an MLP improves on gradient boosting; "
            "how much spatial context is useful; and whether predicting a "
            "documented three-layer vertical profile helps canopy Shannon "
            "prediction."
        ),
        "",
        "## Fair baseline comparisons",
        "",
        f"- {result('conventional_centre_single')}.",
        f"- {result('tessera_1_single')}.",
        f"- {result('conventional_5_single')}.",
        f"- {result('tessera_5_single')}.",
        "",
        comparison_sentence(comparisons.loc["fair_centre_single"]),
        comparison_sentence(comparisons.loc["fair_50m_single"]),
        "",
        "## Did the MLP add value?",
        "",
        f"- {result('tessera_5_hgb')}.",
        f"- {result('tessera_5_single')}.",
        f"- {result('conventional_5_hgb')}.",
        f"- {result('conventional_5_single')}.",
        "",
        comparison_sentence(
            comparisons.loc["mlp_vs_hgb_tessera_50m"]
        ),
        comparison_sentence(
            comparisons.loc["mlp_vs_hgb_conventional_50m"]
        ),
        "",
        "## How much spatial context was useful?",
        "",
    ]
    for model in [
        "tessera_1_single",
        "tessera_3_single",
        "tessera_5_single",
        "tessera_9_single",
    ]:
        lines.append(f"- {result(model)}.")
    lines.extend(
        [
            "",
            (
                f"The lowest primary-target RMSE among the frozen context "
                f"sizes came from {model_label(best_context)}."
            ),
            comparison_sentence(
                comparisons.loc["context_30m_vs_10m"]
            ),
            comparison_sentence(
                comparisons.loc["context_50m_vs_10m"]
            ),
            comparison_sentence(
                comparisons.loc["context_90m_vs_10m"]
            ),
            "",
            "## Did additional structural targets help?",
            "",
            comparison_sentence(
                comparisons.loc["multitask_effect_50m"]
            ),
            comparison_sentence(
                comparisons.loc["profile_aux_effect_50m"]
            ),
            comparison_sentence(
                comparisons.loc["profile_aux_effect_90m"]
            ),
            "",
            "## Was the vertical profile predictable?",
            "",
        ]
    )
    if profile_rows.empty:
        lines.append(
            "No valid vertical-profile predictions were produced."
        )
    else:
        for row in profile_rows.itertuples(index=False):
            lines.append(
                f"- {model_label(str(row.model).replace('__vertical_profile', ''))}: "
                f"three-layer entropy RMSE {float(row.rmse):.3f}, "
                f"R2 {float(row.r2):.3f}, "
                f"Spearman r {float(row.spearman_r):.3f}."
            )
    desired = float(config["interpretation_gates"]["desired_primary_r2"])
    best_primary = primary_rows.sort_values("r2", ascending=False).iloc[0]
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            (
                "The three layer-volume targets describe vegetation volume "
                "from 0-1 m, 1-10 m and above 10 m. They are useful structural "
                "profile targets, but they do not reconstruct the five-bin "
                "canopy Shannon raster supplied by the collaborator."
            ),
            "",
            (
                f"The best tested primary-target R2 was "
                f"{float(best_primary.r2):.3f}; the exploratory usefulness "
                f"target was {desired:.2f}. Results apply to spatially "
                "separated forest units within the Cairngorms and do not, by "
                f"themselves, establish transfer to another landscape."
            ),
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    make_figure(config)
    output_paths = [
        METRICS_PATH,
        POOLED_METRICS_PATH,
        COMPARISON_PATH,
        PREDICTIONS_PATH,
        FIGURE_PATH,
        REPORT_PATH,
    ]
    freeze = {
        "created_utc": utc_now(),
        "status": "complete",
        "config_sha256": sha256(CONFIG_PATH),
        "array_manifest_sha256": sha256(ARRAY_MANIFEST_PATH),
        "primary_target": primary,
        "best_primary_model": str(primary_rows["r2"].idxmax()),
        "best_primary_r2": float(primary_rows["r2"].max()),
        "best_context_model": best_context,
        "outputs": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in output_paths
        },
    }
    atomic_json(RESULT_FREEZE_PATH, freeze)
    print(
        f"report: best primary R2={freeze['best_primary_r2']:.3f}; "
        f"model={freeze['best_primary_model']}",
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
    arguments = parse_args()
    config = load_config()
    runners = {
        "preflight": run_preflight,
        "profiles": run_profiles,
        "patches": run_patches,
        "summaries": run_summaries,
        "conventional": run_conventional,
        "prepare": run_prepare,
        "evaluate": run_evaluate,
        "report": run_report,
    }
    for stage in STAGES:
        print(f"\n=== phase19 {stage} ===", flush=True)
        runners[stage](config)
        if stage == arguments.through:
            break
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        OSError,
        RuntimeError,
        ValueError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1)
