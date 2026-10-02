#!/usr/bin/env python3
"""Run the frozen, resumable Cairngorms multi-target spatial-head study."""

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
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_conventional_predictors as conventional  # noqa: E402
import run_cairngorms_heterogeneity as phase17  # noqa: E402


CONFIG_PATH = ROOT / "configs/cairngorms_spatial_heads.yaml"
PROTOCOL_PATH = (
    ROOT
    / "metadata/project_config_phase18_cairngorms_spatial_heads_protocol_freeze.yaml"
)
PREFLIGHT_PATH = ROOT / "metadata/phase18_cairngorms_preflight.json"
PATCH_MANIFEST_PATH = ROOT / "metadata/phase18_cairngorms_patch_manifest.json"
PATCH_TILE_DIR = ROOT / "data/interim/phase18_cairngorms_patch_tiles"
PATCH_STREAM_DIR = ROOT / "data/interim/phase18_cairngorms_patch_stream"
ARRAY_DIR = ROOT / "data/interim/phase18_cairngorms_neural"
PATCH_QUANTIZED_PATH = ARRAY_DIR / "patch_quantized.npy"
PATCH_SCALES_PATH = ARRAY_DIR / "patch_scales.npy"
PATCH_VALID_PATH = ARRAY_DIR / "patch_valid.npy"
SUMMARY_ARRAY_PATH = ARRAY_DIR / "summary.npy"
CONVENTIONAL_ARRAY_PATH = ARRAY_DIR / "conventional.npy"
ROW_ID_ARRAY_PATH = ARRAY_DIR / "row_id.npy"
ARRAY_MANIFEST_PATH = ARRAY_DIR / "manifest.json"
CONVENTIONAL_PATH = (
    ROOT / "data/processed/phase18_cairngorms_conventional_predictors.parquet"
)
CONVENTIONAL_MANIFEST_PATH = (
    ROOT / "metadata/phase18_cairngorms_conventional_manifest.json"
)
CONVENTIONAL_INVENTORY_PATH = (
    ROOT / "metadata/phase18_cairngorms_conventional_inventory.csv"
)
FOLD_DIR = ROOT / "data/interim/phase18_cairngorms_evaluation_folds"
NEURAL_RESULT_DIR = ROOT / "data/interim/phase18_cairngorms_neural_results"
METRICS_PATH = ROOT / "outputs/tables/phase18_cairngorms_metrics.csv"
POOLED_METRICS_PATH = (
    ROOT / "outputs/tables/phase18_cairngorms_pooled_metrics.csv"
)
BOOTSTRAP_PATH = (
    ROOT / "outputs/tables/phase18_cairngorms_block_bootstrap.csv"
)
PREDICTIONS_PATH = (
    ROOT / "data/processed/phase18_cairngorms_oof_predictions.parquet"
)
FIGURE_PATH = (
    ROOT / "outputs/figures/phase18_cairngorms_spatial_heads.png"
)
REPORT_PATH = (
    ROOT / "outputs/reports/phase18_cairngorms_spatial_heads_results.md"
)
RESULT_FREEZE_PATH = (
    ROOT / "metadata/phase18_cairngorms_result_freeze.json"
)
NEURAL_WORKER_PATH = ROOT / "scripts/cairngorms_spatial_heads_neural_worker.py"

STAGES = ["preflight", "patches", "conventional", "prepare", "evaluate", "report"]
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
        "phase18_cairngorms_spatial_heads"
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
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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


def atomic_npy(values: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npy")
    np.save(temporary, values)
    temporary.replace(path)


def input_path(config: dict[str, Any], key: str) -> Path:
    return ROOT / config["frozen_inputs"][key]


def raw_target_names(config: dict[str, Any]) -> list[str]:
    return [str(value) for value in config["target_panel"]["raw"]]


def target_names(config: dict[str, Any]) -> list[str]:
    return raw_target_names(config) + [
        str(config["target_panel"]["height_adjusted_name"])
    ]


def summary_features(frame: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for prefix in ["centre", "mean", "std", "dx", "dy"]:
        columns.extend(phase17.feature_columns(frame, prefix))
    if len(columns) != 640:
        raise RuntimeError(f"Expected 640 TESSERA summary features, found {len(columns)}")
    return columns


def conventional_features(frame: pd.DataFrame) -> list[str]:
    columns = sorted(
        column
        for column in frame
        if column.startswith(("s1_", "s2_", "terrain_"))
        and not column.endswith("observation_count")
    )
    if not any(column.startswith("s1_") for column in columns):
        raise RuntimeError("Sentinel-1 predictors are missing")
    if not any(column.startswith("s2_") for column in columns):
        raise RuntimeError("Sentinel-2 predictors are missing")
    if not any(column.startswith("terrain_") for column in columns):
        raise RuntimeError("Terrain predictors are missing")
    return columns


def run_preflight(config: dict[str, Any]) -> None:
    if not PROTOCOL_PATH.exists():
        PROTOCOL_PATH.parent.mkdir(parents=True, exist_ok=True)
        PROTOCOL_PATH.write_text(
            CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8"
        )
    if CONFIG_PATH.read_bytes() != PROTOCOL_PATH.read_bytes():
        raise RuntimeError("The Phase 18 configuration differs from its frozen protocol")

    frozen = config["frozen_inputs"]
    checks = {
        "targets": ("targets_path", "targets_sha256"),
        "tessera_summaries": (
            "tessera_summaries_path",
            "tessera_summaries_sha256",
        ),
        "tessera_manifest": (
            "tessera_manifest_path",
            "tessera_manifest_sha256",
        ),
        "lidar_metrics": ("lidar_metrics_path", "lidar_metrics_sha256"),
    }
    records: dict[str, Any] = {}
    for label, (path_key, hash_key) in checks.items():
        path = ROOT / frozen[path_key]
        if not path.exists():
            raise RuntimeError(f"Missing frozen Phase 18 input: {path}")
        observed = sha256(path)
        if observed != str(frozen[hash_key]):
            raise RuntimeError(f"Frozen input hash changed: {path}")
        records[label] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": observed,
            "bytes": path.stat().st_size,
        }

    targets = pd.read_parquet(input_path(config, "targets_path"))
    summaries = pd.read_parquet(input_path(config, "tessera_summaries_path"))
    expected_ids = np.arange(len(targets), dtype=np.int64)
    if not np.array_equal(targets["row_id"].to_numpy(), expected_ids):
        raise RuntimeError("Phase 18 requires consecutive target row IDs")
    if not np.array_equal(summaries["row_id"].to_numpy(), expected_ids):
        raise RuntimeError("TESSERA summary rows do not match target rows")
    summary_features(summaries)
    missing_targets = sorted(set(raw_target_names(config)) - set(targets))
    if missing_targets:
        raise RuntimeError(f"Frozen targets are missing: {missing_targets}")
    if set(targets["spatial_region"]) != set(
        range(int(config["spatial_evaluation"]["region_count"]))
    ):
        raise RuntimeError("The frozen Phase 17 spatial regions are incomplete")

    neural_python = Path(str(config["neural"]["python"]))
    probe = subprocess.run(
        [
            str(neural_python),
            "-c",
            (
                "import torch; "
                "print(torch.__version__); "
                "print(torch.backends.mps.is_available())"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    free_bytes = shutil.disk_usage(ROOT).free
    minimum = (
        int(config["tessera_patches"]["minimum_free_space_after_download_bytes"])
        + 250_000_000
    )
    if free_bytes < minimum:
        raise RuntimeError(
            f"Phase 18 needs at least {minimum / 1e9:.2f} GB free; "
            f"{free_bytes / 1e9:.2f} GB is available"
        )
    record = {
        "created_utc": utc_now(),
        "config_sha256": sha256(CONFIG_PATH),
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "rows": len(targets),
        "spatial_regions": int(targets["spatial_region"].nunique()),
        "summary_features": 640,
        "raw_targets": raw_target_names(config),
        "neural_environment_probe": probe.stdout.strip().splitlines(),
        "free_space_bytes": free_bytes,
        "inputs": records,
    }
    if PREFLIGHT_PATH.exists():
        existing = json.loads(PREFLIGHT_PATH.read_text(encoding="utf-8"))
        for key in ["config_sha256", "protocol_sha256", "rows", "inputs"]:
            if existing[key] != record[key]:
                raise RuntimeError("The Phase 18 preflight checkpoint is stale")
        print("preflight: validated frozen inputs and neural environment", flush=True)
        return
    atomic_json(PREFLIGHT_PATH, record)
    print(
        f"preflight: {len(targets):,} rows, seven raw targets, "
        f"{free_bytes / 1e9:.2f} GB free",
        flush=True,
    )


def patch_points(
    targets: pd.DataFrame, transform: rasterio.Affine, cells: int
) -> dict[str, np.ndarray]:
    points = phase17.patch_sample_points(targets, transform, cells)
    centre = (cells - 1) / 2
    points["local_column"] = np.rint(
        points["offset_x"] + centre
    ).astype(np.int8)
    points["local_row"] = np.rint(
        centre - points["offset_y"]
    ).astype(np.int8)
    return points


def parse_tile_pair(pair: np.ndarray) -> tuple[float, float]:
    return round(float(pair[0]) / 100.0, 2), round(float(pair[1]) / 100.0, 2)


def run_patches(config: dict[str, Any]) -> None:
    paths = [PATCH_QUANTIZED_PATH, PATCH_SCALES_PATH, PATCH_VALID_PATH]
    if PATCH_MANIFEST_PATH.exists() and all(path.exists() for path in paths):
        manifest = json.loads(PATCH_MANIFEST_PATH.read_text(encoding="utf-8"))
        if (
            manifest["config_sha256"] == sha256(CONFIG_PATH)
            and manifest["target_sha256"]
            == sha256(input_path(config, "targets_path"))
            and all(
                manifest["output_sha256"][path.name] == sha256(path)
                for path in paths
            )
        ):
            print("patches: validated ordered 5 by 5 TESSERA checkpoint", flush=True)
            return
        raise RuntimeError("The Phase 18 TESSERA patch checkpoint is stale")

    targets = pd.read_parquet(input_path(config, "targets_path"))
    settings = config["tessera_patches"]
    cells = int(settings["patch_cells"])
    dimensions = int(settings["embedding_dimensions"])
    with rasterio.open(input_path(config, "lidar_metrics_path")) as dataset:
        points = patch_points(targets, dataset.transform, cells)

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
    registry_path, landmask_path = phase17.shared.tessera.ensure_registry(version)
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

    ARRAY_DIR.mkdir(parents=True, exist_ok=True)
    shape = (len(targets), dimensions, cells, cells)
    if not all(path.exists() for path in paths):
        if any(path.exists() for path in paths):
            raise RuntimeError("Only part of the Phase 18 patch array exists")
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
            raise RuntimeError("The resumable patch array has the wrong shape")

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
                raise RuntimeError(f"Stale Phase 18 patch tile marker: {name}")
            tile_records.append(marker)
            print(f"patches [{code + 1}/{len(tiles)}]: resumed {name}", flush=True)
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
                f"Insufficient disk space for {name}: {free / 1e9:.2f} GB "
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
            patch_id = points["patch_id"][source_indices].astype(np.int64)
            local_row = points["local_row"][source_indices].astype(np.int64)
            local_column = points["local_column"][source_indices].astype(np.int64)
            values = quantized[
                rows_index[valid_positions], columns_index[valid_positions]
            ].astype(np.int8)
            quantized_output[patch_id, :, local_row, local_column] = values
            scales_output[patch_id, local_row, local_column] = scales[
                rows_index[valid_positions], columns_index[valid_positions]
            ].astype(np.float16)
            valid_output[patch_id, local_row, local_column] = 1
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
    minimum = int(settings["minimum_valid_pixels_per_patch"])
    if int((valid_count >= minimum).sum()) < int(0.95 * len(targets)):
        raise RuntimeError("Too few ordered TESSERA patches passed the validity gate")
    centre = cells // 2
    centre_valid = np.asarray(valid_output[:, centre, centre], dtype=bool)
    if int(centre_valid.sum()) < int(0.95 * len(targets)):
        raise RuntimeError("Too few ordered TESSERA patches have valid centres")
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
        "minimum_valid_pixels_per_patch": minimum,
        "tile_records": tile_records,
        "output_sha256": {path.name: sha256(path) for path in paths},
    }
    atomic_json(PATCH_MANIFEST_PATH, manifest)
    shutil.rmtree(PATCH_STREAM_DIR, ignore_errors=True)
    print(
        f"patches: retained ordered context for "
        f"{manifest['valid_patch_count']:,}/{len(targets):,} units",
        flush=True,
    )


def run_conventional(config: dict[str, Any]) -> None:
    if CONVENTIONAL_PATH.exists() and CONVENTIONAL_MANIFEST_PATH.exists():
        manifest = json.loads(
            CONVENTIONAL_MANIFEST_PATH.read_text(encoding="utf-8")
        )
        if (
            manifest["config_sha256"] == sha256(CONFIG_PATH)
            and manifest["target_sha256"]
            == sha256(input_path(config, "targets_path"))
            and manifest["output_sha256"] == sha256(CONVENTIONAL_PATH)
        ):
            print("conventional: validated exact 2023 predictor checkpoint", flush=True)
            return
        raise RuntimeError("The Phase 18 conventional checkpoint is stale")

    targets = pd.read_parquet(
        input_path(config, "targets_path"),
        columns=["row_id", "bng_x", "bng_y", "longitude", "latitude"],
    )
    settings = config["conventional"]
    longitude = targets["longitude"].to_numpy(dtype=np.float64)
    latitude = targets["latitude"].to_numpy(dtype=np.float64)
    bbox = [
        float(longitude.min()),
        float(latitude.min()),
        float(longitude.max()),
        float(latitude.max()),
    ]
    catalog = conventional.open_catalog(str(settings["stac_api"]))
    year = int(config["study"]["survey_year"])
    s2_config = settings["sentinel_2"]
    s1_config = settings["sentinel_1"]
    terrain_config = settings["topography"]
    s2_items = conventional.select_sentinel2(
        conventional.collection_items(
            catalog, str(s2_config["collection"]), bbox, year
        ),
        s2_config,
    )
    s1_items = conventional.select_sentinel1(
        conventional.collection_items(
            catalog, str(s1_config["collection"]), bbox, year
        ),
        s1_config,
    )
    dem_items = conventional.collection_items(
        catalog, str(terrain_config["collection"]), bbox, None
    )
    print(
        f"conventional: selected {len(s2_items)} Sentinel-2, "
        f"{len(s1_items)} Sentinel-1, and {len(dem_items)} terrain items",
        flush=True,
    )
    s2_features, s2_inventory = conventional.extract_sentinel2(
        s2_items, longitude, latitude, s2_config
    )
    s1_features, s1_inventory = conventional.extract_sentinel1(
        s1_items, longitude, latitude, s1_config
    )
    terrain_input = targets.rename(
        columns={"bng_x": "x_epsg5070", "bng_y": "y_epsg5070"}
    )
    terrain_features, terrain_inventory = conventional.extract_topography(
        dem_items, terrain_input, bbox, terrain_config
    )
    output = targets[["row_id"]].copy()
    for name, values in sorted(
        {**terrain_features, **s2_features, **s1_features}.items()
    ):
        output[name] = values
    numeric = conventional_features(output)
    s1_count_columns = [
        column
        for column in output
        if column.startswith("s1_") and column.endswith("observation_count")
    ]
    if not s1_count_columns:
        raise RuntimeError("No Sentinel-1 observation-count columns were produced")
    valid = (
        output["s2_valid_observation_count"].to_numpy()
        >= int(s2_config["minimum_valid_observations_per_point"])
    )
    for column in s1_count_columns:
        valid &= output[column].to_numpy() >= int(
            s1_config["minimum_valid_observations_per_point_per_state"]
        )
    valid &= np.isfinite(output[numeric].to_numpy(dtype=np.float32)).all(axis=1)
    output["conventional_valid"] = valid
    atomic_parquet(output, CONVENTIONAL_PATH)
    inventory = pd.DataFrame(
        [
            {"source": "sentinel_2", **record}
            for record in s2_inventory
        ]
        + [
            {"source": "sentinel_1", **record}
            for record in s1_inventory
        ]
        + [
            {"source": "terrain", **record}
            for record in terrain_inventory
        ]
    )
    atomic_csv(inventory, CONVENTIONAL_INVENTORY_PATH)
    atomic_json(
        CONVENTIONAL_MANIFEST_PATH,
        {
            "created_utc": utc_now(),
            "config_sha256": sha256(CONFIG_PATH),
            "target_sha256": sha256(input_path(config, "targets_path")),
            "year": year,
            "spatial_support": settings["spatial_support"],
            "bbox_wgs84": bbox,
            "sentinel_2_item_ids": [item.id for item in s2_items],
            "sentinel_1_item_ids": [item.id for item in s1_items],
            "terrain_item_ids": [item.id for item in dem_items],
            "predictor_count": len(numeric),
            "valid_rows": int(valid.sum()),
            "output_sha256": sha256(CONVENTIONAL_PATH),
            "inventory_sha256": sha256(CONVENTIONAL_INVENTORY_PATH),
        },
    )
    print(
        f"conventional: valid predictors for {int(valid.sum()):,}/"
        f"{len(output):,} units",
        flush=True,
    )


def run_prepare(config: dict[str, Any]) -> None:
    required = [
        PATCH_QUANTIZED_PATH,
        PATCH_SCALES_PATH,
        PATCH_VALID_PATH,
        CONVENTIONAL_PATH,
    ]
    if not all(path.exists() for path in required):
        raise RuntimeError("Phase 18 preparation requires patch and conventional stages")
    output_paths = [
        SUMMARY_ARRAY_PATH,
        CONVENTIONAL_ARRAY_PATH,
        ROW_ID_ARRAY_PATH,
    ]
    if ARRAY_MANIFEST_PATH.exists() and all(path.exists() for path in output_paths):
        manifest = json.loads(ARRAY_MANIFEST_PATH.read_text(encoding="utf-8"))
        if (
            manifest["config_sha256"] == sha256(CONFIG_PATH)
            and all(
                manifest["array_sha256"][path.name] == sha256(path)
                for path in output_paths
            )
            and manifest["patch_manifest_sha256"] == sha256(PATCH_MANIFEST_PATH)
        ):
            print("prepare: validated neural and baseline arrays", flush=True)
            return
        raise RuntimeError("The Phase 18 prepared arrays are stale")

    targets = pd.read_parquet(input_path(config, "targets_path"))
    summaries = pd.read_parquet(input_path(config, "tessera_summaries_path"))
    conventional_frame = pd.read_parquet(CONVENTIONAL_PATH)
    if not targets[["row_id"]].equals(summaries[["row_id"]]):
        raise RuntimeError("TESSERA summary rows are misaligned")
    if not targets[["row_id"]].equals(conventional_frame[["row_id"]]):
        raise RuntimeError("Conventional predictor rows are misaligned")
    summary_columns = summary_features(summaries)
    conventional_columns = conventional_features(conventional_frame)
    atomic_npy(
        summaries[summary_columns].to_numpy(dtype=np.float32),
        SUMMARY_ARRAY_PATH,
    )
    atomic_npy(
        conventional_frame[conventional_columns].to_numpy(dtype=np.float32),
        CONVENTIONAL_ARRAY_PATH,
    )
    atomic_npy(targets["row_id"].to_numpy(dtype=np.int64), ROW_ID_ARRAY_PATH)
    manifest = {
        "created_utc": utc_now(),
        "config_sha256": sha256(CONFIG_PATH),
        "target_names": target_names(config),
        "summary_columns": summary_columns,
        "conventional_columns": conventional_columns,
        "rows": len(targets),
        "patch_manifest_sha256": sha256(PATCH_MANIFEST_PATH),
        "conventional_manifest_sha256": sha256(CONVENTIONAL_MANIFEST_PATH),
        "array_sha256": {
            path.name: sha256(path) for path in output_paths
        },
    }
    atomic_json(ARRAY_MANIFEST_PATH, manifest)
    print(
        f"prepare: wrote {len(summary_columns)} TESSERA and "
        f"{len(conventional_columns)} conventional features",
        flush=True,
    )


def spatial_weights(blocks: pd.Series) -> np.ndarray:
    counts = blocks.value_counts()
    weights = blocks.map(1.0 / counts).to_numpy(dtype=np.float64)
    return weights * len(weights) / weights.sum()


def fit_ridge(
    train_x: np.ndarray,
    train_y: np.ndarray,
    weights: np.ndarray,
    test_x: np.ndarray,
    alpha: float,
) -> np.ndarray:
    scaler = StandardScaler().fit(train_x, sample_weight=weights)
    model = Ridge(alpha=alpha).fit(
        scaler.transform(train_x), train_y, sample_weight=weights
    )
    return model.predict(scaler.transform(test_x))


def fit_nonlinear(
    train_x: np.ndarray,
    train_y: np.ndarray,
    weights: np.ndarray,
    test_x: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    settings = config["baseline_models"]["nonlinear"]
    model = HistGradientBoostingRegressor(
        learning_rate=float(settings["learning_rate"]),
        max_iter=int(settings["max_iter"]),
        max_leaf_nodes=int(settings["max_leaf_nodes"]),
        min_samples_leaf=int(settings["min_samples_leaf"]),
        l2_regularization=float(settings["l2_regularization"]),
        early_stopping=False,
        random_state=int(settings["random_state"]),
    )
    model.fit(train_x, train_y, sample_weight=weights)
    return model.predict(test_x)


def common_valid_mask(
    targets: pd.DataFrame,
    summaries: pd.DataFrame,
    conventional_frame: pd.DataFrame,
    patch_valid: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    mask = (
        summaries["tessera_context_valid"].to_numpy(dtype=bool)
        & summaries["tessera_centre_valid"].to_numpy(dtype=bool)
        & conventional_frame["conventional_valid"].to_numpy(dtype=bool)
        & (
            np.asarray(patch_valid).sum(axis=(1, 2))
            >= int(config["tessera_patches"]["minimum_valid_pixels_per_patch"])
        )
    )
    mask &= np.isfinite(
        summaries[summary_features(summaries)].to_numpy(dtype=np.float32)
    ).all(axis=1)
    mask &= np.isfinite(
        conventional_frame[
            conventional_features(conventional_frame)
        ].to_numpy(dtype=np.float32)
    ).all(axis=1)
    for target in raw_target_names(config):
        values = targets[target].to_numpy(dtype=np.float64)
        lower, upper = config["target_panel"]["valid_ranges"][target]
        mask &= np.isfinite(values) & (values >= lower) & (values <= upper)
    return mask


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
    selected_count = max(
        1,
        int(
            round(
                len(blocks)
                * float(
                    config["spatial_evaluation"]["validation_block_fraction"]
                )
            )
        ),
    )
    validation_blocks = set(
        rng.choice(blocks, size=selected_count, replace=False).tolist()
    )
    validation = train & frame["spatial_block"].isin(
        validation_blocks
    ).to_numpy()
    subtrain = train & ~validation
    return subtrain, validation


def save_fold_arrays(
    fold: int,
    target_matrix: np.ndarray,
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
    model: str,
    test_indices: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    predictions: list[np.ndarray] = []
    for seed in config["neural"]["seeds"]:
        subprocess.run(
            [
                str(config["neural"]["python"]),
                str(NEURAL_WORKER_PATH),
                "--model",
                model,
                "--fold",
                str(fold),
                "--seed",
                str(seed),
            ],
            cwd=ROOT,
            check=True,
        )
        path = NEURAL_RESULT_DIR / f"fold_{fold}_{model}_seed_{seed}.npz"
        with np.load(path) as result:
            if not np.array_equal(result["test_indices"], test_indices):
                raise RuntimeError(
                    f"Neural {model} test rows differ in fold {fold}"
                )
            predictions.append(result["predictions"].astype(np.float64))
    return np.mean(predictions, axis=0)


def metric_rows(
    observed_matrix: np.ndarray,
    prediction_matrix: np.ndarray,
    names: list[str],
    model: str,
    fold: int,
    train_rows: int,
) -> list[dict[str, Any]]:
    return [
        {
            "target": target,
            "model": model,
            "fold": fold,
            "train_rows": train_rows,
            "test_rows": len(observed_matrix),
            **phase17.metric_values(
                observed_matrix[:, index], prediction_matrix[:, index]
            ),
        }
        for index, target in enumerate(names)
    ]


def prediction_frame(
    targets: pd.DataFrame,
    test_indices: np.ndarray,
    observed_matrix: np.ndarray,
    prediction_matrix: np.ndarray,
    names: list[str],
    model: str,
    fold: int,
) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for index, target in enumerate(names):
        records.append(
            pd.DataFrame(
                {
                    "row_id": targets.iloc[test_indices]["row_id"].to_numpy(
                        dtype=np.int64
                    ),
                    "target": target,
                    "model": model,
                    "fold": fold,
                    "observed": observed_matrix[:, index],
                    "predicted": prediction_matrix[:, index],
                    "spatial_block": targets.iloc[test_indices][
                        "spatial_block"
                    ].to_numpy(),
                    "block_x": targets.iloc[test_indices]["block_x"].to_numpy(),
                    "block_y": targets.iloc[test_indices]["block_y"].to_numpy(),
                }
            )
        )
    return pd.concat(records, ignore_index=True)


def run_evaluate_fold(
    fold: int,
    targets: pd.DataFrame,
    summaries: pd.DataFrame,
    conventional_frame: pd.DataFrame,
    summary_array: np.ndarray,
    conventional_array: np.ndarray,
    patch_valid: np.ndarray,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_train, base_test, diagnostics = phase17.shared.buffered_fold_masks(
        targets, fold, config
    )
    valid = common_valid_mask(
        targets, summaries, conventional_frame, patch_valid, config
    )
    train = base_train & valid
    test = base_test & valid
    minimum_train = int(config["spatial_evaluation"]["minimum_training_rows"])
    minimum_test = int(config["spatial_evaluation"]["minimum_test_rows"])
    if train.sum() < minimum_train or test.sum() < minimum_test:
        raise RuntimeError(
            f"Fold {fold} has only {train.sum()} training and "
            f"{test.sum()} common test rows"
        )
    subtrain, validation = validation_split(targets, train, fold, config)
    names = target_names(config)
    raw_names = raw_target_names(config)
    target_matrix = np.full((len(targets), len(names)), np.nan, dtype=np.float32)
    target_matrix[:, : len(raw_names)] = targets[raw_names].to_numpy(
        dtype=np.float32
    )
    height_features = config["target_panel"]["height_adjustment_predictors"]
    height_model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=120,
        max_leaf_nodes=31,
        min_samples_leaf=60,
        l2_regularization=10.0,
        early_stopping=False,
        random_state=int(config["spatial_evaluation"]["seed"]) + fold,
    )
    train_weights = spatial_weights(targets.loc[train, "spatial_block"])
    height_model.fit(
        targets.loc[train, height_features].to_numpy(dtype=np.float32),
        targets.loc[train, raw_names[0]].to_numpy(dtype=np.float32),
        sample_weight=train_weights,
    )
    target_matrix[:, -1] = (
        targets[raw_names[0]].to_numpy(dtype=np.float32)
        - height_model.predict(
            targets[height_features].to_numpy(dtype=np.float32)
        ).astype(np.float32)
    )
    weights = np.ones(len(targets), dtype=np.float32)
    weights[train] = train_weights.astype(np.float32)
    fold_array_path = save_fold_arrays(
        fold,
        target_matrix,
        weights,
        train,
        subtrain,
        validation,
        test,
    )
    print(
        f"evaluate fold {fold}: train={int(train.sum()):,}; "
        f"validation blocks={targets.loc[validation, 'spatial_block'].nunique()}; "
        f"test={int(test.sum()):,}",
        flush=True,
    )

    train_indices = np.flatnonzero(train)
    test_indices = np.flatnonzero(test)
    observed = target_matrix[test_indices].astype(np.float64)
    y_train = target_matrix[train_indices].astype(np.float64)
    models: dict[str, np.ndarray] = {
        "training_region_mean": np.repeat(
            np.average(y_train, axis=0, weights=train_weights)[None, :],
            len(test_indices),
            axis=0,
        )
    }
    alpha = float(config["baseline_models"]["ridge_alpha"])
    summary_train = np.asarray(summary_array[train_indices], dtype=np.float32)
    summary_test = np.asarray(summary_array[test_indices], dtype=np.float32)
    conventional_train = np.asarray(
        conventional_array[train_indices], dtype=np.float32
    )
    conventional_test = np.asarray(
        conventional_array[test_indices], dtype=np.float32
    )
    baseline_specs = [
        (
            "tessera_summary_ridge",
            summary_train,
            summary_test,
            "ridge",
        ),
        (
            "tessera_summary_hist_gradient_boosting",
            summary_train,
            summary_test,
            "nonlinear",
        ),
        (
            "sentinel1_sentinel2_terrain_ridge",
            conventional_train,
            conventional_test,
            "ridge",
        ),
        (
            "sentinel1_sentinel2_terrain_hist_gradient_boosting",
            conventional_train,
            conventional_test,
            "nonlinear",
        ),
    ]
    for model_name, train_x, test_x, kind in baseline_specs:
        print(f"evaluate fold {fold}: fit {model_name}", flush=True)
        columns: list[np.ndarray] = []
        for target_index in range(len(names)):
            if kind == "ridge":
                predicted = fit_ridge(
                    train_x,
                    y_train[:, target_index],
                    train_weights,
                    test_x,
                    alpha,
                )
            else:
                predicted = fit_nonlinear(
                    train_x,
                    y_train[:, target_index],
                    train_weights,
                    test_x,
                    config,
                )
            columns.append(predicted)
        models[model_name] = np.column_stack(columns)
    models["tessera_summary_mlp"] = neural_predictions(
        fold, "mlp", test_indices, config
    )
    models["tessera_patch_cnn"] = neural_predictions(
        fold, "cnn", test_indices, config
    )

    metric_records: list[dict[str, Any]] = []
    prediction_records: list[pd.DataFrame] = []
    for model_name, predicted in models.items():
        metric_records.extend(
            metric_rows(
                observed,
                predicted,
                names,
                model_name,
                fold,
                int(train.sum()),
            )
        )
        prediction_records.append(
            prediction_frame(
                targets,
                test_indices,
                observed,
                predicted,
                names,
                model_name,
                fold,
            )
        )
    metrics = pd.DataFrame(metric_records)
    for key, value in diagnostics.items():
        metrics[f"fold_{key}"] = value
    metrics["common_valid_rows"] = int(valid.sum())
    metrics["fold_array_sha256"] = sha256(fold_array_path)
    return metrics, pd.concat(prediction_records, ignore_index=True)


def block_bootstrap(
    predictions: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    rng = np.random.default_rng(int(config["uncertainty"]["seed"]))
    replicates = int(config["uncertainty"]["block_bootstrap_replicates"])
    records: list[dict[str, Any]] = []
    for (target, model), frame in predictions.groupby(["target", "model"]):
        grouped = {
            block: group[["observed", "predicted"]].to_numpy(dtype=np.float64)
            for block, group in frame.groupby("spatial_block")
        }
        blocks = np.asarray(sorted(grouped))
        draws = np.empty((replicates, 3), dtype=np.float64)
        for replicate in range(replicates):
            selected = rng.choice(blocks, size=len(blocks), replace=True)
            values = np.concatenate([grouped[block] for block in selected])
            metric = phase17.metric_values(values[:, 0], values[:, 1])
            draws[replicate] = [
                metric["rmse"],
                metric["r2"],
                metric["spearman_r"],
            ]
        records.append(
            {
                "target": target,
                "model": model,
                "spatial_blocks": len(blocks),
                "replicates": replicates,
                **{
                    f"{name}_ci_low": float(np.nanquantile(draws[:, index], 0.025))
                    for index, name in enumerate(["rmse", "r2", "spearman_r"])
                },
                **{
                    f"{name}_ci_high": float(np.nanquantile(draws[:, index], 0.975))
                    for index, name in enumerate(["rmse", "r2", "spearman_r"])
                },
            }
        )
    return pd.DataFrame(records)


def run_evaluate(config: dict[str, Any]) -> None:
    complete = [METRICS_PATH, POOLED_METRICS_PATH, BOOTSTRAP_PATH, PREDICTIONS_PATH]
    if all(path.exists() for path in complete):
        print("evaluate: resumed complete multi-target evaluation", flush=True)
        return
    if not ARRAY_MANIFEST_PATH.exists():
        raise RuntimeError("Phase 18 prepared-array manifest is missing")
    targets = pd.read_parquet(input_path(config, "targets_path"))
    summaries = pd.read_parquet(input_path(config, "tessera_summaries_path"))
    conventional_frame = pd.read_parquet(CONVENTIONAL_PATH)
    summary_array = np.load(SUMMARY_ARRAY_PATH, mmap_mode="r")
    conventional_array = np.load(CONVENTIONAL_ARRAY_PATH, mmap_mode="r")
    patch_valid = np.load(PATCH_VALID_PATH, mmap_mode="r")
    input_hashes = {
        "config_sha256": sha256(CONFIG_PATH),
        "array_manifest_sha256": sha256(ARRAY_MANIFEST_PATH),
    }
    FOLD_DIR.mkdir(parents=True, exist_ok=True)
    metric_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    fold_count = int(config["spatial_evaluation"]["region_count"])
    for fold in range(fold_count):
        metric_path = FOLD_DIR / f"fold_{fold}_metrics.csv"
        prediction_path = FOLD_DIR / f"fold_{fold}_predictions.parquet"
        manifest_path = FOLD_DIR / f"fold_{fold}_manifest.json"
        if all(path.exists() for path in [metric_path, prediction_path, manifest_path]):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest["inputs"] != input_hashes
                or manifest["metrics_sha256"] != sha256(metric_path)
                or manifest["predictions_sha256"] != sha256(prediction_path)
            ):
                raise RuntimeError(f"Stale Phase 18 evaluation fold {fold}")
            metrics = pd.read_csv(metric_path)
            predictions = pd.read_parquet(prediction_path)
            print(f"evaluate [{fold + 1}/{fold_count}]: resumed fold {fold}", flush=True)
        else:
            print(f"evaluate [{fold + 1}/{fold_count}]: run fold {fold}", flush=True)
            metrics, predictions = run_evaluate_fold(
                fold,
                targets,
                summaries,
                conventional_frame,
                summary_array,
                conventional_array,
                patch_valid,
                config,
            )
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
        metric_frames.append(metrics)
        prediction_frames.append(predictions)

    metrics = pd.concat(metric_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
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
                **phase17.metric_values(
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
    atomic_csv(block_bootstrap(predictions, config), BOOTSTRAP_PATH)
    print(
        f"evaluate: wrote {len(predictions):,} predictions for "
        f"{pooled['target'].nunique()} targets and "
        f"{pooled['model'].nunique()} models",
        flush=True,
    )


def make_figure(config: dict[str, Any]) -> None:
    pooled = pd.read_csv(POOLED_METRICS_PATH)
    models = [
        "sentinel1_sentinel2_terrain_hist_gradient_boosting",
        "tessera_summary_hist_gradient_boosting",
        "tessera_summary_mlp",
        "tessera_patch_cnn",
    ]
    labels = {
        "sentinel1_sentinel2_terrain_hist_gradient_boosting": "S1/S2 + terrain",
        "tessera_summary_hist_gradient_boosting": "TESSERA summaries",
        "tessera_summary_mlp": "TESSERA MLP",
        "tessera_patch_cnn": "TESSERA patch CNN",
    }
    target_labels = {
        "mean_canopy_shannon_50m": "Shannon entropy",
        "between_cell_height_sd_50m": "Between-cell height SD",
        "mean_within_cell_height_sd_50m": "Within-cell height SD",
        "mean_gap_fraction_50m": "Gap fraction",
        "mean_height_50m": "Mean height",
        "mean_p95_height_50m": "P95 height",
        "mean_cover_50m": "Canopy cover",
        "mean_canopy_shannon_50m_height_adjusted": "Height-adjusted entropy",
    }
    matrix = (
        pooled[pooled["model"].isin(models)]
        .pivot(index="target", columns="model", values="r2")
        .reindex(index=target_names(config), columns=models)
    )
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    image = axes[0].imshow(
        matrix.to_numpy(),
        cmap="RdYlGn",
        vmin=-0.25,
        vmax=0.8,
        aspect="auto",
    )
    axes[0].set_xticks(
        np.arange(len(models)),
        [labels[model] for model in models],
        rotation=35,
        ha="right",
    )
    axes[0].set_yticks(
        np.arange(len(matrix)),
        [target_labels[target] for target in matrix.index],
    )
    axes[0].set_title("a  Spatially withheld performance", loc="left")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix.iloc[row, column]
            axes[0].text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )
    figure.colorbar(image, ax=axes[0], label=r"Pooled $R^2$", fraction=0.045)

    primary = str(config["success"]["primary_target"])
    scoped = pooled[
        pooled["target"].eq(primary) & pooled["model"].isin(models)
    ].set_index("model").reindex(models)
    axes[1].bar(
        np.arange(len(models)),
        scoped["r2"],
        color=["#4472C4", "#7A858C", "#2A9D8F", "#D95F02"],
    )
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[1].axhline(
        float(config["success"]["desired_primary_r2"]),
        color="#555555",
        linestyle="--",
        linewidth=1,
    )
    axes[1].set_xticks(
        np.arange(len(models)),
        [labels[model] for model in models],
        rotation=35,
        ha="right",
    )
    axes[1].set_ylabel(r"Pooled spatially withheld $R^2$")
    axes[1].set_title("b  Canopy Shannon entropy", loc="left")
    axes[1].grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axes[1].set_axisbelow(True)
    figure.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=220, bbox_inches="tight")
    plt.close(figure)


def run_report(config: dict[str, Any]) -> None:
    if REPORT_PATH.exists() and RESULT_FREEZE_PATH.exists():
        freeze = json.loads(RESULT_FREEZE_PATH.read_text(encoding="utf-8"))
        if freeze["report_sha256"] == sha256(REPORT_PATH):
            print("report: validated complete Phase 18 result", flush=True)
            return
    pooled = pd.read_csv(POOLED_METRICS_PATH)
    metrics = pd.read_csv(METRICS_PATH)
    primary = str(config["success"]["primary_target"])
    relevant_models = [
        "sentinel1_sentinel2_terrain_hist_gradient_boosting",
        "tessera_summary_hist_gradient_boosting",
        "tessera_summary_mlp",
        "tessera_patch_cnn",
    ]
    primary_rows = (
        pooled[
            pooled["target"].eq(primary)
            & pooled["model"].isin(relevant_models)
        ]
        .set_index("model")
        .reindex(relevant_models)
    )
    cnn = primary_rows.loc["tessera_patch_cnn"]
    mlp = primary_rows.loc["tessera_summary_mlp"]
    cnn_improvement = (float(mlp.rmse) - float(cnn.rmse)) / float(mlp.rmse)
    positive_folds = int(
        (
            metrics["target"].eq(primary)
            & metrics["model"].eq("tessera_patch_cnn")
            & (metrics["r2"] > 0)
        ).sum()
    )
    success = (
        float(cnn.r2) >= float(config["success"]["desired_primary_r2"])
        and positive_folds
        >= int(config["success"]["minimum_positive_r2_folds"])
        and (cnn_improvement > 0)
    )
    model_labels = {
        "sentinel1_sentinel2_terrain_hist_gradient_boosting": (
            "Sentinel-1, Sentinel-2, and terrain"
        ),
        "tessera_summary_hist_gradient_boosting": (
            "TESSERA summaries with gradient boosting"
        ),
        "tessera_summary_mlp": "TESSERA summary MLP",
        "tessera_patch_cnn": "ordered-patch TESSERA CNN",
    }
    lines = [
        "# Cairngorms multi-target spatial-head results",
        "",
        "## Design",
        "",
        (
            "All models used the same mature-forest cohort and five complete "
            "spatial holdouts with a 1 km training exclusion buffer. Neural "
            "models learned the eight outcomes jointly, while all metrics "
            "were calculated separately for each lidar target."
        ),
        "",
        "## Canopy Shannon entropy",
        "",
    ]
    for model in relevant_models:
        row = primary_rows.loc[model]
        lines.append(
            f"- {model_labels[model]}: RMSE {row.rmse:.3f}, "
            f"R2 {row.r2:.3f}, Spearman r {row.spearman_r:.3f}."
        )
    lines.extend(
        [
            "",
            (
                f"The ordered patch CNN changed RMSE by "
                f"{cnn_improvement * 100:.1f}% relative to the summary MLP "
                f"and had positive R2 in {positive_folds}/5 folds."
            ),
            "",
            "## Predeclared usefulness gate",
            "",
            f"Overall gate passed: **{'yes' if success else 'no'}**.",
            "",
            "## Target-specific results",
            "",
            "| Target | Best tested model | RMSE | R2 | Spearman r |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for target, group in pooled.groupby("target", sort=False):
        tested = group[group["model"].isin(relevant_models)].sort_values("rmse")
        best = tested.iloc[0]
        lines.append(
            f"| {target} | {model_labels[str(best.model)]} | "
            f"{best.rmse:.3f} | {best.r2:.3f} | {best.spearman_r:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            (
                "This is a spatial generalisation test within the Cairngorms "
                "landscape. It establishes whether local TESSERA arrangement "
                "adds predictive information for lidar structure; it does not "
                "establish transfer to another landscape or directly measure "
                "biodiversity."
            ),
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    make_figure(config)
    freeze = {
        "created_utc": utc_now(),
        "status": "complete",
        "config_sha256": sha256(CONFIG_PATH),
        "primary_target": primary,
        "primary_cnn_r2": float(cnn.r2),
        "primary_cnn_rmse": float(cnn.rmse),
        "cnn_rmse_change_from_summary_mlp_fraction": cnn_improvement,
        "success_gate_passed": bool(success),
        "outputs": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [
                METRICS_PATH,
                POOLED_METRICS_PATH,
                BOOTSTRAP_PATH,
                PREDICTIONS_PATH,
                FIGURE_PATH,
            ]
        },
        "report_sha256": sha256(REPORT_PATH),
    }
    atomic_json(RESULT_FREEZE_PATH, freeze)
    print(
        f"report: CNN primary R2={cnn.r2:.3f}; "
        f"gate={'passed' if success else 'not passed'}",
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
        "patches": run_patches,
        "conventional": run_conventional,
        "prepare": run_prepare,
        "evaluate": run_evaluate,
        "report": run_report,
    }
    for stage in STAGES:
        print(f"\n=== phase18 {stage} ===", flush=True)
        runners[stage](config)
        if stage == arguments.through:
            break
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1)
