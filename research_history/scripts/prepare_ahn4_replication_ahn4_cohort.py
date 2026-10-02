#!/usr/bin/env python3
"""Build the frozen Savelsbos AHN4 cohort and spatial folds."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import yaml
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window, from_bounds
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/ahn4_replication_ahn4_deciduous.yaml"


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "phase26_ahn4_deciduous"
    ]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def aligned_window(
    bounds: np.ndarray, dataset: rasterio.DatasetReader, cells: int
) -> Window:
    raw = from_bounds(*bounds, transform=dataset.transform)
    row_start = max(0, (int(math.floor(raw.row_off)) // cells) * cells)
    column_start = max(0, (int(math.floor(raw.col_off)) // cells) * cells)
    row_stop = min(
        dataset.height,
        ((int(math.ceil(raw.row_off + raw.height)) + cells - 1) // cells)
        * cells,
    )
    column_stop = min(
        dataset.width,
        ((int(math.ceil(raw.col_off + raw.width)) + cells - 1) // cells)
        * cells,
    )
    row_stop -= (row_stop - row_start) % cells
    column_stop -= (column_stop - column_start) % cells
    if row_stop <= row_start or column_stop <= column_start:
        raise RuntimeError("The selected forest does not intersect the AHN4 grid")
    return Window(
        column_start,
        row_start,
        column_stop - column_start,
        row_stop - row_start,
    )


def verify_metric_grid(
    dataset: rasterio.DatasetReader,
    reference: rasterio.DatasetReader,
    name: str,
) -> None:
    if (
        dataset.crs != reference.crs
        or dataset.transform != reference.transform
        or dataset.shape != reference.shape
        or dataset.count != 1
    ):
        raise RuntimeError(f"AHN4 metric grid changed for {name}")


def read_float_window(
    path: Path,
    window: Window,
    reference: rasterio.DatasetReader,
) -> np.ndarray:
    with rasterio.open(path) as dataset:
        verify_metric_grid(dataset, reference, path.name)
        values = dataset.read(1, window=window, masked=True).astype(np.float32)
    return values.filled(np.nan)


def read_aligned_mask(
    path: Path,
    window: Window,
    reference: rasterio.DatasetReader,
) -> np.ndarray:
    with rasterio.open(path) as source:
        with WarpedVRT(
            source,
            crs=reference.crs,
            transform=reference.transform,
            width=reference.width,
            height=reference.height,
            resampling=Resampling.nearest,
        ) as aligned:
            values = aligned.read(1, window=window, masked=True)
    # These files mark excluded cells with positive values. Areas outside a
    # sparse mask (for example power lines) are valid and therefore filled 0.
    return np.asarray(values.filled(0) > 0, dtype=bool)


def read_aligned_dates(
    path: Path,
    window: Window,
    reference: rasterio.DatasetReader,
) -> np.ndarray:
    with rasterio.open(path) as source:
        with WarpedVRT(
            source,
            crs=reference.crs,
            transform=reference.transform,
            width=reference.width,
            height=reference.height,
            resampling=Resampling.nearest,
        ) as aligned:
            values = aligned.read(1, window=window, masked=True)
    return values.filled(0).astype(np.int32)


def patch_sum(values: np.ndarray, cells: int) -> np.ndarray:
    rows = values.shape[0] // cells
    columns = values.shape[1] // cells
    return values.reshape(rows, cells, columns, cells).sum(axis=(1, 3))


def patch_masked_mean(
    values: np.ndarray,
    valid: np.ndarray,
    cells: int,
) -> np.ndarray:
    total = patch_sum(np.where(valid, values, 0.0), cells)
    count = patch_sum(valid.astype(np.int16), cells)
    output = np.full(total.shape, np.nan, dtype=np.float32)
    np.divide(total, count, out=output, where=count > 0)
    return output


def patch_bitmask(values: np.ndarray, cells: int) -> np.ndarray:
    rows = values.shape[0] // cells
    columns = values.shape[1] // cells
    grid = values.reshape(rows, cells, columns, cells).transpose(0, 2, 1, 3)
    powers = (1 << np.arange(cells * cells, dtype=np.uint32)).reshape(
        1, 1, cells, cells
    )
    return np.sum(grid.astype(np.uint32) * powers, axis=(2, 3), dtype=np.uint32)


def assign_balanced_folds(
    frame: pd.DataFrame, folds: int, seed: int
) -> dict[str, int]:
    counts = frame.groupby("spatial_block").size().rename("count").reset_index()
    rng = np.random.default_rng(seed)
    counts["tie_break"] = rng.random(len(counts))
    counts = counts.sort_values(
        ["count", "tie_break", "spatial_block"], ascending=[False, True, True]
    )
    totals = np.zeros(folds, dtype=np.int64)
    assignment: dict[str, int] = {}
    for row in counts.itertuples(index=False):
        fold = int(np.argmin(totals))
        assignment[str(row.spatial_block)] = fold
        totals[fold] += int(row.count)
    return assignment


def build_spatial_folds(
    frame: pd.DataFrame,
    output_directory: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    settings = config["evaluation"]
    folds = int(settings["folds"])
    assignment = assign_balanced_folds(
        frame, folds, int(settings["fold_seed"])
    )
    frame["spatial_fold"] = frame["spatial_block"].map(assignment).astype(np.int8)
    coordinates = frame[["rd_x", "rd_y"]].to_numpy(dtype=np.float64)
    buffer_m = float(settings["exclusion_buffer_m"])
    minimum_train = int(settings["minimum_train_rows"])
    minimum_test = int(settings["minimum_test_rows"])
    output_directory.mkdir(parents=True, exist_ok=True)
    records: dict[str, Any] = {}
    for fold in range(folds):
        test = np.flatnonzero(frame["spatial_fold"].to_numpy() == fold)
        candidate_train = np.flatnonzero(frame["spatial_fold"].to_numpy() != fold)
        tree = cKDTree(coordinates[test])
        distance, _ = tree.query(coordinates[candidate_train], k=1)
        train = candidate_train[distance >= buffer_m]
        excluded = candidate_train[distance < buffer_m]
        if len(train) < minimum_train or len(test) < minimum_test:
            raise RuntimeError(
                f"Fold {fold} retains train={len(train)}, test={len(test)}; "
                f"minimums are {minimum_train} and {minimum_test}"
            )
        destination = output_directory / f"fold_{fold}.npz"
        temporary = destination.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            train_indices=train.astype(np.int64),
            test_indices=test.astype(np.int64),
            buffered_indices=excluded.astype(np.int64),
        )
        temporary.replace(destination)
        records[str(fold)] = {
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "buffered_training_rows": int(len(excluded)),
            "test_blocks": int(frame.iloc[test]["spatial_block"].nunique()),
            "file_sha256": sha256(destination),
        }
    return records


def run() -> None:
    config = load_config()
    sources = config["sources"]
    population = config["population"]
    outputs = config["outputs"]
    selected_path = ROOT / str(outputs["selected_site"])
    cohort_path = ROOT / str(outputs["cohort"])
    fold_directory = ROOT / str(outputs["folds"])
    freeze_path = ROOT / str(outputs["design_freeze"])
    if freeze_path.exists():
        raise RuntimeError(
            "The Phase 26 design has already been executed; remove it only for a "
            "documented fresh replication"
        )

    broadleaf = gpd.read_file(
        selected_path, layer="top10nl_broadleaf"
    ).to_crs(config["study"]["analysis_crs"])
    site = gpd.read_file(selected_path, layer="natura_site").to_crs(
        config["study"]["analysis_crs"]
    )
    cells = int(population["output_support_m"] // population["lidar_cell_m"])
    if cells != 5:
        raise RuntimeError("Phase 26 is frozen to 5 x 5 AHN4 cells per unit")

    metric_paths = {
        name: ROOT / str(path)
        for name, path in dict(sources["ahn4_metrics"]).items()
    }
    reference_path = metric_paths["ahn4_mean_height_m"]
    with rasterio.open(reference_path) as reference:
        if str(reference.crs) != str(config["study"]["analysis_crs"]):
            raise RuntimeError(f"Unexpected AHN4 CRS: {reference.crs}")
        if tuple(abs(value) for value in reference.res) != (
            float(population["lidar_cell_m"]),
            float(population["lidar_cell_m"]),
        ):
            raise RuntimeError(f"Unexpected AHN4 resolution: {reference.res}")
        window = aligned_window(broadleaf.total_bounds, reference, cells)
        transform = reference.window_transform(window)
        shape = (int(window.height), int(window.width))
        forest = rasterize(
            ((geometry, 1) for geometry in broadleaf.geometry),
            out_shape=shape,
            transform=transform,
            fill=0,
            all_touched=False,
            dtype="uint8",
        ).astype(bool)
        site_mask = rasterize(
            ((geometry, 1) for geometry in site.geometry),
            out_shape=shape,
            transform=transform,
            fill=0,
            all_touched=False,
            dtype="uint8",
        ).astype(bool)
        metric_values = {
            name: read_float_window(path, window, reference)
            for name, path in metric_paths.items()
        }
        building_road_water = read_aligned_mask(
            ROOT / str(sources["ahn4_building_road_water_mask"]),
            window,
            reference,
        )
        powerline = read_aligned_mask(
            ROOT / str(sources["ahn4_powerline_mask"]), window, reference
        )
        na_mask = read_aligned_mask(
            ROOT / str(sources["ahn4_na_mask"]), window, reference
        )
        dates = read_aligned_dates(
            ROOT / str(sources["ahn4_flighttime"]), window, reference
        )

    expected_dates = np.asarray(
        config["study"]["expected_ahn4_dates"], dtype=np.int32
    )
    correct_date = np.isin(dates, expected_dates)
    target_finite = np.logical_and.reduce(
        [np.isfinite(values) for values in metric_values.values()]
    )
    excluded = building_road_water | powerline | na_mask
    usable = forest & site_mask & correct_date & ~excluded & target_finite

    forest_count = patch_sum((forest & site_mask).astype(np.int16), cells)
    usable_count = patch_sum(usable.astype(np.int16), cells)
    forest_fraction = forest_count / float(cells * cells)
    required_forest = float(population["required_broadleaf_fraction"])
    required_valid_fraction = float(population["minimum_valid_lidar_fraction"])
    candidate = forest_fraction >= required_forest
    sufficient_valid = usable_count >= np.ceil(
        forest_count * required_valid_fraction
    )
    retained = candidate & sufficient_valid

    rows, columns = np.where(retained)
    local_pixel_rows = rows * cells + (cells - 1) / 2
    local_pixel_columns = columns * cells + (cells - 1) / 2
    x = (
        transform.c
        + (local_pixel_columns + 0.5) * transform.a
        + (local_pixel_rows + 0.5) * transform.b
    )
    y = (
        transform.f
        + (local_pixel_columns + 0.5) * transform.d
        + (local_pixel_rows + 0.5) * transform.e
    )
    frame = pd.DataFrame(
        {
            "raster_patch_row": rows.astype(np.int32),
            "raster_patch_column": columns.astype(np.int32),
            "rd_x": x.astype(np.float64),
            "rd_y": y.astype(np.float64),
            "broadleaf_fraction": forest_fraction[retained].astype(np.float32),
            "valid_lidar_cells": usable_count[retained].astype(np.int16),
            "broadleaf_cells": forest_count[retained].astype(np.int16),
            "broadleaf_pixel_mask": patch_bitmask(
                forest & site_mask, cells
            )[retained],
            "ahn4_date": np.asarray(
                [int(expected_dates[0])] * int(retained.sum()), dtype=np.int32
            ),
        }
    )
    for name, values in metric_values.items():
        aggregate = patch_masked_mean(values, usable, cells)
        frame[name] = aggregate[retained].astype(np.float32)

    mature = (
        frame["ahn4_mean_height_m"]
        > float(population["minimum_mean_height_m"])
    ) & (
        frame["ahn4_pulse_penetration"]
        < float(population["maximum_pulse_penetration"])
    )
    frame = frame[mature].reset_index(drop=True)
    minimum_rows = int(config["evaluation"]["minimum_cohort_rows"])
    if len(frame) < minimum_rows:
        raise RuntimeError(
            f"Only {len(frame)} units pass frozen quality rules; minimum is {minimum_rows}"
        )
    frame.insert(0, "row_id", np.arange(len(frame), dtype=np.int64))
    block_m = int(config["evaluation"]["spatial_block_m"])
    block_x = np.floor(frame["rd_x"] / block_m).astype(np.int32)
    block_y = np.floor(frame["rd_y"] / block_m).astype(np.int32)
    frame["spatial_block"] = block_x.astype(str) + "_" + block_y.astype(str)
    evaluation_block_m = int(config["evaluation"]["evaluation_block_m"])
    evaluation_x = np.floor(frame["rd_x"] / evaluation_block_m).astype(np.int32)
    evaluation_y = np.floor(frame["rd_y"] / evaluation_block_m).astype(np.int32)
    frame["evaluation_block"] = (
        evaluation_x.astype(str) + "_" + evaluation_y.astype(str)
    )
    fold_records = build_spatial_folds(frame, fold_directory, config)

    atomic_parquet(frame, cohort_path)
    profile = {
        "created_utc": utc_now(),
        "status": "frozen_design_executed_before_model_evaluation",
        "site": {
            "code": str(config["study"]["expected_site_code"]),
            "name": str(config["study"]["expected_site_name"]),
        },
        "rules": {
            "output_support_m": int(population["output_support_m"]),
            "required_broadleaf_fraction": required_forest,
            "minimum_valid_lidar_fraction": required_valid_fraction,
            "minimum_mean_height_m": float(population["minimum_mean_height_m"]),
            "maximum_pulse_penetration": float(
                population["maximum_pulse_penetration"]
            ),
            "expected_dates": expected_dates.tolist(),
            "spatial_block_m": block_m,
            "exclusion_buffer_m": int(config["evaluation"]["exclusion_buffer_m"]),
        },
        "counts": {
            "candidate_broadleaf_units": int(candidate.sum()),
            "sufficient_lidar_units": int((candidate & sufficient_valid).sum()),
            "mature_forest_units": int(len(frame)),
            "spatial_blocks": int(frame["spatial_block"].nunique()),
        },
        "folds": fold_records,
        "inputs": {
            "config_sha256": sha256(CONFIG_PATH),
            "selected_site_sha256": sha256(selected_path),
            "metric_files": {
                name: {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size}
                for name, path in metric_paths.items()
            },
            "mask_files": {
                key: {
                    "path": str(ROOT / str(sources[key])),
                    "sha256": sha256(ROOT / str(sources[key])),
                }
                for key in [
                    "ahn4_flighttime",
                    "ahn4_building_road_water_mask",
                    "ahn4_powerline_mask",
                    "ahn4_na_mask",
                ]
            },
        },
        "outputs": {
            "cohort": str(cohort_path.relative_to(ROOT)),
            "cohort_sha256": sha256(cohort_path),
            "fold_directory": str(fold_directory.relative_to(ROOT)),
        },
    }
    atomic_json(freeze_path, profile)
    print(
        f"phase26 cohort: retained {len(frame):,} mature broadleaf units "
        f"across {frame['spatial_block'].nunique()} blocks",
        flush=True,
    )


if __name__ == "__main__":
    run()
