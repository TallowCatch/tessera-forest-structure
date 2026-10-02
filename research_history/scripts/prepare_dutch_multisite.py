#!/usr/bin/env python3
"""Prepare the frozen six-site AHN4 cohort and within-site spatial folds."""

from __future__ import annotations

import hashlib
import json
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
from scipy.spatial import cKDTree

from prepare_ahn4_replication_ahn4_cohort import (
    aligned_window,
    assign_balanced_folds,
    patch_bitmask,
    patch_masked_mean,
    patch_sum,
    read_aligned_dates,
    read_aligned_mask,
    read_float_window,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/dutch_multisite_transfer.yaml"


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "phase36_dutch_multisite_transfer"
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
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def read_genus_window(
    paths: list[Path],
    window: rasterio.windows.Window,
    reference: rasterio.DatasetReader,
) -> np.ndarray:
    """Mosaic nearest-neighbour genus classes into one AHN4-aligned window."""
    output = np.full((int(window.height), int(window.width)), 255, dtype=np.uint8)
    for path in paths:
        with rasterio.open(path) as source:
            with WarpedVRT(
                source,
                crs=reference.crs,
                transform=reference.transform,
                width=reference.width,
                height=reference.height,
                src_nodata=255,
                nodata=255,
                resampling=Resampling.nearest,
            ) as aligned:
                values = aligned.read(1, window=window, masked=False)
        valid = values != 255
        output[valid] = values[valid]
    return output


def mode_year(
    dates: np.ndarray, usable: np.ndarray, cells: int
) -> tuple[np.ndarray, np.ndarray]:
    years = dates // 10000
    candidates = [year for year in np.unique(years[usable]) if year > 0]
    if not candidates:
        shape = (dates.shape[0] // cells, dates.shape[1] // cells)
        return np.zeros(shape, dtype=np.int16), np.zeros(shape, dtype=np.float32)
    counts = np.stack(
        [patch_sum((usable & (years == year)).astype(np.int16), cells) for year in candidates]
    )
    winner = np.argmax(counts, axis=0)
    year_values = np.asarray(candidates, dtype=np.int16)[winner]
    winning_count = np.take_along_axis(counts, winner[None], axis=0)[0]
    usable_count = patch_sum(usable.astype(np.int16), cells)
    purity = np.zeros_like(winning_count, dtype=np.float32)
    np.divide(winning_count, usable_count, out=purity, where=usable_count > 0)
    return year_values, purity


def spatially_balanced_cap(
    frame: pd.DataFrame, maximum: int, seed: int
) -> pd.DataFrame:
    if len(frame) <= maximum:
        return frame.copy()
    rng = np.random.default_rng(seed)
    groups: dict[str, list[int]] = {}
    for block, positions in frame.groupby("sampling_block", sort=True).indices.items():
        values = np.asarray(positions, dtype=np.int64)
        rng.shuffle(values)
        groups[str(block)] = values.tolist()
    selected: list[int] = []
    ordered = sorted(groups)
    while len(selected) < maximum:
        active = False
        for block in ordered:
            if groups[block]:
                selected.append(groups[block].pop())
                active = True
                if len(selected) == maximum:
                    break
        if not active:
            break
    return frame.iloc[np.sort(np.asarray(selected, dtype=np.int64))].copy()


def write_site_folds(
    frame: pd.DataFrame,
    directory: Path,
    settings: dict[str, Any],
    seed_offset: int,
) -> dict[str, Any]:
    folds = int(settings["folds"])
    assignment = assign_balanced_folds(
        frame.rename(columns={"fold_block": "spatial_block"}),
        folds,
        int(settings["fold_seed"]) + seed_offset,
    )
    frame["spatial_fold"] = frame["fold_block"].map(assignment).astype(np.int8)
    coordinates = frame[["rd_x", "rd_y"]].to_numpy(dtype=np.float64)
    directory.mkdir(parents=True, exist_ok=True)
    records: dict[str, Any] = {}
    for fold in range(folds):
        test = np.flatnonzero(frame["spatial_fold"].to_numpy() == fold)
        candidate = np.flatnonzero(frame["spatial_fold"].to_numpy() != fold)
        tree = cKDTree(coordinates[test])
        distance, _ = tree.query(coordinates[candidate], k=1)
        train = candidate[distance >= float(settings["exclusion_buffer_m"])]
        buffered = candidate[distance < float(settings["exclusion_buffer_m"])]
        if len(train) < int(settings["minimum_train_rows"]) or len(test) < int(
            settings["minimum_test_rows"]
        ):
            raise RuntimeError(
                f"{frame.iloc[0]['site_code']} fold {fold} is too small: "
                f"train={len(train)}, test={len(test)}"
            )
        path = directory / f"fold_{fold}.npz"
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            train_row_ids=frame.iloc[train]["row_id"].to_numpy(dtype=np.int64),
            test_row_ids=frame.iloc[test]["row_id"].to_numpy(dtype=np.int64),
            buffered_row_ids=frame.iloc[buffered]["row_id"].to_numpy(dtype=np.int64),
        )
        temporary.replace(path)
        records[str(fold)] = {
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "buffered_rows": int(len(buffered)),
            "test_blocks": int(frame.iloc[test]["fold_block"].nunique()),
            "sha256": sha256(path),
        }
    return records


def prepare_site(
    definition: dict[str, str],
    boundary: gpd.GeoDataFrame,
    config: dict[str, Any],
    reference: rasterio.DatasetReader,
    metric_paths: dict[str, Path],
    genus_paths: list[Path],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    population = config["population"]
    genus = config["genus_map"]
    ahn4 = config["ahn4"]
    cells = int(population["output_support_m"] // population["lidar_cell_m"])
    window = aligned_window(boundary.total_bounds, reference, cells)
    transform = reference.window_transform(window)
    shape = (int(window.height), int(window.width))
    site_mask = rasterize(
        ((geometry, 1) for geometry in boundary.geometry),
        out_shape=shape,
        transform=transform,
        fill=0,
        all_touched=False,
        dtype="uint8",
    ).astype(bool)
    classes = read_genus_window(genus_paths, window, reference)
    conifer = np.isin(classes, np.asarray(genus["conifer_classes"], dtype=np.uint8))
    broadleaf = np.isin(
        classes, np.asarray(genus["broadleaf_classes"], dtype=np.uint8)
    )
    tree = conifer | broadleaf
    expected_group = conifer if definition["group"] == "conifer" else broadleaf
    metric_values = {
        name: read_float_window(path, window, reference)
        for name, path in metric_paths.items()
    }
    excluded = np.zeros(shape, dtype=bool)
    for path in ahn4["masks"]:
        excluded |= read_aligned_mask(ROOT / str(path), window, reference)
    dates = read_aligned_dates(ROOT / str(ahn4["flighttime"]), window, reference)
    finite = np.logical_and.reduce(
        [np.isfinite(values) for values in metric_values.values()]
    )
    usable = site_mask & tree & ~excluded & finite & (dates > 0)
    tree_count = patch_sum((site_mask & tree).astype(np.int16), cells)
    group_count = patch_sum((site_mask & expected_group).astype(np.int16), cells)
    usable_count = patch_sum(usable.astype(np.int16), cells)
    tree_fraction = tree_count / float(cells * cells)
    group_fraction = np.zeros_like(tree_fraction, dtype=np.float32)
    np.divide(group_count, tree_count, out=group_fraction, where=tree_count > 0)
    year, year_fraction = mode_year(dates, usable, cells)
    candidate = (
        (tree_fraction >= float(genus["minimum_tree_fraction"]))
        & (group_fraction >= float(genus["minimum_group_fraction_among_trees"]))
        & (
            usable_count
            >= np.ceil(
                tree_count * float(population["minimum_valid_lidar_fraction"])
            )
        )
        & (year_fraction >= float(population["minimum_single_year_fraction"]))
    )
    rows, columns = np.where(candidate)
    pixel_rows = rows * cells + (cells - 1) / 2
    pixel_columns = columns * cells + (cells - 1) / 2
    x = transform.c + (pixel_columns + 0.5) * transform.a
    y = transform.f + (pixel_rows + 0.5) * transform.e
    frame = pd.DataFrame(
        {
            "site_code": definition["code"],
            "site_name": definition["name"],
            "forest_group": definition["group"],
            "rd_x": x.astype(np.float64),
            "rd_y": y.astype(np.float64),
            "tree_fraction": tree_fraction[candidate].astype(np.float32),
            "group_fraction": group_fraction[candidate].astype(np.float32),
            "valid_lidar_cells": usable_count[candidate].astype(np.int16),
            "forest_pixel_mask": patch_bitmask(usable, cells)[candidate],
            "tessera_year": year[candidate].astype(np.int16),
            "ahn4_year_fraction": year_fraction[candidate].astype(np.float32),
        }
    )
    for name, values in metric_values.items():
        frame[name] = patch_masked_mean(values, usable, cells)[candidate].astype(
            np.float32
        )
    mature = (
        frame["ahn4_mean_height_m"] > float(population["minimum_mean_height_m"])
    ) & (
        frame["ahn4_pulse_penetration"]
        < float(population["maximum_pulse_penetration"])
    )
    frame = frame[mature].reset_index(drop=True)
    block_m = int(population["sampling_block_m"])
    frame["sampling_block"] = (
        np.floor(frame["rd_x"] / block_m).astype(np.int32).astype(str)
        + "_"
        + np.floor(frame["rd_y"] / block_m).astype(np.int32).astype(str)
    )
    before_cap = len(frame)
    frame = spatially_balanced_cap(
        frame,
        int(population["maximum_rows_per_site"]),
        int(population["sampling_seed"]) + sum(map(ord, definition["code"])),
    ).reset_index(drop=True)
    if len(frame) < int(population["minimum_rows_per_site"]):
        raise RuntimeError(
            f"{definition['name']} retains only {len(frame)} mature units"
        )
    return frame, {
        "site_code": definition["code"],
        "site_name": definition["name"],
        "forest_group": definition["group"],
        "candidate_units_before_maturity": int(candidate.sum()),
        "mature_units_before_cap": int(before_cap),
        "retained_units": int(len(frame)),
        "sampling_blocks": int(frame["sampling_block"].nunique()),
        "tessera_year_counts": {
            str(int(key)): int(value)
            for key, value in frame["tessera_year"].value_counts().sort_index().items()
        },
        "mean_tree_fraction": float(frame["tree_fraction"].mean()),
        "mean_group_fraction": float(frame["group_fraction"].mean()),
    }


def run() -> None:
    config = load_config()
    outputs = config["outputs"]
    freeze_path = ROOT / str(outputs["preparation_freeze"])
    if freeze_path.exists():
        raise RuntimeError("Phase 36 preparation is already frozen")
    site_source = ROOT / str(config["study"]["site_source"])
    sites = gpd.read_file(site_source).to_crs(config["study"]["analysis_crs"])
    metric_paths = {
        name: ROOT / str(path) for name, path in config["ahn4"]["metrics"].items()
    }
    reference_path = metric_paths["ahn4_mean_height_m"]
    genus_paths = sorted((ROOT / str(config["genus_map"]["tile_directory"])).glob("*.tif"))
    if not genus_paths:
        raise RuntimeError("No extracted European tree-genus tiles were found")
    boundaries: list[gpd.GeoDataFrame] = []
    parts: list[pd.DataFrame] = []
    records: list[dict[str, Any]] = []
    with rasterio.open(reference_path) as reference:
        for index, definition in enumerate(config["study"]["sites"]):
            selected = sites[sites["SITENAME"] == definition["name"]]
            if selected.empty:
                raise RuntimeError(f"Missing Natura 2000 site: {definition['name']}")
            boundary = selected.dissolve(by="SITENAME").reset_index()[["SITENAME", "geometry"]]
            boundary["site_code"] = definition["code"]
            boundary["forest_group"] = definition["group"]
            boundaries.append(boundary)
            frame, record = prepare_site(
                definition, boundary, config, reference, metric_paths, genus_paths
            )
            parts.append(frame)
            records.append(record)
            print(
                f"phase36 prepared {definition['name']}: {len(frame):,} units",
                flush=True,
            )
    cohort = pd.concat(parts, ignore_index=True)
    cohort.insert(0, "row_id", np.arange(len(cohort), dtype=np.int64))
    fold_block_m = int(config["evaluation"]["fold_block_m"])
    cohort["fold_block"] = (
        cohort["site_code"]
        + "_"
        + np.floor(cohort["rd_x"] / fold_block_m).astype(np.int32).astype(str)
        + "_"
        + np.floor(cohort["rd_y"] / fold_block_m).astype(np.int32).astype(str)
    )
    fold_records: dict[str, Any] = {}
    fold_root = ROOT / str(outputs["folds"])
    for index, definition in enumerate(config["study"]["sites"]):
        selected = cohort[cohort["site_code"] == definition["code"]].copy()
        records_for_site = write_site_folds(
            selected,
            fold_root / definition["code"],
            config["evaluation"],
            index * 100,
        )
        cohort.loc[selected.index, "spatial_fold"] = selected["spatial_fold"].to_numpy()
        fold_records[definition["code"]] = records_for_site
    cohort["spatial_fold"] = cohort["spatial_fold"].astype(np.int8)
    cohort_path = ROOT / str(outputs["cohort"])
    atomic_parquet(cohort_path, cohort)
    selected_path = ROOT / str(outputs["selected_sites"])
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    selected_path.unlink(missing_ok=True)
    gpd.GeoDataFrame(pd.concat(boundaries, ignore_index=True), crs=sites.crs).to_file(
        selected_path, layer="sites", driver="GPKG"
    )
    summary = pd.DataFrame(records)
    summary_path = ROOT / str(outputs["site_summary"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    freeze = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "sites_and_population_frozen_before_model_evaluation",
        "sites": records,
        "rows": int(len(cohort)),
        "folds": fold_records,
        "rules": {
            "minimum_tree_fraction": config["genus_map"]["minimum_tree_fraction"],
            "minimum_group_fraction_among_trees": config["genus_map"]["minimum_group_fraction_among_trees"],
            **config["population"],
        },
        "inputs": {
            "config": sha256(CONFIG_PATH),
            "site_source": sha256(site_source),
            "genus_tiles": {str(path.relative_to(ROOT)): sha256(path) for path in genus_paths},
            "ahn4_metrics": {str(path.relative_to(ROOT)): sha256(path) for path in metric_paths.values()},
        },
        "outputs": {
            "cohort": sha256(cohort_path),
            "selected_sites": sha256(selected_path),
            "site_summary": sha256(summary_path),
        },
    }
    atomic_json(freeze_path, freeze)
    print(f"phase36 cohort complete: {len(cohort):,} units", flush=True)


if __name__ == "__main__":
    run()
