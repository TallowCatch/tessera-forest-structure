#!/usr/bin/env python3
"""Screen Dutch forests and freeze the expanded AHN4 transfer cohort."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import yaml
from affine import Affine
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.vrt import WarpedVRT
from scipy.spatial import cKDTree

from prepare_ahn4_replication_ahn4_cohort import aligned_window, assign_balanced_folds
from prepare_dutch_multisite import (
    atomic_json,
    atomic_parquet,
    prepare_site,
    sha256,
    write_site_folds,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/dutch_expanded_transfer.yaml"


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "phase37_dutch_expanded_transfer"
    ]


def site_code(name: str) -> str:
    value = name.lower().replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return value[:64]


def site_seed_offset(code: str) -> int:
    return sum((index + 1) * ord(value) for index, value in enumerate(code)) % 100000


def read_genus_coarse(
    paths: list[Path],
    window: rasterio.windows.Window,
    shape: tuple[int, int],
    reference: rasterio.DatasetReader,
) -> np.ndarray:
    output = np.full(shape, 255, dtype=np.uint8)
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
                values = aligned.read(
                    1,
                    window=window,
                    out_shape=shape,
                    masked=False,
                    resampling=Resampling.nearest,
                )
        valid = values != 255
        output[valid] = values[valid]
    return output


def screen_boundary(
    name: str,
    boundary: gpd.GeoDataFrame,
    config: dict[str, Any],
    reference: rasterio.DatasetReader,
    genus_paths: list[Path],
) -> dict[str, Any]:
    settings = config["study"]["screening"]
    genus = config["genus_map"]
    factor = max(1, int(round(float(settings["resolution_m"]) / abs(reference.res[0]))))
    window = aligned_window(boundary.total_bounds, reference, factor)
    shape = (int(window.height) // factor, int(window.width) // factor)
    transform = reference.window_transform(window) * Affine.scale(factor, factor)
    mask = rasterize(
        ((geometry, 1) for geometry in boundary.geometry),
        out_shape=shape,
        transform=transform,
        fill=0,
        all_touched=False,
        dtype="uint8",
    ).astype(bool)
    classes = read_genus_coarse(genus_paths, window, shape, reference)
    conifer = mask & np.isin(classes, np.asarray(genus["conifer_classes"], dtype=np.uint8))
    broadleaf = mask & np.isin(classes, np.asarray(genus["broadleaf_classes"], dtype=np.uint8))
    conifer_count = int(conifer.sum())
    broadleaf_count = int(broadleaf.sum())
    tree_count = conifer_count + broadleaf_count
    if tree_count == 0:
        dominant_group = "none"
        dominant_share = 0.0
        dominant_pixels = 0
    elif conifer_count >= broadleaf_count:
        dominant_group = "conifer"
        dominant_share = conifer_count / tree_count
        dominant_pixels = conifer_count
    else:
        dominant_group = "broadleaf"
        dominant_share = broadleaf_count / tree_count
        dominant_pixels = broadleaf_count
    centroid = boundary.geometry.union_all().centroid
    return {
        "site_name": name,
        "site_code": site_code(name),
        "dominant_group": dominant_group,
        "dominant_share": float(dominant_share),
        "screen_tree_pixels": tree_count,
        "screen_dominant_pixels": dominant_pixels,
        "screen_resolution_m": int(settings["resolution_m"]),
        "centroid_x": float(centroid.x),
        "centroid_y": float(centroid.y),
    }


def fold_feasible(
    frame: pd.DataFrame,
    code: str,
    settings: dict[str, Any],
    seed_offset: int,
) -> bool:
    work = frame.copy()
    block_m = int(settings["fold_block_m"])
    work["fold_block"] = (
        code
        + "_"
        + np.floor(work["rd_x"] / block_m).astype(np.int32).astype(str)
        + "_"
        + np.floor(work["rd_y"] / block_m).astype(np.int32).astype(str)
    )
    assignment = assign_balanced_folds(
        work.rename(columns={"fold_block": "spatial_block"}),
        int(settings["folds"]),
        int(settings["fold_seed"]) + seed_offset,
    )
    work["spatial_fold"] = work["fold_block"].map(assignment).astype(np.int8)
    coordinates = work[["rd_x", "rd_y"]].to_numpy(dtype=np.float64)
    for fold in range(int(settings["folds"])):
        test = np.flatnonzero(work["spatial_fold"].to_numpy() == fold)
        candidate = np.flatnonzero(work["spatial_fold"].to_numpy() != fold)
        if len(test) < int(settings["minimum_test_rows"]):
            return False
        distance, _ = cKDTree(coordinates[test]).query(coordinates[candidate], k=1)
        train = candidate[distance >= float(settings["exclusion_buffer_m"])]
        if len(train) < int(settings["minimum_train_rows"]):
            return False
    return True


def feasible_seed_offset(
    frame: pd.DataFrame,
    code: str,
    settings: dict[str, Any],
    preferred: list[int],
) -> int | None:
    candidates = preferred + [site_seed_offset(code)] + list(range(0, 10000, 100))
    seen: set[int] = set()
    for offset in candidates:
        if offset in seen:
            continue
        seen.add(offset)
        if fold_feasible(frame, code, settings, offset):
            return offset
    return None


def select_group(
    group: str,
    anchors: list[str],
    prepared: dict[str, dict[str, Any]],
    boundaries: dict[str, gpd.GeoDataFrame],
    target: int,
    minimum_distance_km: float,
    already_selected: list[str],
) -> tuple[list[str], dict[str, str]]:
    selected = [name for name in anchors if name in prepared]
    reasons = {name: "fixed_anchor" for name in selected}
    candidates = [
        name
        for name, value in prepared.items()
        if value["definition"]["group"] == group and name not in selected
    ]
    candidates.sort(
        key=lambda name: (
            -int(prepared[name]["record"]["mature_units_before_cap"]),
            name,
        )
    )

    def overlaps(name: str) -> bool:
        coordinates = set(
            map(
                tuple,
                prepared[name]["frame"][["rd_x", "rd_y"]].to_numpy(dtype=np.float64),
            )
        )
        for other in already_selected + selected:
            if other == name:
                continue
            other_coordinates = set(
                map(
                    tuple,
                    prepared[other]["frame"][["rd_x", "rd_y"]].to_numpy(dtype=np.float64),
                )
            )
            if coordinates.intersection(other_coordinates):
                return True
        return False

    def separated(name: str) -> bool:
        if not selected:
            return True
        centre = np.asarray(
            [prepared[name]["record"]["centroid_x"], prepared[name]["record"]["centroid_y"]]
        )
        distances = []
        for other in selected:
            other_centre = np.asarray(
                [prepared[other]["record"]["centroid_x"], prepared[other]["record"]["centroid_y"]]
            )
            distances.append(float(np.linalg.norm(centre - other_centre) / 1000.0))
        return min(distances) >= minimum_distance_km

    for name in candidates:
        if len(selected) >= target:
            break
        if not overlaps(name) and separated(name):
            selected.append(name)
            reasons[name] = "eligible_area_and_separation"
    for name in candidates:
        if len(selected) >= target:
            break
        if name not in selected and not overlaps(name):
            selected.append(name)
            reasons[name] = "distance_relaxed_to_reach_target"
    return selected, reasons


def run() -> None:
    config = load_config()
    outputs = config["outputs"]
    freeze_path = ROOT / str(outputs["preparation_freeze"])
    if freeze_path.exists():
        raise RuntimeError("Phase 37 preparation is already frozen")
    site_source = ROOT / str(config["study"]["site_source"])
    raw_sites = gpd.read_file(site_source).to_crs(config["study"]["analysis_crs"])
    dissolved = raw_sites.dissolve(by="SITENAME").reset_index()[["SITENAME", "geometry"]]
    boundary_lookup = {
        str(row.SITENAME): gpd.GeoDataFrame(
            {"SITENAME": [str(row.SITENAME)]}, geometry=[row.geometry], crs=dissolved.crs
        )
        for row in dissolved.itertuples(index=False)
    }
    metric_paths = {
        name: ROOT / str(path) for name, path in config["ahn4"]["metrics"].items()
    }
    genus_paths = sorted((ROOT / str(config["genus_map"]["tile_directory"])).glob("*.tif"))
    if not genus_paths:
        raise RuntimeError("No European tree-genus tiles were found")
    anchor_lookup = {value["name"]: value for value in config["study"]["anchors"]}
    screen_records: list[dict[str, Any]] = []
    with rasterio.open(metric_paths["ahn4_mean_height_m"]) as reference:
        for index, (name, boundary) in enumerate(sorted(boundary_lookup.items())):
            try:
                record = screen_boundary(name, boundary, config, reference, genus_paths)
                record["screen_status"] = "complete"
            except Exception as error:
                record = {
                    "site_name": name,
                    "site_code": site_code(name),
                    "dominant_group": "none",
                    "dominant_share": 0.0,
                    "screen_tree_pixels": 0,
                    "screen_dominant_pixels": 0,
                    "screen_status": f"failed: {error}",
                }
            if name in anchor_lookup:
                record["dominant_group"] = anchor_lookup[name]["group"]
                record["anchor"] = True
            else:
                record["anchor"] = False
            screen_records.append(record)
            if (index + 1) % 20 == 0:
                print(f"phase37 screened {index + 1}/{len(boundary_lookup)} regions", flush=True)
        screen = pd.DataFrame(screen_records)
        minimum_share = float(config["study"]["screening"]["minimum_site_group_share"])
        pool_size = int(config["study"]["screening"]["candidate_pool_per_group"])
        pool_names = set(anchor_lookup)
        for group in ["conifer", "broadleaf"]:
            candidates = screen[
                (screen["dominant_group"] == group)
                & (screen["dominant_share"] >= minimum_share)
                & (screen["screen_status"] == "complete")
            ].sort_values(["screen_dominant_pixels", "site_name"], ascending=[False, True])
            pool_names.update(candidates.head(pool_size)["site_name"].astype(str))
        prepared: dict[str, dict[str, Any]] = {}
        for index, name in enumerate(sorted(pool_names)):
            if name not in boundary_lookup:
                continue
            screen_row = screen.loc[screen["site_name"] == name].iloc[0]
            group = anchor_lookup[name]["group"] if name in anchor_lookup else str(screen_row["dominant_group"])
            definition = {
                "name": name,
                "code": anchor_lookup[name]["code"] if name in anchor_lookup else site_code(name),
                "group": group,
            }
            try:
                frame, record = prepare_site(
                    definition,
                    boundary_lookup[name],
                    config,
                    reference,
                    metric_paths,
                    genus_paths,
                )
                record["centroid_x"] = float(frame["rd_x"].mean())
                record["centroid_y"] = float(frame["rd_y"].mean())
                anchor_preferred = []
                if name in anchor_lookup:
                    anchor_preferred = [
                        list(anchor_lookup).index(name) * 100
                    ]
                selected_seed = feasible_seed_offset(
                    frame,
                    definition["code"],
                    config["evaluation"],
                    anchor_preferred,
                )
                if selected_seed is None:
                    raise RuntimeError("no feasible buffered five-fold assignment was found")
                record["fold_seed_offset"] = int(selected_seed)
                prepared[name] = {
                    "definition": definition,
                    "frame": frame,
                    "record": record,
                }
                screen.loc[screen["site_name"] == name, "exact_status"] = "eligible"
                screen.loc[screen["site_name"] == name, "eligible_units"] = len(frame)
                print(f"phase37 eligible {name}: {len(frame):,} {group} units", flush=True)
            except Exception as error:
                screen.loc[screen["site_name"] == name, "exact_status"] = f"excluded: {error}"
                print(f"phase37 excluded {name}: {error}", flush=True)

    screen_path = ROOT / str(outputs["candidate_screen"])
    screen_path.parent.mkdir(parents=True, exist_ok=True)
    screen.to_csv(screen_path, index=False)
    target = int(config["study"]["target_sites_per_group"])
    selected_by_group: dict[str, list[str]] = {}
    previously_selected: list[str] = []
    selection_reasons: dict[str, str] = {}
    for group in ["conifer", "broadleaf"]:
        anchors = [value["name"] for value in config["study"]["anchors"] if value["group"] == group]
        local, reasons = select_group(
            group,
            anchors,
            prepared,
            boundary_lookup,
            target,
            float(config["study"]["screening"]["minimum_centroid_distance_km"]),
            previously_selected,
        )
        selected_by_group[group] = local
        previously_selected.extend(local)
        selection_reasons.update(reasons)
    balanced_count = min(
        target,
        len(selected_by_group["conifer"]),
        len(selected_by_group["broadleaf"]),
    )
    if balanced_count < int(config["study"]["minimum_sites_per_group"]):
        raise RuntimeError(
            "Insufficient balanced sites: "
            f"conifer={len(selected_by_group['conifer'])}, "
            f"broadleaf={len(selected_by_group['broadleaf'])}"
        )
    selected_names = (
        selected_by_group["conifer"][:balanced_count]
        + selected_by_group["broadleaf"][:balanced_count]
    )

    parts: list[pd.DataFrame] = []
    records: list[dict[str, Any]] = []
    selected_boundaries: list[gpd.GeoDataFrame] = []
    for order, name in enumerate(selected_names):
        value = prepared[name]
        frame = value["frame"].copy()
        record = dict(value["record"])
        record["selection_order"] = order
        record["selection_reason"] = selection_reasons[name]
        parts.append(frame)
        records.append(record)
        boundary = boundary_lookup[name].copy()
        boundary["site_code"] = value["definition"]["code"]
        boundary["forest_group"] = value["definition"]["group"]
        boundary["selection_order"] = order
        selected_boundaries.append(boundary)
        screen.loc[screen["site_name"] == name, "selected"] = True
        screen.loc[screen["site_name"] == name, "selection_reason"] = selection_reasons[name]
    screen["selected"] = screen["selected"].fillna(False).astype(bool)
    screen.to_csv(screen_path, index=False)

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
    for index, record in enumerate(records):
        selected = cohort[cohort["site_code"] == record["site_code"]].copy()
        fold_records[record["site_code"]] = write_site_folds(
            selected,
            fold_root / record["site_code"],
            config["evaluation"],
            int(record["fold_seed_offset"]),
        )
        cohort.loc[selected.index, "spatial_fold"] = selected["spatial_fold"].to_numpy()
    cohort["spatial_fold"] = cohort["spatial_fold"].astype(np.int8)
    cohort_path = ROOT / str(outputs["cohort"])
    atomic_parquet(cohort_path, cohort)
    selected_path = ROOT / str(outputs["selected_sites"])
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    selected_path.unlink(missing_ok=True)
    gpd.GeoDataFrame(pd.concat(selected_boundaries, ignore_index=True), crs=dissolved.crs).to_file(
        selected_path, layer="sites", driver="GPKG"
    )
    summary = pd.DataFrame(records).sort_values("selection_order")
    summary_path = ROOT / str(outputs["site_summary"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    freeze = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "sites_population_and_folds_frozen_before_model_evaluation",
        "selection_rule": {
            **config["study"]["screening"],
            "target_sites_per_group": target,
            "balanced_sites_per_group": balanced_count,
            "anchors": config["study"]["anchors"],
        },
        "sites": records,
        "rows": int(len(cohort)),
        "folds": fold_records,
        "inputs": {
            "config": sha256(CONFIG_PATH),
            "site_source": sha256(site_source),
            "genus_tiles": {str(path.relative_to(ROOT)): sha256(path) for path in genus_paths},
            "ahn4_metrics": {str(path.relative_to(ROOT)): sha256(path) for path in metric_paths.values()},
        },
        "outputs": {
            "candidate_screen": sha256(screen_path),
            "cohort": sha256(cohort_path),
            "selected_sites": sha256(selected_path),
            "site_summary": sha256(summary_path),
        },
    }
    atomic_json(freeze_path, freeze)
    counts = summary.groupby("forest_group")["site_code"].count().to_dict()
    print(
        f"phase37 cohort complete: {len(cohort):,} units; "
        f"conifer={counts.get('conifer', 0)}, broadleaf={counts.get('broadleaf', 0)}",
        flush=True,
    )


if __name__ == "__main__":
    run()
