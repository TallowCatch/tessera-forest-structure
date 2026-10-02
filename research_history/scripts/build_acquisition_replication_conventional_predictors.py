#!/usr/bin/env python3
"""Acquire the fair Sentinel-2 and terrain predictors for Phase 13 targets."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_conventional_predictors as base  # noqa: E402
import build_fair_baselines_conventional_predictors as phase8  # noqa: E402
import run_context_height_footprint_height_transfer as common  # noqa: E402


PROTOCOL_PATH = ROOT / "metadata/project_config_phase13_temporal_extension_protocol_freeze.yaml"
TARGET_PATH = ROOT / "data/processed/phase13_tessera_aligned.parquet"
TARGET_FREEZE_PATH = ROOT / "metadata/phase13_combined_target_free_freeze.json"
TESSERA_FREEZE_PATH = ROOT / "metadata/phase13_tessera_alignment_freeze.json"
OUTPUT_PATH = ROOT / "data/processed/phase13_conventional_predictors.parquet"
INVENTORY_PATH = ROOT / "metadata/phase13_conventional_inventory.csv"
FREEZE_PATH = ROOT / "metadata/phase13_conventional_freeze.json"
CHECKPOINT_DIR = ROOT / "data/interim/phase13_conventional_site_chunks"
PHASE13_S2_RADIOMETRY = "earthsearch_per_item_boa_offset_handling"

KEYS = phase8.KEYS
TARGET_COLUMNS = KEYS + ["target_year", "longitude", "latitude", "x_epsg5070", "y_epsg5070"]
MODEL_FEATURES = phase8.TERRAIN_FEATURES + phase8.S2_FEATURES
OUTPUT_COLUMNS = KEYS + MODEL_FEATURES + ["s2_valid_observation_count"]


def expansion_s2_terrain_mixed_years(
    targets: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, list[str]]]:
    catalog = phase8.pystac_client.Client.open(phase8.EARTH_SEARCH)
    predictor = config["conventional_predictors"]
    s2_config = predictor["sentinel_2"]
    terrain_config = dict(predictor["topography"])
    terrain_config["collection"] = "cop-dem-glo-30"
    base.GDAL_OPTIONS["AWS_NO_SIGN_REQUEST"] = "YES"
    output_frames = []
    inventory: list[dict[str, Any]] = []
    selected_items: dict[str, list[str]] = {}
    for site, frame in targets.groupby("site_id", sort=True):
        years = frame["target_year"].unique()
        if len(years) != 1:
            raise RuntimeError(f"Target year is ambiguous for {site}")
        year = int(years[0])
        checkpoint = CHECKPOINT_DIR / f"{site}_{year}_s2_terrain.parquet"
        checkpoint_metadata = CHECKPOINT_DIR / f"{site}_{year}_s2_terrain.json"
        if checkpoint.exists() and checkpoint_metadata.exists():
            metadata = json.loads(checkpoint_metadata.read_text(encoding="utf-8"))
            if (
                metadata.get("sentinel_2_radiometry")
                not in {phase8.EARTH_SEARCH_S2_RADIOMETRY, PHASE13_S2_RADIOMETRY}
                or int(metadata.get("target_year", -1)) != year
            ):
                raise RuntimeError(f"Stale mixed-year Sentinel checkpoint for {site}")
            output_frames.append(pd.read_parquet(checkpoint))
            checkpoint_inventory = metadata["inventory"]
            for record in checkpoint_inventory:
                if record.get("collection") == "sentinel-2-l2a":
                    record.setdefault(
                        "sentinel_2_radiometry",
                        "earthsearch_boa_offset_already_applied",
                    )
            inventory.extend(checkpoint_inventory)
            selected_items[site] = metadata["selected_item_ids"]
            print(f"Resumed Sentinel-2/terrain checkpoint for {site} {year}")
            continue
        longitude = frame["longitude"].to_numpy(dtype=np.float64)
        latitude = frame["latitude"].to_numpy(dtype=np.float64)
        bbox = [
            float(longitude.min()),
            float(latitude.min()),
            float(longitude.max()),
            float(latitude.max()),
        ]
        candidates = base.collection_items(catalog, "sentinel-2-l2a", bbox, year)
        scenes = phase8.select_sentinel2(
            candidates,
            float(s2_config["maximum_item_cloud_cover_percent"]),
            int(s2_config["scenes_per_month_per_mgrs_tile"]),
        )
        dem_items = base.collection_items(catalog, "cop-dem-glo-30", bbox, None)
        print(
            f"Selected {len(scenes)} Sentinel-2 {year} scenes and "
            f"{len(dem_items)} DEM tiles for {site}"
        )
        s2, s2_inventory = phase8.extract_sentinel2(
            scenes,
            longitude,
            latitude,
            s2_config,
            allow_unapplied_boa_offset=True,
        )
        terrain, terrain_inventory = base.extract_topography(
            dem_items, frame, bbox, terrain_config
        )
        site_output = frame[KEYS].copy()
        for name, values in {**terrain, **s2}.items():
            site_output[name] = values
        site_inventory = [
            {"site_id": site, "target_year": year, **record} for record in s2_inventory
        ] + [
            {
                "site_id": site,
                "target_year": year,
                "source": "earth-search",
                "collection": "cop-dem-glo-30",
                "item_id": record["item_id"],
                "datetime": record["datetime"],
                "month": None,
                "selection_group": "dem_tiles",
                "polarization": None,
                "raster_id": None,
            }
            for record in terrain_inventory
        ]
        selected_items[site] = [item.id for item in scenes] + [item.id for item in dem_items]
        base.write_parquet_atomic(site_output, checkpoint)
        phase8.write_json(
            checkpoint_metadata,
            {
                "target_year": year,
                "sentinel_2_radiometry": PHASE13_S2_RADIOMETRY,
                "selected_item_ids": selected_items[site],
                "inventory": site_inventory,
            },
        )
        output_frames.append(site_output)
        inventory.extend(site_inventory)
    return pd.concat(output_frames, ignore_index=True), inventory, selected_items


def main() -> int:
    protected = [OUTPUT_PATH, INVENTORY_PATH, FREEZE_PATH]
    existing = [str(path.relative_to(ROOT)) for path in protected if path.exists()]
    if existing:
        raise RuntimeError(f"Phase 13 conventional outputs exist; refusing to overwrite: {existing}")
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))[
        "phase13_temporal_extension"
    ]
    target_freeze = json.loads(TARGET_FREEZE_PATH.read_text(encoding="utf-8"))
    tessera_freeze = json.loads(TESSERA_FREEZE_PATH.read_text(encoding="utf-8"))
    if tessera_freeze["freeze_basis"]["gedi_freeze_id"] != target_freeze["freeze_id"]:
        raise RuntimeError("Phase 13 TESSERA and GEDI freezes disagree")
    sites = sorted(target_freeze["freeze_basis"]["combined_unique_sites"])
    if len(sites) < int(protocol["combined_replication_gate"]["minimum_unique_forests"]):
        raise RuntimeError("Phase 13 target-free site gate did not pass")
    targets = pd.read_parquet(TARGET_PATH, columns=TARGET_COLUMNS)
    if sorted(targets["site_id"].unique()) != sites or targets.duplicated(KEYS).any():
        raise RuntimeError("Unexpected Phase 13 conventional predictor cohort")
    if "fhd_normal" in targets.columns:
        raise RuntimeError("FHD entered conventional predictor acquisition")

    config = yaml.safe_load(base.CONFIG_PATH.read_text(encoding="utf-8"))
    phase8.CHECKPOINT_DIR = CHECKPOINT_DIR
    predictors, inventory, selected_items = expansion_s2_terrain_mixed_years(targets, config)
    predictors = predictors[OUTPUT_COLUMNS].sort_values(KEYS).reset_index(drop=True)
    if len(predictors) != len(targets) or predictors.duplicated(KEYS).any():
        raise RuntimeError("Phase 13 conventional predictor population is incomplete")
    if predictors["s2_valid_observation_count"].lt(6).any():
        raise RuntimeError("A Phase 13 target has fewer than six valid Sentinel-2 observations")
    if not np.isfinite(predictors[MODEL_FEATURES].to_numpy(dtype=np.float64)).all():
        raise RuntimeError("Phase 13 Sentinel-2 or terrain predictors contain non-finite values")
    medians = predictors.groupby("site_id")[[
        "s2_blue_median", "s2_red_median", "s2_nir_median", "s2_ndvi_median"
    ]].median()
    if not medians[["s2_blue_median", "s2_red_median", "s2_nir_median"]].apply(
        lambda values: values.between(-0.05, 1.0)
    ).all().all():
        raise RuntimeError("Phase 13 Sentinel-2 reflectance failed radiometric QA")
    if not medians["s2_ndvi_median"].between(-1.0, 1.0).all():
        raise RuntimeError("Phase 13 Sentinel-2 NDVI failed radiometric QA")

    inventory_frame = pd.DataFrame(inventory).drop_duplicates().sort_values(
        ["site_id", "collection", "datetime", "item_id"], na_position="last"
    )
    base.write_parquet_atomic(predictors, OUTPUT_PATH)
    base.write_csv_atomic(inventory_frame, INVENTORY_PATH)
    if CHECKPOINT_DIR.exists():
        shutil.rmtree(CHECKPOINT_DIR)
    freeze_basis = {
        "protocol_sha256": common.sha256(PROTOCOL_PATH),
        "target_freeze_id": target_freeze["freeze_id"],
        "tessera_freeze_id": tessera_freeze["freeze_id"],
        "script_sha256": common.sha256(Path(__file__).resolve()),
        "target_input_sha256": common.sha256(TARGET_PATH),
        "target_columns_read": TARGET_COLUMNS,
        "target_fhd_read": False,
        "sites": sites,
        "site_years": target_freeze["freeze_basis"]["site_years"],
        "site_rows": {
            site: int(count) for site, count in predictors.groupby("site_id").size().items()
        },
        "sentinel_2_source": phase8.EARTH_SEARCH,
        "sentinel_2_radiometry": PHASE13_S2_RADIOMETRY,
        "sentinel_2_radiometry_modes": sorted(
            inventory_frame["sentinel_2_radiometry"].dropna().unique().tolist()
        ),
        "selected_item_ids": selected_items,
        "terrain_source": "Element 84 Earth Search cop-dem-glo-30",
        "model_features": MODEL_FEATURES,
        "observation_count_columns": ["s2_valid_observation_count"],
        "observation_counts_are_model_features": False,
        "minimum_clear_observations": int(predictors["s2_valid_observation_count"].min()),
        "remote_source_rasters_retained": False,
        "site_checkpoints_retained": False,
    }
    freeze = {
        "freeze_id": "phase13-conventional-" + common.canonical_hash(freeze_basis)[:12],
        "created_utc": common.utc_now(),
        "status": "frozen_predictors_before_target_outcome_opening",
        "freeze_basis": freeze_basis,
        "outputs": {
            str(path.relative_to(ROOT)): common.sha256(path)
            for path in [OUTPUT_PATH, INVENTORY_PATH]
        },
    }
    common.write_json(FREEZE_PATH, freeze)
    print(
        json.dumps(
            {
                "freeze_id": freeze["freeze_id"],
                "sites": freeze_basis["site_rows"],
                "minimum_clear_observations": freeze_basis["minimum_clear_observations"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
