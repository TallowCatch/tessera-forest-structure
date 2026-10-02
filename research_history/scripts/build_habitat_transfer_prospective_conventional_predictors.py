#!/usr/bin/env python3
"""Acquire corrected Sentinel-2 and terrain predictors for prospective targets."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_conventional_predictors as base  # noqa: E402
import build_fair_baselines_conventional_predictors as phase8  # noqa: E402


PROTOCOL_PATH = ROOT / "metadata/project_config_phase9_prospective_conventional_protocol.yaml"
TARGET_PATH = ROOT / "data/processed/phase9_prospective_tessera_aligned.parquet"
EVALUATION_FREEZE_PATH = ROOT / "metadata/phase9_prospective_evaluation_freeze.json"
OUTPUT_PATH = ROOT / "data/processed/phase9_prospective_conventional_predictors.parquet"
INVENTORY_PATH = ROOT / "metadata/phase9_prospective_conventional_inventory.csv"
FREEZE_PATH = ROOT / "metadata/phase9_prospective_conventional_freeze.json"
CHECKPOINT_DIR = ROOT / "data/interim/phase9_prospective_conventional_site_chunks"

KEYS = phase8.KEYS
TARGET_READ_COLUMNS = KEYS + ["longitude", "latitude", "x_epsg5070", "y_epsg5070"]
MODEL_FEATURES = phase8.TERRAIN_FEATURES + phase8.S2_FEATURES
OUTPUT_COLUMNS = KEYS + MODEL_FEATURES + ["s2_valid_observation_count"]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    protected = [OUTPUT_PATH, INVENTORY_PATH, FREEZE_PATH]
    existing = [str(path.relative_to(ROOT)) for path in protected if path.exists()]
    if existing:
        raise RuntimeError(f"Prospective conventional freeze already exists: {existing}")
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))[
        "phase9_prospective_conventional_comparison"
    ]
    evaluation_freeze = json.loads(EVALUATION_FREEZE_PATH.read_text(encoding="utf-8"))
    if evaluation_freeze["evaluation_id"] != protocol["prospective_evaluation_id"]:
        raise RuntimeError("Conventional protocol points to the wrong prospective evaluation")
    config = yaml.safe_load(base.CONFIG_PATH.read_text(encoding="utf-8"))
    targets = pd.read_parquet(TARGET_PATH, columns=TARGET_READ_COLUMNS)
    if "fhd_normal" in targets.columns or targets.duplicated(KEYS).any():
        raise RuntimeError("Target FHD or duplicate keys entered conventional acquisition")
    expected_sites = sorted(protocol["target_sites"])
    if sorted(targets["site_id"].unique()) != expected_sites:
        raise RuntimeError("Unexpected prospective conventional target cohort")

    phase8.CHECKPOINT_DIR = CHECKPOINT_DIR
    predictors, inventory, selected_items = phase8.expansion_s2_terrain(targets, config)
    predictors = predictors[OUTPUT_COLUMNS].sort_values(KEYS).reset_index(drop=True)
    if len(predictors) != len(targets) or predictors.duplicated(KEYS).any():
        raise RuntimeError("Prospective conventional predictor population is incomplete")
    if predictors["s2_valid_observation_count"].lt(6).any():
        raise RuntimeError("At least one target has fewer than six clear Sentinel-2 observations")
    if not np.isfinite(predictors[MODEL_FEATURES].to_numpy(dtype=np.float64)).all():
        raise RuntimeError("Prospective Sentinel-2 or terrain predictors contain non-finite values")
    medians = predictors.groupby("site_id")[[
        "s2_blue_median", "s2_red_median", "s2_nir_median", "s2_ndvi_median"
    ]].median()
    if not medians[["s2_blue_median", "s2_red_median", "s2_nir_median"]].apply(
        lambda values: values.between(-0.05, 1.0)
    ).all().all():
        raise RuntimeError("Prospective Sentinel-2 reflectance failed radiometric QA")
    if not medians["s2_ndvi_median"].between(-1.0, 1.0).all():
        raise RuntimeError("Prospective Sentinel-2 NDVI failed radiometric QA")

    inventory_frame = pd.DataFrame(inventory).drop_duplicates().sort_values(
        ["site_id", "collection", "datetime", "item_id"], na_position="last"
    )
    base.write_parquet_atomic(predictors, OUTPUT_PATH)
    base.write_csv_atomic(inventory_frame, INVENTORY_PATH)
    outputs = {
        "predictors": {
            "path": str(OUTPUT_PATH.relative_to(ROOT)),
            "rows": len(predictors),
            "sha256": sha256(OUTPUT_PATH),
        },
        "inventory": {
            "path": str(INVENTORY_PATH.relative_to(ROOT)),
            "rows": len(inventory_frame),
            "sha256": sha256(INVENTORY_PATH),
        },
    }
    if CHECKPOINT_DIR.exists():
        shutil.rmtree(CHECKPOINT_DIR)
    freeze_basis = {
        "analysis_timing": protocol["analysis_timing"],
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "prospective_evaluation_id": evaluation_freeze["evaluation_id"],
        "acquisition_script_sha256": sha256(Path(__file__).resolve()),
        "target_input_sha256": sha256(TARGET_PATH),
        "target_columns_read": TARGET_READ_COLUMNS,
        "target_fhd_read": False,
        "sites": expected_sites,
        "site_rows": {
            site: int(count) for site, count in predictors.groupby("site_id").size().items()
        },
        "sentinel_2_source": phase8.EARTH_SEARCH,
        "sentinel_2_radiometry": phase8.EARTH_SEARCH_S2_RADIOMETRY,
        "selected_item_ids": selected_items,
        "terrain_source": "Element 84 Earth Search cop-dem-glo-30",
        "model_features": MODEL_FEATURES,
        "observation_count_columns": ["s2_valid_observation_count"],
        "observation_counts_are_model_features": False,
        "minimum_clear_observations": int(predictors["s2_valid_observation_count"].min()),
        "remote_source_rasters_retained": False,
        "site_checkpoints_retained": False,
    }
    freeze_id = "phase9-prospective-conventional-" + canonical_hash(
        {"freeze_basis": freeze_basis, "outputs": outputs}
    )[:12]
    write_json(FREEZE_PATH, {
        "freeze_id": freeze_id,
        "created_utc": utc_now(),
        "status": "frozen_complete_post_hoc",
        "freeze_basis": freeze_basis,
        "outputs": outputs,
    })
    print(json.dumps({
        "freeze_id": freeze_id,
        "rows": len(predictors),
        "sites": freeze_basis["site_rows"],
        "minimum_clear_observations": freeze_basis["minimum_clear_observations"],
        "site_checkpoints_retained": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
