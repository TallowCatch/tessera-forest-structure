#!/usr/bin/env python3
"""Validate the complete Phase 7 outcome-blind expansion and eight-site LOSO."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import r2_score


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_outputs(freeze: dict[str, Any]) -> None:
    for record in freeze["outputs"].values():
        path = ROOT / record["path"]
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise RuntimeError(f"Missing or changed frozen output: {record['path']}")


def load(relative: str) -> dict[str, Any]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def main() -> int:
    selection = load("metadata/phase7_site_selection_freeze.json")
    if selection["freeze_id"] != "phase7-site-selection-45a1c40b6111":
        raise RuntimeError("Unexpected Phase 7 site selection freeze")
    validate_outputs(selection)
    if not selection["outcome_blind"] or selection["fhd_columns_accessed"]:
        raise RuntimeError("Phase 7 site selection is not outcome-blind")
    selected_sites = ["HARV", "ORNL", "OSBS", "TALL", "NIWO", "WREF", "UNDE"]
    if selection["selected_sites"] != selected_sites:
        raise RuntimeError("Phase 7 preselected site cohort changed")
    screen = pd.read_csv(ROOT / selection["outputs"]["screen"]["path"])
    chosen = screen[screen["selected"]]
    if (
        len(chosen) != 7
        or not chosen["cmr_2024_gedi_v3_granules"].ge(4).all()
        or not chosen["nlcd_2024_forest_fraction"].ge(0.30).all()
        or not chosen["tessera_inventory_complete"].all()
    ):
        raise RuntimeError("Selected-site feasibility evidence changed")

    gedi = load("metadata/phase7_gedi_freeze.json")
    if gedi["freeze_id"] != "phase7-gedi-v3-020c6c80b39f":
        raise RuntimeError("Unexpected Phase 7 GEDI freeze")
    validate_outputs(gedi)
    viability = pd.DataFrame(gedi["freeze_basis"]["viability"]).set_index("site_id")
    viable_sites = ["HARV", "ORNL", "TALL", "WREF", "UNDE"]
    if set(viability.index[viability["viable"]]) != set(viable_sites):
        raise RuntimeError("Phase 7 viable site set changed")
    if int(viability.loc["OSBS", "primary_rows"]) != 54 or int(
        viability.loc["NIWO", "primary_rows"]
    ) != 80:
        raise RuntimeError("Nonviable site counts changed")
    targets = pd.read_parquet(ROOT / gedi["outputs"]["primary_sample"]["path"])
    if len(targets) != 7121 or set(targets["site_id"].unique()) != set(viable_sites):
        raise RuntimeError("Phase 7 primary GEDI population changed")
    if not targets["passes_primary_filters"].all():
        raise RuntimeError("Phase 7 primary table contains a failed filter row")

    alignment = load("metadata/phase7_tessera_alignment_freeze.json")
    if alignment["alignment_id"] != "phase7-tessera-align-6ab0ed696e57":
        raise RuntimeError("Unexpected Phase 7 TESSERA alignment freeze")
    validate_outputs(alignment)
    aligned = pd.read_parquet(ROOT / alignment["outputs"]["aligned_expansion"]["path"])
    if len(aligned) != 7121 or not aligned["passes_tessera_alignment"].all():
        raise RuntimeError("Phase 7 aligned expansion population changed")
    if (
        set(aligned["tessera_dataset_version"].astype(str)) != {"1.0"}
        or set(aligned["tessera_dataset_variant"]) != {"vultr"}
        or set(aligned["tessera_embedding_year"]) != {2024}
    ):
        raise RuntimeError("Expansion TESSERA release changed")
    tile_manifest = load("metadata/phase7_tessera_tile_manifest.json")
    if tile_manifest["tile_download_count"] != 23 or not tile_manifest[
        "local_expansion_tiles_deleted"
    ]:
        raise RuntimeError("Expansion tile streaming provenance changed")

    prediction_freeze = load("metadata/phase7_prospective_prediction_freeze.json")
    if prediction_freeze["freeze_id"] != "phase7-prospective-predictions-a04c5743506e":
        raise RuntimeError("Unexpected prospective prediction freeze")
    validate_outputs(prediction_freeze)
    prediction_basis = prediction_freeze["freeze_basis"]
    if prediction_basis["expansion_columns_contain_fhd_normal"] or prediction_basis[
        "expansion_target_labels_used_for_tuning_training_or_selection"
    ]:
        raise RuntimeError("Prospective prediction stage accessed expansion outcomes")
    prospective_predictions = pd.read_parquet(
        ROOT / prediction_freeze["outputs"]["predictions"]["path"]
    )
    if len(prospective_predictions) != 14242 or prospective_predictions.groupby(
        ["site_id", "shot_number", "method"]
    ).size().ne(1).any():
        raise RuntimeError("Prospective prediction population changed")
    prospective_model = joblib.load(ROOT / prediction_freeze["outputs"]["model"]["path"])
    if prospective_model["expansion_target_outcome_present"] or "fhd_normal" in prospective_model[
        "expansion_target_columns"
    ]:
        raise RuntimeError("Prospective model artifact reports target access")
    original_weight_totals = list(prospective_model["training_site_weight_totals"].values())
    if max(original_weight_totals) - min(original_weight_totals) > 1e-9:
        raise RuntimeError("Original training sites were not equally weighted")

    prospective = load("metadata/phase7_prospective_evaluation_freeze.json")
    if prospective["freeze_id"] != "phase7-prospective-evaluation-d424d39c22bd":
        raise RuntimeError("Unexpected prospective evaluation freeze")
    validate_outputs(prospective)
    prospective_gate = prospective["freeze_basis"]["gate"]
    if (
        prospective_gate["passed"]
        or prospective_gate["positive_r2_sites"]
        or prospective_gate["rmse_below_training_mean_sites"] != ["WREF"]
    ):
        raise RuntimeError("Prospective external result changed")
    prospective_evaluation = pd.read_parquet(
        ROOT / prospective["outputs"]["evaluation"]["path"]
    )
    prospective_metrics = pd.read_csv(ROOT / prospective["outputs"]["metrics"]["path"])
    prospective_site = prospective_metrics[prospective_metrics["scope"].eq("site")]
    for (site, method), frame in prospective_evaluation.groupby(["site_id", "method"]):
        stored = prospective_site[
            prospective_site["site_id"].eq(site) & prospective_site["method"].eq(method)
        ].iloc[0]
        recalculated = float(r2_score(frame["fhd_normal"], frame["prediction"]))
        if not math.isclose(recalculated, float(stored["r2"]), abs_tol=1e-12):
            raise RuntimeError(f"Prospective R2 does not reproduce for {site}/{method}")

    expanded = load("metadata/phase7_expanded_loso_freeze.json")
    if expanded["freeze_id"] != "phase7-expanded-loso-c519234aacb9":
        raise RuntimeError("Unexpected expanded LOSO freeze")
    validate_outputs(expanded)
    expanded_gate = expanded["freeze_basis"]["gate"]
    if expanded_gate["passed"] or set(expanded_gate["positive_r2_sites"]) != {"ORNL", "TALL"}:
        raise RuntimeError("Expanded LOSO positive-site result changed")
    expected_improved = {"BART", "HARV", "ORNL", "SOAP", "TALL", "TEAK", "UNDE"}
    if set(expanded_gate["rmse_below_training_mean_sites"]) != expected_improved:
        raise RuntimeError("Expanded LOSO baseline comparison changed")
    expanded_predictions = pd.read_parquet(ROOT / expanded["outputs"]["predictions"]["path"])
    if len(expanded_predictions) != 20368 or expanded_predictions.groupby(
        ["site_id", "shot_number", "method"]
    ).size().ne(1).any():
        raise RuntimeError("Expanded LOSO prediction population changed")
    artifacts = joblib.load(ROOT / expanded["outputs"]["models"]["path"])
    for held_out, fold in artifacts["outer_folds"].items():
        if fold["target_outcome_present"] or "fhd_normal" in fold["target_columns"]:
            raise RuntimeError(f"Expanded LOSO target entered fitting for {held_out}")
        totals = list(fold["training_weight_totals"].values())
        if max(totals) - min(totals) > 1e-9:
            raise RuntimeError(f"Expanded LOSO training sites not equally weighted for {held_out}")

    staging_files = list((ROOT / "data/raw/phase7_gedi_v3/staging").glob("*"))
    stream_files = list((ROOT / "data/interim/phase7_tessera_stream").rglob("*"))
    if any(path.is_file() for path in staging_files + stream_files):
        raise RuntimeError("Phase 7 source HDF or streamed TESSERA files were retained")

    print("Phase 7 validation passed")
    print("Five outcome-blind viable expansion sites; 7,121 new footprints")
    print("Prospective original-three model: zero positive-R2 expansion sites")
    print("Expanded eight-site LOSO: lower macro RMSE, positive R2 at only ORNL and TALL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
