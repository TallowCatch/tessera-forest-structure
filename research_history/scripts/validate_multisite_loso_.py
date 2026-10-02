#!/usr/bin/env python3
"""Validate the frozen Phase 6 multi-site leave-one-site-out experiment."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml
from sklearn.metrics import r2_score


ROOT = Path(__file__).resolve().parents[1]
FREEZE_PATH = ROOT / "metadata/phase6_multisite_loso_freeze.json"
CONFIG_PATH = ROOT / "configs/project.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_record(record: dict[str, Any]) -> None:
    path = ROOT / record["path"]
    if not path.is_file() or sha256(path) != record["sha256"]:
        raise RuntimeError(f"Missing or changed frozen output: {record['path']}")


def main() -> int:
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if freeze["freeze_id"] != "phase6-multisite-loso-2bec653b4802":
        raise RuntimeError("Unexpected Phase 6 freeze ID")
    for record in freeze["outputs"].values():
        validate_record(record)
    for relative_path, expected_hash in freeze["freeze_basis"]["input_files"].items():
        if sha256(ROOT / relative_path) != expected_hash:
            raise RuntimeError(f"Changed Phase 6 input: {relative_path}")

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    result = config["phase6_multisite_leave_one_site_out"]["result"]
    if result["freeze_id"] != freeze["freeze_id"] or result["generalization_gate_passed"]:
        raise RuntimeError("Project config does not record the frozen failed gate")

    predictions = pd.read_parquet(ROOT / freeze["outputs"]["predictions"]["path"])
    expected_models = {
        "training_mean",
        "terrain_ridge",
        "tessera_area_ridge",
        "tessera_area_topography_ridge",
    }
    if len(predictions) != 12252 or set(predictions["model"].unique()) != expected_models:
        raise RuntimeError("Phase 6 prediction population changed")
    if predictions.groupby(["site_id", "shot_number", "model"]).size().ne(1).any():
        raise RuntimeError("Phase 6 predictions are not unique")
    if predictions.groupby("model").size().ne(3063).any():
        raise RuntimeError("A Phase 6 model does not cover every footprint")
    if not predictions["site_id"].eq(predictions["outer_held_out_site"]).all():
        raise RuntimeError("A prediction is not assigned to its completely held-out site")

    artifacts = joblib.load(ROOT / freeze["outputs"]["models"]["path"])
    for held_out, fold in artifacts["outer_folds"].items():
        metadata = fold["metadata"]
        if metadata["target_outcome_present"] or "fhd_normal" in metadata["target_columns"]:
            raise RuntimeError(f"Held-out outcome entered the fit stage for {held_out}")
        totals = list(metadata["training_weight_total_by_site"].values())
        if len(totals) != 2 or not math.isclose(totals[0], totals[1], abs_tol=1e-9):
            raise RuntimeError(f"Training sites are not equally weighted for {held_out}")

    tuning = pd.read_csv(ROOT / freeze["outputs"]["inner_tuning"]["path"])
    selected = pd.read_csv(ROOT / freeze["outputs"]["selected_alphas"]["path"])
    if len(tuning) != 126 or len(selected) != 9:
        raise RuntimeError("Phase 6 nested tuning grid changed")
    if tuning.groupby(["outer_held_out_site", "model", "alpha"]).size().ne(2).any():
        raise RuntimeError("A reciprocal inner site direction is missing")

    metrics = pd.read_csv(ROOT / freeze["outputs"]["metrics"]["path"])
    site = metrics[metrics["scope"].eq("site")]
    primary = site[site["model"].eq("tessera_area_topography_ridge")]
    if len(primary) != 3 or not primary["r2"].lt(0).all():
        raise RuntimeError("Primary site-level failure no longer reproduces")
    tessera = site[site["model"].eq("tessera_area_ridge")].set_index("held_out_site")
    if set(tessera.index[tessera["r2"].gt(0)]) != {"TEAK"}:
        raise RuntimeError("TESSERA-only positive-site result changed")

    for (held_out, model), frame in predictions.groupby(["outer_held_out_site", "model"]):
        stored = site[
            site["held_out_site"].eq(held_out) & site["model"].eq(model)
        ].iloc[0]
        recalculated = float(r2_score(frame["fhd_normal"], frame["prediction"]))
        if not math.isclose(recalculated, float(stored["r2"]), abs_tol=1e-12):
            raise RuntimeError(f"Stored R2 does not reproduce for {held_out}/{model}")

    gate = freeze["freeze_basis"]["gate"]
    if gate["passed"] or gate["primary_r2_above_zero_on_every_held_out_site"]:
        raise RuntimeError("Frozen generalization gate unexpectedly passes")

    print("Phase 6 validation passed")
    print("3,063 footprints; three complete held-out sites; 12,252 predictions")
    print("Primary R2 negative at SOAP, TEAK, and BART")
    print("TESSERA-only positive only at TEAK; generalization gate failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
