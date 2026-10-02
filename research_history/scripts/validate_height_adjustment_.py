#!/usr/bin/env python3
"""Validate the Phase 10 screen, height target, and retrospective evaluation artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
SCREEN_FREEZE = ROOT / "metadata/phase10_unseen_site_screen_freeze.json"
HEIGHT_FREEZE = ROOT / "metadata/phase10_source_height_model_freeze.json"
EVALUATION_FREEZE = ROOT / "metadata/phase10_height_adjusted_existing_freeze.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_outputs(freeze: dict[str, Any]) -> None:
    for name, output in freeze["outputs"].items():
        path = ROOT / output["path"]
        if not path.exists() or sha256(path) != output["sha256"]:
            raise RuntimeError(f"Frozen output mismatch: {name} ({path})")


def sites_with_acquired_fhd() -> set[str]:
    sites: set[str] = set()
    for path in sorted((ROOT / "data/processed").glob("*.parquet")):
        parquet = pq.ParquetFile(path)
        if not {"site_id", "fhd_normal"} <= set(parquet.schema_arrow.names):
            continue
        sites.update(
            str(value)
            for value in pq.read_table(path, columns=["site_id"])["site_id"].to_pylist()
            if value is not None
        )
    return sites


def main() -> int:
    screen = json.loads(SCREEN_FREEZE.read_text(encoding="utf-8"))
    height = json.loads(HEIGHT_FREEZE.read_text(encoding="utf-8"))
    evaluation = json.loads(EVALUATION_FREEZE.read_text(encoding="utf-8"))
    for freeze in [screen, height, evaluation]:
        validate_outputs(freeze)

    if screen["status"] != "frozen_gate_failed" or screen["gate"]["passed"]:
        raise RuntimeError("Unseen-site screen must remain a failed minimum-five gate")
    selected = screen["freeze_basis"]["selected_sites"]
    if selected != ["CLBJ", "RMNP"]:
        raise RuntimeError("Unexpected outcome-unseen selected sites")
    opened = sites_with_acquired_fhd()
    if set(selected) & opened:
        raise RuntimeError("A protected Phase 10 site now appears in an FHD table")

    if height["freeze_basis"]["target_outcomes_opened"]:
        raise RuntimeError("Source-only height freeze claims target outcome access")
    if not evaluation["freeze_basis"]["retrospective_due_to_target_outcomes_previously_opened_in_phase9"]:
        raise RuntimeError("Height-adjusted evidence must remain labelled retrospective")
    if evaluation["freeze_basis"]["confirmatory_claim_allowed"]:
        raise RuntimeError("Height-adjusted evidence cannot be labelled confirmatory")

    metrics_path = ROOT / evaluation["outputs"]["metrics"]["path"]
    paired_path = ROOT / evaluation["outputs"]["paired"]["path"]
    metrics = pd.read_csv(metrics_path)
    paired = pd.read_csv(paired_path)
    residual = metrics[
        metrics["scope"].eq("macro")
        & metrics["target_variable"].eq("fhd_height_residual")
        & metrics["method"].eq("residual_tessera")
        & metrics["strategy"].eq("all_sources")
    ].iloc[0]
    matched = paired[
        paired["comparison"].eq("height_plus_matched_tessera_minus_height_only")
    ].iloc[0]
    residual_matching = paired[
        paired["comparison"].eq("matched_minus_all_source_residual_tessera")
    ].iloc[0]
    if residual["r2"] >= 0:
        raise RuntimeError("Stored residual result no longer supports the documented conclusion")
    if not matched["bootstrap_95_low"] < 0 < matched["bootstrap_95_high"]:
        raise RuntimeError("Stored full-FHD incremental interval changed")
    if residual_matching["mean_delta_rmse"] <= 0:
        raise RuntimeError("Stored habitat-matched residual comparison changed")

    print(json.dumps({
        "status": "passed",
        "screen_freeze_id": screen["freeze_id"],
        "height_freeze_id": height["freeze_id"],
        "evaluation_freeze_id": evaluation["freeze_id"],
        "protected_unopened_sites": selected,
        "all_source_residual_macro_r2": float(residual["r2"]),
        "matched_full_fhd_mean_delta_rmse": float(matched["mean_delta_rmse"]),
        "matched_residual_mean_delta_rmse": float(residual_matching["mean_delta_rmse"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
