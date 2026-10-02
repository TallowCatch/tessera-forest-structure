#!/usr/bin/env python3
"""Validate Phase 11 acquisition, height-transfer, and diagnostic freezes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_context_height_footprint_height_transfer as base  # noqa: E402


EXPECTED_FREEZES = {
    "metadata/phase11_footprint_height_freeze.json": "phase11-footprint-height-a91c6db1ea71",
    "metadata/phase11_tessera_context_freeze.json": "phase11-tessera-context-93a841199ade",
    "metadata/phase11_context_height_freeze.json": "phase11-context-height-9700619df2d4",
    "metadata/phase11_context_height_diagnostic_freeze.json": "phase11-context-height-diagnostic-ae85abd37b57",
}


def load_and_validate_freeze(relative_path: str, expected_id: str) -> dict:
    path = ROOT / relative_path
    freeze = json.loads(path.read_text(encoding="utf-8"))
    if freeze["freeze_id"] != expected_id:
        raise RuntimeError(f"Unexpected freeze ID in {relative_path}")
    for output, expected_hash in freeze["outputs"].items():
        output_path = ROOT / output
        if not output_path.exists() or base.sha256(output_path) != expected_hash:
            raise RuntimeError(f"Missing or changed Phase 11 output: {output}")
    return freeze


def macro_row(path: Path, model: str) -> pd.Series:
    metrics = pd.read_csv(path)
    rows = metrics[metrics["model"].eq(model) & metrics["scope"].eq("macro")]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one macro row for {model}")
    return rows.iloc[0]


def main() -> int:
    freezes = {
        path: load_and_validate_freeze(path, freeze_id)
        for path, freeze_id in EXPECTED_FREEZES.items()
    }
    context_acquisition = freezes["metadata/phase11_tessera_context_freeze.json"]
    if context_acquisition["freeze_basis"]["target_outcomes_used"]:
        raise RuntimeError("Target-free context acquisition claims target use")
    if context_acquisition["freeze_basis"]["rows"] != 15_326:
        raise RuntimeError("Phase 11 context row count changed")
    if context_acquisition["freeze_basis"]["context_feature_count"] != 1_152:
        raise RuntimeError("Phase 11 context feature count changed")

    footprint = macro_row(
        ROOT / "outputs/tables/phase11_footprint_height_metrics.csv",
        "tessera_area_ridge",
    )
    context = macro_row(
        ROOT / "outputs/tables/phase11_context_height_metrics.csv",
        "tessera_context_ridge",
    )
    nonlinear = macro_row(
        ROOT / "outputs/tables/phase11_context_height_metrics.csv",
        "tessera_context_hgb",
    )
    expected = {
        "footprint_r2": (float(footprint["r2"]), -0.06332237496436542),
        "footprint_rmse": (float(footprint["rmse"]), 8.245269577230156),
        "context_r2": (float(context["r2"]), -0.13777875119608127),
        "context_rmse": (float(context["rmse"]), 8.522062189554493),
        "nonlinear_r2": (float(nonlinear["r2"]), -0.0023325159194682588),
        "nonlinear_rmse": (float(nonlinear["rmse"]), 8.214623633896947),
    }
    for name, (actual, frozen) in expected.items():
        if not np.isclose(actual, frozen, atol=1e-10, rtol=0):
            raise RuntimeError(f"Phase 11 metric changed: {name}")

    height_freeze = freezes["metadata/phase11_context_height_freeze.json"]
    gate = height_freeze["freeze_basis"]["gate"]
    if gate["passed"] or gate["macro_r2_above_zero"]:
        raise RuntimeError("Phase 11 failed height gate changed")
    if height_freeze["freeze_basis"]["fhd_normal_used"]:
        raise RuntimeError("Phase 11 height model claims FHD use")

    bootstrap = pd.read_csv(
        ROOT / "outputs/tables/phase11_context_height_paired_bootstrap.csv"
    )
    rmse = bootstrap[bootstrap["metric"].eq("raw_rmse")].set_index("comparison")
    if int(rmse.loc[
        "tessera_context_ridge_minus_tessera_footprint_ridge", "sites_improved"
    ]) != 4:
        raise RuntimeError("Context Ridge paired result changed")
    if int(rmse.loc[
        "tessera_context_hgb_minus_tessera_footprint_ridge", "sites_improved"
    ]) != 7:
        raise RuntimeError("Nonlinear context paired result changed")
    if (ROOT / "data/interim/phase11_tessera_context_stream").exists():
        raise RuntimeError("Phase 11 raw stream directory was not deleted")

    print("Phase 11 validation passed")
    print("15,326 footprints; 14 complete-site holdouts; no FHD used in height stage")
    print("Context height gate failed as frozen; height-mediated FHD branches remain stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
