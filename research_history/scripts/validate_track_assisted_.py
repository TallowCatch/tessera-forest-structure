#!/usr/bin/env python3
"""Validate Phase 12 track-assisted splits, results, and budget diagnostic."""

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
    "metadata/phase12_track_assisted_prediction_freeze.json":
        "phase12-track-assisted-afc8b3eed44d",
    "metadata/phase12_track_budget_freeze.json":
        "phase12-track-budget-da5043f2c320",
}


def load_and_validate_freeze(relative_path: str, expected_id: str) -> dict:
    path = ROOT / relative_path
    freeze = json.loads(path.read_text(encoding="utf-8"))
    if freeze["freeze_id"] != expected_id:
        raise RuntimeError(f"Unexpected freeze ID in {relative_path}")
    for output, expected_hash in freeze["outputs"].items():
        output_path = ROOT / output
        if not output_path.exists() or base.sha256(output_path) != expected_hash:
            raise RuntimeError(f"Missing or changed Phase 12 output: {output}")
    return freeze


def expect_close(actual: float, expected: float, name: str) -> None:
    if not np.isclose(float(actual), expected, atol=1e-10, rtol=0):
        raise RuntimeError(f"Phase 12 metric changed: {name}")


def main() -> int:
    split = json.loads(
        (ROOT / "metadata/phase12_track_assisted_split_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if split["freeze_id"] != "phase12-track-splits-03eb62d46a19":
        raise RuntimeError("Unexpected Phase 12 split freeze ID")
    basis = split["freeze_basis"]
    if basis["fhd_normal_read"] or "fhd_normal" in basis["columns_read"]:
        raise RuntimeError("Target-free Phase 12 split freeze read FHD")
    if basis["rows"] != 15_326 or basis["primary_folds"] != 37:
        raise RuntimeError("Phase 12 split population changed")
    if len(basis["primary_sites"]) != 10 or basis["buffer_m"] != 500.0:
        raise RuntimeError("Phase 12 primary site or buffer definition changed")

    freezes = {
        path: load_and_validate_freeze(path, freeze_id)
        for path, freeze_id in EXPECTED_FREEZES.items()
    }
    result = freezes["metadata/phase12_track_assisted_prediction_freeze.json"]
    result_basis = result["freeze_basis"]
    if result_basis["test_pass_fhd_used_in_training_or_selection"]:
        raise RuntimeError("Held-out pass FHD entered Phase 12 training or selection")
    if not result_basis["gate"]["passed"]:
        raise RuntimeError("Phase 12 primary gate no longer passes")

    macro = pd.read_csv(
        ROOT / "outputs/tables/phase12_track_assisted_macro_metrics.csv"
    )
    primary = macro[macro["analysis_group"].eq("primary")].set_index("model")
    expected_metrics = {
        ("source_tessera_ridge", "rmse"): 0.3781590731361858,
        ("source_tessera_ridge", "r2"): -0.0678448764214868,
        ("source_plus_local_offset", "rmse"): 0.3454812727334869,
        ("source_plus_local_offset", "r2"): 0.1293332336612904,
        ("source_plus_local_offset", "spearman_r"): 0.4463998728823147,
        ("target_local_mean", "rmse"): 0.4007741303968088,
        ("source_plus_local_residual_ridge", "rmse"): 0.3296628423437238,
        ("source_plus_local_residual_ridge", "r2"): 0.2277670290342281,
    }
    for (model, metric), expected in expected_metrics.items():
        expect_close(primary.loc[model, metric], expected, f"{model}.{metric}")

    bootstrap = pd.read_csv(
        ROOT / "outputs/tables/phase12_track_assisted_bootstrap.csv"
    )
    offset = bootstrap[
        bootstrap["comparison"].eq(
            "source_plus_local_offset_minus_source_tessera_ridge"
        )
        & bootstrap["metric"].eq("rmse")
    ].iloc[0]
    if int(offset["sites_improved"]) != 6:
        raise RuntimeError("Phase 12 offset site count changed")
    expect_close(offset["mean_paired_difference"], -0.0326778004026988, "offset delta")
    expect_close(offset["site_bootstrap_95_low"], -0.0627273710006269, "offset CI low")
    expect_close(offset["site_bootstrap_95_high"], -0.0073084071949786, "offset CI high")

    budget_freeze = freezes["metadata/phase12_track_budget_freeze.json"]
    coverage = budget_freeze["freeze_basis"]["method_fold_coverage"]
    if coverage["offset_50_shots"] != 37 or coverage["offset_250_shots"] != 35:
        raise RuntimeError("Phase 12 budget fold coverage changed")
    if not budget_freeze["freeze_basis"][
        "figure_excludes_incomplete_250_shot_sensitivity"
    ]:
        raise RuntimeError("Incomplete 250-shot sensitivity entered the main figure")

    budget_macro = pd.read_csv(
        ROOT / "outputs/tables/phase12_track_budget_macro_metrics.csv"
    ).set_index("method")
    expected_budget = {
        ("offset_10_shots", "rmse"): 0.3596065035772188,
        ("offset_25_shots", "rmse"): 0.3514152356517851,
        ("offset_50_shots", "rmse"): 0.3482689331724505,
        ("offset_100_shots", "rmse"): 0.3468854880739709,
        ("offset_one_retained_pass", "rmse"): 0.35383465539817,
    }
    for (method, metric), expected in expected_budget.items():
        expect_close(
            budget_macro.loc[method, metric], expected, f"{method}.{metric}"
        )

    budget_bootstrap = pd.read_csv(
        ROOT / "outputs/tables/phase12_track_budget_bootstrap.csv"
    ).set_index("comparison")
    fifty = budget_bootstrap.loc[
        "offset_50_shots_minus_source_only_tessera"
    ]
    expect_close(
        fifty["mean_paired_rmse_difference"], -0.0298901399637352,
        "50-shot paired delta",
    )
    if not (float(fifty["site_bootstrap_95_high"]) < 0):
        raise RuntimeError("Phase 12 50-shot interval no longer favours assistance")

    print("Phase 12 validation passed")
    print("10 forests; 37 complete-pass folds; 500 m local-label exclusion")
    print("Primary local-offset gate passed; 25-100-shot diagnostics beat source-only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
