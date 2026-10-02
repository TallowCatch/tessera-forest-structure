import json
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_phase10_screen_failed_without_opening_target_outcomes():
    freeze = json.loads(
        (ROOT / "metadata/phase10_unseen_site_screen_freeze.json").read_text(encoding="utf-8")
    )
    assert freeze["status"] == "frozen_gate_failed"
    assert freeze["gate"] == {
        "minimum_selected_sites": 5,
        "passed": False,
        "selected_site_count": 2,
    }
    assert freeze["freeze_basis"]["selected_sites"] == ["CLBJ", "RMNP"]
    assert freeze["freeze_basis"]["target_fhd_columns_read"] == []
    assert freeze["freeze_basis"]["target_height_columns_read"] == []
    assert freeze["freeze_basis"]["target_outcomes_opened"] is False


def test_phase10_height_target_and_evidential_status_are_explicit():
    protocol = yaml.safe_load(
        (ROOT / "metadata/project_config_phase10_unseen_replication_protocol_freeze.yaml")
        .read_text(encoding="utf-8")
    )["phase10_unseen_replication"]
    assert protocol["source_only_height_adjustment"]["residual_definition"] == (
        "fhd_normal_minus_source_only_expected_fhd_from_height"
    )
    assert protocol["source_only_height_adjustment"]["target_fhd_or_height_read_before_prediction_freeze"] is False

    freeze = json.loads(
        (ROOT / "metadata/phase10_height_adjusted_existing_freeze.json").read_text(encoding="utf-8")
    )
    basis = freeze["freeze_basis"]
    assert basis["retrospective_due_to_target_outcomes_previously_opened_in_phase9"] is True
    assert basis["confirmatory_claim_allowed"] is False


def test_phase10_height_adjusted_metrics_match_frozen_conclusion():
    metrics = pd.read_csv(ROOT / "outputs/tables/phase10_height_adjusted_existing_metrics.csv")
    macro = metrics[metrics["scope"].eq("macro")]
    height = macro[
        macro["target_variable"].eq("fhd_normal")
        & macro["method"].eq("height_only")
    ].iloc[0]
    residual = macro[
        macro["target_variable"].eq("fhd_height_residual")
        & macro["method"].eq("residual_tessera")
        & macro["strategy"].eq("all_sources")
    ].iloc[0]
    assert abs(height["r2"] - 0.864162) < 1e-6
    assert residual["r2"] < 0
    assert residual["spearman_r"] < 0.12

    paired = pd.read_csv(ROOT / "outputs/tables/phase10_height_adjusted_existing_paired.csv")
    matched = paired[
        paired["comparison"].eq("height_plus_matched_tessera_minus_height_only")
    ].iloc[0]
    residual_matching = paired[
        paired["comparison"].eq("matched_minus_all_source_residual_tessera")
    ].iloc[0]
    assert matched["bootstrap_95_low"] < 0 < matched["bootstrap_95_high"]
    assert residual_matching["mean_delta_rmse"] > 0
