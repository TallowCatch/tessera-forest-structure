import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_height_normalized_proxy_uses_one_metre_fhd_layers() -> None:
    module = load_module(
        ROOT / "scripts/run_transfer_robustness.py",
        "run_transfer_robustness_test",
    )
    frame = pd.DataFrame({
        "fhd_normal": [2.0, 3.0],
        "elev_highestreturn": [14.2, 22.0],
        "elev_lowestmode": [4.0, 2.0],
    })

    result = module.add_targets(frame)

    expected = frame["fhd_normal"].to_numpy() / np.log(np.array([11.0, 20.0]))
    np.testing.assert_allclose(result["height_normalized_entropy_proxy"], expected)


def test_robustness_freezes_keep_retrospective_status_explicit() -> None:
    expected = {
        "robustness_spatial_freeze.json": "retrospective_robustness_not_prospective",
        "robustness_measurement_freeze.json": (
            "retrospective_evaluation_of_unchanged_frozen_predictions"
        ),
        "robustness_transfer_freeze.json": (
            "retrospective_source_only_method_selection_then_known_target_evaluation"
        ),
        "robustness_bart_continuous_freeze.json": (
            "retrospective_local_calibration_case_study"
        ),
    }
    for name, status in expected.items():
        freeze = json.loads((ROOT / "metadata" / name).read_text(encoding="utf-8"))
        assert freeze["status"] == status


def test_spatial_robustness_exposes_scale_dependence() -> None:
    metrics = pd.read_csv(ROOT / "outputs/tables/robustness_spatial_metrics.csv")
    pooled = metrics[metrics["scope"].eq("pooled")].set_index("analysis")

    assert pooled.loc["spatial_buffer_500m", "r2"] > 0.5
    assert pooled.loc["spatial_buffer_3000m", "r2"] > 0.4
    assert pooled.loc["spatial_buffer_6000m", "r2"] < 0.01
    assert pooled.loc["leave_one_gedi_pass_out", "r2"] > 0.5


def test_site_level_inference_and_global_permutation_are_separate() -> None:
    paired = pd.read_csv(ROOT / "outputs/tables/robustness_measurement_paired.csv")
    all_rows = paired[paired["subset"].eq("all_retained")].iloc[0]
    assert all_rows["sites_with_lower_matched_rmse"] == 5
    assert np.isclose(all_rows["exact_two_sided_sign_test_p"], 0.21875)

    permutation = pd.read_csv(
        ROOT / "outputs/tables/robustness_habitat_permutation_summary.csv"
    ).iloc[0]
    assert permutation["replicates"] == 10000
    assert 0.03 < permutation["one_sided_global_permutation_p"] < 0.04


def test_continuous_bart_case_reports_support_and_failed_validation() -> None:
    metrics = pd.read_csv(
        ROOT / "outputs/tables/robustness_bart_continuous_validation.csv"
    ).set_index("comparison")
    support = pd.read_csv(
        ROOT / "outputs/tables/robustness_bart_applicability.csv"
    ).set_index("population")

    assert metrics.loc["local_model_vs_heldout_GEDI_in_complete_tile", "r2"] < 0
    assert metrics.loc["local_model_vs_offtrack_converted_airborne_FHD", "r2"] < 0
    assert support.loc["complete_tile_forest_pixels", "fraction_inside_source_support"] > 0.99
    assert support.loc["offtrack_airborne_lidar_points", "fraction_inside_source_support"] > 0.98
