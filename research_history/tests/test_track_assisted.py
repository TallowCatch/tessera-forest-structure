from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/freeze_track_assisted_splits.py"
MODEL_SCRIPT = ROOT / "scripts/run_track_assisted_prediction.py"
BUDGET_SCRIPT = ROOT / "scripts/analyze_track_assisted_track_budget.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "freeze_track_assisted_splits", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_model_module():
    spec = importlib.util.spec_from_file_location(
        "run_track_assisted_prediction", MODEL_SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_budget_module():
    spec = importlib.util.spec_from_file_location(
        "analyze_track_assisted_track_budget", BUDGET_SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_pass_id_uses_orbit_track_and_utc_date() -> None:
    module = load_module()
    frame = pd.DataFrame(
        {
            "orbit": [123],
            "reference_ground_track": [456],
            "acquisition_datetime": ["2024-07-21T23:00:00Z"],
        }
    )

    assert module.pass_ids(frame).tolist() == ["123_456_2024-07-21"]


def test_buffered_split_holds_complete_pass_and_removes_close_training_rows() -> None:
    module = load_module()
    coordinates = np.asarray([[0.0, 0.0], [0.0, 100.0], [0.0, 400.0], [0.0, 700.0]])
    identifiers = np.asarray(["test", "test", "train", "train"])

    training, test, distances = module.buffered_local_split(
        coordinates, identifiers, "test", 500.0
    )

    assert test.tolist() == [0, 1]
    assert training.tolist() == [3]
    assert np.all(distances >= 500.0)


def test_split_freeze_read_contract_excludes_fhd() -> None:
    module = load_module()

    assert "fhd_normal" not in module.READ_COLUMNS


def test_local_ridge_can_fit_and_predict_a_small_regional_sample() -> None:
    module = load_model_module()
    x = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    y = np.asarray([1.0, 2.0, 3.0, 4.0])

    scaler, model = module.fit_local_ridge(x, y, alpha=0.01)
    prediction = model.predict(scaler.transform(x))

    assert prediction.shape == y.shape
    assert np.sqrt(np.mean((prediction - y) ** 2)) < 0.05


def test_source_alpha_selection_prioritizes_worst_site_error() -> None:
    module = load_model_module()
    tuning = pd.DataFrame(
        {
            "alpha": [1.0, 1.0, 10.0, 10.0],
            "rmse": [1.0, 5.0, 3.0, 4.0],
        }
    )

    selected, ranking = module.select_source_alpha(tuning)

    assert selected == 10.0
    assert ranking.iloc[0]["worst_inner_site_rmse"] == 4.0


def test_primary_gate_requires_tessera_to_beat_source_and_local_mean() -> None:
    module = load_model_module()
    rows = []
    for site in [f"S{i}" for i in range(10)]:
        rows.extend(
            [
                {"analysis_group": "primary", "site_id": site, "model": "source_tessera_ridge", "rmse": 2.0},
                {"analysis_group": "primary", "site_id": site, "model": "target_local_mean", "rmse": 1.5},
                {"analysis_group": "primary", "site_id": site, "model": "source_plus_local_offset", "rmse": 1.0},
            ]
        )
    site_metrics = pd.DataFrame(rows)
    macro_metrics = pd.DataFrame(
        [
            {"analysis_group": "primary", "model": "source_tessera_ridge", "rmse": 2.0, "r2": -1.0},
            {"analysis_group": "primary", "model": "target_local_mean", "rmse": 1.5, "r2": -0.5},
            {"analysis_group": "primary", "model": "source_plus_local_offset", "rmse": 1.0, "r2": 0.2},
        ]
    )

    gate = module.primary_gate(site_metrics, macro_metrics)

    assert gate["passed"] is True
    assert gate["sites_improved_vs_source_tessera"] == 10


def test_budget_sampling_represents_every_retained_pass() -> None:
    module = load_budget_module()
    passes = np.asarray(["A"] * 20 + ["B"] * 5 + ["C"] * 2)

    positions = module.stratified_sample_positions(
        passes, budget=10, rng=np.random.default_rng(7)
    )

    assert len(positions) == 10
    assert len(np.unique(positions)) == 10
    assert set(passes[positions]) == {"A", "B", "C"}


def test_budget_scenario_offset_uses_only_calibration_residual_mean() -> None:
    module = load_budget_module()
    result = module.scenario_metrics(
        observed_test=np.asarray([2.0, 4.0]),
        source_test=np.asarray([1.0, 3.0]),
        observed_calibration=np.asarray([5.0, 7.0]),
        source_calibration=np.asarray([3.0, 5.0]),
    )

    assert np.isclose(result["offset"]["mean_bias"], 1.0)
    assert np.isclose(result["local_mean"]["mean_bias"], 3.0)


def test_budget_comparison_matches_reference_to_candidate_folds() -> None:
    module = load_budget_module()
    fold = pd.DataFrame(
        [
            {"site_id": "A", "held_out_pass": "p1", "method": "offset_10_shots", "rmse": 1.0},
            {"site_id": "A", "held_out_pass": "p1", "method": "local_mean_10_shots", "rmse": 3.0},
        ]
    )
    main_fold = pd.DataFrame(
        [
            {"analysis_group": "primary", "site_id": "A", "held_out_pass": "p1", "model": "source_tessera_ridge", "rmse": 2.0},
            {"analysis_group": "primary", "site_id": "A", "held_out_pass": "p2", "model": "source_tessera_ridge", "rmse": 100.0},
            {"analysis_group": "primary", "site_id": "A", "held_out_pass": "p1", "model": "source_plus_local_offset", "rmse": 0.5},
            {"analysis_group": "primary", "site_id": "A", "held_out_pass": "p2", "model": "source_plus_local_offset", "rmse": 100.0},
        ]
    )

    result = module.bootstrap_comparisons(
        fold, main_fold, replicates=10, seed=7
    ).set_index("comparison")

    assert np.isclose(
        result.loc[
            "offset_10_shots_minus_source_only_tessera",
            "mean_paired_rmse_difference",
        ],
        -1.0,
    )
