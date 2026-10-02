"""Focused tests for the resumable Dutch temporal-transfer workflow."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd

RUNNER = Path(__file__).resolve().parents[1] / "workflows" / "dutch_temporal_transfer" / "run.py"
SPEC = spec_from_file_location("dutch_temporal_transfer", RUNNER)
assert SPEC and SPEC.loader
WORKFLOW = module_from_spec(SPEC)
SPEC.loader.exec_module(WORKFLOW)


def test_acquisition_year_accepts_supported_ahn_encodings() -> None:
    values = np.array([0, 2021, 20210517, 2021123, -9999])

    assert WORKFLOW.acquisition_year(values).tolist() == [0, 2021, 2021, 2021, 0]


def test_patch_bitmask_and_popcount_preserve_forest_pixels() -> None:
    forest = np.zeros((5, 5), dtype=bool)
    forest[0, 0] = True
    forest[4, 4] = True

    masks = WORKFLOW.patch_bitmask(forest, 5)
    counts = WORKFLOW.popcount(pd.Series(masks.reshape(-1)))

    assert masks.shape == (1, 1)
    assert counts.tolist() == [2]


def test_balanced_folds_is_deterministic_and_assigns_every_block() -> None:
    blocks = pd.Series(["a"] * 8 + ["b"] * 5 + ["c"] * 3 + ["d"] * 2)

    first = WORKFLOW.balanced_folds(blocks, folds=3, seed=42)
    second = WORKFLOW.balanced_folds(blocks, folds=3, seed=42)

    assert first == second
    assert set(first) == {"a", "b", "c", "d"}
    assert set(first.values()) <= {0, 1, 2}


def test_constant_prediction_has_undefined_rank_correlation() -> None:
    metrics = WORKFLOW.regression_metrics(np.array([1.0, 2.0, 3.0]), np.zeros(3))

    assert np.isnan(metrics["spearman"])
    assert np.isfinite(metrics["rmse"])


def test_block_bootstrap_matches_single_block_rmse_contrast() -> None:
    predictions = pd.DataFrame(
        {
            "outcome": ["height"] * 3,
            "site_code": ["site"] * 3,
            "fold_block": ["block"] * 3,
            "observed_ahn3": [1.0, 2.0, 3.0],
            "observed_ahn4": [2.0, 3.0, 4.0],
            "predicted_ahn3": [1.0, 2.0, 2.0],
            "predicted_ahn4_temporal": [1.0, 2.0, 3.0],
            "predicted_ahn4_contemporary": [2.0, 3.0, 4.0],
            "observed_change": [1.0, 1.0, 1.0],
            "predicted_change": [0.0, 0.0, 1.0],
        }
    )
    parameters = {"validation": {"bootstrap_seed": 7, "bootstrap_replicates": 20}}

    result = WORKFLOW.bootstrap_penalties(predictions, parameters).set_index("contrast")

    assert np.isclose(
        result.loc["temporal_minus_same_time", "median_delta_rmse"],
        1.0 - np.sqrt(1.0 / 3.0),
    )
    assert np.isclose(result.loc["temporal_minus_contemporary", "median_delta_rmse"], 1.0)
    assert np.isclose(result.loc["temporal_minus_no_change", "median_delta_rmse"], 0.0)
