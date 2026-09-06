from __future__ import annotations

import numpy as np

from tessera_forest_structure.demo import run_demo
from tessera_forest_structure.metrics import spearman_rank_correlation
from tessera_forest_structure.repository import ROOT


def test_reference_correction_improves_absolute_error_without_changing_rank() -> None:
    result = run_demo(ROOT / "examples/demo_data/reference_transfer.csv")
    zero = result["zero_shot"]
    assisted = result["reference_assisted"]
    assert assisted["rmse"] < zero["rmse"]
    assert assisted["r2"] > zero["r2"]
    assert np.isclose(assisted["spearman"], zero["spearman"])
    assert np.isclose(result["estimated_offset"], 5.0, atol=0.02)


def test_spearman_uses_average_ranks_for_ties() -> None:
    observed = np.asarray([1.0, 1.0, 2.0, 3.0])
    predicted = np.asarray([4.0, 4.0, 5.0, 6.0])
    assert np.isclose(spearman_rank_correlation(observed, predicted), 1.0)
