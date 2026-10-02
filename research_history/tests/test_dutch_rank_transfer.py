from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.evaluate_dutch_rank_transfer import (
    exact_budget_available,
    pattern_metrics,
    summarize_metrics,
)


def test_offset_changes_absolute_error_but_preserves_rank() -> None:
    test = pd.DataFrame(
        {
            "fold_block": ["a", "a", "b", "b", "c", "c"],
            "observed": [10.0, 11.0, 20.0, 21.0, 30.0, 31.0],
        }
    )
    shifted = test["observed"].to_numpy() + 5.0
    corrected = shifted - 5.0
    before = pattern_metrics(test, shifted)
    after = pattern_metrics(test, corrected)
    assert np.isclose(before["unit_spearman"], after["unit_spearman"])
    assert np.isclose(before["block_spearman"], after["block_spearman"])
    assert before["rmse"] > after["rmse"]


def test_block_rank_uses_block_means() -> None:
    test = pd.DataFrame(
        {
            "fold_block": ["a", "a", "b", "b", "c", "c"],
            "observed": [0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
        }
    )
    prediction = np.array([2.0, 0.0, 6.0, 4.0, 10.0, 8.0])
    values = pattern_metrics(test, prediction)
    assert np.isclose(values["block_spearman"], 1.0)
    assert values["unit_spearman"] < 1.0


def test_exact_200_budget_is_not_silently_capped() -> None:
    assert exact_budget_available(200, 200)
    assert not exact_budget_available(199, 200)


def test_summary_counts_unique_site_folds_not_reference_repetitions() -> None:
    rows = []
    for replicate in range(10):
        for site, fold in [("a", 0), ("b", 0), ("b", 1)]:
            rows.append(
                {
                    "target_site": site,
                    "forest_group": "broadleaf",
                    "fold": fold,
                    "target": "ahn4_height_cv",
                    "budget": 2,
                    "method": "target_only_ridge",
                    "eligible_200": True,
                    "rmse": 1.0,
                    "r2": 0.1,
                    "unit_spearman": 0.2,
                    "block_spearman": 0.3,
                    "centred_rmse": 0.9,
                    "bias": 0.0,
                }
            )
    metrics = pd.DataFrame(rows)
    config = {"evaluation": {"full_network_max_budget": 100}}
    summary, _, _ = summarize_metrics(metrics, config)
    assert (summary["site_folds"] == 3).all()
