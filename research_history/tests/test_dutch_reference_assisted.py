from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.evaluate_dutch_reference_assisted import (
    affine_correction,
    balanced_sample,
    equal_domain_weights,
    metric_values,
)


def test_balanced_sample_is_reproducible_and_spread_across_blocks() -> None:
    frame = pd.DataFrame(
        {
            "sampling_block": ["a", "a", "b", "b", "c", "c"],
            "rd_x": [0, 1, 1000, 1001, 2000, 2001],
            "rd_y": [0, 1, 0, 1, 0, 1],
        }
    )
    first = balanced_sample(frame.index.to_numpy(), frame, 3, np.random.default_rng(4))
    second = balanced_sample(frame.index.to_numpy(), frame, 3, np.random.default_rng(4))
    assert np.array_equal(first, second)
    assert frame.loc[first, "sampling_block"].nunique() == 3


def test_equal_domain_weights_give_each_domain_half_weight() -> None:
    weights = equal_domain_weights(4, 2)
    assert np.isclose(weights[:4].sum(), 0.5)
    assert np.isclose(weights[4:].sum(), 0.5)


def test_affine_correction_recovers_known_relationship() -> None:
    source_prediction = np.array([1.0, 2.0, 3.0])
    observed = 2.0 + 3.0 * source_prediction
    intercept, slope = affine_correction(source_prediction, observed)
    assert np.isclose(intercept, 2.0)
    assert np.isclose(slope, 3.0)


def test_metric_values_reports_perfect_prediction() -> None:
    observed = np.array([1.0, 2.0, 3.0, 4.0])
    metrics = metric_values(observed, observed.copy(), np.array(["a", "a", "b", "b"]))
    assert np.isclose(metrics["rmse"], 0.0)
    assert np.isclose(metrics["r2"], 1.0)
