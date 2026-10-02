import numpy as np

from scripts.evaluate_scotland_netherlands_transfer import (
    equal_block_weights,
    feature_panels,
    metrics,
)


def test_common_feature_panels_drop_centre_values() -> None:
    summary = np.arange(2 * 640, dtype=np.float32).reshape(2, 640)
    panels = feature_panels(summary)
    assert panels["mean"].shape == (2, 128)
    assert panels["context"].shape == (2, 512)
    np.testing.assert_array_equal(panels["mean"], summary[:, 128:256])
    np.testing.assert_array_equal(panels["context"], summary[:, 128:640])


def test_equal_block_weights_give_equal_total_weight() -> None:
    blocks = np.asarray(["a", "a", "a", "b"])
    weights = equal_block_weights(blocks)
    np.testing.assert_allclose(weights[blocks == "a"].sum(), weights[blocks == "b"].sum())
    np.testing.assert_allclose(weights.mean(), 1.0)


def test_metrics_report_exact_prediction() -> None:
    observed = np.asarray([1.0, 2.0, 3.0, 4.0])
    values = metrics(observed, observed.copy(), np.asarray(["a", "a", "b", "b"]))
    assert values["rmse"] == 0.0
    assert values["bias"] == 0.0
    assert values["r2"] == 1.0
