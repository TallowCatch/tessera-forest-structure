import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts/evaluate_neon_lidar_locked_bart.py"


def load_module():
    spec = importlib.util.spec_from_file_location("evaluate_neon_lidar_locked_bart", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_regression_metrics_uses_prediction_minus_observation_bias() -> None:
    module = load_module()
    observed = np.array([1.0, 2.0, 3.0])
    predicted = observed + 0.5

    metrics = module.regression_metrics(observed, predicted)

    assert np.isclose(metrics["rmse"], 0.5)
    assert np.isclose(metrics["mae"], 0.5)
    assert np.isclose(metrics["mean_bias"], 0.5)
    assert np.isclose(metrics["pearson_r"], 1.0)
