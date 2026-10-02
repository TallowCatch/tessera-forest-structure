import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts/prepare_local_adaptation_bart_coral_predictions.py"


def load_module():
    spec = importlib.util.spec_from_file_location("prepare_local_adaptation_bart_coral_predictions", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_symmetric_matrix_square_root_round_trip() -> None:
    module = load_module()
    matrix = np.array([[2.0, 0.4], [0.4, 1.0]])
    root = module.symmetric_matrix_power(matrix, 0.5, 0.0)

    assert np.allclose(root @ root, matrix, atol=1e-10)


def test_coral_reduces_covariance_distance() -> None:
    module = load_module()
    rng = np.random.default_rng(123)
    source = rng.normal(size=(1000, 3))
    target = rng.normal(size=(800, 3)) @ np.array([
        [2.0, 0.3, 0.0],
        [0.0, 0.5, 0.2],
        [0.0, 0.0, 1.5],
    ]) + np.array([1.0, -2.0, 0.5])

    aligned, _, diagnostics = module.coral_transform(source, target, 1e-6)

    assert diagnostics["source_target_covariance_frobenius_after"] < diagnostics[
        "source_target_covariance_frobenius_before"
    ]
    assert np.allclose(aligned.mean(axis=0), target.mean(axis=0), atol=1e-10)
