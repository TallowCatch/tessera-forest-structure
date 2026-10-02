from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_script():
    specification = importlib.util.spec_from_file_location(
        "phase13_error_decomposition",
        ROOT / "scripts" / "analyze_acquisition_replication_error_decomposition.py",
    )
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    specification.loader.exec_module(module)
    return module


def test_error_components_obey_mse_identity() -> None:
    module = load_script()
    observed = np.array([1.0, 2.0, 4.0, 5.0])
    predicted = np.array([2.0, 2.5, 3.0, 4.0])

    components = module.error_components(observed, predicted)

    assert np.isclose(
        components["total_mse"],
        components["squared_bias"] + components["centered_mse"],
    )


def test_constant_offset_changes_bias_but_not_spatial_error() -> None:
    module = load_script()
    observed = np.array([1.0, 2.0, 4.0, 5.0])
    predicted = np.array([3.0, 3.5, 4.0, 5.0])
    shifted = predicted - 1.5

    original = module.error_components(observed, predicted)
    corrected = module.error_components(observed, shifted)

    assert np.isclose(original["centered_mse"], corrected["centered_mse"])
    assert not np.isclose(original["squared_bias"], corrected["squared_bias"])


def test_local_mean_has_zero_spatial_skill() -> None:
    module = load_script()
    observed = np.array([1.0, 2.0, 4.0, 5.0])
    predicted = np.full_like(observed, 3.0)

    components = module.error_components(observed, predicted)

    assert np.isclose(components["centered_mse"], components["observed_variance"])
