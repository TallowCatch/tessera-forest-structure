import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts/run_local_adaptation_bart_few_shot_adaptation.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_local_adaptation_bart_few_shot_adaptation", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_spatial_sampler_spreads_first_round_across_cells() -> None:
    module = load_module()
    coordinates = np.array([
        [10.0, 10.0], [20.0, 20.0],
        [1010.0, 10.0], [1020.0, 20.0],
        [2010.0, 10.0], [2020.0, 20.0],
    ])
    selected = module.spatially_balanced_sample(
        np.arange(6), coordinates, 3, np.random.default_rng(42)
    )
    cells = {int(coordinates[index, 0] // 1000) for index in selected}

    assert len(selected) == len(set(selected)) == 3
    assert cells == {0, 1, 2}


def test_affine_coefficients_recover_known_transform() -> None:
    module = load_module()
    predicted = np.array([1.0, 2.0, 3.0, 4.0])
    observed = -0.5 + 1.25 * predicted

    intercept, slope = module.affine_coefficients(predicted, observed)

    assert np.isclose(intercept, -0.5)
    assert np.isclose(slope, 1.25)


def test_equal_domain_weights_do_not_depend_on_domain_row_counts() -> None:
    module = load_module()
    source_x = np.zeros((100, 1))
    source_y = np.zeros(100)
    target_x = np.ones((10, 1))
    target_y = np.ones(10)

    scaler, _ = module.fit_equal_domain_ridge(source_x, source_y, target_x, target_y, 1.0)

    assert np.isclose(scaler.mean_[0], 0.5)
