import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts/calibrate_neon_lidar_fhd.py"


def load_module():
    spec = importlib.util.spec_from_file_location("calibrate_neon_lidar_fhd", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_fit_affine_recovers_known_coefficients() -> None:
    module = load_module()
    simulated = np.array([1.0, 2.0, 3.0, 4.0])
    observed = 0.75 + 0.4 * simulated

    intercept, slope = module.fit_affine(simulated, observed)

    assert np.isclose(intercept, 0.75)
    assert np.isclose(slope, 0.4)


def test_leave_one_tile_out_never_trains_on_held_out_tile() -> None:
    module = load_module()
    frame = pd.DataFrame({
        "neon_tile_id": ["A", "A", "B", "B", "C", "C"],
        "simulated_fhd": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "fhd_normal": [1.1, 1.4, 1.8, 2.1, 2.6, 2.9],
    })

    predictions = module.leave_one_tile_out(frame)

    assert predictions["held_out_tile"].eq(predictions["neon_tile_id"]).all()
    assert predictions["training_tile_count"].eq(2).all()
    assert len(predictions) == len(frame)
