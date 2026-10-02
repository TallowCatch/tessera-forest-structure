from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "phase14_scotland",
    ROOT / "scripts/run_scotland_lidar.py",
)
phase14 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(phase14)


def test_quality_mask_rejects_nonforest_and_impossible_height() -> None:
    core = {
        "lidar_maxH": np.array([20.0, 20.0, 80.0]),
        "lidar_meanH": np.array([10.0, 10.0, 10.0]),
        "lidar_p_95": np.array([18.0, 1.0, 18.0]),
        "lidar_p_999": np.array([19.0, 19.0, 70.0]),
        "lidar_Cov": np.array([0.7, 0.7, 0.7]),
        "lidar_canopy_shannon": np.array([1.0, 1.0, 1.0]),
    }
    rules = {
        "minimum_p95_height_m": 3.0,
        "maximum_p999_height_m": 60.0,
        "maximum_return_height_m": 60.0,
        "minimum_canopy_cover": 0.1,
        "maximum_canopy_cover": 1.0,
        "maximum_canopy_shannon": 1.61,
    }
    assert phase14.quality_mask(core, rules).tolist() == [True, False, False]


def test_tile_name_parser_round_trip() -> None:
    name = phase14.tessera.tile_name((-4.15, 57.05))
    assert name == "grid_-4.15_57.05"
    assert phase14.parse_tile_name(name) == (-4.15, 57.05)


def test_metric_values_are_exact_for_perfect_predictions() -> None:
    values = phase14.metric_values(
        np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0])
    )
    assert values["rmse"] == 0.0
    assert values["mae"] == 0.0
    assert values["r2"] == 1.0
    assert values["spearman_r"] == 1.0
    assert values["bias"] == 0.0


def test_metric_values_leave_constant_prediction_correlations_undefined() -> None:
    values = phase14.metric_values(
        np.array([0.0, 1.0, 2.0]), np.array([1.0, 1.0, 1.0])
    )
    assert np.isnan(values["pearson_r"])
    assert np.isnan(values["spearman_r"])


def test_spatial_weights_give_equal_total_weight_to_blocks() -> None:
    blocks = pd.Series(["a", "a", "a", "b"])
    weights = phase14.spatial_weights(blocks)
    assert np.isclose(weights[:3].sum(), weights[3:].sum())


def test_moran_i_positive_for_smooth_block_pattern() -> None:
    frame = pd.DataFrame(
        {
            "block_x": [0, 0, 1, 1],
            "block_y": [0, 1, 0, 1],
            "residual": [1.0, 1.0, -1.0, -1.0],
        }
    )
    value = phase14.moran_i(frame)
    assert np.isfinite(value)
