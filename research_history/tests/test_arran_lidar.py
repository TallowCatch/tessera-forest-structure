from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

CORE_SPEC = importlib.util.spec_from_file_location(
    "phase15_arran_core",
    ROOT / "scripts/arran_lidar_core.py",
)
core = importlib.util.module_from_spec(CORE_SPEC)
assert CORE_SPEC.loader is not None
CORE_SPEC.loader.exec_module(core)

RUNNER_SPEC = importlib.util.spec_from_file_location(
    "phase15_arran_runner",
    ROOT / "scripts/run_arran_lidar.py",
)
runner = importlib.util.module_from_spec(RUNNER_SPEC)
assert RUNNER_SPEC.loader is not None
RUNNER_SPEC.loader.exec_module(runner)


def test_active_config_is_identical_to_protocol_freeze() -> None:
    assert runner.CONFIG_PATH.read_bytes() == runner.PROTOCOL_PATH.read_bytes()
    config = runner.load_config()
    assert config["source"]["survey_year"] == 2025
    assert config["metric_generation"]["metric_resolution_m"] == 10.0
    assert config["tessera"]["dataset_version"] == "1.0"
    assert config["tessera"]["dataset_variant"] == "vultr"


def test_shannon_uses_frozen_right_closed_height_classes() -> None:
    breaks = np.array([-1.0, 2.0, 5.0, 10.0, 15.0, 35.0])
    one_per_class = np.array([2.0, 5.0, 10.0, 15.0, 35.0])
    assert np.isclose(
        core.shannon_from_bins(one_per_class, breaks), np.log(5.0)
    )
    assert core.shannon_from_bins(np.array([2.0, 2.0]), breaks) == 0.0


def test_vertical_complexity_is_one_for_uniform_bins() -> None:
    values = np.arange(0.5, 10.0, 1.0)
    assert np.isclose(
        core.vertical_complexity_index(values, zmax=10.0, bin_width=1.0),
        1.0,
    )


def test_cell_metrics_return_expected_simple_canopy_values() -> None:
    height = np.array([0.0, 1.0, 3.0, 6.0, 12.0, 20.0])
    metrics = core.calculate_cell_metrics(
        height,
        return_number=np.ones(len(height), dtype=np.int16),
        number_of_returns=np.ones(len(height), dtype=np.int16),
        scan_angle_degrees=np.zeros(len(height)),
        h_cutoff=1.3,
        gap_threshold=2.0,
        zmax=35.0,
        shannon_breaks=np.array([-1.0, 2.0, 5.0, 10.0, 15.0, 35.0]),
    )
    assert metrics["lidar_maxH"] == 20.0
    assert metrics["lidar_Cov"] == 4.0 / 6.0
    assert metrics["lidar_gapFrac"] == 2.0 / 6.0
    expected = -((2.0 / 6.0) * np.log(2.0 / 6.0) + 4 * (1.0 / 6.0) * np.log(1.0 / 6.0))
    assert np.isclose(metrics["lidar_canopy_shannon"], expected)


def test_forest_mask_rejects_short_and_impossible_cells() -> None:
    frame = pd.DataFrame(
        {
            "lidar_maxH": np.array([20.0, 20.0, 80.0]),
            "lidar_meanH": np.array([10.0, 10.0, 10.0]),
            "lidar_p_95": np.array([18.0, 1.0, 18.0]),
            "lidar_p_999": np.array([19.0, 19.0, 70.0]),
            "lidar_Cov": np.array([0.7, 0.7, 0.7]),
            "lidar_canopy_shannon": np.array([1.0, 1.0, 1.0]),
        }
    )
    rules = {
        "minimum_p95_height_m": 3.0,
        "maximum_p999_height_m": 60.0,
        "maximum_return_height_m": 60.0,
        "minimum_canopy_cover": 0.1,
        "maximum_canopy_cover": 1.0,
        "maximum_canopy_shannon": 1.61,
    }
    assert core.forest_quality_mask(frame, rules).tolist() == [
        True,
        False,
        False,
    ]
