from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "phase24_height", ROOT / "scripts/analyze_cairngorms_height_diagnostic_cairngorms_height.py"
)
assert SPEC is not None and SPEC.loader is not None
phase24 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(phase24)


def test_target_summary_preserves_cohort_ranges() -> None:
    broad = pd.DataFrame(
        {"lidar_meanH": [2.0, 8.0, 20.0], "lidar_p_95": [4.0, 12.0, 28.0]}
    )
    mature = pd.DataFrame(
        {
            "canopy_mean_height_m": [10.0, 11.0, 12.0],
            "canopy_p95_height_m": [17.0, 18.0, 19.0],
        }
    )
    result = phase24.target_summary(broad, mature).set_index("cohort_target")
    assert result.loc["10 m broad sample: mean height", "sd_m"] > result.loc[
        "50 m mature forest: mean height", "sd_m"
    ]


def test_target_agreement_reports_bias_and_rmse() -> None:
    frame = pd.DataFrame(
        {
            "canopy_mean_height_m": [10.0, 12.0, 14.0],
            "mean_height_50m": [9.0, 11.0, 13.0],
            "canopy_p95_height_m": [18.0, 20.0, 22.0],
            "mean_p95_height_50m": [17.0, 19.0, 21.0],
        }
    )
    result = phase24.target_agreement(frame)
    assert np.allclose(result["pearson_r"], 1.0)
    assert np.allclose(result["mean_difference_m"], 1.0)
    assert np.allclose(result["rmse_difference_m"], 1.0)


def test_quintile_bias_uses_observed_height_bins() -> None:
    observed = np.arange(1.0, 11.0)
    frame = pd.DataFrame(
        {
            "scheme": "balanced",
            "target": "canopy_p95_height_m",
            "model": "model",
            "observed": observed,
            "predicted": np.full_like(observed, observed.mean()),
        }
    )
    result = phase24.quintile_bias(frame, "balanced")
    assert result.iloc[0]["bias_m"] > 0
    assert result.iloc[-1]["bias_m"] < 0
