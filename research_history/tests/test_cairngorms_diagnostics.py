from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_cairngorms_diagnostics.py"
SPEC = importlib.util.spec_from_file_location("phase19_diagnostics", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
phase19 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(phase19)


def test_context_points_remain_centred_on_target() -> None:
    targets = pd.DataFrame(
        {"row_id": [4], "bng_x": [100.0], "bng_y": [200.0]}
    )
    points = phase19.context_points(targets, 3)
    assert len(points["bng_x"]) == 9
    assert points["bng_x"][4] == 100.0
    assert points["bng_y"][4] == 200.0
    assert points["bng_x"].min() == 90.0
    assert points["bng_x"].max() == 110.0
    assert points["bng_y"].min() == 190.0
    assert points["bng_y"].max() == 210.0


def test_patch_summary_recovers_mean_spread_and_gradient() -> None:
    x = np.tile(np.arange(3, dtype=np.int8), (3, 1))
    quantized = x[None, None, :, :]
    scales = np.ones((1, 3, 3), dtype=np.float32)
    valid = np.ones((1, 3, 3), dtype=np.uint8)
    summary = phase19.summarise_patch_batch(quantized, scales, valid)
    assert summary.shape == (1, 5)
    np.testing.assert_allclose(
        summary[0],
        [1.0, 1.0, np.sqrt(2.0 / 3.0), 1.0, 0.0],
        rtol=1e-5,
        atol=1e-6,
    )


def test_conventional_columns_exclude_observation_counts() -> None:
    frame = pd.DataFrame(
        {
            "s2_red_median": [0.1],
            "s2_valid_observation_count": [10],
            "s1_ascending_valid_observation_count": [10],
            "terrain_elevation_m": [300.0],
        }
    )
    assert phase19.conventional_predictor_columns(frame) == [
        "s2_red_median",
        "terrain_elevation_m",
    ]


def test_paired_bootstrap_preserves_complete_blocks() -> None:
    rows = []
    for model, predictions in [
        ("candidate", [0.0, 1.0, 2.0, 3.0]),
        ("reference", [0.5, 1.5, 2.5, 3.5]),
    ]:
        for row_id, predicted in enumerate(predictions):
            rows.append(
                {
                    "row_id": row_id,
                    "target": "target",
                    "model": model,
                    "observed": float(row_id),
                    "predicted": predicted,
                    "spatial_block": "a" if row_id < 2 else "b",
                }
            )
    result = phase19.paired_block_comparison(
        pd.DataFrame(rows), "candidate", "reference", 100, 9
    )
    assert result["rows"] == 4
    assert result["blocks"] == 2
    assert result["candidate_rmse"] == 0.0
    assert result["rmse_delta_ci_upper"] < 0


def test_frozen_design_contains_mlp_and_context_comparisons() -> None:
    config = phase19.load_config()
    names = {value["name"] for value in config["comparisons"]}
    assert "mlp_vs_hgb_tessera_50m" in names
    assert "context_90m_vs_10m" in names
    assert len(config["model_specs"]) == 15
