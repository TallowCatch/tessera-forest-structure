from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import prepare_cairngorms_components as phase20  # noqa: E402
import aggregate_cairngorms_components as aggregate  # noqa: E402


def test_canopy_metrics_have_declared_units() -> None:
    values = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    result = phase20.canopy_metrics(values)
    assert result["valid"] == 5
    assert np.isclose(result["sd"], np.std(values, ddof=1))
    assert np.isclose(result["rcv"], 1.0)
    assert np.isclose(result["gap"], 0.4)


def test_intersect_indices_preserves_frozen_order() -> None:
    indices = np.asarray([4, 1, 3, 0], dtype=np.int64)
    valid = np.asarray([True, False, True, True, True])
    assert phase20.intersect_indices(indices, valid).tolist() == [4, 3, 0]


def test_nonzero_quality_mask_values_are_accepted() -> None:
    values = np.asarray([1.0, 0.0, np.nan])
    accepted = np.isfinite(values) & (values != 0)
    assert accepted.tolist() == [True, False, False]


def test_phase20_config_freezes_three_predictor_sets() -> None:
    config = phase20.load_config()
    assert config["evaluation"]["models"] == ["conventional", "tessera", "fused"]
    assert config["targets"]["primary"] == [
        "canopy_surface_sd_m",
        "canopy_surface_rcv",
        "canopy_open_fraction",
        "mean_local_vci_2m",
    ]


def test_block_bootstrap_point_estimate_uses_equal_block_weight() -> None:
    import pandas as pd

    frame = pd.DataFrame(
        {
            "row_id": [0, 0, 1, 1, 2, 2],
            "spatial_block": ["a", "a", "a", "a", "b", "b"],
            "observed": [0.0, 0.0, 2.0, 2.0, 10.0, 10.0],
            "model": ["candidate", "reference"] * 3,
            "predicted": [0.0, 1.0, 2.0, 3.0, 8.0, 7.0],
        }
    )
    result = aggregate.bootstrap_comparison(
        frame, "candidate", "reference", 20, np.random.default_rng(4)
    )
    expected = np.sqrt((0.0 + 4.0) / 2) - np.sqrt((1.0 + 9.0) / 2)
    assert np.isclose(result["rmse_difference"], expected)
