from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "phase17_cairngorms",
    ROOT / "scripts/run_cairngorms_heterogeneity.py",
)
phase17 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(phase17)


def test_block_mean_uses_available_cells() -> None:
    values = np.arange(25, dtype=np.float32).reshape(5, 5)
    values[0, 0] = np.nan
    mean, count = phase17.block_mean(values, 5)
    assert count.item() == 24
    assert np.isclose(mean.item(), np.arange(1, 25).mean())


def test_block_std_is_zero_for_constant_patch() -> None:
    values = np.full((10, 10), 4.0, dtype=np.float32)
    result = phase17.block_std(values, 5)
    assert result.shape == (2, 2)
    assert np.allclose(result, 0.0)


def test_reduce_tile_embeddings_recovers_linear_gradients() -> None:
    offsets_x = np.tile(np.arange(-2, 3), 5).astype(np.float32)
    offsets_y = np.repeat(np.arange(2, -3, -1), 5).astype(np.float32)
    patch_id = np.zeros(25, dtype=np.int64)
    base = np.arange(phase17.EMBEDDING_DIMENSIONS, dtype=np.float32)
    embeddings = (
        base[None, :]
        + 2.0 * offsets_x[:, None]
        - 3.0 * offsets_y[:, None]
    )
    reduced = phase17.reduce_tile_embeddings(
        patch_id, offsets_x, offsets_y, embeddings
    )
    count = reduced["count"].astype(np.float32)
    mean = reduced["sum_embedding"] / count[:, None]
    denominator_x = (
        reduced["sum_x2"]
        - reduced["sum_x"] * reduced["sum_x"] / count
    )
    denominator_y = (
        reduced["sum_y2"]
        - reduced["sum_y"] * reduced["sum_y"] / count
    )
    gradient_x = (
        reduced["sum_x_embedding"]
        - reduced["sum_x"][:, None] * reduced["sum_embedding"] / count[:, None]
    ) / denominator_x[:, None]
    gradient_y = (
        reduced["sum_y_embedding"]
        - reduced["sum_y"][:, None] * reduced["sum_embedding"] / count[:, None]
    ) / denominator_y[:, None]
    assert np.allclose(mean[0], base)
    assert np.allclose(gradient_x, 2.0)
    assert np.allclose(gradient_y, -3.0)
    assert reduced["centre_valid"].tolist() == [True]
    assert np.allclose(reduced["centre_embedding"][0], base)


def test_metric_values_are_exact_for_perfect_prediction() -> None:
    values = phase17.metric_values(
        np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0])
    )
    assert values["rmse"] == 0.0
    assert values["r2"] == 1.0
    assert values["spearman_r"] == 1.0


def test_feature_columns_ignore_validity_flag() -> None:
    frame = pd.DataFrame(
        {
            "tessera_centre_000": [1.0],
            "tessera_centre_001": [2.0],
            "tessera_centre_valid": [True],
        }
    )
    original = phase17.EMBEDDING_DIMENSIONS
    phase17.EMBEDDING_DIMENSIONS = 2
    try:
        assert phase17.feature_columns(frame, "centre") == [
            "tessera_centre_000",
            "tessera_centre_001",
        ]
    finally:
        phase17.EMBEDDING_DIMENSIONS = original
