from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from reflection_augmentation_flip_mlp_common import (  # noqa: E402
    CHANNELS,
    DX,
    DY,
    SUMMARY_WIDTH,
    reflect_summary,
    reflection_statistics,
)


def test_reflections_only_reverse_corresponding_gradients() -> None:
    values = np.arange(2 * SUMMARY_WIDTH, dtype=np.float32).reshape(2, -1)
    horizontal = reflect_summary(values, -1.0, 1.0)
    vertical = reflect_summary(values, 1.0, -1.0)
    both = reflect_summary(values, -1.0, -1.0)
    np.testing.assert_array_equal(horizontal[:, : CHANNELS * 3], values[:, : CHANNELS * 3])
    np.testing.assert_array_equal(vertical[:, : CHANNELS * 3], values[:, : CHANNELS * 3])
    np.testing.assert_array_equal(horizontal[:, DX], -values[:, DX])
    np.testing.assert_array_equal(horizontal[:, DY], values[:, DY])
    np.testing.assert_array_equal(vertical[:, DX], values[:, DX])
    np.testing.assert_array_equal(vertical[:, DY], -values[:, DY])
    np.testing.assert_array_equal(both[:, DX], -values[:, DX])
    np.testing.assert_array_equal(both[:, DY], -values[:, DY])


def test_vector_reflections_are_applied_per_row() -> None:
    values = np.ones((3, SUMMARY_WIDTH), dtype=np.float32)
    result = reflect_summary(
        values,
        np.asarray([1.0, -1.0, 1.0], dtype=np.float32),
        np.asarray([-1.0, 1.0, -1.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(result[:, DX.start], [1.0, -1.0, 1.0])
    np.testing.assert_array_equal(result[:, DY.start], [-1.0, 1.0, -1.0])


def test_reflection_statistics_match_explicit_four_copy_population() -> None:
    rng = np.random.default_rng(7)
    values = rng.normal(size=(11, SUMMARY_WIDTH)).astype(np.float32)
    indices = np.asarray([0, 2, 3, 6, 8, 10], dtype=np.int64)
    mean, sd = reflection_statistics(values, indices)
    explicit = np.concatenate(
        [
            reflect_summary(values[indices], horizontal, vertical)
            for horizontal, vertical in [
                (1.0, 1.0),
                (-1.0, 1.0),
                (1.0, -1.0),
                (-1.0, -1.0),
            ]
        ],
        axis=0,
    )
    np.testing.assert_allclose(mean, explicit.mean(axis=0), atol=1e-6)
    np.testing.assert_allclose(sd, explicit.std(axis=0), atol=1e-6)

