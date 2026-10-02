from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load_module(
    "dense_prepare", ROOT / "scripts/prepare_cairngorms_dense_10m.py"
)
worker = load_module(
    "dense_worker", ROOT / "scripts/cairngorms_dense_10m_worker.py"
)
aggregate = load_module(
    "dense_aggregate", ROOT / "scripts/aggregate_cairngorms_dense_10m.py"
)


def test_sliding_statistics_use_available_cells() -> None:
    values = np.arange(25, dtype=np.float32).reshape(5, 5)
    values[0, 0] = np.nan
    mean, count = prepare.moving_mean_and_count(values, 5)
    standard_deviation = prepare.moving_std(values, 5)
    assert count[2, 2] == 24
    assert np.isclose(mean[2, 2], np.arange(1, 25).mean())
    assert np.isclose(standard_deviation[2, 2], np.arange(1, 25).std())


def test_models_have_expected_output_shapes() -> None:
    config = worker.load_config()
    scalar = torch.zeros((2, 128 * 5), dtype=torch.float32)
    patch = torch.zeros((2, 129, 5, 5), dtype=torch.float32)
    chip = torch.zeros((2, 129, 16, 16), dtype=torch.float32)
    assert worker.build_model("summary_mlp", config, 4)(scalar).shape == (2, 4)
    assert worker.build_model("residual_patch_cnn", config, 4)(patch).shape == (
        2,
        4,
    )
    assert worker.build_model("compact_unet", config, 4)(chip).shape == (
        2,
        4,
        16,
        16,
    )


def synthetic_arrays() -> dict[str, np.ndarray]:
    embedding = np.zeros((10, 10, 128), dtype=np.int8)
    for row in range(10):
        for column in range(10):
            embedding[row, column] = row + column
    index_grid = np.full((10, 10), -1, dtype=np.int32)
    index_grid[1:9, 1:9] = np.arange(64, dtype=np.int32).reshape(8, 8)
    return {
        "embedding": embedding,
        "scale": np.ones((10, 10), dtype=np.float32),
        "valid": np.ones((10, 10), dtype=np.uint8),
        "row": np.repeat(np.arange(1, 9), 8).astype(np.int32),
        "column": np.tile(np.arange(1, 9), 8).astype(np.int32),
        "block": np.zeros(64, dtype=np.int32),
        "targets": np.arange(64 * 4, dtype=np.float32).reshape(64, 4),
        "index_grid": index_grid,
    }


def test_ordered_patch_and_summary_retain_spatial_signal() -> None:
    arrays = synthetic_arrays()
    selected = np.asarray([27], dtype=np.int64)
    ordered = worker.patch_batch(
        selected,
        arrays,
        np.zeros(128, dtype=np.float32),
        np.ones(128, dtype=np.float32),
    )
    summary = worker.summarize_patch(ordered)
    assert ordered.shape == (1, 129, 5, 5)
    assert summary.shape == (1, 640)
    assert np.allclose(summary[0, :128], ordered[0, :128, 2, 2])
    assert np.all(summary[0, 384:512] > 0)
    assert np.all(summary[0, 512:640] < 0)


def test_unet_batch_masks_targets_without_reordering() -> None:
    arrays = synthetic_arrays()
    membership = np.zeros(64, dtype=bool)
    membership[[0, 9, 18]] = True
    row_weights = np.ones(64, dtype=np.float32)
    x, y, weights = worker.unet_batch(
        [(1, 1)],
        membership,
        row_weights,
        arrays,
        np.zeros(128, dtype=np.float32),
        np.ones(128, dtype=np.float32),
        np.zeros(4, dtype=np.float32),
        np.ones(4, dtype=np.float32),
        8,
    )
    assert x.shape == (1, 129, 8, 8)
    assert y.shape == (1, 4, 8, 8)
    assert int((weights > 0).sum()) == 3
    assert np.allclose(y[0, :, 0, 0].numpy(), arrays["targets"][0])
    assert np.allclose(y[0, :, 1, 1].numpy(), arrays["targets"][9])


def test_block_balanced_metrics_are_exact_for_perfect_prediction() -> None:
    observed = np.arange(8, dtype=np.float64)
    blocks = np.asarray([0, 0, 0, 0, 1, 1, 2, 3])
    metrics = aggregate.metric_values(observed, observed.copy(), blocks)
    assert metrics["rmse"] == 0.0
    assert metrics["r2"] == 1.0
    assert np.isclose(metrics["spearman_r"], 1.0)
