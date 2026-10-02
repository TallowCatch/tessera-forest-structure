from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load_module(
    "profile_prepare", ROOT / "scripts/prepare_cairngorms_dense_profile.py"
)
worker = load_module(
    "profile_worker", ROOT / "scripts/cairngorms_dense_profile_worker.py"
)
aggregate = load_module(
    "profile_aggregate", ROOT / "scripts/aggregate_cairngorms_dense_profile.py"
)


def test_profile_normalization_and_entropy() -> None:
    volumes = np.asarray(
        [[1.0, 0.0], [1.0, 1.0], [2.0, 0.0]], dtype=np.float32
    )
    profile = prepare.normalize_profile(volumes, 1e-6).T
    assert np.allclose(profile.sum(axis=1), 1.0)
    assert np.allclose(profile[0], [0.25, 0.25, 0.5])
    entropy = prepare.profile_entropy(profile)
    assert np.isclose(entropy[1], 0.0)


def test_softmax_profile_is_on_simplex_and_exact_match_has_zero_loss() -> None:
    config = worker.load_config()
    logits = torch.tensor([[0.2, -0.3, 0.7]], dtype=torch.float32)
    observed = torch.softmax(logits, dim=1)
    probabilities = worker.profile_probabilities(logits, 1)
    loss = worker.profile_loss_rows(logits, observed, 1, config)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(1))
    assert float(loss.item()) < 1e-8


def test_profile_metrics_are_exact_for_perfect_prediction() -> None:
    observed = np.asarray(
        [[0.2, 0.3, 0.5], [0.5, 0.2, 0.3], [0.1, 0.7, 0.2]],
        dtype=np.float64,
    )
    blocks = np.asarray([0, 1, 2])
    metrics = aggregate.profile_metrics(observed, observed.copy(), blocks)
    assert metrics["profile_rmse"] == 0.0
    assert metrics["mean_jensen_shannon_distance"] == 0.0
    assert metrics["entropy_r2"] == 1.0


def test_diagnostic_predictions_use_training_rows_only_and_sum_to_one() -> None:
    config = worker.load_config()
    targets = np.asarray(
        [
            [0.2, 0.3, 0.5],
            [0.3, 0.2, 0.5],
            [0.4, 0.4, 0.2],
            [0.1, 0.6, 0.3],
        ],
        dtype=np.float64,
    )
    covariates = np.asarray(
        [[10, 0.5, 0.2], [12, 0.6, 0.1], [20, 0.7, 0.05], [25, 0.8, 0.02]],
        dtype=np.float64,
    )
    predictions = aggregate.diagnostic_predictions(
        "lidar_height_only_ridge",
        np.asarray([0, 1, 2]),
        np.asarray([3]),
        targets,
        covariates,
        np.asarray([0, 1, 2, 3]),
        config,
    )
    assert predictions.shape == (1, 3)
    assert np.all(predictions >= 0)
    assert np.allclose(predictions.sum(axis=1), 1.0)
