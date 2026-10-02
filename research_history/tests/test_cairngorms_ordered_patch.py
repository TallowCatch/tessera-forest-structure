from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


worker = load_module(
    "ordered_patch_worker", ROOT / "scripts/cairngorms_ordered_patch_worker.py"
)
aggregate = load_module(
    "ordered_patch_aggregate",
    ROOT / "scripts/aggregate_cairngorms_ordered_patch.py",
)


def test_ordered_models_accept_complete_patch() -> None:
    config = worker.load_config()
    batch = torch.zeros((3, 129, 5, 5), dtype=torch.float32)
    for name in config["models"]:
        model = worker.build_model(name, config, output_dimensions=8)
        values = batch.flatten(start_dim=1) if "mlp" in name else batch
        assert model(values).shape == (3, 8)


def test_input_batch_retains_order_and_mask() -> None:
    quantized = np.arange(2 * 128 * 5 * 5, dtype=np.uint8).reshape(
        2, 128, 5, 5
    )
    arrays = {
        "patch_quantized": quantized,
        "patch_scales": np.ones((2, 5, 5), dtype=np.float32),
        "patch_valid": np.ones((2, 5, 5), dtype=bool),
    }
    mean = np.zeros(128, dtype=np.float32)
    standard_deviation = np.ones(128, dtype=np.float32)
    cnn = worker.input_batch(
        "residual_patch_cnn",
        np.asarray([0, 1]),
        arrays,
        mean,
        standard_deviation,
    )
    flattened = worker.input_batch(
        "ordered_mlp",
        np.asarray([0, 1]),
        arrays,
        mean,
        standard_deviation,
    )
    assert cnn.shape == (2, 129, 5, 5)
    assert flattened.shape == (2, 129 * 5 * 5)
    assert torch.equal(flattened, cnn.flatten(start_dim=1))


def test_paired_block_bootstrap_preserves_improvement_direction() -> None:
    rows = []
    for block in range(8):
        for row in range(5):
            observed = float(block + row / 10)
            rows.append(
                {
                    "row_id": block * 5 + row,
                    "spatial_block": str(block),
                    "observed": observed,
                    "baseline": observed + 1.0,
                    "candidate": observed + 0.5,
                }
            )
    delta, lower, upper = aggregate.paired_block_bootstrap(
        pd.DataFrame(rows), replicates=100, seed=4
    )
    assert delta < 0
    assert lower < 0
    assert upper < 0
