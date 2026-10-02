from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "phase26_evaluation", ROOT / "scripts/evaluate_ahn4_replication_ahn4.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_equal_block_weights_give_each_block_equal_total_weight() -> None:
    blocks = pd.Series(["a", "a", "a", "b"])
    weights = MODULE.equal_block_weights(blocks)
    assert np.isclose(weights[:3].sum(), weights[3:].sum())


def test_metric_values_uses_weighted_error() -> None:
    result = MODULE.metric_values(
        np.asarray([0.0, 0.0, 1.0, 2.0]),
        np.asarray([0.0, 0.0, 1.0, 1.0]),
        np.asarray([1.0, 1.0, 1.0, 1.0]),
        np.asarray(["a", "a", "b", "b"]),
    )
    assert np.isclose(result["rmse"], 0.5)
