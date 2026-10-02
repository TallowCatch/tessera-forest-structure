from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "phase26_conventional", ROOT / "scripts/acquire_ahn4_replication_ahn4_conventional.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_context_points_preserves_all_pixels_and_marks_forest() -> None:
    frame = pd.DataFrame(
        {
            "row_id": [0],
            "rd_x": [1000.0],
            "rd_y": [2000.0],
            "broadleaf_pixel_mask": [(1 << 0) | (1 << 24)],
        }
    )
    result = MODULE.context_points(frame)
    assert len(result) == 25
    assert result["broadleaf"].sum() == 2
    assert result.iloc[0]["rd_x"] == 980.0
    assert result.iloc[24]["rd_y"] == 1980.0


def test_reduce_features_uses_valid_pixels_only() -> None:
    values = np.arange(50, dtype=np.float32).reshape(25, 2)
    valid = np.ones(25, dtype=bool)
    valid[:5] = False
    rows = np.repeat(np.arange(5), 5).astype(np.float32)
    columns = np.tile(np.arange(5), 5).astype(np.float32)
    result = MODULE.reduce_features(values, valid, columns - 2, 2 - rows, 1)
    assert result["count"].tolist() == [20]
    assert np.allclose(result["mean"][0], values[5:].mean(axis=0))
