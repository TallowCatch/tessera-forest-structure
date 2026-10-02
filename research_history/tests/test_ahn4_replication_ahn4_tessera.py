from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "phase26_tessera", ROOT / "scripts/acquire_ahn4_replication_ahn4_tessera.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_sample_points_respects_broadleaf_bitmask() -> None:
    frame = pd.DataFrame(
        {
            "row_id": [0],
            "rd_x": [1000.0],
            "rd_y": [2000.0],
            "broadleaf_pixel_mask": [(1 << 0) | (1 << 12) | (1 << 24)],
        }
    )
    result = MODULE.sample_points(frame)
    assert result["patch_id"].tolist() == [0, 0, 0]
    assert result["offset_x"].tolist() == [-2, 0, 2]
    assert result["offset_y"].tolist() == [2, 0, -2]


def test_reduce_embeddings_recovers_patch_mean() -> None:
    result = MODULE.reduce_embeddings(
        np.asarray([0, 0, 1]),
        np.asarray([-1, 1, 0], dtype=np.float32),
        np.asarray([0, 0, 0], dtype=np.float32),
        np.stack(
            [
                np.zeros(128, dtype=np.float32),
                np.full(128, 2, dtype=np.float32),
                np.full(128, 5, dtype=np.float32),
            ]
        ),
    )
    assert result["count"].tolist() == [2, 1]
    assert np.allclose(result["sum_embedding"][0] / 2, 1.0)
