from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "phase26_cohort", ROOT / "scripts/prepare_ahn4_replication_ahn4_cohort.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_patch_sum_aggregates_non_overlapping_five_cell_units() -> None:
    values = np.ones((10, 15), dtype=np.int16)
    result = MODULE.patch_sum(values, 5)
    assert result.shape == (2, 3)
    assert np.all(result == 25)


def test_patch_masked_mean_uses_only_valid_forest_cells() -> None:
    values = np.arange(25, dtype=np.float32).reshape(5, 5)
    valid = np.ones((5, 5), dtype=bool)
    valid[0, :5] = False
    result = MODULE.patch_masked_mean(values, valid, 5)
    assert np.isclose(result[0, 0], values[1:].mean())


def test_patch_bitmask_preserves_row_major_pixel_membership() -> None:
    values = np.zeros((5, 5), dtype=bool)
    values[0, 0] = True
    values[2, 3] = True
    result = MODULE.patch_bitmask(values, 5)
    assert int(result[0, 0]) == (1 << 0) + (1 << 13)


def test_balanced_fold_assignment_is_complete_and_deterministic() -> None:
    frame = pd.DataFrame(
        {
            "spatial_block": ["a"] * 7 + ["b"] * 6 + ["c"] * 5 + ["d"] * 4,
        }
    )
    first = MODULE.assign_balanced_folds(frame, folds=3, seed=42)
    second = MODULE.assign_balanced_folds(frame, folds=3, seed=42)
    assert first == second
    assert set(first) == {"a", "b", "c", "d"}
    assert set(first.values()) <= {0, 1, 2}
