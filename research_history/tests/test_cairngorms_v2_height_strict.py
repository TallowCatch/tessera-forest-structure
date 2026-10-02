from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import cairngorms_dense_10m_worker as dense
import cairngorms_v2_height_strict_worker as worker


def config() -> dict:
    return yaml.safe_load(
        (ROOT / "configs/cairngorms_v2_height_strict_unet.yaml").read_text()
    )["phase31_cairngorms_v2_height_strict_unet"]


def test_strict_corrections_are_frozen() -> None:
    values = config()["strict_corrections"]
    assert values["assign_each_target_to_one_chip"] is True
    assert values["supervise_each_target_once_per_epoch"] is True
    assert values["normalize_loss_by_fixed_expected_batch_weight"] is True
    assert values["enforce_minimum_selected_epoch"] is True


def test_every_target_has_one_owner_and_prefers_central_chip() -> None:
    grid = np.full((8, 8), -1, dtype=np.int32)
    grid[3, 3] = 0
    grid[6, 6] = 1
    selected = np.asarray([True, True])
    chips = dense.select_chips(grid, selected, size=4, stride=2, minimum=1)
    owner = worker.assign_target_chips(grid, selected, chips, size=4)
    assert np.all(owner >= 0)
    for target, (row, column) in enumerate([(3, 3), (6, 6)]):
        chip_row, chip_column = chips[int(owner[target])]
        owned_distance = (row - chip_row - 1.5) ** 2 + (column - chip_column - 1.5) ** 2
        possible = [
            (row - rr - 1.5) ** 2 + (column - cc - 1.5) ** 2
            for rr, cc in chips
            if rr <= row < rr + 4 and cc <= column < cc + 4
        ]
        assert owned_distance == min(possible)


def test_strict_batch_supervises_each_owned_target_once() -> None:
    size = 4
    grid = np.full((6, 6), -1, dtype=np.int32)
    coordinates = [(1, 1), (2, 2), (4, 4)]
    for index, (row, column) in enumerate(coordinates):
        grid[row, column] = index
    selected = np.ones(3, dtype=bool)
    chips = dense.select_chips(grid, selected, size=size, stride=2, minimum=1)
    owner = worker.assign_target_chips(grid, selected, chips, size)
    arrays = {
        "embedding": np.zeros((6, 6, 128), dtype=np.int8),
        "scale": np.ones((6, 6), dtype=np.float32),
        "valid": np.ones((6, 6), dtype=bool),
        "index_grid": grid,
        "targets": np.asarray([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], dtype=np.float32),
    }
    stats = {
        "input_mean": np.zeros(128, dtype=np.float32),
        "input_sd": np.ones(128, dtype=np.float32),
        "target_mean": np.zeros(2, dtype=np.float32),
        "target_sd": np.ones(2, dtype=np.float32),
    }
    weights = np.asarray([2.0, 3.0, 5.0], dtype=np.float32)
    active = np.unique(owner)
    _, _, weight_map = worker.strict_batch(
        active, chips, owner, weights, arrays, stats, size
    )
    assert np.isclose(float(weight_map.sum()), float(weights.sum()))
    assert int((weight_map > 0).sum()) == len(weights)


def test_optimizer_denominator_preserves_global_weight_scale() -> None:
    assert worker.optimizer_denominator(12.0, 3) == 4.0
