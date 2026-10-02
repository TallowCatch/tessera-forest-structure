from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import cairngorms_dense_10m_worker as dense
import cairngorms_v2_height_worker as worker


def config() -> dict:
    return yaml.safe_load(
        (ROOT / "configs/cairngorms_v2_height_unet.yaml").read_text()
    )["phase30_cairngorms_v2_height_unet"]


def test_frozen_height_targets_and_primary_outcome() -> None:
    values = config()
    assert values["targets"]["names"] == [
        "canopy_mean_height_m",
        "canopy_p95_height_m",
    ]
    assert values["targets"]["primary"] == "canopy_p95_height_m"
    assert values["model"]["training_chip_minimum_targets"] == 1


def test_selected_epoch_cannot_precede_minimum() -> None:
    history = [
        {"epoch": epoch, "validation_loss": value}
        for epoch, value in enumerate(
            [0.30, 0.20, 0.10, 0.15, 0.17, 0.18, 0.19, 0.20, 0.25, 0.24, 0.23, 0.22],
            start=1,
        )
    ]
    assert worker.selected_epoch(history, minimum=10) == 12


def test_chip_selection_covers_all_selected_rows() -> None:
    grid = np.full((8, 8), -1, dtype=np.int32)
    coordinates = [(0, 0), (3, 3), (7, 7), (0, 7), (7, 0)]
    for index, (row, column) in enumerate(coordinates):
        grid[row, column] = index
    selected = np.ones(len(coordinates), dtype=bool)
    chips = dense.select_chips(grid, selected, size=4, stride=3, minimum=1)
    coverage = worker.coverage_counts(grid, selected, chips, size=4)
    assert len(chips) > 0
    assert np.all(coverage > 0)


def test_overlap_adjustment_equalizes_total_target_weight() -> None:
    base = np.asarray([2.0, 3.0, 5.0, 0.0], dtype=np.float32)
    coverage = np.asarray([1, 2, 4, 0], dtype=np.int16)
    indices = np.asarray([0, 1, 2], dtype=np.int64)
    adjusted = worker.coverage_adjusted_weights(base, coverage, indices)
    assert np.allclose(adjusted[indices] * coverage[indices], base[indices])
    assert adjusted[3] == 0


def test_unet_output_shape_matches_two_height_targets() -> None:
    model = dense.CompactUNet(output_dimensions=2, base=8)
    output = model(torch.zeros((1, 129, 32, 32), dtype=torch.float32))
    assert output.shape == (1, 2, 32, 32)
