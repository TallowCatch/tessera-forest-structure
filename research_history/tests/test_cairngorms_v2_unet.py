from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import acquire_cairngorms_v2_unet_cairngorms_v2_dense as acquisition  # noqa: E402
import cairngorms_v2_unet_worker as worker  # noqa: E402


def config() -> dict:
    return yaml.safe_load(
        (ROOT / "configs/cairngorms_v2_unet.yaml").read_text(encoding="utf-8")
    )["phase28_cairngorms_v2_unet"]


def test_final_design_is_frozen_to_v2_and_phase22_architecture() -> None:
    values = config()
    assert values["study"]["tessera_dataset_version"] == "experimental_v2"
    assert values["model"]["architecture"] == "frozen_phase22_two_level_compact_unet"
    assert values["model"]["input_channels"] == 129
    assert values["model"]["base_channels"] == 24
    assert values["model"]["chip_cells"] == 128
    assert values["model"]["chip_stride_cells"] == 96


def test_primary_gate_is_height_adjusted_corrected_shannon() -> None:
    values = config()
    assert values["targets"]["primary"] == "lidar_canopy_shannon_50m_height_adjusted"
    assert values["evaluation"]["success"]["minimum_primary_rmse_reduction_fraction"] == 0.05
    assert values["evaluation"]["success"]["require_bootstrap_ci_below_zero"] is True


def test_source_coordinates_use_pixel_centres() -> None:
    from affine import Affine

    transform = Affine(10, 0, 100, 0, -10, 200)
    x, y = acquisition.source_coordinates(
        transform, np.asarray([0, 2]), np.asarray([0, 3])
    )
    assert x.tolist() == [105.0, 135.0]
    assert y.tolist() == [195.0, 175.0]


def test_analysis_crs_is_explicit_when_source_metadata_is_absent() -> None:
    assert config()["study"]["analysis_crs"] == "EPSG:27700"


def test_compact_unet_retains_spatial_shape() -> None:
    model = worker.dense.CompactUNet(output_dimensions=8, base=24)
    output = model(worker.torch.zeros((1, 129, 32, 32)))
    assert tuple(output.shape) == (1, 8, 32, 32)


def test_membership_and_chip_selection_are_spatially_explicit() -> None:
    selected = worker.membership(5, np.asarray([1, 3]))
    assert selected.tolist() == [False, True, False, True, False]
    grid = np.full((8, 8), -1, dtype=np.int32)
    grid[2, 2] = 1
    grid[5, 5] = 3
    chips = worker.dense.select_chips(grid, selected, size=4, stride=4, minimum=1)
    assert chips == [(0, 0), (4, 4)]
