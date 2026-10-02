from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import prepare_cairngorms_v2_surface as phase29


def test_phase29_config_freezes_expected_surface_targets() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/cairngorms_v2_surface.yaml").read_text()
    )["phase29_cairngorms_v2_surface"]
    assert config["representation"]["patch_cells"] == 5
    assert config["evaluation"]["folds"] == 5
    assert config["targets"]["raw"] == [
        "canopy_p95_height_m",
        "canopy_surface_sd_m",
        "canopy_surface_cv",
        "canopy_surface_rcv",
        "canopy_rumple",
        "canopy_open_fraction",
        "canopy_height_kurtosis",
    ]


def test_patch_summary_preserves_constant_channels() -> None:
    patch = np.zeros((2, 128, 5, 5), dtype=np.int8)
    patch[:, 0] = 4
    scales = np.ones((2, 5, 5), dtype=np.float32) * 0.5
    valid = np.ones((2, 5, 5), dtype=np.uint8)
    summary = phase29.summarize_patches(patch, scales, valid)
    assert summary.shape == (2, 640)
    np.testing.assert_allclose(summary[:, 0], 2.0)
    np.testing.assert_allclose(summary[:, 128], 2.0)
    np.testing.assert_allclose(summary[:, 256:], 0.0, atol=1e-6)

