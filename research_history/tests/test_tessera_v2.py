from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pyproj import Transformer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import acquire_tessera_v2 as acquisition  # noqa: E402
import tessera_v2_common as common  # noqa: E402


def test_supplied_loader_hash_is_frozen() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/tessera_v2.yaml").read_text(encoding="utf-8")
    )["phase27_tessera_v2"]
    path = ROOT / config["representation"]["supplied_loader"]
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    assert observed == config["representation"]["supplied_loader_sha256"]


def test_tile_center_uses_point_one_degree_grid() -> None:
    assert common.tile_center(5.799, 50.801) == (5.75, 50.85)
    assert common.tile_center(-3.601, 57.299) == (-3.65, 57.25)


def test_batch_sampler_recovers_known_source_pixel() -> None:
    tile = (5.75, 50.85)
    quantized = np.zeros((10, 10, 128), dtype=np.int8)
    quantized[4, 6] = 17
    scales = np.ones((10, 10), dtype=np.float32) * 0.5
    transform, epsg = common.tile_transform_and_epsg(*tile, 10, 10)
    east, north = transform * (6.5, 4.5)
    to_world = Transformer.from_crs(
        f"EPSG:{epsg}", "EPSG:4326", always_xy=True
    )
    lon, lat = to_world.transform(east, north)
    selected, values, local_scales = common.sample_quantized_tile(
        quantized,
        scales,
        tile,
        "EPSG:4326",
        np.asarray([lon]),
        np.asarray([lat]),
    )
    assert selected.tolist() == [0]
    assert np.all(values == 17)
    assert np.allclose(local_scales, 0.5)


def test_ordered_summary_has_expected_constant_patch_statistics() -> None:
    quantized = np.full((2, 128, 5, 5), 2, dtype=np.int8)
    scales = np.full((2, 5, 5), 0.5, dtype=np.float16)
    valid = np.ones((2, 5, 5), dtype=np.uint8)
    summary = common.summarize_ordered_patch(quantized, scales, valid)
    assert summary.shape == (2, 640)
    assert np.allclose(summary[:, :256], 1.0)
    assert np.allclose(summary[:, 256:], 0.0)


def test_savelsbos_points_respect_frozen_broadleaf_mask() -> None:
    cohort = pd.DataFrame(
        {
            "row_id": [0],
            "rd_x": [1000.0],
            "rd_y": [2000.0],
            "broadleaf_pixel_mask": [(1 << 0) | (1 << 12) | (1 << 24)],
        }
    )
    settings = {
        "x_column": "rd_x",
        "y_column": "rd_y",
        "pixel_mask_column": "broadleaf_pixel_mask",
    }
    points = acquisition.build_points(cohort, settings, masked=True)
    assert points["point_index"].tolist() == [0, 12, 24]
    assert points["x"].tolist() == [980.0, 1000.0, 1020.0]
    assert points["y"].tolist() == [2020.0, 2000.0, 1980.0]
