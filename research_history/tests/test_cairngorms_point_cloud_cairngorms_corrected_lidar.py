from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "phase25_prepare", ROOT / "scripts/prepare_cairngorms_point_cloud_cairngorms_corrected_lidar.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_aggregate_band_requires_declared_support() -> None:
    values = np.arange(49, dtype=np.float32).reshape(7, 7)
    values[1, 1] = np.nan
    windows = [(slice(1, 6), slice(1, 6))]
    means, counts = MODULE.aggregate_band(values, windows, minimum_valid=24)
    assert counts.tolist() == [24]
    assert np.isfinite(means[0])
    rejected, _ = MODULE.aggregate_band(values, windows, minimum_valid=25)
    assert np.isnan(rejected[0])


def test_remap_indices_preserves_selected_order() -> None:
    selected = np.array([1, 3, 4, 7], dtype=np.int64)
    source = np.array([0, 1, 2, 3, 7], dtype=np.int64)
    assert MODULE.remap_indices(source, selected).tolist() == [0, 1, 3]


def test_equal_block_weights_gives_equal_block_mass() -> None:
    blocks = np.array([1, 1, 2, 3, 3, 3], dtype=np.int64)
    weights = MODULE.equal_block_weights(blocks)
    totals = [weights[blocks == block].sum() for block in np.unique(blocks)]
    assert np.allclose(totals, totals[0])


def test_verify_replacement_identifies_only_changed_band(tmp_path: Path) -> None:
    profile = {
        "driver": "GTiff",
        "width": 4,
        "height": 4,
        "count": 2,
        "dtype": "float32",
        "transform": from_origin(0, 40, 10, 10),
    }
    previous = tmp_path / "previous.tif"
    corrected = tmp_path / "corrected.tif"
    for path, changed_value in [(previous, 1.0), (corrected, 2.0)]:
        with rasterio.open(path, "w", **profile) as dataset:
            dataset.set_band_description(1, "unchanged")
            dataset.set_band_description(2, "changed")
            dataset.write(np.ones((4, 4), dtype=np.float32), 1)
            dataset.write(np.full((4, 4), changed_value, dtype=np.float32), 2)
    result = MODULE.verify_replacement(previous, corrected, {"changed"})
    assert result == {"changed_bands": ["changed"], "unchanged_band_count": 1}
