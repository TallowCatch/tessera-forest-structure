from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "phase18_cairngorms",
    ROOT / "scripts/run_cairngorms_spatial_heads.py",
)
phase18 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(phase18)


def test_target_panel_contains_structure_controls_and_adjusted_target() -> None:
    config = phase18.load_config()
    names = phase18.target_names(config)
    assert names[:4] == [
        "mean_canopy_shannon_50m",
        "between_cell_height_sd_50m",
        "mean_within_cell_height_sd_50m",
        "mean_gap_fraction_50m",
    ]
    assert "mean_height_50m" in names
    assert names[-1].endswith("_height_adjusted")
    assert len(names) == 8


def test_patch_points_preserve_row_and_column_order() -> None:
    targets = pd.DataFrame(
        {
            "row_id": [0],
            "raster_patch_row": [2],
            "raster_patch_column": [3],
        }
    )
    transform = phase18.rasterio.Affine(10, 0, 0, 0, -10, 1000)
    points = phase18.patch_points(targets, transform, 5)
    assert len(points["patch_id"]) == 25
    assert points["local_row"].tolist() == np.repeat(np.arange(5), 5).tolist()
    assert points["local_column"].tolist() == np.tile(np.arange(5), 5).tolist()


def test_spatial_weights_give_blocks_equal_total_weight() -> None:
    blocks = pd.Series(["a", "a", "a", "b"])
    weights = phase18.spatial_weights(blocks)
    assert np.isclose(weights[:3].sum(), weights[3:].sum())


def test_metric_rows_keep_targets_separate() -> None:
    observed = np.column_stack([np.arange(4), np.arange(4) + 10])
    rows = phase18.metric_rows(
        observed,
        observed,
        ["first", "second"],
        "perfect",
        0,
        10,
    )
    assert [row["target"] for row in rows] == ["first", "second"]
    assert all(row["r2"] == 1.0 for row in rows)
