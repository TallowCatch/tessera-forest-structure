from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from evaluate_dutch_expanded import plot_codes
from prepare_dutch_expanded import fold_feasible, select_group, site_code


def test_site_code_is_stable() -> None:
    assert site_code("Kampina & Oisterwijkse Vennen") == "kampina_and_oisterwijkse_vennen"


def test_plot_codes_are_separate_by_group() -> None:
    sites = pd.DataFrame(
        {
            "site_code": ["c_a", "b_a", "c_b", "b_b"],
            "forest_group": ["conifer", "broadleaf", "conifer", "broadleaf"],
        }
    )
    assert plot_codes(sites) == {"c_a": "C1", "b_a": "B1", "c_b": "C2", "b_b": "B2"}


def test_fold_feasibility_accepts_separated_blocks() -> None:
    x = []
    y = []
    for block in range(5):
        for row in range(50):
            x.append(block * 2000.0 + (row % 5) * 10.0)
            y.append((row // 5) * 10.0)
    frame = pd.DataFrame({"rd_x": x, "rd_y": y})
    settings = {
        "fold_block_m": 500,
        "folds": 5,
        "fold_seed": 7,
        "exclusion_buffer_m": 500,
        "minimum_train_rows": 150,
        "minimum_test_rows": 30,
    }
    assert fold_feasible(frame, "example", settings, 0)


def test_selection_preserves_anchor_and_spacing() -> None:
    names = ["Anchor", "Near", "Far"]
    centres = {"Anchor": (0.0, 0.0), "Near": (1000.0, 0.0), "Far": (30000.0, 0.0)}
    prepared = {}
    boundaries = {}
    for index, name in enumerate(names):
        x, y = centres[name]
        prepared[name] = {
            "definition": {"name": name, "code": name.lower(), "group": "conifer"},
            "frame": pd.DataFrame({"rd_x": [x], "rd_y": [y]}),
            "record": {
                "mature_units_before_cap": 1000 - index,
                "centroid_x": x,
                "centroid_y": y,
            },
        }
        boundaries[name] = gpd.GeoDataFrame(
            {"name": [name]}, geometry=[box(x - 100, y - 100, x + 100, y + 100)], crs="EPSG:28992"
        )
    selected, reasons = select_group(
        "conifer", ["Anchor"], prepared, boundaries, 2, 12.0, []
    )
    assert selected == ["Anchor", "Far"]
    assert reasons["Anchor"] == "fixed_anchor"
    assert reasons["Far"] == "eligible_area_and_separation"
