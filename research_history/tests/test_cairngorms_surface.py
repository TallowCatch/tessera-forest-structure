from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import aggregate_cairngorms_surface as aggregate
import prepare_cairngorms_surface as prepare


def test_frozen_target_order_is_explicit() -> None:
    config = prepare.load_config()
    assert config["study"]["tessera_dataset_version"] == "1.0"
    assert config["study"]["tessera_dataset_variant"] == "vultr"
    assert prepare.all_target_names(config) == [
        "canopy_top_height_m",
        "canopy_surface_sd_m",
        "canopy_surface_cv",
        "canopy_surface_rcv",
        "canopy_rumple",
        "canopy_open_fraction",
        "canopy_surface_sd_m_height_adjusted",
        "canopy_surface_cv_height_adjusted",
        "canopy_surface_rcv_height_adjusted",
        "canopy_rumple_height_adjusted",
        "canopy_open_fraction_height_adjusted",
    ]


def test_intersect_indices_preserves_source_order() -> None:
    valid = np.asarray([True, False, True, False, True])
    observed = prepare.intersect_indices(np.asarray([4, 1, 2, 0]), valid)
    np.testing.assert_array_equal(observed, np.asarray([4, 2, 0]))


def test_target_inventory_has_units_and_provenance() -> None:
    inventory = aggregate.target_inventory()
    assert set(inventory["target"]) == set(
        prepare.load_config()["targets"]["primary"]
    )
    assert inventory["unit"].notna().all()
    assert inventory["source"].str.len().min() > 0


def test_surface_gate_excludes_height_and_openings() -> None:
    surface_targets = prepare.load_config()["interpretation_gates"][
        "surface_targets"
    ]
    assert "canopy_top_height_m" not in surface_targets
    assert "canopy_open_fraction" not in surface_targets
    assert len(surface_targets) == 4
