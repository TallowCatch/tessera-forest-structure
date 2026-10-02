from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "phase16_scotland_evidence",
    ROOT / "scripts/run_scotland_evidence.py",
)
phase16 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(phase16)


def test_config_matches_protocol_freeze() -> None:
    assert phase16.CONFIG_PATH.read_bytes() == phase16.PROTOCOL_PATH.read_bytes()
    config = phase16.load_config()
    assert config["uncertainty"]["replicates"] == 2000
    assert config["full_map"]["map_type"] == "five_region_cross_fitted"


def test_block_bootstrap_preserves_model_order_for_every_replicate() -> None:
    frame = pd.DataFrame(
        {
            "spatial_block": ["a", "a", "b", "b"],
            "observed": [0.0, 1.0, 2.0, 3.0],
            "perfect": [0.0, 1.0, 2.0, 3.0],
            "poor": [1.0, 2.0, 3.0, 4.0],
        }
    )
    result = phase16.block_bootstrap_metrics(
        frame, ["perfect", "poor"], replicates=50, seed=7
    )
    perfect = result[result["model"].eq("perfect")].set_index("replicate")
    poor = result[result["model"].eq("poor")].set_index("replicate")
    assert (perfect["rmse"] == 0).all()
    assert (perfect["rmse"] < poor["rmse"]).all()


def test_reconstructed_region_rule_matches_frozen_sample() -> None:
    sample = pd.read_parquet(
        phase16.ROOT / "data/processed/phase15_arran_lidar_sample.parquet",
        columns=[
            "block_x",
            "block_y",
            "spatial_block",
            "spatial_region",
        ],
    )
    rule = phase16.derive_spatial_region_rule(
        sample, block_size=1000, region_count=5
    )
    blocks = sample[["block_x", "block_y", "spatial_region"]].drop_duplicates()
    observed = phase16.apply_spatial_region_rule(
        blocks, rule, block_size=1000
    )
    assert np.array_equal(observed, blocks["spatial_region"].to_numpy())


def test_phase15_result_inputs_still_match_their_freeze() -> None:
    freeze = phase16.validate_phase15_freeze(phase16.load_config())
    assert freeze["protocol"] == "phase15_arran_lidar"
