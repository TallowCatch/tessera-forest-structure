#!/usr/bin/env python3
"""Run one frozen raw canopy-surface model with TESSERA v2."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

import cairngorms_spatial_unet_spatial_unet_worker as base


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/cairngorms_v2_surface.yaml"


def load_config() -> dict[str, Any]:
    source = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "phase29_cairngorms_v2_surface"
    ]
    model = source["model"]
    return {
        "outputs": {
            "array_directory": source["outputs"]["raw_array_directory"],
            "result_directory": source["outputs"]["raw_result_directory"],
        },
        "models": {
            "hidden_dimensions": model["hidden_dimensions"],
            "dropout": model["dropout"],
            "batch_size": {
                "tessera_mlp_5x5": model["batch_size"],
                "fused_mlp_5x5": model["batch_size"],
            },
            "maximum_epochs": {
                "tessera_mlp_5x5": model["maximum_epochs"],
                "fused_mlp_5x5": model["maximum_epochs"],
            },
            "minimum_epochs": model["minimum_epochs"],
            "early_stopping_patience": model["early_stopping_patience"],
            "learning_rate": model["learning_rate"],
            "weight_decay": model["weight_decay"],
        },
    }


base.CONFIG_PATH = CONFIG_PATH
base.load_config = load_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["tessera_mlp_5x5", "fused_mlp_5x5"], required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    base.run(args.model, "balanced", args.fold, args.seed)

