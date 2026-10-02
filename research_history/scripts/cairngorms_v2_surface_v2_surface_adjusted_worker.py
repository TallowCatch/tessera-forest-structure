#!/usr/bin/env python3
"""Run one frozen height-adjusted canopy-surface model with TESSERA v2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

import cairngorms_components_components_worker as base


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/cairngorms_v2_surface.yaml"


def load_config() -> dict[str, Any]:
    source = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "phase29_cairngorms_v2_surface"
    ]
    model = source["model"]
    return {
        "frozen_inputs": {
            "tessera_features_path": f"{source['outputs']['adjusted_array_directory']}/tessera.npy",
            "conventional_features_path": f"{source['outputs']['adjusted_array_directory']}/conventional.npy",
        },
        "neural": {
            "hidden_dimensions": model["hidden_dimensions"],
            "dropout": model["dropout"],
            "batch_size": model["batch_size"],
            "maximum_epochs": model["maximum_epochs"],
            "minimum_epochs": model["minimum_epochs"],
            "early_stopping_patience": model["early_stopping_patience"],
            "learning_rate": model["learning_rate"],
            "weight_decay": model["weight_decay"],
        },
    }


source_config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
    "phase29_cairngorms_v2_surface"
]
base.CONFIG_PATH = CONFIG_PATH
base.ARRAY_DIR = ROOT / source_config["outputs"]["adjusted_array_directory"]
base.RESULT_DIR = ROOT / source_config["outputs"]["adjusted_result_directory"]
base.load_config = load_config


if __name__ == "__main__":
    args = base.parse_args()
    base.run(args.model, args.fold, args.seed)

