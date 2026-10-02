#!/usr/bin/env python3
"""Fit one frozen Cairngorms model using experimental TESSERA v2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

import cairngorms_components_components_worker as base


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/tessera_v2.yaml"


def load_config() -> dict[str, Any]:
    complete = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "phase27_tessera_v2"
    ]
    site = complete["cairngorms"]
    evaluation = complete["cairngorms_evaluation"]
    return {
        "frozen_inputs": {
            "tessera_features_path": site["v2_features"],
            "conventional_features_path": site["conventional"],
        },
        "neural": {
            key: evaluation[key]
            for key in [
                "hidden_dimensions", "dropout", "batch_size", "maximum_epochs",
                "minimum_epochs", "early_stopping_patience", "learning_rate",
                "weight_decay",
            ]
        },
    }


base.CONFIG_PATH = CONFIG_PATH
base.ARRAY_DIR = ROOT / "data/interim/phase25_cairngorms_corrected_lidar"
base.RESULT_DIR = ROOT / "data/interim/phase27_cairngorms_v2_results"
base.load_config = load_config


if __name__ == "__main__":
    arguments = base.parse_args()
    base.run(arguments.model, arguments.fold, arguments.seed)
