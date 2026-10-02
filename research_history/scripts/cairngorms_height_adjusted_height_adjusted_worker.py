#!/usr/bin/env python3
"""Fit one Phase 23 height-adjusted model, fold, and seed."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

import cairngorms_components_components_worker as base


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/cairngorms_height_adjusted.yaml"


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "phase23_cairngorms_height_adjusted"
    ]


base.CONFIG_PATH = CONFIG_PATH
base.ARRAY_DIR = ROOT / "data/interim/phase23_cairngorms_height_adjusted"
base.RESULT_DIR = ROOT / "data/interim/phase23_cairngorms_height_adjusted_results"
base.load_config = load_config


if __name__ == "__main__":
    args = base.parse_args()
    base.run(args.model, args.fold, args.seed)
