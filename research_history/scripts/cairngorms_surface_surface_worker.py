#!/usr/bin/env python3
"""Fit one frozen Phase 21 model, fold and seed."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

import cairngorms_components_components_worker as base


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/cairngorms_surface.yaml"


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "phase21_cairngorms_surface"
    ]


base.CONFIG_PATH = CONFIG_PATH
base.ARRAY_DIR = ROOT / "data/interim/phase21_cairngorms_surface"
base.RESULT_DIR = ROOT / "data/interim/phase21_cairngorms_surface_results"
base.load_config = load_config


if __name__ == "__main__":
    arguments = base.parse_args()
    base.run(arguments.model, arguments.fold, arguments.seed)
