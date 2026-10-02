from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "phase26_selection", ROOT / "scripts/freeze_ahn4_replication_ahn4_site.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_dominant_habitat_uses_largest_percentage_then_code() -> None:
    table = pd.DataFrame(
        {
            "SITECODE": ["A", "A", "B", "B"],
            "HABITATCODE": ["N17", "N16", "N16", "N08"],
            "DESCRIPTION": ["conifer", "broadleaf", "broadleaf", "heath"],
            "PERCENTAGECOVER": [30, 70, 50, 50],
        }
    )
    result = MODULE.dominant_habitat(table).set_index("SITECODE")
    assert result.loc["A", "HABITATCODE"] == "N16"
    assert result.loc["B", "HABITATCODE"] == "N08"
