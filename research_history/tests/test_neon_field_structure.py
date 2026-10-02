import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts/analyze_neon_field_structure.py"


def load_module():
    spec = importlib.util.spec_from_file_location("analyze_neon_field_structure", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_deduplicate_keeps_latest_and_excludes_ambiguous_latest() -> None:
    module = load_module()
    frame = pd.DataFrame({
        "plotID": ["P1", "P1", "P2", "P2"],
        "individualID": ["I1", "I1", "I2", "I2"],
        "tempStemID": [1, 1, 1, 1],
        "date": ["2024-01-01", "2024-02-01", "2024-03-01", "2024-03-01"],
        "height": [1.0, 2.0, 3.0, 4.0],
    })

    result, ambiguous = module.deduplicate_stem_measurements(frame)

    assert result["height"].tolist() == [2.0]
    assert ambiguous == 1


def test_primary_height_spread_is_quantile_difference() -> None:
    module = load_module()
    frame = pd.DataFrame({"height": np.arange(1.0, 11.0)})

    metrics = module.plot_height_metrics(frame)

    assert np.isclose(
        metrics["height_q90_minus_q10_m"], np.quantile(frame["height"], 0.9) - np.quantile(frame["height"], 0.1)
    )
