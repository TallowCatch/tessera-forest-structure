import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts/build_expanded_loso_gedi_expansion.py"


def load_module():
    spec = importlib.util.spec_from_file_location("build_expanded_loso_gedi_expansion", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_viability_gate_uses_only_rows_and_spatial_cells() -> None:
    module = load_module()
    frame = pd.DataFrame({
        "site_id": ["A"] * 100 + ["B"] * 99,
        "x_epsg5070": list(range(100)) + list(range(99)),
        "y_epsg5070": [0] * 199,
    })
    frame.loc[0:33, "x_epsg5070"] = 10
    frame.loc[34:66, "x_epsg5070"] = 1010
    frame.loc[67:99, "x_epsg5070"] = 2010

    result = module.viability_table(frame, ["A", "B"], 100, 3).set_index("site_id")

    assert bool(result.loc["A", "viable"])
    assert not bool(result.loc["B", "viable"])


def test_phase7_selected_sites_are_outcome_blind_and_new() -> None:
    import yaml

    config = yaml.safe_load((ROOT / "configs/project.yaml").read_text(encoding="utf-8"))
    protocol = config["phase7_site_expansion"]
    selected = set(protocol["selected_expansion_sites"])

    assert protocol["selection_used_fhd_values_or_distributions"] is False
    assert selected.isdisjoint(protocol["original_seen_sites"])
    assert len(selected) == 7
    assert protocol["prospective_external_evaluation"][
        "expansion_target_labels_used_before_prediction_freeze"
    ] is False
