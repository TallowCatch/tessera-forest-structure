import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts/run_multisite_loso.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_multisite_loso", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_equal_site_weights_equalize_domains_and_preserve_total_weight() -> None:
    module = load_module()
    sites = np.array(["A"] * 100 + ["B"] * 10)

    weights = module.equal_site_weights(sites)

    assert np.isclose(weights.sum(), len(sites))
    assert np.isclose(weights[sites == "A"].sum(), weights[sites == "B"].sum())


def test_alpha_selection_prioritizes_worst_site_rmse() -> None:
    module = load_module()
    tuning = pd.DataFrame({
        "alpha": [1.0, 1.0, 10.0, 10.0, 100.0, 100.0],
        "rmse": [0.30, 0.70, 0.48, 0.49, 0.45, 0.55],
    })

    selected, ranking = module.select_alpha(tuning)

    assert selected == 10.0
    assert ranking.iloc[0]["worst_inner_validation_rmse"] == 0.49


def test_outer_fit_rejects_target_outcomes() -> None:
    module = load_module()
    target = pd.DataFrame({
        "site_id": ["C"],
        "shot_number": [1],
        "fhd_normal": [2.0],
    })

    try:
        module.fit_outer_fold(pd.DataFrame(), target, ["A", "B"], "C", [1.0])
    except RuntimeError as error:
        assert "Held-out outcomes" in str(error)
    else:
        raise AssertionError("Target outcomes were accepted by the target-free fit stage")


def test_feature_sets_have_expected_dimensions() -> None:
    module = load_module()

    assert len(module.FEATURE_SETS["terrain_ridge"]) == 4
    assert len(module.FEATURE_SETS["tessera_area_ridge"]) == 128
    assert len(module.FEATURE_SETS["tessera_area_topography_ridge"]) == 132
