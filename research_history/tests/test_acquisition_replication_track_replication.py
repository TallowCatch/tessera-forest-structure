from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, filename: str):
    specification = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    specification.loader.exec_module(module)
    return module


def test_prediction_freeze_has_no_target_outcome_input() -> None:
    text = (ROOT / "scripts/freeze_acquisition_replication_track_predictions.py").read_text(encoding="utf-8")
    assert "phase13_gedi_outcomes_sealed" not in text
    assert '"target_fhd_read": False' in text
    assert "TARGET_TESSERA_PATH" in text
    assert "TARGET_CONVENTIONAL_PATH" in text


def test_sealed_outcome_loader_ignores_empty_granules(tmp_path, monkeypatch) -> None:
    module = load_script("phase13_outcomes", "evaluate_acquisition_replication_track_replication.py")
    empty_path = tmp_path / "empty.parquet"
    populated_path = tmp_path / "populated.parquet"
    pd.DataFrame(columns=["site_id", "shot_number", "fhd_normal"]).to_parquet(empty_path)
    pd.DataFrame(
        {"site_id": ["TEST"], "shot_number": [1], "fhd_normal": [2.5]}
    ).to_parquet(populated_path)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    freeze = {
        "freeze_basis": {
            "sealed_outcome_files": [
                {"path": empty_path.name, "sha256": module.common.sha256(empty_path)},
                {"path": populated_path.name, "sha256": module.common.sha256(populated_path)},
            ]
        }
    }

    outcomes = module.load_sealed_outcomes(freeze)

    assert outcomes["fhd_normal"].dtype.kind == "f"
    assert outcomes.to_dict("records") == [
        {"site_id": "TEST", "shot_number": 1, "fhd_normal": 2.5}
    ]


def test_replication_aggregation_weights_replicates_then_folds_then_sites() -> None:
    module = load_script("phase13_evaluation", "evaluate_acquisition_replication_track_replication.py")
    records = []
    for site, site_value in [("A", 1.0), ("B", 3.0)]:
        for held_out in ["p1", "p2"]:
            for replicate in [0, 1]:
                for model in module.MODELS:
                    record = {
                        "target_year": 2024 if site == "A" else 2025,
                        "site_id": site,
                        "held_out_pass": held_out,
                        "budget": 50,
                        "replicate": replicate,
                        "model": model,
                        "test_rows": 30,
                    }
                    record.update({metric: site_value for metric in module.METRICS})
                    records.append(record)
    fold, site, macro = module.aggregate_metrics(pd.DataFrame(records))
    assert len(fold) == 2 * 2 * len(module.MODELS)
    assert len(site) == 2 * len(module.MODELS)
    assert set(macro["sites"]) == {2}
    assert set(macro["rmse"]) == {2.0}


def test_primary_gate_requires_both_baselines_majority_and_interval() -> None:
    module = load_script("phase13_evaluation_gate", "evaluate_acquisition_replication_track_replication.py")
    macro = pd.DataFrame(
        [
            {"budget": 50, "model": "source_only_tessera", "rmse": 0.5},
            {"budget": 50, "model": "sampled_target_local_mean", "rmse": 0.45},
            {"budget": 50, "model": "tessera_plus_sampled_local_offset", "rmse": 0.4},
        ]
    )
    paired = pd.DataFrame(
        {
            "comparison": ["tessera_offset_minus_source_tessera"] * 5,
            "site_id": list("ABCDE"),
            "rmse_difference": [-0.1, -0.1, -0.1, -0.1, 0.02],
        }
    )
    bootstrap = pd.DataFrame(
        {
            "comparison": ["tessera_offset_minus_source_tessera"],
            "site_bootstrap_95_high": [-0.01],
        }
    )
    gate = module.primary_gate(macro, paired, bootstrap, 50)
    assert gate["passed"]
    assert all(gate["conditions"].values())


def test_map_region_selection_uses_geometry_and_coverage_not_outcomes() -> None:
    module = load_script("phase13_map", "build_acquisition_replication_regional_map.py")
    predictions = pd.DataFrame(
        {
            "site_id": ["SMALL", "SMALL", "LARGE", "LARGE", "LARGE"],
            "pass_id": ["p1", "p1", "p2", "p2", "p2"],
            "x_epsg5070": [0.0, 100.0, 2000.0, 4500.0, 4510.0],
            "y_epsg5070": [0.0, 100.0, 3000.0, 6500.0, 6510.0],
        }
    )
    folds = pd.DataFrame(
        {
            "site_id": ["SMALL", "LARGE"],
            "held_out_pass": ["p1", "p2"],
            "test_rows": [2, 3],
            "eligible": [True, True],
        }
    )
    site, held_out, cell_x, cell_y = module.select_region(predictions, folds)
    assert (site, held_out) == ("LARGE", "p2")
    assert (cell_x, cell_y) == (4, 6)
