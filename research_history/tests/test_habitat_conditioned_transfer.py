import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROFILE_SCRIPT = ROOT / "scripts/build_habitat_transfer_habitat_profiles.py"
TRANSFER_SCRIPT = ROOT / "scripts/run_habitat_transfer_habitat_conditioned_transfer.py"
PROSPECTIVE_GEDI_SCRIPT = ROOT / "scripts/build_habitat_transfer_prospective_gedi.py"
PROSPECTIVE_TESSERA_SCRIPT = ROOT / "scripts/build_habitat_transfer_prospective_tessera_alignment.py"
PROSPECTIVE_PREDICTION_SCRIPT = ROOT / "scripts/freeze_habitat_transfer_prospective_predictions.py"
PROSPECTIVE_EVALUATION_SCRIPT = ROOT / "scripts/evaluate_habitat_transfer_prospective_transfer.py"
MATCHED_SIZE_CONTROL_SCRIPT = ROOT / "scripts/run_habitat_transfer_matched_size_source_control.py"
PROSPECTIVE_CONVENTIONAL_SCRIPT = (
    ROOT / "scripts/build_habitat_transfer_prospective_conventional_predictors.py"
)
PROSPECTIVE_CONVENTIONAL_COMPARISON_SCRIPT = (
    ROOT / "scripts/run_habitat_transfer_prospective_conventional_comparison.py"
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_habitat_distances_are_symmetric_and_ranked() -> None:
    module = load_module(PROFILE_SCRIPT, "build_habitat_transfer_habitat_profiles")
    profiles = pd.DataFrame({
        "site_id": ["A", "A", "B", "B", "C", "C"],
        "hierarchy": ["evt_phys"] * 6,
        "habitat_class": ["x", "y"] * 3,
        "pixel_count": [9, 1, 8, 2, 1, 9],
        "proportion": [0.9, 0.1, 0.8, 0.2, 0.1, 0.9],
    })

    distances = module.build_distances(profiles)
    lookup = distances.set_index(["target_site", "source_site"])["distance"]

    assert np.isclose(lookup.loc[("A", "B")], lookup.loc[("B", "A")])
    assert lookup.loc[("A", "B")] < lookup.loc[("A", "C")]
    ranked = distances[distances["target_site"].eq("A")].sort_values("source_rank")
    assert ranked["source_site"].tolist() == ["B", "C"]


def test_nearest_source_selection_is_target_outcome_free_and_deterministic() -> None:
    module = load_module(PROFILE_SCRIPT, "build_phase9_habitat_profiles_nearest")
    distances = pd.DataFrame({
        "hierarchy": ["evt_phys"] * 4,
        "target_site": ["T"] * 4,
        "source_site": ["D", "B", "A", "C"],
        "distance": [0.4, 0.2, 0.2, 0.3],
        "source_rank": [4, 2, 1, 3],
    })

    selected = module.nearest_sources(
        distances, "T", ["A", "B", "C", "D"], "evt_phys", 3
    )

    assert selected == ["A", "B", "C"]


def test_transfer_source_selection_never_includes_target() -> None:
    module = load_module(TRANSFER_SCRIPT, "run_habitat_transfer_habitat_conditioned_transfer")
    distances = pd.DataFrame({
        "hierarchy": ["evt_phys"] * 4,
        "target_site": ["T"] * 4,
        "source_site": ["A", "B", "C", "D"],
        "distance": [0.3, 0.1, 0.2, 0.4],
        "source_rank": [3, 1, 2, 4],
    })

    selected = module.source_sites_for_strategy(
        distances, "T", ["A", "B", "C", "D"], "nearest5_evt_phys", 3
    )

    assert selected == ["B", "C", "A"]
    assert "T" not in selected


def test_transfer_source_selection_rejects_target_in_source_pool() -> None:
    module = load_module(TRANSFER_SCRIPT, "run_phase9_habitat_conditioned_transfer_reject")

    try:
        module.source_sites_for_strategy(pd.DataFrame(), "T", ["A", "T"], "all_sources", 1)
    except ValueError as error:
        assert "Target site" in str(error)
    else:
        raise AssertionError("Target site was accepted in the source pool")


def test_prospective_gedi_target_lock_and_viability_are_outcome_blind() -> None:
    module = load_module(PROSPECTIVE_GEDI_SCRIPT, "build_habitat_transfer_prospective_gedi")
    import yaml

    protocol = yaml.safe_load(
        (ROOT / "metadata/project_config_phase9_prospective_protocol_freeze.yaml").read_text()
    )["phase9_prospective_habitat_transfer"]
    targets = module.load_locked_targets(protocol)
    frame = pd.DataFrame({
        "site_id": [targets[0]] * 100,
        "x_epsg5070": list(range(0, 100000, 1000)),
        "y_epsg5070": [0] * 100,
    })

    viability = module.viability_table(frame, targets[:1], 100, 3).iloc[0]

    assert targets == ["ABBY", "GRSM", "JERC", "MLBS", "SCBI", "SERC", "STEI"]
    assert bool(viability["viable"])


def test_prospective_tessera_filters_nonviable_sites() -> None:
    module = load_module(
        PROSPECTIVE_TESSERA_SCRIPT,
        "build_habitat_transfer_prospective_tessera_alignment",
    )
    frame = pd.DataFrame({
        "site_id": ["ABBY", "SERC", "STEI"],
        "fhd_normal": [1.0, 99.0, 2.0],
    })

    selected = module.viable_targets(frame, ["ABBY", "STEI"])

    assert selected["site_id"].tolist() == ["ABBY", "STEI"]
    assert selected["fhd_normal"].tolist() == [1.0, 2.0]


def test_prospective_prediction_target_columns_exclude_outcome() -> None:
    module = load_module(
        PROSPECTIVE_PREDICTION_SCRIPT,
        "freeze_habitat_transfer_prospective_predictions",
    )

    assert module.TARGET_READ_COLUMNS == [
        "site_id",
        "shot_number",
        *[f"tessera_area_{index:03d}" for index in range(128)],
    ]
    assert "fhd_normal" not in module.TARGET_READ_COLUMNS


def test_prospective_gate_requires_all_three_predeclared_checks() -> None:
    module = load_module(
        PROSPECTIVE_EVALUATION_SCRIPT,
        "evaluate_habitat_transfer_prospective_transfer",
    )
    paired = pd.DataFrame({
        "strategy": ["nearest5_evt_phys"],
        "site_count": [6],
        "sites_with_lower_rmse": [4],
        "mean_delta_rmse": [-0.02],
        "bootstrap_95_low": [-0.04],
        "bootstrap_95_high": [-0.001],
    })

    passed = module.gate_result(paired)
    assert passed["required_sites_for_majority"] == 4
    assert passed["passed"]

    paired.loc[0, "bootstrap_95_high"] = 0.001
    failed = module.gate_result(paired)
    assert not failed["passed"]
    assert not failed["checks"]["site_bootstrap_95_upper_below_zero"]


def test_matched_size_control_enumerates_all_five_of_eight_subsets() -> None:
    module = load_module(
        MATCHED_SIZE_CONTROL_SCRIPT,
        "run_habitat_transfer_matched_size_source_control",
    )
    source_sites = ["A", "B", "C", "D", "E", "F", "G", "H"]

    subsets = module.enumerate_subsets(source_sites, 5)

    assert len(subsets) == 56
    assert len(set(subsets)) == 56
    assert all(len(subset) == 5 for subset in subsets)


def test_prospective_conventional_acquisition_excludes_target_fhd() -> None:
    module = load_module(
        PROSPECTIVE_CONVENTIONAL_SCRIPT,
        "build_habitat_transfer_prospective_conventional_predictors",
    )

    assert module.TARGET_READ_COLUMNS == [
        "site_id", "shot_number", "longitude", "latitude", "x_epsg5070", "y_epsg5070"
    ]
    assert "fhd_normal" not in module.TARGET_READ_COLUMNS
    assert not any(column.endswith("observation_count") for column in module.MODEL_FEATURES)


def test_prospective_conventional_comparison_excludes_observation_counts() -> None:
    module = load_module(
        PROSPECTIVE_CONVENTIONAL_COMPARISON_SCRIPT,
        "run_habitat_transfer_prospective_conventional_comparison",
    )

    assert len(module.CONVENTIONAL_FEATURES) == 31
    assert module.CONVENTIONAL_FEATURES == (
        module.phase8.TERRAIN_FEATURES + module.phase8.S2_FEATURES
    )
    assert not any("count" in column for column in module.CONVENTIONAL_FEATURES)
