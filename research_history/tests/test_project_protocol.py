from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    return yaml.safe_load((ROOT / "configs/project.yaml").read_text(encoding="utf-8"))


def test_tessera_release_and_site_roles_are_fixed() -> None:
    config = load_config()
    assert config["predictors"]["dataset_version"] == "1.0"
    assert config["predictors"]["dataset_variant"] == "vultr"
    assert config["predictors"]["embedding_year"] == 2024
    assert config["predictors"]["never_mix_versions_or_variants"] is True
    assert config["sites"]["development"] == ["SOAP", "TEAK"]
    assert config["sites"]["locked_transfer_test"] == "BART"


def test_pre_bart_pipeline_selection_is_recorded() -> None:
    config = load_config()
    evaluation = config["phase4_locked_bart_evaluation"]
    assert evaluation["development_transfer_freeze_id"] == "phase4-transfer-841220e4a99a"
    assert evaluation["bart_predictor_freeze_id"] == "conventional-bart-b1e589a8601c"
    assert evaluation["selected_primary_pipeline"] == {
        "model": "tessera_area_topography_ridge",
        "alpha": 100.0,
        "features": "area_weighted_tessera_128_plus_terrain_4",
    }


def test_bart_cannot_be_used_for_tuning() -> None:
    evaluation = load_config()["phase4_locked_bart_evaluation"]
    assert evaluation["tuning_or_model_selection_on_bart"] == "prohibited"
    assert evaluation["predictor_extraction_must_not_read_target"] is True
    assert evaluation["report_all_predeclared_models_regardless_of_result"] is True


def test_corrected_sentinel_model_excludes_count_diagnostics() -> None:
    correction = load_config()["phase4_corrected_sentinel_sensitivity"]
    assert correction["expected_excluded_columns"] == [
        "s1_ascending_valid_observation_count",
        "s1_descending_valid_observation_count",
        "s2_valid_observation_count",
    ]
    assert correction["expected_model_feature_count"] == 43
    assert correction["tuning_or_model_selection_on_bart"] == "prohibited"
    assert correction["interpretation"] == "posthoc_sensitivity_not_replacement_confirmatory_result"


def test_multisite_outer_folds_are_complete_and_target_free() -> None:
    protocol = load_config()["phase6_multisite_leave_one_site_out"]
    assert protocol["sites"] == ["SOAP", "TEAK", "BART"]
    assert protocol["outer_split"] == "leave_one_complete_site_out"
    assert protocol["target_site_labels_used_for_scaling_tuning_or_training"] is False
    observed = {
        (tuple(fold["train_sites"]), fold["test_site"])
        for fold in protocol["outer_folds"]
    }
    assert observed == {
        (("TEAK", "BART"), "SOAP"),
        (("SOAP", "BART"), "TEAK"),
        (("SOAP", "TEAK"), "BART"),
    }
    for train_sites, test_site in observed:
        assert test_site not in train_sites
        assert set(train_sites) | {test_site} == set(protocol["sites"])


def test_multisite_protocol_uses_nested_site_tuning_and_equal_site_weights() -> None:
    protocol = load_config()["phase6_multisite_leave_one_site_out"]
    assert protocol["inner_tuning"]["split"] == "reciprocal_leave_one_training_site_out"
    assert protocol["inner_tuning"]["criterion"] == "lowest_worst_inner_validation_site_rmse"
    assert protocol["outer_training_weighting"] == "equal_total_weight_per_training_site"
    assert protocol["primary_model"] == "tessera_area_topography_ridge"
