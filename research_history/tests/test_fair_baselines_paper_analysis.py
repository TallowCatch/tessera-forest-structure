import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import r2_score


ROOT = Path(__file__).resolve().parents[1]


def test_phase8_protocol_prohibits_observation_counts_and_preserves_site_holdout():
    protocol = yaml.safe_load(
        (ROOT / "metadata/project_config_phase8_paper_analysis_protocol_freeze.yaml").read_text()
    )["phase8_paper_analysis"]
    assert protocol["outer_validation"] == "leave_one_complete_site_out"
    assert protocol["inner_tuning"] == "leave_one_training_site_out"
    assert protocol["equal_total_weight_per_training_site"] is True
    assert protocol["prohibited_model_feature_suffixes"] == ["observation_count"]
    assert len(protocol["sites"]) == 8


def test_fhd_height_proxy_is_rh100_and_strong_within_sites():
    original = pd.read_parquet(
        ROOT / "data/processed/gedi_v3_2024_primary_forest.parquet",
        columns=["site_id", "shot_number", "elev_highestreturn", "elev_lowestmode"],
    )
    expansion = pd.read_parquet(
        ROOT / "data/processed/phase7_gedi_v3_primary_forest.parquet",
        columns=["site_id", "shot_number", "elev_highestreturn", "elev_lowestmode"],
    )
    population = pd.concat([original, expansion], ignore_index=True)
    proxy = population["elev_highestreturn"] - population["elev_lowestmode"]
    assert len(population) == 10184
    assert np.isfinite(proxy).all()
    assert proxy.gt(0).all()

    associations = pd.read_csv(ROOT / "outputs/tables/phase8_fhd_height_associations.csv")
    sites = associations[~associations["scope"].str.startswith("pooled")]
    assert len(sites) == 8
    assert sites["spearman_r"].min() > 0.79
    centered = associations.set_index("scope").loc["pooled_within_site_centered"]
    assert centered["pearson_r"] > 0.84
    assert centered["spearman_r"] > 0.88


def test_fhd_height_loso_metrics_and_target_separation():
    freeze = json.loads(
        (ROOT / "metadata/phase8_fhd_height_dependence_freeze.json").read_text()
    )
    basis = freeze["freeze_basis"]
    assert basis["height_formula"] == "elev_highestreturn - elev_lowestmode"
    assert basis["height_role"] == "explanatory_GEDI_diagnostic_not_wall_to_wall_predictor"
    assert basis["target_site_fhd_used_for_scaling_tuning_or_training"] is False

    predictions = pd.read_parquet(
        ROOT / "data/processed/phase8_fhd_height_loso_predictions.parquet"
    )
    height = predictions[predictions["method"].eq("gedi_height_proxy_ridge")]
    assert len(height) == 10184
    for site, frame in height.groupby("outer_held_out_site"):
        assert set(frame["site_id"]) == {site}
        assert r2_score(frame["fhd_normal"], frame["prediction"]) > 0


def test_fair_baseline_excludes_observation_counts_when_available():
    freeze_path = ROOT / "metadata/phase8_fair_baseline_freeze.json"
    if not freeze_path.exists():
        return
    freeze = json.loads(freeze_path.read_text())
    basis = freeze["freeze_basis"]
    assert basis["observation_counts_used_as_model_features"] is False
    assert sorted(basis["excluded_observation_count_columns"]) == [
        "s1_valid_observation_count",
        "s2_valid_observation_count",
    ]
    assert basis["target_site_fhd_used_for_scaling_tuning_or_training"] is False
    assert basis["phase7_tessera_reproduction_max_metric_delta"] <= 1e-10


def test_all_site_conventional_predictors_are_target_free_and_radiometrically_plausible():
    freeze = json.loads(
        (ROOT / "metadata/phase8_conventional_predictor_all_sites_freeze.json").read_text()
    )
    basis = freeze["freeze_basis"]
    assert basis["target_columns_read"] == []
    assert basis["observation_counts_are_model_features"] is False
    assert (
        basis["sentinel_2_and_dem"]["expansion_radiometry"]
        == "earthsearch_boa_offset_already_applied_cog_dn_times_0.0001"
    )

    predictors = pd.read_parquet(
        ROOT / "data/processed/phase8_conventional_predictors_all_sites.parquet"
    )
    assert len(predictors) == 10184
    assert "fhd_normal" not in predictors.columns
    site_medians = predictors.groupby("site_id")[[
        "s2_blue_median", "s2_red_median", "s2_nir_median", "s2_ndvi_median"
    ]].median()
    assert site_medians[[
        "s2_blue_median", "s2_red_median", "s2_nir_median"
    ]].apply(lambda values: values.between(-0.05, 1.0)).all().all()
    assert site_medians["s2_ndvi_median"].between(-1.0, 1.0).all()
    unde = predictors[predictors["site_id"].eq("UNDE")]
    s1_features = [column for column in predictors if column.startswith("s1_")]
    assert unde[[column for column in s1_features if column != "s1_valid_observation_count"]].isna().all().all()
    assert unde["s1_valid_observation_count"].eq(0).all()


def test_fair_baseline_conclusion_is_frozen():
    metrics = pd.read_csv(ROOT / "outputs/tables/phase8_fair_baseline_metrics.csv")
    macro = metrics[metrics["scope"].eq("macro")].set_index("method")
    sentinel = macro.loc["sentinel2_topography_ridge"]
    tessera = macro.loc["tessera_area_ridge"]
    assert sentinel["rmse"] < tessera["rmse"]
    assert sentinel["r2"] > tessera["r2"]
    assert tessera["spearman_r"] > sentinel["spearman_r"]

    paired = pd.read_csv(
        ROOT / "outputs/tables/phase8_tessera_vs_conventional_paired_differences.csv"
    )
    contrast = paired[
        paired["contrast"].eq(
            "tessera_area_ridge_minus_sentinel2_topography_ridge"
        )
    ].set_index("metric")
    assert contrast.loc["rmse", "ci_low"] < 0 < contrast.loc["rmse", "ci_high"]
    assert contrast.loc["spearman_r", "ci_low"] > 0


def test_radar_sensitivity_does_not_claim_a_gain():
    paired = pd.read_csv(
        ROOT / "outputs/tables/phase8_radar_sensitivity_paired_differences.csv"
    )
    contrast = paired[
        paired["contrast"].eq(
            "sentinel1_sentinel2_topography_ridge_minus_sentinel2_topography_ridge"
        )
    ]
    assert len(contrast) == 3
    assert (contrast["ci_low"] < 0).all()
    assert (contrast["ci_high"] > 0).all()


def test_domain_shift_uses_sites_as_replicates_when_available():
    freeze_path = ROOT / "metadata/phase8_domain_shift_freeze.json"
    if not freeze_path.exists():
        return
    freeze = json.loads(freeze_path.read_text())
    basis = freeze["freeze_basis"]
    assert basis["exploratory"] is True
    assert basis["inference_unit"] == "site"
    assert basis["site_count"] == 8
    assert basis["footprints_treated_as_independent_domain_replicates"] is False

    associations = pd.read_csv(
        ROOT / "outputs/tables/phase8_domain_shift_associations.csv"
    )
    row = associations[
        associations["model"].eq("tessera_area_ridge")
        & associations["outcome"].eq("rmse")
        & associations["shift"].eq("fhd_mean_shift")
    ].iloc[0]
    assert row["spearman_r"] > 0.75
    assert row["exact_two_sided_permutation_p"] < 0.05
