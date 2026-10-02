#!/usr/bin/env python3
"""Validate the Phase 9 habitat profiles and exploratory transfer freezes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
HABITAT_FREEZE_PATH = ROOT / "metadata/phase9_landfire_habitat_freeze.json"
TRANSFER_FREEZE_PATH = ROOT / "metadata/phase9_habitat_conditioned_transfer_freeze.json"
PROSPECTIVE_GEDI_FREEZE_PATH = ROOT / "metadata/phase9_prospective_gedi_freeze.json"
PROSPECTIVE_TESSERA_FREEZE_PATH = ROOT / "metadata/phase9_prospective_tessera_alignment_freeze.json"
PROSPECTIVE_PREDICTION_FREEZE_PATH = ROOT / "metadata/phase9_prospective_prediction_freeze.json"
PROSPECTIVE_EVALUATION_FREEZE_PATH = ROOT / "metadata/phase9_prospective_evaluation_freeze.json"
PROSPECTIVE_EVALUATION_LOCK_PATH = ROOT / "metadata/phase9_prospective_evaluation_execution_lock.json"
MATCHED_SIZE_FREEZE_PATH = ROOT / "metadata/phase9_matched_size_source_control_freeze.json"
PROSPECTIVE_CONVENTIONAL_FREEZE_PATH = (
    ROOT / "metadata/phase9_prospective_conventional_freeze.json"
)
PROSPECTIVE_CONVENTIONAL_COMPARISON_FREEZE_PATH = (
    ROOT / "metadata/phase9_prospective_conventional_comparison_freeze.json"
)
EXPECTED_CURRENT = {"BART", "HARV", "ORNL", "SOAP", "TALL", "TEAK", "UNDE", "WREF"}
EXPECTED_PROSPECTIVE = {"ABBY", "GRSM", "JERC", "MLBS", "SCBI", "SERC", "STEI"}
EXPECTED_VIABLE = {"ABBY", "GRSM", "JERC", "MLBS", "SCBI", "STEI"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_outputs(freeze: dict) -> None:
    for output in freeze["outputs"].values():
        path = ROOT / output["path"]
        if not path.exists() or sha256(path) != output["sha256"]:
            raise RuntimeError(f"Output hash mismatch: {path}")


def main() -> int:
    habitat = json.loads(HABITAT_FREEZE_PATH.read_text(encoding="utf-8"))
    transfer = json.loads(TRANSFER_FREEZE_PATH.read_text(encoding="utf-8"))
    prospective_gedi = json.loads(PROSPECTIVE_GEDI_FREEZE_PATH.read_text(encoding="utf-8"))
    prospective_tessera = json.loads(
        PROSPECTIVE_TESSERA_FREEZE_PATH.read_text(encoding="utf-8")
    )
    prospective_prediction = json.loads(
        PROSPECTIVE_PREDICTION_FREEZE_PATH.read_text(encoding="utf-8")
    )
    prospective_evaluation = json.loads(
        PROSPECTIVE_EVALUATION_FREEZE_PATH.read_text(encoding="utf-8")
    )
    matched_size = json.loads(MATCHED_SIZE_FREEZE_PATH.read_text(encoding="utf-8"))
    prospective_conventional = json.loads(
        PROSPECTIVE_CONVENTIONAL_FREEZE_PATH.read_text(encoding="utf-8")
    )
    prospective_conventional_comparison = json.loads(
        PROSPECTIVE_CONVENTIONAL_COMPARISON_FREEZE_PATH.read_text(encoding="utf-8")
    )
    validate_outputs(habitat)
    validate_outputs(transfer)
    validate_outputs(prospective_gedi)
    validate_outputs(prospective_tessera)
    validate_outputs(prospective_prediction)
    validate_outputs(prospective_evaluation)
    validate_outputs(matched_size)
    validate_outputs(prospective_conventional)
    validate_outputs(prospective_conventional_comparison)
    if habitat["freeze_basis"]["fhd_columns_read"]:
        raise RuntimeError("FHD was read during habitat profiling")
    if habitat["freeze_basis"]["target_outcomes_used_for_habitat_profiles_or_similarity"]:
        raise RuntimeError("Habitat similarity used target outcomes")

    profiles = pd.read_csv(ROOT / habitat["outputs"]["profiles"]["path"])
    sums = profiles.groupby(["hierarchy", "site_id"])["proportion"].sum()
    if not np.allclose(sums.to_numpy(), 1.0, atol=1e-12):
        raise RuntimeError("One or more habitat profiles do not sum to one")
    if set(profiles["hierarchy"]) != {"evt_phys", "evt_group", "evt_system"}:
        raise RuntimeError("The frozen habitat hierarchy is incomplete")

    distances = pd.read_csv(ROOT / habitat["outputs"]["distances"]["path"])
    reverse = distances.merge(
        distances,
        left_on=["hierarchy", "target_site", "source_site"],
        right_on=["hierarchy", "source_site", "target_site"],
        suffixes=("_forward", "_reverse"),
        validate="one_to_one",
    )
    if len(distances) != 19 * 18 * 3 or not np.allclose(
        reverse["distance_forward"], reverse["distance_reverse"], atol=1e-14
    ):
        raise RuntimeError("Habitat distances are incomplete or asymmetric")
    if distances["distance"].lt(0).any() or distances["distance"].gt(1).any():
        raise RuntimeError("Jensen-Shannon distances fall outside [0, 1]")

    points = pd.read_parquet(ROOT / habitat["outputs"]["footprint_habitats"]["path"])
    if "fhd_normal" in points.columns or len(points) != 10184 or points.duplicated(
        ["site_id", "shot_number"]
    ).any():
        raise RuntimeError("Unexpected footprint-habitat table")
    if set(points["site_id"]) != EXPECTED_CURRENT:
        raise RuntimeError("The footprint-habitat table contains the wrong sites")

    prospective = pd.read_csv(ROOT / habitat["outputs"]["prospective_design"]["path"])
    if set(prospective["target_site"]) != EXPECTED_PROSPECTIVE:
        raise RuntimeError("Unexpected prospective habitat target cohort")
    if prospective["target_outcome_opened"].astype(bool).any():
        raise RuntimeError("A prospective target was marked as opened")

    predictions = pd.read_parquet(ROOT / transfer["outputs"]["predictions"]["path"])
    for row in predictions[["outer_held_out_site", "training_sites"]].drop_duplicates().itertuples():
        if row.outer_held_out_site in row.training_sites.split("+"):
            raise RuntimeError("A held-out target entered a training set")
    if not np.isfinite(predictions[["fhd_normal", "prediction"]].to_numpy()).all():
        raise RuntimeError("Transfer predictions contain non-finite values")

    selection = pd.read_csv(ROOT / transfer["outputs"]["source_selection"]["path"])
    totals = selection.groupby(
        ["analysis", "outer_held_out_site", "strategy"]
    )["source_total_fit_weight"].agg(lambda values: float(np.ptp(values)))
    if totals.max() > 1e-8:
        raise RuntimeError("Training sites did not receive equal total fit weight")
    if transfer["freeze_basis"]["target_site_fhd_used_for_habitat_matching_scaling_tuning_or_training"]:
        raise RuntimeError("Target FHD entered source selection or fitting")
    if transfer["freeze_basis"]["all_source_phase7_prediction_max_absolute_delta"] > 1e-10:
        raise RuntimeError("The all-source Phase 7 comparator was not reproduced")

    metrics = pd.read_csv(ROOT / transfer["outputs"]["metrics"]["path"])
    primary = metrics[
        metrics["analysis"].eq("all_quality_filtered")
        & metrics["scope"].eq("site")
        & metrics["method"].eq("tessera_area_ridge")
    ]
    if set(primary["strategy"]) != {
        "all_sources", "nearest5_evt_phys", "nearest5_evt_group", "nearest5_evt_system"
    }:
        raise RuntimeError("The primary strategy set is incomplete")
    if primary.groupby("strategy")["held_out_site"].nunique().ne(8).any():
        raise RuntimeError("One or more strategies lack complete-site results")

    paired = pd.read_csv(ROOT / transfer["outputs"]["paired_summary"]["path"])
    predeclared = paired[
        paired["analysis"].eq("all_quality_filtered")
        & paired["strategy"].eq("nearest5_evt_phys")
    ].iloc[0]
    gate = transfer["freeze_basis"]["gate"]
    if not np.isclose(predeclared["mean_delta_rmse"], gate["mean_delta_rmse"], atol=1e-12):
        raise RuntimeError("Primary paired result differs from the freeze")
    if gate["confirmatory_claim_allowed"] or not gate["exploratory_only"]:
        raise RuntimeError("The exploratory timing limitation was lost")

    gedi_basis = prospective_gedi["freeze_basis"]
    if set(gedi_basis["selected_target_sites"]) != EXPECTED_PROSPECTIVE:
        raise RuntimeError("Prospective GEDI acquisition used the wrong target cohort")
    if set(gedi_basis["viable_sites"]) != EXPECTED_VIABLE:
        raise RuntimeError("Unexpected prospective GEDI viability result")
    if gedi_basis["full_hdf_files_retained"] or gedi_basis["per_granule_interim_files_retained"]:
        raise RuntimeError("Prospective GEDI source files were retained")
    if not prospective_gedi["gate"]["passed"]:
        raise RuntimeError("Prospective GEDI site-count gate did not pass")

    aligned = pd.read_parquet(
        ROOT / prospective_tessera["outputs"]["aligned_prospective"]["path"]
    )
    if len(aligned) != 5142 or set(aligned["site_id"]) != EXPECTED_VIABLE:
        raise RuntimeError("Unexpected prospective TESSERA population")
    if aligned.duplicated(["site_id", "shot_number"]).any():
        raise RuntimeError("Prospective TESSERA keys are not unique")
    if not aligned["passes_tessera_alignment"].all():
        raise RuntimeError("One or more prospective TESSERA alignments failed")
    tessera_basis = prospective_tessera["freeze_basis"]
    if tessera_basis["prospective_tiles_retained"] or tessera_basis["target_outcomes_used_for_alignment"]:
        raise RuntimeError("Prospective TESSERA alignment violated the frozen protocol")

    blind = pd.read_parquet(
        ROOT / prospective_prediction["outputs"]["predictions"]["path"]
    )
    if "fhd_normal" in blind.columns or len(blind) != 5142 * 3:
        raise RuntimeError("Prospective blind prediction table contains target outcomes or wrong rows")
    if set(blind["site_id"]) != EXPECTED_VIABLE or set(blind["strategy"]) != {
        "all_sources", "nearest5_evt_phys", "nearest5_evt_group"
    }:
        raise RuntimeError("Prospective blind strategy population is incomplete")
    if not np.isfinite(blind["prediction"].to_numpy()).all():
        raise RuntimeError("Prospective blind predictions contain non-finite values")
    prediction_basis = prospective_prediction["freeze_basis"]
    if prediction_basis["target_outcomes_opened"] or prediction_basis["target_outcome_columns_read"]:
        raise RuntimeError("Prospective target outcomes were opened before prediction freeze")
    if "fhd_normal" in prediction_basis["target_columns_read"]:
        raise RuntimeError("Prospective FHD was listed among target predictor columns")
    blind_selection = pd.read_csv(
        ROOT / prospective_prediction["outputs"]["source_selection"]["path"]
    )
    weight_spread = blind_selection.groupby(
        ["target_site", "strategy"]
    )["source_total_fit_weight"].agg(lambda values: float(np.ptp(values)))
    if weight_spread.max() > 1e-8:
        raise RuntimeError("Prospective source sites did not receive equal total fit weight")
    if list((ROOT / "data/raw/phase9_prospective_gedi_v3").rglob("*.h5*")):
        raise RuntimeError("A prospective GEDI HDF5 source remains on disk")
    if (ROOT / "data/interim/phase9_prospective_tessera_stream").exists():
        raise RuntimeError("A prospective TESSERA source batch remains on disk")

    evaluation_basis = prospective_evaluation["freeze_basis"]
    if not evaluation_basis["target_outcomes_opened"]:
        raise RuntimeError("Prospective evaluation does not record target outcome opening")
    if evaluation_basis["prediction_id"] != prospective_prediction["prediction_id"]:
        raise RuntimeError("Prospective evaluation used the wrong prediction freeze")
    if evaluation_basis["execution_lock_sha256"] != sha256(PROSPECTIVE_EVALUATION_LOCK_PATH):
        raise RuntimeError("Prospective evaluation execution lock hash changed")
    evaluated = pd.read_parquet(
        ROOT / prospective_evaluation["outputs"]["evaluated_predictions"]["path"]
    )
    if len(evaluated) != 5142 * 3 or evaluated["fhd_normal"].isna().any():
        raise RuntimeError("Prospective evaluated prediction population is incomplete")
    evaluation_metrics = pd.read_csv(
        ROOT / prospective_evaluation["outputs"]["metrics"]["path"]
    )
    site_evaluation = evaluation_metrics[evaluation_metrics["scope"].eq("site")]
    if site_evaluation.groupby("strategy")["site_id"].nunique().ne(6).any():
        raise RuntimeError("Prospective metrics lack complete site results")
    paired_evaluation = pd.read_csv(
        ROOT / prospective_evaluation["outputs"]["paired_summary"]["path"]
    )
    primary_evaluation = paired_evaluation[
        paired_evaluation["strategy"].eq("nearest5_evt_phys")
    ].iloc[0]
    evaluation_gate = evaluation_basis["primary_gate"]
    if not np.isclose(
        primary_evaluation["mean_delta_rmse"], evaluation_gate["mean_delta_rmse"], atol=1e-12
    ):
        raise RuntimeError("Prospective primary effect differs from its freeze")
    if not evaluation_gate["passed"] or not all(evaluation_gate["checks"].values()):
        raise RuntimeError("Prospective primary gate did not pass all frozen checks")
    bootstrap = pd.read_csv(
        ROOT / prospective_evaluation["outputs"]["site_bootstrap"]["path"]
    )
    primary_bootstrap = bootstrap[bootstrap["strategy"].eq("nearest5_evt_phys")]
    if len(primary_bootstrap) != 10000:
        raise RuntimeError("Prospective primary bootstrap has the wrong replicate count")
    bootstrap_interval = np.quantile(primary_bootstrap["mean_delta_rmse"], [0.025, 0.975])
    if not np.allclose(
        bootstrap_interval,
        [evaluation_gate["bootstrap_95_low"], evaluation_gate["bootstrap_95_high"]],
        atol=1e-12,
    ):
        raise RuntimeError("Prospective bootstrap interval differs from its freeze")

    matched_basis = matched_size["freeze_basis"]
    if matched_basis["confirmatory"] or matched_basis["source_subset_count"] != 56:
        raise RuntimeError("Matched-size control timing or subset count changed")
    if matched_basis["frozen_prediction_max_absolute_delta"] > 1e-10:
        raise RuntimeError("Matched-size control did not reproduce frozen predictions")
    matched_metrics = pd.read_csv(ROOT / matched_size["outputs"]["metrics"]["path"])
    if len(matched_metrics) != 6 * 56 or matched_metrics.groupby(
        "site_id"
    )["source_subset"].nunique().ne(56).any():
        raise RuntimeError("Matched-size control does not contain every five-source subset")
    matched_summary = pd.read_csv(ROOT / matched_size["outputs"]["summary"]["path"])
    matched_sites = matched_summary[matched_summary["scope"].eq("site")]
    if len(matched_sites) != 6 or not matched_sites[
        "frozen_minus_median_five_rmse"
    ].lt(0).all():
        raise RuntimeError("Matched-size site summaries differ from the frozen result")
    matched_null = pd.read_csv(
        ROOT / matched_size["outputs"]["random_subset_null"]["path"]
    )
    null_basis = matched_basis["random_subset_null"]
    if len(matched_null) != null_basis["replicates"]:
        raise RuntimeError("Matched-size random-subset null has the wrong replicate count")
    matched_p = (
        np.sum(
            matched_null["random_five_source_macro_rmse"].to_numpy()
            <= null_basis["frozen_phys_macro_rmse"]
        )
        + 1
    ) / (len(matched_null) + 1)
    if not np.isclose(matched_p, null_basis["one_sided_monte_carlo_p"], atol=1e-15):
        raise RuntimeError("Matched-size Monte Carlo result differs from its freeze")

    conventional_basis = prospective_conventional["freeze_basis"]
    if conventional_basis["observation_counts_are_model_features"]:
        raise RuntimeError("Observation counts entered the prospective conventional model")
    if any("count" in name for name in conventional_basis["model_features"]):
        raise RuntimeError("A count field is listed among conventional model features")
    conventional = pd.read_parquet(
        ROOT / prospective_conventional["outputs"]["predictors"]["path"]
    )
    if len(conventional) != 5142 or set(conventional["site_id"]) != EXPECTED_VIABLE:
        raise RuntimeError("Unexpected prospective conventional predictor population")
    if conventional.duplicated(["site_id", "shot_number"]).any():
        raise RuntimeError("Prospective conventional predictor keys are not unique")
    if conventional["s2_valid_observation_count"].min() < 12:
        raise RuntimeError("Prospective conventional clear-observation gate changed")
    if not np.isfinite(
        conventional[conventional_basis["model_features"]].to_numpy()
    ).all():
        raise RuntimeError("Prospective conventional predictors contain non-finite values")

    comparison_basis = prospective_conventional_comparison["freeze_basis"]
    if comparison_basis["confirmatory"] or comparison_basis[
        "observation_counts_used_as_features"
    ]:
        raise RuntimeError("Prospective conventional comparison timing or features changed")
    comparison = pd.read_parquet(
        ROOT
        / prospective_conventional_comparison["outputs"]["predictions"]["path"]
    )
    expected_methods = {"sentinel2_topography_ridge", "tessera_area_ridge"}
    expected_rules = {"all_sources", "nearest5_evt_phys"}
    if len(comparison) != 5142 * 2 * 2 or set(comparison["method"]) != expected_methods:
        raise RuntimeError("Prospective conventional comparison population is incomplete")
    if set(comparison["source_rule"]) != expected_rules:
        raise RuntimeError("Prospective conventional source rules are incomplete")
    group_rows = comparison.groupby(
        ["source_rule", "method", "site_id"]
    ).size()
    expected_rows = evaluated.groupby("site_id").size() / 3
    for (_, _, site_id), rows in group_rows.items():
        if rows != expected_rows.loc[site_id]:
            raise RuntimeError("Prospective comparison site row counts changed")
    frozen_tessera = evaluated[
        evaluated["strategy"].isin(expected_rules)
    ][["site_id", "shot_number", "strategy", "prediction"]].rename(
        columns={"strategy": "source_rule", "prediction": "frozen_prediction"}
    )
    comparison_tessera = comparison[
        comparison["method"].eq("tessera_area_ridge")
    ][["site_id", "shot_number", "source_rule", "prediction"]]
    reproduction = comparison_tessera.merge(
        frozen_tessera,
        on=["site_id", "shot_number", "source_rule"],
        validate="one_to_one",
    )
    if np.max(np.abs(reproduction["prediction"] - reproduction["frozen_prediction"])) > 1e-12:
        raise RuntimeError("Conventional comparison changed a frozen TESSERA prediction")
    comparison_metrics = pd.read_csv(
        ROOT / prospective_conventional_comparison["outputs"]["metrics"]["path"]
    )
    comparison_sites = comparison_metrics[comparison_metrics["scope"].eq("site")]
    if comparison_sites.groupby(["source_rule", "method"])["site_id"].nunique().ne(6).any():
        raise RuntimeError("Prospective baseline metrics lack complete site results")
    comparison_paired = pd.read_csv(
        ROOT / prospective_conventional_comparison["outputs"]["paired"]["path"]
    )
    if len(comparison_paired) != 2 or set(comparison_paired["source_rule"]) != expected_rules:
        raise RuntimeError("Prospective baseline paired comparisons are incomplete")
    comparison_bootstrap = pd.read_csv(
        ROOT / prospective_conventional_comparison["outputs"]["bootstrap"]["path"]
    )
    if comparison_bootstrap.groupby("source_rule").size().ne(10000).any():
        raise RuntimeError("Prospective baseline bootstrap has the wrong replicate count")

    print(json.dumps({
        "habitat_freeze_id": habitat["freeze_id"],
        "transfer_freeze_id": transfer["freeze_id"],
        "profiled_sites": int(profiles["site_id"].nunique()),
        "prospective_targets": sorted(EXPECTED_PROSPECTIVE),
        "primary_mean_delta_rmse": float(predeclared["mean_delta_rmse"]),
        "primary_sites_improved": int(predeclared["sites_with_lower_rmse"]),
        "prospective_gedi_freeze_id": prospective_gedi["freeze_id"],
        "prospective_tessera_alignment_id": prospective_tessera["alignment_id"],
        "prospective_prediction_id": prospective_prediction["prediction_id"],
        "prospective_evaluation_id": prospective_evaluation["evaluation_id"],
        "viable_prospective_sites": sorted(EXPECTED_VIABLE),
        "prospective_rows": len(aligned),
        "target_outcomes_opened": evaluation_basis["target_outcomes_opened"],
        "prospective_primary_gate_passed": evaluation_gate["passed"],
        "prospective_primary_mean_delta_rmse": evaluation_gate["mean_delta_rmse"],
        "matched_size_control_freeze_id": matched_size["freeze_id"],
        "matched_size_monte_carlo_p": null_basis["one_sided_monte_carlo_p"],
        "prospective_conventional_freeze_id": prospective_conventional["freeze_id"],
        "prospective_conventional_comparison_freeze_id": (
            prospective_conventional_comparison["freeze_id"]
        ),
        "physiognomy_matched_tessera_minus_sentinel_rmse": float(
            comparison_paired.set_index("source_rule").loc[
                "nearest5_evt_phys", "mean_delta_rmse"
            ]
        ),
        "status": "validated",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
