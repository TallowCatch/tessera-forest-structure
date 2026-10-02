#!/usr/bin/env python3
"""Validate frozen Phase 5 local and disturbance analyses."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_output_records(freeze: dict) -> None:
    for record in freeze["outputs"].values():
        path = ROOT / record["path"]
        if not path.is_file():
            raise RuntimeError(f"Missing frozen output: {record['path']}")
        if sha256(path) != record["sha256"]:
            raise RuntimeError(f"Checksum mismatch: {record['path']}")


def main() -> int:
    local_freeze = json.loads((ROOT / "metadata/phase5_bart_local_validation_freeze.json").read_text())
    validate_output_records(local_freeze)
    if local_freeze["status"] != "frozen_posthoc_local_validation":
        raise RuntimeError("BART-local result lost its post-hoc label")
    if not local_freeze["freeze_basis"]["no_bart_hyperparameter_tuning"]:
        raise RuntimeError("BART hyperparameter tuning was not prohibited")
    corrected = local_freeze["freeze_basis"]["feature_sets"]["sentinel_topography_ridge_corrected"]
    if any(column.endswith("observation_count") for column in corrected):
        raise RuntimeError("Corrected Sentinel features contain an observation count")
    predictions = pd.read_parquet(ROOT / local_freeze["outputs"]["predictions"]["path"])
    diagnostics = pd.read_csv(ROOT / local_freeze["outputs"]["fold_diagnostics"]["path"])
    pass_metrics = pd.read_csv(ROOT / local_freeze["outputs"]["pass_metrics"]["path"])
    if len(predictions) != 1026 or predictions["held_out_pass"].nunique() != 3:
        raise RuntimeError("BART-local predictions are not the frozen 1,026 rows/three passes")
    if diagnostics["minimum_train_test_distance_m"].min() < 500:
        raise RuntimeError("BART-local buffer is below 500 m")
    prediction_columns = [column for column in predictions if column.startswith("prediction_")]
    if len(prediction_columns) != 6 or not np.isfinite(predictions[prediction_columns].to_numpy()).all():
        raise RuntimeError("BART-local prediction matrix is incomplete or non-finite")
    observed = predictions["fhd_normal"].to_numpy(dtype=float)
    primary = predictions["prediction_tessera_area_topography_ridge"].to_numpy(dtype=float)
    stored = local_freeze["primary_metrics"]
    recalculated = {
        "r2": r2_score(observed, primary),
        "rmse": np.sqrt(np.mean((primary - observed) ** 2)),
        "mae": mean_absolute_error(observed, primary),
        "mean_bias": np.mean(primary - observed),
    }
    if any(not math.isclose(float(stored[key]), float(value), abs_tol=1e-12) for key, value in recalculated.items()):
        raise RuntimeError("BART-local primary metrics do not reproduce")
    primary_passes = pass_metrics[pass_metrics["model"].eq("tessera_area_topography_ridge")]
    if len(primary_passes) != 3 or int((primary_passes["r2"] > 0).sum()) != 1:
        raise RuntimeError("Expected one of three positive primary pass R-squared values")

    disturbance_freeze = json.loads((ROOT / "metadata/phase5_landfire_disturbance_freeze.json").read_text())
    validate_output_records(disturbance_freeze)
    disturbance = pd.read_parquet(ROOT / disturbance_freeze["outputs"]["disturbance_history"]["path"])
    if len(disturbance) != 3063 or int(disturbance["has_recorded_disturbance_1999_2024"].sum()) != 1524:
        raise RuntimeError("LANDFIRE population/count changed")
    if disturbance["is_chronological_stand_age"].any():
        raise RuntimeError("Disturbance history was incorrectly labelled stand age")
    years = disturbance["years_since_latest_recorded_disturbance"].dropna()
    if years.min() < 0 or years.max() > 25:
        raise RuntimeError("Disturbance recency falls outside 1999-2024")
    if disturbance_freeze["freeze_basis"]["source_rasters_retained"]:
        raise RuntimeError("LANDFIRE source rasters were unexpectedly retained")

    variation_freeze = json.loads((ROOT / "metadata/phase5_disturbance_variation_freeze.json").read_text())
    validate_output_records(variation_freeze)
    cohorts = pd.read_parquet(ROOT / variation_freeze["outputs"]["cohorts"]["path"])
    if len(cohorts) != 4089 or cohorts["analysis"].nunique() != 3:
        raise RuntimeError("Disturbance error stratification populations changed")

    field_acquisition = json.loads(
        (ROOT / "metadata/phase5_neon_field_structure_acquisition_freeze.json").read_text()
    )
    if len(field_acquisition["files"]) != 22:
        raise RuntimeError("NEON field acquisition is not the frozen 22-file population")
    for record in field_acquisition["files"]:
        path = ROOT / record["path"]
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise RuntimeError(f"NEON field acquisition mismatch: {record['path']}")

    simulator = json.loads(
        (ROOT / "metadata/phase5_gedi_simulator_smoke_freeze.json").read_text()
    )
    validate_output_records(simulator)
    if simulator["status"] != "passed_synthetic_executable_and_directional_fhd_gate":
        raise RuntimeError("GEDI simulator smoke gate is not passed")

    calibration = json.loads(
        (ROOT / "metadata/phase5_neon_lidar_calibration_freeze.json").read_text()
    )
    validate_output_records(calibration)
    if not calibration["gate"]["passed"]:
        raise RuntimeError("NEON cross-sensor development gate is not passed")
    calibration_predictions = pd.read_parquet(
        ROOT / calibration["outputs"]["cv_predictions"]["path"]
    )
    if len(calibration_predictions) != 295 or calibration_predictions["neon_tile_id"].nunique() != 12:
        raise RuntimeError("NEON calibration is not the frozen 295-row/12-tile population")
    calibration_observed = calibration_predictions["fhd_normal"].to_numpy()
    calibration_predicted = calibration_predictions["calibrated_fhd_cv"].to_numpy()
    calibration_baseline = calibration_predictions["training_mean_baseline_cv"].to_numpy()
    recalculated_calibration_rmse = float(
        np.sqrt(np.mean((calibration_predicted - calibration_observed) ** 2))
    )
    recalculated_baseline_rmse = float(
        np.sqrt(np.mean((calibration_baseline - calibration_observed) ** 2))
    )
    if not math.isclose(
        recalculated_calibration_rmse,
        float(calibration["gate"]["calibration_cv_rmse"]),
        abs_tol=1e-12,
    ) or not math.isclose(
        recalculated_baseline_rmse,
        float(calibration["gate"]["baseline_cv_rmse"]),
        abs_tol=1e-12,
    ):
        raise RuntimeError("NEON development calibration metrics do not reproduce")

    bart_lidar = json.loads(
        (ROOT / "metadata/phase5_neon_lidar_locked_bart_freeze.json").read_text()
    )
    validate_output_records(bart_lidar)
    evaluation = pd.read_parquet(ROOT / bart_lidar["outputs"]["evaluation"]["path"])
    if len(evaluation) != 233 or evaluation["neon_tile_id"].nunique() != 6:
        raise RuntimeError("Locked BART lidar result is not the frozen 233-row/six-tile population")
    coefficients = calibration["final_development_coefficients"]
    recreated = float(coefficients["intercept"]) + float(coefficients["slope"]) * evaluation[
        "simulated_fhd"
    ]
    if not np.allclose(recreated, evaluation["calibrated_neon_fhd"], rtol=0, atol=1e-12):
        raise RuntimeError("Locked BART lidar did not use the frozen development coefficients")
    bart_metrics = pd.read_csv(ROOT / bart_lidar["outputs"]["metrics"]["path"])
    bart_primary = bart_metrics[
        bart_metrics["scope_type"].eq("pooled")
        & bart_metrics["target"].eq("calibrated_NEON_lidar_FHD")
        & bart_metrics["model"].eq("tessera_area_topography_ridge")
    ]
    if len(bart_primary) != 1 or float(bart_primary.iloc[0]["r2"]) >= 0:
        raise RuntimeError("Expected a negative locked primary TESSERA-to-airborne FHD R2")

    field_analysis = json.loads(
        (ROOT / "metadata/phase5_neon_field_structure_analysis_freeze.json").read_text()
    )
    validate_output_records(field_analysis)
    field_plots = pd.read_parquet(ROOT / field_analysis["outputs"]["plots"]["path"])
    if len(field_plots) != 35 or field_plots["field_proxy_is_measured_fhd"].any():
        raise RuntimeError("NEON field result population or FHD limitation changed")
    bart_field = field_plots[field_plots["site_id"].eq("BART")]
    if len(bart_field) != 16 or int(bart_field["nearest_lidar_distance_m"].le(100).sum()) != 2:
        raise RuntimeError("BART field plot population/distance overlap changed")

    few_shot = json.loads(
        (ROOT / "metadata/phase5_bart_few_shot_adaptation_freeze.json").read_text()
    )
    validate_output_records(few_shot)
    few_shot_predictions = pd.read_parquet(ROOT / few_shot["outputs"]["predictions"]["path"])
    if len(few_shot_predictions) != 498636:
        raise RuntimeError("Few-shot prediction population changed")
    if few_shot_predictions["held_out_pass"].nunique() != 3:
        raise RuntimeError("Few-shot experiment lost a complete BART outer pass")
    few_shot_summary = pd.read_csv(ROOT / few_shot["outputs"]["summary"]["path"])
    finite = few_shot_summary[~few_shot_summary["budget_label"].astype(str).eq("all")]
    if finite["adaptation_recovery_gate_passed"].any():
        raise RuntimeError("A finite few-shot budget unexpectedly passes the recovery gate")
    all_budget = few_shot_summary[few_shot_summary["budget_label"].astype(str).eq("all")]
    passing_all = set(all_budget.loc[
        all_budget["adaptation_recovery_gate_passed"], "method"
    ])
    expected_passing_all = {
        "target_only_ridge",
        "equal_domain_weighted_source_plus_target_ridge",
    }
    if passing_all != expected_passing_all:
        raise RuntimeError(f"Unexpected all-label adaptation gates: {passing_all}")
    few_shot_metrics = pd.read_csv(ROOT / few_shot["outputs"]["metrics"]["path"])
    all_pass_metrics = few_shot_metrics[
        few_shot_metrics["scope"].eq("pass")
        & few_shot_metrics["budget_label"].astype(str).eq("all")
        & few_shot_metrics["method"].isin(expected_passing_all)
    ]
    positive_pass_counts = all_pass_metrics.groupby("method")["r2"].apply(lambda values: int((values > 0).sum()))
    if not positive_pass_counts.eq(1).all():
        raise RuntimeError("All-label adaptation is no longer positive on exactly one pass")

    coral_predictions = json.loads(
        (ROOT / "metadata/phase5_bart_coral_prediction_freeze.json").read_text()
    )
    validate_output_records(coral_predictions)
    coral_basis = coral_predictions["freeze_basis"]
    if (
        coral_basis["target_columns_read_contain_fhd_normal"]
        or coral_basis["target_labels_used_for_alignment_training_or_selection"]
    ):
        raise RuntimeError("CORAL target-free stage reports BART-label access")
    remaining = float(coral_basis["diagnostics"]["covariance_distance_fraction_remaining"])
    if remaining >= 0.01:
        raise RuntimeError("CORAL no longer closely aligns source-target covariance")
    coral_evaluation = json.loads(
        (ROOT / "metadata/phase5_bart_coral_evaluation_freeze.json").read_text()
    )
    validate_output_records(coral_evaluation)
    if coral_evaluation["gate_passed"]:
        raise RuntimeError("CORAL unexpectedly passes its predictive gate")
    coral_metrics = pd.read_csv(ROOT / coral_evaluation["outputs"]["metrics"]["path"])
    coral_pooled = coral_metrics[
        coral_metrics["scope"].eq("pooled") & coral_metrics["method"].eq("coral")
    ]
    if len(coral_pooled) != 1 or float(coral_pooled.iloc[0]["r2"]) >= 0:
        raise RuntimeError("Expected negative pooled BART CORAL R2")

    retained_laz = list((ROOT / "data").rglob("*.laz"))
    work_files = list((ROOT / "data/interim/neon_lidar_work").rglob("*"))
    if retained_laz or any(path.is_file() for path in work_files):
        raise RuntimeError("NEON source lidar or temporary processing files were retained")

    print("Phase 5 validation passed")
    print("BART local: 1,026 rows, three buffered passes, one positive pass-specific R2")
    print("Disturbance: 3,063 rows, 1,524 mapped events, explicitly not stand age")
    print("NEON calibration: 295 development footprints, 12 held-out tiles, gate passed")
    print("Locked BART lidar: 233 footprints, six tiles, no BART refitting")
    print("NEON field structure: 35 plots, explicitly not measured FHD")
    print("Few-shot adaptation: no 10-100-label recovery; all-label skill not pass-consistent")
    print("Label-free CORAL: covariance aligned, predictive gate failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
