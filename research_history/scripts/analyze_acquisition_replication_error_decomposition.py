#!/usr/bin/env python3
"""Decompose frozen Phase 13 error into mean-level and spatial components."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_acquisition_replication_track_replication as evaluation  # noqa: E402
import run_context_height_footprint_height_transfer as common  # noqa: E402


PROTOCOL_PATH = ROOT / "metadata/project_config_phase13_track_replication_protocol_freeze.yaml"
TARGET_FREEZE_PATH = ROOT / "metadata/phase13_combined_target_free_freeze.json"
PREDICTION_FREEZE_PATH = ROOT / "metadata/phase13_prediction_freeze.json"
RESULT_FREEZE_PATH = ROOT / "metadata/phase13_track_replication_result_freeze.json"
PREDICTIONS_PATH = ROOT / "data/processed/phase13_frozen_source_predictions.parquet"
SAMPLES_PATH = ROOT / "data/processed/phase13_frozen_local_samples.parquet"
SCENARIO_PATH = ROOT / "outputs/tables/phase13_error_decomposition_scenario.csv"
FOLD_PATH = ROOT / "outputs/tables/phase13_error_decomposition_fold.csv"
SITE_PATH = ROOT / "outputs/tables/phase13_error_decomposition_site.csv"
MACRO_PATH = ROOT / "outputs/tables/phase13_error_decomposition_macro.csv"
FREEZE_PATH = ROOT / "metadata/phase13_error_decomposition_freeze.json"

MODELS = [
    "source_only_tessera",
    "sampled_target_local_mean",
    "tessera_plus_sampled_local_offset",
    "source_only_sentinel2_terrain",
    "sentinel2_terrain_plus_sampled_local_offset",
]
COMPONENTS = ["total_mse", "squared_bias", "centered_mse", "observed_variance"]


def error_components(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if observed.shape != predicted.shape or observed.ndim != 1 or len(observed) == 0:
        raise ValueError("Observed and predicted values must be non-empty aligned vectors")
    if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        raise ValueError("Error decomposition requires finite values")
    residual = predicted - observed
    bias = float(np.mean(residual))
    observed_centered = observed - np.mean(observed)
    predicted_centered = predicted - np.mean(predicted)
    total_mse = float(np.mean(np.square(residual)))
    centered_mse = float(np.mean(np.square(predicted_centered - observed_centered)))
    observed_variance = float(np.mean(np.square(observed_centered)))
    squared_bias = bias**2
    if not np.isclose(total_mse, squared_bias + centered_mse, rtol=1e-10, atol=1e-12):
        raise RuntimeError("MSE decomposition identity failed")
    return {
        "total_mse": total_mse,
        "squared_bias": squared_bias,
        "centered_mse": centered_mse,
        "observed_variance": observed_variance,
    }


def add_derived_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["total_rmse"] = np.sqrt(result["total_mse"])
    result["bias_component_rmse"] = np.sqrt(result["squared_bias"])
    result["centered_rmse"] = np.sqrt(result["centered_mse"])
    result["spatial_skill_over_constant_mean"] = np.where(
        result["observed_variance"].gt(0),
        1.0 - result["centered_mse"] / result["observed_variance"],
        np.nan,
    )
    return result


def aggregate_components(
    frame: pd.DataFrame, group_columns: list[str], count_name: str
) -> pd.DataFrame:
    aggregated = (
        frame.groupby(group_columns, as_index=False)
        .agg(**{count_name: ("total_mse", "size")}, **{name: (name, "mean") for name in COMPONENTS})
    )
    return add_derived_metrics(aggregated)


def main() -> int:
    protected = [SCENARIO_PATH, FOLD_PATH, SITE_PATH, MACRO_PATH, FREEZE_PATH]
    existing = [str(path.relative_to(ROOT)) for path in protected if path.exists()]
    if existing:
        raise RuntimeError(f"Phase 13 decomposition outputs exist; refusing to overwrite: {existing}")

    target_freeze = json.loads(TARGET_FREEZE_PATH.read_text(encoding="utf-8"))
    prediction_freeze = json.loads(PREDICTION_FREEZE_PATH.read_text(encoding="utf-8"))
    result_freeze = json.loads(RESULT_FREEZE_PATH.read_text(encoding="utf-8"))
    if result_freeze["status"] != "complete_gate_failed":
        raise RuntimeError("Phase 13 decomposition expects the completed failed-gate result")
    if result_freeze["freeze_basis"]["prediction_freeze_id"] != prediction_freeze["freeze_id"]:
        raise RuntimeError("Phase 13 result and prediction freezes disagree")

    predictions = pd.read_parquet(PREDICTIONS_PATH)
    samples = pd.read_parquet(SAMPLES_PATH)
    outcomes = evaluation.load_sealed_outcomes(target_freeze)
    target = predictions.merge(outcomes, on=["site_id", "shot_number"], validate="one_to_one")
    if len(target) != len(predictions):
        raise RuntimeError("Frozen Phase 13 predictions lack sealed outcomes")

    records: list[dict[str, Any]] = []
    groups = samples.groupby(
        ["site_id", "held_out_pass", "budget", "replicate"], sort=True
    )
    for (site, held_out, budget, replicate), sampled in groups:
        site_target = target[target["site_id"].eq(site)].set_index("shot_number", drop=False)
        test = site_target[site_target["pass_id"].eq(held_out)]
        local = site_target.loc[sampled["shot_number"].to_numpy()]
        if test.empty or sampled["pass_id"].eq(held_out).any():
            raise RuntimeError("Phase 13 decomposition found local/test overlap")
        observed = test["fhd_normal"].to_numpy(dtype=np.float64)
        tessera_test = test["source_tessera_prediction"].to_numpy(dtype=np.float64)
        sentinel_test = test["source_sentinel2_terrain_prediction"].to_numpy(dtype=np.float64)
        tessera_local = local["source_tessera_prediction"].to_numpy(dtype=np.float64)
        sentinel_local = local["source_sentinel2_terrain_prediction"].to_numpy(dtype=np.float64)
        local_observed = local["fhd_normal"].to_numpy(dtype=np.float64)
        local_mean = float(np.mean(local_observed))
        tessera_offset = float(np.mean(local_observed - tessera_local))
        sentinel_offset = float(np.mean(local_observed - sentinel_local))
        model_predictions = {
            "source_only_tessera": tessera_test,
            "sampled_target_local_mean": np.full(len(test), local_mean),
            "tessera_plus_sampled_local_offset": tessera_test + tessera_offset,
            "source_only_sentinel2_terrain": sentinel_test,
            "sentinel2_terrain_plus_sampled_local_offset": sentinel_test + sentinel_offset,
        }
        for model, predicted in model_predictions.items():
            records.append(
                {
                    "target_year": int(test["target_year"].iloc[0]),
                    "site_id": site,
                    "held_out_pass": held_out,
                    "budget": int(budget),
                    "replicate": int(replicate),
                    "model": model,
                    "test_rows": len(test),
                    **error_components(observed, predicted),
                }
            )

    scenario = add_derived_metrics(pd.DataFrame(records))
    fold = aggregate_components(
        scenario,
        ["target_year", "site_id", "held_out_pass", "budget", "model"],
        "replicates",
    )
    site = aggregate_components(
        fold,
        ["target_year", "site_id", "budget", "model"],
        "folds",
    )
    macro = aggregate_components(site, ["budget", "model"], "sites")

    primary_budget = 50
    primary = macro[macro["budget"].eq(primary_budget)].set_index("model")
    source = primary.loc["source_only_tessera"]
    corrected = primary.loc["tessera_plus_sampled_local_offset"]
    local_mean = primary.loc["sampled_target_local_mean"]
    source_improvement = float(source["total_mse"] - corrected["total_mse"])
    bias_reduction = float(source["squared_bias"] - corrected["squared_bias"])
    findings = {
        "primary_budget": primary_budget,
        "source_to_corrected_total_mse_reduction": source_improvement,
        "source_to_corrected_squared_bias_reduction": bias_reduction,
        "source_to_corrected_centered_mse_change": float(
            corrected["centered_mse"] - source["centered_mse"]
        ),
        "fraction_of_source_improvement_explained_by_bias_reduction": (
            bias_reduction / source_improvement if source_improvement > 0 else None
        ),
        "corrected_minus_local_mean_total_mse": float(
            corrected["total_mse"] - local_mean["total_mse"]
        ),
        "corrected_minus_local_mean_centered_mse": float(
            corrected["centered_mse"] - local_mean["centered_mse"]
        ),
        "tessera_spatial_skill_over_constant_mean": float(
            corrected["spatial_skill_over_constant_mean"]
        ),
    }

    common.write_csv(scenario, SCENARIO_PATH)
    common.write_csv(fold, FOLD_PATH)
    common.write_csv(site, SITE_PATH)
    common.write_csv(macro, MACRO_PATH)
    freeze_basis = {
        "analysis_scope": "post_result_descriptive_error_decomposition_without_model_refitting",
        "protocol_sha256": common.sha256(PROTOCOL_PATH),
        "target_freeze_id": target_freeze["freeze_id"],
        "prediction_freeze_id": prediction_freeze["freeze_id"],
        "result_freeze_id": result_freeze["freeze_id"],
        "script_sha256": common.sha256(Path(__file__).resolve()),
        "aggregation": "replicates_within_fold_then_folds_within_site_then_equal_weight_sites",
        "identity": "total_mse_equals_squared_mean_error_plus_centered_mse",
        "models": MODELS,
        "findings": findings,
    }
    freeze = {
        "freeze_id": "phase13-error-decomposition-" + common.canonical_hash(freeze_basis)[:12],
        "created_utc": common.utc_now(),
        "status": "complete_descriptive_diagnostic",
        "freeze_basis": freeze_basis,
        "outputs": {
            str(path.relative_to(ROOT)): common.sha256(path)
            for path in [SCENARIO_PATH, FOLD_PATH, SITE_PATH, MACRO_PATH]
        },
    }
    common.write_json(FREEZE_PATH, freeze)
    print(json.dumps({"freeze_id": freeze["freeze_id"], "findings": findings}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
