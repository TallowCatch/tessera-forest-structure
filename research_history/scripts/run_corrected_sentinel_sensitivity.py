#!/usr/bin/env python3
"""Rerun Sentinel baselines after excluding acquisition-count diagnostics."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import yaml
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train_baselines import buffered_split, ridge_inner_tuning


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/project.yaml"
DEVELOPMENT_PATH = ROOT / "data/processed/tessera_aligned_development_soap_teak.parquet"
BART_PATH = ROOT / "data/processed/tessera_aligned_locked_bart.parquet"
DEVELOPMENT_CONVENTIONAL_PATH = ROOT / "data/processed/phase3_conventional_predictors_development.parquet"
BART_CONVENTIONAL_PATH = ROOT / "data/processed/phase4_conventional_predictors_bart.parquet"
FOLDS_PATH = ROOT / "metadata/phase3_spatial_folds.parquet"
DEVELOPMENT_CONVENTIONAL_FREEZE_PATH = ROOT / "metadata/phase3_conventional_predictor_freeze.json"
BART_CONVENTIONAL_FREEZE_PATH = ROOT / "metadata/phase4_bart_predictor_freeze.json"
FOLD_FREEZE_PATH = ROOT / "metadata/phase3_spatial_fold_freeze.json"
LOCKED_EVALUATION_FREEZE_PATH = ROOT / "metadata/phase4_locked_bart_evaluation_freeze.json"

DEVELOPMENT_PREDICTIONS_PATH = ROOT / "data/processed/phase4_corrected_sentinel_development_oof.parquet"
TRANSFER_PREDICTIONS_PATH = ROOT / "data/processed/phase4_corrected_sentinel_transfer_predictions.parquet"
BART_PREDICTIONS_PATH = ROOT / "data/processed/phase4_corrected_sentinel_bart_predictions.parquet"
DEVELOPMENT_SUMMARY_PATH = ROOT / "outputs/tables/phase4_corrected_sentinel_development_summary.csv"
TRANSFER_SUMMARY_PATH = ROOT / "outputs/tables/phase4_corrected_sentinel_transfer_summary.csv"
BART_SUMMARY_PATH = ROOT / "outputs/tables/phase4_corrected_sentinel_bart_summary.csv"
TUNING_PATH = ROOT / "outputs/tables/phase4_corrected_sentinel_tuning.csv"
COMPARISON_PATH = ROOT / "outputs/tables/phase4_corrected_sentinel_comparison.csv"
FIGURE_PATH = ROOT / "outputs/figures/phase4_corrected_sentinel_comparison.png"
MODEL_PATH = ROOT / "outputs/models/phase4_corrected_sentinel_ridge.joblib"
FREEZE_PATH = ROOT / "metadata/phase4_corrected_sentinel_sensitivity_freeze.json"

ORIGINAL_DEVELOPMENT_SUMMARY_PATH = ROOT / "outputs/tables/phase3_extended_baseline_summary.csv"
ORIGINAL_TRANSFER_SUMMARY_PATH = ROOT / "outputs/tables/phase4_soap_teak_transfer_summary.csv"
ORIGINAL_BART_SUMMARY_PATH = ROOT / "outputs/tables/phase4_locked_bart_metrics.csv"
METRICS = ["r2", "rmse", "mae", "spearman_r", "mean_bias"]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def joblib_dump(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(value, temporary, compress=3)
    temporary.replace(path)


def ridge_pipeline(alpha: float) -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha))])


def metric_values(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = predicted - observed
    rank = math.nan if np.ptp(predicted) <= 1e-12 else float(spearmanr(observed, predicted).statistic)
    return {
        "r2": float(r2_score(observed, predicted)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(mean_absolute_error(observed, predicted)),
        "spearman_r": rank,
        "mean_bias": float(np.mean(residual)),
    }


def bootstrap_intervals(
    frame: pd.DataFrame,
    prediction_column: str,
    replicates: int,
    seed: int,
    stratify_by_site: bool,
) -> dict[str, tuple[float, float]]:
    random = np.random.default_rng(seed)
    if stratify_by_site:
        strata = {
            site: {
                block: group.index.to_numpy()
                for block, group in site_frame.groupby("spatial_block_id")
            }
            for site, site_frame in frame.groupby("site_id")
        }
    else:
        strata = {
            "ALL": {
                block: group.index.to_numpy()
                for block, group in frame.groupby("spatial_block_id")
            }
        }
    distributions = {metric: [] for metric in METRICS}
    for _ in range(replicates):
        sampled: list[np.ndarray] = []
        for lookup in strata.values():
            block_ids = sorted(lookup)
            selected = random.choice(block_ids, size=len(block_ids), replace=True)
            sampled.extend(lookup[block] for block in selected)
        sample = frame.loc[np.concatenate(sampled)]
        values = metric_values(
            sample["fhd_normal"].to_numpy(dtype=np.float64),
            sample[prediction_column].to_numpy(dtype=np.float64),
        )
        for metric, value in values.items():
            distributions[metric].append(value)
    intervals: dict[str, tuple[float, float]] = {}
    for metric, samples in distributions.items():
        numeric = np.asarray(samples, dtype=np.float64)
        numeric = numeric[np.isfinite(numeric)]
        intervals[metric] = (
            float(np.percentile(numeric, 2.5)),
            float(np.percentile(numeric, 97.5)),
        )
    return intervals


def summary_record(
    frame: pd.DataFrame,
    prediction_column: str,
    scope: str,
    replicates: int,
    seed: int,
    stratify_by_site: bool,
) -> dict[str, Any]:
    intervals = bootstrap_intervals(
        frame, prediction_column, replicates, seed, stratify_by_site
    )
    values = metric_values(
        frame["fhd_normal"].to_numpy(dtype=np.float64),
        frame[prediction_column].to_numpy(dtype=np.float64),
    )
    record: dict[str, Any] = {
        "scope": scope,
        "model": "sentinel_topography_ridge_corrected",
        "rows": len(frame),
        "spatial_blocks": frame["spatial_block_id"].nunique(),
        "bootstrap_replicates": replicates,
    }
    for metric, value in values.items():
        low, high = intervals[metric]
        record[metric] = value
        record[f"{metric}_ci_low"] = low
        record[f"{metric}_ci_high"] = high
    return record


def tune_within_site(
    frame: pd.DataFrame,
    source_mask: np.ndarray,
    features: list[str],
    alphas: list[float],
    buffer_m: float,
    direction: str,
) -> tuple[float, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    for alpha in alphas:
        for fold in sorted(frame.loc[source_mask, "spatial_fold_id"].unique()):
            train, validation, removed, minimum_distance = buffered_split(
                frame, int(fold), buffer_m, source_mask
            )
            model = ridge_pipeline(alpha)
            model.fit(
                frame.iloc[train][features].to_numpy(dtype=np.float32),
                frame.iloc[train]["fhd_normal"].to_numpy(dtype=np.float64),
            )
            prediction = model.predict(
                frame.iloc[validation][features].to_numpy(dtype=np.float32)
            )
            observed = frame.iloc[validation]["fhd_normal"].to_numpy(dtype=np.float64)
            records.append(
                {
                    "stage": "site_transfer_tuning",
                    "scope": direction,
                    "alpha": alpha,
                    "validation_fold": int(fold),
                    "training_rows": len(train),
                    "validation_rows": len(validation),
                    "buffer_removed_rows": removed,
                    "minimum_train_validation_distance_m": minimum_distance,
                    "validation_rmse": float(np.sqrt(np.mean((prediction - observed) ** 2))),
                }
            )
    tuning = pd.DataFrame(records)
    selected = float(
        tuning.groupby("alpha", as_index=False)["validation_rmse"]
        .mean()
        .sort_values(["validation_rmse", "alpha"])
        .iloc[0]["alpha"]
    )
    for record in records:
        record["selected_alpha"] = selected
    return selected, records


def comparison_table(
    corrected_development: pd.DataFrame,
    corrected_transfer: pd.DataFrame,
    corrected_bart: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    original_development = pd.read_csv(ORIGINAL_DEVELOPMENT_SUMMARY_PATH)
    original_transfer = pd.read_csv(ORIGINAL_TRANSFER_SUMMARY_PATH)
    original_bart = pd.read_csv(ORIGINAL_BART_SUMMARY_PATH)
    selections = [
        (
            "development_oof",
            "ALL",
            original_development[
                original_development["scope"].eq("ALL")
                & original_development["model"].eq("sentinel_topography_ridge")
            ].iloc[0],
            corrected_development.iloc[0],
        ),
        *[
            (
                "site_transfer",
                direction,
                original_transfer[
                    original_transfer["direction"].eq(direction)
                    & original_transfer["model"].eq("sentinel_topography_ridge")
                ].iloc[0],
                corrected_transfer[corrected_transfer["scope"].eq(direction)].iloc[0],
            )
            for direction in ["SOAP_to_TEAK", "TEAK_to_SOAP"]
        ],
        (
            "bart_transfer",
            "BART",
            original_bart[original_bart["model"].eq("sentinel_topography_ridge")].iloc[0],
            corrected_bart.iloc[0],
        ),
    ]
    for stage, scope, original, corrected in selections:
        for version, values, valid in [
            ("original_with_count_diagnostics", original, False),
            ("corrected_without_count_diagnostics", corrected, True),
        ]:
            rows.append(
                {
                    "stage": stage,
                    "scope": scope,
                    "version": version,
                    "scientifically_valid_feature_set": valid,
                    **{metric: float(values[metric]) for metric in METRICS},
                }
            )
    return pd.DataFrame(rows)


def plot_comparison(comparison: pd.DataFrame, path: Path) -> None:
    order = ["development_oof:ALL", "site_transfer:SOAP_to_TEAK", "site_transfer:TEAK_to_SOAP", "bart_transfer:BART"]
    comparison = comparison.copy()
    comparison["label"] = comparison["stage"] + ":" + comparison["scope"]
    figure, axis = plt.subplots(figsize=(9, 5))
    width = 0.36
    positions = np.arange(len(order))
    for offset, (version, colour) in zip(
        [-width / 2, width / 2],
        [
            ("original_with_count_diagnostics", "#bb3e03"),
            ("corrected_without_count_diagnostics", "#005f73"),
        ],
        strict=True,
    ):
        lookup = comparison[comparison["version"].eq(version)].set_index("label")["rmse"]
        axis.bar(positions + offset, [lookup[label] for label in order], width, label=version, color=colour)
    axis.set_xticks(positions, [label.replace(":", "\n") for label in order])
    axis.set_ylabel("RMSE")
    axis.set_title("Sentinel + topography correction sensitivity")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.png")
    figure.savefig(temporary, dpi=180, bbox_inches="tight")
    plt.close(figure)
    temporary.replace(path)


def main() -> int:
    protected = [
        DEVELOPMENT_PREDICTIONS_PATH,
        TRANSFER_PREDICTIONS_PATH,
        BART_PREDICTIONS_PATH,
        DEVELOPMENT_SUMMARY_PATH,
        TRANSFER_SUMMARY_PATH,
        BART_SUMMARY_PATH,
        TUNING_PATH,
        COMPARISON_PATH,
        FIGURE_PATH,
        MODEL_PATH,
        FREEZE_PATH,
    ]
    if any(path.exists() for path in protected):
        raise RuntimeError("Corrected Sentinel sensitivity already exists; refusing to overwrite")

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    protocol = config["phase4_corrected_sentinel_sensitivity"]
    development_manifest = json.loads(DEVELOPMENT_CONVENTIONAL_FREEZE_PATH.read_text())
    bart_manifest = json.loads(BART_CONVENTIONAL_FREEZE_PATH.read_text())
    fold_manifest = json.loads(FOLD_FREEZE_PATH.read_text())
    locked_evaluation = json.loads(LOCKED_EVALUATION_FREEZE_PATH.read_text())
    all_features = development_manifest["freeze_basis"]["feature_columns"]
    excluded = [
        feature for feature in all_features
        if any(feature.endswith(suffix) for suffix in protocol["excluded_model_feature_suffixes"])
    ]
    features = [feature for feature in all_features if feature not in excluded]
    if excluded != protocol["expected_excluded_columns"]:
        raise RuntimeError(f"Excluded columns differ from protocol: {excluded}")
    if len(features) != int(protocol["expected_model_feature_count"]):
        raise RuntimeError("Corrected Sentinel feature count differs from protocol")
    if bart_manifest["freeze_basis"]["recipe"] != development_manifest["freeze_basis"]["recipe"]:
        raise RuntimeError("Development and BART predictor recipes differ")

    folds = pd.read_parquet(FOLDS_PATH)
    development = pd.read_parquet(DEVELOPMENT_PATH).merge(
        folds[["site_id", "shot_number", "spatial_block_id", "spatial_fold_id"]],
        on=["site_id", "shot_number"],
        validate="one_to_one",
    ).merge(
        pd.read_parquet(DEVELOPMENT_CONVENTIONAL_PATH)[["site_id", "shot_number", *all_features]],
        on=["site_id", "shot_number"],
        validate="one_to_one",
    )
    bart = pd.read_parquet(BART_PATH).merge(
        pd.read_parquet(BART_CONVENTIONAL_PATH)[["site_id", "shot_number", *all_features]],
        on=["site_id", "shot_number"],
        validate="one_to_one",
    )
    if not np.isfinite(development[features].to_numpy(dtype=float)).all() or not np.isfinite(bart[features].to_numpy(dtype=float)).all():
        raise RuntimeError("Corrected feature set contains non-finite values")

    alphas = [float(value) for value in protocol["ridge_alpha_grid"]]
    buffer_m = float(protocol["exclusion_buffer_m"])
    replicates = int(protocol["uncertainty"]["replicates"])
    seed = int(protocol["uncertainty"]["random_seed"])
    oof = np.full(len(development), np.nan, dtype=np.float64)
    tuning_records: list[dict[str, Any]] = []
    for outer_fold in sorted(development["spatial_fold_id"].unique()):
        train, test, _, _ = buffered_split(development, int(outer_fold), buffer_m)
        selected_alpha, records = ridge_inner_tuning(
            development, features, int(outer_fold), train, alphas, buffer_m
        )
        for record in records:
            record["stage"] = "development_nested_tuning"
            record["scope"] = "ALL"
            record["selected_alpha"] = selected_alpha
        tuning_records.extend(records)
        model = ridge_pipeline(selected_alpha)
        model.fit(
            development.iloc[train][features].to_numpy(dtype=np.float32),
            development.iloc[train]["fhd_normal"].to_numpy(dtype=np.float64),
        )
        oof[test] = model.predict(development.iloc[test][features].to_numpy(dtype=np.float32))
    if not np.isfinite(oof).all():
        raise RuntimeError("Corrected development OOF predictions are incomplete")
    development_predictions = development[
        ["site_id", "shot_number", "spatial_block_id", "spatial_fold_id", "fhd_normal"]
    ].copy()
    development_predictions["prediction_sentinel_topography_ridge_corrected"] = oof
    development_summary = pd.DataFrame(
        [
            summary_record(
                development_predictions,
                "prediction_sentinel_topography_ridge_corrected",
                "ALL",
                replicates,
                seed,
                True,
            )
        ]
    )

    development_tuning = pd.DataFrame(
        [record for record in tuning_records if record["stage"] == "development_nested_tuning"]
    )
    final_alpha = float(
        development_tuning.groupby("alpha", as_index=False)["validation_rmse"]
        .mean()
        .sort_values(["validation_rmse", "alpha"])
        .iloc[0]["alpha"]
    )

    transfer_predictions: list[pd.DataFrame] = []
    transfer_summary_records: list[dict[str, Any]] = []
    transfer_alphas: dict[str, float] = {}
    for number, (train_site, test_site) in enumerate([("SOAP", "TEAK"), ("TEAK", "SOAP")]):
        direction = f"{train_site}_to_{test_site}"
        source_mask = development["site_id"].eq(train_site).to_numpy()
        target_mask = development["site_id"].eq(test_site).to_numpy()
        selected_alpha, records = tune_within_site(
            development, source_mask, features, alphas, buffer_m, direction
        )
        tuning_records.extend(records)
        transfer_alphas[direction] = selected_alpha
        model = ridge_pipeline(selected_alpha)
        model.fit(
            development.loc[source_mask, features].to_numpy(dtype=np.float32),
            development.loc[source_mask, "fhd_normal"].to_numpy(dtype=np.float64),
        )
        output = development.loc[
            target_mask, ["site_id", "shot_number", "spatial_block_id", "fhd_normal"]
        ].copy().reset_index(drop=True)
        output.insert(0, "direction", direction)
        output["prediction_sentinel_topography_ridge_corrected"] = model.predict(
            development.loc[target_mask, features].to_numpy(dtype=np.float32)
        )
        transfer_predictions.append(output)
        record = summary_record(
            output,
            "prediction_sentinel_topography_ridge_corrected",
            direction,
            replicates,
            seed + number + 1,
            False,
        )
        record["selected_alpha"] = selected_alpha
        transfer_summary_records.append(record)
    transfer_prediction_table = pd.concat(transfer_predictions, ignore_index=True)
    transfer_summary = pd.DataFrame(transfer_summary_records)

    final_model = ridge_pipeline(final_alpha)
    final_model.fit(
        development[features].to_numpy(dtype=np.float32),
        development["fhd_normal"].to_numpy(dtype=np.float64),
    )
    bart_predictions = bart[["site_id", "shot_number", "x_epsg5070", "y_epsg5070", "fhd_normal"]].copy()
    block_width = float(config["phase4_locked_bart_evaluation"]["uncertainty"]["spatial_block_width_m"])
    block_x = np.floor(bart_predictions["x_epsg5070"] / block_width).astype(int)
    block_y = np.floor(bart_predictions["y_epsg5070"] / block_width).astype(int)
    bart_predictions["spatial_block_id"] = [
        f"BART_{x}_{y}" for x, y in zip(block_x, block_y, strict=True)
    ]
    bart_predictions["prediction_sentinel_topography_ridge_corrected"] = final_model.predict(
        bart[features].to_numpy(dtype=np.float32)
    )
    bart_summary = pd.DataFrame(
        [
            summary_record(
                bart_predictions,
                "prediction_sentinel_topography_ridge_corrected",
                "BART",
                replicates,
                seed + 3,
                False,
            )
        ]
    )
    bart_summary["selected_alpha_from_development_only"] = final_alpha
    corrected_comparison = comparison_table(development_summary, transfer_summary, bart_summary)

    freeze_basis = {
        "status": protocol["status"],
        "project_config_sha256": sha256(CONFIG_PATH),
        "script_sha256": sha256(Path(__file__).resolve()),
        "helper_script_sha256": sha256(ROOT / "scripts/train_baselines.py"),
        "sklearn_version": sklearn.__version__,
        "development_conventional_predictor_freeze_id": development_manifest["freeze_id"],
        "bart_conventional_predictor_freeze_id": bart_manifest["freeze_id"],
        "spatial_fold_freeze_id": fold_manifest["freeze_id"],
        "prior_locked_bart_evaluation_freeze_id": locked_evaluation["freeze_id"],
        "protocol": protocol,
        "all_predictor_table_columns": all_features,
        "excluded_model_features": excluded,
        "corrected_model_features": features,
        "development_selected_alpha": final_alpha,
        "transfer_selected_alphas": transfer_alphas,
        "bart_tuning_or_model_selection_performed": False,
        "replaces_locked_confirmatory_result": False,
    }
    freeze_basis_sha = canonical_hash(freeze_basis)
    freeze_id = f"sentinel-correction-{freeze_basis_sha[:12]}"
    for frame in [development_predictions, transfer_prediction_table, bart_predictions]:
        frame.insert(0, "sentinel_correction_freeze_id", freeze_id)

    write_parquet(development_predictions, DEVELOPMENT_PREDICTIONS_PATH)
    write_parquet(transfer_prediction_table, TRANSFER_PREDICTIONS_PATH)
    write_parquet(bart_predictions, BART_PREDICTIONS_PATH)
    write_csv(development_summary, DEVELOPMENT_SUMMARY_PATH)
    write_csv(transfer_summary, TRANSFER_SUMMARY_PATH)
    write_csv(bart_summary, BART_SUMMARY_PATH)
    write_csv(pd.DataFrame(tuning_records), TUNING_PATH)
    write_csv(corrected_comparison, COMPARISON_PATH)
    plot_comparison(corrected_comparison, FIGURE_PATH)
    joblib_dump(
        {
            "sentinel_correction_freeze_id": freeze_id,
            "model": final_model,
            "features": features,
            "excluded_diagnostic_columns": excluded,
            "alpha": final_alpha,
            "training_sites": ["SOAP", "TEAK"],
        },
        MODEL_PATH,
    )
    output_paths = {
        "development_predictions": DEVELOPMENT_PREDICTIONS_PATH,
        "transfer_predictions": TRANSFER_PREDICTIONS_PATH,
        "bart_predictions": BART_PREDICTIONS_PATH,
        "development_summary": DEVELOPMENT_SUMMARY_PATH,
        "transfer_summary": TRANSFER_SUMMARY_PATH,
        "bart_summary": BART_SUMMARY_PATH,
        "tuning": TUNING_PATH,
        "comparison": COMPARISON_PATH,
        "figure": FIGURE_PATH,
        "model": MODEL_PATH,
    }
    manifest = {
        "freeze_id": freeze_id,
        "created_at": utc_now(),
        "status": "frozen_posthoc_sensitivity_after_locked_bart_evaluation",
        "freeze_basis_sha256": freeze_basis_sha,
        "freeze_basis": freeze_basis,
        "outputs": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for name, path in output_paths.items()
        },
        "results": {
            "development": development_summary.to_dict(orient="records"),
            "transfer": transfer_summary.to_dict(orient="records"),
            "bart": bart_summary.to_dict(orient="records"),
        },
    }
    json_dump(FREEZE_PATH, manifest)
    print(
        json.dumps(
            {
                "freeze_id": freeze_id,
                "excluded_columns": excluded,
                "model_feature_count": len(features),
                "development_alpha": final_alpha,
                "development": development_summary[["r2", "rmse", "mae", "spearman_r", "mean_bias"]].iloc[0].to_dict(),
                "transfer": transfer_summary[["scope", "selected_alpha", "r2", "rmse", "mae", "spearman_r", "mean_bias"]].to_dict(orient="records"),
                "bart": bart_summary[["r2", "rmse", "mae", "spearman_r", "mean_bias"]].iloc[0].to_dict(),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
