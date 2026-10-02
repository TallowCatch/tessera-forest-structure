#!/usr/bin/env python3
"""Freeze SOAP/TEAK transfer results and select the pre-BART pipeline."""

from __future__ import annotations

import argparse
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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train_baselines import buffered_split


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/project.yaml"
SOURCE_PATH = ROOT / "data/processed/tessera_aligned_development_soap_teak.parquet"
CONVENTIONAL_PATH = ROOT / "data/processed/phase3_conventional_predictors_development.parquet"
FOLDS_PATH = ROOT / "metadata/phase3_spatial_folds.parquet"
PHASE2_FREEZE_PATH = ROOT / "metadata/tessera_phase2_alignment_freeze.json"
FOLD_FREEZE_PATH = ROOT / "metadata/phase3_spatial_fold_freeze.json"
CONVENTIONAL_FREEZE_PATH = ROOT / "metadata/phase3_conventional_predictor_freeze.json"
EXTENDED_FREEZE_PATH = ROOT / "metadata/phase3_extended_development_freeze.json"
EXTENDED_TUNING_PATH = ROOT / "outputs/tables/phase3_extended_nested_tuning.csv"

PREDICTIONS_PATH = ROOT / "data/processed/phase4_soap_teak_transfer_predictions.parquet"
SUMMARY_PATH = ROOT / "outputs/tables/phase4_soap_teak_transfer_summary.csv"
TUNING_PATH = ROOT / "outputs/tables/phase4_soap_teak_transfer_tuning.csv"
FIGURE_PATH = ROOT / "outputs/figures/phase4_soap_teak_transfer.png"
MODEL_PATH = ROOT / "outputs/models/phase4_selected_pipeline_pre_bart.joblib"
FREEZE_PATH = ROOT / "metadata/phase4_development_transfer_freeze.json"

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


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def joblib_dump_atomic(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(value, temporary, compress=3)
    temporary.replace(path)


def ridge_pipeline(alpha: float) -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha))])


def metric_values(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = predicted - observed
    if np.ptp(observed) <= 1e-12 or np.ptp(predicted) <= 1e-12:
        rank_correlation = math.nan
    else:
        rank_correlation = float(spearmanr(observed, predicted).statistic)
    return {
        "r2": float(r2_score(observed, predicted)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(mean_absolute_error(observed, predicted)),
        "spearman_r": rank_correlation,
        "mean_bias": float(np.mean(residual)),
    }


def tune_within_source(
    frame: pd.DataFrame,
    source_mask: np.ndarray,
    features: list[str],
    alphas: list[float],
    buffer_m: float,
    direction: str,
    feature_set: str,
) -> tuple[float, list[dict[str, Any]]]:
    folds = sorted(frame.loc[source_mask, "spatial_fold_id"].unique())
    records: list[dict[str, Any]] = []
    for alpha in alphas:
        for fold in folds:
            train, validation, removed, minimum_distance = buffered_split(
                frame,
                int(fold),
                buffer_m,
                source_mask,
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
                    "direction": direction,
                    "feature_set": feature_set,
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
    ranked = (
        tuning.groupby("alpha", as_index=False)["validation_rmse"]
        .mean()
        .sort_values(["validation_rmse", "alpha"])
    )
    selected = float(ranked.iloc[0]["alpha"])
    for record in records:
        record["selected_alpha"] = selected
    return selected, records


def block_bootstrap_intervals(
    frame: pd.DataFrame,
    prediction_columns: dict[str, str],
    replicates: int,
    seed: int,
) -> dict[tuple[str, str], tuple[float, float]]:
    random = np.random.default_rng(seed)
    blocks = {
        block: group.index.to_numpy()
        for block, group in frame.groupby("spatial_block_id")
    }
    block_ids = sorted(blocks)
    values = {
        (model, metric): [] for model in prediction_columns for metric in METRICS
    }
    for _ in range(replicates):
        selected = random.choice(block_ids, size=len(block_ids), replace=True)
        indices = np.concatenate([blocks[block] for block in selected])
        sample = frame.loc[indices]
        observed = sample["fhd_normal"].to_numpy(dtype=np.float64)
        for model, column in prediction_columns.items():
            metrics = metric_values(observed, sample[column].to_numpy(dtype=np.float64))
            for metric, value in metrics.items():
                values[(model, metric)].append(value)
    intervals: dict[tuple[str, str], tuple[float, float]] = {}
    for key, samples in values.items():
        numeric = np.asarray(samples, dtype=np.float64)
        numeric = numeric[np.isfinite(numeric)]
        if len(numeric) == 0:
            intervals[key] = (math.nan, math.nan)
        else:
            intervals[key] = (
                float(np.percentile(numeric, 2.5)),
                float(np.percentile(numeric, 97.5)),
            )
    return intervals


def select_pipeline(summary: pd.DataFrame, rule: dict[str, Any]) -> dict[str, Any]:
    eligible = list(rule["eligible"])
    worst = (
        summary[summary["model"].isin(eligible)]
        .groupby("model")["rmse"]
        .max()
        .to_dict()
    )
    if set(worst) != set(eligible):
        raise RuntimeError("Transfer metrics do not cover every eligible pipeline")
    ranked = sorted(worst, key=lambda model: (worst[model], model))
    threshold = float(rule["practical_tie_threshold_rmse"])
    if abs(worst[eligible[0]] - worst[eligible[1]]) <= threshold:
        selected = "tessera_area_ridge"
        reason = "practical_tie_predeclared_simpler_model"
    else:
        selected = ranked[0]
        reason = "lowest_worst_direction_rmse"
    return {
        "selected_model": selected,
        "selection_reason": reason,
        "worst_direction_rmse": {key: float(value) for key, value in worst.items()},
        "absolute_worst_rmse_difference": float(abs(worst[eligible[0]] - worst[eligible[1]])),
        "rule": rule,
    }


def combined_development_alphas(
    tuning: pd.DataFrame, ridge_feature_sets: list[str]
) -> dict[str, float]:
    selected: dict[str, float] = {}
    for feature_set in ridge_feature_sets:
        subset = tuning[tuning["feature_set"].eq(feature_set)]
        if subset.empty:
            raise RuntimeError(f"Missing Phase 3 tuning rows for {feature_set}")
        ranked = (
            subset.groupby("alpha", as_index=False)["validation_rmse"]
            .mean()
            .sort_values(["validation_rmse", "alpha"])
        )
        selected[feature_set] = float(ranked.iloc[0]["alpha"])
    return selected


def plot_transfer(summary: pd.DataFrame, output: Path) -> None:
    ordered = (
        summary.groupby("model", as_index=False)["rmse"].max().sort_values("rmse")
    )
    models = ordered["model"].tolist()
    figure, axis = plt.subplots(figsize=(9, 5.5))
    width = 0.36
    positions = np.arange(len(models))
    colours = ["#005f73", "#bb3e03"]
    for offset, (direction, group) in zip(
        [-width / 2, width / 2], summary.groupby("direction", sort=True), strict=True
    ):
        lookup = group.set_index("model")["rmse"]
        axis.bar(
            positions + offset,
            [lookup[model] for model in models],
            width,
            label=direction,
            color=colours.pop(0),
        )
    axis.set_xticks(positions, [model.replace("_", "\n") for model in models])
    axis.set_ylabel("Transfer RMSE")
    axis.set_title("SOAP/TEAK geographic transfer")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.png")
    figure.savefig(temporary, dpi=180, bbox_inches="tight")
    plt.close(figure)
    temporary.replace(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Replace the transfer freeze")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protected = [PREDICTIONS_PATH, SUMMARY_PATH, TUNING_PATH, FIGURE_PATH, MODEL_PATH, FREEZE_PATH]
    if any(path.exists() for path in protected) and not args.force:
        existing = [str(path.relative_to(ROOT)) for path in protected if path.exists()]
        raise RuntimeError(f"Development transfer freeze exists; refusing to overwrite: {existing}")

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    transfer = config["phase4_development_transfer"]
    phase3_models = config["phase3"]["models"]
    source = pd.read_parquet(SOURCE_PATH)
    folds = pd.read_parquet(FOLDS_PATH)
    conventional = pd.read_parquet(CONVENTIONAL_PATH)
    conventional_freeze = json.loads(CONVENTIONAL_FREEZE_PATH.read_text(encoding="utf-8"))
    conventional_features = conventional_freeze["freeze_basis"]["feature_columns"]
    frame = source.merge(
        folds[["site_id", "shot_number", "spatial_block_id", "spatial_fold_id"]],
        on=["site_id", "shot_number"],
        validate="one_to_one",
    ).merge(
        conventional[["site_id", "shot_number", *conventional_features]],
        on=["site_id", "shot_number"],
        validate="one_to_one",
    )
    if len(frame) != 2037 or set(frame["site_id"]) != {"SOAP", "TEAK"}:
        raise RuntimeError("Transfer input is not the exact frozen development population")

    terrain = list(config["phase3_extended_experiments"]["feature_sets"]["terrain"])
    area = [f"tessera_area_{index:03d}" for index in range(128)]
    feature_sets = {
        "terrain_ridge": terrain,
        "sentinel_topography_ridge": conventional_features,
        "tessera_area_ridge": area,
        "tessera_area_topography_ridge": [*area, *terrain],
    }
    for name, columns in feature_sets.items():
        if not np.isfinite(frame[columns].to_numpy(dtype=np.float64)).all():
            raise RuntimeError(f"Non-finite values in {name}")

    nonlinear_parameters = {
        key: phase3_models["tessera_hist_gradient_boosting"][key]
        for key in [
            "learning_rate",
            "max_iter",
            "max_leaf_nodes",
            "min_samples_leaf",
            "l2_regularization",
        ]
    }
    alphas = [float(value) for value in transfer["ridge_alpha_grid"]]
    prediction_frames: list[pd.DataFrame] = []
    tuning_records: list[dict[str, Any]] = []
    direction_alphas: dict[str, dict[str, float]] = {}
    summary_records: list[dict[str, Any]] = []
    for direction_number, definition in enumerate(transfer["directions"]):
        train_site = definition["train_site"]
        test_site = definition["test_site"]
        direction = f"{train_site}_to_{test_site}"
        train_mask = frame["site_id"].eq(train_site).to_numpy()
        test_mask = frame["site_id"].eq(test_site).to_numpy()
        train_indices = np.flatnonzero(train_mask)
        test_indices = np.flatnonzero(test_mask)
        direction_alphas[direction] = {}
        output = frame.iloc[test_indices][
            ["site_id", "shot_number", "spatial_block_id", "fhd_normal"]
        ].copy()
        output.insert(0, "direction", direction)
        output["prediction_training_mean"] = float(frame.iloc[train_indices]["fhd_normal"].mean())

        for feature_set, columns in feature_sets.items():
            selected_alpha, records = tune_within_source(
                frame,
                train_mask,
                columns,
                alphas,
                float(transfer["exclusion_buffer_m"]),
                direction,
                feature_set,
            )
            tuning_records.extend(records)
            direction_alphas[direction][feature_set] = selected_alpha
            model = ridge_pipeline(selected_alpha)
            model.fit(
                frame.iloc[train_indices][columns].to_numpy(dtype=np.float32),
                frame.iloc[train_indices]["fhd_normal"].to_numpy(dtype=np.float64),
            )
            output[f"prediction_{feature_set}"] = model.predict(
                frame.iloc[test_indices][columns].to_numpy(dtype=np.float32)
            )

        nonlinear = HistGradientBoostingRegressor(
            **nonlinear_parameters,
            random_state=int(config["phase3"]["uncertainty"]["random_seed"]),
        )
        nonlinear.fit(
            frame.iloc[train_indices][area].to_numpy(dtype=np.float32),
            frame.iloc[train_indices]["fhd_normal"].to_numpy(dtype=np.float64),
        )
        output["prediction_tessera_hist_gradient_boosting"] = nonlinear.predict(
            frame.iloc[test_indices][area].to_numpy(dtype=np.float32)
        )
        model_columns = {
            model: f"prediction_{model}" for model in transfer["models"]
        }
        intervals = block_bootstrap_intervals(
            output,
            model_columns,
            int(config["phase3"]["uncertainty"]["replicates"]),
            int(config["phase3"]["uncertainty"]["random_seed"]) + direction_number,
        )
        observed = output["fhd_normal"].to_numpy(dtype=np.float64)
        for model, column in model_columns.items():
            metrics = metric_values(observed, output[column].to_numpy(dtype=np.float64))
            record: dict[str, Any] = {
                "direction": direction,
                "train_site": train_site,
                "test_site": test_site,
                "training_rows": int(train_mask.sum()),
                "test_rows": int(test_mask.sum()),
                "test_spatial_blocks": int(output["spatial_block_id"].nunique()),
                "model": model,
                "selected_alpha": direction_alphas[direction].get(model, math.nan),
                "bootstrap_replicates": int(config["phase3"]["uncertainty"]["replicates"]),
            }
            for metric, value in metrics.items():
                low, high = intervals[(model, metric)]
                record[metric] = value
                record[f"{metric}_ci_low"] = low
                record[f"{metric}_ci_high"] = high
            summary_records.append(record)
        prediction_frames.append(output)
        print(f"completed {direction}: train={train_mask.sum():,}, test={test_mask.sum():,}")

    predictions = pd.concat(prediction_frames, ignore_index=True)
    summary = pd.DataFrame(summary_records)
    tuning = pd.DataFrame(tuning_records)
    selection = select_pipeline(summary, transfer["selected_pipeline_rule"])

    phase3_tuning = pd.read_csv(EXTENDED_TUNING_PATH)
    final_alphas = combined_development_alphas(phase3_tuning, list(feature_sets))
    selected_name = selection["selected_model"]
    selected_features = feature_sets[selected_name]
    selected_alpha = final_alphas[selected_name]
    selected_model = ridge_pipeline(selected_alpha)
    selected_model.fit(
        frame[selected_features].to_numpy(dtype=np.float32),
        frame["fhd_normal"].to_numpy(dtype=np.float64),
    )

    phase2_freeze = json.loads(PHASE2_FREEZE_PATH.read_text(encoding="utf-8"))
    fold_freeze = json.loads(FOLD_FREEZE_PATH.read_text(encoding="utf-8"))
    extended_freeze = json.loads(EXTENDED_FREEZE_PATH.read_text(encoding="utf-8"))
    freeze_basis = {
        "phase2_alignment_id": phase2_freeze["alignment_id"],
        "development_input_sha256": sha256(SOURCE_PATH),
        "spatial_fold_freeze_id": fold_freeze["freeze_id"],
        "conventional_predictor_freeze_id": conventional_freeze["freeze_id"],
        "phase3_extended_freeze_id": extended_freeze["freeze_id"],
        "project_config_sha256": sha256(CONFIG_PATH),
        "script_sha256": sha256(Path(__file__).resolve()),
        "helper_script_sha256": sha256(ROOT / "scripts/train_baselines.py"),
        "sklearn_version": sklearn.__version__,
        "protocol": transfer,
        "feature_sets": feature_sets,
        "nonlinear_parameters": nonlinear_parameters,
        "direction_selected_alphas": direction_alphas,
        "combined_development_alphas": final_alphas,
        "pipeline_selection": selection,
        "bart_file_read": False,
        "bart_target_metrics_computed": False,
    }
    freeze_basis_sha = canonical_hash(freeze_basis)
    freeze_id = f"phase4-transfer-{freeze_basis_sha[:12]}"
    predictions.insert(0, "phase4_transfer_freeze_id", freeze_id)

    write_parquet_atomic(predictions, PREDICTIONS_PATH)
    write_csv_atomic(summary, SUMMARY_PATH)
    write_csv_atomic(tuning, TUNING_PATH)
    plot_transfer(summary, FIGURE_PATH)
    joblib_dump_atomic(
        {
            "phase4_transfer_freeze_id": freeze_id,
            "model_name": selected_name,
            "features": selected_features,
            "alpha": selected_alpha,
            "training_sites": ["SOAP", "TEAK"],
            "training_rows": len(frame),
            "model": selected_model,
        },
        MODEL_PATH,
    )
    output_paths = {
        "predictions": PREDICTIONS_PATH,
        "summary": SUMMARY_PATH,
        "tuning": TUNING_PATH,
        "figure": FIGURE_PATH,
        "selected_model": MODEL_PATH,
    }
    manifest = {
        "freeze_id": freeze_id,
        "created_at": utc_now(),
        "status": "frozen_before_bart_predictor_extraction_or_target_access",
        "freeze_basis_sha256": freeze_basis_sha,
        "freeze_basis": freeze_basis,
        "selected_pipeline": {
            "model": selected_name,
            "features": selected_features,
            "alpha": selected_alpha,
        },
        "outputs": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for name, path in output_paths.items()
        },
        "transfer_summary": summary.to_dict(orient="records"),
    }
    json_dump(FREEZE_PATH, manifest)
    print(
        json.dumps(
            {
                "freeze_id": freeze_id,
                "selected_pipeline": manifest["selected_pipeline"],
                "selection": selection,
                "transfer_metrics": summary[
                    ["direction", "model", "r2", "rmse", "mae", "spearman_r", "mean_bias"]
                ].to_dict(orient="records"),
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
