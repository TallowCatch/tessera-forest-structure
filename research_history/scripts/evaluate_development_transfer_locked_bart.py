#!/usr/bin/env python3
"""Run the one-time, predeclared evaluation on locked BART GEDI FHD targets."""

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


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/project.yaml"
DEVELOPMENT_PATH = ROOT / "data/processed/tessera_aligned_development_soap_teak.parquet"
BART_PATH = ROOT / "data/processed/tessera_aligned_locked_bart.parquet"
DEVELOPMENT_CONVENTIONAL_PATH = ROOT / "data/processed/phase3_conventional_predictors_development.parquet"
BART_CONVENTIONAL_PATH = ROOT / "data/processed/phase4_conventional_predictors_bart.parquet"
PHASE2_FREEZE_PATH = ROOT / "metadata/tessera_phase2_alignment_freeze.json"
DEVELOPMENT_CONVENTIONAL_FREEZE_PATH = ROOT / "metadata/phase3_conventional_predictor_freeze.json"
BART_PREDICTOR_FREEZE_PATH = ROOT / "metadata/phase4_bart_predictor_freeze.json"
TRANSFER_FREEZE_PATH = ROOT / "metadata/phase4_development_transfer_freeze.json"

PREDICTIONS_PATH = ROOT / "data/processed/phase4_locked_bart_predictions.parquet"
METRICS_PATH = ROOT / "outputs/tables/phase4_locked_bart_metrics.csv"
FIGURE_PATH = ROOT / "outputs/figures/phase4_locked_bart_evaluation.png"
FREEZE_PATH = ROOT / "metadata/phase4_locked_bart_evaluation_freeze.json"

METRIC_NAMES = ["r2", "rmse", "mae", "spearman_r", "mean_bias"]


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


def ridge_pipeline(alpha: float) -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha))])


def verify_checksum(path: Path, expected: str, label: str) -> None:
    observed = sha256(path)
    if observed != expected:
        raise RuntimeError(f"{label} checksum mismatch: {observed} != {expected}")


def assemble_inputs(include_bart_target: bool) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    phase2 = json.loads(PHASE2_FREEZE_PATH.read_text(encoding="utf-8"))
    development_predictors = json.loads(
        DEVELOPMENT_CONVENTIONAL_FREEZE_PATH.read_text(encoding="utf-8")
    )
    bart_predictors = json.loads(BART_PREDICTOR_FREEZE_PATH.read_text(encoding="utf-8"))
    transfer = json.loads(TRANSFER_FREEZE_PATH.read_text(encoding="utf-8"))

    verify_checksum(
        DEVELOPMENT_PATH,
        phase2["outputs"]["development"]["sha256"],
        "development alignment",
    )
    verify_checksum(
        DEVELOPMENT_CONVENTIONAL_PATH,
        development_predictors["outputs"]["predictors"]["sha256"],
        "development conventional predictors",
    )
    verify_checksum(
        BART_CONVENTIONAL_PATH,
        bart_predictors["outputs"]["predictors"]["sha256"],
        "BART conventional predictors",
    )
    model_path = ROOT / transfer["outputs"]["selected_model"]["path"]
    verify_checksum(model_path, transfer["outputs"]["selected_model"]["sha256"], "selected model")

    if bart_predictors["freeze_basis"]["target_columns_read"] != []:
        raise RuntimeError("BART predictor extraction accessed target columns")
    if bart_predictors["freeze_basis"]["recipe"] != development_predictors["freeze_basis"]["recipe"]:
        raise RuntimeError("BART and development conventional predictor recipes differ")
    if bart_predictors["freeze_basis"]["feature_columns"] != development_predictors["freeze_basis"]["feature_columns"]:
        raise RuntimeError("BART and development conventional feature columns differ")

    conventional_features = development_predictors["freeze_basis"]["feature_columns"]
    development = pd.read_parquet(DEVELOPMENT_PATH).merge(
        pd.read_parquet(DEVELOPMENT_CONVENTIONAL_PATH)[
            ["site_id", "shot_number", *conventional_features]
        ],
        on=["site_id", "shot_number"],
        validate="one_to_one",
    )
    area = [f"tessera_area_{index:03d}" for index in range(128)]
    bart_columns = [
        "site_id",
        "shot_number",
        "x_epsg5070",
        "y_epsg5070",
        *area,
    ]
    if include_bart_target:
        bart_columns.append("fhd_normal")
    bart = pd.read_parquet(BART_PATH, columns=bart_columns).merge(
        pd.read_parquet(BART_CONVENTIONAL_PATH)[
            ["site_id", "shot_number", *conventional_features]
        ],
        on=["site_id", "shot_number"],
        validate="one_to_one",
    )
    if len(development) != 2037 or set(development["site_id"]) != {"SOAP", "TEAK"}:
        raise RuntimeError("Development population is not the frozen SOAP/TEAK sample")
    if len(bart) != 1026 or set(bart["site_id"]) != {"BART"}:
        raise RuntimeError("BART population is not the frozen locked sample")
    overlap = set(development["shot_number"]).intersection(bart["shot_number"])
    if overlap:
        raise RuntimeError(f"Development/BART shot overlap detected: {len(overlap)}")
    return development, bart, {
        "phase2": phase2,
        "development_predictors": development_predictors,
        "bart_predictors": bart_predictors,
        "transfer": transfer,
        "conventional_features": conventional_features,
        "area_features": area,
        "selected_model_path": model_path,
    }


def build_predictions(
    development: pd.DataFrame,
    bart: pd.DataFrame,
    context: dict[str, Any],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    area = context["area_features"]
    conventional = context["conventional_features"]
    terrain = list(config["phase3_extended_experiments"]["feature_sets"]["terrain"])
    feature_sets = {
        "terrain_ridge": terrain,
        "sentinel_topography_ridge": conventional,
        "tessera_area_ridge": area,
        "tessera_area_topography_ridge": [*area, *terrain],
    }
    for name, features in feature_sets.items():
        if not np.isfinite(development[features].to_numpy(dtype=np.float64)).all():
            raise RuntimeError(f"Non-finite development features in {name}")
        if not np.isfinite(bart[features].to_numpy(dtype=np.float64)).all():
            raise RuntimeError(f"Non-finite BART features in {name}")

    transfer = context["transfer"]
    evaluation = config["phase4_locked_bart_evaluation"]
    if transfer["freeze_id"] != evaluation["development_transfer_freeze_id"]:
        raise RuntimeError("Configured transfer freeze does not match the frozen manifest")
    if context["bart_predictors"]["freeze_id"] != evaluation["bart_predictor_freeze_id"]:
        raise RuntimeError("Configured BART predictor freeze does not match the frozen manifest")
    selected = transfer["selected_pipeline"]
    configured_selected = evaluation["selected_primary_pipeline"]
    if selected["model"] != configured_selected["model"] or float(selected["alpha"]) != float(configured_selected["alpha"]):
        raise RuntimeError("Configured primary pipeline differs from the pre-BART selection")

    y_development = development["fhd_normal"].to_numpy(dtype=np.float64)
    output = bart[["site_id", "shot_number", "x_epsg5070", "y_epsg5070"]].copy()
    output["prediction_training_mean"] = float(y_development.mean())
    final_alphas = transfer["freeze_basis"]["combined_development_alphas"]
    selected_artifact = joblib.load(context["selected_model_path"])
    if (
        selected_artifact["model_name"] != selected["model"]
        or selected_artifact["features"] != selected["features"]
        or float(selected_artifact["alpha"]) != float(selected["alpha"])
    ):
        raise RuntimeError("Selected model artifact metadata differs from transfer freeze")

    for model_name, features in feature_sets.items():
        if model_name == selected["model"]:
            model = selected_artifact["model"]
        else:
            model = ridge_pipeline(float(final_alphas[model_name]))
            model.fit(
                development[features].to_numpy(dtype=np.float32),
                y_development,
            )
        output[f"prediction_{model_name}"] = model.predict(
            bart[features].to_numpy(dtype=np.float32)
        )

    nonlinear_config = config["phase3"]["models"]["tessera_hist_gradient_boosting"]
    nonlinear_parameters = {
        key: nonlinear_config[key]
        for key in [
            "learning_rate",
            "max_iter",
            "max_leaf_nodes",
            "min_samples_leaf",
            "l2_regularization",
        ]
    }
    nonlinear = HistGradientBoostingRegressor(
        **nonlinear_parameters,
        random_state=int(config["phase3"]["uncertainty"]["random_seed"]),
    )
    nonlinear.fit(development[area].to_numpy(dtype=np.float32), y_development)
    output["prediction_tessera_hist_gradient_boosting"] = nonlinear.predict(
        bart[area].to_numpy(dtype=np.float32)
    )
    expected_models = evaluation["comparison_models"]
    if {f"prediction_{model}" for model in expected_models} != {
        column for column in output if column.startswith("prediction_")
    }:
        raise RuntimeError("Prediction columns do not match the predeclared model list")
    return output, feature_sets


def block_bootstrap_intervals(
    frame: pd.DataFrame,
    model_columns: dict[str, str],
    replicates: int,
    seed: int,
) -> dict[tuple[str, str], tuple[float, float]]:
    random = np.random.default_rng(seed)
    blocks = {
        block: group.index.to_numpy()
        for block, group in frame.groupby("spatial_block_id")
    }
    block_ids = sorted(blocks)
    distributions = {
        (model, metric): [] for model in model_columns for metric in METRIC_NAMES
    }
    for _ in range(replicates):
        selected = random.choice(block_ids, size=len(block_ids), replace=True)
        sample = frame.loc[np.concatenate([blocks[block] for block in selected])]
        observed = sample["fhd_normal"].to_numpy(dtype=np.float64)
        for model, column in model_columns.items():
            metrics = metric_values(observed, sample[column].to_numpy(dtype=np.float64))
            for metric, value in metrics.items():
                distributions[(model, metric)].append(value)
    intervals: dict[tuple[str, str], tuple[float, float]] = {}
    for key, samples in distributions.items():
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


def evaluate(
    predictions: pd.DataFrame,
    bart: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    evaluation = config["phase4_locked_bart_evaluation"]
    result = predictions.merge(
        bart[["site_id", "shot_number", "fhd_normal"]],
        on=["site_id", "shot_number"],
        validate="one_to_one",
    )
    block_width = float(evaluation["uncertainty"]["spatial_block_width_m"])
    block_x = np.floor(result["x_epsg5070"].to_numpy() / block_width).astype(int)
    block_y = np.floor(result["y_epsg5070"].to_numpy() / block_width).astype(int)
    result["spatial_block_id"] = [
        f"BART_{x}_{y}" for x, y in zip(block_x, block_y, strict=True)
    ]
    if result["spatial_block_id"].nunique() < 2:
        raise RuntimeError("BART spatial block bootstrap requires at least two blocks")
    model_columns = {
        model: f"prediction_{model}" for model in evaluation["comparison_models"]
    }
    intervals = block_bootstrap_intervals(
        result,
        model_columns,
        int(evaluation["uncertainty"]["replicates"]),
        int(evaluation["uncertainty"]["random_seed"]),
    )
    observed = result["fhd_normal"].to_numpy(dtype=np.float64)
    records: list[dict[str, Any]] = []
    for model, column in model_columns.items():
        metrics = metric_values(observed, result[column].to_numpy(dtype=np.float64))
        record: dict[str, Any] = {
            "site_id": "BART",
            "model": model,
            "is_preselected_primary": model == evaluation["selected_primary_pipeline"]["model"],
            "rows": len(result),
            "spatial_blocks": result["spatial_block_id"].nunique(),
            "bootstrap_replicates": int(evaluation["uncertainty"]["replicates"]),
        }
        for metric, value in metrics.items():
            low, high = intervals[(model, metric)]
            record[metric] = value
            record[f"{metric}_ci_low"] = low
            record[f"{metric}_ci_high"] = high
        records.append(record)
    return result, pd.DataFrame(records)


def plot_evaluation(predictions: pd.DataFrame, metrics: pd.DataFrame, output: Path) -> None:
    models = metrics.sort_values("rmse")["model"].tolist()
    low = float(predictions["fhd_normal"].min())
    high = float(predictions["fhd_normal"].max())
    figure, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True)
    for axis, model in zip(axes.flat, models, strict=True):
        row = metrics[metrics["model"].eq(model)].iloc[0]
        axis.scatter(
            predictions["fhd_normal"],
            predictions[f"prediction_{model}"],
            s=8,
            alpha=0.35,
            color="#005f73" if row["is_preselected_primary"] else "#6c757d",
        )
        axis.plot([low, high], [low, high], linestyle="--", linewidth=1, color="#222222")
        axis.set_title(model.replace("_", " "))
        axis.text(
            0.04,
            0.96,
            f"R2 = {row['r2']:.3f}\nRMSE = {row['rmse']:.3f}",
            transform=axis.transAxes,
            va="top",
            fontsize=9,
        )
        axis.grid(alpha=0.2)
    for axis in axes[-1]:
        axis.set_xlabel("Observed GEDI FHD")
    for axis in axes[:, 0]:
        axis.set_ylabel("Prediction")
    figure.suptitle("Locked BART geographic-transfer evaluation")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.png")
    figure.savefig(temporary, dpi=180, bbox_inches="tight")
    plt.close(figure)
    temporary.replace(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Build predictions without reading BART fhd_normal or writing evaluation outputs",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protected = [PREDICTIONS_PATH, METRICS_PATH, FIGURE_PATH, FREEZE_PATH]
    if not args.preflight and any(path.exists() for path in protected):
        existing = [str(path.relative_to(ROOT)) for path in protected if path.exists()]
        raise RuntimeError(f"Locked BART evaluation already exists; refusing to rerun: {existing}")

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    development, bart, context = assemble_inputs(include_bart_target=not args.preflight)
    predictions, feature_sets = build_predictions(development, bart, context, config)
    if args.preflight:
        print(
            json.dumps(
                {
                    "status": "target_free_preflight_passed",
                    "bart_target_loaded": False,
                    "development_rows": len(development),
                    "bart_prediction_rows": len(predictions),
                    "prediction_columns": [
                        column for column in predictions if column.startswith("prediction_")
                    ],
                    "selected_pipeline": context["transfer"]["selected_pipeline"],
                },
                indent=2,
            )
        )
        return 0

    evaluated, metrics = evaluate(predictions, bart, config)
    transfer = context["transfer"]
    predictor_freeze = context["bart_predictors"]
    freeze_basis = {
        "phase2_alignment_id": context["phase2"]["alignment_id"],
        "development_input_sha256": sha256(DEVELOPMENT_PATH),
        "locked_bart_input_sha256": sha256(BART_PATH),
        "development_conventional_predictor_freeze_id": context["development_predictors"]["freeze_id"],
        "bart_predictor_freeze_id": predictor_freeze["freeze_id"],
        "development_transfer_freeze_id": transfer["freeze_id"],
        "selected_pipeline": transfer["selected_pipeline"],
        "project_config_sha256": sha256(CONFIG_PATH),
        "script_sha256": sha256(Path(__file__).resolve()),
        "sklearn_version": sklearn.__version__,
        "protocol": config["phase4_locked_bart_evaluation"],
        "feature_sets": feature_sets,
        "ridge_alphas": transfer["freeze_basis"]["combined_development_alphas"],
        "bart_predictor_target_columns_read": predictor_freeze["freeze_basis"]["target_columns_read"],
        "bart_tuning_or_model_selection_performed": False,
        "target_access_mode": "single_freeze_producing_evaluation_run_after_target_free_preflight",
    }
    freeze_basis_sha = canonical_hash(freeze_basis)
    freeze_id = f"phase4-bart-eval-{freeze_basis_sha[:12]}"
    evaluated.insert(0, "phase4_bart_evaluation_freeze_id", freeze_id)
    write_parquet_atomic(evaluated, PREDICTIONS_PATH)
    write_csv_atomic(metrics, METRICS_PATH)
    plot_evaluation(evaluated, metrics, FIGURE_PATH)
    outputs = {
        "predictions": PREDICTIONS_PATH,
        "metrics": METRICS_PATH,
        "figure": FIGURE_PATH,
    }
    manifest = {
        "freeze_id": freeze_id,
        "created_at": utc_now(),
        "status": "frozen_one_time_locked_bart_evaluation",
        "freeze_basis_sha256": freeze_basis_sha,
        "freeze_basis": freeze_basis,
        "outputs": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for name, path in outputs.items()
        },
        "metrics": metrics.to_dict(orient="records"),
    }
    json_dump(FREEZE_PATH, manifest)
    print(
        json.dumps(
            {
                "freeze_id": freeze_id,
                "preselected_primary": transfer["selected_pipeline"],
                "bart_rows": len(evaluated),
                "bart_spatial_blocks": evaluated["spatial_block_id"].nunique(),
                "metrics": metrics[
                    [
                        "model",
                        "is_preselected_primary",
                        "r2",
                        "rmse",
                        "mae",
                        "spearman_r",
                        "mean_bias",
                    ]
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
