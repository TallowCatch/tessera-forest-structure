#!/usr/bin/env python3
"""Run conventional baselines, alignment sensitivity, and label efficiency."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train_baselines import buffered_split, metric_values, ridge_inner_tuning


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/project.yaml"
SOURCE_PATH = ROOT / "data/processed/tessera_aligned_development_soap_teak.parquet"
CONVENTIONAL_PATH = ROOT / "data/processed/phase3_conventional_predictors_development.parquet"
FOLDS_PATH = ROOT / "metadata/phase3_spatial_folds.parquet"
CORE_PREDICTIONS_PATH = ROOT / "data/processed/phase3_oof_predictions.parquet"
PHASE2_FREEZE_PATH = ROOT / "metadata/tessera_phase2_alignment_freeze.json"
FOLD_FREEZE_PATH = ROOT / "metadata/phase3_spatial_fold_freeze.json"
CONVENTIONAL_FREEZE_PATH = ROOT / "metadata/phase3_conventional_predictor_freeze.json"
CORE_FREEZE_PATH = ROOT / "metadata/phase3_baseline_freeze.json"

PREDICTIONS_PATH = ROOT / "data/processed/phase3_extended_oof_predictions.parquet"
LABEL_PREDICTIONS_PATH = ROOT / "data/processed/phase3_label_efficiency_predictions.parquet"
SUMMARY_PATH = ROOT / "outputs/tables/phase3_extended_baseline_summary.csv"
TUNING_PATH = ROOT / "outputs/tables/phase3_extended_nested_tuning.csv"
ALIGNMENT_PATH = ROOT / "outputs/tables/phase3_alignment_sensitivity.csv"
LABEL_SUMMARY_PATH = ROOT / "outputs/tables/phase3_label_efficiency_summary.csv"
LABEL_REPLICATES_PATH = ROOT / "outputs/tables/phase3_label_efficiency_replicates.csv"
FIGURE_PATH = ROOT / "outputs/figures/phase3_extended_baselines.png"
LABEL_FIGURE_PATH = ROOT / "outputs/figures/phase3_label_efficiency.png"
FREEZE_PATH = ROOT / "metadata/phase3_extended_development_freeze.json"


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


def ridge_pipeline(alpha: float) -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha))])


def run_nested_ridge(
    frame: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    alpha_grid: list[float],
    buffer_m: float,
    number_of_folds: int,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict[str, list[float]]]:
    predictions: dict[str, np.ndarray] = {}
    tuning_records: list[dict[str, Any]] = []
    selected_alphas: dict[str, list[float]] = defaultdict(list)
    for feature_set, columns in feature_sets.items():
        output = np.full(len(frame), np.nan, dtype=np.float64)
        for outer_fold in range(number_of_folds):
            train_indices, test_indices, _, _ = buffered_split(frame, outer_fold, buffer_m)
            selected_alpha, records = ridge_inner_tuning(
                frame,
                columns,
                outer_fold,
                train_indices,
                alpha_grid,
                buffer_m,
            )
            for record in records:
                record["feature_set"] = feature_set
                record["selected_alpha_for_outer_fold"] = selected_alpha
            tuning_records.extend(records)
            model = ridge_pipeline(selected_alpha)
            model.fit(
                frame.iloc[train_indices][columns].to_numpy(dtype=np.float32),
                frame.iloc[train_indices]["fhd_normal"].to_numpy(dtype=np.float64),
            )
            output[test_indices] = model.predict(
                frame.iloc[test_indices][columns].to_numpy(dtype=np.float32)
            )
            selected_alphas[feature_set].append(selected_alpha)
        if not np.isfinite(output).all():
            raise RuntimeError(f"Incomplete out-of-fold predictions for {feature_set}")
        predictions[feature_set] = output
        print(f"completed nested Ridge feature set: {feature_set}")
    return predictions, pd.DataFrame(tuning_records), selected_alphas
def block_bootstrap_intervals(
    frame: pd.DataFrame,
    model_columns: dict[str, str],
    scopes: list[str],
    replicates: int,
    seed: int,
) -> dict[tuple[str, str, str], tuple[float, float]]:
    random = np.random.default_rng(seed)
    metrics = ["r2", "rmse", "mae", "spearman_r", "mean_bias"]
    distributions = {
        (model, scope, metric): []
        for model in model_columns
        for scope in scopes
        for metric in metrics
    }
    for scope in scopes:
        scoped = frame if scope == "ALL" else frame[frame["site_id"].eq(scope)]
        blocks = {
            site: {
                block: group.index.to_numpy()
                for block, group in site_frame.groupby("spatial_block_id")
            }
            for site, site_frame in scoped.groupby("site_id")
        }
        for _ in range(replicates):
            sampled: list[np.ndarray] = []
            for lookup in blocks.values():
                identifiers = sorted(lookup)
                selected = random.choice(identifiers, size=len(identifiers), replace=True)
                sampled.extend(lookup[identifier] for identifier in selected)
            sample = frame.loc[np.concatenate(sampled)]
            observed = sample["fhd_normal"].to_numpy(dtype=np.float64)
            for model, column in model_columns.items():
                values = metric_values(observed, sample[column].to_numpy(dtype=np.float64))
                for metric, value in values.items():
                    distributions[(model, scope, metric)].append(value)
    intervals = {}
    for key, values in distributions.items():
        numeric = np.asarray(values, dtype=np.float64)
        numeric = numeric[np.isfinite(numeric)]
        intervals[key] = (
            float(np.percentile(numeric, 2.5)),
            float(np.percentile(numeric, 97.5)),
        )
    return intervals


def build_summary(
    predictions: pd.DataFrame,
    model_columns: dict[str, str],
    bootstrap_replicates: int,
    seed: int,
) -> pd.DataFrame:
    scopes = ["ALL", "SOAP", "TEAK"]
    intervals = block_bootstrap_intervals(
        predictions, model_columns, scopes, bootstrap_replicates, seed
    )
    records: list[dict[str, Any]] = []
    for scope in scopes:
        scoped = predictions if scope == "ALL" else predictions[predictions["site_id"].eq(scope)]
        observed = scoped["fhd_normal"].to_numpy(dtype=np.float64)
        for model, column in model_columns.items():
            values = metric_values(observed, scoped[column].to_numpy(dtype=np.float64))
            record: dict[str, Any] = {
                "scope": scope,
                "model": model,
                "rows": len(scoped),
                "spatial_blocks": scoped["spatial_block_id"].nunique(),
                "bootstrap_replicates": bootstrap_replicates,
            }
            for metric, value in values.items():
                low, high = intervals[(model, scope, metric)]
                record[metric] = value
                record[f"{metric}_ci_low"] = low
                record[f"{metric}_ci_high"] = high
            records.append(record)
    return pd.DataFrame(records)


def run_label_efficiency(
    frame: pd.DataFrame,
    features: list[str],
    fractions: list[float],
    replicates_below_full: int,
    alpha: float,
    buffer_m: float,
    number_of_folds: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_records: list[pd.DataFrame] = []
    replicate_records: list[dict[str, Any]] = []
    for fraction in fractions:
        replicate_count = 1 if math.isclose(fraction, 1.0) else replicates_below_full
        for replicate in range(replicate_count):
            output = np.full(len(frame), np.nan, dtype=np.float64)
            training_rows: list[int] = []
            training_blocks: list[int] = []
            for outer_fold in range(number_of_folds):
                train_indices, test_indices, _, _ = buffered_split(frame, outer_fold, buffer_m)
                train = frame.iloc[train_indices]
                selected_blocks: list[str] = []
                for site in ("SOAP", "TEAK"):
                    available = sorted(train.loc[train["site_id"].eq(site), "spatial_block_id"].unique())
                    count = min(len(available), max(1, int(math.ceil(fraction * len(available)))))
                    random = np.random.default_rng(
                        seed
                        + int(round(fraction * 1000)) * 100_000
                        + replicate * 100
                        + outer_fold * 10
                        + (0 if site == "SOAP" else 1)
                    )
                    selected_blocks.extend(
                        sorted(random.choice(available, size=count, replace=False).tolist())
                    )
                selected_mask = train["spatial_block_id"].isin(selected_blocks).to_numpy()
                selected_indices = train_indices[np.flatnonzero(selected_mask)]
                model = ridge_pipeline(alpha)
                model.fit(
                    frame.iloc[selected_indices][features].to_numpy(dtype=np.float32),
                    frame.iloc[selected_indices]["fhd_normal"].to_numpy(dtype=np.float64),
                )
                output[test_indices] = model.predict(
                    frame.iloc[test_indices][features].to_numpy(dtype=np.float32)
                )
                training_rows.append(len(selected_indices))
                training_blocks.append(len(selected_blocks))
            if not np.isfinite(output).all():
                raise RuntimeError("Incomplete label-efficiency predictions")
            observed = frame["fhd_normal"].to_numpy(dtype=np.float64)
            replicate_records.append(
                {
                    "training_block_fraction": fraction,
                    "replicate": replicate,
                    "mean_training_rows_per_fold": float(np.mean(training_rows)),
                    "mean_training_blocks_per_fold": float(np.mean(training_blocks)),
                    **metric_values(observed, output),
                }
            )
            record = frame[
                ["site_id", "shot_number", "spatial_block_id", "spatial_fold_id", "fhd_normal"]
            ].copy()
            record["training_block_fraction"] = fraction
            record["replicate"] = replicate
            record["prediction"] = output
            prediction_records.append(record)
        print(f"completed label efficiency fraction: {fraction:g}")
    predictions = pd.concat(prediction_records, ignore_index=True)
    replicates = pd.DataFrame(replicate_records)
    summary = (
        replicates.groupby("training_block_fraction", as_index=False)
        .agg(
            replicates=("replicate", "size"),
            mean_training_rows_per_fold=("mean_training_rows_per_fold", "mean"),
            mean_training_blocks_per_fold=("mean_training_blocks_per_fold", "mean"),
            r2_mean=("r2", "mean"),
            r2_min=("r2", "min"),
            r2_max=("r2", "max"),
            rmse_mean=("rmse", "mean"),
            rmse_min=("rmse", "min"),
            rmse_max=("rmse", "max"),
            mae_mean=("mae", "mean"),
            spearman_r_mean=("spearman_r", "mean"),
        )
        .sort_values("training_block_fraction")
    )
    return predictions, replicates, summary


def plot_baselines(summary: pd.DataFrame, output: Path) -> None:
    overall = summary[summary["scope"].eq("ALL")].sort_values("rmse")
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].barh(overall["model"], overall["r2"], color="#2f6b3c")
    axes[0].axvline(0, color="#333333", linewidth=0.8)
    axes[0].set_xlabel("Spatial out-of-fold R2")
    axes[1].barh(overall["model"], overall["rmse"], color="#bb3e03")
    axes[1].set_xlabel("Spatial out-of-fold RMSE")
    for axis in axes:
        axis.grid(axis="x", alpha=0.2)
    figure.suptitle("SOAP/TEAK extended development baselines")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.png")
    figure.savefig(temporary, dpi=180, bbox_inches="tight")
    plt.close(figure)
    temporary.replace(output)


def plot_label_efficiency(summary: pd.DataFrame, output: Path) -> None:
    x = summary["training_block_fraction"] * 100
    figure, axis = plt.subplots(figsize=(6.5, 4.5))
    axis.plot(x, summary["r2_mean"], marker="o", color="#005f73")
    axis.fill_between(x, summary["r2_min"], summary["r2_max"], color="#94d2bd", alpha=0.45)
    axis.set_xlabel("Training spatial blocks retained (%)")
    axis.set_ylabel("Pooled spatial out-of-fold R2")
    axis.set_xticks(x)
    axis.grid(alpha=0.2)
    axis.set_title("TESSERA Ridge block-based label efficiency")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.png")
    figure.savefig(temporary, dpi=180, bbox_inches="tight")
    plt.close(figure)
    temporary.replace(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Replace the extended freeze")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protected = [
        PREDICTIONS_PATH,
        LABEL_PREDICTIONS_PATH,
        SUMMARY_PATH,
        TUNING_PATH,
        ALIGNMENT_PATH,
        LABEL_SUMMARY_PATH,
        LABEL_REPLICATES_PATH,
        FIGURE_PATH,
        LABEL_FIGURE_PATH,
        FREEZE_PATH,
    ]
    if any(path.exists() for path in protected) and not args.force:
        existing = [str(path.relative_to(ROOT)) for path in protected if path.exists()]
        raise RuntimeError(f"Extended development freeze exists; refusing to overwrite: {existing}")

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    experiment = config["phase3_extended_experiments"]
    source = pd.read_parquet(SOURCE_PATH)
    conventional = pd.read_parquet(CONVENTIONAL_PATH)
    folds = pd.read_parquet(FOLDS_PATH)
    core = pd.read_parquet(CORE_PREDICTIONS_PATH)
    frame = source.merge(
        folds[["site_id", "shot_number", "spatial_block_id", "spatial_fold_id"]],
        on=["site_id", "shot_number"],
        validate="one_to_one",
    )
    conventional_features = [
        column
        for column in conventional.columns
        if column
        not in {
            "conventional_predictor_freeze_id",
            "site_id",
            "shot_number",
            "tessera_alignment_id",
        }
    ]
    frame = frame.merge(
        conventional[["site_id", "shot_number", *conventional_features]],
        on=["site_id", "shot_number"],
        validate="one_to_one",
    )
    frame = frame.merge(
        core[
            [
                "site_id",
                "shot_number",
                "prediction_training_mean",
                "prediction_tessera_hist_gradient_boosting",
            ]
        ],
        on=["site_id", "shot_number"],
        validate="one_to_one",
    )
    if len(frame) != 2037 or set(frame["site_id"]) != {"SOAP", "TEAK"}:
        raise RuntimeError("Extended experiment input is not the frozen development set")

    terrain = list(experiment["feature_sets"]["terrain"])
    area = [f"tessera_area_{index:03d}" for index in range(128)]
    center = [f"tessera_center_{index:03d}" for index in range(128)]
    gaussian = [f"tessera_gaussian_{index:03d}" for index in range(128)]
    feature_sets = {
        "terrain_ridge": terrain,
        "sentinel_topography_ridge": conventional_features,
        "tessera_area_ridge": area,
        "tessera_area_topography_ridge": [*area, *terrain],
        "tessera_center_ridge": center,
        "tessera_gaussian_ridge": gaussian,
    }
    for columns in feature_sets.values():
        if not np.isfinite(frame[columns].to_numpy(dtype=np.float64)).all():
            raise RuntimeError("A feature set contains non-finite values")

    ridge_predictions, tuning, selected_alphas = run_nested_ridge(
        frame,
        feature_sets,
        [float(value) for value in experiment["ridge_alpha_grid"]],
        float(experiment["exclusion_buffer_m"]),
        5,
    )
    predictions = frame[
        [
            "site_id",
            "shot_number",
            "spatial_block_id",
            "spatial_fold_id",
            "fhd_normal",
            "prediction_training_mean",
            "prediction_tessera_hist_gradient_boosting",
        ]
    ].copy()
    model_columns = {
        "training_mean": "prediction_training_mean",
        "tessera_hist_gradient_boosting": "prediction_tessera_hist_gradient_boosting",
    }
    for model, values in ridge_predictions.items():
        column = f"prediction_{model}"
        predictions[column] = values
        model_columns[model] = column

    uncertainty = experiment["uncertainty"]
    summary = build_summary(
        predictions,
        model_columns,
        int(uncertainty["replicates"]),
        int(uncertainty["random_seed"]),
    )
    alignment = summary[
        summary["model"].isin(
            ["tessera_center_ridge", "tessera_area_ridge", "tessera_gaussian_ridge"]
        )
    ].copy()

    label_config = experiment["label_efficiency"]
    label_predictions, label_replicates, label_summary = run_label_efficiency(
        frame,
        area,
        [float(value) for value in label_config["training_block_fractions"]],
        int(label_config["replicates_below_full_fraction"]),
        float(label_config["fixed_alpha_from_phase3_nested_tuning"]),
        float(experiment["exclusion_buffer_m"]),
        5,
        int(label_config["random_seed"]),
    )

    phase2 = json.loads(PHASE2_FREEZE_PATH.read_text(encoding="utf-8"))
    fold_freeze = json.loads(FOLD_FREEZE_PATH.read_text(encoding="utf-8"))
    conventional_freeze = json.loads(CONVENTIONAL_FREEZE_PATH.read_text(encoding="utf-8"))
    core_freeze = json.loads(CORE_FREEZE_PATH.read_text(encoding="utf-8"))
    freeze_basis = {
        "phase2_alignment_id": phase2["alignment_id"],
        "spatial_fold_freeze_id": fold_freeze["freeze_id"],
        "conventional_predictor_freeze_id": conventional_freeze["freeze_id"],
        "core_baseline_freeze_id": core_freeze["freeze_id"],
        "project_config_sha256": sha256(CONFIG_PATH),
        "script_sha256": sha256(Path(__file__).resolve()),
        "helper_script_sha256": sha256(ROOT / "scripts/train_baselines.py"),
        "experiment": experiment,
        "feature_sets": feature_sets,
        "model_columns": model_columns,
        "selected_outer_alphas": selected_alphas,
        "bart_target_metrics_computed": False,
    }
    freeze_basis_sha = canonical_hash(freeze_basis)
    freeze_id = f"phase3-extended-{freeze_basis_sha[:12]}"
    predictions.insert(0, "phase3_extended_freeze_id", freeze_id)
    label_predictions.insert(0, "phase3_extended_freeze_id", freeze_id)

    write_parquet_atomic(predictions, PREDICTIONS_PATH)
    write_parquet_atomic(label_predictions, LABEL_PREDICTIONS_PATH)
    write_csv_atomic(summary, SUMMARY_PATH)
    write_csv_atomic(tuning, TUNING_PATH)
    write_csv_atomic(alignment, ALIGNMENT_PATH)
    write_csv_atomic(label_summary, LABEL_SUMMARY_PATH)
    write_csv_atomic(label_replicates, LABEL_REPLICATES_PATH)
    plot_baselines(summary, FIGURE_PATH)
    plot_label_efficiency(label_summary, LABEL_FIGURE_PATH)

    output_paths = {
        "oof_predictions": PREDICTIONS_PATH,
        "label_efficiency_predictions": LABEL_PREDICTIONS_PATH,
        "summary": SUMMARY_PATH,
        "nested_tuning": TUNING_PATH,
        "alignment_sensitivity": ALIGNMENT_PATH,
        "label_efficiency_summary": LABEL_SUMMARY_PATH,
        "label_efficiency_replicates": LABEL_REPLICATES_PATH,
        "baseline_figure": FIGURE_PATH,
        "label_efficiency_figure": LABEL_FIGURE_PATH,
    }
    manifest = {
        "freeze_id": freeze_id,
        "created_at": utc_now(),
        "status": "frozen",
        "freeze_basis_sha256": freeze_basis_sha,
        "freeze_basis": freeze_basis,
        "outputs": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
            }
            for name, path in output_paths.items()
        },
        "overall_summary": summary[summary["scope"].eq("ALL")].to_dict(orient="records"),
        "label_efficiency_summary": label_summary.to_dict(orient="records"),
    }
    json_dump(FREEZE_PATH, manifest)
    print(
        json.dumps(
            {
                "freeze_id": freeze_id,
                "overall": summary[summary["scope"].eq("ALL")][
                    ["model", "r2", "rmse", "mae", "spearman_r"]
                ].sort_values("rmse").to_dict(orient="records"),
                "label_efficiency": label_summary.to_dict(orient="records"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
