#!/usr/bin/env python3
"""Train Phase 3 SOAP/TEAK baselines with frozen buffered spatial folds."""

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
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/project.yaml"
DEVELOPMENT_PATH = ROOT / "data/processed/tessera_aligned_development_soap_teak.parquet"
PHASE2_FREEZE_PATH = ROOT / "metadata/tessera_phase2_alignment_freeze.json"
FOLD_FREEZE_PATH = ROOT / "metadata/phase3_spatial_fold_freeze.json"

PREDICTIONS_PATH = ROOT / "data/processed/phase3_oof_predictions.parquet"
FOLD_METRICS_PATH = ROOT / "outputs/tables/phase3_baseline_fold_metrics.csv"
SUMMARY_PATH = ROOT / "outputs/tables/phase3_baseline_summary.csv"
TUNING_PATH = ROOT / "outputs/tables/phase3_ridge_nested_tuning.csv"
DIAGNOSTICS_PATH = ROOT / "outputs/tables/phase3_fold_diagnostics.csv"
FIGURE_PATH = ROOT / "outputs/figures/phase3_oof_baselines.png"
MEAN_MODEL_PATH = ROOT / "outputs/models/phase3_training_mean.json"
RIDGE_MODEL_PATH = ROOT / "outputs/models/phase3_tessera_ridge.joblib"
NONLINEAR_MODEL_PATH = ROOT / "outputs/models/phase3_tessera_hist_gradient_boosting.joblib"
FREEZE_PATH = ROOT / "metadata/phase3_baseline_freeze.json"

MODEL_COLUMNS = {
    "training_mean": "prediction_training_mean",
    "tessera_ridge": "prediction_tessera_ridge",
    "tessera_hist_gradient_boosting": "prediction_tessera_hist_gradient_boosting",
}
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


def joblib_dump_atomic(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(value, temporary, compress=3)
    temporary.replace(path)


def buffered_split(
    frame: pd.DataFrame,
    test_fold: int,
    buffer_m: float,
    population: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    if population is None:
        population = np.ones(len(frame), dtype=bool)
    test_mask = population & frame["spatial_fold_id"].eq(test_fold).to_numpy()
    candidate_mask = population & ~frame["spatial_fold_id"].eq(test_fold).to_numpy()
    if not test_mask.any() or not candidate_mask.any():
        raise RuntimeError(f"Empty train or test partition for fold {test_fold}")

    keep = candidate_mask.copy()
    minimum_retained_distance = math.inf
    for site in sorted(frame.loc[test_mask, "site_id"].unique()):
        test_site = test_mask & frame["site_id"].eq(site).to_numpy()
        candidate_site = candidate_mask & frame["site_id"].eq(site).to_numpy()
        candidate_indices = np.flatnonzero(candidate_site)
        if not test_site.any() or len(candidate_indices) == 0:
            continue
        tree = cKDTree(frame.loc[test_site, ["x_epsg5070", "y_epsg5070"]].to_numpy())
        distances, _ = tree.query(
            frame.iloc[candidate_indices][["x_epsg5070", "y_epsg5070"]].to_numpy(),
            k=1,
        )
        retained = distances > buffer_m
        keep[candidate_indices[~retained]] = False
        if retained.any():
            minimum_retained_distance = min(
                minimum_retained_distance, float(distances[retained].min())
            )
    removed = int(candidate_mask.sum() - keep.sum())
    return np.flatnonzero(keep), np.flatnonzero(test_mask), removed, minimum_retained_distance


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


def ridge_inner_tuning(
    frame: pd.DataFrame,
    features: list[str],
    outer_fold: int,
    outer_train_indices: np.ndarray,
    alphas: list[float],
    buffer_m: float,
) -> tuple[float, list[dict[str, Any]]]:
    population = np.zeros(len(frame), dtype=bool)
    population[outer_train_indices] = True
    inner_folds = sorted(frame.loc[population, "spatial_fold_id"].unique())
    records: list[dict[str, Any]] = []
    for alpha in alphas:
        for inner_fold in inner_folds:
            train_indices, validation_indices, removed, minimum_distance = buffered_split(
                frame, int(inner_fold), buffer_m, population
            )
            model = Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("ridge", Ridge(alpha=alpha)),
                ]
            )
            model.fit(
                frame.iloc[train_indices][features].to_numpy(dtype=np.float32),
                frame.iloc[train_indices]["fhd_normal"].to_numpy(dtype=np.float64),
            )
            prediction = model.predict(
                frame.iloc[validation_indices][features].to_numpy(dtype=np.float32)
            )
            records.append(
                {
                    "outer_fold": outer_fold,
                    "inner_validation_fold": int(inner_fold),
                    "alpha": alpha,
                    "training_rows": len(train_indices),
                    "validation_rows": len(validation_indices),
                    "buffer_removed_rows": removed,
                    "minimum_train_validation_distance_m": minimum_distance,
                    "validation_rmse": float(
                        np.sqrt(
                            np.mean(
                                (
                                    prediction
                                    - frame.iloc[validation_indices]["fhd_normal"].to_numpy()
                                )
                                ** 2
                            )
                        )
                    ),
                }
            )
    tuning = pd.DataFrame(records)
    ranked = (
        tuning.groupby("alpha", as_index=False)["validation_rmse"]
        .mean()
        .sort_values(["validation_rmse", "alpha"])
    )
    return float(ranked.iloc[0]["alpha"]), records


def spatial_block_bootstrap(
    frame: pd.DataFrame,
    prediction_columns: dict[str, str],
    scopes: list[str],
    replicates: int,
    seed: int,
) -> dict[tuple[str, str, str], tuple[float, float]]:
    random = np.random.default_rng(seed)
    distributions: dict[tuple[str, str, str], list[float]] = {
        (model, scope, metric): []
        for model in prediction_columns
        for scope in scopes
        for metric in METRIC_NAMES
    }
    for scope in scopes:
        scoped = frame if scope == "ALL" else frame[frame["site_id"].eq(scope)]
        grouped = {
            site: {
                block: group.index.to_numpy()
                for block, group in site_frame.groupby("spatial_block_id")
            }
            for site, site_frame in scoped.groupby("site_id")
        }
        for _ in range(replicates):
            sampled_indices: list[np.ndarray] = []
            for block_lookup in grouped.values():
                block_ids = sorted(block_lookup)
                sampled_blocks = random.choice(block_ids, size=len(block_ids), replace=True)
                sampled_indices.extend(block_lookup[block] for block in sampled_blocks)
            sample = frame.loc[np.concatenate(sampled_indices)]
            observed = sample["fhd_normal"].to_numpy(dtype=np.float64)
            for model, column in prediction_columns.items():
                values = metric_values(observed, sample[column].to_numpy(dtype=np.float64))
                for metric, value in values.items():
                    distributions[(model, scope, metric)].append(value)

    intervals: dict[tuple[str, str, str], tuple[float, float]] = {}
    for key, values in distributions.items():
        numeric = np.asarray(values, dtype=np.float64)
        numeric = numeric[np.isfinite(numeric)]
        intervals[key] = (
            float(np.percentile(numeric, 2.5)),
            float(np.percentile(numeric, 97.5)),
        )
    return intervals


def build_metric_tables(
    predictions: pd.DataFrame,
    bootstrap_replicates: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scopes = ["ALL", "SOAP", "TEAK"]
    fold_records: list[dict[str, Any]] = []
    for fold_id in sorted(predictions["spatial_fold_id"].unique()):
        fold = predictions[predictions["spatial_fold_id"].eq(fold_id)]
        for scope in scopes:
            scoped = fold if scope == "ALL" else fold[fold["site_id"].eq(scope)]
            observed = scoped["fhd_normal"].to_numpy(dtype=np.float64)
            for model, column in MODEL_COLUMNS.items():
                fold_records.append(
                    {
                        "spatial_fold_id": int(fold_id),
                        "scope": scope,
                        "model": model,
                        "rows": len(scoped),
                        **metric_values(
                            observed, scoped[column].to_numpy(dtype=np.float64)
                        ),
                    }
                )
    fold_metrics = pd.DataFrame(fold_records)

    intervals = spatial_block_bootstrap(
        predictions,
        MODEL_COLUMNS,
        scopes,
        bootstrap_replicates,
        random_seed,
    )
    summary_records: list[dict[str, Any]] = []
    for scope in scopes:
        scoped = predictions if scope == "ALL" else predictions[predictions["site_id"].eq(scope)]
        observed = scoped["fhd_normal"].to_numpy(dtype=np.float64)
        for model, column in MODEL_COLUMNS.items():
            metrics = metric_values(observed, scoped[column].to_numpy(dtype=np.float64))
            record: dict[str, Any] = {
                "scope": scope,
                "model": model,
                "rows": len(scoped),
                "spatial_blocks": scoped["spatial_block_id"].nunique(),
                "bootstrap_replicates": bootstrap_replicates,
            }
            for metric, value in metrics.items():
                low, high = intervals[(model, scope, metric)]
                record[metric] = value
                record[f"{metric}_ci_low"] = low
                record[f"{metric}_ci_high"] = high
            summary_records.append(record)
    return fold_metrics, pd.DataFrame(summary_records)


def plot_predictions(predictions: pd.DataFrame, summary: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, len(MODEL_COLUMNS), figsize=(14, 4.3), sharex=True, sharey=True)
    low = float(predictions["fhd_normal"].min())
    high = float(predictions["fhd_normal"].max())
    colours = {"SOAP": "#005f73", "TEAK": "#bb3e03"}
    for axis, (model, column) in zip(axes, MODEL_COLUMNS.items(), strict=True):
        for site, site_frame in predictions.groupby("site_id"):
            axis.scatter(
                site_frame["fhd_normal"],
                site_frame[column],
                s=8,
                alpha=0.38,
                color=colours[site],
                label=site,
            )
        axis.plot([low, high], [low, high], color="#333333", linestyle="--", linewidth=1)
        row = summary[summary["scope"].eq("ALL") & summary["model"].eq(model)].iloc[0]
        axis.set_title(model.replace("_", " "))
        axis.text(
            0.04,
            0.96,
            f"R2 = {row['r2']:.3f}\nRMSE = {row['rmse']:.3f}",
            transform=axis.transAxes,
            va="top",
            fontsize=9,
        )
        axis.set_xlabel("Observed GEDI FHD")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Out-of-fold prediction")
    axes[-1].legend(frameon=False)
    figure.suptitle("SOAP/TEAK buffered spatial out-of-fold baselines")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.png")
    figure.savefig(temporary, dpi=180, bbox_inches="tight")
    plt.close(figure)
    temporary.replace(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Replace the Phase 3 baseline freeze")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protected = [
        PREDICTIONS_PATH,
        FOLD_METRICS_PATH,
        SUMMARY_PATH,
        TUNING_PATH,
        DIAGNOSTICS_PATH,
        MEAN_MODEL_PATH,
        RIDGE_MODEL_PATH,
        NONLINEAR_MODEL_PATH,
        FREEZE_PATH,
    ]
    if any(path.exists() for path in protected) and not args.force:
        existing = [str(path.relative_to(ROOT)) for path in protected if path.exists()]
        raise RuntimeError(f"Phase 3 baseline freeze exists; refusing to overwrite: {existing}")

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    phase3 = config["phase3"]
    folds_config = phase3["folds"]
    models_config = phase3["models"]
    uncertainty_config = phase3["uncertainty"]
    sites = list(phase3["development_sites"])
    feature_columns = [f"{phase3['feature_prefix']}{index:03d}" for index in range(128)]

    source = pd.read_parquet(DEVELOPMENT_PATH)
    fold_freeze = json.loads(FOLD_FREEZE_PATH.read_text(encoding="utf-8"))
    folds_path = ROOT / fold_freeze["outputs"]["folds"]["path"]
    if sha256(folds_path) != fold_freeze["outputs"]["folds"]["sha256"]:
        raise RuntimeError("Frozen fold table checksum mismatch")
    folds = pd.read_parquet(folds_path)
    frame = source.merge(
        folds[
            [
                "site_id",
                "shot_number",
                "spatial_fold_freeze_id",
                "spatial_block_id",
                "spatial_block_x",
                "spatial_block_y",
                "spatial_fold_id",
            ]
        ],
        on=["site_id", "shot_number"],
        how="inner",
        validate="one_to_one",
    )
    if len(frame) != len(source) or set(frame["site_id"]) != set(sites):
        raise RuntimeError("Frozen folds do not cover the exact SOAP/TEAK development input")
    if any(column not in frame for column in feature_columns):
        raise RuntimeError("Primary 128-dimensional area-weighted TESSERA features are incomplete")
    if frame[feature_columns].isna().any().any():
        raise RuntimeError("Primary TESSERA features contain missing values")

    alpha_grid = [float(value) for value in models_config["tessera_ridge"]["alpha_grid"]]
    nonlinear_parameters = {
        key: models_config["tessera_hist_gradient_boosting"][key]
        for key in [
            "learning_rate",
            "max_iter",
            "max_leaf_nodes",
            "min_samples_leaf",
            "l2_regularization",
        ]
    }
    buffer_m = float(folds_config["exclusion_buffer_m"])
    random_seed = int(uncertainty_config["random_seed"])
    prediction_arrays = {
        column: np.full(len(frame), np.nan, dtype=np.float64)
        for column in MODEL_COLUMNS.values()
    }
    outer_alpha = np.full(len(frame), np.nan, dtype=np.float64)
    tuning_records: list[dict[str, Any]] = []
    diagnostic_records: list[dict[str, Any]] = []

    for outer_fold in range(int(folds_config["number_of_folds"])):
        train_indices, test_indices, removed, minimum_distance = buffered_split(
            frame, outer_fold, buffer_m
        )
        selected_alpha, records = ridge_inner_tuning(
            frame,
            feature_columns,
            outer_fold,
            train_indices,
            alpha_grid,
            buffer_m,
        )
        for record in records:
            record["selected_alpha_for_outer_fold"] = selected_alpha
        tuning_records.extend(records)

        x_train = frame.iloc[train_indices][feature_columns].to_numpy(dtype=np.float32)
        y_train = frame.iloc[train_indices]["fhd_normal"].to_numpy(dtype=np.float64)
        x_test = frame.iloc[test_indices][feature_columns].to_numpy(dtype=np.float32)

        prediction_arrays[MODEL_COLUMNS["training_mean"]][test_indices] = float(y_train.mean())
        ridge = Pipeline(
            [
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=selected_alpha)),
            ]
        )
        ridge.fit(x_train, y_train)
        prediction_arrays[MODEL_COLUMNS["tessera_ridge"]][test_indices] = ridge.predict(x_test)

        nonlinear = HistGradientBoostingRegressor(
            **nonlinear_parameters,
            random_state=random_seed,
        )
        nonlinear.fit(x_train, y_train)
        prediction_arrays[MODEL_COLUMNS["tessera_hist_gradient_boosting"]][test_indices] = (
            nonlinear.predict(x_test)
        )
        outer_alpha[test_indices] = selected_alpha
        diagnostic_records.append(
            {
                "spatial_fold_id": outer_fold,
                "training_rows_after_buffer": len(train_indices),
                "test_rows": len(test_indices),
                "buffer_removed_rows": removed,
                "minimum_retained_train_test_distance_m": minimum_distance,
                "selected_ridge_alpha": selected_alpha,
                "test_soap_rows": int(frame.iloc[test_indices]["site_id"].eq("SOAP").sum()),
                "test_teak_rows": int(frame.iloc[test_indices]["site_id"].eq("TEAK").sum()),
            }
        )
        print(
            f"outer fold {outer_fold}: train={len(train_indices):,}, "
            f"test={len(test_indices):,}, ridge alpha={selected_alpha:g}"
        )

    if any(not np.isfinite(values).all() for values in prediction_arrays.values()):
        raise RuntimeError("One or more development rows lack out-of-fold predictions")

    predictions = frame[
        [
            "site_id",
            "shot_number",
            "tessera_alignment_id",
            "spatial_fold_freeze_id",
            "spatial_block_id",
            "spatial_fold_id",
            "x_epsg5070",
            "y_epsg5070",
            "fhd_normal",
        ]
    ].copy()
    for column, values in prediction_arrays.items():
        predictions[column] = values
    predictions["outer_selected_ridge_alpha"] = outer_alpha

    fold_metrics, summary = build_metric_tables(
        predictions,
        int(uncertainty_config["replicates"]),
        random_seed,
    )
    tuning = pd.DataFrame(tuning_records)
    diagnostics = pd.DataFrame(diagnostic_records)
    aggregate_tuning = (
        tuning.groupby("alpha", as_index=False)["validation_rmse"]
        .mean()
        .sort_values(["validation_rmse", "alpha"])
    )
    final_alpha = float(aggregate_tuning.iloc[0]["alpha"])

    phase2_freeze = json.loads(PHASE2_FREEZE_PATH.read_text(encoding="utf-8"))
    freeze_basis = {
        "phase2_alignment_id": phase2_freeze["alignment_id"],
        "development_input_sha256": phase2_freeze["outputs"]["development"]["sha256"],
        "locked_bart_sha256_unchanged": phase2_freeze["outputs"]["locked_bart"]["sha256"],
        "spatial_fold_freeze_id": fold_freeze["freeze_id"],
        "spatial_fold_sha256": fold_freeze["outputs"]["folds"]["sha256"],
        "project_config_sha256": sha256(CONFIG_PATH),
        "training_script_sha256": sha256(Path(__file__).resolve()),
        "sklearn_version": sklearn.__version__,
        "sites": sites,
        "target": phase3["target"],
        "features": feature_columns,
        "outer_validation": {
            "folds": folds_config["number_of_folds"],
            "buffer_m": buffer_m,
        },
        "models": models_config,
        "final_ridge_alpha": final_alpha,
        "uncertainty": uncertainty_config,
        "bart_target_metrics_computed": False,
    }
    freeze_basis_sha = canonical_hash(freeze_basis)
    baseline_id = f"phase3-baselines-{freeze_basis_sha[:12]}"
    predictions.insert(0, "phase3_baseline_freeze_id", baseline_id)

    full_x = frame[feature_columns].to_numpy(dtype=np.float32)
    full_y = frame["fhd_normal"].to_numpy(dtype=np.float64)
    final_ridge = Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=final_alpha)),
        ]
    )
    final_ridge.fit(full_x, full_y)
    final_nonlinear = HistGradientBoostingRegressor(
        **nonlinear_parameters,
        random_state=random_seed,
    )
    final_nonlinear.fit(full_x, full_y)

    write_parquet_atomic(predictions, PREDICTIONS_PATH)
    write_csv_atomic(fold_metrics, FOLD_METRICS_PATH)
    write_csv_atomic(summary, SUMMARY_PATH)
    write_csv_atomic(tuning, TUNING_PATH)
    write_csv_atomic(diagnostics, DIAGNOSTICS_PATH)
    plot_predictions(predictions, summary, FIGURE_PATH)
    json_dump(
        MEAN_MODEL_PATH,
        {
            "phase3_baseline_freeze_id": baseline_id,
            "model": "training_mean",
            "training_rows": len(frame),
            "prediction": float(full_y.mean()),
        },
    )
    joblib_dump_atomic(
        {
            "phase3_baseline_freeze_id": baseline_id,
            "model": "tessera_ridge",
            "feature_columns": feature_columns,
            "pipeline": final_ridge,
        },
        RIDGE_MODEL_PATH,
    )
    joblib_dump_atomic(
        {
            "phase3_baseline_freeze_id": baseline_id,
            "model": "tessera_hist_gradient_boosting",
            "feature_columns": feature_columns,
            "estimator": final_nonlinear,
        },
        NONLINEAR_MODEL_PATH,
    )

    output_paths = {
        "oof_predictions": PREDICTIONS_PATH,
        "fold_metrics": FOLD_METRICS_PATH,
        "summary": SUMMARY_PATH,
        "ridge_tuning": TUNING_PATH,
        "fold_diagnostics": DIAGNOSTICS_PATH,
        "figure": FIGURE_PATH,
        "training_mean_model": MEAN_MODEL_PATH,
        "ridge_model": RIDGE_MODEL_PATH,
        "nonlinear_model": NONLINEAR_MODEL_PATH,
    }
    outputs = {
        name: {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256(path),
            **({"rows": len(predictions)} if name == "oof_predictions" else {}),
        }
        for name, path in output_paths.items()
    }
    manifest = {
        "freeze_id": baseline_id,
        "created_at": utc_now(),
        "status": "frozen",
        "freeze_basis_sha256": freeze_basis_sha,
        "freeze_basis": freeze_basis,
        "outputs": outputs,
        "outer_selected_ridge_alphas": {
            str(int(row.spatial_fold_id)): float(row.selected_ridge_alpha)
            for row in diagnostics.itertuples(index=False)
        },
        "summary": summary.to_dict(orient="records"),
    }
    json_dump(FREEZE_PATH, manifest)
    print(
        json.dumps(
            {
                "freeze_id": baseline_id,
                "development_rows": len(frame),
                "final_ridge_alpha": final_alpha,
                "overall_results": summary[summary["scope"].eq("ALL")][
                    ["model", "r2", "rmse", "mae", "spearman_r", "mean_bias"]
                ].to_dict(orient="records"),
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
