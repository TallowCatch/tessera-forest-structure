#!/usr/bin/env python3
"""Run frozen-hyperparameter, leave-one-pass-out validation within BART."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
BART_PATH = ROOT / "data/processed/tessera_aligned_locked_bart.parquet"
CONVENTIONAL_PATH = ROOT / "data/processed/phase4_conventional_predictors_bart.parquet"
TRANSFER_FREEZE_PATH = ROOT / "metadata/phase4_development_transfer_freeze.json"
SENTINEL_FREEZE_PATH = ROOT / "metadata/phase4_corrected_sentinel_sensitivity_freeze.json"
LOCKED_FREEZE_PATH = ROOT / "metadata/phase4_locked_bart_evaluation_freeze.json"
AUDIT_PATH = ROOT / "metadata/phase4_r2_generalization_audit.json"

PREDICTIONS_PATH = ROOT / "data/processed/phase5_bart_local_oof_predictions.parquet"
FOLD_METRICS_PATH = ROOT / "outputs/tables/phase5_bart_local_pass_metrics.csv"
SUMMARY_PATH = ROOT / "outputs/tables/phase5_bart_local_summary.csv"
FOLD_DIAGNOSTICS_PATH = ROOT / "outputs/tables/phase5_bart_local_fold_diagnostics.csv"
FIGURE_PATH = ROOT / "outputs/figures/phase5_bart_local_validation.png"
FREEZE_PATH = ROOT / "metadata/phase5_bart_local_validation_freeze.json"

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


def write_json(path: Path, value: Any) -> None:
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


def ridge(alpha: float) -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha))])


def buffered_indices(
    coordinates: np.ndarray,
    pass_ids: np.ndarray,
    held_out_pass: str,
    buffer_m: float,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    test = np.flatnonzero(pass_ids == held_out_pass)
    candidates = np.flatnonzero(pass_ids != held_out_pass)
    distances, _ = cKDTree(coordinates[test]).query(coordinates[candidates], k=1)
    train = candidates[distances >= buffer_m]
    if len(train) == 0:
        raise RuntimeError(f"Buffer removed all training rows for {held_out_pass}")
    nearest_test_to_train, _ = cKDTree(coordinates[train]).query(coordinates[test], k=1)
    return train, test, int(len(candidates) - len(train)), float(nearest_test_to_train.min())


def plot_summary(summary: pd.DataFrame, output: Path) -> None:
    ordered = summary.sort_values("rmse")
    labels = [name.replace("_", "\n") for name in ordered["model"]]
    colors = ["#2a9d8f" if "tessera" in name else "#577590" for name in ordered["model"]]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    axes[0].bar(np.arange(len(ordered)), ordered["rmse"], color=colors)
    axes[0].set_xticks(np.arange(len(ordered)), labels, fontsize=8)
    axes[0].set_ylabel("Out-of-pass RMSE")
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].scatter(ordered["observed_mean"], ordered["predicted_mean"], alpha=0)
    predictions = pd.read_parquet(PREDICTIONS_PATH)
    primary = "prediction_tessera_area_topography_ridge"
    for pass_id, group in predictions.groupby("held_out_pass", sort=True):
        axes[1].scatter(group["fhd_normal"], group[primary], s=10, alpha=0.45, label=pass_id)
    low = float(min(predictions["fhd_normal"].min(), predictions[primary].min()))
    high = float(max(predictions["fhd_normal"].max(), predictions[primary].max()))
    axes[1].plot([low, high], [low, high], color="black", linewidth=1)
    axes[1].set_xlabel("GEDI FHD")
    axes[1].set_ylabel("Local TESSERA + terrain prediction")
    axes[1].legend(frameon=False, fontsize=7)
    axes[1].grid(alpha=0.2)
    figure.suptitle("Post-hoc BART leave-one-pass-out validation")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.png")
    figure.savefig(temporary, dpi=180, bbox_inches="tight")
    plt.close(figure)
    temporary.replace(output)


def main() -> int:
    protected = [PREDICTIONS_PATH, FOLD_METRICS_PATH, SUMMARY_PATH, FOLD_DIAGNOSTICS_PATH, FIGURE_PATH, FREEZE_PATH]
    existing = [str(path.relative_to(ROOT)) for path in protected if path.exists()]
    if existing:
        raise RuntimeError(f"BART-local freeze exists; refusing to overwrite: {existing}")

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    protocol = config["phase5_bart_local_validation"]
    if protocol["status"] != "posthoc_after_locked_cross_region_evaluation":
        raise RuntimeError("BART-local analysis must remain labelled post hoc")
    transfer = json.loads(TRANSFER_FREEZE_PATH.read_text(encoding="utf-8"))
    sentinel = json.loads(SENTINEL_FREEZE_PATH.read_text(encoding="utf-8"))
    locked = json.loads(LOCKED_FREEZE_PATH.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    bart = pd.read_parquet(BART_PATH)
    conventional = pd.read_parquet(CONVENTIONAL_PATH)
    frame = bart.merge(conventional, on=["site_id", "shot_number"], validate="one_to_one", suffixes=("", "_conventional"))
    if len(frame) != int(protocol["rows"]) or set(frame["site_id"]) != {"BART"}:
        raise RuntimeError("BART-local input is not the exact frozen BART population")
    if not audit["r2_point_estimate_valid"] or locked["status"] != "frozen_one_time_locked_bart_evaluation":
        raise RuntimeError("Required locked BART audit/evaluation is not valid")

    frame["acquisition_date"] = pd.to_datetime(frame["acquisition_datetime"], utc=True).dt.date.astype(str)
    frame["pass_id"] = (
        frame["orbit"].astype(str)
        + "_"
        + frame["reference_ground_track"].astype(str)
        + "_"
        + frame["acquisition_date"]
    )
    passes = sorted(frame["pass_id"].unique())
    if len(passes) != int(protocol["expected_passes"]):
        raise RuntimeError(f"Expected three BART passes, found {passes}")

    terrain = list(config["phase3_extended_experiments"]["feature_sets"]["terrain"])
    area = [f"tessera_area_{index:03d}" for index in range(128)]
    corrected_sentinel = list(sentinel["freeze_basis"]["corrected_model_features"])
    if any(column.endswith("observation_count") for column in corrected_sentinel):
        raise RuntimeError("Corrected Sentinel feature set still contains observation counts")
    feature_sets = {
        "terrain_ridge": terrain,
        "sentinel_topography_ridge_corrected": corrected_sentinel,
        "tessera_area_ridge": area,
        "tessera_area_topography_ridge": [*area, *terrain],
    }
    alphas = {key: float(value) for key, value in protocol["ridge_alphas_from_development"].items()}
    frozen_alphas = transfer["freeze_basis"]["combined_development_alphas"]
    expected_alphas = {
        "terrain_ridge": float(frozen_alphas["terrain_ridge"]),
        "sentinel_topography_ridge_corrected": float(sentinel["freeze_basis"]["development_selected_alpha"]),
        "tessera_area_ridge": float(frozen_alphas["tessera_area_ridge"]),
        "tessera_area_topography_ridge": float(frozen_alphas["tessera_area_topography_ridge"]),
    }
    if alphas != expected_alphas:
        raise RuntimeError(f"Configured BART-local alphas do not match development freezes: {expected_alphas}")
    for name, columns in feature_sets.items():
        if not np.isfinite(frame[columns].to_numpy(dtype=np.float64)).all():
            raise RuntimeError(f"Non-finite values in {name}")

    nonlinear_parameters = {
        key: config["phase3"]["models"]["tessera_hist_gradient_boosting"][key]
        for key in ["learning_rate", "max_iter", "max_leaf_nodes", "min_samples_leaf", "l2_regularization"]
    }
    prediction_columns = {model: f"prediction_{model}" for model in protocol["models"]}
    predictions = frame[[
        "site_id", "shot_number", "pass_id", "orbit", "reference_ground_track",
        "acquisition_date", "x_epsg5070", "y_epsg5070", "fhd_normal",
    ]].copy()
    predictions = predictions.rename(columns={"pass_id": "held_out_pass"})
    for column in prediction_columns.values():
        predictions[column] = np.nan
    diagnostics: list[dict[str, Any]] = []
    fold_metrics: list[dict[str, Any]] = []
    coordinates = frame[["x_epsg5070", "y_epsg5070"]].to_numpy(dtype=np.float64)
    pass_ids = frame["pass_id"].to_numpy(dtype=str)
    y = frame["fhd_normal"].to_numpy(dtype=np.float64)

    for held_out_pass in passes:
        train, test, removed, minimum_distance = buffered_indices(
            coordinates, pass_ids, held_out_pass, float(protocol["exclusion_buffer_m"])
        )
        diagnostics.append({
            "held_out_pass": held_out_pass,
            "training_rows_before_buffer": int(np.sum(pass_ids != held_out_pass)),
            "training_rows_after_buffer": len(train),
            "test_rows": len(test),
            "buffer_removed_rows": removed,
            "minimum_train_test_distance_m": minimum_distance,
            "training_passes": "|".join(sorted(set(pass_ids[train]))),
        })
        fold_predictions: dict[str, np.ndarray] = {
            "training_mean": np.full(len(test), float(y[train].mean()), dtype=np.float64)
        }
        for model_name, columns in feature_sets.items():
            model = ridge(alphas[model_name])
            model.fit(frame.iloc[train][columns].to_numpy(dtype=np.float32), y[train])
            fold_predictions[model_name] = model.predict(frame.iloc[test][columns].to_numpy(dtype=np.float32))
        nonlinear = HistGradientBoostingRegressor(random_state=20260717, **nonlinear_parameters)
        nonlinear.fit(frame.iloc[train][area].to_numpy(dtype=np.float32), y[train])
        fold_predictions["tessera_hist_gradient_boosting"] = nonlinear.predict(
            frame.iloc[test][area].to_numpy(dtype=np.float32)
        )
        for model_name, values in fold_predictions.items():
            predictions.loc[test, prediction_columns[model_name]] = values
            metrics = metric_values(y[test], values)
            fold_metrics.append({
                "held_out_pass": held_out_pass,
                "model": model_name,
                "rows": len(test),
                **metrics,
            })

    if predictions[list(prediction_columns.values())].isna().any().any():
        raise RuntimeError("Not every BART row received every out-of-pass prediction")
    summary_records: list[dict[str, Any]] = []
    for model_name, column in prediction_columns.items():
        values = predictions[column].to_numpy(dtype=np.float64)
        summary_records.append({
            "model": model_name,
            "rows": len(predictions),
            **metric_values(y, values),
            "observed_mean": float(y.mean()),
            "predicted_mean": float(values.mean()),
            "independent_pass_units": len(passes),
            "confidence_interval_reported": False,
        })
    summary = pd.DataFrame(summary_records).sort_values("rmse").reset_index(drop=True)
    pass_table = pd.DataFrame(fold_metrics).sort_values(["held_out_pass", "rmse"]).reset_index(drop=True)
    diagnostic_table = pd.DataFrame(diagnostics).sort_values("held_out_pass").reset_index(drop=True)
    write_parquet(predictions, PREDICTIONS_PATH)
    write_csv(pass_table, FOLD_METRICS_PATH)
    write_csv(summary, SUMMARY_PATH)
    write_csv(diagnostic_table, FOLD_DIAGNOSTICS_PATH)
    plot_summary(summary, FIGURE_PATH)

    freeze_basis = {
        "protocol": protocol,
        "analysis_label": "posthoc_within_bart_local_validation",
        "input_bart_alignment_sha256": sha256(BART_PATH),
        "input_bart_conventional_sha256": sha256(CONVENTIONAL_PATH),
        "locked_bart_evaluation_freeze_id": locked["freeze_id"],
        "r2_audit_locked_evaluation_freeze_id": audit["locked_evaluation_freeze_id"],
        "development_transfer_freeze_id": transfer["freeze_id"],
        "corrected_sentinel_freeze_id": sentinel["freeze_id"],
        "passes": passes,
        "feature_sets": feature_sets,
        "ridge_alphas": alphas,
        "nonlinear_parameters": nonlinear_parameters,
        "no_bart_hyperparameter_tuning": True,
        "script_sha256": sha256(Path(__file__)),
        "project_config_sha256": sha256(CONFIG_PATH),
        "sklearn_version": sklearn.__version__,
    }
    freeze_id = "phase5-bart-local-" + canonical_hash(freeze_basis)[:12]
    freeze = {
        "freeze_id": freeze_id,
        "created_at": utc_now(),
        "status": "frozen_posthoc_local_validation",
        "freeze_basis": freeze_basis,
        "freeze_basis_sha256": canonical_hash(freeze_basis),
        "outputs": {
            "predictions": {"path": str(PREDICTIONS_PATH.relative_to(ROOT)), "sha256": sha256(PREDICTIONS_PATH), "rows": len(predictions)},
            "pass_metrics": {"path": str(FOLD_METRICS_PATH.relative_to(ROOT)), "sha256": sha256(FOLD_METRICS_PATH), "rows": len(pass_table)},
            "summary": {"path": str(SUMMARY_PATH.relative_to(ROOT)), "sha256": sha256(SUMMARY_PATH), "rows": len(summary)},
            "fold_diagnostics": {"path": str(FOLD_DIAGNOSTICS_PATH.relative_to(ROOT)), "sha256": sha256(FOLD_DIAGNOSTICS_PATH), "rows": len(diagnostic_table)},
            "figure": {"path": str(FIGURE_PATH.relative_to(ROOT)), "sha256": sha256(FIGURE_PATH)},
        },
        "primary_model": "tessera_area_topography_ridge",
        "primary_metrics": summary.set_index("model").loc["tessera_area_topography_ridge", METRICS].to_dict(),
        "best_rmse_model": str(summary.iloc[0]["model"]),
        "interpretation": "Post-hoc within-BART cross-pass interpolation; not cross-region transfer and not a wall-to-wall mapping result.",
    }
    write_json(FREEZE_PATH, freeze)
    print(summary.to_string(index=False))
    print("\n", diagnostic_table.to_string(index=False))
    print(f"\nFrozen {freeze_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
