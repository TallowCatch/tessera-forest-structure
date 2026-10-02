#!/usr/bin/env python3
"""Evaluate already-frozen target-free CORAL predictions on BART outcomes."""

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
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
TARGET_PATH = ROOT / "data/processed/tessera_aligned_locked_bart.parquet"
PREDICTIONS_PATH = ROOT / "data/processed/phase5_bart_coral_target_free_predictions.parquet"
PREDICTION_FREEZE_PATH = ROOT / "metadata/phase5_bart_coral_prediction_freeze.json"
ZERO_SHOT_PATH = ROOT / "data/processed/phase4_locked_bart_predictions.parquet"
METRICS_PATH = ROOT / "outputs/tables/phase5_bart_coral_metrics.csv"
EVALUATION_PATH = ROOT / "data/processed/phase5_bart_coral_evaluation.parquet"
FIGURE_PATH = ROOT / "outputs/figures/phase5_bart_coral_evaluation.png"
FREEZE_PATH = ROOT / "metadata/phase5_bart_coral_evaluation_freeze.json"
METHOD_COLUMNS = {
    "source_training_mean": "prediction_source_training_mean",
    "frozen_source": "prediction_frozen_source_recreated",
    "coral": "prediction_coral",
}


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


def metric_values(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = predicted - observed
    pearson = math.nan
    spearman = math.nan
    if np.ptp(observed) > 1e-12 and np.ptp(predicted) > 1e-12:
        pearson = float(pearsonr(observed, predicted).statistic)
        spearman = float(spearmanr(observed, predicted).statistic)
    return {
        "r2": float(r2_score(observed, predicted)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(mean_absolute_error(observed, predicted)),
        "pearson_r": pearson,
        "spearman_r": spearman,
        "mean_bias": float(np.mean(residual)),
    }


def main() -> int:
    protected = [METRICS_PATH, EVALUATION_PATH, FIGURE_PATH, FREEZE_PATH]
    existing = [str(path.relative_to(ROOT)) for path in protected if path.exists()]
    if existing:
        raise RuntimeError(f"CORAL evaluation freeze exists; refusing to overwrite: {existing}")
    prediction_freeze = json.loads(PREDICTION_FREEZE_PATH.read_text(encoding="utf-8"))
    if prediction_freeze["status"] != "frozen_target_free_coral_predictions_before_bart_outcome_access":
        raise RuntimeError("CORAL predictions were not frozen target-free")
    if prediction_freeze["freeze_basis"]["target_labels_used_for_alignment_training_or_selection"]:
        raise RuntimeError("CORAL prediction freeze reports target-label use")
    if prediction_freeze["outputs"]["predictions"]["sha256"] != sha256(PREDICTIONS_PATH):
        raise RuntimeError("CORAL target-free predictions changed after freeze")

    predictions = pd.read_parquet(PREDICTIONS_PATH)
    target_columns = [
        "site_id", "shot_number", "fhd_normal", "acquisition_datetime", "orbit",
        "reference_ground_track", "x_epsg5070", "y_epsg5070",
    ]
    targets = pd.read_parquet(TARGET_PATH, columns=target_columns)
    evaluation = predictions.merge(targets, on=["site_id", "shot_number"], validate="one_to_one")
    if len(evaluation) != 1026:
        raise RuntimeError("CORAL evaluation population changed")
    evaluation["acquisition_date"] = pd.to_datetime(
        evaluation["acquisition_datetime"], utc=True
    ).dt.date.astype(str)
    evaluation["pass_id"] = (
        evaluation["orbit"].astype(str)
        + "_" + evaluation["reference_ground_track"].astype(str)
        + "_" + evaluation["acquisition_date"]
    )
    if evaluation["pass_id"].nunique() != 3:
        raise RuntimeError("CORAL evaluation no longer contains three BART passes")

    stored_zero_shot = pd.read_parquet(ZERO_SHOT_PATH)[
        ["site_id", "shot_number", "prediction_tessera_area_topography_ridge"]
    ]
    recreation = evaluation.merge(stored_zero_shot, on=["site_id", "shot_number"], validate="one_to_one")
    recreation_difference = float(np.max(np.abs(
        recreation["prediction_frozen_source_recreated"]
        - recreation["prediction_tessera_area_topography_ridge"]
    )))
    if recreation_difference > 1e-6:
        raise RuntimeError("CORAL comparator does not recreate the frozen source model")

    records = []
    scopes = [("pooled", "all_passes", evaluation)]
    scopes.extend(("pass", pass_id, frame) for pass_id, frame in evaluation.groupby("pass_id", sort=True))
    for scope, scope_id, frame in scopes:
        for method, column in METHOD_COLUMNS.items():
            records.append({
                "scope": scope,
                "scope_id": scope_id,
                "method": method,
                "rows": len(frame),
                **metric_values(frame["fhd_normal"].to_numpy(), frame[column].to_numpy()),
            })
    metrics = pd.DataFrame(records)
    pooled = metrics[metrics["scope"].eq("pooled")].set_index("method")
    gate_passed = (
        float(pooled.loc["coral", "r2"]) > 0
        and float(pooled.loc["coral", "rmse"]) < float(pooled.loc["frozen_source", "rmse"])
    )

    EVALUATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    evaluation.to_parquet(EVALUATION_PATH, index=False, compression="zstd")
    metrics.to_csv(METRICS_PATH, index=False)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.8))
    ordered = pooled.loc[["source_training_mean", "frozen_source", "coral"]]
    axes[0].bar(np.arange(3), ordered["rmse"], color=["#6c757d", "#d1495b", "#2a9d8f"])
    axes[0].set_xticks(np.arange(3), ["source mean", "frozen source", "CORAL"])
    axes[0].set_ylabel("BART RMSE")
    axes[0].set_title("Target-free adaptation comparison")
    axes[1].scatter(evaluation["fhd_normal"], evaluation["prediction_coral"], s=12, alpha=0.45)
    low = float(min(evaluation["fhd_normal"].min(), evaluation["prediction_coral"].min()))
    high = float(max(evaluation["fhd_normal"].max(), evaluation["prediction_coral"].max()))
    axes[1].plot([low, high], [low, high], color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("GEDI V3 fhd_normal")
    axes[1].set_ylabel("CORAL prediction")
    axes[1].set_title("Post-hoc label-free CORAL at BART")
    figure.tight_layout()
    temporary = FIGURE_PATH.with_suffix(".tmp.png")
    figure.savefig(temporary, dpi=180, bbox_inches="tight")
    plt.close(figure)
    temporary.replace(FIGURE_PATH)

    freeze_basis = {
        "analysis_label": "posthoc_bart_coral_evaluation",
        "prediction_freeze_id": prediction_freeze["freeze_id"],
        "prediction_freeze_sha256": sha256(PREDICTION_FREEZE_PATH),
        "target_path": str(TARGET_PATH.relative_to(ROOT)),
        "target_sha256": sha256(TARGET_PATH),
        "target_columns_read_after_prediction_freeze": target_columns,
        "frozen_source_predictions_sha256": sha256(ZERO_SHOT_PATH),
        "frozen_source_recreation_max_abs_difference": recreation_difference,
        "target_labels_used_for_coral_alignment_training_or_selection": False,
        "script_sha256": sha256(Path(__file__)),
    }
    freeze_id = "phase5-bart-coral-evaluation-" + canonical_hash(freeze_basis)[:12]
    freeze = {
        "freeze_id": freeze_id,
        "created_at": utc_now(),
        "status": "frozen_posthoc_label_free_coral_evaluation",
        "gate_passed": gate_passed,
        "freeze_basis": freeze_basis,
        "freeze_basis_sha256": canonical_hash(freeze_basis),
        "outputs": {
            "evaluation": {"path": str(EVALUATION_PATH.relative_to(ROOT)), "sha256": sha256(EVALUATION_PATH), "rows": len(evaluation)},
            "metrics": {"path": str(METRICS_PATH.relative_to(ROOT)), "sha256": sha256(METRICS_PATH), "rows": len(metrics)},
            "figure": {"path": str(FIGURE_PATH.relative_to(ROOT)), "sha256": sha256(FIGURE_PATH)},
        },
        "pooled_metrics": pooled.reset_index().to_dict(orient="records"),
        "interpretation_limit": "Post-hoc transductive unsupervised adaptation on one target site.",
    }
    write_json(FREEZE_PATH, freeze)
    print(pooled.reset_index().to_string(index=False))
    print(f"\nCORAL gate passed: {gate_passed}; frozen {freeze_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
