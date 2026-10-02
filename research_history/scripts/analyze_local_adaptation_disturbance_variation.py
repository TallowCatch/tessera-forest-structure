#!/usr/bin/env python3
"""Stratify frozen prediction errors by documented disturbance recency."""

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
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
DISTURBANCE_PATH = ROOT / "data/processed/phase5_landfire_disturbance_history.parquet"
DEVELOPMENT_PATH = ROOT / "data/processed/phase3_extended_oof_predictions.parquet"
LOCKED_BART_PATH = ROOT / "data/processed/phase4_locked_bart_predictions.parquet"
LOCAL_BART_PATH = ROOT / "data/processed/phase5_bart_local_oof_predictions.parquet"

COHORT_PATH = ROOT / "data/processed/phase5_prediction_disturbance_cohorts.parquet"
METRICS_PATH = ROOT / "outputs/tables/phase5_disturbance_cohort_metrics.csv"
COMPARISON_PATH = ROOT / "outputs/tables/phase5_disturbance_primary_comparison.csv"
FIGURE_PATH = ROOT / "outputs/figures/phase5_disturbance_variation.png"
FREEZE_PATH = ROOT / "metadata/phase5_disturbance_variation_freeze.json"

MINIMUM_REPORTABLE_ROWS = 30
PRIMARY_MODEL = "tessera_area_topography_ridge"


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
    if np.ptp(observed) <= 1e-12 or np.ptp(predicted) <= 1e-12:
        rank = math.nan
    else:
        rank = float(spearmanr(observed, predicted).statistic)
    return {
        "r2": float(r2_score(observed, predicted)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(mean_absolute_error(observed, predicted)),
        "spearman_r": rank,
        "mean_bias": float(np.mean(residual)),
        "observed_mean": float(np.mean(observed)),
        "predicted_mean": float(np.mean(predicted)),
    }


def cohort_label(years: pd.Series) -> pd.Series:
    result = pd.Series("no_record_1999_2024", index=years.index, dtype="string")
    result.loc[years.between(0, 5, inclusive="both")] = "recent_0_5_years"
    result.loc[years.between(6, 10, inclusive="both")] = "intermediate_6_10_years"
    result.loc[years.between(11, 25, inclusive="both")] = "older_11_25_years"
    if result.isna().any():
        raise RuntimeError("Every disturbance row must receive a cohort")
    return result


def analysis_frame(path: Path, analysis: str, disturbance: pd.DataFrame) -> pd.DataFrame:
    predictions = pd.read_parquet(path)
    frame = predictions.merge(disturbance, on=["site_id", "shot_number"], validate="one_to_one")
    frame.insert(0, "analysis", analysis)
    frame["disturbance_cohort"] = cohort_label(frame["years_since_latest_recorded_disturbance"])
    frame["disturbance_binary"] = np.where(
        frame["has_recorded_disturbance_1999_2024"], "recorded_1999_2024", "no_record_1999_2024"
    )
    return frame


def main() -> int:
    protected = [COHORT_PATH, METRICS_PATH, COMPARISON_PATH, FIGURE_PATH, FREEZE_PATH]
    existing = [str(path.relative_to(ROOT)) for path in protected if path.exists()]
    if existing:
        raise RuntimeError(f"Disturbance-variation freeze exists; refusing to overwrite: {existing}")
    disturbance = pd.read_parquet(DISTURBANCE_PATH)
    frames = [
        analysis_frame(DEVELOPMENT_PATH, "development_buffered_spatial_oof", disturbance),
        analysis_frame(LOCKED_BART_PATH, "bart_locked_cross_region_transfer", disturbance),
        analysis_frame(LOCAL_BART_PATH, "bart_local_leave_one_pass_out", disturbance),
    ]
    expected_rows = [2037, 1026, 1026]
    if [len(frame) for frame in frames] != expected_rows:
        raise RuntimeError("Prediction/disturbance joins changed frozen populations")

    common_models = [
        "training_mean",
        "terrain_ridge",
        "tessera_area_ridge",
        "tessera_area_topography_ridge",
        "tessera_hist_gradient_boosting",
    ]
    records: list[dict[str, Any]] = []
    retained: list[pd.DataFrame] = []
    for frame in frames:
        analysis = str(frame["analysis"].iloc[0])
        models = list(common_models)
        if analysis == "bart_local_leave_one_pass_out":
            models.append("sentinel_topography_ridge_corrected")
        for grouping_name in ["disturbance_cohort", "disturbance_binary"]:
            for (site, group_name), group in frame.groupby(["site_id", grouping_name], sort=True):
                observed = group["fhd_normal"].to_numpy(dtype=np.float64)
                for model in models:
                    column = f"prediction_{model}"
                    predicted = group[column].to_numpy(dtype=np.float64)
                    records.append({
                        "analysis": analysis,
                        "site_id": site,
                        "grouping": grouping_name,
                        "group": group_name,
                        "model": model,
                        "rows": len(group),
                        "reportable_minimum_30_rows": len(group) >= MINIMUM_REPORTABLE_ROWS,
                        **metric_values(observed, predicted),
                    })
        retained.append(
            frame[[
                "analysis", "site_id", "shot_number", "fhd_normal", "disturbance_cohort",
                "disturbance_binary", "latest_recorded_disturbance_calendar_year",
                "years_since_latest_recorded_disturbance", "disturbance_type",
                *[f"prediction_{model}" for model in models],
            ]]
        )
    metrics = pd.DataFrame(records).sort_values(["analysis", "site_id", "grouping", "group", "rmse"]).reset_index(drop=True)
    cohorts = pd.concat(retained, ignore_index=True)
    primary = metrics[
        metrics["model"].eq(PRIMARY_MODEL)
        & metrics["reportable_minimum_30_rows"]
        & metrics["grouping"].eq("disturbance_cohort")
    ].copy()
    primary = primary.sort_values(["analysis", "site_id", "group"]).reset_index(drop=True)

    COHORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_cohort = COHORT_PATH.with_suffix(".tmp.parquet")
    cohorts.to_parquet(temporary_cohort, index=False, compression="zstd")
    temporary_cohort.replace(COHORT_PATH)
    for table, path in [(metrics, METRICS_PATH), (primary, COMPARISON_PATH)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.csv")
        table.to_csv(temporary, index=False)
        temporary.replace(path)

    figure, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    order = ["recent_0_5_years", "intermediate_6_10_years", "older_11_25_years", "no_record_1999_2024"]
    labels = ["0-5", "6-10", "11-25", "No record\nsince 1999"]
    analyses = [
        ("development_buffered_spatial_oof", "SOAP/TEAK spatial OOF"),
        ("bart_locked_cross_region_transfer", "Locked transfer to BART"),
        ("bart_local_leave_one_pass_out", "BART local cross-pass"),
    ]
    for axis, (analysis, title) in zip(axes, analyses, strict=True):
        subset = primary[primary["analysis"].eq(analysis)]
        for site, group in subset.groupby("site_id", sort=True):
            lookup = group.set_index("group")
            x = [index for index, name in enumerate(order) if name in lookup.index]
            y = [float(lookup.loc[name, "rmse"]) for name in order if name in lookup.index]
            axis.scatter(x, y, s=42, label=site, zorder=3)
            contiguous_runs: list[list[int]] = []
            for position in range(len(x)):
                if not contiguous_runs or x[position] != x[contiguous_runs[-1][-1]] + 1:
                    contiguous_runs.append([position])
                else:
                    contiguous_runs[-1].append(position)
            for run in contiguous_runs:
                if len(run) >= 2:
                    axis.plot([x[position] for position in run], [y[position] for position in run], linewidth=1.8)
        axis.set_xticks(range(len(order)), labels)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False)
    axes[0].set_ylabel("TESSERA + terrain RMSE")
    figure.suptitle("Prediction error by documented LANDFIRE disturbance recency")
    figure.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_figure = FIGURE_PATH.with_suffix(".tmp.png")
    figure.savefig(temporary_figure, dpi=180, bbox_inches="tight")
    plt.close(figure)
    temporary_figure.replace(FIGURE_PATH)

    freeze_basis = {
        "disturbance_input_sha256": sha256(DISTURBANCE_PATH),
        "prediction_inputs": {str(path.relative_to(ROOT)): sha256(path) for path in [DEVELOPMENT_PATH, LOCKED_BART_PATH, LOCAL_BART_PATH]},
        "cohort_definitions": {
            "recent_0_5_years": [0, 5],
            "intermediate_6_10_years": [6, 10],
            "older_11_25_years": [11, 25],
            "no_record_1999_2024": None,
        },
        "minimum_reportable_rows": MINIMUM_REPORTABLE_ROWS,
        "primary_model": PRIMARY_MODEL,
        "invalid_interpretation": "Chronological stand age or a causal disturbance effect.",
        "script_sha256": sha256(Path(__file__)),
    }
    freeze_id = "phase5-disturbance-analysis-" + canonical_hash(freeze_basis)[:12]
    freeze = {
        "freeze_id": freeze_id,
        "created_at": utc_now(),
        "status": "frozen_disturbance_recency_error_stratification",
        "freeze_basis": freeze_basis,
        "freeze_basis_sha256": canonical_hash(freeze_basis),
        "outputs": {
            "cohorts": {"path": str(COHORT_PATH.relative_to(ROOT)), "rows": len(cohorts), "sha256": sha256(COHORT_PATH)},
            "metrics": {"path": str(METRICS_PATH.relative_to(ROOT)), "rows": len(metrics), "sha256": sha256(METRICS_PATH)},
            "primary_comparison": {"path": str(COMPARISON_PATH.relative_to(ROOT)), "rows": len(primary), "sha256": sha256(COMPARISON_PATH)},
            "figure": {"path": str(FIGURE_PATH.relative_to(ROOT)), "sha256": sha256(FIGURE_PATH)},
        },
        "valid_interpretation": "Descriptive model error across documented disturbance-recency groups with at least 30 shots.",
        "invalid_interpretation": "Biological age validation, a causal disturbance effect, or independent samples within each group.",
    }
    write_json(FREEZE_PATH, freeze)
    print(primary[["analysis", "site_id", "group", "rows", "r2", "rmse", "mae", "mean_bias"]].to_string(index=False))
    print(f"\nFrozen {freeze_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
