#!/usr/bin/env python3
"""Run frozen post-hoc few-shot adaptation on spatially separated BART passes."""

from __future__ import annotations

import hashlib
import json
import math
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
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/project.yaml"
PROTOCOL_FREEZE_PATH = ROOT / "metadata/project_config_phase5_adaptation_protocol_freeze.yaml"
SOURCE_PATH = ROOT / "data/processed/tessera_aligned_development_soap_teak.parquet"
SOURCE_CONVENTIONAL_PATH = ROOT / "data/processed/phase3_conventional_predictors_development.parquet"
TARGET_PATH = ROOT / "data/processed/tessera_aligned_locked_bart.parquet"
TARGET_CONVENTIONAL_PATH = ROOT / "data/processed/phase4_conventional_predictors_bart.parquet"
SOURCE_MODEL_PATH = ROOT / "outputs/models/phase4_selected_pipeline_pre_bart.joblib"
ZERO_SHOT_PREDICTIONS_PATH = ROOT / "data/processed/phase4_locked_bart_predictions.parquet"
LOCAL_PREDICTIONS_PATH = ROOT / "data/processed/phase5_bart_local_oof_predictions.parquet"
LOCAL_DIAGNOSTICS_PATH = ROOT / "outputs/tables/phase5_bart_local_fold_diagnostics.csv"
LOCAL_FREEZE_PATH = ROOT / "metadata/phase5_bart_local_validation_freeze.json"
PREDICTIONS_PATH = ROOT / "data/processed/phase5_bart_few_shot_predictions.parquet"
METRICS_PATH = ROOT / "outputs/tables/phase5_bart_few_shot_metrics.csv"
SUMMARY_PATH = ROOT / "outputs/tables/phase5_bart_few_shot_summary.csv"
FIGURE_PATH = ROOT / "outputs/figures/phase5_bart_few_shot_adaptation.png"
FREEZE_PATH = ROOT / "metadata/phase5_bart_few_shot_adaptation_freeze.json"

METHODS = [
    "frozen_source",
    "target_training_mean",
    "source_intercept_correction",
    "source_affine_correction",
    "target_only_ridge",
    "equal_domain_weighted_source_plus_target_ridge",
]


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
    rank = math.nan
    if np.ptp(observed) > 1e-12 and np.ptp(predicted) > 1e-12:
        rank = float(spearmanr(observed, predicted).statistic)
    return {
        "r2": float(r2_score(observed, predicted)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(mean_absolute_error(observed, predicted)),
        "spearman_r": rank,
        "mean_bias": float(np.mean(residual)),
    }


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
    nearest, _ = cKDTree(coordinates[train]).query(coordinates[test], k=1)
    return train, test, int(len(candidates) - len(train)), float(nearest.min())


def spatially_balanced_sample(
    candidate_indices: np.ndarray,
    coordinates: np.ndarray,
    sample_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if sample_size > len(candidate_indices):
        raise ValueError("Sample exceeds adaptation population")
    if sample_size == len(candidate_indices):
        return candidate_indices.copy()
    cells = np.floor(coordinates[candidate_indices] / 1000.0).astype(np.int64)
    cell_keys = [f"{x}_{y}" for x, y in cells]
    groups: dict[str, list[int]] = {}
    for index, key in zip(candidate_indices, cell_keys, strict=True):
        groups.setdefault(key, []).append(int(index))
    for values in groups.values():
        rng.shuffle(values)
    ordered_keys = list(groups)
    rng.shuffle(ordered_keys)
    selected: list[int] = []
    depth = 0
    while len(selected) < sample_size:
        available = [key for key in ordered_keys if depth < len(groups[key])]
        if not available:
            raise RuntimeError("Spatial sampler exhausted candidates unexpectedly")
        rng.shuffle(available)
        for key in available:
            selected.append(groups[key][depth])
            if len(selected) == sample_size:
                break
        depth += 1
    return np.asarray(selected, dtype=np.int64)


def affine_coefficients(predicted: np.ndarray, observed: np.ndarray) -> tuple[float, float]:
    design = np.column_stack([np.ones(len(predicted)), predicted])
    intercept, slope = np.linalg.lstsq(design, observed, rcond=None)[0]
    return float(intercept), float(slope)


def fit_target_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[StandardScaler, Ridge]:
    scaler = StandardScaler().fit(x)
    model = Ridge(alpha=alpha).fit(scaler.transform(x), y)
    return scaler, model


def fit_equal_domain_ridge(
    source_x: np.ndarray,
    source_y: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
    alpha: float,
) -> tuple[StandardScaler, Ridge]:
    combined_x = np.vstack([source_x, target_x])
    combined_y = np.concatenate([source_y, target_y])
    source_weight = 0.5 / len(source_x)
    target_weight = 0.5 / len(target_x)
    weights = np.concatenate([
        np.full(len(source_x), source_weight),
        np.full(len(target_x), target_weight),
    ])
    weights *= len(weights) / weights.sum()
    scaler = StandardScaler().fit(combined_x, sample_weight=weights)
    model = Ridge(alpha=alpha).fit(scaler.transform(combined_x), combined_y, sample_weight=weights)
    return scaler, model


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], list[str]]:
    source = pd.read_parquet(SOURCE_PATH).merge(
        pd.read_parquet(SOURCE_CONVENTIONAL_PATH),
        on=["site_id", "shot_number"], validate="one_to_one", suffixes=("", "_conventional"),
    )
    target = pd.read_parquet(TARGET_PATH).merge(
        pd.read_parquet(TARGET_CONVENTIONAL_PATH),
        on=["site_id", "shot_number"], validate="one_to_one", suffixes=("", "_conventional"),
    )
    pass_assignment = pd.read_parquet(LOCAL_PREDICTIONS_PATH)[
        ["site_id", "shot_number", "held_out_pass"]
    ]
    zero_shot = pd.read_parquet(ZERO_SHOT_PREDICTIONS_PATH)[
        ["site_id", "shot_number", "prediction_tessera_area_topography_ridge"]
    ].rename(columns={"prediction_tessera_area_topography_ridge": "frozen_source_prediction"})
    target = target.merge(pass_assignment, on=["site_id", "shot_number"], validate="one_to_one")
    target = target.merge(zero_shot, on=["site_id", "shot_number"], validate="one_to_one")
    artifact = joblib.load(SOURCE_MODEL_PATH)
    features = list(artifact["features"])
    if artifact["model_name"] != "tessera_area_topography_ridge" or float(artifact["alpha"]) != 100.0:
        raise RuntimeError("Frozen source artifact is not the preselected model")
    recreated = artifact["model"].predict(target[features].to_numpy(dtype=np.float32))
    recreation_difference = float(
        np.max(np.abs(recreated - target["frozen_source_prediction"].to_numpy()))
    )
    if recreation_difference > 1e-6:
        raise RuntimeError(
            f"Frozen source prediction recreation exceeds float32 tolerance: {recreation_difference}"
        )
    artifact["prediction_recreation_max_abs_difference"] = recreation_difference
    if len(source) != 2037 or len(target) != 1026:
        raise RuntimeError("Source or BART population changed")
    return source, target, artifact, features


def predictions_for_fold(
    source_x: np.ndarray,
    source_y: np.ndarray,
    target: pd.DataFrame,
    features: list[str],
    adaptation_indices: np.ndarray,
    test_indices: np.ndarray,
    alpha: float,
) -> dict[str, np.ndarray]:
    target_x = target[features].to_numpy(dtype=np.float64)
    target_y = target["fhd_normal"].to_numpy(dtype=np.float64)
    source_prediction = target["frozen_source_prediction"].to_numpy(dtype=np.float64)
    adapt_y = target_y[adaptation_indices]
    adapt_source_prediction = source_prediction[adaptation_indices]
    outputs = {
        "frozen_source": source_prediction[test_indices],
        "target_training_mean": np.full(len(test_indices), float(adapt_y.mean())),
    }
    correction = float(np.mean(adapt_y - adapt_source_prediction))
    outputs["source_intercept_correction"] = source_prediction[test_indices] + correction
    intercept, slope = affine_coefficients(adapt_source_prediction, adapt_y)
    outputs["source_affine_correction"] = intercept + slope * source_prediction[test_indices]
    scaler, model = fit_target_ridge(target_x[adaptation_indices], adapt_y, alpha)
    outputs["target_only_ridge"] = model.predict(scaler.transform(target_x[test_indices]))
    scaler, model = fit_equal_domain_ridge(
        source_x, source_y, target_x[adaptation_indices], adapt_y, alpha
    )
    outputs["equal_domain_weighted_source_plus_target_ridge"] = model.predict(
        scaler.transform(target_x[test_indices])
    )
    return outputs


def build_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    records = []
    group_columns = ["budget_label", "budget_sort", "replicate", "method"]
    for keys, frame in predictions.groupby(group_columns, sort=True):
        budget_label, budget_sort, replicate, method = keys
        records.append({
            "scope": "pooled",
            "held_out_pass": "all_passes",
            "budget_label": budget_label,
            "budget_sort": budget_sort,
            "replicate": replicate,
            "method": method,
            "rows": len(frame),
            "minimum_adaptation_rows": int(frame["adaptation_rows"].min()),
            "maximum_adaptation_rows": int(frame["adaptation_rows"].max()),
            **metric_values(frame["fhd_normal"].to_numpy(), frame["prediction"].to_numpy()),
        })
        for held_out_pass, pass_frame in frame.groupby("held_out_pass", sort=True):
            records.append({
                "scope": "pass",
                "held_out_pass": held_out_pass,
                "budget_label": budget_label,
                "budget_sort": budget_sort,
                "replicate": replicate,
                "method": method,
                "rows": len(pass_frame),
                "minimum_adaptation_rows": int(pass_frame["adaptation_rows"].min()),
                "maximum_adaptation_rows": int(pass_frame["adaptation_rows"].max()),
                **metric_values(
                    pass_frame["fhd_normal"].to_numpy(), pass_frame["prediction"].to_numpy()
                ),
            })
    return pd.DataFrame(records)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    pooled = metrics[metrics["scope"].eq("pooled")]
    records = []
    for keys, frame in pooled.groupby(["budget_label", "budget_sort", "method"], sort=True):
        budget_label, budget_sort, method = keys
        record: dict[str, Any] = {
            "budget_label": budget_label,
            "budget_sort": budget_sort,
            "method": method,
            "repetitions": len(frame),
        }
        for metric in ["r2", "rmse", "mae", "spearman_r", "mean_bias"]:
            values = frame[metric].to_numpy(dtype=float)
            record[f"{metric}_median"] = float(np.nanmedian(values))
            record[f"{metric}_mean"] = float(np.nanmean(values))
            record[f"{metric}_q05"] = float(np.nanquantile(values, 0.05))
            record[f"{metric}_q95"] = float(np.nanquantile(values, 0.95))
        records.append(record)
    summary = pd.DataFrame(records)
    mean_rmse = summary[summary["method"].eq("target_training_mean")].set_index("budget_label")[
        "rmse_median"
    ]
    summary["target_training_mean_rmse_median"] = summary["budget_label"].map(mean_rmse)
    summary["adaptation_recovery_gate_passed"] = (
        summary["r2_median"].gt(0)
        & summary["rmse_median"].lt(summary["target_training_mean_rmse_median"])
    )
    return summary.sort_values(["budget_sort", "method"]).reset_index(drop=True)


def save_figure(summary: pd.DataFrame) -> None:
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    labels = ["10", "25", "50", "100", "all"]
    x = np.arange(len(labels))
    for method in METHODS:
        frame = summary[summary["method"].eq(method)].set_index("budget_label").loc[labels]
        name = method.replace("_", " ")
        axes[0].plot(x, frame["rmse_median"], marker="o", label=name)
        axes[1].plot(x, frame["r2_median"], marker="o", label=name)
        if frame["repetitions"].max() > 1:
            axes[0].fill_between(x, frame["rmse_q05"], frame["rmse_q95"], alpha=0.08)
            axes[1].fill_between(x, frame["r2_q05"], frame["r2_q95"], alpha=0.08)
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.set_xlabel("BART adaptation labels per outer fold")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Pooled held-out-pass RMSE")
    axes[1].set_ylabel("Pooled held-out-pass R-squared")
    axes[1].axhline(0, color="black", linewidth=1, linestyle="--")
    figure.suptitle("Post-hoc spatially buffered BART few-shot adaptation")
    handles, legend_labels = axes[1].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="lower center", ncol=3, frameon=False, fontsize=8)
    figure.tight_layout(rect=(0, 0.16, 1, 1))
    temporary = FIGURE_PATH.with_suffix(".tmp.png")
    figure.savefig(temporary, dpi=180, bbox_inches="tight")
    plt.close(figure)
    temporary.replace(FIGURE_PATH)


def main() -> int:
    protected = [PREDICTIONS_PATH, METRICS_PATH, SUMMARY_PATH, FIGURE_PATH, FREEZE_PATH]
    existing = [str(path.relative_to(ROOT)) for path in protected if path.exists()]
    if existing:
        raise RuntimeError(f"Few-shot freeze exists; refusing to overwrite: {existing}")
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    protocol = config["phase5_bart_few_shot_adaptation"]
    if protocol["status"] != "posthoc_protocol_frozen_after_zero_shot_and_local_bart_results":
        raise RuntimeError("Few-shot experiment lost its post-hoc status")
    local_freeze = json.loads(LOCAL_FREEZE_PATH.read_text(encoding="utf-8"))
    if local_freeze["freeze_id"] != protocol["outer_split_freeze_id"]:
        raise RuntimeError("Few-shot outer split does not match the frozen local validation")
    source, target, artifact, features = load_inputs()
    coordinates = target[["x_epsg5070", "y_epsg5070"]].to_numpy(dtype=np.float64)
    pass_ids = target["held_out_pass"].to_numpy(dtype=str)
    target_y = target["fhd_normal"].to_numpy(dtype=np.float64)
    source_x = source[features].to_numpy(dtype=np.float64)
    source_y = source["fhd_normal"].to_numpy(dtype=np.float64)
    diagnostics = pd.read_csv(LOCAL_DIAGNOSTICS_PATH).set_index("held_out_pass")
    budgets = list(protocol["label_budgets"])
    finite_repetitions = int(protocol["repetitions_per_finite_budget"])
    random_seed = int(protocol["random_seed"])
    alpha = float(protocol["ridge_alpha_from_development"])
    prediction_frames = []

    for budget_sort, budget in enumerate(budgets):
        budget_label = str(budget)
        repetitions = 1 if budget_label == "all" else finite_repetitions
        for replicate in range(repetitions):
            for pass_position, held_out_pass in enumerate(sorted(set(pass_ids))):
                adaptation_pool, test, removed, minimum_distance = buffered_indices(
                    coordinates, pass_ids, held_out_pass, float(protocol["exclusion_buffer_m"])
                )
                expected = diagnostics.loc[held_out_pass]
                if (
                    len(adaptation_pool) != int(expected["training_rows_after_buffer"])
                    or len(test) != int(expected["test_rows"])
                    or removed != int(expected["buffer_removed_rows"])
                    or not math.isclose(minimum_distance, float(expected["minimum_train_test_distance_m"]), abs_tol=1e-9)
                ):
                    raise RuntimeError(f"Frozen outer fold changed: {held_out_pass}")
                if budget_label == "all":
                    adaptation = adaptation_pool
                else:
                    sample_size = int(budget)
                    rng = np.random.default_rng(
                        random_seed + budget_sort * 100_000 + replicate * 100 + pass_position
                    )
                    adaptation = spatially_balanced_sample(
                        adaptation_pool, coordinates, sample_size, rng
                    )
                fold_outputs = predictions_for_fold(
                    source_x, source_y, target, features, adaptation, test, alpha
                )
                cell_count = len({
                    (int(coordinates[index, 0] // 1000), int(coordinates[index, 1] // 1000))
                    for index in adaptation
                })
                base = target.iloc[test][
                    ["site_id", "shot_number", "held_out_pass", "x_epsg5070", "y_epsg5070", "fhd_normal"]
                ].copy()
                for method, values in fold_outputs.items():
                    output = base.copy()
                    output["budget_label"] = budget_label
                    output["budget_sort"] = budget_sort
                    output["replicate"] = replicate
                    output["method"] = method
                    output["prediction"] = values
                    output["adaptation_rows"] = len(adaptation)
                    output["adaptation_1km_cells"] = cell_count
                    prediction_frames.append(output)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    expected_rows = len(target) * len(METHODS) * (finite_repetitions * (len(budgets) - 1) + 1)
    if len(predictions) != expected_rows:
        raise RuntimeError(f"Few-shot prediction population mismatch: {len(predictions)} != {expected_rows}")
    metrics = build_metrics(predictions)
    summary = summarize(metrics)
    write_parquet(predictions, PREDICTIONS_PATH)
    write_csv(metrics, METRICS_PATH)
    write_csv(summary, SUMMARY_PATH)
    save_figure(summary)

    passing = summary[
        summary["adaptation_recovery_gate_passed"]
        & ~summary["method"].isin(["frozen_source", "target_training_mean"])
    ][["budget_label", "method", "r2_median", "rmse_median"]].to_dict(orient="records")
    freeze_basis = {
        "analysis_label": "posthoc_bart_few_shot_adaptation",
        "protocol": protocol,
        "protocol_config_freeze_sha256": sha256(PROTOCOL_FREEZE_PATH),
        "local_outer_split_freeze_id": local_freeze["freeze_id"],
        "local_outer_split_freeze_sha256": sha256(LOCAL_FREEZE_PATH),
        "source_alignment_sha256": sha256(SOURCE_PATH),
        "source_conventional_sha256": sha256(SOURCE_CONVENTIONAL_PATH),
        "target_alignment_sha256": sha256(TARGET_PATH),
        "target_conventional_sha256": sha256(TARGET_CONVENTIONAL_PATH),
        "source_model_sha256": sha256(SOURCE_MODEL_PATH),
        "source_model_name": artifact["model_name"],
        "source_model_features": features,
        "source_prediction_recreation_max_abs_difference": artifact[
            "prediction_recreation_max_abs_difference"
        ],
        "source_prediction_recreation_tolerance": 1e-6,
        "target_outcomes_used_only_inside_adaptation_folds": True,
        "held_out_outcomes_used_for_fitting_or_selection": False,
        "script_sha256": sha256(Path(__file__)),
        "sklearn_version": sklearn.__version__,
    }
    freeze_id = "phase5-bart-few-shot-" + canonical_hash(freeze_basis)[:12]
    freeze = {
        "freeze_id": freeze_id,
        "created_at": utc_now(),
        "status": "frozen_posthoc_spatially_buffered_few_shot_adaptation",
        "freeze_basis": freeze_basis,
        "freeze_basis_sha256": canonical_hash(freeze_basis),
        "outputs": {
            "predictions": {"path": str(PREDICTIONS_PATH.relative_to(ROOT)), "sha256": sha256(PREDICTIONS_PATH), "rows": len(predictions)},
            "metrics": {"path": str(METRICS_PATH.relative_to(ROOT)), "sha256": sha256(METRICS_PATH), "rows": len(metrics)},
            "summary": {"path": str(SUMMARY_PATH.relative_to(ROOT)), "sha256": sha256(SUMMARY_PATH), "rows": len(summary)},
            "figure": {"path": str(FIGURE_PATH.relative_to(ROOT)), "sha256": sha256(FIGURE_PATH)},
        },
        "passing_adaptation_method_budgets": passing,
        "interpretation_limit": "Post-hoc target-domain adaptation; cannot replace zero-shot transfer or justify wall-to-wall mapping.",
    }
    write_json(FREEZE_PATH, freeze)
    print(summary[[
        "budget_label", "method", "repetitions", "r2_median", "rmse_median",
        "spearman_r_median", "adaptation_recovery_gate_passed",
    ]].to_string(index=False))
    print(f"\nFrozen {freeze_id}; prediction rows: {len(predictions):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
