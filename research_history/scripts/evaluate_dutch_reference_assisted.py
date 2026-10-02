#!/usr/bin/env python3
"""Evaluate reference-assisted transfer on the frozen Phase 37 Dutch cohort."""

from __future__ import annotations

import argparse
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
import yaml
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/dutch_reference_assisted.yaml"
METHODS = [
    "source_ridge",
    "source_mean",
    "target_reference_mean",
    "source_offset",
    "source_affine",
    "target_only_ridge",
    "source_plus_target_ridge",
]
TARGET_LABELS = {
    "ahn4_p95_height_m": "P95 height",
    "ahn4_height_sd_m": "Height SD",
    "ahn4_height_cv": "Height CV",
    "ahn4_entropy": "Return entropy",
    "ahn4_pulse_penetration": "Pulse penetration",
    "ahn4_sigma_z": "Sigma-z",
}


def equal_block_weights(blocks: pd.Series) -> np.ndarray:
    counts = blocks.value_counts()
    weights = blocks.map(1.0 / counts).to_numpy(dtype=np.float64)
    return weights * len(weights) / weights.sum()


def source_weights(frame: pd.DataFrame) -> np.ndarray:
    """Give each source site equal weight and each site block equal weight."""
    weights = np.zeros(len(frame), dtype=np.float64)
    for site, positions in frame.groupby("site_code", sort=True).indices.items():
        local = np.asarray(positions, dtype=np.int64)
        block_weights = equal_block_weights(frame.iloc[local]["fold_block"])
        weights[local] = block_weights / len(local)
    return weights * len(weights) / weights.sum()


def fit_ridge(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    alpha: float,
) -> tuple[StandardScaler, Ridge]:
    scaler = StandardScaler().fit(x, sample_weight=weights)
    model = Ridge(alpha=alpha).fit(scaler.transform(x), y, sample_weight=weights)
    return scaler, model


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["phase38_dutch_reference_assisted"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_frame(config: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    cohort = pd.read_parquet(ROOT / str(config["inputs"]["cohort"]))
    features = pd.read_parquet(ROOT / str(config["inputs"]["features"]))
    frame = cohort.merge(features, on="row_id", validate="one_to_one")
    columns = sorted(column for column in frame if column.startswith("tessera_mean_"))
    if len(columns) != 128:
        raise RuntimeError(f"Expected 128 TESSERA features, found {len(columns)}")
    frame = frame[
        frame["tessera_context_valid"].astype(bool)
        & np.isfinite(frame[columns]).all(axis=1)
    ].reset_index(drop=True)
    return frame, columns


def selected_sites(config: dict[str, Any]) -> pd.DataFrame:
    return pd.read_csv(ROOT / str(config["inputs"]["sites"])).sort_values("selection_order").reset_index(drop=True)


def balanced_sample(
    candidate_indices: np.ndarray,
    frame: pd.DataFrame,
    sample_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if sample_size > len(candidate_indices):
        raise ValueError(f"Requested {sample_size} references from {len(candidate_indices)} candidates")
    if sample_size == len(candidate_indices):
        return candidate_indices.copy()
    groups: dict[str, list[int]] = {}
    for index in candidate_indices:
        groups.setdefault(str(frame.loc[index, "sampling_block"]), []).append(int(index))
    keys = list(groups)
    rng.shuffle(keys)
    for values in groups.values():
        rng.shuffle(values)
    selected: list[int] = []
    depth = 0
    while len(selected) < sample_size:
        available = [key for key in keys if depth < len(groups[key])]
        if not available:
            raise RuntimeError("Spatial sampling exhausted the target reference pool")
        rng.shuffle(available)
        for key in available:
            selected.append(groups[key][depth])
            if len(selected) == sample_size:
                break
        depth += 1
    return np.asarray(selected, dtype=np.int64)


def equal_domain_weights(source_rows: int, target_rows: int) -> np.ndarray:
    if source_rows < 1 or target_rows < 1:
        raise ValueError("Both domains need at least one row")
    weights = np.concatenate([
        np.full(source_rows, 0.5 / source_rows, dtype=np.float64),
        np.full(target_rows, 0.5 / target_rows, dtype=np.float64),
    ])
    return weights


def affine_correction(source_prediction: np.ndarray, observed: np.ndarray) -> tuple[float, float]:
    design = np.column_stack([np.ones(len(source_prediction)), source_prediction])
    intercept, slope = np.linalg.lstsq(design, observed, rcond=None)[0]
    return float(intercept), float(slope)


def metric_values(observed: np.ndarray, predicted: np.ndarray, blocks: np.ndarray) -> dict[str, float]:
    weights = equal_block_weights(pd.Series(blocks))
    residual = predicted - observed
    block_correlations: list[float] = []
    for block in np.unique(blocks):
        selected = blocks == block
        if selected.sum() >= 3 and np.ptp(observed[selected]) > 1e-12 and np.ptp(predicted[selected]) > 1e-12:
            block_correlations.append(float(spearmanr(observed[selected], predicted[selected]).statistic))
    return {
        "rmse": float(np.sqrt(np.average(residual**2, weights=weights))),
        "mae": float(np.average(np.abs(residual), weights=weights)),
        "r2": float(r2_score(observed, predicted, sample_weight=weights)),
        "bias": float(np.average(residual, weights=weights)),
        "spearman": float(np.mean(block_correlations)) if block_correlations else math.nan,
    }


def fit_scaled_ridge(train: pd.DataFrame, features: list[str], target: str, weights: np.ndarray, alpha: float):
    scaler, model = fit_ridge(train[features].to_numpy(dtype=np.float64), train[target].to_numpy(dtype=np.float64), weights, alpha)
    return scaler, model


def source_plus_target_fit(
    source: pd.DataFrame,
    reference: pd.DataFrame,
    features: list[str],
    target: str,
    alpha: float,
):
    source_weights_values = source_weights(source)
    source_weights_values *= 0.5
    target_weights_values = equal_block_weights(reference["sampling_block"])
    target_weights_values *= 0.5
    combined = pd.concat([source, reference], ignore_index=True)
    weights = np.concatenate([source_weights_values, target_weights_values])
    return fit_scaled_ridge(combined, features, target, weights, alpha)


def predictions_for_fold(
    source: pd.DataFrame,
    target_train: pd.DataFrame,
    target_test: pd.DataFrame,
    reference: pd.DataFrame,
    features: list[str],
    target: str,
    alpha: float,
    include_adaptation: bool,
) -> dict[str, np.ndarray]:
    source_weights_values = source_weights(source)
    source_scaler, source_model = fit_scaled_ridge(source, features, target, source_weights_values, alpha)
    test_x = target_test[features].to_numpy(dtype=np.float64)
    source_prediction = source_model.predict(source_scaler.transform(test_x))
    outputs = {
        "source_ridge": source_prediction,
        "source_mean": np.full(len(target_test), np.average(source[target], weights=source_weights_values)),
    }
    if not include_adaptation:
        return outputs
    reference_x = reference[features].to_numpy(dtype=np.float64)
    source_reference_prediction = source_model.predict(source_scaler.transform(reference_x))
    reference_y = reference[target].to_numpy(dtype=np.float64)
    outputs["target_reference_mean"] = np.full(len(target_test), float(reference_y.mean()))
    offset = float(np.mean(reference_y - source_reference_prediction))
    outputs["source_offset"] = source_prediction + offset
    intercept, slope = affine_correction(source_reference_prediction, reference_y)
    outputs["source_affine"] = intercept + slope * source_prediction
    target_weights_values = equal_block_weights(reference["sampling_block"])
    target_scaler, target_model = fit_scaled_ridge(reference, features, target, target_weights_values, alpha)
    outputs["target_only_ridge"] = target_model.predict(target_scaler.transform(test_x))
    combined_scaler, combined_model = source_plus_target_fit(source, reference, features, target, alpha)
    outputs["source_plus_target_ridge"] = combined_model.predict(combined_scaler.transform(test_x))
    return outputs


def worker(index: int) -> None:
    config = load_config()
    sites = selected_sites(config)
    if index >= len(sites):
        raise RuntimeError(f"No selected site at worker index {index}")
    frame, features = load_frame(config)
    site = sites.iloc[index]
    target_site = str(site["site_code"])
    source = frame[frame["site_code"] != target_site].copy()
    target = frame[frame["site_code"] == target_site].copy()
    targets = [str(value) for value in config["evaluation"]["targets"]]
    alpha = float(config["evaluation"]["ridge_alpha"])
    budgets = list(config["evaluation"]["budgets"])
    finite_repetitions = int(config["evaluation"]["finite_repetitions"])
    seed = int(config["evaluation"]["sampling_seed"]) + index * 100000
    fold_root = ROOT / str(config["inputs"]["folds"])
    records: list[dict[str, Any]] = []
    for fold in range(int(config["evaluation"]["fold_count"])):
        with np.load(fold_root / target_site / f"fold_{fold}.npz") as definition:
            train_ids = definition["train_row_ids"].astype(np.int64)
            test_ids = definition["test_row_ids"].astype(np.int64)
        target_train = target[target["row_id"].isin(train_ids)].copy()
        target_test = target[target["row_id"].isin(test_ids)].copy()
        if target_train.empty or target_test.empty:
            raise RuntimeError(f"Empty target train/test split for {target_site} fold {fold}")
        for budget_position, budget_value in enumerate(budgets):
            budget_label = str(budget_value)
            repetitions = 1 if budget_label == "all" else finite_repetitions
            for replicate in range(repetitions):
                if budget_label == "0":
                    reference = target_train.iloc[0:0].copy()
                elif budget_label == "all":
                    reference = target_train.copy()
                else:
                    rng = np.random.default_rng(seed + fold * 1000 + budget_position * 100 + replicate)
                    indices = balanced_sample(
                        target_train.index.to_numpy(dtype=np.int64), target_train, int(budget_value), rng
                    )
                    reference = target_train.loc[indices].copy()
                for target_name in targets:
                    outputs = predictions_for_fold(
                        source,
                        target_train,
                        target_test,
                        reference,
                        features,
                        target_name,
                        alpha,
                        include_adaptation=budget_label != "0",
                    )
                    for method, prediction in outputs.items():
                        metrics = metric_values(
                            target_test[target_name].to_numpy(dtype=np.float64),
                            prediction,
                            target_test["fold_block"].to_numpy(),
                        )
                        records.append({
                            "target_site": target_site,
                            "forest_group": str(site["forest_group"]),
                            "fold": fold,
                            "budget_label": budget_label,
                            "budget_sort": budget_position,
                            "replicate": replicate,
                            "method": method,
                            "target": target_name,
                            "reference_rows": len(reference),
                            "reference_blocks": int(reference["sampling_block"].nunique()) if len(reference) else 0,
                            **metrics,
                        })
    output = pd.DataFrame(records)
    finite_budget_count = sum(str(value) not in {"0", "all"} for value in budgets)
    expected = int(config["evaluation"]["fold_count"]) * len(targets)
    expected *= (
        finite_repetitions * 2
        + finite_repetitions * finite_budget_count * len(METHODS)
        + len(METHODS)
    )
    if len(output) != expected:
        raise RuntimeError(f"Worker row count {len(output)} does not equal expected {expected}")
    output_path = ROOT / str(config["outputs"]["worker_directory"]) / f"worker_{index:02d}_{target_site}.csv"
    write_csv(output_path, output)
    print(f"phase38 worker {index + 1}/{len(sites)} complete: {target_site}; {len(output):,} metric rows", flush=True)


def summarize(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_columns = ["target", "budget_label", "budget_sort", "method"]
    summary = metrics.groupby(group_columns, sort=True).agg(
        repetitions=("replicate", "nunique"),
        fold_records=("fold", "count"),
        reference_rows_median=("reference_rows", "median"),
        reference_blocks_median=("reference_blocks", "median"),
        rmse_mean=("rmse", "mean"),
        rmse_median=("rmse", "median"),
        rmse_q05=("rmse", lambda values: values.quantile(0.05)),
        rmse_q95=("rmse", lambda values: values.quantile(0.95)),
        r2_mean=("r2", "mean"),
        r2_median=("r2", "median"),
        r2_q05=("r2", lambda values: values.quantile(0.05)),
        r2_q95=("r2", lambda values: values.quantile(0.95)),
        bias_mean=("bias", "mean"),
        spearman_mean=("spearman", "mean"),
    ).reset_index()
    baseline = summary[summary["method"].eq("target_reference_mean")][
        ["target", "budget_label", "rmse_median"]
    ].rename(columns={"rmse_median": "reference_mean_rmse_median"})
    summary = summary.merge(baseline, on=["target", "budget_label"], how="left")
    summary["recovery_gate_passed"] = (
        summary["method"].isin(["source_offset", "source_affine", "target_only_ridge", "source_plus_target_ridge"])
        & summary["r2_median"].gt(0)
        & summary["rmse_median"].lt(summary["reference_mean_rmse_median"])
    )
    site_summary = metrics.groupby(
        ["target_site", "forest_group", "target", "budget_label", "budget_sort", "method"], sort=True
    ).agg(
        rmse_mean=("rmse", "mean"),
        r2_mean=("r2", "mean"),
        bias_mean=("bias", "mean"),
        folds=("fold", "nunique"),
    ).reset_index()
    return summary, site_summary


def make_figures(summary: pd.DataFrame, site_summary: pd.DataFrame, output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    plot_methods = ["source_ridge", "source_offset", "source_affine", "target_only_ridge", "source_plus_target_ridge"]
    labels = ["0", "5", "25", "100", "all"]
    colors = {
        "source_ridge": "#3F4B5B",
        "source_offset": "#247A52",
        "source_affine": "#6AAE7B",
        "target_only_ridge": "#B05C35",
        "source_plus_target_ridge": "#3A6EA5",
    }
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True)
    x = np.arange(len(labels))
    for axis, (target, frame) in zip(axes.flat, summary.groupby("target", sort=False), strict=True):
        for method in plot_methods:
            method_frame = frame[frame["method"].eq(method)].set_index("budget_label").reindex(labels)
            axis.plot(x, method_frame["r2_median"], marker="o", color=colors[method], label=method.replace("_", " "))
        axis.axhline(0, color="black", linewidth=0.7, linestyle="--")
        axis.set_title(TARGET_LABELS[target], loc="left", fontweight="bold")
        axis.grid(alpha=0.2)
        axis.set_xticks(x, labels)
    for axis in axes[-1, :]:
        axis.set_xlabel("Target-forest LiDAR reference units")
    for axis in axes[:, 0]:
        axis.set_ylabel("Median held-out $R^2$")
    handles, labels_legend = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_legend, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle("Reference-assisted transfer across the expanded Dutch forest set", y=0.995)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.savefig(output_directory / "phase38_r2_by_reference_budget.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    focus = summary[summary["method"].isin(["source_ridge", "source_offset", "source_affine", "target_only_ridge", "source_plus_target_ridge"])]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for method in plot_methods:
        values = focus[focus["method"].eq(method)].groupby("budget_sort", as_index=True)["rmse_median"].mean()
        values = values.reindex(range(len(labels)))
        axes[0].plot(x, values, marker="o", color=colors[method], label=method.replace("_", " "))
        values_r2 = focus[focus["method"].eq(method)].groupby("budget_sort", as_index=True)["r2_median"].mean().reindex(range(len(labels)))
        axes[1].plot(x, values_r2, marker="o", color=colors[method], label=method.replace("_", " "))
    axes[0].set_ylabel("Mean median RMSE across outcomes")
    axes[1].set_ylabel("Mean median $R^2$ across outcomes")
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.set_xlabel("Target-forest LiDAR reference units")
        axis.grid(alpha=0.2)
    axes[1].axhline(0, color="black", linewidth=0.7, linestyle="--")
    handles, labels_legend = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels_legend, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle("Average reference-assisted performance across six LiDAR outcomes")
    fig.tight_layout(rect=(0, 0.14, 1, 0.96))
    fig.savefig(output_directory / "phase38_average_performance.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    selected = site_summary[
        site_summary["budget_label"].isin(["0", "25", "100"])
        & site_summary["method"].isin(["source_ridge", "source_offset", "source_plus_target_ridge"])
    ].copy()
    selected["label"] = selected["target_site"] + " / " + selected["target"].map(TARGET_LABELS)
    pivot = selected.pivot_table(index="label", columns=["method", "budget_label"], values="rmse_mean")
    if not pivot.empty:
        fig, ax = plt.subplots(figsize=(11, max(5, min(14, 0.16 * len(pivot)))))
        image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="magma")
        ax.set_xticks(range(pivot.shape[1]), [f"{method.replace('_', ' ')}\n{budget}" for method, budget in pivot.columns], rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(pivot.shape[0]), pivot.index, fontsize=6)
        ax.set_title("Held-out RMSE by target forest, outcome, method and reference budget", loc="left", fontweight="bold")
        fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label="RMSE")
        fig.tight_layout()
        fig.savefig(output_directory / "phase38_site_outcome_rmse.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def aggregate() -> None:
    config = load_config()
    outputs = config["outputs"]
    result_path = ROOT / str(outputs["result_freeze"])
    if result_path.exists():
        raise RuntimeError("Phase 38 result freeze already exists")
    sites = selected_sites(config)
    worker_directory = ROOT / str(outputs["worker_directory"])
    paths = [worker_directory / f"worker_{index:02d}_{row.site_code}.csv" for index, row in enumerate(sites.itertuples(index=False))]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise RuntimeError("Missing worker outputs:\n" + "\n".join(missing))
    metrics = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    summary, site_summary = summarize(metrics)
    write_csv(ROOT / str(outputs["metrics"]), metrics)
    write_csv(ROOT / str(outputs["summary"]), summary)
    write_csv(ROOT / str(outputs["site_summary"]), site_summary)
    figure_directory = ROOT / str(outputs["figures"])
    make_figures(summary, site_summary, figure_directory)
    report_lines = [
        "# Phase 38 Dutch reference-assisted transfer",
        "",
        "This post-hoc experiment reused the frozen Phase 37 cohort, public TESSERA v1.0 features, AHN4 outcomes and five spatial folds across 20 Dutch forests. Each target-fold test set was untouched. Reference units were sampled only from the target forest's training side.",
        "",
        "## Pooled summary",
        "",
        "```text",
        summary[["target", "budget_label", "method", "repetitions", "rmse_median", "r2_median", "bias_mean", "recovery_gate_passed"]].to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "```",
        "",
        "## Interpretation",
        "",
        "The zero-reference source Ridge is the strict direct-transfer baseline. Reference-assisted methods use labelled target-forest LiDAR units only from the training side of each fold. A passing recovery gate means positive median R2 and lower median RMSE than the target-reference mean at the same budget; this is an application-oriented reference-assisted result, not zero-shot transfer.",
        "",
        "## Figures",
        "",
        "- `phase38_r2_by_reference_budget.png`: outcome-specific R2 as the number of target references increases.",
        "- `phase38_average_performance.png`: average RMSE and R2 across the six outcomes.",
        "- `phase38_site_outcome_rmse.png`: site- and outcome-level variation in held-out error.",
    ]
    report_path = ROOT / str(outputs["report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    artifacts = {str(path.relative_to(ROOT)): sha256(path) for path in [ROOT / str(outputs["metrics"]), ROOT / str(outputs["summary"]), ROOT / str(outputs["site_summary"]), report_path]}
    artifacts.update({str(path.relative_to(ROOT)): sha256(path) for path in sorted(figure_directory.glob("*.png"))})
    write_json(result_path, {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sites": int(len(sites)),
        "metric_rows": int(len(metrics)),
        "budgets": list(config["evaluation"]["budgets"]),
        "finite_repetitions": int(config["evaluation"]["finite_repetitions"]),
        "source_zero_shot_kept": True,
        "target_test_labels_used_for_fitting": False,
        "artifacts": artifacts,
    })
    print(summary[["target", "budget_label", "method", "rmse_median", "r2_median", "recovery_gate_passed"]].to_string(index=False))
    print("phase38 reference-assisted transfer complete", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["worker", "aggregate"])
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.mode == "worker":
        if args.index is None:
            raise SystemExit("worker mode requires --index")
        worker(args.index)
    else:
        aggregate()


if __name__ == "__main__":
    main()
