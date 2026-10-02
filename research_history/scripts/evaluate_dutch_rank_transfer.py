#!/usr/bin/env python3
"""Evaluate absolute and rank transfer across the expanded Dutch forest set."""

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
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import r2_score

try:
    from evaluate_dutch_reference_assisted import (
        balanced_sample,
        equal_block_weights,
        fit_scaled_ridge,
        load_frame,
        selected_sites,
        sha256,
        source_plus_target_fit,
        source_weights,
        write_csv,
        write_json,
    )
except ModuleNotFoundError:
    from scripts.evaluate_dutch_reference_assisted import (
        balanced_sample,
        equal_block_weights,
        fit_scaled_ridge,
        load_frame,
        selected_sites,
        sha256,
        source_plus_target_fit,
        source_weights,
        write_csv,
        write_json,
    )


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/dutch_rank_transfer.yaml"
TARGET_LABELS = {
    "ahn4_p95_height_m": "P95 height",
    "ahn4_height_sd_m": "Height SD",
    "ahn4_height_cv": "Height CV",
    "ahn4_entropy": "Return entropy",
    "ahn4_pulse_penetration": "Pulse penetration",
    "ahn4_sigma_z": "Sigma-z",
}
ADAPTATION_METHODS = [
    "source_offset",
    "target_only_ridge",
    "source_plus_target_ridge",
    "target_reference_mean",
]


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["phase39_dutch_rank_transfer"]


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def safe_spearman(observed: np.ndarray, predicted: np.ndarray) -> float:
    if len(observed) < 3 or np.ptp(observed) <= 1e-12 or np.ptp(predicted) <= 1e-12:
        return math.nan
    return float(spearmanr(observed, predicted).statistic)


def pattern_metrics(test: pd.DataFrame, predicted: np.ndarray) -> dict[str, float]:
    observed = test["observed"].to_numpy(dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    blocks = test["fold_block"].astype(str)
    weights = equal_block_weights(blocks)
    residual = predicted - observed
    observed_mean = float(np.average(observed, weights=weights))
    predicted_mean = float(np.average(predicted, weights=weights))
    centred_error = (predicted - predicted_mean) - (observed - observed_mean)
    block_frame = pd.DataFrame(
        {"block": blocks.to_numpy(), "observed": observed, "predicted": predicted}
    ).groupby("block", as_index=False).mean()
    return {
        "rmse": float(np.sqrt(np.average(residual**2, weights=weights))),
        "mae": float(np.average(np.abs(residual), weights=weights)),
        "r2": float(r2_score(observed, predicted, sample_weight=weights)),
        "bias": float(np.average(residual, weights=weights)),
        "unit_spearman": safe_spearman(observed, predicted),
        "block_spearman": safe_spearman(
            block_frame["observed"].to_numpy(), block_frame["predicted"].to_numpy()
        ),
        "centred_rmse": float(np.sqrt(np.average(centred_error**2, weights=weights))),
        "test_rows": int(len(test)),
        "test_blocks": int(len(block_frame)),
    }


def exact_budget_available(pool_rows: int, budget: int) -> bool:
    return pool_rows >= budget


def metric_record(
    base: dict[str, Any],
    test: pd.DataFrame,
    observed: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, Any]:
    metric_frame = test[["fold_block"]].copy()
    metric_frame["observed"] = observed
    return {**base, **pattern_metrics(metric_frame, predicted)}


def worker(index: int) -> None:
    config = load_config()
    sites = selected_sites(config)
    if index >= len(sites):
        raise RuntimeError(f"No site at worker index {index}")
    frame, features = load_frame(config)
    site = sites.iloc[index]
    target_site = str(site["site_code"])
    source = frame[frame["site_code"] != target_site].copy()
    target_frame = frame[frame["site_code"] == target_site].copy()
    targets = [str(value) for value in config["evaluation"]["targets"]]
    budgets = [int(value) for value in config["evaluation"]["budgets"]]
    repetitions = int(config["evaluation"]["finite_repetitions"])
    alpha = float(config["evaluation"]["ridge_alpha"])
    seed_base = int(config["evaluation"]["sampling_seed"]) + index * 100000
    source_weight = source_weights(source)
    source_models: dict[str, tuple[Any, Any, float]] = {}
    for target_name in targets:
        scaler, model = fit_scaled_ridge(source, features, target_name, source_weight, alpha)
        source_models[target_name] = (
            scaler,
            model,
            float(np.average(source[target_name], weights=source_weight)),
        )
    fold_root = ROOT / str(config["inputs"]["folds"])
    metric_records: list[dict[str, Any]] = []
    prediction_records: list[pd.DataFrame] = []
    for fold in range(int(config["evaluation"]["fold_count"])):
        with np.load(fold_root / target_site / f"fold_{fold}.npz") as definition:
            train_ids = definition["train_row_ids"].astype(np.int64)
            test_ids = definition["test_row_ids"].astype(np.int64)
        target_train = target_frame[target_frame["row_id"].isin(train_ids)].copy()
        target_test = target_frame[target_frame["row_id"].isin(test_ids)].copy()
        if target_train.empty or target_test.empty:
            raise RuntimeError(f"Empty split for {target_site} fold {fold}")
        pool_rows = len(target_train)
        eligible_200 = exact_budget_available(
            pool_rows, int(config["evaluation"]["exact_200_minimum_pool"])
        )
        test_x = target_test[features].to_numpy(dtype=np.float64)
        for target_name in targets:
            scaler, source_model, source_mean = source_models[target_name]
            observed = target_test[target_name].to_numpy(dtype=np.float64)
            source_prediction = source_model.predict(scaler.transform(test_x))
            common = {
                "target_site": target_site,
                "forest_group": str(site["forest_group"]),
                "fold": fold,
                "target": target_name,
                "reference_pool_rows": pool_rows,
                "eligible_200": eligible_200,
            }
            metric_records.append(
                metric_record(
                    {
                        **common,
                        "budget": 0,
                        "replicate": 0,
                        "method": "source_ridge",
                        "reference_rows": 0,
                        "reference_blocks": 0,
                    },
                    target_test,
                    observed,
                    source_prediction,
                )
            )
            metric_records.append(
                metric_record(
                    {
                        **common,
                        "budget": 0,
                        "replicate": 0,
                        "method": "source_mean",
                        "reference_rows": 0,
                        "reference_blocks": 0,
                    },
                    target_test,
                    observed,
                    np.full(len(target_test), source_mean),
                )
            )
            prediction_records.append(
                pd.DataFrame(
                    {
                        "row_id": target_test["row_id"].to_numpy(dtype=np.int64),
                        "target_site": target_site,
                        "forest_group": str(site["forest_group"]),
                        "fold": fold,
                        "fold_block": target_test["fold_block"].to_numpy(),
                        "rd_x": target_test["rd_x"].to_numpy(dtype=np.float64),
                        "rd_y": target_test["rd_y"].to_numpy(dtype=np.float64),
                        "target": target_name,
                        "observed": observed,
                        "predicted": source_prediction,
                    }
                )
            )
        for budget_position, budget in enumerate(budgets[1:], start=1):
            if not exact_budget_available(pool_rows, budget):
                continue
            for replicate in range(repetitions):
                rng = np.random.default_rng(
                    seed_base + fold * 10000 + budget_position * 100 + replicate
                )
                selected = balanced_sample(
                    target_train.index.to_numpy(dtype=np.int64),
                    target_train,
                    budget,
                    rng,
                )
                reference = target_train.loc[selected].copy()
                reference_x = reference[features].to_numpy(dtype=np.float64)
                target_weight = equal_block_weights(reference["sampling_block"])
                for target_name in targets:
                    scaler, source_model, _ = source_models[target_name]
                    observed = target_test[target_name].to_numpy(dtype=np.float64)
                    source_prediction = source_model.predict(scaler.transform(test_x))
                    source_reference = source_model.predict(scaler.transform(reference_x))
                    reference_y = reference[target_name].to_numpy(dtype=np.float64)
                    offset = float(np.mean(reference_y - source_reference))
                    target_scaler, target_model = fit_scaled_ridge(
                        reference, features, target_name, target_weight, alpha
                    )
                    combined_scaler, combined_model = source_plus_target_fit(
                        source, reference, features, target_name, alpha
                    )
                    outputs = {
                        "target_reference_mean": np.full(len(target_test), float(reference_y.mean())),
                        "source_offset": source_prediction + offset,
                        "target_only_ridge": target_model.predict(target_scaler.transform(test_x)),
                        "source_plus_target_ridge": combined_model.predict(
                            combined_scaler.transform(test_x)
                        ),
                    }
                    common = {
                        "target_site": target_site,
                        "forest_group": str(site["forest_group"]),
                        "fold": fold,
                        "target": target_name,
                        "budget": budget,
                        "replicate": replicate,
                        "reference_rows": len(reference),
                        "reference_blocks": int(reference["sampling_block"].nunique()),
                        "reference_pool_rows": pool_rows,
                        "eligible_200": eligible_200,
                    }
                    for method, prediction in outputs.items():
                        metric_records.append(
                            metric_record(
                                {**common, "method": method},
                                target_test,
                                observed,
                                prediction,
                            )
                        )
    metrics = pd.DataFrame(metric_records)
    predictions = pd.concat(prediction_records, ignore_index=True)
    if len(predictions) != len(target_frame) * len(targets):
        raise RuntimeError(
            f"Zero-shot prediction coverage failed for {target_site}: "
            f"{len(predictions)} != {len(target_frame) * len(targets)}"
        )
    worker_root = ROOT / str(config["outputs"]["worker_directory"])
    write_csv(worker_root / f"metrics_{index:02d}_{target_site}.csv", metrics)
    atomic_parquet(worker_root / f"predictions_{index:02d}_{target_site}.parquet", predictions)
    print(
        f"phase39 worker {index + 1}/{len(sites)} complete: {target_site}; "
        f"{len(metrics):,} metric rows; {len(predictions):,} zero-shot predictions",
        flush=True,
    )


def summarize_metrics(metrics: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = metrics.copy()
    metrics["site_fold"] = (
        metrics["target_site"].astype(str) + ":" + metrics["fold"].astype(str)
    )
    full_max = int(config["evaluation"]["full_network_max_budget"])
    cohorts = [
        ("all_site_folds_to_100", metrics[metrics["budget"] <= full_max]),
        ("matched_200_eligible_folds", metrics[metrics["eligible_200"].astype(bool)]),
    ]
    summaries: list[pd.DataFrame] = []
    for cohort_name, cohort in cohorts:
        grouped = cohort.groupby(["target", "budget", "method"], sort=True).agg(
            sites=("target_site", "nunique"),
            site_folds=("site_fold", "nunique"),
            rmse_median=("rmse", "median"),
            rmse_mean=("rmse", "mean"),
            r2_median=("r2", "median"),
            r2_mean=("r2", "mean"),
            unit_spearman_median=("unit_spearman", "median"),
            unit_spearman_mean=("unit_spearman", "mean"),
            block_spearman_median=("block_spearman", "median"),
            block_spearman_mean=("block_spearman", "mean"),
            centred_rmse_median=("centred_rmse", "median"),
            bias_mean=("bias", "mean"),
        ).reset_index()
        grouped.insert(0, "cohort", cohort_name)
        summaries.append(grouped)
    summary = pd.concat(summaries, ignore_index=True)
    zero = metrics[(metrics["budget"] == 0) & (metrics["method"] == "source_ridge")]
    zero_summary = zero.groupby("target", sort=True).agg(
        sites=("target_site", "nunique"),
        site_folds=("site_fold", "nunique"),
        r2_median=("r2", "median"),
        unit_spearman_median=("unit_spearman", "median"),
        block_spearman_median=("block_spearman", "median"),
        positive_r2_fraction=("r2", lambda values: float((values > 0).mean())),
        positive_unit_rank_fraction=("unit_spearman", lambda values: float((values > 0).mean())),
        positive_block_rank_fraction=("block_spearman", lambda values: float((values > 0).mean())),
        rmse_median=("rmse", "median"),
        centred_rmse_median=("centred_rmse", "median"),
    ).reset_index()
    site_summary = metrics.groupby(
        ["target_site", "forest_group", "target", "budget", "method"], sort=True
    ).agg(
        folds=("fold", "nunique"),
        r2_median=("r2", "median"),
        unit_spearman_median=("unit_spearman", "median"),
        block_spearman_median=("block_spearman", "median"),
        rmse_median=("rmse", "median"),
    ).reset_index()
    return summary, zero_summary, site_summary


def make_figures(
    summary: pd.DataFrame,
    zero_summary: pd.DataFrame,
    site_summary: pd.DataFrame,
    predictions: pd.DataFrame,
    output_directory: Path,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    target_order = list(TARGET_LABELS)
    zero = zero_summary.set_index("target").reindex(target_order)
    x = np.arange(len(target_order))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.bar(x - width, zero["r2_median"], width, label="$R^2$", color="#404C5A")
    ax.bar(x, zero["unit_spearman_median"], width, label="50 m rank", color="#2A8C6A")
    ax.bar(x + width, zero["block_spearman_median"], width, label="500 m spatial rank", color="#D07A43")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, [TARGET_LABELS[value] for value in target_order], rotation=25, ha="right")
    ax.set_ylabel("Median held-out score across site-folds")
    ax.set_title("a  Zero-shot absolute and relative-pattern transfer", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3)
    fig.tight_layout()
    fig.savefig(output_directory / "phase39_zero_shot_rank_vs_absolute.png", dpi=260, bbox_inches="tight")
    plt.close(fig)

    primary = summary[summary["cohort"] == "all_site_folds_to_100"]
    budgets = [0, 2, 5, 10, 25, 50, 100]
    methods = ["source_offset", "target_only_ridge", "source_plus_target_ridge"]
    colors = {
        "source_offset": "#247A52",
        "target_only_ridge": "#B05C35",
        "source_plus_target_ridge": "#3A6EA5",
    }
    for metric, filename, ylabel in [
        ("r2_median", "phase39_reference_absolute_curve.png", "Median held-out $R^2$"),
        ("unit_spearman_median", "phase39_reference_rank_curve.png", "Median 50 m Spearman rank"),
        (
            "block_spearman_median",
            "phase39_reference_block_rank_curve.png",
            "Median 500 m Spearman spatial rank",
        ),
    ]:
        fig, axes = plt.subplots(2, 3, figsize=(13, 7.6), sharex=True)
        for axis, target_name in zip(axes.flat, target_order, strict=True):
            target_frame = primary[primary["target"] == target_name]
            baseline = zero.loc[target_name, metric.replace("_median", "_median")]
            axis.axhline(baseline, color="#404C5A", linewidth=1.5, label="source model, 0 refs")
            for method in methods:
                method_frame = target_frame[target_frame["method"] == method].set_index("budget")
                values = method_frame[metric].reindex(budgets[1:])
                axis.plot(budgets[1:], values, marker="o", color=colors[method], label=method.replace("_", " "))
            axis.axhline(0, color="black", linestyle="--", linewidth=0.7)
            axis.set_title(TARGET_LABELS[target_name], loc="left", fontweight="bold")
            axis.grid(alpha=0.2)
        for axis in axes[-1, :]:
            axis.set_xlabel("Target-forest LiDAR reference units")
        for axis in axes[:, 0]:
            axis.set_ylabel(ylabel)
        handles, legend_labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc="lower center", ncol=2, frameon=False, fontsize=8)
        fig.tight_layout(rect=(0, 0.09, 1, 1))
        fig.savefig(output_directory / filename, dpi=250, bbox_inches="tight")
        plt.close(fig)

    make_matched_reference_figures(summary, output_directory)

    zero_site = site_summary[(site_summary["budget"] == 0) & (site_summary["method"] == "source_ridge")]
    fig, ax = plt.subplots(figsize=(7.6, 6.0))
    for group, group_frame in zero_site.groupby("forest_group"):
        ax.scatter(
            group_frame["r2_median"],
            group_frame["block_spearman_median"],
            s=34,
            alpha=0.7,
            label=group.capitalize(),
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Absolute transfer: median $R^2$")
    ax.set_ylabel("Relative spatial pattern: median 500 m Spearman rank")
    ax.set_title("b  Site-level absolute and spatial-pattern transfer", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(output_directory / "phase39_site_rank_absolute_scatter.png", dpi=260, bbox_inches="tight")
    plt.close(fig)

    selected_sites_list = ["veluwe", "savelsbos"]
    selected_targets = ["ahn4_height_cv", "ahn4_entropy"]
    scoped = predictions[
        predictions["target_site"].isin(selected_sites_list)
        & predictions["target"].isin(selected_targets)
    ].copy()
    if not scoped.empty:
        fig, axes = plt.subplots(4, 2, figsize=(10.5, 15.0))
        rows = []
        for site_code in selected_sites_list:
            for target_name in selected_targets:
                rows.append((site_code, target_name))
        for row, (site_code, target_name) in enumerate(rows):
            group = scoped[(scoped["target_site"] == site_code) & (scoped["target"] == target_name)].copy()
            group["observed_rank"] = rankdata(group["observed"], method="average") / len(group)
            group["predicted_rank"] = rankdata(group["predicted"], method="average") / len(group)
            for column, (field, title) in enumerate([("observed_rank", "LiDAR relative rank"), ("predicted_rank", "TESSERA relative rank")]):
                image = axes[row, column].scatter(
                    group["rd_x"], group["rd_y"], c=group[field], s=7, marker="s", cmap="viridis", vmin=0, vmax=1
                )
                axes[row, column].set_aspect("equal")
                axes[row, column].set_title(
                    f"{site_code.replace('_', ' ').title()} — {TARGET_LABELS[target_name]}\n{title}",
                    fontsize=9,
                )
                axes[row, column].set_xticks([])
                axes[row, column].set_yticks([])
        fig.colorbar(image, ax=axes, fraction=0.018, pad=0.02, label="Within-site percentile")
        fig.suptitle("c  Examples of zero-shot relative spatial pattern", fontweight="bold", y=0.995)
        fig.savefig(output_directory / "phase39_relative_pattern_maps.png", dpi=240, bbox_inches="tight")
        plt.close(fig)


def make_matched_reference_figures(
    summary: pd.DataFrame,
    output_directory: Path,
) -> None:
    """Plot the exact 0--200 curve only on site-folds supporting every budget."""
    target_order = list(TARGET_LABELS)
    matched = summary[summary["cohort"] == "matched_200_eligible_folds"]
    budgets = [0, 2, 5, 10, 25, 50, 100, 200]
    methods = ["source_offset", "target_only_ridge", "source_plus_target_ridge"]
    colors = {
        "source_offset": "#247A52",
        "target_only_ridge": "#B05C35",
        "source_plus_target_ridge": "#3A6EA5",
    }
    specifications = [
        ("r2_median", "phase39_matched_200_absolute_curve.png", "Median held-out $R^2$"),
        (
            "unit_spearman_median",
            "phase39_matched_200_rank_curve.png",
            "Median 50 m Spearman rank",
        ),
        (
            "block_spearman_median",
            "phase39_matched_200_block_rank_curve.png",
            "Median 500 m Spearman spatial rank",
        ),
    ]
    for metric, filename, ylabel in specifications:
        fig, axes = plt.subplots(2, 3, figsize=(13, 7.6), sharex=True)
        for axis, target_name in zip(axes.flat, target_order, strict=True):
            target_frame = matched[matched["target"] == target_name]
            baseline_rows = target_frame[
                (target_frame["method"] == "source_ridge")
                & (target_frame["budget"] == 0)
            ]
            baseline = float(baseline_rows.iloc[0][metric])
            axis.axhline(
                baseline,
                color="#404C5A",
                linewidth=1.5,
                label="source model, 0 refs",
            )
            for method in methods:
                method_frame = target_frame[
                    target_frame["method"] == method
                ].set_index("budget")
                values = method_frame[metric].reindex(budgets[1:])
                axis.plot(
                    budgets[1:],
                    values,
                    marker="o",
                    color=colors[method],
                    label=method.replace("_", " "),
                )
            axis.axhline(0, color="black", linestyle="--", linewidth=0.7)
            axis.set_title(TARGET_LABELS[target_name], loc="left", fontweight="bold")
            axis.grid(alpha=0.2)
        for axis in axes[-1, :]:
            axis.set_xlabel("Target-forest LiDAR reference units")
        for axis in axes[:, 0]:
            axis.set_ylabel(ylabel)
        handles, legend_labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            legend_labels,
            loc="lower center",
            ncol=2,
            frameon=False,
            fontsize=8,
        )
        fig.suptitle(
            "Matched 87 site-folds supporting every reference budget",
            fontsize=11,
            y=0.995,
        )
        fig.tight_layout(rect=(0, 0.09, 1, 0.97))
        fig.savefig(output_directory / filename, dpi=250, bbox_inches="tight")
        plt.close(fig)


def aggregate() -> None:
    config = load_config()
    outputs = config["outputs"]
    result_path = ROOT / str(outputs["result_freeze"])
    if result_path.exists():
        raise RuntimeError("Phase 39 result freeze already exists")
    sites = selected_sites(config)
    worker_root = ROOT / str(outputs["worker_directory"])
    metric_paths = [worker_root / f"metrics_{index:02d}_{row.site_code}.csv" for index, row in enumerate(sites.itertuples(index=False))]
    prediction_paths = [worker_root / f"predictions_{index:02d}_{row.site_code}.parquet" for index, row in enumerate(sites.itertuples(index=False))]
    missing = [str(path) for path in metric_paths + prediction_paths if not path.exists()]
    if missing:
        raise RuntimeError("Missing Phase 39 worker outputs:\n" + "\n".join(missing))
    metrics = pd.concat([pd.read_csv(path) for path in metric_paths], ignore_index=True)
    predictions = pd.concat([pd.read_parquet(path) for path in prediction_paths], ignore_index=True)
    summary, zero_summary, site_summary = summarize_metrics(metrics, config)
    write_csv(ROOT / str(outputs["metrics"]), metrics)
    write_csv(ROOT / str(outputs["summary"]), summary)
    write_csv(ROOT / str(outputs["zero_shot_summary"]), zero_summary)
    write_csv(ROOT / str(outputs["site_summary"]), site_summary)
    atomic_parquet(ROOT / str(outputs["predictions"]), predictions)
    figure_directory = ROOT / str(outputs["figures"])
    make_figures(summary, zero_summary, site_summary, predictions, figure_directory)
    primary = summary[summary["cohort"] == "all_site_folds_to_100"]
    matched = summary[summary["cohort"] == "matched_200_eligible_folds"]
    report = [
        "# Phase 39 Dutch rank and spatial-pattern transfer",
        "",
        "The analysis compares absolute-value transfer with relative within-forest ordering under the frozen Phase 37 spatial folds. The full 20-site network supports exact reference budgets through 100. A matched 87-fold subset supports the exact 200-reference curve.",
        "",
        "## Strict zero-shot source model",
        "",
        "```text",
        zero_summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "```",
        "",
        "## Full-network reference curve through 100",
        "",
        "```text",
        primary[["target", "budget", "method", "sites", "site_folds", "r2_median", "unit_spearman_median", "block_spearman_median", "rmse_median"]].to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "```",
        "",
        "## Matched exact curve through 200",
        "",
        "```text",
        matched[["target", "budget", "method", "sites", "site_folds", "r2_median", "unit_spearman_median", "block_spearman_median", "rmse_median"]].to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "```",
        "",
        "An offset adjustment changes absolute level but preserves the source prediction rank by construction. The target-only and source-plus-target models can change both absolute and rank performance because their fitted relationships use labelled target units.",
    ]
    report_path = ROOT / str(outputs["report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    artifact_paths = [
        ROOT / str(outputs["predictions"]),
        ROOT / str(outputs["metrics"]),
        ROOT / str(outputs["summary"]),
        ROOT / str(outputs["zero_shot_summary"]),
        ROOT / str(outputs["site_summary"]),
        report_path,
    ] + sorted(figure_directory.glob("*.png"))
    write_json(
        result_path,
        {
            "status": "complete",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "sites": int(len(sites)),
            "site_folds": int(metrics[["target_site", "fold"]].drop_duplicates().shape[0]),
            "matched_200_site_folds": int(metrics[metrics["eligible_200"].astype(bool)][["target_site", "fold"]].drop_duplicates().shape[0]),
            "metric_rows": int(len(metrics)),
            "zero_shot_prediction_rows": int(len(predictions)),
            "target_test_labels_used_for_fitting": False,
            "artifacts": {str(path.relative_to(ROOT)): sha256(path) for path in artifact_paths},
        },
    )
    print(zero_summary.to_string(index=False))
    print("phase39 rank-transfer evaluation complete", flush=True)


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
