#!/usr/bin/env python3
"""Run reviewer-requested diagnostics from frozen predictions and cohorts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "outputs" / "reports"
FIGURE_DIR = ROOT / "outputs" / "figures"
PNG_DIR = ROOT / "output" / "premeeting_extended_figures_png"

CAIRN_COHORT = ROOT / "data/processed/phase22_cairngorms_surface_targets.parquet"
CAIRN_SURFACE = ROOT / "data/processed/phase29_cairngorms_v2_surface_predictions.parquet"
CAIRN_HEIGHT = ROOT / "data/processed/phase31_cairngorms_v2_height_strict_predictions.parquet"
SAVEL_COHORT = ROOT / "data/processed/phase26_ahn4_deciduous_cohort.parquet"
SAVEL_PREDICTIONS = ROOT / "data/processed/phase27_savelsbos_v2_predictions.parquet"
TRANSFER_PREDICTIONS = ROOT / "data/processed/phase32_scotland_netherlands_transfer_predictions.parquet"

CAIRN_TARGETS = [
    "canopy_surface_sd_m",
    "canopy_surface_cv",
    "canopy_surface_rcv",
    "canopy_rumple",
    "canopy_open_fraction",
    "canopy_height_kurtosis",
]

TARGET_LABELS = {
    "canopy_mean_height_m": "Mean height (m)",
    "canopy_p95_height_m": "P95 height (m)",
    "canopy_surface_sd_m": "Height SD (m)",
    "canopy_surface_cv": "Height CV",
    "canopy_surface_rcv": "Robust CV",
    "canopy_rumple": "Rumple",
    "canopy_open_fraction": "Canopy openings",
    "canopy_height_kurtosis": "Height kurtosis",
    "ahn4_height_sd_m": "Height SD (m)",
    "ahn4_height_cv": "Height CV",
    "ahn4_entropy": "Return-height entropy",
    "ahn4_pulse_penetration": "Pulse penetration",
    "ahn4_sigma_z": "Sigma-z",
    "mean_height_m": "Mean height (m)",
    "p95_height_m": "P95 height (m)",
    "within_cell_height_sd_m": "Height SD (m)",
    "height_cv": "Height CV",
}

COLORS = {
    "raw": "#188977",
    "adjusted": "#8a4fa3",
    "fit": "#d66b1f",
    "mlp": "#188977",
    "unet": "#3f6fb6",
    "cairngorms": "#247a67",
    "savelsbos": "#7c55b6",
}


def style_axes(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(labelsize=8)


def save_figure(figure: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
    figure.savefig(FIGURE_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    figure.savefig(PNG_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def equal_block_weights(blocks: Iterable[object]) -> np.ndarray:
    values = np.asarray(list(blocks))
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse].astype(np.float64)
    return weights / weights.sum()


def weighted_metrics(
    observed: Iterable[float], predicted: Iterable[float], blocks: Iterable[object]
) -> dict[str, float]:
    observed = np.asarray(list(observed), dtype=np.float64)
    predicted = np.asarray(list(predicted), dtype=np.float64)
    blocks = np.asarray(list(blocks))
    valid = np.isfinite(observed) & np.isfinite(predicted)
    observed, predicted, blocks = observed[valid], predicted[valid], blocks[valid]
    weights = equal_block_weights(blocks)
    error = predicted - observed
    mean = float(np.sum(weights * observed))
    denominator = float(np.sum(weights * np.square(observed - mean)))
    return {
        "n": int(len(observed)),
        "rmse": float(np.sqrt(np.sum(weights * np.square(error)))),
        "mae": float(np.sum(weights * np.abs(error))),
        "bias": float(np.sum(weights * error)),
        "r2": float(1 - np.sum(weights * np.square(error)) / denominator)
        if denominator > 0
        else np.nan,
        "observed_mean": mean,
        "observed_variance": denominator,
    }


def safe_spearman(first: np.ndarray, second: np.ndarray) -> float:
    valid = np.isfinite(first) & np.isfinite(second)
    if valid.sum() < 3 or np.ptp(first[valid]) <= 0 or np.ptp(second[valid]) <= 0:
        return np.nan
    return float(spearmanr(first[valid], second[valid]).statistic)


def safe_pearson(first: np.ndarray, second: np.ndarray) -> float:
    valid = np.isfinite(first) & np.isfinite(second)
    if valid.sum() < 3 or np.ptp(first[valid]) <= 0 or np.ptp(second[valid]) <= 0:
        return np.nan
    return float(pearsonr(first[valid], second[valid]).statistic)


def block_correlation(
    frame: pd.DataFrame, first: str, second: str, block: str
) -> tuple[float, float, int]:
    grouped = frame.groupby(block, as_index=False)[[first, second]].mean()
    return (
        safe_spearman(grouped[first].to_numpy(), grouped[second].to_numpy()),
        safe_pearson(grouped[first].to_numpy(), grouped[second].to_numpy()),
        int(len(grouped)),
    )


def binned_means(frame: pd.DataFrame, x: str, columns: list[str], bins: int = 12) -> pd.DataFrame:
    scoped = frame[[x, *columns]].dropna().copy()
    scoped["bin"] = pd.qcut(scoped[x], q=bins, duplicates="drop")
    return scoped.groupby("bin", observed=False)[[x, *columns]].mean().reset_index(drop=True)


def markdown_table(frame: pd.DataFrame, digits: int = 3) -> str:
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.{digits}f}"
            )
    headers = [str(column) for column in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for values in display.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(map(str, values)) + " |")
    return "\n".join(lines)


def build_cairngorms_adjustment(
    cohort: pd.DataFrame, predictions: pd.DataFrame
) -> pd.DataFrame:
    scoped = predictions[
        (predictions.version == "v2")
        & (predictions.model_family == "tessera")
        & predictions.target.isin(CAIRN_TARGETS)
    ]
    keys = ["row_id", "fold", "target", "spatial_block"]
    raw = scoped[scoped.variant == "raw"][keys + ["observed", "predicted"]].rename(
        columns={"observed": "raw_observed", "predicted": "raw_predicted"}
    )
    adjusted = scoped[scoped.variant == "height_adjusted"][
        keys + ["observed", "predicted"]
    ].rename(
        columns={
            "observed": "adjusted_observed",
            "predicted": "adjusted_predicted",
        }
    )
    frame = raw.merge(adjusted, on=keys, validate="one_to_one")
    frame = frame.merge(
        cohort[
            [
                "row_id",
                "canopy_mean_height_m",
                "canopy_p95_height_m",
                "canopy_open_fraction",
                "bng_x",
                "bng_y",
            ]
        ],
        on="row_id",
        validate="many_to_one",
    )
    frame["height_fitted_value"] = frame.raw_observed - frame.adjusted_observed
    error = np.max(
        np.abs(
            frame.raw_observed
            - frame.height_fitted_value
            - frame.adjusted_observed
        )
    )
    if error > 1e-5:
        raise RuntimeError(f"Cairngorms residual reconstruction failed: {error}")
    return frame


def build_savelsbos_adjustment(
    cohort: pd.DataFrame, predictions: pd.DataFrame
) -> pd.DataFrame:
    scoped = predictions[
        (predictions.model == "tessera_v2")
        & (predictions.target != "ahn4_p95_height_m")
    ]
    keys = ["row_id", "fold", "target", "evaluation_block"]
    raw = scoped[scoped.target_variant == "raw"][keys + ["observed", "predicted"]].rename(
        columns={"observed": "raw_observed", "predicted": "raw_predicted"}
    )
    adjusted = scoped[scoped.target_variant == "height_adjusted"][
        keys + ["observed", "predicted"]
    ].rename(
        columns={
            "observed": "adjusted_observed",
            "predicted": "adjusted_predicted",
        }
    )
    frame = raw.merge(adjusted, on=keys, validate="one_to_one")
    frame = frame.merge(
        cohort[["row_id", "ahn4_mean_height_m", "ahn4_p95_height_m"]],
        on="row_id",
        validate="many_to_one",
    )
    frame["height_fitted_value"] = frame.raw_observed - frame.adjusted_observed
    return frame


def plot_height_covariation(frame: pd.DataFrame, height: str, suffix: str) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(12.2, 7.0))
    for letter, target, axis in zip("abcdef", CAIRN_TARGETS, axes.ravel()):
        scoped = frame[frame.target == target]
        x = scoped[height].to_numpy(dtype=np.float64)
        y = scoped.raw_observed.to_numpy(dtype=np.float64)
        axis.hexbin(x, y, gridsize=38, mincnt=1, bins="log", cmap="viridis")
        binned = binned_means(scoped, height, ["raw_observed", "height_fitted_value"])
        axis.plot(
            binned[height],
            binned.raw_observed,
            color="#252525",
            marker="o",
            markersize=3,
            linewidth=1.4,
            label="Observed bin mean",
        )
        axis.plot(
            binned[height],
            binned.height_fitted_value,
            color=COLORS["fit"],
            linewidth=1.8,
            label="Out-of-fold height fit",
        )
        axis.set_title(f"{letter}  {TARGET_LABELS[target]}", loc="left", fontsize=9, fontweight="bold")
        axis.set_xlabel(TARGET_LABELS[height], fontsize=8)
        axis.set_ylabel("Observed outcome", fontsize=8)
        style_axes(axis)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.suptitle(
        f"Cairngorms structural outcomes in relation to {TARGET_LABELS[height].lower()}",
        y=0.995,
        fontsize=12,
        fontweight="bold",
    )
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=2,
        frameon=False,
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.89))
    save_figure(figure, f"premeeting_extended_height_covariation_{suffix}")


def adjustment_correlation_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target in CAIRN_TARGETS:
        target_frame = frame[frame.target == target]
        for fold, fold_frame in target_frame.groupby("fold"):
            for variant, outcome in [
                ("raw", "raw_observed"),
                ("height_adjusted", "adjusted_observed"),
            ]:
                for height in ["canopy_mean_height_m", "canopy_p95_height_m"]:
                    rho, correlation, blocks = block_correlation(
                        fold_frame, outcome, height, "spatial_block"
                    )
                    rows.append(
                        {
                            "target": target,
                            "variant": variant,
                            "height_predictor": height,
                            "fold": int(fold),
                            "blocks": blocks,
                            "spearman": rho,
                            "pearson": correlation,
                        }
                    )
    fold_table = pd.DataFrame(rows)
    macro = (
        fold_table.groupby(["target", "variant", "height_predictor"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            blocks=("blocks", "sum"),
            spearman=("spearman", "mean"),
            pearson=("pearson", "mean"),
        )
        .assign(fold=-1)
    )
    return pd.concat([fold_table, macro], ignore_index=True, sort=False)


def plot_adjustment_correlations(table: pd.DataFrame) -> None:
    macro = table[table.fold == -1]
    columns = [
        ("raw", "canopy_mean_height_m"),
        ("raw", "canopy_p95_height_m"),
        ("height_adjusted", "canopy_mean_height_m"),
        ("height_adjusted", "canopy_p95_height_m"),
    ]
    matrix = np.asarray(
        [
            [
                macro[
                    (macro.target == target)
                    & (macro.variant == variant)
                    & (macro.height_predictor == height)
                ].iloc[0].spearman
                for variant, height in columns
            ]
            for target in CAIRN_TARGETS
        ]
    )
    figure, axis = plt.subplots(figsize=(8.4, 4.6))
    image = axis.imshow(matrix, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1), aspect="auto")
    axis.set_yticks(np.arange(len(CAIRN_TARGETS)))
    axis.set_yticklabels([TARGET_LABELS[value] for value in CAIRN_TARGETS], fontsize=8)
    axis.set_xticks(np.arange(4))
    axis.set_xticklabels(
        ["Raw vs mean", "Raw vs p95", "Adjusted vs mean", "Adjusted vs p95"],
        rotation=20,
        ha="right",
        fontsize=8,
    )
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            color = "white" if abs(matrix[row, column]) > 0.55 else "#252525"
            axis.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center", fontsize=8, color=color)
    axis.set_title("Equal-fold Spearman correlation across held-out 1 km block means", fontsize=10, fontweight="bold")
    figure.colorbar(image, ax=axis, fraction=0.035, pad=0.03, label="Spearman correlation")
    figure.tight_layout()
    save_figure(figure, "premeeting_extended_height_adjustment_residual_association")


def plot_raw_adjusted_distributions(frame: pd.DataFrame) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(12.2, 6.5))
    for letter, target, axis in zip("abcdef", CAIRN_TARGETS, axes.ravel()):
        scoped = frame[frame.target == target]
        raw = scoped.raw_observed.to_numpy(dtype=np.float64)
        adjusted = scoped.adjusted_observed.to_numpy(dtype=np.float64)
        low = min(np.quantile(raw, 0.005), np.quantile(adjusted, 0.005))
        high = max(np.quantile(raw, 0.995), np.quantile(adjusted, 0.995))
        bins = np.linspace(low, high, 45)
        axis.hist(raw, bins=bins, density=True, histtype="step", linewidth=1.8, color=COLORS["raw"], label="Original")
        axis.hist(adjusted, bins=bins, density=True, histtype="step", linewidth=1.8, color=COLORS["adjusted"], label="Height-adjusted")
        axis.axvline(0, color="#777777", linestyle=":", linewidth=0.8)
        axis.set_title(f"{letter}  {TARGET_LABELS[target]}", loc="left", fontsize=9, fontweight="bold")
        axis.set_ylabel("Density", fontsize=8)
        style_axes(axis)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.suptitle("Original and height-adjusted Cairngorms target distributions", y=0.995, fontsize=12, fontweight="bold")
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=2, frameon=False, fontsize=8)
    figure.tight_layout(rect=(0, 0, 1, 0.89))
    save_figure(figure, "premeeting_extended_raw_adjusted_distributions")


def adjustment_landscape_summary(cairn: pd.DataFrame, savel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    specifications = [
        (
            "Cairngorms",
            cairn,
            "spatial_block",
            "canopy_mean_height_m",
            "canopy_p95_height_m",
        ),
        (
            "Savelsbos",
            savel,
            "evaluation_block",
            "ahn4_mean_height_m",
            "ahn4_p95_height_m",
        ),
    ]
    for site, frame, block, mean_height, p95_height in specifications:
        for target, target_frame in frame.groupby("target"):
            fold_rows = []
            for fold, fold_frame in target_frame.groupby("fold"):
                weights = equal_block_weights(fold_frame[block])
                raw = fold_frame.raw_observed.to_numpy(dtype=np.float64)
                adjusted = fold_frame.adjusted_observed.to_numpy(dtype=np.float64)
                raw_mean = float(np.sum(weights * raw))
                adjusted_mean = float(np.sum(weights * adjusted))
                raw_variance = float(np.sum(weights * np.square(raw - raw_mean)))
                adjusted_variance = float(np.sum(weights * np.square(adjusted - adjusted_mean)))
                raw_metrics = weighted_metrics(raw, fold_frame.raw_predicted, fold_frame[block])
                adjusted_metrics = weighted_metrics(adjusted, fold_frame.adjusted_predicted, fold_frame[block])
                raw_mean_rho, _, _ = block_correlation(fold_frame, "raw_observed", mean_height, block)
                raw_p95_rho, _, _ = block_correlation(fold_frame, "raw_observed", p95_height, block)
                adjusted_mean_rho, _, _ = block_correlation(fold_frame, "adjusted_observed", mean_height, block)
                adjusted_p95_rho, _, _ = block_correlation(fold_frame, "adjusted_observed", p95_height, block)
                fold_rows.append(
                    {
                        "raw_variance": raw_variance,
                        "adjusted_variance": adjusted_variance,
                        "variance_removed": 1 - adjusted_variance / raw_variance,
                        "raw_r2": raw_metrics["r2"],
                        "adjusted_r2": adjusted_metrics["r2"],
                        "raw_mean_height_spearman": raw_mean_rho,
                        "raw_p95_height_spearman": raw_p95_rho,
                        "adjusted_mean_height_spearman": adjusted_mean_rho,
                        "adjusted_p95_height_spearman": adjusted_p95_rho,
                    }
                )
            folds = pd.DataFrame(fold_rows)
            pooled_raw = weighted_metrics(
                target_frame.raw_observed,
                target_frame.raw_predicted,
                target_frame[block],
            )
            pooled_adjusted = weighted_metrics(
                target_frame.adjusted_observed,
                target_frame.adjusted_predicted,
                target_frame[block],
            )
            rows.append(
                {
                    "site": site,
                    "target": target,
                    "n": int(target_frame.row_id.nunique()),
                    "q05": float(target_frame.raw_observed.quantile(0.05)),
                    "q95": float(target_frame.raw_observed.quantile(0.95)),
                    "raw_r2_pooled": pooled_raw["r2"],
                    "adjusted_r2_pooled": pooled_adjusted["r2"],
                    "raw_r2_equal_fold": float(folds.raw_r2.mean()),
                    "adjusted_r2_equal_fold": float(folds.adjusted_r2.mean()),
                    **{
                        column: float(folds[column].mean())
                        for column in folds.columns
                        if column not in {"raw_r2", "adjusted_r2"}
                    },
                }
            )
    return pd.DataFrame(rows)


def plot_landscape_adjustment(table: pd.DataFrame) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12.2, 7.2))
    for column, site in enumerate(["Cairngorms", "Savelsbos"]):
        scoped = table[table.site == site].copy()
        labels = [TARGET_LABELS.get(value, value) for value in scoped.target]
        x = np.arange(len(scoped))
        axes[0, column].bar(x, scoped.variance_removed, color="#737b80")
        axes[0, column].axhline(0, color="#252525", linewidth=0.8)
        axes[0, column].set_ylabel("Fraction of variance removed")
        axes[0, column].set_title(f"{'ab'[column]}  {site}: effect of observed height", loc="left", fontweight="bold")
        width = 0.36
        axes[1, column].bar(x - width / 2, scoped.raw_r2_pooled, width, color=COLORS["raw"], label="Original")
        axes[1, column].bar(x + width / 2, scoped.adjusted_r2_pooled, width, color=COLORS["adjusted"], label="Height-adjusted")
        axes[1, column].axhline(0, color="#252525", linewidth=0.8)
        axes[1, column].set_ylabel("Pooled held-out $R^2$")
        axes[1, column].set_title(f"{'cd'[column]}  {site}: TESSERA v2 prediction", loc="left", fontweight="bold")
        for axis in axes[:, column]:
            axis.set_xticks(x)
            axis.set_xticklabels(labels, rotation=32, ha="right", fontsize=8)
            style_axes(axis)
    handles, labels = axes[1, 0].get_legend_handles_labels()
    figure.suptitle("Height adjustment behaves differently across outcomes and landscapes", y=0.995, fontsize=12, fontweight="bold")
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=2, frameon=False, fontsize=8)
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    save_figure(figure, "premeeting_extended_landscape_height_adjustment_comparison")


def transfer_metrics_and_plot(predictions: pd.DataFrame) -> pd.DataFrame:
    scoped = predictions[
        (predictions.feature_panel == "context")
        & (predictions.model == "transfer")
        & (predictions.training_site != predictions.evaluation_site)
    ]
    directions = [("cairngorms", "savelsbos"), ("savelsbos", "cairngorms")]
    targets = ["mean_height_m", "p95_height_m", "within_cell_height_sd_m", "height_cv"]
    figure, axes = plt.subplots(2, 4, figsize=(13.6, 6.5))
    rows: list[dict[str, object]] = []
    for row, (training, evaluation) in enumerate(directions):
        for column, target in enumerate(targets):
            axis = axes[row, column]
            frame = scoped[
                (scoped.training_site == training)
                & (scoped.evaluation_site == evaluation)
                & (scoped.target == target)
            ]
            metrics = weighted_metrics(frame.observed, frame.predicted, frame.evaluation_block)
            rows.append(
                {
                    "training_site": training,
                    "evaluation_site": evaluation,
                    "target": target,
                    **metrics,
                }
            )
            axis.hexbin(frame.observed, frame.predicted, gridsize=34, mincnt=1, bins="log", cmap="viridis")
            values = np.concatenate([frame.observed.to_numpy(), frame.predicted.to_numpy()])
            low, high = np.quantile(values[np.isfinite(values)], [0.005, 0.995])
            axis.plot([low, high], [low, high], "--", color="#c33d32", linewidth=1)
            axis.text(
                0.03,
                0.96,
                f"$R^2$ = {metrics['r2']:.3f}\nRMSE = {metrics['rmse']:.3f}",
                transform=axis.transAxes,
                va="top",
                fontsize=7,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.8},
            )
            axis.set_title(f"{'abcd'[column] if row == 0 else 'efgh'[column]}  {TARGET_LABELS[target]}", loc="left", fontsize=9, fontweight="bold")
            axis.set_xlabel("Observed", fontsize=8)
            axis.set_ylabel("Transferred prediction", fontsize=8)
            style_axes(axis)
        axes[row, 0].annotate(
            f"Train {training.capitalize()} → evaluate {evaluation.capitalize()}",
            xy=(-0.30, 0.5),
            xycoords="axes fraction",
            rotation=90,
            va="center",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
    figure.suptitle("Observed versus predicted outcomes under direct cross-landscape transfer", y=0.995, fontsize=12, fontweight="bold")
    figure.tight_layout(rect=(0.04, 0, 1, 0.95))
    save_figure(figure, "premeeting_extended_transfer_observed_predicted")
    return pd.DataFrame(rows)


def height_error_regimes(predictions: pd.DataFrame) -> pd.DataFrame:
    model_labels = {
        "tessera_v2_mlp_5x5": "5 x 5 MLP",
        "tessera_v2_unet_strict": "strict U-Net",
    }
    predictions = predictions[predictions.model.isin(model_labels)].copy()
    rows: list[dict[str, object]] = []
    for target, target_frame in predictions.groupby("target"):
        reference = target_frame[target_frame.model == "tessera_v2_mlp_5x5"]
        height_bins = pd.qcut(reference.observed, q=10, duplicates="drop", retbins=True)[1]
        opening_bins = pd.qcut(reference.opening_fraction, q=5, duplicates="drop", retbins=True)[1]
        cover_bins = pd.qcut(reference.mean_cover_50m, q=5, duplicates="drop", retbins=True)[1]
        for model, model_frame in target_frame.groupby("model"):
            frame = model_frame.copy()
            frame["height_bin"] = pd.cut(frame.observed, bins=height_bins, include_lowest=True, labels=False)
            frame["opening_bin"] = pd.cut(frame.opening_fraction, bins=opening_bins, include_lowest=True, labels=False)
            frame["cover_bin"] = pd.cut(frame.mean_cover_50m, bins=cover_bins, include_lowest=True, labels=False)
            for regime, column in [
                ("observed_height_decile", "height_bin"),
                ("opening_quintile", "opening_bin"),
                ("canopy_cover_quintile", "cover_bin"),
                ("spatial_fold", "fold"),
            ]:
                for value, group in frame.groupby(column):
                    metrics = weighted_metrics(group.observed, group.predicted, group.spatial_block)
                    rows.append(
                        {
                            "target": target,
                            "model": model,
                            "model_label": model_labels[model],
                            "regime": regime,
                            "bin": int(value),
                            "opening_mean": float(group.opening_fraction.mean()),
                            "cover_mean": float(group.mean_cover_50m.mean()),
                            **metrics,
                        }
                    )
    return pd.DataFrame(rows)


def plot_height_error_regimes(table: pd.DataFrame) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(15.6, 6.8))
    targets = ["canopy_mean_height_m", "canopy_p95_height_m"]
    regimes = ["observed_height_decile", "opening_quintile", "canopy_cover_quintile", "spatial_fold"]
    titles = ["Observed-height decile", "Canopy-opening quintile", "Canopy-cover quintile", "Spatial fold"]
    for row, target in enumerate(targets):
        for column, (regime, title) in enumerate(zip(regimes, titles)):
            axis = axes[row, column]
            scoped = table[(table.target == target) & (table.regime == regime)]
            for model, color in [("tessera_v2_mlp_5x5", COLORS["mlp"]), ("tessera_v2_unet_strict", COLORS["unet"])]:
                values = scoped[scoped.model == model].sort_values("bin")
                if regime == "opening_quintile":
                    x = values.opening_mean
                elif regime == "canopy_cover_quintile":
                    x = values.cover_mean
                else:
                    x = values.bin + 1
                axis.plot(x, values.rmse, marker="o", linewidth=1.7, color=color, label=values.model_label.iloc[0])
            axis.set_title(f"{'abcdefgh'[row * 4 + column]}  {TARGET_LABELS[target]} by {title.lower()}", loc="left", fontsize=9, fontweight="bold")
            if regime == "opening_quintile":
                xlabel = "Mean opening fraction"
            elif regime == "canopy_cover_quintile":
                xlabel = "Mean supplied canopy cover"
            else:
                xlabel = title
            axis.set_xlabel(xlabel, fontsize=8)
            axis.set_ylabel("RMSE (m)", fontsize=8)
            style_axes(axis)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.suptitle("Canopy-height error varies across forest and spatial regimes", y=0.995, fontsize=12, fontweight="bold")
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=2, frameon=False, fontsize=8)
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    save_figure(figure, "premeeting_extended_height_error_regimes")


def fold_stability(surface: pd.DataFrame, height: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    scoped = surface[
        (surface.version == "v2")
        & (surface.model_family == "tessera")
        & surface.target.isin(CAIRN_TARGETS)
    ]
    for (variant, target, fold), frame in scoped.groupby(["variant", "target", "fold"]):
        metrics = weighted_metrics(frame.observed, frame.predicted, frame.spatial_block)
        rows.append({"analysis": f"surface_{variant}", "model": "TESSERA v2 MLP", "target": target, "fold": int(fold), **metrics})
    height = height[height.model.isin(["tessera_v2_mlp_5x5", "tessera_v2_unet_strict"])]
    for (model, target, fold), frame in height.groupby(["model", "target", "fold"]):
        metrics = weighted_metrics(frame.observed, frame.predicted, frame.spatial_block)
        rows.append({"analysis": "height", "model": model, "target": target, "fold": int(fold), **metrics})
    return pd.DataFrame(rows)


def heatmap(axis: plt.Axes, matrix: np.ndarray, rows: list[str], title: str) -> None:
    image = axis.imshow(matrix, cmap="RdYlGn", vmin=-0.2, vmax=1.0, aspect="auto")
    axis.set_yticks(np.arange(len(rows)))
    axis.set_yticklabels(rows, fontsize=7)
    axis.set_xticks(np.arange(5))
    axis.set_xticklabels(["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"], fontsize=7)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            axis.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=7, color="white" if value < 0.12 or value > 0.78 else "#252525")
    axis.set_title(title, loc="left", fontsize=9, fontweight="bold")
    return image


def plot_fold_stability(table: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 5.2))
    for axis, analysis, title in zip(axes[:2], ["surface_raw", "surface_height_adjusted"], ["a  Original heterogeneity", "b  Height-adjusted heterogeneity"]):
        scoped = table[table.analysis == analysis]
        matrix = np.asarray([[scoped[(scoped.target == target) & (scoped.fold == fold)].iloc[0].r2 for fold in range(5)] for target in CAIRN_TARGETS])
        image = heatmap(axis, matrix, [TARGET_LABELS[value] for value in CAIRN_TARGETS], title)
    height_rows = []
    height_labels = []
    for model, model_label in [("tessera_v2_mlp_5x5", "MLP"), ("tessera_v2_unet_strict", "U-Net")]:
        for target in ["canopy_mean_height_m", "canopy_p95_height_m"]:
            scoped = table[(table.analysis == "height") & (table.model == model) & (table.target == target)]
            height_rows.append([scoped[scoped.fold == fold].iloc[0].r2 for fold in range(5)])
            height_labels.append(f"{model_label}: {TARGET_LABELS[target]}")
    image = heatmap(axes[2], np.asarray(height_rows), height_labels, "c  Canopy height")
    color_axis = figure.add_axes([0.955, 0.18, 0.012, 0.62])
    figure.colorbar(image, cax=color_axis, label="$R^2$")
    figure.suptitle("Fold-level performance under buffered spatial evaluation", y=0.995, fontsize=12, fontweight="bold")
    figure.subplots_adjust(left=0.10, right=0.93, bottom=0.12, top=0.88, wspace=0.48)
    save_figure(figure, "premeeting_extended_fold_stability")


def savelsbos_inventory() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "cairngorms_outcome": "Height SD",
                "savelsbos_status": "Available and used",
                "ahn4_product": "ahn4_10m_std_normalized_height.tif",
                "comparison": "Return-height SD in AHN4; not the same as 1 m CHM surface-height SD.",
            },
            {
                "cairngorms_outcome": "Height CV",
                "savelsbos_status": "Available and used",
                "ahn4_product": "ahn4_10m_coeff_var_normalized_height.tif",
                "comparison": "Return-height CV in AHN4; approximately matched to the Cairngorms transfer target.",
            },
            {
                "cairngorms_outcome": "Robust CV",
                "savelsbos_status": "Derivable but not used",
                "ahn4_product": "AHN4 p25, median and p75 normalized-height rasters",
                "comparison": "Can calculate (p75-p25)/median, but it remains return-based rather than CHM-surface based.",
            },
            {
                "cairngorms_outcome": "Rumple",
                "savelsbos_status": "Not derivable from current products",
                "ahn4_product": "No rumple or high-resolution CHM layer in the selected 10 m metric set",
                "comparison": "Needs a high-resolution canopy surface or point cloud.",
            },
            {
                "cairngorms_outcome": "Canopy openings",
                "savelsbos_status": "No exact equivalent",
                "ahn4_product": "ahn4_10m_pulse_penetration_ratio.tif",
                "comparison": "Pulse penetration is related to openness but is not the fraction of 1 m CHM cells below 2 m.",
            },
            {
                "cairngorms_outcome": "Height kurtosis",
                "savelsbos_status": "Available but not used",
                "ahn4_product": "ahn4_10m_kurto_normalized_height.tif",
                "comparison": "Return-height kurtosis is available; it is not CHM surface-height kurtosis.",
            },
        ]
    )


def code_checks() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "check": "Cairngorms canopy cover",
                "finding": "The cohort averages supplied lidar_Cov across the 25 constituent 10 m cells and requires the 50 m mean to be at least 0.50.",
                "status": "Partly resolved",
                "remaining": "The upstream point-level formula and vegetation-height threshold used to create lidar_Cov are not implemented in this repository and require the lidarSHM production definition.",
                "evidence": "scripts/run_cairngorms_heterogeneity.py:340-364; configs/cairngorms_heterogeneity.yaml:15-21",
            },
            {
                "check": "Invalid TESSERA pixels",
                "finding": "Pixels are invalid when outside the source tile, assigned a non-finite or non-positive embedding scale, or excluded by the TESSERA land mask. V2 preparation also requires a valid centre and at least 20 valid pixels in the 5 x 5 patch.",
                "status": "Resolved",
                "remaining": "None for the implemented validity rule.",
                "evidence": "scripts/run_cairngorms_spatial_heads.py:476-506; scripts/prepare_cairngorms_v2_surface.py:238-245",
            },
            {
                "check": "2,400 of 2,500 CHM cells",
                "finding": "Each 50 m unit contains 2,500 one-metre CHM cells. At least 2,400 finite, non-negative cells are required, retaining at least 96% of the surface support.",
                "status": "Resolved",
                "remaining": "The 96% threshold is a stated quality-control choice, not a value estimated from the data.",
                "evidence": "scripts/prepare_cairngorms_spatial_unet.py:73-89,212-220; configs/cairngorms_spatial_unet.yaml:25-34",
            },
            {
                "check": "24 of 25 LiDAR-metric cells",
                "finding": "The corrected return cohort averages supplied 10 m metrics over each 5 x 5 unit and requires at least 24 valid cells, retaining at least 96% of the return-metric support.",
                "status": "Resolved",
                "remaining": "None for the implemented threshold.",
                "evidence": "configs/cairngorms_point_cloud_cairngorms_corrected_lidar.yaml:29-45; scripts/prepare_cairngorms_point_cloud_cairngorms_corrected_lidar.py:119-147",
            },
            {
                "check": "Different Cairngorms cohorts",
                "finding": "Surface targets use the 1 m CHM and require 2,400 valid pixels; return targets use corrected 10 m point-return metrics and require 24 valid cells. Their separate missing-data and range rules produce 17,316 and 16,831 units, respectively.",
                "status": "Resolved",
                "remaining": "Keep the cohorts and target families explicitly separated in the Methods.",
                "evidence": "data/processed/phase22_cairngorms_surface_targets.parquet; data/processed/phase25_cairngorms_corrected_lidar_targets.parquet",
            },
            {
                "check": "CHM versus point-return products",
                "finding": "Both come from the 2023 Cairngorms survey. The CHM represents the outer canopy surface at 1 m resolution, whereas the 10 m return rasters summarise the distribution of LiDAR returns after the corrected greater-than-1.3 m filter.",
                "status": "Resolved",
                "remaining": "Do not describe these as two independent LiDAR surveys.",
                "evidence": "configs/cairngorms_spatial_unet.yaml; configs/cairngorms_point_cloud_cairngorms_corrected_lidar.yaml",
            },
            {
                "check": "N16",
                "finding": "N16 is the Natura 2000 habitat class 'Broad-leaved deciduous woodland' in the selection table used to identify candidate Dutch sites.",
                "status": "Resolved",
                "remaining": "Name the classification table and class at first use.",
                "evidence": "data/external/ahn4_selection/Natura2000_end2021_HABITATCLASS.csv; configs/ahn4_replication_ahn4_deciduous.yaml:6-12",
            },
            {
                "check": "AHN4",
                "finding": "AHN4 is the fourth Dutch national airborne laser-scanning survey. The public product provides 25 ecosystem-structure metrics at 10 m resolution; Savelsbos was restricted to the 17 February 2021 acquisition.",
                "status": "Resolved",
                "remaining": "Define Actueel Hoogtebestand Nederland and cite the AHN4 data paper and Zenodo record.",
                "evidence": "configs/ahn4_replication_ahn4_deciduous.yaml:3-28; https://doi.org/10.5281/zenodo.8422129",
            },
            {
                "check": "Savelsbos broadleaf mask",
                "finding": "Broadleaf forest is TOP10NL terrain class 'bos: loofbos'. Units require at least 80% broadleaf coverage and at least 95% valid LiDAR coverage over their broadleaf cells.",
                "status": "Resolved",
                "remaining": "Name TOP10NL and the exact class in the Methods.",
                "evidence": "configs/ahn4_replication_ahn4_deciduous.yaml:29-47; scripts/prepare_ahn4_replication_ahn4_cohort.py:261-339",
            },
        ]
    )


def write_report(
    correlations: pd.DataFrame,
    landscape: pd.DataFrame,
    transfer: pd.DataFrame,
    errors: pd.DataFrame,
    folds: pd.DataFrame,
    inventory: pd.DataFrame,
    checks: pd.DataFrame,
) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / "premeeting_extended_diagnostics.md"
    macro_correlations = correlations[correlations.fold == -1]
    raw_abs = float(macro_correlations[macro_correlations.variant == "raw"].spearman.abs().mean())
    adjusted_abs = float(macro_correlations[macro_correlations.variant == "height_adjusted"].spearman.abs().mean())
    positive_folds = folds.groupby(["analysis", "model", "target"]).r2.apply(lambda values: int((values > 0).sum())).reset_index(name="positive_folds")
    tallest = errors[(errors.regime == "observed_height_decile") & (errors.bin == 9)]
    lines = [
        "# Pre-meeting extended diagnostics",
        "",
        "These diagnostics use frozen held-out predictions and do not refit the TESSERA models.",
        "",
        "## Height adjustment",
        "",
        f"Across the six Cairngorms outcomes and two height predictors, the mean absolute held-out block-level Spearman association changed from {raw_abs:.3f} before adjustment to {adjusted_abs:.3f} afterwards.",
        "The adjustment uses observed LiDAR mean and p95 height and is therefore an explanatory diagnostic rather than a deployable prediction step.",
        "",
        "## Height-error regimes",
        "",
    ]
    for row in tallest.itertuples(index=False):
        lines.append(f"- {TARGET_LABELS[row.target]}, {row.model_label}: tallest-decile RMSE {row.rmse:.3f} m; bias {row.bias:.3f} m.")
    lines.extend(
        [
            "",
            "Height errors were also stratified by the supplied LiDAR canopy-cover measure. This is distinct from the CHM-derived opening fraction and therefore provides a separate density check.",
            "",
            "## Fold stability",
            "",
            markdown_table(positive_folds),
            "",
            "## Cross-landscape adjustment comparison",
            "",
            markdown_table(landscape),
            "",
            "## Direct-transfer fits",
            "",
            markdown_table(transfer),
            "",
            "## Savelsbos target inventory",
            "",
            markdown_table(inventory),
            "",
            "## Code checks",
            "",
            markdown_table(checks),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    cairn_cohort = pd.read_parquet(CAIRN_COHORT)
    cairn_surface = pd.read_parquet(CAIRN_SURFACE)
    cairn_height = pd.read_parquet(CAIRN_HEIGHT)
    cairn_height = cairn_height.merge(
        cairn_cohort[["row_id", "mean_cover_50m"]],
        on="row_id",
        how="left",
        validate="many_to_one",
    )
    if cairn_height.mean_cover_50m.isna().any():
        raise RuntimeError("Canopy-cover values are missing after joining the height predictions to the Cairngorms cohort.")
    savel_cohort = pd.read_parquet(SAVEL_COHORT)
    savel_predictions = pd.read_parquet(SAVEL_PREDICTIONS)
    transfer_predictions = pd.read_parquet(TRANSFER_PREDICTIONS)

    cairn_adjustment = build_cairngorms_adjustment(cairn_cohort, cairn_surface)
    savel_adjustment = build_savelsbos_adjustment(savel_cohort, savel_predictions)
    correlations = adjustment_correlation_table(cairn_adjustment)
    landscape = adjustment_landscape_summary(cairn_adjustment, savel_adjustment)
    transfer = transfer_metrics_and_plot(transfer_predictions)
    errors = height_error_regimes(cairn_height)
    folds = fold_stability(cairn_surface, cairn_height)
    inventory = savelsbos_inventory()
    checks = code_checks()

    plot_height_covariation(cairn_adjustment, "canopy_mean_height_m", "mean_height")
    plot_height_covariation(cairn_adjustment, "canopy_p95_height_m", "p95_height")
    plot_raw_adjusted_distributions(cairn_adjustment)
    plot_adjustment_correlations(correlations)
    plot_landscape_adjustment(landscape)
    plot_height_error_regimes(errors)
    plot_fold_stability(folds)

    tables = {
        "premeeting_extended_height_adjustment_correlations.csv": correlations,
        "premeeting_extended_landscape_adjustment.csv": landscape,
        "premeeting_extended_transfer_metrics.csv": transfer,
        "premeeting_extended_height_error_regimes.csv": errors,
        "premeeting_extended_fold_stability.csv": folds,
        "premeeting_extended_savelsbos_target_inventory.csv": inventory,
        "premeeting_extended_code_checks.csv": checks,
    }
    for name, table in tables.items():
        table.to_csv(TABLE_DIR / name, index=False)
    report = write_report(correlations, landscape, transfer, errors, folds, inventory, checks)
    print(
        json.dumps(
            {
                "report": str(report.relative_to(ROOT)),
                "figures": len(list(PNG_DIR.glob("premeeting_extended_*.png"))),
                "tables": len(tables),
                "cairngorms_rows": int(cairn_cohort.row_id.nunique()),
                "savelsbos_rows": int(savel_cohort.row_id.nunique()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    run()
