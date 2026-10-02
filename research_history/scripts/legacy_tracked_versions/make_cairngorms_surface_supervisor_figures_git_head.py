#!/usr/bin/env python3
"""Create figures for the valid Cairngorms canopy-surface supervisor briefing."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "outputs/figures"
METRICS_PATH = ROOT / "outputs/tables/phase21_cairngorms_surface_metrics.csv"
MACRO_PATH = ROOT / "outputs/tables/phase21_cairngorms_surface_macro_metrics.csv"
PREDICTIONS_PATH = ROOT / "data/processed/phase21_cairngorms_surface_predictions.parquet"
TARGETS_PATH = ROOT / "data/processed/phase21_cairngorms_surface_targets.parquet"


TARGETS = [
    "canopy_top_height_m",
    "canopy_surface_sd_m",
    "canopy_surface_cv",
    "canopy_surface_rcv",
    "canopy_rumple",
    "canopy_open_fraction",
]
ADJUSTED = [
    "canopy_surface_sd_m",
    "canopy_surface_cv",
    "canopy_surface_rcv",
    "canopy_rumple",
    "canopy_open_fraction",
]
LABELS = {
    "canopy_top_height_m": "Top height",
    "canopy_surface_sd_m": "Height SD",
    "canopy_surface_cv": "Height CV",
    "canopy_surface_rcv": "Robust CV",
    "canopy_rumple": "Rumple",
    "canopy_open_fraction": "Openings",
}
UNITS = {
    "canopy_top_height_m": "m",
    "canopy_surface_sd_m": "m",
    "canopy_surface_cv": "ratio",
    "canopy_surface_rcv": "ratio",
    "canopy_rumple": "ratio",
    "canopy_open_fraction": "fraction",
}
MODELS = ["conventional", "tessera", "fused"]
MODEL_LABELS = {
    "conventional": "Sentinel + terrain",
    "tessera": "TESSERA",
    "fused": "Combined",
}
COLORS = {
    "conventional": "#4C78A8",
    "tessera": "#148F88",
    "fused": "#E8872D",
}


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 180,
            "savefig.dpi": 300,
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def make_region_map() -> None:
    frame = pd.read_parquet(TARGETS_PATH)
    frame = frame[frame["phase21_valid"]].copy()
    region_colors = ["#3B6FB6", "#5D9E6F", "#C6933B", "#A65A73", "#6E65A8"]
    fig, axis = plt.subplots(figsize=(5.5, 4.0), constrained_layout=True)
    legend_handles = []
    for region, group in frame.groupby("spatial_region", sort=True):
        color = region_colors[int(region)]
        axis.scatter(
            group["bng_x"] / 1000,
            group["bng_y"] / 1000,
            s=2.2,
            alpha=0.62,
            color=color,
            linewidths=0,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.6,
                markersize=7,
                label=f"Region {int(region) + 1} (n={len(group):,})",
            )
        )
    axis.set_xlabel("British National Grid easting (km)")
    axis.set_ylabel("British National Grid northing (km)")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(color="#D8D8D8", linewidth=0.45, alpha=0.7)
    axis.legend(
        handles=legend_handles,
        title="Held-out test region",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        handletextpad=0.5,
        labelspacing=0.7,
    )
    x0, x1 = axis.get_xlim()
    y0, y1 = axis.get_ylim()
    axis.annotate(
        "N",
        xy=(x1 - 0.03 * (x1 - x0), y1 - 0.03 * (y1 - y0)),
        xytext=(x1 - 0.03 * (x1 - x0), y1 - 0.17 * (y1 - y0)),
        ha="center",
        va="bottom",
        fontsize=9,
        arrowprops={"arrowstyle": "-|>", "color": "black", "lw": 1.0},
    )
    save(fig, "cairngorms_surface_supervisor_regions")


def make_performance_figure() -> None:
    macro = pd.read_csv(MACRO_PATH)
    fold_metrics = pd.read_csv(METRICS_PATH)
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.7), constrained_layout=True)
    panels = [
        (axes[0], TARGETS, "Canopy-height-model outcomes"),
        (
            axes[1],
            [f"{target}_height_adjusted" for target in ADJUSTED],
            "Variation beyond average canopy height",
        ),
    ]
    width = 0.23
    for axis, targets, title in panels:
        positions = np.arange(len(targets), dtype=float)
        for model_index, model in enumerate(MODELS):
            offset = (model_index - 1) * width
            means = []
            for target in targets:
                row = macro[(macro["model"] == model) & (macro["target"] == target)]
                means.append(float(row["r2"].iloc[0]))
            axis.bar(
                positions + offset,
                means,
                width=width,
                color=COLORS[model],
                label=MODEL_LABELS[model],
                zorder=2,
            )
            for target_index, target in enumerate(targets):
                values = fold_metrics[
                    (fold_metrics["model"] == model)
                    & (fold_metrics["target"] == target)
                ]["r2"].to_numpy()
                axis.scatter(
                    np.full(len(values), positions[target_index] + offset),
                    values,
                    s=7,
                    color="black",
                    alpha=0.50,
                    linewidths=0,
                    zorder=3,
                )
        axis.axhline(0, color="black", linewidth=0.7)
        axis.set_xticks(
            positions,
            [LABELS[target.replace("_height_adjusted", "")] for target in targets],
            rotation=25,
            ha="right",
        )
        axis.set_ylabel(r"Mean held-out $R^2$")
        axis.set_title(title)
        axis.grid(axis="y", color="#D8D8D8", linewidth=0.5, alpha=0.8, zorder=0)
    axes[0].set_ylim(-0.20, 0.92)
    axes[1].set_ylim(0, 0.84)
    axes[1].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3)
    save(fig, "cairngorms_surface_supervisor_performance")


def make_prediction_figure() -> None:
    predictions = pd.read_parquet(PREDICTIONS_PATH)
    predictions = predictions[
        (predictions["model"] == "tessera") & predictions["target"].isin(TARGETS)
    ]
    macro = pd.read_csv(MACRO_PATH)
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 6.0), constrained_layout=True)
    image = None
    for panel, (axis, target) in enumerate(zip(axes.ravel(), TARGETS)):
        subset = predictions[predictions["target"] == target]
        observed = subset["observed"].to_numpy()
        predicted = subset["predicted"].to_numpy()
        rumple_scaled = target == "canopy_rumple"
        if rumple_scaled:
            observed = 1000.0 * (observed - 1.0)
            predicted = 1000.0 * (predicted - 1.0)
        low = float(min(observed.min(), predicted.min()))
        high = float(max(observed.max(), predicted.max()))
        padding = max((high - low) * 0.035, 1e-6)
        image = axis.hexbin(
            observed,
            predicted,
            gridsize=55,
            mincnt=1,
            cmap="viridis",
            norm=LogNorm(),
            linewidths=0,
            rasterized=True,
        )
        axis.plot([low, high], [low, high], color="#B22222", linestyle="--", linewidth=0.9)
        axis.set_xlim(low - padding, high + padding)
        axis.set_ylim(low - padding, high + padding)
        unit = UNITS[target]
        if rumple_scaled:
            axis.set_xlabel(r"Observed $1000\times(\mathrm{rumple}-1)$")
            axis.set_ylabel(r"Predicted $1000\times(\mathrm{rumple}-1)$")
        else:
            unit_text = f" ({unit})" if unit == "m" else ""
            axis.set_xlabel(f"Observed{unit_text}")
            axis.set_ylabel(f"TESSERA prediction{unit_text}")
        axis.set_title(f"{chr(97 + panel)}  {LABELS[target]}", loc="left", fontweight="bold")
        metric = macro[(macro["model"] == "tessera") & (macro["target"] == target)].iloc[0]
        rmse_unit = " m" if unit == "m" else ""
        rmse_text = (
            rf"RMSE = {metric['rmse']:.4f} (original scale)"
            if rumple_scaled
            else rf"RMSE = {metric['rmse']:.3f}{rmse_unit}"
        )
        axis.text(
            0.04,
            0.96,
            rf"Mean $R^2$ = {metric['r2']:.3f}" + "\n" + rmse_text,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2.0},
        )
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes, shrink=0.72, pad=0.015)
        colorbar.set_label("Observations per hexagon (log scale)")
    save(fig, "cairngorms_surface_supervisor_predictions")


def main() -> None:
    style()
    make_region_map()
    make_performance_figure()
    make_prediction_figure()


if __name__ == "__main__":
    main()
