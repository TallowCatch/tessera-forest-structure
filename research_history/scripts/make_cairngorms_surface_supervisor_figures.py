#!/usr/bin/env python3
"""Create the figures for the current Cairngorms supervisor report."""

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
PHASE22_METRICS = ROOT / "outputs/tables/phase22_cairngorms_spatial_unet_metrics.csv"
PHASE22_MACRO = ROOT / "outputs/tables/phase22_cairngorms_spatial_unet_macro.csv"
PHASE22_PREDICTIONS = ROOT / "data/processed/phase22_cairngorms_spatial_unet_predictions.parquet"
PHASE22_TARGETS = ROOT / "data/processed/phase22_cairngorms_surface_targets.parquet"
SAVELSBOS_COHORT = ROOT / "data/processed/phase26_ahn4_deciduous_cohort.parquet"
PHASE23_METRICS = ROOT / "outputs/tables/phase23_cairngorms_height_adjusted_metrics.csv"
PHASE23_MACRO = ROOT / "outputs/tables/phase23_cairngorms_height_adjusted_macro.csv"
PHASE14_SAMPLE = ROOT / "data/processed/phase14_scotland_lidar_sample.parquet"
PHASE24_DIRECTORY = ROOT / "outputs/phase24_cairngorms_height_diagnostic"


TARGETS = [
    "canopy_p95_height_m",
    "canopy_surface_sd_m",
    "canopy_surface_cv",
    "canopy_surface_rcv",
    "canopy_rumple",
    "canopy_open_fraction",
]
SURFACE_TARGETS = TARGETS[1:]
LABELS = {
    "canopy_p95_height_m": "P95 height",
    "canopy_surface_sd_m": "Height SD",
    "canopy_surface_cv": "Height CV",
    "canopy_surface_rcv": "Robust CV",
    "canopy_rumple": "Rumple",
    "canopy_open_fraction": "Openings",
}
UNITS = {
    "canopy_p95_height_m": "m",
    "canopy_surface_sd_m": "m",
    "canopy_surface_cv": "ratio",
    "canopy_surface_rcv": "ratio",
    "canopy_rumple": "ratio",
    "canopy_open_fraction": "fraction",
}
MODEL_LABELS = {
    "conventional_mlp": "Sentinel + terrain",
    "tessera_mlp_5x5": "TESSERA",
    "fused_mlp_5x5": "Combined",
    "conventional": "Sentinel + terrain",
    "tessera": "TESSERA",
    "fused": "Combined",
}
COLORS = {
    "conventional_mlp": "#4C78A8",
    "tessera_mlp_5x5": "#148F88",
    "fused_mlp_5x5": "#E8872D",
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


def save(figure: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
    figure.savefig(FIGURE_DIR / f"{stem}.png", bbox_inches="tight")
    plt.close(figure)


def balanced_fold(frame: pd.DataFrame) -> np.ndarray:
    block_x = np.floor((frame["bng_x"].to_numpy() - frame["bng_x"].min()) / 6000).astype(int)
    block_y = np.floor((frame["bng_y"].to_numpy() - frame["bng_y"].min()) / 6000).astype(int)
    return (2 * block_x + block_y) % 5


def make_region_map() -> None:
    frame = pd.read_parquet(PHASE22_TARGETS).copy()
    frame["fold"] = balanced_fold(frame)
    fold_colors = ["#3B6FB6", "#5D9E6F", "#C6933B", "#A65A73", "#6E65A8"]
    figure, axis = plt.subplots(figsize=(5.5, 4.0), constrained_layout=True)
    handles = []
    for fold, group in frame.groupby("fold", sort=True):
        color = fold_colors[int(fold)]
        axis.scatter(
            group["bng_x"] / 1000,
            group["bng_y"] / 1000,
            s=2.0,
            alpha=0.62,
            color=color,
            linewidths=0,
        )
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.6,
                markersize=7,
                label=f"Fold {int(fold) + 1} (n={len(group):,})",
            )
        )
    axis.set_xlabel("British National Grid easting (km)")
    axis.set_ylabel("British National Grid northing (km)")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(color="#D8D8D8", linewidth=0.45, alpha=0.7)
    axis.legend(
        handles=handles,
        title="Held-out spatial fold",
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
    save(figure, "cairngorms_surface_supervisor_regions")


def make_savelsbos_region_map() -> None:
    frame = pd.read_parquet(SAVELSBOS_COHORT).copy()
    fold_colors = ["#3B6FB6", "#5D9E6F", "#C6933B", "#A65A73", "#6E65A8"]
    figure, axis = plt.subplots(figsize=(5.5, 4.0), constrained_layout=True)
    handles = []
    for fold, group in frame.groupby("spatial_fold", sort=True):
        color = fold_colors[int(fold)]
        axis.scatter(
            group["rd_x"] / 1000,
            group["rd_y"] / 1000,
            s=8.0,
            alpha=0.72,
            color=color,
            linewidths=0,
        )
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.6,
                markersize=7,
                label=f"Fold {int(fold) + 1} (n={len(group):,})",
            )
        )
    axis.set_xlabel("RD New easting (km)")
    axis.set_ylabel("RD New northing (km)")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(color="#D8D8D8", linewidth=0.45, alpha=0.7)
    axis.legend(
        handles=handles,
        title="Held-out spatial fold",
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
    save(figure, "savelsbos_surface_supervisor_regions")


def draw_performance_panel(
    axis: plt.Axes,
    macro: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    targets: list[str],
    models: list[str],
    title: str,
) -> None:
    width = 0.23
    positions = np.arange(len(targets), dtype=float)
    for model_index, model in enumerate(models):
        offset = (model_index - 1) * width
        means = [
            float(macro[(macro["model"] == model) & (macro["target"] == target)]["r2"].iloc[0])
            for target in targets
        ]
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
                (fold_metrics["model"] == model) & (fold_metrics["target"] == target)
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
    axis.set_xticks(positions, [LABELS[target] for target in targets], rotation=25, ha="right")
    axis.set_ylabel(r"Mean held-out $R^2$")
    axis.set_title(title)
    axis.grid(axis="y", color="#D8D8D8", linewidth=0.5, alpha=0.8, zorder=0)


def make_performance_figure() -> None:
    phase22_macro = pd.read_csv(PHASE22_MACRO)
    phase22_macro = phase22_macro[phase22_macro["scheme"] == "balanced"]
    phase22_metrics = pd.read_csv(PHASE22_METRICS)
    phase22_metrics = phase22_metrics[phase22_metrics["scheme"] == "balanced"]
    phase23_macro = pd.read_csv(PHASE23_MACRO)
    phase23_metrics = pd.read_csv(PHASE23_METRICS)
    figure, axes = plt.subplots(1, 2, figsize=(9.8, 3.7), constrained_layout=True)
    draw_performance_panel(
        axes[0],
        phase22_macro,
        phase22_metrics,
        TARGETS,
        ["conventional_mlp", "tessera_mlp_5x5", "fused_mlp_5x5"],
        "Observed canopy-height-model outcomes",
    )
    draw_performance_panel(
        axes[1],
        phase23_macro,
        phase23_metrics,
        SURFACE_TARGETS,
        ["conventional", "tessera", "fused"],
        "Variation after accounting for canopy height",
    )
    axes[0].set_ylim(-0.08, 0.88)
    axes[1].set_ylim(-0.02, 0.72)
    axes[1].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3)
    save(figure, "cairngorms_surface_supervisor_performance")


def make_prediction_figure() -> None:
    predictions = pd.read_parquet(PHASE22_PREDICTIONS)
    predictions = predictions[
        (predictions["scheme"] == "balanced")
        & (predictions["model"] == "tessera_mlp_5x5")
        & predictions["target"].isin(TARGETS)
    ]
    macro = pd.read_csv(PHASE22_MACRO)
    macro = macro[(macro["scheme"] == "balanced") & (macro["model"] == "tessera_mlp_5x5")]
    figure, axes = plt.subplots(2, 3, figsize=(9.5, 6.0), constrained_layout=True)
    image = None
    for panel, (axis, target) in enumerate(zip(axes.ravel(), TARGETS)):
        subset = predictions[predictions["target"] == target]
        observed = subset["observed"].to_numpy()
        predicted = subset["predicted"].to_numpy()
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
        unit_text = f" ({unit})" if unit == "m" else ""
        axis.set_xlabel(f"Observed{unit_text}")
        axis.set_ylabel(f"TESSERA prediction{unit_text}")
        axis.set_title(f"{chr(97 + panel)}  {LABELS[target]}", loc="left", fontweight="bold")
        metric = macro[macro["target"] == target].iloc[0]
        rmse_unit = " m" if unit == "m" else ""
        axis.text(
            0.04,
            0.96,
            rf"Mean $R^2$ = {metric['r2']:.3f}"
            + "\n"
            + rf"RMSE = {metric['rmse']:.3f}{rmse_unit}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2.0},
        )
    if image is not None:
        colorbar = figure.colorbar(image, ax=axes, shrink=0.72, pad=0.015)
        colorbar.set_label("Observations per hexagon (log scale)")
    save(figure, "cairngorms_surface_supervisor_predictions")


def make_architecture_figure() -> None:
    macro = pd.read_csv(PHASE22_MACRO)
    macro = macro[macro["scheme"] == "balanced"]
    models = ["tessera_mlp_3x3", "tessera_mlp_5x5", "tessera_cnn_5x5", "tessera_unet"]
    model_labels = ["MLP 3 x 3", "MLP 5 x 5", "CNN 5 x 5", "U-Net 128 x 128"]
    colors = ["#8A8F98", "#148F88", "#4C78A8", "#7A4EAB"]
    targets = ["canopy_p95_height_m", "canopy_surface_cv", "canopy_rumple", "canopy_open_fraction"]
    positions = np.arange(len(targets), dtype=float)
    width = 0.19
    figure, axis = plt.subplots(figsize=(7.8, 3.25), constrained_layout=True)
    for index, (model, label, color) in enumerate(zip(models, model_labels, colors)):
        values = [
            float(macro[(macro["model"] == model) & (macro["target"] == target)]["r2"].iloc[0])
            for target in targets
        ]
        offset = (index - 1.5) * width
        axis.bar(positions + offset, values, width=width, color=color, label=label)
    axis.axhline(0, color="black", linewidth=0.7)
    axis.set_xticks(positions, [LABELS[target] for target in targets])
    axis.set_ylabel(r"Mean held-out $R^2$")
    axis.set_ylim(0, 0.84)
    axis.grid(axis="y", color="#D8D8D8", linewidth=0.5, alpha=0.8, zorder=0)
    axis.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    save(figure, "cairngorms_surface_supervisor_architectures")


def make_height_diagnostic() -> None:
    broad = pd.read_parquet(PHASE14_SAMPLE)
    mature = pd.read_parquet(PHASE22_TARGETS)
    predictions = pd.read_parquet(PHASE22_PREDICTIONS)
    selected = predictions[
        (predictions["scheme"] == "balanced")
        & (predictions["model"] == "tessera_mlp_5x5")
        & (predictions["target"] == "canopy_p95_height_m")
    ]
    bias = pd.read_csv(PHASE24_DIRECTORY / "quintile_bias.csv")
    bias = bias[bias["model"] == "tessera_mlp_5x5"]
    ridge = pd.read_csv(PHASE24_DIRECTORY / "ridge.csv")
    ridge_macro = ridge.groupby(["target", "scheme"], as_index=False)["r2"].mean()

    figure, axes = plt.subplots(2, 2, figsize=(8.0, 6.2), constrained_layout=True)
    axes[0, 0].hist(
        broad["lidar_meanH"], bins=50, density=True, alpha=0.55, color="#4C78A8", label="10 m broad sample"
    )
    axes[0, 0].hist(
        mature["canopy_mean_height_m"], bins=40, density=True, alpha=0.65, color="#E8872D", label="50 m mature forest"
    )
    axes[0, 0].set(
        xlabel="Mean canopy height (m)",
        ylabel="Density",
        title="a  Height distributions by cohort",
    )
    axes[0, 0].legend(frameon=False)

    axes[0, 1].hexbin(
        selected["observed"], selected["predicted"], gridsize=42, mincnt=1, cmap="viridis", norm=LogNorm()
    )
    low = min(selected["observed"].min(), selected["predicted"].min())
    high = max(selected["observed"].max(), selected["predicted"].max())
    axes[0, 1].plot([low, high], [low, high], color="#B22222", linewidth=0.9, linestyle="--")
    axes[0, 1].set(
        xlabel="LiDAR p95 height (m)",
        ylabel="Predicted p95 height (m)",
        title="b  P95 height under spatial holdout",
    )

    axes[1, 0].bar(bias["observed_quintile"] + 1, bias["bias_m"], color="#148F88")
    axes[1, 0].axhline(0, color="black", linewidth=0.8)
    axes[1, 0].set(
        xlabel="Observed p95-height quintile",
        ylabel="Mean prediction bias (m)",
        title="c  Bias across the observed height range",
    )

    targets = ["canopy_mean_height_m", "canopy_p95_height_m"]
    x = np.arange(len(targets))
    width = 0.34
    for offset, scheme, label, color in [
        (-width / 2, "buffered_spatial", "Buffered spatial folds", "#148F88"),
        (width / 2, "random", "Random folds", "#8A8F98"),
    ]:
        values = [
            float(
                ridge_macro[(ridge_macro["target"] == target) & (ridge_macro["scheme"] == scheme)]["r2"].iloc[0]
            )
            for target in targets
        ]
        axes[1, 1].bar(x + offset, values, width, label=label, color=color)
    axes[1, 1].axhline(0, color="black", linewidth=0.8)
    axes[1, 1].set_xticks(x, ["Mean height", "P95 height"])
    axes[1, 1].set(ylabel=r"Mean fold $R^2$", title="d  Effect of spatial separation")
    axes[1, 1].legend(frameon=False)
    save(figure, "cairngorms_surface_supervisor_height_diagnostic")


def main() -> None:
    style()
    make_region_map()
    make_savelsbos_region_map()
    make_performance_figure()
    make_prediction_figure()
    make_architecture_figure()
    make_height_diagnostic()


if __name__ == "__main__":
    main()
