#!/usr/bin/env python3
"""Create literature-informed manuscript figure prototypes from frozen outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "figures" / "manuscript_visual_prototypes"

TEAL = "#168A8D"
ORANGE = "#D95F02"
BLUE = "#2C7FB8"
GREY = "#6C757D"


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / f"{stem}.pdf")
    fig.savefig(OUTPUT / f"{stem}.png")
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom",
    )


def spatial_maps() -> None:
    targets = pd.read_parquet(
        ROOT / "data/processed/phase21_cairngorms_surface_targets.parquet"
    )[["row_id", "bng_x", "bng_y", "phase21_valid"]]
    predictions = pd.read_parquet(
        ROOT / "data/processed/phase29_cairngorms_v2_surface_predictions.parquet"
    )
    predictions = predictions[
        (predictions["variant"] == "raw")
        & (predictions["version"] == "v2")
        & (predictions["model_family"] == "tessera")
        & predictions["target"].isin(["canopy_surface_cv", "canopy_rumple"])
    ].merge(targets, on="row_id", how="inner", validate="many_to_one")
    predictions = predictions[predictions["phase21_valid"]].copy()
    predictions["tile_x"] = np.floor(predictions["bng_x"] / 6000).astype(int)
    predictions["tile_y"] = np.floor(predictions["bng_y"] / 6000).astype(int)
    tile = (
        predictions[["row_id", "tile_x", "tile_y"]]
        .drop_duplicates()
        .groupby(["tile_x", "tile_y"])
        .size()
        .idxmax()
    )
    subset = predictions[
        (predictions["tile_x"] == tile[0]) & (predictions["tile_y"] == tile[1])
    ].copy()

    labels = {
        "canopy_surface_cv": "Canopy-height CV",
        "canopy_rumple": "Canopy rumple",
    }
    fig, axes = plt.subplots(2, 3, figsize=(11.3, 7.0), constrained_layout=True)
    letters = iter("abcdef")
    for row, target in enumerate(labels):
        frame = subset[subset["target"] == target]
        low, high = np.nanpercentile(frame["observed"], [2, 98])
        residual = frame["predicted"] - frame["observed"]
        residual_limit = np.nanpercentile(np.abs(residual), 98)
        shared = None
        for col, (field, title) in enumerate(
            [("observed", "LiDAR observed"), ("predicted", "TESSERA predicted")]
        ):
            ax = axes[row, col]
            shared = ax.scatter(
                frame["bng_x"] / 1000,
                frame["bng_y"] / 1000,
                c=frame[field],
                s=8,
                marker="s",
                linewidth=0,
                cmap="viridis",
                vmin=low,
                vmax=high,
                rasterized=True,
            )
            ax.set_title(f"{title}\n{labels[target]}")
            panel_label(ax, next(letters))
        ax = axes[row, 2]
        residual_plot = ax.scatter(
            frame["bng_x"] / 1000,
            frame["bng_y"] / 1000,
            c=residual,
            s=8,
            marker="s",
            linewidth=0,
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-residual_limit, vcenter=0, vmax=residual_limit),
            rasterized=True,
        )
        ax.set_title(f"Prediction residual\n{labels[target]}")
        panel_label(ax, next(letters))
        fig.colorbar(shared, ax=axes[row, :2], shrink=0.74, pad=0.015, label=labels[target])
        fig.colorbar(
            residual_plot,
            ax=axes[row, 2],
            shrink=0.74,
            pad=0.015,
            label="Predicted - observed",
        )

    for ax in axes.flat:
        ax.set_aspect("equal")
        ax.set_xlabel("British National Grid easting (km)")
        ax.set_ylabel("Northing (km)")
        ax.ticklabel_format(style="plain", useOffset=False)

    first = axes[0, 0]
    xmin, xmax = first.get_xlim()
    ymin, ymax = first.get_ylim()
    x0 = xmin + 0.07 * (xmax - xmin)
    y0 = ymin + 0.07 * (ymax - ymin)
    first.plot([x0, x0 + 1], [y0, y0], color="black", lw=2.2)
    first.text(x0 + 0.5, y0 + 0.10, "1 km", ha="center", va="bottom", fontsize=8)
    first.annotate(
        "N",
        xy=(xmin + 0.08 * (xmax - xmin), ymax - 0.08 * (ymax - ymin)),
        xytext=(xmin + 0.08 * (xmax - xmin), ymax - 0.24 * (ymax - ymin)),
        ha="center",
        arrowprops={"arrowstyle": "-|>", "color": "black", "lw": 1.2},
    )
    save(fig, "prototype_spatial_observed_predicted_residual")


def height_adjustment() -> None:
    associations = pd.read_csv(
        ROOT / "outputs/tables/cairngorms_metric_height_associations.csv"
    )
    key_order = ["surface_sd", "surface_cv", "surface_rcv", "rumple", "openings", "kurtosis"]
    display = {
        "surface_sd": "Height SD",
        "surface_cv": "Height CV",
        "surface_rcv": "Robust height CV",
        "rumple": "Rumple",
        "openings": "Canopy openings",
        "kurtosis": "Height kurtosis",
    }
    matrix = (
        associations[associations["metric_key"].isin(key_order)]
        .pivot(index="metric_key", columns="height_measure", values="block_spearman")
        .reindex(key_order)
    )

    macro = pd.read_csv(ROOT / "outputs/tables/phase29_cairngorms_v2_surface_macro.csv")
    macro = macro[
        (macro["version"] == "v2")
        & (macro["model_family"] == "tessera")
        & macro["target"].isin(
            [
                "canopy_surface_sd_m",
                "canopy_surface_cv",
                "canopy_surface_rcv",
                "canopy_rumple",
                "canopy_open_fraction",
                "canopy_height_kurtosis",
            ]
        )
    ].copy()
    target_to_key = {
        "canopy_surface_sd_m": "surface_sd",
        "canopy_surface_cv": "surface_cv",
        "canopy_surface_rcv": "surface_rcv",
        "canopy_rumple": "rumple",
        "canopy_open_fraction": "openings",
        "canopy_height_kurtosis": "kurtosis",
    }
    macro["metric_key"] = macro["target"].map(target_to_key)
    scores = macro.pivot(index="metric_key", columns="variant", values="r2").reindex(key_order)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.6, 4.8), constrained_layout=True)
    image = ax0.imshow(matrix.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax0.set_xticks(range(matrix.shape[1]), matrix.columns)
    ax0.set_yticks(range(len(key_order)), [display[k] for k in key_order])
    ax0.tick_params(axis="x", rotation=18)
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            value = matrix.iloc[y, x]
            ax0.text(x, y, f"{value:.2f}", ha="center", va="center", color="white" if abs(value) > 0.5 else "black")
    fig.colorbar(image, ax=ax0, shrink=0.85, label="Block-level Spearman correlation")
    ax0.set_title("Association of structural outcomes with canopy stature")
    panel_label(ax0, "a")

    y = np.arange(len(key_order))
    raw = scores["raw"].to_numpy()
    adjusted = scores["height_adjusted"].to_numpy()
    for yi, start, end in zip(y, raw, adjusted):
        ax1.plot([start, end], [yi, yi], color="#B8B8B8", lw=2, zorder=1)
    ax1.scatter(raw, y, color=TEAL, s=50, label="Original outcome", zorder=2)
    ax1.scatter(adjusted, y, color=ORANGE, s=50, label="Height-adjusted outcome", zorder=2)
    ax1.axvline(0, color="black", lw=0.8, ls="--")
    ax1.set_yticks(y, [display[k] for k in key_order])
    ax1.invert_yaxis()
    ax1.set_xlim(0, 0.86)
    ax1.set_xlabel("Spatially held-out $R^2$")
    ax1.set_title("Predictive performance before and after height adjustment")
    ax1.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
    )
    panel_label(ax1, "b")
    fig.suptitle("Separating canopy heterogeneity from canopy stature", fontsize=12, fontweight="bold")
    save(fig, "prototype_height_dependence_and_adjustment")


def transfer_diagnostic() -> None:
    predictions = pd.read_parquet(
        ROOT / "data/processed/phase32_scotland_netherlands_transfer_predictions.parquet"
    )
    local = predictions[
        (predictions["model"] == "local_cv")
        & (predictions["training_site"] == predictions["evaluation_site"])
        & (predictions["feature_panel"] == "context")
    ][["row_id", "evaluation_site", "target", "observed"]].drop_duplicates()
    targets = ["mean_height_m", "p95_height_m", "within_cell_height_sd_m", "height_cv"]
    target_labels = {
        "mean_height_m": "Mean height (m)",
        "p95_height_m": "P95 height (m)",
        "within_cell_height_sd_m": "Height SD (m)",
        "height_cv": "Height CV",
    }

    metrics = pd.read_csv(ROOT / "outputs/tables/phase32_scotland_netherlands_transfer_metrics.csv")
    metrics = metrics[metrics["feature_panel"] == "context"]
    local_rmse = {
        (row.evaluation_site, row.target): row.rmse
        for row in metrics.itertuples()
        if row.model == "local_cv" and row.training_site == row.evaluation_site
    }
    transfers = metrics[
        (metrics["model"] == "transfer")
        & (metrics["training_site"] != metrics["evaluation_site"])
        & metrics["target"].isin(targets)
    ].copy()
    transfers["ratio"] = [
        row.rmse / local_rmse[(row.evaluation_site, row.target)]
        for row in transfers.itertuples()
    ]
    directions = [("cairngorms", "savelsbos"), ("savelsbos", "cairngorms")]
    ratio = np.full((2, len(targets)), np.nan)
    for i, (source, destination) in enumerate(directions):
        for j, target in enumerate(targets):
            ratio[i, j] = transfers[
                (transfers["training_site"] == source)
                & (transfers["evaluation_site"] == destination)
                & (transfers["target"] == target)
            ]["ratio"].iloc[0]

    fig = plt.figure(figsize=(11.3, 6.2), constrained_layout=True)
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.82])
    distribution_grid = outer[0].subgridspec(1, 4)
    for col, target in enumerate(targets):
        ax = fig.add_subplot(distribution_grid[0, col])
        for site, colour, label in [
            ("cairngorms", TEAL, "Cairngorms"),
            ("savelsbos", ORANGE, "Savelsbos"),
        ]:
            values = local[(local["evaluation_site"] == site) & (local["target"] == target)]["observed"]
            ax.hist(values, bins=25, density=True, histtype="step", lw=1.8, color=colour, label=label)
        ax.set_title(target_labels[target])
        ax.set_xlabel("Observed value")
        ax.set_ylabel("Density" if col == 0 else "")
        if col == 0:
            ax.legend(frameon=False)
        panel_label(ax, chr(ord("a") + col))

    matrix_grid = outer[1].subgridspec(
        1,
        4,
        width_ratios=[1.05, 3.0, 0.12, 1.05],
    )
    ax = fig.add_subplot(matrix_grid[0, 1])
    colour_axis = fig.add_subplot(matrix_grid[0, 2])
    heat = ax.imshow(ratio, cmap="YlOrRd", vmin=1, vmax=max(3, float(np.nanmax(ratio))), aspect="auto")
    ax.set_xticks(range(len(targets)), [target_labels[t] for t in targets])
    ax.set_yticks([0, 1], ["Scotland to Netherlands", "Netherlands to Scotland"])
    for i in range(ratio.shape[0]):
        for j in range(ratio.shape[1]):
            value = ratio[i, j]
            ax.text(j, i, f"{value:.1f}x", ha="center", va="center", fontweight="bold", color="white" if value > 3 else "black")
    ax.set_title("Transfer error relative to a model trained and tested within the destination landscape")
    fig.colorbar(heat, cax=colour_axis, label="Transfer RMSE / local RMSE")
    panel_label(ax, "e")
    save(fig, "prototype_transfer_distribution_and_error")


def main() -> None:
    style()
    spatial_maps()
    height_adjustment()
    transfer_diagnostic()
    print(f"Wrote figure prototypes to {OUTPUT}")


if __name__ == "__main__":
    main()
