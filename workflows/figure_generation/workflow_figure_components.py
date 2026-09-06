#!/usr/bin/env python3
"""Shared data-driven components for the workflow figure."""

from __future__ import annotations

import string
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/figure_development"

TARGETS = ROOT / "data/processed/cairngorms_surface_targets.parquet"
RETURN_PREDICTIONS = ROOT / "data/processed/cairngorms_unet_predictions.parquet"
HEIGHT_FOLDS = ROOT / "data/interim/cairngorms_height_adjusted"
PATCH_VALUES = ROOT / "data/interim/cairngorms_embedding_patches/patch_quantized.npy"
PATCH_SCALES = ROOT / "data/interim/cairngorms_embedding_patches/patch_scales.npy"
PATCH_IDS = ROOT / "data/interim/cairngorms_embedding_patches/row_id.npy"
PROFILE_VALUES = ROOT / "data/interim/cairngorms_return_profiles/profiles.npy"
PROFILE_IDS = ROOT / "data/interim/cairngorms_return_profiles/row_id.npy"

GREEN = "#2a8c78"
ORANGE = "#d97732"
PURPLE = "#7355a8"
BLUE = "#3f719c"
CHARCOAL = "#42484f"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def actual_embedding_rgb(cohort_ids: np.ndarray) -> np.ndarray:
    patches = np.load(PATCH_VALUES, mmap_mode="r")
    scales = np.load(PATCH_SCALES, mmap_mode="r")
    row_ids = np.load(PATCH_IDS, mmap_mode="r")
    candidates = np.flatnonzero(np.isin(row_ids[::25], cohort_ids))[:500] * 25
    if not len(candidates):
        raise RuntimeError("No aligned embedding patches were found")
    best = int(candidates[0])
    best_score = -np.inf
    for index in candidates:
        values = patches[index].astype(np.float32) * scales[index][None, :, :].astype(np.float32)
        score = float(np.mean(np.std(values, axis=(1, 2))))
        if score > best_score:
            best_score = score
            best = int(index)
    values = patches[best].astype(np.float32) * scales[best][None, :, :].astype(np.float32)
    matrix = values.reshape(128, 25).T
    matrix -= matrix.mean(axis=0, keepdims=True)
    _, _, vectors = np.linalg.svd(matrix, full_matrices=False)
    rgb = (matrix @ vectors[:3].T).reshape(5, 5, 3)
    for channel in range(3):
        low, high = np.percentile(rgb[:, :, channel], [2, 98])
        rgb[:, :, channel] = np.clip((rgb[:, :, channel] - low) / max(high - low, 1e-6), 0, 1)
    return rgb


def representative_profile(cohort_ids: np.ndarray) -> np.ndarray:
    profiles = np.load(PROFILE_VALUES, mmap_mode="r")
    row_ids = np.load(PROFILE_IDS, mmap_mode="r")
    pool = np.asarray(profiles[np.isin(row_ids, cohort_ids)], dtype=np.float64)
    clipped = np.clip(pool, 1e-9, 1.0)
    return pool[np.argmax(-(clipped * np.log(clipped)).sum(axis=1))]


def metric_frame() -> pd.DataFrame:
    targets = pd.read_parquet(TARGETS)
    predictions = pd.read_parquet(RETURN_PREDICTIONS)
    entropy = (
        predictions[
            (predictions["model"] == "tessera_v2")
            & (predictions["target"] == "lidar_canopy_shannon_50m")
        ][["row_id", "observed"]]
        .drop_duplicates("row_id")
        .rename(columns={"observed": "return_entropy"})
    )
    frame = targets.merge(entropy, on="row_id", how="inner", validate="one_to_one")
    columns = [
        "canopy_p95_height_m",
        "canopy_surface_cv",
        "canopy_open_fraction",
        "return_entropy",
    ]
    for column in columns:
        frame[f"{column}_percentile"] = frame[column].rank(method="average", pct=True)
    frame["window_x"] = np.floor(frame["bng_x"] / 2000).astype(int)
    frame["window_y"] = np.floor(frame["bng_y"] / 2000).astype(int)
    # Use a well-sampled window with clear spatial variation across all outcomes.
    selected = frame[(frame["window_x"] == 148) & (frame["window_y"] == 409)].copy()
    if len(selected) < 200:
        raise RuntimeError("The frozen metric-example window is unexpectedly sparse")
    return selected


def draw_metric_map(
    axis: plt.Axes,
    frame: pd.DataFrame,
    column: str,
    label: str,
) -> mpl.collections.PathCollection:
    image = axis.scatter(
        frame["bng_x"],
        frame["bng_y"],
        c=frame[column],
        cmap="viridis",
        vmin=0,
        vmax=1,
        marker="s",
        s=7.0,
        linewidths=0,
        rasterized=True,
    )
    axis.set_xlim(296000, 298000)
    axis.set_ylim(818000, 820000)
    axis.set_aspect("equal")
    axis.axis("off")
    axis.text(0.01, 1.02, label, transform=axis.transAxes, ha="left", va="bottom", fontsize=6.4)
    return image


def make_component_figure() -> None:
    configure_style()
    targets = pd.read_parquet(TARGETS).reset_index(drop=True)
    cohort_ids = targets["row_id"].to_numpy(dtype=np.int64)
    embedding = actual_embedding_rgb(cohort_ids)
    profile = representative_profile(cohort_ids)
    examples = metric_frame()

    fig, axis = plt.subplots(figsize=(11.0, 4.0))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    panel_left = [0.012, 0.195, 0.342, 0.585, 0.745]
    panel_centres = [0.090, 0.260, 0.460, 0.665, 0.865]
    headings = [
        "Aligned annual data",
        "Common 50 m support",
        "LiDAR outcomes",
        "Spatial validation",
        "Models and evaluation",
    ]
    for index, (left, centre, heading) in enumerate(
        zip(panel_left, panel_centres, headings, strict=True)
    ):
        axis.text(centre, 0.965, heading, ha="center", va="top", fontsize=9.0, fontweight="bold")
        axis.text(
            left,
            0.850,
            string.ascii_lowercase[index],
            ha="left",
            va="top",
            fontsize=10.5,
            fontweight="bold",
        )

    for x0, x1 in [(0.176, 0.196), (0.326, 0.346), (0.569, 0.589), (0.729, 0.749)]:
        axis.add_patch(
            mpl.patches.FancyArrowPatch(
                (x0, 0.535),
                (x1, 0.535),
                arrowstyle="-|>",
                mutation_scale=9,
                linewidth=0.8,
                color="#6f7476",
            )
        )

    # (a) Actual aligned input representations with additional separation.
    patch_axis = axis.inset_axes([0.020, 0.420, 0.060, 0.275])
    patch_axis.imshow(embedding, interpolation="nearest")
    patch_axis.set_xticks([])
    patch_axis.set_yticks([])
    for spine in patch_axis.spines.values():
        spine.set_visible(True)
        spine.set_color(CHARCOAL)
        spine.set_linewidth(0.7)
    axis.text(0.050, 0.720, "Tessera v2", ha="center", va="bottom", fontsize=7.2, fontweight="bold")
    axis.text(0.050, 0.385, "128 channels\n$5\\times5$ pixels", ha="center", va="top", fontsize=6.7)

    profile_axis = axis.inset_axes([0.108, 0.420, 0.063, 0.275])
    levels = np.arange(3)
    profile_axis.bar(levels, profile, color=["#b9d8bf", GREEN, "#235640"], width=0.68)
    profile_axis.set_xticks(levels, ["0-1", "1-10", ">10"])
    profile_axis.set_ylim(0, 0.5)
    profile_axis.set_xlabel("height (m)", fontsize=6.0, labelpad=1)
    profile_axis.tick_params(axis="both", labelsize=5.8, length=2)
    axis.text(
        0.140, 0.720, "LiDAR profile", ha="center", va="bottom", fontsize=7.2, fontweight="bold"
    )
    axis.text(
        0.092,
        0.205,
        "Conventional benchmark:\nSentinel-1, Sentinel-2 and terrain",
        ha="center",
        va="center",
        fontsize=6.7,
    )

    # (b) The support scale is labelled outside the raster, not over it.
    support_axis = axis.inset_axes([0.220, 0.420, 0.080, 0.280])
    support_axis.imshow(
        embedding.mean(axis=2), cmap="Greens", vmin=0, vmax=1, interpolation="nearest"
    )
    support_axis.set_xticks(np.arange(-0.5, 5, 1), minor=True)
    support_axis.set_yticks(np.arange(-0.5, 5, 1), minor=True)
    support_axis.grid(which="minor", color="white", linewidth=0.8)
    support_axis.tick_params(
        which="both", bottom=False, left=False, labelbottom=False, labelleft=False
    )
    for spine in support_axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#235640")
        spine.set_linewidth(1.1)
    axis.plot([0.220, 0.300], [0.725, 0.725], color=CHARCOAL, linewidth=0.8)
    axis.plot([0.220, 0.220], [0.715, 0.735], color=CHARCOAL, linewidth=0.8)
    axis.plot([0.300, 0.300], [0.715, 0.735], color=CHARCOAL, linewidth=0.8)
    axis.text(0.260, 0.748, "50 m", ha="center", va="bottom", fontsize=7.0)
    axis.plot([0.220, 0.236], [0.392, 0.392], color=CHARCOAL, linewidth=0.7)
    axis.plot([0.220, 0.220], [0.386, 0.398], color=CHARCOAL, linewidth=0.7)
    axis.plot([0.236, 0.236], [0.386, 0.398], color=CHARCOAL, linewidth=0.7)
    axis.text(0.228, 0.375, "10 m", ha="center", va="top", fontsize=6.1)
    axis.text(
        0.260,
        0.315,
        "centre | mean | SD\nE-W and N-S gradients",
        ha="center",
        va="center",
        fontsize=6.7,
    )
    axis.text(
        0.260,
        0.215,
        "5 summaries $\\times$ 128\n= 640 MLP predictors",
        ha="center",
        va="center",
        fontsize=6.7,
    )

    # (c) Four real outcomes from the same 2 x 2 km Cairngorms window.
    metric_specs = [
        ("canopy_p95_height_m_percentile", "p95 height"),
        ("canopy_surface_cv_percentile", "canopy CV"),
        ("canopy_open_fraction_percentile", "openings"),
        ("return_entropy_percentile", "return entropy"),
    ]
    positions = [
        [0.355, 0.525, 0.087, 0.205],
        [0.458, 0.525, 0.087, 0.205],
        [0.355, 0.295, 0.087, 0.205],
        [0.458, 0.295, 0.087, 0.205],
    ]
    metric_image = None
    for position, (column, label) in zip(positions, metric_specs, strict=True):
        metric_axis = axis.inset_axes(position)
        metric_image = draw_metric_map(metric_axis, examples, column, label)
    color_axis = axis.inset_axes([0.365, 0.215, 0.170, 0.020])
    colorbar = fig.colorbar(metric_image, cax=color_axis, orientation="horizontal")
    colorbar.set_ticks([0, 0.5, 1])
    colorbar.set_ticklabels(["low", "median", "high"])
    colorbar.ax.tick_params(labelsize=5.8, length=2, pad=1)
    colorbar.set_label("within-outcome percentile", fontsize=6.1, labelpad=1)

    # (d) Actual fold ownership rather than a schematic checkerboard.
    fold_owner = np.full(len(targets), -1, dtype=np.int8)
    for fold in range(5):
        with np.load(HEIGHT_FOLDS / f"fold_{fold}.npz") as source:
            fold_owner[source["test_indices"].astype(np.int64)] = fold
    if np.any(fold_owner < 0):
        raise RuntimeError("Every Cairngorms unit must belong to one held-out fold")
    fold_colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#b279a2"]
    fold_axis = axis.inset_axes([0.596, 0.330, 0.135, 0.455])
    fold_axis.scatter(
        targets["bng_x"],
        targets["bng_y"],
        c=fold_owner,
        cmap=mpl.colors.ListedColormap(fold_colors),
        vmin=-0.5,
        vmax=4.5,
        marker="s",
        s=1.5,
        linewidths=0,
        rasterized=True,
    )
    fold_axis.set_aspect("equal")
    fold_axis.axis("off")
    handles = [
        Line2D(
            [],
            [],
            marker="s",
            linestyle="none",
            markersize=4.0,
            markerfacecolor=color,
            markeredgewidth=0,
            label=str(index + 1),
        )
        for index, color in enumerate(fold_colors)
    ]
    fold_axis.legend(
        handles=handles,
        title="fold",
        ncol=5,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        frameon=False,
        handletextpad=0.1,
        columnspacing=0.40,
        borderaxespad=0,
        fontsize=5.8,
        title_fontsize=6.0,
    )
    axis.text(
        0.665,
        0.235,
        "6 km test blocks\n2 km train-test exclusion",
        ha="center",
        va="center",
        fontsize=6.8,
    )

    # (e) Architecture strips use the dimensions of the fitted models.
    x0, x1 = 0.758, 0.990
    row_y = [0.705, 0.505, 0.300]
    row_names = [
        ("Ridge", "benchmark / transfer"),
        ("MLP", "primary model"),
        ("U-Net", "height only"),
    ]
    for (name, purpose), y in zip(row_names, row_y, strict=True):
        axis.text(x0, y + 0.040, name, ha="left", va="center", fontsize=7.5, fontweight="bold")
        axis.text(x0, y - 0.006, purpose, ha="left", va="center", fontsize=5.9, color=CHARCOAL)
    axis.plot([x0, x1], [0.605, 0.605], color="#d7d9d8", linewidth=0.7)
    axis.plot([x0, x1], [0.400, 0.400], color="#d7d9d8", linewidth=0.7)

    # Ridge: a standardised feature vector mapped linearly to unit outcomes.
    ridge_x = [0.846, 0.895, 0.951]
    for index, height in enumerate([0.064, 0.052, 0.040, 0.030]):
        axis.add_patch(
            mpl.patches.Rectangle(
                (ridge_x[0] - 0.012 + index * 0.006, row_y[0] - height / 2),
                0.005,
                height,
                facecolor=BLUE,
                edgecolor="none",
                alpha=0.82,
            )
        )
    axis.text(ridge_x[0], row_y[0] - 0.055, "$p$ features", ha="center", va="top", fontsize=5.7)
    axis.text(
        ridge_x[1],
        row_y[0],
        r"$\mathbf{X}\hat{\beta}$",
        ha="center",
        va="center",
        fontsize=8.0,
        color=ORANGE,
    )
    for start, end in zip(ridge_x[:-1], ridge_x[1:], strict=True):
        axis.add_patch(
            mpl.patches.FancyArrowPatch(
                (start + 0.018, row_y[0]),
                (end - 0.018, row_y[0]),
                arrowstyle="->",
                mutation_scale=7,
                linewidth=0.7,
                color=CHARCOAL,
            )
        )
    for index, length in enumerate([0.038, 0.026, 0.046]):
        axis.plot(
            [ridge_x[2] - length / 2, ridge_x[2] + length / 2],
            [row_y[0] + (index - 1) * 0.022] * 2,
            color=GREEN,
            linewidth=2.2,
            solid_capstyle="butt",
        )
    axis.text(ridge_x[2], row_y[0] - 0.055, "unit outcomes", ha="center", va="top", fontsize=5.7)

    # MLP: layer widths are the exact primary Cairngorms architecture.
    mlp_x = [0.836, 0.876, 0.916, 0.956]
    mlp_labels = ["640", "256", "128", "7 / 4"]
    mlp_heights = [0.090, 0.070, 0.055, 0.036]
    mlp_colors = [GREEN, "#4f9d8d", BLUE, PURPLE]
    for index, (x, label, height, color) in enumerate(
        zip(mlp_x, mlp_labels, mlp_heights, mlp_colors, strict=True)
    ):
        axis.add_patch(
            mpl.patches.Rectangle(
                (x - 0.009, row_y[1] - height / 2),
                0.018,
                height,
                facecolor=color,
                edgecolor="white",
                linewidth=0.45,
            )
        )
        axis.text(
            x,
            row_y[1],
            label,
            ha="center",
            va="center",
            fontsize=5.6,
            color="white",
            fontweight="bold",
        )
        if index:
            axis.add_patch(
                mpl.patches.FancyArrowPatch(
                    (mlp_x[index - 1] + 0.011, row_y[1]),
                    (x - 0.011, row_y[1]),
                    arrowstyle="->",
                    mutation_scale=6.5,
                    linewidth=0.65,
                    color=CHARCOAL,
                )
            )
    axis.text(
        0.896,
        row_y[1] - 0.065,
        "input       hidden layers       outputs",
        ha="center",
        va="top",
        fontsize=5.7,
    )

    # U-Net: encoder, bottleneck, decoder and skip connections.
    unet_x = [0.838, 0.864, 0.890, 0.916, 0.942, 0.968]
    unet_y = [
        row_y[2] + 0.030,
        row_y[2],
        row_y[2] - 0.030,
        row_y[2],
        row_y[2] + 0.030,
        row_y[2] + 0.030,
    ]
    unet_labels = ["129", "24", "48", "96", "48/24", "2"]
    unet_colors = [CHARCOAL, GREEN, BLUE, PURPLE, BLUE, GREEN]
    unet_sizes = [0.017, 0.020, 0.018, 0.016, 0.020, 0.017]
    for index, (x, y, label, color, size) in enumerate(
        zip(unet_x, unet_y, unet_labels, unet_colors, unet_sizes, strict=True)
    ):
        axis.add_patch(
            mpl.patches.Rectangle(
                (x - size / 2, y - size / 2),
                size,
                size,
                facecolor=color,
                edgecolor="white",
                linewidth=0.45,
                zorder=2,
            )
        )
        axis.text(x, y - 0.023, label, ha="center", va="top", fontsize=5.2)
        if index:
            axis.plot(
                unet_x[index - 1 : index + 1],
                unet_y[index - 1 : index + 1],
                color=CHARCOAL,
                linewidth=0.65,
                zorder=1,
            )
    axis.plot(
        [unet_x[1], unet_x[4]],
        [unet_y[1] + 0.024, unet_y[4] + 0.024],
        color="#8f9693",
        linewidth=0.55,
    )
    axis.plot(
        [unet_x[2], unet_x[4] - 0.010],
        [unet_y[2] + 0.021, unet_y[4] - 0.002],
        color="#8f9693",
        linewidth=0.55,
    )
    axis.text(
        0.896,
        0.205,
        "$128\\times128$ pixels; channels shown below",
        ha="center",
        va="center",
        fontsize=5.7,
    )

    axis.text((x0 + x1) / 2, 0.135, "same held-out units", ha="center", va="center", fontsize=6.4)
    axis.text(
        (x0 + x1) / 2,
        0.085,
        r"RMSE   $R^2$   Spearman $\rho$",
        ha="center",
        va="center",
        fontsize=7.2,
        fontweight="bold",
    )

    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.015)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig02_study_design_components.png", dpi=360, bbox_inches="tight")
    fig.savefig(OUT / "fig02_study_design_components.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    make_component_figure()
