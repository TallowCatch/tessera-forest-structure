#!/usr/bin/env python3
"""Build the workflow figure with explicit model families."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from figure_output import save_figure
from matplotlib.lines import Line2D
from workflow_figure_components import (
    BLUE,
    CHARCOAL,
    GREEN,
    HEIGHT_FOLDS,
    ORANGE,
    PURPLE,
    TARGETS,
    actual_embedding_rgb,
    configure_style,
    metric_frame,
    representative_profile,
)

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results/figures/main"


SURFACE_PREDICTIONS = (
    Path(__file__).resolve().parents[2] / "data/processed/cairngorms_surface_predictions.parquet"
)
HEIGHT_PREDICTIONS = (
    Path(__file__).resolve().parents[2] / "data/processed/cairngorms_height_predictions.parquet"
)

FOLD_COLOURS = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#b279a2"]


def arrow(axis: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    axis.add_patch(
        mpl.patches.FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.75,
            color="#707578",
        )
    )


def panel_heading(axis: plt.Axes, letter: str, heading: str, x: float, y: float) -> None:
    axis.text(x, y, letter, ha="left", va="top", fontsize=10.5, fontweight="bold")
    axis.text(x + 0.027, y, heading, ha="left", va="top", fontsize=8.7, fontweight="bold")


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
        s=8.0,
        linewidths=0,
        rasterized=True,
    )
    axis.set_xlim(296000, 298000)
    axis.set_ylim(818000, 820000)
    axis.set_aspect("equal")
    axis.axis("off")
    axis.text(0.01, 1.02, label, transform=axis.transAxes, ha="left", va="bottom", fontsize=6.2)
    return image


def model_output_examples(targets: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    surface = pd.read_parquet(SURFACE_PREDICTIONS)
    surface = surface[
        (surface["variant"] == "raw")
        & (surface["version"] == "v2")
        & (surface["model_family"] == "tessera")
    ].copy()
    surface["display_value"] = surface.groupby("target")["predicted"].rank(pct=True)
    target_order = [
        "canopy_p95_height_m",
        "canopy_surface_sd_m",
        "canopy_surface_cv",
        "canopy_surface_rcv",
        "canopy_rumple",
        "canopy_open_fraction",
        "canopy_height_kurtosis",
    ]
    matrix = surface.pivot(index="row_id", columns="target", values="display_value")
    matrix = matrix.reindex(columns=target_order).dropna()
    locations = targets.set_index("row_id")[["bng_x", "bng_y"]]
    matrix = matrix.join(locations, how="inner")
    matrix = matrix[
        (matrix["bng_x"] >= 296000)
        & (matrix["bng_x"] < 298000)
        & (matrix["bng_y"] >= 818000)
        & (matrix["bng_y"] < 820000)
    ].sort_values(["bng_y", "bng_x"])
    if len(matrix) < 18:
        raise RuntimeError("Too few held-out unit predictions in the example window")
    take = np.linspace(0, len(matrix) - 1, 18).astype(int)
    unit_matrix = matrix.iloc[take][target_order].to_numpy(dtype=float)

    height = pd.read_parquet(HEIGHT_PREDICTIONS)
    height = height[height["model"].str.startswith("tessera_v2_unet")].copy()
    height["display_value"] = height.groupby("target")["predicted"].rank(pct=True)
    height = height.merge(
        targets[["row_id", "bng_x", "bng_y"]],
        on="row_id",
        how="inner",
        validate="many_to_one",
    )
    height = height[
        (height["bng_x"] >= 296000)
        & (height["bng_x"] < 298000)
        & (height["bng_y"] >= 818000)
        & (height["bng_y"] < 820000)
    ]
    if height["target"].nunique() != 2:
        raise RuntimeError("The U-Net example must contain both height outputs")
    return unit_matrix, height


def card(axis: plt.Axes, bounds: tuple[float, float, float, float], colour: str) -> None:
    x, y, width, height = bounds
    axis.add_patch(
        mpl.patches.FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.004,rounding_size=0.004",
            facecolor="white",
            edgecolor="#aeb4b2",
            linewidth=0.75,
        )
    )
    axis.add_patch(
        mpl.patches.Rectangle((x, y + height - 0.012), width, 0.012, color=colour, linewidth=0)
    )


def make_figure() -> None:
    configure_style()
    targets = pd.read_parquet(TARGETS).reset_index(drop=True)
    cohort_ids = targets["row_id"].to_numpy(dtype=np.int64)
    embedding = actual_embedding_rgb(cohort_ids)
    profile = representative_profile(cohort_ids)
    examples = metric_frame()
    unit_outputs, height_outputs = model_output_examples(targets)

    fig, axis = plt.subplots(figsize=(9.0, 6.15))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    # Top row: data, support, outcomes and spatial evaluation.
    top_y = 0.965
    for letter, x in zip("abc", [0.018, 0.290, 0.500], strict=True):
        axis.text(x, top_y, letter, ha="left", va="top", fontsize=10.5, fontweight="bold")

    arrow(axis, (0.225, 0.715), (0.295, 0.715))
    arrow(axis, (0.425, 0.715), (0.535, 0.715))

    patch_axis = axis.inset_axes([0.030, 0.645, 0.075, 0.190])
    patch_axis.imshow(embedding, interpolation="nearest")
    patch_axis.set_xticks([])
    patch_axis.set_yticks([])
    for spine in patch_axis.spines.values():
        spine.set_visible(True)
        spine.set_color(CHARCOAL)
        spine.set_linewidth(0.7)
    axis.text(
        0.0675,
        0.855,
        "Tessera\nembeddings",
        ha="center",
        va="bottom",
        fontsize=6.8,
        fontweight="bold",
        linespacing=0.92,
    )
    axis.text(0.0675, 0.615, "$5\\times5$ pixels", ha="center", va="top", fontsize=6.5)

    profile_axis = axis.inset_axes([0.145, 0.645, 0.065, 0.190])
    levels = np.arange(3)
    profile_axis.bar(levels, profile, color=["#b9d8bf", GREEN, "#235640"], width=0.68)
    profile_axis.set_xticks(levels, ["0-1", "1-10", ">10"])
    profile_axis.set_ylim(0, 0.5)
    profile_axis.set_xlabel("height (m)", fontsize=5.8, labelpad=1)
    profile_axis.tick_params(axis="both", labelsize=5.4, length=2)
    axis.text(
        0.1775, 0.855, "LiDAR profile", ha="center", va="bottom", fontsize=7.0, fontweight="bold"
    )

    support_axis = axis.inset_axes([0.315, 0.645, 0.100, 0.190])
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
        spine.set_linewidth(1.0)
    axis.plot([0.315, 0.415], [0.862, 0.862], color=CHARCOAL, linewidth=0.75)
    axis.plot([0.315, 0.315], [0.854, 0.870], color=CHARCOAL, linewidth=0.75)
    axis.plot([0.415, 0.415], [0.854, 0.870], color=CHARCOAL, linewidth=0.75)
    axis.text(0.365, 0.875, "50 m", ha="center", va="bottom", fontsize=6.8)
    axis.plot([0.315, 0.335], [0.625, 0.625], color=CHARCOAL, linewidth=0.7)
    axis.plot([0.315, 0.315], [0.619, 0.631], color=CHARCOAL, linewidth=0.7)
    axis.plot([0.335, 0.335], [0.619, 0.631], color=CHARCOAL, linewidth=0.7)
    axis.text(0.325, 0.612, "10 m", ha="center", va="top", fontsize=5.8)

    metric_specs = [
        ("canopy_p95_height_m_percentile", "p95 height"),
        ("canopy_surface_cv_percentile", "canopy CV"),
        ("canopy_open_fraction_percentile", "openings"),
        ("return_entropy_percentile", "return entropy"),
    ]
    positions = [
        [0.545, 0.715, 0.160, 0.145],
        [0.710, 0.715, 0.160, 0.145],
        [0.545, 0.540, 0.160, 0.145],
        [0.710, 0.540, 0.160, 0.145],
    ]
    metric_image = None
    for position, (column, label) in zip(positions, metric_specs, strict=True):
        metric_axis = axis.inset_axes(position)
        metric_image = draw_metric_map(metric_axis, examples, column, label)
    colour_axis = axis.inset_axes([0.885, 0.545, 0.012, 0.305])
    colourbar = fig.colorbar(metric_image, cax=colour_axis, orientation="vertical")
    colourbar.set_ticks([0, 0.5, 1])
    colourbar.set_ticklabels(["low", "median", "high"])
    colourbar.ax.tick_params(labelsize=5.5, length=2, pad=2)
    colourbar.set_label("within-outcome percentile", fontsize=5.8, labelpad=4)

    fold_owner = np.full(len(targets), -1, dtype=np.int8)
    for fold in range(5):
        with np.load(HEIGHT_FOLDS / f"fold_{fold}.npz") as source:
            fold_owner[source["test_indices"].astype(np.int64)] = fold
    if np.any(fold_owner < 0):
        raise RuntimeError("Every Cairngorms unit must belong to one held-out fold")
    fold_axis = axis.inset_axes([0.025, 0.080, 0.155, 0.280])
    fold_axis.scatter(
        targets["bng_x"],
        targets["bng_y"],
        c=fold_owner,
        cmap=mpl.colors.ListedColormap(FOLD_COLOURS),
        vmin=-0.5,
        vmax=4.5,
        marker="s",
        s=1.7,
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
            markersize=3.8,
            markerfacecolor=colour,
            markeredgewidth=0,
            label=str(index + 1),
        )
        for index, colour in enumerate(FOLD_COLOURS)
    ]
    fold_axis.legend(
        handles=handles,
        title="fold",
        ncol=5,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.075),
        frameon=False,
        handletextpad=0.1,
        columnspacing=0.38,
        borderaxespad=0,
        fontsize=5.6,
        title_fontsize=5.8,
    )
    # Bottom row: spatial validation, model families and returned predictions.
    axis.text(0.018, 0.390, "d", ha="left", va="top", fontsize=10.5, fontweight="bold")
    axis.text(0.205, 0.395, "e", ha="left", va="top", fontsize=10.5, fontweight="bold")
    axis.text(0.795, 0.390, "f", ha="left", va="top", fontsize=10.5, fontweight="bold")

    axis.add_patch(
        mpl.patches.Rectangle(
            (0.215, 0.075),
            0.550,
            0.295,
            fill=False,
            edgecolor="#9ba19f",
            linewidth=0.65,
        )
    )
    model_axis = axis.inset_axes([0.215, 0.085, 0.550, 0.275])
    model_axis.set_xlim(0, 1)
    model_axis.set_ylim(0, 1)
    model_axis.axis("off")

    def draw_grid(
        left: float,
        bottom: float,
        width: float,
        values: np.ndarray,
        edge_colour: str,
    ) -> None:
        rows, columns = values.shape
        cell = width / columns
        for row in range(rows):
            for column in range(columns):
                model_axis.add_patch(
                    mpl.patches.Rectangle(
                        (left + column * cell, bottom + (rows - row - 1) * cell),
                        cell,
                        cell,
                        facecolor=mpl.colormaps["viridis"](float(values[row, column])),
                        edgecolor="white",
                        linewidth=0.25,
                    )
                )
        model_axis.add_patch(
            mpl.patches.Rectangle(
                (left, bottom),
                width,
                rows * cell,
                fill=False,
                edgecolor=edge_colour,
                linewidth=0.7,
            )
        )

    # Four model families, drawn without decorative headers or separate input arrows.
    ridge_left, ridge_width = 0.005, 0.165
    mlp_left, mlp_width = 0.195, 0.315
    cnn_left, cnn_width = 0.535, 0.180
    unet_left, unet_width = 0.745, 0.245
    ridge_centre = ridge_left + ridge_width / 2
    mlp_centre = mlp_left + mlp_width / 2
    cnn_centre = cnn_left + cnn_width / 2
    unet_centre = unet_left + unet_width / 2
    for centre, label in [
        (ridge_centre, "Ridge"),
        (mlp_centre, "MLP"),
        (cnn_centre, "CNN"),
        (unet_centre, "U-Net"),
    ]:
        model_axis.text(
            centre, 0.965, label, ha="center", va="top", fontsize=7.0, fontweight="bold"
        )

    # Ridge: predictor coefficients and a single fitted outcome.
    ridge_values = np.asarray([0.34, -0.66, 0.52, 0.81, -0.27, 0.45])
    ridge_x = np.linspace(ridge_centre - 0.034, ridge_centre + 0.034, len(ridge_values))
    ridge_zero = 0.620
    for x, value in zip(ridge_x, ridge_values, strict=True):
        model_axis.plot([x, x], [ridge_zero, ridge_zero + value * 0.115], color=BLUE, linewidth=1.8)
    model_axis.plot(
        [ridge_x[0] - 0.008, ridge_x[-1] + 0.008],
        [ridge_zero, ridge_zero],
        color=CHARCOAL,
        linewidth=0.55,
    )
    model_axis.text(ridge_centre, 0.760, "$\\beta$", ha="center", va="center", fontsize=6.2)
    # MLP: 3 x 3 and 5 x 5 spatial summaries feed the same dense network.
    support_values = embedding.mean(axis=2)
    grid_3_left, grid_5_left = mlp_left + 0.075, mlp_left + 0.170
    draw_grid(grid_3_left, 0.650, 0.055, support_values[1:4, 1:4], GREEN)
    draw_grid(grid_5_left, 0.620, 0.078, support_values, GREEN)
    model_axis.text(grid_3_left + 0.0275, 0.620, "$3\\times3$", ha="center", va="top", fontsize=5.0)
    model_axis.text(grid_5_left + 0.039, 0.600, "$5\\times5$", ha="center", va="top", fontsize=5.0)
    layer_x = [mlp_left + 0.045, mlp_left + 0.120, mlp_left + 0.200, mlp_left + 0.280]
    layer_counts = [7, 6, 5, 3]
    layer_labels = ["640", "256", "128", "7/4"]
    layer_nodes: list[list[tuple[float, float]]] = []
    for x, count, label in zip(layer_x, layer_counts, layer_labels, strict=True):
        ys = np.linspace(0.220, 0.390, count)
        nodes = [(x, float(y)) for y in ys]
        layer_nodes.append(nodes)
        for _, y in nodes:
            model_axis.add_patch(
                mpl.patches.Circle(
                    (x, y), 0.0060, facecolor=GREEN, edgecolor="white", linewidth=0.25, zorder=3
                )
            )
        model_axis.text(x, 0.410, label, ha="center", va="bottom", fontsize=4.8)
    for source, destination in zip(layer_nodes[:-1], layer_nodes[1:], strict=True):
        for x0, y0 in source:
            for x1, y1 in destination:
                model_axis.plot(
                    [x0, x1],
                    [y0, y1],
                    color=mpl.colors.to_rgba(GREEN, 0.16),
                    linewidth=0.28,
                    zorder=1,
                )
    # CNN: the 128 embedding channels plus a validity mask form a 129-channel patch.
    for offset in [0.016, 0.008, 0.0]:
        model_axis.add_patch(
            mpl.patches.Rectangle(
                (cnn_left + 0.012 + offset, 0.610 + offset),
                0.060,
                0.060,
                fill=offset != 0.0,
                facecolor=mpl.colors.to_rgba(ORANGE, 0.08),
                edgecolor=ORANGE,
                linewidth=0.6,
            )
        )
    draw_grid(cnn_left + 0.012, 0.610, 0.060, support_values, ORANGE)
    model_axis.text(cnn_left + 0.042, 0.705, "129", ha="center", va="bottom", fontsize=4.8)
    for index in range(3):
        model_axis.add_patch(
            mpl.patches.Rectangle(
                (cnn_left + 0.108 + index * 0.008, 0.600 + index * 0.008),
                0.035,
                0.078,
                facecolor=mpl.colors.to_rgba(ORANGE, 0.12 + index * 0.08),
                edgecolor=ORANGE,
                linewidth=0.55,
            )
        )
    model_axis.text(cnn_left + 0.130, 0.710, "$96\\times3$", ha="center", va="bottom", fontsize=4.8)
    cnn_head_x = [cnn_left + 0.040, cnn_left + 0.095, cnn_left + 0.150]
    for x, label in zip(cnn_head_x, ["256", "128", "7"], strict=True):
        model_axis.add_patch(
            mpl.patches.Circle(
                (x, 0.310), 0.010, facecolor=ORANGE, edgecolor="white", linewidth=0.35
            )
        )
        model_axis.text(x, 0.280, label, ha="center", va="top", fontsize=4.8)
    # U-Net: spatial encoder-decoder widths and skip connections.
    unet_x = np.linspace(unet_left + 0.025, unet_left + 0.205, 5)
    unet_y = np.array([0.645, 0.485, 0.305, 0.485, 0.645])
    unet_labels = ["24", "48", "96", "48", "24"]
    unet_colours = [GREEN, BLUE, PURPLE, BLUE, GREEN]
    unet_sizes = [0.032, 0.029, 0.026, 0.029, 0.032]
    for index, (x, y, label, colour, size) in enumerate(
        zip(unet_x, unet_y, unet_labels, unet_colours, unet_sizes, strict=True)
    ):
        model_axis.add_patch(
            mpl.patches.Rectangle(
                (x - size / 2, y - size / 2),
                size,
                size,
                facecolor=colour,
                edgecolor="white",
                linewidth=0.4,
                zorder=3,
            )
        )
        model_axis.text(x, y - size / 2 - 0.018, label, ha="center", va="top", fontsize=4.6)
        if index:
            model_axis.plot(
                unet_x[index - 1 : index + 1],
                unet_y[index - 1 : index + 1],
                color=CHARCOAL,
                linewidth=0.65,
                zorder=1,
            )
    model_axis.plot([unet_x[0], unet_x[-1]], [0.750, 0.750], color="#8e9693", linewidth=0.55)
    model_axis.plot(
        [unet_x[0], unet_x[0]], [unet_y[0] + 0.020, 0.750], color="#8e9693", linewidth=0.55
    )
    model_axis.plot(
        [unet_x[-1], unet_x[-1]], [unet_y[-1] + 0.020, 0.750], color="#8e9693", linewidth=0.55
    )
    model_axis.plot([unet_x[1], unet_x[-2]], [0.620, 0.620], color="#8e9693", linewidth=0.55)
    model_axis.plot(
        [unet_x[1], unet_x[1]], [unet_y[1] + 0.018, 0.620], color="#8e9693", linewidth=0.55
    )
    model_axis.plot(
        [unet_x[-2], unet_x[-2]], [unet_y[-2] + 0.018, 0.620], color="#8e9693", linewidth=0.55
    )
    model_axis.text(unet_x[0], 0.775, "129", ha="center", va="bottom", fontsize=4.8)
    # Panels d--f now form one compact left-to-right workflow.
    arrow(axis, (0.188, 0.220), (0.212, 0.220))
    arrow(axis, (0.768, 0.220), (0.792, 0.220))

    # Actual held-out outputs: a unit-by-outcome matrix and the two U-Net height maps.
    matrix_axis = axis.inset_axes([0.810, 0.185, 0.055, 0.110])
    matrix_axis.imshow(
        unit_outputs, cmap="viridis", vmin=0, vmax=1, aspect="auto", interpolation="nearest"
    )
    matrix_axis.set_xticks(
        np.arange(7), ["p95", "SD", "CV", "RCV", "rum.", "open.", "kurt."], rotation=45, ha="right"
    )
    matrix_axis.set_yticks([])
    matrix_axis.tick_params(axis="x", labelsize=4.8, length=0, pad=1)
    for spine in matrix_axis.spines.values():
        spine.set_color("#8c9290")
        spine.set_linewidth(0.55)
    dense_positions = [
        (0.875, "canopy_mean_height_m", "mean height"),
        (0.940, "canopy_p95_height_m", "p95 height"),
    ]
    for left, target, label in dense_positions:
        subset = height_outputs[height_outputs["target"] == target]
        dense_axis = axis.inset_axes([left, 0.185, 0.050, 0.110])
        dense_axis.scatter(
            subset["bng_x"],
            subset["bng_y"],
            c=subset["display_value"],
            cmap="viridis",
            vmin=0,
            vmax=1,
            marker="s",
            s=5,
            linewidths=0,
            rasterized=True,
        )
        dense_axis.set_xlim(296000, 298000)
        dense_axis.set_ylim(818000, 820000)
        dense_axis.set_aspect("equal")
        dense_axis.axis("off")
        dense_axis.text(
            0.5, -0.08, label, transform=dense_axis.transAxes, ha="center", va="top", fontsize=5.0
        )

    fig.subplots_adjust(left=0.006, right=0.994, top=0.992, bottom=0.012)
    save_figure(
        fig,
        "fig02_study_design",
        RESULTS_DIR,
        dpi=360,
    )
    plt.close(fig)


if __name__ == "__main__":
    make_figure()
