#!/usr/bin/env python3
"""Build manuscript-ready spatial and height-adjustment figure bundles."""

from __future__ import annotations

import string
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import ConnectionPatch, FancyArrowPatch, Rectangle
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results" / "figures" / "spatial_bundles"

CAIRNGORMS_TARGETS = ROOT / "data/processed/phase22_cairngorms_surface_targets.parquet"
CAIRNGORMS_PREDICTIONS = ROOT / "data/processed/phase29_cairngorms_v2_surface_predictions.parquet"
HEIGHT_ADJUSTMENT_DIR = ROOT / "data/interim/phase23_cairngorms_height_adjusted"
HEIGHT_DIAGNOSTICS = (
    ROOT / "research_history/outputs/tables/premeeting_height_adjustment_diagnostics.csv"
)
SAVELSBOS_TARGETS = ROOT / "data/processed/phase26_ahn4_deciduous_cohort.parquet"
SAVELSBOS_GEOMETRY = ROOT / "data/processed/phase26_ahn4_selected_site.gpkg"
NATURAL_EARTH = (
    ROOT / "data/external/natural_earth_admin0/ne_110m_admin_0_countries.shp"
)
CAIRNGORMS_CHM = ROOT / "data/raw/scotland_lidar/chm/CHM_canopy_metrics_50m_new.tif"
CAIRNGORMS_CONVENTIONAL = ROOT / "data/processed/phase18_cairngorms_conventional_predictors.parquet"
CAIRNGORMS_PATCHES = ROOT / "data/interim/phase18_cairngorms_neural/patch_quantized.npy"
CAIRNGORMS_PATCH_SCALES = ROOT / "data/interim/phase18_cairngorms_neural/patch_scales.npy"
CAIRNGORMS_PATCH_IDS = ROOT / "data/interim/phase18_cairngorms_neural/row_id.npy"
SURFACE_MACRO = ROOT / "research_history/outputs/tables/phase29_cairngorms_v2_surface_macro.csv"

TARGETS = [
    "canopy_surface_sd_m",
    "canopy_surface_cv",
    "canopy_surface_rcv",
    "canopy_rumple",
    "canopy_open_fraction",
    "canopy_height_kurtosis",
]

TARGET_LABELS = {
    "canopy_surface_sd_m": "Canopy-height SD",
    "canopy_surface_cv": "Canopy-height CV",
    "canopy_surface_rcv": "Robust height CV",
    "canopy_rumple": "Canopy rumple",
    "canopy_open_fraction": "Canopy openings",
    "canopy_height_kurtosis": "Height kurtosis",
}

TARGET_UNITS = {
    "canopy_surface_sd_m": "m",
    "canopy_surface_cv": "ratio",
    "canopy_surface_rcv": "ratio",
    "canopy_rumple": "ratio",
    "canopy_open_fraction": "fraction",
    "canopy_height_kurtosis": "unitless",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def panel_label(axis: plt.Axes, letter: str) -> None:
    axis.text(
        -0.10,
        1.04,
        letter,
        transform=axis.transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def add_north_arrow(axis: plt.Axes, x: float = 0.08, y: float = 0.90) -> None:
    axis.annotate(
        "N",
        xy=(x, y),
        xytext=(x, y - 0.13),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        va="center",
        fontsize=8,
        arrowprops={"arrowstyle": "-|>", "color": "black", "lw": 1.2},
    )


def add_scale_bar(
    axis: plt.Axes,
    length_m: float,
    label: str,
    x: float = 0.07,
    y: float = 0.07,
) -> None:
    xmin, xmax = axis.get_xlim()
    ymin, ymax = axis.get_ylim()
    x0 = xmin + x * (xmax - xmin)
    y0 = ymin + y * (ymax - ymin)
    axis.plot([x0, x0 + length_m], [y0, y0], color="black", lw=2.2, solid_capstyle="butt")
    axis.text(x0 + length_m / 2, y0 + 0.018 * (ymax - ymin), label, ha="center", va="bottom", fontsize=8)


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / f"{stem}.png", dpi=320, bbox_inches="tight")
    fig.savefig(OUTPUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def load_height_components(frame: pd.DataFrame) -> np.ndarray:
    components = np.full((len(frame), len(TARGETS)), np.nan, dtype=np.float32)
    ownership = np.zeros(len(frame), dtype=np.int8)
    for fold in range(5):
        with np.load(HEIGHT_ADJUSTMENT_DIR / f"fold_{fold}.npz") as values:
            test = values["test_indices"].astype(np.int64)
            components[test] = values["height_test_predictions"].astype(np.float32)
            ownership[test] += 1
    if not np.all(ownership == 1) or not np.isfinite(components).all():
        raise RuntimeError("The five fold files do not provide exactly one height fit per row")
    return components


def make_joint_height_adjustment() -> None:
    frame = pd.read_parquet(CAIRNGORMS_TARGETS).reset_index(drop=True)
    components = load_height_components(frame)
    diagnostics = pd.read_csv(HEIGHT_DIAGNOSTICS).set_index("target")
    x = frame["canopy_mean_height_m"].to_numpy(float)
    y = frame["canopy_p95_height_m"].to_numpy(float)

    fig, axes = plt.subplots(2, 3, figsize=(11.2, 7.2), sharex=True, sharey=True)
    for position, (axis, target) in enumerate(zip(axes.flat, TARGETS, strict=True)):
        fitted = components[:, position]
        lower, upper = np.nanpercentile(fitted, [1, 99])
        image = axis.hexbin(
            x,
            y,
            C=fitted,
            reduce_C_function=np.mean,
            gridsize=42,
            mincnt=1,
            cmap="viridis",
            vmin=lower,
            vmax=upper,
            linewidths=0,
        )
        density = axis.hexbin(x, y, gridsize=25, mincnt=5, alpha=0)
        offsets = density.get_offsets()
        counts = density.get_array()
        if len(offsets):
            levels = np.nanpercentile(counts, [45, 75, 92])
            levels = np.unique(levels[levels > 0])
            if len(levels):
                axis.tricontour(
                    offsets[:, 0],
                    offsets[:, 1],
                    counts,
                    levels=levels,
                    colors="white",
                    linewidths=0.45,
                    alpha=0.75,
                )
        row = diagnostics.loc[target]
        removed = max(0.0, 100 * row["variance_fraction_removed_by_observed_height"])
        axis.text(
            0.04,
            0.95,
            f"Height-only $R^2$ = {row['height_only_r2']:.2f}\nVariance removed = {removed:.0f}%",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2.5},
        )
        axis.set_title(TARGET_LABELS[target], loc="left", fontweight="bold")
        panel_label(axis, string.ascii_lowercase[position])
        colorbar = fig.colorbar(image, ax=axis, pad=0.015, fraction=0.05)
        colorbar.set_label(f"Out-of-fold fitted component ({TARGET_UNITS[target]})", fontsize=8)
        colorbar.ax.tick_params(labelsize=7)
        axis.grid(color="#d8d8d8", linewidth=0.35, alpha=0.45)

    for axis in axes[-1]:
        axis.set_xlabel("Mean canopy height (m)")
    for axis in axes[:, 0]:
        axis.set_ylabel("P95 canopy height (m)")
    fig.suptitle(
        "Joint mean- and upper-canopy height component removed from each structural outcome",
        y=1.01,
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.005,
        "Colour is the prediction from the fold-specific spline model fitted without the displayed observation; "
        "white contours show sampling density.",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.97))
    save_figure(fig, "joint_height_adjustment_surface")


def make_landscape_context() -> None:
    cairngorms = pd.read_parquet(CAIRNGORMS_TARGETS)
    savelsbos = pd.read_parquet(SAVELSBOS_TARGETS)
    site = gpd.read_file(SAVELSBOS_GEOMETRY, layer="natura_site")
    broadleaf = gpd.read_file(SAVELSBOS_GEOMETRY, layer="top10nl_broadleaf")
    world = gpd.read_file(NATURAL_EARTH)

    fig = plt.figure(figsize=(12.1, 4.9))
    grid = fig.add_gridspec(1, 3, width_ratios=(0.78, 1.35, 1.15), wspace=0.20)
    locator = fig.add_subplot(grid[0, 0])
    cairn_axis = fig.add_subplot(grid[0, 1])
    savels_axis = fig.add_subplot(grid[0, 2])

    europe = world.cx[-12:12, 48:61]
    europe.plot(ax=locator, facecolor="#ecece8", edgecolor="#777772", linewidth=0.55)
    world[world["ADMIN"].isin(["United Kingdom", "Netherlands"])].plot(
        ax=locator, facecolor="#c8d8cf", edgecolor="#315b4b", linewidth=0.8
    )
    locator.scatter(
        [float(cairngorms["longitude"].mean())],
        [float(cairngorms["latitude"].mean())],
        s=45,
        color="#8a2f2b",
        edgecolor="white",
        linewidth=0.8,
        zorder=4,
    )
    savels_point = gpd.GeoSeries(
        gpd.points_from_xy([savelsbos["rd_x"].mean()], [savelsbos["rd_y"].mean()]),
        crs="EPSG:28992",
    ).to_crs("EPSG:4326")
    locator.scatter(
        [savels_point.x.iloc[0]],
        [savels_point.y.iloc[0]],
        s=45,
        color="#8a2f2b",
        edgecolor="white",
        linewidth=0.8,
        zorder=4,
    )
    locator.annotate("Cairngorms", (cairngorms["longitude"].mean(), cairngorms["latitude"].mean()), xytext=(8, 6), textcoords="offset points", fontsize=8)
    locator.annotate("Savelsbos", (savels_point.x.iloc[0], savels_point.y.iloc[0]), xytext=(-48, -14), textcoords="offset points", fontsize=8)
    locator.set_xlim(-12, 12)
    locator.set_ylim(48, 61)
    locator.set_aspect(1.25)
    locator.set_xlabel("Longitude")
    locator.set_ylabel("Latitude")
    locator.set_title("Study locations", loc="left", fontweight="bold")
    panel_label(locator, "a")

    vmin, vmax = 14.0, 38.0
    cairn_scatter = cairn_axis.scatter(
        cairngorms["bng_x"] / 1000,
        cairngorms["bng_y"] / 1000,
        c=cairngorms["canopy_p95_height_m"],
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        s=5.0,
        marker="s",
        linewidths=0,
        rasterized=True,
    )
    cairn_axis.set_aspect("equal")
    cairn_axis.set_xlabel("British National Grid easting (km)")
    cairn_axis.set_ylabel("Northing (km)")
    cairn_axis.set_title("Cairngorms, Scotland\nConifer-dominated forest", loc="left", fontweight="bold")
    panel_label(cairn_axis, "b")
    add_north_arrow(cairn_axis)
    add_scale_bar(cairn_axis, 10, "10 km")

    site_km = site.copy()
    broadleaf_km = broadleaf.copy()
    site_km.geometry = site.geometry.scale(xfact=0.001, yfact=0.001, origin=(0, 0))
    broadleaf_km.geometry = broadleaf.geometry.scale(xfact=0.001, yfact=0.001, origin=(0, 0))
    site_km.plot(ax=savels_axis, facecolor="#edf1ec", edgecolor="#4c554d", linewidth=0.8)
    broadleaf_km.plot(ax=savels_axis, facecolor="#b8cbb8", edgecolor="#58735d", linewidth=0.4, alpha=0.8)
    savels_axis.scatter(
        savelsbos["rd_x"] / 1000,
        savelsbos["rd_y"] / 1000,
        c=savelsbos["ahn4_p95_height_m"],
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        s=15,
        marker="s",
        linewidths=0,
        rasterized=True,
    )
    savels_axis.set_aspect("equal")
    minx, miny, maxx, maxy = site_km.total_bounds
    savels_axis.set_xlim(minx - 0.25, maxx + 0.25)
    savels_axis.set_ylim(miny - 0.25, maxy + 0.25)
    savels_axis.set_xlabel("Dutch RD easting (km)")
    savels_axis.set_ylabel("Northing (km)")
    savels_axis.set_title("Savelsbos, the Netherlands\nBroadleaf-dominated forest", loc="left", fontweight="bold")
    panel_label(savels_axis, "c")
    add_north_arrow(savels_axis)
    add_scale_bar(savels_axis, 1, "1 km")

    colorbar = fig.colorbar(cairn_scatter, ax=[cairn_axis, savels_axis], location="bottom", pad=0.13, fraction=0.055, aspect=35)
    colorbar.set_label("Airborne-LiDAR P95 canopy height (m)")
    fig.suptitle("Contrasting forest landscapes and LiDAR reference units", y=1.01, fontsize=12, fontweight="bold")
    save_figure(fig, "study_landscape_context")


def make_spatial_prediction_bundle() -> None:
    targets = pd.read_parquet(CAIRNGORMS_TARGETS)[["row_id", "bng_x", "bng_y"]]
    predictions = pd.read_parquet(CAIRNGORMS_PREDICTIONS)
    predictions = predictions[
        (predictions["variant"] == "raw")
        & (predictions["version"] == "v2")
        & (predictions["model_family"] == "tessera")
        & predictions["target"].isin(
            ["canopy_surface_cv", "canopy_rumple", "canopy_open_fraction"]
        )
    ].merge(targets, on="row_id", how="left", validate="many_to_one")

    selected = ["canopy_surface_cv", "canopy_rumple", "canopy_open_fraction"]
    fig, axes = plt.subplots(3, 3, figsize=(10.7, 10.3), sharex=True, sharey=True)
    for row_index, target in enumerate(selected):
        data = predictions[predictions["target"] == target].copy()
        if len(data) != len(targets):
            raise RuntimeError(f"Unexpected held-out prediction count for {target}: {len(data)}")
        x = data["bng_x"].to_numpy(float) / 1000
        y = data["bng_y"].to_numpy(float) / 1000
        observed = data["observed"].to_numpy(float)
        predicted = data["predicted"].to_numpy(float)
        residual = predicted - observed
        lower, upper = np.nanpercentile(np.concatenate([observed, predicted]), [1, 99])
        residual_limit = float(np.nanpercentile(np.abs(residual), 98))

        observed_map = axes[row_index, 0].scatter(
            x, y, c=observed, cmap="viridis", vmin=lower, vmax=upper, s=4.5, marker="s", linewidths=0, rasterized=True
        )
        axes[row_index, 1].scatter(
            x, y, c=predicted, cmap="viridis", vmin=lower, vmax=upper, s=4.5, marker="s", linewidths=0, rasterized=True
        )
        residual_map = axes[row_index, 2].scatter(
            x,
            y,
            c=residual,
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-residual_limit, vcenter=0, vmax=residual_limit),
            s=4.5,
            marker="s",
            linewidths=0,
            rasterized=True,
        )
        for column in range(3):
            axes[row_index, column].set_aspect("equal")
            axes[row_index, column].xaxis.set_major_locator(MaxNLocator(5))
            axes[row_index, column].yaxis.set_major_locator(MaxNLocator(5))
        axes[row_index, 0].set_ylabel(f"{TARGET_LABELS[target]}\nNorthing (km)")
        main_colorbar = fig.colorbar(observed_map, ax=axes[row_index, :2], pad=0.012, fraction=0.026)
        main_colorbar.set_label(TARGET_UNITS[target], fontsize=8)
        main_colorbar.ax.tick_params(labelsize=7)
        residual_colorbar = fig.colorbar(residual_map, ax=axes[row_index, 2], pad=0.012, fraction=0.052)
        residual_colorbar.set_label("Predicted - observed", fontsize=8)
        residual_colorbar.ax.tick_params(labelsize=7)

    for index, title in enumerate(["LiDAR observed", "TESSERA v2 predicted", "Held-out residual"]):
        axes[0, index].set_title(title, fontweight="bold")
        panel_label(axes[0, index], string.ascii_lowercase[index])
    for row_index in range(1, 3):
        for column in range(3):
            panel_label(axes[row_index, column], string.ascii_lowercase[row_index * 3 + column])
    for axis in axes[-1]:
        axis.set_xlabel("British National Grid easting (km)")
    add_north_arrow(axes[0, 0])
    add_scale_bar(axes[0, 0], 10, "10 km")
    fig.suptitle(
        "Spatial structure retained in geographically held-out Cairngorms predictions",
        y=1.005,
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    save_figure(fig, "cairngorms_observed_predicted_residual_bundle")


def figure_arrow(fig: plt.Figure, start: tuple[float, float], end: tuple[float, float]) -> None:
    fig.add_artist(
        FancyArrowPatch(
            start,
            end,
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.15,
            color="#303030",
            connectionstyle="arc3,rad=0.0",
        )
    )


def actual_embedding_rgb() -> np.ndarray:
    patches = np.load(CAIRNGORMS_PATCHES, mmap_mode="r")
    scales = np.load(CAIRNGORMS_PATCH_SCALES, mmap_mode="r")
    row_ids = np.load(CAIRNGORMS_PATCH_IDS, mmap_mode="r")
    cohort_ids = set(pd.read_parquet(CAIRNGORMS_TARGETS, columns=["row_id"])["row_id"].astype(int))
    candidates = np.flatnonzero(np.isin(row_ids[::25], list(cohort_ids))) * 25
    candidates = candidates[:500]
    best_index = int(candidates[0])
    best_score = -np.inf
    for index in candidates:
        values = patches[index].astype(np.float32) * scales[index][None, :, :].astype(np.float32)
        score = float(np.mean(np.std(values, axis=(1, 2))))
        if score > best_score:
            best_score = score
            best_index = int(index)
    values = patches[best_index].astype(np.float32) * scales[best_index][None, :, :].astype(np.float32)
    matrix = values.reshape(128, 25).T
    matrix -= matrix.mean(axis=0, keepdims=True)
    _, _, vectors = np.linalg.svd(matrix, full_matrices=False)
    rgb = (matrix @ vectors[:3].T).reshape(5, 5, 3)
    for channel in range(3):
        low, high = np.percentile(rgb[:, :, channel], [2, 98])
        rgb[:, :, channel] = np.clip((rgb[:, :, channel] - low) / max(high - low, 1e-6), 0, 1)
    return rgb


def make_study_workflow_plate() -> None:
    cohort = pd.read_parquet(CAIRNGORMS_TARGETS)
    predictions = pd.read_parquet(CAIRNGORMS_PREDICTIONS)
    cv = predictions[
        (predictions["variant"] == "raw")
        & (predictions["version"] == "v2")
        & (predictions["model_family"] == "tessera")
        & (predictions["target"] == "canopy_surface_cv")
    ].merge(cohort[["row_id", "bng_x", "bng_y"]], on="row_id", validate="one_to_one")
    macro = pd.read_csv(SURFACE_MACRO)
    macro = macro[(macro["version"] == "v2") & (macro["model_family"] == "tessera")]
    fold_owner = np.full(len(cohort), -1, dtype=int)
    for fold in range(5):
        with np.load(HEIGHT_ADJUSTMENT_DIR / f"fold_{fold}.npz") as values:
            fold_owner[values["test_indices"].astype(int)] = fold
    if np.any(fold_owner < 0):
        raise RuntimeError("Incomplete fold ownership")

    fig = plt.figure(figsize=(13.2, 8.2))
    fig.suptitle("From Earth-observation embeddings to spatially independent canopy-structure tests", y=0.975, fontsize=14, fontweight="bold")

    lidar_axis = fig.add_axes([0.045, 0.57, 0.19, 0.29])
    lidar_map = lidar_axis.scatter(
        cohort["bng_x"] / 1000,
        cohort["bng_y"] / 1000,
        c=cohort["canopy_surface_cv"],
        cmap="viridis",
        vmin=0.12,
        vmax=0.95,
        s=3.0,
        marker="s",
        linewidths=0,
        rasterized=True,
    )
    lidar_axis.set_aspect("equal")
    lidar_axis.set_title("Airborne LiDAR reference\n2023 canopy-height CV", fontweight="bold")
    lidar_axis.set_axis_off()
    panel_label(lidar_axis, "a")
    lidar_bar = fig.colorbar(lidar_map, ax=lidar_axis, orientation="horizontal", pad=0.015, fraction=0.05)
    lidar_bar.ax.tick_params(labelsize=6)

    embedding_axis = fig.add_axes([0.065, 0.16, 0.15, 0.25])
    embedding_axis.imshow(actual_embedding_rgb(), interpolation="nearest")
    for boundary in np.arange(-0.5, 5, 1):
        embedding_axis.axhline(boundary, color="white", lw=0.7, alpha=0.8)
        embedding_axis.axvline(boundary, color="white", lw=0.7, alpha=0.8)
    embedding_axis.set_title("TESSERA embedding patch\nPCA display of 128 channels", fontsize=9, fontweight="bold", pad=7)
    embedding_axis.set_xticks([])
    embedding_axis.set_yticks([])
    embedding_axis.text(-0.10, 1.08, "b", transform=embedding_axis.transAxes, fontsize=11, fontweight="bold")

    support_axis = fig.add_axes([0.305, 0.61, 0.18, 0.23])
    support_axis.imshow(np.arange(25).reshape(5, 5), cmap="Greens", alpha=0.75)
    for boundary in np.arange(-0.5, 5, 1):
        support_axis.axhline(boundary, color="white", lw=1.2)
        support_axis.axvline(boundary, color="white", lw=1.2)
    support_axis.add_patch(Rectangle((-0.48, -0.48), 4.96, 4.96, fill=False, lw=2.0, edgecolor="#1c4c38"))
    support_axis.text(2, 2, "50 m\nLiDAR unit", ha="center", va="center", fontsize=11, fontweight="bold", color="white")
    support_axis.set_title("Common spatial support\n5 × 5 TESSERA pixels", fontweight="bold")
    support_axis.set_xticks([])
    support_axis.set_yticks([])
    panel_label(support_axis, "c")

    fold_axis = fig.add_axes([0.285, 0.17, 0.22, 0.28])
    fold_axis.scatter(
        cohort["bng_x"] / 1000,
        cohort["bng_y"] / 1000,
        c=fold_owner,
        cmap=mpl.colors.ListedColormap(["#4c78a8", "#f58518", "#54a24b", "#e45756", "#b279a2"]),
        s=2.5,
        marker="s",
        linewidths=0,
        rasterized=True,
    )
    fold_axis.set_aspect("equal")
    fold_axis.set_axis_off()
    fold_axis.set_title("Buffered spatial evaluation\n5 dispersed test folds; 2 km exclusion", fontsize=9, fontweight="bold", pad=10)
    fold_axis.text(-0.10, 1.10, "d", transform=fold_axis.transAxes, fontsize=11, fontweight="bold")

    model_axis = fig.add_axes([0.515, 0.18, 0.19, 0.63])
    model_axis.set_axis_off()
    model_axis.text(0.5, 1.02, "Model and target tests", ha="center", va="bottom", fontsize=11, fontweight="bold", transform=model_axis.transAxes)
    boxes = [
        (0.25, 0.80, "TESSERA v2\n5 × 5 MLP", "#dcefe6", "#2f6b55"),
        (0.75, 0.80, "Sentinel-1/2 + terrain\nRidge", "#e8edf3", "#48627c"),
        (0.50, 0.48, "Target branches\noriginal | mean + P95 adjusted", "#f7ebd8", "#97672b"),
        (0.50, 0.12, "Held-out predictions\nRMSE, $R^2$, block rank", "#f2e3e3", "#914b4b"),
    ]
    for x, y, text_value, face, edge in boxes:
        model_axis.text(
            x,
            y,
            text_value,
            ha="center",
            va="center",
            fontsize=7.7 if y == 0.80 else 8.6,
            fontweight="bold" if y in (0.80, 0.12) else "normal",
            bbox={"boxstyle": "round,pad=0.48", "facecolor": face, "edgecolor": edge, "linewidth": 1.2},
            transform=model_axis.transAxes,
        )
    model_axis.annotate("", xy=(0.43, 0.22), xytext=(0.25, 0.70), xycoords="axes fraction", arrowprops={"arrowstyle": "-|>", "lw": 1.0})
    model_axis.annotate("", xy=(0.57, 0.22), xytext=(0.75, 0.70), xycoords="axes fraction", arrowprops={"arrowstyle": "-|>", "lw": 1.0})
    model_axis.annotate("", xy=(0.50, 0.22), xytext=(0.50, 0.38), xycoords="axes fraction", arrowprops={"arrowstyle": "-|>", "lw": 1.0})
    panel_label(model_axis, "e")

    observed_axis = fig.add_axes([0.735, 0.59, 0.11, 0.23])
    predicted_axis = fig.add_axes([0.855, 0.59, 0.11, 0.23])
    for axis, field, title in [(observed_axis, "observed", "LiDAR observed"), (predicted_axis, "predicted", "TESSERA predicted")]:
        axis.scatter(cv["bng_x"] / 1000, cv["bng_y"] / 1000, c=cv[field], cmap="viridis", vmin=0.12, vmax=0.95, s=1.8, marker="s", linewidths=0, rasterized=True)
        axis.set_aspect("equal")
        axis.set_axis_off()
        axis.set_title(title, fontsize=9, fontweight="bold")
    panel_label(observed_axis, "f")

    residual_axis = fig.add_axes([0.735, 0.32, 0.11, 0.20])
    residual = cv["predicted"] - cv["observed"]
    limit = np.percentile(np.abs(residual), 98)
    residual_axis.scatter(cv["bng_x"] / 1000, cv["bng_y"] / 1000, c=residual, cmap="RdBu_r", vmin=-limit, vmax=limit, s=1.8, marker="s", linewidths=0, rasterized=True)
    residual_axis.set_aspect("equal")
    residual_axis.set_axis_off()
    residual_axis.set_title("Spatial residual", fontsize=9, fontweight="bold")

    score_axis = fig.add_axes([0.845, 0.20, 0.14, 0.29])
    outcomes = ["canopy_surface_cv", "canopy_rumple", "canopy_open_fraction"]
    labels = ["Height CV", "Rumple", "Openings"]
    raw = [float(macro[(macro["variant"] == "raw") & (macro["target"] == target)]["r2"].iloc[0]) for target in outcomes]
    adjusted = [float(macro[(macro["variant"] == "height_adjusted") & (macro["target"] == target)]["r2"].iloc[0]) for target in outcomes]
    positions = np.arange(3)
    score_axis.bar(positions - 0.17, raw, width=0.34, color="#258f83", label="Original")
    score_axis.bar(positions + 0.17, adjusted, width=0.34, color="#7a55b5", label="Height-adjusted")
    score_axis.set_xticks(positions, labels, rotation=20, ha="right")
    score_axis.set_ylim(0, 0.9)
    score_axis.set_ylabel("Held-out $R^2$")
    score_axis.set_title("What remained predictable?", fontsize=9, fontweight="bold")
    score_axis.legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2)
    score_axis.grid(axis="y", color="#dddddd", lw=0.5)

    figure_arrow(fig, (0.235, 0.69), (0.30, 0.72))
    figure_arrow(fig, (0.215, 0.29), (0.30, 0.68))
    figure_arrow(fig, (0.485, 0.70), (0.515, 0.70))
    figure_arrow(fig, (0.505, 0.31), (0.515, 0.31))
    figure_arrow(fig, (0.705, 0.49), (0.725, 0.61))
    fig.text(0.72, 0.10, "Outputs combine spatial pattern, numerical agreement and height-independent structure.", ha="left", fontsize=8, color="#444444")
    save_figure(fig, "study_workflow_storyboard")


def make_landscape_story_map() -> None:
    cairngorms = pd.read_parquet(CAIRNGORMS_TARGETS)

    with rasterio.open(CAIRNGORMS_CHM) as dataset:
        background = dataset.read(1, masked=True)
        extent = [dataset.bounds.left / 1000, dataset.bounds.right / 1000, dataset.bounds.bottom / 1000, dataset.bounds.top / 1000]

    fig = plt.figure(figsize=(8.4, 7.6))
    main = fig.add_axes([0.10, 0.09, 0.72, 0.84])
    main.imshow(background, extent=extent, origin="upper", cmap="Greys", vmin=10, vmax=38, alpha=0.22)
    main_map = main.scatter(
        cairngorms["bng_x"] / 1000,
        cairngorms["bng_y"] / 1000,
        c=cairngorms["canopy_surface_cv"],
        cmap="viridis",
        vmin=0.12,
        vmax=0.95,
        s=4.2,
        marker="s",
        linewidths=0,
        rasterized=True,
    )
    main.set_aspect("equal")
    main.set_xlabel("British National Grid easting (km)")
    main.set_ylabel("Northing (km)")
    add_north_arrow(main, x=0.93, y=0.91)
    add_scale_bar(main, 10, "10 km", x=0.07, y=0.055)
    color_axis = fig.add_axes([0.86, 0.22, 0.027, 0.56])
    colorbar = fig.colorbar(main_map, cax=color_axis)
    colorbar.set_label("Canopy-height coefficient of variation", labelpad=9)

    zoom_bounds = (281.5, 288.0, 797.0, 804.5)
    main.add_patch(Rectangle((zoom_bounds[0], zoom_bounds[2]), zoom_bounds[1] - zoom_bounds[0], zoom_bounds[3] - zoom_bounds[2], fill=False, edgecolor="#a32625", linewidth=1.1))
    zoom = fig.add_axes([0.13, 0.62, 0.29, 0.27], facecolor="white")
    subset = cairngorms[
        cairngorms["bng_x"].between(zoom_bounds[0] * 1000, zoom_bounds[1] * 1000)
        & cairngorms["bng_y"].between(zoom_bounds[2] * 1000, zoom_bounds[3] * 1000)
    ]
    zoom.scatter(subset["bng_x"] / 1000, subset["bng_y"] / 1000, c=subset["canopy_surface_cv"], cmap="viridis", vmin=0.12, vmax=0.95, s=11, marker="s", linewidths=0.12, edgecolors="white")
    zoom.set_xlim(zoom_bounds[0], zoom_bounds[1])
    zoom.set_ylim(zoom_bounds[2], zoom_bounds[3])
    zoom.set_aspect("equal")
    zoom.set_xticks([])
    zoom.set_yticks([])
    zoom.spines[:].set_color("#a32625")
    fig.add_artist(ConnectionPatch(xyA=(zoom_bounds[1], zoom_bounds[3]), coordsA=main.transData, xyB=(1, 0), coordsB=zoom.transAxes, color="#a32625", lw=0.75))
    save_figure(fig, "landscape_structural_story_map")


def main() -> None:
    configure_style()
    make_joint_height_adjustment()
    make_landscape_context()
    make_spatial_prediction_bundle()
    make_landscape_story_map()
    print(f"Wrote figure bundles to {OUTPUT}")


if __name__ == "__main__":
    main()
