#!/usr/bin/env python3
"""Plot the distributions of every distinct target in the analyses."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from figure_output import save_figure
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "results/figures/main"

GREEN = "#238B72"
PURPLE = "#7652A7"
CHARCOAL = "#303434"
GREY = "#777D7C"

SURFACE_PREDICTIONS = ROOT / "data/processed/cairngorms_surface_predictions.parquet"
POINT_CLOUD_PREDICTIONS = ROOT / "data/derived/cairngorms_point_cloud_predictions.parquet"
DUTCH_PREDICTIONS = ROOT / "data/processed/savelsbos_predictions.parquet"

SURFACE_RAW = [
    ("canopy_mean_height_m", "Mean height", "m"),
    ("canopy_p95_height_m", "P95 height", "m"),
    ("canopy_surface_sd_m", "Height SD", "m"),
    ("canopy_surface_cv", "Height CV", "dimensionless"),
    ("canopy_surface_rcv", "Robust height CV", "dimensionless"),
    ("canopy_rumple", "Rumple", "dimensionless"),
    ("canopy_open_fraction", "Canopy openings", "proportion"),
    ("canopy_height_kurtosis", "Height kurtosis", "dimensionless"),
]
SURFACE_ADJUSTED = SURFACE_RAW[2:]

POINT_CLOUD = [
    ("lidar_canopy_shannon_50m", "Normalised Shannon entropy", "dimensionless"),
    ("lidar_height_cv_50m", "Return-height CV", "dimensionless"),
    ("lidar_rcv_50m", "Robust return-height CV", "dimensionless"),
    ("lidar_rms_50m", "RMS return height", "m"),
]

DUTCH = [
    ("ahn4_p95_height_m", "P95 height", "m"),
    ("ahn4_height_sd_m", "Height SD", "m"),
    ("ahn4_height_cv", "Height CV", "dimensionless"),
    ("ahn4_entropy", "Return entropy", "dimensionless"),
    ("ahn4_pulse_penetration", "Pulse penetration", "proportion"),
    ("ahn4_sigma_z", "Sigma-z", "m"),
]

TITLE_WRAP = {
    "Normalised Shannon entropy": "Normalised Shannon\nentropy",
    "Robust return-height CV": "Robust return-height\nCV",
    "Robust height CV": "Robust height CV",
}


def target_valid(frame: pd.DataFrame) -> pd.Series:
    """Return the configured target-validity mask from a prepared cohort."""
    if "target_valid" in frame:
        return frame["target_valid"].astype(bool)
    candidates = [column for column in frame if column.endswith("_valid")]
    if len(candidates) != 1:
        raise KeyError("Expected one target-validity column")
    return frame[candidates[0]].astype(bool)


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.titlesize": 8.5,
            "axes.labelsize": 7.8,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
        }
    )


def panel(axis: plt.Axes, index: int) -> None:
    axis.text(
        -0.14,
        1.08,
        chr(ord("a") + index),
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.3,
        fontweight="bold",
    )


def distribution(
    axis: plt.Axes,
    values: np.ndarray,
    name: str,
    unit: str,
    color: str,
    adjusted: bool,
) -> None:
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError(f"No finite observations for {name}")
    lower, upper = np.quantile(values, [0.005, 0.995])
    displayed = values[(values >= lower) & (values <= upper)]
    if np.isclose(lower, upper):
        lower, upper = float(values.min()), float(values.max())
        displayed = values
    axis.hist(
        displayed,
        bins=30,
        density=True,
        histtype="stepfilled",
        color=color,
        alpha=0.28,
        edgecolor=color,
        linewidth=0.8,
    )
    axis.axvline(np.median(values), color=CHARCOAL, linestyle="--", linewidth=0.85)
    if adjusted:
        axis.axvline(0, color=GREY, linestyle=":", linewidth=0.75)
    axis.set_xlim(lower, upper)
    axis.set_title(TITLE_WRAP.get(name, name), loc="center", pad=4)
    axis.set_xlabel(unit)
    axis.set_yticks([])
    axis.tick_params(axis="x", length=2.5, width=0.6)


def unique_observed(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    columns = [*keys, "observed"]
    counts = frame[columns].groupby(keys, dropna=False)["observed"].nunique(dropna=False)
    if (counts > 1).any():
        raise ValueError("Observed values are inconsistent across repeated model records")
    return frame[columns].drop_duplicates(keys)


def save(fig: plt.Figure, stem: str) -> None:
    save_figure(fig, stem, FIGURES, dpi=320)
    plt.close(fig)


def row_label(fig: plt.Figure, axes: list[plt.Axes], label: str, color: str) -> None:
    left = min(axis.get_position().x0 for axis in axes)
    centre = np.mean([axis.get_position().y0 + axis.get_position().height / 2 for axis in axes])
    fig.text(
        left - 0.028,
        centre,
        label,
        rotation=90,
        va="center",
        ha="center",
        color=color,
        fontsize=8.4,
        fontweight="bold",
    )


def family_header(axis: plt.Axes, letter: str) -> None:
    axis.set_axis_off()
    axis.text(0.0, 0.5, letter, fontsize=11, fontweight="bold", va="center")


def cairngorms_surface() -> None:
    predictions = pd.read_parquet(SURFACE_PREDICTIONS)
    predictions = predictions[
        predictions["version"].eq("v2") & predictions["model_family"].eq("tessera")
    ]
    observed = unique_observed(predictions, ["row_id", "variant", "target"])

    fig = plt.figure(figsize=(7.2, 7.1))
    grid = fig.add_gridspec(4, 12, hspace=0.78, wspace=0.72)
    raw_axes = [
        fig.add_subplot(grid[row, 3 * column : 3 * column + 3])
        for row in range(2)
        for column in range(4)
    ]
    adjusted_axes = [
        fig.add_subplot(grid[row, start : start + 3]) for row in (2, 3) for start in (1, 5, 9)
    ]

    for index, (axis, (target, label, unit)) in enumerate(zip(raw_axes, SURFACE_RAW, strict=True)):
        if target == "canopy_mean_height_m":
            source = pd.read_parquet(ROOT / "data/processed/cairngorms_surface_targets.parquet")
            values = source.loc[target_valid(source), target].to_numpy(float)
        else:
            values = observed.loc[
                observed["variant"].eq("raw") & observed["target"].eq(target), "observed"
            ].to_numpy(float)
        distribution(axis, values, label, unit, GREEN, adjusted=False)
        panel(axis, index)

    for offset, (axis, (target, label, unit)) in enumerate(
        zip(adjusted_axes, SURFACE_ADJUSTED, strict=True)
    ):
        values = observed.loc[
            observed["variant"].eq("height_adjusted") & observed["target"].eq(target),
            "observed",
        ].to_numpy(float)
        distribution(axis, values, label, unit, PURPLE, adjusted=True)
        panel(axis, len(raw_axes) + offset)

    fig.text(
        0.018,
        0.75,
        "Original",
        rotation=90,
        va="center",
        ha="center",
        color=GREEN,
        fontweight="bold",
    )
    fig.text(
        0.018,
        0.27,
        "Height-adjusted",
        rotation=90,
        va="center",
        ha="center",
        color=PURPLE,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.09, right=0.99, top=0.97, bottom=0.075)
    save(fig, "figS01_target_distributions")


def cairngorms_point_cloud() -> None:
    predictions = pd.read_parquet(POINT_CLOUD_PREDICTIONS)
    predictions = predictions[predictions["model"].eq("tessera_v2")].copy()
    predictions["variant"] = np.where(
        predictions["target"].str.endswith("_height_adjusted"), "height_adjusted", "raw"
    )
    predictions["base_target"] = predictions["target"].str.removesuffix("_height_adjusted")
    observed = unique_observed(predictions, ["row_id", "variant", "base_target"])

    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.3))
    for row, (variant, color) in enumerate([("raw", GREEN), ("height_adjusted", PURPLE)]):
        for column, (target, label, unit) in enumerate(POINT_CLOUD):
            axis = axes[row, column]
            values = observed.loc[
                observed["variant"].eq(variant) & observed["base_target"].eq(target), "observed"
            ].to_numpy(float)
            distribution(axis, values, label, unit, color, adjusted=variant == "height_adjusted")
            panel(axis, row * 4 + column)
    fig.text(
        0.018,
        0.72,
        "Original",
        rotation=90,
        va="center",
        ha="center",
        color=GREEN,
        fontweight="bold",
    )
    fig.text(
        0.018,
        0.27,
        "Height-adjusted",
        rotation=90,
        va="center",
        ha="center",
        color=PURPLE,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.10, right=0.99, top=0.94, bottom=0.13, hspace=0.72, wspace=0.68)
    save(fig, "figS10_point_cloud_target_distributions")


def dutch_targets() -> None:
    predictions = pd.read_parquet(DUTCH_PREDICTIONS)
    predictions = predictions[predictions["model"].eq("tessera_v2")]
    observed = unique_observed(predictions, ["row_id", "target_variant", "target"])

    fig = plt.figure(figsize=(7.2, 7.05))
    grid = fig.add_gridspec(4, 12, hspace=0.78, wspace=0.72)
    raw_axes = [
        fig.add_subplot(grid[row, start : start + 3]) for row in (0, 1) for start in (1, 5, 9)
    ]
    adjusted_axes = [
        *[fig.add_subplot(grid[2, start : start + 3]) for start in (1, 5, 9)],
        *[fig.add_subplot(grid[3, start : start + 3]) for start in (3, 7)],
    ]

    for index, (axis, (target, label, unit)) in enumerate(zip(raw_axes, DUTCH, strict=True)):
        values = observed.loc[
            observed["target_variant"].eq("raw") & observed["target"].eq(target), "observed"
        ].to_numpy(float)
        distribution(axis, values, label, unit, GREEN, adjusted=False)
        panel(axis, index)

    adjusted_specs = [spec for spec in DUTCH if spec[0] != "ahn4_p95_height_m"]
    for offset, (axis, (target, label, unit)) in enumerate(
        zip(adjusted_axes, adjusted_specs, strict=True)
    ):
        values = observed.loc[
            observed["target_variant"].eq("height_adjusted") & observed["target"].eq(target),
            "observed",
        ].to_numpy(float)
        distribution(axis, values, label, unit, PURPLE, adjusted=True)
        panel(axis, len(raw_axes) + offset)

    fig.text(
        0.018,
        0.75,
        "Original",
        rotation=90,
        va="center",
        ha="center",
        color=GREEN,
        fontweight="bold",
    )
    fig.text(
        0.018,
        0.27,
        "Height-adjusted",
        rotation=90,
        va="center",
        ha="center",
        color=PURPLE,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.09, right=0.99, top=0.97, bottom=0.075)
    save(fig, "figS11_dutch_target_distributions")


def combined_figure() -> None:
    surface_predictions = pd.read_parquet(SURFACE_PREDICTIONS)
    surface_predictions = surface_predictions[
        surface_predictions["version"].eq("v2") & surface_predictions["model_family"].eq("tessera")
    ]
    surface = unique_observed(surface_predictions, ["row_id", "variant", "target"])
    surface_targets = pd.read_parquet(ROOT / "data/processed/cairngorms_surface_targets.parquet")

    point_predictions = pd.read_parquet(POINT_CLOUD_PREDICTIONS)
    point_predictions = point_predictions[point_predictions["model"].eq("tessera_v2")].copy()
    point_predictions["variant"] = np.where(
        point_predictions["target"].str.endswith("_height_adjusted"),
        "height_adjusted",
        "raw",
    )
    point_predictions["base_target"] = point_predictions["target"].str.removesuffix(
        "_height_adjusted"
    )
    point = unique_observed(point_predictions, ["row_id", "variant", "base_target"])

    dutch_predictions = pd.read_parquet(DUTCH_PREDICTIONS)
    dutch_predictions = dutch_predictions[dutch_predictions["model"].eq("tessera_v2")]
    dutch = unique_observed(dutch_predictions, ["row_id", "target_variant", "target"])

    fig = plt.figure(figsize=(13.0, 12.6))
    outer = fig.add_gridspec(3, 1, height_ratios=[4.2, 3.35, 3.8], hspace=0.34)

    surface_grid = outer[0].subgridspec(
        3, 16, height_ratios=[0.12, 1.0, 1.0], hspace=0.70, wspace=0.88
    )
    surface_header = fig.add_subplot(surface_grid[0, :])
    family_header(surface_header, "a")
    surface_raw_axes = [
        fig.add_subplot(surface_grid[1, 2 * index : 2 * index + 2]) for index in range(8)
    ]
    surface_adjusted_axes = [
        fig.add_subplot(surface_grid[2, 2 + 2 * index : 4 + 2 * index]) for index in range(6)
    ]
    for axis, (target, label, unit) in zip(surface_raw_axes, SURFACE_RAW, strict=True):
        if target == "canopy_mean_height_m":
            values = surface_targets.loc[target_valid(surface_targets), target].to_numpy(float)
        else:
            values = surface.loc[
                surface["variant"].eq("raw") & surface["target"].eq(target), "observed"
            ].to_numpy(float)
        distribution(axis, values, label, unit, GREEN, adjusted=False)
    for axis, (target, label, unit) in zip(surface_adjusted_axes, SURFACE_ADJUSTED, strict=True):
        values = surface.loc[
            surface["variant"].eq("height_adjusted") & surface["target"].eq(target),
            "observed",
        ].to_numpy(float)
        distribution(axis, values, label, unit, PURPLE, adjusted=True)

    point_grid = outer[1].subgridspec(
        3, 12, height_ratios=[0.12, 1.0, 1.0], hspace=1.30, wspace=0.90
    )
    point_header = fig.add_subplot(point_grid[0, :])
    family_header(point_header, "b")
    point_raw_axes = [fig.add_subplot(point_grid[1, start : start + 2]) for start in (1, 4, 7, 10)]
    point_adjusted_axes = [
        fig.add_subplot(point_grid[2, start : start + 2]) for start in (1, 4, 7, 10)
    ]
    for row_axes, variant, color in [
        (point_raw_axes, "raw", GREEN),
        (point_adjusted_axes, "height_adjusted", PURPLE),
    ]:
        for axis, (target, label, unit) in zip(row_axes, POINT_CLOUD, strict=True):
            values = point.loc[
                point["variant"].eq(variant) & point["base_target"].eq(target), "observed"
            ].to_numpy(float)
            distribution(axis, values, label, unit, color, adjusted=variant == "height_adjusted")

    dutch_grid = outer[2].subgridspec(
        3, 12, height_ratios=[0.12, 1.0, 1.0], hspace=0.72, wspace=0.88
    )
    dutch_header = fig.add_subplot(dutch_grid[0, :])
    family_header(dutch_header, "c")
    dutch_raw_axes = [
        fig.add_subplot(dutch_grid[1, 2 * index : 2 * index + 2]) for index in range(6)
    ]
    dutch_adjusted_axes = [
        fig.add_subplot(dutch_grid[2, 1 + 2 * index : 3 + 2 * index]) for index in range(5)
    ]
    for axis, (target, label, unit) in zip(dutch_raw_axes, DUTCH, strict=True):
        values = dutch.loc[
            dutch["target_variant"].eq("raw") & dutch["target"].eq(target), "observed"
        ].to_numpy(float)
        distribution(axis, values, label, unit, GREEN, adjusted=False)
    adjusted_specs = [spec for spec in DUTCH if spec[0] != "ahn4_p95_height_m"]
    for axis, (target, label, unit) in zip(dutch_adjusted_axes, adjusted_specs, strict=True):
        values = dutch.loc[
            dutch["target_variant"].eq("height_adjusted") & dutch["target"].eq(target),
            "observed",
        ].to_numpy(float)
        distribution(axis, values, label, unit, PURPLE, adjusted=True)

    fig.subplots_adjust(left=0.10, right=0.992, top=0.985, bottom=0.055)
    for raw_axes, adjusted_axes in [
        (surface_raw_axes, surface_adjusted_axes),
        (point_raw_axes, point_adjusted_axes),
        (dutch_raw_axes, dutch_adjusted_axes),
    ]:
        row_label(fig, raw_axes, "Original", GREEN)
        row_label(fig, adjusted_axes, "Height-adjusted", PURPLE)

    for first, second in [(outer[0], outer[1]), (outer[1], outer[2])]:
        y = (first.get_position(fig).y0 + second.get_position(fig).y1) / 2
        fig.add_artist(
            Line2D(
                [0.085, 0.992], [y, y], transform=fig.transFigure, color="#D4D8D7", linewidth=0.8
            )
        )

    stem = "figS01_target_distributions_combined"
    save_figure(
        fig,
        stem,
        FIGURES,
        dpi=320,
        facecolor="white",
    )
    plt.close(fig)


def main() -> None:
    configure()
    combined_figure()


if __name__ == "__main__":
    main()
