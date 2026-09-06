#!/usr/bin/env python3
"""Create observed-versus-predicted panels for corrected point-cloud outcomes."""

from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from figure_output import save_figure
from matplotlib.colors import LogNorm

ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS = ROOT / "data/processed/cairngorms_point_cloud_predictions.parquet"
METRICS = ROOT / "results/tables/cairngorms_point_cloud_performance.csv"
RESULT_FREEZE = {
    PREDICTIONS: "695d361f8456225013bb1a7d54c5132ed6f1aef9c5f33a317f023e5e03529e99",
    METRICS: "d8a3ccba4cb6c2e3c47aecde7bc7238ebb99d30b8041db58f6b033b5d3e62f07",
}
RESULTS_DIR = ROOT / "results/figures/main"

TARGETS = [
    ("lidar_canopy_shannon_50m", "Normalised Shannon entropy", ""),
    ("lidar_height_cv_50m", "Return-height CV", ""),
    ("lidar_rcv_50m", "Robust return-height CV", ""),
    ("lidar_rms_50m", "RMS return height", "m"),
]
VARIANTS = [
    ("", "Original outcome", "viridis"),
    ("_height_adjusted", "Height-adjusted outcome", "viridis"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen_inputs() -> None:
    for path, expected in RESULT_FREEZE.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"Frozen result hash changed for {path}: {actual}")


def panel_limits(frame: pd.DataFrame, adjusted: bool) -> tuple[float, float]:
    values = frame[["observed", "predicted"]].to_numpy(dtype=np.float64).ravel()
    if adjusted:
        extent = float(np.quantile(np.abs(values), 0.999))
        return -extent, extent
    lower, upper = np.quantile(values, [0.001, 0.999])
    margin = 0.025 * float(upper - lower)
    return float(lower - margin), float(upper + margin)


def maximum_hexbin_count(
    frames: list[pd.DataFrame], limits: list[tuple[float, float]], gridsize: int
) -> float:
    figure, axis = plt.subplots()
    maximum = 1.0
    for frame, extent in zip(frames, limits, strict=True):
        artist = axis.hexbin(
            frame["observed"],
            frame["predicted"],
            gridsize=gridsize,
            mincnt=1,
            extent=(*extent, *extent),
        )
        if len(artist.get_array()):
            maximum = max(maximum, float(np.max(artist.get_array())))
        artist.remove()
    plt.close(figure)
    return maximum


def make_figure() -> plt.Figure:
    predictions = pd.read_parquet(PREDICTIONS)
    predictions = predictions[predictions["model"].eq("tessera_v2")].copy()
    metrics = pd.read_csv(METRICS)
    metrics = metrics[metrics["model"].eq("tessera_v2")].set_index("target")

    expected_targets = {target + suffix for target, _, _ in TARGETS for suffix, _, _ in VARIANTS}
    if set(predictions["target"].unique()) != expected_targets:
        raise RuntimeError("Unexpected corrected point-cloud targets")
    if predictions.groupby("target").size().nunique() != 1:
        raise RuntimeError("Point-cloud targets do not contain the same held-out units")

    ordered_frames = []
    ordered_limits = []
    for suffix, _, _ in VARIANTS:
        for target, _, _ in TARGETS:
            frame = predictions[predictions["target"].eq(target + suffix)]
            ordered_frames.append(frame)
            ordered_limits.append(panel_limits(frame, adjusted=bool(suffix)))

    gridsize = 42
    density_norm = LogNorm(
        vmin=1,
        vmax=maximum_hexbin_count(ordered_frames, ordered_limits, gridsize),
    )
    figure, axes = plt.subplots(2, 4, figsize=(11.3, 6.15))
    panel_letters = iter("abcdefgh")
    density_artist = None

    for row, (suffix, row_label, colour_map) in enumerate(VARIANTS):
        for column, (target, target_label, unit) in enumerate(TARGETS):
            axis = axes[row, column]
            full_target = target + suffix
            frame = predictions[predictions["target"].eq(full_target)]
            lower, upper = panel_limits(frame, adjusted=bool(suffix))
            density_artist = axis.hexbin(
                frame["observed"],
                frame["predicted"],
                gridsize=gridsize,
                mincnt=1,
                extent=(lower, upper, lower, upper),
                cmap=colour_map,
                norm=density_norm,
                linewidths=0,
                rasterized=True,
            )
            axis.plot(
                [lower, upper],
                [lower, upper],
                color="#bf4a43" if row == 0 else "#356b9a",
                linestyle="--",
                linewidth=1.0,
            )
            axis.set_xlim(lower, upper)
            axis.set_ylim(lower, upper)
            axis.set_aspect("equal", adjustable="box")
            axis.grid(color="#d8d8d8", linewidth=0.45, alpha=0.65)
            axis.spines[["top", "right"]].set_visible(False)
            axis.tick_params(labelsize=7.5)
            axis.set_title(target_label, fontsize=9.2, fontweight="bold", pad=5)
            axis.set_xlabel("LiDAR observed" + (f" ({unit})" if unit else ""), fontsize=8)
            if column == 0:
                axis.set_ylabel(
                    row_label + "\nTessera predicted" + (f" ({unit})" if unit else ""),
                    fontsize=8,
                )
            score = metrics.loc[full_target]
            axis.text(
                0.04,
                0.96,
                rf"$R^2={float(score['r2']):.2f}$"
                + "\n"
                + rf"RMSE $={float(score['rmse']):.3f}$"
                + (f" {unit}" if unit else "")
                + "\n"
                + rf"$r_s={float(score['block_spearman']):.2f}$",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=7.4,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.84,
                    "pad": 1.5,
                },
            )
            axis.text(
                -0.16,
                1.05,
                next(panel_letters),
                transform=axis.transAxes,
                fontsize=10.5,
                fontweight="bold",
                ha="left",
                va="bottom",
            )

    assert density_artist is not None
    colour_axis = figure.add_axes([0.927, 0.25, 0.012, 0.50])
    colour_bar = figure.colorbar(density_artist, cax=colour_axis)
    colour_bar.set_label("Observations per hexagon", fontsize=8)
    colour_bar.ax.tick_params(labelsize=7.5)
    figure.subplots_adjust(
        left=0.08,
        right=0.905,
        top=0.94,
        bottom=0.09,
        hspace=0.36,
        wspace=0.30,
    )
    return figure


def main() -> None:
    verify_frozen_inputs()
    figure = make_figure()
    save_figure(
        figure,
        "figS09_point_cloud_predictions",
        RESULTS_DIR,
        dpi=300,
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
