#!/usr/bin/env python3
"""Create the supplementary canopy-height model diagnostic."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from figure_output import save_figure

ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS = ROOT / "data/processed/cairngorms_height_predictions.parquet"
RESULTS_DIR = ROOT / "results/figures/main"

MODELS = [
    ("tessera_v2_mlp_5x5", r"$5\times5$ summary MLP"),
    ("tessera_v2_unet_strict", "U-Net"),
]
TARGETS = [
    ("canopy_mean_height_m", "Mean canopy height"),
    ("canopy_p95_height_m", "P95 canopy height"),
]


def equal_fold_metrics(frame: pd.DataFrame) -> tuple[float, float]:
    """Return equal-fold mean metrics with equal spatial-block influence."""
    rows = []
    for _, fold in frame.groupby("fold"):
        observed = fold["observed"].to_numpy(dtype=np.float64)
        predicted = fold["predicted"].to_numpy(dtype=np.float64)
        blocks = fold["spatial_block"].to_numpy()
        _, inverse, counts = np.unique(blocks, return_inverse=True, return_counts=True)
        weights = 1.0 / counts[inverse].astype(np.float64)
        weights /= weights.sum()
        errors = predicted - observed
        observed_mean = np.sum(weights * observed)
        denominator = np.sum(weights * np.square(observed - observed_mean))
        rows.append(
            (
                1 - np.sum(weights * np.square(errors)) / denominator,
                np.sqrt(np.sum(weights * np.square(errors))),
            )
        )
    return tuple(np.mean(rows, axis=0))


def make_figure() -> plt.Figure:
    predictions = pd.read_parquet(PREDICTIONS)
    predictions = predictions[
        predictions["model"].isin([model for model, _ in MODELS])
        & predictions["target"].isin([target for target, _ in TARGETS])
    ].copy()

    figure, axes = plt.subplots(1, 4, figsize=(11.2, 3.1))
    panel_letters = iter("abcd")
    last_hexbin = None

    for row, (target, target_label) in enumerate(TARGETS):
        target_values = predictions[predictions["target"] == target]
        lower = float(np.nanmin(target_values[["observed", "predicted"]].to_numpy()))
        upper = float(np.nanmax(target_values[["observed", "predicted"]].to_numpy()))
        margin = 0.025 * (upper - lower)
        limits = (lower - margin, upper + margin)

        for column, (model, _) in enumerate(MODELS):
            axis = axes[2 * row + column]
            local = target_values[target_values["model"] == model]
            last_hexbin = axis.hexbin(
                local["observed"],
                local["predicted"],
                gridsize=48,
                mincnt=1,
                bins="log",
                cmap="viridis",
                linewidths=0,
            )
            axis.plot(limits, limits, color="#c64b40", linestyle="--", linewidth=1.1)
            axis.set_xlim(limits)
            axis.set_ylim(limits)
            axis.set_aspect("equal", adjustable="box")
            axis.grid(color="#d7d7d7", linewidth=0.45, alpha=0.65)
            axis.spines[["top", "right"]].set_visible(False)
            axis.tick_params(labelsize=8)

            r2, rmse = equal_fold_metrics(local)
            axis.text(
                0.04,
                0.96,
                rf"$R^2={r2:.3f}$" + "\n" + rf"RMSE $={rmse:.3f}$ m",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.82,
                    "pad": 1.8,
                },
            )
            axis.text(
                -0.13,
                1.04,
                next(panel_letters),
                transform=axis.transAxes,
                fontsize=11,
                fontweight="bold",
                ha="left",
                va="bottom",
            )
            axis.set_xlabel(f"Observed {target_label.lower()} (m)", fontsize=8.5)
            axis.set_ylabel(f"Predicted {target_label.lower()} (m)", fontsize=8.5)

    assert last_hexbin is not None
    color_axis = figure.add_axes([0.94, 0.19, 0.012, 0.70])
    colorbar = figure.colorbar(last_hexbin, cax=color_axis)
    colorbar.set_label("Observations per hexagon (log scale)", fontsize=8.5)
    colorbar.ax.tick_params(labelsize=8)
    figure.subplots_adjust(left=0.055, right=0.915, top=0.94, bottom=0.19, wspace=0.38)
    return figure


def main() -> None:
    figure = make_figure()
    save_figure(
        figure,
        "figS08_height_model_predictions",
        RESULTS_DIR,
        dpi=300,
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
