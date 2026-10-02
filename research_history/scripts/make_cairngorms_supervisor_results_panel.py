#!/usr/bin/env python3
"""Build the compact Phase 19 results panel used in the supervisor brief."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
POOLED = ROOT / "outputs/tables/phase19_cairngorms_pooled_metrics.csv"
PAIRED = ROOT / "outputs/tables/phase19_cairngorms_paired_comparisons.csv"
OUT_PNG = ROOT / "outputs/figures/cairngorms_supervisor_results_panel.png"
OUT_PDF = ROOT / "outputs/figures/cairngorms_supervisor_results_panel.pdf"

INK = "#182B36"
TEAL = "#13877B"
TEAL_LIGHT = "#58B7AC"
BLUE = "#4B86D8"
ORANGE = "#D9772B"
GRAY = "#7A858C"
GRID = "#DCE2E5"


def metric(df: pd.DataFrame, target: str, model: str, field: str) -> float:
    rows = df[(df["target"] == target) & (df["model"] == model)]
    if len(rows) != 1:
        raise ValueError(f"Expected one row for target={target!r}, model={model!r}")
    return float(rows.iloc[0][field])


def main() -> None:
    pooled = pd.read_csv(POOLED)
    paired = pd.read_csv(PAIRED)

    primary = "mean_canopy_shannon_50m"
    conventional_r2 = metric(pooled, primary, "conventional_5_single", "r2")
    tessera_single_r2 = metric(pooled, primary, "tessera_5_single", "r2")
    tessera_multi_r2 = metric(pooled, primary, "tessera_5_multi", "r2")

    target_rows = [
        ("Gap fraction", "mean_gap_fraction_50m", "tessera_5_multi"),
        ("Height variation\namong 10 m cells", "between_cell_height_sd_50m", "tessera_5_multi"),
        ("Height-adjusted\ncanopy entropy", "mean_canopy_shannon_50m_structurally_adjusted", "tessera_5_multi"),
        ("Supplied canopy\nentropy", primary, "tessera_5_multi"),
        ("Three-layer\nvertical entropy", "three_layer_volume_entropy", "tessera_5_profile_aux__vertical_profile"),
    ]
    target_labels = [row[0] for row in target_rows]
    target_r2 = [metric(pooled, row[1], row[2], "r2") for row in target_rows]

    context_models = ["tessera_1_single", "tessera_3_single", "tessera_5_single", "tessera_9_single"]
    context_m = [10, 30, 50, 90]
    context_rmse = [metric(pooled, primary, model, "rmse") for model in context_models]

    fair = paired.loc[paired["comparison"] == "fair_50m_single"].iloc[0]
    rmse_reduction = 100.0 * float(fair["rmse_reduction_fraction"])

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.4,
            "axes.titlesize": 9.3,
            "axes.labelsize": 8.6,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.8,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.55), constrained_layout=True)
    fig.patch.set_facecolor("white")

    # a. Matched baseline and exploratory multi-target model.
    ax = axes[0]
    labels = ["Conventional 50 m", "TESSERA 50 m", "TESSERA multi-target"]
    values = [conventional_r2, tessera_single_r2, tessera_multi_r2]
    y = np.arange(len(labels))
    bars = ax.barh(y, values, color=[BLUE, TEAL, TEAL_LIGHT], height=0.62)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_title(r"$\bf{a}$  Canopy-entropy prediction", loc="left", pad=8)
    ax.set_xlabel(r"Out-of-fold $R^2$")
    ax.set_xlim(0, 0.50)
    ax.grid(axis="x", color=GRID, linewidth=0.65)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(value + 0.012, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", ha="left", va="center")
    ax.text(
        0.02,
        -0.23,
        f"Single-target RMSE: {rmse_reduction:.1f}% lower with TESSERA",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.3,
        color=GRAY,
        clip_on=False,
    )

    # b. Which components of structure were recoverable?
    ax = axes[1]
    y = np.arange(len(target_labels))
    colors = [TEAL, TEAL, ORANGE, ORANGE, GRAY]
    bars = ax.barh(y, target_r2, color=colors, height=0.68)
    ax.set_yticks(y, target_labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.85)
    ax.set_xlabel(r"Out-of-fold $R^2$")
    ax.set_title(r"$\bf{b}$  Structural targets", loc="left", pad=8)
    ax.grid(axis="x", color=GRID, linewidth=0.65)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, target_r2):
        ax.text(value + 0.015, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=7.6)

    # c. Context-size ablation.
    ax = axes[2]
    ax.plot(context_m, context_rmse, color=TEAL, marker="o", linewidth=1.8, markersize=4.6)
    best_idx = int(np.argmin(context_rmse))
    ax.scatter([context_m[best_idx]], [context_rmse[best_idx]], color=ORANGE, s=38, zorder=4)
    ax.set_xticks([10, 50, 90])
    ax.set_xlabel("TESSERA context width (m)")
    ax.set_ylabel("Out-of-fold RMSE")
    ax.set_title(r"$\bf{c}$  Spatial context", loc="left", pad=8)
    ax.grid(color=GRID, linewidth=0.65)
    ax.set_axisbelow(True)
    ax.annotate(
        "Best: 50 m",
        xy=(context_m[best_idx], context_rmse[best_idx]),
        xytext=(60, context_rmse[best_idx] + 0.0046),
        arrowprops={"arrowstyle": "-", "color": GRAY, "lw": 0.8},
        fontsize=7.6,
        color=GRAY,
    )
    ax.set_ylim(min(context_rmse) - 0.0012, max(context_rmse) + 0.0022)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_PDF, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
