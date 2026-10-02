#!/usr/bin/env python3
"""Create the grayscale Phase 19 figure for the scientific supervisor note."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
POOLED = ROOT / "outputs/tables/phase19_cairngorms_pooled_metrics.csv"
PAIRED = ROOT / "outputs/tables/phase19_cairngorms_paired_comparisons.csv"
PREDICTIONS = ROOT / "data/processed/phase19_cairngorms_oof_predictions.parquet"
OUT_PDF = ROOT / "outputs/figures/cairngorms_scientific_results.pdf"
OUT_PNG = ROOT / "outputs/figures/cairngorms_scientific_results.png"


def main() -> None:
    pooled = pd.read_csv(POOLED)
    paired = pd.read_csv(PAIRED).set_index("comparison")
    predictions = pd.read_parquet(PREDICTIONS)
    primary = "mean_canopy_shannon_50m"

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 8.2,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.4,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "axes.linewidth": 0.75,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.35))

    # a. Identical-support comparison.
    ax = axes[0, 0]
    models = [
        "conventional_centre_single",
        "tessera_1_single",
        "conventional_5_single",
        "tessera_5_single",
    ]
    labels = ["Conv.\n10 m", "TESSERA\n10 m", "Conv.\n50 m", "TESSERA\n50 m"]
    rows = (
        pooled.loc[pooled["target"].eq(primary) & pooled["model"].isin(models)]
        .set_index("model")
        .reindex(models)
    )
    bars = ax.bar(
        np.arange(4),
        rows["r2"],
        color=["0.80", "0.35", "0.68", "0.10"],
        edgecolor="black",
        linewidth=0.55,
    )
    for index in [0, 2]:
        bars[index].set_hatch("////")
    for bar, value in zip(bars, rows["r2"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.010,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=7.3,
        )
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(np.arange(4), labels)
    ax.set_ylim(0, 0.45)
    ax.set_ylabel(r"Out-of-fold $R^2$")
    ax.set_title(r"$\bf{a}$  Matched spatial support", loc="left")

    # b. Context-size ablation.
    ax = axes[0, 1]
    context_models = [
        "tessera_1_single",
        "tessera_3_single",
        "tessera_5_single",
        "tessera_9_single",
    ]
    context = (
        pooled.loc[
            pooled["target"].eq(primary)
            & pooled["model"].isin(context_models)
        ]
        .set_index("model")
        .reindex(context_models)
    )
    context_m = [10, 30, 50, 90]
    ax.plot(
        context_m,
        context["rmse"],
        color="black",
        marker="o",
        linewidth=1.35,
        markersize=4.0,
    )
    ax.scatter(
        [50],
        [float(context.loc["tessera_5_single", "rmse"])],
        color="white",
        edgecolor="black",
        s=34,
        zorder=3,
    )
    ax.set_xticks(context_m)
    ax.set_xlabel("TESSERA context width (m)")
    ax.set_ylabel("Out-of-fold RMSE")
    ax.set_title(r"$\bf{b}$  Context-size ablation", loc="left")
    ax.grid(axis="y", color="0.86", linewidth=0.55)

    # c. Paired complete-block comparisons.
    ax = axes[1, 0]
    selected_names = [
        "context_30m_vs_10m",
        "context_50m_vs_10m",
        "context_90m_vs_10m",
        "multitask_effect_50m",
        "profile_aux_effect_50m",
        "mlp_vs_hgb_tessera_50m",
    ]
    labels = [
        "30 vs 10 m",
        "50 vs 10 m",
        "90 vs 10 m",
        "Multi-target vs single",
        "Profile-assisted vs single",
        "MLP vs gradient boosting",
    ]
    selected = paired.reindex(selected_names)
    y = np.arange(len(selected))
    point = selected["rmse_delta"].to_numpy()
    lower = point - selected["rmse_delta_ci_lower"].to_numpy()
    upper = selected["rmse_delta_ci_upper"].to_numpy() - point
    ax.errorbar(
        point,
        y,
        xerr=np.vstack([lower, upper]),
        fmt="o",
        color="black",
        ecolor="0.35",
        markersize=3.7,
        capsize=2.4,
        linewidth=0.9,
    )
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Paired RMSE change (candidate - reference)")
    ax.set_title(r"$\bf{c}$  Complete-block comparisons", loc="left")
    ax.grid(axis="x", color="0.88", linewidth=0.5)

    # d. The explicit vertical-profile auxiliary output.
    ax = axes[1, 1]
    entropy = predictions.loc[
        predictions["target"].eq("three_layer_volume_entropy")
        & predictions["model"].eq(
            "tessera_5_profile_aux__vertical_profile"
        )
    ]
    ax.hexbin(
        entropy["observed"],
        entropy["predicted"],
        gridsize=42,
        mincnt=1,
        cmap="Greys",
        linewidths=0,
    )
    lower_bound = float(
        min(entropy["observed"].min(), entropy["predicted"].min())
    )
    upper_bound = float(
        max(entropy["observed"].max(), entropy["predicted"].max())
    )
    ax.plot(
        [lower_bound, upper_bound],
        [lower_bound, upper_bound],
        color="black",
        linestyle="--",
        linewidth=0.8,
    )
    ax.text(
        0.03,
        0.96,
        r"$R^2=0.264$" "\n" r"$r_s=0.561$" "\nRMSE = 0.124",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7.4,
    )
    ax.set_xlabel("Observed three-layer entropy")
    ax.set_ylabel("Predicted three-layer entropy")
    ax.set_title(r"$\bf{d}$  Vertical-profile auxiliary task", loc="left")

    for ax in axes.ravel():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=0.8, w_pad=1.1, h_pad=1.0)
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PDF}")
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
