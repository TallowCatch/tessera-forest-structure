#!/usr/bin/env python3
"""Build the paired AHN3-to-AHN4 temporal-transfer figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from figure_output import save_figure

ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS = ROOT / "data/processed/dutch_temporal_transfer_predictions.parquet"
SUMMARY = ROOT / "results/tables/dutch_temporal_transfer_rmse_ratios.csv"
OUTPUT = ROOT / "results/figures/main"

BLUE = "#3f719c"
GREEN = "#2a8c78"
ORANGE = "#d97732"
CHARCOAL = "#42484f"
LIGHT_GREY = "#e5e7e8"

OUTCOMES = [
    "p95_height_m",
    "height_sd_m",
    "height_cv",
    "entropy",
    "pulse_penetration",
    "sigma_z",
]
LABELS = {
    "p95_height_m": "P95 height",
    "height_sd_m": "Height SD",
    "height_cv": "Height CV",
    "entropy": "Return entropy",
    "pulse_penetration": "Pulse penetration",
    "sigma_z": "Sigma-z",
}
COMPARISONS = {
    "temporal_vs_same_time": (
        "predicted_ahn4_temporal",
        "observed_ahn4",
        "predicted_ahn3",
        "observed_ahn3",
    ),
    "temporal_vs_contemporary": (
        "predicted_ahn4_temporal",
        "observed_ahn4",
        "predicted_ahn4_contemporary",
        "observed_ahn4",
    ),
    "predicted_change_vs_zero": (
        "predicted_change",
        "observed_change",
        None,
        "observed_change",
    ),
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def block_sse(
    frame: pd.DataFrame,
    block_codes: np.ndarray,
    block_count: int,
    predicted: str | None,
    observed: str,
) -> np.ndarray:
    observed_values = frame[observed].to_numpy(dtype=np.float64)
    if predicted is None:
        errors = observed_values
    else:
        errors = frame[predicted].to_numpy(dtype=np.float64) - observed_values
    return np.bincount(
        block_codes,
        weights=np.square(errors),
        minlength=block_count,
    )


def bootstrap_ratios(
    predictions: pd.DataFrame,
    replicates: int = 2000,
    seed: int = 20260918,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records: list[dict[str, float | int | str]] = []
    for outcome in OUTCOMES:
        outcome_frame = predictions[predictions["outcome"].eq(outcome)]
        site_draws: dict[str, list[np.ndarray]] = {comparison: [] for comparison in COMPARISONS}
        for _, site_frame in outcome_frame.groupby("site_code", sort=True):
            block_codes, blocks = pd.factorize(site_frame["fold_block"], sort=True)
            block_count = len(blocks)
            row_count = np.bincount(block_codes, minlength=block_count).astype(np.float64)
            sampled_blocks = rng.integers(0, block_count, size=(replicates, block_count))
            sampled_rows = row_count[sampled_blocks].sum(axis=1)
            for comparison, (
                numerator_prediction,
                numerator_observation,
                denominator_prediction,
                denominator_observation,
            ) in COMPARISONS.items():
                numerator_sse = block_sse(
                    site_frame,
                    block_codes,
                    block_count,
                    numerator_prediction,
                    numerator_observation,
                )
                denominator_sse = block_sse(
                    site_frame,
                    block_codes,
                    block_count,
                    denominator_prediction,
                    denominator_observation,
                )
                numerator_rmse = np.sqrt(numerator_sse[sampled_blocks].sum(axis=1) / sampled_rows)
                denominator_rmse = np.sqrt(
                    denominator_sse[sampled_blocks].sum(axis=1) / sampled_rows
                )
                site_draws[comparison].append(numerator_rmse / denominator_rmse)
        for comparison, values in site_draws.items():
            equal_site_draws = np.median(np.vstack(values), axis=0)
            lower, median, upper = np.quantile(equal_site_draws, [0.025, 0.5, 0.975])
            records.append(
                {
                    "outcome": outcome,
                    "comparison": comparison,
                    "median_rmse_ratio": float(median),
                    "lower_95": float(lower),
                    "upper_95": float(upper),
                    "sites": int(outcome_frame["site_code"].nunique()),
                    "replicates": replicates,
                }
            )
    return pd.DataFrame(records)


def plot_interval(
    axis: plt.Axes,
    frame: pd.DataFrame,
    positions: np.ndarray,
    *,
    offset: float,
    color: str,
    marker: str,
    label: str,
) -> None:
    values = frame.set_index("outcome").reindex(OUTCOMES)
    median = values["median_rmse_ratio"].to_numpy()
    lower = values["lower_95"].to_numpy()
    upper = values["upper_95"].to_numpy()
    axis.errorbar(
        median,
        positions + offset,
        xerr=np.vstack([median - lower, upper - median]),
        fmt=marker,
        color=color,
        markersize=4.5,
        markeredgewidth=0.7,
        capsize=2.0,
        elinewidth=1.0,
        label=label,
        zorder=3,
    )


def make_figure(summary: pd.DataFrame) -> None:
    positions = np.arange(len(OUTCOMES))
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.35), sharey=True)
    for axis in axes:
        axis.axvline(1.0, color=CHARCOAL, linestyle="--", linewidth=0.8, zorder=1)
        axis.set_yticks(positions, [LABELS[outcome] for outcome in OUTCOMES])
        axis.grid(axis="x", color=LIGHT_GREY, linewidth=0.6, zorder=0)
    axes[0].invert_yaxis()

    plot_interval(
        axes[0],
        summary[summary["comparison"].eq("temporal_vs_same_time")],
        positions,
        offset=-0.11,
        color=BLUE,
        marker="o",
        label="Same-time AHN3",
    )
    plot_interval(
        axes[0],
        summary[summary["comparison"].eq("temporal_vs_contemporary")],
        positions,
        offset=0.11,
        color=ORANGE,
        marker="s",
        label="Contemporaneous AHN4",
    )
    plot_interval(
        axes[1],
        summary[summary["comparison"].eq("predicted_change_vs_zero")],
        positions,
        offset=0.0,
        color=GREEN,
        marker="D",
        label="Zero-change baseline",
    )
    axes[0].set_xlabel("Temporal-transfer RMSE / comparator RMSE")
    axes[1].set_xlabel("Predicted-change RMSE / zero-change RMSE")
    axes[0].set_xlim(0.8, 3.25)
    axes[1].set_xlim(0.75, 2.35)
    axes[0].text(
        -0.19,
        1.04,
        "a",
        transform=axes[0].transAxes,
        fontsize=10.5,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    axes[1].text(
        -0.16,
        1.04,
        "b",
        transform=axes[1].transAxes,
        fontsize=10.5,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    handles = [*axes[0].get_legend_handles_labels()[0], *axes[1].get_legend_handles_labels()[0]]
    labels = [*axes[0].get_legend_handles_labels()[1], *axes[1].get_legend_handles_labels()[1]]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.57, 1.01),
        handletextpad=0.4,
        columnspacing=1.4,
    )
    fig.subplots_adjust(left=0.18, right=0.985, top=0.86, bottom=0.20, wspace=0.25)
    save_figure(fig, "figS_temporal_transfer", OUTPUT, dpi=360)
    plt.close(fig)


def main() -> None:
    configure_style()
    predictions = pd.read_parquet(PREDICTIONS)
    summary = bootstrap_ratios(predictions)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY, index=False)
    make_figure(summary)
    print(f"Wrote {OUTPUT / 'figS_temporal_transfer.pdf'}")
    print(f"Wrote {OUTPUT / 'figS_temporal_transfer.png'}")
    print(f"Wrote {SUMMARY}")


if __name__ == "__main__":
    main()
