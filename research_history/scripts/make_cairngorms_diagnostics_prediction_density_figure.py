#!/usr/bin/env python3
"""Create observed-versus-predicted density panels for Phase 19."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_PATH = (
    ROOT / "data/processed/phase19_cairngorms_oof_predictions.parquet"
)
METRICS_PATH = (
    ROOT / "outputs/tables/phase19_cairngorms_pooled_metrics.csv"
)
PNG_PATH = (
    ROOT / "outputs/figures/phase19_cairngorms_prediction_density.png"
)
PDF_PATH = (
    ROOT / "outputs/figures/phase19_cairngorms_prediction_density.pdf"
)
CAPTION_PATH = (
    ROOT
    / "outputs/reports/phase19_cairngorms_prediction_density_caption.txt"
)


PANELS = [
    {
        "letter": "a",
        "title": "Canopy Shannon entropy",
        "target": "mean_canopy_shannon_50m",
        "model": "tessera_5_multi",
        "context": "50 m context",
        "axis_label": "Shannon entropy",
        "unit": "nats",
        "limits": (0.2, 1.55),
        "digits": 3,
    },
    {
        "letter": "b",
        "title": "Mean canopy height",
        "target": "mean_height_50m",
        "model": "tessera_9_multi",
        "context": "90 m context",
        "axis_label": "mean height",
        "unit": "m",
        "limits": (9.5, 32.0),
        "digits": 2,
    },
    {
        "letter": "c",
        "title": "P95 canopy height",
        "target": "mean_p95_height_50m",
        "model": "tessera_9_multi",
        "context": "90 m context",
        "axis_label": "P95 height",
        "unit": "m",
        "limits": (11.5, 40.0),
        "digits": 2,
    },
    {
        "letter": "d",
        "title": "Canopy gap fraction",
        "target": "mean_gap_fraction_50m",
        "model": "tessera_5_multi",
        "context": "50 m context",
        "axis_label": "gap fraction",
        "unit": "proportion",
        "limits": (0.0, 0.56),
        "digits": 3,
    },
]


def panel_data(
    predictions: pd.DataFrame,
    target: str,
    model: str,
) -> pd.DataFrame:
    values = predictions.loc[
        predictions["target"].eq(target)
        & predictions["model"].eq(model),
        ["observed", "predicted"],
    ].dropna()
    if values.empty:
        raise RuntimeError(f"No predictions found for {target}/{model}")
    return values


def metric_row(
    metrics: pd.DataFrame,
    target: str,
    model: str,
) -> pd.Series:
    rows = metrics.loc[
        metrics["target"].eq(target) & metrics["model"].eq(model)
    ]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one metric row for {target}/{model}")
    return rows.iloc[0]


def maximum_bin_count(
    predictions: pd.DataFrame,
    bins: int,
) -> int:
    maximum = 1
    for panel in PANELS:
        values = panel_data(
            predictions,
            str(panel["target"]),
            str(panel["model"]),
        )
        counts, _, _ = np.histogram2d(
            values["observed"],
            values["predicted"],
            bins=bins,
            range=[panel["limits"], panel["limits"]],
        )
        maximum = max(maximum, int(counts.max()))
    return maximum


def annotation(metric: pd.Series, unit: str, digits: int) -> str:
    suffix = "" if unit == "proportion" else f" {unit}"
    return "\n".join(
        [
            rf"$R^2$ = {float(metric.r2):.3f}",
            f"RMSE = {float(metric.rmse):.{digits}f}{suffix}",
            f"Bias = {float(metric.bias):.{digits}f}{suffix}",
            rf"$r_s$ = {float(metric.spearman_r):.3f}",
        ]
    )


def main() -> int:
    if not PREDICTIONS_PATH.exists() or not METRICS_PATH.exists():
        raise RuntimeError("Phase 19 evaluation outputs are missing")
    predictions = pd.read_parquet(PREDICTIONS_PATH)
    metrics = pd.read_csv(METRICS_PATH)
    bins = 54
    norm = LogNorm(vmin=1, vmax=maximum_bin_count(predictions, bins))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
        }
    )
    figure, axes = plt.subplots(
        1,
        4,
        figsize=(14.2, 4.15),
        constrained_layout=True,
    )
    density = None
    for axis, panel in zip(axes, PANELS, strict=True):
        values = panel_data(
            predictions,
            str(panel["target"]),
            str(panel["model"]),
        )
        limits = tuple(float(value) for value in panel["limits"])
        density = axis.hexbin(
            values["observed"],
            values["predicted"],
            gridsize=bins,
            extent=(limits[0], limits[1], limits[0], limits[1]),
            mincnt=1,
            cmap="viridis",
            norm=norm,
            linewidths=0,
            rasterized=True,
        )
        axis.plot(
            limits,
            limits,
            color="#d62728",
            linewidth=1.1,
            linestyle="--",
            label="1:1",
        )
        axis.set_xlim(limits)
        axis.set_ylim(limits)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(color="#d9d9d9", linewidth=0.5, alpha=0.55)
        axis.set_axisbelow(True)
        axis.set_title(
            rf"$\bf{{{panel['letter']}}}$  {panel['title']}"
            f"\n{panel['context']}",
            loc="left",
            pad=8,
        )
        axis.set_xlabel(
            f"Observed {panel['axis_label']}"
            + (
                f" ({panel['unit']})"
                if panel["unit"] != "proportion"
                else " (proportion)"
            )
        )
        axis.set_ylabel(
            f"Predicted {panel['axis_label']}"
            + (
                f" ({panel['unit']})"
                if panel["unit"] != "proportion"
                else " (proportion)"
            )
        )
        metric = metric_row(
            metrics,
            str(panel["target"]),
            str(panel["model"]),
        )
        axis.text(
            0.04,
            0.96,
            annotation(
                metric,
                str(panel["unit"]),
                int(panel["digits"]),
            ),
            transform=axis.transAxes,
            ha="left",
            va="top",
            linespacing=1.35,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.78,
                "pad": 3.5,
            },
        )

    if density is None:
        raise RuntimeError("No density panels were drawn")
    colorbar = figure.colorbar(
        density,
        ax=axes,
        location="right",
        shrink=0.79,
        pad=0.015,
    )
    colorbar.set_label("Forest units per hexagon (log scale)")
    figure.suptitle(
        "Spatially held-out TESSERA predictions of forest structure",
        fontsize=14,
        y=1.04,
    )
    PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(PNG_PATH, dpi=300, bbox_inches="tight")
    figure.savefig(PDF_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)

    caption = (
        "Observed-versus-predicted density plots for the best frozen "
        "multi-target TESSERA models. Predictions are pooled from five "
        "complete spatial holdouts with a 1 km exclusion buffer around each "
        "test region (n = 26,454 non-overlapping 50 m mature-forest units). "
        "Panels show (a) canopy Shannon entropy, expressed in nats; (b) mean "
        "canopy height; (c) mean 95th-percentile canopy height; and (d) canopy "
        "gap fraction. Height panels are expressed in metres, whereas entropy "
        "and gap fraction are dimensionless. Colours show observation density "
        "on a shared logarithmic scale; dashed red lines denote exact 1:1 "
        "agreement."
    )
    CAPTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    CAPTION_PATH.write_text(caption + "\n", encoding="utf-8")
    print(PNG_PATH)
    print(PDF_PATH)
    print(CAPTION_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
