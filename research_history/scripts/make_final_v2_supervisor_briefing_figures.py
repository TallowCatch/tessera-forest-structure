#!/usr/bin/env python3
"""Create publication-style figures for the final three-page v2 briefing."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "outputs/figures"
V1_COLOR = "#1B8E87"
V2_COLOR = "#E98527"
UNET_COLOR = "#855CC7"
GRID_COLOR = "#D9D9D9"


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "figure.dpi": 180,
            "savefig.dpi": 300,
        }
    )


def panel_label(axis: plt.Axes, letter: str, title: str) -> None:
    axis.text(
        -0.12,
        1.05,
        letter,
        transform=axis.transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom",
    )
    axis.set_title(title, loc="left", fontweight="bold", pad=8)


def save(fig: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def make_surface_summary() -> None:
    macro = pd.read_csv(
        ROOT / "outputs/tables/phase29_cairngorms_v2_surface_macro.csv"
    )
    comparison = pd.read_csv(
        ROOT
        / "outputs/tables/phase29_cairngorms_v2_surface_v1_v2_comparison.csv"
    )
    macro = macro[macro["model_family"] == "tessera"]
    comparison = comparison[comparison["model_family"] == "tessera"]
    orders = {
        "raw": [
            "canopy_p95_height_m",
            "canopy_surface_sd_m",
            "canopy_surface_cv",
            "canopy_surface_rcv",
            "canopy_rumple",
            "canopy_open_fraction",
        ],
        "height_adjusted": [
            "canopy_surface_sd_m",
            "canopy_surface_cv",
            "canopy_surface_rcv",
            "canopy_rumple",
            "canopy_open_fraction",
            "canopy_height_kurtosis",
        ],
    }
    labels = {
        "canopy_p95_height_m": "P95\nheight",
        "canopy_surface_sd_m": "Height\nSD",
        "canopy_surface_cv": "Height\nCV",
        "canopy_surface_rcv": "Robust\nCV",
        "canopy_rumple": "Rumple",
        "canopy_open_fraction": "Canopy\nopenings",
        "canopy_height_kurtosis": "Kurtosis",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.65), constrained_layout=True)
    for axis, variant, letter, title in [
        (axes[0], "raw", "a", "Original canopy-surface measurements"),
        (
            axes[1],
            "height_adjusted",
            "b",
            "Remaining variation after accounting for canopy height",
        ),
    ]:
        targets = orders[variant]
        x = np.arange(len(targets))
        width = 0.36
        for offset, version, color in [
            (-width / 2, "v1", V1_COLOR),
            (width / 2, "v2", V2_COLOR),
        ]:
            values = [
                float(
                    macro[
                        (macro.variant == variant)
                        & (macro.version == version)
                        & (macro.target == target)
                    ].r2.iloc[0]
                )
                for target in targets
            ]
            bars = axis.bar(
                x + offset,
                values,
                width,
                color=color,
                label=f"TESSERA {version}",
                edgecolor="none",
            )
            if version == "v2":
                for bar, target in zip(bars, targets):
                    row = comparison[
                        (comparison.variant == variant)
                        & (comparison.target == target)
                    ].iloc[0]
                    if row.ci_high < 0:
                        axis.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.018,
                            "*",
                            ha="center",
                            va="bottom",
                            fontsize=11,
                        )
        panel_label(axis, letter, title)
        axis.axhline(0, color="black", linewidth=0.7)
        axis.set_xticks(x, [labels[target] for target in targets])
        axis.set_ylabel(r"Mean spatially held-out $R^2$")
        axis.grid(axis="y", color=GRID_COLOR, linewidth=0.6, alpha=0.8)
        axis.set_axisbelow(True)
        axis.set_ylim(0, 0.88 if variant == "raw" else 0.74)
    axes[0].legend(frameon=False, ncol=2, loc="upper left")
    axes[1].text(
        0.99,
        0.98,
        "* paired 95% block interval favours v2",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        color="#444444",
    )
    save(fig, "final_v2_surface_summary")


def make_evidence_summary() -> None:
    cairngorms = pd.read_csv(
        ROOT / "outputs/tables/phase27_cairngorms_v2_metrics.csv"
    )
    savelsbos = pd.read_csv(
        ROOT / "outputs/tables/phase27_savelsbos_v2_metrics.csv"
    )
    unet = pd.read_csv(
        ROOT / "outputs/tables/phase28_cairngorms_v2_unet_macro.csv"
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), constrained_layout=True)

    axis = axes[0]
    targets = [
        ("lidar_canopy_shannon_50m", "Raw"),
        ("lidar_canopy_shannon_50m_height_adjusted", "Height-\nadjusted"),
    ]
    x = np.arange(len(targets))
    width = 0.34
    for offset, version, color in [
        (-width / 2, "tessera_v1", V1_COLOR),
        (width / 2, "tessera_v2", V2_COLOR),
    ]:
        values = [
            float(
                cairngorms[
                    (cairngorms.model == version) & (cairngorms.target == target)
                ].r2.iloc[0]
            )
            for target, _ in targets
        ]
        axis.bar(x + offset, values, width, color=color, label=version.replace("tessera_", ""))
    panel_label(axis, "a", "Corrected Cairngorms Shannon entropy")
    axis.set_xticks(x, [label for _, label in targets])
    axis.set_ylabel(r"Mean spatially held-out $R^2$")
    axis.set_ylim(0, 0.24)
    axis.grid(axis="y", color=GRID_COLOR, linewidth=0.6)
    axis.legend(frameon=False, ncol=2, loc="upper left")

    axis = axes[1]
    savelsbos = savelsbos[
        (savelsbos.scope == "pooled")
        & (savelsbos.target_variant == "raw")
        & savelsbos.model.isin(["tessera_v1", "tessera_v2"])
    ]
    targets = [
        ("ahn4_p95_height_m", "P95\nheight"),
        ("ahn4_height_sd_m", "Height\nSD"),
        ("ahn4_height_cv", "Height\nCV"),
        ("ahn4_entropy", "Entropy"),
        ("ahn4_pulse_penetration", "Pulse\npenetration"),
    ]
    x = np.arange(len(targets))
    width = 0.34
    for offset, version, color in [
        (-width / 2, "tessera_v1", V1_COLOR),
        (width / 2, "tessera_v2", V2_COLOR),
    ]:
        values = [
            float(
                savelsbos[
                    (savelsbos.model == version) & (savelsbos.target == target)
                ].r2.iloc[0]
            )
            for target, _ in targets
        ]
        axis.bar(x + offset, values, width, color=color)
    panel_label(axis, "b", "Savelsbos deciduous-forest replication")
    axis.axhline(0, color="black", linewidth=0.7)
    axis.set_xticks(x, [label for _, label in targets])
    axis.set_ylabel(r"Pooled spatially held-out $R^2$")
    axis.set_ylim(-0.16, 0.60)
    axis.grid(axis="y", color=GRID_COLOR, linewidth=0.6)

    axis = axes[2]
    targets = [
        ("lidar_canopy_shannon_50m", "Shannon\nraw"),
        ("lidar_canopy_shannon_50m_height_adjusted", "Shannon\nadjusted"),
        ("lidar_height_cv_50m", "Height CV\nraw"),
        ("lidar_height_cv_50m_height_adjusted", "Height CV\nadjusted"),
    ]
    x = np.arange(len(targets))
    width = 0.34
    for offset, model, color, label in [
        (-width / 2, "tessera_v2", V2_COLOR, "5 x 5 MLP"),
        (width / 2, "tessera_v2_unet", UNET_COLOR, "U-Net"),
    ]:
        values = [
            float(
                unet[(unet.model == model) & (unet.target == target)].r2.iloc[0]
            )
            for target, _ in targets
        ]
        axis.bar(x + offset, values, width, color=color, label=label)
    panel_label(axis, "c", "Final v2 architecture decision")
    axis.axhline(0, color="black", linewidth=0.7)
    axis.set_xticks(x, [label for _, label in targets])
    axis.set_ylabel(r"Mean spatially held-out $R^2$")
    axis.set_ylim(0, 0.42)
    axis.grid(axis="y", color=GRID_COLOR, linewidth=0.6)
    axis.legend(frameon=False, loc="upper right")
    for axis in axes:
        axis.set_axisbelow(True)
    save(fig, "final_v2_cross_dataset_evidence")


def make_prediction_density() -> None:
    predictions = pd.read_parquet(
        ROOT / "data/processed/phase29_cairngorms_v2_surface_predictions.parquet"
    )
    predictions = predictions[
        (predictions["variant"] == "raw")
        & (predictions["version"] == "v2")
        & (predictions["model_family"] == "tessera")
    ]
    macro = pd.read_csv(
        ROOT / "outputs/tables/phase29_cairngorms_v2_surface_macro.csv"
    )
    macro = macro[
        (macro["variant"] == "raw")
        & (macro["version"] == "v2")
        & (macro["model_family"] == "tessera")
    ]
    targets = [
        ("canopy_p95_height_m", "P95 canopy height", "m"),
        ("canopy_surface_sd_m", "Height SD", "m"),
        ("canopy_surface_cv", "Height CV", ""),
        ("canopy_surface_rcv", "Robust CV", ""),
        ("canopy_rumple", "Normalized rumple", ""),
        ("canopy_open_fraction", "Canopy openings", ""),
    ]
    figure, axes = plt.subplots(
        2, 3, figsize=(9.7, 6.0), constrained_layout=True
    )
    image = None
    for panel, (axis, (target, label, unit)) in enumerate(
        zip(axes.ravel(), targets)
    ):
        local = predictions[predictions["target"] == target]
        observed = local["observed"].to_numpy(dtype=np.float64)
        predicted = local["predicted"].to_numpy(dtype=np.float64)
        low = float(min(observed.min(), predicted.min()))
        high = float(max(observed.max(), predicted.max()))
        padding = max((high - low) * 0.035, 1e-6)
        image = axis.hexbin(
            observed,
            predicted,
            gridsize=52,
            mincnt=1,
            cmap="viridis",
            norm=LogNorm(vmin=1, vmax=300),
            linewidths=0,
            rasterized=True,
        )
        axis.plot(
            [low, high],
            [low, high],
            color="#B22222",
            linestyle="--",
            linewidth=0.9,
        )
        axis.set_xlim(low - padding, high + padding)
        axis.set_ylim(low - padding, high + padding)
        unit_suffix = f" ({unit})" if unit else ""
        axis.set_xlabel(f"Observed{unit_suffix}")
        axis.set_ylabel(f"TESSERA v2 prediction{unit_suffix}")
        panel_label(axis, chr(97 + panel), label)
        values = macro[macro["target"] == target].iloc[0]
        rmse_suffix = f" {unit}" if unit else ""
        axis.text(
            0.04,
            0.96,
            rf"Mean $R^2$ = {values.r2:.3f}"
            + "\n"
            + rf"RMSE = {values.rmse:.3f}{rmse_suffix}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=7.5,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.84,
                "pad": 1.8,
            },
        )
        axis.grid(False)
    if image is not None:
        colorbar = figure.colorbar(image, ax=axes, shrink=0.75, pad=0.012)
        colorbar.set_label("Observations per hexagon (log scale)")
    save(figure, "final_v2_prediction_density")

    compact_targets = [targets[0], targets[2], targets[5]]
    figure, axes = plt.subplots(
        1, 3, figsize=(9.7, 3.15), constrained_layout=True
    )
    image = None
    for panel, (axis, (target, label, unit)) in enumerate(
        zip(axes, compact_targets)
    ):
        local = predictions[predictions["target"] == target]
        observed = local["observed"].to_numpy(dtype=np.float64)
        predicted = local["predicted"].to_numpy(dtype=np.float64)
        low = float(min(observed.min(), predicted.min()))
        high = float(max(observed.max(), predicted.max()))
        padding = max((high - low) * 0.035, 1e-6)
        image = axis.hexbin(
            observed,
            predicted,
            gridsize=48,
            mincnt=1,
            cmap="viridis",
            norm=LogNorm(vmin=1, vmax=300),
            linewidths=0,
            rasterized=True,
        )
        axis.plot(
            [low, high], [low, high], color="#B22222", linestyle="--", linewidth=0.9
        )
        axis.set_xlim(low - padding, high + padding)
        axis.set_ylim(low - padding, high + padding)
        unit_suffix = f" ({unit})" if unit else ""
        axis.set_xlabel(f"Observed{unit_suffix}")
        axis.set_ylabel(f"TESSERA v2 prediction{unit_suffix}")
        panel_label(axis, chr(97 + panel), label)
        values = macro[macro["target"] == target].iloc[0]
        rmse_suffix = f" {unit}" if unit else ""
        axis.text(
            0.04,
            0.96,
            rf"Mean $R^2$ = {values.r2:.3f}"
            + "\n"
            + rf"RMSE = {values.rmse:.3f}{rmse_suffix}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=7.5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84, "pad": 1.8},
        )
        axis.grid(False)
    if image is not None:
        colorbar = figure.colorbar(image, ax=axes, shrink=0.78, pad=0.012)
        colorbar.set_label("Observations per hexagon (log scale)")
    save(figure, "final_v2_prediction_density_compact")


def main() -> None:
    style()
    make_surface_summary()
    make_evidence_summary()
    make_prediction_density()


if __name__ == "__main__":
    main()
