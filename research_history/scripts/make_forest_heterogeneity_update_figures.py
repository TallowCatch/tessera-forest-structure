#!/usr/bin/env python3
"""Build figures for the corrected LiDAR, replication and TESSERA v2 briefing."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "outputs" / "figures"
TABLE_DIR = ROOT / "outputs" / "tables"
DATA_DIR = ROOT / "data" / "processed"
FOLD_DIR = ROOT / "data" / "interim" / "phase22_cairngorms_spatial_unet" / "folds"

COLORS = {
    "conventional": "#4C78A8",
    "v1": "#148F88",
    "v2": "#E8872D",
    "unet": "#8E63CE",
}
FOLD_COLORS = ["#4C78A8", "#148F88", "#E8872D", "#8E63CE", "#D4505C"]


def style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.75)
    axis.set_axisbelow(True)


def save(figure: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
    figure.savefig(FIGURE_DIR / f"{stem}.png", dpi=240, bbox_inches="tight")
    plt.close(figure)


def make_study_units() -> None:
    cairngorms = pd.read_parquet(
        DATA_DIR / "phase22_cairngorms_surface_targets.parquet"
    )
    fold = np.full(len(cairngorms), -1, dtype=np.int8)
    for fold_number in range(5):
        with np.load(FOLD_DIR / f"balanced_fold_{fold_number}.npz") as split:
            fold[split["test_indices"]] = fold_number
    if np.any(fold < 0):
        raise RuntimeError("Some Cairngorms rows were not assigned to a test fold")
    cairngorms = cairngorms.assign(spatial_fold=fold)

    savelsbos = pd.read_parquet(DATA_DIR / "phase26_ahn4_deciduous_cohort.parquet")

    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.15))
    figure.subplots_adjust(left=0.09, right=0.98, bottom=0.24, top=0.90, wspace=0.28)
    for fold_number, color in enumerate(FOLD_COLORS):
        selected = cairngorms["spatial_fold"] == fold_number
        axes[0].scatter(
            cairngorms.loc[selected, "bng_x"] / 1000,
            cairngorms.loc[selected, "bng_y"] / 1000,
            s=2.2,
            color=color,
            linewidths=0,
            label=f"Fold {fold_number + 1}",
        )
        selected = savelsbos["spatial_fold"] == fold_number
        axes[1].scatter(
            savelsbos.loc[selected, "rd_x"] / 1000,
            savelsbos.loc[selected, "rd_y"] / 1000,
            s=7,
            color=color,
            linewidths=0,
        )

    axes[0].set_title("a  Cairngorms", loc="left", fontweight="bold")
    axes[1].set_title("b  Savelsbos", loc="left", fontweight="bold")
    axes[0].set_xlabel("British National Grid easting (km)")
    axes[0].set_ylabel("British National Grid northing (km)")
    axes[1].set_xlabel("RD New easting (km)")
    axes[1].set_ylabel("RD New northing (km)")
    for axis in axes:
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(color="#E2E2E2", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    figure.legend(
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )
    save(figure, "forest_heterogeneity_update_study_units")


def phase25_value(frame: pd.DataFrame, model: str, target: str) -> float:
    row = frame[(frame["model"] == model) & (frame["target"] == target)]
    if len(row) != 1:
        raise RuntimeError(f"Expected one Phase 25 row for {model}/{target}")
    return float(row.iloc[0]["r2"])


def phase27_value(
    frame: pd.DataFrame,
    site: str,
    target: str,
    variant: str,
    column: str,
) -> float:
    row = frame[
        (frame["site"] == site)
        & (frame["target"] == target)
        & (frame["target_variant"] == variant)
        & (frame["model_family"] == "tessera")
    ]
    if len(row) != 1:
        raise RuntimeError(f"Expected one Phase 27 row for {site}/{target}/{variant}")
    return float(row.iloc[0][column])


def make_corrected_results() -> None:
    phase25 = pd.read_csv(TABLE_DIR / "phase25_cairngorms_corrected_lidar_macro.csv")
    phase27 = pd.read_csv(TABLE_DIR / "phase27_v1_v2_summary.csv")
    targets = [
        ("lidar_height_cv_50m", "Height CV"),
        ("lidar_rcv_50m", "Robust CV"),
        ("lidar_rms_50m", "Height RMS"),
        ("lidar_canopy_shannon_50m", "Shannon"),
    ]

    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.35), constrained_layout=True)
    x = np.arange(len(targets))
    width = 0.24
    for axis, variant, title in zip(
        axes,
        ["raw", "height_adjusted"],
        ["a  Corrected point-cloud outcomes", "b  After accounting for canopy height"],
        strict=True,
    ):
        suffix = "" if variant == "raw" else "_height_adjusted"
        conventional = [
            phase25_value(phase25, "conventional", target + suffix)
            for target, _ in targets
        ]
        v1 = [
            phase27_value(phase27, "Cairngorms", target, variant, "v1_r2")
            for target, _ in targets
        ]
        v2 = [
            phase27_value(phase27, "Cairngorms", target, variant, "v2_r2")
            for target, _ in targets
        ]
        axis.bar(x - width, conventional, width, color=COLORS["conventional"], label="Sentinel + terrain")
        axis.bar(x, v1, width, color=COLORS["v1"], label="TESSERA v1")
        axis.bar(x + width, v2, width, color=COLORS["v2"], label="TESSERA v2")
        axis.axhline(0, color="#555555", linewidth=0.8)
        axis.set_xticks(x, [label for _, label in targets], rotation=22, ha="right")
        axis.set_ylabel(r"Spatially held-out $R^2$")
        axis.set_title(title, loc="left", fontweight="bold")
        style_axis(axis)
    axes[0].set_ylim(-0.025, 0.40)
    axes[1].set_ylim(-0.03, 0.20)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.08))
    save(figure, "forest_heterogeneity_update_corrected_results")


def make_version_comparison() -> None:
    summary = pd.read_csv(TABLE_DIR / "phase27_v1_v2_summary.csv")
    comparisons = [
        ("Cairngorms", "lidar_canopy_shannon_50m", "raw", "Cairngorms\nShannon"),
        ("Cairngorms", "lidar_height_cv_50m", "raw", "Cairngorms\nheight CV"),
        ("Savelsbos", "ahn4_entropy", "raw", "Savelsbos\nentropy"),
        ("Savelsbos", "ahn4_height_cv", "raw", "Savelsbos\nheight CV"),
    ]
    rows = []
    for site, target, variant, label in comparisons:
        row = summary[
            (summary["site"] == site)
            & (summary["target"] == target)
            & (summary["target_variant"] == variant)
            & (summary["model_family"] == "tessera")
        ].iloc[0]
        rows.append((label, row))

    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.35), constrained_layout=True)
    x = np.arange(len(rows))
    width = 0.34
    axes[0].bar(x - width / 2, [r.v1_r2 for _, r in rows], width, color=COLORS["v1"], label="TESSERA v1")
    axes[0].bar(x + width / 2, [r.v2_r2 for _, r in rows], width, color=COLORS["v2"], label="TESSERA v2")
    axes[0].set_xticks(x, [label for label, _ in rows])
    axes[0].set_ylabel(r"Spatially held-out $R^2$")
    axes[0].set_title("a  Replicated version comparison", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, ncol=2, loc="upper left")
    axes[0].set_ylim(0, 0.61)
    style_axis(axes[0])

    selected = [
        ("Cairngorms\nraw Shannon", "Cairngorms", "lidar_canopy_shannon_50m", "raw"),
        ("Cairngorms\nadjusted Shannon", "Cairngorms", "lidar_canopy_shannon_50m", "height_adjusted"),
        ("Savelsbos\nraw entropy", "Savelsbos", "ahn4_entropy", "raw"),
        ("Savelsbos\nraw height CV", "Savelsbos", "ahn4_height_cv", "raw"),
        ("Savelsbos\nadjusted penetration", "Savelsbos", "ahn4_pulse_penetration", "height_adjusted"),
    ]
    delta_rows = []
    for label, site, target, variant in selected:
        row = summary[
            (summary["site"] == site)
            & (summary["target"] == target)
            & (summary["target_variant"] == variant)
            & (summary["model_family"] == "tessera")
        ].iloc[0]
        delta_rows.append((label, row))
    y = np.arange(len(delta_rows))
    estimates = np.array([row.v2_minus_v1_rmse for _, row in delta_rows])
    lower = np.array([row.ci_low for _, row in delta_rows])
    upper = np.array([row.ci_high for _, row in delta_rows])
    colors = [COLORS["v2"] if value < 0 else "#B24B4B" for value in estimates]
    axes[1].errorbar(
        estimates,
        y,
        xerr=np.vstack([estimates - lower, upper - estimates]),
        fmt="none",
        ecolor="#555555",
        elinewidth=1,
        capsize=2,
        zorder=1,
    )
    axes[1].scatter(estimates, y, color=colors, s=34, zorder=2)
    axes[1].axvline(0, color="#555555", linewidth=0.8, linestyle="--")
    axes[1].set_yticks(y, [label for label, _ in delta_rows])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("V2 minus V1 RMSE (lower favours V2)")
    axes[1].set_title("b  Paired spatial-block comparison", loc="left", fontweight="bold")
    axes[1].grid(axis="x", color="#D9D9D9", linewidth=0.6, alpha=0.75)
    axes[1].spines[["top", "right"]].set_visible(False)
    save(figure, "forest_heterogeneity_update_v1_v2")


def make_unet_comparison() -> None:
    macro = pd.read_csv(TABLE_DIR / "phase28_cairngorms_v2_unet_macro.csv")
    comparison = pd.read_csv(TABLE_DIR / "phase28_cairngorms_v2_unet_comparison.csv")
    targets = [
        ("lidar_canopy_shannon_50m", "Shannon\nraw"),
        ("lidar_canopy_shannon_50m_height_adjusted", "Shannon\nadjusted"),
        ("lidar_height_cv_50m", "Height CV\nraw"),
        ("lidar_height_cv_50m_height_adjusted", "Height CV\nadjusted"),
    ]
    models = [
        ("tessera_v2", "V2 5 x 5 MLP", COLORS["v2"]),
        ("tessera_v2_unet", "V2 U-Net", COLORS["unet"]),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.25), constrained_layout=True)
    x = np.arange(len(targets))
    width = 0.34
    for model_index, (model, label, color) in enumerate(models):
        selected = macro[macro["model"] == model].set_index("target")
        axes[0].bar(
            x + (model_index - 0.5) * width,
            [float(selected.loc[target, "r2"]) for target, _ in targets],
            width,
            color=color,
            label=label,
        )
    axes[0].set_xticks(x, [label for _, label in targets])
    axes[0].set_ylabel(r"Spatially held-out $R^2$")
    axes[0].set_title("a  Final architecture comparison", loc="left", fontweight="bold")
    axes[0].set_ylim(0, 0.40)
    axes[0].legend(frameon=False, ncol=2, loc="upper left")
    style_axis(axes[0])

    comparison = comparison.set_index("target").loc[[target for target, _ in targets]]
    estimates = comparison["rmse_difference"].to_numpy(dtype=float)
    lower = comparison["ci_low"].to_numpy(dtype=float)
    upper = comparison["ci_high"].to_numpy(dtype=float)
    y = np.arange(len(targets))
    axes[1].errorbar(
        estimates,
        y,
        xerr=np.vstack([estimates - lower, upper - estimates]),
        fmt="none",
        ecolor="#555555",
        elinewidth=1,
        capsize=2,
        zorder=1,
    )
    axes[1].scatter(estimates, y, color=COLORS["unet"], s=34, zorder=2)
    axes[1].axvline(0, color="#555555", linewidth=0.8, linestyle="--")
    axes[1].set_yticks(y, [label.replace("\n", " ") for _, label in targets])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("U-Net minus 5 x 5 MLP RMSE\n(left favours U-Net)")
    axes[1].set_title("b  Paired spatial-block difference", loc="left", fontweight="bold")
    axes[1].grid(axis="x", color="#D9D9D9", linewidth=0.6, alpha=0.75)
    axes[1].spines[["top", "right"]].set_visible(False)
    save(figure, "forest_heterogeneity_update_unet")


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.facecolor": "white",
        }
    )
    make_study_units()
    make_corrected_results()
    make_version_comparison()
    make_unet_comparison()


if __name__ == "__main__":
    main()
