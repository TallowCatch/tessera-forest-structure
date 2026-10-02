#!/usr/bin/env python3
"""Create publication-ready multi-panel evidence plates for the manuscript."""

from __future__ import annotations

from pathlib import Path
import string

import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "figures" / "paper_plates"

CAIRN_TARGETS = ROOT / "data/processed/phase22_cairngorms_surface_targets.parquet"
CAIRN_PREDICTIONS = ROOT / "data/processed/phase29_cairngorms_v2_surface_predictions.parquet"
CAIRN_MACRO = ROOT / "research_history/outputs/tables/phase29_cairngorms_v2_surface_macro.csv"
HEIGHT_DIR = ROOT / "data/interim/phase23_cairngorms_height_adjusted"
HEIGHT_DIAGNOSTICS = (
    ROOT / "research_history/outputs/tables/premeeting_height_adjustment_diagnostics.csv"
)

DUTCH_SITE_MAP = (
    ROOT
    / "research_history/outputs/figures/phase37_dutch_expanded/phase37_site_locator_map.png"
)
DUTCH_SITE_SUMMARY = ROOT / "research_history/outputs/tables/phase37_dutch_site_summary.csv"
DUTCH_METRICS = ROOT / "research_history/outputs/tables/phase37_dutch_metrics.csv"
RANK_SUMMARY = ROOT / "results/tables/rank_transfer.csv"
REFERENCE_CURVE = ROOT / "results/tables/local_reference_curve.csv"

TEAL = "#238b75"
ORANGE = "#cc6f32"
PURPLE = "#7552aa"
BLUE = "#396c9e"
CHARCOAL = "#3f4a58"
LIGHT_GREY = "#e8ecef"
MID_GREY = "#8b949e"
RED = "#bd4f4a"

KEY_TARGETS = ["canopy_surface_cv", "canopy_rumple", "canopy_open_fraction"]
TARGET_LABELS = {
    "canopy_surface_cv": "Canopy-height CV",
    "canopy_rumple": "Canopy rumple",
    "canopy_open_fraction": "Canopy openings",
    "canopy_surface_sd_m": "Canopy-height SD",
    "canopy_surface_rcv": "Robust height CV",
    "canopy_height_kurtosis": "Height kurtosis",
}
TARGET_UNITS = {
    "canopy_surface_cv": "ratio",
    "canopy_rumple": "ratio",
    "canopy_open_fraction": "fraction",
}
DUTCH_LABELS = {
    "ahn4_p95_height_m": "P95 height",
    "ahn4_height_sd_m": "Height SD",
    "ahn4_height_cv": "Height CV",
    "ahn4_entropy": "Return entropy",
    "ahn4_pulse_penetration": "Pulse penetration",
    "ahn4_sigma_z": "Sigma-z",
}


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def panel(axis: plt.Axes, index: int) -> None:
    axis.text(
        -0.13,
        1.04,
        string.ascii_lowercase[index],
        transform=axis.transAxes,
        fontweight="bold",
        fontsize=11,
        ha="left",
        va="bottom",
    )


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", dpi=360, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def equal_fold_scores(frame: pd.DataFrame) -> tuple[float, float]:
    scores = []
    errors = []
    for _, fold in frame.groupby("fold"):
        observed = fold["observed"].to_numpy(float)
        predicted = fold["predicted"].to_numpy(float)
        residual = observed - predicted
        denominator = np.sum((observed - observed.mean()) ** 2)
        scores.append(1.0 - np.sum(residual**2) / denominator)
        errors.append(float(np.sqrt(np.mean(residual**2))))
    return float(np.mean(scores)), float(np.mean(errors))


def load_height_components(frame: pd.DataFrame) -> np.ndarray:
    all_targets = [
        "canopy_surface_sd_m",
        "canopy_surface_cv",
        "canopy_surface_rcv",
        "canopy_rumple",
        "canopy_open_fraction",
        "canopy_height_kurtosis",
    ]
    values = np.full((len(frame), len(all_targets)), np.nan, dtype=np.float32)
    ownership = np.zeros(len(frame), dtype=np.int8)
    for fold in range(5):
        with np.load(HEIGHT_DIR / f"fold_{fold}.npz") as source:
            test = source["test_indices"].astype(np.int64)
            values[test] = source["height_test_predictions"].astype(np.float32)
            ownership[test] += 1
    if not np.all(ownership == 1) or not np.isfinite(values).all():
        raise RuntimeError("Height components do not cover every row exactly once")
    return values[:, [all_targets.index(target) for target in KEY_TARGETS]]


def make_cairngorms_prediction_plate() -> None:
    targets = pd.read_parquet(CAIRN_TARGETS)
    predictions = pd.read_parquet(CAIRN_PREDICTIONS)
    macro = pd.read_csv(CAIRN_MACRO)
    selected = predictions[
        (predictions["variant"] == "raw")
        & (predictions["version"] == "v2")
        & (predictions["model_family"] == "tessera")
        & predictions["target"].isin(KEY_TARGETS)
    ].copy()

    fig = plt.figure(figsize=(11.3, 6.8))
    grid = fig.add_gridspec(2, 3, height_ratios=(1.0, 1.0), hspace=0.34, wspace=0.27)

    for index, target in enumerate(KEY_TARGETS):
        axis = fig.add_subplot(grid[0, index])
        data = selected[selected["target"] == target]
        official = macro[
            (macro["variant"] == "raw")
            & (macro["version"] == "v2")
            & (macro["model_family"] == "tessera")
            & (macro["target"] == target)
        ].iloc[0]
        r2, rmse = float(official["r2"]), float(official["rmse"])
        image = axis.hexbin(
            data["observed"],
            data["predicted"],
            gridsize=43,
            bins="log",
            mincnt=1,
            cmap="viridis",
            linewidths=0,
        )
        low = float(min(data["observed"].min(), data["predicted"].min()))
        high = float(max(data["observed"].max(), data["predicted"].max()))
        axis.plot([low, high], [low, high], linestyle="--", color=RED, linewidth=1.0)
        axis.set_xlim(low, high)
        axis.set_ylim(low, high)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(TARGET_LABELS[target], loc="left", fontweight="bold")
        axis.set_xlabel(f"LiDAR observed ({TARGET_UNITS[target]})")
        axis.set_ylabel(f"TESSERA predicted ({TARGET_UNITS[target]})")
        axis.text(
            0.04,
            0.95,
            f"$R^2$ = {r2:.2f}\nRMSE = {rmse:.3f}",
            transform=axis.transAxes,
            va="top",
            ha="left",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2.2},
        )
        panel(axis, index)
        if index == 2:
            colorbar = fig.colorbar(image, ax=axis, fraction=0.048, pad=0.025)
            colorbar.set_label("Observations per bin (log scale)")

    # Use the densest held-out 6 km block, selected independently of model error.
    block = targets["spatial_block"].value_counts().index[0]
    spatial = targets.loc[
        targets["spatial_block"] == block,
        ["row_id", "bng_x", "bng_y", "canopy_rumple"],
    ].copy()
    rumple = selected[selected["target"] == "canopy_rumple"][["row_id", "predicted"]]
    spatial = spatial.merge(rumple, on="row_id", how="inner", validate="one_to_one")
    spatial["residual"] = spatial["predicted"] - spatial["canopy_rumple"]
    common_low, common_high = np.quantile(
        np.concatenate([spatial["canopy_rumple"], spatial["predicted"]]), [0.01, 0.99]
    )
    residual_limit = float(np.quantile(np.abs(spatial["residual"]), 0.99))

    map_specs = [
        ("canopy_rumple", "LiDAR observed rumple", "viridis", common_low, common_high),
        ("predicted", "TESSERA predicted rumple", "viridis", common_low, common_high),
        ("residual", "Prediction minus observation", "RdBu_r", -residual_limit, residual_limit),
    ]
    for offset, (column, title, cmap, lower, upper) in enumerate(map_specs):
        axis = fig.add_subplot(grid[1, offset])
        image = axis.scatter(
            spatial["bng_x"],
            spatial["bng_y"],
            c=spatial[column],
            cmap=cmap,
            vmin=lower,
            vmax=upper,
            s=8,
            marker="s",
            linewidths=0,
        )
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xlabel("Easting (km)")
        if offset == 0:
            axis.set_ylabel("Northing (km)")
        axis.xaxis.set_major_formatter(lambda value, _: f"{value / 1000:.1f}")
        axis.yaxis.set_major_formatter(lambda value, _: f"{value / 1000:.1f}")
        axis.xaxis.set_major_locator(MaxNLocator(4))
        axis.yaxis.set_major_locator(MaxNLocator(4))
        colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.025)
        colorbar.set_label("Rumple" if column != "residual" else "Rumple residual")
        panel(axis, offset + 3)

    fig.subplots_adjust(left=0.065, right=0.97, top=0.97, bottom=0.08)
    save(fig, "cairngorms_prediction_evidence")


def make_height_adjustment_plate() -> None:
    targets = pd.read_parquet(CAIRN_TARGETS).reset_index(drop=True)
    predictions = pd.read_parquet(CAIRN_PREDICTIONS)
    macro = pd.read_csv(CAIRN_MACRO)
    diagnostics = pd.read_csv(HEIGHT_DIAGNOSTICS).set_index("target")
    selected = predictions[
        (predictions["variant"] == "height_adjusted")
        & (predictions["version"] == "v2")
        & (predictions["model_family"] == "tessera")
        & predictions["target"].isin(KEY_TARGETS)
    ].copy()
    components = load_height_components(targets)

    fig = plt.figure(figsize=(11.3, 6.8))
    grid = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.30)
    mean_height = targets["canopy_mean_height_m"].to_numpy(float)
    p95_height = targets["canopy_p95_height_m"].to_numpy(float)

    for index, target in enumerate(KEY_TARGETS):
        axis = fig.add_subplot(grid[0, index])
        fitted = components[:, index]
        lower, upper = np.quantile(fitted, [0.01, 0.99])
        image = axis.hexbin(
            mean_height,
            p95_height,
            C=fitted,
            reduce_C_function=np.mean,
            gridsize=40,
            mincnt=1,
            cmap="viridis",
            vmin=lower,
            vmax=upper,
            linewidths=0,
        )
        diagnostic = diagnostics.loc[target]
        removed = 100.0 * max(
            0.0, float(diagnostic["variance_fraction_removed_by_observed_height"])
        )
        height_r2 = float(diagnostic["height_only_r2"])
        axis.set_title(TARGET_LABELS[target], loc="left", fontweight="bold")
        axis.set_xlabel("Mean canopy height (m)")
        axis.set_ylabel("P95 canopy height (m)")
        axis.text(
            0.04,
            0.95,
            f"Height-only $R^2$ = {height_r2:.2f}\nVariance removed = {removed:.0f}%",
            transform=axis.transAxes,
            va="top",
            ha="left",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2.2},
        )
        colorbar = fig.colorbar(image, ax=axis, fraction=0.048, pad=0.025)
        colorbar.set_label("Height-predicted component")
        panel(axis, index)

    for index, target in enumerate(KEY_TARGETS):
        axis = fig.add_subplot(grid[1, index])
        data = selected[selected["target"] == target]
        official = macro[
            (macro["variant"] == "height_adjusted")
            & (macro["version"] == "v2")
            & (macro["model_family"] == "tessera")
            & (macro["target"] == target)
        ].iloc[0]
        r2, rmse = float(official["r2"]), float(official["rmse"])
        image = axis.hexbin(
            data["observed"],
            data["predicted"],
            gridsize=43,
            bins="log",
            mincnt=1,
            cmap="magma",
            linewidths=0,
        )
        low = float(min(data["observed"].min(), data["predicted"].min()))
        high = float(max(data["observed"].max(), data["predicted"].max()))
        axis.plot([low, high], [low, high], linestyle="--", color="#2c6e9b", linewidth=1.0)
        axis.set_xlim(low, high)
        axis.set_ylim(low, high)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(f"Height-adjusted {TARGET_LABELS[target].lower()}", loc="left", fontweight="bold")
        axis.set_xlabel("Observed residual")
        axis.set_ylabel("Predicted residual")
        axis.text(
            0.04,
            0.95,
            f"$R^2$ = {r2:.2f}\nRMSE = {rmse:.3f}",
            transform=axis.transAxes,
            va="top",
            ha="left",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2.2},
        )
        panel(axis, index + 3)
        if index == 2:
            colorbar = fig.colorbar(image, ax=axis, fraction=0.048, pad=0.025)
            colorbar.set_label("Observations per bin (log scale)")

    fig.subplots_adjust(left=0.065, right=0.97, top=0.97, bottom=0.08)
    save(fig, "height_adjustment_evidence")


def local_relative_rmse() -> pd.DataFrame:
    metrics = pd.read_csv(DUTCH_METRICS)
    grouped = (
        metrics.groupby(["target_site", "target", "scenario"], as_index=False)["rmse"]
        .mean()
    )
    local = grouped[grouped["scenario"] == "local_cv"].rename(columns={"rmse": "local_rmse"})
    comparison = grouped[grouped["scenario"] != "local_cv"].merge(
        local[["target_site", "target", "local_rmse"]],
        on=["target_site", "target"],
        validate="many_to_one",
    )
    comparison["relative_rmse"] = comparison["rmse"] / comparison["local_rmse"]
    return comparison


def make_transferability_plate() -> None:
    rank = pd.read_csv(RANK_SUMMARY)
    curve = pd.read_csv(REFERENCE_CURVE)
    relative = local_relative_rmse()

    fig = plt.figure(figsize=(11.6, 7.8))
    grid = fig.add_gridspec(2, 2, width_ratios=(1.07, 1.0), hspace=0.34, wspace=0.24)

    # Crop the previously validated Dutch locator map to retain the mapped evidence and remove its long key.
    axis = fig.add_subplot(grid[0, 0])
    image = mpimg.imread(DUTCH_SITE_MAP)
    # Keep only the mapped Netherlands panel; rebuild its legend in this figure.
    axis.imshow(
        image[
            int(image.shape[0] * 0.075) : int(image.shape[0] * 0.985),
            int(image.shape[1] * 0.035) : int(image.shape[1] * 0.647),
        ]
    )
    axis.axis("off")
    axis.set_title("Twenty Dutch test forests", loc="left", fontweight="bold")
    axis.legend(
        handles=[Patch(facecolor=TEAL, label="Conifer-dominated"), Patch(facecolor=ORANGE, label="Broadleaf-dominated")],
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        ncol=2,
    )
    panel(axis, 0)

    axis = fig.add_subplot(grid[0, 1])
    scenarios = ["all_other", "same_group", "different_group"]
    scenario_labels = ["All other forests", "Same forest group", "Other forest group"]
    colors = [CHARCOAL, TEAL, ORANGE]
    rng = np.random.default_rng(20260901)
    for position, (scenario, label, color) in enumerate(zip(scenarios, scenario_labels, colors, strict=True)):
        values = relative.loc[relative["scenario"] == scenario, "relative_rmse"].to_numpy(float)
        values = values[np.isfinite(values)]
        violin = axis.violinplot(values, positions=[position], widths=0.72, showextrema=False)
        for body in violin["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor("none")
            body.set_alpha(0.32)
        q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75])
        axis.plot([position, position], [q25, q75], color=color, linewidth=5, solid_capstyle="butt")
        axis.scatter(position, median, color="white", edgecolor=color, linewidth=1.3, s=34, zorder=4)
        sample = rng.choice(values, size=min(70, len(values)), replace=False)
        jitter = rng.normal(position, 0.055, size=len(sample))
        axis.scatter(jitter, sample, color=color, s=7, alpha=0.25, linewidths=0)
    axis.axhline(1.0, linestyle="--", color="black", linewidth=0.9)
    axis.set_xticks(range(3), scenario_labels, rotation=18, ha="right")
    axis.set_ylabel("Transfer RMSE / local-model RMSE")
    axis.set_title("Transfer loss across target forests", loc="left", fontweight="bold")
    axis.set_ylim(bottom=0.75)
    axis.grid(axis="y", color=LIGHT_GREY, linewidth=0.7)
    panel(axis, 1)

    axis = fig.add_subplot(grid[1, 0])
    labels = [DUTCH_LABELS[target] for target in rank["target"]]
    positions = np.arange(len(rank))
    width = 0.24
    axis.bar(positions - width, rank["r2_median"], width=width, color=CHARCOAL, label="Exact values: $R^2$")
    axis.bar(positions, rank["unit_spearman_median"], width=width, color=TEAL, label="50 m ordering")
    axis.bar(positions + width, rank["block_spearman_median"], width=width, color=ORANGE, label="500 m ordering")
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(positions, labels, rotation=27, ha="right")
    axis.set_ylabel("Median held-out score across forests")
    axis.set_title(
        "Exact values versus relative spatial ordering",
        loc="left",
        fontweight="bold",
        pad=36,
    )
    axis.legend(
        frameon=False,
        ncol=3,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        borderaxespad=0,
        columnspacing=1.0,
        handlelength=1.7,
    )
    axis.grid(axis="y", color=LIGHT_GREY, linewidth=0.7)
    panel(axis, 2)

    axis = fig.add_subplot(grid[1, 1])
    matched = curve[curve["cohort"] == "matched_200_eligible_folds"].copy()
    if matched.empty:
        matched = curve[curve["cohort"] == "all_site_folds_to_100"].copy()
    methods = {
        "source_ridge": ("Transferred model", CHARCOAL, "-"),
        "source_offset": ("Transferred + local mean correction", TEAL, "-"),
        "target_only_ridge": ("Locally fitted model", ORANGE, "-"),
    }
    budgets = [0, 2, 5, 10, 25, 50, 100, 200]
    budget_positions = {budget: position for position, budget in enumerate(budgets)}
    for method, (label, color, linestyle) in methods.items():
        subset = matched[matched["method"] == method]
        summary = subset.groupby("budget", as_index=False)["r2_median"].median().sort_values("budget")
        if method == "source_ridge":
            zero = float(summary.iloc[0]["r2_median"])
            axis.axhline(zero, color=color, linewidth=1.8, label=label)
        else:
            x_positions = [budget_positions[int(value)] for value in summary["budget"]]
            axis.plot(
                x_positions,
                summary["r2_median"],
                marker="o",
                markersize=4,
                linewidth=1.8,
                color=color,
                linestyle=linestyle,
                label=label,
            )
    axis.axhline(0, color="black", linewidth=0.8, linestyle="--")
    axis.set_xlabel("LiDAR reference units from the target forest")
    axis.set_ylabel("Median $R^2$ across six outcomes")
    axis.set_title("How local reference data change absolute accuracy", loc="left", fontweight="bold")
    axis.set_xlim(-0.25, len(budgets) - 0.75)
    axis.set_xticks(range(len(budgets)), [str(value) for value in budgets])
    axis.legend(frameon=False, loc="lower right")
    axis.grid(axis="y", color=LIGHT_GREY, linewidth=0.7)
    panel(axis, 3)

    fig.subplots_adjust(left=0.07, right=0.98, top=0.97, bottom=0.085)
    save(fig, "dutch_transferability_story")


def main() -> None:
    style()
    make_cairngorms_prediction_plate()
    make_height_adjustment_plate()
    make_transferability_plate()
    print(f"Created evidence plates in {OUT}")


if __name__ == "__main__":
    main()
