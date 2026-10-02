#!/usr/bin/env python3
"""Build final manuscript result tables and figures from frozen predictions."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
FIGURE_DIR = ROOT / "paper" / "final_manuscript" / "figures"

RAW_PREDICTIONS = (
    ROOT / "data" / "processed" / "phase22_cairngorms_spatial_unet_predictions.parquet"
)
ADJUSTED_PREDICTIONS = (
    ROOT / "data" / "processed" / "phase23_cairngorms_height_adjusted_predictions.parquet"
)
V2_PREDICTIONS = (
    ROOT / "data" / "processed" / "phase29_cairngorms_v2_surface_predictions.parquet"
)
HEIGHT_PREDICTIONS = (
    ROOT / "data" / "processed" / "phase31_cairngorms_v2_height_strict_predictions.parquet"
)
HEIGHT_MACRO = TABLE_DIR / "phase31_cairngorms_v2_height_strict_macro.csv"
SURFACE_MACRO = TABLE_DIR / "phase29_cairngorms_v2_surface_macro.csv"
SAVELSBOS_METRICS = TABLE_DIR / "phase27_savelsbos_v2_metrics.csv"
TRANSFER_METRICS = TABLE_DIR / "phase32_scotland_netherlands_transfer_metrics.csv"

MODEL_COMPARISONS = TABLE_DIR / "final_manuscript_cairngorms_model_comparisons.csv"
HETEROGENEITY_FIGURE = FIGURE_DIR / "cairngorms_v2_heterogeneity_density.pdf"
HEIGHT_FIGURE = FIGURE_DIR / "cairngorms_height_architecture.pdf"
REPLICATION_FIGURE = FIGURE_DIR / "savelsbos_replication_and_transfer.pdf"

COLORS = {
    "conventional": "#6C757D",
    "tessera": "#148F88",
    "fused": "#E8872D",
    "mlp": "#3D7F91",
    "unet": "#C45133",
    "raw": "#148F88",
    "adjusted": "#8A5CC7",
    "transfer": "#D95F02",
}


def load_cairngorms_predictions() -> pd.DataFrame:
    raw = pd.read_parquet(RAW_PREDICTIONS)
    raw = raw[
        (raw["scheme"] == "balanced")
        & (raw["model"] == "conventional_mlp")
    ].copy()
    raw["variant"] = "raw"
    raw["model_family"] = "conventional"

    adjusted = pd.read_parquet(ADJUSTED_PREDICTIONS)
    adjusted = adjusted[adjusted["model"] == "conventional"].copy()
    adjusted["variant"] = "height_adjusted"
    adjusted["model_family"] = "conventional"

    v2 = pd.read_parquet(V2_PREDICTIONS)
    v2 = v2[v2["version"] == "v2"].copy()

    columns = [
        "row_id",
        "variant",
        "model_family",
        "fold",
        "target",
        "observed",
        "predicted",
        "spatial_block",
    ]
    predictions = pd.concat(
        [raw[columns], adjusted[columns], v2[columns]], ignore_index=True
    )
    return predictions


def paired_block_bootstrap(
    local: pd.DataFrame,
    candidate: str,
    reference: str,
    replicates: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    subset = local[local["model_family"].isin([candidate, reference])].copy()
    squared = subset.assign(
        squared_error=np.square(subset["predicted"] - subset["observed"])
    )
    grouped = squared.groupby(
        ["fold", "spatial_block", "model_family"], as_index=False
    )["squared_error"].mean()
    wide = grouped.pivot(
        index=["fold", "spatial_block"],
        columns="model_family",
        values="squared_error",
    ).dropna(subset=[candidate, reference])

    point_by_fold = wide.groupby(level="fold")[[candidate, reference]].mean().pow(0.5)
    point = float((point_by_fold[candidate] - point_by_fold[reference]).mean())

    samples = np.empty(replicates, dtype=np.float64)
    folds = sorted(wide.index.get_level_values("fold").unique())
    for replicate in range(replicates):
        differences: list[float] = []
        for fold in folds:
            fold_values = wide.xs(fold, level="fold")
            selected = rng.integers(0, len(fold_values), len(fold_values))
            rmses = fold_values.iloc[selected][[candidate, reference]].mean().pow(0.5)
            differences.append(float(rmses[candidate] - rmses[reference]))
        samples[replicate] = float(np.mean(differences))

    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "delta_rmse": point,
        "ci_low": float(low),
        "ci_high": float(high),
        "probability_candidate_better": float(np.mean(samples < 0)),
        "paired_blocks": int(len(wide)),
    }


def make_model_comparison_table() -> None:
    predictions = load_cairngorms_predictions()
    rng = np.random.default_rng(20260819)
    rows: list[dict[str, object]] = []
    comparisons = [
        ("tessera", "conventional"),
        ("fused", "conventional"),
        ("fused", "tessera"),
    ]
    for (variant, target), local in predictions.groupby(["variant", "target"]):
        available = set(local["model_family"])
        for candidate, reference in comparisons:
            if not {candidate, reference}.issubset(available):
                continue
            result = paired_block_bootstrap(
                local,
                candidate,
                reference,
                replicates=2_000,
                rng=rng,
            )
            rows.append(
                {
                    "variant": variant,
                    "target": target,
                    "candidate": candidate,
                    "reference": reference,
                    **result,
                }
            )
    MODEL_COMPARISONS.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(MODEL_COMPARISONS, index=False)


def weighted_rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(predicted - observed))))


def make_heterogeneity_figure() -> None:
    predictions = pd.read_parquet(V2_PREDICTIONS)
    predictions = predictions[
        (predictions["version"] == "v2")
        & (predictions["model_family"] == "tessera")
        & (predictions["variant"] == "raw")
    ].copy()
    macro = pd.read_csv(SURFACE_MACRO)
    macro = macro[
        (macro["version"] == "v2")
        & (macro["model_family"] == "tessera")
        & (macro["variant"] == "raw")
    ].set_index("target")

    targets = [
        ("canopy_surface_sd_m", "a  Height SD", "Height SD (m)"),
        ("canopy_surface_cv", "b  Height CV", "Height CV"),
        ("canopy_surface_rcv", "c  Robust height CV", "Robust height CV"),
        ("canopy_rumple", "d  Rumple", "Rumple"),
        ("canopy_open_fraction", "e  Canopy openings", "Canopy opening fraction"),
        ("canopy_height_kurtosis", "f  Height kurtosis", "Height kurtosis"),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(12.8, 8.0), constrained_layout=True)
    last_image = None
    for axis, (target, title, label) in zip(axes.flat, targets, strict=True):
        local = predictions[predictions["target"] == target]
        last_image = axis.hexbin(
            local["observed"],
            local["predicted"],
            gridsize=47,
            mincnt=1,
            bins="log",
            cmap="viridis",
            linewidths=0.08,
        )
        lower = float(min(local["observed"].min(), local["predicted"].min()))
        upper = float(max(local["observed"].max(), local["predicted"].max()))
        padding = 0.02 * (upper - lower)
        lower -= padding
        upper += padding
        axis.plot([lower, upper], [lower, upper], "--", color="#C92A2A", linewidth=1.1)
        axis.set_xlim(lower, upper)
        axis.set_ylim(lower, upper)
        axis.set_xlabel(f"Observed {label}")
        axis.set_ylabel(f"Predicted {label}")
        axis.set_title(title, loc="left", fontweight="bold")
        row = macro.loc[target]
        unit = " m" if target == "canopy_surface_sd_m" else ""
        axis.text(
            0.04,
            0.95,
            f"Mean $R^2$ = {row['r2']:.3f}\nRMSE = {row['rmse']:.3f}{unit}",
            transform=axis.transAxes,
            va="top",
            fontsize=8.7,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
        )
        axis.tick_params(labelsize=8.5)

    if last_image is not None:
        colorbar = figure.colorbar(last_image, ax=axes, fraction=0.022, pad=0.018)
        colorbar.set_label("Observations per hexagon (log scale)")

    HETEROGENEITY_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(HETEROGENEITY_FIGURE, bbox_inches="tight")
    figure.savefig(HETEROGENEITY_FIGURE.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)


def make_height_figure() -> None:
    predictions = pd.read_parquet(HEIGHT_PREDICTIONS)
    macro = pd.read_csv(HEIGHT_MACRO)
    mlp_name = "tessera_v2_mlp_5x5"
    unet_name = "tessera_v2_unet_strict"
    p95 = predictions[
        (predictions["model"] == unet_name)
        & (predictions["target"] == "canopy_p95_height_m")
    ]

    figure, axes = plt.subplots(1, 3, figsize=(12.5, 4.05), constrained_layout=True)

    axis = axes[0]
    image = axis.hexbin(
        p95["observed"],
        p95["predicted"],
        gridsize=42,
        mincnt=1,
        bins="log",
        cmap="viridis",
        linewidths=0.1,
    )
    lower = float(min(p95["observed"].min(), p95["predicted"].min()))
    upper = float(max(p95["observed"].max(), p95["predicted"].max()))
    axis.plot([lower, upper], [lower, upper], "--", color="#C92A2A", linewidth=1.2)
    axis.set_xlabel("Observed p95 height (m)")
    axis.set_ylabel("Predicted p95 height (m)")
    axis.set_title("a  Strict U-Net predictions", loc="left", fontweight="bold")
    axis.text(
        0.04,
        0.95,
        "$R^2=0.378$\nRMSE = 2.220 m",
        transform=axis.transAxes,
        va="top",
        fontsize=9,
    )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.045, pad=0.02)
    colorbar.set_label("Observations per hexagon")

    axis = axes[1]
    targets = ["canopy_mean_height_m", "canopy_p95_height_m"]
    labels = ["Mean height", "P95 height"]
    x = np.arange(len(targets))
    width = 0.34
    for offset, model, label, color in [
        (-width / 2, mlp_name, "5 $\\times$ 5 MLP", COLORS["mlp"]),
        (width / 2, unet_name, "Strict U-Net", COLORS["unet"]),
    ]:
        values = []
        for target in targets:
            row = macro[(macro["model"] == model) & (macro["target"] == target)]
            values.append(float(row["rmse"].iloc[0]))
        axis.bar(x + offset, values, width, label=label, color=color)
    axis.set_xticks(x, labels)
    axis.set_ylabel("RMSE (m)")
    axis.set_title("b  Architecture comparison", loc="left", fontweight="bold")
    axis.legend(frameon=False, loc="upper left")
    axis.grid(axis="y", alpha=0.25)

    axis = axes[2]
    reference = predictions[
        (predictions["model"] == mlp_name)
        & (predictions["target"] == "canopy_p95_height_m")
    ][["row_id", "observed"]].drop_duplicates("row_id")
    reference["height_group"] = pd.qcut(
        reference["observed"],
        q=5,
        labels=["Lowest", "20-40%", "40-60%", "60-80%", "Highest"],
    )
    for model, label, color in [
        (mlp_name, "5 $\\times$ 5 MLP", COLORS["mlp"]),
        (unet_name, "Strict U-Net", COLORS["unet"]),
    ]:
        local = predictions[
            (predictions["model"] == model)
            & (predictions["target"] == "canopy_p95_height_m")
        ].merge(reference[["row_id", "height_group"]], on="row_id", how="left")
        values = []
        for group in reference["height_group"].cat.categories:
            grouped = local[local["height_group"] == group]
            values.append(
                weighted_rmse(grouped["observed"].to_numpy(), grouped["predicted"].to_numpy())
            )
        axis.plot(
            np.arange(5),
            values,
            marker="o",
            linewidth=1.7,
            label=label,
            color=color,
        )
    axis.set_xticks(np.arange(5), reference["height_group"].cat.categories, rotation=25, ha="right")
    axis.set_ylabel("P95-height RMSE (m)")
    axis.set_title("c  Error across observed height", loc="left", fontweight="bold")
    axis.legend(frameon=False, loc="upper left")
    axis.grid(axis="y", alpha=0.25)

    HEIGHT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(HEIGHT_FIGURE, bbox_inches="tight")
    figure.savefig(HEIGHT_FIGURE.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)


def make_replication_figure() -> None:
    savelsbos = pd.read_csv(SAVELSBOS_METRICS)
    savelsbos = savelsbos[
        (savelsbos["scope"] == "pooled") & (savelsbos["model"] == "tessera_v2")
    ].copy()
    transfer = pd.read_csv(TRANSFER_METRICS)
    transfer = transfer[transfer["feature_panel"] == "context"].copy()

    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.25))
    figure.subplots_adjust(left=0.07, right=0.99, bottom=0.22, top=0.78, wspace=0.34)

    target_order = [
        "ahn4_p95_height_m",
        "ahn4_height_sd_m",
        "ahn4_height_cv",
        "ahn4_entropy",
        "ahn4_pulse_penetration",
    ]
    target_labels = ["P95 height", "Height SD", "Height CV", "Entropy", "Penetration"]
    axis = axes[0]
    x = np.arange(len(target_order))
    width = 0.34
    for offset, variant, label, color in [
        (-width / 2, "raw", "Original", COLORS["raw"]),
        (width / 2, "height_adjusted", "Height-adjusted", COLORS["adjusted"]),
    ]:
        values = []
        for target in target_order:
            row = savelsbos[
                (savelsbos["target"] == target) & (savelsbos["target_variant"] == variant)
            ]
            values.append(float(row["r2"].iloc[0]) if len(row) else np.nan)
        axis.bar(x + offset, values, width, label=label, color=color)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(x, target_labels, rotation=25, ha="right")
    axis.set_ylabel("Spatially held-out $R^2$")
    axis.set_title("a  Savelsbos local replication", loc="left", fontweight="bold")
    axis.grid(axis="y", alpha=0.25)

    handles, labels = axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.065, 0.965),
        ncol=2,
        columnspacing=1.1,
        handlelength=1.4,
    )

    transfer_targets = [
        "mean_height_m",
        "p95_height_m",
        "within_cell_height_sd_m",
        "height_cv",
    ]
    transfer_labels = ["Mean height", "P95 height", "Height SD", "Height CV"]
    for axis, source, destination, title in [
        (axes[1], "cairngorms", "savelsbos", "b  Scotland to Savelsbos"),
        (axes[2], "savelsbos", "cairngorms", "c  Savelsbos to Scotland"),
    ]:
        ratios = []
        for target in transfer_targets:
            local = transfer[
                (transfer["training_site"] == destination)
                & (transfer["evaluation_site"] == destination)
                & (transfer["target"] == target)
                & (transfer["model"] == "local_cv")
            ]
            moved = transfer[
                (transfer["training_site"] == source)
                & (transfer["evaluation_site"] == destination)
                & (transfer["target"] == target)
                & (transfer["model"] == "transfer")
            ]
            ratios.append(float(moved["rmse"].iloc[0] / local["rmse"].iloc[0]))
        axis.bar(np.arange(4), ratios, color=COLORS["transfer"], width=0.62)
        axis.axhline(1, color="black", linestyle="--", linewidth=1.0)
        axis.set_xticks(np.arange(4), transfer_labels, rotation=25, ha="right")
        axis.set_ylabel("Transfer RMSE / local RMSE")
        axis.set_title(title, loc="left", fontweight="bold")
        axis.grid(axis="y", alpha=0.25)

    REPLICATION_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(REPLICATION_FIGURE, bbox_inches="tight")
    figure.savefig(REPLICATION_FIGURE.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)


def run() -> None:
    make_model_comparison_table()
    make_heterogeneity_figure()
    make_height_figure()
    make_replication_figure()


if __name__ == "__main__":
    run()
