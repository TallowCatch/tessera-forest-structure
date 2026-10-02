#!/usr/bin/env python3
"""Generate manuscript supplementary tables from frozen result CSV files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "paper"
    / "final_corrected_user"
    / "sections"
    / "supplementary_results_tables.tex"
)

TARGET_LABELS = {
    "canopy_p95_height_m": "P95 height",
    "canopy_surface_sd_m": "Height SD",
    "canopy_surface_cv": "Height CV",
    "canopy_surface_rcv": "Robust height CV",
    "canopy_rumple": "Rumple",
    "canopy_open_fraction": "Canopy openings",
    "canopy_height_kurtosis": "Height kurtosis",
    "ahn4_p95_height_m": "P95 height",
    "ahn4_height_sd_m": "Height SD",
    "ahn4_height_cv": "Height CV",
    "ahn4_entropy": "Return-height entropy",
    "ahn4_pulse_penetration": "Pulse penetration",
    "mean_height_m": "Mean height",
    "p95_height_m": "P95 height",
    "within_cell_height_sd_m": "Height SD",
    "height_cv": "Height CV",
}

TARGET_ORDER = {
    "canopy_p95_height_m": 0,
    "canopy_surface_sd_m": 1,
    "canopy_surface_cv": 2,
    "canopy_surface_rcv": 3,
    "canopy_rumple": 4,
    "canopy_open_fraction": 5,
    "canopy_height_kurtosis": 6,
    "ahn4_p95_height_m": 0,
    "ahn4_height_sd_m": 1,
    "ahn4_height_cv": 2,
    "ahn4_entropy": 3,
    "ahn4_pulse_penetration": 4,
    "mean_height_m": 0,
    "p95_height_m": 1,
    "within_cell_height_sd_m": 2,
    "height_cv": 3,
}


def f3(value: float) -> str:
    return f"{float(value):.3f}"


def f4(value: float) -> str:
    return f"{float(value):.4f}"


def variant_label(value: str) -> str:
    return "Original" if value == "raw" else "Height-adjusted"


def site_label(value: str) -> str:
    return "Cairngorms" if value == "cairngorms" else "Savelsbos"


def cairngorms_performance() -> str:
    data = pd.read_csv(
        ROOT / "outputs/tables/phase29_cairngorms_v2_surface_macro.csv"
    )
    data = data[(data["version"] == "v2") & (data["model_family"] == "tessera")]
    data = data.assign(
        variant_order=data["variant"].map({"raw": 0, "height_adjusted": 1}),
        target_order=data["target"].map(TARGET_ORDER),
    ).sort_values(["variant_order", "target_order"])
    rows = [
        f"{variant_label(row.variant)} & {TARGET_LABELS[row.target]} & "
        f"{f3(row.rmse)} & {f3(row.r2)} & {f3(row.block_spearman)} \\\\"
        for row in data.itertuples(index=False)
    ]
    return r"""
\subsection{Complete Cairngorms performance and uncertainty}

\begin{table}[H]
\centering
\small
\caption{Complete spatially held-out TESSERA v2 performance for the Cairngorms canopy-surface analysis. Values are equal-fold means. $r_s$ denotes block-level Spearman rank correlation.}
\label{tab:supp_cairngorms_complete}
\begin{tabular}{llrrr}
\toprule
Outcome form & Outcome & RMSE & $R^2$ & $r_s$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def comparison_table() -> str:
    comparisons = pd.read_csv(
        ROOT / "outputs/tables/final_manuscript_cairngorms_model_comparisons.csv"
    )
    comparisons = comparisons[
        (comparisons["candidate"] == "tessera")
        & (comparisons["reference"] == "conventional")
    ].assign(
        variant_order=lambda x: x["variant"].map({"raw": 0, "height_adjusted": 1}),
        target_order=lambda x: x["target"].map(TARGET_ORDER),
    ).sort_values(["variant_order", "target_order"])
    rows = [
        f"{variant_label(row.variant)} & {TARGET_LABELS[row.target]} & "
        f"{f4(row.delta_rmse)} & {f4(row.ci_low)} & {f4(row.ci_high)} \\\\"
        for row in comparisons.itertuples(index=False)
    ]
    versions = pd.read_csv(
        ROOT / "outputs/tables/phase29_cairngorms_v2_surface_v1_v2_comparison.csv"
    )
    versions = versions[versions["model_family"] == "tessera"].assign(
        variant_order=lambda x: x["variant"].map({"raw": 0, "height_adjusted": 1}),
        target_order=lambda x: x["target"].map(TARGET_ORDER),
    ).sort_values(["variant_order", "target_order"])
    version_rows = [
        f"{variant_label(row.variant)} & {TARGET_LABELS[row.target]} & "
        f"{f4(row.v2_minus_v1_rmse)} & {f4(row.ci_low)} & {f4(row.ci_high)} \\\\"
        for row in versions.itertuples(index=False)
    ]
    return r"""
\begin{table}[H]
\centering
\small
\caption{Paired Cairngorms comparison of TESSERA v2 with the conventional Sentinel and terrain model. Differences are TESSERA minus conventional RMSE, so negative values favour TESSERA. Intervals are 95\% percentiles from 2,000 paired spatial-block bootstrap replicates.}
\label{tab:supp_tessera_conventional_intervals}
\begin{tabular}{llrrr}
\toprule
Outcome form & Outcome & $\Delta$RMSE & 2.5\% & 97.5\% \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}

\begin{table}[H]
\centering
\small
\caption{Matched Cairngorms comparison of TESSERA v2 with v1.0. Differences are v2 minus v1.0 RMSE, so negative values favour v2. Intervals are 95\% percentiles from 2,000 paired spatial-block bootstrap replicates.}
\label{tab:supp_v1_v2_intervals}
\begin{tabular}{llrrr}
\toprule
Outcome form & Outcome & $\Delta$RMSE & 2.5\% & 97.5\% \\
\midrule
""" + "\n".join(version_rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def savelsbos_tables() -> str:
    data = pd.read_csv(ROOT / "outputs/tables/phase27_savelsbos_v2_metrics.csv")
    targets = [
        "ahn4_p95_height_m",
        "ahn4_height_sd_m",
        "ahn4_height_cv",
        "ahn4_entropy",
        "ahn4_pulse_penetration",
    ]
    models = ["conventional", "tessera_v2", "fused_v2"]
    data = data[
        (data["scope"] == "pooled")
        & data["target"].isin(targets)
        & data["model"].isin(models)
    ]
    data = data.assign(
        variant_order=data["target_variant"].map({"raw": 0, "height_adjusted": 1}),
        target_order=data["target"].map(TARGET_ORDER),
    )
    rows: list[str] = []
    for (variant, target), local in data.groupby(["target_variant", "target"]):
        by_model = local.set_index("model")
        values: list[str] = []
        for model in models:
            row = by_model.loc[model]
            values.extend([f3(row.rmse), f3(row.r2), f3(row.block_spearman)])
        rows.append(
            f"{variant_label(variant)} & {TARGET_LABELS[target]} & "
            + " & ".join(values)
            + " \\\\"
        )
    rows.sort(
        key=lambda line: (
            0 if line.startswith("Original") else 1,
            next(
                rank
                for target, rank in TARGET_ORDER.items()
                if TARGET_LABELS[target] in line
            ),
        )
    )

    context = pd.read_csv(
        ROOT / "outputs/tables/phase32_scotland_netherlands_transfer_metrics.csv"
    )
    context = context[
        (context["training_site"] == "savelsbos")
        & (context["evaluation_site"] == "savelsbos")
        & (context["model"] == "local_cv")
    ].assign(target_order=lambda x: x["target"].map(TARGET_ORDER))
    context_rows: list[str] = []
    for target, local in context.groupby("target"):
        by_panel = local.set_index("feature_panel")
        mean = by_panel.loc["mean"]
        patch = by_panel.loc["context"]
        context_rows.append(
            f"{TARGET_LABELS[target]} & {f3(mean.rmse)} & {f3(mean.r2)} & {f3(mean.block_spearman)} & "
            f"{f3(patch.rmse)} & {f3(patch.r2)} & {f3(patch.block_spearman)} \\\\"
        )
    context_rows.sort(
        key=lambda line: next(
            rank
            for target, rank in TARGET_ORDER.items()
            if TARGET_LABELS[target] in line
        )
    )
    return r"""
\subsection{Complete Savelsbos model and context sensitivity results}

\begin{table}[H]
\centering
\scriptsize
\caption{Complete local Savelsbos results for conventional predictors, TESSERA v2 and their combination. RMSE, $R^2$ and within-block Spearman $r_s$ were calculated from the same 589 held-out observations.}
\label{tab:supp_savelsbos_models}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llrrrrrrrrr}
\toprule
& & \multicolumn{3}{c}{Conventional} & \multicolumn{3}{c}{TESSERA v2} & \multicolumn{3}{c}{Combined} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}
Outcome form & Outcome & RMSE & $R^2$ & $r_s$ & RMSE & $R^2$ & $r_s$ & RMSE & $R^2$ & $r_s$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}%
}
\end{table}

\begin{table}[H]
\centering
\small
\caption{Savelsbos representation sensitivity for the four outcomes shared between landscapes. The 128-value representation contains the mean of each TESSERA channel; the 512-value representation contains channel-wise means, standard deviations and two spatial gradients.}
\label{tab:supp_savelsbos_context}
\begin{tabular}{lrrrrrr}
\toprule
& \multicolumn{3}{c}{128-value mean} & \multicolumn{3}{c}{512-value context} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Outcome & RMSE & $R^2$ & $r_s$ & RMSE & $R^2$ & $r_s$ \\
\midrule
""" + "\n".join(context_rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def transfer_table() -> str:
    metrics = pd.read_csv(
        ROOT / "outputs/tables/phase32_scotland_netherlands_transfer_metrics.csv"
    )
    bootstrap = pd.read_csv(
        ROOT / "outputs/tables/phase32_scotland_netherlands_transfer_bootstrap.csv"
    )
    transfer = metrics[
        (metrics["feature_panel"] == "context") & (metrics["model"] == "transfer")
    ]
    local = metrics[
        (metrics["feature_panel"] == "context") & (metrics["model"] == "local_cv")
    ][["evaluation_site", "target", "rmse"]].rename(columns={"rmse": "local_rmse"})
    boot = bootstrap[bootstrap["feature_panel"] == "context"]
    data = transfer.merge(local, on=["evaluation_site", "target"]).merge(
        boot,
        on=["evaluation_site", "target"],
    )
    data = data.assign(
        direction=data["training_site"].map(site_label)
        + " $\\rightarrow$ "
        + data["evaluation_site"].map(site_label),
        direction_order=data["evaluation_site"].map({"savelsbos": 0, "cairngorms": 1}),
        target_order=data["target"].map(TARGET_ORDER),
    ).sort_values(["direction_order", "target_order"])
    rows = [
        f"{row.direction} & {TARGET_LABELS[row.target]} & {f3(row.local_rmse)} & "
        f"{f3(row.rmse)} & {f3(row.r2)} & {f3(row.block_spearman)} & "
        f"{f3(row.delta_rmse_transfer_minus_local)} & {f3(row.ci_low)} & {f3(row.ci_high)} \\\\"
        for row in data.itertuples(index=False)
    ]
    return r"""
\subsection{Complete direct-transfer and joint-training results}

\begin{table}[H]
\centering
\scriptsize
\caption{Direct cross-landscape transfer for the four approximately harmonised outcomes. $\Delta$RMSE is transfer minus locally trained RMSE; positive values indicate higher error under transfer. Intervals are 95\% percentiles from paired spatial-block bootstrap replicates.}
\label{tab:supp_direct_transfer}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llrrrrrrr}
\toprule
Direction & Outcome & Local RMSE & Transfer RMSE & Transfer $R^2$ & Transfer $r_s$ & $\Delta$RMSE & 2.5\% & 97.5\% \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}%
}
\end{table}
"""


def joint_table() -> str:
    phase32 = pd.read_csv(
        ROOT / "outputs/tables/phase32_scotland_netherlands_transfer_metrics.csv"
    )
    ridge_local = phase32[
        (phase32["feature_panel"] == "context") & (phase32["model"] == "local_cv")
    ][["evaluation_site", "target", "rmse", "r2", "block_spearman"]].rename(
        columns={
            "rmse": "local_rmse",
            "r2": "local_r2",
            "block_spearman": "local_rs",
        }
    )
    ridge_pooled = pd.read_csv(
        ROOT / "outputs/tables/phase34_scotland_netherlands_pooled_metrics.csv"
    ).rename(
        columns={
            "rmse": "pooled_rmse",
            "r2": "pooled_r2",
            "block_spearman": "pooled_rs",
        }
    )
    ridge_boot = pd.read_csv(
        ROOT / "outputs/tables/phase34_scotland_netherlands_pooled_bootstrap.csv"
    )
    ridge = ridge_local.merge(
        ridge_pooled[
            ["evaluation_site", "target", "pooled_rmse", "pooled_r2", "pooled_rs"]
        ],
        on=["evaluation_site", "target"],
    ).merge(ridge_boot, on=["evaluation_site", "target"])
    ridge["model_family"] = "Ridge"

    mlp_metrics = pd.read_csv(
        ROOT / "outputs/tables/phase35_scotland_netherlands_pooled_mlp_metrics.csv"
    )
    mlp_local = mlp_metrics[mlp_metrics["training_scope"].str.startswith("local_")][
        ["evaluation_site", "target", "rmse", "r2", "block_spearman"]
    ].rename(
        columns={
            "rmse": "local_rmse",
            "r2": "local_r2",
            "block_spearman": "local_rs",
        }
    )
    mlp_pooled = mlp_metrics[mlp_metrics["training_scope"] == "pooled_balanced"][[
        "evaluation_site", "target", "rmse", "r2", "block_spearman"
    ]].rename(
        columns={
            "rmse": "pooled_rmse",
            "r2": "pooled_r2",
            "block_spearman": "pooled_rs",
        }
    )
    mlp_boot = pd.read_csv(
        ROOT / "outputs/tables/phase35_scotland_netherlands_pooled_mlp_bootstrap.csv"
    )
    mlp = mlp_local.merge(mlp_pooled, on=["evaluation_site", "target"]).merge(
        mlp_boot,
        on=["evaluation_site", "target"],
    )
    mlp["model_family"] = "MLP"

    data = pd.concat([ridge, mlp], ignore_index=True)
    data["percent_change"] = 100 * (
        data["pooled_rmse"] / data["local_rmse"] - 1
    )
    data = data.assign(
        model_order=data["model_family"].map({"Ridge": 0, "MLP": 1}),
        site_order=data["evaluation_site"].map({"cairngorms": 0, "savelsbos": 1}),
        target_order=data["target"].map(TARGET_ORDER),
    ).sort_values(["model_order", "site_order", "target_order"])
    rows = [
        f"{row.model_family} & {site_label(row.evaluation_site)} & {TARGET_LABELS[row.target]} & "
        f"{f3(row.local_rmse)} & {f3(row.pooled_rmse)} & {row.percent_change:+.1f} & "
        f"{f3(row.local_r2)} & {f3(row.pooled_r2)} & {f3(row.local_rs)} & "
        f"{f3(row.pooled_rs)} & [{f3(row.ci_low)}, {f3(row.ci_high)}] \\\\"
        for row in data.itertuples(index=False)
    ]
    return r"""
\begin{table}[H]
\centering
\scriptsize
\caption{Post hoc joint-training sensitivity analysis. Local and pooled models used the same held-out areas in each landscape. $\Delta$\% is the percentage change in RMSE from local to pooled training; positive values indicate higher pooled-model error. The final column gives the paired 95\% interval for pooled minus local RMSE.}
\label{tab:supp_joint_training}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lllrrrrrrrr}
\toprule
Model & Landscape & Outcome & Local RMSE & Pooled RMSE & $\Delta$\% & Local $R^2$ & Pooled $R^2$ & Local $r_s$ & Pooled $r_s$ & RMSE interval \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}%
}
\end{table}
"""


def main() -> None:
    text = "\n".join(
        [
            cairngorms_performance(),
            comparison_table(),
            savelsbos_tables(),
            transfer_table(),
            joint_table(),
        ]
    )
    OUTPUT.write_text(text.strip() + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
