#!/usr/bin/env python3
"""Run and aggregate the expanded Dutch local and transfer evaluation."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cartopy.io.shapereader as shapereader
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

from evaluate_dutch_multisite_dutch_transfer import (
    add_predictions,
    atomic_csv,
    atomic_parquet,
    fit_predict,
    metrics_from_predictions,
    scenario_bootstrap,
    sha256,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/dutch_expanded_transfer.yaml"
TARGET_LABELS = {
    "ahn4_p95_height_m": "P95 height",
    "ahn4_height_sd_m": "Height SD",
    "ahn4_height_cv": "Height CV",
    "ahn4_entropy": "Return entropy",
    "ahn4_pulse_penetration": "Pulse penetration",
    "ahn4_sigma_z": "Sigma-z",
}
GROUP_COLORS = {"conifer": "#247A52", "broadleaf": "#B05C35"}


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "phase37_dutch_expanded_transfer"
    ]


def load_frame(config: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    cohort = pd.read_parquet(ROOT / str(config["outputs"]["cohort"]))
    features = pd.read_parquet(ROOT / str(config["outputs"]["features"]))
    frame = cohort.merge(features, on="row_id", validate="one_to_one")
    columns = sorted(column for column in frame if column.startswith("tessera_mean_"))
    if len(columns) != 128:
        raise RuntimeError(f"Expected 128 TESSERA mean features, found {len(columns)}")
    frame = frame[
        frame["tessera_context_valid"].astype(bool)
        & np.isfinite(frame[columns]).all(axis=1)
    ].reset_index(drop=True)
    return frame, columns


def selected_sites(config: dict[str, Any]) -> pd.DataFrame:
    sites = pd.read_csv(ROOT / str(config["outputs"]["site_summary"]))
    return sites.sort_values("selection_order").reset_index(drop=True)


def worker(index: int) -> None:
    config = load_config()
    sites = selected_sites(config)
    if index >= len(sites):
        print(f"phase37 worker {index}: no selected site at this index", flush=True)
        return
    frame, columns = load_frame(config)
    target_record = sites.iloc[index]
    target_site = str(target_record["site_code"])
    target_rows = frame[frame["site_code"] == target_site]
    if target_rows.empty:
        raise RuntimeError(f"No valid TESSERA rows for {target_site}")
    site_codes = sites["site_code"].astype(str).tolist()
    groups = dict(zip(sites["site_code"].astype(str), sites["forest_group"].astype(str), strict=True))
    targets = [str(value) for value in config["evaluation"]["targets"]]
    alpha = float(config["evaluation"]["ridge_alpha"])
    fold_root = ROOT / str(config["outputs"]["folds"])
    records: list[pd.DataFrame] = []
    local_parts: list[pd.DataFrame] = []
    for fold in range(int(config["evaluation"]["folds"])):
        with np.load(fold_root / target_site / f"fold_{fold}.npz") as definition:
            train_ids = definition["train_row_ids"].astype(np.int64)
            test_ids = definition["test_row_ids"].astype(np.int64)
        train = frame[frame["row_id"].isin(train_ids)]
        test = frame[frame["row_id"].isin(test_ids)]
        for target in targets:
            ridge, mean = fit_predict(train, test, columns, target, alpha)
            add_predictions(local_parts, test, target, "local_cv", target_site, ridge, mean)
    local = pd.concat(local_parts, ignore_index=True)
    expected = set(target_rows["row_id"])
    for target in targets:
        seen = local[(local["target"] == target) & (local["model"] == "ridge")]["row_id"]
        if len(seen) != len(expected) or set(seen) != expected:
            raise RuntimeError(f"Local CV did not cover {target_site}/{target} exactly once")
    records.append(local)
    for source_site in site_codes:
        if source_site == target_site:
            continue
        train = frame[frame["site_code"] == source_site]
        for target in targets:
            ridge, mean = fit_predict(train, target_rows, columns, target, alpha)
            add_predictions(records, target_rows, target, "single_source", source_site, ridge, mean)
    scenarios = {
        "same_group": [
            site for site in site_codes
            if site != target_site and groups[site] == groups[target_site]
        ],
        "different_group": [
            site for site in site_codes if groups[site] != groups[target_site]
        ],
        "all_other": [site for site in site_codes if site != target_site],
    }
    for scenario, sources in scenarios.items():
        if not sources:
            raise RuntimeError(f"No source sites for {target_site}/{scenario}")
        train = frame[frame["site_code"].isin(sources)]
        for target in targets:
            ridge, mean = fit_predict(train, target_rows, columns, target, alpha)
            add_predictions(records, target_rows, target, scenario, "+".join(sources), ridge, mean)
    output = pd.concat(records, ignore_index=True)
    worker_root = ROOT / str(config["outputs"]["worker_directory"])
    worker_root.mkdir(parents=True, exist_ok=True)
    output_path = worker_root / f"worker_{index:02d}_{target_site}.parquet"
    atomic_parquet(output_path, output)
    print(
        f"phase37 worker {index + 1}/{len(sites)} complete: {target_site}; "
        f"{len(output):,} prediction rows",
        flush=True,
    )


def plot_codes(sites: pd.DataFrame) -> dict[str, str]:
    counters = {"conifer": 0, "broadleaf": 0}
    output: dict[str, str] = {}
    for row in sites.itertuples(index=False):
        counters[row.forest_group] += 1
        prefix = "C" if row.forest_group == "conifer" else "B"
        output[row.site_code] = f"{prefix}{counters[row.forest_group]}"
    return output


def netherlands_outline() -> gpd.GeoDataFrame:
    path = shapereader.natural_earth(
        resolution="10m", category="cultural", name="admin_0_countries"
    )
    countries = gpd.read_file(path)
    key = "ADM0_A3" if "ADM0_A3" in countries else "ISO_A3"
    return countries[countries[key] == "NLD"].to_crs("EPSG:28992")


def locator_map(
    sites: pd.DataFrame,
    boundaries: gpd.GeoDataFrame,
    centres: pd.DataFrame,
    codes: dict[str, str],
    output: Path,
) -> None:
    country = netherlands_outline()
    fig = plt.figure(figsize=(10.8, 7.8))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.6, 1.0], wspace=0.04)
    ax = fig.add_subplot(grid[0, 0])
    country.plot(ax=ax, color="#F2F2EF", edgecolor="#7A7A74", linewidth=0.8)
    for group in ["conifer", "broadleaf"]:
        scoped = boundaries[boundaries["forest_group"] == group]
        scoped.plot(
            ax=ax,
            facecolor=GROUP_COLORS[group],
            edgecolor="white",
            linewidth=0.5,
            alpha=0.65,
        )
    for row in sites.itertuples(index=False):
        centre = centres.loc[row.site_code]
        ax.scatter(
            centre.rd_x,
            centre.rd_y,
            s=42,
            color=GROUP_COLORS[row.forest_group],
            edgecolor="white",
            linewidth=0.7,
            zorder=5,
        )
        ax.annotate(
            codes[row.site_code],
            (centre.rd_x, centre.rd_y),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=7.5,
            fontweight="bold",
            zorder=6,
        )
    ax.set_xlim(0, 290000)
    ax.set_ylim(295000, 620000)
    ax.set_aspect("equal")
    ax.set_xlabel("Dutch RD easting (m)")
    ax.set_ylabel("Dutch RD northing (m)")
    ax.set_title("a  Expanded Dutch forest-site network", loc="left", fontweight="bold")
    ax.legend(
        handles=[
            Patch(facecolor=GROUP_COLORS["conifer"], label="Conifer-dominated"),
            Patch(facecolor=GROUP_COLORS["broadleaf"], label="Broadleaf-dominated"),
        ],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=2,
    )
    key_ax = fig.add_subplot(grid[0, 1])
    key_ax.axis("off")
    key_ax.set_title("Site key", loc="left", fontweight="bold", fontsize=10)
    lines = []
    for group in ["conifer", "broadleaf"]:
        title = "Conifer-dominated" if group == "conifer" else "Broadleaf-dominated"
        lines.append((title, True, group))
        for row in sites[sites["forest_group"] == group].itertuples(index=False):
            lines.append((f"{codes[row.site_code]}   {row.site_name}", False, group))
    y = 0.96
    step = 0.92 / max(len(lines), 1)
    for label, heading, group in lines:
        key_ax.text(
            0.02,
            y,
            label,
            transform=key_ax.transAxes,
            fontsize=8.8 if heading else 7.7,
            fontweight="bold" if heading else "normal",
            color=GROUP_COLORS[group] if heading else "#222222",
            va="top",
        )
        y -= step
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def transfer_map(
    sites: pd.DataFrame,
    centres: pd.DataFrame,
    pairwise: pd.DataFrame,
    codes: dict[str, str],
    output: Path,
) -> None:
    country = netherlands_outline()
    pair = pairwise.groupby(
        ["source_site", "target_site", "same_group"], as_index=False
    )["relative_rmse"].mean()
    pair["unordered"] = pair.apply(
        lambda row: "|".join(sorted([row["source_site"], row["target_site"]])), axis=1
    )
    pair = pair.groupby(["unordered", "same_group"], as_index=False)["relative_rmse"].mean()
    pair[["site_a", "site_b"]] = pair["unordered"].str.split("|", expand=True)
    vmax = max(2.0, float(np.nanquantile(pair["relative_rmse"], 0.95)))
    norm = Normalize(vmin=1.0, vmax=vmax, clip=True)
    cmap = plt.get_cmap("magma")
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 6.2), sharex=True, sharey=True)
    for ax, same, title in zip(
        axes,
        [True, False],
        ["b  Same-group transfers", "c  Different-group transfers"],
        strict=True,
    ):
        country.plot(ax=ax, color="#F4F4F1", edgecolor="#888882", linewidth=0.6)
        scoped = pair[pair["same_group"] == same]
        for row in scoped.itertuples(index=False):
            a = centres.loc[row.site_a]
            b = centres.loc[row.site_b]
            ax.plot(
                [a.rd_x, b.rd_x],
                [a.rd_y, b.rd_y],
                color=cmap(norm(row.relative_rmse)),
                linewidth=0.7,
                alpha=0.48,
                zorder=1,
            )
        for row in sites.itertuples(index=False):
            centre = centres.loc[row.site_code]
            ax.scatter(
                centre.rd_x,
                centre.rd_y,
                s=30,
                color=GROUP_COLORS[row.forest_group],
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
            ax.annotate(codes[row.site_code], (centre.rd_x, centre.rd_y), xytext=(3, 2), textcoords="offset points", fontsize=6.5)
        ax.set_xlim(0, 290000)
        ax.set_ylim(295000, 620000)
        ax.set_aspect("equal")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("Dutch RD easting (m)")
    axes[0].set_ylabel("Dutch RD northing (m)")
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = fig.colorbar(scalar, ax=axes, fraction=0.025, pad=0.02)
    colorbar.set_label("Mean transfer RMSE / local RMSE")
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_figures(
    frame: pd.DataFrame,
    sites: pd.DataFrame,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    pairwise: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    directory = ROOT / str(config["outputs"]["figure_directory"])
    directory.mkdir(parents=True, exist_ok=True)
    codes = plot_codes(sites)
    centres = frame.groupby("site_code")[["rd_x", "rd_y"]].mean()
    boundaries = gpd.read_file(ROOT / str(config["outputs"]["selected_sites"])).to_crs("EPSG:28992")
    locator_map(sites, boundaries, centres, codes, directory / "phase37_site_locator_map.png")
    transfer_map(sites, centres, pairwise, codes, directory / "phase37_geographic_transfer_network.png")

    selected = metrics[
        (metrics["model"] == "ridge")
        & metrics["scenario"].isin(["local_cv", "same_group", "different_group", "all_other"])
    ]
    summary = selected.groupby(["target", "scenario"], as_index=False)["r2"].mean()
    order = list(TARGET_LABELS)
    scenarios = ["local_cv", "same_group", "different_group", "all_other"]
    labels = ["Local", "Same group", "Different group", "All other sites"]
    palette = ["#444444", "#247A52", "#B05C35", "#3A6EA5"]
    x = np.arange(len(order))
    width = 0.19
    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    for index, (scenario, label, color) in enumerate(zip(scenarios, labels, palette, strict=True)):
        values = summary[summary["scenario"] == scenario].set_index("target")["r2"].reindex(order)
        ax.bar(x + (index - 1.5) * width, values, width, label=label, color=color)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, [TARGET_LABELS[value] for value in order], rotation=25, ha="right")
    ax.set_ylabel("Mean held-out $R^2$ across target sites")
    ax.set_title("a  Local prediction and transfer by training source", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.19), ncol=4)
    fig.tight_layout()
    fig.savefig(directory / "phase37_scenario_performance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    pair_mean = pairwise.groupby(["source_site", "target_site"], as_index=False)["relative_rmse"].mean()
    site_order = sites["site_code"].astype(str).tolist()
    matrix = pair_mean.pivot(index="source_site", columns="target_site", values="relative_rmse").reindex(index=site_order, columns=site_order)
    np.fill_diagonal(matrix.values, 1.0)
    vmax = max(2.0, float(np.nanquantile(matrix.to_numpy(), 0.95)))
    fig, ax = plt.subplots(figsize=(11.0, 10.0))
    image = ax.imshow(matrix, cmap="RdYlBu_r", vmin=0.8, vmax=vmax)
    for i in range(len(site_order)):
        for j in range(len(site_order)):
            if np.isfinite(matrix.iloc[i, j]):
                ax.text(j, i, f"{matrix.iloc[i, j]:.1f}", ha="center", va="center", fontsize=4.5)
    ax.set_xticks(range(len(site_order)), [codes[value] for value in site_order], rotation=45, ha="right")
    ax.set_yticks(range(len(site_order)), [codes[value] for value in site_order])
    ax.set_xlabel("Target site")
    ax.set_ylabel("Training site")
    ax.set_title("b  Single-site transfer error relative to local training", loc="left", fontweight="bold")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Mean transfer RMSE / local RMSE")
    fig.tight_layout()
    fig.savefig(directory / "phase37_pairwise_transfer_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    pair_plot = pairwise.groupby(
        ["source_site", "target_site", "same_group", "distance_km"], as_index=False
    )["relative_rmse"].mean()
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for same, group in pair_plot.groupby("same_group"):
        ax.scatter(
            group["distance_km"],
            group["relative_rmse"],
            s=34,
            alpha=0.72,
            color="#247A52" if same else "#B05C35",
            marker="o" if same else "s",
            label="Same forest group" if same else "Different forest group",
        )
    ax.axhline(1, color="black", linestyle="--", linewidth=0.8)
    if float(pair_plot["relative_rmse"].max()) > 10:
        ax.set_yscale("log")
    ax.set_xlabel("Source-target distance (km)")
    ax.set_ylabel("Mean transfer RMSE / local RMSE")
    ax.set_title("c  Transfer penalty, forest group and distance", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2)
    fig.tight_layout()
    fig.savefig(directory / "phase37_distance_group_diagnostic.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    diagnostic_targets = ["ahn4_height_cv", "ahn4_entropy"]
    diagnostic_scenarios = ["local_cv", "same_group", "different_group"]
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 7.0))
    for row, target in enumerate(diagnostic_targets):
        for column, scenario in enumerate(diagnostic_scenarios):
            group = predictions[
                (predictions["model"] == "ridge")
                & (predictions["target"] == target)
                & (predictions["scenario"] == scenario)
            ]
            axes[row, column].hexbin(group["observed"], group["predicted"], gridsize=45, bins="log", mincnt=1, cmap="viridis")
            low = min(group["observed"].min(), group["predicted"].min())
            high = max(group["observed"].max(), group["predicted"].max())
            axes[row, column].plot([low, high], [low, high], color="#C73E3A", linestyle="--", linewidth=1)
            axes[row, column].set_title(f"{TARGET_LABELS[target]}: {scenario.replace('_', ' ')}", fontsize=9)
            axes[row, column].set_xlabel("Observed")
            axes[row, column].set_ylabel("Predicted")
    fig.suptitle("d  Prediction regimes for dispersion and return-profile outcomes", x=0.08, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(directory / "phase37_truth_prediction_diagnostics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def aggregate() -> None:
    config = load_config()
    outputs = config["outputs"]
    result_path = ROOT / str(outputs["result_freeze"])
    if result_path.exists():
        raise RuntimeError("Phase 37 results are already frozen")
    sites = selected_sites(config)
    worker_root = ROOT / str(outputs["worker_directory"])
    expected = [
        worker_root / f"worker_{index:02d}_{row.site_code}.parquet"
        for index, row in enumerate(sites.itertuples(index=False))
    ]
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise RuntimeError("Missing Phase 37 worker outputs:\n" + "\n".join(missing))
    predictions = pd.concat([pd.read_parquet(path) for path in expected], ignore_index=True)
    frame, _ = load_frame(config)
    metrics = metrics_from_predictions(predictions)
    groups = dict(zip(sites["site_code"].astype(str), sites["forest_group"].astype(str), strict=True))
    local_lookup = metrics[
        (metrics["model"] == "ridge") & (metrics["scenario"] == "local_cv")
    ].set_index(["target_site", "target"])
    centres = frame.groupby("site_code")[["rd_x", "rd_y"]].mean()
    pairwise = metrics[
        (metrics["model"] == "ridge") & (metrics["scenario"] == "single_source")
    ].rename(columns={"source_code": "source_site"}).copy()
    pairwise["same_group"] = [
        groups[source] == groups[target]
        for source, target in zip(pairwise["source_site"], pairwise["target_site"], strict=True)
    ]
    pairwise["distance_km"] = [
        float(np.linalg.norm(centres.loc[source].to_numpy() - centres.loc[target].to_numpy()) / 1000.0)
        for source, target in zip(pairwise["source_site"], pairwise["target_site"], strict=True)
    ]
    pairwise["local_rmse"] = [
        float(local_lookup.loc[(site, target), "rmse"])
        for site, target in zip(pairwise["target_site"], pairwise["target"], strict=True)
    ]
    pairwise["relative_rmse"] = pairwise["rmse"] / pairwise["local_rmse"]
    pair_summary = pairwise.groupby(
        ["source_site", "target_site", "same_group", "distance_km"], as_index=False
    )["relative_rmse"].mean()
    x = np.column_stack(
        [np.log1p(pair_summary["distance_km"]), pair_summary["same_group"].astype(float)]
    )
    model = LinearRegression().fit(x, pair_summary["relative_rmse"])
    diagnostic = pd.DataFrame(
        [
            {"statistic": "directed_pairs", "value": len(pair_summary)},
            {"statistic": "mean_relative_rmse_same_group", "value": pair_summary[pair_summary["same_group"]]["relative_rmse"].mean()},
            {"statistic": "mean_relative_rmse_different_group", "value": pair_summary[~pair_summary["same_group"]]["relative_rmse"].mean()},
            {"statistic": "spearman_distance_relative_rmse", "value": spearmanr(pair_summary["distance_km"], pair_summary["relative_rmse"]).statistic},
            {"statistic": "ols_log_distance_coefficient", "value": model.coef_[0]},
            {"statistic": "ols_same_group_coefficient", "value": model.coef_[1]},
            {"statistic": "ols_r2", "value": model.score(x, pair_summary["relative_rmse"])},
        ]
    )
    rng = np.random.default_rng(int(config["evaluation"]["bootstrap_seed"]))
    bootstrap = pd.DataFrame(
        scenario_bootstrap(predictions, "same_group", "different_group", int(config["evaluation"]["bootstrap_replicates"]), rng)
        + scenario_bootstrap(predictions, "all_other", "local_cv", int(config["evaluation"]["bootstrap_replicates"]), rng)
    )
    paths = {
        name: ROOT / str(outputs[name])
        for name in ["predictions", "metrics", "pairwise", "bootstrap", "distance_model", "report"]
    }
    atomic_parquet(paths["predictions"], predictions)
    atomic_csv(paths["metrics"], metrics)
    atomic_csv(paths["pairwise"], pairwise)
    atomic_csv(paths["bootstrap"], bootstrap)
    atomic_csv(paths["distance_model"], diagnostic)
    make_figures(frame, sites, predictions, metrics, pairwise, config)
    main = metrics[
        (metrics["model"] == "ridge")
        & metrics["scenario"].isin(["local_cv", "same_group", "different_group", "all_other"])
    ]
    table = main.groupby(["target", "scenario"], as_index=False).agg(
        r2=("r2", "mean"), rmse=("rmse", "mean")
    )
    site_table = sites[["site_name", "forest_group", "retained_units", "tessera_year_counts"]]
    lines = [
        "# Phase 37 expanded Dutch multi-site transfer experiment",
        "",
        f"{len(sites)} AHN4 forest regions were evaluated under one frozen LiDAR product and matching-year public TESSERA v1.0 embeddings.",
        "",
        "## Selected sites",
        "",
        "```text",
        site_table.to_string(index=False),
        "```",
        "",
        "## Mean performance across target sites",
        "",
        "```text",
        table.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "```",
        "",
        "## Forest group and distance diagnostic",
        "",
        "```text",
        diagnostic.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "```",
        "",
        "This remains a descriptive multi-site experiment: shared forest group and distance do not isolate management, environment or acquisition-year effects.",
    ]
    paths["report"].parent.mkdir(parents=True, exist_ok=True)
    paths["report"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    figure_directory = ROOT / str(outputs["figure_directory"])
    artifacts = {str(path.relative_to(ROOT)): sha256(path) for path in paths.values()}
    artifacts.update(
        {str(path.relative_to(ROOT)): sha256(path) for path in sorted(figure_directory.glob("*.png"))}
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "status": "complete",
                "sites": int(len(sites)),
                "rows": int(len(frame)),
                "artifacts": artifacts,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print("phase37 expanded Dutch transfer evaluation complete", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["worker", "aggregate"])
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.mode == "worker":
        if args.index is None:
            raise SystemExit("worker mode requires --index")
        worker(args.index)
    else:
        aggregate()


if __name__ == "__main__":
    main()
