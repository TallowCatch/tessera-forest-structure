#!/usr/bin/env python3
"""Evaluate local and cross-site TESSERA transfer for six Dutch forests."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

from evaluate_ahn4_replication_ahn4 import equal_block_weights, fit_ridge, metric_values


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/dutch_multisite_transfer.yaml"


TARGET_LABELS = {
    "ahn4_p95_height_m": "P95 height",
    "ahn4_height_sd_m": "Height SD",
    "ahn4_height_cv": "Height CV",
    "ahn4_entropy": "Return entropy",
    "ahn4_pulse_penetration": "Pulse penetration",
    "ahn4_sigma_z": "Sigma-z",
}
SITE_LABELS = {
    "veluwe": "Veluwe",
    "sallandse_heuvelrug": "Sallandse",
    "schoorlse_duinen": "Schoorlse",
    "savelsbos": "Savelsbos",
    "landgoederen_oldenzaal": "Oldenzaal",
    "drentsche_aa": "Drentsche Aa",
}


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "phase36_dutch_multisite_transfer"
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.csv")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def source_weights(frame: pd.DataFrame) -> np.ndarray:
    """Give sites equal weight and blocks equal weight within each site."""
    weights = np.zeros(len(frame), dtype=np.float64)
    for site, positions in frame.groupby("site_code", sort=True).indices.items():
        local = np.asarray(positions, dtype=np.int64)
        block_weights = equal_block_weights(frame.iloc[local]["fold_block"])
        weights[local] = block_weights / len(local)
    return weights * len(weights) / weights.sum()


def fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    target: str,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    weights = source_weights(train)
    scaler, model = fit_ridge(
        train[features].to_numpy(dtype=np.float64),
        train[target].to_numpy(dtype=np.float64),
        weights,
        alpha,
    )
    ridge = model.predict(scaler.transform(test[features].to_numpy(dtype=np.float64)))
    mean = np.full(len(test), np.average(train[target], weights=weights))
    return ridge, mean


def add_predictions(
    records: list[pd.DataFrame],
    test: pd.DataFrame,
    target: str,
    scenario: str,
    source_code: str,
    ridge: np.ndarray,
    mean: np.ndarray,
) -> None:
    common = {
        "row_id": test["row_id"].to_numpy(dtype=np.int64),
        "target_site": test["site_code"].to_numpy(),
        "target_group": test["forest_group"].to_numpy(),
        "fold_block": test["fold_block"].to_numpy(),
        "target": target,
        "scenario": scenario,
        "source_code": source_code,
        "observed": test[target].to_numpy(dtype=np.float64),
    }
    for model_name, values in [("ridge", ridge), ("training_mean", mean)]:
        records.append(pd.DataFrame({**common, "model": model_name, "predicted": values}))


def metrics_from_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    keys = ["target_site", "target", "scenario", "source_code", "model"]
    for values, group in predictions.groupby(keys, sort=True):
        weights = equal_block_weights(group["fold_block"])
        metrics = metric_values(
            group["observed"].to_numpy(dtype=np.float64),
            group["predicted"].to_numpy(dtype=np.float64),
            weights,
            group["fold_block"].to_numpy(),
        )
        records.append(dict(zip(keys, values, strict=True), rows=len(group), **metrics))
    return pd.DataFrame(records)


def scenario_bootstrap(
    predictions: pd.DataFrame,
    candidate: str,
    reference: str,
    replicates: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    scoped = predictions[
        (predictions["model"] == "ridge")
        & predictions["scenario"].isin([candidate, reference])
    ]
    for (site, target), group in scoped.groupby(["target_site", "target"], sort=True):
        pivot = group.pivot_table(
            index=["row_id", "fold_block"],
            columns="scenario",
            values=["observed", "predicted"],
            aggfunc="first",
        ).dropna()
        if pivot.empty or candidate not in pivot["predicted"] or reference not in pivot["predicted"]:
            continue
        observed = pivot[("observed", candidate)].to_numpy(dtype=np.float64)
        frame = pd.DataFrame(
            {
                "block": pivot.index.get_level_values("fold_block"),
                "candidate": np.square(pivot[("predicted", candidate)].to_numpy() - observed),
                "reference": np.square(pivot[("predicted", reference)].to_numpy() - observed),
            }
        ).groupby("block", as_index=False).mean()
        draws = np.empty(replicates, dtype=np.float64)
        for draw in range(replicates):
            selected = rng.integers(0, len(frame), size=len(frame))
            draws[draw] = np.sqrt(frame["candidate"].to_numpy()[selected].mean()) - np.sqrt(
                frame["reference"].to_numpy()[selected].mean()
            )
        records.append(
            {
                "target_site": site,
                "target": target,
                "candidate": candidate,
                "reference": reference,
                "delta_rmse": float(np.sqrt(frame["candidate"].mean()) - np.sqrt(frame["reference"].mean())),
                "ci_low": float(np.quantile(draws, 0.025)),
                "ci_high": float(np.quantile(draws, 0.975)),
                "blocks": int(len(frame)),
            }
        )
    return records


def make_figures(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    pairwise: pd.DataFrame,
    directory: Path,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    colors = {"conifer": "#247A52", "broadleaf": "#B05C35"}
    sites = frame.groupby(["site_code", "forest_group"], as_index=False).agg(
        rd_x=("rd_x", "mean"), rd_y=("rd_y", "mean"), rows=("row_id", "size")
    )
    fig, ax = plt.subplots(figsize=(6.8, 6.4))
    for row in sites.itertuples(index=False):
        ax.scatter(row.rd_x, row.rd_y, s=80, color=colors[row.forest_group], edgecolor="white", linewidth=0.8)
        ax.annotate(SITE_LABELS[row.site_code], (row.rd_x, row.rd_y), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Dutch RD easting (m)")
    ax.set_ylabel("Dutch RD northing (m)")
    ax.set_title("a  Frozen Dutch forest sites", loc="left", fontweight="bold")
    ax.legend(
        handles=[Line2D([0], [0], marker="o", color="none", markerfacecolor=value, label=key.title(), markersize=8) for key, value in colors.items()],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=2,
    )
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(directory / "phase36_site_map.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    selected = metrics[(metrics["model"] == "ridge") & metrics["scenario"].isin(["local_cv", "same_group", "different_group", "all_other"])]
    summary = selected.groupby(["target", "scenario"], as_index=False)["r2"].mean()
    order = list(TARGET_LABELS)
    scenarios = ["local_cv", "same_group", "different_group", "all_other"]
    labels = ["Local", "Same group", "Different group", "All other sites"]
    palette = ["#444444", "#247A52", "#B05C35", "#3A6EA5"]
    x = np.arange(len(order)); width = 0.19
    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    for index, (scenario, label, color) in enumerate(zip(scenarios, labels, palette, strict=True)):
        values = summary[summary["scenario"] == scenario].set_index("target")["r2"].reindex(order)
        ax.bar(x + (index - 1.5) * width, values, width, label=label, color=color)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, [TARGET_LABELS[value] for value in order], rotation=25, ha="right")
    ax.set_ylabel("Mean held-out $R^2$ across target sites")
    ax.set_title("a  Local prediction and transfer by training source", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=4)
    fig.tight_layout()
    fig.savefig(directory / "phase36_scenario_performance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    pair_mean = pairwise.groupby(["source_site", "target_site"], as_index=False)["relative_rmse"].mean()
    site_order = list(SITE_LABELS)
    matrix = pair_mean.pivot(index="source_site", columns="target_site", values="relative_rmse").reindex(index=site_order, columns=site_order)
    np.fill_diagonal(matrix.values, 1.0)
    fig, ax = plt.subplots(figsize=(6.5, 5.8))
    image = ax.imshow(matrix, cmap="RdYlBu_r", vmin=0.8, vmax=max(2.0, float(np.nanquantile(matrix, 0.95))))
    for i in range(len(site_order)):
        for j in range(len(site_order)):
            if np.isfinite(matrix.iloc[i, j]):
                ax.text(j, i, f"{matrix.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(site_order)), [SITE_LABELS[value] for value in site_order], rotation=35, ha="right")
    ax.set_yticks(range(len(site_order)), [SITE_LABELS[value] for value in site_order])
    ax.set_xlabel("Target site")
    ax.set_ylabel("Training site")
    ax.set_title("b  Single-site transfer error relative to local training", loc="left", fontweight="bold")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Mean transfer RMSE / local RMSE")
    fig.tight_layout()
    fig.savefig(directory / "phase36_pairwise_transfer_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    pair_plot = pairwise.groupby(["source_site", "target_site", "same_group", "distance_km"], as_index=False)["relative_rmse"].mean()
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for same, group in pair_plot.groupby("same_group"):
        ax.scatter(
            group["distance_km"], group["relative_rmse"],
            s=55, alpha=0.85,
            color="#247A52" if same else "#B05C35",
            marker="o" if same else "s",
            label="Same forest group" if same else "Different forest group",
        )
    ax.axhline(1, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Source-target distance (km)")
    ax.set_ylabel("Mean transfer RMSE / local RMSE")
    ax.set_title("c  Transfer penalty, forest group and distance", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2)
    fig.tight_layout()
    fig.savefig(directory / "phase36_distance_group_diagnostic.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    diagnostic_targets = ["ahn4_height_cv", "ahn4_entropy"]
    scenarios = ["local_cv", "same_group", "different_group"]
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 7.0))
    for row, target in enumerate(diagnostic_targets):
        for column, scenario in enumerate(scenarios):
            group = predictions[(predictions["model"] == "ridge") & (predictions["target"] == target) & (predictions["scenario"] == scenario)]
            axes[row, column].hexbin(group["observed"], group["predicted"], gridsize=45, bins="log", mincnt=1, cmap="viridis")
            low = min(group["observed"].min(), group["predicted"].min())
            high = max(group["observed"].max(), group["predicted"].max())
            axes[row, column].plot([low, high], [low, high], color="#C73E3A", linestyle="--", linewidth=1)
            axes[row, column].set_title(f"{TARGET_LABELS[target]}: {scenario.replace('_', ' ')}", fontsize=9)
            axes[row, column].set_xlabel("Observed")
            axes[row, column].set_ylabel("Predicted")
    fig.suptitle("d  Prediction regimes for a dispersion and a return-profile metric", x=0.08, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(directory / "phase36_truth_prediction_diagnostics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def run() -> None:
    config = load_config()
    outputs = config["outputs"]
    result_path = ROOT / str(outputs["result_freeze"])
    if result_path.exists():
        raise RuntimeError("Phase 36 results are already frozen")
    cohort_path = ROOT / str(outputs["cohort"])
    feature_path = ROOT / str(outputs["features"])
    cohort = pd.read_parquet(cohort_path)
    features = pd.read_parquet(feature_path)
    frame = cohort.merge(features, on="row_id", validate="one_to_one")
    columns = sorted(column for column in frame if column.startswith("tessera_mean_"))
    if len(columns) != 128:
        raise RuntimeError(f"Expected 128 TESSERA mean features, found {len(columns)}")
    frame = frame[
        frame["tessera_context_valid"].astype(bool)
        & np.isfinite(frame[columns]).all(axis=1)
    ].reset_index(drop=True)
    targets = [str(value) for value in config["evaluation"]["targets"]]
    alpha = float(config["evaluation"]["ridge_alpha"])
    records: list[pd.DataFrame] = []
    site_codes = [str(value["code"]) for value in config["study"]["sites"]]
    groups = {str(value["code"]): str(value["group"]) for value in config["study"]["sites"]}
    fold_root = ROOT / str(outputs["folds"])
    for target_site in site_codes:
        target_rows = frame[frame["site_code"] == target_site]
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
            "same_group": [site for site in site_codes if site != target_site and groups[site] == groups[target_site]],
            "different_group": [site for site in site_codes if groups[site] != groups[target_site]],
            "all_other": [site for site in site_codes if site != target_site],
        }
        for scenario, sources in scenarios.items():
            train = frame[frame["site_code"].isin(sources)]
            for target in targets:
                ridge, mean = fit_predict(train, target_rows, columns, target, alpha)
                add_predictions(records, target_rows, target, scenario, "+".join(sources), ridge, mean)
    predictions = pd.concat(records, ignore_index=True)
    metrics = metrics_from_predictions(predictions)
    local_lookup = metrics[
        (metrics["model"] == "ridge") & (metrics["scenario"] == "local_cv")
    ].set_index(["target_site", "target"])
    site_centres = frame.groupby("site_code")[["rd_x", "rd_y"]].mean()
    pairwise = metrics[
        (metrics["model"] == "ridge") & (metrics["scenario"] == "single_source")
    ].rename(columns={"source_code": "source_site"}).copy()
    pairwise["same_group"] = [
        groups[source] == groups[target]
        for source, target in zip(pairwise["source_site"], pairwise["target_site"], strict=True)
    ]
    pairwise["distance_km"] = [
        float(np.linalg.norm(site_centres.loc[source].to_numpy() - site_centres.loc[target].to_numpy()) / 1000.0)
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
            {"statistic": "pairs", "value": len(pair_summary)},
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
    figure_directory = ROOT / str(outputs["figure_directory"])
    make_figures(frame, predictions, metrics, pairwise, figure_directory)
    main = metrics[(metrics["model"] == "ridge") & metrics["scenario"].isin(["local_cv", "same_group", "different_group", "all_other"])]
    table = main.groupby(["target", "scenario"], as_index=False).agg(r2=("r2", "mean"), rmse=("rmse", "mean"))
    lines = [
        "# Phase 36 Dutch multi-site transfer pilot",
        "",
        "Six AHN4 forest regions were evaluated under one frozen LiDAR product and matching-year public TESSERA v1.0 embeddings.",
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
        "The distance/group analysis is descriptive because six selected sites cannot isolate causal effects of taxonomy, management, geography and acquisition year.",
    ]
    paths["report"].parent.mkdir(parents=True, exist_ok=True)
    paths["report"].write_text("\n".join(lines) + "\n", encoding="utf-8")
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
                "rows": int(len(frame)),
                "artifacts": artifacts,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print("phase36 multi-site transfer evaluation complete", flush=True)


if __name__ == "__main__":
    run()
