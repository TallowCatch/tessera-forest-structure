#!/usr/bin/env python3
"""Fit one site-balanced TESSERA v2 model to Scotland and the Netherlands."""

from __future__ import annotations

import hashlib
import json
import gc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import evaluate_scotland_netherlands_transfer as phase32


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/pooled_transfer_scotland_netherlands_pooled.yaml"
PHASE32_PREDICTIONS = (
    ROOT / "data/processed/phase32_scotland_netherlands_transfer_predictions.parquet"
)
TARGETS = [
    "mean_height_m",
    "p95_height_m",
    "within_cell_height_sd_m",
    "height_cv",
]
TARGET_LABELS = {
    "mean_height_m": "Mean height",
    "p95_height_m": "P95 height",
    "within_cell_height_sd_m": "Height SD",
    "height_cv": "Height CV",
}
SITE_LABELS = {"cairngorms": "Cairngorms", "savelsbos": "Savelsbos"}


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "phase34_scotland_netherlands_pooled"
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp" + path.suffix)
    writer(temporary)
    temporary.replace(path)


def balanced_country_weights(
    sites: list[phase32.SiteData],
    training_indices: list[np.ndarray],
) -> np.ndarray:
    total_rows = sum(len(indices) for indices in training_indices)
    parts = []
    for site, indices in zip(sites, training_indices):
        within_site = phase32.equal_block_weights(site.training_blocks[indices])
        parts.append(within_site * (total_rows / 2.0) / within_site.sum())
    weights = np.concatenate(parts)
    if not np.isclose(weights.sum(), total_rows):
        raise RuntimeError("Country-balanced weights do not have mean one")
    site_totals = [float(values.sum()) for values in parts]
    if not np.isclose(site_totals[0], site_totals[1]):
        raise RuntimeError("Countries do not have equal total training weight")
    return weights


def pooled_predictions(
    sites: list[phase32.SiteData],
    panel: str,
    alpha: float,
    folds: int,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for fold in range(folds):
        training_indices = [site.folds[fold][0] for site in sites]
        test_indices = [site.folds[fold][1] for site in sites]
        x_train = np.concatenate(
            [site.features[panel][indices] for site, indices in zip(sites, training_indices)]
        ).astype(np.float32, copy=False)
        y_train = np.concatenate(
            [
                np.column_stack([site.targets[target][indices] for target in TARGETS])
                for site, indices in zip(sites, training_indices)
            ]
        )
        weights = balanced_country_weights(sites, training_indices)
        scaler = StandardScaler().fit(x_train, sample_weight=weights)
        model = Ridge(
            alpha=alpha,
            solver="lsqr",
            tol=1e-5,
            max_iter=2000,
        ).fit(
            scaler.transform(x_train), y_train, sample_weight=weights
        )
        for site, test in zip(sites, test_indices):
            predicted = model.predict(scaler.transform(site.features[panel][test]))
            for column, target in enumerate(TARGETS):
                parts.append(
                    pd.DataFrame(
                        {
                            "row_id": site.frame.iloc[test]["row_id"].to_numpy(),
                            "fold": fold,
                            "training_sites": "cairngorms+savelsbos",
                            "evaluation_site": site.name,
                            "feature_panel": panel,
                            "target": target,
                            "model": "pooled_balanced",
                            "observed": site.targets[target][test],
                            "predicted": predicted[:, column],
                            "evaluation_block": site.evaluation_blocks[test],
                        }
                    )
                )
        print(f"completed pooled fold {fold + 1}/{folds}", flush=True)
        del x_train, y_train, weights, scaler, model
        gc.collect()
    result = pd.concat(parts, ignore_index=True)
    for site in sites:
        for target in TARGETS:
            subset = result[
                (result["evaluation_site"] == site.name)
                & (result["target"] == target)
            ]
            if len(subset) != len(site.frame) or subset["row_id"].duplicated().any():
                raise RuntimeError(
                    f"Pooled predictions do not cover {site.name}/{target} exactly once"
                )
    return result


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (site, target), group in predictions.groupby(
        ["evaluation_site", "target"], sort=True
    ):
        values = phase32.metrics(
            group["observed"].to_numpy(dtype=np.float64),
            group["predicted"].to_numpy(dtype=np.float64),
            group["evaluation_block"].to_numpy(),
        )
        rows.append(
            {
                "evaluation_site": site,
                "target": target,
                "model": "pooled_balanced",
                "rows": len(group),
                **values,
            }
        )
    return pd.DataFrame(rows)


def local_reference() -> pd.DataFrame:
    frame = pd.read_parquet(PHASE32_PREDICTIONS)
    return frame[
        (frame["feature_panel"] == "context")
        & (frame["model"] == "local_cv")
        & (frame["training_site"] == frame["evaluation_site"])
        & frame["target"].isin(TARGETS)
    ][
        [
            "row_id",
            "fold",
            "evaluation_site",
            "target",
            "observed",
            "predicted",
            "evaluation_block",
        ]
    ].rename(columns={"predicted": "local_predicted"})


def compare_with_local(
    pooled: pd.DataFrame,
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    local = local_reference()
    paired = pooled.merge(
        local,
        on=["row_id", "fold", "evaluation_site", "target"],
        suffixes=("_pooled", "_local"),
        validate="one_to_one",
    )
    if not np.allclose(paired["observed_pooled"], paired["observed_local"]):
        raise RuntimeError("Phase 32 and pooled outcomes differ")
    if not np.array_equal(
        paired["evaluation_block_pooled"], paired["evaluation_block_local"]
    ):
        raise RuntimeError("Phase 32 and pooled evaluation blocks differ")

    rng = np.random.default_rng(seed)
    comparison_rows = []
    bootstrap_rows = []
    for (site, target), group in paired.groupby(
        ["evaluation_site", "target"], sort=True
    ):
        observed = group["observed_pooled"].to_numpy(dtype=np.float64)
        pooled_prediction = group["predicted"].to_numpy(dtype=np.float64)
        local_prediction = group["local_predicted"].to_numpy(dtype=np.float64)
        blocks = group["evaluation_block_pooled"].to_numpy()
        pooled_metrics = phase32.metrics(observed, pooled_prediction, blocks)
        local_metrics = phase32.metrics(observed, local_prediction, blocks)
        comparison_rows.append(
            {
                "evaluation_site": site,
                "target": target,
                "local_rmse": local_metrics["rmse"],
                "pooled_rmse": pooled_metrics["rmse"],
                "pooled_over_local_rmse": pooled_metrics["rmse"]
                / local_metrics["rmse"],
                "local_r2": local_metrics["r2"],
                "pooled_r2": pooled_metrics["r2"],
                "delta_r2_pooled_minus_local": pooled_metrics["r2"]
                - local_metrics["r2"],
            }
        )
        block_errors = pd.DataFrame(
            {
                "block": blocks,
                "local_error": np.square(local_prediction - observed),
                "pooled_error": np.square(pooled_prediction - observed),
            }
        ).groupby("block", as_index=False).mean(numeric_only=True)
        local_error = block_errors["local_error"].to_numpy()
        pooled_error = block_errors["pooled_error"].to_numpy()
        draws = np.empty(replicates, dtype=np.float64)
        for index in range(replicates):
            selected = rng.integers(0, len(block_errors), size=len(block_errors))
            draws[index] = np.sqrt(pooled_error[selected].mean()) - np.sqrt(
                local_error[selected].mean()
            )
        bootstrap_rows.append(
            {
                "evaluation_site": site,
                "target": target,
                "delta_rmse_pooled_minus_local": float(
                    np.sqrt(pooled_error.mean()) - np.sqrt(local_error.mean())
                ),
                "ci_low": float(np.quantile(draws, 0.025)),
                "ci_high": float(np.quantile(draws, 0.975)),
                "evaluation_blocks": len(block_errors),
            }
        )
    return pd.DataFrame(comparison_rows), pd.DataFrame(bootstrap_rows)


def make_figure(comparison: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))
    colors = {"cairngorms": "#168A8D", "savelsbos": "#D95F02"}
    x = np.arange(len(TARGETS))
    width = 0.34
    for offset, site in zip([-0.5, 0.5], ["cairngorms", "savelsbos"]):
        scoped = comparison.set_index(["evaluation_site", "target"]).loc[site]
        axes[0].bar(
            x + offset * width,
            [scoped.loc[target, "pooled_over_local_rmse"] for target in TARGETS],
            width,
            color=colors[site],
            label=SITE_LABELS[site],
        )
        axes[1].bar(
            x + offset * width,
            [scoped.loc[target, "delta_r2_pooled_minus_local"] for target in TARGETS],
            width,
            color=colors[site],
            label=SITE_LABELS[site],
        )
    labels = [TARGET_LABELS[target] for target in TARGETS]
    axes[0].axhline(1, color="black", ls="--", lw=0.9)
    axes[0].set_ylabel("Pooled RMSE / local RMSE")
    axes[0].set_title("a  Error relative to local training", loc="left", fontweight="bold")
    axes[1].axhline(0, color="black", ls="--", lw=0.9)
    axes[1].set_ylabel(r"Pooled minus local $R^2$")
    axes[1].set_title("b  Change in explained variation", loc="left", fontweight="bold")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=25, ha="right")
        axis.grid(axis="y", color="#dddddd", lw=0.7)
        axis.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        "One model trained jointly on Scotland and the Netherlands",
        y=0.99,
        fontweight="bold",
    )
    fig.subplots_adjust(top=0.79, bottom=0.24, left=0.08, right=0.99, wspace=0.18)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=240)
    plt.close(fig)


def write_report(
    path: Path,
    comparison: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> None:
    macro = (
        comparison.groupby("target", as_index=False)[
            ["local_rmse", "pooled_rmse", "local_r2", "pooled_r2"]
        ]
        .mean()
        .assign(
            pooled_over_local_rmse=lambda frame: frame["pooled_rmse"]
            / frame["local_rmse"],
            delta_r2_pooled_minus_local=lambda frame: frame["pooled_r2"]
            - frame["local_r2"],
        )
    )
    lines = [
        "# Joint Scotland-Netherlands TESSERA v2 model",
        "",
        "One Ridge model was fitted per spatial fold using training blocks from both landscapes. Each country received equal total training weight, and blocks within each country received equal influence. Held-out blocks in both countries were predicted by the same fitted model. The comparison uses the identical rows and folds as the Phase 32 locally trained references.",
        "",
        "## Per-landscape comparison",
        "",
        "```text",
        comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "```",
        "",
        "## Equal-country macro comparison",
        "",
        "```text",
        macro.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "```",
        "",
        "## Spatial-block bootstrap",
        "",
        "```text",
        bootstrap.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "```",
        "",
        "A pooled model can improve estimation within a landscape if shared training information is useful. It does not test zero-shot geographic transfer because every fitted model includes training outcomes from both landscapes.",
        "",
    ]
    atomic(path, lambda output: output.write_text("\n".join(lines), encoding="utf-8"))


def run() -> None:
    config = load_config()
    outputs = {key: ROOT / str(value) for key, value in config["outputs"].items()}
    if outputs["result_freeze"].exists():
        raise RuntimeError("Phase 34 pooled result is already frozen")
    phase32_config = phase32.load_config()
    sites = [
        phase32.load_cairngorms(phase32_config),
        phase32.load_savelsbos(phase32_config),
    ]
    predictions = pooled_predictions(
        sites,
        str(config["features"]["panel"]),
        float(config["model"]["alpha"]),
        int(config["model"]["folds"]),
    )
    metric_frame = summarize(predictions)
    comparison, bootstrap = compare_with_local(
        predictions,
        int(config["model"]["bootstrap_replicates"]),
        int(config["model"]["bootstrap_seed"]),
    )
    atomic(outputs["predictions"], lambda path: predictions.to_parquet(path, index=False))
    atomic(outputs["metrics"], lambda path: metric_frame.to_csv(path, index=False))
    atomic(outputs["comparison"], lambda path: comparison.to_csv(path, index=False))
    atomic(outputs["bootstrap"], lambda path: bootstrap.to_csv(path, index=False))
    make_figure(comparison, outputs["figure"])
    write_report(outputs["report"], comparison, bootstrap)
    artifacts = {
        str(path.relative_to(ROOT)): sha256(path)
        for path in outputs.values()
        if path.exists() and path != outputs["result_freeze"]
    }
    inputs = [CONFIG_PATH, phase32.CONFIG_PATH, PHASE32_PREDICTIONS]
    freeze = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation": config["study"]["interpretation"],
        "training_weighting": "equal total weight per country; equal block influence within country",
        "inputs": {str(path.relative_to(ROOT)): sha256(path) for path in inputs},
        "artifacts": artifacts,
    }
    outputs["result_freeze"].parent.mkdir(parents=True, exist_ok=True)
    outputs["result_freeze"].write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("Phase 34 pooled Scotland-Netherlands evaluation complete", flush=True)
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    run()
