#!/usr/bin/env python3
"""Compare local and jointly trained Scotland-Netherlands TESSERA v2 MLPs."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

import evaluate_scotland_netherlands_transfer as phase32


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/pooled_mlp_scotland_netherlands_pooled_mlp.yaml"
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


@dataclass
class TrainingData:
    features: np.ndarray
    targets: np.ndarray
    weights: np.ndarray
    groups: list[np.ndarray]


class MultiTargetMLP(nn.Module):
    def __init__(self, inputs: int, outputs: int, hidden: list[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = inputs
        for next_width in hidden:
            layers.extend(
                [nn.Linear(width, next_width), nn.GELU(), nn.Dropout(dropout)]
            )
            width = next_width
        layers.append(nn.Linear(width, outputs))
        self.network = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "phase35_scotland_netherlands_pooled_mlp"
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def nested_indices(
    site: phase32.SiteData,
    outer_fold: int,
    folds: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    outer_train, outer_test = site.folds[outer_fold]
    inner_train, inner_test = site.folds[(outer_fold + 1) % folds]
    subtrain = np.intersect1d(outer_train, inner_train, assume_unique=True)
    validation = np.intersect1d(outer_train, inner_test, assume_unique=True)
    if len(subtrain) == 0 or len(validation) == 0 or len(outer_test) == 0:
        raise RuntimeError(f"Empty nested split for {site.name} fold {outer_fold}")
    if np.intersect1d(subtrain, validation).size:
        raise RuntimeError("Subtraining and validation rows overlap")
    return subtrain, validation, outer_train, outer_test


def make_training_data(
    sites: list[phase32.SiteData],
    indices: list[np.ndarray],
    panel: str,
) -> TrainingData:
    features = np.concatenate(
        [site.features[panel][rows] for site, rows in zip(sites, indices)]
    ).astype(np.float32, copy=False)
    targets = np.concatenate(
        [
            np.column_stack([site.targets[target][rows] for target in TARGETS])
            for site, rows in zip(sites, indices)
        ]
    ).astype(np.float32, copy=False)
    total = len(features)
    groups = []
    weight_parts = []
    offset = 0
    for site, rows in zip(sites, indices):
        groups.append(np.arange(offset, offset + len(rows), dtype=np.int64))
        local = phase32.equal_block_weights(site.training_blocks[rows])
        weight_parts.append(local)
        offset += len(rows)
    if len(sites) > 1:
        weight_parts = [
            weights * (total / len(sites)) / weights.sum()
            for weights in weight_parts
        ]
    weights = np.concatenate(weight_parts).astype(np.float32)
    if not np.isclose(weights.mean(), 1.0, atol=1e-5):
        raise RuntimeError("Training weights do not have mean one")
    if len(sites) > 1:
        totals = [float(weights[group].sum()) for group in groups]
        if not np.allclose(totals, totals[0], rtol=1e-5):
            raise RuntimeError("Countries do not have equal total weight")
    if not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise RuntimeError("Non-finite values entered the MLP")
    return TrainingData(features, targets, weights, groups)


def weighted_statistics(
    values: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.average(values.astype(np.float64), axis=0, weights=weights).astype(
        np.float32
    )
    variance = np.average(
        np.square(values.astype(np.float64) - mean), axis=0, weights=weights
    )
    sd = np.sqrt(variance).astype(np.float32)
    sd[sd < 1e-6] = 1.0
    return mean, sd


def balanced_batches(
    groups: list[np.ndarray],
    batch_size: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    count = max(1, math.ceil(sum(len(group) for group in groups) / batch_size))
    split_groups = [np.array_split(rng.permutation(group), count) for group in groups]
    batches = []
    for batch in range(count):
        values = np.concatenate([parts[batch] for parts in split_groups])
        if len(values):
            batches.append(rng.permutation(values))
    return batches


def standardized_batch(
    data: TrainingData,
    indices: np.ndarray,
    x_mean: np.ndarray,
    x_sd: np.ndarray,
    y_mean: np.ndarray,
    y_sd: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = (data.features[indices] - x_mean) / x_sd
    y = (data.targets[indices] - y_mean) / y_sd
    return (
        torch.from_numpy(np.ascontiguousarray(x)),
        torch.from_numpy(np.ascontiguousarray(y)),
        torch.from_numpy(np.ascontiguousarray(data.weights[indices])),
    )


def evaluate_loss(
    model: nn.Module,
    data: TrainingData,
    x_mean: np.ndarray,
    x_sd: np.ndarray,
    y_mean: np.ndarray,
    y_sd: np.ndarray,
    batch_size: int,
) -> float:
    model.eval()
    total = 0.0
    total_weight = 0.0
    with torch.no_grad():
        indices = np.arange(len(data.features), dtype=np.int64)
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            x, y, weights = standardized_batch(
                data, selected, x_mean, x_sd, y_mean, y_sd
            )
            per_row = torch.square(model(x) - y).mean(dim=1)
            total += float(torch.sum(per_row * weights))
            total_weight += float(weights.sum())
    return total / total_weight


def train_epochs(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data: TrainingData,
    x_mean: np.ndarray,
    x_sd: np.ndarray,
    y_mean: np.ndarray,
    y_sd: np.ndarray,
    epochs: int,
    batch_size: int,
    rng: np.random.Generator,
) -> None:
    for _ in range(epochs):
        model.train()
        for selected in balanced_batches(data.groups, batch_size, rng):
            x, y, weights = standardized_batch(
                data, selected, x_mean, x_sd, y_mean, y_sd
            )
            optimizer.zero_grad(set_to_none=True)
            per_row = torch.square(model(x) - y).mean(dim=1)
            loss = torch.sum(per_row * weights) / torch.sum(weights)
            loss.backward()
            optimizer.step()


def new_model(settings: dict[str, Any]) -> MultiTargetMLP:
    return MultiTargetMLP(
        512,
        len(TARGETS),
        [int(value) for value in settings["hidden_dimensions"]],
        float(settings["dropout"]),
    )


def new_optimizer(
    model: nn.Module,
    settings: dict[str, Any],
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )


def select_epoch(
    subtrain: TrainingData,
    validation: TrainingData,
    settings: dict[str, Any],
    seed: int,
    label: str,
) -> tuple[int, float]:
    set_seed(seed)
    x_mean, x_sd = weighted_statistics(subtrain.features, subtrain.weights)
    y_mean, y_sd = weighted_statistics(subtrain.targets, subtrain.weights)
    model = new_model(settings)
    optimizer = new_optimizer(model, settings)
    rng = np.random.default_rng(seed)
    best_epoch = 0
    best_loss = math.inf
    stale = 0
    minimum = int(settings["minimum_epochs"])
    maximum = int(settings["maximum_epochs"])
    patience = int(settings["early_stopping_patience"])
    for epoch in range(1, maximum + 1):
        train_epochs(
            model,
            optimizer,
            subtrain,
            x_mean,
            x_sd,
            y_mean,
            y_sd,
            1,
            int(settings["batch_size"]),
            rng,
        )
        loss = evaluate_loss(
            model,
            validation,
            x_mean,
            x_sd,
            y_mean,
            y_sd,
            int(settings["batch_size"]),
        )
        if epoch >= minimum:
            if loss < best_loss - 1e-6:
                best_loss = loss
                best_epoch = epoch
                stale = 0
            else:
                stale += 1
        if epoch % 10 == 0:
            print(f"{label}: epoch={epoch} validation={loss:.6f}", flush=True)
        if epoch >= minimum and stale >= patience:
            break
    if best_epoch < minimum:
        raise RuntimeError("Epoch selection violated the minimum training period")
    return best_epoch, best_loss


def fit_final(
    training: TrainingData,
    settings: dict[str, Any],
    seed: int,
    epochs: int,
) -> tuple[nn.Module, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    set_seed(seed)
    x_mean, x_sd = weighted_statistics(training.features, training.weights)
    y_mean, y_sd = weighted_statistics(training.targets, training.weights)
    model = new_model(settings)
    optimizer = new_optimizer(model, settings)
    train_epochs(
        model,
        optimizer,
        training,
        x_mean,
        x_sd,
        y_mean,
        y_sd,
        epochs,
        int(settings["batch_size"]),
        np.random.default_rng(seed),
    )
    return model, (x_mean, x_sd, y_mean, y_sd)


def predict(
    model: nn.Module,
    features: np.ndarray,
    statistics: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    batch_size: int,
) -> np.ndarray:
    x_mean, x_sd, y_mean, y_sd = statistics
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            x = (np.asarray(features[start : start + batch_size], dtype=np.float32) - x_mean) / x_sd
            standardized = model(torch.from_numpy(np.ascontiguousarray(x))).numpy()
            outputs.append(standardized * y_sd + y_mean)
    return np.concatenate(outputs).astype(np.float32)


def prediction_rows(
    model: nn.Module,
    statistics: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    evaluation_sites: list[phase32.SiteData],
    tests: list[np.ndarray],
    scope: str,
    fold: int,
    seed: int,
    panel: str,
    batch_size: int,
    best_epoch: int,
) -> list[pd.DataFrame]:
    parts = []
    for site, test in zip(evaluation_sites, tests):
        values = predict(model, site.features[panel][test], statistics, batch_size)
        for column, target in enumerate(TARGETS):
            parts.append(
                pd.DataFrame(
                    {
                        "row_id": site.frame.iloc[test]["row_id"].to_numpy(),
                        "fold": fold,
                        "seed": seed,
                        "training_scope": scope,
                        "evaluation_site": site.name,
                        "target": target,
                        "observed": site.targets[target][test],
                        "predicted": values[:, column],
                        "evaluation_block": site.evaluation_blocks[test],
                        "best_epoch": best_epoch,
                    }
                )
            )
    return parts


def run_models(
    sites: list[phase32.SiteData],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    settings = config["model"]
    panel = str(config["features"]["panel"])
    folds = int(settings["folds"])
    seeds = [int(seed) for seed in settings["seeds"]]
    predictions: list[pd.DataFrame] = []
    training_rows = []
    total_runs = folds * len(seeds) * (len(sites) + 1)
    run_number = 0

    for fold in range(folds):
        splits = [nested_indices(site, fold, folds) for site in sites]
        for scope, selected_sites, split_indices in [
            (
                "local_cairngorms",
                [sites[0]],
                [splits[0]],
            ),
            (
                "local_savelsbos",
                [sites[1]],
                [splits[1]],
            ),
            (
                "pooled_balanced",
                sites,
                splits,
            ),
        ]:
            subtrain_indices = [values[0] for values in split_indices]
            validation_indices = [values[1] for values in split_indices]
            outer_train_indices = [values[2] for values in split_indices]
            tests = [values[3] for values in split_indices]
            subtrain = make_training_data(selected_sites, subtrain_indices, panel)
            validation = make_training_data(selected_sites, validation_indices, panel)
            outer_train = make_training_data(selected_sites, outer_train_indices, panel)
            for seed in seeds:
                run_number += 1
                label = f"run {run_number}/{total_runs} {scope} fold={fold} seed={seed}"
                print(f"starting {label}", flush=True)
                best_epoch, best_loss = select_epoch(
                    subtrain, validation, settings, seed, label
                )
                model, statistics = fit_final(
                    outer_train, settings, seed, best_epoch
                )
                predictions.extend(
                    prediction_rows(
                        model,
                        statistics,
                        selected_sites,
                        tests,
                        scope,
                        fold,
                        seed,
                        panel,
                        int(settings["batch_size"]),
                        best_epoch,
                    )
                )
                training_rows.append(
                    {
                        "training_scope": scope,
                        "fold": fold,
                        "seed": seed,
                        "subtrain_rows": len(subtrain.features),
                        "validation_rows": len(validation.features),
                        "outer_train_rows": len(outer_train.features),
                        "best_epoch": best_epoch,
                        "best_validation_loss": best_loss,
                    }
                )
                print(
                    f"completed {label}; best_epoch={best_epoch} validation={best_loss:.6f}",
                    flush=True,
                )
                del model, statistics
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(training_rows)


def ensemble_predictions(seed_predictions: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "row_id",
        "fold",
        "training_scope",
        "evaluation_site",
        "target",
        "evaluation_block",
    ]
    ensemble = (
        seed_predictions.groupby(keys, as_index=False)
        .agg(observed=("observed", "first"), predicted=("predicted", "mean"))
    )
    expected_seeds = seed_predictions["seed"].nunique()
    counts = seed_predictions.groupby(keys).size()
    if not (counts == expected_seeds).all():
        raise RuntimeError("Some ensemble predictions do not contain every seed")
    return ensemble


def summarize(ensemble: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scope, site, target), group in ensemble.groupby(
        ["training_scope", "evaluation_site", "target"], sort=True
    ):
        values = phase32.metrics(
            group["observed"].to_numpy(dtype=np.float64),
            group["predicted"].to_numpy(dtype=np.float64),
            group["evaluation_block"].to_numpy(),
        )
        rows.append(
            {
                "training_scope": scope,
                "evaluation_site": site,
                "target": target,
                "rows": len(group),
                **values,
            }
        )
    return pd.DataFrame(rows)


def compare_local_and_pooled(
    ensemble: pd.DataFrame,
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    comparisons = []
    bootstrap_rows = []
    for site in ["cairngorms", "savelsbos"]:
        local_scope = f"local_{site}"
        for target in TARGETS:
            local = ensemble[
                (ensemble["training_scope"] == local_scope)
                & (ensemble["evaluation_site"] == site)
                & (ensemble["target"] == target)
            ][["row_id", "observed", "predicted", "evaluation_block"]].rename(
                columns={"predicted": "local"}
            )
            pooled = ensemble[
                (ensemble["training_scope"] == "pooled_balanced")
                & (ensemble["evaluation_site"] == site)
                & (ensemble["target"] == target)
            ][["row_id", "predicted"]].rename(columns={"predicted": "pooled"})
            paired = local.merge(pooled, on="row_id", validate="one_to_one")
            observed = paired["observed"].to_numpy(dtype=np.float64)
            blocks = paired["evaluation_block"].to_numpy()
            local_metrics = phase32.metrics(
                observed, paired["local"].to_numpy(dtype=np.float64), blocks
            )
            pooled_metrics = phase32.metrics(
                observed, paired["pooled"].to_numpy(dtype=np.float64), blocks
            )
            comparisons.append(
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
            block_errors = paired.assign(
                local_error=np.square(paired["local"] - paired["observed"]),
                pooled_error=np.square(paired["pooled"] - paired["observed"]),
            ).groupby("evaluation_block", as_index=False)[
                ["local_error", "pooled_error"]
            ].mean()
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
    return pd.DataFrame(comparisons), pd.DataFrame(bootstrap_rows)


def make_figure(comparison: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))
    colors = {"cairngorms": "#168A8D", "savelsbos": "#D95F02"}
    x = np.arange(len(TARGETS))
    width = 0.34
    indexed = comparison.set_index(["evaluation_site", "target"])
    for offset, site in zip([-0.5, 0.5], ["cairngorms", "savelsbos"]):
        axes[0].bar(
            x + offset * width,
            [indexed.loc[(site, target), "pooled_over_local_rmse"] for target in TARGETS],
            width,
            color=colors[site],
            label=SITE_LABELS[site],
        )
        axes[1].bar(
            x + offset * width,
            [indexed.loc[(site, target), "delta_r2_pooled_minus_local"] for target in TARGETS],
            width,
            color=colors[site],
            label=SITE_LABELS[site],
        )
    labels = [TARGET_LABELS[target] for target in TARGETS]
    axes[0].axhline(1, color="black", ls="--", lw=0.9)
    axes[0].set_ylabel("Pooled MLP RMSE / local MLP RMSE")
    axes[0].set_title("a  Error relative to local MLP", loc="left", fontweight="bold")
    axes[1].axhline(0, color="black", ls="--", lw=0.9)
    axes[1].set_ylabel(r"Pooled minus local MLP $R^2$")
    axes[1].set_title("b  Change in explained variation", loc="left", fontweight="bold")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=25, ha="right")
        axis.grid(axis="y", color="#dddddd", lw=0.7)
        axis.set_axisbelow(True)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        "Local and jointly trained TESSERA v2 MLPs",
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
    training: pd.DataFrame,
) -> None:
    lines = [
        "# Joint Scotland-Netherlands TESSERA v2 MLP",
        "",
        "Local and pooled models use the same 512-dimensional TESSERA v2 context, 512-256-128-4 MLP architecture, three random seeds and five outer spatial folds. Epochs were selected using complete inner spatial validation folds before refitting on each outer training set. The pooled model gave each country equal total training weight and gave blocks equal influence within country.",
        "",
        "## Pooled versus local ensemble predictions",
        "",
        "```text",
        comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "```",
        "",
        "## Paired spatial-block bootstrap",
        "",
        "```text",
        bootstrap.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "```",
        "",
        "## Training summary",
        "",
        "```text",
        training.groupby("training_scope")["best_epoch"].describe().to_string(
            float_format=lambda value: f"{value:.2f}"
        ),
        "```",
        "",
        "This experiment evaluates whether sharing one nonlinear mapping improves spatial prediction when local reference observations from both landscapes are available. It is not a zero-shot transfer test.",
        "",
    ]
    atomic(path, lambda output: output.write_text("\n".join(lines), encoding="utf-8"))


def run() -> None:
    config = load_config()
    outputs = {key: ROOT / str(value) for key, value in config["outputs"].items()}
    if outputs["result_freeze"].exists():
        print("Phase 35 pooled MLP result is already frozen", flush=True)
        return
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.set_num_interop_threads(1)
    phase32_config = phase32.load_config()
    sites = [
        phase32.load_cairngorms(phase32_config),
        phase32.load_savelsbos(phase32_config),
    ]
    seed_predictions, training = run_models(sites, config)
    ensemble = ensemble_predictions(seed_predictions)
    metrics = summarize(ensemble)
    comparison, bootstrap = compare_local_and_pooled(
        ensemble,
        int(config["model"]["bootstrap_replicates"]),
        int(config["model"]["bootstrap_seed"]),
    )
    atomic(
        outputs["predictions"],
        lambda path: seed_predictions.to_parquet(path, index=False),
    )
    atomic(outputs["metrics"], lambda path: metrics.to_csv(path, index=False))
    atomic(outputs["comparison"], lambda path: comparison.to_csv(path, index=False))
    atomic(outputs["bootstrap"], lambda path: bootstrap.to_csv(path, index=False))
    atomic(outputs["training"], lambda path: training.to_csv(path, index=False))
    make_figure(comparison, outputs["figure"])
    write_report(outputs["report"], comparison, bootstrap, training)
    artifacts = {
        str(path.relative_to(ROOT)): sha256(path)
        for path in outputs.values()
        if path.exists() and path != outputs["result_freeze"]
    }
    inputs = [
        CONFIG_PATH,
        phase32.CONFIG_PATH,
        Path(__file__).resolve(),
        ROOT / phase32_config["cairngorms"]["surface_cohort"],
        ROOT / phase32_config["cairngorms"]["corrected_cohort"],
        ROOT / phase32_config["cairngorms"]["features"],
        ROOT / phase32_config["cairngorms"]["row_ids"],
        ROOT / phase32_config["savelsbos"]["cohort"],
        ROOT / phase32_config["savelsbos"]["features"],
    ]
    for site_name in ("cairngorms", "savelsbos"):
        site_config = phase32_config[site_name]
        fold_dir = ROOT / site_config["folds"]
        inputs.extend(
            fold_dir / site_config["fold_pattern"].format(fold=fold)
            for fold in range(int(config["model"]["folds"]))
        )
    freeze = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation": config["study"]["interpretation"],
        "architecture": "512-256-128-4 GELU MLP with 0.10 dropout",
        "training_weighting": "equal total country weight for pooled models; equal block influence within country",
        "inputs": {str(path.relative_to(ROOT)): sha256(path) for path in inputs},
        "artifacts": artifacts,
    }
    outputs["result_freeze"].parent.mkdir(parents=True, exist_ok=True)
    outputs["result_freeze"].write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("Phase 35 pooled MLP evaluation complete", flush=True)
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    run()
