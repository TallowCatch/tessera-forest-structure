#!/usr/bin/env python3
"""Aggregate Phase 23 height-adjusted models with paired block uncertainty."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/cairngorms_height_adjusted.yaml"


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "phase23_cairngorms_height_adjusted"
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def weighted_metrics(observed: np.ndarray, predicted: np.ndarray, block: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(observed) & np.isfinite(predicted)
    observed = observed[valid].astype(np.float64)
    predicted = predicted[valid].astype(np.float64)
    block = block[valid]
    _, inverse, counts = np.unique(block, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    weights /= weights.sum()
    error = predicted - observed
    mean = float(np.sum(weights * observed))
    denominator = float(np.sum(weights * np.square(observed - mean)))
    grouped = pd.DataFrame({"observed": observed, "predicted": predicted, "block": block}).groupby("block").mean()
    return {
        "n": int(len(observed)), "blocks": int(len(grouped)),
        "rmse": float(np.sqrt(np.sum(weights * np.square(error)))),
        "r2": float(1 - np.sum(weights * np.square(error)) / denominator) if denominator > 0 else np.nan,
        "block_spearman": float(spearmanr(grouped["observed"], grouped["predicted"]).statistic),
    }


def bootstrap_difference(
    predictions: pd.DataFrame,
    first: str,
    second: str,
    replicates: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    squared = predictions.assign(squared_error=np.square(predictions["predicted"] - predictions["observed"]))
    grouped = squared.groupby(["fold", "spatial_block", "model"], as_index=False)["squared_error"].mean()
    wide = grouped.pivot(index=["fold", "spatial_block"], columns="model", values="squared_error").dropna(subset=[first, second])
    point_fold = wide.groupby(level="fold")[[first, second]].mean().pow(0.5)
    point = float((point_fold[first] - point_fold[second]).mean())
    samples = np.empty(replicates, dtype=np.float64)
    folds = sorted(wide.index.get_level_values("fold").unique())
    for replicate in range(replicates):
        differences = []
        for fold in folds:
            local = wide.xs(fold, level="fold")
            chosen = rng.integers(0, len(local), len(local))
            values = local.iloc[chosen][[first, second]].mean().pow(0.5)
            differences.append(float(values[first] - values[second]))
        samples[replicate] = np.mean(differences)
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "rmse_difference": point,
        "ci_low": float(low),
        "ci_high": float(high),
        "probability_first_better": float(np.mean(samples < 0)),
        "blocks": int(len(wide)),
    }


def make_figure(macro: pd.DataFrame, config: dict[str, Any], path: Path) -> None:
    targets = [str(value) for value in config["targets"]["names"]]
    labels = {
        "canopy_surface_sd_m": "Height SD", "canopy_surface_cv": "Height CV",
        "canopy_surface_rcv": "Robust CV", "canopy_rumple": "Rumple",
        "canopy_open_fraction": "Openings", "canopy_height_kurtosis": "Kurtosis",
    }
    raw_path = ROOT / "outputs/tables/phase22_cairngorms_spatial_unet_macro.csv"
    raw = pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame()
    mapping = {"conventional": "conventional_mlp", "tessera": "tessera_mlp_5x5", "fused": "fused_mlp_5x5"}
    models = ["conventional", "tessera", "fused"]
    colors = {"conventional": "#377eb8", "tessera": "#15958f", "fused": "#e68632"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    x = np.arange(len(targets)); width = 0.24
    for index, model in enumerate(models):
        adjusted_values = [float(macro[(macro.model == model) & (macro.target == target)].r2.iloc[0]) for target in targets]
        axes[1].bar(x + (index - 1) * width, adjusted_values, width, color=colors[model], label=model)
        if len(raw):
            raw_values = [float(raw[(raw.scheme == "balanced") & (raw.model == mapping[model]) & (raw.target == target)].r2.iloc[0]) for target in targets]
            axes[0].bar(x + (index - 1) * width, raw_values, width, color=colors[model], label=model)
    for axis, title in zip(axes, ["Original surface outcomes", "After accounting for canopy height"]):
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(x, [labels[target] for target in targets], rotation=22, ha="right")
        axis.set_ylabel(r"Mean held-out $R^2$")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, loc="upper left")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run() -> None:
    config = load_config()
    outputs = config["outputs"]
    array_dir = ROOT / str(outputs["array_directory"])
    result_dir = ROOT / str(outputs["result_directory"])
    target_names = [str(value) for value in config["targets"]["names"]]
    models = [str(value) for value in config["evaluation"]["models"]]
    seeds = [int(value) for value in config["evaluation"]["seeds"]]
    blocks = np.load(array_dir / "spatial_block.npy")
    row_ids = np.load(array_dir / "row_id.npy")
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    for fold in range(int(config["evaluation"]["folds"])):
        for model in models:
            predictions = []
            observed = None
            test = None
            for seed in seeds:
                path = result_dir / f"{model}_fold_{fold}_seed_{seed}.npz"
                if not path.exists():
                    raise RuntimeError(f"Missing result: {path}")
                with np.load(path) as values:
                    current_test = values["test_indices"].astype(np.int64)
                    if test is not None and not np.array_equal(test, current_test):
                        raise RuntimeError(f"Test rows changed in {path}")
                    test = current_test
                    observed = values["observed"].astype(np.float32)
                    predictions.append(values["predictions"].astype(np.float32))
            assert test is not None and observed is not None
            ensemble = np.mean(predictions, axis=0)
            for column, target in enumerate(target_names):
                metrics = weighted_metrics(observed[:, column], ensemble[:, column], blocks[test])
                metric_rows.append({"fold": fold, "model": model, "target": target, **metrics})
                prediction_rows.append(pd.DataFrame({
                    "row_id": row_ids[test], "fold": fold, "model": model, "target": target,
                    "observed": observed[:, column], "predicted": ensemble[:, column],
                    "spatial_block": blocks[test],
                }))
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    macro = metrics.groupby(["model", "target"], as_index=False).agg(
        folds=("fold", "nunique"), n=("n", "sum"), blocks=("blocks", "sum"),
        rmse=("rmse", "mean"), r2=("r2", "mean"), block_spearman=("block_spearman", "mean"),
        positive_r2_folds=("r2", lambda values: int((values > 0).sum())),
    )
    rng = np.random.default_rng(int(config["evaluation"]["bootstrap_seed"]))
    bootstrap_rows = []
    for target in target_names:
        local = predictions[predictions.target == target]
        for first, second in config["evaluation"]["comparisons"]:
            result = bootstrap_difference(
                local, str(first), str(second), int(config["evaluation"]["bootstrap_replicates"]), rng
            )
            bootstrap_rows.append({"target": target, "first_model": first, "second_model": second, **result})
    bootstrap = pd.DataFrame(bootstrap_rows)
    paths = {
        "predictions": ROOT / str(outputs["prediction_table"]),
        "metrics": ROOT / str(outputs["fold_metrics_table"]),
        "macro": ROOT / str(outputs["macro_table"]),
        "bootstrap": ROOT / str(outputs["bootstrap_table"]),
        "figure": ROOT / str(outputs["figure"]),
        "report": ROOT / str(outputs["report"]),
    }
    for path in paths.values(): path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(paths["predictions"], index=False, compression="zstd")
    metrics.to_csv(paths["metrics"], index=False)
    macro.to_csv(paths["macro"], index=False)
    bootstrap.to_csv(paths["bootstrap"], index=False)
    make_figure(macro, config, paths["figure"])
    lines = [
        "# Phase 23 height-adjusted Cairngorms results", "",
        "Targets were adjusted for mean and p95 canopy height using training rows only. Models use the frozen balanced Phase 22 spatial folds.", "",
        "## Macro results", "", "```text", macro.to_string(index=False, float_format=lambda x: f"{x:.4f}"), "```", "",
        "## Paired block bootstrap", "", "Negative RMSE differences favour the first model.", "", "```text",
        bootstrap.to_string(index=False, float_format=lambda x: f"{x:.4f}"), "```", "",
        "## Scope", "", "These results measure spatial generalisation within the Cairngorms. Height adjustment removes the average training-region relationship with mean and p95 height, but cannot remove every influence of stand development, terrain or management.",
    ]
    paths["report"].write_text("\n".join(lines) + "\n")
    freeze = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": {str(path.relative_to(ROOT)): sha256(path) for path in paths.values()},
    }
    freeze_path = ROOT / str(outputs["result_freeze"])
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n")
    print(json.dumps(freeze, indent=2), flush=True)


if __name__ == "__main__":
    run()
