#!/usr/bin/env python3
"""Aggregate the audited TESSERA v2 canopy-height experiment."""

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
CONFIG_PATH = ROOT / "configs/cairngorms_v2_height_unet.yaml"


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "phase30_cairngorms_v2_height_unet"
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a small results table without pandas' optional tabulate dependency."""
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        formatted = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                formatted.append(f"{float(value):.4f}")
            else:
                formatted.append(str(value))
        lines.append("| " + " | ".join(formatted) + " |")
    return "\n".join(lines)


def weighted_metrics(
    observed: np.ndarray, predicted: np.ndarray, blocks: np.ndarray
) -> dict[str, float]:
    valid = np.isfinite(observed) & np.isfinite(predicted)
    observed = observed[valid].astype(np.float64)
    predicted = predicted[valid].astype(np.float64)
    blocks = blocks[valid]
    unique, inverse, counts = np.unique(blocks, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    weights /= weights.sum()
    errors = predicted - observed
    observed_mean = float(np.sum(weights * observed))
    denominator = float(np.sum(weights * np.square(observed - observed_mean)))
    grouped = pd.DataFrame(
        {"observed": observed, "predicted": predicted, "block": blocks}
    ).groupby("block", as_index=False).mean()
    correlation = spearmanr(grouped["observed"], grouped["predicted"]).statistic
    return {
        "n": int(len(observed)),
        "blocks": int(len(unique)),
        "rmse": float(np.sqrt(np.sum(weights * np.square(errors)))),
        "mae": float(np.sum(weights * np.abs(errors))),
        "bias": float(np.sum(weights * errors)),
        "r2": (
            float(1 - np.sum(weights * np.square(errors)) / denominator)
            if denominator > 0
            else np.nan
        ),
        "block_spearman": float(correlation),
    }


def load_predictions(config: dict[str, Any]) -> pd.DataFrame:
    outputs = config["outputs"]
    array_dir = ROOT / str(outputs["array_directory"])
    result_dir = ROOT / str(outputs["result_directory"])
    row_ids = np.load(array_dir / "row_id.npy", mmap_mode="r")
    blocks = np.load(array_dir / "spatial_block.npy", mmap_mode="r")
    openings = np.load(array_dir / "opening_fraction.npy", mmap_mode="r")
    targets = [str(value) for value in config["targets"]["names"]]
    seeds = [int(value) for value in config["evaluation"]["seeds"]]
    rows: list[pd.DataFrame] = []
    for model in config["evaluation"]["models"]:
        for fold in range(int(config["evaluation"]["folds"])):
            test: np.ndarray | None = None
            observed: np.ndarray | None = None
            predictions: list[np.ndarray] = []
            for seed in seeds:
                path = result_dir / f"{model}_fold_{fold}_seed_{seed}.npz"
                if not path.exists():
                    raise RuntimeError(f"Missing Phase 30 result: {path}")
                with np.load(path) as values:
                    current_test = values["test_indices"].astype(np.int64)
                    current_observed = values["observed"].astype(np.float32)
                    if test is not None and not np.array_equal(test, current_test):
                        raise RuntimeError(f"Test rows changed between seeds: {path}")
                    if observed is not None and not np.array_equal(observed, current_observed):
                        raise RuntimeError(f"Observed values changed between seeds: {path}")
                    test = current_test
                    observed = current_observed
                    predictions.append(values["predictions"].astype(np.float32))
            assert test is not None and observed is not None
            ensemble = np.mean(predictions, axis=0)
            for column, target in enumerate(targets):
                rows.append(
                    pd.DataFrame(
                        {
                            "row_id": row_ids[test],
                            "model": str(model),
                            "fold": fold,
                            "target": target,
                            "observed": observed[:, column],
                            "predicted": ensemble[:, column],
                            "spatial_block": blocks[test],
                            "opening_fraction": openings[test],
                        }
                    )
                )
    return pd.concat(rows, ignore_index=True)


def metric_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for keys, local in predictions.groupby(["model", "target", "fold"], sort=False):
        model, target, fold = keys
        rows.append(
            {
                "model": model,
                "target": target,
                "fold": fold,
                **weighted_metrics(
                    local["observed"].to_numpy(),
                    local["predicted"].to_numpy(),
                    local["spatial_block"].to_numpy(),
                ),
            }
        )
    fold_metrics = pd.DataFrame(rows)
    macro = fold_metrics.groupby(["model", "target"], as_index=False).agg(
        folds=("fold", "nunique"),
        n=("n", "sum"),
        rmse=("rmse", "mean"),
        mae=("mae", "mean"),
        bias=("bias", "mean"),
        r2=("r2", "mean"),
        block_spearman=("block_spearman", "mean"),
        positive_r2_folds=("r2", lambda values: int((values > 0).sum())),
    )
    return fold_metrics, macro


def paired_comparison(
    predictions: pd.DataFrame,
    target: str,
    replicates: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    local = predictions[predictions["target"] == target].copy()
    local["squared_error"] = np.square(local["predicted"] - local["observed"])
    grouped = local.groupby(
        ["fold", "spatial_block", "model"], as_index=False
    )["squared_error"].mean()
    wide = grouped.pivot(
        index=["fold", "spatial_block"], columns="model", values="squared_error"
    ).dropna(subset=["tessera_v2_mlp_5x5", "tessera_v2_unet_audited"])

    def difference(frame: pd.DataFrame) -> float:
        fold_values: list[float] = []
        for fold in sorted(frame.index.get_level_values("fold").unique()):
            values = frame.xs(fold, level="fold").mean().pow(0.5)
            fold_values.append(
                float(values["tessera_v2_unet_audited"] - values["tessera_v2_mlp_5x5"])
            )
        return float(np.mean(fold_values))

    point = difference(wide)
    samples = np.empty(replicates, dtype=np.float64)
    folds = sorted(wide.index.get_level_values("fold").unique())
    for replicate in range(replicates):
        fold_values: list[float] = []
        for fold in folds:
            values = wide.xs(fold, level="fold")
            sampled = values.iloc[rng.integers(0, len(values), len(values))]
            rmse = sampled.mean().pow(0.5)
            fold_values.append(
                float(rmse["tessera_v2_unet_audited"] - rmse["tessera_v2_mlp_5x5"])
            )
        samples[replicate] = np.mean(fold_values)
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "target": target,
        "unet_minus_mlp_rmse": point,
        "ci_low": float(low),
        "ci_high": float(high),
        "probability_unet_better": float(np.mean(samples < 0)),
        "paired_blocks": int(len(wide)),
    }


def opening_group(values: pd.Series, config: dict[str, Any]) -> pd.Series:
    groups = config["evaluation"]["opening_groups"]
    closed = float(groups["closed_maximum"])
    intermediate = float(groups["intermediate_maximum"])
    return pd.cut(
        values,
        bins=[-np.inf, closed, intermediate, np.inf],
        labels=[f"< {closed:.0%}", f"{closed:.0%}-{intermediate:.0%}", f">= {intermediate:.0%}"],
        right=False,
    )


def opening_diagnostic(predictions: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    local = predictions.copy()
    local["opening_group"] = opening_group(local["opening_fraction"], config)
    rows: list[dict[str, Any]] = []
    for keys, group in local.groupby(
        ["model", "target", "opening_group", "fold"], observed=True, sort=False
    ):
        model, target, opening, fold = keys
        rows.append(
            {
                "model": model,
                "target": target,
                "opening_group": str(opening),
                "fold": fold,
                **weighted_metrics(
                    group["observed"].to_numpy(),
                    group["predicted"].to_numpy(),
                    group["spatial_block"].to_numpy(),
                ),
            }
        )
    fold_values = pd.DataFrame(rows)
    return fold_values.groupby(
        ["model", "target", "opening_group"], as_index=False
    ).agg(
        folds=("fold", "nunique"),
        n=("n", "sum"),
        rmse=("rmse", "mean"),
        r2=("r2", "mean"),
        block_spearman=("block_spearman", "mean"),
    )


def make_figure(
    predictions: pd.DataFrame,
    macro: pd.DataFrame,
    opening: pd.DataFrame,
    path: Path,
) -> None:
    primary = "canopy_p95_height_m"
    labels = {
        "tessera_v2_mlp_5x5": "V2 5 x 5 MLP",
        "tessera_v2_unet_audited": "Audited V2 U-Net",
    }
    colors = {
        "tessera_v2_mlp_5x5": "#3F7F93",
        "tessera_v2_unet_audited": "#D87C3C",
    }
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9), constrained_layout=True)
    for axis, model, panel in zip(
        axes[0], labels, ["a", "b"], strict=True
    ):
        local = predictions[
            (predictions["target"] == primary) & (predictions["model"] == model)
        ]
        extent = [
            float(min(local["observed"].min(), local["predicted"].min())),
            float(max(local["observed"].max(), local["predicted"].max())),
        ]
        axis.hexbin(
            local["observed"], local["predicted"], gridsize=55,
            mincnt=1, bins="log", cmap="viridis",
        )
        axis.plot(extent, extent, color="#C63D32", linestyle="--", linewidth=1.2)
        metrics = macro[(macro["target"] == primary) & (macro["model"] == model)].iloc[0]
        axis.text(
            0.04, 0.96,
            f"Mean $R^2$ = {metrics.r2:.3f}\nRMSE = {metrics.rmse:.3f} m",
            transform=axis.transAxes, va="top",
        )
        axis.set_xlabel("Observed p95 canopy height (m)")
        axis.set_ylabel("Predicted p95 canopy height (m)")
        axis.set_title(f"{panel}  {labels[model]}", loc="left", fontweight="bold")

    axis = axes[1, 0]
    targets = ["canopy_mean_height_m", "canopy_p95_height_m"]
    x = np.arange(len(targets))
    for offset, model in zip([-0.18, 0.18], labels, strict=True):
        values = [
            float(macro[(macro.model == model) & (macro.target == target)].rmse.iloc[0])
            for target in targets
        ]
        axis.bar(x + offset, values, 0.36, color=colors[model], label=labels[model])
    axis.set_xticks(x, ["Mean height", "P95 height"])
    axis.set_ylabel("Mean spatially held-out RMSE (m)")
    axis.set_title("c  Height prediction error", loc="left", fontweight="bold")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)

    axis = axes[1, 1]
    groups = list(opening["opening_group"].drop_duplicates())
    x = np.arange(len(groups))
    for offset, model in zip([-0.18, 0.18], labels, strict=True):
        local = opening[(opening.model == model) & (opening.target == primary)]
        values = [float(local[local.opening_group == group].rmse.iloc[0]) for group in groups]
        axis.bar(x + offset, values, 0.36, color=colors[model], label=labels[model])
    axis.set_xticks(x, groups)
    axis.set_xlabel("LiDAR canopy-opening fraction")
    axis.set_ylabel("P95 height RMSE (m)")
    axis.set_title("d  Error by canopy openness", loc="left", fontweight="bold")
    axis.grid(axis="y", alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run() -> None:
    config = load_config()
    outputs = config["outputs"]
    predictions = load_predictions(config)
    fold_metrics, macro = metric_tables(predictions)
    rng = np.random.default_rng(int(config["evaluation"]["bootstrap_seed"]))
    comparisons = pd.DataFrame(
        [
            paired_comparison(
                predictions,
                str(target),
                int(config["evaluation"]["bootstrap_replicates"]),
                rng,
            )
            for target in config["targets"]["names"]
        ]
    )
    opening = opening_diagnostic(predictions, config)

    primary = str(config["targets"]["primary"])
    primary_metrics = macro[macro.target == primary].set_index("model")
    mlp_rmse = float(primary_metrics.loc["tessera_v2_mlp_5x5", "rmse"])
    unet_rmse = float(primary_metrics.loc["tessera_v2_unet_audited", "rmse"])
    reduction = (mlp_rmse - unet_rmse) / mlp_rmse
    primary_comparison = comparisons[comparisons.target == primary].iloc[0]
    success_config = config["evaluation"]["success"]
    success = bool(
        reduction >= float(success_config["minimum_primary_rmse_reduction_fraction"])
        and (
            not bool(success_config["require_bootstrap_ci_below_zero"])
            or float(primary_comparison.ci_high) < 0
        )
    )
    comparisons["rmse_reduction_fraction"] = comparisons["target"].map(
        {
            target: (
                float(macro[(macro.model == "tessera_v2_mlp_5x5") & (macro.target == target)].rmse.iloc[0])
                - float(macro[(macro.model == "tessera_v2_unet_audited") & (macro.target == target)].rmse.iloc[0])
            )
            / float(macro[(macro.model == "tessera_v2_mlp_5x5") & (macro.target == target)].rmse.iloc[0])
            for target in config["targets"]["names"]
        }
    )

    for frame, key in [
        (predictions, "predictions"),
        (fold_metrics, "fold_metrics"),
        (macro, "macro_metrics"),
        (comparisons, "comparison"),
        (opening, "opening_diagnostic"),
    ]:
        path = ROOT / str(outputs[key])
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".parquet":
            frame.to_parquet(path, index=False)
        else:
            frame.to_csv(path, index=False)
    make_figure(predictions, macro, opening, ROOT / str(outputs["figure"]))

    report_path = ROOT / str(outputs["report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 30: audited TESSERA v2 canopy-height experiment",
        "",
        "The V2 5 x 5 MLP and corrected spatial U-Net were trained on identical",
        "Cairngorms rows, targets, spatial folds and random seeds.",
        "",
        "## Equal-fold results",
        "",
        markdown_table(macro),
        "",
        "## Paired U-Net comparison",
        "",
        markdown_table(comparisons),
        "",
        "## Interpretation",
        "",
        f"The frozen p95 success criterion {'passed' if success else 'did not pass'}. ",
        f"The corrected U-Net changed p95 RMSE from {mlp_rmse:.3f} m to "
        f"{unet_rmse:.3f} m ({reduction:+.1%} reduction).",
        "The fixed opening groups are diagnostic only and were not used to select a model.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")

    freeze_path = ROOT / str(outputs["result_freeze"])
    freeze = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "state": "complete",
        "primary_success": success,
        "primary_rmse_reduction_fraction": reduction,
        "config_sha256": sha256(CONFIG_PATH),
        "artifacts": {
            key: {"path": str(outputs[key]), "sha256": sha256(ROOT / str(outputs[key]))}
            for key in [
                "predictions", "fold_metrics", "macro_metrics", "comparison",
                "opening_diagnostic", "figure", "report",
            ]
        },
    }
    atomic_json(freeze_path, freeze)
    print(report_path.read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    run()
