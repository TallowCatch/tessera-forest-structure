#!/usr/bin/env python3
"""Compare reflection-augmented and frozen unaugmented summary MLPs."""

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
CONFIG_PATH = ROOT / "configs/reflection_augmentation_cairngorms_flip_augmentation.yaml"


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "phase33_cairngorms_flip_augmentation"
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def weighted_metrics(
    observed: np.ndarray, predicted: np.ndarray, blocks: np.ndarray
) -> dict[str, float]:
    _, inverse, counts = np.unique(blocks, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse].astype(np.float64)
    weights /= weights.sum()
    error = predicted.astype(np.float64) - observed.astype(np.float64)
    observed_mean = float(np.sum(weights * observed))
    denominator = float(np.sum(weights * np.square(observed - observed_mean)))
    grouped = pd.DataFrame(
        {"observed": observed, "predicted": predicted, "block": blocks}
    ).groupby("block", as_index=False).mean(numeric_only=True)
    return {
        "n": int(len(observed)),
        "blocks": int(len(grouped)),
        "rmse": float(np.sqrt(np.sum(weights * np.square(error)))),
        "r2": (
            float(1.0 - np.sum(weights * np.square(error)) / denominator)
            if denominator > 0
            else float("nan")
        ),
        "block_spearman": float(
            spearmanr(grouped["observed"], grouped["predicted"]).statistic
        ),
    }


def source_paths(config: dict[str, Any], variant: str, fold: int, seed: int):
    frozen = config["frozen_inputs"]
    output = ROOT / str(config["outputs"]["result_directory"])
    if variant == "raw":
        baseline = ROOT / str(frozen["raw_baseline_results"]) / (
            f"balanced_tessera_mlp_5x5_fold{fold}_seed{seed}.npz"
        )
        array_directory = ROOT / str(frozen["raw_array_directory"])
    else:
        baseline = ROOT / str(frozen["adjusted_baseline_results"]) / (
            f"tessera_fold_{fold}_seed_{seed}.npz"
        )
        array_directory = ROOT / str(frozen["adjusted_array_directory"])
    augmented = output / f"{variant}_fold{fold}_seed{seed}.npz"
    return baseline, augmented, array_directory


def collect_predictions(config: dict[str, Any]) -> pd.DataFrame:
    seeds = [int(value) for value in config["evaluation"]["seeds"]]
    rows: list[pd.DataFrame] = []
    for variant in config["evaluation"]["variants"]:
        targets = [str(value) for value in config["targets"][variant]]
        for fold in range(int(config["evaluation"]["folds"])):
            by_method: dict[str, list[np.ndarray]] = {
                "unaugmented": [],
                "flip_augmented": [],
            }
            observed = None
            test = None
            array_directory = None
            for seed in seeds:
                baseline, augmented, array_directory = source_paths(
                    config, variant, fold, seed
                )
                for method, path in [
                    ("unaugmented", baseline),
                    ("flip_augmented", augmented),
                ]:
                    if not path.exists():
                        raise RuntimeError(f"Missing model result: {path}")
                    with np.load(path) as values:
                        current_test = values["test_indices"].astype(np.int64)
                        current_observed = values["observed"].astype(np.float32)
                        if test is not None and not np.array_equal(test, current_test):
                            raise RuntimeError(f"Test rows changed in {path}")
                        if observed is not None and not np.allclose(
                            observed, current_observed, equal_nan=True
                        ):
                            raise RuntimeError(f"Observed outcomes changed in {path}")
                        test = current_test
                        observed = current_observed
                        by_method[method].append(
                            values["predictions"].astype(np.float32)
                        )
            assert test is not None and observed is not None and array_directory is not None
            row_ids = np.load(array_directory / "row_id.npy", mmap_mode="r")
            blocks = np.load(array_directory / "spatial_block.npy", mmap_mode="r")
            for method, predictions in by_method.items():
                ensemble = np.mean(predictions, axis=0)
                for column, target in enumerate(targets):
                    rows.append(
                        pd.DataFrame(
                            {
                                "row_id": row_ids[test],
                                "variant": variant,
                                "method": method,
                                "fold": fold,
                                "target": target,
                                "observed": observed[:, column],
                                "predicted": ensemble[:, column],
                                "spatial_block": blocks[test],
                            }
                        )
                    )
    return pd.concat(rows, ignore_index=True)


def paired_difference(
    predictions: pd.DataFrame, replicates: int, rng: np.random.Generator
) -> dict[str, float]:
    squared = predictions.assign(
        squared_error=np.square(predictions["predicted"] - predictions["observed"])
    )
    grouped = squared.groupby(
        ["fold", "spatial_block", "method"], as_index=False
    )["squared_error"].mean()
    wide = grouped.pivot(
        index=["fold", "spatial_block"], columns="method", values="squared_error"
    ).dropna(subset=["unaugmented", "flip_augmented"])
    folds = sorted(wide.index.get_level_values("fold").unique())
    fold_rmse = wide.groupby(level="fold")[
        ["unaugmented", "flip_augmented"]
    ].mean().pow(0.5)
    point = float(
        (fold_rmse["flip_augmented"] - fold_rmse["unaugmented"]).mean()
    )
    draws = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        differences = []
        for fold in folds:
            local = wide.xs(fold, level="fold")
            selected = rng.integers(0, len(local), size=len(local))
            values = local.iloc[selected][
                ["unaugmented", "flip_augmented"]
            ].mean().pow(0.5)
            differences.append(
                float(values["flip_augmented"] - values["unaugmented"])
            )
        draws[replicate] = np.mean(differences)
    return {
        "delta_rmse_flip_minus_unaugmented": point,
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "probability_flip_better": float(np.mean(draws < 0)),
        "blocks": int(len(wide)),
    }


def make_figure(macro: pd.DataFrame, path: Path) -> None:
    labels = {
        "canopy_p95_height_m": "P95 height",
        "canopy_surface_sd_m": "Height SD",
        "canopy_surface_cv": "Height CV",
        "canopy_surface_rcv": "Robust CV",
        "canopy_rumple": "Rumple",
        "canopy_open_fraction": "Openings",
        "canopy_height_kurtosis": "Kurtosis",
    }
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), constrained_layout=True)
    for axis, variant, title in [
        (axes[0], "raw", "Original canopy-surface outcomes"),
        (axes[1], "height_adjusted", "After accounting for canopy height"),
    ]:
        local = macro[macro.variant == variant]
        targets = list(local["target"].drop_duplicates())
        x = np.arange(len(targets))
        for offset, method, label, color in [
            (-0.18, "unaugmented", "Unaugmented", "#6C757D"),
            (0.18, "flip_augmented", "Reflection augmented", "#148F88"),
        ]:
            values = [
                float(
                    local[(local.target == target) & (local.method == method)]
                    .r2.iloc[0]
                )
                for target in targets
            ]
            axis.bar(x + offset, values, 0.36, label=label, color=color)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(x, [labels[target] for target in targets], rotation=25, ha="right")
        axis.set_ylabel(r"Mean spatially held-out $R^2$")
        axis.set_title(title, loc="left", fontweight="bold")
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, ncol=2, loc="upper left")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def run() -> None:
    config = load_config()
    freeze_path = ROOT / str(config["outputs"]["result_freeze"])
    if freeze_path.exists():
        raise RuntimeError("Phase 33 result is already frozen")
    predictions = collect_predictions(config)
    metric_rows = []
    for keys, local in predictions.groupby(
        ["variant", "method", "target", "fold"], sort=False
    ):
        variant, method, target, fold = keys
        metric_rows.append(
            {
                "variant": variant,
                "method": method,
                "target": target,
                "fold": fold,
                **weighted_metrics(
                    local["observed"].to_numpy(),
                    local["predicted"].to_numpy(),
                    local["spatial_block"].to_numpy(),
                ),
            }
        )
    metrics = pd.DataFrame(metric_rows)
    macro = metrics.groupby(
        ["variant", "method", "target"], as_index=False
    ).agg(
        folds=("fold", "nunique"),
        n=("n", "sum"),
        blocks=("blocks", "sum"),
        rmse=("rmse", "mean"),
        r2=("r2", "mean"),
        block_spearman=("block_spearman", "mean"),
        positive_r2_folds=("r2", lambda values: int((values > 0).sum())),
    )
    rng = np.random.default_rng(int(config["evaluation"]["bootstrap_seed"]))
    comparison_rows = []
    for (variant, target), local in predictions.groupby(
        ["variant", "target"], sort=False
    ):
        result = paired_difference(
            local,
            int(config["evaluation"]["bootstrap_replicates"]),
            rng,
        )
        lookup = macro[
            (macro.variant == variant) & (macro.target == target)
        ].set_index("method")
        comparison_rows.append(
            {
                "variant": variant,
                "target": target,
                "unaugmented_r2": float(lookup.loc["unaugmented", "r2"]),
                "flip_augmented_r2": float(lookup.loc["flip_augmented", "r2"]),
                "r2_change": float(
                    lookup.loc["flip_augmented", "r2"]
                    - lookup.loc["unaugmented", "r2"]
                ),
                **result,
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    paths = {
        "predictions": ROOT / str(config["outputs"]["predictions"]),
        "fold_metrics": ROOT / str(config["outputs"]["fold_metrics"]),
        "macro_metrics": ROOT / str(config["outputs"]["macro_metrics"]),
        "comparison": ROOT / str(config["outputs"]["comparison"]),
        "figure": ROOT / str(config["outputs"]["figure"]),
        "report": ROOT / str(config["outputs"]["report"]),
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(paths["predictions"], index=False, compression="zstd")
    metrics.to_csv(paths["fold_metrics"], index=False)
    macro.to_csv(paths["macro_metrics"], index=False)
    comparison.to_csv(paths["comparison"], index=False)
    make_figure(macro, paths["figure"])
    clear_better = comparison[comparison.ci_high < 0]
    clear_worse = comparison[comparison.ci_low > 0]
    lines = [
        "# Phase 33 reflection-augmentation sensitivity",
        "",
        "Horizontal and vertical reflections were applied to the signed gradient summaries during training. Validation and test predictions average all four reflections.",
        "",
        "Negative RMSE differences favour reflection augmentation.",
        "",
        "```text",
        comparison.to_string(index=False, float_format=lambda value: f"{value:.5f}"),
        "```",
        "",
        f"Clear improvements: {len(clear_better)}.",
        f"Clear degradations: {len(clear_worse)}.",
    ]
    paths["report"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    freeze = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": sha256(CONFIG_PATH),
        "baseline_result_freeze_sha256": sha256(
            ROOT / str(config["frozen_inputs"]["phase29_result_freeze"])
        ),
        "artifacts": {
            str(path.relative_to(ROOT)): sha256(path) for path in paths.values()
        },
    }
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    freeze_path.write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
    print(paths["report"].read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    run()

