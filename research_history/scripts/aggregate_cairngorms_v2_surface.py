#!/usr/bin/env python3
"""Aggregate the frozen v1-v2 Cairngorms canopy-surface comparison."""

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
CONFIG_PATH = ROOT / "configs/cairngorms_v2_surface.yaml"


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "phase29_cairngorms_v2_surface"
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def weighted_metrics(
    observed: np.ndarray, predicted: np.ndarray, block: np.ndarray
) -> dict[str, float]:
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
    grouped = pd.DataFrame(
        {"observed": observed, "predicted": predicted, "block": block}
    ).groupby("block").mean()
    return {
        "n": int(len(observed)),
        "blocks": int(len(grouped)),
        "rmse": float(np.sqrt(np.sum(weights * np.square(error)))),
        "r2": (
            float(1 - np.sum(weights * np.square(error)) / denominator)
            if denominator > 0
            else np.nan
        ),
        "block_spearman": float(
            spearmanr(grouped["observed"], grouped["predicted"]).statistic
        ),
    }


def aggregate_results(
    result_directory: Path,
    array_directory: Path,
    variant: str,
    target_names: list[str],
    seeds: list[int],
    model_names: dict[str, str],
) -> pd.DataFrame:
    row_ids = np.load(array_directory / "row_id.npy", mmap_mode="r")
    blocks = np.load(array_directory / "spatial_block.npy", mmap_mode="r")
    rows: list[pd.DataFrame] = []
    for fold in range(5):
        for family, model in model_names.items():
            predictions: list[np.ndarray] = []
            observed = None
            test = None
            for seed in seeds:
                if variant == "raw":
                    path = result_directory / (
                        f"balanced_{model}_fold{fold}_seed{seed}.npz"
                    )
                else:
                    path = result_directory / f"{model}_fold_{fold}_seed_{seed}.npz"
                if not path.exists():
                    raise RuntimeError(f"Missing Phase 29 result: {path}")
                with np.load(path) as values:
                    current_test = values["test_indices"].astype(np.int64)
                    if test is not None and not np.array_equal(test, current_test):
                        raise RuntimeError(f"Test rows changed in {path}")
                    test = current_test
                    current_observed = values["observed"].astype(np.float32)
                    if observed is not None and not np.allclose(
                        observed, current_observed, equal_nan=True
                    ):
                        raise RuntimeError(f"Observed targets changed in {path}")
                    observed = current_observed
                    predictions.append(values["predictions"].astype(np.float32))
            assert test is not None and observed is not None
            ensemble = np.mean(predictions, axis=0)
            for column, target in enumerate(target_names):
                rows.append(
                    pd.DataFrame(
                        {
                            "row_id": row_ids[test],
                            "variant": variant,
                            "version": "v2",
                            "model_family": family,
                            "fold": fold,
                            "target": target,
                            "observed": observed[:, column],
                            "predicted": ensemble[:, column],
                            "spatial_block": blocks[test],
                        }
                    )
                )
    return pd.concat(rows, ignore_index=True)


def load_v1_predictions(config: dict[str, Any]) -> pd.DataFrame:
    frozen = config["frozen_inputs"]
    raw = pd.read_parquet(ROOT / str(frozen["raw_v1_predictions"]))
    raw = raw[
        (raw["scheme"] == "balanced")
        & raw["model"].isin(["tessera_mlp_5x5", "fused_mlp_5x5"])
        & raw["target"].isin(config["targets"]["raw"])
    ].copy()
    raw["variant"] = "raw"
    raw["version"] = "v1"
    raw["model_family"] = raw["model"].map(
        {"tessera_mlp_5x5": "tessera", "fused_mlp_5x5": "fused"}
    )
    adjusted = pd.read_parquet(ROOT / str(frozen["adjusted_v1_predictions"]))
    adjusted = adjusted[
        adjusted["model"].isin(["tessera", "fused"])
        & adjusted["target"].isin(config["targets"]["height_adjusted"])
    ].copy()
    adjusted["variant"] = "height_adjusted"
    adjusted["version"] = "v1"
    adjusted["model_family"] = adjusted["model"]
    columns = [
        "row_id", "variant", "version", "model_family", "fold", "target",
        "observed", "predicted", "spatial_block",
    ]
    return pd.concat([raw[columns], adjusted[columns]], ignore_index=True)


def paired_version_difference(
    predictions: pd.DataFrame,
    replicates: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    squared = predictions.assign(
        squared_error=np.square(predictions["predicted"] - predictions["observed"])
    )
    grouped = squared.groupby(
        ["fold", "spatial_block", "version"], as_index=False
    )["squared_error"].mean()
    wide = grouped.pivot(
        index=["fold", "spatial_block"], columns="version", values="squared_error"
    ).dropna(subset=["v1", "v2"])
    point_fold = wide.groupby(level="fold")[["v1", "v2"]].mean().pow(0.5)
    point = float((point_fold["v2"] - point_fold["v1"]).mean())
    samples = np.empty(replicates, dtype=np.float64)
    folds = sorted(wide.index.get_level_values("fold").unique())
    for replicate in range(replicates):
        differences = []
        for fold in folds:
            local = wide.xs(fold, level="fold")
            chosen = rng.integers(0, len(local), len(local))
            values = local.iloc[chosen][["v1", "v2"]].mean().pow(0.5)
            differences.append(float(values["v2"] - values["v1"]))
        samples[replicate] = np.mean(differences)
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "v2_minus_v1_rmse": point,
        "ci_low": float(low),
        "ci_high": float(high),
        "probability_v2_better": float(np.mean(samples < 0)),
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
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for axis, variant, title in [
        (axes[0], "raw", "Original canopy-surface outcomes"),
        (axes[1], "height_adjusted", "After accounting for canopy height"),
    ]:
        local = macro[
            (macro["variant"] == variant) & (macro["model_family"] == "tessera")
        ]
        targets = list(local["target"].drop_duplicates())
        x = np.arange(len(targets))
        for offset, version, color in [(-0.18, "v1", "#148F88"), (0.18, "v2", "#E8872D")]:
            values = [
                float(local[(local.target == target) & (local.version == version)].r2.iloc[0])
                for target in targets
            ]
            axis.bar(x + offset, values, 0.36, label=f"TESSERA {version}", color=color)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(x, [labels[target] for target in targets], rotation=25, ha="right")
        axis.set_ylabel(r"Mean spatially held-out $R^2$")
        axis.set_title(title, loc="left", fontweight="bold")
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, ncol=2, loc="upper left")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run() -> None:
    config = load_config()
    outputs = config["outputs"]
    seeds = [int(value) for value in config["evaluation"]["seeds"]]
    v2_raw = aggregate_results(
        ROOT / str(outputs["raw_result_directory"]),
        ROOT / str(outputs["raw_array_directory"]),
        "raw",
        [str(value) for value in config["targets"]["raw"]],
        seeds,
        {"tessera": "tessera_mlp_5x5", "fused": "fused_mlp_5x5"},
    )
    v2_adjusted = aggregate_results(
        ROOT / str(outputs["adjusted_result_directory"]),
        ROOT / str(outputs["adjusted_array_directory"]),
        "height_adjusted",
        [str(value) for value in config["targets"]["height_adjusted"]],
        seeds,
        {"tessera": "tessera", "fused": "fused"},
    )
    predictions = pd.concat(
        [load_v1_predictions(config), v2_raw, v2_adjusted], ignore_index=True
    )

    metric_rows: list[dict[str, Any]] = []
    for keys, local in predictions.groupby(
        ["variant", "version", "model_family", "target", "fold"], sort=False
    ):
        variant, version, family, target, fold = keys
        values = weighted_metrics(
            local["observed"].to_numpy(),
            local["predicted"].to_numpy(),
            local["spatial_block"].to_numpy(),
        )
        metric_rows.append(
            {
                "variant": variant,
                "version": version,
                "model_family": family,
                "target": target,
                "fold": fold,
                **values,
            }
        )
    metrics = pd.DataFrame(metric_rows)
    macro = metrics.groupby(
        ["variant", "version", "model_family", "target"], as_index=False
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
    comparisons: list[dict[str, Any]] = []
    groups = ["variant", "model_family", "target"]
    for keys, local in predictions.groupby(groups, sort=False):
        variant, family, target = keys
        result = paired_version_difference(
            local,
            int(config["evaluation"]["bootstrap_replicates"]),
            rng,
        )
        lookup = macro[
            (macro.variant == variant)
            & (macro.model_family == family)
            & (macro.target == target)
        ].set_index("version")
        comparisons.append(
            {
                "variant": variant,
                "model_family": family,
                "target": target,
                "v1_r2": float(lookup.loc["v1", "r2"]),
                "v2_r2": float(lookup.loc["v2", "r2"]),
                "r2_change": float(lookup.loc["v2", "r2"] - lookup.loc["v1", "r2"]),
                **result,
            }
        )
    comparison = pd.DataFrame(comparisons)

    paths = {
        "predictions": ROOT / str(outputs["predictions"]),
        "metrics": ROOT / str(outputs["fold_metrics"]),
        "macro": ROOT / str(outputs["macro_metrics"]),
        "comparison": ROOT / str(outputs["comparison"]),
        "figure": ROOT / str(outputs["figure"]),
        "report": ROOT / str(outputs["report"]),
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(paths["predictions"], index=False, compression="zstd")
    metrics.to_csv(paths["metrics"], index=False)
    macro.to_csv(paths["macro"], index=False)
    comparison.to_csv(paths["comparison"], index=False)
    make_figure(macro, paths["figure"])

    tessera = comparison[comparison.model_family == "tessera"].copy()
    clear_better = tessera[tessera.ci_high < 0]
    clear_worse = tessera[tessera.ci_low > 0]
    lines = [
        "# Phase 29 frozen TESSERA v2 canopy-surface rerun",
        "",
        "The raw Phase 22 and height-adjusted Phase 23 tasks were rerun without changing targets, folds, buffers, seeds, architecture or hyperparameters.",
        "",
        "## TESSERA-only v1-v2 comparison",
        "",
        "Negative RMSE differences favour v2.",
        "",
        "```text",
        tessera.to_string(index=False, float_format=lambda value: f"{value:.5f}"),
        "```",
        "",
        f"Clear v2 improvements: {len(clear_better)}.",
        f"Clear v2 degradations: {len(clear_worse)}.",
        "",
        "## Scope",
        "",
        "These outcomes describe the visible one-metre canopy-height surface. They are distinct from the corrected point-return outcomes used for the final Shannon analysis.",
    ]
    paths["report"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    freeze = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": sha256(CONFIG_PATH),
        "artifacts": {
            str(path.relative_to(ROOT)): sha256(path) for path in paths.values()
        },
    }
    freeze_path = ROOT / str(outputs["result_freeze"])
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n")
    print(json.dumps(freeze, indent=2), flush=True)


if __name__ == "__main__":
    run()

