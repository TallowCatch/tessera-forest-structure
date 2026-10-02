#!/usr/bin/env python3
"""Aggregate the final v2 U-Net and compare it with the frozen v2 MLP."""

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

from aggregate_cairngorms_height_adjusted import (
    bootstrap_difference,
    weighted_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/cairngorms_v2_unet.yaml"


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "phase28_cairngorms_v2_unet"
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


def make_figure(macro: pd.DataFrame, config: dict[str, Any], path: Path) -> None:
    target_config = config["targets"]
    targets = [
        str(target_config["confirmatory"]),
        str(target_config["primary"]),
        *[str(value) for value in target_config["secondary"]],
    ]
    labels = {
        "lidar_canopy_shannon_50m": "Shannon\n(raw)",
        "lidar_canopy_shannon_50m_height_adjusted": "Shannon\n(height-adjusted)",
        "lidar_height_cv_50m": "Canopy-height CV\n(raw)",
        "lidar_height_cv_50m_height_adjusted": "Canopy-height CV\n(height-adjusted)",
    }
    models = ["tessera_v2", "tessera_v2_unet"]
    names = {"tessera_v2": "v2 5 x 5 MLP", "tessera_v2_unet": "v2 U-Net"}
    colors = {"tessera_v2": "#707780", "tessera_v2_unet": "#168f8b"}
    x = np.arange(len(targets))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), constrained_layout=True)
    for model_index, model in enumerate(models):
        subset = macro[macro["model"] == model].set_index("target")
        offset = (model_index - 0.5) * width
        axes[0].bar(
            x + offset,
            [float(subset.loc[target, "rmse"]) for target in targets],
            width,
            color=colors[model],
            label=names[model],
        )
        axes[1].bar(
            x + offset,
            [float(subset.loc[target, "r2"]) for target in targets],
            width,
            color=colors[model],
            label=names[model],
        )
    for axis, ylabel, panel in zip(
        axes,
        ["Mean held-out RMSE", r"Mean held-out $R^2$"],
        ["a  Absolute error", "b  Explained variation"],
    ):
        axis.set_xticks(x, [labels[target] for target in targets])
        axis.set_ylabel(ylabel)
        axis.set_title(panel, loc="left", fontweight="bold")
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(1.05, 1.20), ncol=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def run() -> None:
    config = load_config()
    outputs = config["outputs"]
    frozen = config["frozen_inputs"]
    evaluation = config["evaluation"]
    target_names = [str(value) for value in config["targets"]["names"]]
    result_dir = ROOT / str(outputs["result_directory"])
    array_dir = ROOT / str(outputs["array_directory"])
    row_ids = np.load(array_dir / "row_id.npy")
    blocks = np.load(array_dir / "spatial_block.npy")
    seeds = [int(value) for value in evaluation["seeds"]]
    parts: list[pd.DataFrame] = []
    for fold in range(int(evaluation["folds"])):
        seed_predictions: list[np.ndarray] = []
        test = None
        observed = None
        for seed in seeds:
            path = result_dir / f"tessera_v2_unet_fold_{fold}_seed_{seed}.npz"
            if not path.exists():
                raise RuntimeError(f"Missing final v2 U-Net result: {path}")
            with np.load(path) as values:
                current_test = values["test_indices"].astype(np.int64)
                current_observed = values["observed"].astype(np.float32)
                if test is not None and not np.array_equal(test, current_test):
                    raise RuntimeError(f"Test indices changed among seeds in fold {fold}")
                if observed is not None and not np.allclose(
                    observed, current_observed, rtol=0, atol=1e-7
                ):
                    raise RuntimeError(f"Observed targets changed among seeds in fold {fold}")
                test = current_test
                observed = current_observed
                seed_predictions.append(values["predictions"].astype(np.float32))
        assert test is not None and observed is not None
        ensemble = np.mean(seed_predictions, axis=0)
        for column, target in enumerate(target_names):
            parts.append(
                pd.DataFrame(
                    {
                        "row_id": row_ids[test],
                        "fold": fold,
                        "model": "tessera_v2_unet",
                        "target": target,
                        "observed": observed[:, column],
                        "predicted": ensemble[:, column],
                        "spatial_block": blocks[test],
                    }
                )
            )
    unet = pd.concat(parts, ignore_index=True)
    baseline = pd.read_parquet(ROOT / str(frozen["corrected_predictions"]))
    baseline = baseline[baseline["model"] == str(evaluation["comparison_model"])].copy()
    baseline = baseline[baseline["target"].isin(target_names)].copy()
    expected = len(row_ids) * len(target_names)
    if len(unet) != expected or len(baseline) != expected:
        raise RuntimeError(
            f"Incomplete comparison: U-Net={len(unet)}, baseline={len(baseline)}, expected={expected}"
        )
    comparison_keys = ["row_id", "fold", "target"]
    checked = unet.merge(
        baseline[comparison_keys + ["observed"]],
        on=comparison_keys,
        suffixes=("_unet", "_baseline"),
        validate="one_to_one",
    )
    if not np.allclose(
        checked["observed_unet"], checked["observed_baseline"], rtol=0, atol=1e-6
    ):
        raise RuntimeError("The U-Net and frozen v2 MLP do not use identical targets")
    predictions = pd.concat([baseline, unet], ignore_index=True)

    metric_rows: list[dict[str, Any]] = []
    for (fold, model, target), group in predictions.groupby(
        ["fold", "model", "target"], sort=True
    ):
        metric_rows.append(
            {
                "fold": int(fold),
                "model": str(model),
                "target": str(target),
                **weighted_metrics(
                    group["observed"].to_numpy(),
                    group["predicted"].to_numpy(),
                    group["spatial_block"].to_numpy(),
                ),
            }
        )
    metrics = pd.DataFrame(metric_rows)
    macro = metrics.groupby(["model", "target"], as_index=False).agg(
        folds=("fold", "nunique"),
        n=("n", "sum"),
        blocks=("blocks", "sum"),
        rmse=("rmse", "mean"),
        r2=("r2", "mean"),
        block_spearman=("block_spearman", "mean"),
        positive_r2_folds=("r2", lambda values: int((values > 0).sum())),
    )
    reported_targets = [
        str(config["targets"]["primary"]),
        str(config["targets"]["confirmatory"]),
        *[str(value) for value in config["targets"]["secondary"]],
    ]
    rng = np.random.default_rng(int(evaluation["bootstrap_seed"]))
    rows: list[dict[str, Any]] = []
    for target in reported_targets:
        local = predictions[predictions["target"] == target]
        result = bootstrap_difference(
            local,
            "tessera_v2_unet",
            str(evaluation["comparison_model"]),
            int(evaluation["bootstrap_replicates"]),
            rng,
        )
        base_rmse = float(
            macro[
                (macro["model"] == str(evaluation["comparison_model"]))
                & (macro["target"] == target)
            ]["rmse"].iloc[0]
        )
        unet_rmse = float(
            macro[
                (macro["model"] == "tessera_v2_unet") & (macro["target"] == target)
            ]["rmse"].iloc[0]
        )
        rows.append(
            {
                "target": target,
                "baseline_rmse": base_rmse,
                "unet_rmse": unet_rmse,
                "rmse_reduction_fraction": (base_rmse - unet_rmse) / base_rmse,
                **result,
            }
        )
    comparison = pd.DataFrame(rows)
    primary = comparison[
        comparison["target"] == str(config["targets"]["primary"])
    ].iloc[0]
    success = config["evaluation"]["success"]
    reduction_pass = float(primary["rmse_reduction_fraction"]) >= float(
        success["minimum_primary_rmse_reduction_fraction"]
    )
    interval_pass = float(primary["ci_high"]) < 0
    passed = reduction_pass and (
        interval_pass if bool(success["require_bootstrap_ci_below_zero"]) else True
    )

    paths = {
        name: ROOT / str(outputs[name])
        for name in [
            "predictions",
            "fold_metrics",
            "macro_metrics",
            "comparison",
            "figure",
            "report",
            "result_freeze",
        ]
    }
    atomic(paths["predictions"], lambda p: predictions.to_parquet(p, index=False))
    atomic(paths["fold_metrics"], lambda p: metrics.to_csv(p, index=False))
    atomic(paths["macro_metrics"], lambda p: macro.to_csv(p, index=False))
    atomic(paths["comparison"], lambda p: comparison.to_csv(p, index=False))
    make_figure(macro, config, paths["figure"])
    lines = [
        "# Final Cairngorms TESSERA v2 U-Net rerun",
        "",
        "The corrected LiDAR cohort, dispersed five-fold spatial test, 2 km exclusion buffers, three seeds and eight outcomes were retained. The comparison changes only the spatial model: the frozen v2 5 x 5 summary MLP versus the frozen two-level U-Net.",
        "",
        "## Pre-declared decision",
        "",
        f"Primary target: `{config['targets']['primary']}`.",
        f"Decision: **{'PASS' if passed else 'NO PASS'}**.",
        f"RMSE reduction: {float(primary['rmse_reduction_fraction']):.1%}; paired 95% interval for U-Net minus MLP RMSE: [{float(primary['ci_low']):.4f}, {float(primary['ci_high']):.4f}].",
        "",
        "## Reported comparisons",
        "",
        "```text",
        comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "```",
        "",
        "## All outcomes",
        "",
        "```text",
        macro.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "```",
        "",
    ]
    atomic(paths["report"], lambda p: p.write_text("\n".join(lines), encoding="utf-8"))
    freeze = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "primary_target": str(config["targets"]["primary"]),
        "decision": "pass" if passed else "no_pass",
        "checks": {
            "minimum_rmse_reduction_pass": reduction_pass,
            "bootstrap_ci_below_zero_pass": interval_pass,
        },
        "primary_result": primary.to_dict(),
        "config_sha256": sha256(CONFIG_PATH),
        "artifacts": {
            str(path.relative_to(ROOT)): sha256(path)
            for name, path in paths.items()
            if name != "result_freeze"
        },
    }
    atomic(
        paths["result_freeze"],
        lambda p: p.write_text(
            json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        ),
    )
    print(json.dumps(freeze, indent=2), flush=True)


if __name__ == "__main__":
    run()
