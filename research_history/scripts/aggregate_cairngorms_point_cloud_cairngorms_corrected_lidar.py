#!/usr/bin/env python3
"""Aggregate the corrected Cairngorms point-cloud metric evaluation."""

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
CONFIG_PATH = ROOT / "configs/cairngorms_point_cloud_cairngorms_corrected_lidar.yaml"


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "phase25_cairngorms_corrected_lidar"
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_figure(macro: pd.DataFrame, config: dict[str, Any], path: Path) -> None:
    raw = [str(value) for value in config["targets"]["primary"]]
    suffix = str(config["targets"]["adjusted_suffix"])
    labels = {
        "lidar_height_cv_50m": "Height CV",
        "lidar_rcv_50m": "Robust CV",
        "lidar_rms_50m": "Height RMS",
        "lidar_canopy_shannon_50m": "Shannon",
    }
    models = ["conventional", "tessera", "fused"]
    colors = {"conventional": "#4c78a8", "tessera": "#15958f", "fused": "#e68632"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for axis, targets, title in [
        (axes[0], raw, "Corrected point-cloud outcomes"),
        (axes[1], [f"{name}{suffix}" for name in raw], "After accounting for canopy height"),
    ]:
        x = np.arange(len(targets))
        width = 0.24
        for model_index, model in enumerate(models):
            values = [
                float(
                    macro[(macro["model"] == model) & (macro["target"] == target)][
                        "r2"
                    ].iloc[0]
                )
                for target in targets
            ]
            axis.bar(
                x + (model_index - 1) * width,
                values,
                width,
                color=colors[model],
                label=model,
            )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(
            x,
            [labels[target.removesuffix(suffix)] for target in targets],
            rotation=24,
            ha="right",
        )
        axis.set_ylabel(r"Mean held-out $R^2$")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, loc="upper left", ncol=3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run() -> None:
    config = load_config()
    outputs = config["outputs"]
    array_dir = ROOT / str(outputs["array_directory"])
    result_dir = ROOT / str(outputs["result_directory"])
    raw_targets = [str(value) for value in config["targets"]["primary"]]
    suffix = str(config["targets"]["adjusted_suffix"])
    target_names = raw_targets + [f"{value}{suffix}" for value in raw_targets]
    models = [str(value) for value in config["evaluation"]["models"]]
    seeds = [int(value) for value in config["evaluation"]["seeds"]]
    blocks = np.load(array_dir / "spatial_block.npy")
    row_ids = np.load(array_dir / "row_id.npy")
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    for fold in range(int(config["evaluation"]["folds"])):
        for model in models:
            seed_predictions = []
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
                    seed_predictions.append(values["predictions"].astype(np.float32))
            assert test is not None and observed is not None
            ensemble = np.mean(seed_predictions, axis=0)
            for column, target in enumerate(target_names):
                metrics = weighted_metrics(observed[:, column], ensemble[:, column], blocks[test])
                metric_rows.append({"fold": fold, "model": model, "target": target, **metrics})
                prediction_rows.append(
                    pd.DataFrame(
                        {
                            "row_id": row_ids[test],
                            "fold": fold,
                            "model": model,
                            "target": target,
                            "observed": observed[:, column],
                            "predicted": ensemble[:, column],
                            "spatial_block": blocks[test],
                        }
                    )
                )
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    macro = metrics.groupby(["model", "target"], as_index=False).agg(
        folds=("fold", "nunique"),
        n=("n", "sum"),
        blocks=("blocks", "sum"),
        rmse=("rmse", "mean"),
        r2=("r2", "mean"),
        block_spearman=("block_spearman", "mean"),
        positive_r2_folds=("r2", lambda values: int((values > 0).sum())),
    )
    rng = np.random.default_rng(int(config["evaluation"]["bootstrap_seed"]))
    bootstrap_rows = []
    for target in target_names:
        local = predictions[predictions["target"] == target]
        for first, second in config["evaluation"]["comparisons"]:
            result = bootstrap_difference(
                local,
                str(first),
                str(second),
                int(config["evaluation"]["bootstrap_replicates"]),
                rng,
            )
            bootstrap_rows.append(
                {
                    "target": target,
                    "first_model": first,
                    "second_model": second,
                    **result,
                }
            )
    bootstrap = pd.DataFrame(bootstrap_rows)

    paths = {
        "predictions": ROOT / str(outputs["prediction_table"]),
        "metrics": ROOT / str(outputs["fold_metrics_table"]),
        "macro": ROOT / str(outputs["macro_table"]),
        "bootstrap": ROOT / str(outputs["bootstrap_table"]),
        "figure": ROOT / str(outputs["figure"]),
        "report": ROOT / str(outputs["report"]),
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(paths["predictions"], index=False, compression="zstd")
    metrics.to_csv(paths["metrics"], index=False)
    macro.to_csv(paths["macro"], index=False)
    bootstrap.to_csv(paths["bootstrap"], index=False)
    make_figure(macro, config, paths["figure"])

    raw_table = macro[macro["target"].isin(raw_targets)].copy()
    adjusted_table = macro[macro["target"].str.endswith(suffix)].copy()
    lines = [
        "# Corrected Cairngorms point-cloud structure results",
        "",
        "Aland's replacement raster applies the greater-than-1.3 m return filter to CV, robust CV, RMS and normalized Shannon. The selected 5 x 5 MLP and dispersed Phase 22 spatial folds were reused without further model selection.",
        "",
        "## Raw outcomes",
        "",
        "```text",
        raw_table.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "```",
        "",
        "## Outcomes after accounting for mean and p95 height",
        "",
        "```text",
        adjusted_table.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "```",
        "",
        "## Scope",
        "",
        "The targets are averages of corrected 10 m point-cloud metrics across each 50 m mature-forest unit. The result is a within-Cairngorms spatial confirmation, not an external geographic-transfer test.",
    ]
    paths["report"].write_text("\n".join(lines) + "\n", encoding="utf-8")
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
