#!/usr/bin/env python3
"""Aggregate frozen Phase 22 Cairngorms spatial-model results."""

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
CONFIG_PATH = ROOT / "configs/cairngorms_spatial_unet.yaml"


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "phase22_cairngorms_spatial_unet"
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
    rmse = float(np.sqrt(np.sum(weights * np.square(error))))
    r2 = float(1 - np.sum(weights * np.square(error)) / denominator) if denominator > 0 else np.nan
    grouped = pd.DataFrame({"observed": observed, "predicted": predicted, "block": block}).groupby("block").mean()
    rank = float(spearmanr(grouped["observed"], grouped["predicted"]).statistic) if len(grouped) > 2 else np.nan
    return {"n": int(len(observed)), "blocks": int(len(grouped)), "rmse": rmse, "r2": r2, "block_spearman": rank}


def write_figure(macro: pd.DataFrame, config: dict[str, Any], path: Path) -> None:
    target_names = [*config["targets"]["primary"], *config["targets"]["secondary"]]
    label = {
        "canopy_p95_height_m": "p95 height",
        "canopy_surface_sd_m": "Height SD",
        "canopy_surface_cv": "Height CV",
        "canopy_surface_rcv": "Robust CV",
        "canopy_rumple": "Rumple",
        "canopy_open_fraction": "Openings",
        "canopy_height_kurtosis": "Kurtosis",
    }
    models = [
        "conventional_mlp", "tessera_mlp_3x3", "tessera_mlp_5x5",
        "tessera_cnn_5x5", "tessera_unet", "fused_mlp_5x5",
    ]
    colors = ["#377eb8", "#70b7a5", "#15958f", "#8b6bb8", "#d65f5f", "#e68632"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8.2), sharex=True, constrained_layout=True)
    width = 0.13
    x = np.arange(len(target_names))
    for axis, scheme in zip(axes, ["balanced", "regional"]):
        subset = macro[macro["scheme"] == scheme]
        for index, (model, color) in enumerate(zip(models, colors)):
            values = []
            for target in target_names:
                match = subset[(subset["model"] == model) & (subset["target"] == target)]
                values.append(float(match["r2"].iloc[0]) if len(match) else np.nan)
            axis.bar(x + (index - 2.5) * width, values, width, color=color, label=model.replace("_", " "))
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_ylabel(r"Mean held-out $R^2$")
        axis.set_title("Dispersed spatial blocks" if scheme == "balanced" else "Contiguous regional stress test")
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    axes[1].set_xticks(x, [label[name] for name in target_names], rotation=20, ha="right")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run() -> None:
    config = load_config()
    outputs = config["outputs"]
    array_dir = ROOT / str(outputs["array_directory"])
    result_dir = ROOT / str(outputs["result_directory"])
    cohort = pd.read_parquet(ROOT / str(outputs["target_table"])).reset_index(drop=True)
    targets = [*config["targets"]["primary"], *config["targets"]["secondary"]]
    models = [*config["models"]["scalar_names"], config["models"]["unet_name"]]
    seeds = [int(value) for value in config["models"]["seeds"]]
    schemes = [str(value) for value in config["spatial_evaluation"]["schemes"]]
    folds = int(config["spatial_evaluation"]["folds"])
    blocks = np.load(array_dir / "spatial_block.npy")
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    for scheme in schemes:
        for fold in range(folds):
            for model in models:
                predictions = []
                observed = None
                test = None
                for seed in seeds:
                    path = result_dir / f"{scheme}_{model}_fold{fold}_seed{seed}.npz"
                    if not path.exists():
                        raise RuntimeError(f"Missing result: {path}")
                    with np.load(path) as values:
                        current_test = values["test_indices"].astype(np.int64)
                        if test is not None and not np.array_equal(test, current_test):
                            raise RuntimeError(f"Test rows changed for {model} {scheme} fold {fold}")
                        test = current_test
                        observed = values["observed"].astype(np.float32)
                        predictions.append(values["predictions"].astype(np.float32))
                assert test is not None and observed is not None
                ensemble = np.mean(predictions, axis=0)
                for column, target in enumerate(targets):
                    result = weighted_metrics(observed[:, column], ensemble[:, column], blocks[test])
                    metric_rows.append({"scheme": scheme, "fold": fold, "model": model, "target": target, **result})
                    prediction_rows.append(pd.DataFrame({
                        "row_id": cohort.iloc[test]["row_id"].to_numpy(),
                        "scheme": scheme,
                        "fold": fold,
                        "model": model,
                        "target": target,
                        "observed": observed[:, column],
                        "predicted": ensemble[:, column],
                        "spatial_block": blocks[test],
                    }))
    metrics = pd.DataFrame(metric_rows)
    macro = metrics.groupby(["scheme", "model", "target"], as_index=False).agg(
        folds=("fold", "nunique"), n=("n", "sum"), blocks=("blocks", "sum"),
        rmse=("rmse", "mean"), r2=("r2", "mean"), block_spearman=("block_spearman", "mean"),
    )
    predictions = pd.concat(prediction_rows, ignore_index=True)
    paths = {
        "predictions": ROOT / str(outputs["prediction_table"]),
        "metrics": ROOT / str(outputs["metrics_table"]),
        "macro": ROOT / str(outputs["macro_table"]),
        "figure": ROOT / str(outputs["figure"]),
        "report": ROOT / str(outputs["report"]),
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(paths["predictions"], index=False, compression="zstd")
    metrics.to_csv(paths["metrics"], index=False)
    macro.to_csv(paths["macro"], index=False)
    write_figure(macro, config, paths["figure"])
    primary = macro[macro["scheme"] == config["spatial_evaluation"]["primary_scheme"]]
    lines = [
        "# Phase 22 Cairngorms spatial validation and U-Net experiment", "",
        "All targets were recomputed from the same 1 m canopy-height raster. The dispersed-block result is primary; the contiguous regional result is a secondary stress test.", "",
        "## Primary macro results", "", "```text",
        primary.to_string(index=False, float_format=lambda value: f"{value:.4f}"), "```", "",
        "## Interpretation limits", "",
        "This is spatial validation within one Cairngorms landscape. It does not establish transfer to a different forest or survey. Kurtosis is secondary. The run uses frozen 2023 TESSERA v1.0/vultr embeddings; a newer embedding release requires a separate complete rerun.",
    ]
    paths["report"].write_text("\n".join(lines) + "\n")
    freeze = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "primary_scheme": config["spatial_evaluation"]["primary_scheme"],
        "artifacts": {str(path.relative_to(ROOT)): sha256(path) for path in paths.values()},
    }
    freeze_path = ROOT / str(outputs["result_freeze"])
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n")
    print(json.dumps(freeze, indent=2), flush=True)


if __name__ == "__main__":
    run()
