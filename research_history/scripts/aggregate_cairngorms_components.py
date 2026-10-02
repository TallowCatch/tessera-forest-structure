#!/usr/bin/env python3
"""Aggregate Phase 20 seed ensembles, metrics, uncertainty, and report."""

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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/cairngorms_components.yaml"
ARRAY_DIR = ROOT / "data/interim/phase20_cairngorms_components"
RESULT_DIR = ROOT / "data/interim/phase20_cairngorms_component_results"
TARGET_PATH = ROOT / "data/processed/phase20_cairngorms_component_targets.parquet"
PREDICTIONS_PATH = ROOT / "data/processed/phase20_cairngorms_component_predictions.parquet"
METRICS_PATH = ROOT / "outputs/tables/phase20_cairngorms_component_metrics.csv"
MACRO_PATH = ROOT / "outputs/tables/phase20_cairngorms_component_macro_metrics.csv"
COMPARISON_PATH = ROOT / "outputs/tables/phase20_cairngorms_component_comparisons.csv"
FIGURE_PATH = ROOT / "outputs/figures/phase20_cairngorms_component_evaluation.png"
REPORT_PATH = ROOT / "outputs/reports/phase20_cairngorms_component_results.md"
FREEZE_PATH = ROOT / "metadata/phase20_cairngorms_result_freeze.json"


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "phase20_cairngorms_components"
    ]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_values(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(observed) & np.isfinite(predicted)
    observed = observed[valid]
    predicted = predicted[valid]
    return {
        "n": int(len(observed)),
        "rmse": float(np.sqrt(mean_squared_error(observed, predicted))),
        "mae": float(mean_absolute_error(observed, predicted)),
        "r2": float(r2_score(observed, predicted)),
        "spearman_r": float(spearmanr(observed, predicted).statistic),
        "bias": float(np.mean(predicted - observed)),
    }


def bootstrap_comparison(
    frame: pd.DataFrame,
    candidate: str,
    reference: str,
    replicates: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    pivot = frame.pivot(index=["row_id", "spatial_block", "observed"], columns="model", values="predicted").reset_index()
    pivot = pivot.dropna(subset=[candidate, reference])
    pivot["candidate_squared_error"] = np.square(pivot[candidate] - pivot["observed"])
    pivot["reference_squared_error"] = np.square(pivot[reference] - pivot["observed"])
    block_errors = pivot.groupby("spatial_block", sort=False)[
        ["candidate_squared_error", "reference_squared_error"]
    ].mean()
    candidate_error = block_errors["candidate_squared_error"].to_numpy(dtype=np.float64)
    reference_error = block_errors["reference_squared_error"].to_numpy(dtype=np.float64)
    point = float(np.sqrt(candidate_error.mean()) - np.sqrt(reference_error.mean()))
    sample_indices = rng.integers(
        0, len(block_errors), size=(replicates, len(block_errors))
    )
    draws = np.sqrt(candidate_error[sample_indices].mean(axis=1)) - np.sqrt(
        reference_error[sample_indices].mean(axis=1)
    )
    return {
        "rmse_difference": point,
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "probability_candidate_better": float(np.mean(draws < 0)),
        "blocks": int(len(block_errors)),
    }


def atomic_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.suffix == ".csv":
        frame.to_csv(temporary, index=False)
    else:
        frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def make_figure(macro: pd.DataFrame, target_names: list[str]) -> None:
    labels = {
        "canopy_surface_sd_m": "Surface SD",
        "canopy_surface_rcv": "Relative surface variation",
        "canopy_open_fraction": "Canopy opening",
        "mean_local_vci_2m": "Mean local VCI",
    }
    models = ["conventional", "tessera", "fused"]
    colors = {"conventional": "#377eb8", "tessera": "#15958f", "fused": "#e68632"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for axis, suffix, title in zip(axes, ["", "_height_adjusted"], ["Raw structural outcomes", "After canopy-height adjustment"]):
        x = np.arange(len(target_names))
        width = 0.24
        for model_index, model in enumerate(models):
            values = []
            for target in target_names:
                name = target + suffix
                match = macro[(macro["model"] == model) & (macro["target"] == name)]
                values.append(float(match["r2"].iloc[0]) if len(match) else np.nan)
            axis.bar(x + (model_index - 1) * width, values, width, label=model, color=colors[model])
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(x, [labels[target] for target in target_names], rotation=20, ha="right")
        axis.set_ylabel(r"Mean held-out $R^2$")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run() -> None:
    config = load_config()
    target_names = [str(value) for value in config["targets"]["primary"]]
    all_names = target_names + [f"{name}_height_adjusted" for name in target_names]
    models = [str(value) for value in config["evaluation"]["models"]]
    seeds = [int(value) for value in config["evaluation"]["seeds"]]
    folds = int(config["evaluation"]["folds"])
    cohort = pd.read_parquet(TARGET_PATH).set_index("row_id")

    prediction_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    for fold in range(folds):
        with np.load(ARRAY_DIR / f"fold_{fold}.npz") as fold_values:
            test = fold_values["test_indices"].astype(np.int64)
            height_predictions = fold_values["height_test_predictions"].astype(np.float32)
            observed_raw = fold_values["final_targets"][test, : len(target_names)].astype(np.float32)
        for model in models:
            seed_predictions: list[np.ndarray] = []
            observed: np.ndarray | None = None
            for seed in seeds:
                path = RESULT_DIR / f"{model}_fold_{fold}_seed_{seed}.npz"
                if not path.exists():
                    raise RuntimeError(f"Missing model result: {path}")
                with np.load(path) as values:
                    if not np.array_equal(test, values["test_indices"]):
                        raise RuntimeError(f"Test rows differ in {path}")
                    seed_predictions.append(values["predictions"].astype(np.float32))
                    observed = values["observed"].astype(np.float32)
            assert observed is not None
            ensemble = np.mean(seed_predictions, axis=0)
            for target_index, target in enumerate(all_names):
                metrics = metric_values(observed[:, target_index], ensemble[:, target_index])
                metric_rows.append({"fold": fold, "model": model, "target": target, **metrics})
                prediction_rows.append(pd.DataFrame({
                    "row_id": test,
                    "fold": fold,
                    "spatial_block": cohort.loc[test, "spatial_block"].to_numpy(),
                    "model": model,
                    "target": target,
                    "observed": observed[:, target_index],
                    "predicted": ensemble[:, target_index],
                }))
        for target_index, target in enumerate(target_names):
            metrics = metric_values(observed_raw[:, target_index], height_predictions[:, target_index])
            metric_rows.append({"fold": fold, "model": "height_only", "target": target, **metrics})
            prediction_rows.append(pd.DataFrame({
                "row_id": test,
                "fold": fold,
                "spatial_block": cohort.loc[test, "spatial_block"].to_numpy(),
                "model": "height_only",
                "target": target,
                "observed": observed_raw[:, target_index],
                "predicted": height_predictions[:, target_index],
            }))

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    macro = metrics.groupby(["model", "target"], as_index=False).agg(
        folds=("fold", "nunique"),
        n=("n", "sum"),
        rmse=("rmse", "mean"),
        mae=("mae", "mean"),
        r2=("r2", "mean"),
        spearman_r=("spearman_r", "mean"),
        bias=("bias", "mean"),
        positive_r2_folds=("r2", lambda values: int(np.sum(np.asarray(values) > 0))),
    )

    rng = np.random.default_rng(int(config["evaluation"]["bootstrap_seed"]))
    comparisons: list[dict[str, Any]] = []
    for target in all_names:
        subset = predictions[predictions["target"] == target]
        for candidate, reference in config["evaluation"]["comparisons"]:
            result = bootstrap_comparison(
                subset, str(candidate), str(reference),
                int(config["evaluation"]["bootstrap_replicates"]), rng,
            )
            comparisons.append({"target": target, "candidate": candidate, "reference": reference, **result})
    comparison_frame = pd.DataFrame(comparisons)

    atomic_table(metrics, METRICS_PATH)
    atomic_table(macro, MACRO_PATH)
    atomic_table(comparison_frame, COMPARISON_PATH)
    atomic_table(predictions, PREDICTIONS_PATH)
    make_figure(macro, target_names)

    vertical_name = str(config["interpretation_gates"]["vertical_claim_target"])
    vertical = macro[(macro["model"] == "tessera") & (macro["target"] == vertical_name)].iloc[0]
    vertical_comparison = comparison_frame[(comparison_frame["target"] == vertical_name) & (comparison_frame["candidate"] == "tessera")].iloc[0]
    supports_vertical = (
        int(vertical["positive_r2_folds"]) >= int(config["interpretation_gates"]["minimum_folds_with_positive_r2"])
        and float(vertical_comparison["ci_high"]) < 0
    )
    interpretation = (
        "The frozen criteria support a vertical-complexity interpretation."
        if supports_vertical
        else "The frozen criteria do not support a broad vertical-complexity claim; interpret positive results as canopy-surface or opening structure."
    )
    lines = [
        "# Cairngorms component-resolved heterogeneity results",
        "",
        f"Completed: {utc_now()}",
        "",
        interpretation,
        "",
        "## Equal-fold mean metrics",
        "",
        "```text",
        macro.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "```",
        "",
        "## Paired 1 km block-bootstrap comparisons",
        "",
        "Negative RMSE differences favour the candidate model.",
        "",
        "```text",
        comparison_frame.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "```",
        "",
        "## Scope",
        "",
        "The CHM-derived outcomes describe canopy-surface variation and openings. `mean_local_vci_2m` is the mean of an existing local LiDAR vertical-evenness metric; it is not a fine-bin return-height profile. Detailed foliage-height profiles still require the original point cloud or height histograms.",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    freeze = {
        "created_utc": utc_now(),
        "supports_vertical_complexity_claim": bool(supports_vertical),
        "interpretation": interpretation,
        "artifacts": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [TARGET_PATH, PREDICTIONS_PATH, METRICS_PATH, MACRO_PATH, COMPARISON_PATH, FIGURE_PATH, REPORT_PATH]
        },
    }
    FREEZE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FREEZE_PATH.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n")
    print(json.dumps(freeze, indent=2), flush=True)


if __name__ == "__main__":
    run()
