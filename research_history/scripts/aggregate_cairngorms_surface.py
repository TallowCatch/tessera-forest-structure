#!/usr/bin/env python3
"""Aggregate the frozen Cairngorms canopy-surface experiment."""

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

from aggregate_cairngorms_components import (
    atomic_table,
    bootstrap_comparison,
    metric_values,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/cairngorms_surface.yaml"
ARRAY_DIR = ROOT / "data/interim/phase21_cairngorms_surface"
RESULT_DIR = ROOT / "data/interim/phase21_cairngorms_surface_results"
TARGET_PATH = ROOT / "data/processed/phase21_cairngorms_surface_targets.parquet"
PREDICTIONS_PATH = ROOT / "data/processed/phase21_cairngorms_surface_predictions.parquet"
METRICS_PATH = ROOT / "outputs/tables/phase21_cairngorms_surface_metrics.csv"
MACRO_PATH = ROOT / "outputs/tables/phase21_cairngorms_surface_macro_metrics.csv"
COMPARISON_PATH = ROOT / "outputs/tables/phase21_cairngorms_surface_comparisons.csv"
INVENTORY_PATH = ROOT / "outputs/tables/phase21_cairngorms_target_inventory.csv"
FIGURE_PATH = ROOT / "outputs/figures/phase21_cairngorms_surface_evaluation.png"
REPORT_PATH = ROOT / "outputs/reports/phase21_cairngorms_surface_results.md"
FREEZE_PATH = ROOT / "metadata/phase21_cairngorms_result_freeze.json"


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "phase21_cairngorms_surface"
    ]


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def target_inventory() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "target": "canopy_top_height_m",
                "unit": "m",
                "source": "2023 1 m CHM",
                "definition": "99.9th percentile of valid CHM values in the 50 m unit",
            },
            {
                "target": "canopy_surface_sd_m",
                "unit": "m",
                "source": "2023 1 m CHM",
                "definition": "standard deviation of valid CHM values in the 50 m unit",
            },
            {
                "target": "canopy_surface_cv",
                "unit": "ratio",
                "source": "supplied CHM 50 m product",
                "definition": "chm_height_cv; exact production definition awaiting confirmation",
            },
            {
                "target": "canopy_surface_rcv",
                "unit": "ratio",
                "source": "2023 1 m CHM",
                "definition": "interquartile range divided by median CHM height",
            },
            {
                "target": "canopy_rumple",
                "unit": "ratio",
                "source": "supplied CHM 50 m product",
                "definition": "chm_rumple_norm; exact production definition awaiting confirmation",
            },
            {
                "target": "canopy_open_fraction",
                "unit": "fraction",
                "source": "2023 1 m CHM",
                "definition": "fraction of valid CHM values below 2 m",
            },
        ]
    )


def make_figure(
    macro: pd.DataFrame, raw_names: list[str], adjusted_names: list[str]
) -> None:
    labels = {
        "canopy_top_height_m": "Top height",
        "canopy_surface_sd_m": "Surface SD",
        "canopy_surface_cv": "CV",
        "canopy_surface_rcv": "Robust CV",
        "canopy_rumple": "Rumple",
        "canopy_open_fraction": "Openings",
    }
    models = ["conventional", "tessera", "fused"]
    colors = {
        "conventional": "#377eb8",
        "tessera": "#15958f",
        "fused": "#e68632",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    panels = [
        (axes[0], raw_names, "Raw LiDAR outcomes"),
        (
            axes[1],
            [f"{name}_height_adjusted" for name in adjusted_names],
            "Variation remaining after height adjustment",
        ),
    ]
    width = 0.24
    for axis, names, title in panels:
        x = np.arange(len(names))
        for model_index, model in enumerate(models):
            values = []
            for name in names:
                match = macro[(macro["model"] == model) & (macro["target"] == name)]
                values.append(float(match["r2"].iloc[0]) if len(match) else np.nan)
            axis.bar(
                x + (model_index - 1) * width,
                values,
                width,
                label=model,
                color=colors[model],
            )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(
            x,
            [labels[name.replace("_height_adjusted", "")] for name in names],
            rotation=22,
            ha="right",
        )
        axis.set_ylabel(r"Mean held-out $R^2$")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run() -> None:
    config = load_config()
    raw_names = [str(value) for value in config["targets"]["primary"]]
    adjusted_base_names = [
        str(value) for value in config["targets"]["height_adjusted"]
    ]
    adjusted_names = [f"{value}_height_adjusted" for value in adjusted_base_names]
    all_names = raw_names + adjusted_names
    models = [str(value) for value in config["evaluation"]["models"]]
    seeds = [int(value) for value in config["evaluation"]["seeds"]]
    folds = int(config["evaluation"]["folds"])
    cohort = pd.read_parquet(TARGET_PATH).set_index("row_id")

    prediction_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    for fold in range(folds):
        with np.load(ARRAY_DIR / f"fold_{fold}.npz") as fold_values:
            test = fold_values["test_indices"].astype(np.int64)
            final_targets = fold_values["final_targets"].astype(np.float32)
            height_predictions = fold_values[
                "height_reference_predictions"
            ].astype(np.float32)
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
                metrics = metric_values(
                    observed[:, target_index], ensemble[:, target_index]
                )
                metric_rows.append(
                    {"fold": fold, "model": model, "target": target, **metrics}
                )
                prediction_rows.append(
                    pd.DataFrame(
                        {
                            "row_id": test,
                            "fold": fold,
                            "spatial_block": cohort.loc[
                                test, "spatial_block"
                            ].to_numpy(),
                            "model": model,
                            "target": target,
                            "observed": observed[:, target_index],
                            "predicted": ensemble[:, target_index],
                        }
                    )
                )
        for adjusted_index, target in enumerate(adjusted_base_names):
            raw_index = raw_names.index(target)
            observed_raw = final_targets[test, raw_index]
            metrics = metric_values(
                observed_raw, height_predictions[:, adjusted_index]
            )
            metric_rows.append(
                {
                    "fold": fold,
                    "model": "height_only",
                    "target": target,
                    **metrics,
                }
            )
            prediction_rows.append(
                pd.DataFrame(
                    {
                        "row_id": test,
                        "fold": fold,
                        "spatial_block": cohort.loc[
                            test, "spatial_block"
                        ].to_numpy(),
                        "model": "height_only",
                        "target": target,
                        "observed": observed_raw,
                        "predicted": height_predictions[:, adjusted_index],
                    }
                )
            )

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
        positive_r2_folds=(
            "r2",
            lambda values: int(np.sum(np.asarray(values) > 0)),
        ),
    )

    rng = np.random.default_rng(int(config["evaluation"]["bootstrap_seed"]))
    comparison_rows: list[dict[str, Any]] = []
    for target in all_names:
        subset = predictions[predictions["target"] == target]
        for candidate, reference in config["evaluation"]["comparisons"]:
            result = bootstrap_comparison(
                subset,
                str(candidate),
                str(reference),
                int(config["evaluation"]["bootstrap_replicates"]),
                rng,
            )
            comparison_rows.append(
                {
                    "target": target,
                    "candidate": candidate,
                    "reference": reference,
                    **result,
                }
            )
    comparisons = pd.DataFrame(comparison_rows)
    inventory = target_inventory()

    atomic_table(metrics, METRICS_PATH)
    atomic_table(macro, MACRO_PATH)
    atomic_table(comparisons, COMPARISON_PATH)
    atomic_table(inventory, INVENTORY_PATH)
    atomic_table(predictions, PREDICTIONS_PATH)
    make_figure(macro, raw_names, adjusted_base_names)

    gate = config["interpretation_gates"]
    surface_targets = [str(value) for value in gate["surface_targets"]]
    recovered: list[str] = []
    added_value: list[str] = []
    for target in surface_targets:
        row = macro[(macro["model"] == "tessera") & (macro["target"] == target)]
        if len(row) and int(row.iloc[0]["positive_r2_folds"]) >= int(
            gate["minimum_folds_with_positive_r2"]
        ):
            recovered.append(target)
        comparison = comparisons[
            (comparisons["target"] == target)
            & (comparisons["candidate"] == "tessera")
            & (comparisons["reference"] == "conventional")
        ]
        if len(comparison) and float(comparison.iloc[0]["ci_high"]) < 0:
            added_value.append(target)
    supports_recovery = len(recovered) >= int(
        gate["minimum_surface_targets_recovered"]
    )
    supports_added_value = len(added_value) >= int(
        gate["minimum_surface_targets_with_added_value"]
    )

    top_height = macro[
        (macro["model"] == "tessera")
        & (macro["target"] == "canopy_top_height_m")
    ].iloc[0]
    lines = [
        "# Cairngorms canopy height and surface heterogeneity results",
        "",
        f"Completed: {utc_now()}",
        "",
        "## Direct answer",
        "",
        (
            "The frozen spatial tests support recovery of canopy-surface heterogeneity."
            if supports_recovery
            else "The frozen spatial tests do not meet the pre-declared recovery rule for canopy-surface heterogeneity."
        ),
        (
            "TESSERA also meets the separate added-value rule against Sentinel-1, Sentinel-2 and terrain."
            if supports_added_value
            else "TESSERA does not meet the separate added-value rule against Sentinel-1, Sentinel-2 and terrain."
        ),
        "",
        f"TESSERA top-height RMSE was {float(top_height['rmse']):.3f} m with R-squared {float(top_height['r2']):.3f} and mean bias {float(top_height['bias']):.3f} m.",
        f"Surface targets satisfying the recovery rule: {', '.join(recovered) if recovered else 'none'}.",
        f"Surface targets with a paired RMSE interval favouring TESSERA: {', '.join(added_value) if added_value else 'none'}.",
        "",
        "## Equal-region mean metrics",
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
        comparisons.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "```",
        "",
        "## Limits",
        "",
        "This is a retrospective Cairngorms method-development analysis, not an independent geographic replication. The exact collaborator-side production definitions of `chm_height_cv` and `chm_rumple_norm` remain to be confirmed. Height-adjusted outcomes are secondary diagnostics; they remove the average training-region relationship with mean and p95 height but do not remove every possible effect of stand development or terrain.",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    freeze = {
        "created_utc": utc_now(),
        "supports_surface_recovery": bool(supports_recovery),
        "supports_added_value_over_conventional": bool(supports_added_value),
        "recovered_surface_targets": recovered,
        "surface_targets_with_added_value": added_value,
        "artifacts": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [
                TARGET_PATH,
                PREDICTIONS_PATH,
                METRICS_PATH,
                MACRO_PATH,
                COMPARISON_PATH,
                INVENTORY_PATH,
                FIGURE_PATH,
                REPORT_PATH,
            ]
        },
    }
    FREEZE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FREEZE_PATH.write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(freeze, indent=2), flush=True)


if __name__ == "__main__":
    run()
