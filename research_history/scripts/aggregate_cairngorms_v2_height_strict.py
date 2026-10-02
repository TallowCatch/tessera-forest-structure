#!/usr/bin/env python3
"""Aggregate the strict-ownership V2 canopy-height U-Net rerun."""

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

import aggregate_cairngorms_v2_height as phase30


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/cairngorms_v2_height_strict_unet.yaml"


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "phase31_cairngorms_v2_height_strict_unet"
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


def load_predictions(config: dict[str, Any]) -> pd.DataFrame:
    frozen = config["frozen_inputs"]
    previous = pd.read_parquet(ROOT / str(frozen["phase30_predictions"]))
    previous = previous[
        previous["model"].isin(
            ["tessera_v2_mlp_5x5", "tessera_v2_unet_audited"]
        )
    ].copy()
    array_dir = ROOT / str(frozen["phase30_array_directory"])
    result_dir = ROOT / str(config["outputs"]["result_directory"])
    row_ids = np.load(array_dir / "row_id.npy", mmap_mode="r")
    blocks = np.load(array_dir / "spatial_block.npy", mmap_mode="r")
    openings = np.load(array_dir / "opening_fraction.npy", mmap_mode="r")
    seeds = [int(value) for value in config["evaluation"]["seeds"]]
    target_names = [str(value) for value in config["targets"]["names"]]
    rows: list[pd.DataFrame] = []
    for fold in range(int(config["evaluation"]["folds"])):
        test: np.ndarray | None = None
        observed: np.ndarray | None = None
        predictions: list[np.ndarray] = []
        for seed in seeds:
            path = result_dir / f"tessera_v2_unet_strict_fold_{fold}_seed_{seed}.npz"
            if not path.exists():
                raise RuntimeError(f"Missing Phase 31 result: {path}")
            with np.load(path) as values:
                current_test = values["test_indices"].astype(np.int64)
                current_observed = values["observed"].astype(np.float32)
                if test is not None and not np.array_equal(test, current_test):
                    raise RuntimeError(f"Test rows changed between strict U-Net seeds: {path}")
                if observed is not None and not np.array_equal(observed, current_observed):
                    raise RuntimeError(f"Observed values changed between strict U-Net seeds: {path}")
                test = current_test
                observed = current_observed
                predictions.append(values["predictions"].astype(np.float32))
        assert test is not None and observed is not None
        ensemble = np.mean(predictions, axis=0)
        for column, target in enumerate(target_names):
            rows.append(
                pd.DataFrame(
                    {
                        "row_id": row_ids[test],
                        "model": "tessera_v2_unet_strict",
                        "fold": fold,
                        "target": target,
                        "observed": observed[:, column],
                        "predicted": ensemble[:, column],
                        "spatial_block": blocks[test],
                        "opening_fraction": openings[test],
                    }
                )
            )
    strict = pd.concat(rows, ignore_index=True)
    columns = [
        "row_id", "model", "fold", "target", "observed", "predicted",
        "spatial_block", "opening_fraction",
    ]
    return pd.concat([previous[columns], strict[columns]], ignore_index=True)


def metric_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for keys, local in predictions.groupby(["model", "target", "fold"], sort=False):
        model, target, fold = keys
        rows.append(
            {
                "model": model,
                "target": target,
                "fold": fold,
                **phase30.weighted_metrics(
                    local["observed"].to_numpy(),
                    local["predicted"].to_numpy(),
                    local["spatial_block"].to_numpy(),
                ),
            }
        )
    folds = pd.DataFrame(rows)
    macro = folds.groupby(["model", "target"], as_index=False).agg(
        folds=("fold", "nunique"),
        n=("n", "sum"),
        rmse=("rmse", "mean"),
        mae=("mae", "mean"),
        bias=("bias", "mean"),
        r2=("r2", "mean"),
        block_spearman=("block_spearman", "mean"),
        positive_r2_folds=("r2", lambda values: int((values > 0).sum())),
    )
    return folds, macro


def paired_difference(
    predictions: pd.DataFrame,
    target: str,
    reference: str,
    candidate: str,
    replicates: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    local = predictions[
        (predictions["target"] == target)
        & predictions["model"].isin([reference, candidate])
    ].copy()
    local["squared_error"] = np.square(local["predicted"] - local["observed"])
    grouped = local.groupby(
        ["fold", "spatial_block", "model"], as_index=False
    )["squared_error"].mean()
    wide = grouped.pivot(
        index=["fold", "spatial_block"], columns="model", values="squared_error"
    ).dropna(subset=[reference, candidate])
    folds = sorted(wide.index.get_level_values("fold").unique())

    def fold_difference(values: pd.DataFrame) -> float:
        differences = []
        for fold in folds:
            rmse = values.xs(fold, level="fold")[[reference, candidate]].mean().pow(0.5)
            differences.append(float(rmse[candidate] - rmse[reference]))
        return float(np.mean(differences))

    point = fold_difference(wide)
    samples = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        differences = []
        for fold in folds:
            local_fold = wide.xs(fold, level="fold")
            sampled = local_fold.iloc[rng.integers(0, len(local_fold), len(local_fold))]
            rmse = sampled[[reference, candidate]].mean().pow(0.5)
            differences.append(float(rmse[candidate] - rmse[reference]))
        samples[replicate] = np.mean(differences)
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "target": target,
        "reference": reference,
        "candidate": candidate,
        "candidate_minus_reference_rmse": point,
        "ci_low": float(low),
        "ci_high": float(high),
        "probability_candidate_better": float(np.mean(samples < 0)),
        "paired_blocks": int(len(wide)),
    }


def opening_diagnostic(predictions: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    local = predictions.copy()
    local["opening_group"] = phase30.opening_group(local["opening_fraction"], config)
    rows: list[dict[str, Any]] = []
    for keys, values in local.groupby(
        ["model", "target", "opening_group", "fold"], observed=True, sort=False
    ):
        model, target, opening, fold = keys
        rows.append(
            {
                "model": model,
                "target": target,
                "opening_group": str(opening),
                "fold": fold,
                **phase30.weighted_metrics(
                    values["observed"].to_numpy(),
                    values["predicted"].to_numpy(),
                    values["spatial_block"].to_numpy(),
                ),
            }
        )
    folds = pd.DataFrame(rows)
    return folds.groupby(["model", "target", "opening_group"], as_index=False).agg(
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
    models = [
        "tessera_v2_mlp_5x5",
        "tessera_v2_unet_audited",
        "tessera_v2_unet_strict",
    ]
    labels = {
        "tessera_v2_mlp_5x5": "V2 5 x 5 MLP",
        "tessera_v2_unet_audited": "Overlap-weighted U-Net",
        "tessera_v2_unet_strict": "Strict-ownership U-Net",
    }
    colors = {
        "tessera_v2_mlp_5x5": "#3F7F93",
        "tessera_v2_unet_audited": "#D8A03C",
        "tessera_v2_unet_strict": "#C4583C",
    }
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    strict = predictions[
        (predictions.target == primary) & (predictions.model == "tessera_v2_unet_strict")
    ]
    extent = [
        float(min(strict.observed.min(), strict.predicted.min())),
        float(max(strict.observed.max(), strict.predicted.max())),
    ]
    axes[0].hexbin(strict.observed, strict.predicted, gridsize=55, mincnt=1, bins="log", cmap="viridis")
    axes[0].plot(extent, extent, color="#C63D32", linestyle="--", linewidth=1.2)
    values = macro[(macro.target == primary) & (macro.model == "tessera_v2_unet_strict")].iloc[0]
    axes[0].text(
        0.04, 0.96, f"Mean $R^2$ = {values.r2:.3f}\nRMSE = {values.rmse:.3f} m",
        transform=axes[0].transAxes, va="top",
    )
    axes[0].set_xlabel("Observed p95 height (m)")
    axes[0].set_ylabel("Predicted p95 height (m)")
    axes[0].set_title("a  Strict U-Net predictions", loc="left", fontweight="bold")

    targets = ["canopy_mean_height_m", "canopy_p95_height_m"]
    x = np.arange(len(targets))
    for offset, model in zip([-0.25, 0.0, 0.25], models, strict=True):
        scores = [
            float(macro[(macro.model == model) & (macro.target == target)].rmse.iloc[0])
            for target in targets
        ]
        axes[1].bar(x + offset, scores, 0.24, color=colors[model], label=labels[model])
    axes[1].set_xticks(x, ["Mean height", "P95 height"])
    axes[1].set_ylabel("Mean spatially held-out RMSE (m)")
    axes[1].set_title("b  Architecture comparison", loc="left", fontweight="bold")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)

    groups = list(opening.opening_group.drop_duplicates())
    x = np.arange(len(groups))
    for offset, model in zip([-0.25, 0.0, 0.25], models, strict=True):
        local = opening[(opening.model == model) & (opening.target == primary)]
        scores = [float(local[local.opening_group == group].rmse.iloc[0]) for group in groups]
        axes[2].bar(x + offset, scores, 0.24, color=colors[model], label=labels[model])
    axes[2].set_xticks(x, groups)
    axes[2].set_xlabel("LiDAR canopy-opening fraction")
    axes[2].set_ylabel("P95 height RMSE (m)")
    axes[2].set_title("c  Error by canopy openness", loc="left", fontweight="bold")
    axes[2].grid(axis="y", alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run() -> None:
    config = load_config()
    outputs = config["outputs"]
    predictions = load_predictions(config)
    folds, macro = metric_tables(predictions)
    opening = opening_diagnostic(predictions, config)
    rng = np.random.default_rng(int(config["evaluation"]["bootstrap_seed"]))
    comparisons: list[dict[str, Any]] = []
    for target in config["targets"]["names"]:
        for reference in ["tessera_v2_mlp_5x5", "tessera_v2_unet_audited"]:
            comparisons.append(
                paired_difference(
                    predictions,
                    str(target),
                    reference,
                    "tessera_v2_unet_strict",
                    int(config["evaluation"]["bootstrap_replicates"]),
                    rng,
                )
            )
    comparison = pd.DataFrame(comparisons)
    comparison["rmse_reduction_fraction"] = [
        -float(row.candidate_minus_reference_rmse)
        / float(macro[(macro.model == row.reference) & (macro.target == row.target)].rmse.iloc[0])
        for row in comparison.itertuples()
    ]

    primary = str(config["targets"]["primary"])
    primary_row = comparison[
        (comparison.target == primary)
        & (comparison.reference == "tessera_v2_mlp_5x5")
    ].iloc[0]
    success_config = config["evaluation"]["success"]
    success = bool(
        float(primary_row.rmse_reduction_fraction)
        >= float(success_config["minimum_primary_rmse_reduction_fraction"])
        and (
            not bool(success_config["require_bootstrap_ci_below_zero"])
            or float(primary_row.ci_high) < 0
        )
    )

    for frame, name in [
        (predictions, "predictions"),
        (folds, "fold_metrics"),
        (macro, "macro_metrics"),
        (comparison, "comparison"),
        (opening, "opening_diagnostic"),
    ]:
        path = ROOT / str(outputs[name])
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".parquet":
            frame.to_parquet(path, index=False)
        else:
            frame.to_csv(path, index=False)
    make_figure(predictions, macro, opening, ROOT / str(outputs["figure"]))

    strict_primary = macro[
        (macro.model == "tessera_v2_unet_strict") & (macro.target == primary)
    ].iloc[0]
    old_primary = macro[
        (macro.model == "tessera_v2_unet_audited") & (macro.target == primary)
    ].iloc[0]
    mlp_primary = macro[
        (macro.model == "tessera_v2_mlp_5x5") & (macro.target == primary)
    ].iloc[0]
    report = "\n".join(
        [
            "# Phase 31: strict-ownership TESSERA v2 canopy-height rerun",
            "",
            "Every U-Net training target was assigned to exactly one best-centred chip",
            "and therefore contributed exactly once per epoch. The target rows, spatial",
            "folds, seeds and architecture were otherwise unchanged.",
            "",
            "## Equal-fold results",
            "",
            phase30.markdown_table(macro),
            "",
            "## Paired comparisons",
            "",
            phase30.markdown_table(comparison),
            "",
            "## Interpretation",
            "",
            f"The frozen p95 success criterion {'passed' if success else 'did not pass'}.",
            f"P95 RMSE was {mlp_primary.rmse:.3f} m for the fixed MLP, "
            f"{old_primary.rmse:.3f} m for the overlap-weighted U-Net and "
            f"{strict_primary.rmse:.3f} m for the strict-ownership U-Net.",
            "The strict result isolates the effect of removing duplicate target supervision",
            "from overlapping training chips.",
            "",
        ]
    )
    report_path = ROOT / str(outputs["report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    freeze_path = ROOT / str(outputs["result_freeze"])
    freeze = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "state": "complete",
        "primary_success": success,
        "config_sha256": sha256(CONFIG_PATH),
        "phase30_result_freeze_sha256": sha256(
            ROOT / str(config["frozen_inputs"]["phase30_result_freeze"])
        ),
        "artifacts": {
            name: {"path": str(outputs[name]), "sha256": sha256(ROOT / str(outputs[name]))}
            for name in [
                "predictions", "fold_metrics", "macro_metrics", "comparison",
                "opening_diagnostic", "figure", "report",
            ]
        },
    }
    atomic_json(freeze_path, freeze)
    print(report, flush=True)


if __name__ == "__main__":
    run()
