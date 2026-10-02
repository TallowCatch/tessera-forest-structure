#!/usr/bin/env python3
"""Aggregate ordered-patch folds and compare them with the frozen baseline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/cairngorms_ordered_patch.yaml"
RESULT_DIR = ROOT / "data/interim/cairngorms_ordered_patch_results"
PREDICTION_PATH = (
    ROOT / "data/processed/cairngorms_ordered_patch_oof_predictions.parquet"
)
METRIC_PATH = ROOT / "outputs/tables/cairngorms_ordered_patch_metrics.csv"
POOLED_PATH = (
    ROOT / "outputs/tables/cairngorms_ordered_patch_pooled_metrics.csv"
)
COMPARISON_PATH = (
    ROOT / "outputs/tables/cairngorms_ordered_patch_comparisons.csv"
)
FREEZE_PATH = ROOT / "metadata/cairngorms_ordered_patch_result_freeze.json"


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "cairngorms_ordered_patch"
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_record(
    observed: np.ndarray,
    predicted: np.ndarray,
    target: str,
    model: str,
    fold: int | str,
) -> dict[str, Any]:
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    residual = predicted - observed
    total_variation = float(np.square(observed - observed.mean()).sum())
    r2 = (
        1.0 - float(np.square(residual).sum()) / total_variation
        if total_variation > 0
        else np.nan
    )
    return {
        "target": target,
        "model": model,
        "fold": fold,
        "rows": len(observed),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "mae": float(np.mean(np.abs(residual))),
        "r2": r2,
        "pearson_r": float(pearsonr(observed, predicted).statistic),
        "spearman_r": float(spearmanr(observed, predicted).statistic),
        "bias": float(residual.mean()),
    }


def load_new_predictions(config: dict[str, Any]) -> pd.DataFrame:
    source = config["source"]
    array_directory = ROOT / source["array_directory"]
    row_ids = np.load(array_directory / "row_id.npy")
    targets = pd.read_parquet(ROOT / source["target_table"])[
        ["row_id", "spatial_block", "block_x", "block_y"]
    ]
    block_lookup = targets.set_index("row_id")
    target_names = list(config["evaluation"]["target_names"])
    records: list[pd.DataFrame] = []

    for model in config["models"]:
        for fold in config["evaluation"]["folds"]:
            seed_predictions: list[np.ndarray] = []
            test_indices: np.ndarray | None = None
            fold_targets: np.ndarray | None = None
            for seed in config["evaluation"]["seeds"]:
                path = RESULT_DIR / f"fold_{fold}_{model}_seed_{seed}.npz"
                manifest_path = path.with_suffix(".json")
                if not path.exists() or not manifest_path.exists():
                    raise RuntimeError(f"Missing ordered-patch result: {path.name}")
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("output_sha256") != sha256(path):
                    raise RuntimeError(f"Result checksum failed: {path.name}")
                with np.load(path) as result:
                    scoped_indices = result["test_indices"].astype(np.int64)
                    predictions = result["predictions"].astype(np.float64)
                if test_indices is None:
                    test_indices = scoped_indices
                elif not np.array_equal(test_indices, scoped_indices):
                    raise RuntimeError("Seed test indices do not agree")
                seed_predictions.append(predictions)

            fold_path = array_directory / f"fold_{fold}.npz"
            with np.load(fold_path) as values:
                fold_targets = values["targets"].astype(np.float64)[test_indices]
            averaged = np.mean(seed_predictions, axis=0)
            scoped_row_ids = row_ids[test_indices]
            scoped_blocks = block_lookup.loc[scoped_row_ids]
            for column, target in enumerate(target_names):
                records.append(
                    pd.DataFrame(
                        {
                            "row_id": scoped_row_ids,
                            "target": target,
                            "model": model,
                            "fold": fold,
                            "observed": fold_targets[:, column],
                            "predicted": averaged[:, column],
                            "spatial_block": scoped_blocks[
                                "spatial_block"
                            ].to_numpy(),
                            "block_x": scoped_blocks["block_x"].to_numpy(),
                            "block_y": scoped_blocks["block_y"].to_numpy(),
                        }
                    )
                )
    return pd.concat(records, ignore_index=True)


def calculate_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (target, model, fold), group in predictions.groupby(
        ["target", "model", "fold"], sort=True
    ):
        records.append(
            metric_record(
                group["observed"].to_numpy(),
                group["predicted"].to_numpy(),
                str(target),
                str(model),
                int(fold),
            )
        )
    for (target, model), group in predictions.groupby(
        ["target", "model"], sort=True
    ):
        records.append(
            metric_record(
                group["observed"].to_numpy(),
                group["predicted"].to_numpy(),
                str(target),
                str(model),
                "pooled",
            )
        )
    return pd.DataFrame(records)


def paired_block_bootstrap(
    merged: pd.DataFrame,
    replicates: int,
    seed: int,
) -> tuple[float, float, float]:
    grouped = (
        merged.assign(
            candidate_square=np.square(
                merged["candidate"] - merged["observed"]
            ),
            baseline_square=np.square(
                merged["baseline"] - merged["observed"]
            ),
        )
        .groupby("spatial_block", sort=False)
        .agg(
            candidate_square=("candidate_square", "sum"),
            baseline_square=("baseline_square", "sum"),
            rows=("row_id", "size"),
        )
    )
    values = grouped.to_numpy(dtype=np.float64)
    observed_delta = float(
        np.sqrt(values[:, 0].sum() / values[:, 2].sum())
        - np.sqrt(values[:, 1].sum() / values[:, 2].sum())
    )
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        selected = rng.integers(0, len(values), size=len(values))
        sample = values[selected]
        rows = sample[:, 2].sum()
        deltas[index] = np.sqrt(sample[:, 0].sum() / rows) - np.sqrt(
            sample[:, 1].sum() / rows
        )
    lower, upper = np.quantile(deltas, [0.025, 0.975])
    return observed_delta, float(lower), float(upper)


def comparisons(
    config: dict[str, Any], predictions: pd.DataFrame
) -> pd.DataFrame:
    source = config["source"]
    baseline_model = str(config["evaluation"]["baseline_model"])
    baseline = pd.read_parquet(ROOT / source["baseline_predictions"])
    baseline = baseline.loc[
        baseline["model"].eq(baseline_model),
        ["row_id", "target", "observed", "predicted", "spatial_block"],
    ].rename(columns={"predicted": "baseline"})
    records: list[dict[str, Any]] = []
    for target in config["evaluation"]["target_names"]:
        baseline_target = baseline.loc[baseline["target"].eq(target)]
        for model in config["models"]:
            candidate = predictions.loc[
                predictions["target"].eq(target)
                & predictions["model"].eq(model),
                ["row_id", "target", "predicted"],
            ].rename(columns={"predicted": "candidate"})
            merged = baseline_target.merge(
                candidate, on=["row_id", "target"], validate="one_to_one"
            )
            if len(merged) != int(config["source"]["expected_rows"]):
                raise RuntimeError("Baseline and candidate rows do not align")
            delta, lower, upper = paired_block_bootstrap(
                merged,
                int(config["evaluation"]["bootstrap_replicates"]),
                int(config["evaluation"]["bootstrap_seed"]),
            )
            baseline_rmse = float(
                np.sqrt(np.mean(np.square(merged["baseline"] - merged["observed"])))
            )
            candidate_rmse = float(
                np.sqrt(np.mean(np.square(merged["candidate"] - merged["observed"])))
            )
            records.append(
                {
                    "target": target,
                    "candidate": model,
                    "reference": baseline_model,
                    "candidate_rmse": candidate_rmse,
                    "reference_rmse": baseline_rmse,
                    "rmse_delta": delta,
                    "rmse_delta_ci_lower": lower,
                    "rmse_delta_ci_upper": upper,
                    "rmse_reduction_fraction": (
                        baseline_rmse - candidate_rmse
                    )
                    / baseline_rmse,
                }
            )
    return pd.DataFrame(records)


def main() -> None:
    config = load_config()
    predictions = load_new_predictions(config)
    metrics = calculate_metrics(predictions)
    pooled = metrics.loc[metrics["fold"].eq("pooled")].drop(columns="fold")
    comparison = comparisons(config, predictions)

    PREDICTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    FREEZE_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(PREDICTION_PATH, index=False)
    metrics.to_csv(METRIC_PATH, index=False)
    pooled.to_csv(POOLED_PATH, index=False)
    comparison.to_csv(COMPARISON_PATH, index=False)

    primary = str(config["evaluation"]["primary_target"])
    primary_metrics = pooled.loc[pooled["target"].eq(primary)].sort_values(
        "rmse"
    )
    primary_comparisons = comparison.loc[
        comparison["target"].eq(primary)
    ].sort_values("candidate_rmse")
    freeze = {
        "config_sha256": sha256(CONFIG_PATH),
        "source_hashes": {
            name: sha256(ROOT / path)
            for name, path in {
                "patch_manifest": config["source"]["patch_manifest"],
                "baseline_predictions": config["source"]["baseline_predictions"],
                "baseline_metrics": config["source"]["baseline_metrics"],
            }.items()
        },
        "output_hashes": {
            path.name: sha256(path)
            for path in [
                PREDICTION_PATH,
                METRIC_PATH,
                POOLED_PATH,
                COMPARISON_PATH,
            ]
        },
        "primary_metrics": primary_metrics.to_dict(orient="records"),
        "primary_comparisons": primary_comparisons.to_dict(orient="records"),
        "interpretation": config["interpretation"],
    }
    FREEZE_PATH.write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(primary_metrics.to_string(index=False), flush=True)
    print(primary_comparisons.to_string(index=False), flush=True)
    print(f"Wrote {FREEZE_PATH}", flush=True)


if __name__ == "__main__":
    main()
