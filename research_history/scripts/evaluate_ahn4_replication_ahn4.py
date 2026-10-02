#!/usr/bin/env python3
"""Run the frozen spatial evaluation for the AHN4 Savelsbos replication."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/ahn4_replication_ahn4_deciduous.yaml"


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "phase26_ahn4_deciduous"
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


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.csv")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def equal_block_weights(blocks: pd.Series) -> np.ndarray:
    counts = blocks.value_counts()
    weights = blocks.map(1.0 / counts).to_numpy(dtype=np.float64)
    return weights * len(weights) / weights.sum()


def metric_values(
    observed: np.ndarray,
    predicted: np.ndarray,
    weights: np.ndarray,
    blocks: np.ndarray,
) -> dict[str, float]:
    error = predicted - observed
    correlations: list[float] = []
    for block in np.unique(blocks):
        selected = blocks == block
        if selected.sum() < 3:
            continue
        x = observed[selected]
        y = predicted[selected]
        if np.ptp(x) <= 1e-12 or np.ptp(y) <= 1e-12:
            continue
        correlations.append(float(spearmanr(x, y).statistic))
    return {
        "rmse": float(np.sqrt(np.average(error**2, weights=weights))),
        "mae": float(np.average(np.abs(error), weights=weights)),
        "r2": float(r2_score(observed, predicted, sample_weight=weights)),
        "bias": float(np.average(error, weights=weights)),
        "block_spearman": (
            float(np.mean(correlations)) if correlations else float("nan")
        ),
    }


def fit_ridge(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    alpha: float,
) -> tuple[StandardScaler, Ridge]:
    scaler = StandardScaler().fit(x, sample_weight=weights)
    model = Ridge(alpha=alpha).fit(
        scaler.transform(x), y, sample_weight=weights
    )
    return scaler, model


def height_adjustment_model(config: dict[str, Any]):
    settings = config["evaluation"]
    return make_pipeline(
        StandardScaler(),
        SplineTransformer(
            n_knots=int(settings["height_adjustment_spline_knots"]),
            degree=3,
            include_bias=False,
        ),
        Ridge(alpha=float(settings["height_adjustment_ridge_alpha"])),
    )


def feature_columns(frame: pd.DataFrame, prefix: str) -> list[str]:
    return sorted(column for column in frame if column.startswith(prefix))


def fold_positions(
    frame: pd.DataFrame,
    cohort: pd.DataFrame,
    fold_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(fold_path) as fold:
        train_ids = set(
            cohort.iloc[fold["train_indices"].astype(np.int64)]["row_id"].tolist()
        )
        test_ids = set(
            cohort.iloc[fold["test_indices"].astype(np.int64)]["row_id"].tolist()
        )
    ids = frame["row_id"].to_numpy(dtype=np.int64)
    return (
        np.flatnonzero(np.isin(ids, list(train_ids))),
        np.flatnonzero(np.isin(ids, list(test_ids))),
    )


def paired_block_bootstrap(
    predictions: pd.DataFrame,
    candidate: str,
    reference: str,
    replicates: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for (target, variant), group in predictions.groupby(
        ["target", "target_variant"], sort=True
    ):
        scoped = group[group["model"].isin([candidate, reference])]
        pivot = scoped.pivot(
            index=["row_id", "evaluation_block"],
            columns="model",
            values=["observed", "predicted"],
        )
        if candidate not in pivot["predicted"] or reference not in pivot["predicted"]:
            continue
        observed = pivot[("observed", candidate)].to_numpy(dtype=np.float64)
        candidate_error = (
            pivot[("predicted", candidate)].to_numpy(dtype=np.float64) - observed
        ) ** 2
        reference_error = (
            pivot[("predicted", reference)].to_numpy(dtype=np.float64) - observed
        ) ** 2
        block_frame = pd.DataFrame(
            {
                "block": pivot.index.get_level_values("evaluation_block"),
                "candidate": candidate_error,
                "reference": reference_error,
            }
        ).groupby("block", as_index=False).mean()
        blocks = len(block_frame)
        draws = np.empty(replicates, dtype=np.float64)
        for draw in range(replicates):
            selected = rng.integers(0, blocks, size=blocks)
            candidate_rmse = np.sqrt(block_frame["candidate"].to_numpy()[selected].mean())
            reference_rmse = np.sqrt(block_frame["reference"].to_numpy()[selected].mean())
            draws[draw] = candidate_rmse - reference_rmse
        observed_delta = float(
            np.sqrt(block_frame["candidate"].mean())
            - np.sqrt(block_frame["reference"].mean())
        )
        records.append(
            {
                "target": target,
                "target_variant": variant,
                "candidate": candidate,
                "reference": reference,
                "delta_rmse": observed_delta,
                "ci_low": float(np.quantile(draws, 0.025)),
                "ci_high": float(np.quantile(draws, 0.975)),
                "evaluation_blocks": int(blocks),
            }
        )
    return records


def run() -> None:
    config = load_config()
    outputs = config["outputs"]
    paths = {
        name: ROOT / str(outputs[name])
        for name in [
            "cohort",
            "folds",
            "tessera",
            "conventional",
            "predictions",
            "metrics",
            "bootstrap",
            "report",
            "result_freeze",
        ]
    }
    if paths["result_freeze"].exists():
        raise RuntimeError("Phase 26 results are already frozen")
    cohort = pd.read_parquet(paths["cohort"])
    tessera = pd.read_parquet(paths["tessera"])
    conventional = pd.read_parquet(paths["conventional"])
    frame = cohort.merge(tessera, on="row_id", validate="one_to_one").merge(
        conventional, on="row_id", validate="one_to_one"
    )
    tessera_mean = feature_columns(frame, "tessera_mean_")
    tessera_context = sum(
        [
            feature_columns(frame, f"tessera_{prefix}_")
            for prefix in config["predictors"]["tessera_sensitivity_features"]
        ],
        [],
    )
    conventional_columns = feature_columns(frame, "conventional_")
    conventional_columns = [
        column
        for column in conventional_columns
        if column
        not in {"conventional_context_valid", "conventional_valid_pixel_count"}
    ]
    valid = (
        frame["tessera_context_valid"].astype(bool)
        & frame["conventional_context_valid"].astype(bool)
        & np.isfinite(frame[tessera_context].to_numpy()).all(axis=1)
        & np.isfinite(frame[conventional_columns].to_numpy()).all(axis=1)
    )
    frame = frame[valid].reset_index(drop=True)
    if len(frame) < int(config["evaluation"]["minimum_cohort_rows"]):
        raise RuntimeError(
            f"Only {len(frame)} units have all frozen predictors; minimum is "
            f"{config['evaluation']['minimum_cohort_rows']}"
        )

    feature_sets = {
        "tessera_mean_ridge": tessera_mean,
        "tessera_context_ridge": tessera_context,
        "conventional_ridge": conventional_columns,
        "fused_ridge": tessera_mean + conventional_columns,
    }
    targets = list(config["evaluation"]["primary_targets"])
    controls = list(config["evaluation"]["height_adjustment_predictors"])
    ridge_alpha = float(config["evaluation"]["ridge_alpha"])
    prediction_records: list[pd.DataFrame] = []
    metric_records: list[dict[str, Any]] = []
    for fold in range(int(config["evaluation"]["folds"])):
        train, test = fold_positions(
            frame, cohort, paths["folds"] / f"fold_{fold}.npz"
        )
        if (
            len(train) < int(config["evaluation"]["minimum_train_rows"])
            or len(test) < int(config["evaluation"]["minimum_test_rows"])
        ):
            raise RuntimeError(
                f"Common-predictor fold {fold} has train={len(train)}, test={len(test)}"
            )
        train_weights = equal_block_weights(frame.iloc[train]["spatial_block"])
        test_weights = equal_block_weights(frame.iloc[test]["evaluation_block"])
        for target in targets:
            variants: dict[str, tuple[np.ndarray, np.ndarray]] = {
                "raw": (
                    frame.iloc[train][target].to_numpy(dtype=np.float64),
                    frame.iloc[test][target].to_numpy(dtype=np.float64),
                )
            }
            if target != "ahn4_p95_height_m":
                adjustment = height_adjustment_model(config)
                adjustment.fit(
                    frame.iloc[train][controls].to_numpy(dtype=np.float64),
                    frame.iloc[train][target].to_numpy(dtype=np.float64),
                    ridge__sample_weight=train_weights,
                )
                variants["height_adjusted"] = (
                    frame.iloc[train][target].to_numpy(dtype=np.float64)
                    - adjustment.predict(
                        frame.iloc[train][controls].to_numpy(dtype=np.float64)
                    ),
                    frame.iloc[test][target].to_numpy(dtype=np.float64)
                    - adjustment.predict(
                        frame.iloc[test][controls].to_numpy(dtype=np.float64)
                    ),
                )
            for variant, (train_y, test_y) in variants.items():
                models: dict[str, np.ndarray] = {
                    "training_mean": np.full(
                        len(test), np.average(train_y, weights=train_weights)
                    )
                }
                for model_name, columns in feature_sets.items():
                    train_x = frame.iloc[train][columns].to_numpy(dtype=np.float64)
                    test_x = frame.iloc[test][columns].to_numpy(dtype=np.float64)
                    scaler, model = fit_ridge(
                        train_x, train_y, train_weights, ridge_alpha
                    )
                    models[model_name] = model.predict(scaler.transform(test_x))
                for model_name, prediction in models.items():
                    metrics = metric_values(
                        test_y,
                        prediction,
                        test_weights,
                        frame.iloc[test]["evaluation_block"].to_numpy(),
                    )
                    metric_records.append(
                        {
                            "scope": "fold",
                            "fold": fold,
                            "target": target,
                            "target_variant": variant,
                            "model": model_name,
                            "rows": int(len(test)),
                            **metrics,
                        }
                    )
                    prediction_records.append(
                        pd.DataFrame(
                            {
                                "row_id": frame.iloc[test]["row_id"].to_numpy(),
                                "fold": fold,
                                "spatial_block": frame.iloc[test]["spatial_block"].to_numpy(),
                                "evaluation_block": frame.iloc[test]["evaluation_block"].to_numpy(),
                                "target": target,
                                "target_variant": variant,
                                "model": model_name,
                                "observed": test_y,
                                "predicted": prediction,
                            }
                        )
                    )
        print(
            f"phase26 evaluate: fold {fold} train={len(train):,} test={len(test):,}",
            flush=True,
        )
    predictions = pd.concat(prediction_records, ignore_index=True)
    metrics = pd.DataFrame(metric_records)
    pooled_records: list[dict[str, Any]] = []
    for (target, variant, model), group in predictions.groupby(
        ["target", "target_variant", "model"], sort=True
    ):
        weights = equal_block_weights(group["evaluation_block"])
        values = metric_values(
            group["observed"].to_numpy(dtype=np.float64),
            group["predicted"].to_numpy(dtype=np.float64),
            weights,
            group["evaluation_block"].to_numpy(),
        )
        pooled_records.append(
            {
                "scope": "pooled",
                "fold": -1,
                "target": target,
                "target_variant": variant,
                "model": model,
                "rows": int(len(group)),
                **values,
            }
        )
    metrics = pd.concat([metrics, pd.DataFrame(pooled_records)], ignore_index=True)
    rng = np.random.default_rng(int(config["evaluation"]["bootstrap_seed"]))
    bootstrap_records: list[dict[str, Any]] = []
    for candidate, reference in [
        ("tessera_mean_ridge", "conventional_ridge"),
        ("tessera_context_ridge", "tessera_mean_ridge"),
        ("fused_ridge", "tessera_mean_ridge"),
        ("fused_ridge", "conventional_ridge"),
    ]:
        bootstrap_records.extend(
            paired_block_bootstrap(
                predictions,
                candidate,
                reference,
                int(config["evaluation"]["bootstrap_replicates"]),
                rng,
            )
        )
    bootstrap = pd.DataFrame(bootstrap_records)
    atomic_parquet(predictions, paths["predictions"])
    atomic_csv(metrics, paths["metrics"])
    atomic_csv(bootstrap, paths["bootstrap"])

    pooled = metrics[metrics["scope"] == "pooled"].sort_values(
        ["target_variant", "target", "rmse"]
    )
    lines = [
        "# Phase 26 AHN4 Savelsbos replication",
        "",
        f"Frozen common cohort: {len(frame):,} 50 m broadleaf units.",
        "",
        "## Pooled spatially held-out results",
        "",
        "```text",
        pooled.to_string(index=False),
        "```",
        "",
        "The site was selected and the evaluation rules were frozen before AHN4 outcome values were inspected.",
    ]
    paths["report"].parent.mkdir(parents=True, exist_ok=True)
    paths["report"].write_text("\n".join(lines) + "\n")
    freeze = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete_frozen_replication",
        "common_cohort_rows": int(len(frame)),
        "feature_dimensions": {
            "tessera_mean": len(tessera_mean),
            "tessera_context": len(tessera_context),
            "conventional": len(conventional_columns),
            "fused": len(tessera_mean) + len(conventional_columns),
        },
        "inputs": {
            "cohort_sha256": sha256(paths["cohort"]),
            "tessera_sha256": sha256(paths["tessera"]),
            "conventional_sha256": sha256(paths["conventional"]),
            "config_sha256": sha256(CONFIG_PATH),
        },
        "outputs": {
            "predictions_sha256": sha256(paths["predictions"]),
            "metrics_sha256": sha256(paths["metrics"]),
            "bootstrap_sha256": sha256(paths["bootstrap"]),
        },
        "pooled_metrics": pooled.to_dict(orient="records"),
    }
    atomic_json(paths["result_freeze"], freeze)
    print("phase26 evaluation: results frozen", flush=True)


if __name__ == "__main__":
    run()
