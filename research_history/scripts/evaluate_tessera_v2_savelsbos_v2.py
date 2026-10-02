#!/usr/bin/env python3
"""Compare v1.0 and experimental v2 on the frozen Savelsbos test."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from evaluate_ahn4_replication_ahn4 import (
    equal_block_weights,
    fit_ridge,
    fold_positions,
    metric_values,
    paired_block_bootstrap,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/tessera_v2.yaml"


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "phase27_tessera_v2"
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feature_columns(frame: pd.DataFrame, prefix: str) -> list[str]:
    return sorted(column for column in frame if column.startswith(prefix))


def height_model(settings: dict[str, Any]):
    return make_pipeline(
        StandardScaler(),
        SplineTransformer(
            n_knots=int(settings["height_adjustment_spline_knots"]),
            degree=3,
            include_bias=False,
        ),
        Ridge(alpha=float(settings["height_adjustment_ridge_alpha"])),
    )


def atomic(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp" + path.suffix)
    writer(temporary)
    temporary.replace(path)


def run() -> None:
    config = load_config()
    site = config["savelsbos"]
    evaluation = config["savelsbos_evaluation"]
    paths = {
        key: ROOT / str(site[key])
        for key in [
            "cohort", "v1_features", "v2_features", "conventional", "folds",
            "predictions", "metrics", "bootstrap", "comparison", "report",
            "result_freeze",
        ]
    }
    if paths["result_freeze"].exists():
        raise RuntimeError("Phase 27 Savelsbos result is already frozen")
    cohort = pd.read_parquet(paths["cohort"])
    v1 = pd.read_parquet(paths["v1_features"]).rename(
        columns=lambda value: value if value == "row_id" else f"v1_{value}"
    )
    v2 = pd.read_parquet(paths["v2_features"]).rename(
        columns=lambda value: value if value == "row_id" else f"v2_{value}"
    )
    conventional = pd.read_parquet(paths["conventional"])
    frame = cohort.merge(v1, on="row_id", validate="one_to_one").merge(
        v2, on="row_id", validate="one_to_one"
    ).merge(conventional, on="row_id", validate="one_to_one")
    v1_mean = feature_columns(frame, "v1_tessera_mean_")
    v2_mean = feature_columns(frame, "v2_tessera_mean_")
    conventional_columns = [
        column
        for column in feature_columns(frame, "conventional_")
        if column
        not in {"conventional_context_valid", "conventional_valid_pixel_count"}
    ]
    if len(v1_mean) != 128 or len(v2_mean) != 128:
        raise RuntimeError("Both TESSERA versions must expose 128 mean features")
    valid = (
        frame["v1_tessera_context_valid"].astype(bool)
        & frame["v2_tessera_context_valid"].astype(bool)
        & frame["conventional_context_valid"].astype(bool)
        & np.isfinite(frame[v1_mean + v2_mean + conventional_columns]).all(axis=1)
    )
    frame = frame[valid].reset_index(drop=True)
    feature_sets = {
        "tessera_v1": v1_mean,
        "tessera_v2": v2_mean,
        "conventional": conventional_columns,
        "fused_v1": v1_mean + conventional_columns,
        "fused_v2": v2_mean + conventional_columns,
    }
    targets = [str(value) for value in evaluation["targets"]]
    controls = [str(value) for value in evaluation["height_predictors"]]
    prediction_parts: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    for fold in range(int(evaluation["folds"])):
        train, test = fold_positions(
            frame, cohort, paths["folds"] / f"fold_{fold}.npz"
        )
        if (
            len(train) < int(evaluation["minimum_train_rows"])
            or len(test) < int(evaluation["minimum_test_rows"])
        ):
            raise RuntimeError(f"Savelsbos v2 fold {fold} is too small")
        train_weights = equal_block_weights(frame.iloc[train]["spatial_block"])
        test_weights = equal_block_weights(frame.iloc[test]["evaluation_block"])
        for target in targets:
            variants = {
                "raw": (
                    frame.iloc[train][target].to_numpy(dtype=np.float64),
                    frame.iloc[test][target].to_numpy(dtype=np.float64),
                )
            }
            if target != "ahn4_p95_height_m":
                adjustment = height_model(evaluation)
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
                for model_name, columns in feature_sets.items():
                    scaler, model = fit_ridge(
                        frame.iloc[train][columns].to_numpy(dtype=np.float64),
                        train_y,
                        train_weights,
                        float(evaluation["ridge_alpha"]),
                    )
                    predicted = model.predict(
                        scaler.transform(
                            frame.iloc[test][columns].to_numpy(dtype=np.float64)
                        )
                    )
                    values = metric_values(
                        test_y,
                        predicted,
                        test_weights,
                        frame.iloc[test]["evaluation_block"].to_numpy(),
                    )
                    metric_rows.append(
                        {
                            "scope": "fold", "fold": fold, "target": target,
                            "target_variant": variant, "model": model_name,
                            "rows": len(test), **values,
                        }
                    )
                    prediction_parts.append(
                        pd.DataFrame(
                            {
                                "row_id": frame.iloc[test]["row_id"].to_numpy(),
                                "fold": fold,
                                "target": target,
                                "target_variant": variant,
                                "model": model_name,
                                "observed": test_y,
                                "predicted": predicted,
                                "evaluation_block": frame.iloc[test][
                                    "evaluation_block"
                                ].to_numpy(),
                            }
                        )
                    )
    predictions = pd.concat(prediction_parts, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    pooled_rows = []
    for (target, variant, model), group in predictions.groupby(
        ["target", "target_variant", "model"], sort=True
    ):
        weights = equal_block_weights(group["evaluation_block"])
        values = metric_values(
            group["observed"].to_numpy(),
            group["predicted"].to_numpy(),
            weights,
            group["evaluation_block"].to_numpy(),
        )
        pooled_rows.append(
            {
                "scope": "pooled", "fold": -1, "target": target,
                "target_variant": variant, "model": model,
                "rows": len(group), **values,
            }
        )
    metrics = pd.concat([metrics, pd.DataFrame(pooled_rows)], ignore_index=True)
    rng = np.random.default_rng(int(evaluation["bootstrap_seed"]))
    bootstrap = pd.DataFrame(
        paired_block_bootstrap(
            predictions, "tessera_v2", "tessera_v1",
            int(evaluation["bootstrap_replicates"]), rng,
        )
        + paired_block_bootstrap(
            predictions, "fused_v2", "fused_v1",
            int(evaluation["bootstrap_replicates"]), rng,
        )
        + paired_block_bootstrap(
            predictions, "tessera_v2", "conventional",
            int(evaluation["bootstrap_replicates"]), rng,
        )
    )
    comparison = bootstrap[
        bootstrap["reference"].isin(["tessera_v1", "fused_v1"])
    ].reset_index(drop=True)
    atomic(paths["predictions"], lambda p: predictions.to_parquet(p, index=False))
    atomic(paths["metrics"], lambda p: metrics.to_csv(p, index=False))
    atomic(paths["bootstrap"], lambda p: bootstrap.to_csv(p, index=False))
    atomic(paths["comparison"], lambda p: comparison.to_csv(p, index=False))
    pooled = metrics[metrics["scope"] == "pooled"]
    lines = [
        "# Savelsbos experimental TESSERA v2 comparison", "",
        "The frozen 2021 broadleaf cohort, spatial folds, LiDAR targets and Ridge settings were retained. Only the TESSERA representation changed.",
        "", "## Pooled held-out metrics", "", "```text",
        pooled.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "```", "", "## Paired v2 minus v1 RMSE", "", "```text",
        comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "```", "",
    ]
    atomic(paths["report"], lambda p: p.write_text("\n".join(lines), encoding="utf-8"))
    artifacts = {
        str(path.relative_to(ROOT)): sha256(path)
        for key, path in paths.items()
        if key in {"predictions", "metrics", "bootstrap", "comparison", "report"}
    }
    paths["result_freeze"].parent.mkdir(parents=True, exist_ok=True)
    paths["result_freeze"].write_text(
        json.dumps(
            {"created_utc": datetime.now(timezone.utc).isoformat(), "artifacts": artifacts},
            indent=2, sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    print("phase27 Savelsbos v2 evaluation complete", flush=True)


if __name__ == "__main__":
    run()
