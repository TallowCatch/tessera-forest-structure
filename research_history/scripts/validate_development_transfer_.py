#!/usr/bin/env python3
"""Validate pre-BART selection, BART predictors, and locked Phase 4 evaluation."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    rank = math.nan if np.ptp(predicted) <= 1e-12 else float(spearmanr(observed, predicted).statistic)
    residual_sum_of_squares = float(np.sum((predicted - observed) ** 2))
    total_sum_of_squares = float(np.sum((observed - observed.mean()) ** 2))
    return {
        "r2": 1.0 - residual_sum_of_squares / total_sum_of_squares,
        "rmse": float(np.sqrt(np.mean((predicted - observed) ** 2))),
        "mae": float(mean_absolute_error(observed, predicted)),
        "spearman_r": rank,
        "mean_bias": float(np.mean(predicted - observed)),
    }


def validate_outputs(manifest: dict) -> None:
    for output in manifest["outputs"].values():
        path = ROOT / output["path"]
        assert path.exists() and sha256(path) == output["sha256"]


def main() -> int:
    development_predictors = json.loads(
        (ROOT / "metadata/phase3_conventional_predictor_freeze.json").read_text()
    )
    transfer = json.loads((ROOT / "metadata/phase4_development_transfer_freeze.json").read_text())
    bart_predictors = json.loads((ROOT / "metadata/phase4_bart_predictor_freeze.json").read_text())
    evaluation = json.loads((ROOT / "metadata/phase4_locked_bart_evaluation_freeze.json").read_text())
    validate_outputs(transfer)
    validate_outputs(bart_predictors)
    validate_outputs(evaluation)

    snapshots = [
        (transfer["freeze_basis"]["project_config_sha256"], "metadata/project_config_phase4_development_transfer_freeze.yaml"),
        (bart_predictors["freeze_basis"]["project_config_sha256"], "metadata/project_config_phase4_bart_protocol_freeze.yaml"),
        (evaluation["freeze_basis"]["project_config_sha256"], "metadata/project_config_phase4_locked_bart_evaluation_freeze.yaml"),
    ]
    for expected, relative_path in snapshots:
        assert sha256(ROOT / relative_path) == expected

    assert transfer["freeze_basis"]["bart_file_read"] is False
    assert transfer["freeze_basis"]["bart_target_metrics_computed"] is False
    assert transfer["selected_pipeline"]["model"] == "tessera_area_topography_ridge"
    assert transfer["selected_pipeline"]["alpha"] == 100.0
    transfer_predictions = pd.read_parquet(ROOT / transfer["outputs"]["predictions"]["path"])
    transfer_summary = pd.read_csv(ROOT / transfer["outputs"]["summary"]["path"])
    for row in transfer_summary.itertuples(index=False):
        subset = transfer_predictions[transfer_predictions["direction"].eq(row.direction)]
        calculated = metrics(
            subset["fhd_normal"].to_numpy(dtype=float),
            subset[f"prediction_{row.model}"].to_numpy(dtype=float),
        )
        for name, value in calculated.items():
            stored = getattr(row, name)
            assert (math.isnan(value) and math.isnan(stored)) or math.isclose(
                value, stored, rel_tol=1e-10, abs_tol=1e-10
            )

    assert bart_predictors["freeze_basis"]["target_columns_read"] == []
    assert bart_predictors["freeze_basis"]["recipe"] == development_predictors["freeze_basis"]["recipe"]
    assert bart_predictors["freeze_basis"]["feature_columns"] == development_predictors["freeze_basis"]["feature_columns"]
    bart_features = pd.read_parquet(ROOT / bart_predictors["outputs"]["predictors"]["path"])
    assert len(bart_features) == 1026 and set(bart_features["site_id"]) == {"BART"}
    assert "fhd_normal" not in bart_features

    assert evaluation["freeze_basis"]["development_transfer_freeze_id"] == transfer["freeze_id"]
    assert evaluation["freeze_basis"]["bart_predictor_freeze_id"] == bart_predictors["freeze_id"]
    assert evaluation["freeze_basis"]["bart_tuning_or_model_selection_performed"] is False
    predictions = pd.read_parquet(ROOT / evaluation["outputs"]["predictions"]["path"])
    metric_table = pd.read_csv(ROOT / evaluation["outputs"]["metrics"]["path"])
    assert len(predictions) == 1026 and set(predictions["site_id"]) == {"BART"}
    assert predictions["spatial_block_id"].nunique() == 4
    assert metric_table["is_preselected_primary"].sum() == 1
    primary = metric_table[metric_table["is_preselected_primary"]].iloc[0]
    assert primary["model"] == transfer["selected_pipeline"]["model"]
    observed = predictions["fhd_normal"].to_numpy(dtype=float)
    for row in metric_table.itertuples(index=False):
        calculated = metrics(observed, predictions[f"prediction_{row.model}"].to_numpy(dtype=float))
        for name, value in calculated.items():
            stored = getattr(row, name)
            assert (math.isnan(value) and math.isnan(stored)) or math.isclose(
                value, stored, rel_tol=1e-10, abs_tol=1e-10
            )
    print("Phase 4 freeze validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
