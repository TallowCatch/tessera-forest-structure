#!/usr/bin/env python3
"""Validate the post-hoc corrected Sentinel-plus-topography sensitivity freeze."""

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
MANIFEST_PATH = ROOT / "metadata/phase4_corrected_sentinel_sensitivity_freeze.json"
SNAPSHOT_PATH = ROOT / "metadata/project_config_phase4_corrected_sentinel_protocol_freeze.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_values(frame: pd.DataFrame) -> dict[str, float]:
    observed = frame["fhd_normal"].to_numpy(dtype=float)
    predicted = frame["prediction_sentinel_topography_ridge_corrected"].to_numpy(dtype=float)
    residual_sum_of_squares = float(np.sum((predicted - observed) ** 2))
    total_sum_of_squares = float(np.sum((observed - observed.mean()) ** 2))
    return {
        "r2": 1.0 - residual_sum_of_squares / total_sum_of_squares,
        "rmse": float(np.sqrt(np.mean((predicted - observed) ** 2))),
        "mae": float(mean_absolute_error(observed, predicted)),
        "spearman_r": float(spearmanr(observed, predicted).statistic),
        "mean_bias": float(np.mean(predicted - observed)),
    }


def assert_metrics(frame: pd.DataFrame, row: pd.Series) -> None:
    for metric, value in metric_values(frame).items():
        assert math.isclose(value, float(row[metric]), rel_tol=1e-10, abs_tol=1e-10)


def main() -> int:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    basis = manifest["freeze_basis"]
    assert manifest["status"] == "frozen_posthoc_sensitivity_after_locked_bart_evaluation"
    assert basis["replaces_locked_confirmatory_result"] is False
    assert basis["bart_tuning_or_model_selection_performed"] is False
    assert sha256(SNAPSHOT_PATH) == basis["project_config_sha256"]
    assert basis["excluded_model_features"] == [
        "s1_ascending_valid_observation_count",
        "s1_descending_valid_observation_count",
        "s2_valid_observation_count",
    ]
    assert len(basis["corrected_model_features"]) == 43
    assert all(not feature.endswith("observation_count") for feature in basis["corrected_model_features"])
    for output in manifest["outputs"].values():
        path = ROOT / output["path"]
        assert path.exists() and sha256(path) == output["sha256"]

    development = pd.read_parquet(ROOT / manifest["outputs"]["development_predictions"]["path"])
    transfer = pd.read_parquet(ROOT / manifest["outputs"]["transfer_predictions"]["path"])
    bart = pd.read_parquet(ROOT / manifest["outputs"]["bart_predictions"]["path"])
    development_summary = pd.read_csv(ROOT / manifest["outputs"]["development_summary"]["path"])
    transfer_summary = pd.read_csv(ROOT / manifest["outputs"]["transfer_summary"]["path"])
    bart_summary = pd.read_csv(ROOT / manifest["outputs"]["bart_summary"]["path"])
    assert len(development) == 2037 and len(transfer) == 2037 and len(bart) == 1026
    for frame in [development, transfer, bart]:
        assert set(frame["sentinel_correction_freeze_id"]) == {manifest["freeze_id"]}
    assert_metrics(development, development_summary.iloc[0])
    for direction in ["SOAP_to_TEAK", "TEAK_to_SOAP"]:
        assert_metrics(
            transfer[transfer["direction"].eq(direction)],
            transfer_summary[transfer_summary["scope"].eq(direction)].iloc[0],
        )
    assert_metrics(bart, bart_summary.iloc[0])
    comparison = pd.read_csv(ROOT / manifest["outputs"]["comparison"]["path"])
    assert len(comparison) == 8
    assert comparison.loc[
        comparison["version"].eq("corrected_without_count_diagnostics"),
        "scientifically_valid_feature_set",
    ].all()
    print(f"Corrected Sentinel sensitivity validation passed: {manifest['freeze_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
