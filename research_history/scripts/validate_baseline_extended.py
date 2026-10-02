#!/usr/bin/env python3
"""Validate the conventional predictors and extended Phase 3 development freeze."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(observed, predicted)),
        "rmse": float(np.sqrt(np.mean((predicted - observed) ** 2))),
        "mae": float(mean_absolute_error(observed, predicted)),
        "spearman_r": float(spearmanr(observed, predicted).statistic),
        "mean_bias": float(np.mean(predicted - observed)),
    }


def main() -> int:
    conventional = json.loads((ROOT / "metadata/phase3_conventional_predictor_freeze.json").read_text())
    extended = json.loads((ROOT / "metadata/phase3_extended_development_freeze.json").read_text())
    snapshots = {
        conventional["freeze_basis"]["project_config_sha256"]:
            ROOT / "metadata/project_config_phase3_conventional_features_freeze.yaml",
        extended["freeze_basis"]["project_config_sha256"]:
            ROOT / "metadata/project_config_phase3_extended_development_freeze.yaml",
    }
    for expected, path in snapshots.items():
        assert sha256(path) == expected
    for manifest in [conventional, extended]:
        for output in manifest["outputs"].values():
            path = ROOT / output["path"]
            assert path.exists() and sha256(path) == output["sha256"]

    predictor_path = ROOT / conventional["outputs"]["predictors"]["path"]
    predictors = pd.read_parquet(predictor_path)
    feature_columns = conventional["freeze_basis"]["feature_columns"]
    assert len(predictors) == 2037 and set(predictors["site_id"]) == {"SOAP", "TEAK"}
    assert "fhd_normal" not in predictors and conventional["freeze_basis"]["target_columns_read"] == []
    assert np.isfinite(predictors[feature_columns].to_numpy(dtype=float)).all()
    assert predictors["s2_valid_observation_count"].min() >= 6
    assert predictors["s1_ascending_valid_observation_count"].min() >= 6
    assert predictors["s1_descending_valid_observation_count"].min() >= 6

    predictions = pd.read_parquet(ROOT / extended["outputs"]["oof_predictions"]["path"])
    summary = pd.read_csv(ROOT / extended["outputs"]["summary"]["path"])
    assert len(predictions) == 2037 and set(predictions["site_id"]) == {"SOAP", "TEAK"}
    assert set(predictions["phase3_extended_freeze_id"]) == {extended["freeze_id"]}
    assert "BART" not in set(predictions["site_id"])
    observed = predictions["fhd_normal"].to_numpy(dtype=float)
    for row in summary[summary["scope"].eq("ALL")].itertuples(index=False):
        calculated = metrics(observed, predictions[f"prediction_{row.model}"].to_numpy(dtype=float))
        for name, value in calculated.items():
            assert math.isclose(value, getattr(row, name), rel_tol=1e-10, abs_tol=1e-10)
    label = pd.read_parquet(ROOT / extended["outputs"]["label_efficiency_predictions"]["path"])
    assert set(label["phase3_extended_freeze_id"]) == {extended["freeze_id"]}
    assert set(label["training_block_fraction"]) == {0.1, 0.25, 0.5, 1.0}
    print("Phase 3 extended freeze validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
