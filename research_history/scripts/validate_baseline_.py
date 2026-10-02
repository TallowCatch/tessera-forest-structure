#!/usr/bin/env python3
"""Validate frozen Phase 3 spatial folds, baselines, and BART separation."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
CONFIG_SNAPSHOT_PATH = ROOT / "metadata/project_config_phase3_spatial_baselines_freeze.yaml"
FOLD_SCRIPT_PATH = ROOT / "scripts/freeze_baseline_spatial_folds.py"
TRAIN_SCRIPT_PATH = ROOT / "scripts/train_baselines.py"
PHASE2_FREEZE_PATH = ROOT / "metadata/tessera_phase2_alignment_freeze.json"
FOLD_FREEZE_PATH = ROOT / "metadata/phase3_spatial_fold_freeze.json"
BASELINE_FREEZE_PATH = ROOT / "metadata/phase3_baseline_freeze.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_hashes(manifest: dict[str, object]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name, record in manifest["outputs"].items():
        path = ROOT / record["path"]
        require(path.is_file(), f"Missing output {name}: {path}")
        require(sha256(path) == record["sha256"], f"Checksum mismatch: {path}")
        paths[name] = path
    return paths


def validate_fold_freeze() -> tuple[dict[str, object], pd.DataFrame]:
    manifest = json.loads(FOLD_FREEZE_PATH.read_text(encoding="utf-8"))
    require(manifest["status"] == "frozen", "Phase 3 spatial folds are not frozen")
    require(
        manifest["freeze_id"] == f"spatial-folds-{manifest['freeze_basis_sha256'][:12]}",
        "Spatial fold ID does not match its freeze basis",
    )
    basis = manifest["freeze_basis"]
    require(
        sha256(CONFIG_SNAPSHOT_PATH) == basis["project_config_sha256"],
        "Phase 3 config snapshot changed",
    )
    require(sha256(FOLD_SCRIPT_PATH) == basis["fold_script_sha256"], "Fold script changed")
    paths = validate_hashes(manifest)
    folds = pd.read_parquet(paths["folds"])

    require(len(folds) == 2037, "Expected 2,037 development fold rows")
    require(set(folds["site_id"]) == {"SOAP", "TEAK"}, "Unexpected site in fold table")
    require(not folds.duplicated(["site_id", "shot_number"]).any(), "Duplicate fold rows")
    require(folds["spatial_fold_freeze_id"].eq(manifest["freeze_id"]).all(), "Fold ID mismatch")
    require(set(folds["spatial_fold_id"]) == set(range(5)), "Expected five spatial folds")
    require(
        folds.groupby("spatial_fold_id")["site_id"].nunique().eq(2).all(),
        "Every fold must contain SOAP and TEAK",
    )
    require(
        folds.groupby("spatial_block_id")["spatial_fold_id"].nunique().eq(1).all(),
        "A spatial block was split across folds",
    )

    block_width = int(basis["block_width_m"])
    require(block_width == 7000, f"Unexpected frozen block width: {block_width}")
    expected_x = np.floor(folds["x_epsg5070"] / block_width).astype(int)
    expected_y = np.floor(folds["y_epsg5070"] / block_width).astype(int)
    require(expected_x.equals(folds["spatial_block_x"]), "Spatial block x index mismatch")
    require(expected_y.equals(folds["spatial_block_y"]), "Spatial block y index mismatch")

    variograms = pd.read_csv(paths["variogram_summary"])
    ranges = dict(zip(variograms["site_id"], variograms["practical_range_m"], strict=True))
    require(6000.0 < ranges["SOAP"] < 6100.0, "SOAP variogram range changed")
    require(350.0 < ranges["TEAK"] < 400.0, "TEAK variogram range changed")
    require(
        math.ceil(max(ranges.values()) / 1000.0) * 1000 == block_width,
        "Block width does not follow the frozen variogram rule",
    )
    return manifest, folds


def recompute_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = predicted - observed
    correlation = float(spearmanr(observed, predicted).statistic)
    return {
        "r2": float(r2_score(observed, predicted)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(mean_absolute_error(observed, predicted)),
        "spearman_r": correlation,
        "mean_bias": float(np.mean(residual)),
    }


def validate_buffer(predictions: pd.DataFrame, diagnostics: pd.DataFrame, buffer_m: float) -> None:
    for fold_id in range(5):
        test_mask = predictions["spatial_fold_id"].eq(fold_id).to_numpy()
        candidate_mask = ~test_mask
        keep = candidate_mask.copy()
        minimum_distance = math.inf
        for site in ("SOAP", "TEAK"):
            test_site = test_mask & predictions["site_id"].eq(site).to_numpy()
            candidate_site = candidate_mask & predictions["site_id"].eq(site).to_numpy()
            candidate_indices = np.flatnonzero(candidate_site)
            tree = cKDTree(
                predictions.loc[test_site, ["x_epsg5070", "y_epsg5070"]].to_numpy()
            )
            distances, _ = tree.query(
                predictions.iloc[candidate_indices][["x_epsg5070", "y_epsg5070"]].to_numpy(),
                k=1,
            )
            retained = distances > buffer_m
            keep[candidate_indices[~retained]] = False
            minimum_distance = min(minimum_distance, float(distances[retained].min()))
        row = diagnostics[diagnostics["spatial_fold_id"].eq(fold_id)].iloc[0]
        require(int(keep.sum()) == row["training_rows_after_buffer"], "Buffered row count mismatch")
        require(int(test_mask.sum()) == row["test_rows"], "Test row count mismatch")
        require(minimum_distance > buffer_m, "Training row violates the 500 m buffer")
        require(
            abs(minimum_distance - row["minimum_retained_train_test_distance_m"]) < 1e-6,
            "Minimum train/test distance mismatch",
        )


def validate_baseline_freeze(
    fold_manifest: dict[str, object],
    folds: pd.DataFrame,
) -> dict[str, object]:
    manifest = json.loads(BASELINE_FREEZE_PATH.read_text(encoding="utf-8"))
    require(manifest["status"] == "frozen", "Phase 3 baselines are not frozen")
    require(
        manifest["freeze_id"] == f"phase3-baselines-{manifest['freeze_basis_sha256'][:12]}",
        "Baseline ID does not match its freeze basis",
    )
    basis = manifest["freeze_basis"]
    require(
        sha256(CONFIG_SNAPSHOT_PATH) == basis["project_config_sha256"],
        "Phase 3 config snapshot changed",
    )
    require(sha256(TRAIN_SCRIPT_PATH) == basis["training_script_sha256"], "Training script changed")
    require(
        basis["spatial_fold_freeze_id"] == fold_manifest["freeze_id"],
        "Baseline uses the wrong spatial fold freeze",
    )
    require(not basis["bart_target_metrics_computed"], "BART was used for Phase 3 metrics")
    require(len(basis["features"]) == 128, "Baseline does not use 128 TESSERA features")
    require(
        basis["features"] == [f"tessera_area_{index:03d}" for index in range(128)],
        "Baseline feature set is not the primary footprint aggregation",
    )
    paths = validate_hashes(manifest)
    predictions = pd.read_parquet(paths["oof_predictions"])
    require(len(predictions) == 2037, "Expected 2,037 out-of-fold predictions")
    require(set(predictions["site_id"]) == {"SOAP", "TEAK"}, "BART or unknown site in predictions")
    require(not predictions.duplicated(["site_id", "shot_number"]).any(), "Duplicate predictions")
    require(predictions["phase3_baseline_freeze_id"].eq(manifest["freeze_id"]).all(), "Baseline ID mismatch")
    require(
        set(zip(predictions["site_id"], predictions["shot_number"], strict=True))
        == set(zip(folds["site_id"], folds["shot_number"], strict=True)),
        "Predictions do not cover the exact frozen folds",
    )
    prediction_columns = {
        "training_mean": "prediction_training_mean",
        "tessera_ridge": "prediction_tessera_ridge",
        "tessera_hist_gradient_boosting": "prediction_tessera_hist_gradient_boosting",
    }
    require(
        np.isfinite(predictions[list(prediction_columns.values())].to_numpy()).all(),
        "Non-finite predictions",
    )

    diagnostics = pd.read_csv(paths["fold_diagnostics"])
    validate_buffer(predictions, diagnostics, float(basis["outer_validation"]["buffer_m"]))
    summary = pd.read_csv(paths["summary"])
    observed = predictions["fhd_normal"].to_numpy(dtype=np.float64)
    for model, column in prediction_columns.items():
        reported = summary[summary["scope"].eq("ALL") & summary["model"].eq(model)].iloc[0]
        recalculated = recompute_metrics(
            observed, predictions[column].to_numpy(dtype=np.float64)
        )
        for metric, value in recalculated.items():
            require(abs(value - reported[metric]) < 1e-12, f"Metric mismatch: {model} {metric}")

    phase2 = json.loads(PHASE2_FREEZE_PATH.read_text(encoding="utf-8"))
    locked_path = ROOT / phase2["outputs"]["locked_bart"]["path"]
    require(
        sha256(locked_path) == basis["locked_bart_sha256_unchanged"],
        "Locked BART table changed during Phase 3",
    )
    locked_ids = set(pd.read_parquet(locked_path, columns=["shot_number"])["shot_number"])
    require(
        locked_ids.isdisjoint(set(predictions["shot_number"])),
        "A locked BART shot appears in Phase 3 predictions",
    )

    mean_model = json.loads(paths["training_mean_model"].read_text(encoding="utf-8"))
    ridge_model = joblib.load(paths["ridge_model"])
    nonlinear_model = joblib.load(paths["nonlinear_model"])
    require(mean_model["phase3_baseline_freeze_id"] == manifest["freeze_id"], "Mean model ID mismatch")
    require(ridge_model["phase3_baseline_freeze_id"] == manifest["freeze_id"], "Ridge model ID mismatch")
    require(nonlinear_model["phase3_baseline_freeze_id"] == manifest["freeze_id"], "Nonlinear model ID mismatch")
    require(ridge_model["feature_columns"] == basis["features"], "Ridge feature metadata mismatch")
    require(nonlinear_model["feature_columns"] == basis["features"], "Nonlinear feature metadata mismatch")
    return manifest


def main() -> None:
    fold_manifest, folds = validate_fold_freeze()
    baseline_manifest = validate_baseline_freeze(fold_manifest, folds)
    overall = {
        row["model"]: {"r2": row["r2"], "rmse": row["rmse"]}
        for row in baseline_manifest["summary"]
        if row["scope"] == "ALL"
    }
    print(
        json.dumps(
            {
                "status": "valid",
                "spatial_fold_freeze_id": fold_manifest["freeze_id"],
                "phase3_baseline_freeze_id": baseline_manifest["freeze_id"],
                "development_rows": len(folds),
                "spatial_blocks": folds["spatial_block_id"].nunique(),
                "block_width_m": fold_manifest["freeze_basis"]["block_width_m"],
                "bart_target_metrics_computed": False,
                "overall": overall,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
