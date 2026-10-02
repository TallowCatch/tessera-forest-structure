#!/usr/bin/env python3
"""Prepare label-free CORAL predictions for BART and freeze before target access."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/project.yaml"
PROTOCOL_FREEZE_PATH = ROOT / "metadata/project_config_phase5_adaptation_protocol_freeze.yaml"
SOURCE_PATH = ROOT / "data/processed/tessera_aligned_development_soap_teak.parquet"
SOURCE_CONVENTIONAL_PATH = ROOT / "data/processed/phase3_conventional_predictors_development.parquet"
TARGET_PATH = ROOT / "data/processed/tessera_aligned_locked_bart.parquet"
TARGET_CONVENTIONAL_PATH = ROOT / "data/processed/phase4_conventional_predictors_bart.parquet"
SOURCE_MODEL_PATH = ROOT / "outputs/models/phase4_selected_pipeline_pre_bart.joblib"
PREDICTIONS_PATH = ROOT / "data/processed/phase5_bart_coral_target_free_predictions.parquet"
MODEL_PATH = ROOT / "outputs/models/phase5_bart_coral.joblib"
FREEZE_PATH = ROOT / "metadata/phase5_bart_coral_prediction_freeze.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def write_joblib(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(value, temporary)
    temporary.replace(path)


def symmetric_matrix_power(matrix: np.ndarray, power: float, regularization: float) -> np.ndarray:
    symmetric = (matrix + matrix.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    adjusted = np.maximum(eigenvalues, 0.0) + regularization
    return (eigenvectors * np.power(adjusted, power)) @ eigenvectors.T


def coral_transform(
    source: np.ndarray,
    target: np.ndarray,
    regularization: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    source_covariance = np.cov(source_centered, rowvar=False)
    target_covariance = np.cov(target_centered, rowvar=False)
    transform = symmetric_matrix_power(
        source_covariance, -0.5, regularization
    ) @ symmetric_matrix_power(target_covariance, 0.5, regularization)
    aligned = source_centered @ transform + target_mean
    aligned_covariance = np.cov(aligned, rowvar=False)
    before = float(np.linalg.norm(source_covariance - target_covariance, ord="fro"))
    after = float(np.linalg.norm(aligned_covariance - target_covariance, ord="fro"))
    diagnostics = {
        "source_target_mean_distance_before": float(np.linalg.norm(source_mean - target_mean)),
        "source_target_mean_distance_after": float(np.linalg.norm(aligned.mean(axis=0) - target_mean)),
        "source_target_covariance_frobenius_before": before,
        "source_target_covariance_frobenius_after": after,
        "covariance_distance_fraction_remaining": after / before,
        "source_covariance_min_eigenvalue": float(np.linalg.eigvalsh(source_covariance).min()),
        "target_covariance_min_eigenvalue": float(np.linalg.eigvalsh(target_covariance).min()),
    }
    return aligned, transform, diagnostics


def main() -> int:
    protected = [PREDICTIONS_PATH, MODEL_PATH, FREEZE_PATH]
    existing = [str(path.relative_to(ROOT)) for path in protected if path.exists()]
    if existing:
        raise RuntimeError(f"CORAL target-free freeze exists; refusing to overwrite: {existing}")
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    protocol = config["phase5_bart_unsupervised_coral"]
    if protocol["status"] != "posthoc_protocol_frozen_before_coral_execution":
        raise RuntimeError("CORAL experiment lost its frozen post-hoc status")
    source_model = joblib.load(SOURCE_MODEL_PATH)
    features = list(source_model["features"])
    area = [column for column in features if column.startswith("tessera_area_")]
    terrain = [column for column in features if column.startswith("terrain_")]
    if len(area) != 128 or len(terrain) != 4 or float(source_model["alpha"]) != 100.0:
        raise RuntimeError("Unexpected frozen source feature recipe")

    source_alignment_columns = ["site_id", "shot_number", "fhd_normal", *area]
    source_conventional_columns = ["site_id", "shot_number", *terrain]
    target_alignment_columns = ["site_id", "shot_number", *area]
    target_conventional_columns = ["site_id", "shot_number", *terrain]
    source = pd.read_parquet(SOURCE_PATH, columns=source_alignment_columns).merge(
        pd.read_parquet(SOURCE_CONVENTIONAL_PATH, columns=source_conventional_columns),
        on=["site_id", "shot_number"], validate="one_to_one",
    )
    target = pd.read_parquet(TARGET_PATH, columns=target_alignment_columns).merge(
        pd.read_parquet(TARGET_CONVENTIONAL_PATH, columns=target_conventional_columns),
        on=["site_id", "shot_number"], validate="one_to_one",
    )
    if len(source) != 2037 or set(source["site_id"]) != {"SOAP", "TEAK"}:
        raise RuntimeError("CORAL source population changed")
    if len(target) != 1026 or set(target["site_id"]) != {"BART"}:
        raise RuntimeError("CORAL target predictor population changed")
    source_x = source[features].to_numpy(dtype=np.float64)
    target_x = target[features].to_numpy(dtype=np.float64)
    source_y = source["fhd_normal"].to_numpy(dtype=np.float64)
    if not np.isfinite(source_x).all() or not np.isfinite(target_x).all():
        raise RuntimeError("CORAL features contain non-finite values")

    scaler = StandardScaler().fit(source_x)
    source_standardized = scaler.transform(source_x)
    target_standardized = scaler.transform(target_x)
    regularization = float(protocol["covariance_regularization"])
    source_aligned, transform, diagnostics = coral_transform(
        source_standardized, target_standardized, regularization
    )
    if diagnostics["covariance_distance_fraction_remaining"] >= 1.0:
        raise RuntimeError("CORAL did not reduce source-target covariance distance")
    coral_model = Ridge(alpha=float(protocol["ridge_alpha_from_development"])).fit(
        source_aligned, source_y
    )
    coral_prediction = coral_model.predict(target_standardized)
    frozen_source_prediction = source_model["model"].predict(target_x.astype(np.float32))
    predictions = target[["site_id", "shot_number"]].copy()
    predictions["prediction_frozen_source_recreated"] = frozen_source_prediction
    predictions["prediction_coral"] = coral_prediction
    predictions["prediction_source_training_mean"] = float(source_y.mean())
    if not np.isfinite(predictions.filter(like="prediction_").to_numpy()).all():
        raise RuntimeError("CORAL produced non-finite predictions")

    artifact = {
        "method": "linear_correlation_alignment",
        "features": features,
        "source_scaler": scaler,
        "coral_transform": transform,
        "ridge": coral_model,
        "covariance_regularization": regularization,
        "source_sites": ["SOAP", "TEAK"],
        "unlabeled_target_site": "BART",
        "target_labels_used": False,
    }
    write_parquet(predictions, PREDICTIONS_PATH)
    write_joblib(artifact, MODEL_PATH)

    freeze_basis = {
        "analysis_label": "posthoc_transductive_label_free_coral",
        "protocol": protocol,
        "protocol_config_freeze_sha256": sha256(PROTOCOL_FREEZE_PATH),
        "source_alignment_path": str(SOURCE_PATH.relative_to(ROOT)),
        "source_alignment_sha256": sha256(SOURCE_PATH),
        "source_alignment_columns_read": source_alignment_columns,
        "source_conventional_path": str(SOURCE_CONVENTIONAL_PATH.relative_to(ROOT)),
        "source_conventional_sha256": sha256(SOURCE_CONVENTIONAL_PATH),
        "source_conventional_columns_read": source_conventional_columns,
        "target_alignment_path": str(TARGET_PATH.relative_to(ROOT)),
        "target_alignment_sha256": sha256(TARGET_PATH),
        "target_alignment_columns_read": target_alignment_columns,
        "target_conventional_path": str(TARGET_CONVENTIONAL_PATH.relative_to(ROOT)),
        "target_conventional_sha256": sha256(TARGET_CONVENTIONAL_PATH),
        "target_conventional_columns_read": target_conventional_columns,
        "target_columns_read_contain_fhd_normal": "fhd_normal" in target_alignment_columns,
        "target_labels_used_for_alignment_training_or_selection": False,
        "source_model_sha256": sha256(SOURCE_MODEL_PATH),
        "features": features,
        "diagnostics": diagnostics,
        "script_sha256": sha256(Path(__file__)),
        "sklearn_version": sklearn.__version__,
    }
    freeze_id = "phase5-bart-coral-predictions-" + canonical_hash(freeze_basis)[:12]
    freeze = {
        "freeze_id": freeze_id,
        "created_at": utc_now(),
        "status": "frozen_target_free_coral_predictions_before_bart_outcome_access",
        "freeze_basis": freeze_basis,
        "freeze_basis_sha256": canonical_hash(freeze_basis),
        "outputs": {
            "predictions": {"path": str(PREDICTIONS_PATH.relative_to(ROOT)), "sha256": sha256(PREDICTIONS_PATH), "rows": len(predictions)},
            "model": {"path": str(MODEL_PATH.relative_to(ROOT)), "sha256": sha256(MODEL_PATH)},
        },
    }
    write_json(FREEZE_PATH, freeze)
    print(json.dumps({
        "freeze_id": freeze_id,
        "target_rows": len(predictions),
        "target_labels_used": False,
        **diagnostics,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
