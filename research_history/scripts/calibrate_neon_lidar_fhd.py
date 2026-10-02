#!/usr/bin/env python3
"""Calibrate simulated NEON lidar FHD to GEDI with frozen tile-level CV."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "metadata/phase5_neon_lidar_manifest.csv"
TILE_ROOT = ROOT / "data/processed/neon_lidar_tile_metrics"
TILE_FREEZE_ROOT = ROOT / "metadata/phase5_neon_lidar_tiles"
PROTOCOL_PATH = ROOT / "docs/phase5_neon_cross_sensor_calibration_protocol.md"
CONFIG_FREEZE_PATH = ROOT / "metadata/project_config_phase5_neon_cross_sensor_calibration_freeze.yaml"
PREDICTIONS_PATH = ROOT / "data/processed/neon_lidar_development_cv_predictions.parquet"
METRICS_PATH = ROOT / "outputs/tables/phase5_neon_lidar_calibration_metrics.csv"
FIGURE_PATH = ROOT / "outputs/figures/phase5_neon_lidar_calibration_cv.png"
FREEZE_PATH = ROOT / "metadata/phase5_neon_lidar_calibration_freeze.json"
DEVELOPMENT_SITES = ("SOAP", "TEAK")
EXPECTED_TILES_PER_SITE = 6


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


def fit_affine(simulated: np.ndarray, observed: np.ndarray) -> tuple[float, float]:
    design = np.column_stack([np.ones(len(simulated)), simulated])
    intercept, slope = np.linalg.lstsq(design, observed, rcond=None)[0]
    return float(intercept), float(slope)


def regression_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = predicted - observed
    pearson = math.nan
    spearman = math.nan
    if len(observed) >= 2 and np.ptp(observed) > 0 and np.ptp(predicted) > 0:
        pearson = float(pearsonr(observed, predicted).statistic)
        spearman = float(spearmanr(observed, predicted).statistic)
    return {
        "n": int(len(observed)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "mean_bias": float(np.mean(residual)),
        "r2": float(r2_score(observed, predicted)),
        "pearson_r": pearson,
        "spearman_r": spearman,
    }


def tile_tag(row: Any) -> str:
    return (
        f"{row.site_id}_{int(row.selection_order):02d}_"
        f"{int(row.tile_easting)}_{int(row.tile_northing)}"
    )


def load_development_tiles() -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    manifest = pd.read_csv(MANIFEST_PATH)
    selected = manifest[manifest["site_id"].isin(DEVELOPMENT_SITES)].copy()
    counts = selected.groupby("site_id").size().to_dict()
    expected_counts = {site: EXPECTED_TILES_PER_SITE for site in DEVELOPMENT_SITES}
    if counts != expected_counts:
        raise RuntimeError(f"Development manifest changed: {counts} != {expected_counts}")

    frames = []
    inputs = []
    for row in selected.sort_values(["site_id", "selection_order"]).itertuples(index=False):
        tag = tile_tag(row)
        metrics_path = TILE_ROOT / f"{tag}.parquet"
        freeze_path = TILE_FREEZE_ROOT / f"{tag}.json"
        if not metrics_path.is_file() or not freeze_path.is_file():
            raise RuntimeError(f"Missing frozen development tile: {tag}")
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        basis = freeze["freeze_basis"]
        if (
            basis["site_id"] != row.site_id
            or int(basis["selection_order"]) != int(row.selection_order)
            or int(basis["tile_easting"]) != int(row.tile_easting)
            or int(basis["tile_northing"]) != int(row.tile_northing)
        ):
            raise RuntimeError(f"Tile freeze identity mismatch: {tag}")
        if freeze["outputs"]["metrics"]["sha256"] != sha256(metrics_path):
            raise RuntimeError(f"Tile metrics checksum mismatch: {tag}")
        frame = pd.read_parquet(metrics_path)
        if len(frame) != int(row.eligible_gedi_rows):
            raise RuntimeError(f"Tile row population mismatch: {tag}")
        required = {"shot_number", "site_id", "fhd_normal", "simulated_fhd"}
        missing = required - set(frame.columns)
        if missing:
            raise RuntimeError(f"Tile {tag} lacks columns: {sorted(missing)}")
        if not frame["site_id"].eq(row.site_id).all():
            raise RuntimeError(f"Footprint site mismatch: {tag}")
        frame["neon_tile_id"] = tag
        frame["neon_tile_selection_order"] = int(row.selection_order)
        frame["neon_tile_easting"] = int(row.tile_easting)
        frame["neon_tile_northing"] = int(row.tile_northing)
        frames.append(frame)
        inputs.append({
            "tile_id": tag,
            "freeze_id": freeze["freeze_id"],
            "freeze_sha256": sha256(freeze_path),
            "metrics_sha256": sha256(metrics_path),
            "rows": len(frame),
        })

    combined = pd.concat(frames, ignore_index=True)
    if combined["shot_number"].duplicated().any():
        raise RuntimeError("GEDI shot numbers are duplicated across development tiles")
    if not np.isfinite(combined[["fhd_normal", "simulated_fhd"]].to_numpy()).all():
        raise RuntimeError("Development FHD values include non-finite values")
    return combined, inputs


def leave_one_tile_out(data: pd.DataFrame) -> pd.DataFrame:
    predictions = []
    for held_out_tile in sorted(data["neon_tile_id"].unique()):
        train = data[~data["neon_tile_id"].eq(held_out_tile)]
        test = data[data["neon_tile_id"].eq(held_out_tile)].copy()
        intercept, slope = fit_affine(
            train["simulated_fhd"].to_numpy(), train["fhd_normal"].to_numpy()
        )
        test["calibrated_fhd_cv"] = intercept + slope * test["simulated_fhd"]
        test["training_mean_baseline_cv"] = float(train["fhd_normal"].mean())
        test["fold_intercept"] = intercept
        test["fold_slope"] = slope
        test["held_out_tile"] = held_out_tile
        test["training_tile_count"] = int(train["neon_tile_id"].nunique())
        predictions.append(test)
    return pd.concat(predictions, ignore_index=True)


def metrics_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scopes = [("pooled", "all_development_tiles", predictions)]
    scopes.extend(
        ("tile", tile, frame)
        for tile, frame in predictions.groupby("neon_tile_id", sort=True)
    )
    for scope_type, scope_id, frame in scopes:
        observed = frame["fhd_normal"].to_numpy()
        for model, column in (
            ("affine_calibration", "calibrated_fhd_cv"),
            ("training_mean_baseline", "training_mean_baseline_cv"),
        ):
            rows.append({
                "scope_type": scope_type,
                "scope_id": scope_id,
                "model": model,
                **regression_metrics(observed, frame[column].to_numpy()),
            })
    return pd.DataFrame(rows)


def save_figure(predictions: pd.DataFrame) -> None:
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.4, 5.4))
    for site, frame in predictions.groupby("site_id", sort=True):
        axis.scatter(
            frame["fhd_normal"], frame["calibrated_fhd_cv"], s=24, alpha=0.7, label=site
        )
    lower = float(min(predictions["fhd_normal"].min(), predictions["calibrated_fhd_cv"].min()))
    upper = float(max(predictions["fhd_normal"].max(), predictions["calibrated_fhd_cv"].max()))
    axis.plot([lower, upper], [lower, upper], color="black", linewidth=1, linestyle="--")
    axis.set_xlabel("GEDI V3 fhd_normal")
    axis.set_ylabel("Cross-validated calibrated NEON lidar FHD")
    axis.set_title("Leave-one-NEON-tile-out development calibration")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(FIGURE_PATH, dpi=200)
    plt.close(figure)


def main() -> int:
    for path in (PREDICTIONS_PATH, METRICS_PATH, FIGURE_PATH, FREEZE_PATH):
        if path.exists():
            raise RuntimeError(f"Refusing to overwrite frozen calibration output: {path}")
    data, tile_inputs = load_development_tiles()
    predictions = leave_one_tile_out(data)
    metrics = metrics_table(predictions)
    pooled = metrics[metrics["scope_type"].eq("pooled")].set_index("model")
    calibration_rmse = float(pooled.loc["affine_calibration", "rmse"])
    baseline_rmse = float(pooled.loc["training_mean_baseline", "rmse"])
    calibration_r2 = float(pooled.loc["affine_calibration", "r2"])
    gate_passed = calibration_rmse < baseline_rmse and calibration_r2 > 0
    final_intercept, final_slope = fit_affine(
        data["simulated_fhd"].to_numpy(), data["fhd_normal"].to_numpy()
    )

    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(PREDICTIONS_PATH, index=False)
    metrics.to_csv(METRICS_PATH, index=False)
    save_figure(predictions)

    freeze_basis = {
        "protocol_path": str(PROTOCOL_PATH.relative_to(ROOT)),
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "config_freeze_path": str(CONFIG_FREEZE_PATH.relative_to(ROOT)),
        "config_freeze_sha256": sha256(CONFIG_FREEZE_PATH),
        "manifest_path": str(MANIFEST_PATH.relative_to(ROOT)),
        "manifest_sha256": sha256(MANIFEST_PATH),
        "development_sites": list(DEVELOPMENT_SITES),
        "tiles_per_site": EXPECTED_TILES_PER_SITE,
        "tile_inputs": tile_inputs,
        "target": "GEDI_V3_fhd_normal",
        "predictor": "gediMetric_FHD_from_NEON_airborne_lidar",
        "calibration": "ordinary_least_squares_affine_with_intercept",
        "validation": "leave_one_complete_NEON_tile_out",
        "baseline": "training_fold_mean_GEDI_fhd_normal",
        "gate": "pooled_CV_RMSE_below_baseline_and_pooled_CV_R2_above_zero",
        "script_sha256": sha256(Path(__file__)),
    }
    freeze_id = "phase5-neon-lidar-calibration-" + canonical_hash(freeze_basis)[:12]
    freeze = {
        "freeze_id": freeze_id,
        "created_at": utc_now(),
        "status": (
            "frozen_development_calibration_gate_passed_bart_unlocked"
            if gate_passed
            else "frozen_development_calibration_gate_failed_bart_locked"
        ),
        "gate": {
            "passed": gate_passed,
            "calibration_cv_rmse": calibration_rmse,
            "baseline_cv_rmse": baseline_rmse,
            "calibration_cv_r2": calibration_r2,
        },
        "final_development_coefficients": {
            "intercept": final_intercept,
            "slope": final_slope,
            "fit_rows": len(data),
            "fit_tiles": int(data["neon_tile_id"].nunique()),
        },
        "freeze_basis": freeze_basis,
        "freeze_basis_sha256": canonical_hash(freeze_basis),
        "outputs": {
            "cv_predictions": {
                "path": str(PREDICTIONS_PATH.relative_to(ROOT)),
                "rows": len(predictions),
                "sha256": sha256(PREDICTIONS_PATH),
            },
            "metrics": {
                "path": str(METRICS_PATH.relative_to(ROOT)),
                "rows": len(metrics),
                "sha256": sha256(METRICS_PATH),
            },
            "figure": {
                "path": str(FIGURE_PATH.relative_to(ROOT)),
                "sha256": sha256(FIGURE_PATH),
            },
        },
    }
    write_json(FREEZE_PATH, freeze)
    print(json.dumps({
        "freeze_id": freeze_id,
        "rows": len(data),
        "tiles": int(data["neon_tile_id"].nunique()),
        "gate_passed": gate_passed,
        "calibration_cv_rmse": calibration_rmse,
        "baseline_cv_rmse": baseline_rmse,
        "calibration_cv_r2": calibration_r2,
        "final_intercept": final_intercept,
        "final_slope": final_slope,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
