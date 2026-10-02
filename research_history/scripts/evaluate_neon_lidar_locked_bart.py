#!/usr/bin/env python3
"""Run the locked BART evaluation against calibrated NEON airborne lidar FHD."""

from __future__ import annotations

import hashlib
import json
import math
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
CALIBRATION_FREEZE_PATH = ROOT / "metadata/phase5_neon_lidar_calibration_freeze.json"
PHASE4_PREDICTIONS_PATH = ROOT / "data/processed/phase4_locked_bart_predictions.parquet"
PHASE4_FREEZE_PATH = ROOT / "metadata/phase4_locked_bart_evaluation_freeze.json"
CORRECTED_SENTINEL_PATH = ROOT / "data/processed/phase4_corrected_sentinel_bart_predictions.parquet"
CORRECTED_SENTINEL_FREEZE_PATH = ROOT / "metadata/phase4_corrected_sentinel_sensitivity_freeze.json"
OUTPUT_PATH = ROOT / "data/processed/phase5_neon_lidar_locked_bart_evaluation.parquet"
METRICS_PATH = ROOT / "outputs/tables/phase5_neon_lidar_locked_bart_metrics.csv"
FIGURE_PATH = ROOT / "outputs/figures/phase5_neon_lidar_locked_bart_evaluation.png"
FREEZE_PATH = ROOT / "metadata/phase5_neon_lidar_locked_bart_freeze.json"
EXPECTED_BART_TILES = 6
PRIMARY_MODEL = "tessera_area_topography_ridge"
MODEL_COLUMNS = {
    "training_mean": "prediction_training_mean",
    "terrain_ridge": "prediction_terrain_ridge",
    "sentinel_topography_ridge_corrected": "prediction_sentinel_topography_ridge_corrected",
    "tessera_area_ridge": "prediction_tessera_area_ridge",
    "tessera_area_topography_ridge": "prediction_tessera_area_topography_ridge",
    "tessera_hist_gradient_boosting": "prediction_tessera_hist_gradient_boosting",
}


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
        f"BART_{int(row.selection_order):02d}_"
        f"{int(row.tile_easting)}_{int(row.tile_northing)}"
    )


def load_bart_lidar() -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    manifest = pd.read_csv(MANIFEST_PATH)
    selected = manifest[manifest["site_id"].eq("BART")].sort_values("selection_order")
    if len(selected) != EXPECTED_BART_TILES:
        raise RuntimeError(f"Expected {EXPECTED_BART_TILES} frozen BART tiles, found {len(selected)}")
    if selected["month"].astype(str).nunique() != 1 or str(selected["month"].iloc[0]) != "2024-08":
        raise RuntimeError("BART lidar acquisition month changed from frozen August 2024")

    frames = []
    inputs = []
    for row in selected.itertuples(index=False):
        tag = tile_tag(row)
        metrics_path = TILE_ROOT / f"{tag}.parquet"
        freeze_path = TILE_FREEZE_ROOT / f"{tag}.json"
        if not metrics_path.is_file() or not freeze_path.is_file():
            raise RuntimeError(f"Missing frozen BART tile: {tag}")
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        basis = freeze["freeze_basis"]
        if basis["site_id"] != "BART" or int(basis["selection_order"]) != int(row.selection_order):
            raise RuntimeError(f"BART tile freeze identity mismatch: {tag}")
        if freeze["outputs"]["metrics"]["sha256"] != sha256(metrics_path):
            raise RuntimeError(f"BART tile metrics checksum mismatch: {tag}")
        frame = pd.read_parquet(metrics_path)
        if len(frame) != int(row.eligible_gedi_rows):
            raise RuntimeError(f"BART tile population mismatch: {tag}")
        frame["neon_tile_id"] = tag
        frame["neon_tile_selection_order"] = int(row.selection_order)
        frame["neon_lidar_month"] = "2024-08"
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
        raise RuntimeError("BART lidar tiles contain duplicate GEDI shot numbers")
    return combined, inputs


def join_frozen_predictions(lidar: pd.DataFrame) -> pd.DataFrame:
    phase4 = pd.read_parquet(PHASE4_PREDICTIONS_PATH)
    sentinel = pd.read_parquet(CORRECTED_SENTINEL_PATH)[
        ["site_id", "shot_number", "prediction_sentinel_topography_ridge_corrected"]
    ]
    if phase4.duplicated(["site_id", "shot_number"]).any() or sentinel.duplicated(
        ["site_id", "shot_number"]
    ).any():
        raise RuntimeError("Frozen Phase 4 prediction keys are not unique")
    phase4 = phase4.drop(columns=["prediction_sentinel_topography_ridge"])
    predictions = phase4.merge(sentinel, on=["site_id", "shot_number"], validate="one_to_one")
    joined = lidar.merge(
        predictions,
        on=["site_id", "shot_number"],
        suffixes=("", "_phase4"),
        validate="one_to_one",
    )
    if len(joined) != len(lidar):
        raise RuntimeError("Not every BART lidar footprint has a frozen Phase 4 prediction")
    target_difference = np.max(np.abs(joined["fhd_normal"] - joined["fhd_normal_phase4"]))
    if float(target_difference) != 0.0:
        raise RuntimeError("BART lidar and Phase 4 GEDI targets differ")
    return joined.drop(columns=["fhd_normal_phase4"])


def build_metrics(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scopes = [("pooled", "all_bart_tiles", data)]
    scopes.extend(("tile", tile, frame) for tile, frame in data.groupby("neon_tile_id", sort=True))
    comparisons = [
        ("GEDI_V3_fhd_normal", "calibrated_neon_lidar", "calibrated_neon_fhd"),
        ("GEDI_V3_fhd_normal", "raw_simulated_neon_lidar", "simulated_fhd"),
    ]
    comparisons.extend(
        ("calibrated_NEON_lidar_FHD", model, column)
        for model, column in MODEL_COLUMNS.items()
    )
    comparisons.extend(
        ("GEDI_V3_fhd_normal_subset", model, column)
        for model, column in MODEL_COLUMNS.items()
    )
    for scope_type, scope_id, frame in scopes:
        for target_name, model, column in comparisons:
            observed_column = (
                "calibrated_neon_fhd"
                if target_name == "calibrated_NEON_lidar_FHD"
                else "fhd_normal"
            )
            rows.append({
                "scope_type": scope_type,
                "scope_id": scope_id,
                "target": target_name,
                "model": model,
                "is_preselected_primary_tessera_model": model == PRIMARY_MODEL,
                **regression_metrics(
                    frame[observed_column].to_numpy(), frame[column].to_numpy()
                ),
            })
    return pd.DataFrame(rows)


def save_figure(data: pd.DataFrame) -> None:
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 5.0))
    panels = [
        (axes[0], "fhd_normal", "calibrated_neon_fhd", "GEDI V3 fhd_normal", "Calibrated NEON lidar FHD"),
        (
            axes[1],
            "calibrated_neon_fhd",
            MODEL_COLUMNS[PRIMARY_MODEL],
            "Calibrated NEON lidar FHD",
            "Locked TESSERA + topography prediction",
        ),
    ]
    for axis, observed_column, predicted_column, x_label, y_label in panels:
        for tile, frame in data.groupby("neon_tile_id", sort=True):
            axis.scatter(frame[observed_column], frame[predicted_column], s=22, alpha=0.65, label=tile)
        lower = float(min(data[observed_column].min(), data[predicted_column].min()))
        upper = float(max(data[observed_column].max(), data[predicted_column].max()))
        axis.plot([lower, upper], [lower, upper], color="black", linewidth=1, linestyle="--")
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
    axes[0].set_title("Locked cross-sensor transfer at BART")
    axes[1].set_title("TESSERA model against airborne lidar reference")
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8)
    figure.tight_layout(rect=(0, 0.12, 1, 1))
    figure.savefig(FIGURE_PATH, dpi=200)
    plt.close(figure)


def main() -> int:
    for path in (OUTPUT_PATH, METRICS_PATH, FIGURE_PATH, FREEZE_PATH):
        if path.exists():
            raise RuntimeError(f"Refusing to overwrite locked BART output: {path}")
    calibration = json.loads(CALIBRATION_FREEZE_PATH.read_text(encoding="utf-8"))
    if not calibration["gate"]["passed"]:
        raise RuntimeError("Development calibration gate failed; BART evaluation is prohibited")
    coefficients = calibration["final_development_coefficients"]
    lidar, tile_inputs = load_bart_lidar()
    lidar["calibrated_neon_fhd"] = (
        float(coefficients["intercept"]) + float(coefficients["slope"]) * lidar["simulated_fhd"]
    )
    joined = join_frozen_predictions(lidar)
    metrics = build_metrics(joined)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joined.to_parquet(OUTPUT_PATH, index=False)
    metrics.to_csv(METRICS_PATH, index=False)
    save_figure(joined)

    phase4_freeze = json.loads(PHASE4_FREEZE_PATH.read_text(encoding="utf-8"))
    sentinel_freeze = json.loads(CORRECTED_SENTINEL_FREEZE_PATH.read_text(encoding="utf-8"))
    freeze_basis = {
        "bart_tile_inputs": tile_inputs,
        "bart_lidar_month": "2024-08",
        "calibration_freeze_id": calibration["freeze_id"],
        "calibration_freeze_sha256": sha256(CALIBRATION_FREEZE_PATH),
        "calibration_coefficients_used_without_refitting": coefficients,
        "phase4_bart_evaluation_freeze_id": phase4_freeze["freeze_id"],
        "phase4_bart_predictions_sha256": sha256(PHASE4_PREDICTIONS_PATH),
        "corrected_sentinel_freeze_id": sentinel_freeze["freeze_id"],
        "corrected_sentinel_predictions_sha256": sha256(CORRECTED_SENTINEL_PATH),
        "primary_tessera_model": PRIMARY_MODEL,
        "model_selection_or_refitting_on_bart": False,
        "all_predeclared_models_reported": True,
        "evaluation_groups": "six_complete_NEON_lidar_tiles",
        "script_sha256": sha256(Path(__file__)),
    }
    freeze_id = "phase5-neon-lidar-bart-" + canonical_hash(freeze_basis)[:12]
    freeze = {
        "freeze_id": freeze_id,
        "created_at": utc_now(),
        "status": "frozen_locked_bart_neon_lidar_evaluation",
        "freeze_basis": freeze_basis,
        "freeze_basis_sha256": canonical_hash(freeze_basis),
        "outputs": {
            "evaluation": {
                "path": str(OUTPUT_PATH.relative_to(ROOT)),
                "rows": len(joined),
                "sha256": sha256(OUTPUT_PATH),
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
    pooled = metrics[metrics["scope_type"].eq("pooled")]
    print(pooled[["target", "model", "n", "rmse", "r2", "pearson_r", "spearman_r"]].to_string(index=False))
    print(f"\nFrozen {freeze_id}: {len(joined)} BART footprints across {EXPECTED_BART_TILES} tiles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
