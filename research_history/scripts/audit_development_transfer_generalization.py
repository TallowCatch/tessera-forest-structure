#!/usr/bin/env python3
"""Independently audit Phase 4 R2 arithmetic and generalization assumptions."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from freeze_baseline_spatial_folds import estimate_variogram


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_PATH = ROOT / "data/processed/phase4_locked_bart_predictions.parquet"
METRICS_PATH = ROOT / "outputs/tables/phase4_locked_bart_metrics.csv"
BART_SOURCE_PATH = ROOT / "data/processed/tessera_aligned_locked_bart.parquet"
DEVELOPMENT_SOURCE_PATH = ROOT / "data/processed/tessera_aligned_development_soap_teak.parquet"
BART_CONVENTIONAL_PATH = ROOT / "data/processed/phase4_conventional_predictors_bart.parquet"
TRANSFER_FREEZE_PATH = ROOT / "metadata/phase4_development_transfer_freeze.json"
EVALUATION_FREEZE_PATH = ROOT / "metadata/phase4_locked_bart_evaluation_freeze.json"

MANUAL_PATH = ROOT / "outputs/tables/phase4_r2_manual_audit.csv"
SUBSET_PATH = ROOT / "outputs/tables/phase4_r2_subset_audit.csv"
AUDIT_PATH = ROOT / "metadata/phase4_r2_generalization_audit.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def manual_values(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = predicted - observed
    residual_sum_of_squares = float(np.sum(residual**2))
    total_sum_of_squares = float(np.sum((observed - observed.mean()) ** 2))
    if total_sum_of_squares <= 0:
        raise RuntimeError("R2 is undefined for a constant target")
    if np.ptp(predicted) <= 1e-12:
        pearson = math.nan
        spearman = math.nan
    else:
        pearson = float(pearsonr(observed, predicted).statistic)
        spearman = float(spearmanr(observed, predicted).statistic)
    bias = float(residual.mean())
    intercept_recalibrated = predicted - bias
    return {
        "residual_sum_of_squares": residual_sum_of_squares,
        "total_sum_of_squares": total_sum_of_squares,
        "manual_r2": 1.0 - residual_sum_of_squares / total_sum_of_squares,
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mean_bias": bias,
        "pearson_r": pearson,
        "pearson_r_squared": pearson**2 if math.isfinite(pearson) else math.nan,
        "spearman_r": spearman,
        "posthoc_intercept_recalibrated_r2": 1.0
        - float(np.sum((intercept_recalibrated - observed) ** 2)) / total_sum_of_squares,
    }


def main() -> int:
    protected = [MANUAL_PATH, SUBSET_PATH, AUDIT_PATH]
    if any(path.exists() for path in protected):
        raise RuntimeError("Phase 4 R2 audit already exists; refusing to overwrite")

    frozen = pd.read_parquet(PREDICTIONS_PATH)
    stored = pd.read_csv(METRICS_PATH)
    source_columns = [
        "site_id",
        "shot_number",
        "fhd_normal",
        "x_epsg5070",
        "y_epsg5070",
        "orbit",
        "reference_ground_track",
        "acquisition_datetime",
        "is_nighttime",
        "sensitivity",
        "nlcd_center_class",
    ]
    source = pd.read_parquet(BART_SOURCE_PATH, columns=source_columns)
    development = pd.read_parquet(
        DEVELOPMENT_SOURCE_PATH, columns=["site_id", "shot_number"]
    )
    if frozen.duplicated(["site_id", "shot_number"]).any() or source.duplicated(
        ["site_id", "shot_number"]
    ).any():
        raise RuntimeError("Duplicate BART evaluation keys")
    if set(development["shot_number"]).intersection(source["shot_number"]):
        raise RuntimeError("Development and BART shot numbers overlap")
    joined = frozen.merge(
        source,
        on=["site_id", "shot_number"],
        suffixes=("_frozen", "_source"),
        validate="one_to_one",
    )
    target_difference = float(
        np.max(np.abs(joined["fhd_normal_frozen"] - joined["fhd_normal_source"]))
    )
    if target_difference != 0.0:
        raise RuntimeError("Frozen evaluation target differs from the source target")

    manual_records: list[dict[str, Any]] = []
    maximum_r2_difference = 0.0
    for row in stored.itertuples(index=False):
        values = manual_values(
            frozen["fhd_normal"].to_numpy(dtype=float),
            frozen[f"prediction_{row.model}"].to_numpy(dtype=float),
        )
        difference = abs(values["manual_r2"] - float(row.r2))
        maximum_r2_difference = max(maximum_r2_difference, difference)
        manual_records.append(
            {
                "model": row.model,
                "stored_r2": float(row.r2),
                "absolute_r2_difference": difference,
                "observed_mean": float(frozen["fhd_normal"].mean()),
                "prediction_mean": float(frozen[f"prediction_{row.model}"].mean()),
                "prediction_standard_deviation": float(
                    frozen[f"prediction_{row.model}"].std(ddof=0)
                ),
                **values,
            }
        )
    manual = pd.DataFrame(manual_records)
    if maximum_r2_difference > 1e-12:
        raise RuntimeError("Stored R2 differs from independent SSE/SST arithmetic")

    transfer = json.loads(TRANSFER_FREEZE_PATH.read_text(encoding="utf-8"))
    selected_path = ROOT / transfer["outputs"]["selected_model"]["path"]
    selected = joblib.load(selected_path)
    features = selected["features"]
    tessera = pd.read_parquet(
        BART_SOURCE_PATH,
        columns=["site_id", "shot_number", *[feature for feature in features if feature.startswith("tessera_")]],
    )
    terrain_features = [feature for feature in features if feature.startswith("terrain_")]
    terrain = pd.read_parquet(BART_CONVENTIONAL_PATH)[
        ["site_id", "shot_number", *terrain_features]
    ]
    model_frame = tessera.merge(
        terrain, on=["site_id", "shot_number"], validate="one_to_one"
    )
    recreated = selected["model"].predict(
        model_frame[features].to_numpy(dtype=np.float32)
    )
    recreated_frame = model_frame[["site_id", "shot_number"]].copy()
    recreated_frame["recreated"] = recreated
    recreation = frozen[
        ["site_id", "shot_number", "prediction_tessera_area_topography_ridge"]
    ].merge(recreated_frame, on=["site_id", "shot_number"], validate="one_to_one")
    maximum_prediction_difference = float(
        np.max(
            np.abs(
                recreation["prediction_tessera_area_topography_ridge"]
                - recreation["recreated"]
            )
        )
    )
    if maximum_prediction_difference > 1e-12:
        raise RuntimeError("Frozen primary predictions cannot be recreated")

    analysis = frozen.merge(
        source.drop(columns=["fhd_normal", "x_epsg5070", "y_epsg5070"]),
        on=["site_id", "shot_number"],
        validate="one_to_one",
    )
    analysis["orbit_track"] = (
        analysis["orbit"].astype(str)
        + "_"
        + analysis["reference_ground_track"].astype(str)
    )
    analysis["acquisition_month"] = pd.to_datetime(
        analysis["acquisition_datetime"]
    ).dt.month
    primary_column = "prediction_tessera_area_topography_ridge"
    subset_records: list[dict[str, Any]] = []
    subset_definitions: list[tuple[str, str, pd.Series]] = [
        ("overall", "ALL", pd.Series(True, index=analysis.index)),
        ("quality", "nighttime", analysis["is_nighttime"]),
        ("quality", "sensitivity_at_least_0.98", analysis["sensitivity"].ge(0.98)),
    ]
    for block in sorted(analysis["spatial_block_id"].unique()):
        subset_definitions.append(
            ("spatial_block", block, analysis["spatial_block_id"].eq(block))
        )
    for track in sorted(analysis["orbit_track"].unique()):
        subset_definitions.append(
            ("orbit_track", track, analysis["orbit_track"].eq(track))
        )
    for nlcd_class in sorted(analysis["nlcd_center_class"].unique()):
        subset_definitions.append(
            ("nlcd_class", str(nlcd_class), analysis["nlcd_center_class"].eq(nlcd_class))
        )
    for family, identifier, mask in subset_definitions:
        subset = analysis[mask]
        if len(subset) < 10 or subset["fhd_normal"].nunique() < 2:
            continue
        values = manual_values(
            subset["fhd_normal"].to_numpy(dtype=float),
            subset[primary_column].to_numpy(dtype=float),
        )
        subset_records.append(
            {
                "subset_family": family,
                "subset": identifier,
                "rows": len(subset),
                **values,
            }
        )
    subsets = pd.DataFrame(subset_records)

    _, variogram = estimate_variogram(
        source[["site_id", "x_epsg5070", "y_epsg5070", "fhd_normal"]],
        "BART",
        250.0,
        30,
    )
    orbit_track_count = int(analysis["orbit_track"].nunique())
    spatial_block_count = int(analysis["spatial_block_id"].nunique())
    all_blocks_negative = bool(
        (subsets.loc[subsets["subset_family"].eq("spatial_block"), "manual_r2"] < 0).all()
    )
    all_tracks_negative = bool(
        (subsets.loc[subsets["subset_family"].eq("orbit_track"), "manual_r2"] < 0).all()
    )

    write_csv(manual, MANUAL_PATH)
    write_csv(subsets, SUBSET_PATH)
    evaluation = json.loads(EVALUATION_FREEZE_PATH.read_text(encoding="utf-8"))
    report = {
        "created_at": utc_now(),
        "status": "frozen_posthoc_evaluation_audit",
        "locked_evaluation_freeze_id": evaluation["freeze_id"],
        "r2_point_estimate_valid": True,
        "checks": {
            "rows": len(frozen),
            "unique_evaluation_keys": int(frozen[["site_id", "shot_number"]].drop_duplicates().shape[0]),
            "development_bart_shot_overlap": 0,
            "maximum_source_target_difference": target_difference,
            "maximum_manual_stored_r2_difference": maximum_r2_difference,
            "maximum_recreated_prediction_difference": maximum_prediction_difference,
            "all_bart_spatial_blocks_have_negative_primary_r2": all_blocks_negative,
            "all_bart_orbit_tracks_have_negative_primary_r2": all_tracks_negative,
        },
        "interpretation": {
            "valid_claim": (
                "The frozen SOAP/TEAK model performs worse than the BART test-set mean "
                "for the 1,026 quality-filtered GEDI shots sampled by three 2024 passes."
            ),
            "invalid_claim": (
                "The four-block bootstrap provides precise uncertainty for every BART "
                "forest location or all times of year."
            ),
            "negative_r2_meaning": (
                "Prediction SSE exceeds the SSE of a constant predictor equal to the "
                "BART observed mean; R2 is not squared Pearson correlation."
            ),
        },
        "uncertainty_limitations": {
            "spatial_blocks": spatial_block_count,
            "orbit_track_passes": orbit_track_count,
            "acquisition_dates": int(pd.to_datetime(analysis["acquisition_datetime"]).dt.date.nunique()),
            "configured_bootstrap_block_width_m": 7000,
            "posthoc_bart_variogram_practical_range_m": variogram["practical_range_m"],
            "variogram_range_hit_upper_fit_bound": math.isclose(
                variogram["practical_range_m"], variogram["maximum_fitted_lag_m"] * 2.0
            ),
            "conclusion": (
                "The point metrics are exact for the sampled shots, but confidence "
                "intervals and wider BART population generalization are low precision."
            ),
        },
        "outputs": {
            "manual_metrics": {
                "path": str(MANUAL_PATH.relative_to(ROOT)),
                "sha256": sha256(MANUAL_PATH),
            },
            "subset_metrics": {
                "path": str(SUBSET_PATH.relative_to(ROOT)),
                "sha256": sha256(SUBSET_PATH),
            },
        },
        "script_sha256": sha256(Path(__file__).resolve()),
    }
    write_json(report, AUDIT_PATH)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
