#!/usr/bin/env python3
"""Freeze the source-only height expectation model and residual-FHD training target."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "metadata/project_config_phase10_unseen_replication_protocol_freeze.yaml"
SOURCE_PATHS = [
    ROOT / "data/processed/tessera_aligned_development_soap_teak.parquet",
    ROOT / "data/processed/tessera_aligned_locked_bart.parquet",
    ROOT / "data/processed/phase7_tessera_aligned_expansion.parquet",
]
OUTPUT_PATH = ROOT / "data/processed/phase10_source_height_residuals.parquet"
TUNING_PATH = ROOT / "outputs/tables/phase10_source_height_model_tuning.csv"
FREEZE_PATH = ROOT / "metadata/phase10_source_height_model_freeze.json"
KEYS = ["site_id", "shot_number"]
HEIGHT_INPUTS = ["elev_highestreturn", "elev_lowestmode"]
FEATURES = [f"tessera_area_{index:03d}" for index in range(128)]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def height_features(height: np.ndarray) -> np.ndarray:
    if not np.isfinite(height).all() or (height < 0).any():
        raise RuntimeError("Canopy-height proxy must be finite and nonnegative")
    logged = np.log1p(height)
    return np.column_stack([logged, logged**2, logged**3])


def equal_site_weights(site_ids: np.ndarray) -> np.ndarray:
    site_ids = np.asarray(site_ids)
    unique, counts = np.unique(site_ids, return_counts=True)
    count_by_site = dict(zip(unique, counts, strict=True))
    return np.asarray([1.0 / count_by_site[site] for site in site_ids], dtype=np.float64)


def fit_model(frame: pd.DataFrame, alpha: float) -> tuple[StandardScaler, Ridge]:
    x = height_features(frame["gedi_canopy_height_proxy_m"].to_numpy(dtype=np.float64))
    y = frame["fhd_normal"].to_numpy(dtype=np.float64)
    weights = equal_site_weights(frame["site_id"].to_numpy())
    scaler = StandardScaler().fit(x, sample_weight=weights)
    model = Ridge(alpha=alpha).fit(scaler.transform(x), y, sample_weight=weights)
    return scaler, model


def main() -> int:
    protected = [OUTPUT_PATH, TUNING_PATH, FREEZE_PATH]
    existing = [str(path.relative_to(ROOT)) for path in protected if path.exists()]
    if existing:
        raise RuntimeError(f"Phase 10 source-height freeze exists; refusing to overwrite: {existing}")

    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))["phase10_unseen_replication"]
    source_sites = sorted(protocol["source_sites"])
    columns = KEYS + ["fhd_normal", *HEIGHT_INPUTS, *FEATURES]
    sources = pd.concat(
        [pd.read_parquet(path, columns=columns) for path in SOURCE_PATHS], ignore_index=True
    )
    if sorted(sources["site_id"].unique()) != source_sites or sources.duplicated(KEYS).any():
        raise RuntimeError("Unexpected original eight-site source population")
    sources["gedi_canopy_height_proxy_m"] = (
        sources["elev_highestreturn"] - sources["elev_lowestmode"]
    )
    required = ["fhd_normal", "gedi_canopy_height_proxy_m", *FEATURES]
    if not np.isfinite(sources[required].to_numpy(dtype=np.float64)).all():
        raise RuntimeError("Source model inputs contain non-finite values")
    if (sources["gedi_canopy_height_proxy_m"] < 0).any():
        raise RuntimeError("Source canopy-height proxy contains negative values")

    alphas = [float(value) for value in protocol["source_only_height_adjustment"]["expected_fhd_alpha_grid"]]
    tuning_records: list[dict[str, Any]] = []
    for alpha in alphas:
        site_rmse = []
        for held_out in source_sites:
            training = sources[~sources["site_id"].eq(held_out)]
            testing = sources[sources["site_id"].eq(held_out)]
            scaler, model = fit_model(training, alpha)
            prediction = model.predict(scaler.transform(height_features(
                testing["gedi_canopy_height_proxy_m"].to_numpy(dtype=np.float64)
            )))
            rmse = float(mean_squared_error(testing["fhd_normal"], prediction) ** 0.5)
            site_rmse.append(rmse)
            tuning_records.append({
                "alpha": alpha,
                "held_out_site": held_out,
                "rmse": rmse,
            })
        tuning_records.append({
            "alpha": alpha,
            "held_out_site": "__summary__",
            "rmse": float(max(site_rmse)),
            "macro_mean_rmse": float(np.mean(site_rmse)),
        })
    tuning = pd.DataFrame(tuning_records)
    summaries = tuning[tuning["held_out_site"].eq("__summary__")].sort_values(
        ["rmse", "macro_mean_rmse", "alpha"]
    )
    selected_alpha = float(summaries.iloc[0]["alpha"])
    scaler, model = fit_model(sources, selected_alpha)
    expected = model.predict(scaler.transform(height_features(
        sources["gedi_canopy_height_proxy_m"].to_numpy(dtype=np.float64)
    )))
    output = sources.copy()
    output["height_expected_fhd"] = expected
    output["fhd_height_residual"] = output["fhd_normal"] - expected
    output = output.sort_values(KEYS).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    TUNING_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(OUTPUT_PATH.with_suffix(".parquet.tmp"), index=False, compression="zstd")
    OUTPUT_PATH.with_suffix(".parquet.tmp").replace(OUTPUT_PATH)
    tuning.to_csv(TUNING_PATH.with_suffix(".csv.tmp"), index=False)
    TUNING_PATH.with_suffix(".csv.tmp").replace(TUNING_PATH)

    freeze_basis = {
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "script_sha256": sha256(Path(__file__).resolve()),
        "source_files": {str(path.relative_to(ROOT)): sha256(path) for path in SOURCE_PATHS},
        "source_sites": source_sites,
        "source_rows": len(sources),
        "height_proxy": "elev_highestreturn - elev_lowestmode",
        "feature_transform": ["log1p(height)", "log1p(height)^2", "log1p(height)^3"],
        "model": "standardized_equal-site-weighted_ridge",
        "alpha_grid": alphas,
        "alpha_selection": "minimum worst-site LOSO RMSE; ties by macro RMSE then alpha",
        "selected_alpha": selected_alpha,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "model_intercept": float(model.intercept_),
        "model_coefficients": model.coef_.tolist(),
        "target_outcome_columns_read": [],
        "target_height_columns_read": [],
        "target_outcomes_opened": False,
    }
    freeze_id = "phase10-source-height-" + canonical_hash(freeze_basis)[:12]
    write_json(FREEZE_PATH, {
        "freeze_id": freeze_id,
        "created_utc": utc_now(),
        "status": "frozen_before_new_target_screen_or_acquisition",
        "freeze_basis": freeze_basis,
        "outputs": {
            "source_residuals": {"path": str(OUTPUT_PATH.relative_to(ROOT)), "rows": len(output), "sha256": sha256(OUTPUT_PATH)},
            "tuning": {"path": str(TUNING_PATH.relative_to(ROOT)), "rows": len(tuning), "sha256": sha256(TUNING_PATH)},
        },
    })
    print(json.dumps({
        "freeze_id": freeze_id,
        "source_rows": len(output),
        "selected_alpha": selected_alpha,
        "residual_sd": float(output["fhd_height_residual"].std(ddof=1)),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
