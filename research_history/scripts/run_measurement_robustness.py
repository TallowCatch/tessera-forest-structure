#!/usr/bin/env python3
"""Evaluate frozen six-site predictions under GEDI measurement-condition filters."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_multisite_loso as loso  # noqa: E402


TARGET_PATH = ROOT / "data/processed/phase9_prospective_tessera_aligned.parquet"
CONVENTIONAL_PATH = ROOT / "data/processed/phase9_prospective_conventional_predictors.parquet"
PREDICTIONS_PATH = ROOT / "data/processed/phase9_prospective_evaluated_predictions.parquet"
METRICS_PATH = ROOT / "outputs/tables/robustness_measurement_metrics.csv"
PAIRED_PATH = ROOT / "outputs/tables/robustness_measurement_paired.csv"
CALIBRATION_PATH = ROOT / "outputs/tables/robustness_calibration_diagnostics.csv"
ASSOCIATIONS_PATH = ROOT / "outputs/tables/robustness_measurement_associations.csv"
FREEZE_PATH = ROOT / "metadata/robustness_measurement_freeze.json"
KEYS = ["site_id", "shot_number"]
STRATEGIES = ["all_sources", "nearest5_evt_phys"]
MINIMUM_SITE_ROWS = 30


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


def add_filter_flags(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["day_of_year"] = result["acquisition_datetime"].dt.dayofyear
    result["cell_30m"] = (
        np.floor(result["x_epsg5070"] / 30.0).astype(int).astype(str)
        + "_"
        + np.floor(result["y_epsg5070"] / 30.0).astype(int).astype(str)
    )
    earliest = (
        result.sort_values(["acquisition_datetime", "shot_number"])
        .drop_duplicates(["site_id", "cell_30m"], keep="first")
        .set_index(KEYS).index
    )
    key_index = pd.MultiIndex.from_frame(result[KEYS])
    result["first_observation_per_30m_cell"] = key_index.isin(earliest)
    return result


def masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "all_retained": np.ones(len(frame), dtype=bool),
        "sensitivity_ge_0.95": frame["sensitivity"].ge(0.95).to_numpy(),
        "sensitivity_ge_0.98": frame["sensitivity"].ge(0.98).to_numpy(),
        "nighttime": frame["is_nighttime"].astype(bool).to_numpy(),
        "slope_le_15deg": frame["terrain_slope_degrees"].le(15.0).to_numpy(),
        "slope_le_30deg": frame["terrain_slope_degrees"].le(30.0).to_numpy(),
        "may_to_september": frame["day_of_year"].between(122, 274).to_numpy(),
        "first_observation_per_30m_cell": frame[
            "first_observation_per_30m_cell"
        ].to_numpy(),
    }


def metric_table(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for subset_name, selected in masks(frame).items():
        subset = frame.loc[selected]
        for (strategy, site), scoped in subset.groupby(["strategy", "site_id"], sort=True):
            if len(scoped) < MINIMUM_SITE_ROWS:
                continue
            records.append({
                "subset": subset_name,
                "strategy": strategy,
                "scope": "site",
                "site_id": site,
                "rows": len(scoped),
                **loso.metric_values(
                    scoped["fhd_normal"].to_numpy(), scoped["prediction"].to_numpy()
                ),
            })
    site = pd.DataFrame(records)
    macro_records = []
    for (subset_name, strategy), scoped in site.groupby(["subset", "strategy"], sort=True):
        macro_records.append({
            "subset": subset_name,
            "strategy": strategy,
            "scope": "macro",
            "site_id": "macro_mean",
            "rows": int(scoped["rows"].sum()),
            "site_count": len(scoped),
            **{metric: float(scoped[metric].mean()) for metric in loso.METRICS},
        })
    site["site_count"] = 1
    return pd.concat([site, pd.DataFrame(macro_records)], ignore_index=True)


def paired_table(metrics: pd.DataFrame) -> pd.DataFrame:
    site = metrics[metrics["scope"].eq("site")]
    records: list[dict[str, Any]] = []
    for subset_name, scoped in site.groupby("subset", sort=True):
        all_source = scoped[scoped["strategy"].eq("all_sources")].set_index("site_id")
        matched = scoped[scoped["strategy"].eq("nearest5_evt_phys")].set_index("site_id")
        common = sorted(set(all_source.index) & set(matched.index))
        if len(common) < 2:
            continue
        delta = (
            matched.loc[common, "rmse"].to_numpy()
            - all_source.loc[common, "rmse"].to_numpy()
        )
        improved = int(np.sum(delta < 0))
        jackknife = np.array([
            np.mean(np.delete(delta, index)) for index in range(len(delta))
        ])
        records.append({
            "subset": subset_name,
            "site_count": len(common),
            "sites": "+".join(common),
            "sites_with_lower_matched_rmse": improved,
            "mean_matched_minus_all_rmse": float(np.mean(delta)),
            "median_matched_minus_all_rmse": float(np.median(delta)),
            "exact_two_sided_sign_test_p": float(
                binomtest(improved, len(delta), 0.5, alternative="two-sided").pvalue
            ),
            "jackknife_min_mean_delta": float(jackknife.min()),
            "jackknife_max_mean_delta": float(jackknife.max()),
        })
    return pd.DataFrame(records)


def calibration_table(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (strategy, site), scoped in frame.groupby(["strategy", "site_id"], sort=True):
        observed = scoped["fhd_normal"].to_numpy(dtype=np.float64)
        predicted = scoped["prediction"].to_numpy(dtype=np.float64)
        slope, intercept = np.polyfit(predicted, observed, 1)
        observed_sd = float(np.std(observed, ddof=1))
        predicted_sd = float(np.std(predicted, ddof=1))
        centered_error = (predicted - predicted.mean()) - (observed - observed.mean())
        records.append({
            "strategy": strategy,
            "site_id": site,
            "rows": len(scoped),
            "mean_bias": float(np.mean(predicted - observed)),
            "observed_sd": observed_sd,
            "predicted_sd": predicted_sd,
            "predicted_to_observed_sd_ratio": predicted_sd / observed_sd,
            "calibration_intercept_observed_on_predicted": float(intercept),
            "calibration_slope_observed_on_predicted": float(slope),
            "centered_rmse": float(np.sqrt(np.mean(centered_error**2))),
            "spearman_r": float(spearmanr(observed, predicted).statistic),
        })
    return pd.DataFrame(records)


def association_table(frame: pd.DataFrame) -> pd.DataFrame:
    variables = [
        "sensitivity", "day_of_year", "terrain_slope_degrees", "solar_elevation"
    ]
    records = []
    unique = frame.drop_duplicates(KEYS)
    for site, scoped in unique.groupby("site_id", sort=True):
        for variable in variables:
            values = scoped[variable].to_numpy(dtype=np.float64)
            outcome = scoped["fhd_normal"].to_numpy(dtype=np.float64)
            coefficient = math.nan
            if np.ptp(values) > 1e-12:
                coefficient = float(spearmanr(values, outcome).statistic)
            records.append({
                "site_id": site,
                "variable": variable,
                "rows": len(scoped),
                "unique_values": int(scoped[variable].nunique()),
                "spearman_with_fhd": coefficient,
            })
    return pd.DataFrame(records)


def main() -> int:
    target_columns = [
        *KEYS, "acquisition_datetime", "sensitivity", "is_nighttime",
        "solar_elevation", "x_epsg5070", "y_epsg5070", "fhd_normal",
    ]
    targets = pd.read_parquet(TARGET_PATH, columns=target_columns)
    terrain = pd.read_parquet(
        CONVENTIONAL_PATH, columns=KEYS + ["terrain_slope_degrees"]
    )
    predictions = pd.read_parquet(PREDICTIONS_PATH)
    predictions = predictions[
        predictions["strategy"].isin(STRATEGIES)
        & predictions["method"].eq("tessera_area_ridge")
    ]
    frame = predictions.drop(columns=["fhd_normal"]).merge(
        targets, on=KEYS, validate="many_to_one"
    ).merge(terrain, on=KEYS, validate="many_to_one")
    frame = add_filter_flags(frame)

    metrics = metric_table(frame)
    paired = paired_table(metrics)
    calibration = calibration_table(frame)
    associations = association_table(frame)
    for table, path in [
        (metrics, METRICS_PATH),
        (paired, PAIRED_PATH),
        (calibration, CALIBRATION_PATH),
        (associations, ASSOCIATIONS_PATH),
    ]:
        write_csv(table, path)

    freeze = {
        "freeze_id": "retrospective-measurement-robustness-" + hashlib.sha256(
            metrics.to_csv(index=False).encode("utf-8")
        ).hexdigest()[:12],
        "created_utc": utc_now(),
        "status": "retrospective_evaluation_of_unchanged_frozen_predictions",
        "minimum_site_rows": MINIMUM_SITE_ROWS,
        "phenology_proxy": "day_of_year_122_through_274",
        "input_hashes": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [TARGET_PATH, CONVENTIONAL_PATH, PREDICTIONS_PATH]
        },
        "outputs": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [METRICS_PATH, PAIRED_PATH, CALIBRATION_PATH, ASSOCIATIONS_PATH]
        },
    }
    FREEZE_PATH.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n")
    print(metrics[metrics["scope"].eq("macro")].to_string(index=False))
    print("\nPaired matched-source effects")
    print(paired.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
