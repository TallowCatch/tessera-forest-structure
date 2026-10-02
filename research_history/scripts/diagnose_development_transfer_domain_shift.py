#!/usr/bin/env python3
"""Record post-hoc feature-domain diagnostics without changing locked BART metrics."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT_PATH = ROOT / "data/processed/tessera_aligned_development_soap_teak.parquet"
BART_PATH = ROOT / "data/processed/tessera_aligned_locked_bart.parquet"
DEVELOPMENT_CONVENTIONAL_PATH = ROOT / "data/processed/phase3_conventional_predictors_development.parquet"
BART_CONVENTIONAL_PATH = ROOT / "data/processed/phase4_conventional_predictors_bart.parquet"
CONVENTIONAL_FREEZE_PATH = ROOT / "metadata/phase3_conventional_predictor_freeze.json"
TRANSFER_FREEZE_PATH = ROOT / "metadata/phase4_development_transfer_freeze.json"
EVALUATION_FREEZE_PATH = ROOT / "metadata/phase4_locked_bart_evaluation_freeze.json"

SITE_PATH = ROOT / "outputs/tables/phase4_posthoc_site_distributions.csv"
SHIFT_PATH = ROOT / "outputs/tables/phase4_posthoc_feature_domain_shift.csv"
CONTRIBUTION_PATH = ROOT / "outputs/tables/phase4_posthoc_sentinel_ridge_contributions.csv"
MANIFEST_PATH = ROOT / "metadata/phase4_posthoc_domain_shift_diagnostics.json"


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


def main() -> int:
    protected = [SITE_PATH, SHIFT_PATH, CONTRIBUTION_PATH, MANIFEST_PATH]
    if any(path.exists() for path in protected):
        raise RuntimeError("Post-hoc diagnostic freeze already exists; refusing to overwrite")

    conventional_manifest = json.loads(CONVENTIONAL_FREEZE_PATH.read_text(encoding="utf-8"))
    transfer = json.loads(TRANSFER_FREEZE_PATH.read_text(encoding="utf-8"))
    evaluation = json.loads(EVALUATION_FREEZE_PATH.read_text(encoding="utf-8"))
    features = conventional_manifest["freeze_basis"]["feature_columns"]
    area = [f"tessera_area_{index:03d}" for index in range(128)]

    development = pd.read_parquet(DEVELOPMENT_PATH).merge(
        pd.read_parquet(DEVELOPMENT_CONVENTIONAL_PATH)[["site_id", "shot_number", *features]],
        on=["site_id", "shot_number"],
        validate="one_to_one",
    )
    bart = pd.read_parquet(BART_PATH).merge(
        pd.read_parquet(BART_CONVENTIONAL_PATH)[["site_id", "shot_number", *features]],
        on=["site_id", "shot_number"],
        validate="one_to_one",
    )

    site_rows: list[dict[str, Any]] = []
    for site, frame in [("SOAP", development[development["site_id"].eq("SOAP")]),
                        ("TEAK", development[development["site_id"].eq("TEAK")]),
                        ("BART", bart)]:
        for variable in ["fhd_normal", "terrain_elevation_m", "terrain_slope_degrees"]:
            values = frame[variable]
            site_rows.append(
                {
                    "site_id": site,
                    "variable": variable,
                    "rows": len(values),
                    "minimum": float(values.min()),
                    "p10": float(values.quantile(0.1)),
                    "median": float(values.median()),
                    "mean": float(values.mean()),
                    "p90": float(values.quantile(0.9)),
                    "maximum": float(values.max()),
                }
            )
    site_summary = pd.DataFrame(site_rows)

    shift_rows: list[dict[str, Any]] = []
    for feature in [*features, *area]:
        development_values = development[feature]
        bart_values = bart[feature]
        standard_deviation = float(development_values.std(ddof=0))
        standardized_mean_shift = (
            float((bart_values.mean() - development_values.mean()) / standard_deviation)
            if standard_deviation > 0
            else np.nan
        )
        shift_rows.append(
            {
                "feature": feature,
                "feature_family": "tessera" if feature.startswith("tessera_") else "conventional",
                "development_mean": float(development_values.mean()),
                "development_std": standard_deviation,
                "development_minimum": float(development_values.min()),
                "development_maximum": float(development_values.max()),
                "bart_mean": float(bart_values.mean()),
                "bart_minimum": float(bart_values.min()),
                "bart_maximum": float(bart_values.max()),
                "standardized_mean_shift": standardized_mean_shift,
                "bart_fraction_outside_development_range": float(
                    (
                        (bart_values < development_values.min())
                        | (bart_values > development_values.max())
                    ).mean()
                ),
            }
        )
    shifts = pd.DataFrame(shift_rows).sort_values(
        "standardized_mean_shift", key=lambda values: values.abs(), ascending=False
    )

    alpha = float(
        transfer["freeze_basis"]["combined_development_alphas"]["sentinel_topography_ridge"]
    )
    model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha))])
    model.fit(
        development[features].to_numpy(dtype=np.float32),
        development["fhd_normal"].to_numpy(dtype=np.float64),
    )
    standardized_bart = model["scale"].transform(bart[features].to_numpy(dtype=np.float32))
    contributions = standardized_bart * model["ridge"].coef_
    contribution_table = pd.DataFrame(
        {
            "feature": features,
            "ridge_coefficient": model["ridge"].coef_,
            "bart_mean_standardized_value": standardized_bart.mean(axis=0),
            "bart_mean_prediction_contribution": contributions.mean(axis=0),
            "bart_mean_absolute_prediction_contribution": np.abs(contributions).mean(axis=0),
        }
    ).sort_values("bart_mean_absolute_prediction_contribution", ascending=False)

    count_feature = "s1_descending_valid_observation_count"
    count_shift = shifts[shifts["feature"].eq(count_feature)].iloc[0]
    count_contribution = contribution_table[
        contribution_table["feature"].eq(count_feature)
    ].iloc[0]
    tessera_shifts = shifts[shifts["feature_family"].eq("tessera")][
        "standardized_mean_shift"
    ].abs()
    diagnostic = {
        "status": "frozen_posthoc_after_locked_bart_evaluation",
        "created_at": utc_now(),
        "locked_evaluation_freeze_id": evaluation["freeze_id"],
        "locked_metrics_modified": False,
        "model_selection_or_tuning_performed": False,
        "findings": {
            "bart_elevation_range_m": [
                float(bart["terrain_elevation_m"].min()),
                float(bart["terrain_elevation_m"].max()),
            ],
            "development_elevation_range_m": [
                float(development["terrain_elevation_m"].min()),
                float(development["terrain_elevation_m"].max()),
            ],
            "bart_elevation_fraction_outside_development_range": float(
                (
                    (bart["terrain_elevation_m"] < development["terrain_elevation_m"].min())
                    | (bart["terrain_elevation_m"] > development["terrain_elevation_m"].max())
                ).mean()
            ),
            "median_absolute_tessera_standardized_mean_shift": float(tessera_shifts.median()),
            "maximum_absolute_tessera_standardized_mean_shift": float(tessera_shifts.max()),
            "sentinel_count_feature": count_feature,
            "sentinel_count_standardized_mean_shift": float(count_shift["standardized_mean_shift"]),
            "sentinel_count_mean_prediction_contribution": float(
                count_contribution["bart_mean_prediction_contribution"]
            ),
            "sentinel_baseline_valid_for_scientific_comparison": False,
            "sentinel_invalidity_reason": (
                "Acquisition observation count was included as a predictor; near-zero development "
                "variance caused extreme BART extrapolation."
            ),
        },
    }
    diagnostic["diagnostic_sha256"] = canonical_hash(diagnostic)
    write_csv(site_summary, SITE_PATH)
    write_csv(shifts, SHIFT_PATH)
    write_csv(contribution_table, CONTRIBUTION_PATH)
    diagnostic["outputs"] = {
        "site_distributions": {"path": str(SITE_PATH.relative_to(ROOT)), "sha256": sha256(SITE_PATH)},
        "feature_domain_shift": {"path": str(SHIFT_PATH.relative_to(ROOT)), "sha256": sha256(SHIFT_PATH)},
        "sentinel_contributions": {"path": str(CONTRIBUTION_PATH.relative_to(ROOT)), "sha256": sha256(CONTRIBUTION_PATH)},
    }
    write_json(diagnostic, MANIFEST_PATH)
    print(json.dumps(diagnostic["findings"], indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
