#!/usr/bin/env python3
"""Diagnose the frozen Phase 11 context-height transfer result."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_context_height_footprint_height_transfer as base  # noqa: E402


CONTEXT_FREEZE_PATH = ROOT / "metadata/phase11_context_height_freeze.json"
CONTEXT_PREDICTIONS_PATH = ROOT / "data/processed/phase11_context_height_predictions.parquet"
FOOTPRINT_PREDICTIONS_PATH = ROOT / "data/processed/phase11_footprint_height_predictions.parquet"
DECOMPOSITION_PATH = ROOT / "outputs/tables/phase11_context_height_error_decomposition.csv"
PAIRED_SITE_PATH = ROOT / "outputs/tables/phase11_context_height_paired_sites.csv"
BOOTSTRAP_PATH = ROOT / "outputs/tables/phase11_context_height_paired_bootstrap.csv"
FREEZE_PATH = ROOT / "metadata/phase11_context_height_diagnostic_freeze.json"

MODEL_NAMES = [
    "tessera_footprint_ridge",
    "tessera_context_ridge",
    "tessera_context_hgb",
]
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 20260721


def offset_corrected_predictions(observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Apply an oracle forest-mean correction for diagnosis, never deployment."""
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    return predicted - float(np.mean(predicted - observed))


def assemble_predictions() -> pd.DataFrame:
    context = pd.read_parquet(CONTEXT_PREDICTIONS_PATH)
    context = context[context["model"].isin(["tessera_context_ridge", "tessera_context_hgb"])]
    footprint = pd.read_parquet(FOOTPRINT_PREDICTIONS_PATH)
    footprint = footprint[footprint["model"].eq("tessera_area_ridge")].copy()
    footprint["model"] = "tessera_footprint_ridge"
    columns = [*base.KEYS, "observed_rh100_m", "model", "prediction"]
    combined = pd.concat([footprint[columns], context[columns]], ignore_index=True)
    expected_rows = 15_326 * len(MODEL_NAMES)
    if len(combined) != expected_rows or combined[[*base.KEYS, "model"]].duplicated().any():
        raise RuntimeError("Phase 11 diagnostic prediction population changed")
    if sorted(combined["model"].unique()) != sorted(MODEL_NAMES):
        raise RuntimeError("Phase 11 diagnostic models changed")
    return combined


def error_decomposition(predictions: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for model, model_frame in predictions.groupby("model", sort=True):
        site_records = []
        for site, frame in model_frame.groupby("site_id", sort=True):
            observed = frame["observed_rh100_m"].to_numpy(dtype=np.float64)
            predicted = frame["prediction"].to_numpy(dtype=np.float64)
            corrected = offset_corrected_predictions(observed, predicted)
            raw = base.metric_values(observed, predicted)
            centered = base.metric_values(observed, corrected)
            row = {
                "model": model,
                "scope": "site",
                "site_id": site,
                "rows": len(frame),
                "site_mean_error_m": raw["mean_bias"],
                "absolute_site_mean_error_m": abs(raw["mean_bias"]),
                **{f"raw_{key}": value for key, value in raw.items()},
                **{f"known_mean_corrected_{key}": value for key, value in centered.items()},
            }
            records.append(row)
            site_records.append(row)
        sites = pd.DataFrame(site_records)
        metric_columns = [
            column
            for column in sites.columns
            if column not in {"model", "scope", "site_id", "rows"}
        ]
        records.append(
            {
                "model": model,
                "scope": "macro",
                "site_id": "macro_mean",
                "rows": int(sites["rows"].sum()),
                **{column: float(sites[column].mean()) for column in metric_columns},
            }
        )
    return pd.DataFrame(records)


def paired_site_table(decomposition: pd.DataFrame) -> pd.DataFrame:
    sites = decomposition[decomposition["scope"].eq("site")].copy()
    metrics = ["raw_rmse", "raw_r2", "raw_spearman_r", "absolute_site_mean_error_m"]
    reference = sites[sites["model"].eq("tessera_footprint_ridge")].set_index("site_id")
    records = []
    for model in ["tessera_context_ridge", "tessera_context_hgb"]:
        candidate = sites[sites["model"].eq(model)].set_index("site_id")
        for site in sorted(reference.index):
            record: dict[str, Any] = {
                "comparison": f"{model}_minus_tessera_footprint_ridge",
                "candidate_model": model,
                "reference_model": "tessera_footprint_ridge",
                "site_id": site,
            }
            for metric in metrics:
                record[f"candidate_{metric}"] = float(candidate.loc[site, metric])
                record[f"reference_{metric}"] = float(reference.loc[site, metric])
                record[f"delta_{metric}"] = float(
                    candidate.loc[site, metric] - reference.loc[site, metric]
                )
            records.append(record)
    return pd.DataFrame(records)


def paired_bootstrap(paired: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    records = []
    directions = {
        "raw_rmse": "lower",
        "raw_r2": "higher",
        "raw_spearman_r": "higher",
        "absolute_site_mean_error_m": "lower",
    }
    for comparison, frame in paired.groupby("comparison", sort=True):
        for metric, beneficial_direction in directions.items():
            differences = frame[f"delta_{metric}"].to_numpy(dtype=np.float64)
            draws = np.mean(
                rng.choice(
                    differences,
                    size=(BOOTSTRAP_REPLICATES, len(differences)),
                    replace=True,
                ),
                axis=1,
            )
            beneficial = draws < 0 if beneficial_direction == "lower" else draws > 0
            site_beneficial = differences < 0 if beneficial_direction == "lower" else differences > 0
            records.append(
                {
                    "comparison": comparison,
                    "metric": metric,
                    "beneficial_direction": beneficial_direction,
                    "sites": len(differences),
                    "sites_improved": int(np.sum(site_beneficial)),
                    "mean_paired_difference": float(np.mean(differences)),
                    "site_bootstrap_95_low": float(np.quantile(draws, 0.025)),
                    "site_bootstrap_95_high": float(np.quantile(draws, 0.975)),
                    "bootstrap_probability_beneficial": float(np.mean(beneficial)),
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    "bootstrap_seed": BOOTSTRAP_SEED,
                }
            )
    return pd.DataFrame(records)


def main() -> int:
    protected = [DECOMPOSITION_PATH, PAIRED_SITE_PATH, BOOTSTRAP_PATH, FREEZE_PATH]
    existing = [str(path.relative_to(ROOT)) for path in protected if path.exists()]
    if existing:
        raise RuntimeError(f"Phase 11 context-height diagnostics already exist: {existing}")
    context_freeze = json.loads(CONTEXT_FREEZE_PATH.read_text(encoding="utf-8"))
    if context_freeze["freeze_id"] != "phase11-context-height-9700619df2d4":
        raise RuntimeError("Unexpected Phase 11 context-height freeze")
    for path in [CONTEXT_PREDICTIONS_PATH]:
        expected = context_freeze["outputs"][str(path.relative_to(ROOT))]
        if base.sha256(path) != expected:
            raise RuntimeError("Frozen context-height predictions changed")

    predictions = assemble_predictions()
    decomposition = error_decomposition(predictions)
    paired = paired_site_table(decomposition)
    bootstrap = paired_bootstrap(paired)
    base.write_csv(decomposition, DECOMPOSITION_PATH)
    base.write_csv(paired, PAIRED_SITE_PATH)
    base.write_csv(bootstrap, BOOTSTRAP_PATH)
    freeze_basis = {
        "analysis_role": "post_outcome_diagnostic_not_model_selection",
        "context_height_freeze_id": context_freeze["freeze_id"],
        "script_sha256": base.sha256(Path(__file__).resolve()),
        "input_hashes": {
            str(path.relative_to(ROOT)): base.sha256(path)
            for path in [CONTEXT_PREDICTIONS_PATH, FOOTPRINT_PREDICTIONS_PATH]
        },
        "models": MODEL_NAMES,
        "sites": sorted(predictions["site_id"].unique()),
        "rows_per_model": 15_326,
        "oracle_mean_correction_uses_target_site_height": True,
        "oracle_mean_correction_is_deployable": False,
        "bootstrap_unit": "complete_site",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "fhd_normal_used": False,
    }
    freeze = {
        "freeze_id": "phase11-context-height-diagnostic-" + base.canonical_hash(freeze_basis)[:12],
        "created_utc": base.utc_now(),
        "status": "complete_post_outcome_diagnostic",
        "freeze_basis": freeze_basis,
        "outputs": {
            str(path.relative_to(ROOT)): base.sha256(path)
            for path in [DECOMPOSITION_PATH, PAIRED_SITE_PATH, BOOTSTRAP_PATH]
        },
    }
    base.write_json(FREEZE_PATH, freeze)
    macro = decomposition[decomposition["scope"].eq("macro")].set_index("model")
    print(
        json.dumps(
            {
                "freeze_id": freeze["freeze_id"],
                "macro_error_decomposition": macro[
                    [
                        "raw_rmse",
                        "known_mean_corrected_rmse",
                        "raw_r2",
                        "known_mean_corrected_r2",
                        "raw_spearman_r",
                        "absolute_site_mean_error_m",
                    ]
                ].to_dict(orient="index"),
                "rmse_bootstrap": bootstrap[bootstrap["metric"].eq("raw_rmse")].to_dict(orient="records"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
