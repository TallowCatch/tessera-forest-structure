#!/usr/bin/env python3
"""Freeze expansion-site predictions before reading any expansion FHD outcomes."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_multisite_loso as loso  # noqa: E402


CONFIG_PATH = ROOT / "configs/project.yaml"
PROTOCOL_PATH = ROOT / "metadata/project_config_phase7_site_expansion_protocol_freeze.yaml"
ORIGINAL_DEVELOPMENT_PATH = ROOT / "data/processed/tessera_aligned_development_soap_teak.parquet"
ORIGINAL_BART_PATH = ROOT / "data/processed/tessera_aligned_locked_bart.parquet"
EXPANSION_PATH = ROOT / "data/processed/phase7_tessera_aligned_expansion.parquet"
EXPANSION_ALIGNMENT_FREEZE_PATH = ROOT / "metadata/phase7_tessera_alignment_freeze.json"
PREDICTIONS_PATH = ROOT / "data/processed/phase7_prospective_target_free_predictions.parquet"
TUNING_PATH = ROOT / "outputs/tables/phase7_prospective_original_site_tuning.csv"
MODEL_PATH = ROOT / "outputs/models/phase7_prospective_original_site_model.joblib"
FREEZE_PATH = ROOT / "metadata/phase7_prospective_prediction_freeze.json"
FEATURES = [f"tessera_area_{index:03d}" for index in range(128)]
KEYS = ["site_id", "shot_number"]


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


def main() -> int:
    protected = [PREDICTIONS_PATH, TUNING_PATH, MODEL_PATH, FREEZE_PATH]
    existing = [str(path.relative_to(ROOT)) for path in protected if path.exists()]
    if existing:
        raise RuntimeError(f"Prospective prediction freeze exists; refusing to overwrite: {existing}")
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))["phase7_site_expansion"]
    evaluation = protocol["prospective_external_evaluation"]
    original_sites = list(evaluation["training_sites"])
    original = pd.concat([
        pd.read_parquet(ORIGINAL_DEVELOPMENT_PATH, columns=KEYS + ["fhd_normal"] + FEATURES),
        pd.read_parquet(ORIGINAL_BART_PATH, columns=KEYS + ["fhd_normal"] + FEATURES),
    ], ignore_index=True)
    expansion_columns = KEYS + FEATURES
    expansion = pd.read_parquet(EXPANSION_PATH, columns=expansion_columns)
    if "fhd_normal" in expansion.columns:
        raise RuntimeError("Expansion FHD entered the target-free prediction stage")
    alignment_freeze = json.loads(EXPANSION_ALIGNMENT_FREEZE_PATH.read_text(encoding="utf-8"))
    expansion_sites = list(alignment_freeze["freeze_basis"]["sites"])
    if set(original["site_id"].unique()) != set(original_sites):
        raise RuntimeError("Original training population changed")
    if set(expansion["site_id"].unique()) != set(expansion_sites):
        raise RuntimeError("Expansion predictor population changed")

    alphas = [float(value) for value in evaluation["alpha_grid"]]
    tuning_records: list[dict[str, Any]] = []
    for alpha in alphas:
        for held_out in original_sites:
            training = original[~original["site_id"].eq(held_out)]
            validation = original[original["site_id"].eq(held_out)]
            weights = loso.equal_site_weights(training["site_id"].to_numpy())
            scaler, model = loso.fit_weighted_ridge(
                training[FEATURES].to_numpy(dtype=np.float64),
                training["fhd_normal"].to_numpy(dtype=np.float64),
                alpha,
                weights,
            )
            prediction = model.predict(
                scaler.transform(validation[FEATURES].to_numpy(dtype=np.float64))
            )
            tuning_records.append({
                "alpha": alpha,
                "held_out_original_site": held_out,
                "training_sites": "+".join(sorted(set(original_sites) - {held_out})),
                "training_rows": len(training),
                "validation_rows": len(validation),
                **loso.metric_values(
                    validation["fhd_normal"].to_numpy(dtype=np.float64), prediction
                ),
            })
    tuning = pd.DataFrame(tuning_records)
    selected_alpha, ranking = loso.select_alpha(tuning)
    tuning["selected_alpha"] = selected_alpha

    weights = loso.equal_site_weights(original["site_id"].to_numpy())
    scaler, model = loso.fit_weighted_ridge(
        original[FEATURES].to_numpy(dtype=np.float64),
        original["fhd_normal"].to_numpy(dtype=np.float64),
        selected_alpha,
        weights,
    )
    prediction = model.predict(scaler.transform(expansion[FEATURES].to_numpy(dtype=np.float64)))
    baseline_value = float(original.groupby("site_id")["fhd_normal"].mean().mean())
    model_predictions = expansion[KEYS].copy()
    model_predictions["method"] = "tessera_area_ridge"
    model_predictions["prediction"] = prediction
    baseline = expansion[KEYS].copy()
    baseline["method"] = "original_site_mean"
    baseline["prediction"] = baseline_value
    predictions = pd.concat([baseline, model_predictions], ignore_index=True).sort_values(
        ["site_id", "method", "shot_number"]
    ).reset_index(drop=True)

    loso.write_parquet(predictions, PREDICTIONS_PATH)
    loso.write_csv(tuning, TUNING_PATH)
    loso.write_joblib({
        "features": FEATURES,
        "selected_alpha": selected_alpha,
        "scaler": scaler,
        "model": model,
        "baseline_value": baseline_value,
        "training_sites": original_sites,
        "training_site_weight_totals": {
            site: float(weights[original["site_id"].eq(site).to_numpy()].sum())
            for site in original_sites
        },
        "expansion_target_columns": expansion_columns,
        "expansion_target_outcome_present": False,
    }, MODEL_PATH)
    outputs = {
        "predictions": {"path": str(PREDICTIONS_PATH.relative_to(ROOT)), "rows": len(predictions), "sha256": sha256(PREDICTIONS_PATH)},
        "tuning": {"path": str(TUNING_PATH.relative_to(ROOT)), "rows": len(tuning), "sha256": sha256(TUNING_PATH)},
        "model": {"path": str(MODEL_PATH.relative_to(ROOT)), "sha256": sha256(MODEL_PATH)},
    }
    freeze_basis = {
        "protocol_snapshot_sha256": sha256(PROTOCOL_PATH),
        "expansion_alignment_id": alignment_freeze["alignment_id"],
        "expansion_predictor_sha256": sha256(EXPANSION_PATH),
        "original_input_hashes": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [ORIGINAL_DEVELOPMENT_PATH, ORIGINAL_BART_PATH]
        },
        "training_sites": original_sites,
        "training_rows": len(original),
        "expansion_sites": expansion_sites,
        "expansion_rows": len(expansion),
        "expansion_columns_read": expansion_columns,
        "expansion_columns_contain_fhd_normal": False,
        "expansion_target_labels_used_for_tuning_training_or_selection": False,
        "features": FEATURES,
        "alpha_grid": alphas,
        "alpha_selection": evaluation["alpha_selection"],
        "selected_alpha": selected_alpha,
        "selection_ranking": ranking.to_dict(orient="records"),
        "equal_total_training_weight_per_original_site": True,
        "baseline_value": baseline_value,
        "software": {"numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
    }
    freeze_id = "phase7-prospective-predictions-" + canonical_hash(
        {"freeze_basis": freeze_basis, "outputs": outputs}
    )[:12]
    write_json(FREEZE_PATH, {
        "freeze_id": freeze_id,
        "created_utc": utc_now(),
        "freeze_basis": freeze_basis,
        "outputs": outputs,
    })
    print(json.dumps({
        "freeze_id": freeze_id,
        "selected_alpha": selected_alpha,
        "training_rows": len(original),
        "expansion_rows": len(expansion),
        "expansion_sites": expansion_sites,
        "expansion_labels_used": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
