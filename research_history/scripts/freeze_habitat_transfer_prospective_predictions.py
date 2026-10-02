#!/usr/bin/env python3
"""Fit source-only models and freeze prospective predictions before outcome access."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_multisite_loso as loso  # noqa: E402
import run_habitat_transfer_habitat_conditioned_transfer as habitat  # noqa: E402


PROTOCOL_PATH = ROOT / "metadata/project_config_phase9_prospective_protocol_freeze.yaml"
HABITAT_FREEZE_PATH = ROOT / "metadata/phase9_landfire_habitat_freeze.json"
GEDI_FREEZE_PATH = ROOT / "metadata/phase9_prospective_gedi_freeze.json"
TESSERA_FREEZE_PATH = ROOT / "metadata/phase9_prospective_tessera_alignment_freeze.json"
DISTANCE_PATH = ROOT / "metadata/phase9_landfire_habitat_distances.csv"
TARGET_PATH = ROOT / "data/processed/phase9_prospective_tessera_aligned.parquet"
SOURCE_PATHS = [
    ROOT / "data/processed/tessera_aligned_development_soap_teak.parquet",
    ROOT / "data/processed/tessera_aligned_locked_bart.parquet",
    ROOT / "data/processed/phase7_tessera_aligned_expansion.parquet",
]

PREDICTIONS_PATH = ROOT / "data/processed/phase9_prospective_predictions_blind.parquet"
TUNING_PATH = ROOT / "outputs/tables/phase9_prospective_inner_tuning_blind.csv"
SELECTION_PATH = ROOT / "outputs/tables/phase9_prospective_source_selection_blind.csv"
FREEZE_PATH = ROOT / "metadata/phase9_prospective_prediction_freeze.json"

KEYS = ["site_id", "shot_number"]
FEATURES = [f"tessera_area_{index:03d}" for index in range(128)]
TARGET_READ_COLUMNS = KEYS + FEATURES
STRATEGIES = ["all_sources", "nearest5_evt_phys", "nearest5_evt_group"]


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


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def load_target_predictors(path: Path) -> pd.DataFrame:
    target = pd.read_parquet(path, columns=TARGET_READ_COLUMNS)
    if "fhd_normal" in target.columns:
        raise RuntimeError("Prospective target FHD entered the predictor table")
    if target.duplicated(KEYS).any():
        raise RuntimeError("Prospective target predictor keys are not unique")
    return target


def tune_source_only(
    training: pd.DataFrame,
    source_sites: list[str],
    target_site: str,
    strategy: str,
    alphas: list[float],
) -> tuple[float, pd.DataFrame]:
    selected, tuning = habitat.tune_alpha(
        training,
        source_sites,
        target_site,
        strategy,
        "prospective_blind",
        alphas,
    )
    return selected, tuning


def main() -> int:
    protected = [PREDICTIONS_PATH, TUNING_PATH, SELECTION_PATH, FREEZE_PATH]
    existing = [str(path.relative_to(ROOT)) for path in protected if path.exists()]
    if existing:
        raise RuntimeError(f"Prospective prediction freeze exists; refusing to overwrite: {existing}")

    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))[
        "phase9_prospective_habitat_transfer"
    ]
    habitat_freeze = json.loads(HABITAT_FREEZE_PATH.read_text(encoding="utf-8"))
    gedi_freeze = json.loads(GEDI_FREEZE_PATH.read_text(encoding="utf-8"))
    tessera_freeze = json.loads(TESSERA_FREEZE_PATH.read_text(encoding="utf-8"))
    source_sites = sorted(protocol["source_sites"])
    target_sites = list(gedi_freeze["freeze_basis"]["viable_sites"])
    if sorted(tessera_freeze["freeze_basis"]["sites"]) != sorted(target_sites):
        raise RuntimeError("TESSERA and GEDI prospective site freezes differ")

    sources = pd.concat(
        [pd.read_parquet(path, columns=KEYS + ["fhd_normal"] + FEATURES) for path in SOURCE_PATHS],
        ignore_index=True,
    )
    if sorted(sources["site_id"].unique()) != source_sites or sources.duplicated(KEYS).any():
        raise RuntimeError("Unexpected source training population")
    targets = load_target_predictors(TARGET_PATH)
    if sorted(targets["site_id"].unique()) != sorted(target_sites):
        raise RuntimeError("Unexpected prospective target predictor population")
    if not np.isfinite(sources[["fhd_normal", *FEATURES]].to_numpy(dtype=np.float64)).all():
        raise RuntimeError("Non-finite source training values")
    if not np.isfinite(targets[FEATURES].to_numpy(dtype=np.float64)).all():
        raise RuntimeError("Non-finite prospective predictors")

    distances = pd.read_csv(DISTANCE_PATH)
    alphas = [float(value) for value in protocol["locked_prediction"]["alpha_grid"]]
    prediction_frames: list[pd.DataFrame] = []
    tuning_frames: list[pd.DataFrame] = []
    selection_records: list[dict[str, Any]] = []
    for target_site in target_sites:
        target = targets[targets["site_id"].eq(target_site)].copy()
        for strategy in STRATEGIES:
            selected_sources = habitat.source_sites_for_strategy(
                distances,
                target_site,
                source_sites,
                strategy,
                5,
            )
            training = sources[sources["site_id"].isin(selected_sources)].copy()
            selected_alpha, tuning = tune_source_only(
                training,
                selected_sources,
                target_site,
                strategy,
                alphas,
            )
            tuning_frames.append(tuning)
            weights = loso.equal_site_weights(training["site_id"].to_numpy())
            scaler, model = loso.fit_weighted_ridge(
                training[FEATURES].to_numpy(dtype=np.float64),
                training["fhd_normal"].to_numpy(dtype=np.float64),
                selected_alpha,
                weights,
            )
            predicted = target[KEYS].copy()
            predicted["strategy"] = strategy
            predicted["method"] = "tessera_area_ridge"
            predicted["training_sites"] = "+".join(selected_sources)
            predicted["selected_alpha"] = selected_alpha
            predicted["prediction"] = model.predict(
                scaler.transform(target[FEATURES].to_numpy(dtype=np.float64))
            )
            prediction_frames.append(predicted)
            hierarchy = strategy.removeprefix("nearest5_") if strategy != "all_sources" else None
            for source_site in selected_sources:
                scoped = distances[
                    distances["target_site"].eq(target_site)
                    & distances["source_site"].eq(source_site)
                    & distances["hierarchy"].eq(hierarchy)
                ] if hierarchy else pd.DataFrame()
                selection_records.append(
                    {
                        "target_site": target_site,
                        "strategy": strategy,
                        "source_site": source_site,
                        "habitat_hierarchy": hierarchy,
                        "habitat_distance": float(scoped.iloc[0]["distance"]) if len(scoped) else np.nan,
                        "source_rank": int(scoped.iloc[0]["source_rank"]) if len(scoped) else np.nan,
                        "source_rows": int(training["site_id"].eq(source_site).sum()),
                        "source_total_fit_weight": float(
                            weights[training["site_id"].eq(source_site).to_numpy()].sum()
                        ),
                        "selected_alpha": selected_alpha,
                    }
                )
            print(
                f"{target_site} {strategy}: alpha={selected_alpha:g}; sources={'+'.join(selected_sources)}",
                flush=True,
            )

    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["site_id", "strategy", "shot_number"]
    )
    tuning = pd.concat(tuning_frames, ignore_index=True)
    selections = pd.DataFrame(selection_records)
    if "fhd_normal" in predictions.columns or len(predictions) != len(targets) * len(STRATEGIES):
        raise RuntimeError("Prospective prediction table is invalid")

    write_parquet(predictions, PREDICTIONS_PATH)
    write_csv(tuning, TUNING_PATH)
    write_csv(selections, SELECTION_PATH)
    freeze_basis = {
        "protocol_snapshot_sha256": sha256(PROTOCOL_PATH),
        "habitat_freeze_id": habitat_freeze["freeze_id"],
        "gedi_freeze_id": gedi_freeze["freeze_id"],
        "tessera_alignment_id": tessera_freeze["alignment_id"],
        "prediction_script_sha256": sha256(Path(__file__).resolve()),
        "sklearn_version": sklearn.__version__,
        "source_input_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in SOURCE_PATHS},
        "target_predictor_sha256": sha256(TARGET_PATH),
        "target_columns_read": TARGET_READ_COLUMNS,
        "target_outcome_columns_read": [],
        "target_outcomes_opened": False,
        "source_sites": source_sites,
        "target_sites": target_sites,
        "strategies": STRATEGIES,
        "features": FEATURES,
        "model": "standardized_equal-site-weighted_ridge",
        "alpha_grid": alphas,
        "inner_tuning": "leave_one_selected_source_site_out_minimax_rmse",
        "rows_per_strategy": len(targets),
        "source_selection": {
            f"{target}/{strategy}": selections[
                selections["target_site"].eq(target) & selections["strategy"].eq(strategy)
            ]["source_site"].tolist()
            for target in target_sites
            for strategy in STRATEGIES
        },
    }
    prediction_id = "phase9-prospective-prediction-" + canonical_hash(freeze_basis)[:12]
    freeze = {
        "prediction_id": prediction_id,
        "created_utc": utc_now(),
        "status": "frozen_before_target_evaluation",
        "freeze_basis": freeze_basis,
        "outputs": {
            "predictions": {
                "path": str(PREDICTIONS_PATH.relative_to(ROOT)),
                "rows": len(predictions),
                "sha256": sha256(PREDICTIONS_PATH),
            },
            "inner_tuning": {
                "path": str(TUNING_PATH.relative_to(ROOT)),
                "rows": len(tuning),
                "sha256": sha256(TUNING_PATH),
            },
            "source_selection": {
                "path": str(SELECTION_PATH.relative_to(ROOT)),
                "rows": len(selections),
                "sha256": sha256(SELECTION_PATH),
            },
        },
    }
    write_json(FREEZE_PATH, freeze)
    print(
        json.dumps(
            {
                "prediction_id": prediction_id,
                "target_sites": target_sites,
                "target_rows": len(targets),
                "prediction_rows": len(predictions),
                "target_outcomes_opened": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
