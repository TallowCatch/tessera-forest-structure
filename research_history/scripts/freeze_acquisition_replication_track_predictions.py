#!/usr/bin/env python3
"""Freeze Phase 13 source predictions, folds, and local-shot samples before outcomes."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_track_assisted_track_budget as budget_tools  # noqa: E402
import build_fair_baselines_conventional_predictors as phase8  # noqa: E402
import freeze_track_assisted_splits as split_tools  # noqa: E402
import run_context_height_footprint_height_transfer as common  # noqa: E402


PROTOCOL_PATH = ROOT / "metadata/project_config_phase13_track_replication_protocol_freeze.yaml"
TARGET_FREEZE_PATH = ROOT / "metadata/phase13_combined_target_free_freeze.json"
TESSERA_FREEZE_PATH = ROOT / "metadata/phase13_tessera_alignment_freeze.json"
CONVENTIONAL_FREEZE_PATH = ROOT / "metadata/phase13_conventional_freeze.json"
FOLD_PATH = ROOT / "metadata/phase13_combined_target_free_fold_summary.csv"
TARGET_TESSERA_PATH = ROOT / "data/processed/phase13_tessera_aligned.parquet"
TARGET_CONVENTIONAL_PATH = ROOT / "data/processed/phase13_conventional_predictors.parquet"
PREDICTIONS_PATH = ROOT / "data/processed/phase13_frozen_source_predictions.parquet"
LOCAL_SAMPLES_PATH = ROOT / "data/processed/phase13_frozen_local_samples.parquet"
TUNING_PATH = ROOT / "outputs/tables/phase13_source_model_tuning.csv"
SELECTION_PATH = ROOT / "outputs/tables/phase13_source_model_selection.csv"
FREEZE_PATH = ROOT / "metadata/phase13_prediction_freeze.json"

SOURCE_TESSERA_PATHS = [
    ROOT / "data/processed/tessera_aligned_development_soap_teak.parquet",
    ROOT / "data/processed/tessera_aligned_locked_bart.parquet",
    ROOT / "data/processed/phase7_tessera_aligned_expansion.parquet",
    ROOT / "data/processed/phase9_prospective_tessera_aligned.parquet",
]
SOURCE_CONVENTIONAL_PATHS = [
    ROOT / "data/processed/phase8_conventional_predictors_all_sites.parquet",
    ROOT / "data/processed/phase9_prospective_conventional_predictors.parquet",
]
KEYS = ["site_id", "shot_number"]
TESSERA_FEATURES = [f"tessera_area_{index:03d}" for index in range(128)]
CONVENTIONAL_FEATURES = phase8.TERRAIN_FEATURES + phase8.S2_FEATURES


def tune_source(
    source: pd.DataFrame, features: list[str], alphas: list[float], model_name: str
) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    for held_out in sorted(source["site_id"].unique()):
        training = source[~source["site_id"].eq(held_out)]
        validation = source[source["site_id"].eq(held_out)]
        for alpha in alphas:
            scaler, model = common.fit_ridge(training, features, "fhd_normal", alpha)
            predicted = model.predict(
                scaler.transform(validation[features].to_numpy(dtype=np.float64))
            )
            records.append(
                {
                    "model": model_name,
                    "held_out_source_site": held_out,
                    "alpha": alpha,
                    "training_rows": len(training),
                    "validation_rows": len(validation),
                    **common.metric_values(
                        validation["fhd_normal"].to_numpy(dtype=np.float64), predicted
                    ),
                }
            )
    tuning = pd.DataFrame(records)
    selected, ranking = common.select_alpha(tuning)
    tuning["selected_alpha"] = selected
    ranking.insert(0, "model", model_name)
    ranking["selected_alpha"] = selected
    ranking["selected"] = np.isclose(ranking["alpha"], selected)
    return selected, tuning, ranking


def verify_fold(
    target: pd.DataFrame, fold: pd.Series, buffer_m: float
) -> tuple[np.ndarray, np.ndarray]:
    coordinates = target[["x_epsg5070", "y_epsg5070"]].to_numpy(dtype=np.float64)
    identifiers = target["pass_id"].to_numpy(dtype=str)
    local, test, _ = split_tools.buffered_local_split(
        coordinates, identifiers, str(fold["held_out_pass"]), buffer_m
    )
    if len(test) != int(fold["test_rows"]) or len(local) != int(fold["local_rows_after_buffer"]):
        raise RuntimeError("Phase 13 fold row counts changed")
    if split_tools.shot_hash(target.iloc[test]["shot_number"].to_numpy()) != fold["test_shot_sha256"]:
        raise RuntimeError("Phase 13 test shot set changed")
    if split_tools.shot_hash(target.iloc[local]["shot_number"].to_numpy()) != fold["local_shot_sha256"]:
        raise RuntimeError("Phase 13 local shot set changed")
    return local, test


def main() -> int:
    protected = [PREDICTIONS_PATH, LOCAL_SAMPLES_PATH, TUNING_PATH, SELECTION_PATH, FREEZE_PATH]
    existing = [str(path.relative_to(ROOT)) for path in protected if path.exists()]
    if existing:
        raise RuntimeError(f"Phase 13 prediction outputs exist; refusing to overwrite: {existing}")
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))[
        "phase13_track_assisted_replication"
    ]
    target_freeze = json.loads(TARGET_FREEZE_PATH.read_text(encoding="utf-8"))
    tessera_freeze = json.loads(TESSERA_FREEZE_PATH.read_text(encoding="utf-8"))
    conventional_freeze = json.loads(CONVENTIONAL_FREEZE_PATH.read_text(encoding="utf-8"))
    if not target_freeze["freeze_basis"]["gate"]["passed"]:
        raise RuntimeError("Phase 13 target-free GEDI gate failed")
    if tessera_freeze["freeze_basis"]["target_fhd_read"]:
        raise RuntimeError("Target FHD entered TESSERA alignment")
    if conventional_freeze["freeze_basis"]["target_fhd_read"]:
        raise RuntimeError("Target FHD entered conventional predictors")
    sites = sorted(target_freeze["freeze_basis"]["combined_unique_sites"])

    source_tessera = pd.concat(
        [
            pd.read_parquet(path, columns=KEYS + ["fhd_normal", *TESSERA_FEATURES])
            for path in SOURCE_TESSERA_PATHS
        ],
        ignore_index=True,
    )
    source_conventional = pd.concat(
        [pd.read_parquet(path, columns=KEYS + CONVENTIONAL_FEATURES) for path in SOURCE_CONVENTIONAL_PATHS],
        ignore_index=True,
    )
    source = source_tessera.merge(source_conventional, on=KEYS, validate="one_to_one")
    expected_sources = sorted(protocol["source_training"]["sites"])
    if sorted(source["site_id"].unique()) != expected_sources:
        raise RuntimeError("Phase 13 source training population changed")
    if not np.isfinite(
        source[["fhd_normal", *TESSERA_FEATURES, *CONVENTIONAL_FEATURES]].to_numpy(
            dtype=np.float64
        )
    ).all():
        raise RuntimeError("Phase 13 source training data contain non-finite values")

    target_tessera = pd.read_parquet(TARGET_TESSERA_PATH)
    if "fhd_normal" in target_tessera.columns:
        raise RuntimeError("Target FHD entered the prediction freeze")
    target_conventional = pd.read_parquet(
        TARGET_CONVENTIONAL_PATH, columns=KEYS + CONVENTIONAL_FEATURES
    )
    target = target_tessera.merge(target_conventional, on=KEYS, validate="one_to_one")
    target = target[target["site_id"].isin(sites)].copy()
    target["acquisition_datetime"] = pd.to_datetime(target["acquisition_datetime"], utc=True)
    target["pass_id"] = split_tools.pass_ids(target)
    if sorted(target["site_id"].unique()) != sites or target.duplicated(KEYS).any():
        raise RuntimeError("Phase 13 target predictor population changed")
    if not np.isfinite(target[[*TESSERA_FEATURES, *CONVENTIONAL_FEATURES]].to_numpy()).all():
        raise RuntimeError("Phase 13 target predictors contain non-finite values")

    alphas = [float(value) for value in protocol["source_training"]["alpha_grid"]]
    model_specs = {
        "tessera": TESSERA_FEATURES,
        "sentinel2_terrain": CONVENTIONAL_FEATURES,
    }
    prediction = target[
        KEYS
        + [
            "acquisition_datetime",
            "target_year",
            "orbit",
            "reference_ground_track",
            "pass_id",
            "longitude",
            "latitude",
            "x_epsg5070",
            "y_epsg5070",
        ]
    ].copy()
    tuning_frames = []
    selection_frames = []
    selected_alphas: dict[str, float] = {}
    for model_name, features in model_specs.items():
        selected, tuning, ranking = tune_source(source, features, alphas, model_name)
        scaler, model = common.fit_ridge(source, features, "fhd_normal", selected)
        prediction[f"source_{model_name}_prediction"] = model.predict(
            scaler.transform(target[features].to_numpy(dtype=np.float64))
        )
        selected_alphas[model_name] = selected
        tuning_frames.append(tuning)
        selection_frames.append(ranking)
        print(f"{model_name}: selected source alpha={selected:g}", flush=True)

    folds = pd.read_csv(FOLD_PATH)
    folds = folds[folds["eligible"].astype(bool) & folds["site_id"].isin(sites)].copy()
    expected_folds = int(target_freeze["freeze_basis"]["eligible_folds"])
    if len(folds) != expected_folds:
        raise RuntimeError("Phase 13 eligible fold count changed")
    local_rules = protocol["local_reference"]
    budgets = [int(local_rules["secondary_budget"]), int(local_rules["primary_budget"])]
    repeats = int(local_rules["replicates_per_fold"])
    seed = int(local_rules["sampling_seed"])
    buffer_m = float(protocol["complete_pass_split"]["exclusion_buffer_m"])
    sample_frames = []
    for site in sites:
        site_target = target[target["site_id"].eq(site)].copy().reset_index(drop=True)
        for _, fold in folds[folds["site_id"].eq(site)].sort_values("held_out_pass").iterrows():
            local_index, _ = verify_fold(site_target, fold, buffer_m)
            local = site_target.iloc[local_index]
            local_pass_ids = local["pass_id"].to_numpy(dtype=str)
            for shot_budget in budgets:
                for repeat in range(repeats):
                    rng = np.random.default_rng(
                        budget_tools.stable_seed(
                            site,
                            fold["held_out_pass"],
                            shot_budget,
                            repeat,
                            base_seed=seed,
                        )
                    )
                    positions = budget_tools.stratified_sample_positions(
                        local_pass_ids, shot_budget, rng
                    )
                    sample = local.iloc[positions][KEYS + ["target_year", "pass_id"]].copy()
                    sample["held_out_pass"] = fold["held_out_pass"]
                    sample["budget"] = shot_budget
                    sample["replicate"] = repeat
                    sample_frames.append(sample)
    samples = pd.concat(sample_frames, ignore_index=True)
    scenario_counts = samples.groupby(["site_id", "held_out_pass", "budget", "replicate"]).size()
    if not all(int(index[2]) == int(count) for index, count in scenario_counts.items()):
        raise RuntimeError("A frozen Phase 13 local sample has the wrong budget")
    if samples.groupby(["site_id", "held_out_pass", "budget", "replicate"])["pass_id"].nunique().lt(
        int(local_rules["minimum_retained_passes_represented"])
    ).any():
        raise RuntimeError("A Phase 13 local sample represents too few passes")

    common.write_parquet(prediction, PREDICTIONS_PATH)
    common.write_parquet(samples, LOCAL_SAMPLES_PATH)
    common.write_csv(pd.concat(tuning_frames, ignore_index=True), TUNING_PATH)
    common.write_csv(pd.concat(selection_frames, ignore_index=True), SELECTION_PATH)
    freeze_basis = {
        "protocol_sha256": common.sha256(PROTOCOL_PATH),
        "script_sha256": common.sha256(Path(__file__).resolve()),
        "target_freeze_id": target_freeze["freeze_id"],
        "tessera_freeze_id": tessera_freeze["freeze_id"],
        "conventional_freeze_id": conventional_freeze["freeze_id"],
        "source_input_hashes": {
            str(path.relative_to(ROOT)): common.sha256(path)
            for path in [*SOURCE_TESSERA_PATHS, *SOURCE_CONVENTIONAL_PATHS]
        },
        "target_input_hashes": {
            str(path.relative_to(ROOT)): common.sha256(path)
            for path in [TARGET_TESSERA_PATH, TARGET_CONVENTIONAL_PATH, FOLD_PATH]
        },
        "target_fhd_read": False,
        "source_sites": expected_sources,
        "target_sites": sites,
        "target_site_years": target_freeze["freeze_basis"]["site_years"],
        "eligible_folds": len(folds),
        "selected_alphas": selected_alphas,
        "budgets": budgets,
        "replicates_per_fold": repeats,
        "local_sampling_seed": seed,
        "frozen_local_scenarios": int(len(scenario_counts)),
    }
    freeze = {
        "freeze_id": "phase13-predictions-" + common.canonical_hash(freeze_basis)[:12],
        "created_utc": common.utc_now(),
        "status": "frozen_before_target_fhd_opening",
        "freeze_basis": freeze_basis,
        "outputs": {
            str(path.relative_to(ROOT)): common.sha256(path)
            for path in [PREDICTIONS_PATH, LOCAL_SAMPLES_PATH, TUNING_PATH, SELECTION_PATH]
        },
    }
    common.write_json(FREEZE_PATH, freeze)
    print(
        json.dumps(
            {
                "freeze_id": freeze["freeze_id"],
                "target_sites": sites,
                "eligible_folds": len(folds),
                "local_scenarios": len(scenario_counts),
                "target_fhd_read": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
