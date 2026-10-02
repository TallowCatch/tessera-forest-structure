#!/usr/bin/env python3
"""Prepare frozen Cairngorms canopy-height and surface targets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

import prepare_cairngorms_components as common


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/cairngorms_surface.yaml"
OUTPUT_DIR = ROOT / "data/interim/phase21_cairngorms_surface"
TARGET_PATH = ROOT / "data/processed/phase21_cairngorms_surface_targets.parquet"
STATUS_PATH = ROOT / "metadata/phase21_cairngorms_preparation_status.json"
FREEZE_PATH = ROOT / "metadata/phase21_cairngorms_input_freeze.json"


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "phase21_cairngorms_surface"
    ]


def intersect_indices(indices: np.ndarray, valid: np.ndarray) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    return indices[valid[indices]]


def all_target_names(config: dict[str, Any]) -> list[str]:
    raw = [str(value) for value in config["targets"]["primary"]]
    adjusted = [
        f"{value}_height_adjusted"
        for value in config["targets"]["height_adjusted"]
    ]
    return raw + adjusted


def build_fold(
    fold: int,
    frame: pd.DataFrame,
    valid: np.ndarray,
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    frozen = config["frozen_inputs"]
    source_path = ROOT / str(frozen["fold_directory"]) / f"fold_{fold}.npz"
    with np.load(source_path) as source:
        weights = source["weights"].astype(np.float32)
        train = intersect_indices(source["train_indices"], valid)
        subtrain = intersect_indices(source["subtrain_indices"], valid)
        validation = intersect_indices(source["validation_indices"], valid)
        test = intersect_indices(source["test_indices"], valid)
    minimum = int(config["quality_control"]["minimum_test_rows_per_fold"])
    if len(test) < minimum:
        raise RuntimeError(
            f"Fold {fold} retains only {len(test)} test rows; minimum is {minimum}"
        )

    raw_names = [str(value) for value in config["targets"]["primary"]]
    adjusted_names = [str(value) for value in config["targets"]["height_adjusted"]]
    raw = frame[raw_names].to_numpy(dtype=np.float32)
    controls = frame[
        [str(value) for value in config["targets"]["height_adjustment_predictors"]]
    ].to_numpy(dtype=np.float32)
    width = len(raw_names) + len(adjusted_names)
    tuning = np.full((len(frame), width), np.nan, dtype=np.float32)
    final = np.full_like(tuning, np.nan)
    tuning[:, : len(raw_names)] = raw
    final[:, : len(raw_names)] = raw
    height_reference_predictions = np.full(
        (len(test), len(adjusted_names)), np.nan, dtype=np.float32
    )
    knots = int(config["targets"]["spline_knots"])
    alpha = float(config["targets"]["ridge_alpha"])

    for adjusted_index, target_name in enumerate(adjusted_names):
        raw_index = raw_names.index(target_name)
        output_index = len(raw_names) + adjusted_index
        tune_model = common.spline_height_model(knots, alpha).fit(
            controls[subtrain], raw[subtrain, raw_index]
        )
        tune_rows = np.concatenate([subtrain, validation])
        tuning[tune_rows, output_index] = (
            raw[tune_rows, raw_index] - tune_model.predict(controls[tune_rows])
        )
        final_model = common.spline_height_model(knots, alpha).fit(
            controls[train], raw[train, raw_index]
        )
        final_rows = np.concatenate([train, test])
        predictions = final_model.predict(controls[final_rows])
        final[final_rows, output_index] = raw[final_rows, raw_index] - predictions
        height_reference_predictions[:, adjusted_index] = final_model.predict(
            controls[test]
        ).astype(np.float32)

    return {
        "weights": weights,
        "train_indices": train,
        "subtrain_indices": subtrain,
        "validation_indices": validation,
        "test_indices": test,
        "tuning_targets": tuning,
        "final_targets": final,
        "height_reference_predictions": height_reference_predictions,
    }


def run() -> None:
    config = load_config()
    frozen = config["frozen_inputs"]
    records: dict[str, Any] = {}
    for key in [
        "phase20_targets",
        "chm_metrics",
        "tessera_features",
        "conventional_features",
        "row_id",
        "common_valid",
    ]:
        records[key] = common.verify_file(
            ROOT / str(frozen[f"{key}_path"]),
            expected_hash=str(frozen[f"{key}_sha256"]),
        )
    for fold, expected_hash in enumerate(frozen["fold_sha256"]):
        path = ROOT / str(frozen["fold_directory"]) / f"fold_{fold}.npz"
        records[f"fold_{fold}"] = common.verify_file(
            path, expected_hash=str(expected_hash)
        )

    frame = (
        pd.read_parquet(ROOT / str(frozen["phase20_targets_path"]))
        .sort_values("row_id")
        .reset_index(drop=True)
    )
    row_ids = np.load(ROOT / str(frozen["row_id_path"]))
    if not np.array_equal(row_ids, frame["row_id"].to_numpy()):
        raise RuntimeError("Feature row IDs no longer match the target cohort")

    supplied = common.sample_supplied_metrics(
        ROOT / str(frozen["chm_metrics_path"]),
        frame,
        config["targets"]["supplied_bands"],
    )
    for name, values in supplied.items():
        frame[name] = values
    frame["canopy_top_height_m"] = frame["chm_p999_m"].astype(np.float32)
    agreement_rows = np.isfinite(frame["canopy_top_height_m"]) & np.isfinite(
        frame["supplied_chm_max_m"]
    )
    top_height_agreement = {
        "rows": int(agreement_rows.sum()),
        "spearman_r": float(
            spearmanr(
                frame.loc[agreement_rows, "canopy_top_height_m"],
                frame.loc[agreement_rows, "supplied_chm_max_m"],
            ).statistic
        ),
        "rmse_m": float(
            np.sqrt(
                np.mean(
                    np.square(
                        frame.loc[agreement_rows, "canopy_top_height_m"]
                        - frame.loc[agreement_rows, "supplied_chm_max_m"]
                    )
                )
            )
        ),
    }
    if top_height_agreement["spearman_r"] < float(
        config["quality_control"]["minimum_top_height_agreement_spearman"]
    ):
        raise RuntimeError("Robust top height does not agree with supplied maximum height")

    qc = config["quality_control"]
    common_valid = np.load(ROOT / str(frozen["common_valid_path"])).astype(bool)
    raw_names = [str(value) for value in config["targets"]["primary"]]
    finite_targets = np.all(
        np.isfinite(frame[raw_names].to_numpy(dtype=np.float64)), axis=1
    )
    valid = (
        common_valid
        & finite_targets
        & (frame["chm_valid_pixels"].to_numpy() >= int(qc["minimum_valid_chm_pixels"]))
        & frame["canopy_top_height_m"].between(
            0, float(qc["maximum_top_height_m"]), inclusive="both"
        ).to_numpy()
        & frame["canopy_surface_sd_m"].between(
            0, float(qc["maximum_surface_sd_m"]), inclusive="both"
        ).to_numpy()
        & frame["canopy_surface_cv"].between(
            0, float(qc["maximum_surface_cv"]), inclusive="both"
        ).to_numpy()
        & frame["canopy_surface_rcv"].between(
            0, float(qc["maximum_surface_rcv"]), inclusive="both"
        ).to_numpy()
        & frame["canopy_rumple"].between(
            float(qc["minimum_rumple"]),
            float(qc["maximum_rumple"]),
            inclusive="both",
        ).to_numpy()
        & frame["canopy_open_fraction"].between(0, 1, inclusive="both").to_numpy()
        & (
            frame["qa_flagged_fraction"].to_numpy()
            <= float(qc["maximum_flagged_fraction"])
        )
    )
    frame["phase21_valid"] = valid

    common.atomic_parquet(frame, TARGET_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fold_test_rows: dict[str, int] = {}
    for fold in range(int(config["evaluation"]["folds"])):
        values = build_fold(fold, frame, valid, config)
        temporary = OUTPUT_DIR / f"fold_{fold}.tmp.npz"
        np.savez_compressed(temporary, **values)
        temporary.replace(OUTPUT_DIR / f"fold_{fold}.npz")
        fold_test_rows[str(fold)] = int(len(values["test_indices"]))

    status = {
        "created_utc": common.utc_now(),
        "state": "complete",
        "rows_total": int(len(frame)),
        "rows_valid": int(valid.sum()),
        "raw_target_names": raw_names,
        "height_adjusted_names": [
            f"{value}_height_adjusted"
            for value in config["targets"]["height_adjusted"]
        ],
        "top_height_agreement": top_height_agreement,
        "fold_test_rows": fold_test_rows,
        "output_sha256": common.sha256(TARGET_PATH),
    }
    common.atomic_json(STATUS_PATH, status)
    common.atomic_json(
        FREEZE_PATH,
        {
            "created_utc": common.utc_now(),
            "config_sha256": common.sha256(CONFIG_PATH),
            "inputs": records,
            "outputs": status,
        },
    )
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    run()
