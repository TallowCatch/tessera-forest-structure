#!/usr/bin/env python3
"""Prepare leakage-safe height-adjusted Phase 23 targets and features."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

import prepare_cairngorms_components as common


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/cairngorms_height_adjusted.yaml"


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "phase23_cairngorms_height_adjusted"
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def subset_features(source: Path, source_ids: Path, target_ids: np.ndarray, output: Path) -> None:
    row_ids = np.load(source_ids, mmap_mode="r")
    order = np.argsort(row_ids)
    positions = np.searchsorted(row_ids[order], target_ids)
    if np.any(positions >= len(order)) or not np.array_equal(row_ids[order][positions], target_ids):
        raise RuntimeError(f"Feature rows do not cover the Phase 23 cohort: {source}")
    values = np.load(source, mmap_mode="r")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.npy")
    np.save(temporary, np.asarray(values[order[positions]], dtype=np.float32))
    temporary.replace(output)


def equal_block_weights(block: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(block, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse].astype(np.float32)
    return weights * len(weights) / weights.sum()


def run() -> None:
    config = load_config()
    frozen = config["frozen_inputs"]
    outputs = config["outputs"]
    array_dir = ROOT / str(outputs["array_directory"])
    array_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(ROOT / str(frozen["target_table_path"])).reset_index(drop=True)
    target_ids = frame["row_id"].to_numpy(dtype=np.int64)
    subset_features(
        ROOT / str(frozen["tessera_source_path"]), ROOT / str(frozen["tessera_row_id_path"]),
        target_ids, ROOT / str(frozen["tessera_features_path"]),
    )
    subset_features(
        ROOT / str(frozen["conventional_source_path"]), ROOT / str(frozen["conventional_row_id_path"]),
        target_ids, ROOT / str(frozen["conventional_features_path"]),
    )
    target_names = [str(value) for value in config["targets"]["names"]]
    controls = frame[[str(value) for value in config["targets"]["height_predictors"]]].to_numpy(dtype=np.float32)
    raw = frame[target_names].to_numpy(dtype=np.float32)
    block = np.load(ROOT / str(frozen["spatial_block_path"])).astype(np.int64)
    if len(block) != len(frame):
        raise RuntimeError("Spatial blocks do not match the corrected Phase 22 cohort")
    weights = equal_block_weights(block)
    np.save(array_dir / "spatial_block.npy", block)
    np.save(array_dir / "row_id.npy", target_ids)

    knots = int(config["targets"]["spline_knots"])
    alpha = float(config["targets"]["ridge_alpha"])
    fold_counts: dict[str, Any] = {}
    for fold in range(int(config["evaluation"]["folds"])):
        source_path = ROOT / str(frozen["source_fold_directory"]) / f"balanced_fold_{fold}.npz"
        with np.load(source_path) as source:
            train = source["train_indices"].astype(np.int64)
            subtrain = source["subtrain_indices"].astype(np.int64)
            validation = source["validation_indices"].astype(np.int64)
            test = source["test_indices"].astype(np.int64)
        tuning = np.full_like(raw, np.nan)
        final = np.full_like(raw, np.nan)
        height_test_predictions = np.full((len(test), len(target_names)), np.nan, dtype=np.float32)
        for column in range(len(target_names)):
            tune_model = common.spline_height_model(knots, alpha).fit(controls[subtrain], raw[subtrain, column])
            tune_rows = np.concatenate([subtrain, validation])
            tuning[tune_rows, column] = raw[tune_rows, column] - tune_model.predict(controls[tune_rows])
            final_model = common.spline_height_model(knots, alpha).fit(controls[train], raw[train, column])
            final_rows = np.concatenate([train, test])
            prediction = final_model.predict(controls[final_rows])
            final[final_rows, column] = raw[final_rows, column] - prediction
            height_test_predictions[:, column] = final_model.predict(controls[test]).astype(np.float32)
        np.savez_compressed(
            array_dir / f"fold_{fold}.npz",
            weights=weights,
            train_indices=train,
            subtrain_indices=subtrain,
            validation_indices=validation,
            test_indices=test,
            tuning_targets=tuning,
            final_targets=final,
            height_test_predictions=height_test_predictions,
        )
        fold_counts[str(fold)] = {
            "train": int(len(train)), "subtrain": int(len(subtrain)),
            "validation": int(len(validation)), "test": int(len(test)),
        }
    status = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "state": "complete", "rows": int(len(frame)), "targets": target_names,
        "fold_counts": fold_counts,
        "inputs": {
            "target_table_sha256": sha256(ROOT / str(frozen["target_table_path"])),
            "spatial_block_sha256": sha256(ROOT / str(frozen["spatial_block_path"])),
        },
    }
    atomic_json(ROOT / str(outputs["preparation_status"]), status)
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    run()
