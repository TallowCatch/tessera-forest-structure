#!/usr/bin/env python3
"""Prepare the frozen inputs for the audited TESSERA v2 height experiment."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/cairngorms_v2_height_unet.yaml"


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "phase30_cairngorms_v2_height_unet"
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


def link(source: Path, target: Path) -> None:
    source = source.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        if target.resolve() == source:
            return
        raise RuntimeError(f"Existing link points to another source: {target}")
    if target.exists():
        raise RuntimeError(f"Refusing to replace existing Phase 30 input: {target}")
    target.symlink_to(os.path.relpath(source, target.parent))


def save_exact(path: Path, values: np.ndarray) -> None:
    if path.exists():
        existing = np.load(path, mmap_mode="r")
        if existing.shape != values.shape or not np.array_equal(existing, values):
            raise RuntimeError(f"Existing Phase 30 array differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npy")
    np.save(temporary, values)
    temporary.replace(path)


def validate_fold(values: dict[str, np.ndarray], count: int) -> dict[str, int]:
    groups = {
        name: np.asarray(values[name], dtype=np.int64)
        for name in [
            "train_indices",
            "subtrain_indices",
            "validation_indices",
            "test_indices",
        ]
    }
    for name, indices in groups.items():
        if len(indices) == 0 or indices.min() < 0 or indices.max() >= count:
            raise RuntimeError(f"Invalid {name}")
        if len(np.unique(indices)) != len(indices):
            raise RuntimeError(f"Duplicate rows in {name}")
    if np.intersect1d(groups["train_indices"], groups["test_indices"]).size:
        raise RuntimeError("Training and test rows overlap")
    if np.intersect1d(groups["subtrain_indices"], groups["validation_indices"]).size:
        raise RuntimeError("Subtraining and validation rows overlap")
    tuning = np.sort(
        np.concatenate([groups["subtrain_indices"], groups["validation_indices"]])
    )
    if not np.array_equal(tuning, np.sort(groups["train_indices"])):
        raise RuntimeError("Subtraining and validation rows do not partition training")
    return {name.removesuffix("_indices"): int(len(indices)) for name, indices in groups.items()}


def run() -> None:
    config = load_config()
    frozen = config["frozen_inputs"]
    outputs = config["outputs"]
    output_dir = ROOT / str(outputs["array_directory"])
    status_path = ROOT / str(outputs["preparation_status"])
    config_hash = sha256(CONFIG_PATH)
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("config_sha256") != config_hash:
            raise RuntimeError("Existing Phase 30 preparation uses another config")
        print(json.dumps(status, indent=2), flush=True)
        return

    cohort_path = ROOT / str(frozen["target_table"])
    cohort = pd.read_parquet(cohort_path)
    row_ids = cohort["row_id"].to_numpy(dtype=np.int64)
    raw_source = ROOT / str(frozen["v2_raw_array_directory"])
    source_ids = np.load(raw_source / "row_id.npy", mmap_mode="r")
    if not np.array_equal(source_ids, row_ids):
        raise RuntimeError("The target table and V2 arrays do not have identical rows")

    target_names = [str(value) for value in config["targets"]["names"]]
    targets = cohort[target_names].to_numpy(dtype=np.float32)
    openings = cohort["canopy_open_fraction"].to_numpy(dtype=np.float32)
    if not np.isfinite(targets).all() or not np.isfinite(openings).all():
        raise RuntimeError("A frozen target or opening value is non-finite")
    if np.any(targets < 0) or np.any((openings < 0) | (openings > 1)):
        raise RuntimeError("A frozen height or opening value is outside its valid range")

    output_dir.mkdir(parents=True, exist_ok=True)
    save_exact(output_dir / "targets.npy", targets)
    save_exact(output_dir / "opening_fraction.npy", openings)
    for name in [
        "embedding_int8.npy",
        "embedding_scale.npy",
        "tessera_valid.npy",
        "row.npy",
        "column.npy",
        "spatial_block.npy",
        "index_grid.npy",
        "row_id.npy",
    ]:
        link(raw_source / name, output_dir / name)
    link(
        ROOT / str(frozen["v2_summary_array"]),
        output_dir / "tessera_summary.npy",
    )

    fold_counts: dict[str, dict[str, int]] = {}
    fold_dir = output_dir / "folds"
    fold_dir.mkdir(parents=True, exist_ok=True)
    for fold in range(int(config["evaluation"]["folds"])):
        source_path = raw_source / "folds" / f"balanced_fold_{fold}.npz"
        target_path = fold_dir / f"fold_{fold}.npz"
        with np.load(source_path) as source:
            copied = {
                name: source[name]
                for name in [
                    "train_indices",
                    "subtrain_indices",
                    "validation_indices",
                    "test_indices",
                ]
            }
        fold_counts[str(fold)] = validate_fold(copied, len(cohort))
        if target_path.exists():
            with np.load(target_path) as existing:
                if any(not np.array_equal(existing[name], value) for name, value in copied.items()):
                    raise RuntimeError(f"Existing Phase 30 fold differs: {target_path}")
        else:
            np.savez_compressed(target_path, **copied)

    status = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "state": "complete",
        "config_sha256": config_hash,
        "rows": int(len(cohort)),
        "targets": target_names,
        "fold_counts": fold_counts,
        "inputs": {
            "target_table_sha256": sha256(cohort_path),
            "dense_v2_manifest_sha256": sha256(
                ROOT / str(frozen["dense_v2_manifest"])
            ),
            "source_row_id_sha256": sha256(raw_source / "row_id.npy"),
            "v2_summary_sha256": sha256(ROOT / str(frozen["v2_summary_array"])),
        },
        "outputs": {
            "targets_sha256": sha256(output_dir / "targets.npy"),
            "opening_fraction_sha256": sha256(output_dir / "opening_fraction.npy"),
        },
    }
    atomic_json(status_path, status)
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    run()
