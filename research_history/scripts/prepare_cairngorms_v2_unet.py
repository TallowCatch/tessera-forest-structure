#!/usr/bin/env python3
"""Align corrected LiDAR rows, frozen folds and the dense v2 grid."""

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
CONFIG_PATH = ROOT / "configs/cairngorms_v2_unet.yaml"


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "phase28_cairngorms_v2_unet"
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


def aligned_positions(source_ids: np.ndarray, target_ids: np.ndarray) -> np.ndarray:
    order = np.argsort(source_ids)
    locations = np.searchsorted(source_ids[order], target_ids)
    if (
        np.any(locations >= len(order))
        or not np.array_equal(source_ids[order][locations], target_ids)
    ):
        raise RuntimeError("Source row IDs do not cover the corrected cohort")
    return order[locations]


def link(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(os.path.relpath(source, target.parent))


def run() -> None:
    config = load_config()
    frozen = config["frozen_inputs"]
    acquisition = config["acquisition"]
    outputs = config["outputs"]
    manifest_path = ROOT / str(outputs["acquisition_manifest"])
    if not manifest_path.exists():
        raise RuntimeError("Dense TESSERA v2 acquisition has not completed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("config_sha256") != sha256(CONFIG_PATH):
        raise RuntimeError("Dense v2 acquisition was made with another config")

    cohort = pd.read_parquet(ROOT / str(frozen["corrected_cohort"]))
    cohort = cohort.sort_values("row_id").reset_index(drop=True)
    target_ids = cohort["row_id"].to_numpy(dtype=np.int64)
    source_dir = ROOT / str(frozen["source_unet_directory"])
    source_ids = np.load(source_dir / "row_id.npy", mmap_mode="r")
    positions = aligned_positions(source_ids, target_ids)
    rows = np.asarray(np.load(source_dir / "row.npy", mmap_mode="r")[positions], dtype=np.int32)
    columns = np.asarray(
        np.load(source_dir / "column.npy", mmap_mode="r")[positions], dtype=np.int32
    )

    dense_dir = ROOT / str(acquisition["dense_directory"])
    dense_embedding = np.load(dense_dir / "embedding_int8.npy", mmap_mode="r")
    dense_scale = np.load(dense_dir / "embedding_scale.npy", mmap_mode="r")
    dense_valid = np.load(dense_dir / "tessera_valid.npy", mmap_mode="r")
    if not np.all(dense_valid[rows, columns]):
        missing = int((dense_valid[rows, columns] == 0).sum())
        raise RuntimeError(f"Dense v2 misses {missing} corrected target centres")

    patch_dir = ROOT / str(frozen["source_v2_patch_directory"])
    patch_ids = np.load(patch_dir / "row_id.npy", mmap_mode="r")
    patch_positions = aligned_positions(patch_ids, target_ids)
    patch = np.load(patch_dir / "patch_quantized.npy", mmap_mode="r")
    patch_scale = np.load(patch_dir / "patch_scales.npy", mmap_mode="r")
    patch_valid = np.load(patch_dir / "patch_valid.npy", mmap_mode="r")
    embedding_checks: list[np.ndarray] = []
    scale_checks: list[np.ndarray] = []
    valid_checks: list[np.ndarray] = []
    for patch_row, row_offset in enumerate(range(-2, 3)):
        for patch_column, column_offset in enumerate(range(-2, 3)):
            local_valid = np.asarray(
                dense_valid[rows + row_offset, columns + column_offset], dtype=bool
            )
            frozen_valid = np.asarray(
                patch_valid[patch_positions, patch_row, patch_column], dtype=bool
            )
            valid_checks.append(local_valid == frozen_valid)
            both = local_valid & frozen_valid
            if both.any():
                local_embedding = np.asarray(
                    dense_embedding[
                        rows[both] + row_offset, columns[both] + column_offset
                    ],
                    dtype=np.int8,
                )
                frozen_embedding = np.asarray(
                    patch[patch_positions[both], :, patch_row, patch_column],
                    dtype=np.int8,
                )
                embedding_checks.append(
                    np.all(local_embedding == frozen_embedding, axis=1)
                )
                scale_checks.append(
                    np.isclose(
                        np.asarray(
                            dense_scale[
                                rows[both] + row_offset, columns[both] + column_offset
                            ],
                            dtype=np.float32,
                        ),
                        np.asarray(
                            patch_scale[patch_positions[both], patch_row, patch_column],
                            dtype=np.float32,
                        ),
                        rtol=1e-3,
                        atol=1e-5,
                    )
                )
    exact_embedding_match = float(np.mean(np.concatenate(embedding_checks)))
    scale_match = float(np.mean(np.concatenate(scale_checks)))
    valid_match = float(np.mean(np.concatenate(valid_checks)))
    if min(exact_embedding_match, scale_match, valid_match) < 0.999:
        raise RuntimeError(
            "Dense and frozen 5 x 5 v2 inputs do not align: "
            f"embedding={exact_embedding_match:.5f}, scale={scale_match:.5f}, "
            f"valid={valid_match:.5f}"
        )

    array_dir = ROOT / str(outputs["array_directory"])
    array_dir.mkdir(parents=True, exist_ok=True)
    for name in ["embedding_int8.npy", "embedding_scale.npy", "tessera_valid.npy"]:
        link(dense_dir / name, array_dir / name)
    np.save(array_dir / "row_id.npy", target_ids)
    np.save(array_dir / "row.npy", rows)
    np.save(array_dir / "column.npy", columns)
    corrected_dir = ROOT / str(frozen["corrected_array_directory"])
    blocks = np.asarray(
        np.load(corrected_dir / "spatial_block.npy", mmap_mode="r"), dtype=np.int64
    )
    corrected_ids = np.load(corrected_dir / "row_id.npy", mmap_mode="r")
    if not np.array_equal(corrected_ids, target_ids):
        raise RuntimeError("Corrected fold arrays no longer match the corrected cohort")
    np.save(array_dir / "spatial_block.npy", blocks)
    index_grid = np.full(dense_valid.shape, -1, dtype=np.int32)
    index_grid[rows, columns] = np.arange(len(target_ids), dtype=np.int32)
    np.save(array_dir / "index_grid.npy", index_grid)

    folds_dir = array_dir / "folds"
    folds_dir.mkdir(exist_ok=True)
    fold_counts: dict[str, dict[str, int]] = {}
    for fold in range(int(config["evaluation"]["folds"])):
        source = corrected_dir / f"fold_{fold}.npz"
        target = folds_dir / f"fold_{fold}.npz"
        link(source, target)
        with np.load(source) as values:
            fold_counts[str(fold)] = {
                name: int(len(values[name]))
                for name in [
                    "train_indices",
                    "subtrain_indices",
                    "validation_indices",
                    "test_indices",
                ]
            }
            if values["final_targets"].shape != (len(target_ids), 8):
                raise RuntimeError("Corrected fold target dimensions changed")

    status = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "state": "complete",
        "rows": int(len(target_ids)),
        "dense_shape": list(dense_valid.shape),
        "dense_valid_cells": int(np.asarray(dense_valid).sum()),
        "centre_alignment": {
            "exact_embedding_fraction": exact_embedding_match,
            "scale_match_fraction": scale_match,
            "valid_match_fraction": valid_match,
        },
        "fold_counts": fold_counts,
        "config_sha256": sha256(CONFIG_PATH),
        "acquisition_manifest_sha256": sha256(manifest_path),
    }
    atomic_json(ROOT / str(outputs["preparation_status"]), status)
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    run()
