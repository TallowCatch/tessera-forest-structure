#!/usr/bin/env python3
"""Prepare frozen v2 patches for the Cairngorms canopy-surface rerun."""

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
from numpy.lib.format import open_memmap


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/cairngorms_v2_surface.yaml"


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "phase29_cairngorms_v2_surface"
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
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        if target.resolve() == source:
            return
        raise RuntimeError(f"Existing link points to another source: {target}")
    if target.exists():
        raise RuntimeError(f"Refusing to replace existing Phase 29 input: {target}")
    target.symlink_to(os.path.relpath(source, target.parent))


def build_patch_arrays(
    dense_directory: Path,
    rows: np.ndarray,
    columns: np.ndarray,
    output_directory: Path,
) -> tuple[Path, Path, Path]:
    patch_path = output_directory / "patch_quantized.npy"
    scale_path = output_directory / "patch_scales.npy"
    valid_path = output_directory / "patch_valid.npy"
    if all(path.exists() for path in [patch_path, scale_path, valid_path]):
        return patch_path, scale_path, valid_path
    if any(path.exists() for path in [patch_path, scale_path, valid_path]):
        raise RuntimeError("Partial Phase 29 patch arrays already exist")

    dense_embedding = np.load(dense_directory / "embedding_int8.npy", mmap_mode="r")
    dense_scale = np.load(dense_directory / "embedding_scale.npy", mmap_mode="r")
    dense_valid = np.load(dense_directory / "tessera_valid.npy", mmap_mode="r")
    if dense_embedding.shape[:2] != dense_valid.shape or dense_scale.shape != dense_valid.shape:
        raise RuntimeError("Dense v2 arrays have incompatible shapes")
    if rows.min() < 2 or columns.min() < 2:
        raise RuntimeError("A target centre is too close to the dense-grid boundary")
    if rows.max() + 2 >= dense_valid.shape[0] or columns.max() + 2 >= dense_valid.shape[1]:
        raise RuntimeError("A target centre is too close to the dense-grid boundary")

    output_directory.mkdir(parents=True, exist_ok=True)
    patch = open_memmap(
        patch_path, mode="w+", dtype=np.int8, shape=(len(rows), 128, 5, 5)
    )
    scales = open_memmap(
        scale_path, mode="w+", dtype=np.float16, shape=(len(rows), 5, 5)
    )
    valid = open_memmap(
        valid_path, mode="w+", dtype=np.uint8, shape=(len(rows), 5, 5)
    )
    chunk = 2048
    for start in range(0, len(rows), chunk):
        stop = min(start + chunk, len(rows))
        local_rows = rows[start:stop]
        local_columns = columns[start:stop]
        for patch_row, row_offset in enumerate(range(-2, 3)):
            for patch_column, column_offset in enumerate(range(-2, 3)):
                rr = local_rows + row_offset
                cc = local_columns + column_offset
                patch[start:stop, :, patch_row, patch_column] = dense_embedding[rr, cc]
                scales[start:stop, patch_row, patch_column] = dense_scale[rr, cc]
                valid[start:stop, patch_row, patch_column] = dense_valid[rr, cc]
        print(f"prepare patches {stop:,}/{len(rows):,}", flush=True)
    patch.flush()
    scales.flush()
    valid.flush()
    return patch_path, scale_path, valid_path


def summarize_patches(
    patch: np.ndarray,
    scales: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Return centre, mean, SD and x/y gradients for dequantized 5 x 5 patches."""
    embeddings = np.asarray(patch, dtype=np.float32)
    embeddings *= np.asarray(scales, dtype=np.float32)[:, None]
    mask = np.asarray(valid, dtype=bool)
    count = mask.sum(axis=(1, 2)).astype(np.float32)
    if np.any(count < 1):
        raise RuntimeError("A v2 patch has no valid pixels")
    total = np.where(mask[:, None], embeddings, 0).sum(axis=(2, 3))
    square = np.where(mask[:, None], np.square(embeddings), 0).sum(axis=(2, 3))
    mean = total / count[:, None]
    standard_deviation = np.sqrt(
        np.maximum(square / count[:, None] - np.square(mean), 0)
    )
    offsets = np.arange(5, dtype=np.float32) - 2
    xx = np.broadcast_to(offsets[None], (5, 5))
    yy = np.broadcast_to(-offsets[:, None], (5, 5))
    sum_x = np.where(mask, xx[None], 0).sum(axis=(1, 2))
    sum_y = np.where(mask, yy[None], 0).sum(axis=(1, 2))
    sum_x2 = np.where(mask, np.square(xx)[None], 0).sum(axis=(1, 2))
    sum_y2 = np.where(mask, np.square(yy)[None], 0).sum(axis=(1, 2))
    sum_x_embedding = np.where(
        mask[:, None], embeddings * xx[None, None], 0
    ).sum(axis=(2, 3))
    sum_y_embedding = np.where(
        mask[:, None], embeddings * yy[None, None], 0
    ).sum(axis=(2, 3))
    denominator_x = sum_x2 - np.square(sum_x) / count
    denominator_y = sum_y2 - np.square(sum_y) / count
    gradient_x = np.zeros_like(mean)
    gradient_y = np.zeros_like(mean)
    good_x = denominator_x > 0
    good_y = denominator_y > 0
    gradient_x[good_x] = (
        sum_x_embedding[good_x]
        - sum_x[good_x, None] * total[good_x] / count[good_x, None]
    ) / denominator_x[good_x, None]
    gradient_y[good_y] = (
        sum_y_embedding[good_y]
        - sum_y[good_y, None] * total[good_y] / count[good_y, None]
    ) / denominator_y[good_y, None]
    centre = embeddings[:, :, 2, 2]
    return np.concatenate(
        [centre, mean, standard_deviation, gradient_x, gradient_y], axis=1
    ).astype(np.float32)


def build_summary(
    patch_path: Path,
    scale_path: Path,
    valid_path: Path,
    output_path: Path,
) -> None:
    if output_path.exists():
        return
    patch = np.load(patch_path, mmap_mode="r")
    scales = np.load(scale_path, mmap_mode="r")
    valid = np.load(valid_path, mmap_mode="r")
    summary = open_memmap(
        output_path, mode="w+", dtype=np.float32, shape=(len(patch), 640)
    )
    chunk = 1024
    for start in range(0, len(patch), chunk):
        stop = min(start + chunk, len(patch))
        summary[start:stop] = summarize_patches(
            patch[start:stop], scales[start:stop], valid[start:stop]
        )
        print(f"prepare summaries {stop:,}/{len(patch):,}", flush=True)
    summary.flush()


def prepare_raw_folds(source: Path, target: Path, patch_path: Path, scale_path: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    patch = np.load(patch_path, mmap_mode="r")
    scales = np.load(scale_path, mmap_mode="r")
    for fold in range(5):
        output = target / f"balanced_fold_{fold}.npz"
        if output.exists():
            continue
        with np.load(source / f"balanced_fold_{fold}.npz") as values:
            copied = {name: values[name] for name in values.files}
        subtrain = copied["subtrain_indices"].astype(np.int64)
        centre = (
            np.asarray(patch[subtrain, :, 2, 2], dtype=np.float32)
            * np.asarray(scales[subtrain, 2, 2], dtype=np.float32)[:, None]
        )
        input_mean = centre.mean(axis=0, dtype=np.float64).astype(np.float32)
        input_sd = centre.std(axis=0, dtype=np.float64).astype(np.float32)
        input_sd[input_sd < 1e-6] = 1.0
        copied["input_mean"] = input_mean
        copied["input_sd"] = input_sd
        np.savez_compressed(output, **copied)


def run() -> None:
    config = load_config()
    frozen = config["frozen_inputs"]
    outputs = config["outputs"]
    status_path = ROOT / str(outputs["preparation_status"])
    config_hash = sha256(CONFIG_PATH)
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("config_sha256") != config_hash:
            raise RuntimeError("Existing Phase 29 preparation uses another config")
        print(json.dumps(status, indent=2), flush=True)
        return

    cohort = pd.read_parquet(ROOT / str(frozen["target_table"]))
    target_ids = cohort["row_id"].to_numpy(dtype=np.int64)
    raw_source = ROOT / str(frozen["raw_source_arrays"])
    adjusted_source = ROOT / str(frozen["adjusted_source_arrays"])
    source_ids = np.load(raw_source / "row_id.npy", mmap_mode="r")
    adjusted_ids = np.load(adjusted_source / "row_id.npy", mmap_mode="r")
    if not np.array_equal(source_ids, target_ids) or not np.array_equal(adjusted_ids, target_ids):
        raise RuntimeError("Frozen raw, adjusted and target rows are not identical")
    rows = np.asarray(np.load(raw_source / "row.npy", mmap_mode="r"), dtype=np.int32)
    columns = np.asarray(np.load(raw_source / "column.npy", mmap_mode="r"), dtype=np.int32)

    raw_dir = ROOT / str(outputs["raw_array_directory"])
    adjusted_dir = ROOT / str(outputs["adjusted_array_directory"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    adjusted_dir.mkdir(parents=True, exist_ok=True)
    patch_path, scale_path, valid_path = build_patch_arrays(
        ROOT / str(frozen["dense_v2_directory"]), rows, columns, raw_dir
    )
    valid = np.load(valid_path, mmap_mode="r")
    valid_counts = np.asarray(valid, dtype=np.uint8).sum(axis=(1, 2))
    minimum = int(config["representation"]["minimum_valid_patch_cells"])
    if np.any(valid_counts < minimum):
        raise RuntimeError(
            f"V2 retains fewer than {minimum} cells in {int((valid_counts < minimum).sum())} patches"
        )
    if not np.all(valid[:, 2, 2]):
        raise RuntimeError("A frozen target centre is missing from v2")

    for name in [
        "conventional.npy", "row_id.npy", "target_rows.npy", "row.npy", "column.npy",
        "spatial_block.npy", "index_grid.npy",
    ]:
        link(raw_source / name, raw_dir / name)
    dense_dir = ROOT / str(frozen["dense_v2_directory"])
    for name in ["embedding_int8.npy", "embedding_scale.npy", "tessera_valid.npy"]:
        link(dense_dir / name, raw_dir / name)
    prepare_raw_folds(raw_source / "folds", raw_dir / "folds", patch_path, scale_path)

    summary_path = adjusted_dir / "tessera.npy"
    build_summary(patch_path, scale_path, valid_path, summary_path)
    for name in ["conventional.npy", "spatial_block.npy", "row_id.npy"]:
        link(adjusted_source / name, adjusted_dir / name)
    for fold in range(5):
        link(adjusted_source / f"fold_{fold}.npz", adjusted_dir / f"fold_{fold}.npz")

    status = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "state": "complete",
        "config_sha256": config_hash,
        "rows": int(len(cohort)),
        "valid_patch_cells_minimum": int(valid_counts.min()),
        "valid_patch_cells_median": float(np.median(valid_counts)),
        "inputs": {
            "target_table_sha256": sha256(ROOT / str(frozen["target_table"])),
            "dense_v2_manifest_sha256": sha256(ROOT / str(frozen["dense_v2_manifest"])),
            "supplied_loader_sha256": sha256(ROOT / str(config["representation"]["supplied_loader"])),
        },
        "outputs": {
            "patch_sha256": sha256(patch_path),
            "scale_sha256": sha256(scale_path),
            "valid_sha256": sha256(valid_path),
            "summary_sha256": sha256(summary_path),
        },
    }
    if status["inputs"]["supplied_loader_sha256"] != config["representation"]["supplied_loader_sha256"]:
        raise RuntimeError("The supplied experimental v2 loader has changed")
    atomic_json(status_path, status)
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    run()

