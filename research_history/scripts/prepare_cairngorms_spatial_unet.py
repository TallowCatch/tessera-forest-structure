#!/usr/bin/env python3
"""Prepare consistent CHM targets and frozen spatial folds for Phase 22."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import yaml
from rasterio.windows import from_bounds
from scipy.spatial import cKDTree
from scipy.stats import kurtosis

import prepare_cairngorms_components as common


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/cairngorms_spatial_unet.yaml"


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "phase22_cairngorms_spatial_unet"
    ]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


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


def surface_area_ratio(grid: np.ndarray) -> float:
    """Return triangulated surface area divided by horizontal area."""
    z00 = grid[:-1, :-1]
    z10 = grid[:-1, 1:]
    z01 = grid[1:, :-1]
    z11 = grid[1:, 1:]
    valid = np.isfinite(z00) & np.isfinite(z10) & np.isfinite(z01) & np.isfinite(z11)
    if int(valid.sum()) < 0.95 * valid.size:
        return np.nan
    area_a = 0.5 * np.sqrt(
        1.0 + np.square(z10 - z00) + np.square(z11 - z10)
    )
    area_b = 0.5 * np.sqrt(
        1.0 + np.square(z11 - z01) + np.square(z01 - z00)
    )
    return float(np.mean((area_a + area_b)[valid]))


def metrics(values: np.ndarray, opening_threshold: float) -> dict[str, float]:
    flat = values[np.isfinite(values) & (values >= 0)].astype(np.float64)
    if len(flat) < 2:
        return {}
    q25, median, q75, p95 = np.quantile(flat, [0.25, 0.50, 0.75, 0.95])
    mean = float(np.mean(flat))
    sd = float(np.std(flat, ddof=1))
    return {
        "chm_valid_pixels": float(len(flat)),
        "canopy_mean_height_m": mean,
        "canopy_p95_height_m": float(p95),
        "canopy_surface_sd_m": sd,
        "canopy_surface_cv": float(sd / mean) if mean > 1e-6 else np.nan,
        "canopy_surface_rcv": float((q75 - q25) / median) if median > 1e-6 else np.nan,
        "canopy_rumple": surface_area_ratio(values),
        "canopy_open_fraction": float(np.mean(flat < opening_threshold)),
        "canopy_height_kurtosis": float(kurtosis(flat, fisher=True, bias=False)),
    }


def extract_targets(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    path = ROOT / str(config["inputs"]["chm_path"])
    target_config = config["targets"]
    columns = [
        "chm_valid_pixels",
        "canopy_mean_height_m",
        *target_config["primary"],
        *target_config["secondary"],
    ]
    output = {name: np.full(len(frame), np.nan, dtype=np.float32) for name in columns}
    with rasterio.open(path) as dataset:
        mode, bands = common.identify_chm_bands(dataset)
        for position, row in enumerate(frame.itertuples(index=False)):
            half = 25.0
            window = from_bounds(
                float(row.bng_x) - half,
                float(row.bng_y) - half,
                float(row.bng_x) + half,
                float(row.bng_y) + half,
                dataset.transform,
            ).round_offsets().round_lengths()
            values = common.read_chm_values(dataset, mode, bands, window)
            expected = int(window.height) * int(window.width)
            grid = values.reshape(int(window.height), int(window.width))
            if expected == 0:
                continue
            result = metrics(grid, float(target_config["opening_threshold_m"]))
            for name, value in result.items():
                output[name][position] = value
            if (position + 1) % 500 == 0:
                print(f"targets {position + 1:,}/{len(frame):,}", flush=True)
    for name, values in output.items():
        frame[name] = values
    return frame


def split_validation(
    train: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    fold: int,
    settings: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    block_size = float(settings["validation_block_size_m"])
    bx = np.floor((x - x.min()) / block_size).astype(np.int64)
    by = np.floor((y - y.min()) / block_size).astype(np.int64)
    block = bx * 100000 + by
    unique = np.unique(block[train])
    rng = np.random.default_rng(int(settings["validation_seed"]) + fold)
    rng.shuffle(unique)
    count = max(1, int(np.ceil(len(unique) * float(settings["validation_fraction"]))))
    selected = np.isin(block, unique[:count])
    validation = train[selected[train]]
    subtrain = train[~selected[train]]
    return subtrain, validation


def fold_indices(
    scheme: str,
    fold: int,
    frame: pd.DataFrame,
    settings: dict[str, Any],
) -> dict[str, np.ndarray]:
    x = frame["bng_x"].to_numpy(dtype=np.float64)
    y = frame["bng_y"].to_numpy(dtype=np.float64)
    if scheme == "balanced":
        size = float(settings["balance_block_size_m"])
        bx = np.floor((x - x.min()) / size).astype(np.int64)
        by = np.floor((y - y.min()) / size).astype(np.int64)
        assignment = (2 * bx + by) % int(settings["folds"])
        buffer_m = float(settings["balanced_exclusion_buffer_m"])
    elif scheme == "regional":
        assignment = frame["spatial_region"].to_numpy(dtype=np.int64)
        buffer_m = float(settings["regional_exclusion_buffer_m"])
    else:
        raise ValueError(f"Unknown scheme: {scheme}")
    test = np.flatnonzero(assignment == fold)
    candidate = np.flatnonzero(assignment != fold)
    tree = cKDTree(np.column_stack([x[test], y[test]]))
    distance, _ = tree.query(np.column_stack([x[candidate], y[candidate]]), k=1)
    train = candidate[distance > buffer_m]
    subtrain, validation = split_validation(train, x, y, fold, settings)
    if len(test) < int(settings["minimum_test_rows"]):
        raise RuntimeError(f"{scheme} fold {fold} has only {len(test)} test rows")
    return {
        "train_indices": train.astype(np.int64),
        "subtrain_indices": subtrain.astype(np.int64),
        "validation_indices": validation.astype(np.int64),
        "test_indices": test.astype(np.int64),
    }


def save_subset(source: Path, indices: np.ndarray, target: Path) -> None:
    values = np.load(source, mmap_mode="r")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp.npy")
    np.save(temporary, np.asarray(values[indices]))
    temporary.replace(target)


def link_dense_files(source: Path, target: Path) -> None:
    for name in ["embedding_int8.npy", "embedding_scale.npy", "tessera_valid.npy"]:
        destination = target / name
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        destination.symlink_to(os.path.relpath(source / name, target))


def run() -> None:
    config = load_config()
    inputs = config["inputs"]
    outputs = config["outputs"]
    array_dir = ROOT / str(outputs["array_directory"])
    array_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(ROOT / str(inputs["cohort_path"])).sort_values("row_id").reset_index(drop=True)
    frame = extract_targets(frame, config)
    names = [*config["targets"]["primary"], *config["targets"]["secondary"]]
    target_values = frame[names].to_numpy(dtype=np.float64)
    tc = config["targets"]
    valid = (
        frame["phase20_valid"].fillna(False).to_numpy(dtype=bool)
        & np.all(np.isfinite(target_values), axis=1)
        & (frame["chm_valid_pixels"].to_numpy() >= int(tc["minimum_valid_pixels"]))
        & frame["canopy_p95_height_m"].between(0, float(tc["maximum_height_m"])).to_numpy()
        & frame["canopy_surface_cv"].between(0, float(tc["maximum_cv"])).to_numpy()
        & frame["canopy_surface_rcv"].between(0, float(tc["maximum_rcv"])).to_numpy()
        & frame["canopy_rumple"].between(1, float(tc["maximum_rumple"])).to_numpy()
        & frame["canopy_open_fraction"].between(0, 1).to_numpy()
    )
    frame["phase22_valid"] = valid
    selected = np.flatnonzero(valid)
    cohort = frame.iloc[selected].reset_index(drop=True)
    target_path = ROOT / str(outputs["target_table"])
    target_path.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_parquet(target_path, index=False, compression="zstd")

    row_ids = np.load(ROOT / str(inputs["patch_row_id_path"]), mmap_mode="r")
    if not np.array_equal(row_ids, frame["row_id"].to_numpy()):
        raise RuntimeError("Patch rows no longer match the cohort")
    conventional_ids = np.load(ROOT / str(inputs["conventional_row_id_path"]), mmap_mode="r")
    if not np.array_equal(conventional_ids, row_ids):
        raise RuntimeError("Conventional and TESSERA row IDs differ")
    for source_key, name in [
        ("patch_path", "patch_quantized.npy"),
        ("patch_scale_path", "patch_scales.npy"),
        ("patch_valid_path", "patch_valid.npy"),
        ("conventional_path", "conventional.npy"),
    ]:
        save_subset(ROOT / str(inputs[source_key]), selected, array_dir / name)
    np.save(array_dir / "row_id.npy", cohort["row_id"].to_numpy(dtype=np.int64))
    targets = cohort[names].to_numpy(dtype=np.float32)
    np.save(array_dir / "target_rows.npy", targets)
    # The cohort indices identify non-overlapping 50 m blocks on the 10 m
    # source grid. U-Net supervision belongs at each block's centre pixel.
    rows = cohort["raster_patch_row"].to_numpy(dtype=np.int32) * 5 + 2
    columns = cohort["raster_patch_column"].to_numpy(dtype=np.int32) * 5 + 2
    np.save(array_dir / "row.npy", rows)
    np.save(array_dir / "column.npy", columns)
    metric_size = float(config["spatial_evaluation"]["metric_block_size_m"])
    metric_block = (
        np.floor(cohort["bng_x"].to_numpy() / metric_size).astype(np.int64) * 100000
        + np.floor(cohort["bng_y"].to_numpy() / metric_size).astype(np.int64)
    )
    np.save(array_dir / "spatial_block.npy", metric_block)

    dense_source = ROOT / str(inputs["dense_source_directory"])
    link_dense_files(dense_source, array_dir)
    dense_valid = np.load(dense_source / "tessera_valid.npy", mmap_mode="r")
    dense_embedding = np.load(dense_source / "embedding_int8.npy", mmap_mode="r")
    dense_scale = np.load(dense_source / "embedding_scale.npy", mmap_mode="r")
    rng = np.random.default_rng(20260804)
    check = rng.choice(len(cohort), min(2000, len(cohort)), replace=False)
    patch_centre = (
        np.asarray(patch[check, :, 2, 2], dtype=np.float32)
        * np.asarray(scales[check, 2, 2], dtype=np.float32)[:, None]
    )
    dense_centre = (
        np.asarray(dense_embedding[rows[check], columns[check]], dtype=np.float32)
        * np.asarray(dense_scale[rows[check], columns[check]], dtype=np.float32)[:, None]
    )
    alignment_correlation = float(
        np.corrcoef(patch_centre.reshape(-1), dense_centre.reshape(-1))[0, 1]
    )
    if alignment_correlation < 0.98:
        raise RuntimeError(
            f"Dense-grid and ordered-patch TESSERA alignment failed: {alignment_correlation:.4f}"
        )
    index_grid = np.full(dense_valid.shape, -1, dtype=np.int32)
    index_grid[rows, columns] = np.arange(len(cohort), dtype=np.int32)
    np.save(array_dir / "index_grid.npy", index_grid)
    del index_grid

    patch = np.load(array_dir / "patch_quantized.npy", mmap_mode="r")
    scales = np.load(array_dir / "patch_scales.npy", mmap_mode="r")
    fold_rows: dict[str, Any] = {}
    settings = config["spatial_evaluation"]
    fold_dir = array_dir / "folds"
    fold_dir.mkdir(exist_ok=True)
    for scheme in settings["schemes"]:
        for fold in range(int(settings["folds"])):
            values = fold_indices(str(scheme), fold, cohort, settings)
            subtrain = values["subtrain_indices"]
            centre = (
                np.asarray(patch[subtrain, :, 2, 2], dtype=np.float32)
                * np.asarray(scales[subtrain, 2, 2], dtype=np.float32)[:, None]
            )
            input_mean = centre.mean(axis=0, dtype=np.float64).astype(np.float32)
            input_sd = centre.std(axis=0, dtype=np.float64).astype(np.float32)
            input_sd[input_sd < 1e-6] = 1.0
            target_mean = targets[subtrain].mean(axis=0, dtype=np.float64).astype(np.float32)
            target_sd = targets[subtrain].std(axis=0, dtype=np.float64).astype(np.float32)
            target_sd[target_sd < 1e-6] = 1.0
            path = fold_dir / f"{scheme}_fold_{fold}.npz"
            np.savez_compressed(
                path,
                **values,
                input_mean=input_mean,
                input_sd=input_sd,
                target_mean=target_mean,
                target_sd=target_sd,
            )
            fold_rows[f"{scheme}_{fold}"] = {key: int(len(value)) for key, value in values.items()}

    status = {
        "created_utc": utc_now(),
        "state": "complete",
        "rows_source": int(len(frame)),
        "rows_valid": int(len(cohort)),
        "target_names": names,
        "dense_patch_alignment_correlation": alignment_correlation,
        "fold_rows": fold_rows,
        "target_sha256": sha256(target_path),
        "config_sha256": sha256(CONFIG_PATH),
    }
    atomic_json(ROOT / str(outputs["preparation_status"]), status)
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    run()
