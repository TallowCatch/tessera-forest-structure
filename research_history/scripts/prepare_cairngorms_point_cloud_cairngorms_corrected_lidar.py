#!/usr/bin/env python3
"""Prepare corrected point-cloud targets using the frozen Cairngorms design."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import yaml
from rasterio.windows import from_bounds
from rasterio.windows import Window

import prepare_cairngorms_components as common


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/cairngorms_point_cloud_cairngorms_corrected_lidar.yaml"


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "phase25_cairngorms_corrected_lidar"
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


def subset_features(
    source: Path,
    source_ids_path: Path,
    target_ids: np.ndarray,
    output: Path,
) -> None:
    source_ids = np.load(source_ids_path, mmap_mode="r")
    order = np.argsort(source_ids)
    positions = np.searchsorted(source_ids[order], target_ids)
    if (
        np.any(positions >= len(order))
        or not np.array_equal(source_ids[order][positions], target_ids)
    ):
        raise RuntimeError(f"Feature rows do not cover the corrected cohort: {source}")
    values = np.load(source, mmap_mode="r")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.npy")
    np.save(temporary, np.asarray(values[order[positions]], dtype=np.float32))
    temporary.replace(output)


def equal_block_weights(block: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(block, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse].astype(np.float32)
    return weights * len(weights) / weights.sum()


def remap_indices(indices: np.ndarray, selected: np.ndarray) -> np.ndarray:
    mapping = np.full(int(selected.max()) + 1, -1, dtype=np.int64)
    mapping[selected] = np.arange(len(selected), dtype=np.int64)
    retained = indices[indices < len(mapping)]
    retained = mapping[retained]
    return retained[retained >= 0]


def band_lookup(dataset: rasterio.DatasetReader) -> dict[str, int]:
    descriptions = list(dataset.descriptions)
    if any(value is None for value in descriptions):
        raise RuntimeError("Corrected LiDAR raster contains an unnamed band")
    if len(set(descriptions)) != len(descriptions):
        raise RuntimeError("Corrected LiDAR raster contains duplicate band names")
    return {str(value): index for index, value in enumerate(descriptions, start=1)}


def support_windows(
    dataset: rasterio.DatasetReader, frame: pd.DataFrame
) -> list[tuple[slice, slice]]:
    windows: list[tuple[slice, slice]] = []
    for row in frame.itertuples(index=False):
        window = from_bounds(
            float(row.bng_x) - 25.0,
            float(row.bng_y) - 25.0,
            float(row.bng_x) + 25.0,
            float(row.bng_y) + 25.0,
            dataset.transform,
        ).round_offsets().round_lengths()
        if int(window.height) != 5 or int(window.width) != 5:
            raise RuntimeError("A 50 m cohort unit does not cover exactly 5 x 5 LiDAR cells")
        row_start = int(window.row_off)
        column_start = int(window.col_off)
        windows.append(
            (
                slice(row_start, row_start + 5),
                slice(column_start, column_start + 5),
            )
        )
    return windows


def aggregate_band(
    values: np.ndarray,
    windows: list[tuple[slice, slice]],
    minimum_valid: int,
) -> tuple[np.ndarray, np.ndarray]:
    means = np.full(len(windows), np.nan, dtype=np.float32)
    counts = np.zeros(len(windows), dtype=np.int16)
    for position, window in enumerate(windows):
        local = np.asarray(values[window], dtype=np.float64)
        valid = np.isfinite(local)
        counts[position] = int(valid.sum())
        if counts[position] >= minimum_valid:
            means[position] = float(local[valid].mean())
    return means, counts


def extract_metrics(
    path: Path,
    frame: pd.DataFrame,
    source_bands: dict[str, str],
    minimum_valid: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = pd.DataFrame(index=frame.index)
    with rasterio.open(path) as dataset:
        lookup = band_lookup(dataset)
        missing = sorted(set(source_bands.values()) - set(lookup))
        if missing:
            raise RuntimeError(f"Corrected LiDAR raster is missing bands: {missing}")
        windows = support_windows(dataset, frame)
        for output_name, source_name in source_bands.items():
            values = dataset.read(lookup[source_name])
            means, counts = aggregate_band(values, windows, minimum_valid)
            result[output_name] = means
            result[f"{output_name}_valid_10m_cells"] = counts
            print(f"aggregated {source_name}", flush=True)
        profile = {
            "bands": list(dataset.descriptions),
            "count": int(dataset.count),
            "height": int(dataset.height),
            "width": int(dataset.width),
            "transform": tuple(dataset.transform),
            "bounds": tuple(dataset.bounds),
            "resolution": tuple(dataset.res),
            "crs": None if dataset.crs is None else str(dataset.crs),
        }
    return result, profile


def verify_replacement(
    previous: Path,
    corrected: Path,
    expected_changed: set[str],
) -> dict[str, Any]:
    with rasterio.open(previous) as old, rasterio.open(corrected) as new:
        if (
            old.count != new.count
            or old.shape != new.shape
            or old.transform != new.transform
            or old.descriptions != new.descriptions
        ):
            raise RuntimeError("Corrected LiDAR raster geometry or bands changed")
        lookup = band_lookup(new)
        changed: list[str] = []
        for name in new.descriptions:
            assert name is not None
            band = lookup[name]
            differs = False
            for row_start in range(0, new.height, 256):
                window = Window(
                    col_off=0,
                    row_off=row_start,
                    width=new.width,
                    height=min(256, new.height - row_start),
                )
                x = old.read(band, window=window)
                y = new.read(band, window=window)
                if not np.array_equal(x, y, equal_nan=True):
                    differs = True
                    break
            if differs:
                changed.append(name)
        if set(changed) != expected_changed:
            raise RuntimeError(
                f"Unexpected changed bands: observed={changed}, expected={sorted(expected_changed)}"
            )
    return {"changed_bands": changed, "unchanged_band_count": int(new.count - len(changed))}


def build_folds(
    selected: np.ndarray,
    raw: np.ndarray,
    controls: np.ndarray,
    blocks: np.ndarray,
    config: dict[str, Any],
    output_directory: Path,
) -> dict[str, Any]:
    settings = config["evaluation"]
    target_config = config["targets"]
    suffix = str(target_config["adjusted_suffix"])
    width = raw.shape[1]
    weights = equal_block_weights(blocks)
    fold_counts: dict[str, Any] = {}
    for fold in range(int(settings["folds"])):
        source_path = (
            ROOT
            / str(config["frozen_inputs"]["source_fold_directory"])
            / f"balanced_fold_{fold}.npz"
        )
        with np.load(source_path) as source:
            train = remap_indices(source["train_indices"].astype(np.int64), selected)
            subtrain = remap_indices(source["subtrain_indices"].astype(np.int64), selected)
            validation = remap_indices(source["validation_indices"].astype(np.int64), selected)
            test = remap_indices(source["test_indices"].astype(np.int64), selected)
        if len(test) < int(config["quality_control"]["minimum_test_rows"]):
            raise RuntimeError(f"Corrected fold {fold} has only {len(test)} test rows")
        tuning = np.full((len(raw), width * 2), np.nan, dtype=np.float32)
        final = np.full_like(tuning, np.nan)
        tuning[:, :width] = raw
        final[:, :width] = raw
        for column in range(width):
            tune_model = common.spline_height_model(
                int(target_config["spline_knots"]),
                float(target_config["ridge_alpha"]),
            ).fit(controls[subtrain], raw[subtrain, column])
            tune_rows = np.concatenate([subtrain, validation])
            tuning[tune_rows, width + column] = (
                raw[tune_rows, column] - tune_model.predict(controls[tune_rows])
            )
            final_model = common.spline_height_model(
                int(target_config["spline_knots"]),
                float(target_config["ridge_alpha"]),
            ).fit(controls[train], raw[train, column])
            final_rows = np.concatenate([train, test])
            final[final_rows, width + column] = (
                raw[final_rows, column] - final_model.predict(controls[final_rows])
            )
        np.savez_compressed(
            output_directory / f"fold_{fold}.npz",
            weights=weights,
            train_indices=train,
            subtrain_indices=subtrain,
            validation_indices=validation,
            test_indices=test,
            tuning_targets=tuning,
            final_targets=final,
        )
        fold_counts[str(fold)] = {
            "train": int(len(train)),
            "subtrain": int(len(subtrain)),
            "validation": int(len(validation)),
            "test": int(len(test)),
        }
    return {
        "fold_counts": fold_counts,
        "raw_targets": [str(value) for value in target_config["primary"]],
        "adjusted_targets": [f"{value}{suffix}" for value in target_config["primary"]],
    }


def run() -> None:
    config = load_config()
    frozen = config["frozen_inputs"]
    outputs = config["outputs"]
    corrected = ROOT / str(frozen["corrected_lidar_path"])
    previous = ROOT / str(frozen["previous_lidar_path"])
    if sha256(corrected) != str(frozen["corrected_lidar_sha256"]):
        raise RuntimeError("Corrected LiDAR checksum does not match the frozen input")
    if sha256(previous) != str(frozen["previous_lidar_sha256"]):
        raise RuntimeError("Previous LiDAR checksum does not match the recorded input")
    replacement = verify_replacement(
        previous,
        corrected,
        {"lidar_height_cv", "lidar_rcv", "lidar_rms", "lidar_canopy_shannon"},
    )

    frame = pd.read_parquet(ROOT / str(frozen["cohort_path"])).reset_index(drop=True)
    source_bands = {
        str(name): str(value) for name, value in config["targets"]["source_bands"].items()
    }
    extracted, profile = extract_metrics(
        corrected,
        frame,
        source_bands,
        int(config["targets"]["minimum_valid_10m_cells"]),
    )
    for name in extracted:
        frame[name] = extracted[name].to_numpy()

    primary = [str(value) for value in config["targets"]["primary"]]
    controls_names = [str(value) for value in config["targets"]["height_predictors"]]
    valid = np.all(np.isfinite(frame[primary + controls_names].to_numpy(dtype=np.float64)), axis=1)
    for name in primary:
        low, high = [float(value) for value in config["quality_control"][name]]
        valid &= frame[name].between(low, high, inclusive="both").to_numpy()
    selected = np.flatnonzero(valid)
    cohort = frame.iloc[selected].reset_index(drop=True)
    cohort["phase25_valid"] = True

    array_dir = ROOT / str(outputs["array_directory"])
    array_dir.mkdir(parents=True, exist_ok=True)
    target_path = ROOT / str(outputs["target_table"])
    target_path.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_parquet(target_path, index=False, compression="zstd")
    target_ids = cohort["row_id"].to_numpy(dtype=np.int64)
    subset_features(
        ROOT / str(frozen["tessera_source_path"]),
        ROOT / str(frozen["tessera_row_id_path"]),
        target_ids,
        ROOT / str(frozen["tessera_features_path"]),
    )
    subset_features(
        ROOT / str(frozen["conventional_source_path"]),
        ROOT / str(frozen["conventional_row_id_path"]),
        target_ids,
        ROOT / str(frozen["conventional_features_path"]),
    )
    source_blocks = np.load(ROOT / str(frozen["source_spatial_block_path"])).astype(np.int64)
    blocks = source_blocks[selected]
    np.save(array_dir / "spatial_block.npy", blocks)
    np.save(array_dir / "row_id.npy", target_ids)
    raw = cohort[primary].to_numpy(dtype=np.float32)
    controls = cohort[controls_names].to_numpy(dtype=np.float32)
    fold_status = build_folds(selected, raw, controls, blocks, config, array_dir)

    status = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "state": "complete",
        "rows_source": int(len(frame)),
        "rows_valid": int(len(cohort)),
        "replacement_verification": replacement,
        "raster_profile": profile,
        **fold_status,
    }
    atomic_json(ROOT / str(outputs["preparation_status"]), status)
    atomic_json(
        ROOT / str(outputs["input_freeze"]),
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "config_sha256": sha256(CONFIG_PATH),
            "corrected_lidar_sha256": sha256(corrected),
            "previous_lidar_sha256": sha256(previous),
            "target_table_sha256": sha256(target_path),
            "status": status,
        },
    )
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    run()
