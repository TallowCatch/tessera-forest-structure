#!/usr/bin/env python3
"""Derive three-layer LiDAR profile targets for the frozen dense grid."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import yaml

import prepare_cairngorms_dense_10m as dense_prepare


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/cairngorms_dense_profile.yaml"


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "cairngorms_dense_profile"
    ]


def profile_paths(config: dict[str, Any]) -> dict[str, Path]:
    directory = ROOT / str(config["outputs"]["profile_directory"])
    dense = ROOT / str(config["source"]["dense_directory"])
    return {
        "directory": directory,
        "targets": directory / "profile_targets.npy",
        "covariates": directory / "diagnostic_covariates.npy",
        "manifest": directory / "manifest.json",
        "dense": dense,
        "rows": dense / "row.npy",
        "columns": dense / "column.npy",
        "dense_targets": dense / "target_rows.npy",
        "dense_manifest": ROOT / str(config["source"]["dense_manifest"]),
        "status": ROOT / str(config["outputs"]["preparation_status"]),
        "freeze": ROOT / str(config["outputs"]["input_freeze"]),
    }


def normalize_profile(volumes: np.ndarray, minimum_total: float) -> np.ndarray:
    if volumes.ndim < 2 or volumes.shape[0] != 3:
        raise ValueError("Profile volumes must have three layers on axis zero")
    valid = np.isfinite(volumes).all(axis=0) & (volumes >= 0).all(axis=0)
    total = np.where(valid, volumes.sum(axis=0), np.nan)
    valid &= total > minimum_total
    proportions = np.full(volumes.shape, np.nan, dtype=np.float32)
    proportions[:, valid] = volumes[:, valid] / total[valid]
    return proportions


def profile_entropy(proportions: np.ndarray) -> np.ndarray:
    values = np.asarray(proportions, dtype=np.float64)
    terms = np.where(values > 0, values * np.log(np.maximum(values, 1e-12)), 0.0)
    return -terms.sum(axis=-1)


def validate_inputs(config: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    source = ROOT / str(config["source"]["lidar_metrics_path"])
    required = [
        source,
        paths["rows"],
        paths["columns"],
        paths["dense_targets"],
        paths["dense_manifest"],
    ]
    for fold in range(int(config["spatial_evaluation"]["region_count"])):
        required.append(paths["dense"] / "folds" / f"fold_{fold}.npz")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("Missing frozen dense inputs: " + ", ".join(missing))
    if sha256(source) != str(config["source"]["expected_source_sha256"]):
        raise RuntimeError("The Cairngorms LiDAR source hash changed")
    manifest = json.loads(paths["dense_manifest"].read_text(encoding="utf-8"))
    expected_rows = int(config["source"]["expected_dense_rows"])
    if int(manifest["eligible_rows"]) != expected_rows:
        raise RuntimeError("Dense eligible-row count differs from the frozen experiment")
    rows = np.load(paths["rows"], mmap_mode="r")
    columns = np.load(paths["columns"], mmap_mode="r")
    if len(rows) != expected_rows or len(columns) != expected_rows:
        raise RuntimeError("Dense row/column arrays have an unexpected length")
    return manifest


def main() -> None:
    config = load_config()
    paths = profile_paths(config)
    paths["directory"].mkdir(parents=True, exist_ok=True)
    if paths["targets"].exists() and paths["covariates"].exists() and paths["manifest"].exists():
        print("profile preparation resumed: outputs already complete", flush=True)
        return

    dense_manifest = validate_inputs(config, paths)
    atomic_json(
        paths["status"],
        {"updated_utc": utc_now(), "stage": "derive", "message": "Reading LiDAR bands"},
    )
    rows = np.asarray(np.load(paths["rows"], mmap_mode="r"), dtype=np.int64)
    columns = np.asarray(np.load(paths["columns"], mmap_mode="r"), dtype=np.int64)
    old_targets = np.load(paths["dense_targets"], mmap_mode="r")
    cells = int(config["profile"]["structural_window_cells"])
    source = ROOT / str(config["source"]["lidar_metrics_path"])

    with rasterio.open(source) as dataset:
        lookup = {str(name): index for index, name in enumerate(dataset.descriptions, 1)}
        required_bands = [
            *[str(value) for value in config["profile"]["source_bands"]],
            *[str(value) for value in config["profile"]["covariate_bands"].values()],
        ]
        missing = sorted(set(required_bands) - set(lookup))
        if missing:
            raise RuntimeError("LiDAR raster lacks bands: " + ", ".join(missing))

        layer_rows: list[np.ndarray] = []
        for name in config["profile"]["source_bands"]:
            values = dense_prepare.read_band(dataset, lookup, str(name))
            rolling, _ = dense_prepare.moving_mean_and_count(values, cells)
            layer_rows.append(rolling[rows, columns].astype(np.float32))
            del values, rolling
        volume_rows = np.stack(layer_rows, axis=0)
        profile = normalize_profile(
            volume_rows, float(config["profile"]["minimum_total_volume"])
        ).T
        del volume_rows, layer_rows

        covariate_rows: list[np.ndarray] = []
        for name in config["profile"]["covariate_bands"].values():
            values = dense_prepare.read_band(dataset, lookup, str(name))
            rolling, _ = dense_prepare.moving_mean_and_count(values, cells)
            covariate_rows.append(rolling[rows, columns].astype(np.float32))
            del values, rolling
        covariates = np.column_stack(covariate_rows).astype(np.float32)

    if not np.isfinite(profile).all() or (profile < 0).any():
        raise RuntimeError("Frozen dense rows include invalid profile proportions")
    sum_error = np.abs(profile.sum(axis=1) - 1.0)
    if float(sum_error.max()) > 1e-5:
        raise RuntimeError("Profile proportions do not sum to one")
    reconstructed_entropy = profile_entropy(profile)
    entropy_error = np.abs(reconstructed_entropy - np.asarray(old_targets[:, 3]))
    tolerance = float(config["profile"]["entropy_consistency_tolerance"])
    if float(entropy_error.max()) > tolerance:
        raise RuntimeError(
            f"Profile entropy differs from frozen dense target by {entropy_error.max():.8f}"
        )
    gap_error = np.abs(covariates[:, 2] - np.asarray(old_targets[:, 2]))
    if float(gap_error.max()) > tolerance:
        raise RuntimeError(
            f"Rolling gap fraction differs from frozen dense target by {gap_error.max():.8f}"
        )

    for path, values in [(paths["targets"], profile), (paths["covariates"], covariates)]:
        temporary = path.with_suffix(".tmp.npy")
        np.save(temporary, values)
        temporary.replace(path)

    manifest = {
        "created_utc": utc_now(),
        "config_sha256": sha256(CONFIG_PATH),
        "dense_manifest_sha256": sha256(paths["dense_manifest"]),
        "rows": len(profile),
        "profile_layer_labels": config["profile"]["layer_labels"],
        "covariate_names": list(config["profile"]["covariate_bands"].keys()),
        "profile_sum_max_absolute_error": float(sum_error.max()),
        "entropy_consistency_max_absolute_error": float(entropy_error.max()),
        "gap_consistency_max_absolute_error": float(gap_error.max()),
        "profile_quantiles": {
            str(label): np.quantile(profile[:, index], [0.01, 0.25, 0.5, 0.75, 0.99]).tolist()
            for index, label in enumerate(config["profile"]["layer_labels"])
        },
        "dense_fold_records": dense_manifest["folds"],
    }
    atomic_json(paths["manifest"], manifest)
    atomic_json(
        paths["freeze"],
        {
            "created_utc": utc_now(),
            "config_sha256": sha256(CONFIG_PATH),
            "lidar_sha256": sha256(source),
            "dense_manifest_sha256": sha256(paths["dense_manifest"]),
            "profile_targets_sha256": sha256(paths["targets"]),
            "diagnostic_covariates_sha256": sha256(paths["covariates"]),
        },
    )
    atomic_json(
        paths["status"],
        {
            "updated_utc": utc_now(),
            "stage": "complete",
            "message": f"Prepared {len(profile):,} three-layer profiles",
            "rows": len(profile),
        },
    )
    print(f"prepared {len(profile):,} three-layer profile targets", flush=True)


if __name__ == "__main__":
    main()
