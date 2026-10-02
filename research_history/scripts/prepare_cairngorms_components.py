#!/usr/bin/env python3
"""Prepare independently reproducible Cairngorms structure targets and folds."""

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
from rasterio.windows import Window, from_bounds
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/cairngorms_components.yaml"
OUTPUT_DIR = ROOT / "data/interim/phase20_cairngorms_components"
TARGET_PATH = ROOT / "data/processed/phase20_cairngorms_component_targets.parquet"
STATUS_PATH = ROOT / "metadata/phase20_cairngorms_preparation_status.json"
FREEZE_PATH = ROOT / "metadata/phase20_cairngorms_input_freeze.json"


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "phase20_cairngorms_components"
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


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def verify_file(path: Path, expected_bytes: int | None = None, expected_hash: str | None = None) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Missing required input: {path}")
    size = path.stat().st_size
    if expected_bytes is not None and size != expected_bytes:
        raise RuntimeError(f"Unexpected size for {path}: {size} != {expected_bytes}")
    observed_hash = sha256(path)
    if expected_hash is not None and observed_hash != expected_hash:
        raise RuntimeError(f"Frozen input changed: {path}")
    return {"path": str(path.relative_to(ROOT)), "bytes": size, "sha256": observed_hash}


def band_lookup(dataset: rasterio.DatasetReader) -> dict[str, int]:
    return {
        str(description).strip().lower(): index
        for index, description in enumerate(dataset.descriptions, 1)
        if description
    }


def identify_chm_bands(dataset: rasterio.DatasetReader) -> tuple[str, tuple[int, ...]]:
    lookup = band_lookup(dataset)
    direct = [index for name, index in lookup.items() if name == "chm" or "canopy height model" in name]
    if len(direct) == 1:
        return "direct", (direct[0],)
    if dataset.count == 1:
        return "direct", (1,)
    dsm = [index for name, index in lookup.items() if name == "dsm" or "surface model" in name]
    dtm = [index for name, index in lookup.items() if name == "dtm" or "terrain model" in name]
    if len(dsm) == 1 and len(dtm) == 1:
        return "difference", (dsm[0], dtm[0])
    if dataset.count == 3:
        sample = dataset.read(
            out_shape=(3, min(dataset.height, 512), min(dataset.width, 512)),
            masked=True,
        ).filled(np.nan).astype(np.float64)
        candidates: list[tuple[float, int]] = []
        for candidate in range(3):
            candidate_values = sample[candidate][np.isfinite(sample[candidate])]
            if not len(candidate_values):
                continue
            # A CHM is near zero over open ground and much smaller than the
            # absolute DSM/DTM elevations. Confirm this with DSM - DTM too.
            if np.nanmedian(candidate_values) >= 50 or np.nanquantile(candidate_values, 0.99) >= 100:
                continue
            others = [index for index in range(3) if index != candidate]
            for dsm_index, dtm_index in [others, others[::-1]]:
                valid = (
                    np.isfinite(sample[candidate])
                    & np.isfinite(sample[dsm_index])
                    & np.isfinite(sample[dtm_index])
                )
                residual = np.abs(
                    sample[candidate][valid]
                    - (sample[dsm_index][valid] - sample[dtm_index][valid])
                )
                if len(residual) and np.median(residual) < 0.5 and np.mean(residual < 2.0) > 0.99:
                    candidates.append((float(np.median(residual)), candidate + 1))
        if candidates:
            candidates.sort()
            return "direct", (candidates[0][1],)
    raise RuntimeError(
        "Could not establish the CHM band from raster descriptions; "
        f"descriptions={dataset.descriptions}"
    )


def metric_window(dataset: rasterio.DatasetReader, x: float, y: float, size_m: float = 50.0) -> Window:
    half = size_m / 2.0
    return from_bounds(x - half, y - half, x + half, y + half, dataset.transform).round_offsets().round_lengths()


def read_chm_values(
    dataset: rasterio.DatasetReader,
    mode: str,
    bands: tuple[int, ...],
    window: Window,
) -> np.ndarray:
    if mode == "direct":
        values = dataset.read(bands[0], window=window, boundless=True, masked=True)
        result = np.asarray(values.filled(np.nan), dtype=np.float32)
    else:
        values = dataset.read(list(bands), window=window, boundless=True, masked=True)
        data = np.asarray(values.filled(np.nan), dtype=np.float32)
        result = data[0] - data[1]
    return result.reshape(-1)


def accepted_mask(dataset: rasterio.DatasetReader, window: Window) -> np.ndarray:
    values = dataset.read(1, window=window, boundless=True, masked=True)
    data = np.asarray(values.filled(np.nan), dtype=np.float32).reshape(-1)
    return np.isfinite(data) & (data != 0)


def canopy_metrics(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values) & (values >= 0)]
    if not len(values):
        return {"valid": 0, "p999": np.nan, "sd": np.nan, "rcv": np.nan, "gap": np.nan}
    q25, median, q75 = np.quantile(values, [0.25, 0.50, 0.75])
    return {
        "valid": int(len(values)),
        "p999": float(np.quantile(values, 0.999)),
        "sd": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
        "rcv": float((q75 - q25) / median) if median > 1e-6 else np.nan,
        "gap": float(np.mean(values < 2.0)),
    }


def sample_supplied_metrics(path: Path, frame: pd.DataFrame, mapping: dict[str, str]) -> dict[str, np.ndarray]:
    output = {name: np.full(len(frame), np.nan, dtype=np.float32) for name in mapping}
    with rasterio.open(path) as dataset:
        lookup = band_lookup(dataset)
        for position, row in enumerate(frame.itertuples(index=False)):
            raster_row, raster_col = dataset.index(float(row.bng_x), float(row.bng_y))
            if not (0 <= raster_row < dataset.height and 0 <= raster_col < dataset.width):
                continue
            window = Window(raster_col, raster_row, 1, 1)
            for target, description in mapping.items():
                key = str(description).lower()
                if key not in lookup:
                    raise RuntimeError(f"Missing supplied 50 m band: {description}")
                value = dataset.read(lookup[key], window=window, masked=True)
                if value.count():
                    output[target][position] = float(value.compressed()[0])
    return output


def aggregate_local_vci(path: Path, frame: pd.DataFrame) -> np.ndarray:
    with rasterio.open(path) as dataset:
        lookup = band_lookup(dataset)
        if "lidar_vci_2m" not in lookup:
            raise RuntimeError("LiDAR raster does not contain lidar_vci_2m")
        values = dataset.read(lookup["lidar_vci_2m"], masked=True)
        data = np.asarray(values.filled(np.nan), dtype=np.float32)
    result = np.full(len(frame), np.nan, dtype=np.float32)
    for position, row in enumerate(frame.itertuples(index=False)):
        r0 = int(row.raster_patch_row) * 5
        c0 = int(row.raster_patch_column) * 5
        patch = data[r0 : r0 + 5, c0 : c0 + 5]
        valid = patch[np.isfinite(patch)]
        if len(valid) >= 20:
            result[position] = float(np.mean(valid))
    return result


def spline_height_model(knots: int, alpha: float):
    return make_pipeline(
        StandardScaler(),
        SplineTransformer(n_knots=knots, degree=3, include_bias=False),
        Ridge(alpha=alpha),
    )


def intersect_indices(indices: np.ndarray, valid: np.ndarray) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    return indices[valid[indices]]


def build_fold(
    fold: int,
    targets: np.ndarray,
    controls: np.ndarray,
    valid: np.ndarray,
    settings: dict[str, Any],
) -> dict[str, Any]:
    source_path = ROOT / str(settings["frozen_inputs"]["fold_directory"]) / f"fold_{fold}.npz"
    with np.load(source_path) as source:
        weights = source["weights"].astype(np.float32)
        train = intersect_indices(source["train_indices"], valid)
        subtrain = intersect_indices(source["subtrain_indices"], valid)
        validation = intersect_indices(source["validation_indices"], valid)
        test = intersect_indices(source["test_indices"], valid)
    minimum = int(settings["quality_control"]["minimum_test_rows_per_fold"])
    if len(test) < minimum:
        raise RuntimeError(f"Fold {fold} retains only {len(test)} test rows; minimum is {minimum}")

    tuning = np.full((len(targets), targets.shape[1] * 2), np.nan, dtype=np.float32)
    final = np.full_like(tuning, np.nan)
    tuning[:, : targets.shape[1]] = targets
    final[:, : targets.shape[1]] = targets
    height_test_predictions = np.full((len(test), targets.shape[1]), np.nan, dtype=np.float32)
    knots = int(settings["targets"]["spline_knots"])
    alpha = float(settings["targets"]["ridge_alpha"])

    for column in range(targets.shape[1]):
        tune_model = spline_height_model(knots, alpha).fit(controls[subtrain], targets[subtrain, column])
        tune_rows = np.concatenate([subtrain, validation])
        tuning[tune_rows, targets.shape[1] + column] = (
            targets[tune_rows, column] - tune_model.predict(controls[tune_rows])
        )
        final_model = spline_height_model(knots, alpha).fit(controls[train], targets[train, column])
        final_rows = np.concatenate([train, test])
        prediction = final_model.predict(controls[final_rows])
        final[final_rows, targets.shape[1] + column] = targets[final_rows, column] - prediction
        height_test_predictions[:, column] = final_model.predict(controls[test]).astype(np.float32)

    return {
        "weights": weights,
        "train_indices": train,
        "subtrain_indices": subtrain,
        "validation_indices": validation,
        "test_indices": test,
        "tuning_targets": tuning,
        "final_targets": final,
        "height_test_predictions": height_test_predictions,
    }


def run() -> None:
    config = load_config()
    frozen = config["frozen_inputs"]
    records: dict[str, Any] = {}
    for key in ["cohort", "lidar_metrics", "tessera_features", "conventional_features", "row_id", "common_valid"]:
        records[key] = verify_file(
            ROOT / str(frozen[f"{key}_path"]), expected_hash=str(frozen[f"{key}_sha256"])
        )
    for fold in range(int(config["evaluation"]["folds"])):
        path = ROOT / str(frozen["fold_directory"]) / f"fold_{fold}.npz"
        records[f"fold_{fold}"] = verify_file(path)
    for key, item in config["downloads"].items():
        records[key] = verify_file(ROOT / str(item["path"]), expected_bytes=int(item["expected_bytes"]))

    cohort = pd.read_parquet(ROOT / str(frozen["cohort_path"])).sort_values("row_id").reset_index(drop=True)
    row_ids = np.load(ROOT / str(frozen["row_id_path"]))
    if not np.array_equal(row_ids, cohort["row_id"].to_numpy()):
        raise RuntimeError("Feature row IDs no longer match the frozen cohort")

    chm_path = ROOT / str(config["downloads"]["chm_1m"]["path"])
    mask_paths = [
        ROOT / str(config["downloads"]["slope_mask"]["path"]),
        ROOT / str(config["downloads"]["slope_of_slope_mask"]["path"]),
    ]
    recomputed = {key: np.full(len(cohort), np.nan, dtype=np.float32) for key in [
        "chm_valid_pixels", "chm_p999_m", "canopy_surface_sd_m", "canopy_surface_rcv", "canopy_open_fraction", "qa_flagged_fraction"
    ]}
    with rasterio.open(chm_path) as chm, rasterio.open(mask_paths[0]) as mask_a, rasterio.open(mask_paths[1]) as mask_b:
        mode, bands = identify_chm_bands(chm)
        for position, row in enumerate(cohort.itertuples(index=False)):
            window = metric_window(chm, float(row.bng_x), float(row.bng_y))
            metrics = canopy_metrics(read_chm_values(chm, mode, bands, window))
            recomputed["chm_valid_pixels"][position] = metrics["valid"]
            recomputed["chm_p999_m"][position] = metrics["p999"]
            recomputed["canopy_surface_sd_m"][position] = metrics["sd"]
            recomputed["canopy_surface_rcv"][position] = metrics["rcv"]
            recomputed["canopy_open_fraction"][position] = metrics["gap"]
            accepted_a = accepted_mask(mask_a, metric_window(mask_a, float(row.bng_x), float(row.bng_y)))
            accepted_b = accepted_mask(mask_b, metric_window(mask_b, float(row.bng_x), float(row.bng_y)))
            if accepted_a.shape != accepted_b.shape or not len(accepted_a):
                raise RuntimeError("Terrain quality masks do not align at a cohort cell")
            recomputed["qa_flagged_fraction"][position] = float(
                np.mean(~(accepted_a & accepted_b))
            )
            if (position + 1) % 500 == 0:
                print(f"targets {position + 1:,}/{len(cohort):,}", flush=True)

    for name, values in recomputed.items():
        cohort[name] = values
    cohort["mean_local_vci_2m"] = aggregate_local_vci(ROOT / str(frozen["lidar_metrics_path"]), cohort)
    supplied = sample_supplied_metrics(
        ROOT / str(config["downloads"]["chm_metrics_50m"]["path"]),
        cohort,
        config["targets"]["supplied_50m_comparison"],
    )
    comparison: dict[str, Any] = {}
    for target, values in supplied.items():
        column = f"supplied_{target}"
        cohort[column] = values
        valid_pair = np.isfinite(cohort[target]) & np.isfinite(values)
        rho = float(spearmanr(cohort.loc[valid_pair, target], values[valid_pair]).statistic)
        comparison[target] = {
            "rows": int(valid_pair.sum()),
            "spearman_r": rho,
            "mae": float(np.mean(np.abs(cohort.loc[valid_pair, target] - values[valid_pair]))),
        }
    required = ["canopy_surface_sd_m", "canopy_open_fraction"]
    threshold = float(config["quality_control"]["minimum_supplied_recomputed_spearman"])
    for target in required:
        if comparison[target]["spearman_r"] < threshold:
            raise RuntimeError(f"Recomputed {target} does not reproduce the supplied metric closely enough")

    qc = config["quality_control"]
    common_valid = np.load(ROOT / str(frozen["common_valid_path"])).astype(bool)
    valid = (
        common_valid
        & (cohort["chm_valid_pixels"].to_numpy() >= int(config["targets"]["minimum_valid_chm_pixels"]))
        & (cohort["chm_p999_m"].to_numpy() <= float(qc["maximum_chm_height_m"]))
        & cohort["canopy_surface_sd_m"].between(0, float(qc["maximum_surface_sd_m"]), inclusive="both").to_numpy()
        & cohort["canopy_surface_rcv"].between(0, float(qc["maximum_surface_rcv"]), inclusive="both").to_numpy()
        & cohort["canopy_open_fraction"].between(0, 1, inclusive="both").to_numpy()
        & cohort["mean_local_vci_2m"].between(0, 1, inclusive="both").to_numpy()
        & (cohort["qa_flagged_fraction"].to_numpy() <= float(qc["maximum_flagged_fraction"]))
    )
    cohort["phase20_valid"] = valid
    target_names = [str(value) for value in config["targets"]["primary"]]
    target_matrix = cohort[target_names].to_numpy(dtype=np.float32)
    control_names = [str(value) for value in config["targets"]["height_adjustment_predictors"]]
    controls = cohort[control_names].to_numpy(dtype=np.float32)

    # Preserve the expensive 1 m target extraction even if a later fold gate
    # identifies a problem.
    atomic_parquet(cohort, TARGET_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for fold in range(int(config["evaluation"]["folds"])):
        values = build_fold(fold, target_matrix, controls, valid, config)
        temporary = OUTPUT_DIR / f"fold_{fold}.tmp.npz"
        np.savez_compressed(temporary, **values)
        temporary.replace(OUTPUT_DIR / f"fold_{fold}.npz")
    sensitivity = {
        str(value): int((common_valid & (cohort["qa_flagged_fraction"].to_numpy() <= float(value))).sum())
        for value in qc["sensitivity_thresholds"]
    }
    status = {
        "created_utc": utc_now(),
        "state": "complete",
        "rows_total": len(cohort),
        "rows_valid": int(valid.sum()),
        "chm_mode": mode,
        "chm_bands": list(bands),
        "target_names": target_names,
        "height_adjusted_names": [f"{name}_height_adjusted" for name in target_names],
        "supplied_recomputed_comparison": comparison,
        "qa_sensitivity_rows": sensitivity,
        "fold_test_rows": {
            str(fold): int(np.load(OUTPUT_DIR / f"fold_{fold}.npz")["test_indices"].size)
            for fold in range(int(config["evaluation"]["folds"]))
        },
        "output_sha256": sha256(TARGET_PATH),
    }
    atomic_json(STATUS_PATH, status)
    atomic_json(FREEZE_PATH, {"created_utc": utc_now(), "config_sha256": sha256(CONFIG_PATH), "inputs": records, "outputs": status})
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    run()
