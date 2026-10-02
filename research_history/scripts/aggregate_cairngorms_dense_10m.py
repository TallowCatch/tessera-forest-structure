#!/usr/bin/env python3
"""Aggregate dense Cairngorms folds and write block-balanced results and maps."""

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
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/cairngorms_dense_10m.yaml"


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


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "cairngorms_dense_10m"
    ]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def weighted_correlation(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    weights = weights / weights.sum()
    x_mean = np.sum(weights * x)
    y_mean = np.sum(weights * y)
    covariance = np.sum(weights * (x - x_mean) * (y - y_mean))
    x_variance = np.sum(weights * np.square(x - x_mean))
    y_variance = np.sum(weights * np.square(y - y_mean))
    if x_variance <= 0 or y_variance <= 0:
        return float("nan")
    return float(covariance / np.sqrt(x_variance * y_variance))


def metric_values(
    observed: np.ndarray, predicted: np.ndarray, blocks: np.ndarray
) -> dict[str, float]:
    unique, inverse, counts = np.unique(blocks, return_inverse=True, return_counts=True)
    del unique
    weights = 1.0 / counts[inverse].astype(np.float64)
    weights /= weights.sum()
    residual = predicted - observed
    observed_mean = np.sum(weights * observed)
    denominator = np.sum(weights * np.square(observed - observed_mean))
    rmse = np.sqrt(np.sum(weights * np.square(residual)))
    return {
        "rmse": float(rmse),
        "mae": float(np.sum(weights * np.abs(residual))),
        "r2": float(1.0 - np.sum(weights * np.square(residual)) / denominator),
        "pearson_r": weighted_correlation(observed, predicted, weights),
        "spearman_r": weighted_correlation(
            rankdata(observed), rankdata(predicted), weights
        ),
        "bias": float(np.sum(weights * residual)),
    }


def read_seed_results(
    result_dir: Path,
    model: str,
    fold: int,
    seeds: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    test_indices: np.ndarray | None = None
    observed: np.ndarray | None = None
    predictions: list[np.ndarray] = []
    for seed in seeds:
        path = result_dir / f"{model}_fold{fold}_seed{seed}.npz"
        if not path.exists():
            raise RuntimeError(f"Missing result: {path}")
        with np.load(path) as result:
            local_indices = result["test_indices"].astype(np.int64)
            local_observed = result["observed"].astype(np.float64)
            local_prediction = result["predicted"].astype(np.float64)
        if test_indices is None:
            test_indices = local_indices
            observed = local_observed
        elif not np.array_equal(test_indices, local_indices):
            raise RuntimeError(f"Seed test rows differ for {model} fold {fold}")
        elif not np.allclose(observed, local_observed, equal_nan=True):
            raise RuntimeError(f"Seed observations differ for {model} fold {fold}")
        predictions.append(local_prediction)
    assert test_indices is not None and observed is not None
    stacked = np.stack(predictions, axis=0)
    return (
        test_indices,
        observed,
        stacked.mean(axis=0),
        stacked.std(axis=0, ddof=1),
    )


def block_bootstrap_comparison(
    observed: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    blocks: np.ndarray,
    replicates: int,
    seed: int,
) -> tuple[float, float, float]:
    unique, inverse = np.unique(blocks, return_inverse=True)
    candidate_sse = np.bincount(
        inverse, weights=np.square(candidate - observed), minlength=len(unique)
    )
    reference_sse = np.bincount(
        inverse, weights=np.square(reference - observed), minlength=len(unique)
    )
    count = np.bincount(inverse, minlength=len(unique)).astype(np.float64)
    candidate_mse = candidate_sse / count
    reference_mse = reference_sse / count
    observed_delta = float(
        np.sqrt(candidate_mse.mean()) - np.sqrt(reference_mse.mean())
    )
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        selected = rng.integers(0, len(unique), size=len(unique))
        draws[replicate] = np.sqrt(candidate_mse[selected].mean()) - np.sqrt(
            reference_mse[selected].mean()
        )
    return (
        observed_delta,
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def write_maps(
    config: dict[str, Any],
    model_frame: pd.DataFrame,
    target_names: list[str],
    shape: tuple[int, int],
    rows: np.ndarray,
    columns: np.ndarray,
) -> dict[str, str]:
    map_dir = ROOT / str(config["outputs"]["map_directory"])
    map_dir.mkdir(parents=True, exist_ok=True)
    source = ROOT / str(config["source"]["lidar_metrics_path"])
    with rasterio.open(source) as dataset:
        profile = dataset.profile.copy()
        transform = dataset.transform
    profile.update(
        driver="GTiff",
        dtype="float32",
        count=len(target_names),
        nodata=-9999.0,
        crs=config["study"]["analysis_crs"],
        transform=transform,
        compress="deflate",
        predictor=3,
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    prediction_path = map_dir / "selected_model_oof_prediction.tif"
    uncertainty_path = map_dir / "selected_model_seed_sd.tif"
    selected_indices = model_frame["row_index"].to_numpy(dtype=np.int64)
    for path, prefix in [
        (prediction_path, "predicted_"),
        (uncertainty_path, "seed_sd_"),
    ]:
        temporary = path.with_suffix(".tmp.tif")
        with rasterio.open(temporary, "w", **profile) as output:
            for band, target in enumerate(target_names, start=1):
                raster = np.full(shape, -9999.0, dtype=np.float32)
                raster[rows[selected_indices], columns[selected_indices]] = model_frame[
                    f"{prefix}{target}"
                ].to_numpy(dtype=np.float32)
                output.write(raster, band)
                output.set_band_description(band, target)
        temporary.replace(path)
    return {
        "prediction_map": str(prediction_path.relative_to(ROOT)),
        "prediction_map_sha256": sha256(prediction_path),
        "seed_sd_map": str(uncertainty_path.relative_to(ROOT)),
        "seed_sd_map_sha256": sha256(uncertainty_path),
    }


def main() -> None:
    config = load_config()
    dense_dir = ROOT / str(config["outputs"]["dense_directory"])
    result_dir = ROOT / str(config["outputs"]["result_directory"])
    manifest = json.loads((dense_dir / "manifest.json").read_text(encoding="utf-8"))
    target_names = [str(value) for value in config["targets"]["names"]]
    models = [str(value) for value in config["models"]["names"]]
    seeds = [int(value) for value in config["models"]["seeds"]]
    region_count = int(config["spatial_evaluation"]["region_count"])
    blocks = np.load(dense_dir / "spatial_block.npy", mmap_mode="r")
    rows = np.load(dense_dir / "row.npy", mmap_mode="r")
    columns = np.load(dense_dir / "column.npy", mmap_mode="r")
    records: list[pd.DataFrame] = []
    seed_sd_by_model: dict[str, list[pd.DataFrame]] = {model: [] for model in models}
    metric_records: list[dict[str, Any]] = []
    for model in models:
        for fold in range(region_count):
            indices, observed, predicted, seed_sd = read_seed_results(
                result_dir, model, fold, seeds
            )
            frame = pd.DataFrame(
                {
                    "model": model,
                    "fold": fold,
                    "row_index": indices,
                    "spatial_block": np.asarray(blocks[indices], dtype=np.int32),
                }
            )
            for target_index, target in enumerate(target_names):
                frame[f"observed_{target}"] = observed[:, target_index].astype(np.float32)
                frame[f"predicted_{target}"] = predicted[:, target_index].astype(np.float32)
                frame[f"seed_sd_{target}"] = seed_sd[:, target_index].astype(np.float32)
                metric_records.append(
                    {
                        "model": model,
                        "fold": fold,
                        "target": target,
                        "rows": len(indices),
                        **metric_values(
                            observed[:, target_index],
                            predicted[:, target_index],
                            np.asarray(blocks[indices]),
                        ),
                    }
                )
            records.append(frame)
            seed_sd_by_model[model].append(frame)
    prediction_frame = pd.concat(records, ignore_index=True)
    prediction_path = ROOT / str(config["outputs"]["prediction_table"])
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = prediction_path.with_suffix(".tmp.parquet")
    prediction_frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(prediction_path)

    metrics = pd.DataFrame(metric_records)
    metrics_path = ROOT / str(config["outputs"]["metrics_table"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_path, index=False)
    pooled_records: list[dict[str, Any]] = []
    for model in models:
        local = prediction_frame[prediction_frame["model"] == model]
        for target in target_names:
            pooled_records.append(
                {
                    "model": model,
                    "target": target,
                    "rows": len(local),
                    **metric_values(
                        local[f"observed_{target}"].to_numpy(dtype=np.float64),
                        local[f"predicted_{target}"].to_numpy(dtype=np.float64),
                        local["spatial_block"].to_numpy(dtype=np.int64),
                    ),
                }
            )
    pooled = pd.DataFrame(pooled_records)
    pooled_path = ROOT / str(config["outputs"]["pooled_metrics_table"])
    pooled.to_csv(pooled_path, index=False)

    primary = str(config["targets"]["primary"])
    reference = str(config["selection"]["reference_model"])
    reference_frame = prediction_frame[prediction_frame["model"] == reference].sort_values(
        "row_index"
    )
    comparisons: list[dict[str, Any]] = []
    for model in models:
        if model == reference:
            continue
        candidate_frame = prediction_frame[prediction_frame["model"] == model].sort_values(
            "row_index"
        )
        if not np.array_equal(
            reference_frame["row_index"].to_numpy(),
            candidate_frame["row_index"].to_numpy(),
        ):
            raise RuntimeError(f"OOF rows differ for {model} and {reference}")
        observed = reference_frame[f"observed_{primary}"].to_numpy(dtype=np.float64)
        reference_prediction = reference_frame[f"predicted_{primary}"].to_numpy(
            dtype=np.float64
        )
        candidate_prediction = candidate_frame[f"predicted_{primary}"].to_numpy(
            dtype=np.float64
        )
        delta, lower, upper = block_bootstrap_comparison(
            observed,
            candidate_prediction,
            reference_prediction,
            reference_frame["spatial_block"].to_numpy(dtype=np.int64),
            int(config["uncertainty"]["block_bootstrap_replicates"]),
            int(config["uncertainty"]["seed"]),
        )
        reference_rmse = float(
            pooled[(pooled["model"] == reference) & (pooled["target"] == primary)][
                "rmse"
            ].iloc[0]
        )
        candidate_rmse = float(
            pooled[(pooled["model"] == model) & (pooled["target"] == primary)][
                "rmse"
            ].iloc[0]
        )
        reduction = (reference_rmse - candidate_rmse) / reference_rmse
        comparisons.append(
            {
                "candidate": model,
                "reference": reference,
                "target": primary,
                "candidate_rmse": candidate_rmse,
                "reference_rmse": reference_rmse,
                "rmse_reduction_fraction": reduction,
                "paired_rmse_delta": delta,
                "paired_rmse_delta_ci_lower": lower,
                "paired_rmse_delta_ci_upper": upper,
                "material_gate_passed": bool(
                    reduction
                    >= float(
                        config["selection"][
                            "minimum_material_rmse_reduction_fraction"
                        ]
                    )
                    and upper < 0
                ),
            }
        )
    comparison_frame = pd.DataFrame(comparisons)
    comparison_path = ROOT / str(config["outputs"]["comparisons_table"])
    comparison_frame.to_csv(comparison_path, index=False)

    primary_metrics = pooled[pooled["target"] == primary].sort_values("rmse")
    selected_model = str(primary_metrics.iloc[0]["model"])
    selected_frame = prediction_frame[
        prediction_frame["model"] == selected_model
    ].sort_values("row_index")
    map_records = write_maps(
        config,
        selected_frame,
        target_names,
        tuple(int(value) for value in manifest["shape"]),
        rows,
        columns,
    )
    freeze_path = ROOT / str(config["outputs"]["result_freeze"])
    atomic_json(
        freeze_path,
        {
            "created_utc": utc_now(),
            "config_sha256": sha256(CONFIG_PATH),
            "dense_manifest_sha256": sha256(dense_dir / "manifest.json"),
            "models": models,
            "seeds": seeds,
            "target_names": target_names,
            "primary_target": primary,
            "selected_model": selected_model,
            "selection_note": "Selected by lowest block-balanced out-of-fold primary-target RMSE; map remains retrospective.",
            "primary_metrics": primary_metrics.to_dict(orient="records"),
            "comparison_records": comparison_frame.to_dict(orient="records"),
            "prediction_table": str(prediction_path.relative_to(ROOT)),
            "prediction_table_sha256": sha256(prediction_path),
            "metrics_table_sha256": sha256(metrics_path),
            "pooled_metrics_table_sha256": sha256(pooled_path),
            "comparisons_table_sha256": sha256(comparison_path),
            **map_records,
        },
    )
    print("primary target metrics", flush=True)
    print(primary_metrics[["model", "rmse", "r2", "spearman_r"]].to_string(index=False), flush=True)
    print(f"selected_model={selected_model}", flush=True)


if __name__ == "__main__":
    main()
