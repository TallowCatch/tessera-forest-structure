#!/usr/bin/env python3
"""Aggregate direct-profile models, diagnostics, strata, and maps."""

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
from sklearn.linear_model import Ridge


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


def block_weights(blocks: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(blocks, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse].astype(np.float64)
    return weights / weights.sum()


def weighted_correlation(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    x_mean = float(np.sum(weights * x))
    y_mean = float(np.sum(weights * y))
    x_centered = x - x_mean
    y_centered = y - y_mean
    denominator = np.sqrt(
        np.sum(weights * np.square(x_centered))
        * np.sum(weights * np.square(y_centered))
    )
    if denominator <= 0:
        return float("nan")
    return float(np.sum(weights * x_centered * y_centered) / denominator)


def scalar_metrics(
    observed: np.ndarray, predicted: np.ndarray, blocks: np.ndarray
) -> dict[str, float]:
    weights = block_weights(blocks)
    residual = predicted - observed
    observed_mean = float(np.sum(weights * observed))
    denominator = float(np.sum(weights * np.square(observed - observed_mean)))
    sse = float(np.sum(weights * np.square(residual)))
    return {
        "rmse": float(np.sqrt(sse)),
        "mae": float(np.sum(weights * np.abs(residual))),
        "r2": float(1.0 - sse / denominator) if denominator > 0 else float("nan"),
        "pearson_r": weighted_correlation(observed, predicted, weights),
        "spearman_r": weighted_correlation(
            rankdata(observed), rankdata(predicted), weights
        ),
        "bias": float(np.sum(weights * residual)),
    }


def profile_entropy(profile: np.ndarray) -> np.ndarray:
    values = np.asarray(profile, dtype=np.float64)
    return -np.where(
        values > 0,
        values * np.log(np.maximum(values, 1e-12)),
        0.0,
    ).sum(axis=1)


def jensen_shannon_divergence(observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    mixture = 0.5 * (observed + predicted)
    observed_term = np.where(
        observed > 0,
        observed * (np.log(np.maximum(observed, 1e-12)) - np.log(np.maximum(mixture, 1e-12))),
        0.0,
    )
    predicted_term = np.where(
        predicted > 0,
        predicted * (np.log(np.maximum(predicted, 1e-12)) - np.log(np.maximum(mixture, 1e-12))),
        0.0,
    )
    return 0.5 * (observed_term + predicted_term).sum(axis=1)


def profile_metrics(
    observed: np.ndarray, predicted: np.ndarray, blocks: np.ndarray
) -> dict[str, float]:
    weights = block_weights(blocks)
    squared = np.square(predicted - observed).mean(axis=1)
    js = jensen_shannon_divergence(observed, predicted)
    entropy_metrics = scalar_metrics(
        profile_entropy(observed), profile_entropy(predicted), blocks
    )
    return {
        "profile_rmse": float(np.sqrt(np.sum(weights * squared))),
        "profile_mae": float(np.sum(weights * np.abs(predicted - observed).mean(axis=1))),
        "mean_js_divergence": float(np.sum(weights * js)),
        "mean_jensen_shannon_distance": float(np.sum(weights * np.sqrt(np.maximum(js, 0.0)))),
        "entropy_rmse": entropy_metrics["rmse"],
        "entropy_r2": entropy_metrics["r2"],
        "entropy_spearman_r": entropy_metrics["spearman_r"],
        "maximum_profile_sum_error": float(np.max(np.abs(predicted.sum(axis=1) - 1.0))),
    }


def read_seed_results(
    result_directory: Path, model: str, fold: int, seeds: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    indices: np.ndarray | None = None
    observed: np.ndarray | None = None
    predictions: list[np.ndarray] = []
    for seed in seeds:
        path = result_directory / f"{model}_fold{fold}_seed{seed}.npz"
        if not path.exists():
            raise RuntimeError(f"Missing result: {path}")
        with np.load(path) as result:
            local_indices = result["test_indices"].astype(np.int64)
            local_observed = result["observed"].astype(np.float64)
            local_prediction = result["predicted"].astype(np.float64)
        if indices is None:
            indices = local_indices
            observed = local_observed
        elif not np.array_equal(indices, local_indices):
            raise RuntimeError(f"Seed test rows differ for {model} fold {fold}")
        elif not np.allclose(observed, local_observed):
            raise RuntimeError(f"Seed observations differ for {model} fold {fold}")
        predictions.append(local_prediction)
    assert indices is not None and observed is not None
    stacked = np.stack(predictions, axis=0)
    mean = stacked.mean(axis=0)
    mean /= mean.sum(axis=1, keepdims=True)
    return indices, observed, mean, stacked.std(axis=0, ddof=1)


def normalize_predictions(values: np.ndarray) -> np.ndarray:
    clipped = np.maximum(np.asarray(values, dtype=np.float64), 1e-8)
    return clipped / clipped.sum(axis=1, keepdims=True)


def diagnostic_predictions(
    model: str,
    train: np.ndarray,
    test: np.ndarray,
    targets: np.ndarray,
    covariates: np.ndarray,
    blocks: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    weights = block_weights(blocks[train]) * len(train)
    if model == "training_profile_mean":
        mean = np.average(targets[train], axis=0, weights=weights)
        return np.repeat(mean[None], len(test), axis=0)
    columns = [0] if model == "lidar_height_only_ridge" else [0, 1, 2]
    x_train = covariates[train][:, columns].astype(np.float64)
    x_test = covariates[test][:, columns].astype(np.float64)
    mean = np.average(x_train, axis=0, weights=weights)
    variance = np.average(np.square(x_train - mean), axis=0, weights=weights)
    sd = np.sqrt(np.maximum(variance, 1e-12))
    estimator = Ridge(alpha=float(config["diagnostic_baselines"]["ridge_alpha"]))
    estimator.fit((x_train - mean) / sd, targets[train], sample_weight=weights)
    return normalize_predictions(estimator.predict((x_test - mean) / sd))


def result_frame(
    model: str,
    fold: int,
    indices: np.ndarray,
    observed: np.ndarray,
    predicted: np.ndarray,
    seed_sd: np.ndarray,
    blocks: np.ndarray,
    covariates: np.ndarray,
    labels: list[str],
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "model": model,
            "fold": fold,
            "row_index": indices,
            "spatial_block": blocks[indices].astype(np.int32),
            "mean_canopy_height_m": covariates[indices, 0].astype(np.float32),
            "canopy_cover": covariates[indices, 1].astype(np.float32),
            "gap_fraction": covariates[indices, 2].astype(np.float32),
        }
    )
    for layer_index, label in enumerate(labels):
        frame[f"observed_{label}"] = observed[:, layer_index].astype(np.float32)
        frame[f"predicted_{label}"] = predicted[:, layer_index].astype(np.float32)
        frame[f"seed_sd_{label}"] = seed_sd[:, layer_index].astype(np.float32)
    frame["observed_profile_entropy"] = profile_entropy(observed).astype(np.float32)
    frame["predicted_profile_entropy"] = profile_entropy(predicted).astype(np.float32)
    frame["jensen_shannon_distance"] = np.sqrt(
        np.maximum(jensen_shannon_divergence(observed, predicted), 0.0)
    ).astype(np.float32)
    return frame


def profile_arrays(frame: pd.DataFrame, labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    observed = frame[[f"observed_{label}" for label in labels]].to_numpy(dtype=np.float64)
    predicted = frame[[f"predicted_{label}" for label in labels]].to_numpy(dtype=np.float64)
    return observed, predicted


def block_bootstrap_delta(
    frame: pd.DataFrame,
    candidate_column: str,
    reference_column: str,
    replicates: int,
    seed: int,
) -> tuple[float, float, float]:
    grouped = frame.groupby("spatial_block", sort=False)[
        [candidate_column, reference_column]
    ].mean()
    differences = grouped[candidate_column].to_numpy() - grouped[reference_column].to_numpy()
    observed = float(differences.mean())
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        selected = rng.integers(0, len(differences), size=len(differences))
        draws[replicate] = differences[selected].mean()
    return observed, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def write_maps(
    config: dict[str, Any],
    frame: pd.DataFrame,
    labels: list[str],
    rows: np.ndarray,
    columns: np.ndarray,
    shape: tuple[int, int],
) -> dict[str, str]:
    directory = ROOT / str(config["outputs"]["map_directory"])
    directory.mkdir(parents=True, exist_ok=True)
    source = ROOT / str(config["source"]["lidar_metrics_path"])
    with rasterio.open(source) as dataset:
        profile = dataset.profile.copy()
    profile.update(
        driver="GTiff",
        dtype="float32",
        count=4,
        nodata=-9999.0,
        crs=config["study"]["analysis_crs"],
        compress="deflate",
        predictor=3,
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    path = directory / "selected_model_oof_profile.tif"
    temporary = path.with_suffix(".tmp.tif")
    ordered = frame.sort_values("row_index")
    indices = ordered["row_index"].to_numpy(dtype=np.int64)
    names = [*labels, "profile_entropy"]
    values = [
        *[ordered[f"predicted_{label}"].to_numpy(dtype=np.float32) for label in labels],
        ordered["predicted_profile_entropy"].to_numpy(dtype=np.float32),
    ]
    with rasterio.open(temporary, "w", **profile) as output:
        for band, (name, local_values) in enumerate(zip(names, values), start=1):
            raster = np.full(shape, -9999.0, dtype=np.float32)
            raster[rows[indices], columns[indices]] = local_values
            output.write(raster, band)
            output.set_band_description(band, name)
    temporary.replace(path)
    return {
        "selected_model_oof_map": str(path.relative_to(ROOT)),
        "selected_model_oof_map_sha256": sha256(path),
    }


def main() -> None:
    config = load_config()
    dense = ROOT / str(config["source"]["dense_directory"])
    profile_directory = ROOT / str(config["outputs"]["profile_directory"])
    result_directory = ROOT / str(config["outputs"]["result_directory"])
    manifest = json.loads((profile_directory / "manifest.json").read_text(encoding="utf-8"))
    dense_manifest = json.loads((dense / "manifest.json").read_text(encoding="utf-8"))
    labels = [str(value) for value in config["profile"]["layer_labels"]]
    tessera_models = [str(value) for value in config["models"]["names"]]
    diagnostic_models = [str(value) for value in config["diagnostic_baselines"]["names"]]
    seeds = [int(value) for value in config["models"]["seeds"]]
    folds = int(config["spatial_evaluation"]["region_count"])
    targets = np.load(profile_directory / "profile_targets.npy", mmap_mode="r")
    covariates = np.load(profile_directory / "diagnostic_covariates.npy", mmap_mode="r")
    blocks = np.asarray(np.load(dense / "spatial_block.npy", mmap_mode="r"))
    rows = np.asarray(np.load(dense / "row.npy", mmap_mode="r"))
    columns = np.asarray(np.load(dense / "column.npy", mmap_mode="r"))

    frames: list[pd.DataFrame] = []
    metric_records: list[dict[str, Any]] = []
    for fold in range(folds):
        with np.load(dense / "folds" / f"fold_{fold}.npz") as source:
            fold_data = {name: source[name] for name in source.files}
        test = fold_data["test_indices"].astype(np.int64)
        train = fold_data["train_indices"].astype(np.int64)
        for model in tessera_models:
            indices, observed, predicted, seed_sd = read_seed_results(
                result_directory, model, fold, seeds
            )
            if not np.array_equal(indices, test):
                raise RuntimeError(f"Frozen test rows differ for {model} fold {fold}")
            frame = result_frame(
                model,
                fold,
                indices,
                observed,
                predicted,
                seed_sd,
                blocks,
                covariates,
                labels,
            )
            frames.append(frame)
        for model in diagnostic_models:
            predicted = diagnostic_predictions(
                model, train, test, targets, covariates, blocks, config
            )
            frame = result_frame(
                model,
                fold,
                test,
                np.asarray(targets[test]),
                predicted,
                np.zeros_like(predicted),
                blocks,
                covariates,
                labels,
            )
            frames.append(frame)

    predictions = pd.concat(frames, ignore_index=True)
    for (model, fold), frame in predictions.groupby(["model", "fold"], sort=False):
        observed, predicted = profile_arrays(frame, labels)
        local_blocks = frame["spatial_block"].to_numpy(dtype=np.int64)
        record: dict[str, Any] = {
            "model": model,
            "fold": int(fold),
            "rows": len(frame),
            **profile_metrics(observed, predicted, local_blocks),
        }
        for layer_index, label in enumerate(labels):
            for key, value in scalar_metrics(
                observed[:, layer_index], predicted[:, layer_index], local_blocks
            ).items():
                record[f"{label}_{key}"] = value
        metric_records.append(record)

    prediction_path = ROOT / str(config["outputs"]["prediction_table"])
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = prediction_path.with_suffix(".tmp.parquet")
    predictions.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(prediction_path)
    metrics = pd.DataFrame(metric_records)
    metrics_path = ROOT / str(config["outputs"]["metrics_table"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_path, index=False)

    pooled_records: list[dict[str, Any]] = []
    all_models = [*tessera_models, *diagnostic_models]
    for model in all_models:
        frame = predictions[predictions["model"] == model]
        observed, predicted = profile_arrays(frame, labels)
        local_blocks = frame["spatial_block"].to_numpy(dtype=np.int64)
        record = {
            "model": model,
            "rows": len(frame),
            **profile_metrics(observed, predicted, local_blocks),
        }
        for layer_index, label in enumerate(labels):
            for key, value in scalar_metrics(
                observed[:, layer_index], predicted[:, layer_index], local_blocks
            ).items():
                record[f"{label}_{key}"] = value
        pooled_records.append(record)
    pooled = pd.DataFrame(pooled_records)
    height_baseline_rmse = float(
        pooled.loc[
            pooled["model"] == "lidar_height_only_ridge", "profile_rmse"
        ].iloc[0]
    )
    pooled["incremental_r2_over_lidar_height"] = 1.0 - np.square(
        pooled["profile_rmse"]
    ) / np.square(height_baseline_rmse)
    pooled_path = ROOT / str(config["outputs"]["pooled_metrics_table"])
    pooled.to_csv(pooled_path, index=False)

    strata_records: list[dict[str, Any]] = []
    intervals = config["profile"]["height_strata_m"]
    stratum_labels = config["profile"]["height_stratum_labels"]
    for model in all_models:
        frame = predictions[predictions["model"] == model]
        for stratum, interval in zip(stratum_labels, intervals):
            lower, upper = (float(value) for value in interval)
            selected = frame[
                (frame["mean_canopy_height_m"] >= lower)
                & (frame["mean_canopy_height_m"] < upper)
            ]
            observed, predicted = profile_arrays(selected, labels)
            strata_records.append(
                {
                    "model": model,
                    "height_stratum": stratum,
                    "lower_m": lower,
                    "upper_m": upper,
                    "rows": len(selected),
                    **profile_metrics(
                        observed,
                        predicted,
                        selected["spatial_block"].to_numpy(dtype=np.int64),
                    ),
                }
            )
    strata = pd.DataFrame(strata_records)
    strata_path = ROOT / str(config["outputs"]["height_strata_table"])
    strata.to_csv(strata_path, index=False)

    wide_js = predictions.pivot(
        index=["row_index", "spatial_block"], columns="model", values="jensen_shannon_distance"
    ).reset_index()
    reference = "summary_mlp"
    comparisons: list[dict[str, Any]] = []
    for model in ["residual_patch_cnn", "compact_unet"]:
        delta, lower, upper = block_bootstrap_delta(
            wide_js,
            model,
            reference,
            int(config["uncertainty"]["block_bootstrap_replicates"]),
            int(config["uncertainty"]["seed"]),
        )
        reference_value = float(
            pooled.loc[pooled["model"] == reference, "mean_jensen_shannon_distance"].iloc[0]
        )
        candidate_value = float(
            pooled.loc[pooled["model"] == model, "mean_jensen_shannon_distance"].iloc[0]
        )
        reduction = (reference_value - candidate_value) / reference_value
        comparisons.append(
            {
                "candidate": model,
                "reference": reference,
                "metric": "mean_jensen_shannon_distance",
                "candidate_value": candidate_value,
                "reference_value": reference_value,
                "reduction_fraction": reduction,
                "paired_delta": delta,
                "paired_delta_ci_lower": lower,
                "paired_delta_ci_upper": upper,
                "material_gate_passed": bool(
                    reduction
                    >= float(config["selection"]["minimum_material_reduction_fraction"])
                    and upper < 0
                ),
            }
        )
    comparison_frame = pd.DataFrame(comparisons)
    comparison_path = ROOT / str(config["outputs"]["comparisons_table"])
    comparison_frame.to_csv(comparison_path, index=False)

    entropy_records: list[dict[str, Any]] = []
    old_prediction_path = ROOT / str(config["source"]["dense_prediction_table"])
    if old_prediction_path.exists():
        old = pd.read_parquet(
            old_prediction_path,
            columns=[
                "model",
                "row_index",
                "spatial_block",
                "observed_three_layer_volume_entropy_50m_sliding",
                "predicted_three_layer_volume_entropy_50m_sliding",
            ],
        )
        for model in tessera_models:
            profile_frame = predictions[predictions["model"] == model][
                ["row_index", "predicted_profile_entropy"]
            ]
            direct_frame = old[old["model"] == model]
            merged = direct_frame.merge(profile_frame, on="row_index", validate="one_to_one")
            observed = merged[
                "observed_three_layer_volume_entropy_50m_sliding"
            ].to_numpy(dtype=np.float64)
            direct = merged[
                "predicted_three_layer_volume_entropy_50m_sliding"
            ].to_numpy(dtype=np.float64)
            derived = merged["predicted_profile_entropy"].to_numpy(dtype=np.float64)
            local_blocks = merged["spatial_block"].to_numpy(dtype=np.int64)
            direct_metrics = scalar_metrics(observed, direct, local_blocks)
            derived_metrics = scalar_metrics(observed, derived, local_blocks)
            entropy_records.extend(
                [
                    {"model": model, "training_target": "direct_entropy", **direct_metrics},
                    {"model": model, "training_target": "profile_derived_entropy", **derived_metrics},
                ]
            )
    entropy_frame = pd.DataFrame(entropy_records)
    entropy_path = ROOT / str(config["outputs"]["entropy_comparison_table"])
    entropy_frame.to_csv(entropy_path, index=False)

    selected_model = str(
        pooled[pooled["model"].isin(tessera_models)]
        .sort_values("mean_jensen_shannon_distance")
        .iloc[0]["model"]
    )
    selected = predictions[predictions["model"] == selected_model]
    map_records = write_maps(
        config,
        selected,
        labels,
        rows,
        columns,
        tuple(int(value) for value in dense_manifest["shape"]),
    )
    freeze_path = ROOT / str(config["outputs"]["result_freeze"])
    atomic_json(
        freeze_path,
        {
            "created_utc": utc_now(),
            "config_sha256": sha256(CONFIG_PATH),
            "profile_manifest_sha256": sha256(profile_directory / "manifest.json"),
            "profile_layers": labels,
            "models": tessera_models,
            "diagnostic_baselines": diagnostic_models,
            "seeds": seeds,
            "rows": int(manifest["rows"]),
            "selected_model": selected_model,
            "selection_note": "Lowest block-balanced out-of-fold mean Jensen-Shannon distance among TESSERA models; the map is retrospective.",
            "pooled_records": pooled.to_dict(orient="records"),
            "comparison_records": comparison_frame.to_dict(orient="records"),
            "prediction_table": str(prediction_path.relative_to(ROOT)),
            "prediction_table_sha256": sha256(prediction_path),
            "metrics_table_sha256": sha256(metrics_path),
            "pooled_metrics_table_sha256": sha256(pooled_path),
            "height_strata_table_sha256": sha256(strata_path),
            "comparisons_table_sha256": sha256(comparison_path),
            "entropy_comparison_table_sha256": sha256(entropy_path),
            **map_records,
        },
    )
    print("pooled profile metrics", flush=True)
    print(
        pooled[
            [
                "model",
                "mean_jensen_shannon_distance",
                "profile_rmse",
                "entropy_r2",
                "incremental_r2_over_lidar_height",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print(f"selected_model={selected_model}", flush=True)


if __name__ == "__main__":
    main()
