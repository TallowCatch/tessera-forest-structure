#!/usr/bin/env python3
"""Quantify spatial-support uncertainty in Cairngorms CHM metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import yaml
from rasterio.windows import from_bounds
from scipy.stats import kurtosis

try:
    import prepare_cairngorms_components as common
except ModuleNotFoundError:
    from scripts import prepare_cairngorms_components as common


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/reference_uncertainty_cairngorms_reference_uncertainty.yaml"
LABELS = {
    "canopy_mean_height_m": "Mean height",
    "canopy_p95_height_m": "P95 height",
    "canopy_surface_sd_m": "Height SD",
    "canopy_surface_cv": "Height CV",
    "canopy_surface_rcv": "Robust CV",
    "canopy_rumple": "Rumple",
    "canopy_open_fraction": "Openings",
    "canopy_height_kurtosis": "Kurtosis",
}


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "phase40_cairngorms_reference_uncertainty"
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def surface_area_ratio(grid: np.ndarray) -> float:
    z00 = grid[:-1, :-1]
    z10 = grid[:-1, 1:]
    z01 = grid[1:, :-1]
    z11 = grid[1:, 1:]
    valid = np.isfinite(z00) & np.isfinite(z10) & np.isfinite(z01) & np.isfinite(z11)
    if not valid.any():
        return math.nan
    area_a = 0.5 * np.sqrt(1.0 + np.square(z10 - z00) + np.square(z11 - z10))
    area_b = 0.5 * np.sqrt(1.0 + np.square(z11 - z01) + np.square(z01 - z00))
    return float(np.mean((area_a + area_b)[valid]))


def metric_values(grid: np.ndarray, opening_threshold: float) -> dict[str, float]:
    flat = grid[np.isfinite(grid) & (grid >= 0)].astype(np.float64)
    if len(flat) < 4:
        return {name: math.nan for name in LABELS}
    q25, median, q75, p95 = np.quantile(flat, [0.25, 0.50, 0.75, 0.95])
    mean = float(np.mean(flat))
    sd = float(np.std(flat, ddof=1))
    return {
        "canopy_mean_height_m": mean,
        "canopy_p95_height_m": float(p95),
        "canopy_surface_sd_m": sd,
        "canopy_surface_cv": float(sd / mean) if mean > 1e-6 else math.nan,
        "canopy_surface_rcv": float((q75 - q25) / median) if median > 1e-6 else math.nan,
        "canopy_rumple": surface_area_ratio(grid),
        "canopy_open_fraction": float(np.mean(flat < opening_threshold)),
        "canopy_height_kurtosis": float(kurtosis(flat, fisher=True, bias=False)),
    }


def block_labels(grid_size: int, block_size: int) -> np.ndarray:
    if grid_size % block_size:
        raise ValueError("Grid size must be divisible by block size")
    axis = np.arange(grid_size, dtype=np.int16) // block_size
    count = grid_size // block_size
    return axis[:, None] * count + axis[None, :]


def jackknife_metrics(
    grid: np.ndarray,
    block_size: int,
    opening_threshold: float,
) -> tuple[dict[str, float], dict[str, float], int]:
    labels = block_labels(grid.shape[0], block_size)
    unique = np.unique(labels)
    full = metric_values(grid, opening_threshold)
    leave_values = {name: [] for name in LABELS}
    for block in unique:
        omitted = grid.copy()
        omitted[labels == block] = np.nan
        values = metric_values(omitted, opening_threshold)
        for name, value in values.items():
            leave_values[name].append(value)
    standard_errors: dict[str, float] = {}
    b = len(unique)
    for name, values in leave_values.items():
        array = np.asarray(values, dtype=np.float64)
        if not np.isfinite(array).all():
            standard_errors[name] = math.nan
            continue
        mean = float(np.mean(array))
        standard_errors[name] = float(
            np.sqrt((b - 1.0) / b * np.sum(np.square(array - mean)))
        )
    return full, standard_errors, b


def worker(index: int) -> None:
    config = load_config()
    settings = config["analysis"]
    workers = int(settings["workers"])
    targets = pd.read_parquet(ROOT / str(config["inputs"]["targets"]))
    targets = targets[targets["phase22_valid"].astype(bool)].reset_index(drop=True)
    selected = targets.iloc[np.arange(len(targets)) % workers == index].copy()
    if selected.empty:
        raise RuntimeError(f"Worker {index} received no rows")
    records: list[dict[str, Any]] = []
    chm_path = ROOT / str(config["inputs"]["chm"])
    with rasterio.open(chm_path) as dataset:
        mode, bands = common.identify_chm_bands(dataset)
        for position, row in enumerate(selected.itertuples(index=False)):
            half = 25.0
            window = from_bounds(
                float(row.bng_x) - half,
                float(row.bng_y) - half,
                float(row.bng_x) + half,
                float(row.bng_y) + half,
                dataset.transform,
            ).round_offsets().round_lengths()
            values = common.read_chm_values(dataset, mode, bands, window)
            expected = int(settings["expected_grid_pixels"])
            if int(window.height) != expected or int(window.width) != expected:
                raise RuntimeError(
                    f"Unexpected CHM window for row {row.row_id}: "
                    f"{window.height} x {window.width}"
                )
            grid = values.reshape(expected, expected).astype(np.float64)
            full, standard_errors, blocks = jackknife_metrics(
                grid,
                int(settings["block_size_pixels"]),
                float(settings["opening_threshold_m"]),
            )
            for name in settings["metrics"]:
                stored = float(getattr(row, name))
                recalculated = float(full[name])
                records.append(
                    {
                        "row_id": int(row.row_id),
                        "fold_block": str(row.spatial_block),
                        "metric": name,
                        "stored_value": stored,
                        "recalculated_value": recalculated,
                        "absolute_recalculation_difference": abs(stored - recalculated),
                        "jackknife_se": float(standard_errors[name]),
                        "jackknife_variance": float(standard_errors[name] ** 2),
                        "jackknife_blocks": blocks,
                        "valid_pixels": int(np.isfinite(grid).sum()),
                    }
                )
            if (position + 1) % 100 == 0:
                print(
                    f"worker {index}: {position + 1:,}/{len(selected):,} units",
                    flush=True,
                )
    output = pd.DataFrame(records)
    output_root = ROOT / str(config["outputs"]["worker_directory"])
    atomic_parquet(output_root / f"worker_{index:02d}.parquet", output)
    print(
        f"phase40 worker {index + 1}/{workers} complete: "
        f"{len(selected):,} units; {len(output):,} metric records",
        flush=True,
    )


def equal_block_weights(blocks: pd.Series) -> np.ndarray:
    counts = blocks.astype(str).value_counts()
    weights = blocks.astype(str).map(1.0 / counts).to_numpy(dtype=np.float64)
    return weights * len(weights) / weights.sum()


def fold_comparison(
    uncertainty: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    merged = predictions.merge(
        uncertainty[["row_id", "metric", "jackknife_variance"]],
        left_on=["row_id", "target"],
        right_on=["row_id", "metric"],
        validate="one_to_one",
    )
    records: list[dict[str, Any]] = []
    for (target, fold), group in merged.groupby(["target", "fold"], sort=True):
        weights = equal_block_weights(group["spatial_block"])
        observed = group["observed"].to_numpy(dtype=np.float64)
        predicted = group["predicted"].to_numpy(dtype=np.float64)
        residual_square = np.square(predicted - observed)
        reference_variance = group["jackknife_variance"].to_numpy(dtype=np.float64)
        observed_mean = float(np.average(observed, weights=weights))
        observed_variance = float(
            np.average(np.square(observed - observed_mean), weights=weights)
        )
        model_mse = float(np.average(residual_square, weights=weights))
        reference_mse = float(np.average(reference_variance, weights=weights))
        latent_residual = max(model_mse - reference_mse, 0.0)
        latent_variance = max(observed_variance - reference_mse, 1e-12)
        positive = reference_variance[reference_variance > 0]
        variance_floor = float(np.quantile(positive, 0.05)) if len(positive) else 1e-12
        diagnostic_chi2 = float(
            np.average(
                residual_square / np.maximum(reference_variance, variance_floor),
                weights=weights,
            )
        )
        records.append(
            {
                "target": target,
                "fold": int(fold),
                "rows": len(group),
                "blocks": int(group["spatial_block"].nunique()),
                "model_rmse": math.sqrt(model_mse),
                "rms_jackknife_se": math.sqrt(reference_mse),
                "rms_jackknife_se_to_model_rmse": math.sqrt(reference_mse / model_mse),
                "observed_variance": observed_variance,
                "mean_reference_variance": reference_mse,
                "approximate_reliability": max(
                    0.0, min(1.0, 1.0 - reference_mse / observed_variance)
                ),
                "observed_r2": 1.0 - model_mse / observed_variance,
                "optimistic_error_adjusted_r2": 1.0
                - latent_residual / latent_variance,
                "diagnostic_reduced_chi2": diagnostic_chi2,
            }
        )
    return pd.DataFrame(records)


def make_figure(
    uncertainty: pd.DataFrame,
    summary: pd.DataFrame,
    predictions: pd.DataFrame,
    output_path: Path,
) -> None:
    order = list(LABELS)
    normalized = uncertainty.merge(
        summary[["metric", "between_unit_sd"]], on="metric", validate="many_to_one"
    )
    normalized["normalised_se"] = normalized["jackknife_se"] / normalized["between_unit_sd"]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8))
    data = [
        normalized.loc[normalized["metric"] == name, "normalised_se"].to_numpy()
        for name in order
    ]
    axes[0].boxplot(data, showfliers=False)
    axes[0].set_xticks(np.arange(1, len(order) + 1), [LABELS[name] for name in order], rotation=40, ha="right")
    axes[0].set_ylabel("Jackknife SE / between-unit SD")
    axes[0].set_title("a  Relative reference instability", loc="left", fontweight="bold")
    x = np.arange(len(order))
    selected_summary = summary.set_index("metric").reindex(order)
    axes[1].bar(x, selected_summary["rms_jackknife_se_to_model_rmse"], color="#2A8C6A")
    axes[1].set_xticks(x, [LABELS[name] for name in order], rotation=40, ha="right")
    axes[1].axhline(1, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("RMS jackknife SE / model RMSE")
    axes[1].set_title("b  Uncertainty relative to TESSERA error", loc="left", fontweight="bold")
    scoped = predictions[predictions["target"] == "canopy_height_kurtosis"].merge(
        uncertainty[uncertainty["metric"] == "canopy_height_kurtosis"][["row_id", "jackknife_se"]],
        on="row_id",
        validate="one_to_one",
    )
    rng = np.random.default_rng(20260830)
    if len(scoped) > 220:
        quantile = pd.qcut(scoped["observed"], 10, labels=False, duplicates="drop")
        sample_indices = []
        for value in sorted(quantile.dropna().unique()):
            candidates = scoped.index[quantile == value].to_numpy()
            sample_indices.extend(rng.choice(candidates, min(22, len(candidates)), replace=False))
        scoped = scoped.loc[sample_indices]
    axes[2].errorbar(
        scoped["observed"],
        scoped["predicted"],
        xerr=1.96 * scoped["jackknife_se"],
        fmt="o",
        markersize=2.5,
        alpha=0.30,
        ecolor="#8AA5B5",
        color="#315D73",
        linewidth=0.5,
    )
    low = min(scoped["observed"].min(), scoped["predicted"].min())
    high = max(scoped["observed"].max(), scoped["predicted"].max())
    axes[2].plot([low, high], [low, high], linestyle="--", color="#B14E3B", linewidth=1)
    axes[2].set_xlabel("Observed CHM kurtosis")
    axes[2].set_ylabel("Held-out TESSERA prediction")
    axes[2].set_title("c  Kurtosis with 95% jackknife intervals", loc="left", fontweight="bold")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def aggregate() -> None:
    config = load_config()
    result_path = ROOT / str(config["outputs"]["result_freeze"])
    if result_path.exists():
        raise RuntimeError("Phase 40 result freeze already exists")
    worker_root = ROOT / str(config["outputs"]["worker_directory"])
    workers = int(config["analysis"]["workers"])
    paths = [worker_root / f"worker_{index:02d}.parquet" for index in range(workers)]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise RuntimeError("Missing worker outputs:\n" + "\n".join(missing))
    uncertainty = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    expected = 17316 * len(config["analysis"]["metrics"])
    if len(uncertainty) != expected:
        raise RuntimeError(f"Unexpected uncertainty rows: {len(uncertainty)} != {expected}")
    maximum_difference = float(uncertainty["absolute_recalculation_difference"].max())
    if maximum_difference > 1e-3:
        raise RuntimeError(f"CHM target recalculation mismatch: {maximum_difference}")
    prediction = pd.read_parquet(ROOT / str(config["inputs"]["predictions"]))
    prediction = prediction[
        (prediction["variant"] == config["analysis"]["prediction_variant"])
        & (prediction["version"] == config["analysis"]["prediction_version"])
        & (prediction["model_family"] == config["analysis"]["prediction_model_family"])
        & prediction["target"].isin(config["analysis"]["metrics"])
    ].copy()
    height_prediction = pd.read_parquet(
        ROOT / str(config["inputs"]["height_predictions"])
    )
    height_prediction = height_prediction[
        (height_prediction["model"] == config["analysis"]["height_prediction_model"])
        & (height_prediction["target"] == "canopy_mean_height_m")
    ][
        ["row_id", "fold", "target", "observed", "predicted", "spatial_block"]
    ].copy()
    prediction = pd.concat(
        [
            prediction[
                ["row_id", "fold", "target", "observed", "predicted", "spatial_block"]
            ],
            height_prediction,
        ],
        ignore_index=True,
    )
    duplicated = prediction.duplicated(["row_id", "target"], keep=False)
    if duplicated.any():
        raise RuntimeError("Duplicate held-out predictions in Phase 40 model set")
    fold = fold_comparison(uncertainty, prediction)
    unit_summary = uncertainty.groupby("metric", sort=True).agg(
        units=("row_id", "nunique"),
        valid_pixels_median=("valid_pixels", "median"),
        jackknife_se_median=("jackknife_se", "median"),
        jackknife_se_p95=("jackknife_se", lambda values: float(np.quantile(values, 0.95))),
        mean_reference_variance=("jackknife_variance", "mean"),
        maximum_recalculation_difference=("absolute_recalculation_difference", "max"),
    ).reset_index()
    between = uncertainty.groupby("metric")["stored_value"].std(ddof=1).rename("between_unit_sd").reset_index()
    fold_summary = fold.groupby("target", sort=True).agg(
        model_rmse=("model_rmse", "mean"),
        rms_jackknife_se=("rms_jackknife_se", "mean"),
        rms_jackknife_se_to_model_rmse=("rms_jackknife_se_to_model_rmse", "mean"),
        approximate_reliability=("approximate_reliability", "mean"),
        observed_r2=("observed_r2", "mean"),
        optimistic_error_adjusted_r2=("optimistic_error_adjusted_r2", "mean"),
        diagnostic_reduced_chi2=("diagnostic_reduced_chi2", "mean"),
    ).reset_index().rename(columns={"target": "metric"})
    summary = unit_summary.merge(between, on="metric", validate="one_to_one").merge(
        fold_summary, on="metric", validate="one_to_one"
    )
    summary["median_se_to_between_sd"] = summary["jackknife_se_median"] / summary["between_unit_sd"]
    atomic_parquet(ROOT / str(config["outputs"]["unit_uncertainty"]), uncertainty)
    atomic_csv(ROOT / str(config["outputs"]["summary"]), summary)
    atomic_csv(ROOT / str(config["outputs"]["fold_comparison"]), fold)
    make_figure(uncertainty, summary, prediction, ROOT / str(config["outputs"]["figure"]))
    report_lines = [
        "# Phase 40 Cairngorms reference uncertainty",
        "",
        "The delete-one 10 m spatial-block jackknife quantifies finite-support and within-window spatial-composition instability. It does not include airborne-LiDAR acquisition or CHM-production error.",
        "",
        "```text",
        summary.to_string(index=False, float_format=lambda value: f"{value:.5f}"),
        "```",
        "",
        "The optimistic error-adjusted R-squared is a sensitivity bound under independent additive reference error, not a replacement for the observed held-out score. Diagnostic reduced chi-squared values exclude model predictive variance and should not be treated as formal goodness-of-fit tests.",
    ]
    report_path = ROOT / str(config["outputs"]["report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    artifacts = [
        ROOT / str(config["outputs"]["unit_uncertainty"]),
        ROOT / str(config["outputs"]["summary"]),
        ROOT / str(config["outputs"]["fold_comparison"]),
        ROOT / str(config["outputs"]["figure"]),
        report_path,
    ]
    atomic_json(
        result_path,
        {
            "status": "complete",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "units": int(uncertainty["row_id"].nunique()),
            "metrics": int(uncertainty["metric"].nunique()),
            "jackknife_blocks_per_unit": int(uncertainty["jackknife_blocks"].iloc[0]),
            "maximum_target_recalculation_difference": maximum_difference,
            "scope": "finite-support and spatial-composition uncertainty only",
            "artifacts": {
                str(path.relative_to(ROOT)): sha256(path) for path in artifacts
            },
        },
    )
    print(summary.to_string(index=False), flush=True)
    print("phase40 reference-uncertainty analysis complete", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["worker", "aggregate"])
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.mode == "worker":
        if args.index is None:
            raise SystemExit("worker mode requires --index")
        worker(args.index)
    else:
        aggregate()


if __name__ == "__main__":
    main()
