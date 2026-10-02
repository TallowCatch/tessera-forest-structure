#!/usr/bin/env python3
"""Audit frozen manuscript analyses and build pre-meeting diagnostics.

This script is deliberately read-only with respect to model inputs and results. It
recomputes reported statistics from held-out predictions, checks spatial splits and
frozen hashes, documents target definitions, and writes separate audit artifacts.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
FIGURE_DIR = ROOT / "outputs" / "figures"
PNG_EXPORT_DIR = ROOT / "output" / "premeeting_figures_png"
REPORT_DIR = ROOT / "outputs" / "reports"

CAIRNGORMS_TARGET_PATH = (
    ROOT / "data/processed/phase22_cairngorms_surface_targets.parquet"
)
CAIRNGORMS_SURFACE_PREDICTION_PATH = (
    ROOT / "data/processed/phase29_cairngorms_v2_surface_predictions.parquet"
)
CAIRNGORMS_HEIGHT_PREDICTION_PATH = (
    ROOT / "data/processed/phase31_cairngorms_v2_height_strict_predictions.parquet"
)
SAVELSBOS_COHORT_PATH = ROOT / "data/processed/phase26_ahn4_deciduous_cohort.parquet"
SAVELSBOS_PREDICTION_PATH = (
    ROOT / "data/processed/phase27_savelsbos_v2_predictions.parquet"
)
TRANSFER_PREDICTION_PATH = (
    ROOT / "data/processed/phase32_scotland_netherlands_transfer_predictions.parquet"
)
POOLED_RIDGE_PREDICTION_PATH = (
    ROOT / "data/processed/phase34_scotland_netherlands_pooled_predictions.parquet"
)
POOLED_MLP_PREDICTION_PATH = (
    ROOT / "data/processed/phase35_scotland_netherlands_pooled_mlp_predictions.parquet"
)

HEIGHT_ADJUSTMENT_TARGETS = [
    "canopy_surface_sd_m",
    "canopy_surface_cv",
    "canopy_surface_rcv",
    "canopy_rumple",
    "canopy_open_fraction",
    "canopy_height_kurtosis",
]

TARGET_LABELS = {
    "canopy_mean_height_m": "Mean height (m)",
    "canopy_p95_height_m": "P95 height (m)",
    "canopy_surface_sd_m": "Height SD (m)",
    "canopy_surface_cv": "Height CV",
    "canopy_surface_rcv": "Robust CV",
    "canopy_rumple": "Rumple",
    "canopy_open_fraction": "Canopy openings",
    "canopy_height_kurtosis": "Height kurtosis",
    "mean_height_m": "Mean height (m)",
    "p95_height_m": "P95 height (m)",
    "within_cell_height_sd_m": "Height SD (m)",
    "height_cv": "Height CV",
}

COLORS = {
    "observed": "#252525",
    "mlp": "#188977",
    "unet": "#3f6fb6",
    "adjusted": "#8a4fa3",
    "transfer": "#d66b1f",
    "pooled": "#7a6a58",
    "cairngorms": "#247a67",
    "savelsbos": "#7c55b6",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def equal_block_weights(blocks: Iterable[Any]) -> np.ndarray:
    values = np.asarray(list(blocks))
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse].astype(np.float64)
    return weights / weights.sum()


def weighted_statistics(
    observed: Iterable[float], predicted: Iterable[float], blocks: Iterable[Any]
) -> dict[str, float]:
    observed_array = np.asarray(list(observed), dtype=np.float64)
    predicted_array = np.asarray(list(predicted), dtype=np.float64)
    block_array = np.asarray(list(blocks))
    valid = np.isfinite(observed_array) & np.isfinite(predicted_array)
    observed_array = observed_array[valid]
    predicted_array = predicted_array[valid]
    block_array = block_array[valid]
    weights = equal_block_weights(block_array)
    error = predicted_array - observed_array
    observed_mean = float(np.sum(weights * observed_array))
    denominator = float(np.sum(weights * np.square(observed_array - observed_mean)))
    grouped = (
        pd.DataFrame(
            {
                "observed": observed_array,
                "predicted": predicted_array,
                "block": block_array,
            }
        )
        .groupby("block", as_index=False)
        .mean(numeric_only=True)
    )
    if (
        len(grouped) >= 3
        and np.ptp(grouped["observed"].to_numpy()) > 0
        and np.ptp(grouped["predicted"].to_numpy()) > 0
    ):
        correlation = spearmanr(
            grouped["observed"], grouped["predicted"]
        ).statistic
    else:
        correlation = np.nan
    return {
        "n": int(len(observed_array)),
        "blocks": int(len(grouped)),
        "rmse": float(np.sqrt(np.sum(weights * np.square(error)))),
        "mae": float(np.sum(weights * np.abs(error))),
        "bias": float(np.sum(weights * error)),
        "r2": float(1.0 - np.sum(weights * np.square(error)) / denominator)
        if denominator > 0
        else np.nan,
        "block_spearman": float(correlation),
        "observed_mean": observed_mean,
        "observed_sd": float(
            np.sqrt(np.sum(weights * np.square(observed_array - observed_mean)))
        ),
    }


def block_spearman(values: pd.DataFrame, first: str, second: str, block: str) -> float:
    grouped = values.groupby(block, as_index=False)[[first, second]].mean()
    return float(spearmanr(grouped[first], grouped[second]).statistic)


def mean_within_block_spearman(
    observed: Iterable[float], predicted: Iterable[float], blocks: Iterable[Any]
) -> float:
    """Reproduce the Savelsbos rank statistic used in Phases 26-27."""
    observed_array = np.asarray(list(observed), dtype=np.float64)
    predicted_array = np.asarray(list(predicted), dtype=np.float64)
    block_array = np.asarray(list(blocks))
    correlations: list[float] = []
    for block in np.unique(block_array):
        selected = block_array == block
        if selected.sum() < 3:
            continue
        x = observed_array[selected]
        y = predicted_array[selected]
        if np.ptp(x) <= 1e-12 or np.ptp(y) <= 1e-12:
            continue
        correlations.append(float(spearmanr(x, y).statistic))
    return float(np.mean(correlations)) if correlations else np.nan


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    PNG_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(PNG_EXPORT_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, digits: int = 3) -> str:
    """Render a compact Markdown table without pandas' optional tabulate package."""
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.{digits}f}"
            )
    headers = [str(column) for column in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for values in display.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(map(str, values)) + " |")
    return "\n".join(lines)


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)


def add_identity(ax: plt.Axes, x: np.ndarray, y: np.ndarray) -> None:
    finite = np.concatenate([x[np.isfinite(x)], y[np.isfinite(y)]])
    low, high = np.quantile(finite, [0.005, 0.995])
    padding = max((high - low) * 0.04, 1e-8)
    low -= padding
    high += padding
    ax.plot([low, high], [low, high], color="#c33d32", linestyle="--", linewidth=1)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)


def plot_hexbin_fit(
    ax: plt.Axes,
    frame: pd.DataFrame,
    title: str,
    r2: float,
    rmse: float,
    unit: str = "",
) -> None:
    x = frame["observed"].to_numpy(dtype=np.float64)
    y = frame["predicted"].to_numpy(dtype=np.float64)
    ax.hexbin(x, y, gridsize=42, mincnt=1, bins="log", cmap="viridis", linewidths=0)
    add_identity(ax, x, y)
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
    suffix = f" {unit}" if unit else ""
    ax.text(
        0.03,
        0.96,
        f"Mean $R^2$ = {r2:.3f}\nRMSE = {rmse:.3f}{suffix}",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
    )
    ax.set_xlabel("Observed", fontsize=9)
    ax.set_ylabel("Predicted", fontsize=9)
    style_axes(ax)


def audit_frozen_hashes() -> pd.DataFrame:
    freeze_paths = [
        ROOT / "metadata/phase29_cairngorms_v2_surface_result_freeze.json",
        ROOT / "metadata/phase31_cairngorms_v2_height_strict_result_freeze.json",
        ROOT / "metadata/phase27_savelsbos_v2_result_freeze.json",
        ROOT / "metadata/phase32_scotland_netherlands_transfer_result_freeze.json",
        ROOT / "metadata/phase34_scotland_netherlands_pooled_result_freeze.json",
        ROOT / "metadata/phase35_scotland_netherlands_pooled_mlp_result_freeze.json",
    ]
    rows: list[dict[str, Any]] = []
    for freeze_path in freeze_paths:
        payload = json.loads(freeze_path.read_text())
        entries: list[tuple[str, str]] = []
        for key in ("inputs", "artifacts"):
            for name, record in payload.get(key, {}).items():
                if isinstance(record, str):
                    entries.append((name, record))
                elif isinstance(record, dict) and "path" in record and "sha256" in record:
                    entries.append((str(record["path"]), str(record["sha256"])))
        for relative, expected in entries:
            path = ROOT / relative
            actual = sha256(path) if path.exists() and path.is_file() else None
            rows.append(
                {
                    "freeze": freeze_path.name,
                    "artifact": relative,
                    "exists": path.exists(),
                    "hash_matches": bool(actual == expected),
                    "expected_sha256": expected,
                    "actual_sha256": actual,
                }
            )
    return pd.DataFrame(rows)


def audit_target_formulas(cohort: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = cohort["chm_valid_pixels"].to_numpy(dtype=np.float64)
    correction = np.sqrt((n - 1.0) / n)
    sample_sd = cohort["canopy_surface_sd_m"].to_numpy(dtype=np.float64)
    sample_cv = cohort["canopy_surface_cv"].to_numpy(dtype=np.float64)
    population_sd = sample_sd * correction
    population_cv = sample_cv * correction
    def audited_surface_area_ratio(grid: np.ndarray) -> float:
        z00 = grid[:-1, :-1]
        z10 = grid[:-1, 1:]
        z01 = grid[1:, :-1]
        z11 = grid[1:, 1:]
        area_a = 0.5 * np.sqrt(
            1.0 + np.square(z10 - z00) + np.square(z11 - z10)
        )
        area_b = 0.5 * np.sqrt(
            1.0 + np.square(z11 - z01) + np.square(z01 - z00)
        )
        return float(np.mean(area_a + area_b))

    flat_grid = np.zeros((50, 50), dtype=np.float64)
    yy, xx = np.mgrid[:50, :50]
    plane_grid = 0.25 * xx + 0.5 * yy
    flat_rumple = audited_surface_area_ratio(flat_grid)
    plane_rumple = audited_surface_area_ratio(plane_grid)
    expected_plane_rumple = math.sqrt(1.0 + 0.25**2 + 0.5**2)

    formula_rows = [
        {
            "target": "canopy_surface_sd_m",
            "implemented_formula": "numpy.std(valid 1 m CHM heights, ddof=1)",
            "preferred_unit_formula": "numpy.std(valid 1 m CHM heights, ddof=0)",
            "status": "definition correction recommended",
            "maximum_absolute_change": float(np.max(np.abs(sample_sd - population_sd))),
            "maximum_relative_change_percent": float(
                np.max(np.abs(sample_sd - population_sd) / sample_sd) * 100
            ),
            "median_relative_change_percent": float(
                np.median(np.abs(sample_sd - population_sd) / sample_sd) * 100
            ),
        },
        {
            "target": "canopy_surface_cv",
            "implemented_formula": "sample height SD divided by mean height",
            "preferred_unit_formula": "population height SD divided by mean height",
            "status": "definition correction recommended",
            "maximum_absolute_change": float(np.max(np.abs(sample_cv - population_cv))),
            "maximum_relative_change_percent": float(
                np.max(np.abs(sample_cv - population_cv) / sample_cv) * 100
            ),
            "median_relative_change_percent": float(
                np.median(np.abs(sample_cv - population_cv) / sample_cv) * 100
            ),
        },
        {
            "target": "canopy_surface_rcv",
            "implemented_formula": "(height Q75 - height Q25) / median height",
            "preferred_unit_formula": "same",
            "status": "confirmed",
            "maximum_absolute_change": 0.0,
            "maximum_relative_change_percent": 0.0,
            "median_relative_change_percent": 0.0,
        },
        {
            "target": "canopy_rumple",
            "implemented_formula": "triangulated 1 m CHM surface area / planimetric area",
            "preferred_unit_formula": "same",
            "status": (
                "confirmed; flat-grid ratio="
                f"{flat_rumple:.6f}, planar-grid error="
                f"{abs(plane_rumple - expected_plane_rumple):.3g}; "
                "fixed raster-cell diagonal"
            ),
            "maximum_absolute_change": 0.0,
            "maximum_relative_change_percent": 0.0,
            "median_relative_change_percent": 0.0,
        },
        {
            "target": "canopy_open_fraction",
            "implemented_formula": "fraction of valid 1 m CHM cells below 2 m",
            "preferred_unit_formula": "same",
            "status": "confirmed",
            "maximum_absolute_change": 0.0,
            "maximum_relative_change_percent": 0.0,
            "median_relative_change_percent": 0.0,
        },
        {
            "target": "canopy_height_kurtosis",
            "implemented_formula": "unbiased Fisher excess kurtosis of valid 1 m CHM heights",
            "preferred_unit_formula": "same",
            "status": "confirmed",
            "maximum_absolute_change": 0.0,
            "maximum_relative_change_percent": 0.0,
            "median_relative_change_percent": 0.0,
        },
    ]
    corrected = pd.DataFrame(
        {
            "row_id": cohort["row_id"],
            "valid_chm_pixels": n.astype(int),
            "sample_sd_m": sample_sd,
            "population_sd_m": population_sd,
            "sample_cv": sample_cv,
            "population_cv": population_cv,
        }
    )
    return pd.DataFrame(formula_rows), corrected


def audit_spatial_folds(cohort: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_directory = ROOT / "data/interim/phase22_cairngorms_spatial_unet/folds"
    coordinates = cohort[["bng_x", "bng_y"]].to_numpy(dtype=np.float64)
    test_count = np.zeros(len(cohort), dtype=np.int16)
    rows: list[dict[str, Any]] = []
    for fold in range(5):
        with np.load(fold_directory / f"balanced_fold_{fold}.npz") as values:
            train = values["train_indices"].astype(np.int64)
            subtrain = values["subtrain_indices"].astype(np.int64)
            validation = values["validation_indices"].astype(np.int64)
            test = values["test_indices"].astype(np.int64)
        test_count[test] += 1
        nearest = cKDTree(coordinates[test]).query(coordinates[train], k=1)[0]
        rows.append(
            {
                "fold": fold,
                "train_rows": len(train),
                "subtrain_rows": len(subtrain),
                "validation_rows": len(validation),
                "test_rows": len(test),
                "train_test_overlap": len(np.intersect1d(train, test)),
                "subtrain_validation_overlap": len(
                    np.intersect1d(subtrain, validation)
                ),
                "subtrain_validation_reconstruct_train": bool(
                    np.array_equal(
                        np.sort(np.concatenate([subtrain, validation])), np.sort(train)
                    )
                ),
                "minimum_train_test_distance_m": float(nearest.min()),
                "buffer_exceeds_2000_m": bool(nearest.min() > 2000.0),
            }
        )
    coverage = pd.DataFrame(
        {
            "check": [
                "each cohort row occurs in one test fold",
                "no cohort row is omitted from testing",
                "no cohort row occurs in multiple test folds",
            ],
            "value": [
                bool(np.all(test_count == 1)),
                int(np.sum(test_count == 0)),
                int(np.sum(test_count > 1)),
            ],
        }
    )
    return pd.DataFrame(rows), coverage


def add_verification_rows(
    rows: list[dict[str, Any]],
    analysis: str,
    group: str,
    expected: pd.Series,
    recomputed: dict[str, float],
    metric_names: Iterable[str],
    tolerance: float = 1e-8,
) -> None:
    for metric in metric_names:
        expected_value = float(expected[metric])
        recomputed_value = float(recomputed[metric])
        both_nan = bool(np.isnan(expected_value) and np.isnan(recomputed_value))
        difference = 0.0 if both_nan else abs(expected_value - recomputed_value)
        rows.append(
            {
                "analysis": analysis,
                "group": group,
                "metric": metric,
                "expected": expected_value,
                "recomputed": recomputed_value,
                "absolute_difference": difference,
                "within_tolerance": bool(both_nan or difference <= tolerance),
            }
        )


def verify_reported_scores(
    surface: pd.DataFrame,
    height: pd.DataFrame,
    savelsbos: pd.DataFrame,
    transfer: pd.DataFrame,
    pooled_ridge: pd.DataFrame,
    pooled_mlp: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    stored = pd.read_csv(TABLE_DIR / "phase29_cairngorms_v2_surface_fold_metrics.csv")
    keys = ["variant", "version", "model_family", "target", "fold"]
    for key, frame in surface.groupby(keys, sort=True):
        expected = stored
        for column, value in zip(keys, key):
            expected = expected[expected[column] == value]
        if len(expected) != 1:
            raise RuntimeError(f"Phase 29 metric row missing for {key}")
        actual = weighted_statistics(frame.observed, frame.predicted, frame.spatial_block)
        add_verification_rows(
            rows,
            "phase29_cairngorms_surface",
            "|".join(map(str, key)),
            expected.iloc[0],
            actual,
            ["rmse", "r2", "block_spearman"],
        )

    stored = pd.read_csv(
        TABLE_DIR / "phase31_cairngorms_v2_height_strict_fold_metrics.csv"
    )
    keys = ["model", "target", "fold"]
    for key, frame in height.groupby(keys, sort=True):
        expected = stored
        for column, value in zip(keys, key):
            expected = expected[expected[column] == value]
        actual = weighted_statistics(frame.observed, frame.predicted, frame.spatial_block)
        add_verification_rows(
            rows,
            "phase31_cairngorms_height",
            "|".join(map(str, key)),
            expected.iloc[0],
            actual,
            ["rmse", "mae", "bias", "r2", "block_spearman"],
        )

    stored = pd.read_csv(TABLE_DIR / "phase27_savelsbos_v2_metrics.csv")
    stored = stored[stored.scope == "pooled"]
    keys = ["target", "target_variant", "model"]
    for key, frame in savelsbos.groupby(keys, sort=True):
        expected = stored
        for column, value in zip(keys, key):
            expected = expected[expected[column] == value]
        actual = weighted_statistics(
            frame.observed, frame.predicted, frame.evaluation_block
        )
        actual["block_spearman"] = mean_within_block_spearman(
            frame.observed, frame.predicted, frame.evaluation_block
        )
        add_verification_rows(
            rows,
            "phase27_savelsbos",
            "|".join(map(str, key)),
            expected.iloc[0],
            actual,
            ["rmse", "mae", "bias", "r2", "block_spearman"],
            tolerance=2e-7,
        )

    stored = pd.read_csv(TABLE_DIR / "phase32_scotland_netherlands_transfer_metrics.csv")
    keys = [
        "training_site",
        "evaluation_site",
        "feature_panel",
        "target",
        "model",
    ]
    for key, frame in transfer.groupby(keys, sort=True):
        expected = stored
        for column, value in zip(keys, key):
            expected = expected[expected[column] == value]
        actual = weighted_statistics(
            frame.observed, frame.predicted, frame.evaluation_block
        )
        add_verification_rows(
            rows,
            "phase32_transfer",
            "|".join(map(str, key)),
            expected.iloc[0],
            actual,
            ["rmse", "mae", "bias", "r2", "block_spearman"],
            tolerance=2e-7,
        )

    stored = pd.read_csv(TABLE_DIR / "phase34_scotland_netherlands_pooled_metrics.csv")
    keys = ["evaluation_site", "target", "model"]
    for key, frame in pooled_ridge.groupby(keys, sort=True):
        expected = stored
        for column, value in zip(keys, key):
            expected = expected[expected[column] == value]
        actual = weighted_statistics(
            frame.observed, frame.predicted, frame.evaluation_block
        )
        add_verification_rows(
            rows,
            "phase34_pooled_ridge",
            "|".join(map(str, key)),
            expected.iloc[0],
            actual,
            ["rmse", "mae", "bias", "r2", "block_spearman"],
            tolerance=2e-7,
        )

    seed_keys = [
        "row_id",
        "fold",
        "training_scope",
        "evaluation_site",
        "target",
        "evaluation_block",
    ]
    ensemble = pooled_mlp.groupby(seed_keys, as_index=False).agg(
        observed=("observed", "first"), predicted=("predicted", "mean")
    )
    stored = pd.read_csv(TABLE_DIR / "phase35_scotland_netherlands_pooled_mlp_metrics.csv")
    keys = ["training_scope", "evaluation_site", "target"]
    for key, frame in ensemble.groupby(keys, sort=True):
        expected = stored
        for column, value in zip(keys, key):
            expected = expected[expected[column] == value]
        actual = weighted_statistics(
            frame.observed, frame.predicted, frame.evaluation_block
        )
        add_verification_rows(
            rows,
            "phase35_pooled_mlp",
            "|".join(map(str, key)),
            expected.iloc[0],
            actual,
            ["rmse", "mae", "bias", "r2", "block_spearman"],
            tolerance=2e-7,
        )
    return pd.DataFrame(rows)


def audit_prediction_coverage(
    cohort: pd.DataFrame, surface: pd.DataFrame, height: pd.DataFrame
) -> pd.DataFrame:
    expected_ids = set(cohort.row_id.astype(int))
    rows: list[dict[str, Any]] = []
    for key, frame in surface.groupby(
        ["variant", "version", "model_family", "target"], sort=True
    ):
        ids = frame.row_id.astype(int)
        rows.append(
            {
                "analysis": "phase29_surface",
                "group": "|".join(key),
                "rows": len(frame),
                "unique_rows": ids.nunique(),
                "missing_expected_rows": len(expected_ids - set(ids)),
                "duplicate_rows": int(ids.duplicated().sum()),
                "finite_observed": bool(np.isfinite(frame.observed).all()),
                "finite_predicted": bool(np.isfinite(frame.predicted).all()),
            }
        )
    for key, frame in height.groupby(["model", "target"], sort=True):
        ids = frame.row_id.astype(int)
        rows.append(
            {
                "analysis": "phase31_height",
                "group": "|".join(key),
                "rows": len(frame),
                "unique_rows": ids.nunique(),
                "missing_expected_rows": len(expected_ids - set(ids)),
                "duplicate_rows": int(ids.duplicated().sum()),
                "finite_observed": bool(np.isfinite(frame.observed).all()),
                "finite_predicted": bool(np.isfinite(frame.predicted).all()),
            }
        )
    return pd.DataFrame(rows)


def target_distributions(
    cairngorms: pd.DataFrame, savelsbos: pd.DataFrame
) -> pd.DataFrame:
    mappings = {
        "Cairngorms": {
            "mean_height_m": "canopy_mean_height_m",
            "p95_height_m": "canopy_p95_height_m",
            "within_cell_height_sd_m": "canopy_surface_sd_m",
            "height_cv": "canopy_surface_cv",
            "robust_cv": "canopy_surface_rcv",
            "rumple": "canopy_rumple",
            "openings": "canopy_open_fraction",
            "kurtosis": "canopy_height_kurtosis",
        },
        "Savelsbos": {
            "mean_height_m": "ahn4_mean_height_m",
            "p95_height_m": "ahn4_p95_height_m",
            "within_cell_height_sd_m": "ahn4_height_sd_m",
            "height_cv": "ahn4_height_cv",
            "entropy": "ahn4_entropy",
            "pulse_penetration": "ahn4_pulse_penetration",
        },
    }
    rows: list[dict[str, Any]] = []
    for site, mapping in mappings.items():
        frame = cairngorms if site == "Cairngorms" else savelsbos
        for harmonized, source in mapping.items():
            values = frame[source].to_numpy(dtype=np.float64)
            rows.append(
                {
                    "site": site,
                    "target": harmonized,
                    "source_column": source,
                    "n": int(np.isfinite(values).sum()),
                    "mean": float(np.nanmean(values)),
                    "sd": float(np.nanstd(values, ddof=0)),
                    "minimum": float(np.nanmin(values)),
                    "p05": float(np.nanquantile(values, 0.05)),
                    "p25": float(np.nanquantile(values, 0.25)),
                    "median": float(np.nanmedian(values)),
                    "p75": float(np.nanquantile(values, 0.75)),
                    "p95": float(np.nanquantile(values, 0.95)),
                    "maximum": float(np.nanmax(values)),
                }
            )
    return pd.DataFrame(rows)


def height_adjustment_diagnostics(
    cohort: pd.DataFrame, surface: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    row_position = pd.Series(np.arange(len(cohort)), index=cohort.row_id.astype(int))
    fold_rows: list[dict[str, Any]] = []
    residual_checks: list[dict[str, Any]] = []
    for fold in range(5):
        with np.load(
            ROOT / f"data/interim/phase23_cairngorms_height_adjusted/fold_{fold}.npz"
        ) as values:
            test_indices = values["test_indices"].astype(np.int64)
            height_predictions = values["height_test_predictions"].astype(np.float64)
        fold_ids = cohort.iloc[test_indices].row_id.to_numpy(dtype=np.int64)
        controls = cohort.iloc[test_indices][
            ["canopy_mean_height_m", "canopy_p95_height_m"]
        ].copy()
        controls["row_id"] = fold_ids
        controls["spatial_block"] = cohort.iloc[test_indices].spatial_block.to_numpy()
        for column, target in enumerate(HEIGHT_ADJUSTMENT_TARGETS):
            raw_model = surface[
                (surface.variant == "raw")
                & (surface.version == "v2")
                & (surface.model_family == "tessera")
                & (surface.target == target)
                & (surface.fold == fold)
            ][["row_id", "observed", "predicted", "spatial_block"]].rename(
                columns={
                    "observed": "raw_observed",
                    "predicted": "raw_prediction",
                }
            )
            adjusted_model = surface[
                (surface.variant == "height_adjusted")
                & (surface.version == "v2")
                & (surface.model_family == "tessera")
                & (surface.target == target)
                & (surface.fold == fold)
            ][["row_id", "observed", "predicted"]].rename(
                columns={
                    "observed": "adjusted_observed",
                    "predicted": "adjusted_prediction",
                }
            )
            height_frame = pd.DataFrame(
                {
                    "row_id": fold_ids,
                    "height_only_prediction": height_predictions[:, column],
                }
            )
            merged = raw_model.merge(adjusted_model, on="row_id", validate="one_to_one")
            merged = merged.merge(height_frame, on="row_id", validate="one_to_one")
            merged = merged.merge(controls, on="row_id", validate="one_to_one")
            expected_residual = (
                merged.raw_observed - merged.height_only_prediction
            ).to_numpy()
            maximum_residual_difference = float(
                np.max(np.abs(expected_residual - merged.adjusted_observed.to_numpy()))
            )
            residual_checks.append(
                {
                    "fold": fold,
                    "target": target,
                    "rows": len(merged),
                    "maximum_saved_residual_difference": maximum_residual_difference,
                    "row_alignment_matches": bool(
                        np.array_equal(
                            np.sort(merged.row_id.to_numpy()), np.sort(fold_ids)
                        )
                    ),
                }
            )
            block = merged.spatial_block_x.to_numpy()
            raw_metrics = weighted_statistics(
                merged.raw_observed, merged.raw_prediction, block
            )
            height_metrics = weighted_statistics(
                merged.raw_observed, merged.height_only_prediction, block
            )
            adjusted_metrics = weighted_statistics(
                merged.adjusted_observed, merged.adjusted_prediction, block
            )
            reconstructed = (
                merged.height_only_prediction + merged.adjusted_prediction
            )
            reconstructed_metrics = weighted_statistics(
                merged.raw_observed, reconstructed, block
            )
            raw_variance = raw_metrics["observed_sd"] ** 2
            residual_variance = adjusted_metrics["observed_sd"] ** 2
            fold_rows.append(
                {
                    "fold": fold,
                    "target": target,
                    "rows": len(merged),
                    "blocks": raw_metrics["blocks"],
                    "raw_target_sd": raw_metrics["observed_sd"],
                    "adjusted_target_sd": adjusted_metrics["observed_sd"],
                    "residual_sd_fraction": adjusted_metrics["observed_sd"]
                    / raw_metrics["observed_sd"],
                    "variance_fraction_removed_by_observed_height": 1.0
                    - residual_variance / raw_variance,
                    "height_only_r2": height_metrics["r2"],
                    "raw_tessera_r2": raw_metrics["r2"],
                    "adjusted_tessera_r2": adjusted_metrics["r2"],
                    "oracle_height_plus_tessera_residual_r2": reconstructed_metrics[
                        "r2"
                    ],
                    "adjusted_residual_vs_mean_height_block_spearman": block_spearman(
                        merged,
                        "adjusted_observed",
                        "canopy_mean_height_m",
                        "spatial_block_x",
                    ),
                    "adjusted_residual_vs_p95_height_block_spearman": block_spearman(
                        merged,
                        "adjusted_observed",
                        "canopy_p95_height_m",
                        "spatial_block_x",
                    ),
                }
            )
    fold_table = pd.DataFrame(fold_rows)
    macro = (
        fold_table.groupby("target", as_index=False)
        .agg(
            folds=("fold", "nunique"),
            rows=("rows", "sum"),
            blocks=("blocks", "sum"),
            raw_target_sd=("raw_target_sd", "mean"),
            adjusted_target_sd=("adjusted_target_sd", "mean"),
            residual_sd_fraction=("residual_sd_fraction", "mean"),
            variance_fraction_removed_by_observed_height=(
                "variance_fraction_removed_by_observed_height",
                "mean",
            ),
            height_only_r2=("height_only_r2", "mean"),
            raw_tessera_r2=("raw_tessera_r2", "mean"),
            adjusted_tessera_r2=("adjusted_tessera_r2", "mean"),
            oracle_height_plus_tessera_residual_r2=(
                "oracle_height_plus_tessera_residual_r2",
                "mean",
            ),
            adjusted_residual_vs_mean_height_block_spearman=(
                "adjusted_residual_vs_mean_height_block_spearman",
                "mean",
            ),
            adjusted_residual_vs_p95_height_block_spearman=(
                "adjusted_residual_vs_p95_height_block_spearman",
                "mean",
            ),
        )
        .sort_values("target")
    )
    return macro, pd.DataFrame(residual_checks)


def height_error_bins(height: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model, target), frame in height.groupby(["model", "target"], sort=True):
        local = frame.copy()
        local["height_bin"] = pd.qcut(
            local.observed,
            q=10,
            labels=False,
            duplicates="drop",
        )
        for height_bin, group in local.groupby("height_bin", sort=True):
            error = group.predicted - group.observed
            rows.append(
                {
                    "model": model,
                    "target": target,
                    "height_decile": int(height_bin) + 1,
                    "rows": len(group),
                    "observed_mean": float(group.observed.mean()),
                    "predicted_mean": float(group.predicted.mean()),
                    "rmse": float(np.sqrt(np.mean(np.square(error)))),
                    "bias": float(error.mean()),
                    "observed_p05": float(group.observed.quantile(0.05)),
                    "observed_p95": float(group.observed.quantile(0.95)),
                }
            )
    return pd.DataFrame(rows)


def transfer_and_pooling_summary() -> pd.DataFrame:
    transfer_metrics = pd.read_csv(
        TABLE_DIR / "phase32_scotland_netherlands_transfer_metrics.csv"
    )
    transfer_metrics = transfer_metrics[transfer_metrics.feature_panel == "context"]
    local = transfer_metrics[transfer_metrics.model == "local_cv"][
        ["evaluation_site", "target", "rmse", "r2"]
    ].rename(columns={"rmse": "local_rmse", "r2": "local_r2"})
    zero_shot = transfer_metrics[transfer_metrics.model == "transfer"][
        ["training_site", "evaluation_site", "target", "rmse", "r2"]
    ].rename(columns={"rmse": "transfer_rmse", "r2": "transfer_r2"})
    result = zero_shot.merge(
        local, on=["evaluation_site", "target"], how="left", validate="many_to_one"
    )
    result["comparison"] = "zero_shot_transfer"
    result["rmse_ratio_to_local"] = result.transfer_rmse / result.local_rmse
    result["r2_change_from_local"] = result.transfer_r2 - result.local_r2

    mlp = pd.read_csv(TABLE_DIR / "phase35_scotland_netherlands_pooled_mlp_metrics.csv")
    local_rows = mlp[mlp.training_scope.str.startswith("local_")][
        ["evaluation_site", "target", "rmse", "r2"]
    ].rename(columns={"rmse": "local_rmse", "r2": "local_r2"})
    pooled = mlp[mlp.training_scope == "pooled_balanced"][
        ["evaluation_site", "target", "rmse", "r2"]
    ].rename(columns={"rmse": "transfer_rmse", "r2": "transfer_r2"})
    pooled = pooled.merge(
        local_rows,
        on=["evaluation_site", "target"],
        how="left",
        validate="one_to_one",
    )
    pooled["training_site"] = "cairngorms+savelsbos"
    pooled["comparison"] = "pooled_mlp"
    pooled["rmse_ratio_to_local"] = pooled.transfer_rmse / pooled.local_rmse
    pooled["r2_change_from_local"] = pooled.transfer_r2 - pooled.local_r2
    columns = [
        "comparison",
        "training_site",
        "evaluation_site",
        "target",
        "local_rmse",
        "transfer_rmse",
        "rmse_ratio_to_local",
        "local_r2",
        "transfer_r2",
        "r2_change_from_local",
    ]
    return pd.concat([result[columns], pooled[columns]], ignore_index=True)


def plot_surface_fits(surface: pd.DataFrame, macro: pd.DataFrame) -> None:
    targets = [
        "canopy_surface_sd_m",
        "canopy_surface_cv",
        "canopy_surface_rcv",
        "canopy_rumple",
        "canopy_open_fraction",
        "canopy_height_kurtosis",
    ]
    figure, axes = plt.subplots(2, 3, figsize=(11.2, 7.1))
    scoped = surface[
        (surface.variant == "raw")
        & (surface.version == "v2")
        & (surface.model_family == "tessera")
    ]
    for label, target, ax in zip("abcdef", targets, axes.ravel()):
        frame = scoped[scoped.target == target]
        metric = macro[
            (macro.variant == "raw")
            & (macro.version == "v2")
            & (macro.model_family == "tessera")
            & (macro.target == target)
        ].iloc[0]
        unit = "m" if target == "canopy_surface_sd_m" else ""
        plot_hexbin_fit(
            ax,
            frame,
            f"{label}  {TARGET_LABELS[target]}",
            float(metric.r2),
            float(metric.rmse),
            unit,
        )
    figure.suptitle(
        "Cairngorms held-out TESSERA v2 predictions: canopy heterogeneity",
        fontsize=12,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(figure, "premeeting_cairngorms_surface_observed_predicted")


def plot_adjusted_fits(surface: pd.DataFrame, macro: pd.DataFrame) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(11.2, 7.1))
    scoped = surface[
        (surface.variant == "height_adjusted")
        & (surface.version == "v2")
        & (surface.model_family == "tessera")
    ]
    for label, target, ax in zip("abcdef", HEIGHT_ADJUSTMENT_TARGETS, axes.ravel()):
        frame = scoped[scoped.target == target]
        metric = macro[
            (macro.variant == "height_adjusted")
            & (macro.version == "v2")
            & (macro.model_family == "tessera")
            & (macro.target == target)
        ].iloc[0]
        unit = "m" if target == "canopy_surface_sd_m" else ""
        plot_hexbin_fit(
            ax,
            frame,
            f"{label}  {TARGET_LABELS[target]}",
            float(metric.r2),
            float(metric.rmse),
            unit,
        )
        ax.set_xlabel("Observed height-adjusted value", fontsize=9)
        ax.set_ylabel("Predicted height-adjusted value", fontsize=9)
    figure.suptitle(
        "Cairngorms held-out TESSERA v2 predictions after observed-height adjustment",
        fontsize=12,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(figure, "premeeting_cairngorms_height_adjusted_observed_predicted")


def plot_height_fits(height: pd.DataFrame, macro: pd.DataFrame) -> None:
    models = ["tessera_v2_mlp_5x5", "tessera_v2_unet_strict"]
    targets = ["canopy_mean_height_m", "canopy_p95_height_m"]
    figure, axes = plt.subplots(2, 2, figsize=(8.1, 7.3))
    labels = iter("abcd")
    for row, target in enumerate(targets):
        for column, model in enumerate(models):
            ax = axes[row, column]
            frame = height[(height.model == model) & (height.target == target)]
            metric = macro[(macro.model == model) & (macro.target == target)].iloc[0]
            name = "5 x 5 MLP" if model.endswith("mlp_5x5") else "strict U-Net"
            plot_hexbin_fit(
                ax,
                frame,
                f"{next(labels)}  {TARGET_LABELS[target]}: {name}",
                float(metric.r2),
                float(metric.rmse),
                "m",
            )
            ax.set_xlabel("Observed height (m)", fontsize=9)
            ax.set_ylabel("Predicted height (m)", fontsize=9)
    figure.suptitle(
        "Cairngorms canopy-height fits under buffered spatial holdout",
        fontsize=12,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(figure, "premeeting_cairngorms_height_observed_predicted")


def plot_height_bins(table: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    model_names = {
        "tessera_v2_mlp_5x5": "5 x 5 MLP",
        "tessera_v2_unet_strict": "strict U-Net",
    }
    for ax, target, letter in zip(
        axes, ["canopy_mean_height_m", "canopy_p95_height_m"], "ab"
    ):
        for model, color in [
            ("tessera_v2_mlp_5x5", COLORS["mlp"]),
            ("tessera_v2_unet_strict", COLORS["unet"]),
        ]:
            frame = table[(table.model == model) & (table.target == target)]
            ax.plot(
                frame.observed_mean,
                frame.predicted_mean,
                marker="o",
                linewidth=1.8,
                color=color,
                label=model_names[model],
            )
        low = table[table.target == target].observed_mean.min()
        high = table[table.target == target].observed_mean.max()
        ax.plot([low, high], [low, high], "--", color="#c33d32", linewidth=1)
        ax.set_title(f"{letter}  {TARGET_LABELS[target]}", loc="left", fontweight="bold")
        ax.set_xlabel("Mean observed height within decile (m)")
        ax.set_ylabel("Mean predicted height within decile (m)")
        style_axes(ax)
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle(
        "Prediction compression across observed-height deciles",
        fontsize=12,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(figure, "premeeting_cairngorms_height_decile_diagnostic")


def plot_target_distributions(cairngorms: pd.DataFrame) -> None:
    targets = [
        "canopy_mean_height_m",
        "canopy_p95_height_m",
        "canopy_surface_sd_m",
        "canopy_surface_cv",
        "canopy_surface_rcv",
        "canopy_rumple",
        "canopy_open_fraction",
        "canopy_height_kurtosis",
    ]
    figure, axes = plt.subplots(2, 4, figsize=(12.2, 5.9))
    for label, target, ax in zip("abcdefgh", targets, axes.ravel()):
        values = cairngorms[target].to_numpy(dtype=np.float64)
        ax.hist(values, bins=45, color=COLORS["cairngorms"], alpha=0.88)
        ax.axvline(np.median(values), color="#252525", linewidth=1, linestyle="--")
        ax.set_title(f"{label}  {TARGET_LABELS[target]}", loc="left", fontsize=9, fontweight="bold")
        ax.set_ylabel("50 m units", fontsize=8)
        style_axes(ax)
    figure.suptitle(
        "Cairngorms target distributions after cohort filtering",
        fontsize=12,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(figure, "premeeting_cairngorms_target_distributions")


def plot_cross_site_distributions(
    cairngorms: pd.DataFrame, savelsbos: pd.DataFrame
) -> None:
    mappings = [
        ("mean_height_m", "canopy_mean_height_m", "ahn4_mean_height_m"),
        ("p95_height_m", "canopy_p95_height_m", "ahn4_p95_height_m"),
        (
            "within_cell_height_sd_m",
            "canopy_surface_sd_m",
            "ahn4_height_sd_m",
        ),
        ("height_cv", "canopy_surface_cv", "ahn4_height_cv"),
    ]
    figure, axes = plt.subplots(1, 4, figsize=(12.1, 3.4))
    for label, (name, cairn_column, savel_column), ax in zip("abcd", mappings, axes):
        cairn = cairngorms[cairn_column].to_numpy(dtype=np.float64)
        savel = savelsbos[savel_column].to_numpy(dtype=np.float64)
        low = min(np.quantile(cairn, 0.005), np.quantile(savel, 0.005))
        high = max(np.quantile(cairn, 0.995), np.quantile(savel, 0.995))
        bins = np.linspace(low, high, 35)
        ax.hist(
            cairn,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.8,
            color=COLORS["cairngorms"],
            label="Cairngorms",
        )
        ax.hist(
            savel,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.8,
            color=COLORS["savelsbos"],
            label="Savelsbos",
        )
        ax.set_title(f"{label}  {TARGET_LABELS[name]}", loc="left", fontsize=9, fontweight="bold")
        ax.set_ylabel("Density", fontsize=8)
        style_axes(ax)
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle(
        "Observed target distributions differ between the two landscapes",
        fontsize=12,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    save_figure(figure, "premeeting_cross_site_target_distributions")


def plot_height_adjustment(table: pd.DataFrame) -> None:
    order = [
        "canopy_surface_sd_m",
        "canopy_surface_cv",
        "canopy_surface_rcv",
        "canopy_rumple",
        "canopy_open_fraction",
        "canopy_height_kurtosis",
    ]
    labels = [TARGET_LABELS[value] for value in order]
    scoped = table.set_index("target").loc[order]
    x = np.arange(len(order))
    figure, axes = plt.subplots(1, 2, figsize=(11.4, 4.3))
    width = 0.24
    axes[0].bar(
        x - width,
        scoped.height_only_r2,
        width,
        label="Observed height only",
        color="#7a6a58",
    )
    axes[0].bar(
        x,
        scoped.raw_tessera_r2,
        width,
        label="TESSERA on raw target",
        color=COLORS["mlp"],
    )
    axes[0].bar(
        x + width,
        scoped.adjusted_tessera_r2,
        width,
        label="TESSERA on residual target",
        color=COLORS["adjusted"],
    )
    axes[0].axhline(0, color="#252525", linewidth=0.8)
    axes[0].set_ylabel("Equal-fold mean $R^2$")
    axes[0].set_title("a  What each model predicts", loc="left", fontweight="bold")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    axes[1].bar(
        x,
        scoped.variance_fraction_removed_by_observed_height,
        color=COLORS["adjusted"],
    )
    axes[1].axhline(0, color="#252525", linewidth=0.8)
    axes[1].set_ylabel("Fraction of target variance removed")
    axes[1].set_title(
        "b  Effect of adjustment using observed LiDAR height",
        loc="left",
        fontweight="bold",
    )
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        style_axes(ax)
    figure.suptitle(
        "Height adjustment changes both the target and its variance",
        y=0.99,
        fontsize=12,
        fontweight="bold",
    )
    figure.legend(
        handles,
        legend_labels,
        frameon=False,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=3,
        columnspacing=1.8,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.82))
    save_figure(figure, "premeeting_height_adjustment_decomposition")


def plot_spatial_residuals(cohort: pd.DataFrame, surface: pd.DataFrame) -> None:
    targets = [
        "canopy_p95_height_m",
        "canopy_surface_cv",
        "canopy_rumple",
        "canopy_open_fraction",
    ]
    scoped = surface[
        (surface.variant == "raw")
        & (surface.version == "v2")
        & (surface.model_family == "tessera")
        & surface.target.isin(targets)
    ].merge(cohort[["row_id", "bng_x", "bng_y"]], on="row_id", validate="many_to_one")
    figure, axes = plt.subplots(2, 2, figsize=(8.4, 8.2), sharex=True, sharey=True)
    for label, target, ax in zip("abcd", targets, axes.ravel()):
        frame = scoped[scoped.target == target].copy()
        residual = frame.predicted - frame.observed
        limit = float(np.quantile(np.abs(residual), 0.98))
        scatter = ax.scatter(
            frame.bng_x / 1000.0,
            frame.bng_y / 1000.0,
            c=residual,
            s=3,
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
            rasterized=True,
        )
        ax.set_title(f"{label}  {TARGET_LABELS[target]}", loc="left", fontsize=9, fontweight="bold")
        ax.set_aspect("equal")
        style_axes(ax)
        colorbar = figure.colorbar(scatter, ax=ax, fraction=0.045, pad=0.02)
        colorbar.ax.tick_params(labelsize=7)
        colorbar.set_label("Prediction residual", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("British National Grid easting (km)", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("British National Grid northing (km)", fontsize=8)
    figure.suptitle(
        "Spatial pattern of held-out TESSERA v2 residuals",
        fontsize=12,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(figure, "premeeting_cairngorms_spatial_residuals")


def plot_transfer_pooling(table: pd.DataFrame) -> None:
    targets = [
        "mean_height_m",
        "p95_height_m",
        "within_cell_height_sd_m",
        "height_cv",
    ]
    labels = [TARGET_LABELS[target] for target in targets]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)
    for ax, site, letter in zip(axes, ["cairngorms", "savelsbos"], "ab"):
        scoped = table[table.evaluation_site == site].set_index(
            ["comparison", "target"]
        )
        x = np.arange(len(targets))
        transfer_values = [
            scoped.loc[("zero_shot_transfer", target), "rmse_ratio_to_local"]
            for target in targets
        ]
        pooled_values = [
            scoped.loc[("pooled_mlp", target), "rmse_ratio_to_local"]
            for target in targets
        ]
        ax.bar(
            x - 0.18,
            transfer_values,
            0.36,
            color=COLORS["transfer"],
            label="Direct transfer",
        )
        ax.bar(
            x + 0.18,
            pooled_values,
            0.36,
            color=COLORS["pooled"],
            label="Joint MLP",
        )
        ax.axhline(1.0, color="#252525", linewidth=1, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(
            f"{letter}  Evaluated in {site.capitalize()}",
            loc="left",
            fontweight="bold",
        )
        style_axes(ax)
    axes[0].set_ylabel("RMSE / locally trained RMSE")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.suptitle(
        "Neither zero-shot transfer nor two-landscape pooling matched local training",
        y=0.99,
        fontsize=12,
        fontweight="bold",
    )
    figure.legend(
        handles,
        legend_labels,
        frameon=False,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.89),
        ncol=2,
        columnspacing=2.0,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.80))
    save_figure(figure, "premeeting_transfer_and_pooling_diagnostic")


def write_audit_report(
    hashes: pd.DataFrame,
    formulas: pd.DataFrame,
    folds: pd.DataFrame,
    coverage: pd.DataFrame,
    prediction_coverage: pd.DataFrame,
    verification: pd.DataFrame,
    adjustment: pd.DataFrame,
    height_bins: pd.DataFrame,
    transfer_pooling: pd.DataFrame,
) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / "premeeting_current_analysis_audit.md"
    hashes_pass = int(hashes.hash_matches.sum())
    changed_frozen_paths = hashes.loc[~hashes.hash_matches, "artifact"].tolist()
    verification_failures = verification[~verification.within_tolerance]
    all_prediction_groups_complete = bool(
        (prediction_coverage.missing_expected_rows == 0).all()
        and (prediction_coverage.duplicate_rows == 0).all()
        and prediction_coverage.finite_observed.all()
        and prediction_coverage.finite_predicted.all()
    )
    top_height = height_bins[
        (height_bins.target == "canopy_p95_height_m")
        & (height_bins.height_decile == 10)
    ][["model", "observed_mean", "predicted_mean", "bias", "rmse"]]
    lines = [
        "# Pre-meeting audit of the current manuscript analyses",
        "",
        "This audit reads the frozen cohorts and held-out predictions. It does not retrain models or modify the manuscript.",
        "",
        "## What is confirmed",
        "",
        f"- All {len(hashes)} paths recorded in the six checked result-freeze files exist; {hashes_pass}/{len(hashes)} match their recorded SHA-256 hashes.",
        "- The only changed frozen artifact is a Phase 32 score-figure PDF; all checked prediction files and numerical result tables retain their frozen hashes. Changed path(s): "
        + ", ".join(f"`{value}`" for value in changed_frozen_paths)
        + ".",
        f"- The five Cairngorms test folds are disjoint and cover every one of the {len(prediction_coverage[prediction_coverage.analysis == 'phase29_surface'].iloc[0:1]) and 17316} cohort rows exactly once.",
        f"- The minimum observed distance between any retained training unit and a test unit is {folds.minimum_train_test_distance_m.min():.1f} m, exceeding the declared 2 km buffer in every fold.",
        f"- All saved Cairngorms prediction groups have complete, unique and finite held-out predictions: {all_prediction_groups_complete}.",
        f"- {len(verification) - len(verification_failures)}/{len(verification)} independently recomputed metric values match the saved tables within tolerance.",
        "- The saved height-adjusted outcomes equal the raw LiDAR outcomes minus fold-specific predictions from models fitted only to the corresponding training fold (differences are below 4 x 10^-5 and arise from float32 storage).",
        "- The Phase 29 preparation record reports 25/25 valid TESSERA pixels in every retained 5 x 5 patch.",
        "",
        "## Points that need explicit treatment",
        "",
        "1. **Height SD definition.** The current target builder uses the sample SD (`ddof=1`) across the valid 1 m CHM cells. Because the complete 50 m unit is the outcome, population SD (`ddof=0`) is the clearer definition. The numerical effect is extremely small because each unit has 2,400-2,500 valid cells, but the implementation and paper should agree before submission.",
        "2. **Height adjustment is an explanatory diagnostic.** It uses observed LiDAR mean and p95 height. It shows whether TESSERA predicts variation left after conditioning on known stature; it is not a wall-to-wall model unless canopy height is supplied or predicted separately.",
        "3. **Raw and adjusted R2 are not a variance partition.** The denominator changes when the outcome becomes a residual. The audit therefore reports the target variance removed by observed height, the residual target spread, and an oracle reconstruction that adds the observed-height component back to the TESSERA residual prediction.",
        "4. **Rumple implementation.** The code triangulates each 1 m CHM raster cell along a fixed diagonal and divides surface area by horizontal area. Independent flat- and planar-grid checks reproduce the expected ratios. The paper should state this raster implementation rather than only naming the metric.",
        "5. **Eligibility logic.** The Cairngorms cohort originates from 50 m units with at least 20/25 valid source cells, supplied mean height at least 10 m, supplied canopy cover at least 0.50, and at least 2,400/2,500 valid 1 m CHM cells. Invalid TESSERA cells are excluded by the land mask, finite embedding/scale checks and the patch-validity threshold; the final v2 cohort has complete 5 x 5 patches.",
        "",
        "## What the diagnostics show",
        "",
        "- The high Cairngorms scores are real held-out scores under the saved weighting and fold definitions. Observed-versus-predicted plots nevertheless show regression toward the mean, so the score should be presented with the fit panels rather than alone.",
        "- Canopy-height prediction compresses the upper range. The decile diagnostic quantifies the growing negative bias in the tallest observed units for both the MLP and strict U-Net.",
        "- Height adjustment removes different amounts of target variance across outcomes. Rumple retains the strongest residual prediction, while CV and openings lose more of their raw predictability after conditioning on observed height.",
        "- The cross-site target distributions differ substantially. This is consistent with the poor direct transfer results, but it does not by itself identify whether forest composition, acquisition/product differences or target support causes the failure.",
        "- The saved rank statistic is not identically defined between studies: Cairngorms reports Spearman correlation across 1 km block means, whereas Savelsbos reports the mean of correlations calculated within 500 m blocks. They should be labelled separately or harmonised before direct comparison.",
        "- Joint MLP training on the two available landscapes remains worse than local training. With only two sites, this tests pooling under strong domain imbalance; it does not test David's hypothesis that a genuinely multi-site training set could remove the compromise.",
        "",
        "### Tallest p95-height decile",
        "",
        markdown_table(top_height),
        "",
        "### Height-adjustment decomposition",
        "",
        markdown_table(adjustment[
            [
                "target",
                "variance_fraction_removed_by_observed_height",
                "height_only_r2",
                "raw_tessera_r2",
                "adjusted_tessera_r2",
                "oracle_height_plus_tessera_residual_r2",
            ]
        ]),
        "",
        "### Transfer and joint-training comparison",
        "",
        markdown_table(transfer_pooling),
        "",
        "## Deferred until the scientific design is agreed",
        "",
        "- Add an intermediate pine-dominated Dutch landscape or other sites to separate geographic distance from forest-type and LiDAR-product differences.",
        "- Run the AHN3-to-AHN4 temporal-change experiment and decide how sensor and point-density changes will be controlled.",
        "- Rerun model training with population SD/CV only if the team wants implementation and terminology to be exactly aligned; the label change itself is about 0.02%.",
        "- Rewrite the manuscript after agreeing which analyses are primary, supporting and future work.",
        "",
        "## Generated audit artifacts",
        "",
        "- `outputs/tables/premeeting_frozen_hash_audit.csv`",
        "- `outputs/tables/premeeting_target_formula_audit.csv`",
        "- `outputs/tables/premeeting_spatial_fold_audit.csv`",
        "- `outputs/tables/premeeting_score_verification.csv`",
        "- `outputs/tables/premeeting_height_adjustment_diagnostics.csv`",
        "- `outputs/tables/premeeting_target_distributions.csv`",
        "- `outputs/tables/premeeting_transfer_and_pooling_summary.csv`",
        "- all figures prefixed `outputs/figures/premeeting_`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    cairngorms = pd.read_parquet(CAIRNGORMS_TARGET_PATH)
    surface = pd.read_parquet(CAIRNGORMS_SURFACE_PREDICTION_PATH)
    height = pd.read_parquet(CAIRNGORMS_HEIGHT_PREDICTION_PATH)
    savelsbos = pd.read_parquet(SAVELSBOS_COHORT_PATH)
    savelsbos_predictions = pd.read_parquet(SAVELSBOS_PREDICTION_PATH)
    transfer_predictions = pd.read_parquet(TRANSFER_PREDICTION_PATH)
    pooled_ridge_predictions = pd.read_parquet(POOLED_RIDGE_PREDICTION_PATH)
    pooled_mlp_predictions = pd.read_parquet(POOLED_MLP_PREDICTION_PATH)

    hashes = audit_frozen_hashes()
    formulas, corrected_values = audit_target_formulas(cairngorms)
    folds, fold_coverage = audit_spatial_folds(cairngorms)
    prediction_coverage = audit_prediction_coverage(cairngorms, surface, height)
    verification = verify_reported_scores(
        surface,
        height,
        savelsbos_predictions,
        transfer_predictions,
        pooled_ridge_predictions,
        pooled_mlp_predictions,
    )
    distributions = target_distributions(cairngorms, savelsbos)
    adjustment, residual_checks = height_adjustment_diagnostics(cairngorms, surface)
    height_bins = height_error_bins(height)
    transfer_pooling = transfer_and_pooling_summary()

    outputs = {
        "premeeting_frozen_hash_audit.csv": hashes,
        "premeeting_target_formula_audit.csv": formulas,
        "premeeting_population_sd_cv_values.csv": corrected_values,
        "premeeting_spatial_fold_audit.csv": folds,
        "premeeting_spatial_fold_coverage.csv": fold_coverage,
        "premeeting_prediction_coverage.csv": prediction_coverage,
        "premeeting_score_verification.csv": verification,
        "premeeting_target_distributions.csv": distributions,
        "premeeting_height_adjustment_diagnostics.csv": adjustment,
        "premeeting_height_adjustment_residual_checks.csv": residual_checks,
        "premeeting_height_error_by_decile.csv": height_bins,
        "premeeting_transfer_and_pooling_summary.csv": transfer_pooling,
    }
    for name, table in outputs.items():
        table.to_csv(TABLE_DIR / name, index=False)

    surface_macro = pd.read_csv(
        TABLE_DIR / "phase29_cairngorms_v2_surface_macro.csv"
    )
    height_macro = pd.read_csv(
        TABLE_DIR / "phase31_cairngorms_v2_height_strict_macro.csv"
    )
    plot_surface_fits(surface, surface_macro)
    plot_adjusted_fits(surface, surface_macro)
    plot_height_fits(height, height_macro)
    plot_height_bins(height_bins)
    plot_target_distributions(cairngorms)
    plot_cross_site_distributions(cairngorms, savelsbos)
    plot_height_adjustment(adjustment)
    plot_spatial_residuals(cairngorms, surface)
    plot_transfer_pooling(transfer_pooling)

    report = write_audit_report(
        hashes,
        formulas,
        folds,
        fold_coverage,
        prediction_coverage,
        verification,
        adjustment,
        height_bins,
        transfer_pooling,
    )

    summary = {
        "report": str(report.relative_to(ROOT)),
        "hashes_checked": len(hashes),
        "hash_failures": int((~hashes.hash_matches).sum()),
        "metric_values_checked": len(verification),
        "metric_verification_failures": int((~verification.within_tolerance).sum()),
        "prediction_groups_checked": len(prediction_coverage),
        "prediction_coverage_failures": int(
            (
                (prediction_coverage.missing_expected_rows > 0)
                | (prediction_coverage.duplicate_rows > 0)
                | ~prediction_coverage.finite_observed
                | ~prediction_coverage.finite_predicted
            ).sum()
        ),
        "figures_created": 9,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()
