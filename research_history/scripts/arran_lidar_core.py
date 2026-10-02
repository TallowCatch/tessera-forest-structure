#!/usr/bin/env python3
"""Core utilities for the frozen Arran airborne-LiDAR replication."""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import laspy
import numpy as np
import pandas as pd
import rasterio


METRIC_COLUMNS = [
    "lidar_maxH",
    "lidar_meanH",
    "lidar_stdH",
    "lidar_p_05",
    "lidar_p_10",
    "lidar_p_25",
    "lidar_p_50",
    "lidar_p_75",
    "lidar_p_90",
    "lidar_p_95",
    "lidar_p_999",
    "lidar_vci_2m",
    "lidar_vci_5m",
    "lidar_vci_10m",
    "lidar_vci_15m",
    "lidar_Cov",
    "lidar_gapFrac",
    "lidar_grndFrac",
    "lidar_height_cv",
    "lidar_rcv",
    "lidar_rms",
    "lidar_canopy_shannon",
    "lidar_ePAI",
]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def request_json(
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    user_agent: str,
    attempts: int = 6,
    retry_delay_seconds: int = 10,
) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {
        "Accept": "application/json",
        "User-Agent": user_agent,
    }
    if body is not None:
        headers["Content-Type"] = "application/json"
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(
                url,
                data=body,
                headers=headers,
                method="GET" if body is None else "POST",
            )
            with urllib.request.urlopen(request, timeout=180) as response:
                return json.loads(response.read().decode("utf-8"))
        except (
            TimeoutError,
            urllib.error.HTTPError,
            urllib.error.URLError,
        ) as error:
            last_error = error
            if attempt < attempts:
                time.sleep(retry_delay_seconds * attempt)
    raise RuntimeError(f"Request failed after {attempts} attempts: {url}") from last_error


def download_resumable(
    url: str,
    destination: Path,
    expected_size: int,
    *,
    user_agent: str,
    attempts: int,
    retry_delay_seconds: int,
) -> None:
    """Download one immutable source object with byte-range resume."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    if destination.exists():
        if destination.stat().st_size == expected_size:
            return
        destination.unlink()
    if partial.exists() and partial.stat().st_size > expected_size:
        partial.unlink()

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        current = partial.stat().st_size if partial.exists() else 0
        if current == expected_size:
            partial.replace(destination)
            return
        headers = {"User-Agent": user_agent}
        if current:
            headers["Range"] = f"bytes={current}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                status = getattr(response, "status", response.getcode())
                mode = "ab"
                if current and status != 206:
                    mode = "wb"
                    current = 0
                with partial.open(mode) as target:
                    while True:
                        chunk = response.read(8 * 1024 * 1024)
                        if not chunk:
                            break
                        target.write(chunk)
            observed = partial.stat().st_size
            if observed == expected_size:
                partial.replace(destination)
                return
            raise RuntimeError(
                f"Incomplete download for {destination.name}: "
                f"{observed} of {expected_size} bytes"
            )
        except (
            OSError,
            TimeoutError,
            urllib.error.HTTPError,
            urllib.error.URLError,
        ) as error:
            last_error = error
            if attempt < attempts:
                time.sleep(retry_delay_seconds * attempt)
    raise RuntimeError(f"Failed to download {url}") from last_error


def shannon_from_bins(values: np.ndarray, breaks: np.ndarray) -> float:
    """Match lidarSHM::roughness_metrics_f and R cut(..., right=TRUE)."""
    values = np.asarray(values, dtype=np.float64)
    counts = np.asarray(
        [
            np.count_nonzero((values > lower) & (values <= upper))
            for lower, upper in zip(breaks[:-1], breaks[1:], strict=True)
        ],
        dtype=np.float64,
    )
    total = counts.sum()
    if total <= 0:
        return 0.0
    probabilities = counts[counts > 0] / total
    return float(-np.sum(probabilities * np.log(probabilities)))


def vertical_complexity_index(
    values: np.ndarray,
    *,
    zmax: float,
    bin_width: float,
) -> float:
    """Match lidR::VCI: Shannon entropy divided by the uniform-bin entropy."""
    values = np.asarray(values, dtype=np.float64)
    values = values[(values >= 0) & (values < zmax)]
    number_of_bins = int(math.ceil(zmax / bin_width))
    if len(values) == 0 or number_of_bins <= 1:
        return float("nan")
    indices = np.floor(values / bin_width).astype(np.int64)
    counts = np.bincount(indices, minlength=number_of_bins)[:number_of_bins]
    probabilities = counts[counts > 0].astype(np.float64)
    probabilities /= probabilities.sum()
    entropy = -np.sum(probabilities * np.log(probabilities))
    return float(entropy / math.log(number_of_bins))


def _finite_divide(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def calculate_cell_metrics(
    height: np.ndarray,
    return_number: np.ndarray,
    number_of_returns: np.ndarray,
    scan_angle_degrees: np.ndarray,
    *,
    h_cutoff: float,
    gap_threshold: float,
    zmax: float,
    shannon_breaks: np.ndarray,
) -> dict[str, float]:
    """Calculate the preregistered lidarSHM-compatible metrics for one 10 m cell."""
    height = np.asarray(height, dtype=np.float64)
    return_number = np.asarray(return_number, dtype=np.int16)
    number_of_returns = np.asarray(number_of_returns, dtype=np.float64)
    scan_angle_degrees = np.asarray(scan_angle_degrees, dtype=np.float64)
    valid = (
        np.isfinite(height)
        & np.isfinite(number_of_returns)
        & (number_of_returns >= 1)
        & np.isfinite(scan_angle_degrees)
    )
    height = height[valid]
    return_number = return_number[valid]
    number_of_returns = number_of_returns[valid]
    scan_angle_degrees = scan_angle_degrees[valid]
    if len(height) == 0:
        return {name: float("nan") for name in METRIC_COLUMNS}

    canopy = height[height > h_cutoff]
    first = return_number == 1
    first_count = int(first.sum())
    if len(canopy) == 0 or first_count == 0:
        return {name: float("nan") for name in METRIC_COLUMNS}

    quantiles = np.quantile(
        canopy,
        [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.999],
        method="linear",
    )
    mean_height = float(np.mean(canopy))
    standard_deviation = (
        float(np.std(canopy, ddof=1)) if len(canopy) > 1 else float("nan")
    )
    all_mean = float(np.mean(height))
    all_standard_deviation = (
        float(np.std(height, ddof=1)) if len(height) > 1 else float("nan")
    )
    all_median = float(np.median(height))
    all_iqr = float(np.quantile(height, 0.75) - np.quantile(height, 0.25))

    average_angle = float(np.mean(np.abs(scan_angle_degrees)))
    angle_radians = math.radians(average_angle)
    cosine = math.cos(angle_radians)
    safe_returns = 1.0 / number_of_returns
    weighted_gap = float(np.sum(safe_returns[height < gap_threshold]))
    gap_ratio = _finite_divide(weighted_gap, first_count)
    gap_fraction = (
        float(gap_ratio**cosine)
        if np.isfinite(gap_ratio) and gap_ratio >= 0
        else float("nan")
    )
    weighted_canopy = float(np.sum(safe_returns[height > h_cutoff]))
    canopy_ratio = _finite_divide(weighted_canopy, first_count)
    x_parameter = 2.0
    extinction = math.sqrt(x_parameter**2 + math.tan(angle_radians) ** 2) / (
        x_parameter + 1.744 * (x_parameter + 1.182) ** (-0.733)
    )
    epai = (
        -math.log1p(-canopy_ratio) * cosine / extinction
        if np.isfinite(canopy_ratio) and 0 <= canopy_ratio < 1
        else float("nan")
    )

    return {
        "lidar_maxH": float(np.max(canopy)),
        "lidar_meanH": mean_height,
        "lidar_stdH": standard_deviation,
        "lidar_p_05": float(quantiles[0]),
        "lidar_p_10": float(quantiles[1]),
        "lidar_p_25": float(quantiles[2]),
        "lidar_p_50": float(quantiles[3]),
        "lidar_p_75": float(quantiles[4]),
        "lidar_p_90": float(quantiles[5]),
        "lidar_p_95": float(quantiles[6]),
        "lidar_p_999": float(quantiles[7]),
        "lidar_vci_2m": vertical_complexity_index(
            canopy, zmax=zmax, bin_width=2.0
        ),
        "lidar_vci_5m": vertical_complexity_index(
            canopy, zmax=zmax, bin_width=5.0
        ),
        "lidar_vci_10m": vertical_complexity_index(
            canopy, zmax=zmax, bin_width=10.0
        ),
        "lidar_vci_15m": vertical_complexity_index(
            canopy, zmax=zmax, bin_width=15.0
        ),
        "lidar_Cov": float(np.mean(height[first] > h_cutoff)),
        "lidar_gapFrac": gap_fraction,
        "lidar_grndFrac": float(np.mean(height[first] < 0.5)),
        "lidar_height_cv": _finite_divide(all_standard_deviation, all_mean),
        "lidar_rcv": _finite_divide(all_iqr, all_median),
        "lidar_rms": float(np.sqrt(np.mean((height - all_mean) ** 2))),
        "lidar_canopy_shannon": shannon_from_bins(height, shannon_breaks),
        "lidar_ePAI": epai,
    }


def forest_quality_mask(
    frame: pd.DataFrame,
    rules: dict[str, Any],
) -> np.ndarray:
    required = [
        "lidar_maxH",
        "lidar_meanH",
        "lidar_p_95",
        "lidar_p_999",
        "lidar_Cov",
        "lidar_canopy_shannon",
    ]
    valid = np.isfinite(frame[required].to_numpy(dtype=np.float64)).all(axis=1)
    valid &= frame["lidar_meanH"].to_numpy() >= 1.3
    valid &= frame["lidar_meanH"].to_numpy() <= frame["lidar_maxH"].to_numpy()
    valid &= frame["lidar_p_95"].to_numpy() >= float(
        rules["minimum_p95_height_m"]
    )
    valid &= frame["lidar_p_95"].to_numpy() <= frame["lidar_p_999"].to_numpy()
    valid &= frame["lidar_p_999"].to_numpy() <= float(
        rules["maximum_p999_height_m"]
    )
    valid &= frame["lidar_maxH"].to_numpy() <= float(
        rules["maximum_return_height_m"]
    )
    valid &= frame["lidar_Cov"].to_numpy() >= float(
        rules["minimum_canopy_cover"]
    )
    valid &= frame["lidar_Cov"].to_numpy() <= float(
        rules["maximum_canopy_cover"]
    )
    valid &= frame["lidar_canopy_shannon"].to_numpy() >= 0
    valid &= frame["lidar_canopy_shannon"].to_numpy() <= float(
        rules["maximum_canopy_shannon"]
    )
    return valid


def _scan_angle_degrees(points: laspy.ScaleAwarePointRecord, point_format: int) -> np.ndarray:
    if point_format >= 6:
        return np.asarray(points.scan_angle, dtype=np.float32) * 0.006
    return np.asarray(points.scan_angle_rank, dtype=np.float32)


def process_laz_tile(
    laz_path: Path,
    dtm_path: Path,
    *,
    tile_name: str,
    metric_resolution_m: float,
    expected_dtm_resolution_m: float,
    expected_crs_epsg: int,
    chunk_points: int,
    metric_settings: dict[str, Any],
    forest_rules: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read raw LAZ directly and return quality-controlled 10 m cell metrics."""
    with rasterio.open(dtm_path) as terrain:
        if terrain.crs is None or terrain.crs.to_epsg() != expected_crs_epsg:
            raise RuntimeError(f"{tile_name}: unexpected DTM CRS {terrain.crs}")
        if not np.allclose(
            terrain.res,
            [expected_dtm_resolution_m, expected_dtm_resolution_m],
            atol=1e-9,
        ):
            raise RuntimeError(f"{tile_name}: unexpected DTM resolution {terrain.res}")
        ground = terrain.read(1).astype(np.float32)
        transform = terrain.transform
        bounds = terrain.bounds
        nodata = terrain.nodata

    width_m = float(bounds.right - bounds.left)
    height_m = float(bounds.top - bounds.bottom)
    cells_x = int(round(width_m / metric_resolution_m))
    cells_y = int(round(height_m / metric_resolution_m))
    if cells_x <= 0 or cells_y <= 0:
        raise RuntimeError(f"{tile_name}: invalid DTM bounds {bounds}")

    cell_parts: list[np.ndarray] = []
    height_parts: list[np.ndarray] = []
    return_parts: list[np.ndarray] = []
    returns_parts: list[np.ndarray] = []
    angle_parts: list[np.ndarray] = []
    raw_points = 0
    retained_points = 0
    class_counts: dict[int, int] = {}

    with laspy.open(laz_path) as source:
        point_format = int(source.header.point_format.id)
        source_crs = source.header.parse_crs()
        if source_crs is None or source_crs.to_epsg() != expected_crs_epsg:
            raise RuntimeError(f"{tile_name}: unexpected LAZ CRS {source_crs}")
        for points in source.chunk_iterator(chunk_points):
            raw_points += len(points)
            classes, counts = np.unique(points.classification, return_counts=True)
            for classification, count in zip(classes, counts, strict=True):
                key = int(classification)
                class_counts[key] = class_counts.get(key, 0) + int(count)

            x = np.asarray(points.x, dtype=np.float64)
            y = np.asarray(points.y, dtype=np.float64)
            z = np.asarray(points.z, dtype=np.float64)
            terrain_columns = np.floor((x - transform.c) / transform.a).astype(
                np.int64
            )
            terrain_rows = np.floor((y - transform.f) / transform.e).astype(
                np.int64
            )
            inside = (
                (x >= bounds.left)
                & (x < bounds.right)
                & (y >= bounds.bottom)
                & (y < bounds.top)
                & (terrain_rows >= 0)
                & (terrain_rows < ground.shape[0])
                & (terrain_columns >= 0)
                & (terrain_columns < ground.shape[1])
            )
            indices = np.flatnonzero(inside)
            if len(indices) == 0:
                continue
            local_ground = ground[
                terrain_rows[indices],
                terrain_columns[indices],
            ]
            usable = np.isfinite(local_ground)
            if nodata is not None:
                usable &= ~np.isclose(local_ground, nodata)
            indices = indices[usable]
            local_ground = local_ground[usable]
            if len(indices) == 0:
                continue

            columns_10m = np.floor(
                (x[indices] - bounds.left) / metric_resolution_m
            ).astype(np.int32)
            rows_10m = np.floor(
                (y[indices] - bounds.bottom) / metric_resolution_m
            ).astype(np.int32)
            valid_cells = (
                (columns_10m >= 0)
                & (columns_10m < cells_x)
                & (rows_10m >= 0)
                & (rows_10m < cells_y)
            )
            indices = indices[valid_cells]
            local_ground = local_ground[valid_cells]
            columns_10m = columns_10m[valid_cells]
            rows_10m = rows_10m[valid_cells]
            if len(indices) == 0:
                continue
            cell_parts.append(
                (rows_10m * cells_x + columns_10m).astype(np.int32)
            )
            height_parts.append((z[indices] - local_ground).astype(np.float32))
            return_parts.append(
                np.asarray(points.return_number[indices], dtype=np.uint8)
            )
            returns_parts.append(
                np.asarray(points.number_of_returns[indices], dtype=np.uint8)
            )
            angle_parts.append(
                _scan_angle_degrees(points, point_format)[indices]
            )
            retained_points += len(indices)

    if not cell_parts:
        empty = pd.DataFrame(
            columns=[
                "source_tile",
                "cell_index",
                "bng_x",
                "bng_y",
                "point_count",
                "canopy_point_count",
                *METRIC_COLUMNS,
            ]
        )
        return empty, {
            "raw_points": raw_points,
            "retained_points_after_dtm_alignment": 0,
            "eligible_forest_cells": 0,
            "classification_counts": class_counts,
        }

    cell = np.concatenate(cell_parts)
    height = np.concatenate(height_parts)
    return_number = np.concatenate(return_parts)
    number_of_returns = np.concatenate(returns_parts)
    scan_angle = np.concatenate(angle_parts)
    order = np.argsort(cell, kind="stable")
    cell = cell[order]
    height = height[order]
    return_number = return_number[order]
    number_of_returns = number_of_returns[order]
    scan_angle = scan_angle[order]
    unique_cells, starts, counts = np.unique(
        cell,
        return_index=True,
        return_counts=True,
    )

    h_cutoff = float(metric_settings["h_cutoff_m"])
    rows: list[dict[str, Any]] = []
    breaks = np.asarray(metric_settings["shannon_breaks_m"], dtype=np.float64)
    for cell_index, start, count in zip(
        unique_cells, starts, counts, strict=True
    ):
        stop = start + count
        local_height = height[start:stop]
        metrics = calculate_cell_metrics(
            local_height,
            return_number[start:stop],
            number_of_returns[start:stop],
            scan_angle[start:stop],
            h_cutoff=h_cutoff,
            gap_threshold=float(metric_settings["gap_threshold_m"]),
            zmax=float(metric_settings["vci_zmax_m"]),
            shannon_breaks=breaks,
        )
        local_row = int(cell_index) // cells_x
        local_column = int(cell_index) % cells_x
        rows.append(
            {
                "source_tile": tile_name,
                "cell_index": int(cell_index),
                "bng_x": bounds.left
                + (local_column + 0.5) * metric_resolution_m,
                "bng_y": bounds.bottom
                + (local_row + 0.5) * metric_resolution_m,
                "point_count": int(count),
                "canopy_point_count": int(np.count_nonzero(local_height > h_cutoff)),
                **metrics,
            }
        )

    frame = pd.DataFrame(rows)
    quality = forest_quality_mask(frame, forest_rules)
    frame = frame.loc[quality].reset_index(drop=True)
    float_columns = ["bng_x", "bng_y", *METRIC_COLUMNS]
    frame[float_columns] = frame[float_columns].astype(np.float32)
    summary = {
        "raw_points": raw_points,
        "retained_points_after_dtm_alignment": retained_points,
        "metric_cells_with_returns": int(len(unique_cells)),
        "eligible_forest_cells": int(len(frame)),
        "classification_counts": class_counts,
        "point_format": point_format,
        "metric_resolution_m": metric_resolution_m,
        "normalization": "official_same_capture_dtm_containing_50cm_cell",
    }
    return frame, summary
