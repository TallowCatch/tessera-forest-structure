#!/usr/bin/env python3
"""Rebuild the selected Dutch 50 m LiDAR units used before TESSERA filtering.

The original prepared cohort was intentionally removed during local storage cleanup.
This lightweight recovery reads only the two AHN4 layers needed for eligibility and
checks every reconstructed site against the frozen preparation summary.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import yaml
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window, from_bounds

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "workflows/dutch_multiforest/parameters.yaml"
SUMMARY = ROOT / "results/tables/dutch_forest_sites.csv"
OUTPUT = ROOT / "data/processed/dutch_eligible_units_map.parquet"
RECONSTRUCTION_CHECK = ROOT / "results/tables/diagnostics/dutch_eligible_unit_reconstruction.csv"
MEAN_LOCAL = ROOT / "data/raw/ahn4/ahn4_10m_mean_normalized_height.tif"
PULSE_LOCAL = ROOT / "data/raw/ahn4/ahn4_10m_pulse_penetration_ratio.tif"
MEAN_URL = (
    "/vsicurl/https://zenodo.org/api/records/8422129/files/"
    "ahn4_10m_mean_normalized_height.tif/content"
)
PULSE_URL = (
    "/vsicurl/https://zenodo.org/api/records/8422129/files/"
    "ahn4_10m_pulse_penetration_ratio.tif/content"
)
AUXILIARY_FILENAMES = [
    "ahn4_10m_perc_95_normalized_height.tif",
    "ahn4_10m_std_normalized_height.tif",
    "ahn4_10m_coeff_var_normalized_height.tif",
    "ahn4_10m_entropy_normalized_height.tif",
    "ahn4_10m_sigma_z.tif",
]


def aligned_window(bounds: np.ndarray, dataset: rasterio.DatasetReader, cells: int) -> Window:
    raw = from_bounds(*bounds, transform=dataset.transform)
    row_start = max(0, (int(math.floor(raw.row_off)) // cells) * cells)
    column_start = max(0, (int(math.floor(raw.col_off)) // cells) * cells)
    row_stop = min(
        dataset.height,
        ((int(math.ceil(raw.row_off + raw.height)) + cells - 1) // cells) * cells,
    )
    column_stop = min(
        dataset.width,
        ((int(math.ceil(raw.col_off + raw.width)) + cells - 1) // cells) * cells,
    )
    row_stop -= (row_stop - row_start) % cells
    column_stop -= (column_stop - column_start) % cells
    if row_stop <= row_start or column_stop <= column_start:
        raise RuntimeError("Selected site does not intersect the AHN4 grid")
    return Window(column_start, row_start, column_stop - column_start, row_stop - row_start)


def patch_sum(values: np.ndarray, cells: int) -> np.ndarray:
    rows = values.shape[0] // cells
    columns = values.shape[1] // cells
    return values.reshape(rows, cells, columns, cells).sum(axis=(1, 3))


def patch_mean(values: np.ndarray, valid: np.ndarray, cells: int) -> np.ndarray:
    total = patch_sum(np.where(valid, values, 0.0), cells)
    count = patch_sum(valid.astype(np.int16), cells)
    output = np.full(total.shape, np.nan, dtype=np.float32)
    np.divide(total, count, out=output, where=count > 0)
    return output


def read_aligned(path: str | Path, window: Window, reference: rasterio.DatasetReader) -> np.ndarray:
    with rasterio.open(path) as source:
        with WarpedVRT(
            source,
            crs=reference.crs,
            transform=reference.transform,
            width=reference.width,
            height=reference.height,
            resampling=Resampling.nearest,
        ) as aligned:
            values = aligned.read(1, window=window, masked=True)
    return values


def read_remote_window(
    path: str, window: Window, reference: rasterio.DatasetReader, attempts: int = 8
) -> np.ma.MaskedArray:
    """Read a same-grid remote COG window with retries for transient HTTP failures."""
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            with rasterio.open(path) as source:
                if (
                    source.crs != reference.crs
                    or source.transform != reference.transform
                    or source.shape != reference.shape
                ):
                    raise RuntimeError("Remote AHN4 grids are not aligned")
                return source.read(1, window=window, masked=True)
        except rasterio.errors.RasterioIOError as caught:
            error = caught
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Remote AHN4 window failed after {attempts} attempts") from error


def read_genus(paths: list[Path], window: Window, reference: rasterio.DatasetReader) -> np.ndarray:
    output = np.full((int(window.height), int(window.width)), 255, dtype=np.uint8)
    for path in paths:
        with rasterio.open(path) as source:
            with WarpedVRT(
                source,
                crs=reference.crs,
                transform=reference.transform,
                width=reference.width,
                height=reference.height,
                src_nodata=255,
                nodata=255,
                resampling=Resampling.nearest,
            ) as aligned:
                values = aligned.read(1, window=window, masked=False).astype(np.uint8)
        valid = values != 255
        output[valid] = values[valid]
    return output


def mode_year(dates: np.ndarray, usable: np.ndarray, cells: int) -> tuple[np.ndarray, np.ndarray]:
    years = dates // 10000
    candidates = [year for year in np.unique(years[usable]) if year > 0]
    if not candidates:
        shape = (dates.shape[0] // cells, dates.shape[1] // cells)
        return np.zeros(shape, dtype=np.int16), np.zeros(shape, dtype=np.float32)
    counts = np.stack(
        [patch_sum((usable & (years == year)).astype(np.int16), cells) for year in candidates]
    )
    winner = np.argmax(counts, axis=0)
    year_values = np.asarray(candidates, dtype=np.int16)[winner]
    winning_count = np.take_along_axis(counts, winner[None], axis=0)[0]
    usable_count = patch_sum(usable.astype(np.int16), cells)
    purity = np.zeros_like(winning_count, dtype=np.float32)
    np.divide(winning_count, usable_count, out=purity, where=usable_count > 0)
    return year_values, purity


def spatially_balanced_cap(frame: pd.DataFrame, maximum: int, seed: int) -> pd.DataFrame:
    if len(frame) <= maximum:
        return frame.copy()
    rng = np.random.default_rng(seed)
    groups: dict[str, list[int]] = {}
    for block, positions in frame.groupby("sampling_block", sort=True).indices.items():
        values = np.asarray(positions, dtype=np.int64)
        rng.shuffle(values)
        groups[str(block)] = values.tolist()
    selected: list[int] = []
    ordered = sorted(groups)
    while len(selected) < maximum:
        active = False
        for block in ordered:
            if groups[block]:
                selected.append(groups[block].pop())
                active = True
                if len(selected) == maximum:
                    break
        if not active:
            break
    return frame.iloc[np.sort(np.asarray(selected, dtype=np.int64))].copy()


def rebuild_site(
    summary: pd.Series,
    boundary: gpd.GeoDataFrame,
    config: dict,
    reference: rasterio.DatasetReader,
    genus_paths: list[Path],
) -> pd.DataFrame:
    population = config["population"]
    genus = config["genus_map"]
    cells = int(population["output_support_m"] // population["lidar_cell_m"])
    window = aligned_window(boundary.total_bounds, reference, cells)
    transform = reference.window_transform(window)
    shape = (int(window.height), int(window.width))
    site_mask = rasterize(
        ((geometry, 1) for geometry in boundary.geometry),
        out_shape=shape,
        transform=transform,
        fill=0,
        all_touched=False,
        dtype="uint8",
    ).astype(bool)

    classes = read_genus(genus_paths, window, reference)
    conifer = np.isin(classes, np.asarray(genus["conifer_classes"], dtype=np.uint8))
    broadleaf = np.isin(classes, np.asarray(genus["broadleaf_classes"], dtype=np.uint8))
    tree = conifer | broadleaf
    expected = conifer if summary.forest_group == "conifer" else broadleaf

    mean_source = str(MEAN_LOCAL) if MEAN_LOCAL.exists() else MEAN_URL
    pulse_source = str(PULSE_LOCAL) if PULSE_LOCAL.exists() else PULSE_URL
    mean_height = (
        read_remote_window(mean_source, window, reference).filled(np.nan).astype(np.float32)
    )
    penetration = (
        read_remote_window(pulse_source, window, reference).filled(np.nan).astype(np.float32)
    )
    excluded = np.zeros(shape, dtype=bool)
    for relative_path in config["ahn4"]["masks"]:
        values = read_aligned(ROOT / relative_path, window, reference)
        excluded |= np.asarray(values.filled(0) > 0, dtype=bool)
    dates = (
        read_aligned(ROOT / config["ahn4"]["flighttime"], window, reference)
        .filled(0)
        .astype(np.int32)
    )
    finite = np.isfinite(mean_height) & np.isfinite(penetration)
    height_cv: np.ndarray | None = None
    for filename in AUXILIARY_FILENAMES:
        local_path = ROOT / "data/raw/ahn4" / filename
        auxiliary_source = (
            str(local_path)
            if local_path.exists()
            else "/vsicurl/https://zenodo.org/api/records/8422129/files/" + filename + "/content"
        )
        auxiliary = read_remote_window(auxiliary_source, window, reference).filled(np.nan)
        finite &= np.isfinite(auxiliary)
        if filename == "ahn4_10m_coeff_var_normalized_height.tif":
            height_cv = auxiliary.astype(np.float32)
    if height_cv is None:
        raise RuntimeError("AHN4 height-CV layer was not read")
    usable = site_mask & tree & ~excluded & finite & (dates > 0)

    tree_count = patch_sum((site_mask & tree).astype(np.int16), cells)
    group_count = patch_sum((site_mask & expected).astype(np.int16), cells)
    usable_count = patch_sum(usable.astype(np.int16), cells)
    tree_fraction = tree_count / float(cells * cells)
    group_fraction = np.zeros_like(tree_fraction, dtype=np.float32)
    np.divide(group_count, tree_count, out=group_fraction, where=tree_count > 0)
    year, year_fraction = mode_year(dates, usable, cells)
    candidate = (
        (tree_fraction >= float(genus["minimum_tree_fraction"]))
        & (group_fraction >= float(genus["minimum_group_fraction_among_trees"]))
        & (usable_count >= np.ceil(tree_count * float(population["minimum_valid_lidar_fraction"])))
        & (year_fraction >= float(population["minimum_single_year_fraction"]))
    )

    rows, columns = np.where(candidate)
    pixel_rows = rows * cells + (cells - 1) / 2
    pixel_columns = columns * cells + (cells - 1) / 2
    x = transform.c + (pixel_columns + 0.5) * transform.a
    y = transform.f + (pixel_rows + 0.5) * transform.e
    mean_50 = patch_mean(mean_height, usable, cells)[candidate]
    penetration_50 = patch_mean(penetration, usable, cells)[candidate]
    height_cv_50 = patch_mean(height_cv, usable, cells)[candidate]
    frame = pd.DataFrame(
        {
            "site_code": summary.site_code,
            "site_name": summary.site_name,
            "forest_group": summary.forest_group,
            "rd_x": x.astype(np.float64),
            "rd_y": y.astype(np.float64),
            "tessera_year": year[candidate].astype(np.int16),
            "tree_fraction": tree_fraction[candidate].astype(np.float32),
            "group_fraction": group_fraction[candidate].astype(np.float32),
            "ahn4_mean_height_m": mean_50.astype(np.float32),
            "ahn4_height_cv": height_cv_50.astype(np.float32),
            "ahn4_pulse_penetration": penetration_50.astype(np.float32),
        }
    )
    frame = frame[
        (frame["ahn4_mean_height_m"] > float(population["minimum_mean_height_m"]))
        & (frame["ahn4_pulse_penetration"] < float(population["maximum_pulse_penetration"]))
    ].reset_index(drop=True)
    block_m = int(population["sampling_block_m"])
    frame["sampling_block"] = (
        np.floor(frame["rd_x"] / block_m).astype(np.int32).astype(str)
        + "_"
        + np.floor(frame["rd_y"] / block_m).astype(np.int32).astype(str)
    )
    frame = spatially_balanced_cap(
        frame,
        int(population["maximum_rows_per_site"]),
        int(population["sampling_seed"]) + sum(map(ord, str(summary.site_code))),
    ).reset_index(drop=True)

    return frame


def main() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    summary = pd.read_csv(SUMMARY).sort_values("selection_order").reset_index(drop=True)
    boundaries = gpd.read_file(ROOT / config["study"]["site_source"]).to_crs(
        config["study"]["analysis_crs"]
    )
    boundaries = boundaries[boundaries["SITENAME"].isin(summary["site_name"])].dissolve(
        by="SITENAME"
    )
    genus_paths = sorted((ROOT / config["genus_map"]["tile_directory"]).glob("*.tif"))
    if not genus_paths:
        raise FileNotFoundError("Extract the required European tree-genus GeoTIFFs first")

    parts: list[pd.DataFrame] = []
    checks: list[dict[str, object]] = []
    with rasterio.Env(
        GDAL_HTTP_MULTIRANGE="YES",
        GDAL_HTTP_MAX_RETRY="8",
        GDAL_HTTP_RETRY_DELAY="2",
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    ):
        mean_source = str(MEAN_LOCAL) if MEAN_LOCAL.exists() else MEAN_URL
        with rasterio.open(mean_source) as reference:
            for row in summary.itertuples(index=False):
                boundary = boundaries.loc[[row.site_name]].reset_index()
                frame = rebuild_site(
                    pd.Series(row._asdict()), boundary, config, reference, genus_paths
                )
                parts.append(frame)
                expected = int(row.retained_units)
                marker = "" if len(frame) == expected else f" (expected {expected:,})"
                checks.append(
                    {
                        "site_name": row.site_name,
                        "reconstructed_units": len(frame),
                        "frozen_units": expected,
                        "difference": len(frame) - expected,
                    }
                )
                print(f"rebuilt {row.site_name}: {len(frame):,} units{marker}", flush=True)
    output = pd.concat(parts, ignore_index=True)
    output.insert(0, "row_id", np.arange(len(output), dtype=np.int64))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(OUTPUT, index=False, compression="zstd")
    RECONSTRUCTION_CHECK.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(checks).to_csv(RECONSTRUCTION_CHECK, index=False)
    frozen_total = int(summary["retained_units"].sum())
    print(
        f"wrote {OUTPUT.relative_to(ROOT)} with {len(output):,} reconstructed units "
        f"(frozen total {frozen_total:,})",
        flush=True,
    )


if __name__ == "__main__":
    main()
