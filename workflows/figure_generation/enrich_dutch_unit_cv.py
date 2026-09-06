#!/usr/bin/env python3
"""Attach AHN4 height CV to the reconstructed Dutch eligible-unit map."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import yaml
from rasterio.features import rasterize
from rebuild_dutch_eligible_units import (
    CONFIG,
    MEAN_LOCAL,
    OUTPUT,
    PULSE_LOCAL,
    ROOT,
    aligned_window,
    patch_mean,
    read_aligned,
    read_genus,
)

CV_LOCAL = ROOT / "data/raw/ahn4/ahn4_10m_coeff_var_normalized_height.tif"
SAVELSBOS_REFERENCE = ROOT / "data/processed/savelsbos_targets.parquet"
VALIDATION_OUTPUT = ROOT / "results/tables/diagnostics/dutch_height_cv_reconstruction.csv"


def local_float(path: Path, window: rasterio.windows.Window) -> np.ndarray:
    with rasterio.open(path) as source:
        return source.read(1, window=window, masked=True).filled(np.nan).astype(np.float32)


def enrich_site(
    frame: pd.DataFrame,
    boundary: gpd.GeoDataFrame,
    config: dict,
    reference: rasterio.DatasetReader,
    genus_paths: list[Path],
) -> np.ndarray:
    population = config["population"]
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
    genus = config["genus_map"]
    tree = np.isin(
        classes,
        np.asarray(genus["conifer_classes"] + genus["broadleaf_classes"], dtype=np.uint8),
    )
    mean_height = reference.read(1, window=window, masked=True).filled(np.nan).astype(np.float32)
    penetration = local_float(PULSE_LOCAL, window)
    height_cv = local_float(CV_LOCAL, window)

    excluded = np.zeros(shape, dtype=bool)
    for relative_path in config["ahn4"]["masks"]:
        values = read_aligned(ROOT / relative_path, window, reference)
        excluded |= np.asarray(values.filled(0) > 0, dtype=bool)
    dates = read_aligned(ROOT / config["ahn4"]["flighttime"], window, reference).filled(0)
    usable = (
        site_mask
        & tree
        & ~excluded
        & np.isfinite(mean_height)
        & np.isfinite(penetration)
        & np.isfinite(height_cv)
        & (dates > 0)
    )
    cv_50 = patch_mean(height_cv, usable, cells)

    inverse = ~transform
    indices: list[tuple[int, int]] = []
    for x, y in frame[["rd_x", "rd_y"]].to_numpy(dtype=float):
        pixel_column, pixel_row = inverse * (x, y)
        indices.append((int(np.floor(pixel_row)) // cells, int(np.floor(pixel_column)) // cells))
    return np.asarray([cv_50[row, column] for row, column in indices], dtype=np.float32)


def validate_savelsbos(units: pd.DataFrame) -> pd.DataFrame:
    reference = pd.read_parquet(SAVELSBOS_REFERENCE)
    required = {"rd_x", "rd_y", "ahn4_height_cv"}
    if not required.issubset(reference.columns):
        return pd.DataFrame()
    current = units[units["site_name"] == "Savelsbos"][["rd_x", "rd_y", "ahn4_height_cv"]]
    merged = current.merge(
        reference[["rd_x", "rd_y", "ahn4_height_cv"]],
        on=["rd_x", "rd_y"],
        suffixes=("_reconstructed", "_retained"),
    )
    difference = merged["ahn4_height_cv_reconstructed"] - merged["ahn4_height_cv_retained"]
    return pd.DataFrame(
        {
            "matched_units": [len(merged)],
            "mean_absolute_difference": [float(np.mean(np.abs(difference)))],
            "maximum_absolute_difference": [float(np.max(np.abs(difference)))],
            "pearson_correlation": [
                float(
                    np.corrcoef(
                        merged["ahn4_height_cv_reconstructed"], merged["ahn4_height_cv_retained"]
                    )[0, 1]
                )
            ],
        }
    )


def main() -> None:
    if not CV_LOCAL.exists():
        raise FileNotFoundError(CV_LOCAL)
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    units = pd.read_parquet(OUTPUT)
    boundaries = gpd.read_file(ROOT / config["study"]["site_source"]).to_crs(
        config["study"]["analysis_crs"]
    )
    boundaries = boundaries[boundaries["SITENAME"].isin(units["site_name"])].dissolve(by="SITENAME")
    genus_paths = sorted((ROOT / config["genus_map"]["tile_directory"]).glob("*.tif"))
    values = np.full(len(units), np.nan, dtype=np.float32)
    with rasterio.open(MEAN_LOCAL) as reference:
        for site_name, positions in units.groupby("site_name", sort=False).indices.items():
            frame = units.iloc[positions]
            boundary = boundaries.loc[[site_name]].reset_index()
            values[np.asarray(positions)] = enrich_site(
                frame, boundary, config, reference, genus_paths
            )
            print(f"sampled {site_name}: {len(frame):,} units", flush=True)
    if not np.isfinite(values).all():
        raise RuntimeError(f"Height CV was missing for {int((~np.isfinite(values)).sum())} units")
    units["ahn4_height_cv"] = values
    units.to_parquet(OUTPUT, index=False, compression="zstd")

    validation = validate_savelsbos(units)
    VALIDATION_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    validation.to_csv(VALIDATION_OUTPUT, index=False)
    print(validation.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
