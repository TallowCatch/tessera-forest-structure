#!/usr/bin/env python3
"""Run the resumable AHN3-to-AHN4 Dutch temporal-transfer experiment."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
import yaml
from affine import Affine
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import rowcol
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window, from_bounds
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
PARAMETERS = Path(__file__).with_name("parameters.yaml")
S3_ROOT = "https://s3.us-west-2.amazonaws.com/tessera-embeddings"
EMBEDDINGS_SUBDIR = "global_0.1_degree_representation"
LANDMASKS_SUBDIR = "global_0.1_degree_tiff_all"


def log(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{stamp}] {message}", flush=True)


def load_parameters() -> dict[str, Any]:
    return yaml.safe_load(PARAMETERS.read_text(encoding="utf-8"))


def rooted(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


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


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.csv")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def patch_matrix(values: np.ndarray, cells: int) -> np.ndarray:
    rows = values.shape[0] // cells
    columns = values.shape[1] // cells
    trimmed = values[: rows * cells, : columns * cells]
    return trimmed.reshape(rows, cells, columns, cells).transpose(0, 2, 1, 3)


def patch_sum(values: np.ndarray, cells: int) -> np.ndarray:
    return patch_matrix(values, cells).sum(axis=(2, 3))


def patch_bitmask(values: np.ndarray, cells: int) -> np.ndarray:
    patches = patch_matrix(values.astype(np.uint32), cells).reshape(
        values.shape[0] // cells, values.shape[1] // cells, cells * cells
    )
    bits = 1 << np.arange(cells * cells, dtype=np.uint32)
    return (patches * bits).sum(axis=2, dtype=np.uint32)


def aligned_window(
    bounds: Iterable[float], dataset: rasterio.DatasetReader, factor: int
) -> Window:
    raw = from_bounds(*bounds, transform=dataset.transform)
    column_start = max(0, int(math.floor(raw.col_off / factor) * factor))
    row_start = max(0, int(math.floor(raw.row_off / factor) * factor))
    column_stop = min(
        dataset.width,
        int(math.ceil((raw.col_off + raw.width) / factor) * factor),
    )
    row_stop = min(
        dataset.height,
        int(math.ceil((raw.row_off + raw.height) / factor) * factor),
    )
    column_stop -= (column_stop - column_start) % factor
    row_stop -= (row_stop - row_start) % factor
    return Window(
        column_start,
        row_start,
        column_stop - column_start,
        row_stop - row_start,
    )


def read_genus_window(
    paths: list[Path], window: Window, reference: rasterio.DatasetReader
) -> np.ndarray:
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
                values = aligned.read(1, window=window, masked=False)
        valid = values != 255
        output[valid] = values[valid]
    return output


def point_arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    local_rows = np.repeat(np.arange(5, dtype=np.int8), 5)
    local_columns = np.tile(np.arange(5, dtype=np.int8), 5)
    offset_x = local_columns - 2
    offset_y = 2 - local_rows
    bits = 1 << np.arange(25, dtype=np.uint32)
    masks = frame["forest_pixel_mask"].to_numpy(dtype=np.uint32)[:, None]
    keep = ((masks & bits[None, :]) > 0).reshape(-1)
    owners = np.repeat(np.arange(len(frame), dtype=np.int64), 25)[keep]
    x = (
        np.repeat(frame["rd_x"].to_numpy(dtype=np.float64), 25)
        + np.tile(offset_x.astype(np.float64) * 10.0, len(frame))
    )[keep]
    y = (
        np.repeat(frame["rd_y"].to_numpy(dtype=np.float64), 25)
        + np.tile(offset_y.astype(np.float64) * 10.0, len(frame))
    )[keep]
    return owners, x, y


def popcount(values: pd.Series) -> np.ndarray:
    return np.fromiter(
        (int(value).bit_count() for value in values),
        dtype=np.int16,
        count=len(values),
    )


def candidate_units(parameters: dict[str, Any], envelopes: gpd.GeoDataFrame) -> pd.DataFrame:
    paths = parameters["study"]
    support = parameters["support"]
    sites = pd.read_csv(rooted(paths["site_table"]))[
        ["site_code", "site_name", "forest_group"]
    ]
    boundaries = gpd.read_file(rooted(paths["site_boundaries"])).to_crs(
        paths["analysis_crs"]
    )
    genus_paths = sorted(rooted(paths["genus_tiles"]).glob("*.tif"))
    if not genus_paths:
        raise FileNotFoundError("No tree-genus raster tiles are available")
    conifer_classes = np.asarray([0, 1, 2, 5], dtype=np.uint8)
    broadleaf_classes = np.asarray([3, 4, 6], dtype=np.uint8)
    cells = int(support["unit_m"] // support["lidar_cell_m"])
    frames: list[pd.DataFrame] = []
    with rasterio.open(rooted(paths["reference_raster"])) as reference:
        for site in sites.itertuples(index=False):
            boundary = boundaries[boundaries["SITENAME"].eq(site.site_name)]
            if boundary.empty:
                raise RuntimeError(f"Missing Natura 2000 boundary for {site.site_name}")
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
            classes = read_genus_window(genus_paths, window, reference)
            conifer = np.isin(classes, conifer_classes)
            broadleaf = np.isin(classes, broadleaf_classes)
            tree = conifer | broadleaf
            expected = conifer if site.forest_group == "conifer" else broadleaf
            forest = site_mask & tree
            tree_count = patch_sum(forest.astype(np.int16), cells)
            group_count = patch_sum((site_mask & expected).astype(np.int16), cells)
            tree_fraction = tree_count / float(cells * cells)
            group_fraction = np.zeros_like(tree_fraction, dtype=np.float32)
            np.divide(group_count, tree_count, out=group_fraction, where=tree_count > 0)
            candidate = (
                (tree_fraction >= float(support["minimum_tree_fraction"]))
                & (
                    group_fraction
                    >= float(support["minimum_group_fraction_among_trees"])
                )
            )
            rows, columns = np.where(candidate)
            pixel_rows = rows * cells + (cells - 1) / 2
            pixel_columns = columns * cells + (cells - 1) / 2
            x = transform.c + (pixel_columns + 0.5) * transform.a
            y = transform.f + (pixel_rows + 0.5) * transform.e
            frame = pd.DataFrame(
                {
                    "site_code": site.site_code,
                    "site_name": site.site_name,
                    "forest_group": site.forest_group,
                    "rd_x": x.astype(np.float64),
                    "rd_y": y.astype(np.float64),
                    "tree_fraction": tree_fraction[candidate].astype(np.float32),
                    "group_fraction": group_fraction[candidate].astype(np.float32),
                    "forest_pixel_mask": patch_bitmask(forest, cells)[candidate],
                }
            )
            frame = assign_ahn3_year(frame, envelopes)
            minimum_year = int(parameters["ahn3"]["minimum_tessera_year"])
            maximum_year = int(parameters["ahn3"]["maximum_tessera_year"])
            frame = frame[
                frame["ahn3_year"].between(minimum_year, maximum_year)
            ].reset_index(drop=True)
            log(
                f"candidate grid {site.site_code}: {len(frame):,} units in "
                f"AHN3 years {minimum_year}-{maximum_year}"
            )
            frames.append(frame)
    output = pd.concat(frames, ignore_index=True)
    output.insert(0, "candidate_id", np.arange(len(output), dtype=np.int64))
    return output


def assign_ahn3_year(
    frame: pd.DataFrame, envelopes: gpd.GeoDataFrame, chunk_size: int = 20000
) -> pd.DataFrame:
    years = np.zeros(len(frame), dtype=np.int16)
    for start in range(0, len(frame), chunk_size):
        stop = min(start + chunk_size, len(frame))
        local = frame.iloc[start:stop]
        points = gpd.GeoDataFrame(
            {"position": np.arange(start, stop, dtype=np.int64)},
            geometry=gpd.points_from_xy(local["rd_x"], local["rd_y"]),
            crs="EPSG:28992",
        )
        joined = gpd.sjoin(
            points,
            envelopes[["year", "DATE", "geometry"]],
            how="left",
            predicate="within",
        )
        joined = joined.sort_values(["position", "DATE"], na_position="first")
        joined = joined.drop_duplicates("position", keep="last")
        valid = joined["year"].notna()
        years[joined.loc[valid, "position"].to_numpy(dtype=np.int64)] = joined.loc[
            valid, "year"
        ].to_numpy(dtype=np.int16)
    result = frame.copy()
    result["ahn3_year"] = years
    return result


def record_inventory(record_id: int, work: Path) -> dict[str, dict[str, Any]]:
    path = work / f"zenodo_{record_id}.json"
    if not path.exists():
        response = requests.get(
            f"https://zenodo.org/api/records/{record_id}", timeout=120
        )
        response.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(response.json(), indent=2) + "\n")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {item["key"]: item for item in payload["files"]}


def checked_download(url: str, destination: Path, expected_size: int | None = None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and (expected_size is None or destination.stat().st_size == expected_size):
        return
    subprocess.run(
        [
            "curl",
            "--location",
            "--fail",
            "--retry",
            "12",
            "--retry-all-errors",
            "--retry-delay",
            "5",
            "--continue-at",
            "-",
            "--output",
            str(destination),
            url,
        ],
        check=True,
    )
    if expected_size is not None and destination.stat().st_size != expected_size:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"Wrong download size for {destination.name}")


def metric_source(
    record_id: int,
    filename: str,
    work: Path,
    local: Path | None = None,
) -> tuple[Path, bool]:
    if local is not None and local.exists():
        return local, False
    inventory = record_inventory(record_id, work)
    if filename not in inventory:
        raise RuntimeError(f"{filename} is absent from Zenodo record {record_id}")
    item = inventory[filename]
    path = work / "downloads" / filename
    log(f"downloading {filename} ({int(item['size']) / 1e9:.2f} GB)")
    checked_download(item["links"]["self"], path, int(item["size"]))
    return path, True


def sample_raster(frame: pd.DataFrame, path: Path) -> tuple[np.ndarray, np.ndarray]:
    sums = np.zeros(len(frame), dtype=np.float64)
    counts = np.zeros(len(frame), dtype=np.int16)
    with rasterio.open(path) as dataset:
        if dataset.crs is None:
            raise RuntimeError(f"Raster has no CRS: {path}")
        for _, positions in frame.groupby("site_code", sort=False).indices.items():
            positions = np.asarray(positions, dtype=np.int64)
            local_frame = frame.iloc[positions].reset_index(drop=True)
            owner, x, y = point_arrays(local_frame)
            row, column = rowcol(dataset.transform, x, y)
            row = np.asarray(row, dtype=np.int64)
            column = np.asarray(column, dtype=np.int64)
            inside = (
                (row >= 0)
                & (column >= 0)
                & (row < dataset.height)
                & (column < dataset.width)
            )
            if not inside.any():
                continue
            row = row[inside]
            column = column[inside]
            local_owner = owner[inside]
            row_start, row_stop = int(row.min()), int(row.max()) + 1
            column_start, column_stop = int(column.min()), int(column.max()) + 1
            window = Window(
                column_start,
                row_start,
                column_stop - column_start,
                row_stop - row_start,
            )
            array = dataset.read(1, window=window, masked=False)
            values = array[row - row_start, column - column_start].astype(np.float64)
            valid = np.isfinite(values)
            if dataset.nodata is not None:
                valid &= values != float(dataset.nodata)
            global_owner = positions[local_owner[valid]]
            np.add.at(sums, global_owner, values[valid])
            np.add.at(counts, global_owner, 1)
    means = np.full(len(frame), np.nan, dtype=np.float32)
    valid = counts > 0
    means[valid] = (sums[valid] / counts[valid]).astype(np.float32)
    return means, counts


def acquisition_year(values: np.ndarray) -> np.ndarray:
    """Convert AHN date encodings (YYYY, YYYYMMDD, or YYYYDDD) to years."""
    raw = np.asarray(values, dtype=np.int64)
    years = np.zeros(raw.shape, dtype=np.int16)
    calendar_dates = raw >= 10_000_000
    ordinal_dates = (raw >= 1_000_000) & ~calendar_dates
    plain_years = (raw >= 1900) & (raw <= 2100)
    years[calendar_dates] = (raw[calendar_dates] // 10_000).astype(np.int16)
    years[ordinal_dates] = (raw[ordinal_dates] // 1_000).astype(np.int16)
    years[plain_years] = raw[plain_years].astype(np.int16)
    return years


def sample_year_raster(frame: pd.DataFrame, path: Path) -> tuple[np.ndarray, np.ndarray]:
    years = np.zeros(len(frame), dtype=np.int16)
    fractions = np.zeros(len(frame), dtype=np.float32)
    with rasterio.open(path) as dataset:
        for _, positions in frame.groupby("site_code", sort=False).indices.items():
            positions = np.asarray(positions, dtype=np.int64)
            local_frame = frame.iloc[positions].reset_index(drop=True)
            owner, x, y = point_arrays(local_frame)
            row, column = rowcol(dataset.transform, x, y)
            row = np.asarray(row, dtype=np.int64)
            column = np.asarray(column, dtype=np.int64)
            inside = (
                (row >= 0)
                & (column >= 0)
                & (row < dataset.height)
                & (column < dataset.width)
            )
            values = np.zeros(len(owner), dtype=np.int32)
            if inside.any():
                rr, cc = row[inside], column[inside]
                row_start, row_stop = int(rr.min()), int(rr.max()) + 1
                column_start, column_stop = int(cc.min()), int(cc.max()) + 1
                array = dataset.read(
                    1,
                    window=Window(
                        column_start,
                        row_start,
                        column_stop - column_start,
                        row_stop - row_start,
                    ),
                    masked=False,
                )
                values[inside] = array[rr - row_start, cc - column_start].astype(
                    np.int32
                )
            values = acquisition_year(values)
            for local_position in range(len(local_frame)):
                observed = values[(owner == local_position) & (values > 0)]
                if len(observed) == 0:
                    continue
                unique, count = np.unique(observed, return_counts=True)
                winner = int(np.argmax(count))
                years[positions[local_position]] = int(unique[winner])
                fractions[positions[local_position]] = float(count[winner] / len(observed))
    return years, fractions


def spatially_balanced_cap(
    frame: pd.DataFrame, maximum: int, seed: int
) -> pd.DataFrame:
    if len(frame) <= maximum:
        return frame.copy()
    rng = np.random.default_rng(seed)
    groups: dict[str, list[int]] = {}
    for block, positions in frame.groupby("sampling_block", sort=True).indices.items():
        values = np.asarray(positions, dtype=np.int64)
        rng.shuffle(values)
        groups[str(block)] = values.tolist()
    selected: list[int] = []
    while len(selected) < maximum:
        active = False
        for block in sorted(groups):
            if groups[block]:
                selected.append(groups[block].pop())
                active = True
                if len(selected) == maximum:
                    break
        if not active:
            break
    return frame.iloc[np.sort(np.asarray(selected, dtype=np.int64))].copy()


def prepare_targets(parameters: dict[str, Any]) -> pd.DataFrame:
    outputs = parameters["outputs"]
    cohort_path = rooted(outputs["cohort"])
    if cohort_path.exists():
        log(f"resuming completed target cohort: {cohort_path}")
        return pd.read_parquet(cohort_path)
    work = rooted(outputs["work_directory"])
    work.mkdir(parents=True, exist_ok=True)
    candidate_path = work / "ahn3_candidate_units.parquet"
    envelope_path = work / "AHN3_pseudo_omhullen.gpkg"
    if not envelope_path.exists():
        log("downloading AHN3 acquisition envelopes")
        checked_download(parameters["ahn3"]["acquisition_envelopes_url"], envelope_path)
    envelopes = gpd.read_file(envelope_path).to_crs(parameters["study"]["analysis_crs"])
    envelopes["year"] = pd.to_numeric(envelopes["year"], errors="coerce")
    envelopes["DATE"] = pd.to_numeric(envelopes["DATE"], errors="coerce")
    envelopes = envelopes[envelopes["year"].notna()].copy()
    if candidate_path.exists():
        frame = pd.read_parquet(candidate_path)
        log(f"resuming {len(frame):,} AHN3 candidate units")
    else:
        frame = candidate_units(parameters, envelopes)
        atomic_parquet(candidate_path, frame)
    support = parameters["support"]
    denominator = popcount(frame["forest_pixel_mask"])
    ahn3_files = {
        **parameters["ahn3"]["screening_metrics"],
        **parameters["ahn3"]["metrics"],
    }
    for outcome, filename in ahn3_files.items():
        value_column = f"ahn3_{outcome}"
        fraction_column = f"ahn3_{outcome}_valid_fraction"
        if value_column in frame.columns and fraction_column in frame.columns:
            log(f"resuming sampled {value_column}")
            continue
        path, temporary = metric_source(
            int(parameters["ahn3"]["record_id"]), filename, work
        )
        try:
            values, counts = sample_raster(frame, path)
            frame[value_column] = values
            frame[fraction_column] = counts / denominator
            atomic_parquet(candidate_path, frame)
            log(f"sampled {value_column}: {np.isfinite(values).sum():,} units")
        finally:
            if temporary:
                path.unlink(missing_ok=True)
                gc.collect()
    required = [
        f"ahn3_{name}_valid_fraction"
        for name in parameters["ahn3"]["metrics"]
    ] + ["ahn3_mean_height_m_valid_fraction"]
    valid = np.logical_and.reduce(
        [
            frame[column].to_numpy()
            >= float(support["minimum_valid_metric_fraction"])
            for column in required
        ]
    )
    valid &= frame["ahn3_mean_height_m"].to_numpy() > float(
        support["minimum_baseline_mean_height_m"]
    )
    valid &= frame["ahn3_pulse_penetration"].to_numpy() < float(
        support["maximum_baseline_pulse_penetration"]
    )
    frame = frame[valid].reset_index(drop=True)
    block_m = int(support["sampling_block_m"])
    frame["sampling_block"] = (
        np.floor(frame["rd_x"] / block_m).astype(np.int32).astype(str)
        + "_"
        + np.floor(frame["rd_y"] / block_m).astype(np.int32).astype(str)
    )
    retained: list[pd.DataFrame] = []
    for site_code, local in frame.groupby("site_code", sort=False):
        local = spatially_balanced_cap(
            local.reset_index(drop=True),
            int(support["maximum_units_per_site"]),
            int(support["sampling_seed"]) + sum(map(ord, str(site_code))),
        )
        if len(local) < int(support["minimum_units_per_site"]):
            log(f"dropping {site_code}: only {len(local):,} AHN3 baseline units")
            continue
        retained.append(local)
        log(f"AHN3 baseline {site_code}: {len(local):,} retained units")
    frame = pd.concat(retained, ignore_index=True)
    frame.insert(0, "unit_id", np.arange(len(frame), dtype=np.int64))
    paired_checkpoint = work / "paired_targets_checkpoint.parquet"
    if paired_checkpoint.exists():
        checkpoint = pd.read_parquet(paired_checkpoint)
        if checkpoint["candidate_id"].tolist() != frame["candidate_id"].tolist():
            raise RuntimeError(
                "The paired-target checkpoint does not match the current AHN3 "
                "baseline cohort; remove the checkpoint before changing parameters"
            )
        frame = checkpoint
        log(f"resuming {len(frame):,} paired target units")
    if "ahn4_year" not in frame or "ahn4_year_fraction" not in frame:
        ahn4_year, ahn4_year_fraction = sample_year_raster(
            frame, rooted(parameters["ahn4"]["flighttime"])
        )
        frame["ahn4_year"] = ahn4_year
        frame["ahn4_year_fraction"] = ahn4_year_fraction
        atomic_parquet(paired_checkpoint, frame)
    denominator = popcount(frame["forest_pixel_mask"])
    for outcome, filename in parameters["ahn4"]["metrics"].items():
        value_column = f"ahn4_{outcome}"
        fraction_column = f"ahn4_{outcome}_valid_fraction"
        if value_column in frame.columns and fraction_column in frame.columns:
            continue
        local_path = parameters["ahn4"].get("local_metrics", {}).get(outcome)
        path, temporary = metric_source(
            int(parameters["ahn4"]["record_id"]),
            filename,
            work,
            rooted(local_path) if local_path else None,
        )
        try:
            values, counts = sample_raster(frame, path)
            frame[value_column] = values
            frame[fraction_column] = counts / denominator
            atomic_parquet(paired_checkpoint, frame)
            log(f"sampled {value_column}: {np.isfinite(values).sum():,} units")
        finally:
            if temporary:
                path.unlink(missing_ok=True)
                gc.collect()
    valid = frame["ahn4_year_fraction"].to_numpy() >= float(
        support["minimum_single_year_fraction"]
    )
    valid &= frame["ahn4_year"].between(2020, 2022).to_numpy()
    for outcome in parameters["ahn4"]["metrics"]:
        valid &= (
            frame[f"ahn4_{outcome}_valid_fraction"].to_numpy()
            >= float(support["minimum_valid_metric_fraction"])
        )
    frame = frame[valid].reset_index(drop=True)
    frame["unit_id"] = np.arange(len(frame), dtype=np.int64)
    atomic_parquet(cohort_path, frame)
    inventory = (
        frame.groupby(["site_code", "site_name", "forest_group"], as_index=False)
        .agg(
            units=("unit_id", "size"),
            ahn3_year_min=("ahn3_year", "min"),
            ahn3_year_max=("ahn3_year", "max"),
            ahn4_year_min=("ahn4_year", "min"),
            ahn4_year_max=("ahn4_year", "max"),
        )
        .sort_values(["forest_group", "site_code"])
    )
    atomic_csv(rooted(outputs["site_inventory"]), inventory)
    log(f"paired target cohort complete: {len(frame):,} units across {frame.site_code.nunique()} sites")
    return frame


def version_path(version: str) -> str:
    cleaned = version.removeprefix("v")
    major, *minor = cleaned.split(".")
    return f"v{major}" if not minor or minor[0] == "0" else f"v{major}.{minor[0]}"


def tile_from_world(lon: float, lat: float) -> tuple[float, float]:
    return (
        round(math.floor(lon * 10) / 10 + 0.05, 2),
        round(math.floor(lat * 10) / 10 + 0.05, 2),
    )


def tile_name(tile: tuple[float, float]) -> str:
    return f"grid_{tile[0]:.2f}_{tile[1]:.2f}"


def s3_url(key: str) -> str:
    prefix = "s3://tessera-embeddings/"
    if not key.startswith(prefix):
        raise RuntimeError(f"Unexpected TESSERA path: {key}")
    return f"{S3_ROOT}/{key[len(prefix):]}"


def ensure_registry(version: str) -> tuple[Path, Path]:
    cache = ROOT / "data/external/geotessera_cache" / version_path(version)
    manifest = cache / "manifest.parquet"
    landmasks = cache / "landmasks.parquet"
    for name, path in (("manifest.parquet", manifest), ("landmasks.parquet", landmasks)):
        if not path.exists():
            checked_download(f"{S3_ROOT}/{version_path(version)}/{name}", path)
    return manifest, landmasks


def lookup_embeddings(
    manifest: Path,
    tiles: list[tuple[float, float]],
    year: int,
    version: str,
    variant: str,
) -> pd.DataFrame:
    import pyarrow.parquet as pq

    lon_values = sorted({round(lon * 100) for lon, _ in tiles})
    lat_values = sorted({round(lat * 100) for _, lat in tiles})
    frame = pq.read_table(
        manifest,
        filters=[
            ("year", "=", year),
            ("lon_i", "in", lon_values),
            ("lat_i", "in", lat_values),
        ],
    ).to_pandas()
    wanted = {(round(lon * 100), round(lat * 100)) for lon, lat in tiles}
    frame = frame[
        [
            (int(lon), int(lat)) in wanted
            for lon, lat in zip(frame["lon_i"], frame["lat_i"], strict=True)
        ]
    ]
    return frame[
        frame["version"].astype(str).eq(version)
        & frame["variant"].astype(str).eq(variant)
    ].sort_values(["lon", "lat"])


def lookup_landmasks(
    manifest: Path, tiles: list[tuple[float, float]]
) -> pd.DataFrame:
    import pyarrow.parquet as pq

    lon_values = sorted({round(lon * 100) for lon, _ in tiles})
    lat_values = sorted({round(lat * 100) for _, lat in tiles})
    frame = pq.read_table(
        manifest,
        filters=[("lon_i", "in", lon_values), ("lat_i", "in", lat_values)],
    ).to_pandas()
    wanted = {(round(lon * 100), round(lat * 100)) for lon, lat in tiles}
    return frame[
        [
            (int(lon), int(lat)) in wanted
            for lon, lat in zip(frame["lon_i"], frame["lat_i"], strict=True)
        ]
    ].sort_values(["lon", "lat"])


def embedding_points(rows: pd.DataFrame) -> dict[str, np.ndarray]:
    owner, x, y = point_arrays(rows)
    local_rows = np.repeat(np.arange(5, dtype=np.int8), 5)
    local_columns = np.tile(np.arange(5, dtype=np.int8), 5)
    masks = rows["forest_pixel_mask"].to_numpy(dtype=np.uint32)[:, None]
    keep = ((masks & (1 << np.arange(25, dtype=np.uint32))[None, :]) > 0).reshape(-1)
    return {
        "owner": owner,
        "rd_x": x,
        "rd_y": y,
        "offset_x": np.tile(local_columns - 2, len(rows))[keep].astype(np.float32),
        "offset_y": np.tile(2 - local_rows, len(rows))[keep].astype(np.float32),
        "year": np.repeat(rows["tessera_year"].to_numpy(dtype=np.int16), 25)[keep],
    }


def reduce_embedding_chunk(
    owners: np.ndarray, embeddings: np.ndarray
) -> dict[str, np.ndarray]:
    order = np.argsort(owners, kind="stable")
    owners = owners[order]
    embeddings = embeddings[order].astype(np.float32)
    unique, starts, count = np.unique(owners, return_index=True, return_counts=True)
    return {
        "row_id": unique.astype(np.int64),
        "count": count.astype(np.int16),
        "sum": np.add.reduceat(embeddings, starts, axis=0).astype(np.float32),
    }


def build_tessera_rows(cohort: pd.DataFrame, output: Path) -> pd.DataFrame:
    if output.exists():
        return pd.read_parquet(output)
    shared = [
        "unit_id",
        "site_code",
        "site_name",
        "forest_group",
        "rd_x",
        "rd_y",
        "forest_pixel_mask",
    ]
    ahn3 = cohort[shared].copy()
    ahn3["epoch"] = "ahn3"
    ahn3["tessera_year"] = cohort["ahn3_year"].to_numpy(dtype=np.int16)
    ahn4 = cohort[shared].copy()
    ahn4["epoch"] = "ahn4"
    ahn4["tessera_year"] = cohort["ahn4_year"].to_numpy(dtype=np.int16)
    rows = pd.concat([ahn3, ahn4], ignore_index=True)
    rows.insert(0, "row_id", np.arange(len(rows), dtype=np.int64))
    atomic_parquet(output, rows)
    return rows


def acquire_tessera(parameters: dict[str, Any], cohort: pd.DataFrame) -> pd.DataFrame:
    outputs = parameters["outputs"]
    output = rooted(outputs["tessera_features"])
    if output.exists():
        log(f"resuming completed TESSERA features: {output}")
        return pd.read_parquet(output)
    rows_path = rooted(outputs["tessera_rows"])
    rows = build_tessera_rows(cohort, rows_path)
    points = embedding_points(rows)
    transformer = Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)
    longitude, latitude = transformer.transform(points["rd_x"], points["rd_y"])
    tiles = [
        tile_from_world(float(lon), float(lat))
        for lon, lat in zip(longitude, latitude, strict=True)
    ]
    groups = sorted(
        {
            (int(year), tile)
            for year, tile in zip(points["year"], tiles, strict=True)
        }
    )
    lookup = {group: index for index, group in enumerate(groups)}
    group_code = np.asarray(
        [
            lookup[(int(year), tile)]
            for year, tile in zip(points["year"], tiles, strict=True)
        ],
        dtype=np.int16,
    )
    settings = parameters["tessera"]
    version = str(settings["dataset_version"])
    variant = str(settings["dataset_variant"])
    manifest, landmasks = ensure_registry(version)
    inventory: dict[tuple[int, tuple[float, float]], tuple[Any, Any]] = {}
    for year in sorted(set(points["year"].astype(int))):
        year_tiles = sorted({tile for local_year, tile in groups if local_year == year})
        embedding_rows = lookup_embeddings(
            manifest, year_tiles, year, version, variant
        )
        mask_rows = lookup_landmasks(landmasks, year_tiles)
        found = {
            (round(float(row.lon), 2), round(float(row.lat), 2))
            for row in embedding_rows.itertuples(index=False)
        }
        found_masks = {
            (round(float(row.lon), 2), round(float(row.lat), 2))
            for row in mask_rows.itertuples(index=False)
        }
        wanted = set(year_tiles)
        if found != wanted or found_masks != wanted:
            raise RuntimeError(
                f"Incomplete TESSERA coverage for {year}; "
                f"embeddings={sorted(wanted-found)}, masks={sorted(wanted-found_masks)}"
            )
        by_tile = {
            (round(float(row.lon), 2), round(float(row.lat), 2)): row
            for row in embedding_rows.itertuples(index=False)
        }
        masks_by_tile = {
            (round(float(row.lon), 2), round(float(row.lat), 2)): row
            for row in mask_rows.itertuples(index=False)
        }
        for tile in year_tiles:
            inventory[(year, tile)] = (by_tile[tile], masks_by_tile[tile])
    work = rooted(outputs["work_directory"])
    chunk_root = work / "tessera_chunks"
    stream_root = work / "tessera_stream"
    chunk_root.mkdir(parents=True, exist_ok=True)
    stream_root.mkdir(parents=True, exist_ok=True)
    for code, (year, tile) in enumerate(groups):
        name = f"{year}_{tile_name(tile)}"
        chunk = chunk_root / f"{name}.npz"
        if chunk.exists():
            log(f"TESSERA [{code + 1}/{len(groups)}] resume {name}")
            continue
        indices = np.flatnonzero(group_code == code)
        embedding_row, mask_row = inventory[(year, tile)]
        directory = stream_root / EMBEDDINGS_SUBDIR / str(year) / tile_name(tile)
        paths = {
            "embedding": directory / f"{tile_name(tile)}.npy",
            "scales": directory / f"{tile_name(tile)}_scales.npy",
            "landmask": stream_root
            / LANDMASKS_SUBDIR
            / f"{tile_name(tile)}.tiff",
        }
        sources = {
            "embedding": str(embedding_row.grid_path),
            "scales": str(embedding_row.scales_path),
            "landmask": str(mask_row.key),
        }
        sizes = {
            "embedding": int(embedding_row.grid_size),
            "scales": int(embedding_row.scales_size),
            "landmask": int(mask_row.file_size),
        }
        required = sum(sizes.values()) + 512 * 1024 * 1024
        if shutil.disk_usage(ROOT).free < required:
            raise RuntimeError(f"Insufficient free space for TESSERA tile {name}")
        log(f"TESSERA [{code + 1}/{len(groups)}] acquire {name}")
        try:
            for kind in ("embedding", "scales", "landmask"):
                checked_download(s3_url(sources[kind]), paths[kind], sizes[kind])
            quantized = np.load(paths["embedding"], mmap_mode="r")
            scales = np.load(paths["scales"], mmap_mode="r")
            with rasterio.open(paths["landmask"]) as dataset:
                landmask = dataset.read(1)
                to_tile = Transformer.from_crs(
                    "EPSG:28992", dataset.crs, always_xy=True
                )
                x, y = to_tile.transform(
                    points["rd_x"][indices], points["rd_y"][indices]
                )
                rr, cc = rowcol(dataset.transform, x, y)
            rr = np.asarray(rr, dtype=np.int64)
            cc = np.asarray(cc, dtype=np.int64)
            inside = (
                (rr >= 0)
                & (cc >= 0)
                & (rr < quantized.shape[0])
                & (cc < quantized.shape[1])
            )
            local = np.flatnonzero(inside)
            usable = (
                np.isfinite(scales[rr[local], cc[local]])
                & (scales[rr[local], cc[local]] > 0)
                & (landmask[rr[local], cc[local]] > 0)
            )
            local = local[usable]
            embeddings = (
                quantized[rr[local], cc[local]].astype(np.float32)
                * scales[rr[local], cc[local]].astype(np.float32)[:, None]
            )
            finite = np.isfinite(embeddings).all(axis=1)
            local = local[finite]
            embeddings = embeddings[finite]
            reduced = reduce_embedding_chunk(
                points["owner"][indices[local]], embeddings
            )
            temporary = chunk.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, **reduced)
            temporary.replace(chunk)
        finally:
            for path in paths.values():
                path.unlink(missing_ok=True)
            gc.collect()
    count = np.zeros(len(rows), dtype=np.int16)
    sums = np.zeros((len(rows), int(settings["dimensions"])), dtype=np.float32)
    for year, tile in groups:
        with np.load(chunk_root / f"{year}_{tile_name(tile)}.npz") as chunk:
            ids = chunk["row_id"].astype(np.int64)
            count[ids] += chunk["count"]
            sums[ids] += chunk["sum"]
    mean = np.full_like(sums, np.nan)
    usable = count > 0
    mean[usable] = sums[usable] / count[usable, None]
    result: dict[str, Any] = {
        "row_id": rows["row_id"].to_numpy(dtype=np.int64),
        "tessera_valid_pixel_count": count,
        "tessera_context_valid": count >= int(settings["minimum_valid_pixels"]),
    }
    for dimension in range(int(settings["dimensions"])):
        result[f"tessera_mean_{dimension:03d}"] = mean[:, dimension]
    frame = pd.DataFrame(result)
    atomic_parquet(output, frame)
    shutil.rmtree(chunk_root, ignore_errors=True)
    shutil.rmtree(stream_root, ignore_errors=True)
    log(
        f"TESSERA features complete: {frame.tessera_context_valid.sum():,}/"
        f"{len(frame):,} epoch rows"
    )
    return frame


def balanced_folds(blocks: pd.Series, folds: int, seed: int) -> dict[str, int]:
    counts = blocks.value_counts().rename_axis("block").reset_index(name="rows")
    rng = np.random.default_rng(seed)
    counts["tie"] = rng.random(len(counts))
    counts = counts.sort_values(["rows", "tie"], ascending=[False, True])
    totals = np.zeros(folds, dtype=np.int64)
    assignment: dict[str, int] = {}
    for row in counts.itertuples(index=False):
        fold = int(np.argmin(totals))
        assignment[str(row.block)] = fold
        totals[fold] += int(row.rows)
    return assignment


def regression_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(observed) & np.isfinite(predicted)
    observed = observed[finite]
    predicted = predicted[finite]
    if len(observed) < 3:
        return {"rows": len(observed), "rmse": np.nan, "mae": np.nan, "r2": np.nan, "bias": np.nan, "spearman": np.nan}
    # Rank correlation is undefined for constant baselines such as zero change.
    # Record that case as missing instead of emitting one warning per fold.
    if np.ptp(observed) == 0 or np.ptp(predicted) == 0:
        correlation = np.nan
    else:
        correlation = spearmanr(observed, predicted).statistic
    return {
        "rows": int(len(observed)),
        "rmse": float(np.sqrt(mean_squared_error(observed, predicted))),
        "mae": float(mean_absolute_error(observed, predicted)),
        "r2": float(r2_score(observed, predicted)),
        "bias": float(np.mean(predicted - observed)),
        "spearman": float(correlation),
    }


def evaluate(parameters: dict[str, Any], cohort: pd.DataFrame, features: pd.DataFrame) -> None:
    outputs = parameters["outputs"]
    predictions_path = rooted(outputs["predictions"])
    summary_path = rooted(outputs["summary"])
    if predictions_path.exists() and summary_path.exists():
        log("resuming completed model predictions")
        prediction_frame = pd.read_parquet(predictions_path)
        summary = pd.read_csv(summary_path)
        bootstrap_path = rooted(outputs["bootstrap"])
        if bootstrap_path.exists():
            bootstrap = pd.read_csv(bootstrap_path)
        else:
            bootstrap = bootstrap_penalties(prediction_frame, parameters)
            atomic_csv(bootstrap_path, bootstrap)
        write_report(parameters, cohort, prediction_frame, summary, bootstrap)
        log(f"temporal evaluation complete: {len(prediction_frame):,} predictions")
        return
    rows = pd.read_parquet(rooted(outputs["tessera_rows"]))
    merged = rows.merge(features, on="row_id", validate="one_to_one")
    feature_columns = [
        f"tessera_mean_{dimension:03d}"
        for dimension in range(int(parameters["tessera"]["dimensions"]))
    ]
    valid_rows = merged["tessera_context_valid"].astype(bool)
    valid_rows &= np.isfinite(merged[feature_columns]).all(axis=1)
    valid_units = set(
        merged.loc[valid_rows]
        .groupby("unit_id")["epoch"]
        .nunique()
        .loc[lambda value: value.eq(2)]
        .index
    )
    cohort = cohort[cohort["unit_id"].isin(valid_units)].reset_index(drop=True)
    merged = merged[merged["unit_id"].isin(valid_units)].copy()
    ahn3 = merged[merged["epoch"].eq("ahn3")].set_index("unit_id")
    ahn4 = merged[merged["epoch"].eq("ahn4")].set_index("unit_id")
    validation = parameters["validation"]
    block_m = int(validation["fold_block_m"])
    cohort["fold_block"] = (
        cohort["site_code"].astype(str)
        + "_"
        + np.floor(cohort["rd_x"] / block_m).astype(np.int32).astype(str)
        + "_"
        + np.floor(cohort["rd_y"] / block_m).astype(np.int32).astype(str)
    )
    cohort["spatial_fold"] = -1
    for site_index, (site_code, positions) in enumerate(
        cohort.groupby("site_code", sort=True).indices.items()
    ):
        positions = np.asarray(positions, dtype=np.int64)
        assignment = balanced_folds(
            cohort.iloc[positions]["fold_block"],
            int(validation["folds"]),
            int(validation["fold_seed"]) + site_index,
        )
        cohort.loc[positions, "spatial_fold"] = (
            cohort.iloc[positions]["fold_block"].map(assignment).to_numpy()
        )
    targets = list(parameters["ahn3"]["metrics"])
    predictions: list[pd.DataFrame] = []
    fold_records: list[dict[str, Any]] = []
    for site_code, site_frame in cohort.groupby("site_code", sort=True):
        site_ids = site_frame["unit_id"].to_numpy(dtype=np.int64)
        coordinates = site_frame[["rd_x", "rd_y"]].to_numpy(dtype=np.float64)
        site_folds = site_frame["spatial_fold"].to_numpy(dtype=np.int8)
        x3_all = ahn3.loc[site_ids, feature_columns].to_numpy(dtype=np.float64)
        x4_all = ahn4.loc[site_ids, feature_columns].to_numpy(dtype=np.float64)
        for fold in range(int(validation["folds"])):
            test = np.flatnonzero(site_folds == fold)
            candidate = np.flatnonzero(site_folds != fold)
            if len(test) < int(validation["minimum_test_rows"]):
                log(f"skip {site_code} fold {fold}: only {len(test)} test units")
                continue
            distance, _ = cKDTree(coordinates[test]).query(coordinates[candidate], k=1)
            train = candidate[distance >= float(validation["exclusion_buffer_m"])]
            if len(train) < int(validation["minimum_train_rows"]):
                log(f"skip {site_code} fold {fold}: only {len(train)} buffered train units")
                continue
            test_ids = site_ids[test]
            for outcome in targets:
                y3 = site_frame[f"ahn3_{outcome}"].to_numpy(dtype=np.float64)
                y4 = site_frame[f"ahn4_{outcome}"].to_numpy(dtype=np.float64)
                baseline_model = make_pipeline(
                    StandardScaler(), Ridge(alpha=float(validation["ridge_alpha"]))
                )
                baseline_model.fit(x3_all[train], y3[train])
                predicted3 = baseline_model.predict(x3_all[test])
                predicted4 = baseline_model.predict(x4_all[test])
                contemporary_model = make_pipeline(
                    StandardScaler(), Ridge(alpha=float(validation["ridge_alpha"]))
                )
                contemporary_model.fit(x4_all[train], y4[train])
                contemporary4 = contemporary_model.predict(x4_all[test])
                observed_change = y4[test] - y3[test]
                predicted_change = predicted4 - predicted3
                methods = {
                    "same_time_ahn3": (y3[test], predicted3),
                    "temporal_transfer_ahn4": (y4[test], predicted4),
                    "contemporary_ahn4": (y4[test], contemporary4),
                    "no_change_ahn4": (y4[test], y3[test]),
                    "predicted_change": (observed_change, predicted_change),
                    "zero_change": (observed_change, np.zeros_like(observed_change)),
                }
                for comparison, (observed, predicted) in methods.items():
                    fold_records.append(
                        {
                            "site_code": site_code,
                            "site_name": site_frame.iloc[0]["site_name"],
                            "forest_group": site_frame.iloc[0]["forest_group"],
                            "outcome": outcome,
                            "fold": fold,
                            "comparison": comparison,
                            "train_rows": int(len(train)),
                            "test_rows": int(len(test)),
                            **regression_metrics(observed, predicted),
                        }
                    )
                predictions.append(
                    pd.DataFrame(
                        {
                            "unit_id": test_ids,
                            "site_code": site_code,
                            "forest_group": site_frame.iloc[0]["forest_group"],
                            "fold": fold,
                            "fold_block": site_frame.iloc[test]["fold_block"].to_numpy(),
                            "outcome": outcome,
                            "observed_ahn3": y3[test],
                            "observed_ahn4": y4[test],
                            "predicted_ahn3": predicted3,
                            "predicted_ahn4_temporal": predicted4,
                            "predicted_ahn4_contemporary": contemporary4,
                            "observed_change": observed_change,
                            "predicted_change": predicted_change,
                        }
                    )
                )
        log(f"evaluated temporal transfer for {site_code}")
    prediction_frame = pd.concat(predictions, ignore_index=True)
    fold_frame = pd.DataFrame(fold_records)
    atomic_parquet(predictions_path, prediction_frame)
    atomic_csv(rooted(outputs["fold_metrics"]), fold_frame)
    site_metrics = (
        fold_frame.groupby(["site_code", "outcome", "comparison"], as_index=False)[
            ["rmse", "mae", "r2", "bias", "spearman"]
        ]
        .mean()
    )
    summary = (
        site_metrics.groupby(["outcome", "comparison"], as_index=False)
        .agg(
            sites=("site_code", "nunique"),
            median_rmse=("rmse", "median"),
            mean_rmse=("rmse", "mean"),
            median_r2=("r2", "median"),
            median_bias=("bias", "median"),
            median_spearman=("spearman", "median"),
        )
        .sort_values(["outcome", "comparison"])
    )
    atomic_csv(summary_path, summary)
    bootstrap = bootstrap_penalties(prediction_frame, parameters)
    atomic_csv(rooted(outputs["bootstrap"]), bootstrap)
    write_report(parameters, cohort, prediction_frame, summary, bootstrap)
    log(f"temporal evaluation complete: {len(prediction_frame):,} predictions")


def bootstrap_penalties(
    predictions: pd.DataFrame, parameters: dict[str, Any]
) -> pd.DataFrame:
    settings = parameters["validation"]
    rng = np.random.default_rng(int(settings["bootstrap_seed"]))
    repetitions = int(settings["bootstrap_replicates"])
    records: list[dict[str, Any]] = []
    comparisons = {
        "temporal_minus_same_time": (
            "predicted_ahn4_temporal",
            "predicted_ahn3",
            "observed_ahn4",
            "observed_ahn3",
        ),
        "temporal_minus_contemporary": (
            "predicted_ahn4_temporal",
            "predicted_ahn4_contemporary",
            "observed_ahn4",
            "observed_ahn4",
        ),
        "temporal_minus_no_change": (
            "predicted_ahn4_temporal",
            "observed_ahn3",
            "observed_ahn4",
            "observed_ahn4",
        ),
        "predicted_change_minus_zero": (
            "predicted_change",
            None,
            "observed_change",
            "observed_change",
        ),
    }
    for outcome, outcome_frame in predictions.groupby("outcome", sort=True):
        site_draws: dict[str, list[np.ndarray]] = {
            key: [] for key in comparisons
        }
        for _, local in outcome_frame.groupby("site_code", sort=True):
            block_codes, blocks = pd.factorize(local["fold_block"], sort=True)
            block_count = len(blocks)
            row_count = np.bincount(block_codes, minlength=block_count).astype(
                np.float64
            )
            sampled_blocks = rng.integers(
                0, block_count, size=(repetitions, block_count)
            )
            sampled_rows = row_count[sampled_blocks].sum(axis=1)
            for name, (
                first,
                second,
                first_observed,
                second_observed,
            ) in comparisons.items():
                first_error = (
                    local[first].to_numpy(dtype=np.float64)
                    - local[first_observed].to_numpy(dtype=np.float64)
                )
                if second is None:
                    second_error = local[second_observed].to_numpy(dtype=np.float64)
                else:
                    second_error = (
                        local[second].to_numpy(dtype=np.float64)
                        - local[second_observed].to_numpy(dtype=np.float64)
                    )
                first_sse = np.bincount(
                    block_codes,
                    weights=np.square(first_error),
                    minlength=block_count,
                )
                second_sse = np.bincount(
                    block_codes,
                    weights=np.square(second_error),
                    minlength=block_count,
                )
                first_rmse = np.sqrt(
                    first_sse[sampled_blocks].sum(axis=1) / sampled_rows
                )
                second_rmse = np.sqrt(
                    second_sse[sampled_blocks].sum(axis=1) / sampled_rows
                )
                site_draws[name].append(first_rmse - second_rmse)
        for name, values in site_draws.items():
            draws = np.median(np.vstack(values), axis=0)
            lower, median, upper = np.quantile(draws, [0.025, 0.5, 0.975])
            records.append(
                {
                    "outcome": outcome,
                    "contrast": name,
                    "median_delta_rmse": median,
                    "lower_95": lower,
                    "upper_95": upper,
                    "replicates": repetitions,
                }
            )
    return pd.DataFrame(records)


def write_report(
    parameters: dict[str, Any],
    cohort: pd.DataFrame,
    predictions: pd.DataFrame,
    summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> None:
    path = rooted(parameters["outputs"]["report"])
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# AHN3-to-AHN4 temporal transfer",
        "",
        f"The AHN3-defined cohort contains {len(cohort):,} paired 50 m units across "
        f"{cohort.site_code.nunique()} forests. AHN4 outcomes were not used for maturity selection.",
        f"Held-out predictions cover {predictions.unit_id.nunique():,} units across "
        f"{predictions[['site_code', 'fold']].drop_duplicates().shape[0]} site-folds. "
        "One fold was omitted because fewer than 150 buffered training units remained.",
        "",
        "## Median performance across sites",
        "",
        "```text",
        summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "```",
        "",
        "## Paired spatial-block bootstrap",
        "",
        "Positive delta-RMSE values mean that temporal transfer was worse than the comparator.",
        "",
        "```text",
        bootstrap.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "```",
        "",
        "## Interpretation boundary",
        "",
        "The temporal penalty includes ecological change and differences between AHN3 and AHN4 "
        "acquisition conditions. It is not a pure estimate of biological change.",
        "",
    ]
    temporary = path.with_suffix(".tmp.md")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)


def write_manifest(parameters: dict[str, Any]) -> None:
    outputs = parameters["outputs"]
    paths = [
        PARAMETERS,
        Path(__file__),
        rooted(outputs["cohort"]),
        rooted(outputs["tessera_rows"]),
        rooted(outputs["tessera_features"]),
        rooted(outputs["predictions"]),
        rooted(outputs["fold_metrics"]),
        rooted(outputs["summary"]),
        rooted(outputs["bootstrap"]),
        rooted(outputs["site_inventory"]),
        rooted(outputs["report"]),
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(f"Cannot freeze incomplete run; missing {missing}")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "design": "AHN3-defined paired-unit temporal transfer to AHN4",
        "files": {str(path.relative_to(ROOT)): sha256(path) for path in paths},
    }
    atomic_json(rooted(outputs["manifest"]), manifest)


def check_inputs(parameters: dict[str, Any]) -> None:
    required = [
        rooted(parameters["study"]["site_table"]),
        rooted(parameters["study"]["site_boundaries"]),
        rooted(parameters["study"]["reference_raster"]),
        rooted(parameters["ahn4"]["flighttime"]),
    ]
    required.extend(rooted(parameters["study"]["genus_tiles"]).glob("*.tif"))
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required local inputs: {missing}")
    if shutil.disk_usage(ROOT).free < 2_000_000_000:
        raise RuntimeError("At least 2 GB of free disk space is required")
    log(f"input check passed with {shutil.disk_usage(ROOT).free / 1e9:.1f} GB free")


def run(stage: str) -> None:
    parameters = load_parameters()
    check_inputs(parameters)
    if stage == "check":
        return
    cohort = prepare_targets(parameters)
    if stage == "prepare":
        return
    features = acquire_tessera(parameters, cohort)
    if stage == "tessera":
        return
    evaluate(parameters, cohort, features)
    write_manifest(parameters)
    log("temporal-transfer workflow finished successfully")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["check", "prepare", "tessera", "all"],
        default="all",
        help="Stop after the selected resumable stage.",
    )
    args = parser.parse_args()
    run(args.stage)


if __name__ == "__main__":
    main()
