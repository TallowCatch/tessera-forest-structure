#!/usr/bin/env python3
"""Acquire, subset, filter, and freeze the 2024 GEDI V3 sample.

The script deliberately processes one protected GEDI HDF5 granule at a time. Each full
granule is checksum-verified, clipped to the exact NEON Priority-1 polygons, converted to
a compact Parquet subset, and deleted before the next download begins.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import rasterio
import requests
import yaml
from pyproj import Transformer
from rasterio.windows import Window
from shapely import contains_xy
from shapely.geometry import Point, box, shape


ROOT = Path(__file__).resolve().parents[1]
CMR_GRANULES_URL = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"
NLCD_WCS_URL = (
    "https://dmsdata.cr.usgs.gov/geoserver/"
    "mrlc_Land-Cover-Native_conus_year_data/ows"
)
NLCD_COVERAGE = (
    "mrlc_Land-Cover-Native_conus_year_data:"
    "Land-Cover-Native_conus_year_data"
)
NLCD_CRS = "EPSG:5070"
NLCD_PIXEL_SIZE_M = 30.0
NLCD_ORIGIN_LEFT = -2_415_585.0
NLCD_ORIGIN_TOP = 3_314_805.0
HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
DOWNLOAD_MARGIN_BYTES = 512 * 1024 * 1024

HDF_FIELDS = {
    "shot_number": "shot_number",
    "delta_time": "delta_time",
    "longitude": "lon_lowestmode",
    "latitude": "lat_lowestmode",
    "fhd_normal": "fhd_normal",
    "l2b_quality_flag_rel3": "l2b_quality_flag_rel3",
    "degrade_flag": "geolocation/degrade_flag",
    "sensitivity": "sensitivity",
    "solar_elevation": "geolocation/solar_elevation",
    "beam_id": "beam",
    "cover": "cover",
    "pai": "pai",
    "elev_lowestmode": "elev_lowestmode",
    "elev_highestreturn": "elev_highestreturn",
    "digital_elevation_model": "digital_elevation_model",
    "landsat_treecover": "land_cover_data/landsat_treecover",
    "worldcover_class": "land_cover_data/worldcover_class",
}

FLOAT_FILL_FIELDS = {
    "fhd_normal",
    "cover",
    "pai",
    "digital_elevation_model",
    "landsat_treecover",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_hashes(path: Path) -> dict[str, str]:
    sha256 = hashlib.sha256()
    sha512 = hashlib.sha512()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            sha256.update(chunk)
            sha512.update(chunk)
    return {"sha256": sha256.hexdigest(), "sha512": sha512.hexdigest()}


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_project_config() -> dict[str, Any]:
    with (ROOT / "configs/project.yaml").open(encoding="utf-8") as source:
        return yaml.safe_load(source)


def load_boundaries(path: Path, requested_sites: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    features = {
        feature["properties"]["siteID"]: feature
        for feature in document["features"]
        if feature["properties"]["siteID"] in requested_sites
    }
    missing = sorted(set(requested_sites) - set(features))
    if missing:
        raise RuntimeError(f"Missing site boundaries: {', '.join(missing)}")
    polygons = {site: shape(feature["geometry"]) for site, feature in features.items()}
    return features, polygons


def cmr_checksum(umm: dict[str, Any], filename: str) -> tuple[int, str, str]:
    records = umm["DataGranule"]["ArchiveAndDistributionInformation"]
    record = next(item for item in records if item["Name"] == filename)
    checksum = record["Checksum"]
    return int(record["SizeInBytes"]), checksum["Algorithm"].upper(), checksum["Value"].lower()


def get_attribute(umm: dict[str, Any], name: str) -> str:
    for attribute in umm.get("AdditionalAttributes", []):
        if attribute.get("Name") == name:
            return str(attribute["Values"][0])
    raise RuntimeError(f"CMR granule is missing {name}")


def concise_granule(item: dict[str, Any], site_id: str) -> dict[str, Any]:
    umm = item["umm"]
    granule_id = umm["GranuleUR"]
    filename = f"{granule_id}.h5"
    urls = [
        value["URL"]
        for value in umm.get("RelatedUrls", [])
        if value.get("Type") == "GET DATA" and value["URL"].endswith(".h5")
    ]
    if len(urls) != 1:
        raise RuntimeError(f"Expected one HDF5 download URL for {granule_id}, found {len(urls)}")
    size, algorithm, checksum = cmr_checksum(umm, filename)
    if algorithm != "SHA-512":
        raise RuntimeError(f"Expected SHA-512 for {granule_id}, received {algorithm}")
    date_range = umm["TemporalExtent"]["RangeDateTime"]
    return {
        "granule_id": granule_id,
        "cmr_concept_id": item["meta"]["concept-id"],
        "cmr_revision_id": item["meta"]["revision-id"],
        "cmr_revision_date": item["meta"]["revision-date"],
        "collection_concept_id": item["meta"]["collection-concept-id"],
        "download_url": urls[0],
        "filename": filename,
        "size_bytes": size,
        "checksum_algorithm": algorithm,
        "checksum": checksum,
        "beginning_datetime": date_range["BeginningDateTime"],
        "ending_datetime": date_range["EndingDateTime"],
        "orbit": int(umm["OrbitCalculatedSpatialDomains"][0]["BeginOrbitNumber"]),
        "reference_ground_track": int(get_attribute(umm, "Reference_Ground_Track")),
        "producer_doi": get_attribute(umm, "identifier_product_doi"),
        "sites": [site_id],
    }


def query_cmr(
    session: requests.Session,
    collection_id: str,
    year: int,
    polygons: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    queries: dict[str, Any] = {}
    temporal = f"{year}-01-01T00:00:00Z,{year}-12-31T23:59:59Z"
    for site_id, polygon in polygons.items():
        bbox = ",".join(f"{value:.12f}" for value in polygon.bounds)
        params = {
            "collection_concept_id": collection_id,
            "bounding_box": bbox,
            "temporal": temporal,
            "page_size": 2000,
        }
        response = session.get(CMR_GRANULES_URL, params=params, timeout=90)
        response.raise_for_status()
        items = response.json()["items"]
        hits = int(response.headers.get("CMR-Hits", len(items)))
        if hits != len(items):
            raise RuntimeError(f"CMR pagination would truncate {site_id}: {hits} hits, {len(items)} returned")
        queries[site_id] = {"bounding_box": bbox, "temporal": temporal, "hits": hits}
        for item in items:
            entry = concise_granule(item, site_id)
            existing = merged.get(entry["granule_id"])
            if existing is None:
                merged[entry["granule_id"]] = entry
            else:
                if existing["checksum"] != entry["checksum"]:
                    raise RuntimeError(f"Inconsistent CMR checksum for {entry['granule_id']}")
                existing["sites"].append(site_id)
    granules = sorted(merged.values(), key=lambda item: (item["beginning_datetime"], item["granule_id"]))
    return granules, queries


def remaining_download_bytes(path: Path, expected_size: int) -> int:
    partial = path.with_suffix(path.suffix + ".part")
    existing = partial.stat().st_size if partial.exists() else 0
    if existing > expected_size:
        partial.unlink()
        existing = 0
    return expected_size - existing


def ensure_disk_space(path: Path, required_bytes: int) -> None:
    free = shutil.disk_usage(path.parent).free
    minimum = required_bytes + DOWNLOAD_MARGIN_BYTES
    if free < minimum:
        raise RuntimeError(
            f"Insufficient free storage: {free / 1e9:.2f} GB available, "
            f"{minimum / 1e9:.2f} GB required"
        )


def download_granule(session: requests.Session, entry: dict[str, Any], destination: Path) -> dict[str, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    if destination.exists():
        hashes = file_hashes(destination)
        if destination.stat().st_size == entry["size_bytes"] and hashes["sha512"] == entry["checksum"]:
            return hashes
        destination.unlink()

    ensure_disk_space(destination, remaining_download_bytes(destination, entry["size_bytes"]))
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    mode = "ab" if offset else "wb"

    with session.get(entry["download_url"], headers=headers, stream=True, timeout=(30, 300)) as response:
        if response.status_code in {401, 403}:
            raise RuntimeError("Earthdata rejected the credentials in ~/.netrc")
        if offset and response.status_code != 206:
            offset = 0
            mode = "wb"
        response.raise_for_status()
        with partial.open(mode) as output:
            for chunk in response.iter_content(8 * 1024 * 1024):
                if chunk:
                    output.write(chunk)
            output.flush()
            os.fsync(output.fileno())

    if partial.stat().st_size != entry["size_bytes"]:
        raise RuntimeError(
            f"Wrong size for {entry['granule_id']}: "
            f"{partial.stat().st_size} != {entry['size_bytes']}"
        )
    with partial.open("rb") as source:
        signature = source.read(8)
    if signature != HDF5_SIGNATURE:
        raise RuntimeError(f"Downloaded object is not HDF5: {entry['granule_id']}")
    hashes = file_hashes(partial)
    if hashes["sha512"] != entry["checksum"]:
        raise RuntimeError(f"SHA-512 verification failed for {entry['granule_id']}")
    partial.replace(destination)
    return hashes


def decode_attribute(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def selected_values(group: h5py.Group, dataset_path: str, indices: np.ndarray) -> np.ndarray:
    dataset = group[dataset_path]
    values = dataset[indices]
    if dataset_path.split("/")[-1] in FLOAT_FILL_FIELDS:
        values = values.astype("float64", copy=False)
        fill = dataset.attrs.get("_FillValue")
        if fill is not None:
            values[np.isclose(values, float(fill))] = np.nan
    return values


def parse_granule_tokens(granule_id: str) -> None:
    if not re.fullmatch(r"GEDI02_B_\d{13}_O\d{5}_\d{2}_T\d{5}_\d{2}_\d{3}_\d{2}_V003", granule_id):
        raise RuntimeError(f"Unexpected GEDI V3 granule identifier: {granule_id}")


def extract_granule(
    hdf_path: Path,
    entry: dict[str, Any],
    features: dict[str, Any],
    polygons: dict[str, Any],
    source_hashes: dict[str, str],
) -> pd.DataFrame:
    parse_granule_tokens(entry["granule_id"])
    frames: list[pd.DataFrame] = []
    with h5py.File(hdf_path, "r") as hdf:
        if decode_attribute(hdf.attrs.get("short_name", "")) != "GEDI_L2B":
            raise RuntimeError(f"Not a GEDI L2B file: {hdf_path}")
        beams = sorted(key for key in hdf.keys() if key.startswith("BEAM"))
        if not beams:
            raise RuntimeError(f"No GEDI beam groups in {hdf_path}")
        for beam_name in beams:
            group = hdf[beam_name]
            missing = [path for path in HDF_FIELDS.values() if path not in group]
            if missing:
                raise RuntimeError(f"{entry['granule_id']} {beam_name} is missing: {missing}")
            description = decode_attribute(group.attrs.get("description", ""))
            beam_power = "full" if description.strip().lower() == "full power beam" else "coverage"
            longitude = group[HDF_FIELDS["longitude"]][:]
            latitude = group[HDF_FIELDS["latitude"]][:]
            for site_id in entry["sites"]:
                inside = contains_xy(polygons[site_id], longitude, latitude)
                indices = np.flatnonzero(inside)
                if indices.size == 0:
                    continue
                values = {
                    column: selected_values(group, dataset_path, indices)
                    for column, dataset_path in HDF_FIELDS.items()
                }
                frame = pd.DataFrame(values)
                frame.insert(0, "site_id", site_id)
                frame.insert(1, "flight_box_id", features[site_id]["properties"]["flightbxID"])
                frame.insert(2, "granule_id", entry["granule_id"])
                frame.insert(3, "cmr_concept_id", entry["cmr_concept_id"])
                frame.insert(4, "collection_concept_id", entry["collection_concept_id"])
                frame.insert(5, "product", "GEDI02_B")
                frame.insert(6, "product_version", "003")
                frame.insert(7, "acquisition_datetime", entry["beginning_datetime"])
                frame.insert(8, "orbit", entry["orbit"])
                frame.insert(9, "reference_ground_track", entry["reference_ground_track"])
                frame.insert(10, "beam_name", beam_name)
                frame.insert(11, "beam_power", beam_power)
                frame["source_hdf_sha512"] = source_hashes["sha512"]
                frames.append(frame)
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    result["acquisition_datetime"] = pd.to_datetime(result["acquisition_datetime"], utc=True)
    return result


def write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def process_granules(
    session: requests.Session,
    granules: list[dict[str, Any]],
    features: dict[str, Any],
    polygons: dict[str, Any],
    staging_dir: Path,
    interim_dir: Path,
    progress_path: Path,
) -> list[Path]:
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {"granules": {}}
    outputs: list[Path] = []
    for position, entry in enumerate(granules, start=1):
        granule_id = entry["granule_id"]
        output = interim_dir / f"{granule_id}.parquet"
        outputs.append(output)
        if output.exists():
            print(f"[{position}/{len(granules)}] reuse {output.name}", flush=True)
            continue
        print(f"[{position}/{len(granules)}] download {granule_id}", flush=True)
        hdf_path = staging_dir / entry["filename"]
        source_hashes = download_granule(session, entry, hdf_path)
        try:
            print(f"[{position}/{len(granules)}] extract {', '.join(entry['sites'])}", flush=True)
            subset = extract_granule(hdf_path, entry, features, polygons, source_hashes)
            write_parquet_atomic(subset, output)
            progress["granules"][granule_id] = {
                "completed_at": utc_now(),
                "source_size_bytes": entry["size_bytes"],
                "source_sha256": source_hashes["sha256"],
                "source_sha512": source_hashes["sha512"],
                "subset_rows": int(len(subset)),
                "subset_sha256": file_hashes(output)["sha256"],
                "sites": entry["sites"],
            }
            progress["updated_at"] = utc_now()
            json_dump(progress_path, progress)
        finally:
            if hdf_path.exists():
                hdf_path.unlink()
        print(f"[{position}/{len(granules)}] retained {len(subset):,} rows; source deleted", flush=True)
    return outputs


def aligned_nlcd_grid(polygon: Any, transformer: Transformer, radius_m: float) -> tuple[float, float, float, float, int, int]:
    minx, miny, maxx, maxy = transformer.transform_bounds(*polygon.bounds, densify_pts=21)
    minx -= radius_m
    miny -= radius_m
    maxx += radius_m
    maxy += radius_m
    first_col = math.floor((minx - NLCD_ORIGIN_LEFT) / NLCD_PIXEL_SIZE_M)
    last_col = math.ceil((maxx - NLCD_ORIGIN_LEFT) / NLCD_PIXEL_SIZE_M)
    first_row = math.floor((NLCD_ORIGIN_TOP - maxy) / NLCD_PIXEL_SIZE_M)
    last_row = math.ceil((NLCD_ORIGIN_TOP - miny) / NLCD_PIXEL_SIZE_M)
    left = NLCD_ORIGIN_LEFT + first_col * NLCD_PIXEL_SIZE_M
    right = NLCD_ORIGIN_LEFT + last_col * NLCD_PIXEL_SIZE_M
    top = NLCD_ORIGIN_TOP - first_row * NLCD_PIXEL_SIZE_M
    bottom = NLCD_ORIGIN_TOP - last_row * NLCD_PIXEL_SIZE_M
    return left, bottom, right, top, last_col - first_col, last_row - first_row


def acquire_nlcd_raster(
    session: requests.Session,
    site_id: str,
    polygon: Any,
    year: int,
    path: Path,
) -> dict[str, Any]:
    transformer = Transformer.from_crs("EPSG:4326", NLCD_CRS, always_xy=True)
    left, bottom, right, top, width, height = aligned_nlcd_grid(polygon, transformer, 15.0)
    params = {
        "service": "WCS",
        "version": "1.0.0",
        "request": "GetCoverage",
        "coverage": NLCD_COVERAGE,
        "crs": NLCD_CRS,
        "response_crs": NLCD_CRS,
        "bbox": f"{left},{bottom},{right},{top}",
        "width": width,
        "height": height,
        "format": "GeoTIFF",
        "time": f"{year}-01-01T00:00:00.000Z",
    }
    if not path.exists():
        print(f"Acquire Annual NLCD {year} raw classes for {site_id}", flush=True)
        response = session.get(NLCD_WCS_URL, params=params, timeout=180)
        response.raise_for_status()
        if not response.content.startswith((b"II*\x00", b"MM\x00*")):
            raise RuntimeError(f"Annual NLCD WCS did not return a GeoTIFF for {site_id}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(response.content)
        temporary.replace(path)
    with rasterio.open(path) as dataset:
        if dataset.count != 1 or dataset.dtypes[0] != "uint8":
            raise RuntimeError(f"Unexpected Annual NLCD raster type for {site_id}: {dataset.profile}")
        if dataset.crs.to_epsg() != 5070:
            raise RuntimeError(f"Unexpected Annual NLCD CRS for {site_id}: {dataset.crs}")
        if not np.allclose(dataset.res, (30.0, 30.0), atol=1e-5):
            raise RuntimeError(f"Unexpected Annual NLCD resolution for {site_id}: {dataset.res}")
        unique = sorted(int(value) for value in np.unique(dataset.read(1)))
    return {
        "site_id": site_id,
        "path": str(path.relative_to(ROOT)),
        "sha256": file_hashes(path)["sha256"],
        "size_bytes": path.stat().st_size,
        "request_url": requests.Request("GET", NLCD_WCS_URL, params=params).prepare().url,
        "bbox_epsg5070": [left, bottom, right, top],
        "width": width,
        "height": height,
        "unique_classes": unique,
    }


def footprint_forest_fraction(
    dataset: rasterio.io.DatasetReader,
    x: float,
    y: float,
    forest_classes: set[int],
    radius_m: float,
) -> tuple[int, float, float]:
    center_value = int(next(dataset.sample([(x, y)]))[0])
    circle = Point(x, y).buffer(radius_m, quad_segs=32)
    minx, miny, maxx, maxy = circle.bounds
    row_a, col_a = dataset.index(minx, maxy)
    row_b, col_b = dataset.index(maxx, miny)
    row_min = max(0, min(row_a, row_b))
    row_max = min(dataset.height - 1, max(row_a, row_b))
    col_min = max(0, min(col_a, col_b))
    col_max = min(dataset.width - 1, max(col_a, col_b))
    forest_area = 0.0
    valid_area = 0.0
    for row in range(row_min, row_max + 1):
        for col in range(col_min, col_max + 1):
            left, bottom, right, top = rasterio.windows.bounds(Window(col, row, 1, 1), dataset.transform)
            area = circle.intersection(box(left, bottom, right, top)).area
            if area <= 0:
                continue
            value = int(dataset.read(1, window=Window(col, row, 1, 1))[0, 0])
            if dataset.nodata is None or value != int(dataset.nodata):
                valid_area += area
                if value in forest_classes:
                    forest_area += area
    return center_value, forest_area / circle.area, valid_area / circle.area


def add_nlcd_fields(
    candidates: pd.DataFrame,
    raster_paths: dict[str, Path],
    forest_classes: set[int],
    radius_m: float,
) -> pd.DataFrame:
    result = candidates.copy()
    result["x_epsg5070"] = np.nan
    result["y_epsg5070"] = np.nan
    result["nlcd_center_class"] = pd.Series(pd.NA, index=result.index, dtype="Int16")
    result["nlcd_forest_fraction"] = np.nan
    result["nlcd_valid_fraction"] = np.nan
    transformer = Transformer.from_crs("EPSG:4326", NLCD_CRS, always_xy=True)
    for site_id, raster_path in raster_paths.items():
        mask = result["site_id"].eq(site_id)
        indices = result.index[mask]
        x_values, y_values = transformer.transform(
            result.loc[indices, "longitude"].to_numpy(),
            result.loc[indices, "latitude"].to_numpy(),
        )
        result.loc[indices, "x_epsg5070"] = x_values
        result.loc[indices, "y_epsg5070"] = y_values
        center_values: list[int] = []
        forest_fractions: list[float] = []
        valid_fractions: list[float] = []
        with rasterio.open(raster_path) as dataset:
            for x, y in zip(x_values, y_values, strict=True):
                center, forest, valid = footprint_forest_fraction(
                    dataset, x, y, forest_classes, radius_m
                )
                center_values.append(center)
                forest_fractions.append(forest)
                valid_fractions.append(valid)
        result.loc[indices, "nlcd_center_class"] = center_values
        result.loc[indices, "nlcd_forest_fraction"] = forest_fractions
        result.loc[indices, "nlcd_valid_fraction"] = valid_fractions
    return result


def add_filter_fields(candidates: pd.DataFrame, minimum_forest_fraction: float) -> pd.DataFrame:
    result = candidates.copy()
    result["is_finite_fhd"] = np.isfinite(result["fhd_normal"]) & result["fhd_normal"].ge(0)
    result["passes_l2b_quality"] = result["l2b_quality_flag_rel3"].eq(1)
    result["passes_geolocation"] = result["degrade_flag"].eq(0)
    result["is_full_power"] = result["beam_power"].eq("full")
    result["is_nighttime"] = result["solar_elevation"].lt(0)
    result["passes_forest_mask"] = (
        result["nlcd_valid_fraction"].ge(0.999)
        & result["nlcd_forest_fraction"].ge(minimum_forest_fraction)
    )
    result["passes_primary_filters"] = (
        result["is_finite_fhd"]
        & result["passes_l2b_quality"]
        & result["passes_geolocation"]
        & result["is_full_power"]
        & result["passes_forest_mask"]
    )
    return result


def exclusion_table(candidates: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for site_id, frame in candidates.groupby("site_id", sort=True):
        remaining = pd.Series(True, index=frame.index)
        records.append({"site_id": site_id, "stage": "exact_priority1_candidates", "count": int(remaining.sum())})
        stages = [
            ("excluded_invalid_fhd", frame["is_finite_fhd"]),
            ("excluded_l2b_quality", frame["passes_l2b_quality"]),
            ("excluded_degraded_geolocation", frame["passes_geolocation"]),
            ("excluded_not_full_power", frame["is_full_power"]),
            ("excluded_forest_fraction", frame["passes_forest_mask"]),
        ]
        for label, passes in stages:
            excluded = remaining & ~passes
            records.append({"site_id": site_id, "stage": label, "count": int(excluded.sum())})
            remaining &= passes
        records.append({"site_id": site_id, "stage": "retained_primary", "count": int(remaining.sum())})
    return pd.DataFrame(records)


def inventory_table(candidates: pd.DataFrame) -> pd.DataFrame:
    inventory = candidates.copy()
    inventory["acquisition_date"] = inventory["acquisition_datetime"].dt.date.astype(str)
    columns = ["site_id", "acquisition_date", "beam_name", "beam_power", "is_nighttime"]
    return (
        inventory.groupby(columns, dropna=False, observed=True)
        .agg(candidate_count=("shot_number", "size"), primary_count=("passes_primary_filters", "sum"))
        .reset_index()
        .sort_values(columns)
    )


def freeze_outputs(
    candidates: pd.DataFrame,
    granules: list[dict[str, Any]],
    cmr_queries: dict[str, Any],
    nlcd_records: list[dict[str, Any]],
    config: dict[str, Any],
    boundary_path: Path,
    candidate_path: Path,
    primary_path: Path,
    manifest_path: Path,
    exclusion_path: Path,
    inventory_path: Path,
    force: bool,
) -> dict[str, Any]:
    protected = [candidate_path, primary_path, manifest_path, exclusion_path, inventory_path]
    if any(path.exists() for path in protected) and not force:
        existing = [str(path.relative_to(ROOT)) for path in protected if path.exists()]
        raise RuntimeError(f"Frozen outputs already exist; refusing to overwrite: {existing}")

    freeze_basis = {
        "collection_concept_id": config["target"]["cmr_collection_concept_id"],
        "product_version": config["target"]["product_version"],
        "year": config["target"]["primary_year"],
        "project_config_sha256": file_hashes(ROOT / "configs/project.yaml")["sha256"],
        "acquisition_script_sha256": file_hashes(Path(__file__).resolve())["sha256"],
        "boundary_sha256": file_hashes(boundary_path)["sha256"],
        "source_granules": [
            {"granule_id": item["granule_id"], "sha512": item["checksum"], "sites": item["sites"]}
            for item in granules
        ],
        "nlcd": [{"site_id": item["site_id"], "sha256": item["sha256"]} for item in nlcd_records],
        "filters": {
            "fhd": "finite_and_nonnegative",
            "l2b_quality_flag_rel3": 1,
            "degrade_flag": 0,
            "beam_power": "HDF_group_description_equals_Full_power_beam",
            "forest_classes": config["forest_mask"]["forest_classes"],
            "minimum_footprint_forest_fraction": config["forest_mask"]["minimum_footprint_forest_fraction"],
            "minimum_nlcd_valid_fraction": 0.999,
            "gedi_footprint_radius_m": config["target"]["footprint_diameter_m_approx"] / 2,
        },
    }
    freeze_basis_sha256 = canonical_hash(freeze_basis)
    freeze_id = f"gedi-v3-2024-{freeze_basis_sha256[:12]}"
    frozen = candidates.copy()
    frozen.insert(0, "freeze_id", freeze_id)
    primary = frozen[frozen["passes_primary_filters"]].copy()
    if primary.empty:
        raise RuntimeError("Primary sample is empty")
    if primary["site_id"].nunique() != len(config["sites"]["development"]) + 1:
        raise RuntimeError("At least one configured site has no primary GEDI shots")
    if primary.duplicated(["site_id", "shot_number"]).any():
        raise RuntimeError("Duplicate site/shot_number records in the primary sample")

    for path in protected:
        path.parent.mkdir(parents=True, exist_ok=True)
    write_parquet_atomic(frozen, candidate_path)
    write_parquet_atomic(primary, primary_path)
    exclusions = exclusion_table(frozen)
    inventory = inventory_table(frozen)
    exclusions.to_csv(exclusion_path, index=False)
    inventory.to_csv(inventory_path, index=False)

    site_summary: dict[str, Any] = {}
    for site_id, frame in primary.groupby("site_id", sort=True):
        grid_cells = set(
            zip(
                np.floor(frame["x_epsg5070"] / 1000).astype(int),
                np.floor(frame["y_epsg5070"] / 1000).astype(int),
                strict=True,
            )
        )
        site_summary[site_id] = {
            "candidate_rows": int(frozen["site_id"].eq(site_id).sum()),
            "primary_rows": int(len(frame)),
            "dates": int(frame["acquisition_datetime"].dt.date.nunique()),
            "orbits": int(frame["orbit"].nunique()),
            "beams": sorted(frame["beam_name"].unique().tolist()),
            "occupied_1km_cells": len(grid_cells),
            "fhd_min": float(frame["fhd_normal"].min()),
            "fhd_max": float(frame["fhd_normal"].max()),
            "fhd_mean": float(frame["fhd_normal"].mean()),
        }
    gate_passed = all(summary["occupied_1km_cells"] >= 3 for summary in site_summary.values())
    manifest = {
        "freeze_id": freeze_id,
        "created_at": utc_now(),
        "status": "frozen" if gate_passed else "frozen_gate_failed",
        "freeze_basis_sha256": freeze_basis_sha256,
        "freeze_basis": freeze_basis,
        "cmr_query_url": CMR_GRANULES_URL,
        "cmr_queries": cmr_queries,
        "granule_count": len(granules),
        "source_granules": granules,
        "nlcd_rasters": nlcd_records,
        "outputs": {
            "all_candidates": {
                "path": str(candidate_path.relative_to(ROOT)),
                "rows": int(len(frozen)),
                "sha256": file_hashes(candidate_path)["sha256"],
            },
            "primary_sample": {
                "path": str(primary_path.relative_to(ROOT)),
                "rows": int(len(primary)),
                "sha256": file_hashes(primary_path)["sha256"],
            },
            "exclusion_counts": {
                "path": str(exclusion_path.relative_to(ROOT)),
                "sha256": file_hashes(exclusion_path)["sha256"],
            },
            "inventory": {
                "path": str(inventory_path.relative_to(ROOT)),
                "sha256": file_hashes(inventory_path)["sha256"],
            },
        },
        "site_summary": site_summary,
        "phase1_gate": {
            "rule": "each_site_has_at_least_3_occupied_1km_cells",
            "passed": gate_passed,
        },
    }
    json_dump(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Replace existing final frozen outputs")
    parser.add_argument(
        "--sites",
        nargs="+",
        default=["SOAP", "TEAK", "BART"],
        choices=["SOAP", "TEAK", "BART"],
        help="Sites to include; the publication sample requires all three",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if set(args.sites) != {"SOAP", "TEAK", "BART"}:
        raise RuntimeError("A frozen publication sample requires SOAP, TEAK, and BART")
    config = load_project_config()
    year = int(config["target"]["primary_year"])
    boundary_path = ROOT / "metadata/neon_aop_priority1_2024.geojson"
    features, polygons = load_boundaries(boundary_path, args.sites)

    session = requests.Session()
    session.headers.update({"User-Agent": "tessera-gedi-fhd/0.1 (research sample acquisition)"})
    print("Query exact GEDI02_B.003 inventory from NASA CMR", flush=True)
    granules, cmr_queries = query_cmr(
        session,
        config["target"]["cmr_collection_concept_id"],
        year,
        polygons,
    )
    if not granules:
        raise RuntimeError("CMR returned no GEDI V3 granules")
    json_dump(
        ROOT / "metadata/gedi_v3_2024_cmr_inventory.json",
        {"queried_at": utc_now(), "queries": cmr_queries, "granules": granules},
    )
    print(f"CMR returned {len(granules)} unique source granules", flush=True)

    nlcd_records: list[dict[str, Any]] = []
    raster_paths: dict[str, Path] = {}
    for site_id in args.sites:
        raster_path = ROOT / f"data/external/nlcd/Annual_NLCD_Land_Cover_{year}_{site_id}.tif"
        raster_paths[site_id] = raster_path
        nlcd_records.append(acquire_nlcd_raster(session, site_id, polygons[site_id], year, raster_path))

    parquet_paths = process_granules(
        session,
        granules,
        features,
        polygons,
        ROOT / "data/raw/gedi_v3/staging",
        ROOT / "data/interim/gedi_v3_exact_p1",
        ROOT / "metadata/gedi_v3_2024_acquisition_progress.json",
    )
    frames = [pd.read_parquet(path) for path in parquet_paths if path.exists()]
    if len(frames) != len(parquet_paths):
        raise RuntimeError("One or more per-granule subsets are missing")
    candidates = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
    candidates = candidates.drop_duplicates(["site_id", "shot_number"], keep="first")
    candidates = candidates.sort_values(["site_id", "acquisition_datetime", "beam_name", "shot_number"])
    candidates = add_nlcd_fields(
        candidates,
        raster_paths,
        set(config["forest_mask"]["forest_classes"]),
        config["target"]["footprint_diameter_m_approx"] / 2,
    )
    candidates = add_filter_fields(
        candidates,
        float(config["forest_mask"]["minimum_footprint_forest_fraction"]),
    )
    manifest = freeze_outputs(
        candidates,
        granules,
        cmr_queries,
        nlcd_records,
        config,
        boundary_path,
        ROOT / "data/processed/gedi_v3_2024_all_candidates.parquet",
        ROOT / "data/processed/gedi_v3_2024_primary_forest.parquet",
        ROOT / "metadata/gedi_v3_2024_freeze.json",
        ROOT / "metadata/gedi_v3_2024_exclusion_counts.csv",
        ROOT / "metadata/gedi_v3_2024_inventory.csv",
        args.force,
    )
    if list((ROOT / "data/raw/gedi_v3/staging").glob("*.h5")):
        raise RuntimeError("One or more full GEDI source granules remain in staging")
    print(json.dumps({"freeze_id": manifest["freeze_id"], "status": manifest["status"], "site_summary": manifest["site_summary"]}, indent=2))
    return 0 if manifest["phase1_gate"]["passed"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, requests.RequestException, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
