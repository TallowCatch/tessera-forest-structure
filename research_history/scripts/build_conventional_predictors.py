#!/usr/bin/env python3
"""Build streamed topography and Sentinel predictors for development or locked BART."""

from __future__ import annotations

import argparse
import calendar
import hashlib
import importlib.metadata
import json
import math
import sys
import time
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import rasterio
import yaml
from pyproj import Transformer
from pystac_client.exceptions import APIError
from rasterio.merge import merge
from rasterio.transform import array_bounds, rowcol
from rasterio.warp import Resampling, calculate_default_transform, reproject


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/project.yaml"
PHASE2_FREEZE_PATH = ROOT / "metadata/tessera_phase2_alignment_freeze.json"
PHASE7_FREEZE_PATH = ROOT / "metadata/phase7_tessera_alignment_freeze.json"
DEVELOPMENT_PATH = ROOT / "data/processed/tessera_aligned_development_soap_teak.parquet"
LOCKED_BART_PATH = ROOT / "data/processed/tessera_aligned_locked_bart.parquet"
EXPANSION_PATH = ROOT / "data/processed/phase7_tessera_aligned_expansion.parquet"

PARTITION_PATHS = {
    "development": {
        "input": DEVELOPMENT_PATH,
        "output": ROOT / "data/processed/phase3_conventional_predictors_development.parquet",
        "inventory": ROOT / "metadata/phase3_conventional_stac_inventory.csv",
        "manifest": ROOT / "metadata/phase3_conventional_predictor_freeze.json",
        "expected_sites": {"SOAP", "TEAK"},
        "id_prefix": "conventional-dev",
    },
    "bart": {
        "input": LOCKED_BART_PATH,
        "output": ROOT / "data/processed/phase4_conventional_predictors_bart.parquet",
        "inventory": ROOT / "metadata/phase4_bart_stac_inventory.csv",
        "manifest": ROOT / "metadata/phase4_bart_predictor_freeze.json",
        "expected_sites": {"BART"},
        "id_prefix": "conventional-bart",
    },
    "expansion": {
        "input": EXPANSION_PATH,
        "output": ROOT / "data/processed/phase8_conventional_predictors_expansion.parquet",
        "inventory": ROOT / "metadata/phase8_conventional_stac_inventory.csv",
        "manifest": ROOT / "metadata/phase8_conventional_predictor_freeze.json",
        "expected_sites": {"HARV", "ORNL", "TALL", "UNDE", "WREF"},
        "id_prefix": "phase8-conventional-expansion",
    },
}

GDAL_OPTIONS = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MULTIRANGE": "YES",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": 64 * 1024 * 1024,
}
EXPANSION_CHECKPOINT_DIR = ROOT / "data/interim/phase8_conventional_predictor_site_chunks"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def open_catalog(url: str) -> pystac_client.Client:
    return pystac_client.Client.open(url)


def item_datetime(item: Any) -> pd.Timestamp:
    return pd.Timestamp(item.properties["datetime"])


def collection_items(
    catalog: pystac_client.Client,
    collection: str,
    bbox: list[float],
    year: int | None,
) -> list[Any]:
    intervals = [None]
    if year is not None:
        intervals = [
            (
                f"{year}-{month:02d}-01T00:00:00Z/"
                f"{year}-{month:02d}-{calendar.monthrange(year, month)[1]:02d}T23:59:59Z"
            )
            for month in range(1, 13)
        ]
    items: dict[str, Any] = {}
    for interval in intervals:
        parameters: dict[str, Any] = {"collections": [collection], "bbox": bbox}
        if interval is not None:
            parameters["datetime"] = interval
        last_error: Exception | None = None
        for attempt in range(5):
            try:
                for item in catalog.search(**parameters).items():
                    items[item.id] = item
                break
            except APIError as error:
                last_error = error
                time.sleep(2**attempt)
        else:
            raise RuntimeError(
                f"STAC search failed after retries for {collection} {interval}: {last_error}"
            )
    return sorted(items.values(), key=lambda item: item.id)


def select_sentinel2(items: list[Any], config: dict[str, Any]) -> list[Any]:
    maximum_cloud = float(config["maximum_item_cloud_cover_percent"])
    per_group = int(config["scenes_per_month_per_mgrs_tile"])
    records = []
    for item in items:
        cloud = item.properties.get("eo:cloud_cover")
        cloud = float(cloud) if cloud is not None else 100.0
        if cloud > maximum_cloud:
            continue
        timestamp = item_datetime(item)
        records.append(
            {
                "item": item,
                "month": int(timestamp.month),
                "tile": str(item.properties["s2:mgrs_tile"]),
                "cloud": cloud,
                "datetime": timestamp,
            }
        )
    selected: list[Any] = []
    frame = pd.DataFrame(records)
    if frame.empty:
        raise RuntimeError("No Sentinel-2 items passed the frozen cloud threshold")
    for _, group in frame.groupby(["month", "tile"], sort=True):
        chosen = group.sort_values(["cloud", "datetime"], kind="stable").head(per_group)
        selected.extend(chosen["item"].tolist())
    return sorted({item.id: item for item in selected}.values(), key=lambda item: item.id)


def select_evenly(values: list[str], count: int) -> list[str]:
    if len(values) <= count:
        return values
    positions = np.linspace(0, len(values) - 1, count)
    return [values[int(round(position))] for position in positions]


def select_sentinel1(items: list[Any], config: dict[str, Any]) -> list[Any]:
    per_group = int(config["scenes_per_month_per_orbit_state"])
    required_assets = set(config["polarizations"])
    grouped: dict[tuple[int, str], dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))
    for item in items:
        if not required_assets.issubset(item.assets):
            continue
        timestamp = item_datetime(item)
        state = str(item.properties.get("sat:orbit_state", "unknown")).lower()
        grouped[(int(timestamp.month), state)][timestamp.date().isoformat()].append(item)
    selected: list[Any] = []
    for dates in grouped.values():
        chosen_dates = select_evenly(sorted(dates), per_group)
        for date in chosen_dates:
            selected.extend(dates[date])
    return sorted({item.id: item for item in selected}.values(), key=lambda item: item.id)


def sample_asset(item: Any, asset_key: str, longitude: np.ndarray, latitude: np.ndarray) -> np.ndarray:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            # STAC responses can contain a cached SAS query that has expired
            # while a queued batch job was waiting. Strip any existing query
            # before asking Planetary Computer for a fresh signed URL.
            unsigned = item.clone()
            for asset in unsigned.assets.values():
                parts = urlsplit(asset.href)
                asset.href = urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
            signed = planetary_computer.sign(unsigned)
            with rasterio.Env(**GDAL_OPTIONS):
                with rasterio.open(signed.assets[asset_key].href) as source:
                    transformer = Transformer.from_crs("EPSG:4326", source.crs, always_xy=True)
                    x, y = transformer.transform(longitude, latitude)
                    x = np.asarray(x)
                    y = np.asarray(y)
                    inside = (
                        (x >= source.bounds.left)
                        & (x <= source.bounds.right)
                        & (y >= source.bounds.bottom)
                        & (y <= source.bounds.top)
                    )
                    output = np.full(len(longitude), np.nan, dtype=np.float32)
                    indices = np.flatnonzero(inside)
                    if len(indices):
                        samples = np.asarray(
                            list(source.sample(zip(x[indices], y[indices], strict=True))),
                            dtype=np.float32,
                        )[:, 0]
                        if source.nodata is not None:
                            samples[np.isclose(samples, source.nodata)] = np.nan
                        output[indices] = samples
                    return output
        except (OSError, rasterio.errors.RasterioError) as error:
            last_error = error
            time.sleep(2**attempt)
    raise RuntimeError(f"Failed to sample {item.id}/{asset_key}: {last_error}")


def finite_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    output = np.full(len(numerator), np.nan, dtype=np.float32)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (np.abs(denominator) > 1e-6)
    output[valid] = numerator[valid] / denominator[valid]
    return output


def summarise_observations(
    observations: list[np.ndarray],
    prefix: str,
    output: dict[str, np.ndarray],
) -> None:
    if not observations:
        raise RuntimeError(f"No observations available for {prefix}")
    values = np.stack(observations, axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        output[f"{prefix}_p10"] = np.nanpercentile(values, 10, axis=1).astype(np.float32)
        output[f"{prefix}_median"] = np.nanmedian(values, axis=1).astype(np.float32)
        output[f"{prefix}_p90"] = np.nanpercentile(values, 90, axis=1).astype(np.float32)


def extract_sentinel2(
    items: list[Any],
    longitude: np.ndarray,
    latitude: np.ndarray,
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    observations: dict[str, list[np.ndarray]] = defaultdict(list)
    inventory: list[dict[str, Any]] = []
    valid_scl = np.asarray(config["valid_scl_classes"], dtype=np.int16)
    offset = float(config["processing_baseline_offset_dn"])
    scale = float(config["reflectance_scale"])
    bands = dict(config["bands"])

    for position, item in enumerate(items, start=1):
        baseline = float(item.properties["s2:processing_baseline"])
        if baseline < 4.0:
            raise RuntimeError(f"Unexpected pre-04.00 Sentinel-2 baseline in 2024: {item.id}")
        scl = sample_asset(item, "SCL", longitude, latitude)
        scl_classes = np.full(len(scl), -9999, dtype=np.int16)
        finite_scl = np.isfinite(scl)
        scl_classes[finite_scl] = scl[finite_scl].astype(np.int16)
        clear = np.isin(scl_classes, valid_scl)
        reflectance: dict[str, np.ndarray] = {}
        for asset_key, name in bands.items():
            raw = sample_asset(item, asset_key, longitude, latitude)
            value = (raw - offset) * scale
            value[~clear | ~np.isfinite(raw) | (raw <= 0)] = np.nan
            value[(value < -0.2) | (value > 1.6)] = np.nan
            reflectance[name] = value.astype(np.float32)
            observations[f"s2_{name}"].append(reflectance[name])
        observations["s2_ndvi"].append(
            finite_ratio(reflectance["nir"] - reflectance["red"], reflectance["nir"] + reflectance["red"])
        )
        observations["s2_ndmi"].append(
            finite_ratio(reflectance["nir"] - reflectance["swir1"], reflectance["nir"] + reflectance["swir1"])
        )
        observations["s2_nbr"].append(
            finite_ratio(reflectance["nir"] - reflectance["swir2"], reflectance["nir"] + reflectance["swir2"])
        )
        timestamp = item_datetime(item)
        inventory.append(
            {
                "collection": config["collection"],
                "item_id": item.id,
                "datetime": timestamp.isoformat(),
                "month": int(timestamp.month),
                "selection_group": f"{timestamp.month:02d}_{item.properties['s2:mgrs_tile']}",
                "mgrs_tile": item.properties["s2:mgrs_tile"],
                "orbit_state": item.properties.get("sat:orbit_state"),
                "item_cloud_cover_percent": item.properties.get("eo:cloud_cover"),
                "processing_baseline": item.properties["s2:processing_baseline"],
            }
        )
        print(f"Sentinel-2 {position}/{len(items)}: {item.id}")

    output: dict[str, np.ndarray] = {}
    for name, arrays in sorted(observations.items()):
        summarise_observations(arrays, name, output)
    output["s2_valid_observation_count"] = np.sum(
        np.isfinite(np.stack(observations["s2_nir"], axis=1)), axis=1
    ).astype(np.int16)
    return output, inventory


def extract_sentinel1(
    items: list[Any],
    longitude: np.ndarray,
    latitude: np.ndarray,
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    observations: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    inventory: list[dict[str, Any]] = []
    for position, item in enumerate(items, start=1):
        state = str(item.properties["sat:orbit_state"]).lower()
        if state not in {"ascending", "descending"}:
            raise RuntimeError(f"Unexpected Sentinel-1 orbit state: {state}")
        for polarization in config["polarizations"]:
            linear = sample_asset(item, polarization, longitude, latitude)
            valid = np.isfinite(linear) & (linear > 0)
            decibels = np.full(len(linear), np.nan, dtype=np.float32)
            decibels[valid] = 10.0 * np.log10(linear[valid])
            observations[(state, polarization)].append(decibels)
        timestamp = item_datetime(item)
        inventory.append(
            {
                "collection": config["collection"],
                "item_id": item.id,
                "datetime": timestamp.isoformat(),
                "month": int(timestamp.month),
                "selection_group": f"{timestamp.month:02d}_{state}_{timestamp.date().isoformat()}",
                "mgrs_tile": None,
                "orbit_state": state,
                "item_cloud_cover_percent": None,
                "processing_baseline": None,
            }
        )
        print(f"Sentinel-1 {position}/{len(items)}: {item.id}")

    output: dict[str, np.ndarray] = {}
    for (state, polarization), arrays in sorted(observations.items()):
        prefix = f"s1_{state}_{polarization}_db"
        summarise_observations(arrays, prefix, output)
        output[f"s1_{state}_valid_observation_count"] = np.sum(
            np.isfinite(np.stack(arrays, axis=1)), axis=1
        ).astype(np.int16)
    return output, inventory


def extract_topography(
    items: list[Any],
    frame: pd.DataFrame,
    bbox: list[float],
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    expanded = [bbox[0] - 0.01, bbox[1] - 0.01, bbox[2] + 0.01, bbox[3] + 0.01]
    sources = []
    inventory = []
    try:
        with rasterio.Env(**GDAL_OPTIONS):
            for item in items:
                signed = planetary_computer.sign(item)
                sources.append(rasterio.open(signed.assets["data"].href))
                inventory.append(
                    {
                        "collection": config["collection"],
                        "item_id": item.id,
                        "datetime": item_datetime(item).isoformat(),
                        "month": None,
                        "selection_group": "dem_tiles",
                        "mgrs_tile": None,
                        "orbit_state": None,
                        "item_cloud_cover_percent": None,
                        "processing_baseline": None,
                    }
                )
            mosaic, source_transform = merge(
                sources,
                bounds=expanded,
                nodata=np.nan,
                dtype="float32",
                resampling=Resampling.bilinear,
            )
    finally:
        for source in sources:
            source.close()

    elevation_wgs84 = mosaic[0]
    source_bounds = array_bounds(
        elevation_wgs84.shape[0], elevation_wgs84.shape[1], source_transform
    )
    destination_crs = config["analysis_crs"]
    resolution = float(config["analysis_resolution_m"])
    destination_transform, width, height = calculate_default_transform(
        "EPSG:4326",
        destination_crs,
        elevation_wgs84.shape[1],
        elevation_wgs84.shape[0],
        *source_bounds,
        resolution=resolution,
    )
    elevation = np.full((height, width), np.nan, dtype=np.float32)
    reproject(
        source=elevation_wgs84,
        destination=elevation,
        src_transform=source_transform,
        src_crs="EPSG:4326",
        src_nodata=np.nan,
        dst_transform=destination_transform,
        dst_crs=destination_crs,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    dz_dy, dz_dx = np.gradient(elevation.astype(np.float64), resolution, resolution)
    slope = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
    aspect = np.arctan2(-dz_dx, dz_dy)
    rows, columns = rowcol(
        destination_transform,
        frame["x_epsg5070"].to_numpy(),
        frame["y_epsg5070"].to_numpy(),
    )
    rows = np.asarray(rows)
    columns = np.asarray(columns)
    if (
        (rows < 0).any()
        or (rows >= height).any()
        or (columns < 0).any()
        or (columns >= width).any()
    ):
        raise RuntimeError("A GEDI point falls outside the projected DEM mosaic")
    return (
        {
            "terrain_elevation_m": elevation[rows, columns].astype(np.float32),
            "terrain_slope_degrees": slope[rows, columns].astype(np.float32),
            "terrain_aspect_sin": np.sin(aspect[rows, columns]).astype(np.float32),
            "terrain_aspect_cos": np.cos(aspect[rows, columns]).astype(np.float32),
        },
        inventory,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", choices=sorted(PARTITION_PATHS), required=True)
    parser.add_argument("--force", action="store_true", help="Replace an existing freeze")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = PARTITION_PATHS[args.partition]
    protected = [paths["output"], paths["inventory"], paths["manifest"]]
    if any(path.exists() for path in protected) and not args.force:
        existing = [str(path.relative_to(ROOT)) for path in protected if path.exists()]
        raise RuntimeError(f"Predictor freeze exists; refusing to overwrite: {existing}")

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    predictor_config = config["conventional_predictors"]
    input_columns = [
        "site_id",
        "shot_number",
        "longitude",
        "latitude",
        "x_epsg5070",
        "y_epsg5070",
        "tessera_alignment_id",
    ]
    frame = pd.read_parquet(paths["input"], columns=input_columns)
    if set(frame["site_id"]) != paths["expected_sites"]:
        raise RuntimeError(f"Unexpected sites in {args.partition} predictor input")
    if frame.duplicated(["site_id", "shot_number"]).any():
        raise RuntimeError("Duplicate predictor input rows")
    catalog = open_catalog(predictor_config["stac_api"])
    year = int(predictor_config["year"])
    s2_config = predictor_config["sentinel_2"]
    s1_config = predictor_config["sentinel_1"]
    terrain_config = predictor_config["topography"]
    scopes = (
        [(site, group.copy()) for site, group in frame.groupby("site_id", sort=True)]
        if args.partition == "expansion"
        else [(args.partition, frame)]
    )
    output_frames: list[pd.DataFrame] = []
    inventory_records: list[dict[str, Any]] = []
    bboxes: dict[str, list[float]] = {}
    selected_item_ids: dict[str, dict[str, list[str]]] = {}
    for scope, scoped in scopes:
        checkpoint_path = EXPANSION_CHECKPOINT_DIR / f"{scope}_predictors.parquet"
        checkpoint_metadata_path = EXPANSION_CHECKPOINT_DIR / f"{scope}_metadata.json"
        if (
            args.partition == "expansion"
            and checkpoint_path.exists()
            and checkpoint_metadata_path.exists()
        ):
            checkpoint_metadata = json.loads(
                checkpoint_metadata_path.read_text(encoding="utf-8")
            )
            checkpoint = pd.read_parquet(checkpoint_path)
            expected_keys = scoped[["site_id", "shot_number"]].sort_values(
                ["site_id", "shot_number"]
            ).reset_index(drop=True)
            observed_keys = checkpoint[["site_id", "shot_number"]].sort_values(
                ["site_id", "shot_number"]
            ).reset_index(drop=True)
            if not expected_keys.equals(observed_keys):
                raise RuntimeError(f"Checkpoint keys differ from frozen input for {scope}")
            output_frames.append(checkpoint)
            inventory_records.extend(checkpoint_metadata["inventory_records"])
            bboxes[scope] = checkpoint_metadata["bbox_wgs84"]
            selected_item_ids[scope] = checkpoint_metadata["selected_item_ids"]
            print(f"Resumed frozen-input predictor checkpoint for {scope}")
            continue
        longitude = scoped["longitude"].to_numpy(dtype=np.float64)
        latitude = scoped["latitude"].to_numpy(dtype=np.float64)
        bbox = [
            float(longitude.min()),
            float(latitude.min()),
            float(longitude.max()),
            float(latitude.max()),
        ]
        bboxes[scope] = bbox
        s2_candidates = collection_items(catalog, s2_config["collection"], bbox, year)
        s1_candidates = collection_items(catalog, s1_config["collection"], bbox, year)
        dem_items = collection_items(catalog, terrain_config["collection"], bbox, None)
        s2_items = select_sentinel2(s2_candidates, s2_config)
        s1_items = select_sentinel1(s1_candidates, s1_config)
        print(
            f"Selected {len(s2_items)} Sentinel-2 items, {len(s1_items)} Sentinel-1 items, "
            f"and {len(dem_items)} DEM tiles for {scope}"
        )
        s2_features, s2_inventory = extract_sentinel2(
            s2_items, longitude, latitude, s2_config
        )
        s1_features, s1_inventory = extract_sentinel1(
            s1_items, longitude, latitude, s1_config
        )
        terrain_features, terrain_inventory = extract_topography(
            dem_items, scoped, bbox, terrain_config
        )
        features = {**terrain_features, **s2_features, **s1_features}
        scoped_output = scoped[["site_id", "shot_number", "tessera_alignment_id"]].copy()
        for name, values in sorted(features.items()):
            scoped_output[name] = values
        output_frames.append(scoped_output)
        for record in s2_inventory + s1_inventory + terrain_inventory:
            inventory_records.append({"site_scope": scope, **record})
        selected_item_ids[scope] = {
            "sentinel_2": [item.id for item in s2_items],
            "sentinel_1": [item.id for item in s1_items],
            "topography": [item.id for item in dem_items],
        }
        if args.partition == "expansion":
            write_parquet_atomic(scoped_output, checkpoint_path)
            json_dump(
                checkpoint_metadata_path,
                {
                    "site_id": scope,
                    "input_rows": len(scoped),
                    "input_key_sha256": canonical_hash(
                        [
                            [str(row.site_id), int(row.shot_number)]
                            for row in scoped[["site_id", "shot_number"]]
                            .sort_values(["site_id", "shot_number"])
                            .itertuples(index=False)
                        ]
                    ),
                    "bbox_wgs84": bbox,
                    "selected_item_ids": selected_item_ids[scope],
                    "inventory_records": [
                        record
                        for record in inventory_records
                        if record["site_scope"] == scope
                    ],
                },
            )
            print(f"Checkpointed target-free conventional predictors for {scope}")
    output = pd.concat(output_frames, ignore_index=True).sort_values(
        ["site_id", "shot_number"]
    ).reset_index(drop=True)

    minimum_s2 = int(s2_config["minimum_valid_observations_per_point"])
    minimum_s1 = int(s1_config["minimum_valid_observations_per_point_per_state"])
    if (output["s2_valid_observation_count"] < minimum_s2).any():
        raise RuntimeError("Sentinel-2 valid-observation gate failed")
    for state in ("ascending", "descending"):
        column = f"s1_{state}_valid_observation_count"
        if column not in output or (output[column] < minimum_s1).any():
            raise RuntimeError(f"Sentinel-1 valid-observation gate failed for {state}")
    numeric_features = [column for column in output if column not in {"site_id", "shot_number", "tessera_alignment_id"}]
    if not np.isfinite(output[numeric_features].to_numpy(dtype=np.float64)).all():
        raise RuntimeError("Conventional predictor table contains non-finite values")

    inventory = pd.DataFrame(inventory_records)
    inventory.insert(0, "partition", args.partition)
    if args.partition == "expansion":
        alignment = json.loads(PHASE7_FREEZE_PATH.read_text(encoding="utf-8"))
        input_sha256 = alignment["outputs"]["aligned_expansion"]["sha256"]
    else:
        alignment = json.loads(PHASE2_FREEZE_PATH.read_text(encoding="utf-8"))
        input_role = "development" if args.partition == "development" else "locked_bart"
        input_sha256 = alignment["outputs"][input_role]["sha256"]
    freeze_basis = {
        "partition": args.partition,
        "input_path": str(paths["input"].relative_to(ROOT)),
        "input_sha256": input_sha256,
        "tessera_alignment_id": alignment["alignment_id"],
        "project_config_sha256": sha256(CONFIG_PATH),
        "script_sha256": sha256(Path(__file__).resolve()),
        "planetary_computer_version": importlib.metadata.version("planetary-computer"),
        "pystac_client_version": importlib.metadata.version("pystac-client"),
        "rasterio_version": rasterio.__version__,
        "recipe": predictor_config,
        "bbox_wgs84_by_scope": bboxes,
        "selected_item_ids_by_scope": selected_item_ids,
        "feature_columns": numeric_features,
        "target_columns_read": [],
    }
    freeze_basis_sha = canonical_hash(freeze_basis)
    freeze_id = f"{paths['id_prefix']}-{freeze_basis_sha[:12]}"
    output.insert(0, "conventional_predictor_freeze_id", freeze_id)
    write_parquet_atomic(output, paths["output"])
    write_csv_atomic(inventory, paths["inventory"])

    count_columns = [column for column in output if column.endswith("observation_count")]
    missingness = {
        column: {
            "minimum": int(output[column].min()),
            "median": float(output[column].median()),
            "maximum": int(output[column].max()),
        }
        for column in count_columns
    }
    manifest = {
        "freeze_id": freeze_id,
        "created_at": utc_now(),
        "status": "frozen",
        "freeze_basis_sha256": freeze_basis_sha,
        "freeze_basis": freeze_basis,
        "outputs": {
            "predictors": {
                "path": str(paths["output"].relative_to(ROOT)),
                "rows": len(output),
                "columns": len(output.columns),
                "sha256": sha256(paths["output"]),
            },
            "inventory": {
                "path": str(paths["inventory"].relative_to(ROOT)),
                "rows": len(inventory),
                "sha256": sha256(paths["inventory"]),
            },
        },
        "valid_observation_summary": missingness,
    }
    json_dump(paths["manifest"], manifest)
    print(
        json.dumps(
            {
                "freeze_id": freeze_id,
                "partition": args.partition,
                "rows": len(output),
                "feature_columns": len(numeric_features),
                "selected_items_by_scope": {
                    scope: {sensor: len(items) for sensor, items in sensors.items()}
                    for scope, sensors in selected_item_ids.items()
                },
                "valid_observation_summary": missingness,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
