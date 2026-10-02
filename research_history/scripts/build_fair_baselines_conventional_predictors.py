#!/usr/bin/env python3
"""Build the frozen Phase 8 conventional predictors without reading GEDI FHD."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import sys
import time
from collections import defaultdict
from datetime import date, datetime, time as datetime_time, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pystac_client
import requests


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_conventional_predictors as base  # noqa: E402


ALIGNED_PATHS = {
    "development": ROOT / "data/processed/tessera_aligned_development_soap_teak.parquet",
    "bart": ROOT / "data/processed/tessera_aligned_locked_bart.parquet",
    "expansion": ROOT / "data/processed/phase7_tessera_aligned_expansion.parquet",
}
EXISTING_PATHS = {
    "development": ROOT / "data/processed/phase3_conventional_predictors_development.parquet",
    "bart": ROOT / "data/processed/phase4_conventional_predictors_bart.parquet",
}
EXISTING_INVENTORIES = {
    "development": ROOT / "metadata/phase3_conventional_stac_inventory.csv",
    "bart": ROOT / "metadata/phase4_bart_stac_inventory.csv",
}
OUTPUT_PATH = ROOT / "data/processed/phase8_conventional_predictors_all_sites.parquet"
INVENTORY_PATH = ROOT / "metadata/phase8_conventional_predictor_inventory.csv"
FREEZE_PATH = ROOT / "metadata/phase8_conventional_predictor_all_sites_freeze.json"
CHECKPOINT_DIR = ROOT / "data/interim/phase8_conventional_fallback_site_chunks"
EARTH_SEARCH = "https://earth-search.aws.element84.com/v1"
EARTH_SEARCH_S2_RADIOMETRY = (
    "earthsearch_boa_offset_already_applied_cog_dn_times_0.0001"
)
OPERA_SERVICE = (
    "https://gis.earthdata.nasa.gov/image/rest/services/"
    "OPERA_L2_RTC_S1_V1/OPERA_L2_RTC_S1_V1_{polarization}/ImageServer"
)
EXPECTED_SITES = {"BART", "HARV", "ORNL", "SOAP", "TALL", "TEAK", "UNDE", "WREF"}
EXPANSION_SITES = {"HARV", "ORNL", "TALL", "UNDE", "WREF"}
S1_MISSING_SITE = "UNDE"
KEYS = ["site_id", "shot_number"]
TERRAIN_FEATURES = [
    "terrain_elevation_m",
    "terrain_slope_degrees",
    "terrain_aspect_sin",
    "terrain_aspect_cos",
]
S2_VARIABLES = ["blue", "green", "red", "nir", "swir1", "swir2", "ndvi", "ndmi", "nbr"]
S2_FEATURES = [
    f"s2_{variable}_{statistic}"
    for variable in S2_VARIABLES
    for statistic in ["p10", "median", "p90"]
]
S1_FEATURES = [
    f"s1_{polarization}_db_{statistic}"
    for polarization in ["vh", "vv"]
    for statistic in ["p10", "median", "p90"]
]
COUNT_COLUMNS = ["s1_valid_observation_count", "s2_valid_observation_count"]
FEATURE_COLUMNS = TERRAIN_FEATURES + S2_FEATURES + S1_FEATURES + COUNT_COLUMNS
EARTH_S2_BANDS = {
    "blue": "blue",
    "green": "green",
    "red": "red",
    "nir": "nir",
    "swir16": "swir1",
    "swir22": "swir2",
}


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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def item_tile(item: Any) -> str:
    code = str(item.properties["grid:code"])
    if not code.startswith("MGRS-"):
        raise RuntimeError(f"Unexpected Earth Search grid code: {code}")
    return code.removeprefix("MGRS-")


def select_sentinel2(items: list[Any], maximum_cloud: float, per_group: int) -> list[Any]:
    records = []
    for item in items:
        cloud = float(item.properties.get("eo:cloud_cover", 100.0))
        if cloud <= maximum_cloud:
            timestamp = base.item_datetime(item)
            records.append(
                {
                    "item": item,
                    "month": int(timestamp.month),
                    "tile": item_tile(item),
                    "cloud": cloud,
                    "datetime": timestamp,
                }
            )
    frame = pd.DataFrame(records)
    if frame.empty:
        raise RuntimeError("No Earth Search Sentinel-2 scenes passed the cloud threshold")
    selected = []
    for _, group in frame.groupby(["month", "tile"], sort=True):
        selected.extend(
            group.sort_values(["cloud", "datetime"], kind="stable").head(per_group)["item"]
        )
    return sorted({item.id: item for item in selected}.values(), key=lambda item: item.id)


def extract_sentinel2(
    items: list[Any],
    longitude: np.ndarray,
    latitude: np.ndarray,
    config: dict[str, Any],
    allow_unapplied_boa_offset: bool = False,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    valid_scl = np.asarray(config["valid_scl_classes"], dtype=np.int16)
    scale = float(config["reflectance_scale"])

    def extract_item(item: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        baseline = float(item.properties["s2:processing_baseline"])
        offset_applied = bool(item.properties.get("earthsearch:boa_offset_applied"))
        if baseline < 4.0 or (not offset_applied and not allow_unapplied_boa_offset):
            raise RuntimeError(f"Unexpected Sentinel-2 radiometry metadata: {item.id}")
        scl = base.sample_asset(item, "scl", longitude, latitude)
        scl_classes = np.full(len(scl), -9999, dtype=np.int16)
        finite_scl = np.isfinite(scl)
        scl_classes[finite_scl] = scl[finite_scl].astype(np.int16)
        clear = np.isin(scl_classes, valid_scl)
        reflectance: dict[str, np.ndarray] = {}
        item_observations: dict[str, np.ndarray] = {}
        for asset_key, name in EARTH_S2_BANDS.items():
            raw = base.sample_asset(item, asset_key, longitude, latitude)
            if offset_applied:
                # Earth Search's COG conversion has already applied the
                # post-2022 BOA offset, so only the DN scale remains.
                value = raw * scale
                radiometry_mode = "earthsearch_boa_offset_already_applied"
            else:
                bands = item.assets[asset_key].extra_fields.get("raster:bands", [])
                if len(bands) != 1:
                    raise RuntimeError(f"Missing raster radiometry metadata: {item.id}")
                asset_scale = float(bands[0].get("scale", np.nan))
                asset_offset = float(bands[0].get("offset", np.nan))
                if not np.isclose(asset_scale, scale) or not np.isclose(asset_offset, -0.1):
                    raise RuntimeError(f"Unexpected raster radiometry metadata: {item.id}")
                value = raw * asset_scale + asset_offset
                radiometry_mode = "earthsearch_asset_scale_and_boa_offset_applied_locally"
            value[~clear | ~np.isfinite(raw) | (raw <= 0)] = np.nan
            value[(value < -0.2) | (value > 1.6)] = np.nan
            reflectance[name] = value.astype(np.float32)
            item_observations[f"s2_{name}"] = reflectance[name]
        item_observations["s2_ndvi"] = base.finite_ratio(
            reflectance["nir"] - reflectance["red"],
            reflectance["nir"] + reflectance["red"],
        )
        item_observations["s2_ndmi"] = base.finite_ratio(
            reflectance["nir"] - reflectance["swir1"],
            reflectance["nir"] + reflectance["swir1"],
        )
        item_observations["s2_nbr"] = base.finite_ratio(
            reflectance["nir"] - reflectance["swir2"],
            reflectance["nir"] + reflectance["swir2"],
        )
        timestamp = base.item_datetime(item)
        return item_observations, {
            "source": "earth-search",
            "collection": "sentinel-2-l2a",
            "item_id": item.id,
            "datetime": timestamp.isoformat(),
            "month": int(timestamp.month),
            "selection_group": f"{timestamp.month:02d}_{item_tile(item)}",
            "polarization": None,
            "raster_id": None,
            "sentinel_2_radiometry": radiometry_mode,
        }

    observations: dict[str, list[np.ndarray]] = defaultdict(list)
    inventory: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        for position, (item_observations, item_inventory) in enumerate(
            executor.map(extract_item, items), start=1
        ):
            for name, values in item_observations.items():
                observations[name].append(values)
            inventory.append(item_inventory)
            print(f"Sentinel-2 {position}/{len(items)}: {item_inventory['item_id']}")
    output: dict[str, np.ndarray] = {}
    for name, arrays in sorted(observations.items()):
        base.summarise_observations(arrays, name, output)
    output["s2_valid_observation_count"] = np.sum(
        np.isfinite(np.stack(observations["s2_nir"], axis=1)), axis=1
    ).astype(np.int16)
    return output, inventory


def select_dates(values: list[date], count: int) -> list[date]:
    unique = sorted(set(values))
    if len(unique) <= count:
        return unique
    positions = np.linspace(0, len(unique) - 1, count)
    return [unique[int(round(position))] for position in positions]


def opera_dates(
    session: requests.Session, bbox: list[float], per_month: int
) -> tuple[list[date], list[dict[str, Any]]]:
    expanded = [bbox[0] - 0.05, bbox[1] - 0.05, bbox[2] + 0.05, bbox[3] + 0.05]
    parameters = {
        "where": "category = 1 AND startdate >= DATE '2024-01-01' "
        "AND startdate < DATE '2025-01-01'",
        "geometry": ",".join(map(str, expanded)),
        "geometryType": "esriGeometryEnvelope",
        "inSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "objectid,name,startdate",
        "returnGeometry": "false",
        "f": "json",
    }
    response = session.get(
        OPERA_SERVICE.format(polarization="VV") + "/query",
        params=parameters,
        timeout=90,
    )
    response.raise_for_status()
    payload = response.json()
    if "error" in payload:
        raise RuntimeError(f"OPERA catalog query failed: {payload['error']}")
    records = []
    for feature in payload.get("features", []):
        attributes = feature["attributes"]
        name = str(attributes["name"])
        if "OPERA_L2_RTC-S1_" not in name:
            continue
        timestamp = datetime.fromtimestamp(attributes["startdate"] / 1000, timezone.utc)
        records.append(
            {
                "date": timestamp.date(),
                "datetime": timestamp,
                "raster_id": int(attributes["objectid"]),
                "item_id": name,
            }
        )
    selected = []
    for month in range(1, 13):
        selected.extend(select_dates([record["date"] for record in records if record["date"].month == month], per_month))
    return sorted(set(selected)), records


def sample_opera_date(
    session: requests.Session,
    polarization: str,
    acquisition_date: date,
    longitude: np.ndarray,
    latitude: np.ndarray,
    batch_size: int = 1000,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    output = np.full(len(longitude), np.nan, dtype=np.float32)
    raster_records: dict[int, dict[str, Any]] = {}
    start = datetime.combine(acquisition_date, datetime_time.min, tzinfo=timezone.utc)
    end = datetime.combine(acquisition_date, datetime_time.max, tzinfo=timezone.utc)
    time_extent = f"{int(start.timestamp() * 1000)},{int(end.timestamp() * 1000)}"
    url = OPERA_SERVICE.format(polarization=polarization.upper()) + "/getSamples"
    for first in range(0, len(longitude), batch_size):
        last = min(first + batch_size, len(longitude))
        points = [
            [float(x), float(y)]
            for x, y in zip(longitude[first:last], latitude[first:last], strict=True)
        ]
        geometry = json.dumps(
            {"points": points, "spatialReference": {"wkid": 4326}},
            separators=(",", ":"),
        )
        parameters = {
            "geometry": geometry,
            "geometryType": "esriGeometryMultipoint",
            "time": time_extent,
            "outFields": "objectid,name,startdate",
            "f": "json",
        }
        payload = None
        for attempt in range(5):
            try:
                response = requests.post(url, data=parameters, timeout=90)
                response.raise_for_status()
                payload = response.json()
                if "error" not in payload:
                    break
            except (requests.RequestException, ValueError):
                payload = None
            time.sleep(2**attempt)
        if payload is None or "error" in payload:
            continue
        for sample in payload.get("samples", []):
            index = first + int(sample["locationId"])
            value = float(str(sample["value"]).split()[0])
            if np.isfinite(value) and value > 0:
                output[index] = 10.0 * np.log10(value)
            attributes = sample.get("attributes", {})
            if attributes.get("objectid") is not None:
                raster_id = int(attributes["objectid"])
                raster_records[raster_id] = {
                    "raster_id": raster_id,
                    "item_id": attributes.get("name"),
                    "datetime": datetime.fromtimestamp(
                        attributes["startdate"] / 1000, timezone.utc
                    ).isoformat()
                    if attributes.get("startdate") is not None
                    else start.isoformat(),
                }
    return output, list(raster_records.values())


def extract_opera(
    session: requests.Session,
    site: str,
    longitude: np.ndarray,
    latitude: np.ndarray,
    per_month: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], list[str]]:
    bbox = [
        float(longitude.min()),
        float(latitude.min()),
        float(longitude.max()),
        float(latitude.max()),
    ]
    dates, _ = opera_dates(session, bbox, per_month)
    if not dates:
        if site != S1_MISSING_SITE:
            raise RuntimeError(f"No 2024 OPERA RTC-S1 dates found for {site}")
        return (
            {feature: np.full(len(longitude), np.nan, dtype=np.float32) for feature in S1_FEATURES}
            | {"s1_valid_observation_count": np.zeros(len(longitude), dtype=np.int16)},
            [],
            [],
        )
    observations: dict[str, list[np.ndarray]] = defaultdict(list)
    inventory: list[dict[str, Any]] = []
    tasks = [
        (acquisition_date, polarization)
        for acquisition_date in dates
        for polarization in ["vv", "vh"]
    ]

    def sample_task(
        task: tuple[date, str]
    ) -> tuple[date, str, np.ndarray, list[dict[str, Any]]]:
        acquisition_date, polarization = task
        values, rasters = sample_opera_date(
            session, polarization, acquisition_date, longitude, latitude
        )
        return acquisition_date, polarization, values, rasters

    with ThreadPoolExecutor(max_workers=6) as executor:
        for position, (acquisition_date, polarization, values, rasters) in enumerate(
            executor.map(sample_task, tasks), start=1
        ):
            observations[polarization].append(values)
            for raster in rasters:
                inventory.append(
                    {
                        "source": "nasa-earthdata-egis",
                        "collection": "OPERA_L2_RTC-S1_V1",
                        "item_id": raster["item_id"],
                        "datetime": raster["datetime"],
                        "month": acquisition_date.month,
                        "selection_group": acquisition_date.isoformat(),
                        "polarization": polarization.upper(),
                        "raster_id": raster["raster_id"],
                    }
                )
            print(
                f"OPERA RTC-S1 {site} {position}/{len(tasks)}: "
                f"{acquisition_date} {polarization.upper()}"
            )
    output: dict[str, np.ndarray] = {}
    for polarization in ["vh", "vv"]:
        base.summarise_observations(
            observations[polarization], f"s1_{polarization}_db", output
        )
    output["s1_valid_observation_count"] = np.sum(
        np.isfinite(np.stack(observations["vv"], axis=1)), axis=1
    ).astype(np.int16)
    return output, inventory, [value.isoformat() for value in dates]


def existing_s2_terrain() -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames = []
    inventory = []
    columns = KEYS + TERRAIN_FEATURES + S2_FEATURES + ["s2_valid_observation_count"]
    for partition, path in EXISTING_PATHS.items():
        frames.append(pd.read_parquet(path, columns=columns))
        site_scope = "BART" if partition == "bart" else "SOAP+TEAK"
        source = pd.read_csv(EXISTING_INVENTORIES[partition])
        source = source[source["collection"].isin(["sentinel-2-l2a", "cop-dem-glo-30"])]
        for record in source.to_dict("records"):
            inventory.append(
                {
                    "site_id": site_scope,
                    "source": "planetary-computer-existing-freeze",
                    "collection": record["collection"],
                    "item_id": record["item_id"],
                    "datetime": record["datetime"],
                    "month": record["month"],
                    "selection_group": record["selection_group"],
                    "polarization": None,
                    "raster_id": None,
                }
            )
    return pd.concat(frames, ignore_index=True), inventory


def expansion_s2_terrain(
    aligned: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, list[str]]]:
    catalog = pystac_client.Client.open(EARTH_SEARCH)
    predictor = config["conventional_predictors"]
    s2_config = predictor["sentinel_2"]
    terrain_config = dict(predictor["topography"])
    terrain_config["collection"] = "cop-dem-glo-30"
    base.GDAL_OPTIONS["AWS_NO_SIGN_REQUEST"] = "YES"
    output_frames = []
    inventory: list[dict[str, Any]] = []
    selected_items: dict[str, list[str]] = {}
    for site, frame in aligned.groupby("site_id", sort=True):
        checkpoint = CHECKPOINT_DIR / f"{site}_s2_terrain.parquet"
        checkpoint_metadata = CHECKPOINT_DIR / f"{site}_s2_terrain.json"
        if checkpoint.exists() and checkpoint_metadata.exists():
            metadata = json.loads(checkpoint_metadata.read_text(encoding="utf-8"))
            if metadata.get("sentinel_2_radiometry") != EARTH_SEARCH_S2_RADIOMETRY:
                raise RuntimeError(
                    f"Stale Sentinel-2 radiometry checkpoint for {site}: {checkpoint}"
                )
            output_frames.append(pd.read_parquet(checkpoint))
            inventory.extend(metadata["inventory"])
            selected_items[site] = metadata["selected_item_ids"]
            print(f"Resumed Sentinel-2/terrain checkpoint for {site}")
            continue
        longitude = frame["longitude"].to_numpy(dtype=np.float64)
        latitude = frame["latitude"].to_numpy(dtype=np.float64)
        bbox = [
            float(longitude.min()), float(latitude.min()),
            float(longitude.max()), float(latitude.max()),
        ]
        candidates = base.collection_items(catalog, "sentinel-2-l2a", bbox, 2024)
        scenes = select_sentinel2(
            candidates,
            float(s2_config["maximum_item_cloud_cover_percent"]),
            int(s2_config["scenes_per_month_per_mgrs_tile"]),
        )
        dem_items = base.collection_items(catalog, "cop-dem-glo-30", bbox, None)
        print(f"Selected {len(scenes)} Sentinel-2 scenes and {len(dem_items)} DEM tiles for {site}")
        s2, s2_inventory = extract_sentinel2(scenes, longitude, latitude, s2_config)
        terrain, terrain_inventory = base.extract_topography(
            dem_items, frame, bbox, terrain_config
        )
        site_output = frame[KEYS].copy()
        for name, values in {**terrain, **s2}.items():
            site_output[name] = values
        site_inventory = [
            {"site_id": site, **record} for record in s2_inventory
        ] + [
            {
                "site_id": site,
                "source": "earth-search",
                "collection": "cop-dem-glo-30",
                "item_id": record["item_id"],
                "datetime": record["datetime"],
                "month": None,
                "selection_group": "dem_tiles",
                "polarization": None,
                "raster_id": None,
            }
            for record in terrain_inventory
        ]
        selected_items[site] = [item.id for item in scenes] + [item.id for item in dem_items]
        base.write_parquet_atomic(site_output, checkpoint)
        write_json(
            checkpoint_metadata,
            {
                "sentinel_2_radiometry": EARTH_SEARCH_S2_RADIOMETRY,
                "selected_item_ids": selected_items[site],
                "inventory": site_inventory,
            },
        )
        output_frames.append(site_output)
        inventory.extend(site_inventory)
    return pd.concat(output_frames, ignore_index=True), inventory, selected_items


def add_opera(
    frame: pd.DataFrame, per_month: int
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, list[str]]]:
    session = requests.Session()
    output_frames = []
    inventory = []
    selected_dates: dict[str, list[str]] = {}
    for site, scoped in frame.groupby("site_id", sort=True):
        checkpoint = CHECKPOINT_DIR / f"{site}_opera_s1.parquet"
        checkpoint_metadata = CHECKPOINT_DIR / f"{site}_opera_s1.json"
        if checkpoint.exists() and checkpoint_metadata.exists():
            output_frames.append(pd.read_parquet(checkpoint))
            metadata = json.loads(checkpoint_metadata.read_text(encoding="utf-8"))
            inventory.extend(metadata["inventory"])
            selected_dates[site] = metadata["selected_dates"]
            print(f"Resumed OPERA RTC-S1 checkpoint for {site}")
            continue
        s1, site_inventory, dates = extract_opera(
            session,
            site,
            scoped["longitude"].to_numpy(dtype=np.float64),
            scoped["latitude"].to_numpy(dtype=np.float64),
            per_month,
        )
        site_output = scoped[KEYS].copy()
        for name, values in s1.items():
            site_output[name] = values
        site_inventory = [{"site_id": site, **record} for record in site_inventory]
        base.write_parquet_atomic(site_output, checkpoint)
        write_json(
            checkpoint_metadata,
            {"selected_dates": dates, "inventory": site_inventory},
        )
        output_frames.append(site_output)
        inventory.extend(site_inventory)
        selected_dates[site] = dates
    return pd.concat(output_frames, ignore_index=True), inventory, selected_dates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Replace final outputs; keep site checkpoints")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protected = [OUTPUT_PATH, INVENTORY_PATH, FREEZE_PATH]
    if not args.force and any(path.exists() for path in protected):
        raise RuntimeError("Phase 8 all-site conventional predictor freeze already exists")
    config = base.yaml.safe_load(base.CONFIG_PATH.read_text(encoding="utf-8"))
    aligned = pd.concat(
        [
            pd.read_parquet(
                path,
                columns=KEYS + ["longitude", "latitude", "x_epsg5070", "y_epsg5070"],
            )
            for path in ALIGNED_PATHS.values()
        ],
        ignore_index=True,
    )
    if len(aligned) != 10184 or set(aligned["site_id"]) != EXPECTED_SITES:
        raise RuntimeError("Unexpected Phase 8 aligned predictor population")
    if aligned.duplicated(KEYS).any():
        raise RuntimeError("Duplicate all-site predictor keys")

    existing, inventory = existing_s2_terrain()
    expansion_input = aligned[aligned["site_id"].isin(EXPANSION_SITES)].copy()
    expansion, expansion_inventory, selected_items = expansion_s2_terrain(
        expansion_input, config
    )
    s2_terrain = pd.concat([existing, expansion], ignore_index=True)
    s1, s1_inventory, selected_dates = add_opera(
        aligned, int(config["conventional_predictors"]["sentinel_1"]["scenes_per_month_per_orbit_state"])
    )
    output = aligned[KEYS].merge(s2_terrain, on=KEYS, validate="one_to_one").merge(
        s1, on=KEYS, validate="one_to_one"
    )
    output = output[KEYS + FEATURE_COLUMNS].sort_values(KEYS).reset_index(drop=True)
    if (output["s2_valid_observation_count"] < 6).any():
        raise RuntimeError("At least one point has fewer than six clear Sentinel-2 observations")
    non_s1 = [feature for feature in FEATURE_COLUMNS if feature not in S1_FEATURES]
    if not np.isfinite(output[non_s1].to_numpy(dtype=np.float64)).all():
        raise RuntimeError("Eight-site Sentinel-2/terrain predictors contain non-finite values")
    site_optical_medians = output.groupby("site_id")[[
        "s2_blue_median",
        "s2_red_median",
        "s2_nir_median",
        "s2_ndvi_median",
    ]].median()
    if not site_optical_medians[[
        "s2_blue_median", "s2_red_median", "s2_nir_median"
    ]].apply(lambda values: values.between(-0.05, 1.0)).all().all():
        raise RuntimeError("Sentinel-2 site median reflectance failed radiometric QA")
    if not site_optical_medians["s2_ndvi_median"].between(-1.0, 1.0).all():
        raise RuntimeError("Sentinel-2 site median NDVI failed radiometric QA")
    available = output[~output["site_id"].eq(S1_MISSING_SITE)]
    if not np.isfinite(available[S1_FEATURES].to_numpy(dtype=np.float64)).all():
        raise RuntimeError("A site with 2024 Sentinel-1 coverage has non-finite OPERA features")
    missing = output[output["site_id"].eq(S1_MISSING_SITE)]
    if not missing[S1_FEATURES].isna().all().all() or missing["s1_valid_observation_count"].ne(0).any():
        raise RuntimeError("UNDE must be explicitly represented as lacking 2024 Sentinel-1 coverage")

    all_inventory = pd.DataFrame(inventory + expansion_inventory + s1_inventory)
    all_inventory = all_inventory.drop_duplicates().sort_values(
        ["site_id", "collection", "datetime", "item_id"], na_position="last"
    )
    base.write_parquet_atomic(output, OUTPUT_PATH)
    base.write_csv_atomic(all_inventory, INVENTORY_PATH)
    outputs = {
        "predictors": {"path": str(OUTPUT_PATH.relative_to(ROOT)), "sha256": sha256(OUTPUT_PATH)},
        "inventory": {"path": str(INVENTORY_PATH.relative_to(ROOT)), "sha256": sha256(INVENTORY_PATH)},
    }
    freeze_basis = {
        "script_sha256": sha256(Path(__file__).resolve()),
        "input_hashes": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in list(ALIGNED_PATHS.values())
            + list(EXISTING_PATHS.values())
            + list(EXISTING_INVENTORIES.values())
        },
        "rows": len(output),
        "sites": sorted(EXPECTED_SITES),
        "target_columns_read": [],
        "sentinel_2_and_dem": {
            "existing_sites": ["BART", "SOAP", "TEAK"],
            "existing_source": "Microsoft Planetary Computer frozen products",
            "expansion_sites": sorted(EXPANSION_SITES),
            "expansion_source": "Element 84 Earth Search public mirrors of Sentinel-2 L2A and Copernicus DEM GLO-30",
            "expansion_radiometry": EARTH_SEARCH_S2_RADIOMETRY,
            "selected_item_ids": selected_items,
        },
        "sentinel_1": {
            "source": "NASA OPERA_L2_RTC-S1_V1 through Earthdata EGIS ImageServer",
            "radiometry": "30 m gamma-0 power converted to decibels",
            "temporal_statistics": ["p10", "median", "p90"],
            "selected_dates": selected_dates,
            "site_without_2024_coverage": S1_MISSING_SITE,
            "missing_s1_values_imputed": False,
        },
        "feature_columns": FEATURE_COLUMNS,
        "primary_eight_site_features": TERRAIN_FEATURES + S2_FEATURES,
        "seven_site_radar_sensitivity_features": TERRAIN_FEATURES + S2_FEATURES + S1_FEATURES,
        "observation_count_columns": COUNT_COLUMNS,
        "observation_counts_are_model_features": False,
    }
    freeze_id = "phase8-conventional-all-sites-" + canonical_hash(
        {"freeze_basis": freeze_basis, "outputs": outputs}
    )[:12]
    write_json(
        FREEZE_PATH,
        {
            "freeze_id": freeze_id,
            "created_utc": utc_now(),
            "freeze_basis": freeze_basis,
            "outputs": outputs,
        },
    )
    print(
        json.dumps(
            {
                "freeze_id": freeze_id,
                "rows": len(output),
                "sites": output.groupby("site_id").size().to_dict(),
                "s1_valid_observation_count_minimum_by_site": output.groupby("site_id")[
                    "s1_valid_observation_count"
                ].min().to_dict(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
