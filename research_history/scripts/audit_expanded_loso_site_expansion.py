#!/usr/bin/env python3
"""Freeze outcome-blind feasibility evidence for the Phase 7 site expansion."""

from __future__ import annotations

import hashlib
import io
import json
import math
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests
import yaml
from pyproj import Transformer
from rasterio.io import MemoryFile
from rasterio.mask import mask
from shapely.geometry import mapping


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/project.yaml"
FLIGHT_BOX_URL = "https://www.neonscience.org/sites/default/files/AOP_flightBoxes_0.zip"
CMR_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
NLCD_WCS_URL = "https://dmsdata.cr.usgs.gov/geoserver/mrlc_Land-Cover-Native_conus_year_data/ows"
NLCD_COVERAGE = "mrlc_Land-Cover-Native_conus_year_data:Land-Cover-Native_conus_year_data"
MANIFEST_PATH = ROOT / "data/external/geotessera_cache/v1/manifest.parquet"
SCREEN_PATH = ROOT / "metadata/phase7_site_screen.csv"
BOUNDARY_PATH = ROOT / "metadata/neon_aop_priority1_expansion_2024.geojson"
FREEZE_PATH = ROOT / "metadata/phase7_site_selection_freeze.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def required_tile_indices(bounds: tuple[float, float, float, float]) -> set[tuple[int, int]]:
    west, south, east, north = bounds
    longitude = [value * 10 + 5 for value in range(math.floor(west * 10), math.floor(east * 10) + 1)]
    latitude = [value * 10 + 5 for value in range(math.floor(south * 10), math.floor(north * 10) + 1)]
    return {(lon, lat) for lon in longitude for lat in latitude}


def nlcd_forest_fraction(session: requests.Session, geometry: Any) -> float:
    transformer = Transformer.from_crs(4326, 5070, always_xy=True)
    minx, miny, maxx, maxy = transformer.transform_bounds(*geometry.bounds, densify_pts=21)
    width = max(1, math.ceil((maxx - minx) / 30))
    height = max(1, math.ceil((maxy - miny) / 30))
    params = {
        "service": "WCS",
        "version": "1.0.0",
        "request": "GetCoverage",
        "coverage": NLCD_COVERAGE,
        "crs": "EPSG:5070",
        "response_crs": "EPSG:5070",
        "bbox": f"{minx},{miny},{maxx},{maxy}",
        "width": width,
        "height": height,
        "format": "GeoTIFF",
        "time": "2024-01-01T00:00:00.000Z",
    }
    response = session.get(NLCD_WCS_URL, params=params, timeout=180)
    response.raise_for_status()
    projected = gpd.GeoSeries([geometry], crs=4326).to_crs(5070).iloc[0]
    with MemoryFile(response.content) as memory, memory.open() as dataset:
        values, _ = mask(dataset, [mapping(projected)], crop=True, filled=False)
    pixels = values[0].compressed()
    return float(np.isin(pixels, [41, 42, 43]).mean())


def main() -> int:
    protected = [SCREEN_PATH, BOUNDARY_PATH, FREEZE_PATH]
    existing = [str(path.relative_to(ROOT)) for path in protected if path.exists()]
    if existing:
        raise RuntimeError(f"Phase 7 site selection exists; refusing to overwrite: {existing}")
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    protocol = config["phase7_site_expansion"]
    candidates = list(protocol["candidate_sites"])
    selected = list(protocol["selected_expansion_sites"])
    if protocol["selection_used_fhd_values_or_distributions"]:
        raise RuntimeError("Site selection must be outcome-blind")

    session = requests.Session()
    session.headers.update({"User-Agent": "tessera-gedi-fhd/0.1 outcome-blind-site-screen"})
    response = session.get(FLIGHT_BOX_URL, timeout=90)
    response.raise_for_status()
    zip_bytes = response.content
    with tempfile.TemporaryDirectory() as temporary:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
            archive.extractall(temporary)
        shapefile = next(Path(temporary).rglob("AOP_flightboxesAllSites.shp"))
        boundaries = gpd.read_file(shapefile).to_crs(4326)
    boundaries = boundaries[
        boundaries["siteID"].isin(candidates)
        & boundaries["priority"].eq(1)
        & boundaries["sampleType"].eq("Terrestrial")
    ].dissolve(by="siteID", aggfunc="first").reset_index()
    if set(boundaries["siteID"]) != set(candidates):
        raise RuntimeError("Candidate Priority-1 terrestrial boundary set is incomplete")

    all_required = {site: required_tile_indices(geometry.bounds) for site, geometry in zip(
        boundaries["siteID"], boundaries.geometry, strict=True
    )}
    lon_values = sorted({lon for values in all_required.values() for lon, _ in values})
    lat_values = sorted({lat for values in all_required.values() for _, lat in values})
    inventory = pq.read_table(
        MANIFEST_PATH,
        columns=["lon_i", "lat_i"],
        filters=[
            ("version", "=", "1.0"),
            ("variant", "=", "vultr"),
            ("year", "=", 2024),
            ("lon_i", "in", lon_values),
            ("lat_i", "in", lat_values),
        ],
    )
    available = set(zip(inventory["lon_i"].to_pylist(), inventory["lat_i"].to_pylist()))
    records: list[dict[str, Any]] = []
    for row in boundaries.sort_values("siteID").itertuples():
        bbox = ",".join(f"{value:.12f}" for value in row.geometry.bounds)
        cmr = session.get(CMR_URL, params={
            "collection_concept_id": config["target"]["cmr_collection_concept_id"],
            "bounding_box": bbox,
            "temporal": "2024-01-01T00:00:00Z,2024-12-31T23:59:59Z",
            "page_size": 0,
        }, timeout=90)
        cmr.raise_for_status()
        required = all_required[row.siteID]
        records.append({
            "site_id": row.siteID,
            "selected": row.siteID in selected,
            "domain": row.domain,
            "domain_name": row.domainName,
            "site_name": row.siteName,
            "flight_box_id": row.flightbxID,
            "cmr_2024_gedi_v3_granules": int(cmr.headers["CMR-Hits"]),
            "nlcd_2024_forest_fraction": nlcd_forest_fraction(session, row.geometry),
            "required_tessera_tiles": len(required),
            "available_tessera_tiles": len(required & available),
            "tessera_inventory_complete": required <= available,
            "bbox_wgs84": json.dumps(list(row.geometry.bounds)),
        })
    screen = pd.DataFrame(records).sort_values("site_id").reset_index(drop=True)
    selected_screen = screen[screen["selected"]]
    evidence = protocol["selection_evidence"]
    gate = (
        selected_screen["cmr_2024_gedi_v3_granules"].ge(
            evidence["minimum_2024_gedi_v3_intersecting_granules"]
        ).all()
        and selected_screen["nlcd_2024_forest_fraction"].ge(
            evidence["minimum_annual_nlcd_2024_forest_fraction_in_flight_box"]
        ).all()
        and selected_screen["tessera_inventory_complete"].all()
    )
    if not gate:
        raise RuntimeError("One or more preselected expansion sites failed feasibility")
    selected_boundaries = boundaries[boundaries["siteID"].isin(selected)].copy()
    selected_boundaries = selected_boundaries.sort_values("siteID")
    write_text(SCREEN_PATH, screen.to_csv(index=False))
    write_text(BOUNDARY_PATH, selected_boundaries.to_json(drop_id=True) + "\n")
    freeze = {
        "freeze_id": "phase7-site-selection-" + hashlib.sha256(
            screen.to_csv(index=False).encode("utf-8")
        ).hexdigest()[:12],
        "created_utc": utc_now(),
        "outcome_blind": True,
        "fhd_columns_accessed": [],
        "source": {
            "flight_box_url": FLIGHT_BOX_URL,
            "flight_box_zip_sha256": sha256_bytes(zip_bytes),
            "cmr_url": CMR_URL,
            "nlcd_wcs_url": NLCD_WCS_URL,
            "tessera_manifest_sha256": sha256(MANIFEST_PATH),
        },
        "selected_sites": selected,
        "selection_evidence": evidence,
        "gate_passed": bool(gate),
        "outputs": {
            "screen": {"path": str(SCREEN_PATH.relative_to(ROOT)), "sha256": sha256(SCREEN_PATH)},
            "boundaries": {"path": str(BOUNDARY_PATH.relative_to(ROOT)), "sha256": sha256(BOUNDARY_PATH)},
        },
    }
    write_text(FREEZE_PATH, json.dumps(freeze, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "freeze_id": freeze["freeze_id"],
        "selected": selected,
        "gate_passed": bool(gate),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
