#!/usr/bin/env python3
"""Freeze an outcome-blind screen of every remaining official NEON forest site."""

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
PROTOCOL_PATH = ROOT / "metadata/project_config_phase10_unseen_replication_protocol_freeze.yaml"
FLIGHT_BOX_URL = "https://www.neonscience.org/sites/default/files/AOP_flightBoxes_0.zip"
CMR_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
NLCD_WCS_URL = "https://dmsdata.cr.usgs.gov/geoserver/mrlc_Land-Cover-Native_conus_year_data/ows"
NLCD_COVERAGE = "mrlc_Land-Cover-Native_conus_year_data:Land-Cover-Native_conus_year_data"
MANIFEST_PATH = ROOT / "data/external/geotessera_cache/v1/manifest.parquet"
SCREEN_PATH = ROOT / "metadata/phase10_unseen_site_screen.csv"
BOUNDARY_PATH = ROOT / "metadata/phase10_unseen_site_boundaries.geojson"
FREEZE_PATH = ROOT / "metadata/phase10_unseen_site_screen_freeze.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def required_tile_indices(bounds: tuple[float, float, float, float]) -> set[tuple[int, int]]:
    west, south, east, north = bounds
    longitudes = [value * 10 + 5 for value in range(math.floor(west * 10), math.floor(east * 10) + 1)]
    latitudes = [value * 10 + 5 for value in range(math.floor(south * 10), math.floor(north * 10) + 1)]
    return {(lon, lat) for lon in longitudes for lat in latitudes}


def nlcd_forest_fraction(session: requests.Session, geometry: Any) -> tuple[float, bool]:
    transformer = Transformer.from_crs(4326, 5070, always_xy=True)
    minx, miny, maxx, maxy = transformer.transform_bounds(*geometry.bounds, densify_pts=21)
    width = max(1, math.ceil((maxx - minx) / 30))
    height = max(1, math.ceil((maxy - miny) / 30))
    response = session.get(NLCD_WCS_URL, params={
        "service": "WCS", "version": "1.0.0", "request": "GetCoverage",
        "coverage": NLCD_COVERAGE, "crs": "EPSG:5070", "response_crs": "EPSG:5070",
        "bbox": f"{minx},{miny},{maxx},{maxy}", "width": width, "height": height,
        "format": "GeoTIFF", "time": "2024-01-01T00:00:00.000Z",
    }, timeout=180)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "").lower()
    if "tiff" not in content_type and not response.content.startswith((b"II*\x00", b"MM\x00*")):
        return float("nan"), False
    projected = gpd.GeoSeries([geometry], crs=4326).to_crs(5070).iloc[0]
    with MemoryFile(response.content) as memory, memory.open() as dataset:
        values, _ = mask(dataset, [mapping(projected)], crop=True, filled=False)
    pixels = values[0].compressed()
    return float(np.isin(pixels, [41, 42, 43]).mean()), True


def discovered_opened_sites() -> tuple[set[str], dict[str, list[str]]]:
    opened: set[str] = set()
    provenance: dict[str, list[str]] = {}
    for path in sorted((ROOT / "data/processed").glob("*.parquet")):
        parquet = pq.ParquetFile(path)
        names = set(parquet.schema_arrow.names)
        if not {"site_id", "fhd_normal"} <= names:
            continue
        sites = sorted({str(value) for value in pq.read_table(path, columns=["site_id"])["site_id"].to_pylist() if value is not None})
        opened.update(sites)
        provenance[str(path.relative_to(ROOT))] = sites
    return opened, provenance


def main() -> int:
    protected = [SCREEN_PATH, BOUNDARY_PATH, FREEZE_PATH]
    existing = [str(path.relative_to(ROOT)) for path in protected if path.exists()]
    if existing:
        raise RuntimeError(f"Phase 10 unseen-site screen exists; refusing to overwrite: {existing}")
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))["phase10_unseen_replication"]
    rules = protocol["outcome_blind_screen"]
    opened, opened_provenance = discovered_opened_sites()
    expected_opened = set(protocol["target_population"]["expected_previously_opened_sites"])
    if opened != expected_opened:
        raise RuntimeError(f"Previously opened site inventory changed: discovered={sorted(opened)} expected={sorted(expected_opened)}")

    session = requests.Session()
    session.headers.update({"User-Agent": "tessera-gedi-fhd/0.1 phase10-outcome-blind-screen"})
    response = session.get(FLIGHT_BOX_URL, timeout=90)
    response.raise_for_status()
    zip_bytes = response.content
    with tempfile.TemporaryDirectory() as temporary:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
            archive.extractall(temporary)
        shapefile = next(Path(temporary).rglob("AOP_flightboxesAllSites.shp"))
        boundaries = gpd.read_file(shapefile).to_crs(4326)
    boundaries = boundaries[
        boundaries["priority"].eq(1) & boundaries["sampleType"].eq("Terrestrial")
    ].dissolve(by="siteID", aggfunc="first").reset_index().sort_values("siteID")
    if len(boundaries) < 40 or boundaries["siteID"].duplicated().any():
        raise RuntimeError("Official Priority-1 terrestrial boundary population is incomplete")

    candidates = boundaries[~boundaries["siteID"].isin(opened)].copy()
    required_by_site = {
        row.siteID: required_tile_indices(row.geometry.bounds) for row in candidates.itertuples()
    }
    lon_values = sorted({lon for tiles in required_by_site.values() for lon, _ in tiles})
    lat_values = sorted({lat for tiles in required_by_site.values() for _, lat in tiles})
    inventory = pq.read_table(MANIFEST_PATH, columns=["lon_i", "lat_i"], filters=[
        ("version", "=", str(rules["tessera_dataset_version"])),
        ("variant", "=", str(rules["tessera_dataset_variant"])),
        ("year", "=", int(rules["tessera_year"])),
        ("lon_i", "in", lon_values), ("lat_i", "in", lat_values),
    ])
    available = set(zip(inventory["lon_i"].to_pylist(), inventory["lat_i"].to_pylist()))

    records: list[dict[str, Any]] = []
    for position, row in enumerate(candidates.itertuples(), start=1):
        bbox = ",".join(f"{value:.12f}" for value in row.geometry.bounds)
        cmr = session.get(CMR_URL, params={
            "collection_concept_id": rules["collection_concept_id"],
            "bounding_box": bbox,
            "temporal": "2024-01-01T00:00:00Z,2024-12-31T23:59:59Z",
            "page_size": 0,
        }, timeout=90)
        cmr.raise_for_status()
        required = required_by_site[row.siteID]
        fraction, nlcd_available = nlcd_forest_fraction(session, row.geometry)
        record = {
            "site_id": row.siteID,
            "domain": row.domain,
            "domain_name": row.domainName,
            "site_name": row.siteName,
            "flight_box_id": row.flightbxID,
            "cmr_2024_gedi_v3_granules": int(cmr.headers["CMR-Hits"]),
            "nlcd_2024_forest_fraction": fraction,
            "nlcd_2024_available": nlcd_available,
            "required_tessera_tiles": len(required),
            "available_tessera_tiles": len(required & available),
            "tessera_inventory_complete": required <= available,
            "bbox_wgs84": json.dumps(list(row.geometry.bounds)),
        }
        record["passes_screen"] = bool(
            record["cmr_2024_gedi_v3_granules"] >= int(rules["minimum_intersecting_granules"])
            and nlcd_available
            and fraction >= float(rules["minimum_forest_fraction"])
            and record["tessera_inventory_complete"]
        )
        records.append(record)
        print(f"[{position}/{len(candidates)}] {row.siteID}: granules={record['cmr_2024_gedi_v3_granules']} forest={fraction:.3f} pass={record['passes_screen']}", flush=True)

    screen = pd.DataFrame(records).sort_values("site_id").reset_index(drop=True)
    selected_sites = screen.loc[screen["passes_screen"], "site_id"].tolist()
    gate = len(selected_sites) >= int(rules["minimum_screened_target_sites"])
    selected_boundaries = candidates[candidates["siteID"].isin(selected_sites)].sort_values("siteID")
    write_text(SCREEN_PATH, screen.to_csv(index=False))
    write_text(BOUNDARY_PATH, selected_boundaries.to_json(drop_id=True) + "\n")
    freeze_basis = {
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "script_sha256": sha256(Path(__file__).resolve()),
        "official_priority1_terrestrial_sites": boundaries["siteID"].tolist(),
        "candidate_sites_before_screen": candidates["siteID"].tolist(),
        "previously_opened_sites": sorted(opened),
        "opened_site_discovery_read_columns": ["site_id"],
        "opened_site_provenance": opened_provenance,
        "selection_rules": rules,
        "selected_sites": selected_sites,
        "target_fhd_columns_read": [],
        "target_height_columns_read": [],
        "target_outcomes_opened": False,
        "source": {
            "flight_box_zip_sha256": sha256_bytes(zip_bytes),
            "tessera_manifest_sha256": sha256(MANIFEST_PATH),
        },
    }
    freeze_id = "phase10-unseen-screen-" + canonical_hash(freeze_basis)[:12]
    write_text(FREEZE_PATH, json.dumps({
        "freeze_id": freeze_id,
        "created_utc": utc_now(),
        "status": "frozen" if gate else "frozen_gate_failed",
        "freeze_basis": freeze_basis,
        "gate": {"minimum_selected_sites": int(rules["minimum_screened_target_sites"]), "selected_site_count": len(selected_sites), "passed": gate},
        "outputs": {
            "screen": {"path": str(SCREEN_PATH.relative_to(ROOT)), "rows": len(screen), "sha256": sha256(SCREEN_PATH)},
            "boundaries": {"path": str(BOUNDARY_PATH.relative_to(ROOT)), "rows": len(selected_boundaries), "sha256": sha256(BOUNDARY_PATH)},
        },
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"freeze_id": freeze_id, "selected_sites": selected_sites, "gate_passed": gate}, indent=2))
    return 0 if gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
