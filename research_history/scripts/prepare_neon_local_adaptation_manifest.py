#!/usr/bin/env python3
"""Preflight NEON access and freeze bounded lidar/field file manifests."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import yaml
from pyproj import Transformer


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/project.yaml"
GEDI_PATH = ROOT / "data/processed/gedi_v3_2024_primary_forest.parquet"
PREFLIGHT_PATH = ROOT / "metadata/phase5_neon_access_preflight.json"
LIDAR_MANIFEST_PATH = ROOT / "metadata/phase5_neon_lidar_manifest.csv"
FIELD_MANIFEST_PATH = ROOT / "metadata/phase5_neon_field_structure_manifest.csv"
FREEZE_PATH = ROOT / "metadata/phase5_neon_acquisition_manifest_freeze.json"

PRODUCT_API = "https://data.neonscience.org/api/v0/products/{product}"
DATA_API = "https://data.neonscience.org/api/v0/data/{product}/{site}/{month}"
LIDAR_PRODUCT = "DP1.30003.001"
FIELD_PRODUCT = "DP1.10098.001"
LIDAR_MONTHS = {"SOAP": "2024-06", "TEAK": "2024-06", "BART": "2024-08"}
FIELD_MONTHS = {
    "SOAP": ["2023-09", "2023-10"],
    "TEAK": ["2024-08", "2024-09"],
    "BART": ["2024-04", "2024-07", "2024-08", "2024-09", "2024-10"],
}
SITE_UTM = {"SOAP": 32611, "TEAK": 32611, "BART": 32619}
TILES_PER_SITE = 6
EDGE_BUFFER_M = 20.0
DISK_RESERVE_BYTES = 2_000_000_000
MAX_WORKING_MULTIPLIER = 6.0
LIDAR_PATTERN = re.compile(
    r"DP1_(\d{6})_(\d{7})_classified_point_cloud(?:_colorized)?\.laz$",
    re.IGNORECASE,
)
FIELD_TABLE_MARKERS = ["vst_apparentindividual", "vst_mappingandtagging", "vst_perplotperyear"]


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


def build_storage_gate(free_bytes: int, maximum_tile_bytes: int) -> dict[str, Any]:
    estimated_working_bytes = int(math.ceil(maximum_tile_bytes * MAX_WORKING_MULTIPLIER))
    required_free_bytes = estimated_working_bytes + DISK_RESERVE_BYTES
    return {
        "passed": required_free_bytes <= free_bytes,
        "free_bytes": free_bytes,
        "maximum_tile_bytes": maximum_tile_bytes,
        "estimated_maximum_working_bytes": estimated_working_bytes,
        "disk_reserve_bytes": DISK_RESERVE_BYTES,
        "required_free_bytes": required_free_bytes,
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def token_from_environment() -> str | None:
    return os.getenv("NEON_TOKEN") or os.getenv("NEON_PAT")


def product_months(session: requests.Session, product: str, site: str) -> list[str]:
    response = session.get(PRODUCT_API.format(product=product), timeout=60)
    response.raise_for_status()
    sites = response.json()["data"]["siteCodes"]
    record = next((item for item in sites if item["siteCode"] == site), None)
    return list(record["availableMonths"]) if record else []


def api_files(session: requests.Session, token: str, product: str, site: str, month: str) -> list[dict[str, Any]]:
    response = session.get(
        DATA_API.format(product=product, site=site, month=month),
        headers={"X-API-TOKEN": token},
        timeout=120,
    )
    if response.status_code == 403:
        raise RuntimeError("NEON rejected the API token or the token has expired")
    response.raise_for_status()
    return list(response.json()["data"]["files"])


def candidate_tiles(gedi: pd.DataFrame, site: str) -> pd.DataFrame:
    subset = gedi[gedi["site_id"].eq(site)].copy()
    transformer = Transformer.from_crs(4326, SITE_UTM[site], always_xy=True)
    x, y = transformer.transform(subset["longitude"].to_numpy(), subset["latitude"].to_numpy())
    subset["utm_x"] = x
    subset["utm_y"] = y
    subset["tile_easting"] = (np.floor(subset["utm_x"] / 1000) * 1000).astype(int)
    subset["tile_northing"] = (np.floor(subset["utm_y"] / 1000) * 1000).astype(int)
    offset_x = subset["utm_x"] - subset["tile_easting"]
    offset_y = subset["utm_y"] - subset["tile_northing"]
    subset = subset[
        offset_x.between(EDGE_BUFFER_M, 1000 - EDGE_BUFFER_M)
        & offset_y.between(EDGE_BUFFER_M, 1000 - EDGE_BUFFER_M)
    ]
    return (
        subset.groupby(["tile_easting", "tile_northing"], as_index=False)
        .agg(
            eligible_gedi_rows=("shot_number", "size"),
            mean_utm_x=("utm_x", "mean"),
            mean_utm_y=("utm_y", "mean"),
        )
        .sort_values(["eligible_gedi_rows", "tile_easting", "tile_northing"], ascending=[False, True, True])
        .reset_index(drop=True)
    )


def select_spread_tiles(candidates: pd.DataFrame, count: int) -> pd.DataFrame:
    if len(candidates) < count:
        raise RuntimeError(f"Only {len(candidates)} eligible lidar tiles; need {count}")
    remaining = candidates.copy()
    selected_indices: list[int] = [int(remaining.index[0])]
    while len(selected_indices) < count:
        selected = remaining.loc[selected_indices, ["mean_utm_x", "mean_utm_y"]].to_numpy(dtype=float)
        best_index = None
        best_score = None
        for index, row in remaining.drop(index=selected_indices).iterrows():
            point = row[["mean_utm_x", "mean_utm_y"]].to_numpy(dtype=float)
            minimum_distance = float(np.sqrt(((selected - point) ** 2).sum(axis=1)).min())
            score = minimum_distance * math.sqrt(float(row["eligible_gedi_rows"]))
            tie = (int(row["tile_easting"]), int(row["tile_northing"]))
            key = (score, -tie[0], -tie[1])
            if best_score is None or key > best_score:
                best_score = key
                best_index = int(index)
        if best_index is None:
            raise RuntimeError("Unable to select a spatially spread tile set")
        selected_indices.append(best_index)
    selected = remaining.loc[selected_indices].copy()
    selected["selection_order"] = range(1, len(selected) + 1)
    return selected.sort_values("selection_order")


def build_preflight(session: requests.Session, token_present: bool) -> dict[str, Any]:
    availability: dict[str, Any] = {}
    for product, requested in [(LIDAR_PRODUCT, {site: [month] for site, month in LIDAR_MONTHS.items()}), (FIELD_PRODUCT, FIELD_MONTHS)]:
        availability[product] = {}
        for site, months in requested.items():
            available = product_months(session, product, site)
            availability[product][site] = {
                "requested_months": months,
                "all_requested_available": all(month in available for month in months),
                "available_months": [month for month in available if month.startswith(("2023-", "2024-"))],
            }
    return {
        "checked_at": utc_now(),
        "token_present": token_present,
        "token_value_recorded": False,
        "availability": availability,
        "status": "ready_for_authenticated_manifest" if token_present else "blocked_missing_neon_token",
        "required_environment_variable": "NEON_TOKEN",
    }


def main() -> int:
    session = requests.Session()
    session.headers.update({"Accept": "application/json", "User-Agent": "tessera-gedi-fhd/0.1"})
    token = token_from_environment()
    preflight = build_preflight(session, token is not None)
    write_json(PREFLIGHT_PATH, preflight)
    unavailable = [
        f"{product}/{site}"
        for product, sites in preflight["availability"].items()
        for site, record in sites.items()
        if not record["all_requested_available"]
    ]
    if unavailable:
        raise RuntimeError(f"Required official NEON months are unavailable: {unavailable}")
    if token is None:
        print("NEON metadata availability passed, but downloads are blocked: NEON_TOKEN is absent.")
        print(f"Preflight written to {PREFLIGHT_PATH.relative_to(ROOT)}")
        return 2
    protected = [LIDAR_MANIFEST_PATH, FIELD_MANIFEST_PATH, FREEZE_PATH]
    existing = [str(path.relative_to(ROOT)) for path in protected if path.exists()]
    if existing:
        raise RuntimeError(f"NEON acquisition manifest exists; refusing to overwrite: {existing}")

    gedi = pd.read_parquet(GEDI_PATH)
    lidar_records: list[dict[str, Any]] = []
    tile_selection_basis: dict[str, Any] = {}
    for site, month in LIDAR_MONTHS.items():
        files = api_files(session, token, LIDAR_PRODUCT, site, month)
        available_tiles: dict[tuple[int, int], dict[str, Any]] = {}
        for item in files:
            match = LIDAR_PATTERN.search(str(item["name"]))
            if match:
                available_tiles[(int(match.group(1)), int(match.group(2)))] = item
        candidates = candidate_tiles(gedi, site)
        selected = select_spread_tiles(candidates, TILES_PER_SITE)
        tile_selection_basis[site] = selected.to_dict(orient="records")
        for row in selected.itertuples(index=False):
            tile = (int(row.tile_easting), int(row.tile_northing))
            item = available_tiles.get(tile)
            if item is None:
                raise RuntimeError(f"Selected {site} tile {tile} is absent from the official 2024 file list")
            lidar_records.append({
                "site_id": site,
                "month": month,
                "selection_order": int(row.selection_order),
                "tile_easting": tile[0],
                "tile_northing": tile[1],
                "eligible_gedi_rows": int(row.eligible_gedi_rows),
                "file_name": item["name"],
                "size_bytes": int(item["size"]),
                "crc32c": item.get("crc32c"),
                "url": item["url"],
            })
    lidar_manifest = pd.DataFrame(lidar_records).sort_values(["site_id", "selection_order"]).reset_index(drop=True)

    field_records: list[dict[str, Any]] = []
    for site, months in FIELD_MONTHS.items():
        for month in months:
            files = api_files(session, token, FIELD_PRODUCT, site, month)
            selected_files = [item for item in files if any(marker in str(item["name"]).lower() for marker in FIELD_TABLE_MARKERS)]
            if not selected_files:
                raise RuntimeError(f"No required vegetation-structure tables found for {site} {month}")
            for item in selected_files:
                field_records.append({
                    "site_id": site,
                    "month": month,
                    "file_name": item["name"],
                    "size_bytes": int(item["size"]),
                    "crc32c": item.get("crc32c"),
                    "url": item["url"],
                })
    field_manifest = pd.DataFrame(field_records).drop_duplicates(["site_id", "month", "file_name"]).sort_values(["site_id", "month", "file_name"]).reset_index(drop=True)

    free_bytes = shutil.disk_usage(ROOT).free
    maximum_tile_bytes = int(lidar_manifest["size_bytes"].max())
    storage_gate = build_storage_gate(free_bytes, maximum_tile_bytes)
    for frame, path in [(lidar_manifest, LIDAR_MANIFEST_PATH), (field_manifest, FIELD_MANIFEST_PATH)]:
        temporary = path.with_suffix(".tmp.csv")
        frame.to_csv(temporary, index=False)
        temporary.replace(path)

    freeze_basis = {
        "gedi_input_sha256": sha256(GEDI_PATH),
        "products": {"lidar": LIDAR_PRODUCT, "field_structure": FIELD_PRODUCT},
        "lidar_months": LIDAR_MONTHS,
        "field_months": FIELD_MONTHS,
        "tiles_per_site": TILES_PER_SITE,
        "edge_buffer_m": EDGE_BUFFER_M,
        "tile_selection_rule": "first_maximum_rows_then_maximum_minimum_distance_times_sqrt_rows",
        "tile_selection_basis": tile_selection_basis,
        "one_source_tile_at_a_time": True,
        "disk_reserve_bytes": DISK_RESERVE_BYTES,
        "working_size_multiplier": MAX_WORKING_MULTIPLIER,
        "storage_gate": storage_gate,
        "token_value_recorded": False,
        "script_sha256": sha256(Path(__file__)),
        "project_config_sha256": sha256(CONFIG_PATH),
    }
    freeze_id = "phase5-neon-manifest-" + canonical_hash(freeze_basis)[:12]
    freeze = {
        "freeze_id": freeze_id,
        "created_at": utc_now(),
        "status": (
            "frozen_authenticated_file_manifest_download_ready"
            if storage_gate["passed"]
            else "frozen_authenticated_file_manifest_download_blocked_storage"
        ),
        "freeze_basis": freeze_basis,
        "freeze_basis_sha256": canonical_hash(freeze_basis),
        "outputs": {
            "lidar_manifest": {"path": str(LIDAR_MANIFEST_PATH.relative_to(ROOT)), "rows": len(lidar_manifest), "sha256": sha256(LIDAR_MANIFEST_PATH), "total_size_bytes": int(lidar_manifest["size_bytes"].sum())},
            "field_manifest": {"path": str(FIELD_MANIFEST_PATH.relative_to(ROOT)), "rows": len(field_manifest), "sha256": sha256(FIELD_MANIFEST_PATH), "total_size_bytes": int(field_manifest["size_bytes"].sum())},
        },
    }
    write_json(FREEZE_PATH, freeze)
    print(lidar_manifest[["site_id", "selection_order", "tile_easting", "tile_northing", "eligible_gedi_rows", "size_bytes"]].to_string(index=False))
    print(f"\nFrozen {freeze_id}; no science files downloaded")
    if not storage_gate["passed"]:
        print(
            "Download remains blocked by the storage gate: "
            f"requires {storage_gate['required_free_bytes'] / 1e9:.2f} GB free, "
            f"found {storage_gate['free_bytes'] / 1e9:.2f} GB"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
