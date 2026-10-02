#!/usr/bin/env python3
"""Select the Dutch replication site without inspecting AHN4 outcomes."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
import yaml
from rasterio.mask import mask
from rasterio.features import rasterize
from rasterio.windows import Window, from_bounds


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/ahn4_replication_ahn4_deciduous.yaml"


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "phase26_ahn4_deciduous"
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dominant_habitat(table: pd.DataFrame) -> pd.DataFrame:
    required = {"SITECODE", "HABITATCODE", "DESCRIPTION", "PERCENTAGECOVER"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise RuntimeError(f"Natura habitat table is missing columns: {missing}")
    ordered = table.sort_values(
        ["SITECODE", "PERCENTAGECOVER", "HABITATCODE"],
        ascending=[True, False, True],
    )
    return ordered.drop_duplicates("SITECODE")


def fetch_top10nl(
    site_wgs84: gpd.GeoDataFrame, endpoint: str, cache_path: Path
) -> gpd.GeoDataFrame:
    if cache_path.exists():
        frame = gpd.read_file(cache_path)
        frame.attrs["pages"] = int(frame.attrs.get("pages", 0))
        return frame
    bounds = site_wgs84.total_bounds
    url: str | None = endpoint
    parameters: dict[str, Any] | None = {
        "f": "json",
        "bbox": ",".join(f"{value:.10f}" for value in bounds),
        "limit": 1000,
    }
    features: list[dict[str, Any]] = []
    pages = 0
    while url is not None:
        response = requests.get(url, params=parameters, timeout=120)
        response.raise_for_status()
        page = response.json()
        features.extend(page.get("features", []))
        url = next(
            (link["href"] for link in page.get("links", []) if link.get("rel") == "next"),
            None,
        )
        parameters = None
        pages += 1
        if pages > 100:
            raise RuntimeError("TOP10NL pagination exceeded 100 pages")
    if not features:
        raise RuntimeError("TOP10NL returned no terrain features for the selected site")
    frame = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    if "id" in frame.columns:
        frame = frame.drop_duplicates("id").reset_index(drop=True)
    frame.attrs["pages"] = pages
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_file(cache_path, driver="GeoJSON")
    return frame


def broadleaf_for_site(
    site: gpd.GeoDataFrame,
    endpoint: str,
    land_use_field: str,
    broadleaf_value: str,
    cache_path: Path,
    analysis_crs: str,
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    top10 = fetch_top10nl(site.to_crs("EPSG:4326"), endpoint, cache_path)
    pages = int(top10.attrs.get("pages", 0))
    broadleaf = top10[top10[land_use_field] == broadleaf_value].to_crs(analysis_crs)
    broadleaf = gpd.clip(broadleaf, site[["geometry"]], keep_geom_type=True)
    broadleaf = broadleaf[~broadleaf.geometry.is_empty].copy()
    area = float(broadleaf.geometry.area.sum() / 1_000_000.0)
    return broadleaf, {
        "site_code": str(site.iloc[0]["SITECODE"]),
        "site_name": str(site.iloc[0]["SITENAME"]),
        "site_area_km2": float(site.iloc[0]["area_km2"]),
        "dominant_n16_percent": float(site.iloc[0]["PERCENTAGECOVER"]),
        "features_in_bbox": int(len(top10)),
        "broadleaf_features_after_clip": int(len(broadleaf)),
        "broadleaf_area_km2": area,
        "api_pages": pages,
    }


def flight_dates(path: Path, site: gpd.GeoDataFrame) -> tuple[list[int], int]:
    with rasterio.open(path) as dataset:
        local = site.to_crs(dataset.crs)
        values, _ = mask(dataset, local.geometry, crop=True, filled=False)
        dates = values[0].compressed().astype(np.int64)
        if dataset.nodata is not None:
            dates = dates[dates != int(dataset.nodata)]
    return sorted(int(value) for value in np.unique(dates)), int(len(dates))


def candidate_support(
    flight_path: Path,
    broadleaf: gpd.GeoDataFrame,
    required_fraction: float,
    block_size_m: int,
) -> dict[str, int]:
    with rasterio.open(flight_path) as dataset:
        forest = broadleaf.to_crs(dataset.crs)
        raw = from_bounds(*forest.total_bounds, transform=dataset.transform)
        row_start = (int(np.floor(raw.row_off)) // 5) * 5
        column_start = (int(np.floor(raw.col_off)) // 5) * 5
        row_stop = ((int(np.ceil(raw.row_off + raw.height)) + 4) // 5) * 5
        column_stop = ((int(np.ceil(raw.col_off + raw.width)) + 4) // 5) * 5
        window = Window(
            column_start,
            row_start,
            column_stop - column_start,
            row_stop - row_start,
        )
        transform = dataset.window_transform(window)
        forest_mask = rasterize(
            ((geometry, 1) for geometry in forest.geometry),
            out_shape=(int(window.height), int(window.width)),
            transform=transform,
            fill=0,
            all_touched=False,
            dtype="uint8",
        )
    fractions = forest_mask.reshape(
        forest_mask.shape[0] // 5,
        5,
        forest_mask.shape[1] // 5,
        5,
    ).mean(axis=(1, 3))
    local_rows, local_columns = np.where(fractions >= required_fraction)
    x = transform.c + (local_columns * 5 + 2.5) * transform.a
    y = transform.f + (local_rows * 5 + 2.5) * transform.e
    blocks = set(zip(np.floor(x / block_size_m), np.floor(y / block_size_m)))
    return {
        "candidate_50m_units": int(len(local_rows)),
        "fully_broadleaf_50m_units": int(np.sum(fractions == 1.0)),
        "spatial_blocks": int(len(blocks)),
    }


def run() -> None:
    config = load_config()
    sources = config["sources"]
    study = config["study"]
    habitat_path = ROOT / str(sources["natura_habitat_table"])
    boundary_path = ROOT / str(sources["natura_site_boundaries"])
    flight_path = ROOT / str(sources["ahn4_flighttime"])

    habitats = pd.read_csv(habitat_path)
    sites = gpd.read_file(boundary_path).to_crs(study["analysis_crs"])
    sites["area_km2"] = sites.geometry.area / 1_000_000.0
    dominant = dominant_habitat(habitats)
    candidates = sites.merge(
        dominant[["SITECODE", "HABITATCODE", "DESCRIPTION", "PERCENTAGECOVER"]],
        on="SITECODE",
        how="left",
    )
    candidates = candidates[candidates["HABITATCODE"] == "N16"].copy()
    candidates = candidates.sort_values(
        ["area_km2", "SITECODE"], ascending=[False, True]
    ).reset_index(drop=True)
    if candidates.empty:
        raise RuntimeError("No Natura 2000 site has dominant N16 habitat")
    cache_directory = ROOT / "data/external/ahn4_selection/top10nl_candidates"
    candidate_records: list[dict[str, Any]] = []
    broadleaf_by_code: dict[str, gpd.GeoDataFrame] = {}
    for row in candidates.itertuples(index=False):
        local = candidates[candidates["SITECODE"] == row.SITECODE].iloc[[0]].copy()
        cache_path = cache_directory / f"{row.SITECODE}.geojson"
        broadleaf, record = broadleaf_for_site(
            local,
            str(sources["top10nl_items"]),
            str(sources["top10nl_land_use_field"]),
            str(sources["top10nl_broadleaf_value"]),
            cache_path,
            str(study["analysis_crs"]),
        )
        broadleaf_by_code[str(row.SITECODE)] = broadleaf
        dates, dated_cells = flight_dates(flight_path, local)
        years = sorted({int(value // 10000) for value in dates})
        record["ahn4_dates_yyyymmdd"] = dates
        record["ahn4_years"] = years
        record["single_acquisition_year"] = len(years) == 1
        record["dated_10m_cells"] = dated_cells
        candidate_records.append(record)
        print(
            f"candidate {row.SITECODE}: {record['broadleaf_area_km2']:.3f} km2 broadleaf",
            flush=True,
        )
    candidate_table = pd.DataFrame(candidate_records)
    eligible_table = candidate_table[candidate_table["single_acquisition_year"]].copy()
    eligible_table = eligible_table.sort_values(
        ["broadleaf_area_km2", "site_code"], ascending=[False, True]
    )
    if eligible_table.empty:
        raise RuntimeError("No N16 candidate was acquired within one AHN4 calendar year")
    selected_code = str(eligible_table.iloc[0]["site_code"])
    selected = candidates[candidates["SITECODE"] == selected_code].iloc[[0]].copy()
    code = str(selected.iloc[0]["SITECODE"])
    name = str(selected.iloc[0]["SITENAME"])
    expected_code = study.get("expected_site_code")
    expected_name = study.get("expected_site_name")
    if expected_code is not None and code != str(expected_code):
        raise RuntimeError(f"Outcome-blind site code changed: {code}")
    if expected_name is not None and name != str(expected_name):
        raise RuntimeError(f"Outcome-blind site name changed: {name}")

    selected_record = eligible_table.iloc[0].to_dict()
    minimum_area = float(config["population"]["minimum_broadleaf_area_km2"])
    if float(selected_record["broadleaf_area_km2"]) < minimum_area:
        raise RuntimeError(
            f"Largest eligible broadleaf area is below {minimum_area:.1f} km2"
        )

    dates = [int(value) for value in selected_record["ahn4_dates_yyyymmdd"]]
    years = [int(value) for value in selected_record["ahn4_years"]]
    dated_cells = int(selected_record["dated_10m_cells"])
    expected_dates = study.get("expected_ahn4_dates")
    if expected_dates is not None and dates != [int(value) for value in expected_dates]:
        raise RuntimeError(f"AHN4 acquisition dates changed: {dates}")
    if len(years) != 1:
        raise RuntimeError(f"Selected site spans multiple AHN4 years: {years}")
    if years[0] != int(config["tessera"]["year"]):
        raise RuntimeError(
            f"Frozen TESSERA year {config['tessera']['year']} does not match AHN4 {years[0]}"
        )

    broadleaf = broadleaf_by_code[code]
    if broadleaf.empty:
        raise RuntimeError("TOP10NL contains no broadleaf polygons in the selected site")
    broadleaf_area = float(broadleaf.geometry.area.sum() / 1_000_000.0)
    support = candidate_support(
        flight_path,
        broadleaf,
        float(config["population"]["required_broadleaf_fraction"]),
        int(config["evaluation"]["spatial_block_m"]),
    )
    if support["candidate_50m_units"] < 500 or support["spatial_blocks"] < 3:
        raise RuntimeError(f"Selected site has insufficient spatial support: {support}")

    selected_output = ROOT / str(config["outputs"]["selected_site"])
    selected_output.parent.mkdir(parents=True, exist_ok=True)
    if selected_output.exists():
        selected_output.unlink()
    selected.to_file(selected_output, layer="natura_site", driver="GPKG")
    broadleaf.to_file(selected_output, layer="top10nl_broadleaf", driver="GPKG")

    collection_response = requests.get(
        str(sources["top10nl_collection"]), params={"f": "json"}, timeout=120
    )
    collection_response.raise_for_status()
    collection = collection_response.json()
    updated = next(
        (
            link.get("updated")
            for link in collection.get("links", [])
            if link.get("rel") == "self"
        ),
        None,
    )
    freeze = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_status": "frozen_before_ahn4_outcome_inspection",
        "selection_rule": str(study["selection_rule"]),
        "candidate_count": int(len(candidates)),
        "selected_site": {
            "code": code,
            "name": name,
            "area_km2": float(selected.iloc[0]["area_km2"]),
            "dominant_habitat_code": str(selected.iloc[0]["HABITATCODE"]),
            "dominant_habitat_description": str(selected.iloc[0]["DESCRIPTION"]),
            "dominant_habitat_percent": float(selected.iloc[0]["PERCENTAGECOVER"]),
        },
        "pixel_mask": {
            "source": str(sources["top10nl_collection"]),
            "source_updated": updated,
            "land_use_field": str(sources["top10nl_land_use_field"]),
            "land_use_value": str(sources["top10nl_broadleaf_value"]),
            "features_in_bbox": int(selected_record["features_in_bbox"]),
            "broadleaf_features_after_clip": int(
                selected_record["broadleaf_features_after_clip"]
            ),
            "broadleaf_area_km2": broadleaf_area,
            "api_pages": int(selected_record["api_pages"]),
            **support,
        },
        "candidate_broadleaf_areas": candidate_table.sort_values(
            ["broadleaf_area_km2", "site_code"], ascending=[False, True]
        ).to_dict(orient="records"),
        "ahn4": {
            "acquisition_dates_yyyymmdd": dates,
            "dated_10m_cells_in_site": dated_cells,
            "selected_tessera_year": years[0],
        },
        "inputs": {
            str(habitat_path.relative_to(ROOT)): sha256(habitat_path),
            str(boundary_path.relative_to(ROOT)): sha256(boundary_path),
            str(flight_path.relative_to(ROOT)): sha256(flight_path),
            str(CONFIG_PATH.relative_to(ROOT)): sha256(CONFIG_PATH),
        },
        "selected_site_sha256": sha256(selected_output),
    }
    freeze_path = ROOT / str(config["outputs"]["selection_freeze"])
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n")
    print(json.dumps(freeze, indent=2), flush=True)


if __name__ == "__main__":
    run()
