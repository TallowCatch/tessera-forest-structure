#!/usr/bin/env python3
"""Stream LANDFIRE annual disturbance rasters and sample frozen GEDI centres."""

from __future__ import annotations

import hashlib
import io
import json
import re
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import requests
import yaml
from pyproj import Transformer


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/project.yaml"
GEDI_PATH = ROOT / "data/processed/gedi_v3_2024_primary_forest.parquet"
OUTPUT_PATH = ROOT / "data/processed/phase5_landfire_disturbance_history.parquet"
SUMMARY_PATH = ROOT / "outputs/tables/phase5_landfire_disturbance_summary.csv"
FREEZE_PATH = ROOT / "metadata/phase5_landfire_disturbance_freeze.json"

SERVICE_ROOT = "https://lfps.usgs.gov/arcgis/rest/services/Landfire_Disturbance"
ATTRIBUTE_ZIP = "https://www.landfire.gov/sites/default/files/CSV/AllYears/DistCSV_AllYears.zip"
OBSERVATION_YEAR = 2024
YEARS = list(range(1999, 2025))

SITE_BBOXES = {
    "SOAP": (-119.2919464457, 37.0142750706, -119.1767316705, 37.1042875314),
    "TEAK": (-119.0962033206, 36.9514910125, -118.9715140645, 37.0427798538),
    "BART": (-71.3342560121, 43.9919475560, -71.2094969977, 44.0819645961),
}


def service_name(year: int) -> str:
    prefix = {
        **{value: "LF2001" for value in range(1999, 2008)},
        2008: "LF2008",
        2009: "LF2010",
        2010: "LF2010",
        2011: "LF2012",
        2012: "LF2012",
        2013: "LF2014",
        2014: "LF2014",
        2015: "LF2016",
        2016: "LF2016",
        2017: "LF2020",
        2018: "LF2020",
        2019: "LF2020",
        2020: "LF2020",
        2021: "LF2022",
        2022: "LF2022",
        2023: "LF2023",
        2024: "LF2024",
    }[year]
    return f"{prefix}_Dist{year % 100:02d}_CONUS"


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


def fetch_attribute_tables(session: requests.Session) -> tuple[dict[int, pd.DataFrame], str]:
    response = session.get(ATTRIBUTE_ZIP, timeout=120)
    response.raise_for_status()
    checksum = hashlib.sha256(response.content).hexdigest()
    tables: dict[int, pd.DataFrame] = {}
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        for name in archive.namelist():
            match = re.search(r"Dist(\d{2})_.*\.csv$", name)
            if not match:
                continue
            short_year = int(match.group(1))
            year = 1900 + short_year if short_year == 99 else 2000 + short_year
            if year not in YEARS or year in tables:
                continue
            with archive.open(name) as handle:
                table = pd.read_csv(handle, encoding="utf-8-sig")
            table.columns = [str(column).strip() for column in table.columns]
            calendar_year_column = next(
                (column for column in ["CALENDAR_YR", "DIST_YEAR", "YEAR"] if column in table),
                None,
            )
            if calendar_year_column is None:
                raise RuntimeError(f"No event-year column in LANDFIRE table {name}")
            table["NORMALIZED_CALENDAR_YEAR"] = pd.to_numeric(
                table[calendar_year_column], errors="coerce"
            ).astype("Int64")
            table["NORMALIZED_PRIMARY_SOURCE"] = table[
                "PRIMARY_SOURCE" if "PRIMARY_SOURCE" in table else "SOURCE1"
            ].astype("string")
            secondary_column = "SECONDARY_SOURCE" if "SECONDARY_SOURCE" in table else "SOURCE2"
            table["NORMALIZED_SECONDARY_SOURCE"] = table[secondary_column].astype("string")
            description_column = "DESCRIPTION" if "DESCRIPTION" in table else "DESCRIPTIO"
            table["NORMALIZED_DESCRIPTION"] = table[description_column].astype("string")
            tables[year] = table
    missing = sorted(set(YEARS) - set(tables))
    if missing:
        raise RuntimeError(f"LANDFIRE attribute archive is missing years: {missing}")
    return tables, checksum


def download_export(
    session: requests.Session,
    site: str,
    year: int,
    output: Path,
) -> dict[str, Any]:
    name = service_name(year)
    base = f"{SERVICE_ROOT}/{name}/ImageServer"
    metadata_response = session.get(base, params={"f": "json"}, timeout=60)
    metadata_response.raise_for_status()
    metadata = metadata_response.json()
    if metadata.get("pixelType") != "S16" or metadata.get("spatialReference", {}).get("wkid") != 5070:
        raise RuntimeError(f"Unexpected LANDFIRE service metadata for {name}")
    raster_response = session.get(
        f"{base}/exportImage",
        params={
            "bbox": ",".join(str(value) for value in SITE_BBOXES[site]),
            "bboxSR": 4326,
            "imageSR": 5070,
            "format": "tiff",
            "pixelType": "S16",
            "interpolation": "RSP_NearestNeighbor",
            "f": "image",
        },
        timeout=120,
    )
    raster_response.raise_for_status()
    if not raster_response.content.startswith((b"II*\x00", b"MM\x00*")):
        raise RuntimeError(f"LANDFIRE export did not return a TIFF for {site} {year}")
    output.write_bytes(raster_response.content)
    return {
        "site": site,
        "nominal_year": year,
        "service": name,
        "service_url": base,
        "download_bytes": len(raster_response.content),
    }


def main() -> int:
    protected = [OUTPUT_PATH, SUMMARY_PATH, FREEZE_PATH]
    existing = [str(path.relative_to(ROOT)) for path in protected if path.exists()]
    if existing:
        raise RuntimeError(f"Disturbance freeze exists; refusing to overwrite: {existing}")
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    protocol = config["phase5_age_and_stand_development"]["disturbance_history"]
    if list(protocol["years"]) != [1999, 2024] or protocol["retain_source_rasters"]:
        raise RuntimeError("LANDFIRE protocol must cover 1999-2024 and delete source rasters")
    gedi = pd.read_parquet(GEDI_PATH)
    if len(gedi) != 3063 or set(gedi["site_id"]) != set(SITE_BBOXES):
        raise RuntimeError("Input is not the exact frozen GEDI population")
    output = gedi[["site_id", "shot_number", "longitude", "latitude"]].copy()
    output["latest_recorded_disturbance_calendar_year"] = pd.Series(pd.NA, index=output.index, dtype="Int64")
    output["latest_recorded_disturbance_product_year"] = pd.Series(pd.NA, index=output.index, dtype="Int64")
    for column in ["disturbance_type", "disturbance_severity", "disturbance_primary_source", "disturbance_secondary_source", "disturbance_description"]:
        output[column] = pd.Series(pd.NA, index=output.index, dtype="string")
    output["disturbance_value"] = pd.Series(pd.NA, index=output.index, dtype="Int64")

    session = requests.Session()
    session.headers.update({"User-Agent": "tessera-gedi-fhd/0.1", "Accept": "application/json"})
    tables, attribute_sha256 = fetch_attribute_tables(session)
    transformer = Transformer.from_crs(4326, 5070, always_xy=True)
    inventory: list[dict[str, Any]] = []
    total_temporary_bytes = 0
    with tempfile.TemporaryDirectory(prefix="landfire-disturbance-") as temporary_directory:
        temporary_root = Path(temporary_directory)
        for nominal_year in YEARS:
            table = tables[nominal_year].copy()
            table["VALUE"] = pd.to_numeric(table["VALUE"], errors="raise").astype(int)
            lookup = table.set_index("VALUE")
            for site in SITE_BBOXES:
                raster_path = temporary_root / f"{site}_{nominal_year}.tif"
                item = download_export(session, site, nominal_year, raster_path)
                total_temporary_bytes += int(item["download_bytes"])
                site_indices = output.index[output["site_id"].eq(site)].to_numpy()
                x, y = transformer.transform(
                    output.loc[site_indices, "longitude"].to_numpy(dtype=float),
                    output.loc[site_indices, "latitude"].to_numpy(dtype=float),
                )
                with rasterio.open(raster_path) as dataset:
                    values = np.asarray([value[0] for value in dataset.sample(zip(x, y, strict=True))], dtype=int)
                    item["width"] = int(dataset.width)
                    item["height"] = int(dataset.height)
                    item["sample_resolution_x_m"] = float(abs(dataset.transform.a))
                    item["sample_resolution_y_m"] = float(abs(dataset.transform.e))
                    item["raster_crs"] = dataset.crs.to_string()
                raster_path.unlink()
                non_background = values != 0
                known = np.isin(values, lookup.index.to_numpy(dtype=int))
                invalid = non_background & ~known
                if invalid.any():
                    raise RuntimeError(f"Unmapped LANDFIRE codes at {site} {nominal_year}: {sorted(set(values[invalid]))}")
                event_positions = np.flatnonzero(non_background)
                item["non_background_sample_rows"] = int(len(event_positions))
                inventory.append(item)
                for position in event_positions:
                    row_index = site_indices[position]
                    code = int(values[position])
                    attributes = lookup.loc[code]
                    disturbance_type = str(attributes.get("DIST_TYPE", ""))
                    if disturbance_type in {"NA", "Water", "Fill-NoData"}:
                        continue
                    if pd.isna(attributes["NORMALIZED_CALENDAR_YEAR"]):
                        continue
                    calendar_year = int(attributes["NORMALIZED_CALENDAR_YEAR"])
                    current = output.at[row_index, "latest_recorded_disturbance_calendar_year"]
                    current_product = output.at[row_index, "latest_recorded_disturbance_product_year"]
                    should_replace = pd.isna(current) or calendar_year > int(current) or (
                        calendar_year == int(current) and (pd.isna(current_product) or nominal_year > int(current_product))
                    )
                    if should_replace:
                        output.at[row_index, "latest_recorded_disturbance_calendar_year"] = calendar_year
                        output.at[row_index, "latest_recorded_disturbance_product_year"] = nominal_year
                        output.at[row_index, "disturbance_type"] = disturbance_type
                        output.at[row_index, "disturbance_severity"] = str(attributes.get("SEVERITY", ""))
                        output.at[row_index, "disturbance_primary_source"] = str(attributes["NORMALIZED_PRIMARY_SOURCE"])
                        output.at[row_index, "disturbance_secondary_source"] = str(attributes["NORMALIZED_SECONDARY_SOURCE"])
                        output.at[row_index, "disturbance_description"] = str(attributes["NORMALIZED_DESCRIPTION"])
                        output.at[row_index, "disturbance_value"] = code

    output["has_recorded_disturbance_1999_2024"] = output["latest_recorded_disturbance_calendar_year"].notna()
    output["years_since_latest_recorded_disturbance"] = (
        OBSERVATION_YEAR - output["latest_recorded_disturbance_calendar_year"]
    ).astype("Int64")
    output["disturbance_time_label"] = output["years_since_latest_recorded_disturbance"].astype("string")
    output.loc[~output["has_recorded_disturbance_1999_2024"], "disturbance_time_label"] = "no_record_1999_2024"
    output["is_chronological_stand_age"] = False

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = OUTPUT_PATH.with_suffix(".tmp.parquet")
    output.to_parquet(temporary_output, index=False, compression="zstd")
    temporary_output.replace(OUTPUT_PATH)
    summary = (
        output.groupby(["site_id", "has_recorded_disturbance_1999_2024"], dropna=False)
        .agg(
            rows=("shot_number", "size"),
            minimum_years_since_recorded_disturbance=("years_since_latest_recorded_disturbance", "min"),
            median_years_since_recorded_disturbance=("years_since_latest_recorded_disturbance", "median"),
            maximum_years_since_recorded_disturbance=("years_since_latest_recorded_disturbance", "max"),
        )
        .reset_index()
    )
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_summary = SUMMARY_PATH.with_suffix(".tmp.csv")
    summary.to_csv(temporary_summary, index=False)
    temporary_summary.replace(SUMMARY_PATH)

    freeze_basis = {
        "protocol": protocol,
        "gedi_input_sha256": sha256(GEDI_PATH),
        "attribute_archive_url": ATTRIBUTE_ZIP,
        "attribute_archive_sha256": attribute_sha256,
        "service_root": SERVICE_ROOT,
        "nominal_product_years": YEARS,
        "services": {str(year): service_name(year) for year in YEARS},
        "site_bboxes": SITE_BBOXES,
        "inventory": inventory,
        "source_rasters_retained": False,
        "total_temporary_download_bytes": total_temporary_bytes,
        "script_sha256": sha256(Path(__file__)),
        "project_config_sha256": sha256(CONFIG_PATH),
    }
    freeze_id = "phase5-disturbance-" + canonical_hash(freeze_basis)[:12]
    freeze = {
        "freeze_id": freeze_id,
        "created_at": utc_now(),
        "status": "frozen_disturbance_history_not_stand_age",
        "freeze_basis": freeze_basis,
        "freeze_basis_sha256": canonical_hash(freeze_basis),
        "outputs": {
            "disturbance_history": {"path": str(OUTPUT_PATH.relative_to(ROOT)), "rows": len(output), "sha256": sha256(OUTPUT_PATH)},
            "summary": {"path": str(SUMMARY_PATH.relative_to(ROOT)), "rows": len(summary), "sha256": sha256(SUMMARY_PATH)},
        },
        "recorded_disturbance_rows": int(output["has_recorded_disturbance_1999_2024"].sum()),
        "no_record_rows": int((~output["has_recorded_disturbance_1999_2024"]).sum()),
        "valid_interpretation": "Time since latest mapped LANDFIRE disturbance within the 1999-2024 observation window.",
        "invalid_interpretation": "Chronological biological or stand age.",
    }
    write_json(FREEZE_PATH, freeze)
    print(summary.to_string(index=False))
    print(f"\nRecorded disturbance rows: {freeze['recorded_disturbance_rows']}/{len(output)}")
    print(f"Temporary rasters streamed and deleted: {total_temporary_bytes / 1_000_000:.1f} MB")
    print(f"Frozen {freeze_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
