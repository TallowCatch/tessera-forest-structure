#!/usr/bin/env python3
"""Acquire and freeze the outcome-blind GEDI pass cohort for Phase 13.

FHD values are mechanically copied to separate sealed Parquet files while each source
granule is streamed. Site and fold eligibility use only coordinates, acquisition
metadata, quality flags, forest cover, and a target-availability boolean.
"""

from __future__ import annotations

import io
import hashlib
import json
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import yaml
from scipy.spatial import cKDTree
from shapely.geometry import shape


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_gedi_v3_sample as gedi  # noqa: E402
import freeze_track_assisted_splits as split_tools  # noqa: E402
import run_context_height_footprint_height_transfer as common  # noqa: E402


PROTOCOL_PATH = ROOT / "metadata/project_config_phase13_track_replication_protocol_freeze.yaml"
PHASE10_SCREEN_PATH = ROOT / "metadata/phase10_unseen_site_screen.csv"
PHASE10_FREEZE_PATH = ROOT / "metadata/phase10_unseen_site_screen_freeze.json"
BOUNDARY_PATH = ROOT / "metadata/phase13_candidate_boundaries.geojson"
CMR_PATH = ROOT / "metadata/phase13_target_free_cmr_inventory.json"
PROGRESS_PATH = ROOT / "metadata/phase13_target_free_acquisition_progress.json"
ALL_METADATA_PATH = ROOT / "data/processed/phase13_target_free_gedi_metadata.parquet"
PRIMARY_METADATA_PATH = ROOT / "data/processed/phase13_target_free_primary_forest_metadata.parquet"
SITE_PATH = ROOT / "metadata/phase13_target_free_site_summary.csv"
FOLD_PATH = ROOT / "metadata/phase13_target_free_fold_summary.csv"
FREEZE_PATH = ROOT / "metadata/phase13_target_free_gedi_freeze.json"
STAGING_DIR = ROOT / "data/raw/gedi_v3/phase13_staging"
INTERIM_METADATA_DIR = ROOT / "data/interim/phase13_gedi_metadata_exact_p1"
SEALED_OUTCOME_DIR = ROOT / "data/interim/phase13_gedi_outcomes_sealed"
FLIGHT_BOX_URL = "https://www.neonscience.org/sites/default/files/AOP_flightBoxes_0.zip"

TARGET_FREE_FIELDS = {
    "shot_number": "shot_number",
    "delta_time": "delta_time",
    "longitude": "lon_lowestmode",
    "latitude": "lat_lowestmode",
    "fhd_normal": "fhd_normal",
    "l2b_quality_flag_rel3": "l2b_quality_flag_rel3",
    "degrade_flag": "geolocation/degrade_flag",
    "solar_elevation": "geolocation/solar_elevation",
    "beam_id": "beam",
}
METADATA_COLUMNS = [
    "site_id",
    "flight_box_id",
    "granule_id",
    "cmr_concept_id",
    "collection_concept_id",
    "product",
    "product_version",
    "acquisition_datetime",
    "orbit",
    "reference_ground_track",
    "beam_name",
    "beam_power",
    "shot_number",
    "delta_time",
    "longitude",
    "latitude",
    "l2b_quality_flag_rel3",
    "degrade_flag",
    "solar_elevation",
    "beam_id",
    "source_hdf_sha512",
]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def load_protocol() -> dict[str, Any]:
    return yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))[
        "phase13_track_assisted_replication"
    ]


def official_boundaries(site_ids: list[str]) -> tuple[dict[str, Any], dict[str, Any], str]:
    response = requests.get(FLIGHT_BOX_URL, timeout=90)
    response.raise_for_status()
    archive_bytes = response.content
    with tempfile.TemporaryDirectory() as temporary:
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            archive.extractall(temporary)
        shapefile = next(Path(temporary).rglob("AOP_flightboxesAllSites.shp"))
        boundaries = gpd.read_file(shapefile).to_crs(4326)
    boundaries = (
        boundaries[
            boundaries["priority"].eq(1)
            & boundaries["sampleType"].eq("Terrestrial")
            & boundaries["siteID"].isin(site_ids)
        ]
        .dissolve(by="siteID", aggfunc="first")
        .reset_index()
        .sort_values("siteID")
    )
    if boundaries["siteID"].tolist() != sorted(site_ids):
        raise RuntimeError("Official Phase 13 boundary population is incomplete")
    write_text(BOUNDARY_PATH, boundaries.to_json(drop_id=True) + "\n")
    document = json.loads(BOUNDARY_PATH.read_text(encoding="utf-8"))
    features = {feature["properties"]["siteID"]: feature for feature in document["features"]}
    polygons = {site: shape(feature["geometry"]) for site, feature in features.items()}
    return features, polygons, sha256_bytes(archive_bytes)


def broad_candidates(protocol: dict[str, Any]) -> list[str]:
    phase10 = json.loads(PHASE10_FREEZE_PATH.read_text(encoding="utf-8"))
    expected_freeze = protocol["prerequisites"]["phase10_outcome_blind_screen"]
    if phase10["freeze_id"] != expected_freeze:
        raise RuntimeError("Unexpected Phase 10 outcome-blind screen freeze")
    screen = pd.read_csv(PHASE10_SCREEN_PATH)
    rules = protocol["target_population"]["broad_screen"]
    selected = screen[
        screen["cmr_2024_gedi_v3_granules"].ge(
            int(rules["minimum_intersecting_gedi_v3_granules"])
        )
        & screen["nlcd_2024_available"].eq(True)
        & screen["nlcd_2024_forest_fraction"].ge(
            float(rules["minimum_whole_flight_box_forest_fraction"])
        )
        & screen["tessera_inventory_complete"].eq(True)
    ]["site_id"].sort_values().tolist()
    expected = sorted(rules["expected_candidates"])
    if selected != expected:
        raise RuntimeError(f"Phase 13 broad candidate population changed: {selected}")
    return selected


def process_granules(
    session: requests.Session,
    granules: list[dict[str, Any]],
    features: dict[str, Any],
    polygons: dict[str, Any],
) -> tuple[list[Path], list[Path]]:
    progress = (
        json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
        if PROGRESS_PATH.exists()
        else {"granules": {}}
    )
    metadata_paths: list[Path] = []
    outcome_paths: list[Path] = []
    original_fields = gedi.HDF_FIELDS
    original_fill = gedi.FLOAT_FILL_FIELDS
    gedi.HDF_FIELDS = TARGET_FREE_FIELDS
    gedi.FLOAT_FILL_FIELDS = {"fhd_normal"}
    try:
        for position, entry in enumerate(granules, start=1):
            metadata_path = INTERIM_METADATA_DIR / f"{entry['granule_id']}.parquet"
            outcome_path = SEALED_OUTCOME_DIR / f"{entry['granule_id']}.parquet"
            metadata_paths.append(metadata_path)
            outcome_paths.append(outcome_path)
            if metadata_path.exists() and outcome_path.exists():
                print(f"[{position}/{len(granules)}] reuse {entry['granule_id']}", flush=True)
                continue
            hdf_path = STAGING_DIR / entry["filename"]
            print(f"[{position}/{len(granules)}] download {entry['granule_id']}", flush=True)
            hashes = gedi.download_granule(session, entry, hdf_path)
            try:
                subset = gedi.extract_granule(hdf_path, entry, features, polygons, hashes)
                if subset.empty:
                    metadata = pd.DataFrame(columns=[*METADATA_COLUMNS, "target_available"])
                    outcomes = pd.DataFrame(columns=["site_id", "shot_number", "fhd_normal"])
                else:
                    available = np.isfinite(subset["fhd_normal"]) & subset["fhd_normal"].ge(0)
                    metadata = subset[METADATA_COLUMNS].copy()
                    metadata["target_available"] = available.to_numpy(dtype=bool)
                    outcomes = subset[["site_id", "shot_number", "fhd_normal"]].copy()
                gedi.write_parquet_atomic(metadata, metadata_path)
                gedi.write_parquet_atomic(outcomes, outcome_path)
                progress["granules"][entry["granule_id"]] = {
                    "completed_at": gedi.utc_now(),
                    "source_size_bytes": entry["size_bytes"],
                    "source_sha512": hashes["sha512"],
                    "metadata_rows": int(len(metadata)),
                    "metadata_sha256": common.sha256(metadata_path),
                    "sealed_outcome_rows": int(len(outcomes)),
                    "sealed_outcome_sha256": common.sha256(outcome_path),
                    "sites": entry["sites"],
                }
                progress["updated_at"] = gedi.utc_now()
                gedi.json_dump(PROGRESS_PATH, progress)
            finally:
                if hdf_path.exists():
                    hdf_path.unlink()
            print(f"[{position}/{len(granules)}] retained {len(metadata):,} rows", flush=True)
    finally:
        gedi.HDF_FIELDS = original_fields
        gedi.FLOAT_FILL_FIELDS = original_fill
    return metadata_paths, outcome_paths


def target_free_filters(frame: pd.DataFrame, protocol: dict[str, Any]) -> pd.DataFrame:
    rules = protocol["gedi"]["quality_filters"]
    result = frame.copy()
    result["passes_target_availability"] = result["target_available"].eq(True)
    result["passes_l2b_quality"] = result["l2b_quality_flag_rel3"].eq(
        int(rules["l2b_quality_flag_rel3"])
    )
    result["passes_geolocation"] = result["degrade_flag"].eq(int(rules["degrade_flag"]))
    result["is_full_power"] = result["beam_power"].eq(str(rules["beam_power"]))
    result["passes_forest_mask"] = (
        result["nlcd_valid_fraction"].ge(float(rules["minimum_nlcd_valid_fraction"]))
        & result["nlcd_forest_fraction"].ge(
            float(rules["minimum_25m_footprint_forest_fraction"])
        )
    )
    result["passes_primary_filters"] = (
        result["passes_target_availability"]
        & result["passes_l2b_quality"]
        & result["passes_geolocation"]
        & result["is_full_power"]
        & result["passes_forest_mask"]
    )
    return result


def summarize_passes(
    primary: pd.DataFrame, protocol: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    rules = protocol["complete_pass_split"]
    buffer_m = float(rules["exclusion_buffer_m"])
    minimum_test = int(rules["minimum_test_rows"])
    minimum_local = int(rules["minimum_retained_local_rows"])
    primary = primary.copy()
    primary["pass_id"] = split_tools.pass_ids(primary)
    site_records: list[dict[str, Any]] = []
    fold_records: list[dict[str, Any]] = []
    for site, site_frame in primary.groupby("site_id", sort=True):
        site_frame = site_frame.reset_index(drop=True)
        coordinates = site_frame[["x_epsg5070", "y_epsg5070"]].to_numpy(dtype=np.float64)
        pass_ids = site_frame["pass_id"].to_numpy(dtype=str)
        eligible_folds = 0
        for held_out in sorted(site_frame["pass_id"].unique()):
            test = np.flatnonzero(pass_ids == held_out)
            local_candidates = np.flatnonzero(pass_ids != held_out)
            if len(test) and len(local_candidates):
                distance, _ = cKDTree(coordinates[test]).query(coordinates[local_candidates], k=1)
                local = local_candidates[distance >= buffer_m]
            else:
                local = local_candidates[:0]
            local_passes = int(site_frame.iloc[local]["pass_id"].nunique()) if len(local) else 0
            eligible = (
                len(test) >= minimum_test
                and len(local) >= minimum_local
                and local_passes >= int(protocol["local_reference"]["minimum_retained_passes_represented"])
            )
            eligible_folds += int(eligible)
            fold_records.append(
                {
                    "site_id": site,
                    "held_out_pass": held_out,
                    "total_site_passes": int(site_frame["pass_id"].nunique()),
                    "test_rows": len(test),
                    "local_rows_after_buffer": len(local),
                    "local_passes_after_buffer": local_passes,
                    "test_shot_sha256": split_tools.shot_hash(
                        site_frame.iloc[test]["shot_number"].to_numpy()
                    ),
                    "local_shot_sha256": split_tools.shot_hash(
                        site_frame.iloc[local]["shot_number"].to_numpy()
                    ),
                    "eligible": eligible,
                }
            )
        total_passes = int(site_frame["pass_id"].nunique())
        viable = (
            total_passes >= int(rules["minimum_total_passes_per_site"])
            and eligible_folds >= int(rules["minimum_eligible_held_out_passes_per_site"])
        )
        site_records.append(
            {
                "site_id": site,
                "primary_rows": len(site_frame),
                "total_passes": total_passes,
                "eligible_folds": eligible_folds,
                "occupied_1km_cells": int(
                    len(
                        set(
                            zip(
                                np.floor(site_frame["x_epsg5070"] / 1000).astype(int),
                                np.floor(site_frame["y_epsg5070"] / 1000).astype(int),
                                strict=True,
                            )
                        )
                    )
                ),
                "viable_replication_site": viable,
            }
        )
    sites = pd.DataFrame(site_records)
    folds = pd.DataFrame(fold_records)
    viable_sites = sorted(sites.loc[sites["viable_replication_site"], "site_id"].tolist())
    return sites, folds, viable_sites


def main() -> int:
    protected = [ALL_METADATA_PATH, PRIMARY_METADATA_PATH, SITE_PATH, FOLD_PATH, FREEZE_PATH]
    existing = [str(path.relative_to(ROOT)) for path in protected if path.exists()]
    if existing:
        raise RuntimeError(f"Phase 13 target-free outputs exist; refusing to overwrite: {existing}")
    protocol = load_protocol()
    candidates = broad_candidates(protocol)
    features, polygons, boundary_archive_sha256 = official_boundaries(candidates)
    session = requests.Session()
    session.headers.update({"User-Agent": "tessera-gedi-fhd/0.1 phase13-target-free"})
    granules, queries = gedi.query_cmr(
        session,
        str(protocol["gedi"]["collection_concept_id"]),
        int(protocol["gedi"]["year"]),
        polygons,
    )
    gedi.json_dump(CMR_PATH, {"queried_at": gedi.utc_now(), "queries": queries, "granules": granules})
    print(
        f"CMR returned {len(granules)} unique granules; "
        f"largest is {max(item['size_bytes'] for item in granules) / 1e9:.2f} GB",
        flush=True,
    )

    nlcd_records = []
    raster_paths: dict[str, Path] = {}
    for site in candidates:
        raster = ROOT / f"data/external/nlcd/Annual_NLCD_Land_Cover_2024_{site}.tif"
        raster_paths[site] = raster
        nlcd_records.append(gedi.acquire_nlcd_raster(session, site, polygons[site], 2024, raster))

    metadata_paths, outcome_paths = process_granules(session, granules, features, polygons)
    metadata_frames = [pd.read_parquet(path) for path in metadata_paths if path.exists()]
    if len(metadata_frames) != len(metadata_paths):
        raise RuntimeError("One or more Phase 13 metadata subsets are missing")
    nonempty = [frame for frame in metadata_frames if not frame.empty]
    if not nonempty:
        raise RuntimeError("Phase 13 exact flight-box extraction returned no shots")
    metadata = pd.concat(nonempty, ignore_index=True)
    metadata = metadata.drop_duplicates(["site_id", "shot_number"], keep="first")
    metadata["acquisition_datetime"] = pd.to_datetime(metadata["acquisition_datetime"], utc=True)
    metadata = metadata.sort_values(
        ["site_id", "acquisition_datetime", "beam_name", "shot_number"]
    ).reset_index(drop=True)
    rules = protocol["gedi"]["quality_filters"]
    metadata = gedi.add_nlcd_fields(
        metadata,
        raster_paths,
        set(int(value) for value in rules["nlcd_2024_forest_classes"]),
        12.5,
    )
    metadata = target_free_filters(metadata, protocol)
    primary = metadata[metadata["passes_primary_filters"]].copy()
    sites, folds, viable_sites = summarize_passes(primary, protocol)
    minimum_sites = int(protocol["complete_pass_split"]["minimum_viable_replication_sites"])
    gate_passed = len(viable_sites) >= minimum_sites

    common.write_parquet(metadata, ALL_METADATA_PATH)
    common.write_parquet(primary, PRIMARY_METADATA_PATH)
    common.write_csv(sites, SITE_PATH)
    common.write_csv(folds, FOLD_PATH)
    freeze_basis = {
        "protocol_sha256": common.sha256(PROTOCOL_PATH),
        "script_sha256": common.sha256(Path(__file__).resolve()),
        "phase10_screen_sha256": common.sha256(PHASE10_SCREEN_PATH),
        "phase10_freeze_id": protocol["prerequisites"]["phase10_outcome_blind_screen"],
        "boundary_archive_sha256": boundary_archive_sha256,
        "boundary_geojson_sha256": common.sha256(BOUNDARY_PATH),
        "broad_candidates": candidates,
        "source_granules": [
            {"granule_id": item["granule_id"], "sha512": item["checksum"], "sites": item["sites"]}
            for item in granules
        ],
        "metadata_columns_used_for_selection": [
            column for column in metadata.columns if column != "fhd_normal"
        ],
        "fhd_values_read_by_selection_or_summarization": False,
        "target_availability_boolean_used_for_missingness_qc": True,
        "sealed_outcome_files": [
            {"path": str(path.relative_to(ROOT)), "sha256": common.sha256(path)}
            for path in outcome_paths
        ],
        "nlcd": [{"site_id": item["site_id"], "sha256": item["sha256"]} for item in nlcd_records],
        "viable_sites": viable_sites,
        "eligible_folds": int(folds["eligible"].sum()),
        "gate": {
            "minimum_viable_sites": minimum_sites,
            "observed_viable_sites": len(viable_sites),
            "passed": gate_passed,
        },
    }
    freeze = {
        "freeze_id": "phase13-target-free-" + common.canonical_hash(freeze_basis)[:12],
        "created_utc": common.utc_now(),
        "status": "frozen_gate_passed" if gate_passed else "frozen_gate_failed",
        "freeze_basis": freeze_basis,
        "outputs": {
            str(path.relative_to(ROOT)): common.sha256(path)
            for path in [BOUNDARY_PATH, CMR_PATH, ALL_METADATA_PATH, PRIMARY_METADATA_PATH, SITE_PATH, FOLD_PATH]
        },
    }
    common.write_json(FREEZE_PATH, freeze)
    if list(STAGING_DIR.glob("*.h5")):
        raise RuntimeError("A full Phase 13 GEDI HDF remains in staging")
    print(
        json.dumps(
            {
                "freeze_id": freeze["freeze_id"],
                "gate_passed": gate_passed,
                "viable_sites": viable_sites,
                "eligible_folds": int(folds["eligible"].sum()),
            },
            indent=2,
        )
    )
    return 0 if gate_passed else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, requests.RequestException, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
