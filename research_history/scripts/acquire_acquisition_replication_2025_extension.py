#!/usr/bin/env python3
"""Acquire the frozen 2025 extension for nonviable Phase 13 forests."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import acquire_acquisition_replication_target_free_gedi as phase13  # noqa: E402
import build_gedi_v3_sample as gedi  # noqa: E402
import run_context_height_footprint_height_transfer as common  # noqa: E402


PROTOCOL_PATH = ROOT / "metadata/project_config_phase13_temporal_extension_protocol_freeze.yaml"
PREREQUISITE_PATH = ROOT / "metadata/phase13_target_free_gedi_freeze.json"
BOUNDARY_PATH = ROOT / "metadata/phase13_2025_candidate_boundaries.geojson"
CMR_PATH = ROOT / "metadata/phase13_2025_target_free_cmr_inventory.json"
PROGRESS_PATH = ROOT / "metadata/phase13_2025_target_free_acquisition_progress.json"
ALL_METADATA_PATH = ROOT / "data/processed/phase13_2025_target_free_gedi_metadata.parquet"
PRIMARY_METADATA_PATH = ROOT / "data/processed/phase13_2025_target_free_primary_forest_metadata.parquet"
SITE_PATH = ROOT / "metadata/phase13_2025_target_free_site_summary.csv"
FOLD_PATH = ROOT / "metadata/phase13_2025_target_free_fold_summary.csv"
FREEZE_PATH = ROOT / "metadata/phase13_2025_target_free_gedi_freeze.json"
STAGING_DIR = ROOT / "data/raw/gedi_v3/phase13_2025_staging"
INTERIM_METADATA_DIR = ROOT / "data/interim/phase13_2025_gedi_metadata_exact_p1"
SEALED_OUTCOME_DIR = ROOT / "data/interim/phase13_2025_gedi_outcomes_sealed"


def load_protocol() -> dict[str, Any]:
    return yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))[
        "phase13_temporal_extension"
    ]


def configure_shared_paths() -> None:
    phase13.BOUNDARY_PATH = BOUNDARY_PATH
    phase13.PROGRESS_PATH = PROGRESS_PATH
    phase13.STAGING_DIR = STAGING_DIR
    phase13.INTERIM_METADATA_DIR = INTERIM_METADATA_DIR
    phase13.SEALED_OUTCOME_DIR = SEALED_OUTCOME_DIR


def main() -> int:
    protected = [ALL_METADATA_PATH, PRIMARY_METADATA_PATH, SITE_PATH, FOLD_PATH, FREEZE_PATH]
    existing = [str(path.relative_to(ROOT)) for path in protected if path.exists()]
    if existing:
        raise RuntimeError(f"Phase 13 2025 extension exists; refusing to overwrite: {existing}")
    protocol = load_protocol()
    prerequisite = json.loads(PREREQUISITE_PATH.read_text(encoding="utf-8"))
    if prerequisite["freeze_id"] != protocol["prerequisite_2024_target_free_freeze"]:
        raise RuntimeError("Unexpected Phase 13 2024 target-free prerequisite")
    if prerequisite["freeze_basis"]["gate"]["passed"]:
        raise RuntimeError("The 2025 extension is prohibited because the 2024 gate passed")
    existing_viable = sorted(prerequisite["freeze_basis"]["viable_sites"])
    expected_existing = sorted(
        protocol["combined_replication_gate"]["existing_2024_viable_forests"]
    )
    if existing_viable != expected_existing:
        raise RuntimeError("The frozen 2024 viable forest set changed")
    candidates = sorted(protocol["target_population"]["candidates"])
    configure_shared_paths()
    features, polygons, boundary_archive_sha256 = phase13.official_boundaries(candidates)
    session = requests.Session()
    session.headers.update({"User-Agent": "tessera-gedi-fhd/0.1 phase13-2025-target-free"})
    _, preliminary_queries = gedi.query_cmr(
        session,
        str(protocol["gedi"]["collection_concept_id"]),
        int(protocol["gedi"]["year"]),
        polygons,
    )
    minimum_granules = int(
        protocol["target_population"]["minimum_2025_intersecting_gedi_v3_granules_before_download"]
    )
    acquisition_sites = sorted(
        site for site in candidates if int(preliminary_queries[site]["hits"]) >= minimum_granules
    )
    if not acquisition_sites:
        raise RuntimeError("No frozen 2025 extension candidate has four intersecting granules")
    acquisition_features = {site: features[site] for site in acquisition_sites}
    acquisition_polygons = {site: polygons[site] for site in acquisition_sites}
    granules, queries = gedi.query_cmr(
        session,
        str(protocol["gedi"]["collection_concept_id"]),
        int(protocol["gedi"]["year"]),
        acquisition_polygons,
    )
    gedi.json_dump(
        CMR_PATH,
        {
            "queried_at": gedi.utc_now(),
            "preliminary_queries": preliminary_queries,
            "minimum_granules": minimum_granules,
            "acquisition_sites": acquisition_sites,
            "queries": queries,
            "granules": granules,
        },
    )
    print(
        f"2025 extension: {len(acquisition_sites)} sites, {len(granules)} unique granules, "
        f"{sum(item['size_bytes'] for item in granules) / 1e9:.1f} GB streamed",
        flush=True,
    )

    nlcd_records = []
    raster_paths: dict[str, Path] = {}
    for site in acquisition_sites:
        raster = ROOT / f"data/external/nlcd/Annual_NLCD_Land_Cover_2024_{site}.tif"
        raster_paths[site] = raster
        nlcd_records.append(
            gedi.acquire_nlcd_raster(
                session, site, acquisition_polygons[site], 2024, raster
            )
        )
    metadata_paths, outcome_paths = phase13.process_granules(
        session, granules, acquisition_features, acquisition_polygons
    )
    frames = [pd.read_parquet(path) for path in metadata_paths if path.exists()]
    if len(frames) != len(metadata_paths):
        raise RuntimeError("One or more 2025 target-free metadata subsets are missing")
    nonempty = [frame for frame in frames if not frame.empty]
    if not nonempty:
        raise RuntimeError("The 2025 exact flight-box extraction returned no shots")
    metadata = pd.concat(nonempty, ignore_index=True)
    metadata = metadata.drop_duplicates(["site_id", "shot_number"], keep="first")
    metadata["acquisition_datetime"] = pd.to_datetime(metadata["acquisition_datetime"], utc=True)
    metadata["target_year"] = 2025
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
    metadata = phase13.target_free_filters(metadata, protocol)
    primary = metadata[metadata["passes_primary_filters"]].copy()
    sites, folds, viable_sites = phase13.summarize_passes(primary, protocol)
    combined_sites = sorted(set(existing_viable) | set(viable_sites))
    combined_minimum = int(protocol["combined_replication_gate"]["minimum_unique_forests"])
    gate_passed = bool(viable_sites) and len(combined_sites) >= combined_minimum

    common.write_parquet(metadata, ALL_METADATA_PATH)
    common.write_parquet(primary, PRIMARY_METADATA_PATH)
    common.write_csv(sites, SITE_PATH)
    common.write_csv(folds, FOLD_PATH)
    freeze_basis = {
        "protocol_sha256": common.sha256(PROTOCOL_PATH),
        "script_sha256": common.sha256(Path(__file__).resolve()),
        "prerequisite_2024_freeze_id": prerequisite["freeze_id"],
        "boundary_archive_sha256": boundary_archive_sha256,
        "boundary_geojson_sha256": common.sha256(BOUNDARY_PATH),
        "candidate_sites": candidates,
        "preliminary_cmr_hits": {
            site: int(preliminary_queries[site]["hits"]) for site in candidates
        },
        "acquisition_sites": acquisition_sites,
        "source_granules": [
            {"granule_id": item["granule_id"], "sha512": item["checksum"], "sites": item["sites"]}
            for item in granules
        ],
        "target_year": 2025,
        "forest_mask_year": 2024,
        "fhd_values_read_by_selection_or_summarization": False,
        "target_availability_boolean_used_for_missingness_qc": True,
        "sealed_outcome_files": [
            {"path": str(path.relative_to(ROOT)), "sha256": common.sha256(path)}
            for path in outcome_paths
        ],
        "nlcd": [{"site_id": item["site_id"], "sha256": item["sha256"]} for item in nlcd_records],
        "viable_2025_sites": viable_sites,
        "eligible_2025_folds": int(folds["eligible"].sum()) if not folds.empty else 0,
        "combined_unique_sites": combined_sites,
        "gate": {
            "minimum_combined_unique_sites": combined_minimum,
            "observed_combined_unique_sites": len(combined_sites),
            "require_new_2025_site": True,
            "observed_new_2025_sites": len(viable_sites),
            "passed": gate_passed,
        },
    }
    freeze = {
        "freeze_id": "phase13-2025-target-free-" + common.canonical_hash(freeze_basis)[:12],
        "created_utc": common.utc_now(),
        "status": "frozen_combined_gate_passed" if gate_passed else "frozen_combined_gate_failed",
        "freeze_basis": freeze_basis,
        "outputs": {
            str(path.relative_to(ROOT)): common.sha256(path)
            for path in [BOUNDARY_PATH, CMR_PATH, ALL_METADATA_PATH, PRIMARY_METADATA_PATH, SITE_PATH, FOLD_PATH]
        },
    }
    common.write_json(FREEZE_PATH, freeze)
    if list(STAGING_DIR.glob("*.h5")):
        raise RuntimeError("A full 2025 GEDI HDF remains in staging")
    print(
        json.dumps(
            {
                "freeze_id": freeze["freeze_id"],
                "gate_passed": gate_passed,
                "viable_2025_sites": viable_sites,
                "combined_unique_sites": combined_sites,
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
