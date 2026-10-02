#!/usr/bin/env python3
"""Acquire and freeze 2024 GEDI V3 targets for the outcome-blind site expansion."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_gedi_v3_sample as base  # noqa: E402


CONFIG_PATH = ROOT / "configs/project.yaml"
PROTOCOL_PATH = ROOT / "metadata/project_config_phase7_site_expansion_protocol_freeze.yaml"
SELECTION_PATH = ROOT / "metadata/phase7_site_selection_freeze.json"
BOUNDARY_PATH = ROOT / "metadata/neon_aop_priority1_expansion_2024.geojson"
CMR_INVENTORY_PATH = ROOT / "metadata/phase7_gedi_cmr_inventory.json"
PROGRESS_PATH = ROOT / "metadata/phase7_gedi_acquisition_progress.json"
CANDIDATE_PATH = ROOT / "data/processed/phase7_gedi_v3_all_candidates.parquet"
PRIMARY_PATH = ROOT / "data/processed/phase7_gedi_v3_primary_forest.parquet"
FREEZE_PATH = ROOT / "metadata/phase7_gedi_freeze.json"
EXCLUSION_PATH = ROOT / "metadata/phase7_gedi_exclusion_counts.csv"
INVENTORY_PATH = ROOT / "metadata/phase7_gedi_inventory.csv"


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


def viability_table(
    primary: pd.DataFrame,
    selected_sites: list[str],
    minimum_rows: int,
    minimum_cells: int,
) -> pd.DataFrame:
    records = []
    for site in selected_sites:
        frame = primary[primary["site_id"].eq(site)]
        cells = set(zip(
            np.floor(frame["x_epsg5070"] / 1000).astype(int),
            np.floor(frame["y_epsg5070"] / 1000).astype(int),
            strict=True,
        ))
        rows = len(frame)
        records.append({
            "site_id": site,
            "primary_rows": rows,
            "occupied_1km_cells": len(cells),
            "passes_minimum_rows": rows >= minimum_rows,
            "passes_minimum_cells": len(cells) >= minimum_cells,
            "viable": rows >= minimum_rows and len(cells) >= minimum_cells,
        })
    return pd.DataFrame(records)


def main() -> int:
    protected = [CANDIDATE_PATH, PRIMARY_PATH, FREEZE_PATH, EXCLUSION_PATH, INVENTORY_PATH]
    existing = [str(path.relative_to(ROOT)) for path in protected if path.exists()]
    if existing:
        raise RuntimeError(f"Phase 7 GEDI freeze exists; refusing to overwrite: {existing}")
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    frozen_protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    protocol = frozen_protocol["phase7_site_expansion"]
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    sites = list(protocol["selected_expansion_sites"])
    if sites != selection["selected_sites"] or not selection["outcome_blind"]:
        raise RuntimeError("Acquisition sites differ from the outcome-blind selection freeze")
    features, polygons = base.load_boundaries(BOUNDARY_PATH, sites)

    session = requests.Session()
    session.headers.update({"User-Agent": "tessera-gedi-fhd/0.1 phase7-gedi-expansion"})
    granules, cmr_queries = base.query_cmr(
        session,
        config["target"]["cmr_collection_concept_id"],
        int(config["target"]["primary_year"]),
        polygons,
    )
    if not granules:
        raise RuntimeError("CMR returned no expansion GEDI V3 granules")
    write_json(CMR_INVENTORY_PATH, {
        "queried_at": utc_now(), "queries": cmr_queries, "granules": granules
    })
    print(f"CMR returned {len(granules)} unique expansion granules", flush=True)

    nlcd_records = []
    raster_paths: dict[str, Path] = {}
    year = int(config["target"]["primary_year"])
    for site in sites:
        raster_path = ROOT / f"data/external/nlcd/Annual_NLCD_Land_Cover_{year}_{site}.tif"
        raster_paths[site] = raster_path
        nlcd_records.append(base.acquire_nlcd_raster(
            session, site, polygons[site], year, raster_path
        ))

    parquet_paths = base.process_granules(
        session,
        granules,
        features,
        polygons,
        ROOT / "data/raw/phase7_gedi_v3/staging",
        ROOT / "data/interim/phase7_gedi_v3_exact_p1",
        PROGRESS_PATH,
    )
    frames = [pd.read_parquet(path) for path in parquet_paths if path.exists()]
    nonempty = [frame for frame in frames if not frame.empty]
    if len(frames) != len(parquet_paths) or not nonempty:
        raise RuntimeError("One or more expansion per-granule subsets are missing")
    candidates = pd.concat(nonempty, ignore_index=True)
    candidates = candidates.drop_duplicates(["site_id", "shot_number"], keep="first")
    candidates = candidates.sort_values(
        ["site_id", "acquisition_datetime", "beam_name", "shot_number"]
    ).reset_index(drop=True)
    candidates = base.add_nlcd_fields(
        candidates,
        raster_paths,
        set(config["forest_mask"]["forest_classes"]),
        float(config["target"]["footprint_diameter_m_approx"]) / 2,
    )
    candidates = base.add_filter_fields(
        candidates, float(config["forest_mask"]["minimum_footprint_forest_fraction"])
    )
    primary_all = candidates[candidates["passes_primary_filters"]].copy()
    acquisition = protocol["acquisition"]
    viability = viability_table(
        primary_all,
        sites,
        int(acquisition["minimum_primary_rows_per_site"]),
        int(acquisition["minimum_occupied_1km_cells_per_site"]),
    )
    viable_sites = viability.loc[viability["viable"], "site_id"].tolist()
    minimum_sites = int(acquisition["minimum_viable_expansion_sites"])
    gate_passed = len(viable_sites) >= minimum_sites

    freeze_basis = {
        "selection_freeze_id": selection["freeze_id"],
        "protocol_snapshot_sha256": sha256(PROTOCOL_PATH),
        "boundary_sha256": sha256(BOUNDARY_PATH),
        "collection_concept_id": config["target"]["cmr_collection_concept_id"],
        "product_version": str(config["target"]["product_version"]),
        "year": year,
        "selected_sites": sites,
        "source_granules": [
            {"granule_id": item["granule_id"], "sha512": item["checksum"], "sites": item["sites"]}
            for item in granules
        ],
        "nlcd": [{"site_id": item["site_id"], "sha256": item["sha256"]} for item in nlcd_records],
        "filters_identical_to_original": {
            "fhd": "finite_and_nonnegative",
            "l2b_quality_flag_rel3": 1,
            "degrade_flag": 0,
            "beam_power": "HDF_group_description_equals_Full_power_beam",
            "forest_classes": config["forest_mask"]["forest_classes"],
            "minimum_footprint_forest_fraction": config["forest_mask"]["minimum_footprint_forest_fraction"],
            "minimum_nlcd_valid_fraction": 0.999,
            "gedi_footprint_radius_m": float(config["target"]["footprint_diameter_m_approx"]) / 2,
        },
        "viability_rule": acquisition,
        "viability": viability.to_dict(orient="records"),
        "viable_sites": viable_sites,
        "viability_did_not_use_model_performance": True,
    }
    freeze_id = "phase7-gedi-v3-" + canonical_hash(freeze_basis)[:12]
    frozen_candidates = candidates.copy()
    frozen_candidates.insert(0, "freeze_id", freeze_id)
    primary = frozen_candidates[
        frozen_candidates["passes_primary_filters"]
        & frozen_candidates["site_id"].isin(viable_sites)
    ].copy()
    base.write_parquet_atomic(frozen_candidates, CANDIDATE_PATH)
    base.write_parquet_atomic(primary, PRIMARY_PATH)
    exclusions = base.exclusion_table(frozen_candidates)
    exclusions.to_csv(EXCLUSION_PATH, index=False)
    inventory = base.inventory_table(frozen_candidates)
    inventory.to_csv(INVENTORY_PATH, index=False)
    outputs = {
        "all_candidates": {"path": str(CANDIDATE_PATH.relative_to(ROOT)), "rows": len(frozen_candidates), "sha256": sha256(CANDIDATE_PATH)},
        "primary_sample": {"path": str(PRIMARY_PATH.relative_to(ROOT)), "rows": len(primary), "sha256": sha256(PRIMARY_PATH)},
        "exclusion_counts": {"path": str(EXCLUSION_PATH.relative_to(ROOT)), "sha256": sha256(EXCLUSION_PATH)},
        "inventory": {"path": str(INVENTORY_PATH.relative_to(ROOT)), "sha256": sha256(INVENTORY_PATH)},
    }
    manifest = {
        "freeze_id": freeze_id,
        "created_utc": utc_now(),
        "status": "frozen" if gate_passed else "frozen_gate_failed",
        "freeze_basis": freeze_basis,
        "outputs": outputs,
        "gate": {
            "minimum_viable_expansion_sites": minimum_sites,
            "viable_site_count": len(viable_sites),
            "passed": gate_passed,
        },
    }
    write_json(FREEZE_PATH, manifest)
    retained_hdf = list((ROOT / "data/raw/phase7_gedi_v3/staging").glob("*.h5"))
    if retained_hdf:
        raise RuntimeError("One or more full expansion GEDI granules remain in staging")
    print(json.dumps({
        "freeze_id": freeze_id,
        "source_granules": len(granules),
        "viability": freeze_basis["viability"],
        "primary_rows": len(primary),
        "gate_passed": gate_passed,
    }, indent=2))
    return 0 if gate_passed else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, requests.RequestException, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
