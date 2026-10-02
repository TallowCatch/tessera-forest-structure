#!/usr/bin/env python3
"""Validate the integrity and scientific filters of the frozen GEDI V3 sample."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "metadata/gedi_v3_2024_freeze.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    require(manifest["status"] == "frozen", f"Unexpected freeze status: {manifest['status']}")
    require(manifest["phase1_gate"]["passed"], "Phase 1 spatial-coverage gate failed")
    require(manifest["granule_count"] == 15, "Expected 15 unique CMR source granules")

    output_paths: dict[str, Path] = {}
    for name, record in manifest["outputs"].items():
        path = ROOT / record["path"]
        require(path.exists(), f"Missing frozen output: {path}")
        require(sha256(path) == record["sha256"], f"SHA-256 mismatch: {path}")
        output_paths[name] = path

    candidates = pd.read_parquet(output_paths["all_candidates"])
    primary = pd.read_parquet(output_paths["primary_sample"])
    freeze_id = manifest["freeze_id"]
    expected_sites = {"SOAP", "TEAK", "BART"}
    require(len(candidates) == manifest["outputs"]["all_candidates"]["rows"], "Candidate row-count mismatch")
    require(len(primary) == manifest["outputs"]["primary_sample"]["rows"], "Primary row-count mismatch")
    require(set(candidates["site_id"]) == expected_sites, "Candidate sites do not match the fixed design")
    require(set(primary["site_id"]) == expected_sites, "Primary sites do not match the fixed design")
    require(candidates["freeze_id"].eq(freeze_id).all(), "Candidate freeze_id mismatch")
    require(primary["freeze_id"].eq(freeze_id).all(), "Primary freeze_id mismatch")
    require(not candidates.duplicated(["site_id", "shot_number"]).any(), "Duplicate candidate shots")
    require(not primary.duplicated(["site_id", "shot_number"]).any(), "Duplicate primary shots")
    require(primary["passes_primary_filters"].all(), "Primary table contains rejected rows")
    require(primary["is_finite_fhd"].all(), "Primary table contains invalid FHD")
    require(primary["l2b_quality_flag_rel3"].eq(1).all(), "Primary table contains low-quality shots")
    require(primary["degrade_flag"].eq(0).all(), "Primary table contains degraded geolocation")
    require(primary["is_full_power"].all(), "Primary table contains coverage-beam shots")
    require(primary["nlcd_valid_fraction"].ge(0.999).all(), "Primary table has incomplete NLCD support")
    require(primary["nlcd_forest_fraction"].ge(0.8).all(), "Primary table violates forest threshold")
    require(primary["nlcd_center_class"].isin([41, 42, 43]).all(), "Primary center class is not forest")

    for site_id, summary in manifest["site_summary"].items():
        site_primary = primary[primary["site_id"].eq(site_id)]
        require(len(site_primary) == summary["primary_rows"], f"Primary count mismatch for {site_id}")
        require(summary["occupied_1km_cells"] >= 3, f"Insufficient spatial support for {site_id}")

    staging = ROOT / "data/raw/gedi_v3/staging"
    require(not list(staging.glob("*.h5")), "Full GEDI HDF5 files remain in staging")
    require(not list(staging.glob("*.part")), "Partial GEDI downloads remain in staging")

    print(f"VALID: {freeze_id}")
    print(f"Candidates: {len(candidates):,}")
    print(f"Primary forest sample: {len(primary):,}")
    print(primary.groupby("site_id").size().to_string())


if __name__ == "__main__":
    main()

