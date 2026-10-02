#!/usr/bin/env python3
"""Stream TESSERA tiles and align them to viable Phase 7 GEDI expansion sites."""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_tessera_alignment as base  # noqa: E402


CONFIG_PATH = ROOT / "configs/project.yaml"
PROTOCOL_PATH = ROOT / "metadata/project_config_phase7_site_expansion_protocol_freeze.yaml"
GEDI_PATH = ROOT / "data/processed/phase7_gedi_v3_primary_forest.parquet"
GEDI_FREEZE_PATH = ROOT / "metadata/phase7_gedi_freeze.json"
OUTPUT_PATH = ROOT / "data/processed/phase7_tessera_aligned_expansion.parquet"
MISSINGNESS_PATH = ROOT / "metadata/phase7_tessera_missingness.csv"
TILE_MANIFEST_PATH = ROOT / "metadata/phase7_tessera_tile_manifest.json"
FREEZE_PATH = ROOT / "metadata/phase7_tessera_alignment_freeze.json"
STREAM_ROOT = ROOT / "data/interim/phase7_tessera_stream"


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


def main() -> int:
    protected = [OUTPUT_PATH, MISSINGNESS_PATH, TILE_MANIFEST_PATH, FREEZE_PATH]
    existing = [str(path.relative_to(ROOT)) for path in protected if path.exists()]
    if existing:
        raise RuntimeError(f"Phase 7 TESSERA alignment exists; refusing to overwrite: {existing}")
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))["phase7_site_expansion"]
    gedi_freeze = json.loads(GEDI_FREEZE_PATH.read_text(encoding="utf-8"))
    viable_sites = list(gedi_freeze["freeze_basis"]["viable_sites"])
    if len(viable_sites) < int(protocol["acquisition"]["minimum_viable_expansion_sites"]):
        raise RuntimeError("GEDI expansion did not pass its frozen site viability gate")
    targets = pd.read_parquet(GEDI_PATH)
    if set(targets["site_id"].unique()) != set(viable_sites):
        raise RuntimeError("GEDI expansion table differs from its viable site freeze")
    predictors = config["predictors"]
    if (str(predictors["dataset_version"]), str(predictors["dataset_variant"])) != ("1.0", "vultr"):
        raise RuntimeError("Expansion must use the original TESSERA 1.0/vultr release")

    selected_manifest, selected_landmasks = base.ensure_registry("1.0")
    radius = float(config["target"]["footprint_diameter_m_approx"]) / 2
    aligned_frames = []
    tile_records_all: list[dict[str, Any]] = []
    total_tiles = 0
    if STREAM_ROOT.exists():
        shutil.rmtree(STREAM_ROOT)

    try:
        for position, site in enumerate(viable_sites, start=1):
            site_targets = targets[targets["site_id"].eq(site)].copy()
            base.EXPECTED_SITES = {site}
            tiles, by_site = base.required_tiles(site_targets, radius)
            selected_rows = base.lookup_embedding_rows(
                selected_manifest,
                tiles,
                int(predictors["embedding_year"]),
                "1.0",
                "vultr",
            )
            landmask_rows = base.lookup_landmask_rows(selected_landmasks, tiles)
            base.assert_complete_inventory(selected_rows, landmask_rows, tiles)
            print(
                f"[{position}/{len(viable_sites)}] {site}: stream {len(tiles)} TESSERA tiles",
                flush=True,
            )
            base.check_storage(
                STREAM_ROOT,
                selected_rows,
                landmask_rows,
                int(predictors["embedding_year"]),
            )
            tile_records = base.download_tiles(
                STREAM_ROOT,
                selected_rows,
                landmask_rows,
                int(predictors["embedding_year"]),
                config,
            )
            loaded = base.load_tiles(tile_records)
            aligned, _ = base.align_targets(site_targets, loaded, config)
            if not aligned["passes_tessera_alignment"].all():
                raise RuntimeError(f"One or more {site} footprints failed TESSERA alignment")
            aligned_frames.append(aligned)
            total_tiles += len(tile_records)
            for record in tile_records:
                provenance = dict(record)
                provenance["site_id"] = site
                provenance["retained_after_alignment"] = False
                tile_records_all.append(provenance)
            del loaded
            gc.collect()
            shutil.rmtree(STREAM_ROOT)
            print(f"[{position}/{len(viable_sites)}] {site}: aligned and source tiles deleted", flush=True)
    finally:
        if STREAM_ROOT.exists():
            shutil.rmtree(STREAM_ROOT)

    aligned = pd.concat(aligned_frames, ignore_index=True)
    missingness = base.missingness_table(aligned)
    if len(aligned) != len(targets) or not aligned["passes_tessera_alignment"].all():
        raise RuntimeError("Expansion alignment population is incomplete")
    freeze_basis = {
        "gedi_freeze_id": gedi_freeze["freeze_id"],
        "gedi_primary_sha256": sha256(GEDI_PATH),
        "protocol_snapshot_sha256": sha256(PROTOCOL_PATH),
        "alignment_script_sha256": sha256(Path(__file__).resolve()),
        "geotessera_version": importlib.metadata.version("geotessera"),
        "registry_manifest_sha256": sha256(selected_manifest),
        "landmask_manifest_sha256": sha256(selected_landmasks),
        "dataset_version": str(predictors["dataset_version"]),
        "dataset_variant": str(predictors["dataset_variant"]),
        "embedding_year": int(predictors["embedding_year"]),
        "sites": viable_sites,
        "site_rows": {
            site: int(count) for site, count in aligned.groupby("site_id").size().items()
        },
        "site_tile_download_counts": {
            site: sum(record["site_id"] == site for record in tile_records_all)
            for site in viable_sites
        },
        "streamed_tile_downloads": total_tiles,
        "expansion_tiles_retained": False,
        "alignment": {
            "footprint_radius_m": radius,
            "primary": "pixel_area_weighted_mean_inside_circular_footprint",
            "gaussian_sigma_m": config["alignment"]["gaussian_sigma_m"],
            "minimum_valid_footprint_fraction": config["alignment"]["minimum_valid_footprint_fraction"],
            "tile_edge_rule": "clip_pixel_overlap_to_projected_0.1_degree_tile_domain",
        },
    }
    alignment_id = "phase7-tessera-align-" + canonical_hash(freeze_basis)[:12]
    aligned.insert(0, "tessera_alignment_id", alignment_id)
    base.write_parquet_atomic(aligned, OUTPUT_PATH)
    missingness.to_csv(MISSINGNESS_PATH, index=False)
    write_json(TILE_MANIFEST_PATH, {
        "created_utc": utc_now(),
        "dataset_version": predictors["dataset_version"],
        "dataset_variant": predictors["dataset_variant"],
        "year": predictors["embedding_year"],
        "tile_download_count": total_tiles,
        "unique_tile_count": len({record["tile"] for record in tile_records_all}),
        "tiles": tile_records_all,
        "local_expansion_tiles_deleted": True,
    })
    manifest = {
        "alignment_id": alignment_id,
        "created_utc": utc_now(),
        "status": "frozen",
        "freeze_basis": freeze_basis,
        "outputs": {
            "aligned_expansion": {"path": str(OUTPUT_PATH.relative_to(ROOT)), "rows": len(aligned), "sha256": sha256(OUTPUT_PATH)},
            "missingness": {"path": str(MISSINGNESS_PATH.relative_to(ROOT)), "sha256": sha256(MISSINGNESS_PATH)},
            "tile_manifest": {"path": str(TILE_MANIFEST_PATH.relative_to(ROOT)), "sha256": sha256(TILE_MANIFEST_PATH)},
        },
    }
    write_json(FREEZE_PATH, manifest)
    if STREAM_ROOT.exists() or list((ROOT / "data/interim").rglob("phase7_tessera_stream/**/*.npy")):
        raise RuntimeError("Expansion TESSERA stream files were retained")
    print(json.dumps({
        "alignment_id": alignment_id,
        "rows": len(aligned),
        "sites": viable_sites,
        "site_rows": freeze_basis["site_rows"],
        "tile_downloads": total_tiles,
        "expansion_tiles_retained": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
