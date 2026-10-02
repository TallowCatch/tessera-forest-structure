#!/usr/bin/env python3
"""Stream TESSERA tiles and align them to the frozen Phase 13 footprints."""

from __future__ import annotations

import base64
import gc
import importlib.metadata
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import botocore.session
import awscrt.checksums
import pandas as pd
import yaml
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ConnectionClosedError, EndpointConnectionError, ReadTimeoutError
import geotessera.registry as tessera_registry


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_tessera_alignment as base  # noqa: E402
import run_context_height_footprint_height_transfer as common  # noqa: E402


CONFIG_PATH = ROOT / "configs/project.yaml"
PROTOCOL_PATH = ROOT / "metadata/project_config_phase13_temporal_extension_protocol_freeze.yaml"
GEDI_PATH = ROOT / "data/processed/phase13_combined_target_free_primary_metadata.parquet"
GEDI_FREEZE_PATH = ROOT / "metadata/phase13_combined_target_free_freeze.json"
OUTPUT_PATH = ROOT / "data/processed/phase13_tessera_aligned.parquet"
MISSINGNESS_PATH = ROOT / "metadata/phase13_tessera_missingness.csv"
TILE_MANIFEST_PATH = ROOT / "metadata/phase13_tessera_tile_manifest.json"
FREEZE_PATH = ROOT / "metadata/phase13_tessera_alignment_freeze.json"
STREAM_ROOT = ROOT / "data/interim/phase13_tessera_stream"
DOWNLOAD_ATTEMPTS = 4
PARALLEL_RANGE_COUNT = 4


def configure_tessera_downloads() -> None:
    """Use a long read timeout for the unusually slow public S3 stream."""
    tessera_registry._S3_CLIENTS["us-west-2"] = (  # noqa: SLF001
        botocore.session.get_session().create_client(
            "s3",
            region_name="us-west-2",
            config=Config(
                signature_version=UNSIGNED,
                connect_timeout=30,
                read_timeout=900,
                retries={"mode": "standard", "total_max_attempts": 8},
            ),
        )
    )


def download_with_retry(
    url: str,
    progress_callback=None,
    cache_path: Path | None = None,
) -> str:
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            return tessera_registry.download_file_to_temp(
                url,
                progress_callback=progress_callback,
                cache_path=cache_path,
            )
        except (ConnectionClosedError, EndpointConnectionError, ReadTimeoutError):
            if attempt == DOWNLOAD_ATTEMPTS:
                raise
            delay = 15 * attempt
            print(
                f"TESSERA stream interrupted; retry {attempt + 1}/{DOWNLOAD_ATTEMPTS} "
                f"in {delay}s",
                flush=True,
            )
            time.sleep(delay)


def crc64nvme(path: Path) -> int:
    checksum = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            checksum = awscrt.checksums.crc64nvme(chunk, checksum)
    return checksum


def remote_object_metadata(url: str) -> tuple[int, int]:
    response = subprocess.run(
        ["curl", "--fail", "--silent", "--show-error", "--head", "--location",
         "--header", "x-amz-checksum-mode: ENABLED", url],
        check=True,
        capture_output=True,
        text=True,
    )
    headers: dict[str, str] = {}
    for line in response.stdout.splitlines():
        key, separator, value = line.partition(":")
        if separator:
            headers[key.strip().lower()] = value.strip()
    size = int(headers["content-length"])
    encoded_checksum = headers["x-amz-checksum-crc64nvme"]
    checksum = int.from_bytes(base64.b64decode(encoded_checksum), "big")
    return size, checksum


def download_resumable(
    url: str,
    progress_callback=None,
    cache_path: Path | None = None,
) -> str:
    if cache_path is None:
        return download_with_retry(url, progress_callback=progress_callback)
    if cache_path.exists():
        return str(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    partial = cache_path.with_name(f".{cache_path.name}.partial")
    expected_size, expected_checksum = remote_object_metadata(url)
    boundaries = [
        expected_size * index // PARALLEL_RANGE_COUNT
        for index in range(PARALLEL_RANGE_COUNT + 1)
    ]
    parts = [
        cache_path.with_name(f".{cache_path.name}.part{index:02d}")
        for index in range(PARALLEL_RANGE_COUNT)
    ]
    if partial.exists():
        first_size = boundaries[1] - boundaries[0]
        if not parts[0].exists() and partial.stat().st_size <= first_size:
            partial.replace(parts[0])
        else:
            partial.unlink()

    jobs: list[tuple[subprocess.Popen[str], Path, Path, int]] = []
    for index, part in enumerate(parts):
        start = boundaries[index]
        end = boundaries[index + 1] - 1
        part_size = end - start + 1
        if part.exists() and part.stat().st_size > part_size:
            part.unlink()
        existing = part.stat().st_size if part.exists() else 0
        if existing == part_size:
            continue
        continuation = part.with_suffix(part.suffix + ".download")
        if continuation.exists():
            continuation.unlink()
        request_start = start + existing
        process = subprocess.Popen(
            [
                "curl", "--fail", "--silent", "--show-error", "--location",
                "--range", f"{request_start}-{end}", "--retry", "8",
                "--retry-all-errors", "--retry-delay", "5",
                "--connect-timeout", "30", "--output", str(continuation), url,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        jobs.append((process, part, continuation, part_size - existing))

    failures: list[str] = []
    for process, part, continuation, expected_continuation_size in jobs:
        _, error = process.communicate()
        if process.returncode != 0:
            failures.append(error.strip() or f"curl exited {process.returncode}")
            continue
        if continuation.stat().st_size != expected_continuation_size:
            failures.append(f"wrong ranged size for {continuation.name}")
            continue
        with part.open("ab") as destination, continuation.open("rb") as source:
            shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
        continuation.unlink()
    if failures:
        raise RuntimeError("TESSERA ranged download failed: " + "; ".join(failures))
    if any(
        part.stat().st_size != boundaries[index + 1] - boundaries[index]
        for index, part in enumerate(parts)
    ):
        raise RuntimeError(f"Incomplete TESSERA ranged object: {url}")

    with partial.open("wb") as destination:
        for part in parts:
            with part.open("rb") as source:
                shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
    if partial.stat().st_size != expected_size or crc64nvme(partial) != expected_checksum:
        partial.unlink()
        for part in parts:
            part.unlink(missing_ok=True)
        raise RuntimeError(f"CRC64NVME mismatch for TESSERA object: {url}")
    partial.replace(cache_path)
    for part in parts:
        part.unlink()
    return str(cache_path)


def load_protocol() -> dict[str, Any]:
    return yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))[
        "phase13_temporal_extension"
    ]


def main() -> int:
    protected = [OUTPUT_PATH, MISSINGNESS_PATH, TILE_MANIFEST_PATH, FREEZE_PATH]
    existing = [str(path.relative_to(ROOT)) for path in protected if path.exists()]
    if existing:
        raise RuntimeError(f"Phase 13 TESSERA outputs exist; refusing to overwrite: {existing}")
    configure_tessera_downloads()
    base.download_file_to_temp = download_resumable
    protocol = load_protocol()
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    gedi_freeze = json.loads(GEDI_FREEZE_PATH.read_text(encoding="utf-8"))
    gate = gedi_freeze["freeze_basis"]["gate"]
    if not gate["passed"]:
        raise RuntimeError("Phase 13 target-free GEDI gate did not pass")
    viable_sites = list(gedi_freeze["freeze_basis"]["combined_unique_sites"])
    minimum_sites = int(protocol["combined_replication_gate"]["minimum_unique_forests"])
    if len(viable_sites) < minimum_sites:
        raise RuntimeError("Phase 13 has too few viable target sites")

    targets = pd.read_parquet(GEDI_PATH)
    targets = targets[targets["site_id"].isin(viable_sites)].copy()
    if set(targets["site_id"].unique()) != set(viable_sites):
        raise RuntimeError("Phase 13 target metadata is missing a viable site")
    if "fhd_normal" in targets.columns:
        raise RuntimeError("FHD entered the Phase 13 TESSERA alignment")
    predictors = config["predictors"]
    release = (str(predictors["dataset_version"]), str(predictors["dataset_variant"]))
    if release != ("1.0", "vultr") or set(targets["target_year"].unique()) - {2024, 2025}:
        raise RuntimeError("Phase 13 requires year-matched TESSERA 1.0/vultr targets")

    selected_manifest, selected_landmasks = base.ensure_registry("1.0")
    radius = float(config["target"]["footprint_diameter_m_approx"]) / 2
    aligned_frames: list[pd.DataFrame] = []
    tile_records_all: list[dict[str, Any]] = []
    try:
        for position, site in enumerate(viable_sites, start=1):
            site_targets = targets[targets["site_id"].eq(site)].copy()
            target_years = site_targets["target_year"].unique()
            if len(target_years) != 1:
                raise RuntimeError(f"Phase 13 target year is ambiguous for {site}")
            target_year = int(target_years[0])
            base.EXPECTED_SITES = {site}
            tiles, _ = base.required_tiles(site_targets, radius)
            selected_rows = base.lookup_embedding_rows(
                selected_manifest, tiles, target_year, "1.0", "vultr"
            )
            landmask_rows = base.lookup_landmask_rows(selected_landmasks, tiles)
            base.assert_complete_inventory(selected_rows, landmask_rows, tiles)
            print(
                f"[{position}/{len(viable_sites)}] {site}: stream {len(tiles)} TESSERA tiles",
                flush=True,
            )
            base.check_storage(STREAM_ROOT, selected_rows, landmask_rows, target_year)
            tile_records = base.download_tiles(
                STREAM_ROOT, selected_rows, landmask_rows, target_year, config
            )
            loaded = base.load_tiles(tile_records)
            aligned, _ = base.align_targets(site_targets, loaded, config)
            aligned["tessera_embedding_year"] = target_year
            if not aligned["passes_tessera_alignment"].all():
                raise RuntimeError(f"One or more {site} footprints failed TESSERA alignment")
            aligned_frames.append(aligned)
            for record in tile_records:
                provenance = dict(record)
                provenance["site_id"] = site
                provenance["retained_after_alignment"] = False
                tile_records_all.append(provenance)
            del loaded
            gc.collect()
            shutil.rmtree(STREAM_ROOT)
            print(f"[{position}/{len(viable_sites)}] {site}: aligned; tiles deleted", flush=True)
    finally:
        resumable_partials = (
            list(STREAM_ROOT.rglob("*.partial")) if STREAM_ROOT.exists() else []
        )
        if STREAM_ROOT.exists() and not resumable_partials:
            shutil.rmtree(STREAM_ROOT)

    aligned = pd.concat(aligned_frames, ignore_index=True)
    if len(aligned) != len(targets) or not aligned["passes_tessera_alignment"].all():
        raise RuntimeError("Phase 13 TESSERA alignment population is incomplete")
    if "fhd_normal" in aligned.columns:
        raise RuntimeError("FHD entered the aligned Phase 13 predictor table")
    missingness = base.missingness_table(aligned)
    freeze_basis = {
        "gedi_freeze_id": gedi_freeze["freeze_id"],
        "gedi_primary_sha256": common.sha256(GEDI_PATH),
        "protocol_sha256": common.sha256(PROTOCOL_PATH),
        "script_sha256": common.sha256(Path(__file__).resolve()),
        "geotessera_version": importlib.metadata.version("geotessera"),
        "registry_manifest_sha256": common.sha256(selected_manifest),
        "landmask_manifest_sha256": common.sha256(selected_landmasks),
        "releases": ["1.0/vultr/2024", "1.0/vultr/2025"],
        "sites": viable_sites,
        "site_years": gedi_freeze["freeze_basis"]["site_years"],
        "site_rows": {
            site: int(count) for site, count in aligned.groupby("site_id").size().items()
        },
        "site_tile_download_counts": {
            site: sum(record["site_id"] == site for record in tile_records_all)
            for site in viable_sites
        },
        "target_fhd_read": False,
        "source_tiles_retained": False,
        "alignment": {
            "footprint_radius_m": radius,
            "primary": "pixel_area_weighted_mean_inside_circular_footprint",
            "minimum_valid_footprint_fraction": config["alignment"][
                "minimum_valid_footprint_fraction"
            ],
        },
    }
    alignment_id = "phase13-tessera-" + common.canonical_hash(freeze_basis)[:12]
    aligned.insert(0, "tessera_alignment_id", alignment_id)
    base.write_parquet_atomic(aligned, OUTPUT_PATH)
    common.write_csv(missingness, MISSINGNESS_PATH)
    common.write_json(
        TILE_MANIFEST_PATH,
        {
            "created_utc": common.utc_now(),
            "releases": ["1.0/vultr/2024", "1.0/vultr/2025"],
            "tile_download_count": len(tile_records_all),
            "unique_tile_count": len({record["tile"] for record in tile_records_all}),
            "tiles": tile_records_all,
            "local_tiles_deleted": True,
        },
    )
    freeze = {
        "freeze_id": alignment_id,
        "created_utc": common.utc_now(),
        "status": "frozen_predictors_before_target_outcome_opening",
        "freeze_basis": freeze_basis,
        "outputs": {
            str(path.relative_to(ROOT)): common.sha256(path)
            for path in [OUTPUT_PATH, MISSINGNESS_PATH, TILE_MANIFEST_PATH]
        },
    }
    common.write_json(FREEZE_PATH, freeze)
    print(
        json.dumps(
            {
                "freeze_id": alignment_id,
                "sites": viable_sites,
                "rows": len(aligned),
                "unique_tiles": len({record["tile"] for record in tile_records_all}),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
