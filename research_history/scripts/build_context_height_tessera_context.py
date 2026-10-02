#!/usr/bin/env python3
"""Stream TESSERA tiles and freeze multi-scale context features for Phase 11."""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from pyproj import Transformer
from shapely.geometry import box


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_tessera_alignment as base  # noqa: E402


CONFIG_PATH = ROOT / "configs/project.yaml"
PROTOCOL_PATH = ROOT / "metadata/project_config_phase11_height_context_protocol_freeze.yaml"
INPUT_PATHS = [
    ROOT / "data/processed/tessera_aligned_development_soap_teak.parquet",
    ROOT / "data/processed/tessera_aligned_locked_bart.parquet",
    ROOT / "data/processed/phase7_tessera_aligned_expansion.parquet",
    ROOT / "data/processed/phase9_prospective_tessera_aligned.parquet",
]
OUTPUT_PATH = ROOT / "data/processed/phase11_tessera_context_features.parquet"
MISSINGNESS_PATH = ROOT / "outputs/tables/phase11_tessera_context_missingness.csv"
TILE_MANIFEST_PATH = ROOT / "metadata/phase11_tessera_context_tile_manifest.json"
FREEZE_PATH = ROOT / "metadata/phase11_tessera_context_freeze.json"
STREAM_ROOT = ROOT / "data/interim/phase11_tessera_context_stream"
CHUNK_ROOT = ROOT / "data/interim/phase11_tessera_context_chunks"

KEYS = ["site_id", "shot_number"]
AREA_FEATURES = [f"tessera_area_{index:03d}" for index in range(128)]
TARGET_READ_COLUMNS = KEYS + ["longitude", "latitude", *AREA_FEATURES]
WINDOW_WIDTHS_M = [30, 70, 150]
SUMMARY_NAMES = ["mean", "std", "delta"]


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


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def context_feature_columns(widths: list[int] = WINDOW_WIDTHS_M) -> list[str]:
    return [
        f"tessera_ctx{width:03d}_{summary}_{dimension:03d}"
        for width in widths
        for summary in SUMMARY_NAMES
        for dimension in range(128)
    ]


def summarize_context(
    matrix: np.ndarray,
    footprint_embedding: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if matrix.ndim != 2 or matrix.shape[1] != 128 or len(matrix) == 0:
        raise ValueError("Context matrix must contain one or more 128-D embeddings")
    mean = np.mean(matrix, axis=0, dtype=np.float64).astype(np.float32)
    std = np.std(matrix, axis=0, ddof=0, dtype=np.float64).astype(np.float32)
    delta = np.asarray(footprint_embedding, dtype=np.float32) - mean
    return mean, std, delta


def pixel_matrix_in_window(
    lon: float,
    lat: float,
    width_m: int,
    tiles: dict[tuple[float, float], base.TileData],
) -> tuple[np.ndarray, int, list[str]]:
    center_key = base.tile_from_world(lon, lat)
    center_tile = tiles[center_key]
    forward = Transformer.from_crs("EPSG:4326", center_tile.crs, always_xy=True)
    center_x, center_y = forward.transform(lon, lat)
    half_width = float(width_m) / 2.0
    window = box(
        center_x - half_width,
        center_y - half_width,
        center_x + half_width,
        center_y + half_width,
    )
    touched = base.footprint_tiles(lon, lat, half_width)
    missing = touched - set(tiles)
    if missing:
        raise RuntimeError(f"Context window requires unavailable tiles: {sorted(missing)}")
    if len({tiles[key].crs.to_string() for key in touched}) != 1:
        raise RuntimeError("A context window crosses TESSERA UTM zones")

    vectors: list[np.ndarray] = []
    used_tiles: set[str] = set()
    for key in touched:
        tile = tiles[key]
        clipped = window.intersection(tile.domain)
        if clipped.is_empty:
            continue
        rows, columns = base.pixel_ranges(tile, clipped.bounds)
        for row in rows:
            for column in columns:
                pixel_x, pixel_y = base.xy(tile.transform, row, column, offset="center")
                if abs(pixel_x - center_x) > half_width or abs(pixel_y - center_y) > half_width:
                    continue
                if not base.valid_pixel(tile, row, column):
                    continue
                vectors.append(base.embedding_at(tile, row, column))
                used_tiles.add(tile.name)
    if not vectors:
        raise RuntimeError(f"No valid TESSERA pixels in {width_m} m context window")
    return np.stack(vectors).astype(np.float32), len(vectors), sorted(used_tiles)


def extract_site_context(
    targets: pd.DataFrame,
    tiles: dict[tuple[float, float], base.TileData],
) -> pd.DataFrame:
    feature_columns = context_feature_columns()
    feature_values = np.empty((len(targets), len(feature_columns)), dtype=np.float32)
    counts = np.empty((len(targets), len(WINDOW_WIDTHS_M)), dtype=np.int16)
    tile_lists: list[list[str]] = []
    for position, row in enumerate(targets.itertuples(index=False), start=0):
        footprint = np.asarray(
            [getattr(row, feature) for feature in AREA_FEATURES], dtype=np.float32
        )
        offset = 0
        used_for_row: set[str] = set()
        for width_index, width in enumerate(WINDOW_WIDTHS_M):
            matrix, count, used = pixel_matrix_in_window(
                float(row.longitude), float(row.latitude), width, tiles
            )
            mean, std, delta = summarize_context(matrix, footprint)
            for values in [mean, std, delta]:
                feature_values[position, offset : offset + 128] = values
                offset += 128
            counts[position, width_index] = count
            used_for_row.update(used)
        tile_lists.append(sorted(used_for_row))
        if (position + 1) % 250 == 0 or position + 1 == len(targets):
            print(f"  context {position + 1:,}/{len(targets):,}", flush=True)

    output = targets[KEYS].reset_index(drop=True).copy()
    output = pd.concat(
        [output, pd.DataFrame(feature_values, columns=feature_columns)], axis=1
    )
    for index, width in enumerate(WINDOW_WIDTHS_M):
        output[f"tessera_ctx{width:03d}_valid_pixel_count"] = counts[:, index]
    output["tessera_context_tiles_used"] = [";".join(values) for values in tile_lists]
    return output


def validate_chunk(chunk: pd.DataFrame, expected: pd.DataFrame, site: str) -> None:
    if len(chunk) != len(expected) or set(chunk["site_id"].unique()) != {site}:
        raise RuntimeError(f"Wrong context chunk population for {site}")
    if chunk[KEYS].duplicated().any():
        raise RuntimeError(f"Duplicate context keys for {site}")
    expected_keys = set(map(tuple, expected[KEYS].to_numpy()))
    actual_keys = set(map(tuple, chunk[KEYS].to_numpy()))
    if actual_keys != expected_keys:
        raise RuntimeError(f"Context keys changed for {site}")
    features = context_feature_columns()
    if not np.isfinite(chunk[features].to_numpy(dtype=np.float32)).all():
        raise RuntimeError(f"Non-finite context features for {site}")


def main() -> int:
    protected = [OUTPUT_PATH, MISSINGNESS_PATH, TILE_MANIFEST_PATH, FREEZE_PATH]
    existing = [str(path.relative_to(ROOT)) for path in protected if path.exists()]
    if existing:
        raise RuntimeError(f"Phase 11 context outputs already exist: {existing}")
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))[
        "phase11_height_context_transfer"
    ]
    predictors = config["predictors"]
    release = (
        str(predictors["dataset_version"]),
        str(predictors["dataset_variant"]),
        int(predictors["embedding_year"]),
    )
    if release != ("1.0", "vultr", 2024):
        raise RuntimeError("Phase 11 context must remain in TESSERA 1.0/vultr/2024")
    targets = pd.concat(
        [pd.read_parquet(path, columns=TARGET_READ_COLUMNS) for path in INPUT_PATHS],
        ignore_index=True,
    )
    sites = sorted(protocol["cohort"]["sites"])
    if len(targets) != int(protocol["cohort"]["expected_rows"]):
        raise RuntimeError("Phase 11 context population row count changed")
    if sorted(targets["site_id"].unique()) != sites or targets[KEYS].duplicated().any():
        raise RuntimeError("Phase 11 context population sites or keys changed")

    selected_manifest, selected_landmasks = base.ensure_registry("1.0")
    CHUNK_ROOT.mkdir(parents=True, exist_ok=True)
    chunk_frames: list[pd.DataFrame] = []
    tile_records_all: list[dict[str, Any]] = []
    try:
        for site_index, site in enumerate(sites, start=1):
            site_targets = targets[targets["site_id"].eq(site)].copy()
            chunk_path = CHUNK_ROOT / f"{site}.parquet"
            chunk_manifest_path = CHUNK_ROOT / f"{site}.json"
            if chunk_path.exists() and chunk_manifest_path.exists():
                chunk = pd.read_parquet(chunk_path)
                validate_chunk(chunk, site_targets, site)
                chunk_manifest = json.loads(chunk_manifest_path.read_text(encoding="utf-8"))
                if chunk_manifest["chunk_sha256"] != sha256(chunk_path):
                    raise RuntimeError(f"Context chunk checksum changed for {site}")
                tile_records_all.extend(chunk_manifest["tiles"])
                chunk_frames.append(chunk)
                print(f"[{site_index:02d}/{len(sites)}] {site}: resume verified chunk", flush=True)
                continue

            if STREAM_ROOT.exists():
                shutil.rmtree(STREAM_ROOT)
            base.EXPECTED_SITES = {site}
            required, _ = base.required_tiles(site_targets, max(WINDOW_WIDTHS_M) / 2)
            selected_rows = base.lookup_embedding_rows(
                selected_manifest, required, 2024, "1.0", "vultr"
            )
            landmask_rows = base.lookup_landmask_rows(selected_landmasks, required)
            base.assert_complete_inventory(selected_rows, landmask_rows, required)
            print(
                f"[{site_index:02d}/{len(sites)}] {site}: stream {len(required)} tiles",
                flush=True,
            )
            base.check_storage(STREAM_ROOT, selected_rows, landmask_rows, 2024)
            tile_records = base.download_tiles(
                STREAM_ROOT, selected_rows, landmask_rows, 2024, config
            )
            loaded = base.load_tiles(tile_records)
            chunk = extract_site_context(site_targets, loaded)
            validate_chunk(chunk, site_targets, site)
            write_parquet(chunk, chunk_path)
            site_records = []
            for record in tile_records:
                provenance = dict(record)
                provenance["site_id"] = site
                provenance["retained_after_context_extraction"] = False
                site_records.append(provenance)
            write_json(
                chunk_manifest_path,
                {
                    "site_id": site,
                    "rows": len(chunk),
                    "chunk_sha256": sha256(chunk_path),
                    "tiles": site_records,
                },
            )
            chunk_frames.append(chunk)
            tile_records_all.extend(site_records)
            del loaded
            gc.collect()
            shutil.rmtree(STREAM_ROOT)
            print(f"[{site_index:02d}/{len(sites)}] {site}: compact chunk frozen; tiles deleted", flush=True)
    finally:
        if STREAM_ROOT.exists():
            shutil.rmtree(STREAM_ROOT)

    context = pd.concat(chunk_frames, ignore_index=True).sort_values(KEYS).reset_index(drop=True)
    if len(context) != len(targets) or context[KEYS].duplicated().any():
        raise RuntimeError("Combined context population is incomplete")
    features = context_feature_columns()
    if len(features) != 1152 or not np.isfinite(context[features].to_numpy(dtype=np.float32)).all():
        raise RuntimeError("Combined context feature matrix is incomplete")
    write_parquet(context, OUTPUT_PATH)

    missingness_records = []
    for site, frame in context.groupby("site_id", sort=True):
        for width in WINDOW_WIDTHS_M:
            counts = frame[f"tessera_ctx{width:03d}_valid_pixel_count"]
            missingness_records.append(
                {
                    "site_id": site,
                    "window_width_m": width,
                    "rows": len(frame),
                    "minimum_valid_pixels": int(counts.min()),
                    "median_valid_pixels": float(counts.median()),
                    "maximum_valid_pixels": int(counts.max()),
                }
            )
    missingness = pd.DataFrame(missingness_records)
    MISSINGNESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    missingness.to_csv(MISSINGNESS_PATH, index=False)
    write_json(
        TILE_MANIFEST_PATH,
        {
            "created_utc": utc_now(),
            "dataset_version": "1.0",
            "dataset_variant": "vultr",
            "year": 2024,
            "window_widths_m": WINDOW_WIDTHS_M,
            "tile_download_count": len(tile_records_all),
            "unique_tile_count": len({record["tile"] for record in tile_records_all}),
            "tiles": tile_records_all,
            "local_stream_tiles_deleted": True,
        },
    )
    freeze_basis = {
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "script_sha256": sha256(Path(__file__).resolve()),
        "input_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in INPUT_PATHS},
        "registry_manifest_sha256": sha256(selected_manifest),
        "landmask_manifest_sha256": sha256(selected_landmasks),
        "dataset_version": "1.0",
        "dataset_variant": "vultr",
        "year": 2024,
        "rows": len(context),
        "sites": sites,
        "window_widths_m": WINDOW_WIDTHS_M,
        "context_feature_count": len(features),
        "summaries": SUMMARY_NAMES,
        "target_columns_read": [],
        "target_outcomes_used": False,
        "raw_stream_tiles_retained": False,
        "geotessera_version": importlib.metadata.version("geotessera"),
    }
    freeze = {
        "freeze_id": "phase11-tessera-context-" + canonical_hash(freeze_basis)[:12],
        "created_utc": utc_now(),
        "status": "complete_target_free_context_features",
        "freeze_basis": freeze_basis,
        "outputs": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [OUTPUT_PATH, MISSINGNESS_PATH, TILE_MANIFEST_PATH]
        },
    }
    write_json(FREEZE_PATH, freeze)
    shutil.rmtree(CHUNK_ROOT)
    print(
        json.dumps(
            {
                "freeze_id": freeze["freeze_id"],
                "rows": len(context),
                "sites": len(sites),
                "context_features": len(features),
                "tile_downloads": len(tile_records_all),
                "unique_tiles": len({record["tile"] for record in tile_records_all}),
                "raw_tiles_retained": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
