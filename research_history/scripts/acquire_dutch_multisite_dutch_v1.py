#!/usr/bin/env python3
"""Acquire matching-year public TESSERA v1.0 inputs for Phase 36."""

from __future__ import annotations

import gc
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import yaml
from pyproj import Transformer
from rasterio.transform import rowcol

from acquire_ahn4_replication_ahn4_tessera import (
    assert_complete_inventory,
    atomic_npz,
    download_checked,
    ensure_registry,
    finalise_summary,
    local_tile_paths,
    lookup_embedding_rows,
    lookup_landmask_rows,
    reduce_embeddings,
    s3_url,
    tile_from_world,
    tile_name,
)
from tessera_v2_common import sha256


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / os.environ.get(
    "DUTCH_MULTISITE_CONFIG", "configs/dutch_multisite_transfer.yaml"
)
CONFIG_KEY = os.environ.get(
    "DUTCH_MULTISITE_CONFIG_KEY", "phase36_dutch_multisite_transfer"
)
PHASE_TAG = os.environ.get("DUTCH_MULTISITE_PHASE", "phase36")
DIMENSIONS = 128


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[CONFIG_KEY]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sample_points(cohort: pd.DataFrame) -> dict[str, np.ndarray]:
    rows = np.repeat(np.arange(5, dtype=np.int8), 5)
    columns = np.tile(np.arange(5, dtype=np.int8), 5)
    offset_x = columns - 2
    offset_y = 2 - rows
    positions = np.arange(25, dtype=np.uint32)
    keep = (
        cohort["forest_pixel_mask"].to_numpy(dtype=np.uint32)[:, None]
        & (1 << positions)[None, :]
    ) > 0
    selected = keep.reshape(-1)
    return {
        "patch_id": np.repeat(cohort["row_id"].to_numpy(dtype=np.int64), 25)[selected],
        "rd_x": (
            np.repeat(cohort["rd_x"].to_numpy(dtype=np.float64), 25)
            + np.tile(offset_x.astype(np.float64) * 10.0, len(cohort))
        )[selected],
        "rd_y": (
            np.repeat(cohort["rd_y"].to_numpy(dtype=np.float64), 25)
            + np.tile(offset_y.astype(np.float64) * 10.0, len(cohort))
        )[selected],
        "offset_x": np.tile(offset_x, len(cohort))[selected].astype(np.float32),
        "offset_y": np.tile(offset_y, len(cohort))[selected].astype(np.float32),
        "year": np.repeat(cohort["tessera_year"].to_numpy(dtype=np.int16), 25)[selected],
    }


def run() -> None:
    config = load_config()
    settings = config["tessera"]
    outputs = config["outputs"]
    cohort_path = ROOT / str(outputs["cohort"])
    output_path = ROOT / str(outputs["features"])
    manifest_path = ROOT / str(outputs["acquisition_freeze"])
    cohort = pd.read_parquet(cohort_path).sort_values("row_id").reset_index(drop=True)
    if output_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest["cohort_sha256"] == sha256(cohort_path)
            and manifest["config_sha256"] == sha256(CONFIG_PATH)
            and manifest["output_sha256"] == sha256(output_path)
        ):
            print(f"{PHASE_TAG} validated existing public TESSERA v1 acquisition", flush=True)
            return
        raise RuntimeError(f"Existing {PHASE_TAG} public TESSERA acquisition is stale")
    points = sample_points(cohort)
    to_wgs84 = Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)
    longitude, latitude = to_wgs84.transform(points["rd_x"], points["rd_y"])
    tiles = [
        tile_from_world(float(lon), float(lat))
        for lon, lat in zip(longitude, latitude, strict=True)
    ]
    groups = sorted(
        {
            (int(year), tile)
            for year, tile in zip(points["year"], tiles, strict=True)
        }
    )
    lookup = {group: index for index, group in enumerate(groups)}
    group_code = np.asarray(
        [lookup[(int(year), tile)] for year, tile in zip(points["year"], tiles, strict=True)],
        dtype=np.int16,
    )
    version = str(settings["dataset_version"])
    variant = str(settings["dataset_variant"])
    registry_path, landmask_path = ensure_registry(version)
    inventory: dict[tuple[int, tuple[float, float]], tuple[Any, Any]] = {}
    for year in sorted(set(points["year"].astype(int))):
        year_tiles = sorted({tile for local_year, tile in groups if local_year == year})
        rows = lookup_embedding_rows(registry_path, year_tiles, year, version, variant)
        masks = lookup_landmask_rows(landmask_path, year_tiles)
        assert_complete_inventory(rows, masks, year_tiles)
        rows_by_tile = {
            (round(float(row.lon), 2), round(float(row.lat), 2)): row
            for row in rows.itertuples(index=False)
        }
        masks_by_tile = {
            (round(float(row.lon), 2), round(float(row.lat), 2)): row
            for row in masks.itertuples(index=False)
        }
        for tile in year_tiles:
            inventory[(year, tile)] = (rows_by_tile[tile], masks_by_tile[tile])
    chunk_root = ROOT / str(
        settings.get("chunk_directory", f"data/interim/{PHASE_TAG}_v1_chunks")
    )
    stream_root = ROOT / str(
        settings.get("stream_directory", f"data/interim/{PHASE_TAG}_v1_stream")
    )
    chunk_root.mkdir(parents=True, exist_ok=True)
    config_hash = sha256(CONFIG_PATH)
    records: list[dict[str, Any]] = []
    for code, (year, tile) in enumerate(groups):
        name = f"{year}_{tile_name(tile)}"
        indices = np.flatnonzero(group_code == code)
        chunk_path = chunk_root / f"{name}.npz"
        if chunk_path.exists():
            records.append({"year": year, "tile": tile_name(tile), "source_points": len(indices), "resumed": True})
            print(f"{PHASE_TAG} v1 [{code + 1}/{len(groups)}]: resumed {name}", flush=True)
            continue
        row, mask_row = inventory[(year, tile)]
        paths = local_tile_paths(stream_root, year, tile)
        sources = {
            "embedding": str(row.grid_path),
            "scales": str(row.scales_path),
            "landmask": str(mask_row.key),
        }
        sizes = {
            "embedding": int(row.grid_size),
            "scales": int(row.scales_size),
            "landmask": int(mask_row.file_size),
        }
        print(
            f"{PHASE_TAG} v1 [{code + 1}/{len(groups)}]: acquire {name} for {len(indices):,} pixels",
            flush=True,
        )
        try:
            if shutil.disk_usage(ROOT).free < sum(sizes.values()) + 512 * 1024 * 1024:
                raise RuntimeError(f"Insufficient free space for {name}")
            for kind in ["embedding", "scales", "landmask"]:
                download_checked(s3_url(sources[kind]), paths[kind], sizes[kind])
            quantized = np.load(paths["embedding"], mmap_mode="r")
            scales = np.load(paths["scales"], mmap_mode="r")
            with rasterio.open(paths["landmask"]) as dataset:
                landmask = dataset.read(1)
                to_tile = Transformer.from_crs("EPSG:28992", dataset.crs, always_xy=True)
                x, y = to_tile.transform(points["rd_x"][indices], points["rd_y"][indices])
                rr, cc = rowcol(dataset.transform, x, y)
            rr = np.asarray(rr, dtype=np.int64)
            cc = np.asarray(cc, dtype=np.int64)
            inside = (
                (rr >= 0) & (cc >= 0) & (rr < quantized.shape[0]) & (cc < quantized.shape[1])
            )
            local = np.flatnonzero(inside)
            usable = (
                np.isfinite(scales[rr[local], cc[local]])
                & (scales[rr[local], cc[local]] > 0)
                & (landmask[rr[local], cc[local]] > 0)
            )
            local = local[usable]
            embeddings = (
                quantized[rr[local], cc[local]].astype(np.float32)
                * scales[rr[local], cc[local]].astype(np.float32)[:, None]
            )
            finite = np.isfinite(embeddings).all(axis=1)
            local = local[finite]
            embeddings = embeddings[finite]
            source_indices = indices[local]
            partial = reduce_embeddings(
                points["patch_id"][source_indices],
                points["offset_x"][source_indices],
                points["offset_y"][source_indices],
                embeddings,
            )
            partial["config_sha256"] = np.asarray([config_hash])
            atomic_npz(chunk_path, partial)
            records.append(
                {
                    "year": year,
                    "tile": tile_name(tile),
                    "source_points": int(len(indices)),
                    "valid_points": int(len(source_indices)),
                    "resumed": False,
                }
            )
        finally:
            for path in paths.values():
                path.unlink(missing_ok=True)
            gc.collect()
    count = np.zeros(len(cohort), dtype=np.int16)
    sum_x = np.zeros(len(cohort), dtype=np.float32)
    sum_y = np.zeros(len(cohort), dtype=np.float32)
    sum_x2 = np.zeros(len(cohort), dtype=np.float32)
    sum_y2 = np.zeros(len(cohort), dtype=np.float32)
    sums = {
        key: np.zeros((len(cohort), DIMENSIONS), dtype=np.float32)
        for key in ["sum_embedding", "sum_embedding_sq", "sum_x_embedding", "sum_y_embedding"]
    }
    for year, tile in groups:
        with np.load(chunk_root / f"{year}_{tile_name(tile)}.npz") as chunk:
            ids = chunk["patch_id"].astype(np.int64)
            count[ids] += chunk["count"]
            sum_x[ids] += chunk["sum_x"]
            sum_y[ids] += chunk["sum_y"]
            sum_x2[ids] += chunk["sum_x2"]
            sum_y2[ids] += chunk["sum_y2"]
            for key in sums:
                sums[key][ids] += chunk[key]
    summaries = finalise_summary(count, sum_x, sum_y, sum_x2, sum_y2, **sums)
    minimum = int(settings["minimum_valid_pixels"])
    output: dict[str, Any] = {
        "row_id": cohort["row_id"].to_numpy(dtype=np.int64),
        "tessera_valid_pixel_count": count,
        "tessera_context_valid": count >= minimum,
    }
    for dimension in range(DIMENSIONS):
        output[f"tessera_mean_{dimension:03d}"] = summaries["mean"][:, dimension]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.parquet")
    pd.DataFrame(output).to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(output_path)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cohort_sha256": sha256(cohort_path),
        "config_sha256": config_hash,
        "dataset_version": version,
        "dataset_variant": variant,
        "year_tile_groups": records,
        "valid_rows": int((count >= minimum).sum()),
        "rows": int(len(cohort)),
        "output_sha256": sha256(output_path),
    }
    atomic_json(manifest_path, manifest)
    shutil.rmtree(chunk_root, ignore_errors=True)
    shutil.rmtree(stream_root, ignore_errors=True)
    print(
        f"{PHASE_TAG} public v1 complete: {manifest['valid_rows']:,}/{len(cohort):,} valid units",
        flush=True,
    )


if __name__ == "__main__":
    run()
