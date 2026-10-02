#!/usr/bin/env python3
"""Stream and aggregate 2021 TESSERA embeddings for frozen Savelsbos units."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import yaml
from pyproj import Transformer
from rasterio.transform import rowcol


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/ahn4_replication_ahn4_deciduous.yaml"
CHUNK_DIR = ROOT / "data/interim/phase26_ahn4_tessera_chunks"
STREAM_DIR = ROOT / "data/interim/phase26_ahn4_tessera_stream"
MANIFEST_PATH = ROOT / "metadata/phase26_ahn4_tessera_manifest.json"
DIMENSIONS = 128
S3_ROOT = "https://s3.us-west-2.amazonaws.com/tessera-embeddings"
EMBEDDINGS_SUBDIR = "global_0.1_degree_representation"
LANDMASKS_SUBDIR = "global_0.1_degree_tiff_all"


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


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def atomic_npz(path: Path, values: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **values)
    temporary.replace(path)


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "curl",
            "--location",
            "--fail",
            "--retry",
            "8",
            "--retry-all-errors",
            "--continue-at",
            "-",
            "--output",
            str(path),
            url,
        ],
        check=True,
    )


def download_checked(url: str, path: Path, size: int) -> None:
    if path.exists() and path.stat().st_size == size:
        return
    download(url, path)
    if path.stat().st_size != size:
        path.unlink(missing_ok=True)
        raise RuntimeError(f"Wrong download size for {path.name}")


def version_path(version: str) -> str:
    cleaned = version.removeprefix("v")
    major, *minor = cleaned.split(".")
    return f"v{major}" if not minor or minor[0] == "0" else f"v{major}.{minor[0]}"


def tile_from_world(lon: float, lat: float) -> tuple[float, float]:
    return (
        round(math.floor(lon * 10) / 10 + 0.05, 2),
        round(math.floor(lat * 10) / 10 + 0.05, 2),
    )


def tile_name(tile: tuple[float, float]) -> str:
    return f"grid_{tile[0]:.2f}_{tile[1]:.2f}"


def s3_url(key: str) -> str:
    prefix = "s3://tessera-embeddings/"
    if not key.startswith(prefix):
        raise RuntimeError(f"Unexpected TESSERA source path: {key}")
    return f"{S3_ROOT}/{key[len(prefix):]}"


def ensure_registry(version: str) -> tuple[Path, Path]:
    cache = ROOT / "data/external/geotessera_cache" / version_path(version)
    manifest = cache / "manifest.parquet"
    landmasks = cache / "landmasks.parquet"
    for name, path in [("manifest.parquet", manifest), ("landmasks.parquet", landmasks)]:
        if not path.exists():
            download(f"{S3_ROOT}/{version_path(version)}/{name}", path)
    return manifest, landmasks


def lookup_embedding_rows(
    manifest: Path,
    tiles: list[tuple[float, float]],
    year: int,
    version: str,
    variant: str,
) -> pd.DataFrame:
    import pyarrow.parquet as pq

    lon_values = sorted({round(lon * 100) for lon, _ in tiles})
    lat_values = sorted({round(lat * 100) for _, lat in tiles})
    frame = pq.read_table(
        manifest,
        filters=[
            ("year", "=", year),
            ("lon_i", "in", lon_values),
            ("lat_i", "in", lat_values),
        ],
    ).to_pandas()
    wanted = {(round(lon * 100), round(lat * 100)) for lon, lat in tiles}
    frame = frame[
        [
            (int(lon), int(lat)) in wanted
            for lon, lat in zip(frame["lon_i"], frame["lat_i"], strict=True)
        ]
    ]
    return frame[
        frame["version"].astype(str).eq(version)
        & frame["variant"].astype(str).eq(variant)
    ].sort_values(["lon", "lat"])


def lookup_landmask_rows(
    manifest: Path, tiles: list[tuple[float, float]]
) -> pd.DataFrame:
    import pyarrow.parquet as pq

    lon_values = sorted({round(lon * 100) for lon, _ in tiles})
    lat_values = sorted({round(lat * 100) for _, lat in tiles})
    frame = pq.read_table(
        manifest,
        filters=[("lon_i", "in", lon_values), ("lat_i", "in", lat_values)],
    ).to_pandas()
    wanted = {(round(lon * 100), round(lat * 100)) for lon, lat in tiles}
    return frame[
        [
            (int(lon), int(lat)) in wanted
            for lon, lat in zip(frame["lon_i"], frame["lat_i"], strict=True)
        ]
    ].sort_values(["lon", "lat"])


def assert_complete_inventory(
    rows: pd.DataFrame,
    masks: pd.DataFrame,
    tiles: list[tuple[float, float]],
) -> None:
    wanted = {(round(lon, 2), round(lat, 2)) for lon, lat in tiles}
    found = {
        (round(float(lon), 2), round(float(lat), 2))
        for lon, lat in zip(rows["lon"], rows["lat"], strict=True)
    }
    found_masks = {
        (round(float(lon), 2), round(float(lat), 2))
        for lon, lat in zip(masks["lon"], masks["lat"], strict=True)
    }
    if found != wanted or found_masks != wanted:
        raise RuntimeError(
            f"Incomplete TESSERA inventory; embeddings={sorted(wanted-found)}, "
            f"landmasks={sorted(wanted-found_masks)}"
        )


def local_tile_paths(
    root: Path, year: int, tile: tuple[float, float]
) -> dict[str, Path]:
    name = tile_name(tile)
    directory = root / EMBEDDINGS_SUBDIR / str(year) / name
    return {
        "embedding": directory / f"{name}.npy",
        "scales": directory / f"{name}_scales.npy",
        "landmask": root / LANDMASKS_SUBDIR / f"{name}.tiff",
    }


def sample_points(cohort: pd.DataFrame) -> dict[str, np.ndarray]:
    local_rows = np.repeat(np.arange(5, dtype=np.int8), 5)
    local_columns = np.tile(np.arange(5, dtype=np.int8), 5)
    offsets_x = local_columns - 2
    offsets_y = 2 - local_rows
    bits = (1 << np.arange(25, dtype=np.uint32))[None, :]
    masks = cohort["broadleaf_pixel_mask"].to_numpy(dtype=np.uint32)[:, None]
    keep = (masks & bits) > 0
    patch_id = np.repeat(cohort["row_id"].to_numpy(dtype=np.int64), 25)[
        keep.reshape(-1)
    ]
    x = (
        np.repeat(cohort["rd_x"].to_numpy(dtype=np.float64), 25)
        + np.tile(offsets_x.astype(np.float64) * 10.0, len(cohort))
    )[keep.reshape(-1)]
    y = (
        np.repeat(cohort["rd_y"].to_numpy(dtype=np.float64), 25)
        + np.tile(offsets_y.astype(np.float64) * 10.0, len(cohort))
    )[keep.reshape(-1)]
    return {
        "patch_id": patch_id,
        "rd_x": x,
        "rd_y": y,
        "offset_x": np.tile(offsets_x, len(cohort))[keep.reshape(-1)].astype(
            np.float32
        ),
        "offset_y": np.tile(offsets_y, len(cohort))[keep.reshape(-1)].astype(
            np.float32
        ),
    }


def reduce_embeddings(
    patch_id: np.ndarray,
    offset_x: np.ndarray,
    offset_y: np.ndarray,
    embeddings: np.ndarray,
) -> dict[str, np.ndarray]:
    order = np.argsort(patch_id, kind="stable")
    patch_id = patch_id[order]
    offset_x = offset_x[order]
    offset_y = offset_y[order]
    embeddings = embeddings[order].astype(np.float32)
    unique, starts, count = np.unique(
        patch_id, return_index=True, return_counts=True
    )

    def grouped_sum(values: np.ndarray) -> np.ndarray:
        return np.add.reduceat(values, starts, axis=0)

    return {
        "patch_id": unique.astype(np.int64),
        "count": count.astype(np.int16),
        "sum_x": grouped_sum(offset_x).astype(np.float32),
        "sum_y": grouped_sum(offset_y).astype(np.float32),
        "sum_x2": grouped_sum(offset_x * offset_x).astype(np.float32),
        "sum_y2": grouped_sum(offset_y * offset_y).astype(np.float32),
        "sum_embedding": grouped_sum(embeddings).astype(np.float32),
        "sum_embedding_sq": grouped_sum(embeddings * embeddings).astype(
            np.float32
        ),
        "sum_x_embedding": grouped_sum(embeddings * offset_x[:, None]).astype(
            np.float32
        ),
        "sum_y_embedding": grouped_sum(embeddings * offset_y[:, None]).astype(
            np.float32
        ),
    }


def finalise_summary(
    count: np.ndarray,
    sum_x: np.ndarray,
    sum_y: np.ndarray,
    sum_x2: np.ndarray,
    sum_y2: np.ndarray,
    sum_embedding: np.ndarray,
    sum_embedding_sq: np.ndarray,
    sum_x_embedding: np.ndarray,
    sum_y_embedding: np.ndarray,
) -> dict[str, np.ndarray]:
    shape = sum_embedding.shape
    mean = np.full(shape, np.nan, dtype=np.float32)
    standard_deviation = np.full(shape, np.nan, dtype=np.float32)
    dx = np.full(shape, np.nan, dtype=np.float32)
    dy = np.full(shape, np.nan, dtype=np.float32)
    usable = count > 0
    mean[usable] = sum_embedding[usable] / count[usable, None]
    variance = np.maximum(
        sum_embedding_sq[usable] / count[usable, None]
        - mean[usable] * mean[usable],
        0.0,
    )
    standard_deviation[usable] = np.sqrt(variance)
    denominator_x = np.zeros(len(count), dtype=np.float32)
    denominator_y = np.zeros(len(count), dtype=np.float32)
    denominator_x[usable] = (
        sum_x2[usable] - sum_x[usable] ** 2 / count[usable]
    )
    denominator_y[usable] = (
        sum_y2[usable] - sum_y[usable] ** 2 / count[usable]
    )
    valid_x = denominator_x > 0
    valid_y = denominator_y > 0
    dx[valid_x] = (
        sum_x_embedding[valid_x]
        - sum_x[valid_x, None]
        * sum_embedding[valid_x]
        / count[valid_x, None]
    ) / denominator_x[valid_x, None]
    dy[valid_y] = (
        sum_y_embedding[valid_y]
        - sum_y[valid_y, None]
        * sum_embedding[valid_y]
        / count[valid_y, None]
    ) / denominator_y[valid_y, None]
    return {"mean": mean, "std": standard_deviation, "dx": dx, "dy": dy}


def run() -> None:
    config = load_config()
    output_path = ROOT / str(config["outputs"]["tessera"])
    cohort_path = ROOT / str(config["outputs"]["cohort"])
    if output_path.exists() and MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if (
            manifest["cohort_sha256"] == sha256(cohort_path)
            and manifest["output_sha256"] == sha256(output_path)
        ):
            print("phase26 TESSERA: validated existing checkpoint", flush=True)
            return
        raise RuntimeError("The existing Phase 26 TESSERA checkpoint is stale")

    cohort = pd.read_parquet(cohort_path)
    points = sample_points(cohort)
    transformer = Transformer.from_crs(
        config["study"]["analysis_crs"], "EPSG:4326", always_xy=True
    )
    longitude, latitude = transformer.transform(points["rd_x"], points["rd_y"])
    tile_pairs = [
        tile_from_world(float(lon), float(lat))
        for lon, lat in zip(longitude, latitude, strict=True)
    ]
    tiles = sorted(set(tile_pairs))
    tile_codes = np.asarray([tiles.index(tile) for tile in tile_pairs], dtype=np.int16)
    settings = config["tessera"]
    year = int(settings["year"])
    version = str(settings["dataset_version"])
    variant = str(settings["dataset_variant"])
    registry_path, landmask_path = ensure_registry(version)
    rows = lookup_embedding_rows(
        registry_path, tiles, year, version, variant
    )
    masks = lookup_landmask_rows(landmask_path, tiles)
    assert_complete_inventory(rows, masks, tiles)
    rows_by_tile = {
        (round(float(row.lon), 2), round(float(row.lat), 2)): row
        for row in rows.itertuples(index=False)
    }
    masks_by_tile = {
        (round(float(row.lon), 2), round(float(row.lat), 2)): row
        for row in masks.itertuples(index=False)
    }
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    config_hash = sha256(CONFIG_PATH)
    for code, tile in enumerate(tiles):
        name = tile_name(tile)
        indices = np.flatnonzero(tile_codes == code)
        chunk_path = CHUNK_DIR / f"{name}.npz"
        if chunk_path.exists():
            records.append(
                {"tile": name, "source_points": int(len(indices)), "resumed": True}
            )
            print(f"TESSERA [{code + 1}/{len(tiles)}]: resumed {name}", flush=True)
            continue
        row = rows_by_tile[tile]
        mask_row = masks_by_tile[tile]
        paths = local_tile_paths(STREAM_DIR, year, tile)
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
        required = sum(sizes.values()) + 512 * 1024 * 1024
        if shutil.disk_usage(ROOT).free < required:
            raise RuntimeError(f"Insufficient free space for {name}")
        print(
            f"TESSERA [{code + 1}/{len(tiles)}]: acquire {name} for "
            f"{len(indices):,} broadleaf pixels",
            flush=True,
        )
        try:
            for kind in ["embedding", "scales", "landmask"]:
                download_checked(s3_url(sources[kind]), paths[kind], sizes[kind])
            quantized = np.load(paths["embedding"], mmap_mode="r")
            scales = np.load(paths["scales"], mmap_mode="r")
            with rasterio.open(paths["landmask"]) as dataset:
                landmask = dataset.read(1)
                target_crs = dataset.crs
                target_transform = dataset.transform
            to_tile = Transformer.from_crs(
                config["study"]["analysis_crs"], target_crs, always_xy=True
            )
            x, y = to_tile.transform(points["rd_x"][indices], points["rd_y"][indices])
            rr, cc = rowcol(target_transform, x, y)
            rr = np.asarray(rr, dtype=np.int64)
            cc = np.asarray(cc, dtype=np.int64)
            inside = (
                (rr >= 0)
                & (cc >= 0)
                & (rr < quantized.shape[0])
                & (cc < quantized.shape[1])
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
                    "tile": name,
                    "source_points": int(len(indices)),
                    "valid_points": int(len(source_indices)),
                    "source_sizes": sizes,
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
    shape = (len(cohort), DIMENSIONS)
    sums = {
        "sum_embedding": np.zeros(shape, dtype=np.float32),
        "sum_embedding_sq": np.zeros(shape, dtype=np.float32),
        "sum_x_embedding": np.zeros(shape, dtype=np.float32),
        "sum_y_embedding": np.zeros(shape, dtype=np.float32),
    }
    for tile in tiles:
        with np.load(CHUNK_DIR / f"{tile_name(tile)}.npz") as chunk:
            ids = chunk["patch_id"].astype(np.int64)
            count[ids] += chunk["count"]
            sum_x[ids] += chunk["sum_x"]
            sum_y[ids] += chunk["sum_y"]
            sum_x2[ids] += chunk["sum_x2"]
            sum_y2[ids] += chunk["sum_y2"]
            for key in sums:
                sums[key][ids] += chunk[key]
    summaries = finalise_summary(
        count, sum_x, sum_y, sum_x2, sum_y2, **sums
    )
    minimum = int(config["predictors"]["minimum_valid_tessera_pixels"])
    valid = count >= minimum
    output = pd.DataFrame(
        {
            "row_id": cohort["row_id"].to_numpy(dtype=np.int64),
            "tessera_valid_pixel_count": count,
            "tessera_context_valid": valid,
        }
    )
    for prefix, values in summaries.items():
        for dimension in range(DIMENSIONS):
            output[f"tessera_{prefix}_{dimension:03d}"] = values[:, dimension]
    atomic_parquet(output, output_path)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cohort_sha256": sha256(cohort_path),
        "config_sha256": config_hash,
        "year": year,
        "dataset_version": version,
        "dataset_variant": variant,
        "aggregation": "mean, standard deviation and planar x/y gradients over mapped broadleaf 10 m cells",
        "tile_count": int(len(tiles)),
        "tiles": records,
        "valid_rows": int(valid.sum()),
        "output_sha256": sha256(output_path),
    }
    atomic_json(MANIFEST_PATH, manifest)
    shutil.rmtree(CHUNK_DIR, ignore_errors=True)
    shutil.rmtree(STREAM_DIR, ignore_errors=True)
    print(
        f"phase26 TESSERA: {valid.sum():,}/{len(valid):,} valid units across "
        f"{len(tiles)} tiles",
        flush=True,
    )


if __name__ == "__main__":
    run()
