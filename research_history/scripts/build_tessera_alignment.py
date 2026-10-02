#!/usr/bin/env python3
"""Acquire the exact TESSERA tiles and align them to the frozen GEDI sample."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import rasterio
import yaml
from geotessera.registry import download_file_to_temp, write_tessera_metadata
from pyproj import CRS, Transformer
from rasterio.transform import rowcol, xy
from rasterio.windows import Window
from shapely.geometry import Point, Polygon, box


ROOT = Path(__file__).resolve().parents[1]
S3_HTTPS_ROOT = "https://s3.us-west-2.amazonaws.com/tessera-embeddings"
EMBEDDINGS_SUBDIR = "global_0.1_degree_representation"
LANDMASKS_SUBDIR = "global_0.1_degree_tiff_all"
DOWNLOAD_MARGIN_BYTES = 512 * 1024 * 1024
EXPECTED_SITES = {"SOAP", "TEAK", "BART"}


@dataclass
class TileData:
    lon: float
    lat: float
    name: str
    quantized: np.ndarray
    scales: np.ndarray
    landmask: np.ndarray
    crs: CRS
    transform: rasterio.Affine
    height: int
    width: int
    domain: Polygon


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def version_path(version: str) -> str:
    cleaned = version.removeprefix("v")
    major, *minor = cleaned.split(".")
    return f"v{major}" if not minor or minor[0] == "0" else f"v{major}.{minor[0]}"


def tile_from_world(lon: float, lat: float) -> tuple[float, float]:
    tile_lon = math.floor(lon * 10) / 10 + 0.05
    tile_lat = math.floor(lat * 10) / 10 + 0.05
    return round(tile_lon, 2), round(tile_lat, 2)


def tile_name(tile: tuple[float, float]) -> str:
    return f"grid_{tile[0]:.2f}_{tile[1]:.2f}"


def load_config() -> dict[str, Any]:
    with (ROOT / "configs/project.yaml").open(encoding="utf-8") as source:
        return yaml.safe_load(source)


def projected_transformers(lon: float, lat: float) -> tuple[Transformer, Transformer, CRS]:
    zone = max(1, min(60, math.floor((lon + 180) / 6) + 1))
    epsg = (32600 if lat >= 0 else 32700) + zone
    crs = CRS.from_epsg(epsg)
    return (
        Transformer.from_crs("EPSG:4326", crs, always_xy=True),
        Transformer.from_crs(crs, "EPSG:4326", always_xy=True),
        crs,
    )


def footprint_tiles(lon: float, lat: float, radius_m: float) -> set[tuple[float, float]]:
    forward, backward, _ = projected_transformers(lon, lat)
    x, y = forward.transform(lon, lat)
    points = [(x, y)] + [
        (x + dx, y + dy)
        for dx, dy in [
            (-radius_m, -radius_m),
            (-radius_m, radius_m),
            (radius_m, -radius_m),
            (radius_m, radius_m),
        ]
    ]
    return {tile_from_world(*backward.transform(px, py)) for px, py in points}


def required_tiles(targets: pd.DataFrame, radius_m: float) -> tuple[list[tuple[float, float]], dict[str, list[str]]]:
    tiles: set[tuple[float, float]] = set()
    by_site: dict[str, set[tuple[float, float]]] = {site: set() for site in EXPECTED_SITES}
    for row in targets[["site_id", "longitude", "latitude"]].itertuples(index=False):
        touched = footprint_tiles(float(row.longitude), float(row.latitude), radius_m)
        tiles.update(touched)
        by_site[row.site_id].update(touched)
    return sorted(tiles), {
        site: [tile_name(tile) for tile in sorted(site_tiles)]
        for site, site_tiles in sorted(by_site.items())
    }


def s3_url(key: str) -> str:
    if not key.startswith("s3://tessera-embeddings/"):
        raise RuntimeError(f"Unexpected TESSERA source path: {key}")
    return f"{S3_HTTPS_ROOT}/{key.split('/', 3)[3]}"


def ensure_registry(version: str) -> tuple[Path, Path]:
    path_component = version_path(version)
    cache = ROOT / "data/external/geotessera_cache" / path_component
    manifest = cache / "manifest.parquet"
    landmasks = cache / "landmasks.parquet"
    for name, destination in [("manifest.parquet", manifest), ("landmasks.parquet", landmasks)]:
        url = f"{S3_HTTPS_ROOT}/{path_component}/{name}"
        download_file_to_temp(url, cache_path=destination)
    return manifest, landmasks


def lookup_embedding_rows(
    manifest: Path,
    tiles: list[tuple[float, float]],
    year: int,
    version: str,
    variant: str,
) -> pd.DataFrame:
    lon_values = sorted({round(lon * 100) for lon, _ in tiles})
    lat_values = sorted({round(lat * 100) for _, lat in tiles})
    table = pq.read_table(
        manifest,
        columns=[
            "version",
            "variant",
            "year",
            "lon",
            "lat",
            "grid_path",
            "scales_path",
            "grid_mtime",
            "scales_mtime",
            "grid_size",
            "scales_size",
            "lon_i",
            "lat_i",
        ],
        filters=[
            ("year", "=", year),
            ("lon_i", "in", lon_values),
            ("lat_i", "in", lat_values),
        ],
    )
    frame = table.to_pandas()
    wanted = {(round(lon * 100), round(lat * 100)) for lon, lat in tiles}
    mask = [
        (int(lon_i), int(lat_i)) in wanted
        for lon_i, lat_i in zip(frame["lon_i"], frame["lat_i"], strict=True)
    ]
    frame = frame.loc[mask]
    frame = frame[
        frame["version"].astype(str).eq(version)
        & frame["variant"].astype(str).eq(variant)
    ].copy()
    return frame.sort_values(["lon", "lat"]).reset_index(drop=True)


def lookup_landmask_rows(landmasks: Path, tiles: list[tuple[float, float]]) -> pd.DataFrame:
    lon_values = sorted({round(lon * 100) for lon, _ in tiles})
    lat_values = sorted({round(lat * 100) for _, lat in tiles})
    table = pq.read_table(
        landmasks,
        columns=["lon", "lat", "file_size", "mtime", "key", "lon_i", "lat_i"],
        filters=[("lon_i", "in", lon_values), ("lat_i", "in", lat_values)],
    )
    frame = table.to_pandas()
    wanted = {(round(lon * 100), round(lat * 100)) for lon, lat in tiles}
    mask = [
        (int(lon_i), int(lat_i)) in wanted
        for lon_i, lat_i in zip(frame["lon_i"], frame["lat_i"], strict=True)
    ]
    return frame.loc[mask].sort_values(["lon", "lat"]).reset_index(drop=True)


def assert_complete_inventory(
    rows: pd.DataFrame,
    landmasks: pd.DataFrame,
    tiles: list[tuple[float, float]],
) -> None:
    wanted = {(round(lon, 2), round(lat, 2)) for lon, lat in tiles}
    found = {(round(lon, 2), round(lat, 2)) for lon, lat in zip(rows.lon, rows.lat, strict=True)}
    found_masks = {
        (round(lon, 2), round(lat, 2))
        for lon, lat in zip(landmasks.lon, landmasks.lat, strict=True)
    }
    if found != wanted:
        raise RuntimeError(f"Missing TESSERA embedding tiles: {sorted(wanted - found)}")
    if found_masks != wanted:
        raise RuntimeError(f"Missing TESSERA landmasks: {sorted(wanted - found_masks)}")


def json_value(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def preflight_record(
    selected_manifest: Path,
    rejected_manifest: Path,
    selected_rows: pd.DataFrame,
    rejected_rows: pd.DataFrame,
    tiles: list[tuple[float, float]],
    by_site: dict[str, list[str]],
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "checked_at": utc_now(),
        "required_tile_count": len(tiles),
        "required_tiles": [tile_name(tile) for tile in tiles],
        "required_tiles_by_site": by_site,
        "rejected_release": {
            "dataset_version": "1.1",
            "dataset_variant": "cambridge",
            "manifest_sha256": sha256(rejected_manifest),
            "available_required_tiles": int(len(rejected_rows)),
            "missing_required_tiles": [tile_name(tile) for tile in tiles],
        },
        "selected_release": {
            "dataset_version": config["predictors"]["dataset_version"],
            "dataset_variant": config["predictors"]["dataset_variant"],
            "manifest_sha256": sha256(selected_manifest),
            "available_required_tiles": int(len(selected_rows)),
            "download_bytes_embeddings_and_scales": int(
                selected_rows["grid_size"].sum() + selected_rows["scales_size"].sum()
            ),
        },
        "decision": "repin_entire_project_to_available_frozen_release_do_not_mix_feature_spaces",
    }


def local_tile_paths(root: Path, year: int, tile: tuple[float, float]) -> dict[str, Path]:
    name = tile_name(tile)
    directory = root / EMBEDDINGS_SUBDIR / str(year) / name
    return {
        "embedding": directory / f"{name}.npy",
        "scales": directory / f"{name}_scales.npy",
        "landmask": root / LANDMASKS_SUBDIR / f"{name}.tiff",
    }


def check_storage(root: Path, rows: pd.DataFrame, landmasks: pd.DataFrame, year: int) -> None:
    remaining = 0
    for row in rows.itertuples(index=False):
        paths = local_tile_paths(root, year, (float(row.lon), float(row.lat)))
        if not paths["embedding"].exists():
            remaining += int(row.grid_size)
        if not paths["scales"].exists():
            remaining += int(row.scales_size)
    for row in landmasks.itertuples(index=False):
        path = local_tile_paths(root, year, (float(row.lon), float(row.lat)))["landmask"]
        if not path.exists():
            remaining += int(row.file_size)
    root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(root).free
    if free < remaining + DOWNLOAD_MARGIN_BYTES:
        raise RuntimeError(
            f"Insufficient storage: {free / 1e9:.2f} GB free, "
            f"{(remaining + DOWNLOAD_MARGIN_BYTES) / 1e9:.2f} GB required"
        )


def download_tiles(
    root: Path,
    rows: pd.DataFrame,
    landmasks: pd.DataFrame,
    year: int,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    landmask_index = {
        (round(float(row.lon), 2), round(float(row.lat), 2)): row
        for row in landmasks.itertuples(index=False)
    }
    records: list[dict[str, Any]] = []
    for position, row in enumerate(rows.itertuples(index=False), start=1):
        tile = (round(float(row.lon), 2), round(float(row.lat), 2))
        name = tile_name(tile)
        paths = local_tile_paths(root, year, tile)
        mask_row = landmask_index[tile]
        sources = {
            "embedding": str(row.grid_path),
            "scales": str(row.scales_path),
            "landmask": str(mask_row.key),
        }
        expected_sizes = {
            "embedding": int(row.grid_size),
            "scales": int(row.scales_size),
            "landmask": int(mask_row.file_size),
        }
        print(f"[{position}/{len(rows)}] acquire {name}", flush=True)
        for kind in ["embedding", "scales", "landmask"]:
            download_file_to_temp(s3_url(sources[kind]), cache_path=paths[kind])
            if paths[kind].stat().st_size != expected_sizes[kind]:
                raise RuntimeError(f"Wrong {kind} size for {name}")
        quantized = np.load(paths["embedding"], mmap_mode="r")
        scales = np.load(paths["scales"], mmap_mode="r")
        with rasterio.open(paths["landmask"]) as dataset:
            profile = {
                "shape": [dataset.height, dataset.width],
                "crs": dataset.crs.to_string(),
                "transform": list(dataset.transform)[:6],
                "bounds": list(dataset.bounds),
            }
        if quantized.dtype != np.int8 or quantized.ndim != 3 or quantized.shape[2] != 128:
            raise RuntimeError(f"Unexpected embedding array for {name}: {quantized.shape} {quantized.dtype}")
        if scales.shape != quantized.shape[:2] or profile["shape"] != list(quantized.shape[:2]):
            raise RuntimeError(f"Grid shape mismatch for {name}")
        records.append(
            {
                "tile": name,
                "lon": tile[0],
                "lat": tile[1],
                "year": year,
                "dataset_version": config["predictors"]["dataset_version"],
                "dataset_variant": config["predictors"]["dataset_variant"],
                "sources": sources,
                "source_mtime": {
                    "embedding": json_value(row.grid_mtime),
                    "scales": json_value(row.scales_mtime),
                    "landmask": json_value(mask_row.mtime),
                },
                "source_integrity": "S3_CRC64NVME_validated_by_botocore",
                "local_files": {
                    kind: {
                        "path": str(path.relative_to(ROOT)),
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                    for kind, path in paths.items()
                },
                "embedding_shape": list(quantized.shape),
                "embedding_dtype": str(quantized.dtype),
                "scales_dtype": str(scales.dtype),
                **profile,
            }
        )
    write_tessera_metadata(
        root,
        config["predictors"]["dataset_version"],
        config["predictors"]["dataset_variant"],
        extra={"year": year, "required_tile_count": len(records)},
    )
    return records


def tile_domain_polygon(lon: float, lat: float, transformer: Transformer) -> Polygon:
    west, east = lon - 0.05, lon + 0.05
    south, north = lat - 0.05, lat + 0.05
    steps = np.linspace(0, 1, 33)
    coordinates: list[tuple[float, float]] = []
    coordinates.extend((west + (east - west) * t, north) for t in steps)
    coordinates.extend((east, north - (north - south) * t) for t in steps[1:])
    coordinates.extend((east - (east - west) * t, south) for t in steps[1:])
    coordinates.extend((west, south + (north - south) * t) for t in steps[1:])
    xs, ys = transformer.transform(
        [coordinate[0] for coordinate in coordinates],
        [coordinate[1] for coordinate in coordinates],
    )
    return Polygon(zip(xs, ys, strict=True))


def load_tiles(records: list[dict[str, Any]]) -> dict[tuple[float, float], TileData]:
    result: dict[tuple[float, float], TileData] = {}
    for record in records:
        paths = {kind: ROOT / value["path"] for kind, value in record["local_files"].items()}
        quantized = np.load(paths["embedding"], mmap_mode="r")
        scales = np.load(paths["scales"], mmap_mode="r")
        with rasterio.open(paths["landmask"]) as dataset:
            landmask = dataset.read(1)
            crs = dataset.crs
            transform = dataset.transform
            height, width = dataset.height, dataset.width
        forward = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        key = (round(float(record["lon"]), 2), round(float(record["lat"]), 2))
        result[key] = TileData(
            lon=key[0],
            lat=key[1],
            name=record["tile"],
            quantized=quantized,
            scales=scales,
            landmask=landmask,
            crs=crs,
            transform=transform,
            height=height,
            width=width,
            domain=tile_domain_polygon(key[0], key[1], forward),
        )
    return result


def valid_pixel(tile: TileData, row: int, col: int) -> bool:
    scale = float(tile.scales[row, col])
    return bool(tile.landmask[row, col] > 0 and math.isfinite(scale) and scale > 0)


def embedding_at(tile: TileData, row: int, col: int) -> np.ndarray:
    return tile.quantized[row, col].astype(np.float32) * np.float32(tile.scales[row, col])


def pixel_ranges(tile: TileData, bounds: tuple[float, float, float, float]) -> tuple[range, range]:
    minx, miny, maxx, maxy = bounds
    row_a, col_a = rowcol(tile.transform, minx, maxy)
    row_b, col_b = rowcol(tile.transform, maxx, miny)
    row_min = max(0, min(int(row_a), int(row_b)))
    row_max = min(tile.height - 1, max(int(row_a), int(row_b)))
    col_min = max(0, min(int(col_a), int(col_b)))
    col_max = min(tile.width - 1, max(int(col_a), int(col_b)))
    return range(row_min, row_max + 1), range(col_min, col_max + 1)


def align_one(
    lon: float,
    lat: float,
    tiles: dict[tuple[float, float], TileData],
    radius_m: float,
    gaussian_sigma_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    center_key = tile_from_world(lon, lat)
    center_tile = tiles[center_key]
    forward = Transformer.from_crs("EPSG:4326", center_tile.crs, always_xy=True)
    x_center, y_center = forward.transform(lon, lat)
    center_row, center_col = rowcol(center_tile.transform, x_center, y_center)
    center_valid = (
        0 <= center_row < center_tile.height
        and 0 <= center_col < center_tile.width
        and valid_pixel(center_tile, center_row, center_col)
    )
    center_vector = (
        embedding_at(center_tile, center_row, center_col)
        if center_valid
        else np.full(128, np.nan, dtype=np.float32)
    )

    touched = footprint_tiles(lon, lat, radius_m)
    missing = touched - set(tiles)
    if missing:
        raise RuntimeError(f"Footprint requires unavailable tiles: {sorted(missing)}")
    touched_crs = {tiles[key].crs.to_string() for key in touched}
    if len(touched_crs) != 1:
        raise RuntimeError("A GEDI footprint crosses TESSERA UTM zones")

    circle = Point(x_center, y_center).buffer(radius_m, quad_segs=32)
    vectors: list[np.ndarray] = []
    areas: list[float] = []
    gaussian_weights: list[float] = []
    intersecting_pixels = 0
    valid_pixels = 0
    geometric_area = 0.0
    used_tiles: set[str] = set()
    for key in touched:
        tile = tiles[key]
        footprint_in_tile = circle.intersection(tile.domain)
        if footprint_in_tile.is_empty:
            continue
        rows, columns = pixel_ranges(tile, footprint_in_tile.bounds)
        for row in rows:
            for col in columns:
                left, bottom, right, top = rasterio.windows.bounds(
                    Window(col, row, 1, 1), tile.transform
                )
                overlap = footprint_in_tile.intersection(box(left, bottom, right, top)).area
                if overlap <= 1e-8:
                    continue
                intersecting_pixels += 1
                geometric_area += overlap
                if not valid_pixel(tile, row, col):
                    continue
                valid_pixels += 1
                used_tiles.add(tile.name)
                vector = embedding_at(tile, row, col)
                pixel_x, pixel_y = xy(tile.transform, row, col, offset="center")
                distance = math.hypot(pixel_x - x_center, pixel_y - y_center)
                vectors.append(vector)
                areas.append(overlap)
                gaussian_weights.append(
                    overlap * math.exp(-0.5 * (distance / gaussian_sigma_m) ** 2)
                )
    if not vectors:
        area_vector = np.full(128, np.nan, dtype=np.float32)
        gaussian_vector = np.full(128, np.nan, dtype=np.float32)
        valid_area = 0.0
    else:
        matrix = np.stack(vectors).astype(np.float32)
        area_weights = np.asarray(areas, dtype=np.float64)
        gaussian = np.asarray(gaussian_weights, dtype=np.float64)
        area_vector = np.average(matrix, axis=0, weights=area_weights).astype(np.float32)
        gaussian_vector = np.average(matrix, axis=0, weights=gaussian).astype(np.float32)
        valid_area = float(area_weights.sum())
    metadata = {
        "tessera_center_tile": center_tile.name,
        "tessera_center_pixel_row": int(center_row),
        "tessera_center_pixel_col": int(center_col),
        "tessera_crs": center_tile.crs.to_string(),
        "tessera_center_valid": bool(center_valid),
        "tessera_intersecting_pixel_count": intersecting_pixels,
        "tessera_valid_pixel_count": valid_pixels,
        "tessera_geometric_footprint_fraction": geometric_area / circle.area,
        "tessera_valid_footprint_fraction": valid_area / circle.area,
        "tessera_crosses_tile_boundary": len(touched) > 1,
        "tessera_tiles_used": ";".join(sorted(used_tiles)),
    }
    return center_vector, area_vector, gaussian_vector, metadata


def align_targets(
    targets: pd.DataFrame,
    tiles: dict[tuple[float, float], TileData],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, Counter[str]]:
    count = len(targets)
    center = np.full((count, 128), np.nan, dtype=np.float32)
    area = np.full((count, 128), np.nan, dtype=np.float32)
    gaussian = np.full((count, 128), np.nan, dtype=np.float32)
    metadata: list[dict[str, Any]] = []
    usage: Counter[str] = Counter()
    radius = float(config["target"]["footprint_diameter_m_approx"]) / 2
    sigma = float(config["alignment"]["gaussian_sigma_m"])
    for index, row in enumerate(targets[["longitude", "latitude"]].itertuples(index=False)):
        center[index], area[index], gaussian[index], record = align_one(
            float(row.longitude), float(row.latitude), tiles, radius, sigma
        )
        metadata.append(record)
        for name in record["tessera_tiles_used"].split(";"):
            if name:
                usage[name] += 1
        if (index + 1) % 500 == 0 or index + 1 == count:
            print(f"aligned {index + 1:,}/{count:,} GEDI footprints", flush=True)
    embedding_frames = [
        pd.DataFrame(center, columns=[f"tessera_center_{index:03d}" for index in range(128)]),
        pd.DataFrame(area, columns=[f"tessera_area_{index:03d}" for index in range(128)]),
        pd.DataFrame(gaussian, columns=[f"tessera_gaussian_{index:03d}" for index in range(128)]),
    ]
    result = pd.concat(
        [targets.reset_index(drop=True), pd.DataFrame(metadata), *embedding_frames], axis=1
    )
    result["tessera_dataset_version"] = str(config["predictors"]["dataset_version"])
    result["tessera_dataset_variant"] = str(config["predictors"]["dataset_variant"])
    result["tessera_embedding_year"] = int(config["predictors"]["embedding_year"])
    result["tessera_gaussian_sigma_m"] = sigma
    threshold = float(config["alignment"]["minimum_valid_footprint_fraction"])
    result["passes_tessera_alignment"] = (
        result["tessera_center_valid"]
        & result["tessera_valid_footprint_fraction"].ge(threshold)
        & np.isfinite(area).all(axis=1)
        & np.isfinite(gaussian).all(axis=1)
    )
    return result, usage


def missingness_table(aligned: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for site, frame in aligned.groupby("site_id", sort=True):
        records.append(
            {
                "site_id": site,
                "rows": len(frame),
                "failed_alignment_rows": int((~frame["passes_tessera_alignment"]).sum()),
                "invalid_center_rows": int((~frame["tessera_center_valid"]).sum()),
                "cross_tile_footprints": int(frame["tessera_crosses_tile_boundary"].sum()),
                "minimum_valid_footprint_fraction": float(frame["tessera_valid_footprint_fraction"].min()),
                "mean_valid_footprint_fraction": float(frame["tessera_valid_footprint_fraction"].mean()),
                "minimum_geometric_footprint_fraction": float(frame["tessera_geometric_footprint_fraction"].min()),
                "minimum_valid_pixel_count": int(frame["tessera_valid_pixel_count"].min()),
                "maximum_valid_pixel_count": int(frame["tessera_valid_pixel_count"].max()),
            }
        )
    return pd.DataFrame(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Replace an existing alignment freeze")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config()
    predictors = config["predictors"]
    if (str(predictors["dataset_version"]), str(predictors["dataset_variant"])) != ("1.0", "vultr"):
        raise RuntimeError("Phase 2 is availability-pinned to TESSERA 1.0/vultr")
    target_path = ROOT / "data/processed/gedi_v3_2024_primary_forest.parquet"
    targets = pd.read_parquet(target_path)
    if set(targets["site_id"]) != EXPECTED_SITES or targets["freeze_id"].nunique() != 1:
        raise RuntimeError("Input is not the frozen three-site GEDI sample")

    radius = float(config["target"]["footprint_diameter_m_approx"]) / 2
    tiles, by_site = required_tiles(targets, radius)
    if len(tiles) != 8:
        raise RuntimeError(f"Expected exactly 8 required TESSERA tiles, found {len(tiles)}")

    selected_manifest, selected_landmasks = ensure_registry("1.0")
    rejected_manifest, _ = ensure_registry("1.1")
    selected_rows = lookup_embedding_rows(
        selected_manifest, tiles, int(predictors["embedding_year"]), "1.0", "vultr"
    )
    rejected_rows = lookup_embedding_rows(
        rejected_manifest, tiles, int(predictors["embedding_year"]), "1.1", "cambridge"
    )
    landmask_rows = lookup_landmask_rows(selected_landmasks, tiles)
    assert_complete_inventory(selected_rows, landmask_rows, tiles)
    if not rejected_rows.empty:
        raise RuntimeError("The v1.1/cambridge availability decision has changed; review before proceeding")
    preflight = preflight_record(
        selected_manifest,
        rejected_manifest,
        selected_rows,
        rejected_rows,
        tiles,
        by_site,
        config,
    )
    json_dump(ROOT / "metadata/tessera_phase2_preflight.json", preflight)

    embeddings_root = ROOT / "data/external/geotessera_embeddings"
    check_storage(
        embeddings_root,
        selected_rows,
        landmask_rows,
        int(predictors["embedding_year"]),
    )
    tile_records = download_tiles(
        embeddings_root,
        selected_rows,
        landmask_rows,
        int(predictors["embedding_year"]),
        config,
    )
    tile_manifest_path = ROOT / "metadata/tessera_v1_2024_tile_manifest.json"
    json_dump(
        tile_manifest_path,
        {
            "created_at": utc_now(),
            "dataset_version": predictors["dataset_version"],
            "dataset_variant": predictors["dataset_variant"],
            "year": predictors["embedding_year"],
            "geotessera_version": importlib.metadata.version("geotessera"),
            "tile_count": len(tile_records),
            "tiles": tile_records,
        },
    )

    development_path = ROOT / "data/processed/tessera_aligned_development_soap_teak.parquet"
    locked_path = ROOT / "data/processed/tessera_aligned_locked_bart.parquet"
    freeze_path = ROOT / "metadata/tessera_phase2_alignment_freeze.json"
    missingness_path = ROOT / "metadata/tessera_phase2_missingness.csv"
    usage_path = ROOT / "metadata/tessera_phase2_tile_usage.csv"
    protected = [development_path, locked_path, freeze_path, missingness_path, usage_path]
    if any(path.exists() for path in protected) and not args.force:
        existing = [str(path.relative_to(ROOT)) for path in protected if path.exists()]
        raise RuntimeError(f"Alignment freeze exists; refusing to overwrite: {existing}")

    loaded_tiles = load_tiles(tile_records)
    aligned, usage = align_targets(targets, loaded_tiles, config)
    if not aligned["passes_tessera_alignment"].all():
        failure = missingness_table(aligned)
        failure.to_csv(missingness_path, index=False)
        raise RuntimeError("One or more GEDI footprints failed TESSERA alignment")

    gedi_manifest = json.loads((ROOT / "metadata/gedi_v3_2024_freeze.json").read_text())
    freeze_basis = {
        "gedi_freeze_id": targets["freeze_id"].iloc[0],
        "gedi_primary_sha256": gedi_manifest["outputs"]["primary_sample"]["sha256"],
        "project_config_sha256": sha256(ROOT / "configs/project.yaml"),
        "alignment_script_sha256": sha256(Path(__file__).resolve()),
        "geotessera_version": importlib.metadata.version("geotessera"),
        "registry_manifest_sha256": sha256(selected_manifest),
        "landmask_manifest_sha256": sha256(selected_landmasks),
        "dataset_version": predictors["dataset_version"],
        "dataset_variant": predictors["dataset_variant"],
        "embedding_year": predictors["embedding_year"],
        "tiles": [
            {
                "tile": record["tile"],
                "local_files": {
                    kind: value["sha256"] for kind, value in record["local_files"].items()
                },
            }
            for record in tile_records
        ],
        "alignment": {
            "footprint_radius_m": radius,
            "primary": "pixel_area_weighted_mean_inside_circular_footprint",
            "center": "containing_10m_pixel",
            "gaussian": "overlap_area_times_gaussian_at_pixel_center",
            "gaussian_sigma_m": config["alignment"]["gaussian_sigma_m"],
            "minimum_valid_footprint_fraction": config["alignment"]["minimum_valid_footprint_fraction"],
            "tile_edge_rule": "clip_pixel_overlap_to_projected_0.1_degree_tile_domain",
        },
    }
    alignment_id = f"tessera-align-{canonical_hash(freeze_basis)[:12]}"
    aligned.insert(0, "tessera_alignment_id", alignment_id)
    development = aligned[aligned["site_id"].isin(config["sites"]["development"])].copy()
    locked = aligned[aligned["site_id"].eq(config["sites"]["locked_transfer_test"])].copy()
    if set(development["site_id"]) != {"SOAP", "TEAK"} or set(locked["site_id"]) != {"BART"}:
        raise RuntimeError("Development/locked partition is incorrect")
    write_parquet_atomic(development, development_path)
    write_parquet_atomic(locked, locked_path)

    missingness = missingness_table(aligned)
    missingness.to_csv(missingness_path, index=False)
    usage_frame = pd.DataFrame(
        [{"tile": name, "footprints_using_tile": count} for name, count in sorted(usage.items())]
    )
    usage_frame.to_csv(usage_path, index=False)
    manifest = {
        "alignment_id": alignment_id,
        "created_at": utc_now(),
        "status": "frozen",
        "freeze_basis_sha256": canonical_hash(freeze_basis),
        "freeze_basis": freeze_basis,
        "partition_rule": {
            "development": config["sites"]["development"],
            "locked_transfer_test": config["sites"]["locked_transfer_test"],
            "bart_must_not_be_used_for_model_selection": True,
        },
        "outputs": {
            "development": {
                "path": str(development_path.relative_to(ROOT)),
                "rows": len(development),
                "sha256": sha256(development_path),
            },
            "locked_bart": {
                "path": str(locked_path.relative_to(ROOT)),
                "rows": len(locked),
                "sha256": sha256(locked_path),
            },
            "missingness": {
                "path": str(missingness_path.relative_to(ROOT)),
                "sha256": sha256(missingness_path),
            },
            "tile_usage": {
                "path": str(usage_path.relative_to(ROOT)),
                "sha256": sha256(usage_path),
            },
        },
        "site_summary": missingness.to_dict(orient="records"),
    }
    json_dump(freeze_path, manifest)
    print(
        json.dumps(
            {
                "alignment_id": alignment_id,
                "development_rows": len(development),
                "locked_bart_rows": len(locked),
                "tile_count": len(tile_records),
                "site_summary": manifest["site_summary"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
