#!/usr/bin/env python3
"""Shared access helpers for the experimental source.coop TESSERA v2 tiles."""

from __future__ import annotations

import hashlib
import math
import subprocess
from pathlib import Path

import numpy as np
from affine import Affine
from pyproj import Transformer
from rasterio.transform import rowcol


S3_HTTPS_ROOT = (
    "https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop"
)
S3_PREFIX = "tessera/tessera/npy/v2"
DIMENSIONS = 128


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tile_center(lon: float, lat: float) -> tuple[float, float]:
    """Return the 0.1 degree tile centre used by the supplied v2 loader."""
    tile_lon = np.floor(lon * 10) / 10 + 0.05
    tile_lat = np.floor(lat * 10) / 10 + 0.05
    return round(float(tile_lon), 2), round(float(tile_lat), 2)


def tile_name(tile: tuple[float, float]) -> str:
    return f"grid_{tile[0]:.2f}_{tile[1]:.2f}"


def utm_epsg(lon: float, lat: float) -> int:
    zone = max(1, min(60, int(math.floor((lon + 180.0) / 6.0)) + 1))
    return (32600 if lat >= 0 else 32700) + zone


def tile_transform_and_epsg(
    lon: float, lat: float, height: int, width: int
) -> tuple[Affine, int]:
    """Reproduce the source-tile geometry in Aneesh's supplied loader."""
    epsg = utm_epsg(lon, lat)
    projection = Transformer.from_crs(
        "EPSG:4326", f"EPSG:{epsg}", always_xy=True
    )
    west, east = lon - 0.05, lon + 0.05
    south, north = lat - 0.05, lat + 0.05
    ul_e, ul_n = projection.transform(west, north)
    ur_e, ur_n = projection.transform(east, north)
    ll_e, ll_n = projection.transform(west, south)
    lr_e, lr_n = projection.transform(east, south)
    origin_e = min(ul_e, ll_e)
    origin_n = max(ul_n, ur_n)
    max_e = max(ur_e, lr_e)
    min_n = min(ll_n, lr_n)
    transform = Affine(
        (max_e - origin_e) / width,
        0.0,
        origin_e,
        0.0,
        -(origin_n - min_n) / height,
        origin_n,
    )
    return transform, epsg


def source_urls(year: int, tile: tuple[float, float]) -> dict[str, str]:
    name = tile_name(tile)
    base = f"{S3_HTTPS_ROOT}/{S3_PREFIX}/{year}/{name}"
    return {
        "embedding": f"{base}/{name}.npy",
        "scales": f"{base}/{name}_scales.npy",
    }


def download(url: str, path: Path, attempts: int = 8, delay: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "curl",
            "--location",
            "--fail",
            "--retry",
            str(attempts),
            "--retry-delay",
            str(delay),
            "--retry-all-errors",
            "--continue-at",
            "-",
            "--output",
            str(path),
            url,
        ],
        check=True,
    )


def sample_quantized_tile(
    quantized: np.ndarray,
    scales: np.ndarray,
    tile: tuple[float, float],
    source_crs: str,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample requested coordinates without dequantising the complete tile."""
    if quantized.ndim != 3 or quantized.shape[2] != DIMENSIONS:
        raise RuntimeError(f"Unexpected v2 embedding shape: {quantized.shape}")
    if scales.ndim != 2 or scales.shape != quantized.shape[:2]:
        raise RuntimeError(f"Unexpected v2 scale shape: {scales.shape}")
    transform, epsg = tile_transform_and_epsg(
        tile[0], tile[1], quantized.shape[0], quantized.shape[1]
    )
    projection = Transformer.from_crs(
        source_crs, f"EPSG:{epsg}", always_xy=True
    )
    east, north = projection.transform(x, y)
    rows, columns = rowcol(transform, east, north)
    rows = np.asarray(rows, dtype=np.int64)
    columns = np.asarray(columns, dtype=np.int64)
    inside = (
        (rows >= 0)
        & (columns >= 0)
        & (rows < quantized.shape[0])
        & (columns < quantized.shape[1])
    )
    selected = np.flatnonzero(inside)
    local_scales = np.asarray(
        scales[rows[selected], columns[selected]], dtype=np.float32
    )
    # The supplied loader treats every finite cell in an available tile as
    # covered; v2 does not provide the separate land mask used by v1.
    valid = np.isfinite(local_scales)
    selected = selected[valid]
    values = np.asarray(
        quantized[rows[selected], columns[selected]], dtype=np.int8
    )
    return selected, values, local_scales[valid]


def summarize_ordered_patch(
    quantized: np.ndarray, scales: np.ndarray, valid: np.ndarray
) -> np.ndarray:
    """Return centre, mean, SD and planar gradients for ordered 5 x 5 patches."""
    values = quantized.astype(np.float32) * scales[:, None].astype(np.float32)
    mask = valid.astype(bool)
    values = np.where(mask[:, None], values, 0.0)
    count = np.maximum(mask.sum(axis=(1, 2)), 1).astype(np.float32)
    total = values.sum(axis=(2, 3))
    square = np.square(values).sum(axis=(2, 3))
    mean = total / count[:, None]
    standard_deviation = np.sqrt(
        np.maximum(square / count[:, None] - np.square(mean), 0.0)
    )
    cells = values.shape[-1]
    offsets = np.arange(cells, dtype=np.float32) - (cells - 1) / 2
    xx = np.broadcast_to(offsets[None], (cells, cells))
    yy = np.broadcast_to(-offsets[:, None], (cells, cells))
    sx = np.where(mask, xx[None], 0).sum(axis=(1, 2))
    sy = np.where(mask, yy[None], 0).sum(axis=(1, 2))
    sx2 = np.where(mask, np.square(xx)[None], 0).sum(axis=(1, 2))
    sy2 = np.where(mask, np.square(yy)[None], 0).sum(axis=(1, 2))
    sxv = (values * xx[None, None]).sum(axis=(2, 3))
    syv = (values * yy[None, None]).sum(axis=(2, 3))
    denominator_x = sx2 - np.square(sx) / count
    denominator_y = sy2 - np.square(sy) / count
    dx = np.zeros_like(mean)
    dy = np.zeros_like(mean)
    usable_x = denominator_x > 0
    usable_y = denominator_y > 0
    dx[usable_x] = (
        sxv[usable_x]
        - sx[usable_x, None] * total[usable_x] / count[usable_x, None]
    ) / denominator_x[usable_x, None]
    dy[usable_y] = (
        syv[usable_y]
        - sy[usable_y, None] * total[usable_y] / count[usable_y, None]
    ) / denominator_y[usable_y, None]
    centre = values[:, :, cells // 2, cells // 2]
    return np.concatenate(
        [centre, mean, standard_deviation, dx, dy], axis=1
    ).astype(np.float32)
