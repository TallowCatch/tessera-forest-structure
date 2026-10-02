#!/usr/bin/env python3
"""Stream TESSERA v2 onto the frozen dense Cairngorms 10 m grid."""

from __future__ import annotations

import gc
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import yaml
from numpy.lib.format import open_memmap
from pyproj import Transformer

import tessera_v2_common as v2


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/cairngorms_v2_unet.yaml"


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "phase28_cairngorms_v2_unet"
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


def source_coordinates(
    transform: rasterio.Affine, rows: np.ndarray, columns: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    x = transform.c + (columns.astype(np.float64) + 0.5) * transform.a
    y = transform.f + (rows.astype(np.float64) + 0.5) * transform.e
    return x, y


def initialize_outputs(
    directory: Path, shape: tuple[int, int], state_path: Path, config_hash: str
) -> None:
    paths = {
        "embedding": directory / "embedding_int8.npy",
        "scale": directory / "embedding_scale.npy",
        "valid": directory / "tessera_valid.npy",
    }
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("config_sha256") != config_hash:
            raise RuntimeError("Existing Phase 28 acquisition uses another config")
        if not all(path.exists() for path in paths.values()):
            raise RuntimeError("Phase 28 dense acquisition is incomplete and inconsistent")
        return
    if any(path.exists() for path in paths.values()):
        raise RuntimeError("Untracked Phase 28 dense outputs already exist")
    directory.mkdir(parents=True, exist_ok=True)
    embedding = open_memmap(
        paths["embedding"], mode="w+", dtype=np.int8, shape=(*shape, v2.DIMENSIONS)
    )
    scale = open_memmap(paths["scale"], mode="w+", dtype=np.float16, shape=shape)
    valid = open_memmap(paths["valid"], mode="w+", dtype=np.uint8, shape=shape)
    embedding[:] = 0
    scale[:] = 0
    valid[:] = 0
    embedding.flush()
    scale.flush()
    valid.flush()
    atomic_json(
        state_path,
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "config_sha256": config_hash,
            "shape": [*shape, v2.DIMENSIONS],
        },
    )


def run() -> None:
    config = load_config()
    frozen = config["frozen_inputs"]
    acquisition = config["acquisition"]
    outputs = config["outputs"]
    loader = ROOT / str(config["representation"]["supplied_loader"])
    if sha256(loader) != str(config["representation"]["supplied_loader_sha256"]):
        raise RuntimeError("The supplied TESSERA v2 loader has changed")

    source_dense = ROOT / str(frozen["source_dense_directory"])
    source_valid_path = source_dense / "tessera_valid.npy"
    source_valid = np.load(source_valid_path, mmap_mode="r")
    rows, columns = np.nonzero(source_valid)
    shape = tuple(int(value) for value in source_valid.shape)
    with rasterio.open(ROOT / str(frozen["source_raster"])) as dataset:
        if dataset.shape != shape:
            raise RuntimeError(
                f"Dense grid {shape} does not match source raster {dataset.shape}"
            )
        transform = dataset.transform
        source_crs = (
            str(dataset.crs)
            if dataset.crs is not None
            else str(config["study"]["analysis_crs"])
        )
    x, y = source_coordinates(transform, rows, columns)
    to_world = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)
    longitude, latitude = to_world.transform(x, y)
    longitude_bin = np.floor(np.asarray(longitude) * 10).astype(np.int16)
    latitude_bin = np.floor(np.asarray(latitude) * 10).astype(np.int16)
    tile_codes = np.column_stack([longitude_bin, latitude_bin])
    tiles = np.unique(tile_codes, axis=0)

    dense_dir = ROOT / str(acquisition["dense_directory"])
    marker_dir = ROOT / str(acquisition["marker_directory"])
    stream_dir = ROOT / str(acquisition["stream_directory"])
    state_path = dense_dir / "acquisition_state.json"
    config_hash = sha256(CONFIG_PATH)
    initialize_outputs(dense_dir, shape, state_path, config_hash)
    marker_dir.mkdir(parents=True, exist_ok=True)
    embedding_output = np.load(dense_dir / "embedding_int8.npy", mmap_mode="r+")
    scale_output = np.load(dense_dir / "embedding_scale.npy", mmap_mode="r+")
    valid_output = np.load(dense_dir / "tessera_valid.npy", mmap_mode="r+")

    records: list[dict[str, Any]] = []
    year = int(config["study"]["tessera_year"])
    for tile_number, (lon_bin, lat_bin) in enumerate(tiles, start=1):
        tile = (round(float(lon_bin) / 10 + 0.05, 2), round(float(lat_bin) / 10 + 0.05, 2))
        name = v2.tile_name(tile)
        marker = marker_dir / f"{name}.json"
        chosen = np.flatnonzero(
            (longitude_bin == lon_bin) & (latitude_bin == lat_bin)
        )
        if marker.exists():
            record = json.loads(marker.read_text(encoding="utf-8"))
            if record.get("config_sha256") != config_hash:
                raise RuntimeError(f"Stale Phase 28 tile marker: {marker}")
            records.append(record)
            print(f"v2 dense [{tile_number}/{len(tiles)}]: resumed {name}", flush=True)
            continue
        free = shutil.disk_usage(ROOT).free
        if free < int(acquisition["minimum_free_space_after_download_bytes"]):
            raise RuntimeError("Insufficient free GWS space before v2 tile download")
        urls = v2.source_urls(year, tile)
        local_dir = stream_dir / name
        local = {kind: local_dir / Path(url).name for kind, url in urls.items()}
        print(
            f"v2 dense [{tile_number}/{len(tiles)}]: acquire {name} for "
            f"{len(chosen):,} valid grid cells",
            flush=True,
        )
        quantized = None
        scales = None
        try:
            for kind in ["embedding", "scales"]:
                v2.download(
                    urls[kind],
                    local[kind],
                    int(acquisition["download_attempts"]),
                    int(acquisition["retry_delay_seconds"]),
                )
            quantized = np.load(local["embedding"], mmap_mode="r")
            scales = np.load(local["scales"], mmap_mode="r")
            selected, values, local_scales = v2.sample_quantized_tile(
                quantized,
                scales,
                tile,
                source_crs,
                x[chosen],
                y[chosen],
            )
            destination = chosen[selected]
            embedding_output[rows[destination], columns[destination]] = values
            scale_output[rows[destination], columns[destination]] = local_scales.astype(
                np.float16
            )
            valid_output[rows[destination], columns[destination]] = 1
            embedding_output.flush()
            scale_output.flush()
            valid_output.flush()
            record = {
                "tile": name,
                "requested_cells": int(len(chosen)),
                "sampled_cells": int(len(destination)),
                "config_sha256": config_hash,
            }
            atomic_json(marker, record)
            records.append(record)
        finally:
            del quantized, scales
            gc.collect()
            for path in local.values():
                if path.exists():
                    path.unlink()
            if local_dir.exists():
                local_dir.rmdir()

    valid_count = int(np.asarray(valid_output).sum())
    requested_count = int(len(rows))
    coverage = valid_count / requested_count
    if coverage < float(acquisition["minimum_dense_coverage_fraction"]):
        raise RuntimeError(f"Dense v2 coverage is only {coverage:.3%}")
    manifest_path = ROOT / str(outputs["acquisition_manifest"])
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "state": "complete",
        "config_sha256": config_hash,
        "supplied_loader_sha256": sha256(loader),
        "source_valid_sha256": sha256(source_valid_path),
        "shape": [*shape, v2.DIMENSIONS],
        "requested_cells": requested_count,
        "valid_cells": valid_count,
        "coverage_fraction": coverage,
        "tiles": records,
        "outputs": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for name, path in {
                "embedding": dense_dir / "embedding_int8.npy",
                "scale": dense_dir / "embedding_scale.npy",
                "valid": dense_dir / "tessera_valid.npy",
            }.items()
        },
    }
    atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    run()
