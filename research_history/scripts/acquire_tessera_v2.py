#!/usr/bin/env python3
"""Stream experimental TESSERA v2 inputs for one frozen forest cohort."""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from numpy.lib.format import open_memmap
from pyproj import Transformer

import tessera_v2_common as v2


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/tessera_v2.yaml"


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "phase27_tessera_v2"
    ]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npy")
    np.save(temporary, values)
    temporary.replace(path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_points(
    cohort: pd.DataFrame, settings: dict[str, Any], masked: bool
) -> dict[str, np.ndarray]:
    rows = np.repeat(np.arange(5, dtype=np.int8), 5)
    columns = np.tile(np.arange(5, dtype=np.int8), 5)
    offset_x = columns.astype(np.float64) - 2.0
    offset_y = 2.0 - rows.astype(np.float64)
    keep = np.ones((len(cohort), 25), dtype=bool)
    if masked:
        bits = (1 << np.arange(25, dtype=np.uint32))[None, :]
        masks = cohort[str(settings["pixel_mask_column"])].to_numpy(
            dtype=np.uint32
        )[:, None]
        keep = (masks & bits) > 0
    full_index = np.arange(len(cohort) * 25, dtype=np.int64)
    selected = keep.reshape(-1)
    x = (
        np.repeat(
            cohort[str(settings["x_column"])].to_numpy(dtype=np.float64), 25
        )
        + np.tile(offset_x * 10.0, len(cohort))
    )[selected]
    y = (
        np.repeat(
            cohort[str(settings["y_column"])].to_numpy(dtype=np.float64), 25
        )
        + np.tile(offset_y * 10.0, len(cohort))
    )[selected]
    return {"point_index": full_index[selected], "x": x, "y": y}


def preflight_sources(year: int, tiles: list[tuple[float, float]]) -> None:
    missing: list[str] = []
    for tile in tiles:
        for url in v2.source_urls(year, tile).values():
            completed = subprocess.run(
                ["curl", "--silent", "--fail", "--head", "--max-time", "30", url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if completed.returncode != 0:
                missing.append(url)
    if missing:
        raise RuntimeError(
            "Experimental v2 coverage is missing before acquisition:\n"
            + "\n".join(missing)
        )


def save_chunk(
    path: Path,
    point_index: np.ndarray,
    quantized: np.ndarray,
    scales: np.ndarray,
    config_hash: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        point_index=point_index,
        quantized=quantized,
        scales=scales,
        config_sha256=np.asarray([config_hash]),
    )
    temporary.replace(path)


def feature_frame(
    cohort: pd.DataFrame,
    summary: np.ndarray,
    valid: np.ndarray,
    minimum: int,
) -> pd.DataFrame:
    counts = valid.sum(axis=(1, 2)).astype(np.int16)
    output: dict[str, Any] = {
        "row_id": cohort["row_id"].to_numpy(dtype=np.int64),
        "tessera_valid_pixel_count": counts,
        "tessera_context_valid": counts >= minimum,
    }
    slices = {
        "mean": slice(128, 256),
        "std": slice(256, 384),
        "dx": slice(384, 512),
        "dy": slice(512, 640),
    }
    for prefix, selected in slices.items():
        values = summary[:, selected]
        for dimension in range(v2.DIMENSIONS):
            output[f"tessera_{prefix}_{dimension:03d}"] = values[:, dimension]
    return pd.DataFrame(output)


def output_paths(site: str, config: dict[str, Any]) -> dict[str, Path]:
    settings = config[site]
    if site == "savelsbos":
        return {"features": ROOT / str(settings["v2_features"])}
    directory = ROOT / str(settings["v2_patch_directory"])
    return {
        "features": ROOT / str(settings["v2_features"]),
        "patch": directory / "patch_quantized.npy",
        "scales": directory / "patch_scales.npy",
        "valid": directory / "patch_valid.npy",
        "row_id": directory / "row_id.npy",
    }


def run(site: str) -> None:
    config = load_config()
    settings = config[site]
    acquisition = config["acquisition"]
    cohort_key = "corrected_cohort" if site == "cairngorms" else "cohort"
    cohort_path = ROOT / str(settings[cohort_key])
    cohort = pd.read_parquet(cohort_path).sort_values("row_id").reset_index(drop=True)
    paths = output_paths(site, config)
    manifest_path = ROOT / f"metadata/phase27_{site}_v2_acquisition.json"
    if manifest_path.exists() and all(path.exists() for path in paths.values()):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest["cohort_sha256"] == v2.sha256(cohort_path)
            and manifest["config_sha256"] == v2.sha256(CONFIG_PATH)
            and all(
                manifest["output_sha256"][name] == v2.sha256(path)
                for name, path in paths.items()
            )
        ):
            print(f"phase27 {site}: validated existing acquisition", flush=True)
            return
        raise RuntimeError(f"Existing Phase 27 {site} acquisition is stale")

    masked = site == "savelsbos"
    points = build_points(cohort, settings, masked)
    to_world = Transformer.from_crs(
        str(settings["analysis_crs"]), "EPSG:4326", always_xy=True
    )
    longitude, latitude = to_world.transform(points["x"], points["y"])
    tile_pairs = [
        v2.tile_center(float(lon), float(lat))
        for lon, lat in zip(longitude, latitude, strict=True)
    ]
    tiles = sorted(set(tile_pairs))
    tile_lookup = {tile: code for code, tile in enumerate(tiles)}
    tile_code = np.asarray([tile_lookup[tile] for tile in tile_pairs], dtype=np.int16)
    year = int(settings["year"])
    preflight_sources(year, tiles)

    chunk_root = (
        ROOT / str(acquisition["chunk_directory"]) / site
    )
    stream_root = (
        ROOT / str(acquisition["stream_directory"]) / site
    )
    config_hash = v2.sha256(CONFIG_PATH)
    records: list[dict[str, Any]] = []
    for code, tile in enumerate(tiles):
        name = v2.tile_name(tile)
        point_rows = np.flatnonzero(tile_code == code)
        chunk_path = chunk_root / f"{name}.npz"
        if chunk_path.exists():
            with np.load(chunk_path) as chunk:
                if (
                    str(chunk["config_sha256"][0]) != config_hash
                    or len(chunk["point_index"]) > len(point_rows)
                ):
                    raise RuntimeError(f"Stale v2 tile chunk: {chunk_path}")
            records.append(
                {"tile": name, "source_points": int(len(point_rows)), "resumed": True}
            )
            print(f"v2 {site} [{code + 1}/{len(tiles)}]: resumed {name}", flush=True)
            continue

        urls = v2.source_urls(year, tile)
        local = {
            kind: stream_root / name / Path(url).name
            for kind, url in urls.items()
        }
        free = shutil.disk_usage(ROOT).free
        if free < int(acquisition["minimum_free_space_after_download_bytes"]):
            raise RuntimeError("Insufficient free disk space before v2 download")
        print(
            f"v2 {site} [{code + 1}/{len(tiles)}]: acquire {name} for "
            f"{len(point_rows):,} requested pixels",
            flush=True,
        )
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
                str(settings["analysis_crs"]),
                points["x"][point_rows],
                points["y"][point_rows],
            )
            source_rows = point_rows[selected]
            save_chunk(
                chunk_path,
                points["point_index"][source_rows],
                values,
                local_scales,
                config_hash,
            )
            records.append(
                {
                    "tile": name,
                    "source_points": int(len(point_rows)),
                    "valid_points": int(len(source_rows)),
                    "embedding_sha256": v2.sha256(local["embedding"]),
                    "scales_sha256": v2.sha256(local["scales"]),
                    "resumed": False,
                }
            )
        finally:
            for path in local.values():
                path.unlink(missing_ok=True)
            gc.collect()

    directory = paths["features"].parent
    directory.mkdir(parents=True, exist_ok=True)
    patch_path = paths.get("patch", directory / "phase27_patch_quantized.npy")
    scale_path = paths.get("scales", directory / "phase27_patch_scales.npy")
    valid_path = paths.get("valid", directory / "phase27_patch_valid.npy")
    temporary_patch = patch_path.with_suffix(".tmp.npy")
    temporary_scale = scale_path.with_suffix(".tmp.npy")
    temporary_valid = valid_path.with_suffix(".tmp.npy")
    shape = (len(cohort), v2.DIMENSIONS, 5, 5)
    patch = open_memmap(temporary_patch, mode="w+", dtype=np.int8, shape=shape)
    scales = open_memmap(
        temporary_scale, mode="w+", dtype=np.float16, shape=(len(cohort), 5, 5)
    )
    valid = open_memmap(
        temporary_valid, mode="w+", dtype=np.uint8, shape=(len(cohort), 5, 5)
    )
    patch[:] = 0
    scales[:] = 0
    valid[:] = 0
    for tile in tiles:
        with np.load(chunk_root / f"{v2.tile_name(tile)}.npz") as chunk:
            point_index = chunk["point_index"].astype(np.int64)
            patch_id = point_index // 25
            position = point_index % 25
            local_row = position // 5
            local_column = position % 5
            patch[patch_id, :, local_row, local_column] = chunk["quantized"]
            scales[patch_id, local_row, local_column] = chunk["scales"].astype(
                np.float16
            )
            valid[patch_id, local_row, local_column] = 1
    patch.flush()
    scales.flush()
    valid.flush()
    summary = v2.summarize_ordered_patch(patch, scales, valid)
    valid_count = np.asarray(valid).sum(axis=(1, 2))
    minimum = int(settings["minimum_valid_pixels"])
    if int((valid_count >= minimum).sum()) < int(0.95 * len(cohort)):
        raise RuntimeError(
            f"Only {(valid_count >= minimum).sum():,}/{len(cohort):,} v2 patches "
            "passed the frozen validity gate"
        )

    if site == "savelsbos":
        atomic_parquet(paths["features"], feature_frame(cohort, summary, valid, minimum))
        temporary_patch.unlink(missing_ok=True)
        temporary_scale.unlink(missing_ok=True)
        temporary_valid.unlink(missing_ok=True)
    else:
        del patch, scales, valid
        gc.collect()
        temporary_patch.replace(paths["patch"])
        temporary_scale.replace(paths["scales"])
        temporary_valid.replace(paths["valid"])
        atomic_npy(paths["features"], summary)
        atomic_npy(paths["row_id"], cohort["row_id"].to_numpy(dtype=np.int64))

    manifest = {
        "created_utc": utc_now(),
        "site": site,
        "year": year,
        "representation": "experimental-v2",
        "dimensions": v2.DIMENSIONS,
        "supplied_loader_sha256": config["representation"][
            "supplied_loader_sha256"
        ],
        "cohort_sha256": v2.sha256(cohort_path),
        "config_sha256": config_hash,
        "aggregation": (
            "ordered 5 x 5 centre, mean, SD and planar gradients"
            if site == "cairngorms"
            else "mean, SD and planar gradients over mapped broadleaf pixels"
        ),
        "tiles": records,
        "valid_rows": int((valid_count >= minimum).sum()),
        "output_sha256": {
            name: v2.sha256(path) for name, path in paths.items()
        },
    }
    atomic_json(manifest_path, manifest)
    shutil.rmtree(chunk_root, ignore_errors=True)
    shutil.rmtree(stream_root, ignore_errors=True)
    print(
        f"phase27 {site}: {manifest['valid_rows']:,}/{len(cohort):,} valid rows "
        f"across {len(tiles)} v2 tiles",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", choices=["cairngorms", "savelsbos"], required=True)
    run(parser.parse_args().site)


if __name__ == "__main__":
    main()
