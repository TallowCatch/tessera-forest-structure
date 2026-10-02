#!/usr/bin/env python3
"""Acquire matching-year TESSERA v2 patches for the Phase 36 cohort."""

from __future__ import annotations

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
CONFIG_PATH = ROOT / "configs/dutch_multisite_transfer.yaml"


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "phase36_dutch_multisite_transfer"
    ]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def build_points(cohort: pd.DataFrame) -> dict[str, np.ndarray]:
    positions = np.arange(25, dtype=np.uint32)
    keep = (
        cohort["forest_pixel_mask"].to_numpy(dtype=np.uint32)[:, None]
        & (1 << positions)[None, :]
    ) > 0
    rows = np.repeat(np.arange(5, dtype=np.int8), 5)
    columns = np.tile(np.arange(5, dtype=np.int8), 5)
    selected = keep.reshape(-1)
    full_index = np.arange(len(cohort) * 25, dtype=np.int64)
    x = (
        np.repeat(cohort["rd_x"].to_numpy(dtype=np.float64), 25)
        + np.tile((columns - 2).astype(np.float64) * 10.0, len(cohort))
    )[selected]
    y = (
        np.repeat(cohort["rd_y"].to_numpy(dtype=np.float64), 25)
        + np.tile((2 - rows).astype(np.float64) * 10.0, len(cohort))
    )[selected]
    year = np.repeat(cohort["tessera_year"].to_numpy(dtype=np.int16), 25)[selected]
    return {"point_index": full_index[selected], "x": x, "y": y, "year": year}


def preflight(groups: list[tuple[int, tuple[float, float]]]) -> None:
    missing: list[str] = []
    for year, tile in groups:
        for url in v2.source_urls(year, tile).values():
            result = subprocess.run(
                ["curl", "--silent", "--fail", "--head", "--max-time", "30", url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if result.returncode != 0:
                missing.append(url)
    if missing:
        raise RuntimeError("Missing TESSERA v2 inputs:\n" + "\n".join(missing))


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


def run() -> None:
    config = load_config()
    settings = config["tessera"]
    outputs = config["outputs"]
    cohort_path = ROOT / str(outputs["cohort"])
    feature_path = ROOT / str(outputs["features"])
    manifest_path = ROOT / str(outputs["acquisition_freeze"])
    cohort = pd.read_parquet(cohort_path).sort_values("row_id").reset_index(drop=True)
    if manifest_path.exists() and feature_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest["cohort_sha256"] == v2.sha256(cohort_path)
            and manifest["config_sha256"] == v2.sha256(CONFIG_PATH)
            and manifest["features_sha256"] == v2.sha256(feature_path)
        ):
            print("phase36 validated existing TESSERA v2 acquisition", flush=True)
            return
        raise RuntimeError("Existing Phase 36 acquisition is stale")
    points = build_points(cohort)
    transform = Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)
    longitude, latitude = transform.transform(points["x"], points["y"])
    tiles = [
        v2.tile_center(float(lon), float(lat))
        for lon, lat in zip(longitude, latitude, strict=True)
    ]
    groups = sorted(set(zip(points["year"].astype(int), tiles, strict=True)))
    preflight(groups)
    group_lookup = {group: index for index, group in enumerate(groups)}
    group_code = np.asarray(
        [group_lookup[(int(year), tile)] for year, tile in zip(points["year"], tiles, strict=True)],
        dtype=np.int16,
    )
    chunk_root = ROOT / str(settings["chunk_directory"])
    stream_root = ROOT / str(settings["stream_directory"])
    config_hash = v2.sha256(CONFIG_PATH)
    records: list[dict[str, Any]] = []
    for code, (year, tile) in enumerate(groups):
        name = f"{year}_{v2.tile_name(tile)}"
        selected_points = np.flatnonzero(group_code == code)
        chunk_path = chunk_root / f"{name}.npz"
        if chunk_path.exists():
            records.append(
                {"year": year, "tile": v2.tile_name(tile), "requested": len(selected_points), "resumed": True}
            )
            print(f"phase36 v2 [{code + 1}/{len(groups)}]: resumed {name}", flush=True)
            continue
        urls = v2.source_urls(year, tile)
        local = {kind: stream_root / name / Path(url).name for kind, url in urls.items()}
        print(
            f"phase36 v2 [{code + 1}/{len(groups)}]: {name} "
            f"for {len(selected_points):,} pixels",
            flush=True,
        )
        try:
            if shutil.disk_usage(ROOT).free < 512 * 1024 * 1024:
                raise RuntimeError("Less than 512 MiB remains before a v2 tile download")
            for kind in ["embedding", "scales"]:
                v2.download(
                    urls[kind],
                    local[kind],
                    int(settings["download_attempts"]),
                    int(settings["retry_delay_seconds"]),
                )
            quantized = np.load(local["embedding"], mmap_mode="r")
            scales = np.load(local["scales"], mmap_mode="r")
            inside, values, local_scales = v2.sample_quantized_tile(
                quantized,
                scales,
                tile,
                "EPSG:28992",
                points["x"][selected_points],
                points["y"][selected_points],
            )
            source_rows = selected_points[inside]
            save_chunk(
                chunk_path,
                points["point_index"][source_rows],
                values,
                local_scales,
                config_hash,
            )
            records.append(
                {
                    "year": year,
                    "tile": v2.tile_name(tile),
                    "requested": int(len(selected_points)),
                    "valid": int(len(source_rows)),
                    "resumed": False,
                }
            )
        finally:
            for path in local.values():
                path.unlink(missing_ok=True)
            gc.collect()
    work = ROOT / "data/interim/phase36_v2_assembly"
    work.mkdir(parents=True, exist_ok=True)
    patch_path = work / "patch.npy"
    scale_path = work / "scales.npy"
    valid_path = work / "valid.npy"
    patch = open_memmap(
        patch_path, mode="w+", dtype=np.int8, shape=(len(cohort), 128, 5, 5)
    )
    scales = open_memmap(
        scale_path, mode="w+", dtype=np.float16, shape=(len(cohort), 5, 5)
    )
    valid = open_memmap(
        valid_path, mode="w+", dtype=np.uint8, shape=(len(cohort), 5, 5)
    )
    patch[:] = 0
    scales[:] = 0
    valid[:] = 0
    for year, tile in groups:
        with np.load(chunk_root / f"{year}_{v2.tile_name(tile)}.npz") as chunk:
            point_index = chunk["point_index"].astype(np.int64)
            row_id = point_index // 25
            position = point_index % 25
            row = position // 5
            column = position % 5
            patch[row_id, :, row, column] = chunk["quantized"]
            scales[row_id, row, column] = chunk["scales"].astype(np.float16)
            valid[row_id, row, column] = 1
    patch.flush()
    scales.flush()
    valid.flush()
    summary = v2.summarize_ordered_patch(patch, scales, valid)
    count = np.asarray(valid).sum(axis=(1, 2)).astype(np.int16)
    output: dict[str, Any] = {
        "row_id": cohort["row_id"].to_numpy(dtype=np.int64),
        "tessera_valid_pixel_count": count,
        "tessera_context_valid": count >= int(settings["minimum_valid_pixels"]),
    }
    mean = summary[:, 128:256]
    for dimension in range(128):
        output[f"tessera_v2_mean_{dimension:03d}"] = mean[:, dimension]
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = feature_path.with_suffix(".tmp.parquet")
    pd.DataFrame(output).to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(feature_path)
    del summary, patch, scales, valid
    gc.collect()
    for path in [patch_path, scale_path, valid_path]:
        path.unlink(missing_ok=True)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cohort_sha256": v2.sha256(cohort_path),
        "config_sha256": v2.sha256(CONFIG_PATH),
        "features_sha256": v2.sha256(feature_path),
        "rows": int(len(cohort)),
        "valid_context_rows": int((count >= int(settings["minimum_valid_pixels"])).sum()),
        "year_tile_groups": records,
    }
    atomic_json(manifest_path, manifest)
    print(
        f"phase36 v2 complete: {manifest['valid_context_rows']:,}/{len(cohort):,} contexts valid",
        flush=True,
    )


if __name__ == "__main__":
    run()
