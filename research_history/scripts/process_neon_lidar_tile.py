#!/usr/bin/env python3
"""Process one frozen NEON LAZ tile into GEDI-footprint simulator metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import laspy
import numpy as np
import pandas as pd
import requests
from pyproj import Transformer
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "metadata/phase5_neon_lidar_manifest.csv"
GEDI_PATH = ROOT / "data/processed/gedi_v3_2024_primary_forest.parquet"
SIMULATOR_FREEZE_PATH = ROOT / "metadata/phase5_gedi_simulator_smoke_freeze.json"
GEDIRAT = ROOT / "data/external/gedisimulator/gediRat"
GEDIMETRIC = ROOT / "data/external/gedisimulator/gediMetric"
PULSE_ROOT = ROOT / "data/external/gedisimulator/pulse_shapes"
WORK_ROOT = ROOT / "data/interim/neon_lidar_work"
OUTPUT_ROOT = ROOT / "data/processed/neon_lidar_tile_metrics"
FREEZE_ROOT = ROOT / "metadata/phase5_neon_lidar_tiles"
SITE_UTM = {"SOAP": 32611, "TEAK": 32611, "BART": 32619}
EXTRACTION_RADIUS_M = 20.0
EDGE_BUFFER_M = 20.0
CHUNK_POINTS = 1_000_000
DISK_RESERVE_BYTES = 2_000_000_000
WORKING_MARGIN_BYTES = 500_000_000


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


def token_from_environment() -> str | None:
    return os.getenv("NEON_TOKEN") or os.getenv("NEON_PAT")


def metric_column(metric_path: Path, label: str) -> float:
    lines = metric_path.read_text(encoding="utf-8").splitlines()
    header = next(line for line in lines if line.startswith("#"))
    match = re.search(rf"(?:^|, )(\d+) {re.escape(label)}(?:,|$)", header)
    if match is None:
        raise RuntimeError(f"gediMetric output does not identify {label}")
    values = next(line for line in lines if line and not line.startswith("#")).split()
    value = float(values[int(match.group(1)) - 1])
    if not math.isfinite(value):
        raise RuntimeError(f"Non-finite gediMetric {label}: {metric_path}")
    return value


def selected_gedi_rows(gedi: pd.DataFrame, site: str, easting: int, northing: int) -> pd.DataFrame:
    subset = gedi[gedi["site_id"].eq(site)].copy()
    transformer = Transformer.from_crs(4326, SITE_UTM[site], always_xy=True)
    x, y = transformer.transform(subset["longitude"].to_numpy(), subset["latitude"].to_numpy())
    subset["utm_x"] = x
    subset["utm_y"] = y
    in_tile = (
        subset["utm_x"].between(easting + EDGE_BUFFER_M, easting + 1000 - EDGE_BUFFER_M)
        & subset["utm_y"].between(northing + EDGE_BUFFER_M, northing + 1000 - EDGE_BUFFER_M)
    )
    return subset.loc[in_tile].sort_values("shot_number").reset_index(drop=True)


def download_source(session: requests.Session, token: str, url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with session.get(url, headers={"X-API-TOKEN": token}, stream=True, timeout=300) as response:
        if response.status_code == 403:
            raise RuntimeError("NEON rejected the API token or the token has expired")
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                if chunk:
                    handle.write(chunk)
    temporary.replace(path)


def extract_footprints(source_path: Path, footprints: pd.DataFrame, subset_root: Path) -> pd.DataFrame:
    subset_root.mkdir(parents=True, exist_ok=True)
    centers = footprints[["utm_x", "utm_y"]].to_numpy(dtype=np.float64)
    point_counts = np.zeros(len(footprints), dtype=np.int64)
    ground_counts = np.zeros(len(footprints), dtype=np.int64)
    paths = [subset_root / f"shot_{int(shot)}.las" for shot in footprints["shot_number"]]
    with laspy.open(source_path) as reader, ExitStack() as stack:
        writers = [
            stack.enter_context(laspy.open(path, mode="w", header=reader.header, do_compress=False))
            for path in paths
        ]
        for points in reader.chunk_iterator(CHUNK_POINTS):
            xy = np.column_stack((np.asarray(points.x), np.asarray(points.y)))
            tree = cKDTree(xy)
            memberships = tree.query_ball_point(centers, r=EXTRACTION_RADIUS_M)
            classifications = np.asarray(points.classification)
            for index, member_indices in enumerate(memberships):
                if not member_indices:
                    continue
                selected = np.asarray(member_indices, dtype=np.int64)
                writers[index].write_points(points[selected])
                point_counts[index] += len(selected)
                ground_counts[index] += int((classifications[selected] == 2).sum())
    result = footprints.copy()
    result["subset_las_path"] = [str(path) for path in paths]
    result["als_point_count_20m"] = point_counts
    result["als_ground_point_count_20m"] = ground_counts
    if (point_counts == 0).any() or (ground_counts == 0).any():
        failed = result.loc[(point_counts == 0) | (ground_counts == 0), "shot_number"].tolist()
        raise RuntimeError(f"Missing ALS or ground returns for shots: {failed}")
    return result


def simulate_footprint(row: Any, work_root: Path) -> dict[str, Any]:
    shot_number = int(row.shot_number)
    wave_path = work_root / f"shot_{shot_number}.wave"
    metric_root = work_root / f"shot_{shot_number}"
    metric_path = work_root / f"shot_{shot_number}.metric.txt"
    pulse_path = PULSE_ROOT / f"meanPulse.{row.beam_name}.txt"
    if not pulse_path.is_file():
        raise RuntimeError(f"Missing beam pulse: {pulse_path}")
    subprocess.run([
        str(GEDIRAT), "-input", str(row.subset_las_path),
        "-coord", f"{row.utm_x:.3f}", f"{row.utm_y:.3f}",
        "-output", str(wave_path), "-ground", "-checkCover",
        "-readPulse", str(pulse_path), "-pBuff", "0.1",
    ], check=True, capture_output=True, text=True)
    subprocess.run([
        str(GEDIMETRIC), "-input", str(wave_path), "-outRoot", str(metric_root),
        "-ground", "-fhdHistRes", "1", "-laiRes", "5",
    ], check=True, capture_output=True, text=True)
    return {
        "simulated_fhd": metric_column(metric_path, "FHD"),
        "simulated_true_ground_m": metric_column(metric_path, "true ground"),
        "simulated_true_top_m": metric_column(metric_path, "true top"),
        "simulated_als_cover": metric_column(metric_path, "ALS cover"),
        "simulated_point_density": metric_column(metric_path, "pointDense"),
        "simulated_beam_density": metric_column(metric_path, "beamDense"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", required=True, choices=sorted(SITE_UTM))
    parser.add_argument("--selection-order", required=True, type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = token_from_environment()
    if not token:
        raise RuntimeError("NEON_TOKEN is required")
    manifest = pd.read_csv(MANIFEST_PATH)
    selected = manifest[
        manifest["site_id"].eq(args.site)
        & manifest["selection_order"].eq(args.selection_order)
    ]
    if len(selected) != 1:
        raise RuntimeError("Requested lidar tile is absent or non-unique in the frozen manifest")
    tile = selected.iloc[0]
    tile_tag = f"{args.site}_{args.selection_order:02d}_{int(tile.tile_easting)}_{int(tile.tile_northing)}"
    output_path = OUTPUT_ROOT / f"{tile_tag}.parquet"
    freeze_path = FREEZE_ROOT / f"{tile_tag}.json"
    if output_path.exists() or freeze_path.exists():
        raise RuntimeError(f"Tile output exists; refusing to overwrite: {tile_tag}")
    free_bytes = shutil.disk_usage(ROOT).free
    required_free_bytes = int(tile.size_bytes) + WORKING_MARGIN_BYTES + DISK_RESERVE_BYTES
    if free_bytes < required_free_bytes:
        raise RuntimeError(
            f"Tile run needs {required_free_bytes / 1e9:.2f} GB free including reserve; "
            f"found {free_bytes / 1e9:.2f} GB"
        )

    gedi = pd.read_parquet(GEDI_PATH)
    footprints = selected_gedi_rows(
        gedi, args.site, int(tile.tile_easting), int(tile.tile_northing)
    )
    if len(footprints) != int(tile.eligible_gedi_rows):
        raise RuntimeError(
            f"GEDI tile population changed: expected {int(tile.eligible_gedi_rows)}, found {len(footprints)}"
        )
    work_root = WORK_ROOT / tile_tag
    source_path = work_root / Path(str(tile.file_name)).name
    source_sha256 = None
    try:
        session = requests.Session()
        session.headers.update({"User-Agent": "tessera-gedi-fhd/0.1"})
        download_source(session, token, str(tile.url), source_path)
        if source_path.stat().st_size != int(tile.size_bytes):
            raise RuntimeError("Downloaded NEON LAZ does not match the published size")
        source_sha256 = sha256(source_path)
        extracted = extract_footprints(source_path, footprints, work_root / "footprints")
        simulator_records = []
        for index, row in enumerate(extracted.itertuples(index=False), start=1):
            simulator_records.append(simulate_footprint(row, work_root))
            print(f"simulated {args.site} tile {args.selection_order}: {index}/{len(extracted)}", flush=True)
        simulation = pd.DataFrame(simulator_records)
        result = pd.concat([extracted.drop(columns=["subset_las_path"]), simulation], axis=1)
        result["neon_tile_id"] = tile_tag
        result["neon_tile_selection_order"] = args.selection_order
        result["neon_tile_easting"] = int(tile.tile_easting)
        result["neon_tile_northing"] = int(tile.tile_northing)
        result["simulated_canopy_height_m"] = (
            result["simulated_true_top_m"] - result["simulated_true_ground_m"]
        )
        result["simulated_minus_gedi_fhd"] = result["simulated_fhd"] - result["fhd_normal"]
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(".tmp.parquet")
        result.to_parquet(temporary, index=False)
        temporary.replace(output_path)

        simulator_freeze = json.loads(SIMULATOR_FREEZE_PATH.read_text(encoding="utf-8"))
        freeze_basis = {
            "source_manifest_path": str(MANIFEST_PATH.relative_to(ROOT)),
            "source_manifest_sha256": sha256(MANIFEST_PATH),
            "gedi_input_path": str(GEDI_PATH.relative_to(ROOT)),
            "gedi_input_sha256": sha256(GEDI_PATH),
            "site_id": args.site,
            "selection_order": args.selection_order,
            "tile_easting": int(tile.tile_easting),
            "tile_northing": int(tile.tile_northing),
            "source_file_name": str(tile.file_name),
            "source_size_bytes": int(tile.size_bytes),
            "source_laz_sha256": source_sha256,
            "source_laz_retained": False,
            "extraction_radius_m": EXTRACTION_RADIUS_M,
            "edge_buffer_m": EDGE_BUFFER_M,
            "chunk_points": CHUNK_POINTS,
            "simulator_smoke_freeze_id": simulator_freeze["freeze_id"],
            "simulator_binary_sha256": simulator_freeze["freeze_basis"]["binary_sha256"],
            "beam_specific_pulse_shape": True,
            "metric_primary": "gediMetric_FHD",
            "metric_equivalence_gate": "must_compare_with_collocated_GEDI_fhd_normal_before_external_claim",
            "token_value_recorded": False,
            "script_sha256": sha256(Path(__file__)),
        }
        freeze_id = "phase5-neon-lidar-tile-" + canonical_hash(freeze_basis)[:12]
        freeze = {
            "freeze_id": freeze_id,
            "created_at": utc_now(),
            "status": "frozen_collocated_neon_lidar_simulation_pending_cross_sensor_gate",
            "freeze_basis": freeze_basis,
            "freeze_basis_sha256": canonical_hash(freeze_basis),
            "outputs": {
                "metrics": {
                    "path": str(output_path.relative_to(ROOT)),
                    "rows": len(result),
                    "sha256": sha256(output_path),
                }
            },
        }
        write_json(freeze_path, freeze)
        print(f"Frozen {freeze_id}: {len(result)} collocated footprints")
    finally:
        if work_root.exists():
            shutil.rmtree(work_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
