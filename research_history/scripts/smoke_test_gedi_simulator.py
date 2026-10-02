#!/usr/bin/env python3
"""Freeze a deterministic gediRat/gediMetric synthetic-forest smoke test."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import laspy
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SIMULATOR_ROOT = ROOT / "data/external/gedisimulator"
LIBCLIDAR_ROOT = ROOT / "data/external/libclidar"
TOOLS_ROOT = ROOT / "data/external/hancock-tools"
GEDIRAT = SIMULATOR_ROOT / "gediRat"
GEDIMETRIC = SIMULATOR_ROOT / "gediMetric"
WORK_ROOT = ROOT / "data/interim/phase5_gedi_simulator_smoke"
TABLE_PATH = ROOT / "outputs/tables/phase5_gedi_simulator_smoke.csv"
FREEZE_PATH = ROOT / "metadata/phase5_gedi_simulator_smoke_freeze.json"
EXPECTED_SIMULATOR_COMMIT = "49ad4f26b03083b61ee0bd424e8a7e4a8f2c301d"
RANDOM_SEED = 20260717


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


def git_commit(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def make_forest_las(path: Path, multilayer: bool) -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    ground_count = 4_000
    canopy_count = 12_000
    point_count = ground_count + canopy_count
    radius = 18 * np.sqrt(rng.random(point_count))
    angle = 2 * np.pi * rng.random(point_count)
    x = 500_000 + radius * np.cos(angle)
    y = 4_100_000 + radius * np.sin(angle)
    ground = 100 + 0.015 * (x[:ground_count] - 500_000) + rng.normal(0, 0.08, ground_count)
    if multilayer:
        levels = rng.choice([4.0, 12.0, 24.0], size=canopy_count, p=[0.25, 0.35, 0.40])
    else:
        levels = np.full(canopy_count, 12.0)
    canopy = (
        100
        + 0.015 * (x[ground_count:] - 500_000)
        + levels
        + rng.normal(0, 0.8, canopy_count)
    )
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.01, 0.01, 0.01])
    header.offsets = np.array([500_000, 4_100_000, 0])
    las = laspy.LasData(header)
    las.x = x
    las.y = y
    las.z = np.concatenate([ground, canopy])
    las.classification = np.concatenate([
        np.full(ground_count, 2, dtype=np.uint8),
        np.ones(canopy_count, dtype=np.uint8),
    ])
    las.intensity = np.full(point_count, 100, dtype=np.uint16)
    las.return_number = np.ones(point_count, dtype=np.uint8)
    las.number_of_returns = np.ones(point_count, dtype=np.uint8)
    las.write(path)


def parse_fhd(metric_path: Path) -> float:
    lines = metric_path.read_text(encoding="utf-8").splitlines()
    header = next(line for line in lines if line.startswith("#"))
    match = re.search(r"(?:^|, )(\d+) FHD(?:,|$)", header)
    if match is None:
        raise RuntimeError("gediMetric output does not identify the FHD column")
    values = next(line for line in lines if line and not line.startswith("#")).split()
    return float(values[int(match.group(1)) - 1])


def simulate(name: str, multilayer: bool) -> dict[str, Any]:
    las_path = WORK_ROOT / f"{name}.las"
    wave_path = WORK_ROOT / f"{name}.wave"
    metric_root = WORK_ROOT / name
    metric_path = WORK_ROOT / f"{name}.metric.txt"
    make_forest_las(las_path, multilayer)
    subprocess.run([
        str(GEDIRAT), "-input", str(las_path), "-coord", "500000", "4100000",
        "-output", str(wave_path), "-ground", "-pBuff", "0.1",
    ], check=True)
    subprocess.run([
        str(GEDIMETRIC), "-input", str(wave_path), "-outRoot", str(metric_root),
        "-ground", "-fhdHistRes", "1", "-laiRes", "5",
    ], check=True)
    fhd = parse_fhd(metric_path)
    if not math.isfinite(fhd):
        raise RuntimeError(f"Non-finite synthetic FHD for {name}")
    return {
        "scenario": name,
        "multilayer": multilayer,
        "fhd": fhd,
        "las_sha256": sha256(las_path),
        "wave_sha256": sha256(wave_path),
        "metric_sha256": sha256(metric_path),
    }


def main() -> int:
    if FREEZE_PATH.exists():
        raise RuntimeError(f"Smoke-test freeze exists; refusing to overwrite: {FREEZE_PATH.relative_to(ROOT)}")
    simulator_commit = git_commit(SIMULATOR_ROOT)
    if simulator_commit != EXPECTED_SIMULATOR_COMMIT:
        raise RuntimeError(f"Unexpected GEDI simulator commit: {simulator_commit}")
    for binary in [GEDIRAT, GEDIMETRIC]:
        if not binary.is_file():
            raise RuntimeError(f"Missing simulator binary: {binary}")
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    records = [
        simulate("single_layer", multilayer=False),
        simulate("multi_layer", multilayer=True),
    ]
    table = pd.DataFrame(records)
    single_fhd = float(table.loc[~table["multilayer"], "fhd"].iloc[0])
    multi_fhd = float(table.loc[table["multilayer"], "fhd"].iloc[0])
    if multi_fhd <= single_fhd:
        raise RuntimeError("Synthetic multilayer canopy did not produce higher FHD")
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLE_PATH, index=False)

    freeze_basis = {
        "random_seed": RANDOM_SEED,
        "expected_simulator_commit": EXPECTED_SIMULATOR_COMMIT,
        "source_commits": {
            "gedisimulator": simulator_commit,
            "libclidar": git_commit(LIBCLIDAR_ROOT),
            "hancock_tools": git_commit(TOOLS_ROOT),
        },
        "binary_sha256": {"gediRat": sha256(GEDIRAT), "gediMetric": sha256(GEDIMETRIC)},
        "gate": "both_fhd_finite_and_multilayer_fhd_greater_than_single_layer_fhd",
        "not_validated_by_this_test": "numerical_equivalence_to_GEDI_L2B_fhd_normal",
        "script_sha256": sha256(Path(__file__)),
    }
    freeze_id = "phase5-gedi-simulator-smoke-" + canonical_hash(freeze_basis)[:12]
    freeze = {
        "freeze_id": freeze_id,
        "created_at": utc_now(),
        "status": "passed_synthetic_executable_and_directional_fhd_gate",
        "freeze_basis": freeze_basis,
        "freeze_basis_sha256": canonical_hash(freeze_basis),
        "results": {"single_layer_fhd": single_fhd, "multi_layer_fhd": multi_fhd},
        "outputs": {
            "table": {
                "path": str(TABLE_PATH.relative_to(ROOT)),
                "rows": len(table),
                "sha256": sha256(TABLE_PATH),
            }
        },
    }
    write_json(FREEZE_PATH, freeze)
    print(f"Frozen {freeze_id}: single={single_fhd:.6f}, multilayer={multi_fhd:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
