#!/usr/bin/env python3
"""Validate the frozen Phase 2 GEDI-to-TESSERA alignment independently."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FREEZE_PATH = ROOT / "metadata/tessera_phase2_alignment_freeze.json"
TILE_MANIFEST_PATH = ROOT / "metadata/tessera_v1_2024_tile_manifest.json"
CONFIG_SNAPSHOT_PATH = ROOT / "metadata/project_config_phase2_tessera_alignment_freeze.yaml"
BUILD_SCRIPT_PATH = ROOT / "scripts/build_tessera_alignment.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_output_file(
    output: dict[str, object],
    expected_sites: set[str],
    freeze: dict[str, object],
) -> pd.DataFrame:
    path = ROOT / str(output["path"])
    require(path.is_file(), f"Missing frozen output: {path}")
    require(sha256(path) == output["sha256"], f"Checksum mismatch: {path}")

    frame = pd.read_parquet(path)
    require(len(frame) == output["rows"], f"Row count mismatch: {path}")
    require(set(frame["site_id"].unique()) == expected_sites, f"Wrong sites: {path}")
    require(
        not frame.duplicated(["site_id", "shot_number"]).any(),
        f"Duplicate site/shot rows: {path}",
    )
    require(frame["passes_tessera_alignment"].all(), f"Failed alignments: {path}")

    basis = freeze["freeze_basis"]
    expected_constants = {
        "tessera_alignment_id": freeze["alignment_id"],
        "freeze_id": basis["gedi_freeze_id"],
        "tessera_dataset_version": basis["dataset_version"],
        "tessera_dataset_variant": basis["dataset_variant"],
        "tessera_embedding_year": basis["embedding_year"],
    }
    for column, expected in expected_constants.items():
        require(column in frame, f"Missing column {column}: {path}")
        values = set(frame[column].unique())
        require(values == {expected}, f"Unexpected {column} values in {path}: {values}")

    require(frame["tessera_center_valid"].all(), f"Invalid centre pixels: {path}")
    threshold = basis["alignment"]["minimum_valid_footprint_fraction"]
    require(
        (frame["tessera_valid_footprint_fraction"] >= threshold).all(),
        f"Footprint coverage below {threshold}: {path}",
    )

    for prefix in ("tessera_center_", "tessera_area_", "tessera_gaussian_"):
        columns = [f"{prefix}{index:03d}" for index in range(128)]
        require(all(column in frame for column in columns), f"Incomplete {prefix} vector")
        values = frame[columns].to_numpy(dtype=np.float32, copy=False)
        require(np.isfinite(values).all(), f"Non-finite values in {prefix}: {path}")
        require(np.any(values != 0.0), f"All-zero values in {prefix}: {path}")

    return frame


def validate_source_tiles(freeze: dict[str, object]) -> None:
    manifest = json.loads(TILE_MANIFEST_PATH.read_text())
    basis = freeze["freeze_basis"]
    require(manifest["tile_count"] == 8, "Expected exactly eight source tiles")
    require(manifest["dataset_version"] == basis["dataset_version"], "Tile version mismatch")
    require(manifest["dataset_variant"] == basis["dataset_variant"], "Tile variant mismatch")
    require(manifest["year"] == basis["embedding_year"], "Tile year mismatch")

    manifest_tiles = {item["tile"]: item for item in manifest["tiles"]}
    frozen_tiles = {item["tile"]: item for item in basis["tiles"]}
    require(manifest_tiles.keys() == frozen_tiles.keys(), "Frozen tile set mismatch")

    for tile_name, tile in manifest_tiles.items():
        require(tile["embedding_shape"][2] == 128, f"Wrong dimension: {tile_name}")
        require(tile["source_integrity"] == "S3_CRC64NVME_validated_by_botocore", f"Unverified source: {tile_name}")
        for kind, file_record in tile["local_files"].items():
            path = ROOT / file_record["path"]
            require(path.is_file(), f"Missing {kind} file: {path}")
            require(path.stat().st_size == file_record["size_bytes"], f"Size mismatch: {path}")
            expected = frozen_tiles[tile_name]["local_files"][kind]
            require(file_record["sha256"] == expected, f"Manifest hash mismatch: {path}")
            require(sha256(path) == expected, f"Local hash mismatch: {path}")


def main() -> None:
    freeze = json.loads(FREEZE_PATH.read_text())
    require(freeze["status"] == "frozen", "Alignment manifest is not frozen")
    require(
        freeze["alignment_id"] == f"tessera-align-{freeze['freeze_basis_sha256'][:12]}",
        "Alignment ID does not match freeze basis",
    )
    basis = freeze["freeze_basis"]
    require(
        sha256(CONFIG_SNAPSHOT_PATH) == basis["project_config_sha256"],
        "Phase 2 config snapshot changed",
    )
    require(sha256(BUILD_SCRIPT_PATH) == basis["alignment_script_sha256"], "Build script changed")
    require((basis["dataset_version"], basis["dataset_variant"]) == ("1.0", "vultr"), "Wrong release")

    for output_name in ("missingness", "tile_usage"):
        output = freeze["outputs"][output_name]
        path = ROOT / output["path"]
        require(path.is_file(), f"Missing frozen metadata: {path}")
        require(sha256(path) == output["sha256"], f"Checksum mismatch: {path}")

    development = validate_output_file(
        freeze["outputs"]["development"], {"SOAP", "TEAK"}, freeze
    )
    locked = validate_output_file(
        freeze["outputs"]["locked_bart"], {"BART"}, freeze
    )
    require(
        set(development["shot_number"]).isdisjoint(set(locked["shot_number"])),
        "Development and locked partitions overlap",
    )
    validate_source_tiles(freeze)

    print(
        json.dumps(
            {
                "status": "valid",
                "alignment_id": freeze["alignment_id"],
                "release": "1.0/vultr/2024",
                "source_tiles": len(basis["tiles"]),
                "development_rows": len(development),
                "locked_bart_rows": len(locked),
                "failed_alignment_rows": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
