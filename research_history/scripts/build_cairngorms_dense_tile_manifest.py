#!/usr/bin/env python3
"""Extend the sampled Cairngorms tile manifest for the complete dense grid."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "metadata/phase18_cairngorms_patch_manifest.json"
REGISTRY = ROOT / "data/external/geotessera_cache/v1/manifest.parquet"
LANDMASKS = ROOT / "data/external/geotessera_cache/v1/landmasks.parquet"
OUTPUT = ROOT / "metadata/cairngorms_dense_10m_tessera_tile_manifest.json"
REQUIRED_ADDITIONS = [(-4.05, 56.95), (-3.85, 56.95)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tile_name(lon: float, lat: float) -> str:
    return f"grid_{lon:.2f}_{lat:.2f}"


def main() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    registry = pd.read_parquet(REGISTRY)
    landmasks = pd.read_parquet(LANDMASKS)
    records = {str(record["tile"]): record for record in source["tile_records"]}
    additions: list[str] = []
    for lon, lat in REQUIRED_ADDITIONS:
        name = tile_name(lon, lat)
        if name in records:
            continue
        match = registry[
            (registry["version"].astype(str) == str(source["dataset_version"]))
            & (registry["variant"].astype(str) == str(source["dataset_variant"]))
            & (registry["year"] == int(source["embedding_year"]))
            & (registry["lon_i"] == round(lon * 100))
            & (registry["lat_i"] == round(lat * 100))
        ]
        mask = landmasks[
            (landmasks["lon_i"] == round(lon * 100))
            & (landmasks["lat_i"] == round(lat * 100))
        ]
        if len(match) != 1 or len(mask) != 1:
            raise RuntimeError(f"Registry lookup was not unique for {name}")
        row = match.iloc[0]
        mask_row = mask.iloc[0]
        records[name] = {
            "created_utc": datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
            "tile": name,
            "source_points": 0,
            "valid_points": 0,
            "embedding_source": str(row["grid_path"]),
            "scales_source": str(row["scales_path"]),
            "landmask_source": str(mask_row["key"]),
            "source_sizes": {
                "embedding": int(row["grid_size"]),
                "scales": int(row["scales_size"]),
                "landmask": int(mask_row["file_size"]),
            },
        }
        additions.append(name)
    output = {
        **{key: value for key, value in source.items() if key != "tile_records"},
        "created_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "source_manifest": str(SOURCE.relative_to(ROOT)),
        "source_manifest_sha256": sha256(SOURCE),
        "registry_sha256": sha256(REGISTRY),
        "landmask_registry_sha256": sha256(LANDMASKS),
        "dense_grid_additions": additions,
        "tile_count": len(records),
        "tile_records": [records[name] for name in sorted(records)],
    }
    temporary = OUTPUT.with_suffix(".tmp.json")
    temporary.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(OUTPUT)
    print(f"wrote {OUTPUT} with {len(records)} tiles; additions={additions}")


if __name__ == "__main__":
    main()
