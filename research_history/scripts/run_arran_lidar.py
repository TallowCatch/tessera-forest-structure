#!/usr/bin/env python3
"""Run the frozen, resumable Arran airborne-LiDAR replication and transfer."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import yaml
from pyproj import Transformer
from rasterio.transform import rowcol
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr
from shapely.geometry import mapping, shape
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_conventional_predictors as conventional  # noqa: E402
import build_tessera_alignment as tessera  # noqa: E402
import arran_lidar_core as core  # noqa: E402
import run_scotland_lidar as phase14  # noqa: E402


CONFIG_PATH = ROOT / "configs/arran_lidar.yaml"
PROTOCOL_PATH = (
    ROOT / "metadata/project_config_phase15_arran_lidar_protocol_freeze.yaml"
)
BOUNDARY_PATH = ROOT / "data/raw/phase15_arran/arran_boundary.geojson"
BOUNDARY_FREEZE_PATH = ROOT / "metadata/phase15_arran_boundary_freeze.json"
INVENTORY_PATH = ROOT / "metadata/phase15_arran_lidar_inventory.json"
METRIC_TILE_DIR = ROOT / "data/interim/phase15_arran_lidar_metric_tiles"
METRIC_TILE_METADATA_DIR = ROOT / "metadata/phase15_arran_lidar_tiles"
SOURCE_STREAM_DIR = ROOT / "data/interim/phase15_arran_lidar_stream"
LIDAR_STATUS_PATH = ROOT / "outputs/reports/phase15_arran_lidar_status.json"
SAMPLE_PATH = ROOT / "data/processed/phase15_arran_lidar_sample.parquet"
SAMPLE_QC_PATH = ROOT / "metadata/phase15_arran_lidar_sample_qc.json"
TESSERA_CHUNK_DIR = ROOT / "data/interim/phase15_arran_tessera_chunks"
TESSERA_STREAM_DIR = ROOT / "data/interim/phase15_arran_tessera_stream"
TESSERA_PATH = ROOT / "data/processed/phase15_arran_tessera.parquet"
TESSERA_MANIFEST_PATH = ROOT / "metadata/phase15_arran_tessera_manifest.json"
ARRAN_CONVENTIONAL_PATH = (
    ROOT / "data/processed/phase15_arran_conventional_predictors.parquet"
)
ARRAN_CONVENTIONAL_MANIFEST_PATH = (
    ROOT / "metadata/phase15_arran_conventional_freeze.json"
)
ARRAN_CONVENTIONAL_INVENTORY_PATH = (
    ROOT / "metadata/phase15_arran_conventional_inventory.csv"
)
CAIRNGORMS_CONVENTIONAL_PATH = (
    ROOT / "data/processed/phase15_cairngorms_conventional_predictors.parquet"
)
CAIRNGORMS_CONVENTIONAL_MANIFEST_PATH = (
    ROOT / "metadata/phase15_cairngorms_conventional_freeze.json"
)
CAIRNGORMS_CONVENTIONAL_INVENTORY_PATH = (
    ROOT / "metadata/phase15_cairngorms_conventional_inventory.csv"
)
SPATIAL_METRICS_PATH = (
    ROOT / "outputs/tables/phase15_arran_spatial_metrics.csv"
)
SPATIAL_PREDICTIONS_PATH = (
    ROOT / "data/processed/phase15_arran_primary_oof_predictions.parquet"
)
TRANSFER_METRICS_PATH = (
    ROOT / "outputs/tables/phase15_cairngorms_to_arran_metrics.csv"
)
TRANSFER_PREDICTIONS_PATH = (
    ROOT / "data/processed/phase15_cairngorms_to_arran_predictions.parquet"
)
FIGURE_PATH = ROOT / "outputs/figures/phase15_arran_replication_transfer.png"
REPORT_PATH = ROOT / "outputs/reports/phase15_arran_results.md"
RESULT_FREEZE_PATH = ROOT / "metadata/phase15_arran_result_freeze.json"
RUN_STATUS_PATH = ROOT / "outputs/reports/phase15_arran_pipeline_status.json"

TESSERA_FEATURES = [f"tessera_{index:03d}" for index in range(128)]
STAGES = [
    "boundary",
    "inventory",
    "lidar_metrics",
    "sample",
    "tessera",
    "conventional",
    "evaluate",
    "transfer",
    "report",
]


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def load_config() -> dict[str, Any]:
    if sha256(CONFIG_PATH) != sha256(PROTOCOL_PATH):
        raise RuntimeError(
            "The active Arran config differs from the frozen protocol copy"
        )
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "phase15_arran_lidar"
    ]


def update_run_status(
    stage: str,
    state: str,
    *,
    detail: dict[str, Any] | None = None,
) -> None:
    existing: dict[str, Any] = {}
    if RUN_STATUS_PATH.exists():
        existing = json.loads(RUN_STATUS_PATH.read_text(encoding="utf-8"))
    existing.update(
        {
            "updated_utc": utc_now(),
            "current_stage": stage,
            "state": state,
            "stages": STAGES,
        }
    )
    if detail is not None:
        existing["detail"] = detail
    core.atomic_json(RUN_STATUS_PATH, existing)


def run_boundary(config: dict[str, Any]) -> None:
    expected = config["study"]
    if BOUNDARY_PATH.exists() and BOUNDARY_FREEZE_PATH.exists():
        freeze = json.loads(BOUNDARY_FREEZE_PATH.read_text(encoding="utf-8"))
        if (
            freeze["boundary_sha256"] == sha256(BOUNDARY_PATH)
            and freeze["config_sha256"] == sha256(CONFIG_PATH)
        ):
            print("boundary: validated frozen Arran boundary", flush=True)
            return
        raise RuntimeError("The frozen Arran boundary or config changed")

    response = core.request_json(
        expected["boundary_source_url"],
        user_agent=config["source"]["user_agent"],
    )
    features = response.get("features", [])
    matching = [
        feature
        for feature in features
        if feature.get("properties", {}).get("osm_type")
        == expected["expected_osm_type"]
        and int(feature.get("properties", {}).get("osm_id", -1))
        == int(expected["expected_osm_id"])
    ]
    if len(matching) != 1:
        raise RuntimeError("The expected Arran OSM island relation was not returned")
    feature = matching[0]
    geometry = shape(feature["geometry"])
    if geometry.geom_type not in {"Polygon", "MultiPolygon"} or not geometry.is_valid:
        raise RuntimeError("The Arran boundary geometry is invalid")
    bounds = np.asarray(geometry.bounds, dtype=np.float64)
    if not np.allclose(
        bounds,
        np.asarray(expected["expected_boundary_bounds_wgs84"], dtype=np.float64),
        atol=0.001,
    ):
        raise RuntimeError(f"Unexpected Arran boundary bounds: {bounds.tolist()}")
    frozen_feature = {
        "type": "Feature",
        "properties": {
            "name": "Arran",
            "osm_type": expected["expected_osm_type"],
            "osm_id": int(expected["expected_osm_id"]),
            "source_url": expected["boundary_source_url"],
        },
        "geometry": mapping(geometry),
    }
    BOUNDARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = BOUNDARY_PATH.with_suffix(".tmp.geojson")
    temporary.write_text(
        json.dumps(frozen_feature, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(BOUNDARY_PATH)
    core.atomic_json(
        BOUNDARY_FREEZE_PATH,
        {
            "created_utc": utc_now(),
            "name": "Arran",
            "osm_type": expected["expected_osm_type"],
            "osm_id": int(expected["expected_osm_id"]),
            "bounds_wgs84": list(geometry.bounds),
            "area_square_degrees": geometry.area,
            "boundary_path": str(BOUNDARY_PATH.relative_to(ROOT)),
            "boundary_sha256": sha256(BOUNDARY_PATH),
            "config_sha256": sha256(CONFIG_PATH),
            "protocol_sha256": sha256(PROTOCOL_PATH),
        },
    )
    print(f"boundary: froze Arran relation {expected['expected_osm_id']}", flush=True)


def catalogue_products(
    config: dict[str, Any],
    collection: str,
    footprint_wkt: str,
) -> list[dict[str, Any]]:
    endpoint = f"{config['source']['catalogue_api']}/search/product"
    limit = 200
    offset = 0
    products: dict[str, dict[str, Any]] = {}
    while True:
        response = core.request_json(
            endpoint,
            payload={
                "collections": [collection],
                "footprint": footprint_wkt,
                "spatialop": "intersects",
                "limit": limit,
                "offset": offset,
            },
            user_agent=config["source"]["user_agent"],
            attempts=int(config["source"]["download_attempts"]),
            retry_delay_seconds=int(config["source"]["retry_delay_seconds"]),
        )
        result = response.get("result", [])
        for product in result:
            products[str(product["id"])] = product
        print(
            f"inventory: {collection} offset {offset}, received {len(result)}",
            flush=True,
        )
        if len(result) < limit:
            break
        offset += limit
    return sorted(products.values(), key=lambda item: item["name"])


def _product_record(product: dict[str, Any]) -> dict[str, Any]:
    http = product["data"]["product"]["http"]
    metadata = product["metadata"]
    return {
        "id": str(product["id"]),
        "name": str(product["name"]),
        "title": str(metadata["title"]),
        "url": str(http["url"]),
        "size_bytes": int(http["size"]),
        "grid_reference": str(product["properties"]["osgbGridRef"]).upper(),
        "bounds_wgs84": metadata["boundingBox"],
        "temporal_extent": metadata["temporalExtent"],
        "spatial_reference_system": str(metadata["spatialReferenceSystem"]),
        "use_constraints": str(metadata["useConstraints"]),
    }


def run_inventory(config: dict[str, Any]) -> None:
    if INVENTORY_PATH.exists():
        inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
        if (
            inventory["boundary_sha256"] == sha256(BOUNDARY_PATH)
            and inventory["config_sha256"] == sha256(CONFIG_PATH)
            and inventory["protocol_sha256"] == sha256(PROTOCOL_PATH)
        ):
            print(
                f"inventory: validated {inventory['tile_count']} paired tiles",
                flush=True,
            )
            return
        raise RuntimeError("The Arran inventory does not match the frozen inputs")

    feature = json.loads(BOUNDARY_PATH.read_text(encoding="utf-8"))
    geometry = shape(feature["geometry"])
    laz_products = catalogue_products(
        config,
        config["source"]["laz_collection"],
        geometry.wkt,
    )
    dtm_products = catalogue_products(
        config,
        config["source"]["dtm_collection"],
        geometry.wkt,
    )
    laz_by_grid = {
        record["grid_reference"]: record
        for record in map(_product_record, laz_products)
    }
    dtm_by_grid = {
        record["grid_reference"]: record
        for record in map(_product_record, dtm_products)
    }
    if set(laz_by_grid) != set(dtm_by_grid):
        missing_dtm = sorted(set(laz_by_grid) - set(dtm_by_grid))
        missing_laz = sorted(set(dtm_by_grid) - set(laz_by_grid))
        raise RuntimeError(
            f"Unpaired Arran source tiles; missing DTM={missing_dtm}, "
            f"missing LAZ={missing_laz}"
        )
    survey_year = int(config["source"]["survey_year"])
    tiles = []
    for grid_reference in sorted(laz_by_grid):
        laz = laz_by_grid[grid_reference]
        dtm = dtm_by_grid[grid_reference]
        for record in [laz, dtm]:
            if str(survey_year) not in str(record["title"]):
                raise RuntimeError(
                    f"{grid_reference}: source title is not from {survey_year}"
                )
            if record["spatial_reference_system"] != "EPSG:27700":
                raise RuntimeError(
                    f"{grid_reference}: source CRS is "
                    f"{record['spatial_reference_system']}"
                )
        tiles.append(
            {
                "grid_reference": grid_reference,
                "laz": laz,
                "dtm": dtm,
            }
        )
    if len(tiles) < 100:
        raise RuntimeError(
            f"Only {len(tiles)} source tiles intersect Arran; expected island coverage"
        )
    inventory = {
        "created_utc": utc_now(),
        "study_area": "Arran",
        "survey_year": survey_year,
        "boundary_sha256": sha256(BOUNDARY_PATH),
        "config_sha256": sha256(CONFIG_PATH),
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "tile_count": len(tiles),
        "total_laz_bytes": sum(tile["laz"]["size_bytes"] for tile in tiles),
        "total_dtm_bytes": sum(tile["dtm"]["size_bytes"] for tile in tiles),
        "tiles": tiles,
    }
    core.atomic_json(INVENTORY_PATH, inventory)
    print(
        f"inventory: froze {len(tiles)} LAZ/DTM pairs; "
        f"{inventory['total_laz_bytes'] / 1e9:.1f} GB compressed LAZ",
        flush=True,
    )


def metric_checkpoint_valid(
    tile: dict[str, Any],
    parquet_path: Path,
    metadata_path: Path,
) -> bool:
    if not parquet_path.exists() or not metadata_path.exists():
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return bool(
        metadata["grid_reference"] == tile["grid_reference"]
        and metadata["laz_url"] == tile["laz"]["url"]
        and metadata["laz_size_bytes"] == tile["laz"]["size_bytes"]
        and metadata["dtm_url"] == tile["dtm"]["url"]
        and metadata["dtm_size_bytes"] == tile["dtm"]["size_bytes"]
        and metadata["config_sha256"] == sha256(CONFIG_PATH)
        and metadata["protocol_sha256"] == sha256(PROTOCOL_PATH)
        and metadata["output_sha256"] == sha256(parquet_path)
    )


def run_lidar_metrics(config: dict[str, Any], smoke_tile: str | None = None) -> None:
    inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    tiles = inventory["tiles"]
    if smoke_tile is not None:
        tiles = [
            tile
            for tile in tiles
            if tile["grid_reference"].upper() == smoke_tile.upper()
        ]
        if len(tiles) != 1:
            raise RuntimeError(f"Smoke tile {smoke_tile} is not in the Arran inventory")
    METRIC_TILE_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_TILE_METADATA_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_STREAM_DIR.mkdir(parents=True, exist_ok=True)
    completed = 0
    eligible_rows = 0
    completed_source_bytes = 0

    for position, tile in enumerate(tiles, start=1):
        grid = tile["grid_reference"].upper()
        parquet_path = METRIC_TILE_DIR / f"{grid}.parquet"
        metadata_path = METRIC_TILE_METADATA_DIR / f"{grid}.json"
        if metric_checkpoint_valid(tile, parquet_path, metadata_path):
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            completed += 1
            eligible_rows += int(metadata["summary"]["eligible_forest_cells"])
            completed_source_bytes += int(tile["laz"]["size_bytes"]) + int(
                tile["dtm"]["size_bytes"]
            )
            print(
                f"lidar [{position}/{len(tiles)}]: resumed {grid}, "
                f"{metadata['summary']['eligible_forest_cells']:,} forest cells",
                flush=True,
            )
            continue

        required = (
            int(tile["laz"]["size_bytes"])
            + int(tile["dtm"]["size_bytes"])
            + int(
                config["source"]["minimum_free_space_after_source_download_bytes"]
            )
        )
        free = shutil.disk_usage(ROOT).free
        if free < required:
            raise RuntimeError(
                f"{grid}: insufficient free space for one streamed tile; "
                f"{free / 1e9:.2f} GB free, {required / 1e9:.2f} GB required"
            )
        laz_path = SOURCE_STREAM_DIR / f"{grid}.laz"
        dtm_path = SOURCE_STREAM_DIR / f"{grid}_DTM.tif"
        update_run_status(
            "lidar_metrics",
            "running",
            detail={
                "current_tile": grid,
                "tile_position": position,
                "tile_count": len(tiles),
                "completed_tiles": completed,
                "eligible_forest_cells": eligible_rows,
            },
        )
        print(
            f"lidar [{position}/{len(tiles)}]: acquire {grid} "
            f"({tile['laz']['size_bytes'] / 1e6:.1f} MB LAZ)",
            flush=True,
        )
        core.download_resumable(
            tile["laz"]["url"],
            laz_path,
            int(tile["laz"]["size_bytes"]),
            user_agent=config["source"]["user_agent"],
            attempts=int(config["source"]["download_attempts"]),
            retry_delay_seconds=int(config["source"]["retry_delay_seconds"]),
        )
        core.download_resumable(
            tile["dtm"]["url"],
            dtm_path,
            int(tile["dtm"]["size_bytes"]),
            user_agent=config["source"]["user_agent"],
            attempts=int(config["source"]["download_attempts"]),
            retry_delay_seconds=int(config["source"]["retry_delay_seconds"]),
        )
        laz_hash = sha256(laz_path)
        dtm_hash = sha256(dtm_path)
        frame, summary = core.process_laz_tile(
            laz_path,
            dtm_path,
            tile_name=grid,
            metric_resolution_m=float(
                config["metric_generation"]["metric_resolution_m"]
            ),
            expected_dtm_resolution_m=float(
                config["source"]["expected_dtm_resolution_m"]
            ),
            expected_crs_epsg=int(config["source"]["expected_crs_epsg"]),
            chunk_points=int(config["metric_generation"]["point_chunk_size"]),
            metric_settings=config["metric_generation"],
            forest_rules=config["sampling"]["forest_mask"],
        )
        core.atomic_parquet(frame, parquet_path)
        metadata = {
            "created_utc": utc_now(),
            "grid_reference": grid,
            "laz_url": tile["laz"]["url"],
            "laz_size_bytes": int(tile["laz"]["size_bytes"]),
            "laz_sha256": laz_hash,
            "dtm_url": tile["dtm"]["url"],
            "dtm_size_bytes": int(tile["dtm"]["size_bytes"]),
            "dtm_sha256": dtm_hash,
            "config_sha256": sha256(CONFIG_PATH),
            "protocol_sha256": sha256(PROTOCOL_PATH),
            "lidarshm_reference_commit": config["metric_generation"][
                "lidarshm_commit"
            ],
            "summary": summary,
            "output_path": str(parquet_path.relative_to(ROOT)),
            "output_sha256": sha256(parquet_path),
        }
        core.atomic_json(metadata_path, metadata)
        laz_path.unlink()
        dtm_path.unlink()
        completed += 1
        eligible_rows += int(summary["eligible_forest_cells"])
        completed_source_bytes += int(tile["laz"]["size_bytes"]) + int(
            tile["dtm"]["size_bytes"]
        )
        core.atomic_json(
            LIDAR_STATUS_PATH,
            {
                "updated_utc": utc_now(),
                "state": "running" if position < len(tiles) else "complete",
                "current_tile": grid,
                "completed_tiles": completed,
                "tile_count": len(tiles),
                "completion_fraction": completed / len(tiles),
                "eligible_forest_cells": eligible_rows,
                "completed_source_bytes": completed_source_bytes,
                "total_source_bytes": sum(
                    int(item["laz"]["size_bytes"])
                    + int(item["dtm"]["size_bytes"])
                    for item in tiles
                ),
            },
        )
        print(
            f"lidar [{position}/{len(tiles)}]: checkpointed {grid}, "
            f"{summary['eligible_forest_cells']:,} forest cells",
            flush=True,
        )
        gc.collect()

    if smoke_tile is None:
        if completed != len(tiles):
            raise RuntimeError("The LiDAR metric stage ended before every tile completed")
        core.atomic_json(
            LIDAR_STATUS_PATH,
            {
                "updated_utc": utc_now(),
                "state": "complete",
                "completed_tiles": completed,
                "tile_count": len(tiles),
                "completion_fraction": 1.0,
                "eligible_forest_cells": eligible_rows,
                "completed_source_bytes": completed_source_bytes,
                "total_source_bytes": completed_source_bytes,
            },
        )


def run_sample(config: dict[str, Any]) -> None:
    if SAMPLE_PATH.exists() and SAMPLE_QC_PATH.exists():
        qc = json.loads(SAMPLE_QC_PATH.read_text(encoding="utf-8"))
        if (
            qc["config_sha256"] == sha256(CONFIG_PATH)
            and qc["output_sha256"] == sha256(SAMPLE_PATH)
        ):
            print("sample: validated complete Arran sample checkpoint", flush=True)
            return
        raise RuntimeError("The Arran sample checkpoint differs from the protocol")
    inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    frames = []
    tile_hashes = {}
    for tile in inventory["tiles"]:
        grid = tile["grid_reference"].upper()
        parquet_path = METRIC_TILE_DIR / f"{grid}.parquet"
        metadata_path = METRIC_TILE_METADATA_DIR / f"{grid}.json"
        if not metric_checkpoint_valid(tile, parquet_path, metadata_path):
            raise RuntimeError(f"Missing or stale metric checkpoint for {grid}")
        frames.append(pd.read_parquet(parquet_path))
        tile_hashes[grid] = sha256(parquet_path)
    population = pd.concat(frames, ignore_index=True)
    if population.empty:
        raise RuntimeError("No Arran cells passed the frozen forest quality rules")
    if population.duplicated(["bng_x", "bng_y"]).any():
        raise RuntimeError("Duplicate 10 m cell centres occur across Arran source tiles")
    population = population.sort_values(["bng_x", "bng_y"]).reset_index(drop=True)
    eligible_count = len(population)
    maximum = int(config["sampling"]["maximum_rows"])
    if eligible_count > maximum:
        rng = np.random.default_rng(int(config["sampling"]["seed"]))
        selected = np.sort(rng.choice(eligible_count, size=maximum, replace=False))
        frame = population.iloc[selected].copy().reset_index(drop=True)
    else:
        frame = population.copy()
    transformer = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
    longitude, latitude = transformer.transform(
        frame["bng_x"].to_numpy(dtype=np.float64),
        frame["bng_y"].to_numpy(dtype=np.float64),
    )
    frame["longitude"] = longitude
    frame["latitude"] = latitude
    frame["tessera_tile"] = [
        tessera.tile_name(tessera.tile_from_world(float(lon), float(lat)))
        for lon, lat in zip(longitude, latitude, strict=True)
    ]
    frame = phase14.assign_spatial_regions(frame, config)
    frame.insert(0, "row_id", np.arange(len(frame), dtype=np.int64))
    core.atomic_parquet(frame, SAMPLE_PATH)
    core.atomic_json(
        SAMPLE_QC_PATH,
        {
            "created_utc": utc_now(),
            "eligible_forest_cells_before_sampling": eligible_count,
            "sample_rows": len(frame),
            "sample_seed": int(config["sampling"]["seed"]),
            "sample_is_complete_population": eligible_count <= maximum,
            "spatial_blocks": int(frame["spatial_block"].nunique()),
            "spatial_regions": {
                str(region): int(count)
                for region, count in frame.groupby("spatial_region").size().items()
            },
            "tessera_tiles": sorted(frame["tessera_tile"].unique().tolist()),
            "metric_tile_hashes": tile_hashes,
            "config_sha256": sha256(CONFIG_PATH),
            "protocol_sha256": sha256(PROTOCOL_PATH),
            "output_sha256": sha256(SAMPLE_PATH),
        },
    )
    print(
        f"sample: wrote {len(frame):,} of {eligible_count:,} eligible Arran cells "
        f"across {frame['spatial_block'].nunique()} 1 km blocks",
        flush=True,
    )


def _download_checked(
    url: str,
    destination: Path,
    expected_size: int,
    config: dict[str, Any],
) -> None:
    core.download_resumable(
        url,
        destination,
        expected_size,
        user_agent=config["source"]["user_agent"],
        attempts=int(config["tessera"]["download_attempts"]),
        retry_delay_seconds=int(config["tessera"]["retry_delay_seconds"]),
    )


def _parse_tessera_tile(name: str) -> tuple[float, float]:
    prefix, longitude, latitude = name.split("_")
    if prefix != "grid":
        raise RuntimeError(f"Unexpected TESSERA tile name: {name}")
    return float(longitude), float(latitude)


def run_tessera(config: dict[str, Any]) -> None:
    if TESSERA_PATH.exists() and TESSERA_MANIFEST_PATH.exists():
        manifest = json.loads(TESSERA_MANIFEST_PATH.read_text(encoding="utf-8"))
        if (
            manifest["sample_sha256"] == sha256(SAMPLE_PATH)
            and manifest["output_sha256"] == sha256(TESSERA_PATH)
            and manifest["dataset_version"]
            == str(config["tessera"]["dataset_version"])
            and manifest["dataset_variant"]
            == str(config["tessera"]["dataset_variant"])
        ):
            print("tessera: validated complete Arran alignment checkpoint", flush=True)
            return
        raise RuntimeError("The Arran TESSERA checkpoint differs from the protocol")
    sample = pd.read_parquet(
        SAMPLE_PATH,
        columns=[
            "row_id",
            "bng_x",
            "bng_y",
            "longitude",
            "latitude",
            "tessera_tile",
        ],
    )
    setting = config["tessera"]
    version = str(setting["dataset_version"])
    variant = str(setting["dataset_variant"])
    year = int(setting["embedding_year"])
    tile_names = sorted(sample["tessera_tile"].unique())
    tiles = [_parse_tessera_tile(name) for name in tile_names]
    manifest_path, landmask_path = tessera.ensure_registry(version)
    rows = tessera.lookup_embedding_rows(
        manifest_path, tiles, year, version, variant
    )
    masks = tessera.lookup_landmask_rows(landmask_path, tiles)
    tessera.assert_complete_inventory(rows, masks, tiles)
    rows_by_tile = {
        tessera.tile_name((float(row.lon), float(row.lat))): row
        for row in rows.itertuples(index=False)
    }
    masks_by_tile = {
        tessera.tile_name((float(row.lon), float(row.lat))): row
        for row in masks.itertuples(index=False)
    }
    TESSERA_CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    TESSERA_STREAM_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for position, name in enumerate(tile_names, start=1):
        scoped = sample[sample["tessera_tile"].eq(name)].copy()
        checkpoint = TESSERA_CHUNK_DIR / f"{name}.parquet"
        if checkpoint.exists():
            chunk = pd.read_parquet(checkpoint)
            if len(chunk) != len(scoped) or set(chunk["row_id"]) != set(
                scoped["row_id"]
            ):
                raise RuntimeError(f"Stale Arran TESSERA chunk: {name}")
            records.append(
                {
                    "tile": name,
                    "sample_rows": len(chunk),
                    "valid_rows": int(chunk["tessera_valid"].sum()),
                    "chunk_sha256": sha256(checkpoint),
                    "resumed": True,
                }
            )
            print(
                f"tessera [{position}/{len(tile_names)}]: resumed {name}",
                flush=True,
            )
            continue
        row = rows_by_tile[name]
        mask_row = masks_by_tile[name]
        tile = _parse_tessera_tile(name)
        paths = tessera.local_tile_paths(TESSERA_STREAM_DIR, year, tile)
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
        required = sum(sizes.values()) + int(
            config["source"]["minimum_free_space_after_source_download_bytes"]
        )
        if shutil.disk_usage(ROOT).free < required:
            raise RuntimeError(
                f"{name}: insufficient free space for one TESSERA tile"
            )
        print(
            f"tessera [{position}/{len(tile_names)}]: acquire {name} "
            f"for {len(scoped):,} cells",
            flush=True,
        )
        try:
            for kind in ["embedding", "scales", "landmask"]:
                _download_checked(
                    tessera.s3_url(sources[kind]),
                    paths[kind],
                    sizes[kind],
                    config,
                )
            quantized = np.load(paths["embedding"], mmap_mode="r")
            scales = np.load(paths["scales"], mmap_mode="r")
            with rasterio.open(paths["landmask"]) as land:
                landmask = land.read(1)
                destination_crs = land.crs
                transform = land.transform
            transformer = Transformer.from_crs(
                "EPSG:27700", destination_crs, always_xy=True
            )
            x, y = transformer.transform(
                scoped["bng_x"].to_numpy(),
                scoped["bng_y"].to_numpy(),
            )
            pixel_rows, pixel_columns = rowcol(transform, x, y)
            pixel_rows = np.asarray(pixel_rows, dtype=np.int64)
            pixel_columns = np.asarray(pixel_columns, dtype=np.int64)
            inside = (
                (pixel_rows >= 0)
                & (pixel_columns >= 0)
                & (pixel_rows < quantized.shape[0])
                & (pixel_columns < quantized.shape[1])
            )
            embeddings = np.full((len(scoped), 128), np.nan, dtype=np.float32)
            valid = np.zeros(len(scoped), dtype=bool)
            indices = np.flatnonzero(inside)
            if len(indices):
                local_scales = scales[
                    pixel_rows[indices], pixel_columns[indices]
                ].astype(np.float32)
                usable = (
                    np.isfinite(local_scales)
                    & (local_scales > 0)
                    & (
                        landmask[
                            pixel_rows[indices],
                            pixel_columns[indices],
                        ]
                        > 0
                    )
                )
                valid_indices = indices[usable]
                embeddings[valid_indices] = (
                    quantized[
                        pixel_rows[valid_indices],
                        pixel_columns[valid_indices],
                    ].astype(np.float32)
                    * scales[
                        pixel_rows[valid_indices],
                        pixel_columns[valid_indices],
                    ].astype(np.float32)[:, None]
                )
                valid[valid_indices] = np.isfinite(
                    embeddings[valid_indices]
                ).all(axis=1)
            output = pd.DataFrame(
                embeddings, columns=TESSERA_FEATURES, dtype=np.float32
            )
            output.insert(0, "tessera_valid", valid)
            output.insert(0, "row_id", scoped["row_id"].to_numpy(dtype=np.int64))
            core.atomic_parquet(output, checkpoint)
            records.append(
                {
                    "tile": name,
                    "sample_rows": len(output),
                    "valid_rows": int(valid.sum()),
                    "embedding_source": sources["embedding"],
                    "scales_source": sources["scales"],
                    "landmask_source": sources["landmask"],
                    "source_sizes": sizes,
                    "chunk_sha256": sha256(checkpoint),
                    "resumed": False,
                }
            )
        finally:
            for path in paths.values():
                path.unlink(missing_ok=True)
            gc.collect()
    aligned = pd.concat(
        [pd.read_parquet(TESSERA_CHUNK_DIR / f"{name}.parquet") for name in tile_names],
        ignore_index=True,
    ).sort_values("row_id")
    if len(aligned) != len(sample) or aligned["row_id"].duplicated().any():
        raise RuntimeError("Arran TESSERA alignment does not match the sample")
    valid_fraction = float(aligned["tessera_valid"].mean())
    if valid_fraction < float(setting["minimum_valid_alignment_fraction"]):
        raise RuntimeError(
            f"Arran TESSERA valid fraction {valid_fraction:.3f} is below threshold"
        )
    core.atomic_parquet(aligned, TESSERA_PATH)
    core.atomic_json(
        TESSERA_MANIFEST_PATH,
        {
            "created_utc": utc_now(),
            "sample_sha256": sha256(SAMPLE_PATH),
            "registry_manifest_sha256": sha256(manifest_path),
            "landmask_manifest_sha256": sha256(landmask_path),
            "dataset_version": version,
            "dataset_variant": variant,
            "embedding_year": year,
            "tile_count": len(tile_names),
            "sample_rows": len(aligned),
            "valid_rows": int(aligned["tessera_valid"].sum()),
            "valid_fraction": valid_fraction,
            "streamed_source_tiles_retained": False,
            "chunks": records,
            "output_sha256": sha256(TESSERA_PATH),
        },
    )
    print(
        f"tessera: aligned {int(aligned['tessera_valid'].sum()):,}/"
        f"{len(aligned):,} Arran cells",
        flush=True,
    )


def _conventional_features(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    excluded = tuple(
        load_config()["conventional"]["exclude_feature_suffixes"]
    )
    terrain = sorted(
        column for column in frame if column.startswith("terrain_")
    )
    sentinel = sorted(
        column
        for column in frame
        if column.startswith("s2_")
        and not column.endswith(excluded)
    )
    if not terrain or not sentinel:
        raise RuntimeError("The conventional predictor checkpoint is incomplete")
    return terrain, sentinel


def _build_conventional_partition(
    *,
    sample_path: Path,
    output_path: Path,
    inventory_path: Path,
    manifest_path: Path,
    year: int,
    landscape: str,
    config: dict[str, Any],
) -> None:
    if output_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest["sample_sha256"] == sha256(sample_path)
            and manifest["output_sha256"] == sha256(output_path)
            and int(manifest["year"]) == year
        ):
            print(
                f"conventional: validated {landscape} checkpoint",
                flush=True,
            )
            return
        raise RuntimeError(
            f"The {landscape} conventional checkpoint differs from the protocol"
        )

    columns = ["row_id", "bng_x", "bng_y", "longitude", "latitude"]
    sample = pd.read_parquet(sample_path, columns=columns)
    if sample["row_id"].duplicated().any():
        raise RuntimeError(f"Duplicate row IDs in the {landscape} sample")
    longitude = sample["longitude"].to_numpy(dtype=np.float64)
    latitude = sample["latitude"].to_numpy(dtype=np.float64)
    bbox = [
        float(longitude.min()),
        float(latitude.min()),
        float(longitude.max()),
        float(latitude.max()),
    ]
    catalog = conventional.open_catalog(config["conventional"]["stac_api"])
    s2_config = config["conventional"]["sentinel_2"]
    terrain_config = config["conventional"]["topography"]
    s2_candidates = conventional.collection_items(
        catalog, s2_config["collection"], bbox, year
    )
    s2_items = conventional.select_sentinel2(s2_candidates, s2_config)
    dem_items = conventional.collection_items(
        catalog, terrain_config["collection"], bbox, None
    )
    print(
        f"conventional: {landscape} selected {len(s2_items)} Sentinel-2 "
        f"scenes and {len(dem_items)} DEM tiles",
        flush=True,
    )
    s2_features, s2_inventory = conventional.extract_sentinel2(
        s2_items, longitude, latitude, s2_config
    )
    # The shared terrain routine expects projected coordinate columns under
    # legacy names. Values remain unmodified British National Grid metres.
    terrain_input = sample.rename(
        columns={"bng_x": "x_epsg5070", "bng_y": "y_epsg5070"}
    )
    terrain_features, terrain_inventory = conventional.extract_topography(
        dem_items, terrain_input, bbox, terrain_config
    )
    output = sample[["row_id"]].copy()
    for name, values in sorted({**terrain_features, **s2_features}.items()):
        output[name] = values
    minimum = int(
        s2_config["minimum_valid_observations_per_point"]
    )
    output["conventional_valid"] = (
        output["s2_valid_observation_count"].to_numpy(dtype=np.int64)
        >= minimum
    ) & np.isfinite(
        output[
            [
                column
                for column in output
                if column.startswith("terrain_")
            ]
        ].to_numpy(dtype=np.float64)
    ).all(axis=1)
    core.atomic_parquet(output, output_path)
    inventory = pd.DataFrame(
        [
            {"landscape": landscape, **record}
            for record in s2_inventory + terrain_inventory
        ]
    )
    write_csv(inventory, inventory_path)
    core.atomic_json(
        manifest_path,
        {
            "created_utc": utc_now(),
            "landscape": landscape,
            "year": year,
            "sample_path": str(sample_path.relative_to(ROOT)),
            "sample_sha256": sha256(sample_path),
            "bbox_wgs84": bbox,
            "sentinel_2_item_ids": [item.id for item in s2_items],
            "dem_item_ids": [item.id for item in dem_items],
            "sentinel_2_recipe": s2_config,
            "topography_recipe": terrain_config,
            "minimum_valid_observations": minimum,
            "valid_rows": int(output["conventional_valid"].sum()),
            "sample_rows": len(output),
            "output_path": str(output_path.relative_to(ROOT)),
            "output_sha256": sha256(output_path),
            "inventory_sha256": sha256(inventory_path),
            "config_sha256": sha256(CONFIG_PATH),
            "protocol_sha256": sha256(PROTOCOL_PATH),
        },
    )
    print(
        f"conventional: {landscape} aligned "
        f"{int(output['conventional_valid'].sum()):,}/{len(output):,} cells",
        flush=True,
    )


def run_conventional(config: dict[str, Any]) -> None:
    _build_conventional_partition(
        sample_path=SAMPLE_PATH,
        output_path=ARRAN_CONVENTIONAL_PATH,
        inventory_path=ARRAN_CONVENTIONAL_INVENTORY_PATH,
        manifest_path=ARRAN_CONVENTIONAL_MANIFEST_PATH,
        year=int(config["source"]["survey_year"]),
        landscape="Arran",
        config=config,
    )
    source_sample = ROOT / config["cross_landscape"]["source_sample"]
    _build_conventional_partition(
        sample_path=source_sample,
        output_path=CAIRNGORMS_CONVENTIONAL_PATH,
        inventory_path=CAIRNGORMS_CONVENTIONAL_INVENTORY_PATH,
        manifest_path=CAIRNGORMS_CONVENTIONAL_MANIFEST_PATH,
        year=int(config["cross_landscape"]["source_year"]),
        landscape="Cairngorms Connect",
        config=config,
    )


def _joined_landscape(
    sample_path: Path,
    tessera_path: Path,
    conventional_path: Path,
) -> pd.DataFrame:
    sample = pd.read_parquet(sample_path)
    embeddings = pd.read_parquet(tessera_path)
    conventional_frame = pd.read_parquet(conventional_path)
    frame = sample.merge(
        embeddings, on="row_id", how="inner", validate="one_to_one"
    ).merge(
        conventional_frame, on="row_id", how="inner", validate="one_to_one"
    )
    if len(frame) != len(sample):
        raise RuntimeError("Predictor alignment dropped frozen sample rows")
    return frame.sort_values("row_id").reset_index(drop=True)


def _predict_ridge(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_weight: np.ndarray,
    test_x: np.ndarray,
    alpha: float,
) -> np.ndarray:
    scaler, model = phase14.fit_ridge(
        train_x, train_y, train_weight, alpha
    )
    return model.predict(scaler.transform(test_x))


def _fit_height_adjustment(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    height_features: list[str],
    weights: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    train_x = train[height_features].to_numpy(dtype=np.float64)
    test_x = test[height_features].to_numpy(dtype=np.float64)
    train_y = train[target].to_numpy(dtype=np.float64)
    test_y = test[target].to_numpy(dtype=np.float64)
    scaler, model = phase14.fit_ridge(train_x, train_y, weights, alpha)
    return (
        train_y - model.predict(scaler.transform(train_x)),
        test_y - model.predict(scaler.transform(test_x)),
    )


def _append_metrics(
    records: list[dict[str, Any]],
    *,
    experiment: str,
    target: str,
    model: str,
    fold: int | str,
    observed: np.ndarray,
    predicted: np.ndarray,
    train_rows: int,
    test_rows: int,
) -> None:
    records.append(
        {
            "experiment": experiment,
            "target": target,
            "model": model,
            "fold": fold,
            "train_rows": train_rows,
            "test_rows": test_rows,
            **phase14.metric_values(observed, predicted),
        }
    )


def _common_valid_mask(
    frame: pd.DataFrame,
    target: str,
    terrain_features: list[str],
    sentinel_features: list[str],
    config: dict[str, Any],
) -> np.ndarray:
    return (
        phase14.valid_target_mask(frame, target, config)
        & frame["tessera_valid"].to_numpy(dtype=bool)
        & frame["conventional_valid"].to_numpy(dtype=bool)
        & np.isfinite(
            frame[TESSERA_FEATURES].to_numpy(dtype=np.float32)
        ).all(axis=1)
        & np.isfinite(
            frame[terrain_features + sentinel_features].to_numpy(
                dtype=np.float32
            )
        ).all(axis=1)
    )


def _model_predictions(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_y: np.ndarray,
    weights: np.ndarray,
    terrain_features: list[str],
    sentinel_features: list[str],
    config: dict[str, Any],
    nonlinear: bool,
) -> dict[str, np.ndarray]:
    alpha = float(config["models"]["ridge_alpha"])
    outputs: dict[str, np.ndarray] = {
        "training_mean": np.full(len(test), np.average(train_y, weights=weights))
    }
    coordinate_features = ["bng_x", "bng_y"]
    coordinate_model = make_pipeline(
        PolynomialFeatures(
            degree=int(config["models"]["coordinate_polynomial_degree"]),
            include_bias=False,
        ),
        StandardScaler(),
        Ridge(alpha=alpha),
    )
    coordinate_model.fit(
        train[coordinate_features].to_numpy(dtype=np.float64),
        train_y,
        ridge__sample_weight=weights,
    )
    outputs["coordinates"] = coordinate_model.predict(
        test[coordinate_features].to_numpy(dtype=np.float64)
    )
    feature_sets = {
        "terrain_ridge": terrain_features,
        "sentinel2_terrain_ridge": terrain_features + sentinel_features,
        "tessera_ridge": TESSERA_FEATURES,
    }
    for model_name, columns in feature_sets.items():
        outputs[model_name] = _predict_ridge(
            train[columns].to_numpy(dtype=np.float32),
            train_y,
            weights,
            test[columns].to_numpy(dtype=np.float32),
            alpha,
        )
    if nonlinear:
        for model_name, columns in {
            "sentinel2_terrain_nonlinear": terrain_features
            + sentinel_features,
            "tessera_nonlinear": TESSERA_FEATURES,
        }.items():
            model = phase14.fit_histogram_model(
                train[columns].to_numpy(dtype=np.float32),
                train_y,
                weights,
                config,
            )
            outputs[model_name] = model.predict(
                test[columns].to_numpy(dtype=np.float32)
            )
    return outputs


def run_evaluate(config: dict[str, Any]) -> None:
    if SPATIAL_METRICS_PATH.exists() and SPATIAL_PREDICTIONS_PATH.exists():
        print("evaluate: validated existing Arran spatial outputs", flush=True)
        return
    frame = _joined_landscape(
        SAMPLE_PATH, TESSERA_PATH, ARRAN_CONVENTIONAL_PATH
    )
    terrain_features, sentinel_features = _conventional_features(frame)
    targets = list(config["targets"]["representative"])
    nonlinear_targets = set(config["targets"]["nonlinear"])
    records: list[dict[str, Any]] = []
    primary_records: list[pd.DataFrame] = []
    region_count = int(config["spatial_evaluation"]["region_count"])
    minimum_train = int(
        config["spatial_evaluation"]["minimum_training_rows_per_target"]
    )
    minimum_test = int(
        config["spatial_evaluation"]["minimum_test_rows_per_target"]
    )

    for fold in range(region_count):
        base_train, base_test, fold_info = phase14.buffered_fold_masks(
            frame, fold, config
        )
        for target in targets:
            common = _common_valid_mask(
                frame,
                target,
                terrain_features,
                sentinel_features,
                config,
            )
            train_mask = base_train & common
            test_mask = base_test & common
            if train_mask.sum() < minimum_train or test_mask.sum() < minimum_test:
                raise RuntimeError(
                    f"{target} fold {fold} has {train_mask.sum()} train and "
                    f"{test_mask.sum()} test rows after common filtering"
                )
            train = frame.loc[train_mask]
            test = frame.loc[test_mask]
            train_y = train[target].to_numpy(dtype=np.float64)
            test_y = test[target].to_numpy(dtype=np.float64)
            weights = phase14.spatial_weights(train["spatial_block"])
            predictions = _model_predictions(
                train=train,
                test=test,
                train_y=train_y,
                weights=weights,
                terrain_features=terrain_features,
                sentinel_features=sentinel_features,
                config=config,
                nonlinear=target in nonlinear_targets,
            )
            for model_name, predicted in predictions.items():
                _append_metrics(
                    records,
                    experiment="arran_buffered_spatial",
                    target=target,
                    model=model_name,
                    fold=fold,
                    observed=test_y,
                    predicted=predicted,
                    train_rows=len(train),
                    test_rows=len(test),
                )
            if target == config["targets"]["primary"]:
                output = test[
                    ["row_id", "bng_x", "bng_y", "spatial_block"]
                ].copy()
                output["fold"] = fold
                output["observed"] = test_y
                for model_name, predicted in predictions.items():
                    output[model_name] = predicted
                primary_records.append(output)

                height_features = list(
                    config["targets"]["height_adjustment_predictors"]
                )
                adjusted_train, adjusted_test = _fit_height_adjustment(
                    train,
                    test,
                    target,
                    height_features,
                    weights,
                    float(config["models"]["ridge_alpha"]),
                )
                adjusted_predictions = _model_predictions(
                    train=train,
                    test=test,
                    train_y=adjusted_train,
                    weights=weights,
                    terrain_features=terrain_features,
                    sentinel_features=sentinel_features,
                    config=config,
                    nonlinear=True,
                )
                for model_name, predicted in adjusted_predictions.items():
                    _append_metrics(
                        records,
                        experiment="arran_height_adjusted_spatial",
                        target=f"{target}_height_adjusted",
                        model=model_name,
                        fold=fold,
                        observed=adjusted_test,
                        predicted=predicted,
                        train_rows=len(train),
                        test_rows=len(test),
                    )
        print(
            f"evaluate: completed Arran fold {fold + 1}/{region_count}; "
            f"buffer excluded {fold_info['buffered_training_rows']:,} rows",
            flush=True,
        )

    metrics = pd.DataFrame(records)
    predictions = pd.concat(primary_records, ignore_index=True).sort_values(
        "row_id"
    )
    write_csv(metrics, SPATIAL_METRICS_PATH)
    core.atomic_parquet(predictions, SPATIAL_PREDICTIONS_PATH)
    print(
        f"evaluate: wrote {len(metrics):,} fold-level metric records",
        flush=True,
    )


def run_transfer(config: dict[str, Any]) -> None:
    if TRANSFER_METRICS_PATH.exists() and TRANSFER_PREDICTIONS_PATH.exists():
        print("transfer: validated existing cross-landscape outputs", flush=True)
        return
    source = _joined_landscape(
        ROOT / config["cross_landscape"]["source_sample"],
        ROOT / config["cross_landscape"]["source_tessera"],
        CAIRNGORMS_CONVENTIONAL_PATH,
    )
    target = _joined_landscape(
        SAMPLE_PATH, TESSERA_PATH, ARRAN_CONVENTIONAL_PATH
    )
    terrain_features, sentinel_features = _conventional_features(target)
    if _conventional_features(source) != (terrain_features, sentinel_features):
        raise RuntimeError(
            "Cairngorms and Arran conventional feature schemas differ"
        )
    target_name = str(config["targets"]["primary"])
    source_valid = _common_valid_mask(
        source,
        target_name,
        terrain_features,
        sentinel_features,
        config,
    )
    target_valid = _common_valid_mask(
        target,
        target_name,
        terrain_features,
        sentinel_features,
        config,
    )
    source = source.loc[source_valid].copy()
    target = target.loc[target_valid].copy()
    source_y = source[target_name].to_numpy(dtype=np.float64)
    weights = phase14.spatial_weights(source["spatial_block"])
    predictions = _model_predictions(
        train=source,
        test=target,
        train_y=source_y,
        weights=weights,
        terrain_features=terrain_features,
        sentinel_features=sentinel_features,
        config=config,
        nonlinear=True,
    )
    output = target[
        ["row_id", "bng_x", "bng_y", "spatial_region", "spatial_block"]
    ].copy()
    output["observed"] = target[target_name].to_numpy(dtype=np.float64)
    for model_name, predicted in predictions.items():
        output[model_name] = predicted
    records: list[dict[str, Any]] = []
    scopes: list[tuple[str, np.ndarray]] = [
        ("all_arran", np.ones(len(output), dtype=bool))
    ]
    scopes.extend(
        (
            f"arran_region_{region}",
            output["spatial_region"].to_numpy() == region,
        )
        for region in sorted(output["spatial_region"].unique())
    )
    for scope, mask in scopes:
        observed = output.loc[mask, "observed"].to_numpy(dtype=np.float64)
        for model_name, predicted in predictions.items():
            _append_metrics(
                records,
                experiment="cairngorms_2023_to_arran_2025",
                target=target_name,
                model=model_name,
                fold=scope,
                observed=observed,
                predicted=np.asarray(predicted)[mask],
                train_rows=len(source),
                test_rows=int(mask.sum()),
            )
    write_csv(pd.DataFrame(records), TRANSFER_METRICS_PATH)
    core.atomic_parquet(output, TRANSFER_PREDICTIONS_PATH)
    print(
        f"transfer: evaluated {len(source):,} Cairngorms source cells "
        f"against {len(target):,} Arran cells",
        flush=True,
    )


def _macro_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = ["rmse", "mae", "r2", "pearson_r", "spearman_r", "bias"]
    return (
        frame.groupby(["experiment", "target", "model"], as_index=False)[
            metrics
        ]
        .mean()
        .sort_values(["experiment", "target", "rmse", "model"])
    )


def run_report(config: dict[str, Any]) -> None:
    spatial = pd.read_csv(SPATIAL_METRICS_PATH)
    transfer = pd.read_csv(TRANSFER_METRICS_PATH)
    macro = _macro_metrics(spatial)
    primary = str(config["targets"]["primary"])
    local = macro[
        macro["experiment"].eq("arran_buffered_spatial")
        & macro["target"].eq(primary)
    ].copy()
    overall_transfer = transfer[
        transfer["fold"].eq("all_arran")
        & transfer["target"].eq(primary)
    ].copy()
    if local.empty or overall_transfer.empty:
        raise RuntimeError("Primary Arran results are missing")

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    selected_models = [
        "training_mean",
        "sentinel2_terrain_ridge",
        "sentinel2_terrain_nonlinear",
        "tessera_ridge",
        "tessera_nonlinear",
    ]
    labels = ["Mean", "S2 ridge", "S2 nonlinear", "TESSERA ridge", "TESSERA nonlinear"]
    local_index = local.set_index("model")
    transfer_index = overall_transfer.set_index("model")
    for axis, table, title in [
        (axes[0], local_index, "Within Arran: buffered spatial folds"),
        (axes[1], transfer_index, "Cairngorms 2023 to Arran 2025"),
    ]:
        available = [
            (model, label)
            for model, label in zip(selected_models, labels, strict=True)
            if model in table.index
        ]
        axis.bar(
            np.arange(len(available)),
            [table.loc[model, "rmse"] for model, _ in available],
            color=["#777777", "#4477AA", "#66CCEE", "#228833", "#009988"][
                : len(available)
            ],
        )
        axis.set_xticks(
            np.arange(len(available)),
            [label for _, label in available],
            rotation=30,
            ha="right",
        )
        axis.set_ylabel("RMSE")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(FIGURE_PATH, dpi=240, bbox_inches="tight")
    plt.close(figure)

    best_local = local.sort_values("rmse").iloc[0]
    best_transfer = overall_transfer.sort_values("rmse").iloc[0]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            [
                "# Arran airborne-LiDAR replication and transfer",
                "",
                "## Frozen design",
                "",
                (
                    "The analysis derives published lidarSHM-compatible structural "
                    "metrics directly from the 2025 Scottish national airborne-LiDAR "
                    "LAZ tiles at 10 m. Heights are normalized with the paired 50 cm "
                    "terrain model from the same survey. No LAS conversion is used."
                ),
                "",
                (
                    "Models are assessed with five 1 km buffered spatial regions "
                    "within Arran. A separate test trains on Cairngorms Connect 2023 "
                    "and evaluates Arran 2025; that result combines geographic and "
                    "temporal transfer."
                ),
                "",
                "## Primary results",
                "",
                (
                    f"The lowest mean within-Arran RMSE was {best_local.rmse:.3f} "
                    f"for `{best_local.model}` (mean fold R2 "
                    f"{best_local.r2:.3f}, Spearman {best_local.spearman_r:.3f})."
                ),
                "",
                (
                    f"The lowest Cairngorms-to-Arran RMSE was "
                    f"{best_transfer.rmse:.3f} for `{best_transfer.model}` "
                    f"(R2 {best_transfer.r2:.3f}, Spearman "
                    f"{best_transfer.spearman_r:.3f})."
                ),
                "",
                (
                    "The cross-landscape comparison is interpreted cautiously because "
                    "the source and target surveys differ in both landscape and year, "
                    "and their height-normalization workflows are documented separately."
                ),
                "",
                f"![Arran results](../figures/{FIGURE_PATH.name})",
                "",
            ]
        ),
        encoding="utf-8",
    )
    inputs = [
        CONFIG_PATH,
        PROTOCOL_PATH,
        BOUNDARY_PATH,
        INVENTORY_PATH,
        SAMPLE_PATH,
        TESSERA_PATH,
        ARRAN_CONVENTIONAL_PATH,
        CAIRNGORMS_CONVENTIONAL_PATH,
        SPATIAL_METRICS_PATH,
        SPATIAL_PREDICTIONS_PATH,
        TRANSFER_METRICS_PATH,
        TRANSFER_PREDICTIONS_PATH,
        FIGURE_PATH,
        REPORT_PATH,
    ]
    core.atomic_json(
        RESULT_FREEZE_PATH,
        {
            "created_utc": utc_now(),
            "protocol": "phase15_arran_lidar",
            "files": {
                str(path.relative_to(ROOT)): sha256(path) for path in inputs
            },
            "primary_target": primary,
            "within_arran_best_model": best_local.to_dict(),
            "cross_landscape_best_model": best_transfer.to_dict(),
        },
    )
    print(f"report: wrote {REPORT_PATH.relative_to(ROOT)}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--start-at", choices=STAGES)
    parser.add_argument("--stop-after", choices=STAGES)
    parser.add_argument(
        "--smoke-tile",
        help="Process only one named LiDAR tile; valid with --stage lidar_metrics",
    )
    return parser.parse_args()


def selected_stages(args: argparse.Namespace) -> list[str]:
    if args.stage:
        return [args.stage]
    start = STAGES.index(args.start_at) if args.start_at else 0
    stop = STAGES.index(args.stop_after) + 1 if args.stop_after else len(STAGES)
    if stop <= start:
        raise RuntimeError("--stop-after must not precede --start-at")
    return STAGES[start:stop]


def main() -> int:
    args = parse_args()
    if args.smoke_tile and args.stage != "lidar_metrics":
        raise RuntimeError("--smoke-tile requires --stage lidar_metrics")
    config = load_config()
    functions = {
        "boundary": run_boundary,
        "inventory": run_inventory,
        "lidar_metrics": lambda value: run_lidar_metrics(
            value, smoke_tile=args.smoke_tile
        ),
        "sample": run_sample,
        "tessera": run_tessera,
        "conventional": run_conventional,
        "evaluate": run_evaluate,
        "transfer": run_transfer,
        "report": run_report,
    }
    for stage in selected_stages(args):
        update_run_status(stage, "running")
        print(f"\\n=== {stage} ===", flush=True)
        try:
            functions[stage](config)
        except Exception as error:
            update_run_status(
                stage,
                "failed",
                detail={"error_type": type(error).__name__, "error": str(error)},
            )
            raise
        update_run_status(stage, "complete")
    update_run_status(
        selected_stages(args)[-1],
        "complete",
        detail={"completed_stages": selected_stages(args)},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
