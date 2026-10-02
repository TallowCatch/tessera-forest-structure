#!/usr/bin/env python3
"""Acquire year-matched Sentinel and terrain predictors for Savelsbos."""

from __future__ import annotations

import hashlib
import json
import sys
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from pyproj import Transformer


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/ahn4_replication_ahn4_deciduous.yaml"
MANIFEST_PATH = ROOT / "metadata/phase26_ahn4_conventional_manifest.json"


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "phase26_ahn4_deciduous"
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


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def context_points(cohort: pd.DataFrame) -> pd.DataFrame:
    local_rows = np.repeat(np.arange(5, dtype=np.int8), 5)
    local_columns = np.tile(np.arange(5, dtype=np.int8), 5)
    offset_x = local_columns - 2
    offset_y = 2 - local_rows
    bits = (1 << np.arange(25, dtype=np.uint32))[None, :]
    masks = cohort["broadleaf_pixel_mask"].to_numpy(dtype=np.uint32)[:, None]
    return pd.DataFrame(
        {
            "row_id": np.repeat(cohort["row_id"].to_numpy(dtype=np.int64), 25),
            "rd_x": np.repeat(cohort["rd_x"].to_numpy(dtype=np.float64), 25)
            + np.tile(offset_x.astype(np.float64) * 10.0, len(cohort)),
            "rd_y": np.repeat(cohort["rd_y"].to_numpy(dtype=np.float64), 25)
            + np.tile(offset_y.astype(np.float64) * 10.0, len(cohort)),
            "offset_x": np.tile(offset_x, len(cohort)).astype(np.float32),
            "offset_y": np.tile(offset_y, len(cohort)).astype(np.float32),
            "broadleaf": ((masks & bits) > 0).reshape(-1),
        }
    )


def summarise_observations(
    observations: list[np.ndarray], prefix: str, output: dict[str, np.ndarray]
) -> None:
    values = np.stack(observations, axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        output[f"{prefix}_p10"] = np.nanpercentile(values, 10, axis=1).astype(
            np.float32
        )
        output[f"{prefix}_median"] = np.nanmedian(values, axis=1).astype(
            np.float32
        )
        output[f"{prefix}_p90"] = np.nanpercentile(values, 90, axis=1).astype(
            np.float32
        )


def extract_sentinel2(
    conventional: Any,
    items: list[Any],
    longitude: np.ndarray,
    latitude: np.ndarray,
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    observations: dict[str, list[np.ndarray]] = defaultdict(list)
    inventory: list[dict[str, Any]] = []
    valid_scl = np.asarray(config["valid_scl_classes"], dtype=np.int16)
    scale = float(config["reflectance_scale"])
    for position, item in enumerate(items, start=1):
        baseline = float(item.properties["s2:processing_baseline"])
        offset = (
            float(config["processing_baseline_offset_dn"])
            if baseline >= 4.0
            else 0.0
        )
        scl = conventional.sample_asset(item, "SCL", longitude, latitude)
        classes = np.full(len(scl), -9999, dtype=np.int16)
        classes[np.isfinite(scl)] = scl[np.isfinite(scl)].astype(np.int16)
        clear = np.isin(classes, valid_scl)
        reflectance: dict[str, np.ndarray] = {}
        for asset, name in dict(config["bands"]).items():
            raw = conventional.sample_asset(item, asset, longitude, latitude)
            value = (raw - offset) * scale
            value[~clear | ~np.isfinite(raw) | (raw <= 0)] = np.nan
            value[(value < -0.2) | (value > 1.6)] = np.nan
            reflectance[name] = value.astype(np.float32)
            observations[f"s2_{name}"].append(reflectance[name])
        observations["s2_ndvi"].append(
            conventional.finite_ratio(
                reflectance["nir"] - reflectance["red"],
                reflectance["nir"] + reflectance["red"],
            )
        )
        observations["s2_ndmi"].append(
            conventional.finite_ratio(
                reflectance["nir"] - reflectance["swir1"],
                reflectance["nir"] + reflectance["swir1"],
            )
        )
        observations["s2_nbr"].append(
            conventional.finite_ratio(
                reflectance["nir"] - reflectance["swir2"],
                reflectance["nir"] + reflectance["swir2"],
            )
        )
        timestamp = conventional.item_datetime(item)
        inventory.append(
            {
                "item_id": item.id,
                "datetime": timestamp.isoformat(),
                "processing_baseline": baseline,
                "offset_dn_applied": offset,
            }
        )
        print(f"Sentinel-2 {position}/{len(items)}: {item.id}", flush=True)
    output: dict[str, np.ndarray] = {}
    for name, arrays in sorted(observations.items()):
        summarise_observations(arrays, name, output)
    output["s2_valid_observation_count"] = np.sum(
        np.isfinite(np.stack(observations["s2_nir"], axis=1)), axis=1
    ).astype(np.int16)
    return output, inventory


def reduce_features(
    values: np.ndarray,
    valid: np.ndarray,
    offset_x: np.ndarray,
    offset_y: np.ndarray,
    rows: int,
) -> dict[str, np.ndarray]:
    features = values.shape[1]
    grid = values.reshape(rows, 25, features)
    mask = valid.reshape(rows, 25)
    x = offset_x.reshape(rows, 25)
    y = offset_y.reshape(rows, 25)
    count = mask.sum(axis=1).astype(np.int16)
    masked = np.where(mask[:, :, None], grid, 0.0)
    total = masked.sum(axis=1)
    mean = np.full((rows, features), np.nan, dtype=np.float32)
    np.divide(total, count[:, None], out=mean, where=count[:, None] > 0)
    variance = np.full_like(mean, np.nan)
    np.divide(
        (np.where(mask[:, :, None], (grid - mean[:, None, :]) ** 2, 0.0)).sum(
            axis=1
        ),
        count[:, None],
        out=variance,
        where=count[:, None] > 0,
    )
    standard_deviation = np.sqrt(np.maximum(variance, 0.0))

    def slope(offset: np.ndarray) -> np.ndarray:
        local_mean = np.divide(
            np.where(mask, offset, 0.0).sum(axis=1),
            count,
            out=np.zeros(rows, dtype=np.float32),
            where=count > 0,
        )
        centred = offset - local_mean[:, None]
        denominator = np.where(mask, centred**2, 0.0).sum(axis=1)
        numerator = np.where(
            mask[:, :, None], centred[:, :, None] * grid, 0.0
        ).sum(axis=1)
        result = np.full((rows, features), np.nan, dtype=np.float32)
        np.divide(
            numerator,
            denominator[:, None],
            out=result,
            where=denominator[:, None] > 0,
        )
        return result

    return {
        "count": count,
        "mean": mean,
        "std": standard_deviation,
        "dx": slope(x),
        "dy": slope(y),
    }


def run() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_conventional_predictors as conventional

    config = load_config()
    settings = config["predictors"]
    output_path = ROOT / str(config["outputs"]["conventional"])
    cohort_path = ROOT / str(config["outputs"]["cohort"])
    if output_path.exists() and MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if (
            manifest["cohort_sha256"] == sha256(cohort_path)
            and manifest["output_sha256"] == sha256(output_path)
        ):
            print("phase26 conventional: validated existing checkpoint", flush=True)
            return
        raise RuntimeError("The existing Phase 26 conventional checkpoint is stale")

    cohort = pd.read_parquet(cohort_path)
    points = context_points(cohort)
    to_wgs84 = Transformer.from_crs(
        config["study"]["analysis_crs"], "EPSG:4326", always_xy=True
    )
    longitude, latitude = to_wgs84.transform(points["rd_x"], points["rd_y"])
    longitude = np.asarray(longitude, dtype=np.float64)
    latitude = np.asarray(latitude, dtype=np.float64)
    bbox = [
        float(longitude.min()),
        float(latitude.min()),
        float(longitude.max()),
        float(latitude.max()),
    ]
    catalog = conventional.open_catalog(str(settings["stac_api"]))
    year = int(settings["conventional_year"])
    s2_candidates = conventional.collection_items(
        catalog, str(settings["sentinel_2"]["collection"]), bbox, year
    )
    s2_items = conventional.select_sentinel2(
        s2_candidates, settings["sentinel_2"]
    )
    s1_candidates = conventional.collection_items(
        catalog, str(settings["sentinel_1"]["collection"]), bbox, year
    )
    s1_items = conventional.select_sentinel1(
        s1_candidates, settings["sentinel_1"]
    )
    dem_items = conventional.collection_items(
        catalog, str(settings["topography"]["collection"]), bbox, None
    )
    print(
        f"phase26 conventional: selected {len(s2_items)} Sentinel-2, "
        f"{len(s1_items)} Sentinel-1 and {len(dem_items)} DEM items",
        flush=True,
    )
    s2, s2_inventory = extract_sentinel2(
        conventional,
        s2_items,
        longitude,
        latitude,
        settings["sentinel_2"],
    )
    s1, s1_inventory = conventional.extract_sentinel1(
        s1_items, longitude, latitude, settings["sentinel_1"]
    )
    terrain, terrain_inventory = conventional.extract_topography(
        dem_items,
        pd.DataFrame(
            {
                "x_epsg5070": points["rd_x"],
                "y_epsg5070": points["rd_y"],
            }
        ),
        bbox,
        settings["topography"],
    )
    pixel = pd.DataFrame({**terrain, **s2, **s1})
    feature_columns = sorted(
        column for column in pixel if "observation_count" not in column
    )
    values = pixel[feature_columns].to_numpy(dtype=np.float32)
    valid = points["broadleaf"].to_numpy(dtype=bool)
    valid &= pixel["s2_valid_observation_count"].to_numpy() >= int(
        settings["sentinel_2"]["minimum_valid_observations_per_point"]
    )
    s1_counts = sorted(
        column
        for column in pixel
        if column.startswith("s1_") and column.endswith("observation_count")
    )
    if not s1_counts:
        raise RuntimeError("No Sentinel-1 observation counts were produced")
    for column in s1_counts:
        valid &= pixel[column].to_numpy() >= int(
            settings["sentinel_1"]["minimum_valid_observations_per_point_per_state"]
        )
    valid &= np.isfinite(values).all(axis=1)
    summary = reduce_features(
        values,
        valid,
        points["offset_x"].to_numpy(dtype=np.float32),
        points["offset_y"].to_numpy(dtype=np.float32),
        len(cohort),
    )
    minimum = int(settings["minimum_valid_tessera_pixels"])
    context_valid = summary["count"] >= minimum
    output = pd.DataFrame(
        {
            "row_id": cohort["row_id"].to_numpy(dtype=np.int64),
            "conventional_valid_pixel_count": summary["count"],
            "conventional_context_valid": context_valid,
        }
    )
    for statistic in ["mean", "std", "dx", "dy"]:
        for column, values_column in zip(
            feature_columns, summary[statistic].T, strict=True
        ):
            output[f"conventional_{statistic}_{column}"] = values_column
    atomic_parquet(output, output_path)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cohort_sha256": sha256(cohort_path),
        "config_sha256": sha256(CONFIG_PATH),
        "year": year,
        "sentinel_2": s2_inventory,
        "sentinel_1": s1_inventory,
        "terrain": terrain_inventory,
        "feature_columns": feature_columns,
        "valid_rows": int(context_valid.sum()),
        "output_sha256": sha256(output_path),
    }
    atomic_json(MANIFEST_PATH, manifest)
    print(
        f"phase26 conventional: {context_valid.sum():,}/{len(context_valid):,} "
        "valid units",
        flush=True,
    )


if __name__ == "__main__":
    run()
