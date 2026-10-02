#!/usr/bin/env python3
"""Map a complete BART tile and test locally calibrated predictions away from GEDI tracks."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import requests
from pyproj import Transformer
from rasterio.transform import from_origin, rowcol
from rasterio.warp import Resampling, reproject
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import process_neon_lidar_tile as lidar  # noqa: E402
import run_multisite_loso as loso  # noqa: E402


BART_PATH = ROOT / "data/processed/tessera_aligned_locked_bart.parquet"
MANIFEST_PATH = ROOT / "metadata/phase5_neon_lidar_manifest.csv"
CALIBRATION_PATH = ROOT / "metadata/phase5_neon_lidar_calibration_freeze.json"
NLCD_PATH = ROOT / "data/external/nlcd/Annual_NLCD_Land_Cover_2024_BART.tif"
TESSERA_ROOT = ROOT / "data/external/geotessera_embeddings/global_0.1_degree_representation/2024"
LANDMASK_ROOT = ROOT / "data/external/geotessera_embeddings/global_0.1_degree_tiff_all"
OFFTRACK_PATH = ROOT / "data/processed/robustness_bart_offtrack_als.parquet"
MAP_PATH = ROOT / "outputs/maps/robustness_bart_local_fhd.tif"
SUPPORT_PATH = ROOT / "outputs/maps/robustness_bart_area_of_applicability.tif"
METRICS_PATH = ROOT / "outputs/tables/robustness_bart_continuous_validation.csv"
SUPPORT_SUMMARY_PATH = ROOT / "outputs/tables/robustness_bart_applicability.csv"
FIGURE_PATH = ROOT / "outputs/figures/robustness_bart_continuous_validation.png"
FREEZE_PATH = ROOT / "metadata/robustness_bart_continuous_freeze.json"
WORK_ROOT = ROOT / "data/interim/robustness_bart_offtrack_work"

FEATURES = [f"tessera_center_{index:03d}" for index in range(128)]
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
TILE_SELECTION_ORDER = 3
GRID_SPACING_M = 100.0
MINIMUM_GEDI_DISTANCE_M = 150.0
MAP_RESOLUTION_M = 10.0


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def load_tessera_tile(tile_name: str):
    directory = TESSERA_ROOT / tile_name
    quantized = np.load(directory / f"{tile_name}.npy", mmap_mode="r")
    scales = np.load(directory / f"{tile_name}_scales.npy", mmap_mode="r")
    with rasterio.open(LANDMASK_ROOT / f"{tile_name}.tiff") as dataset:
        transform = dataset.transform
        crs = dataset.crs
    return quantized, scales, transform, crs


def embeddings_at_xy(
    x: np.ndarray,
    y: np.ndarray,
    quantized: np.ndarray,
    scales: np.ndarray,
    transform,
) -> np.ndarray:
    rows, columns = rowcol(transform, x, y)
    rows = np.asarray(rows)
    columns = np.asarray(columns)
    if (
        (rows < 0).any() or (columns < 0).any()
        or (rows >= quantized.shape[0]).any() or (columns >= quantized.shape[1]).any()
    ):
        raise RuntimeError("Requested map coordinates fall outside the TESSERA tile")
    return quantized[rows, columns].astype(np.float32) * scales[rows, columns, None]


def nlcd_on_grid(transform, crs, width: int, height: int) -> np.ndarray:
    destination = np.zeros((height, width), dtype=np.uint8)
    with rasterio.open(NLCD_PATH) as source:
        reproject(
            source=rasterio.band(source, 1),
            destination=destination,
            src_transform=source.transform,
            src_crs=source.crs,
            dst_transform=transform,
            dst_crs=crs,
            resampling=Resampling.nearest,
        )
    return destination


def tune_local_alpha(training: pd.DataFrame) -> float:
    records = []
    granules = sorted(training["granule_id"].unique())
    for alpha in ALPHAS:
        for granule in granules:
            fit = training[~training["granule_id"].eq(granule)]
            validation = training[training["granule_id"].eq(granule)]
            scaler = StandardScaler().fit(fit[FEATURES].to_numpy(dtype=np.float64))
            model = Ridge(alpha=alpha).fit(
                scaler.transform(fit[FEATURES].to_numpy(dtype=np.float64)),
                fit["fhd_normal"].to_numpy(),
            )
            prediction = model.predict(
                scaler.transform(validation[FEATURES].to_numpy(dtype=np.float64))
            )
            records.append({
                "alpha": alpha,
                "granule": granule,
                "rmse": loso.metric_values(validation["fhd_normal"].to_numpy(), prediction)["rmse"],
            })
    ranking = (
        pd.DataFrame(records).groupby("alpha", as_index=False)
        .agg(worst_rmse=("rmse", "max"), mean_rmse=("rmse", "mean"))
        .sort_values(["worst_rmse", "mean_rmse", "alpha"])
    )
    return float(ranking.iloc[0]["alpha"])


def build_offtrack_centres(
    bart: pd.DataFrame,
    easting: int,
    northing: int,
    nlcd: np.ndarray,
    map_transform,
) -> pd.DataFrame:
    coordinates = np.array([
        (x, y)
        for y in np.arange(northing + GRID_SPACING_M / 2, northing + 1000, GRID_SPACING_M)
        for x in np.arange(easting + GRID_SPACING_M / 2, easting + 1000, GRID_SPACING_M)
    ])
    tree = cKDTree(bart[["utm_x", "utm_y"]].to_numpy())
    distance, _ = tree.query(coordinates)
    rows, columns = rowcol(map_transform, coordinates[:, 0], coordinates[:, 1])
    forest = np.isin(nlcd[np.asarray(rows), np.asarray(columns)], [41, 42, 43])
    selected = coordinates[(distance >= MINIMUM_GEDI_DISTANCE_M) & forest]
    frame = pd.DataFrame(selected, columns=["utm_x", "utm_y"])
    frame["shot_number"] = np.arange(9_000_000_000, 9_000_000_000 + len(frame), dtype=np.int64)
    frame["beam_name"] = "BEAM0110"
    frame["nearest_gedi_distance_m"] = cKDTree(
        bart[["utm_x", "utm_y"]].to_numpy()
    ).query(selected)[0]
    return frame


def acquire_offtrack_als(centres: pd.DataFrame, tile: pd.Series) -> pd.DataFrame:
    if OFFTRACK_PATH.exists():
        return pd.read_parquet(OFFTRACK_PATH)
    token = os.getenv("NEON_TOKEN") or os.getenv("NEON_PAT")
    if not token:
        raise RuntimeError("NEON_TOKEN is required to acquire the off-track lidar tile")
    required = int(tile.size_bytes) + 2_500_000_000
    if shutil.disk_usage(ROOT).free < required:
        raise RuntimeError("Insufficient temporary storage for the off-track lidar validation")
    source_path = WORK_ROOT / str(tile.file_name)
    try:
        session = requests.Session()
        session.headers.update({"User-Agent": "tessera-gedi-fhd/robustness"})
        lidar.download_source(session, token, str(tile.url), source_path)
        if source_path.stat().st_size != int(tile.size_bytes):
            raise RuntimeError("Downloaded NEON lidar tile has an unexpected size")
        extracted = lidar.extract_footprints(source_path, centres, WORK_ROOT / "footprints")
        simulations = []
        for index, row in enumerate(extracted.itertuples(index=False), 1):
            simulations.append(lidar.simulate_footprint(row, WORK_ROOT))
            print(f"off-track ALS simulation {index}/{len(extracted)}", flush=True)
        result = pd.concat([
            extracted.drop(columns=["subset_las_path"]).reset_index(drop=True),
            pd.DataFrame(simulations),
        ], axis=1)
        OFFTRACK_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary = OFFTRACK_PATH.with_suffix(".tmp.parquet")
        result.to_parquet(temporary, index=False, compression="zstd")
        temporary.replace(OFFTRACK_PATH)
        return result
    finally:
        if WORK_ROOT.exists():
            shutil.rmtree(WORK_ROOT)


def write_raster(values: np.ndarray, path: Path, transform, crs, nodata: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=values.shape[0], width=values.shape[1],
        count=1, dtype="float32", crs=crs, transform=transform,
        nodata=nodata, compress="deflate", tiled=True,
    ) as dataset:
        dataset.write(values.astype(np.float32), 1)


def main() -> int:
    manifest = pd.read_csv(MANIFEST_PATH)
    tile = manifest[
        manifest["site_id"].eq("BART")
        & manifest["selection_order"].eq(TILE_SELECTION_ORDER)
    ].iloc[0]
    easting = int(tile.tile_easting)
    northing = int(tile.tile_northing)
    tile_name = "grid_-71.25_44.05"
    quantized, scales, tessera_transform, crs = load_tessera_tile(tile_name)

    bart = pd.read_parquet(BART_PATH)
    transformer = Transformer.from_crs(4326, crs, always_xy=True)
    bart["utm_x"], bart["utm_y"] = transformer.transform(
        bart["longitude"].to_numpy(), bart["latitude"].to_numpy()
    )
    in_validation_tile = (
        bart["utm_x"].between(easting, easting + 1000)
        & bart["utm_y"].between(northing, northing + 1000)
    )
    training = bart[~in_validation_tile].copy()
    heldout_gedi = bart[in_validation_tile].copy()
    alpha = tune_local_alpha(training)
    scaler = StandardScaler().fit(training[FEATURES].to_numpy(dtype=np.float64))
    model = Ridge(alpha=alpha).fit(
        scaler.transform(training[FEATURES].to_numpy(dtype=np.float64)),
        training["fhd_normal"].to_numpy(),
    )

    map_transform = from_origin(easting, northing + 1000, MAP_RESOLUTION_M, MAP_RESOLUTION_M)
    width = height = int(1000 / MAP_RESOLUTION_M)
    columns, rows = np.meshgrid(np.arange(width), np.arange(height))
    x = easting + (columns.ravel() + 0.5) * MAP_RESOLUTION_M
    y = northing + 1000 - (rows.ravel() + 0.5) * MAP_RESOLUTION_M
    map_embeddings = embeddings_at_xy(x, y, quantized, scales, tessera_transform)
    nlcd = nlcd_on_grid(map_transform, crs, width, height)
    forest = np.isin(nlcd.ravel(), [41, 42, 43])
    prediction = np.full(len(x), -9999.0, dtype=np.float32)
    prediction[forest] = model.predict(scaler.transform(map_embeddings[forest]))
    prediction_grid = prediction.reshape(height, width)
    write_raster(prediction_grid, MAP_PATH, map_transform, crs, -9999.0)

    # Area of applicability is based on distance to BART training embeddings in a
    # source-fitted 20-dimensional PCA space.
    train_scaled = scaler.transform(training[FEATURES].to_numpy(dtype=np.float64))
    pca = PCA(n_components=20, random_state=0).fit(train_scaled)
    train_scores = pca.transform(train_scaled)
    reference_distances = []
    for granule in sorted(training["granule_id"].unique()):
        fit_mask = ~training["granule_id"].eq(granule).to_numpy()
        test_mask = ~fit_mask
        neighbours = NearestNeighbors(n_neighbors=1).fit(train_scores[fit_mask])
        reference_distances.extend(neighbours.kneighbors(train_scores[test_mask])[0][:, 0])
    threshold = float(np.quantile(reference_distances, 0.95))
    neighbours = NearestNeighbors(n_neighbors=1).fit(train_scores)
    map_scores = pca.transform(scaler.transform(map_embeddings[forest]))
    map_distance = neighbours.kneighbors(map_scores)[0][:, 0]
    support = np.full(len(x), -9999.0, dtype=np.float32)
    support[forest] = map_distance
    support_grid = support.reshape(height, width)
    write_raster(support_grid, SUPPORT_PATH, map_transform, crs, -9999.0)

    centres = build_offtrack_centres(bart, easting, northing, nlcd, map_transform)
    offtrack = acquire_offtrack_als(centres, tile)
    offtrack_embeddings = embeddings_at_xy(
        offtrack["utm_x"].to_numpy(), offtrack["utm_y"].to_numpy(),
        quantized, scales, tessera_transform,
    )
    offtrack["local_tessera_prediction"] = model.predict(scaler.transform(offtrack_embeddings))
    calibration = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))[
        "final_development_coefficients"
    ]
    offtrack["converted_als_fhd"] = (
        float(calibration["intercept"])
        + float(calibration["slope"]) * offtrack["simulated_fhd"]
    )
    offtrack_scores = pca.transform(scaler.transform(offtrack_embeddings))
    offtrack["applicability_distance"] = neighbours.kneighbors(offtrack_scores)[0][:, 0]
    offtrack["inside_source_support"] = offtrack["applicability_distance"].le(threshold)
    temporary = OFFTRACK_PATH.with_suffix(".tmp.parquet")
    offtrack.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(OFFTRACK_PATH)

    heldout_prediction = model.predict(
        scaler.transform(heldout_gedi[FEATURES].to_numpy(dtype=np.float64))
    )
    metrics = pd.DataFrame([
        {
            "comparison": "local_model_vs_heldout_GEDI_in_complete_tile",
            "rows": len(heldout_gedi),
            **loso.metric_values(heldout_gedi["fhd_normal"].to_numpy(), heldout_prediction),
        },
        {
            "comparison": "local_model_vs_offtrack_converted_airborne_FHD",
            "rows": len(offtrack),
            **loso.metric_values(
                offtrack["converted_als_fhd"].to_numpy(),
                offtrack["local_tessera_prediction"].to_numpy(),
            ),
        },
    ])
    write_csv(metrics, METRICS_PATH)
    support_summary = pd.DataFrame([
        {
            "population": "complete_tile_forest_pixels",
            "rows": int(forest.sum()),
            "support_threshold": threshold,
            "fraction_inside_source_support": float(np.mean(map_distance <= threshold)),
            "median_applicability_distance": float(np.median(map_distance)),
        },
        {
            "population": "offtrack_airborne_lidar_points",
            "rows": len(offtrack),
            "support_threshold": threshold,
            "fraction_inside_source_support": float(offtrack["inside_source_support"].mean()),
            "median_applicability_distance": float(offtrack["applicability_distance"].median()),
        },
    ])
    write_csv(support_summary, SUPPORT_SUMMARY_PATH)

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    masked_prediction = np.ma.masked_equal(prediction_grid, -9999.0)
    axes[0].imshow(masked_prediction, extent=[easting, easting + 1000, northing, northing + 1000], cmap="viridis", origin="upper")
    axes[0].set_title("Locally calibrated FHD map")
    axes[0].set_xlabel("UTM easting (m)")
    axes[0].set_ylabel("UTM northing (m)")
    masked_support = np.ma.masked_equal(support_grid, -9999.0)
    axes[1].imshow(masked_support, extent=[easting, easting + 1000, northing, northing + 1000], cmap="magma", origin="upper")
    axes[1].contour(
        support_grid <= threshold, levels=[0.5], extent=[easting, easting + 1000, northing, northing + 1000],
        colors="white", linewidths=0.8,
    )
    axes[1].set_title("Distance from source support")
    axes[1].set_xlabel("UTM easting (m)")
    axes[2].scatter(
        offtrack["converted_als_fhd"], offtrack["local_tessera_prediction"],
        c=offtrack["applicability_distance"], cmap="magma", s=25, alpha=0.8,
    )
    low = min(offtrack["converted_als_fhd"].min(), offtrack["local_tessera_prediction"].min())
    high = max(offtrack["converted_als_fhd"].max(), offtrack["local_tessera_prediction"].max())
    axes[2].plot([low, high], [low, high], "k--", linewidth=1)
    axes[2].set_xlabel("Converted airborne-lidar FHD")
    axes[2].set_ylabel("Local TESSERA prediction")
    axes[2].set_title("Off-track validation")
    figure.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=220, bbox_inches="tight")
    plt.close(figure)

    freeze = {
        "freeze_id": "retrospective-bart-continuous-" + hashlib.sha256(
            metrics.to_csv(index=False).encode("utf-8")
        ).hexdigest()[:12],
        "created_utc": utc_now(),
        "status": "retrospective_local_calibration_case_study",
        "training_population": "BART_GEDI_outside_complete_1km_validation_tile",
        "validation_tile": f"{easting}_{northing}",
        "offtrack_minimum_distance_from_GEDI_m": MINIMUM_GEDI_DISTANCE_M,
        "selected_alpha": alpha,
        "airborne_conversion_warning": "SOAP_TEAK_conversion_has_weak_BART_R2_and_limits_absolute_validation",
        "outputs": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [OFFTRACK_PATH, MAP_PATH, SUPPORT_PATH, METRICS_PATH,
                         SUPPORT_SUMMARY_PATH, FIGURE_PATH]
        },
    }
    FREEZE_PATH.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n")
    print(metrics.to_string(index=False))
    print(support_summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
