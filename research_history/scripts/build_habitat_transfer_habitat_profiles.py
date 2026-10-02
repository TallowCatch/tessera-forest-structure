#!/usr/bin/env python3
"""Build outcome-blind LANDFIRE habitat profiles for current and candidate NEON sites."""

from __future__ import annotations

import hashlib
import io
import json
import math
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import requests
import yaml
from pyproj import Transformer
from rasterio.mask import mask
from requests.adapters import HTTPAdapter
from scipy.spatial.distance import jensenshannon
from shapely.geometry import mapping
from urllib3.util.retry import Retry


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/project.yaml"
PHASE7_SCREEN_PATH = ROOT / "metadata/phase7_site_screen.csv"
FLIGHT_BOX_URL = "https://www.neonscience.org/sites/default/files/AOP_flightBoxes_0.zip"
IMAGE_SERVER = (
    "https://lfps.usgs.gov/arcgis/rest/services/Landfire_LF2024/"
    "LF2024_EVT_CONUS/ImageServer"
)
CROSSWALK_URL = "https://landfire.gov/sites/default/files/CSV/2024/LF2024_EVT.csv"
EXTERNAL_DIR = ROOT / "data/external/landfire_evt_2024"
CROSSWALK_PATH = EXTERNAL_DIR / "LF2024_EVT.csv"
BOUNDARY_PATH = ROOT / "metadata/phase9_neon_habitat_sites_2024.geojson"
PROFILE_PATH = ROOT / "metadata/phase9_landfire_habitat_site_profiles.csv"
SITE_SUMMARY_PATH = ROOT / "metadata/phase9_landfire_habitat_site_summary.csv"
DISTANCE_PATH = ROOT / "metadata/phase9_landfire_habitat_distances.csv"
PROSPECTIVE_PATH = ROOT / "metadata/phase9_habitat_prospective_design.csv"
POINT_PATH = ROOT / "data/processed/phase9_landfire_evt_footprints.parquet"
FIGURE_PATH = ROOT / "outputs/figures/phase9_landfire_habitat_profiles.png"
PROTOCOL_PATH = ROOT / "metadata/project_config_phase9_habitat_transfer_protocol_freeze.yaml"
FREEZE_PATH = ROOT / "metadata/phase9_landfire_habitat_freeze.json"

POPULATION_PATHS = [
    ROOT / "data/processed/tessera_aligned_development_soap_teak.parquet",
    ROOT / "data/processed/tessera_aligned_locked_bart.parquet",
    ROOT / "data/processed/phase7_tessera_aligned_expansion.parquet",
]
KEYS = ["site_id", "shot_number"]
HIERARCHY_COLUMNS = {
    "evt_phys": "evt_phys",
    "evt_group": "evt_gp_n",
    "evt_system": "evt_name",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def make_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=4,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.headers.update({"User-Agent": "tessera-gedi-fhd/0.1 phase9-habitat-profile"})
    return session


def download_file(session: requests.Session, url: str, path: Path, timeout: int = 180) -> bytes:
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    content = response.content
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)
    return content


def export_evt_raster(
    session: requests.Session,
    site_id: str,
    geometry: Any,
    output_path: Path,
) -> None:
    projected = gpd.GeoSeries([geometry], crs=4326).to_crs(5070).iloc[0]
    minx, miny, maxx, maxy = projected.bounds
    width = max(1, math.ceil((maxx - minx) / 30.0))
    height = max(1, math.ceil((maxy - miny) / 30.0))
    response = session.get(
        f"{IMAGE_SERVER}/exportImage",
        params={
            "bbox": f"{minx},{miny},{maxx},{maxy}",
            "bboxSR": 5070,
            "imageSR": 5070,
            "size": f"{width},{height}",
            "format": "tiff",
            "pixelType": "S16",
            "interpolation": "RSP_NearestNeighbor",
            "f": "image",
        },
        timeout=240,
    )
    response.raise_for_status()
    if "tiff" not in response.headers.get("content-type", "").lower():
        raise RuntimeError(f"LANDFIRE export for {site_id} was not a TIFF")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.tif")
    temporary.write_bytes(response.content)
    with rasterio.open(temporary) as dataset:
        if dataset.crs.to_epsg() != 5070 or dataset.count != 1:
            raise RuntimeError(f"Unexpected LANDFIRE raster contract for {site_id}")
    temporary.replace(output_path)


def habitat_profile_rows(
    site_id: str,
    values: np.ndarray,
    crosswalk: pd.DataFrame,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    counts = pd.Series(values.astype(np.int64)).value_counts().rename_axis("value").reset_index(name="pixel_count")
    decoded = counts.merge(crosswalk, on="value", how="left", validate="many_to_one")
    if decoded["evt_name"].isna().any():
        missing = decoded.loc[decoded["evt_name"].isna(), "value"].tolist()
        raise RuntimeError(f"LANDFIRE codes missing from the 2024 crosswalk at {site_id}: {missing}")
    tree = decoded[decoded["evt_lf"].eq("Tree")].copy()
    tree_pixels = int(tree["pixel_count"].sum())
    if tree_pixels == 0:
        raise RuntimeError(f"No LANDFIRE tree pixels found at {site_id}")
    rows: list[dict[str, Any]] = []
    for hierarchy, column in HIERARCHY_COLUMNS.items():
        grouped = tree.groupby(column, as_index=False, dropna=False)["pixel_count"].sum()
        for row in grouped.itertuples(index=False):
            rows.append({
                "site_id": site_id,
                "hierarchy": hierarchy,
                "habitat_class": str(getattr(row, column)),
                "pixel_count": int(row.pixel_count),
                "proportion": float(row.pixel_count / tree_pixels),
            })
    summary = {
        "site_id": site_id,
        "flight_box_pixels": int(len(values)),
        "tree_pixels": tree_pixels,
        "tree_pixel_fraction": float(tree_pixels / len(values)),
        "unique_evt_codes": int(len(decoded)),
        "unique_tree_evt_codes": int(len(tree)),
    }
    return rows, summary


def build_distances(profiles: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for hierarchy, frame in profiles.groupby("hierarchy", sort=True):
        matrix = frame.pivot_table(
            index="site_id", columns="habitat_class", values="proportion", fill_value=0.0
        ).sort_index()
        for target_site in matrix.index:
            distances = []
            for source_site in matrix.index:
                if source_site == target_site:
                    continue
                distance = float(jensenshannon(
                    matrix.loc[target_site].to_numpy(dtype=np.float64),
                    matrix.loc[source_site].to_numpy(dtype=np.float64),
                    base=2,
                ))
                distances.append((source_site, distance))
            for rank, (source_site, distance) in enumerate(
                sorted(distances, key=lambda item: (item[1], item[0])), start=1
            ):
                records.append({
                    "hierarchy": hierarchy,
                    "target_site": target_site,
                    "source_site": source_site,
                    "distance": distance,
                    "source_rank": rank,
                })
    return pd.DataFrame(records)


def nearest_sources(
    distances: pd.DataFrame,
    target_site: str,
    source_pool: list[str],
    hierarchy: str,
    count: int,
) -> list[str]:
    scoped = distances[
        distances["hierarchy"].eq(hierarchy)
        & distances["target_site"].eq(target_site)
        & distances["source_site"].isin(source_pool)
    ].sort_values(["distance", "source_site"])
    if len(scoped) < count:
        raise RuntimeError(f"Only {len(scoped)} sources available for {target_site}/{hierarchy}")
    return scoped.head(count)["source_site"].tolist()


def save_figure(profiles: pd.DataFrame, distances: pd.DataFrame, current_sites: list[str]) -> None:
    primary = profiles[profiles["hierarchy"].eq("evt_phys")]
    totals = primary.groupby("habitat_class")["pixel_count"].sum().sort_values(ascending=False)
    shown = totals.head(7).index.tolist()
    composition = primary.copy()
    composition["plot_class"] = np.where(
        composition["habitat_class"].isin(shown), composition["habitat_class"], "Other"
    )
    composition = composition.groupby(["site_id", "plot_class"], as_index=False)["proportion"].sum()
    pivot = composition.pivot(index="site_id", columns="plot_class", values="proportion").fillna(0)
    pivot = pivot.reindex(current_sites)

    distance_frame = distances[
        distances["hierarchy"].eq("evt_phys")
        & distances["target_site"].isin(current_sites)
        & distances["source_site"].isin(current_sites)
    ].pivot(index="target_site", columns="source_site", values="distance").reindex(
        index=current_sites, columns=current_sites
    ).copy()
    distance_values = distance_frame.to_numpy(copy=True)
    np.fill_diagonal(distance_values, 0.0)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.7), gridspec_kw={"width_ratios": [1.25, 1]})
    bottom = np.zeros(len(pivot))
    colours = plt.get_cmap("tab10")(np.linspace(0, 1, len(pivot.columns)))
    for colour, column in zip(colours, pivot.columns, strict=True):
        values = pivot[column].to_numpy()
        axes[0].bar(pivot.index, values, bottom=bottom, label=column, color=colour)
        bottom += values
    axes[0].set_ylabel("Proportion of LANDFIRE tree pixels")
    axes[0].set_xlabel("NEON site")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8)
    image = axes[1].imshow(distance_values, cmap="viridis", vmin=0, vmax=1)
    axes[1].set_xticks(range(len(current_sites)), current_sites, rotation=45, ha="right")
    axes[1].set_yticks(range(len(current_sites)), current_sites)
    axes[1].set_title("Physiognomy Jensen-Shannon distance")
    figure.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
    figure.suptitle("LANDFIRE 2024 habitat profiles for the frozen eight-site cohort")
    figure.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = FIGURE_PATH.with_suffix(".tmp.png")
    figure.savefig(temporary, dpi=180, bbox_inches="tight")
    plt.close(figure)
    temporary.replace(FIGURE_PATH)


def main() -> int:
    protected = [
        BOUNDARY_PATH, PROFILE_PATH, SITE_SUMMARY_PATH, DISTANCE_PATH, PROSPECTIVE_PATH,
        POINT_PATH, FIGURE_PATH, PROTOCOL_PATH, FREEZE_PATH,
    ]
    existing = [str(path.relative_to(ROOT)) for path in protected if path.exists()]
    if FREEZE_PATH.exists():
        raise RuntimeError(f"Phase 9 habitat freeze exists; refusing to overwrite: {existing}")
    if existing:
        print(f"Resuming an incomplete Phase 9 habitat build: {existing}")

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    protocol = config["phase9_habitat_conditioned_transfer"]
    phase7 = config["phase7_site_expansion"]
    current_sites = sorted(protocol["current_sites"])
    all_sites = sorted(set(current_sites) | set(phase7["candidate_sites"]))
    if protocol["similarity"]["target_outcomes_used"]:
        raise RuntimeError("Habitat similarity must remain target-outcome blind")

    session = make_session()
    response = session.get(FLIGHT_BOX_URL, timeout=120)
    response.raise_for_status()
    boundary_zip = response.content
    with tempfile.TemporaryDirectory() as temporary:
        with zipfile.ZipFile(io.BytesIO(boundary_zip)) as archive:
            archive.extractall(temporary)
        shapefile = next(Path(temporary).rglob("AOP_flightboxesAllSites.shp"))
        boundaries = gpd.read_file(shapefile).to_crs(4326)
    boundaries = boundaries[
        boundaries["siteID"].isin(all_sites)
        & boundaries["priority"].eq(1)
        & boundaries["sampleType"].eq("Terrestrial")
    ].dissolve(by="siteID", aggfunc="first").reset_index()
    if set(boundaries["siteID"]) != set(all_sites):
        raise RuntimeError("The official NEON boundary set is incomplete")
    boundaries = boundaries.sort_values("siteID").reset_index(drop=True)
    write_text(BOUNDARY_PATH, boundaries.to_json(drop_id=True) + "\n")

    crosswalk_bytes = download_file(session, CROSSWALK_URL, CROSSWALK_PATH)
    crosswalk = pd.read_csv(CROSSWALK_PATH)
    crosswalk.columns = [column.lower() for column in crosswalk.columns]
    required_columns = {"value", "evt_name", "evt_lf", "evt_phys", "evt_gp_n"}
    if not required_columns <= set(crosswalk.columns) or crosswalk["value"].duplicated().any():
        raise RuntimeError("Unexpected LANDFIRE 2024 crosswalk schema")

    profile_records: list[dict[str, Any]] = []
    summary_records: list[dict[str, Any]] = []
    raster_paths: dict[str, Path] = {}
    for row in boundaries.itertuples():
        raster_path = EXTERNAL_DIR / f"LF2024_EVT_{row.siteID}.tif"
        if not raster_path.exists():
            export_evt_raster(session, row.siteID, row.geometry, raster_path)
        raster_paths[row.siteID] = raster_path
        projected = gpd.GeoSeries([row.geometry], crs=4326).to_crs(5070).iloc[0]
        with rasterio.open(raster_path) as dataset:
            values, _ = mask(dataset, [mapping(projected)], crop=True, filled=False)
        pixels = values[0].compressed()
        rows, summary = habitat_profile_rows(row.siteID, pixels, crosswalk)
        profile_records.extend(rows)
        summary_records.append(summary)

    profiles = pd.DataFrame(profile_records).sort_values(
        ["hierarchy", "site_id", "habitat_class"]
    ).reset_index(drop=True)
    summaries = pd.DataFrame(summary_records).sort_values("site_id").reset_index(drop=True)
    distances = build_distances(profiles).sort_values(
        ["hierarchy", "target_site", "source_rank"]
    ).reset_index(drop=True)

    population = pd.concat([
        pd.read_parquet(path, columns=KEYS + ["longitude", "latitude"])
        for path in POPULATION_PATHS
    ], ignore_index=True)
    if sorted(population["site_id"].unique()) != current_sites or population.duplicated(KEYS).any():
        raise RuntimeError("Unexpected eight-site footprint population")
    point_frames: list[pd.DataFrame] = []
    transformer = Transformer.from_crs(4326, 5070, always_xy=True)
    for site_id, frame in population.groupby("site_id", sort=True):
        x, y = transformer.transform(frame["longitude"].to_numpy(), frame["latitude"].to_numpy())
        with rasterio.open(raster_paths[site_id]) as dataset:
            sampled = np.asarray([value[0] for value in dataset.sample(zip(x, y, strict=True))], dtype=np.int64)
        decoded = pd.DataFrame({"value": sampled}).merge(
            crosswalk[list(required_columns)], on="value", how="left", validate="many_to_one"
        )
        if decoded["evt_name"].isna().any():
            raise RuntimeError(f"One or more {site_id} footprints could not be decoded")
        output = frame[KEYS + ["longitude", "latitude"]].reset_index(drop=True).copy()
        output["landfire_evt_value"] = decoded["value"]
        output["landfire_evt_name"] = decoded["evt_name"]
        output["landfire_evt_lifeform"] = decoded["evt_lf"]
        output["landfire_evt_phys"] = decoded["evt_phys"]
        output["landfire_evt_group"] = decoded["evt_gp_n"]
        point_frames.append(output)
    points = pd.concat(point_frames, ignore_index=True).sort_values(KEYS).reset_index(drop=True)

    screen = pd.read_csv(PHASE7_SCREEN_PATH)
    evidence = phase7["selection_evidence"]
    prospective = screen[
        ~screen["selected"].astype(bool)
        & screen["cmr_2024_gedi_v3_granules"].ge(
            evidence["minimum_2024_gedi_v3_intersecting_granules"]
        )
        & screen["nlcd_2024_forest_fraction"].ge(
            evidence["minimum_annual_nlcd_2024_forest_fraction_in_flight_box"]
        )
        & screen["tessera_inventory_complete"].astype(bool)
    ].copy()
    prospective_records = []
    for row in prospective.sort_values("site_id").itertuples():
        selected_sources = nearest_sources(distances, row.site_id, current_sites, "evt_phys", 5)
        scoped = distances[
            distances["hierarchy"].eq("evt_phys")
            & distances["target_site"].eq(row.site_id)
            & distances["source_site"].isin(selected_sources)
        ]
        prospective_records.append({
            "target_site": row.site_id,
            "target_outcome_opened": False,
            "nearest_current_site": selected_sources[0],
            "nearest5_current_sites": "+".join(selected_sources),
            "nearest5_mean_distance": float(scoped["distance"].mean()),
            "cmr_2024_gedi_v3_granules": int(row.cmr_2024_gedi_v3_granules),
            "nlcd_2024_forest_fraction": float(row.nlcd_2024_forest_fraction),
            "required_tessera_tiles": int(row.required_tessera_tiles),
        })
    prospective_design = pd.DataFrame(prospective_records)

    write_text(PROFILE_PATH, profiles.to_csv(index=False))
    write_text(SITE_SUMMARY_PATH, summaries.to_csv(index=False))
    write_text(DISTANCE_PATH, distances.to_csv(index=False))
    write_text(PROSPECTIVE_PATH, prospective_design.to_csv(index=False))
    write_parquet(points, POINT_PATH)
    save_figure(profiles, distances, current_sites)
    write_text(PROTOCOL_PATH, yaml.safe_dump(
        {"phase9_habitat_conditioned_transfer": protocol}, sort_keys=False
    ))

    raster_manifest = {
        site: {
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for site, path in sorted(raster_paths.items())
    }
    outputs = {
        "boundaries": {"path": str(BOUNDARY_PATH.relative_to(ROOT)), "sha256": sha256(BOUNDARY_PATH)},
        "profiles": {"path": str(PROFILE_PATH.relative_to(ROOT)), "rows": len(profiles), "sha256": sha256(PROFILE_PATH)},
        "site_summary": {"path": str(SITE_SUMMARY_PATH.relative_to(ROOT)), "rows": len(summaries), "sha256": sha256(SITE_SUMMARY_PATH)},
        "distances": {"path": str(DISTANCE_PATH.relative_to(ROOT)), "rows": len(distances), "sha256": sha256(DISTANCE_PATH)},
        "prospective_design": {"path": str(PROSPECTIVE_PATH.relative_to(ROOT)), "rows": len(prospective_design), "sha256": sha256(PROSPECTIVE_PATH)},
        "footprint_habitats": {"path": str(POINT_PATH.relative_to(ROOT)), "rows": len(points), "sha256": sha256(POINT_PATH)},
        "figure": {"path": str(FIGURE_PATH.relative_to(ROOT)), "sha256": sha256(FIGURE_PATH)},
        "protocol": {"path": str(PROTOCOL_PATH.relative_to(ROOT)), "sha256": sha256(PROTOCOL_PATH)},
    }
    freeze_basis = {
        "timing": protocol["timing"],
        "fhd_columns_read": [],
        "target_outcomes_used_for_habitat_profiles_or_similarity": False,
        "current_sites": current_sites,
        "all_profiled_sites": all_sites,
        "prospective_target_sites": prospective_design["target_site"].tolist(),
        "habitat_product": protocol["habitat_source"],
        "similarity": protocol["similarity"],
        "source_hashes": {
            "flight_box_zip_sha256": sha256_bytes(boundary_zip),
            "landfire_crosswalk_sha256": sha256_bytes(crosswalk_bytes),
            "phase7_screen_sha256": sha256(PHASE7_SCREEN_PATH),
        },
        "raster_manifest": raster_manifest,
    }
    freeze_id = "phase9-habitat-" + canonical_hash({"freeze_basis": freeze_basis, "outputs": outputs})[:12]
    write_text(FREEZE_PATH, json.dumps({
        "freeze_id": freeze_id,
        "created_utc": utc_now(),
        "freeze_basis": freeze_basis,
        "outputs": outputs,
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "freeze_id": freeze_id,
        "profiled_sites": len(all_sites),
        "current_footprints": len(points),
        "prospective_targets": prospective_design["target_site"].tolist(),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
