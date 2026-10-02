#!/usr/bin/env python3
"""Build manuscript-specific figures from frozen project outputs."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from urllib.parse import urlencode

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
import pandas as pd
import rasterio
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.transform import array_bounds, from_bounds
from rasterio.warp import calculate_default_transform, reproject
from shapely.geometry import mapping


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "paper" / "figures"
TMP_DIR = ROOT / "tmp" / "paper"

COLORS = {
    "development": "#087E8B",
    "heldout": "#D95F02",
    "prospective": "#2C7FB8",
    "habitat": "#7B2CBF",
    "replication": "#E9C46A",
    "excluded": "#C62828",
    "mean": "#6C757D",
    "sentinel": "#3B82F6",
    "tessera": "#129490",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )


def _web_mercator_bbox(
    west: float, south: float, east: float, north: float
) -> tuple[float, float, float, float]:
    radius = 6378137.0
    return (
        radius * np.deg2rad(west),
        radius * np.log(np.tan(np.pi / 4 + np.deg2rad(south) / 2)),
        radius * np.deg2rad(east),
        radius * np.log(np.tan(np.pi / 4 + np.deg2rad(north) / 2)),
    )


CONUS_BBOX_3857 = _web_mercator_bbox(-128.0, 24.0, -66.0, 50.5)
CONUS_CRS = "EPSG:5070"


def _download_state_boundaries() -> gpd.GeoDataFrame:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    path = TMP_DIR / "cb_2024_us_state_5m.zip"
    if not path.exists():
        url = "https://www2.census.gov/geo/tiger/GENZ2024/shp/cb_2024_us_state_5m.zip"
        urllib.request.urlretrieve(url, path)
    states = gpd.read_file(f"zip://{path}")
    non_conterminous = {"AK", "HI", "PR", "GU", "VI", "MP", "AS"}
    return states.loc[~states["STUSPS"].isin(non_conterminous)].to_crs(CONUS_CRS)


def _download_sentinel_context() -> Path:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    path = TMP_DIR / "eox_s2cloudless_2024_conus_wide_1900x1016.jpg"
    if path.exists():
        return path
    params = {
        "SERVICE": "WMS",
        "REQUEST": "GetMap",
        "VERSION": "1.1.1",
        "LAYERS": "s2cloudless-2024_3857",
        "STYLES": "",
        "SRS": "EPSG:3857",
        "BBOX": ",".join(str(value) for value in CONUS_BBOX_3857),
        "WIDTH": "1900",
        "HEIGHT": "1016",
        "FORMAT": "image/jpeg",
    }
    url = f"https://tiles.maps.eox.at/map?{urlencode(params)}"
    urllib.request.urlretrieve(url, path)
    return path


def _project_sentinel_cutout(
    image: np.ndarray, conus: object
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Reproject the continental mosaic and mask it to the US land geometry."""
    height, width = image.shape[:2]
    src_transform = from_bounds(*CONUS_BBOX_3857, width, height)
    dst_transform, dst_width, dst_height = calculate_default_transform(
        "EPSG:3857",
        CONUS_CRS,
        width,
        height,
        *CONUS_BBOX_3857,
    )
    source = image[..., :3].astype(np.float32)
    if source.max() > 1.0:
        source /= 255.0
    projected = np.zeros((dst_height, dst_width, 3), dtype=np.float32)
    for band in range(3):
        reproject(
            source=source[..., band],
            destination=projected[..., band],
            src_transform=src_transform,
            src_crs="EPSG:3857",
            dst_transform=dst_transform,
            dst_crs=CONUS_CRS,
            resampling=Resampling.bilinear,
        )
    land_mask = geometry_mask(
        [mapping(conus)],
        out_shape=(dst_height, dst_width),
        transform=dst_transform,
        invert=True,
        all_touched=True,
    )
    rgba = np.dstack([np.clip(projected, 0.0, 1.0), land_mask.astype(np.float32)])
    west, south, east, north = array_bounds(dst_height, dst_width, dst_transform)
    return rgba, (west, east, south, north)


def _site_centroids() -> gpd.GeoDataFrame:
    original = gpd.read_file(
        ROOT / "metadata" / "phase9_neon_habitat_sites_2024.geojson"
    )
    replication_2024 = gpd.read_file(
        ROOT / "metadata" / "phase13_candidate_boundaries.geojson"
    )
    replication_2025 = gpd.read_file(
        ROOT / "metadata" / "phase13_2025_candidate_boundaries.geojson"
    )
    replication_sites = ["CLBJ", "KONZ", "ONAQ", "RMNP", "YELL"]
    replication = pd.concat(
        [
            replication_2024.loc[
                replication_2024["siteID"].isin(["CLBJ", "KONZ", "RMNP", "YELL"])
            ],
            replication_2025.loc[replication_2025["siteID"].eq("ONAQ")],
        ],
        ignore_index=True,
    )
    sites = gpd.GeoDataFrame(
        pd.concat(
            [original.loc[~original["siteID"].isin(replication_sites)], replication],
            ignore_index=True,
        ),
        geometry="geometry",
        crs=4326,
    )
    sites = gpd.GeoDataFrame(sites, geometry="geometry", crs=4326).to_crs(CONUS_CRS)
    sites["geometry"] = sites.geometry.centroid
    group = {
        "SOAP": "development",
        "TEAK": "development",
        "BART": "heldout",
        "HARV": "prospective",
        "ORNL": "prospective",
        "TALL": "prospective",
        "UNDE": "prospective",
        "WREF": "prospective",
        "ABBY": "habitat",
        "GRSM": "habitat",
        "JERC": "habitat",
        "MLBS": "habitat",
        "SCBI": "habitat",
        "STEI": "habitat",
        "CLBJ": "replication",
        "KONZ": "replication",
        "ONAQ": "replication",
        "RMNP": "replication",
        "YELL": "replication",
        "NIWO": "excluded",
        "OSBS": "excluded",
        "SERC": "excluded",
    }
    sites["group"] = sites["siteID"].map(group)
    return sites.loc[sites["group"].notna()].copy()


def make_study_map() -> None:
    states = _download_state_boundaries()
    sites = _site_centroids()
    image = plt.imread(_download_sentinel_context())
    conus = states.geometry.union_all()
    image, image_extent = _project_sentinel_cutout(image, conus)

    fig, ax = plt.subplots(figsize=(7.25, 3.75))
    ax.set_facecolor("#FFFFFF")
    ax.imshow(image, extent=image_extent, origin="upper", zorder=0)
    states.boundary.plot(ax=ax, color="#F4F4F4", linewidth=0.38, alpha=0.78, zorder=1)
    gpd.GeoSeries([conus], crs=CONUS_CRS).boundary.plot(
        ax=ax,
        color="#FFFFFF",
        linewidth=1.0,
        alpha=0.95,
        zorder=2,
    )
    for group in [
        "development", "heldout", "prospective", "habitat", "replication", "excluded"
    ]:
        subset = sites.loc[sites["group"] == group]
        marker = "x" if group == "excluded" else "o"
        scatter_kwargs = {
            "s": 31 if group != "excluded" else 45,
            "c": COLORS[group],
            "marker": marker,
            "linewidth": 1.15,
            "zorder": 3,
        }
        if group != "excluded":
            scatter_kwargs["edgecolor"] = "#FFFFFF"
        ax.scatter(subset.geometry.x, subset.geometry.y, **scatter_kwargs)

    label_positions = {
        "SOAP": (-4, -4, "right", "top"),
        "TEAK": (-4, 4, "right", "bottom"),
        "BART": (4, 4, "left", "bottom"),
        "HARV": (-4, -4, "right", "top"),
        "ORNL": (-4, 4, "right", "bottom"),
        "TALL": (-4, 4, "right", "bottom"),
        "UNDE": (-4, 4, "right", "bottom"),
        "WREF": (4, 4, "left", "bottom"),
        "ABBY": (-4, -4, "right", "top"),
        "GRSM": (4, -4, "left", "top"),
        "JERC": (4, -4, "left", "top"),
        "MLBS": (4, -4, "left", "top"),
        "SCBI": (-4, 4, "right", "bottom"),
        "SERC": (4, -4, "left", "top"),
        "STEI": (4, -4, "left", "top"),
        "CLBJ": (-4, -4, "right", "top"),
        "KONZ": (4, -4, "left", "top"),
        "ONAQ": (-4, -4, "right", "top"),
        "RMNP": (-5, 5, "right", "bottom"),
        "YELL": (-4, 4, "right", "bottom"),
        "NIWO": (5, -5, "left", "top"),
        "OSBS": (4, -4, "left", "top"),
    }
    for row in sites.itertuples():
        dx, dy, horizontal_alignment, vertical_alignment = label_positions[row.siteID]
        label = ax.annotate(
            row.siteID,
            xy=(row.geometry.x, row.geometry.y),
            xytext=(dx, dy),
            textcoords="offset points",
            color=COLORS["excluded"] if row.group == "excluded" else "#FFFFFF",
            fontsize=8.3,
            weight="bold",
            ha=horizontal_alignment,
            va=vertical_alignment,
            zorder=5,
        )
        label.set_path_effects(
            [path_effects.withStroke(linewidth=1.8, foreground="#111111")]
        )

    min_x, min_y, max_x, max_y = conus.bounds
    x_pad = (max_x - min_x) * 0.018
    y_pad = (max_y - min_y) * 0.035
    ax.set_xlim(min_x - x_pad, max_x + x_pad)
    ax.set_ylim(min_y - y_pad, max_y + y_pad)
    ax.set_aspect("equal")
    ax.set_axis_off()
    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["development"], markeredgecolor="white", markersize=7, label="Development"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["heldout"], markeredgecolor="white", markersize=7, label="Initial held-out site"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["prospective"], markeredgecolor="white", markersize=7, label="First prospective expansion"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["habitat"], markeredgecolor="white", markersize=7, label="Forest-form test"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["replication"], markeredgecolor="white", markersize=7, label="Independent track replication"),
        Line2D([0], [0], marker="x", color=COLORS["excluded"], linestyle="none", markersize=6, label="Below sample threshold"),
    ]
    fig.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.008),
        frameon=False,
        ncol=3,
        columnspacing=0.9,
        handletextpad=0.4,
    )
    attribution = ax.text(
        0.995,
        0.012,
        "EOxCloudless 2024 | modified Copernicus Sentinel data",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.2,
        color="#4B5563",
        zorder=5,
    )
    fig.subplots_adjust(left=0.015, right=0.995, top=0.995, bottom=0.17)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / "paper_study_sites.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def make_transfer_evidence() -> None:
    prospective = pd.read_csv(ROOT / "outputs" / "tables" / "phase7_prospective_metrics.csv")
    prospective = prospective.loc[prospective["scope"] == "site"]
    fair = pd.read_csv(ROOT / "outputs" / "tables" / "phase8_fair_baseline_metrics.csv")
    fair = fair.loc[fair["scope"] == "site"]

    prospective_sites = ["HARV", "ORNL", "TALL", "UNDE", "WREF"]
    fair_sites = ["BART", "HARV", "ORNL", "SOAP", "TALL", "TEAK", "UNDE", "WREF"]
    x1 = np.arange(len(prospective_sites))
    x2 = np.arange(len(fair_sites))
    width = 0.36

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.25, 2.75),
        gridspec_kw={"wspace": 0.42, "width_ratios": [0.9, 1.35, 1.35]},
    )

    ax = axes[0]
    for offset, method, label, color in [
        (-width / 2, "original_site_mean", "original-site mean", COLORS["mean"]),
        (width / 2, "tessera_area_ridge", "TESSERA", COLORS["tessera"]),
    ]:
        values = prospective.set_index(["site_id", "method"]).loc[
            [(s, method) for s in prospective_sites], "rmse"
        ].to_numpy()
        ax.bar(x1 + offset, values, width, label=label, color=color)
    ax.set_xticks(x1, prospective_sites, rotation=35, ha="right")
    ax.set_ylabel("RMSE")
    ax.set_title("a  Prospective test", loc="left", weight="bold")

    ax = axes[1]
    for offset, method, label, color in [
        (-width / 2, "sentinel2_topography_ridge", "Sentinel-2 + terrain", COLORS["sentinel"]),
        (width / 2, "tessera_area_ridge", "TESSERA", COLORS["tessera"]),
    ]:
        values = fair.set_index(["held_out_site", "method"]).loc[
            [(s, method) for s in fair_sites], "rmse"
        ].to_numpy()
        ax.bar(x2 + offset, values, width, label=label, color=color)
    ax.set_xticks(x2, fair_sites, rotation=45, ha="right")
    ax.set_ylabel("RMSE")
    ax.set_title("b  Complete-site error", loc="left", weight="bold")

    ax = axes[2]
    for offset, method, label, color in [
        (-width / 2, "sentinel2_topography_ridge", "Sentinel-2 + terrain", COLORS["sentinel"]),
        (width / 2, "tessera_area_ridge", "TESSERA", COLORS["tessera"]),
    ]:
        values = fair.set_index(["held_out_site", "method"]).loc[
            [(s, method) for s in fair_sites], "spearman_r"
        ].to_numpy()
        ax.bar(x2 + offset, values, width, label=label, color=color)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x2, fair_sites, rotation=45, ha="right")
    ax.set_ylabel("Spearman $r_s$")
    ax.set_title("c  Complete-site rank", loc="left", weight="bold")

    for ax in axes:
        ax.grid(axis="y", color="#D8DDE2", linewidth=0.5, alpha=0.8)
        ax.set_axisbelow(True)

    fig.legend(
        handles=[
            Patch(facecolor=COLORS["mean"], label="Original-site mean"),
            Patch(facecolor=COLORS["sentinel"], label="Sentinel-2 + terrain"),
            Patch(facecolor=COLORS["tessera"], label="TESSERA"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
        columnspacing=1.4,
        handletextpad=0.5,
    )
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.24, top=0.78)
    fig.savefig(FIGURE_DIR / "paper_transfer_evidence.pdf", bbox_inches="tight")
    plt.close(fig)


def _one_to_one_line(ax, x: pd.Series, y: pd.Series) -> None:
    low = float(min(x.min(), y.min()))
    high = float(max(x.max(), y.max()))
    margin = (high - low) * 0.06
    ax.plot(
        [low - margin, high + margin],
        [low - margin, high + margin],
        linestyle="--",
        color="#333333",
        linewidth=0.8,
        zorder=1,
    )
    ax.set_xlim(low - margin, high + margin)
    ax.set_ylim(low - margin, high + margin)


def make_habitat_evaluation() -> None:
    metrics = pd.read_csv(
        ROOT / "outputs" / "tables" / "phase9_prospective_evaluation_metrics.csv"
    )
    metrics = metrics.loc[metrics["scope"] == "site"]
    paired = pd.read_csv(
        ROOT / "outputs" / "tables" / "phase9_prospective_evaluation_paired.csv"
    )
    sites = ["ABBY", "GRSM", "JERC", "MLBS", "SCBI", "STEI"]
    baseline = metrics.loc[metrics["strategy"] == "all_sources"].set_index("site_id").loc[sites]
    matched = metrics.loc[metrics["strategy"] == "nearest5_evt_phys"].set_index("site_id").loc[sites]
    delta = matched["rmse"] - baseline["rmse"]
    x = np.arange(len(sites))
    width = 0.36

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.25, 2.7),
        gridspec_kw={"wspace": 0.42, "width_ratios": [1.25, 1.05, 0.9]},
    )
    axes[0].bar(x - width / 2, baseline["rmse"], width, color=COLORS["mean"])
    axes[0].bar(x + width / 2, matched["rmse"], width, color=COLORS["tessera"])
    axes[0].set_xticks(x, sites, rotation=40, ha="right")
    axes[0].set_ylabel("RMSE")
    axes[0].set_title("a  Prospective site error", loc="left", weight="bold")

    axes[1].bar(
        x,
        delta,
        color=np.where(delta.to_numpy() < 0, COLORS["tessera"], COLORS["heldout"]),
    )
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[1].set_xticks(x, sites, rotation=40, ha="right")
    axes[1].set_ylabel("Selected minus all-source RMSE")
    axes[1].set_title("b  Paired site change", loc="left", weight="bold")

    order = ["nearest5_evt_phys", "nearest5_evt_group"]
    summary = paired.set_index("strategy").loc[order]
    labels = ["Physiognomy", "Vegetation\ngroup"]
    axes[2].bar(labels, summary["mean_delta_rmse"], color=[COLORS["tessera"], "#4E8098"])
    axes[2].errorbar(
        labels,
        summary["mean_delta_rmse"],
        yerr=np.vstack(
            [
                summary["mean_delta_rmse"] - summary["bootstrap_95_low"],
                summary["bootstrap_95_high"] - summary["mean_delta_rmse"],
            ]
        ),
        fmt="none",
        color="#202124",
        capsize=2.5,
        linewidth=0.9,
    )
    axes[2].axhline(0, color="#333333", linewidth=0.8)
    axes[2].set_ylabel("Mean paired RMSE change")
    axes[2].set_title("c  Descriptive site interval", loc="left", weight="bold")

    for ax in axes:
        ax.grid(axis="y", color="#D8DDE2", linewidth=0.5, alpha=0.8)
        ax.set_axisbelow(True)
    fig.legend(
        handles=[
            Patch(facecolor=COLORS["mean"], label="All eight sources"),
            Patch(facecolor=COLORS["tessera"], label="Five forest-form-selected sources"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        frameon=False,
        columnspacing=1.4,
        handletextpad=0.5,
    )
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.25, top=0.76)
    fig.savefig(FIGURE_DIR / "paper_habitat_evaluation.pdf", bbox_inches="tight")
    plt.close(fig)


def make_height_dependence() -> None:
    paths = [
        ROOT / "data" / "processed" / "tessera_aligned_development_soap_teak.parquet",
        ROOT / "data" / "processed" / "tessera_aligned_locked_bart.parquet",
        ROOT / "data" / "processed" / "phase7_tessera_aligned_expansion.parquet",
    ]
    source_population = pd.concat(
        [
            pd.read_parquet(
                path,
                columns=["site_id", "fhd_normal", "elev_highestreturn", "elev_lowestmode"],
            )
            for path in paths
        ],
        ignore_index=True,
    )
    source_population["cohort"] = "source"
    habitat_population = pd.read_parquet(
        ROOT / "data" / "processed" / "phase9_prospective_gedi_v3_primary_forest.parquet",
        columns=["site_id", "fhd_normal", "elev_highestreturn", "elev_lowestmode"],
    )
    habitat_population["cohort"] = "habitat"
    population = pd.concat([source_population, habitat_population], ignore_index=True)
    population["height"] = population["elev_highestreturn"] - population["elev_lowestmode"]
    source_sites = ["BART", "HARV", "ORNL", "SOAP", "TALL", "TEAK", "UNDE", "WREF"]
    habitat_sites = ["ABBY", "GRSM", "JERC", "MLBS", "SCBI", "STEI"]
    sites = source_sites + habitat_sites
    if set(population["site_id"]) != set(sites) or len(population) != 15_326:
        raise RuntimeError("Unexpected 14-site FHD-height population")
    cohort_colors = {"source": COLORS["mean"], "habitat": COLORS["habitat"]}
    associations = (
        population.groupby(["site_id", "cohort"], sort=False)
        .apply(
            lambda frame: frame[["height", "fhd_normal"]]
            .corr(method="spearman")
            .iloc[0, 1],
            include_groups=False,
        )
        .rename("spearman_r")
        .reset_index()
        .set_index("site_id")
        .loc[sites]
    )
    quintile_frames = []
    for site, frame in population.groupby("site_id", sort=False):
        scoped = frame.copy()
        scoped["height_quintile"] = pd.qcut(
            scoped["height"], 5, labels=False, duplicates="raise"
        ) + 1
        quintile_frames.append(
            scoped.groupby("height_quintile", as_index=False)
            .agg(fhd_mean=("fhd_normal", "mean"))
            .assign(site_id=site, cohort=scoped["cohort"].iloc[0])
        )
    quintiles = pd.concat(quintile_frames, ignore_index=True)
    adjusted = pd.read_csv(
        ROOT / "outputs" / "tables" / "phase10_height_adjusted_existing_paired.csv"
    ).set_index("comparison")
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.15), gridspec_kw={"hspace": 0.42, "wspace": 0.30})

    for cohort in ["source", "habitat"]:
        frame = population.loc[population["cohort"] == cohort]
        axes[0, 0].scatter(
            frame["height"],
            frame["fhd_normal"],
            s=2.2,
            alpha=0.11,
            color=cohort_colors[cohort],
        )
    axes[0, 0].set_xlabel("GEDI canopy-height proxy (m)")
    axes[0, 0].set_ylabel("GEDI V3 fhd_normal")
    axes[0, 0].set_title("a  FHD and canopy height", loc="left", weight="bold")

    axes[0, 1].bar(
        np.arange(len(sites)),
        associations["spearman_r"],
        color=[cohort_colors[associations.loc[site, "cohort"]] for site in sites],
    )
    axes[0, 1].set_xticks(np.arange(len(sites)), sites, rotation=50, ha="right")
    axes[0, 1].tick_params(axis="x", labelsize=6.7)
    axes[0, 1].set_ylabel("Within-site Spearman $r_s$")
    axes[0, 1].set_title("b  Within-site height association", loc="left", weight="bold")

    for site in sites:
        frame = quintiles.loc[quintiles["site_id"] == site]
        cohort = frame["cohort"].iloc[0]
        axes[1, 0].plot(
            frame["height_quintile"],
            frame["fhd_mean"],
            marker="o",
            markersize=2.0,
            linewidth=0.7,
            alpha=0.32,
            color=cohort_colors[cohort],
        )
    for cohort in ["source", "habitat"]:
        mean_profile = (
            quintiles.loc[quintiles["cohort"] == cohort]
            .groupby("height_quintile", as_index=False)["fhd_mean"]
            .mean()
        )
        axes[1, 0].plot(
            mean_profile["height_quintile"],
            mean_profile["fhd_mean"],
            marker="o",
            markersize=3.4,
            linewidth=1.8,
            color=cohort_colors[cohort],
        )
    axes[1, 0].set_xlabel("Within-site canopy-height quintile")
    axes[1, 0].set_ylabel("Mean GEDI V3 fhd_normal")
    axes[1, 0].set_title("c  FHD across height quintiles", loc="left", weight="bold")

    comparisons = [
        "height_plus_all_source_tessera_minus_height_only",
        "height_plus_matched_tessera_minus_height_only",
        "matched_minus_all_source_residual_tessera",
    ]
    adjusted = adjusted.loc[comparisons]
    labels = [
        "All-source\nvs height",
        "Forest-form\nvs height",
        "Forest-form\nvs all-source",
    ]
    values = adjusted["mean_delta_rmse"].to_numpy()
    axes[1, 1].bar(
        np.arange(len(labels)),
        values,
        color=[COLORS["tessera"], COLORS["habitat"], COLORS["heldout"]],
        width=0.62,
    )
    axes[1, 1].errorbar(
        np.arange(len(labels)),
        values,
        yerr=np.vstack(
            [
                values - adjusted["bootstrap_95_low"].to_numpy(),
                adjusted["bootstrap_95_high"].to_numpy() - values,
            ]
        ),
        fmt="none",
        color="#202124",
        capsize=2.5,
        linewidth=0.9,
    )
    axes[1, 1].axhline(0, color="#333333", linestyle="--", linewidth=0.8)
    axes[1, 1].set_xticks(np.arange(len(labels)), labels)
    axes[1, 1].tick_params(axis="x", labelsize=7.2)
    axes[1, 1].set_ylabel("Mean paired RMSE change")
    axes[1, 1].set_title("d  Height-adjusted prediction", loc="left", weight="bold")

    for ax in axes.flat:
        ax.grid(color="#D8DDE2", linewidth=0.45, alpha=0.75)
        ax.set_axisbelow(True)
    fig.legend(
        handles=[
            Patch(facecolor=cohort_colors["source"], label="Eight source sites"),
            Patch(facecolor=cohort_colors["habitat"], label="Six forest-form test sites"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        frameon=False,
        columnspacing=1.4,
        handletextpad=0.5,
    )
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.105, top=0.90)
    fig.savefig(FIGURE_DIR / "paper_height_dependence.pdf", bbox_inches="tight")
    plt.close(fig)


def make_robustness_summary() -> None:
    spatial = pd.read_csv(ROOT / "outputs" / "tables" / "robustness_spatial_metrics.csv")
    measurement = pd.read_csv(
        ROOT / "outputs" / "tables" / "robustness_measurement_paired.csv"
    )
    calibration = pd.read_csv(
        ROOT / "outputs" / "tables" / "robustness_calibration_diagnostics.csv"
    )
    targets = pd.read_csv(
        ROOT / "outputs" / "tables" / "robustness_model_and_target_metrics.csv"
    )

    fig, axes = plt.subplots(
        2, 2, figsize=(7.25, 5.15), gridspec_kw={"hspace": 0.50, "wspace": 0.32}
    )

    pooled = spatial.loc[spatial["scope"] == "pooled"].copy()
    buffer_rows = pooled.loc[pooled["analysis"].str.startswith("spatial_buffer")].copy()
    buffer_rows["buffer_km"] = (
        buffer_rows["analysis"].str.extract(r"_(\d+)m$")[0].astype(float) / 1000.0
    )
    buffer_rows = buffer_rows.sort_values("buffer_km")
    axes[0, 0].plot(
        buffer_rows["buffer_km"], buffer_rows["r2"], marker="o", color=COLORS["tessera"]
    )
    pass_r2 = float(
        pooled.loc[pooled["analysis"] == "leave_one_gedi_pass_out", "r2"].iloc[0]
    )
    axes[0, 0].axhline(pass_r2, color=COLORS["heldout"], linestyle="--", linewidth=1.1)
    axes[0, 0].text(
        5.95, pass_r2 + 0.025, "complete-pass holdout", color=COLORS["heldout"],
        ha="right", va="bottom", fontsize=7.4,
    )
    axes[0, 0].axhline(0, color="#333333", linewidth=0.7)
    axes[0, 0].set_xlabel("Training exclusion distance (km)")
    axes[0, 0].set_ylabel("Pooled $R^2$")
    axes[0, 0].set_title("a  Spatial separation", loc="left", weight="bold")

    subset_order = [
        "all_retained", "sensitivity_ge_0.95", "sensitivity_ge_0.98",
        "slope_le_15deg", "slope_le_30deg", "may_to_september", "nighttime",
    ]
    labels = ["All", "Sens. .95", "Sens. .98", "Slope 15", "Slope 30", "May--Sep", "Night"]
    measurement = measurement.set_index("subset").loc[subset_order]
    axes[0, 1].bar(
        np.arange(len(labels)), measurement["mean_matched_minus_all_rmse"],
        color=COLORS["tessera"],
    )
    axes[0, 1].axhline(0, color="#333333", linewidth=0.7)
    axes[0, 1].set_xticks(np.arange(len(labels)), labels, rotation=38, ha="right")
    axes[0, 1].set_ylabel("Selected minus all-source RMSE")
    axes[0, 1].set_title("b  Measurement-condition checks", loc="left", weight="bold")

    sites = ["ABBY", "GRSM", "JERC", "MLBS", "SCBI", "STEI"]
    x = np.arange(len(sites))
    width = 0.36
    for offset, strategy, label, color in [
        (-width / 2, "all_sources", "All sources", COLORS["mean"]),
        (width / 2, "nearest5_evt_phys", "Forest-form selected", COLORS["tessera"]),
    ]:
        values = calibration.loc[calibration["strategy"] == strategy].set_index("site_id").loc[
            sites, "predicted_to_observed_sd_ratio"
        ]
        axes[1, 0].bar(x + offset, values, width, color=color, label=label)
    axes[1, 0].axhline(1, color="#333333", linestyle="--", linewidth=0.8)
    axes[1, 0].set_xticks(x, sites, rotation=40, ha="right")
    axes[1, 0].set_ylabel("Predicted / observed SD")
    axes[1, 0].set_title("c  Prediction shrinkage", loc="left", weight="bold")
    axes[1, 0].legend(frameon=False, fontsize=7.2, loc="upper left")

    macro = targets.loc[targets["scope"] == "macro"].copy()
    specifications = [
        ("fhd_normal", "ridge", "all_sources", "FHD\nRidge", COLORS["mean"]),
        ("fhd_normal", "ridge", "locked_nearest5_forest_form", "FHD\nforest-form", COLORS["tessera"]),
        ("fhd_normal", "hist_gradient_boosting", "all_sources", "FHD\nnonlinear", COLORS["sentinel"]),
        ("height_normalized_entropy_proxy", "ridge", "all_sources", "Height-normalized\nproxy", COLORS["habitat"]),
    ]
    values = []
    for target_name, model, strategy, _, _ in specifications:
        values.append(float(macro.loc[
            macro["target_variable"].eq(target_name)
            & macro["model"].eq(model)
            & macro["strategy"].eq(strategy), "r2"
        ].iloc[0]))
    axes[1, 1].bar(
        np.arange(len(values)), values, color=[item[4] for item in specifications]
    )
    axes[1, 1].axhline(0, color="#333333", linewidth=0.7)
    axes[1, 1].set_xticks(
        np.arange(len(values)), [item[3] for item in specifications], rotation=28, ha="right"
    )
    axes[1, 1].set_ylabel("Six-site mean $R^2$")
    axes[1, 1].set_title("d  Target and model sensitivity", loc="left", weight="bold")

    for axis in axes.flat:
        axis.grid(axis="y", color="#D8DDE2", linewidth=0.45, alpha=0.8)
        axis.set_axisbelow(True)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.12, top=0.97)
    fig.savefig(FIGURE_DIR / "paper_robustness_summary.pdf", bbox_inches="tight")
    plt.close(fig)


def make_continuous_validation() -> None:
    prediction_path = ROOT / "outputs" / "maps" / "robustness_bart_local_fhd.tif"
    support_path = ROOT / "outputs" / "maps" / "robustness_bart_area_of_applicability.tif"
    offtrack = pd.read_parquet(
        ROOT / "data" / "processed" / "robustness_bart_offtrack_als.parquet"
    )
    support_summary = pd.read_csv(
        ROOT / "outputs" / "tables" / "robustness_bart_applicability.csv"
    )
    threshold = float(support_summary["support_threshold"].iloc[0])
    with rasterio.open(prediction_path) as dataset:
        prediction = dataset.read(1, masked=True)
        bounds = dataset.bounds
    with rasterio.open(support_path) as dataset:
        support = dataset.read(1, masked=True)

    extent = [0.0, 1.0, 0.0, 1.0]
    offtrack_x = (offtrack["utm_x"] - bounds.left) / 1000.0
    offtrack_y = (offtrack["utm_y"] - bounds.bottom) / 1000.0
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.45), gridspec_kw={"wspace": 0.54})
    image = axes[0].imshow(prediction, extent=extent, origin="upper", cmap="viridis")
    axes[0].scatter(offtrack_x, offtrack_y, s=5, c="white", edgecolor="#222222", linewidth=0.2)
    axes[0].set_title("a  Complete-tile FHD", loc="left", weight="bold", fontsize=8.5)
    axes[0].set_xlabel("Tile easting (km)")
    axes[0].set_ylabel("Tile northing (km)")
    colorbar = fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.025)
    colorbar.set_label("Predicted FHD", rotation=270, labelpad=10, fontsize=7)
    colorbar.ax.tick_params(labelsize=6.5)

    image = axes[1].imshow(support, extent=extent, origin="upper", cmap="magma")
    axes[1].contour(
        np.asarray(support) <= threshold, levels=[0.5], extent=extent,
        colors="white", linewidths=0.65,
    )
    axes[1].set_title("b  Distance from training", loc="left", weight="bold", fontsize=8.5)
    axes[1].set_xlabel("Tile easting (km)")
    colorbar = fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.025)
    colorbar.set_label("Support distance", rotation=270, labelpad=10, fontsize=7)
    colorbar.ax.tick_params(labelsize=6.5)

    axes[2].scatter(
        offtrack["converted_als_fhd"], offtrack["local_tessera_prediction"],
        c=offtrack["applicability_distance"], cmap="magma", s=15, alpha=0.82,
        edgecolor="white", linewidth=0.2,
    )
    _one_to_one_line(axes[2], offtrack["converted_als_fhd"], offtrack["local_tessera_prediction"])
    axes[2].set_xlabel("Converted airborne-lidar FHD")
    axes[2].set_ylabel("Local prediction")
    axes[2].set_title("c  Off-track lidar check", loc="left", weight="bold", fontsize=8.5)
    axes[2].grid(color="#D8DDE2", linewidth=0.45, alpha=0.75)
    axes[2].set_axisbelow(True)

    for axis in axes[:2]:
        axis.set_xticks([0.0, 0.5, 1.0])
        axis.set_yticks([0.0, 0.5, 1.0])
        axis.tick_params(labelsize=7)
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.19, top=0.91)
    fig.savefig(FIGURE_DIR / "paper_bart_continuous_validation.pdf", bbox_inches="tight")
    plt.close(fig)


def make_track_assisted() -> None:
    retrospective = pd.read_csv(
        ROOT / "outputs" / "tables" / "phase12_track_assisted_macro_metrics.csv"
    )
    retrospective = retrospective[
        retrospective["analysis_group"].eq("primary")
    ].set_index("model")
    independent = pd.read_csv(
        ROOT / "outputs" / "tables" / "phase13_track_replication_macro_metrics.csv"
    )
    independent = independent[independent["budget"].eq(50)].set_index("model")
    paired = pd.read_csv(
        ROOT / "outputs" / "tables" / "phase13_track_replication_paired_sites.csv"
    )
    paired = paired[
        paired["comparison"].eq("tessera_offset_minus_local_mean")
    ].sort_values("site_id")
    paired_bootstrap = pd.read_csv(
        ROOT / "outputs" / "tables" / "phase13_track_replication_bootstrap.csv"
    ).set_index("comparison").loc["tessera_offset_minus_local_mean"]
    decomposition = pd.read_csv(
        ROOT / "outputs" / "tables" / "phase13_error_decomposition_macro.csv"
    )
    decomposition = decomposition[decomposition["budget"].eq(50)].set_index("model")

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.25, 2.8),
        gridspec_kw={"wspace": 0.44, "width_ratios": [1.08, 1.0, 1.05]},
    )
    cohorts = ["Retrospective", "Independent"]
    x = np.arange(2)
    width = 0.24
    model_specs = [
        (
            "Source-only TESSERA",
            "source_tessera_ridge",
            "source_only_tessera",
            COLORS["mean"],
        ),
        (
            "Local GEDI mean",
            "target_local_mean",
            "sampled_target_local_mean",
            "#D6A756",
        ),
        (
            "TESSERA + local correction",
            "source_plus_local_offset",
            "tessera_plus_sampled_local_offset",
            COLORS["tessera"],
        ),
    ]
    for index, (label, old_name, new_name, color) in enumerate(model_specs):
        values = [retrospective.loc[old_name, "rmse"], independent.loc[new_name, "rmse"]]
        bars = axes[0].bar(
            x + (index - 1) * width, values, width, color=color, label=label
        )
        axes[0].bar_label(bars, fmt="%.3f", padding=1.5, fontsize=6.3, rotation=90)
    axes[0].set_xticks(x, cohorts)
    axes[0].set_ylim(0.0, 1.24)
    axes[0].set_ylabel("Equal-forest RMSE")
    axes[0].set_title("a  Replication", loc="left", weight="bold")
    axes[0].legend(
        frameon=False,
        fontsize=6.3,
        loc="upper left",
        bbox_to_anchor=(-0.03, 1.01),
        handlelength=1.2,
        handletextpad=0.4,
    )

    sites = paired["site_id"].tolist()
    differences = paired["rmse_difference"].to_numpy(dtype=float)
    site_x = np.arange(len(sites))
    axes[1].bar(
        site_x,
        differences,
        color=np.where(differences < 0, COLORS["tessera"], COLORS["heldout"]),
    )
    axes[1].axhline(0.0, color="#444444", linewidth=0.8)
    axes[1].axhline(
        paired_bootstrap["mean_paired_rmse_difference"],
        color="#111111",
        linestyle="--",
        linewidth=0.8,
    )
    axes[1].set_xticks(site_x, sites, rotation=45, ha="right")
    axes[1].set_ylabel("Corrected TESSERA minus local-mean RMSE")
    axes[1].set_title("b  Independent forests", loc="left", weight="bold")

    decomposition_models = [
        "sampled_target_local_mean",
        "tessera_plus_sampled_local_offset",
        "sentinel2_terrain_plus_sampled_local_offset",
    ]
    decomposition_labels = ["Local\nmean", "TESSERA +\ncorrection", "Sentinel-2 +\ncorrection"]
    bias = decomposition.loc[decomposition_models, "squared_bias"].to_numpy(dtype=float)
    centered = decomposition.loc[decomposition_models, "centered_mse"].to_numpy(dtype=float)
    component_x = np.arange(len(decomposition_models))
    axes[2].bar(component_x, bias, color="#9ECAE1", label="Squared mean error")
    axes[2].bar(
        component_x,
        centered,
        bottom=bias,
        color="#6A51A3",
        label="Spatial-pattern error",
    )
    axes[2].set_xticks(component_x, decomposition_labels)
    axes[2].set_ylabel("Equal-forest mean squared error")
    axes[2].set_title("c  Error decomposition", loc="left", weight="bold")
    axes[2].legend(frameon=False, fontsize=6.2, loc="upper left")
    axes[2].set_ylim(0.0, 0.61)

    for axis in axes:
        axis.grid(axis="y", color="#D8DDE2", linewidth=0.5, alpha=0.8)
        axis.set_axisbelow(True)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.24, top=0.94)
    fig.savefig(FIGURE_DIR / "paper_track_assisted.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    make_study_map()
    make_transfer_evidence()
    make_habitat_evaluation()
    make_height_dependence()
    make_robustness_summary()
    make_continuous_validation()
    make_track_assisted()
    print(
        json.dumps(
            {
                "figures": [
                    "paper_study_sites.png",
                    "paper_transfer_evidence.pdf",
                    "paper_habitat_evaluation.pdf",
                    "paper_height_dependence.pdf",
                    "paper_robustness_summary.pdf",
                    "paper_bart_continuous_validation.pdf",
                    "paper_track_assisted.pdf",
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
