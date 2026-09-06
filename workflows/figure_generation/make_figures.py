#!/usr/bin/env python3
"""Build the main multi-panel and supplementary analysis figures."""

from __future__ import annotations

import string
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from figure_output import save_figure
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Rectangle
from matplotlib.ticker import MaxNLocator
from PIL import Image
from pyproj import Transformer
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/figures/main"

CAIRN_TARGETS = ROOT / "data/processed/cairngorms_surface_targets.parquet"
CAIRN_PREDICTIONS = ROOT / "data/processed/cairngorms_surface_predictions.parquet"
CAIRN_MACRO = ROOT / "results/tables/cairngorms_performance.csv"
HEIGHT_DIR = ROOT / "data/interim/cairngorms_height_adjusted"
HEIGHT_DIAGNOSTICS = ROOT / "results/tables/diagnostics/cairngorms_height_adjustment.csv"

DUTCH_SITES = ROOT / "results/tables/dutch_forest_sites.csv"
DUTCH_METRICS = ROOT / "results/tables/dutch_transfer_fold_metrics.csv"
DUTCH_PAIRWISE = ROOT / "results/tables/dutch_pairwise_transfer.csv"
DUTCH_BOUNDARIES = ROOT / "data/external/ahn4_selection/Nature2000_NL_RDnew.shp"
DUTCH_COHORT = ROOT / "data/processed/dutch_eligible_units_map.parquet"
NATURAL_EARTH_10M = ROOT / "data/external/natural_earth_admin0_10m/ne_10m_admin_0_countries.shp"
NATURAL_EARTH_110M = ROOT / "data/external/natural_earth_admin0/ne_110m_admin_0_countries.shp"
NATURAL_EARTH = NATURAL_EARTH_10M if NATURAL_EARTH_10M.exists() else NATURAL_EARTH_110M
SATELLITE_CACHE = ROOT / "data/external/eox_s2cloudless"
RANK_SUMMARY = ROOT / "results/tables/rank_transfer.csv"
RANK_FOLD_METRICS = ROOT / "results/tables/dutch_rank_transfer_by_fold.csv"
REFERENCE_CURVE = ROOT / "results/tables/local_reference_curve.csv"
REFERENCE_UNCERTAINTY = ROOT / "results/tables/lidar_reference_uncertainty.csv"
VERSION_COMPARISON = ROOT / "results/tables/diagnostics/cairngorms_embedding_version_comparison.csv"
SAVELSBOS_TARGETS = ROOT / "data/processed/savelsbos_targets.parquet"
SAVELSBOS_GEOMETRY = ROOT / "data/processed/savelsbos_site.gpkg"

PATCH_VALUES = ROOT / "data/interim/cairngorms_embedding_patches/patch_quantized.npy"
PATCH_SCALES = ROOT / "data/interim/cairngorms_embedding_patches/patch_scales.npy"
PATCH_IDS = ROOT / "data/interim/cairngorms_embedding_patches/row_id.npy"
PROFILE_VALUES = ROOT / "data/interim/cairngorms_return_profiles/profiles.npy"
PROFILE_IDS = ROOT / "data/interim/cairngorms_return_profiles/row_id.npy"

GREEN = "#2a8c78"
ORANGE = "#d97732"
PURPLE = "#7355a8"
BLUE = "#3f719c"
CHARCOAL = "#42484f"
RED = "#b54845"
LIGHT_GREY = "#e5e7e8"

RAW_TARGETS = [
    "canopy_p95_height_m",
    "canopy_surface_sd_m",
    "canopy_surface_cv",
    "canopy_surface_rcv",
    "canopy_rumple",
    "canopy_open_fraction",
    "canopy_height_kurtosis",
]
ADJUSTED_TARGETS = RAW_TARGETS[1:]
DIST_TARGETS = ["canopy_mean_height_m", *RAW_TARGETS]
CORE_TARGETS = ["canopy_surface_cv", "canopy_rumple", "canopy_open_fraction"]

LABELS = {
    "canopy_mean_height_m": "Mean canopy height",
    "canopy_p95_height_m": "P95 canopy height",
    "canopy_surface_sd_m": "Canopy-height SD",
    "canopy_surface_cv": "Canopy-height CV",
    "canopy_surface_rcv": "Robust height CV",
    "canopy_rumple": "Canopy rumple",
    "canopy_open_fraction": "Canopy openings",
    "canopy_height_kurtosis": "Height kurtosis",
}
UNITS = {
    "canopy_mean_height_m": "m",
    "canopy_p95_height_m": "m",
    "canopy_surface_sd_m": "m",
    "canopy_surface_cv": "ratio",
    "canopy_surface_rcv": "ratio",
    "canopy_rumple": "ratio",
    "canopy_open_fraction": "fraction",
    "canopy_height_kurtosis": "unitless",
}
DUTCH_LABELS = {
    "ahn4_p95_height_m": "P95 height",
    "ahn4_height_sd_m": "Height SD",
    "ahn4_height_cv": "Height CV",
    "ahn4_entropy": "Return entropy",
    "ahn4_pulse_penetration": "Pulse penetration",
    "ahn4_sigma_z": "Sigma-z",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel(axis: plt.Axes, index: int, x: float = -0.13, y: float = 1.04) -> None:
    axis.text(
        x,
        y,
        string.ascii_lowercase[index],
        transform=axis.transAxes,
        fontsize=10.5,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def save(fig: plt.Figure, stem: str) -> None:
    save_figure(fig, stem, OUT, dpi=360)
    plt.close(fig)


def add_scale_bar(axis: plt.Axes, length_m: float, label: str) -> None:
    xmin, xmax = axis.get_xlim()
    ymin, ymax = axis.get_ylim()
    x0 = xmin + 0.07 * (xmax - xmin)
    y0 = ymin + 0.055 * (ymax - ymin)
    axis.plot([x0, x0 + length_m], [y0, y0], color="black", linewidth=2.0, solid_capstyle="butt")
    axis.text(x0 + length_m / 2, y0 + 0.018 * (ymax - ymin), label, ha="center", va="bottom")


def satellite_context(
    bounds: tuple[float, float, float, float],
    source_crs: str,
    year: int,
    zoom: int = 14,
) -> np.ndarray:
    """Return a cropped EOX Sentinel-2 cloudless image for projected bounds."""
    xmin, xmax, ymin, ymax = bounds
    transformer = Transformer.from_crs(source_crs, "EPSG:3857", always_xy=True)
    corners = [
        transformer.transform(x, y)
        for x, y in [(xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)]
    ]
    mercator_bounds = (
        min(point[0] for point in corners),
        max(point[0] for point in corners),
        min(point[1] for point in corners),
        max(point[1] for point in corners),
    )

    half_world = 20037508.342789244
    tile_span = (2 * half_world) / (2**zoom)
    mxmin, mxmax, mymin, mymax = mercator_bounds
    col_min = int(np.floor((mxmin + half_world) / tile_span))
    col_max = int(np.floor((mxmax + half_world) / tile_span))
    row_min = int(np.floor((half_world - mymax) / tile_span))
    row_max = int(np.floor((half_world - mymin) / tile_span))

    tile_size = 256
    mosaic = Image.new(
        "RGB",
        ((col_max - col_min + 1) * tile_size, (row_max - row_min + 1) * tile_size),
    )
    SATELLITE_CACHE.mkdir(parents=True, exist_ok=True)
    for row in range(row_min, row_max + 1):
        for col in range(col_min, col_max + 1):
            cache_path = SATELLITE_CACHE / f"s2cloudless_{year}_z{zoom}_x{col}_y{row}.jpg"
            if not cache_path.exists():
                url = (
                    "https://tiles.maps.eox.at/wmts/1.0.0/"
                    f"s2cloudless-{year}_3857/default/g/{zoom}/{row}/{col}.jpg"
                )
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                cache_path.write_bytes(response.content)
            with Image.open(cache_path) as tile:
                mosaic.paste(
                    tile.convert("RGB"),
                    ((col - col_min) * tile_size, (row - row_min) * tile_size),
                )

    mosaic_bounds = (
        col_min * tile_span - half_world,
        (col_max + 1) * tile_span - half_world,
        half_world - (row_max + 1) * tile_span,
        half_world - row_min * tile_span,
    )
    width, height = mosaic.size
    left = int(np.floor((mxmin - mosaic_bounds[0]) / (mosaic_bounds[1] - mosaic_bounds[0]) * width))
    right = int(np.ceil((mxmax - mosaic_bounds[0]) / (mosaic_bounds[1] - mosaic_bounds[0]) * width))
    top = int(np.floor((mosaic_bounds[3] - mymax) / (mosaic_bounds[3] - mosaic_bounds[2]) * height))
    bottom = int(
        np.ceil((mosaic_bounds[3] - mymin) / (mosaic_bounds[3] - mosaic_bounds[2]) * height)
    )
    return np.asarray(mosaic.crop((left, top, right, bottom)))


def add_country_locator(
    axis: plt.Axes,
    world: gpd.GeoDataFrame,
    country: str,
    longitude: float,
    latitude: float,
    extent: tuple[float, float, float, float],
) -> None:
    """Add a restrained country locator without creating another panel."""
    locator = axis.inset_axes([0.755, 0.735, 0.225, 0.225])
    xmin, xmax, ymin, ymax = extent
    region = world.cx[xmin:xmax, ymin:ymax]
    region.plot(ax=locator, facecolor="#f2f2ef", edgecolor="#a7aca8", linewidth=0.35)
    world[world["ADMIN"] == country].plot(
        ax=locator,
        facecolor="#c9ddd2",
        edgecolor="#4c6d60",
        linewidth=0.55,
    )
    locator.scatter(
        longitude, latitude, s=10, color=RED, edgecolor="white", linewidth=0.4, zorder=5
    )
    locator.set_xlim(xmin, xmax)
    locator.set_ylim(ymin, ymax)
    locator.set_aspect("equal")
    locator.set_xticks([])
    locator.set_yticks([])
    for spine in locator.spines.values():
        spine.set_color("#8f9692")
        spine.set_linewidth(0.45)


def small_map_scale(width_m: float) -> float:
    """Return a readable scale-bar length occupying roughly one quarter of a map."""
    target = max(width_m * 0.26, 1.0)
    magnitude = 10 ** np.floor(np.log10(target))
    candidates = np.asarray([1.0, 2.0, 5.0, 10.0]) * magnitude
    return float(candidates[candidates <= target][-1])


def site_codes(sites: pd.DataFrame) -> dict[str, str]:
    counters = {"conifer": 0, "broadleaf": 0}
    codes: dict[str, str] = {}
    for row in sites.sort_values("selection_order").itertuples(index=False):
        counters[row.forest_group] += 1
        prefix = "C" if row.forest_group == "conifer" else "B"
        codes[str(row.site_name)] = f"{prefix}{counters[row.forest_group]}"
    return codes


def site_label_offset(code: str) -> tuple[float, float, str]:
    """Keep labels for the tightly clustered southern sites legible."""
    offsets = {
        "B1": (-15000.0, 1800.0, "right"),
        "B9": (3500.0, 6500.0, "left"),
        "B10": (3500.0, -6500.0, "left"),
    }
    return offsets.get(code, (2400.0, 1800.0, "left"))


def selected_dutch_boundaries() -> tuple[pd.DataFrame, gpd.GeoDataFrame, dict[str, str]]:
    sites = pd.read_csv(DUTCH_SITES).sort_values("selection_order").reset_index(drop=True)
    boundaries = gpd.read_file(DUTCH_BOUNDARIES).to_crs("EPSG:28992")
    boundaries = boundaries.dissolve(by="SITENAME").reset_index()
    boundaries = boundaries[boundaries["SITENAME"].isin(sites["site_name"])].copy()
    boundaries = boundaries.merge(
        sites[["site_name", "site_code", "forest_group", "selection_order", "retained_units"]],
        left_on="SITENAME",
        right_on="site_name",
        validate="one_to_one",
    )
    return sites, boundaries, site_codes(sites)


def country_outline() -> gpd.GeoDataFrame:
    world = gpd.read_file(NATURAL_EARTH)
    name_column = "ADMIN" if "ADMIN" in world else "NAME"
    return world[world[name_column] == "Netherlands"].to_crs("EPSG:28992")


def actual_embedding_rgb() -> np.ndarray:
    patches = np.load(PATCH_VALUES, mmap_mode="r")
    scales = np.load(PATCH_SCALES, mmap_mode="r")
    row_ids = np.load(PATCH_IDS, mmap_mode="r")
    cohort = set(pd.read_parquet(CAIRN_TARGETS, columns=["row_id"])["row_id"].astype(int))
    candidates = np.flatnonzero(np.isin(row_ids[::25], list(cohort)))[:500] * 25
    best = int(candidates[0])
    best_score = -np.inf
    for index in candidates:
        values = patches[index].astype(np.float32) * scales[index][None, :, :].astype(np.float32)
        score = float(np.mean(np.std(values, axis=(1, 2))))
        if score > best_score:
            best_score = score
            best = int(index)
    values = patches[best].astype(np.float32) * scales[best][None, :, :].astype(np.float32)
    matrix = values.reshape(128, 25).T
    matrix -= matrix.mean(axis=0, keepdims=True)
    _, _, vectors = np.linalg.svd(matrix, full_matrices=False)
    rgb = (matrix @ vectors[:3].T).reshape(5, 5, 3)
    for channel in range(3):
        low, high = np.percentile(rgb[:, :, channel], [2, 98])
        rgb[:, :, channel] = np.clip((rgb[:, :, channel] - low) / max(high - low, 1e-6), 0, 1)
    return rgb


def height_components(frame: pd.DataFrame, targets: list[str]) -> np.ndarray:
    all_targets = ADJUSTED_TARGETS
    values = np.full((len(frame), len(all_targets)), np.nan, dtype=np.float32)
    owner = np.zeros(len(frame), dtype=np.int8)
    for fold in range(5):
        with np.load(HEIGHT_DIR / f"fold_{fold}.npz") as source:
            test = source["test_indices"].astype(np.int64)
            values[test] = source["height_test_predictions"].astype(np.float32)
            owner[test] += 1
    if not np.all(owner == 1) or not np.isfinite(values).all():
        raise RuntimeError("Height-adjustment components do not cover each row exactly once")
    return values[:, [all_targets.index(target) for target in targets]]


def core_predictions(variant: str) -> pd.DataFrame:
    predictions = pd.read_parquet(CAIRN_PREDICTIONS)
    return predictions[
        (predictions["variant"] == variant)
        & (predictions["version"] == "v2")
        & (predictions["model_family"] == "tessera")
        & predictions["target"].isin(CORE_TARGETS)
    ].copy()


def macro_row(target: str, variant: str) -> pd.Series:
    macro = pd.read_csv(CAIRN_MACRO)
    return macro[
        (macro["variant"] == variant)
        & (macro["version"] == "v2")
        & (macro["model_family"] == "tessera")
        & (macro["target"] == target)
    ].iloc[0]


def observed_prediction(axis: plt.Axes, data: pd.DataFrame, target: str, variant: str) -> None:
    image = axis.hexbin(
        data["observed"],
        data["predicted"],
        gridsize=43,
        bins="log",
        mincnt=1,
        cmap="viridis" if variant == "raw" else "magma",
        linewidths=0,
    )
    low = float(min(data["observed"].min(), data["predicted"].min()))
    high = float(max(data["observed"].max(), data["predicted"].max()))
    axis.plot(
        [low, high],
        [low, high],
        linestyle="--",
        color=RED if variant == "raw" else BLUE,
        linewidth=0.9,
    )
    axis.set_xlim(low, high)
    axis.set_ylim(low, high)
    axis.set_aspect("equal", adjustable="box")
    score = macro_row(target, variant)
    axis.text(
        0.04,
        0.95,
        f"{LABELS[target]}\n$R^2$ = {float(score['r2']):.2f}; RMSE = {float(score['rmse']):.3f}",
        transform=axis.transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84, "pad": 2.0},
    )
    axis._density_artist = image  # type: ignore[attr-defined]


def make_landscape_figure() -> None:
    cairn = pd.read_parquet(CAIRN_TARGETS)
    savelsbos = pd.read_parquet(SAVELSBOS_TARGETS)
    world = gpd.read_file(NATURAL_EARTH).to_crs("EPSG:4326")
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.7))
    cairn_axis, savelsbos_axis, cairn_detail, savelsbos_detail = axes.flat

    # Bounds follow the outside edges of the 50 m units, keeping the inset
    # frames clear of the outermost displayed cells.
    cairn_zoom = (281110.0, 283110.0, 799520.0, 801520.0)
    savelsbos_zoom = (179842.19, 181842.19, 312983.87, 314983.87)

    def draw_units(
        axis: plt.Axes,
        frame: pd.DataFrame,
        x_column: str,
        y_column: str,
        value_column: str,
        letter_index: int,
        scale_m: float,
        bounds: tuple[float, float, float, float] | None = None,
        point_size: float = 3.0,
    ) -> mpl.collections.PathCollection:
        if bounds is None:
            xmin = float(frame[x_column].min())
            xmax = float(frame[x_column].max())
            ymin = float(frame[y_column].min())
            ymax = float(frame[y_column].max())
        else:
            xmin, xmax, ymin, ymax = bounds
        plotted = frame
        if bounds is not None:
            plotted = frame[
                frame[x_column].between(xmin, xmax) & frame[y_column].between(ymin, ymax)
            ]
        span = 1.18 * max(xmax - xmin, ymax - ymin)
        xmid = 0.5 * (xmin + xmax)
        ymid = 0.5 * (ymin + ymax)
        axis.set_xlim(xmid - span / 2, xmid + span / 2)
        axis.set_ylim(ymid - 0.56 * span, ymid + 0.44 * span)
        axis.set_aspect("equal")
        axis.set_axis_off()
        image = axis.scatter(
            plotted[x_column],
            plotted[y_column],
            c=plotted[value_column],
            cmap="viridis",
            vmin=0.12,
            vmax=0.95,
            s=point_size,
            marker="s",
            linewidths=0,
            rasterized=True,
        )
        x0 = xmid - 0.43 * span
        y0 = ymid - 0.485 * span
        axis.plot([x0, x0 + scale_m], [y0, y0], color="black", linewidth=1.4, solid_capstyle="butt")
        scale_label = f"{scale_m / 1000:g} km" if scale_m >= 1000 else f"{scale_m:g} m"
        axis.text(
            x0 + scale_m / 2, y0 - 0.018 * span, scale_label, ha="center", va="top", fontsize=6.0
        )
        axis.text(
            0.01,
            0.99,
            string.ascii_lowercase[letter_index],
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9.2,
            fontweight="bold",
        )
        return image

    def draw_satellite_units(
        axis: plt.Axes,
        frame: pd.DataFrame,
        x_column: str,
        y_column: str,
        bounds: tuple[float, float, float, float],
        source_crs: str,
        year: int,
        letter_index: int,
    ) -> None:
        xmin, xmax, ymin, ymax = bounds
        context = satellite_context(bounds, source_crs, year)
        axis.imshow(context, extent=(xmin, xmax, ymin, ymax), origin="upper", zorder=0)
        plotted = frame[frame[x_column].between(xmin, xmax) & frame[y_column].between(ymin, ymax)]
        axis.scatter(
            plotted[x_column],
            plotted[y_column],
            marker="s",
            s=9.0,
            facecolors="none",
            edgecolors="white",
            linewidths=0.28,
            alpha=0.82,
            rasterized=True,
            zorder=2,
        )
        axis.set_xlim(xmin, xmax)
        axis.set_ylim(ymin, ymax)
        axis.set_aspect("equal")
        axis.set_axis_off()
        axis.text(
            0.01,
            0.99,
            string.ascii_lowercase[letter_index],
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9.2,
            fontweight="bold",
            color="black",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.2},
            zorder=5,
        )
        x0 = xmin + 0.06 * (xmax - xmin)
        y0 = ymin + 0.055 * (ymax - ymin)
        axis.plot(
            [x0, x0 + 500], [y0, y0], color="white", linewidth=2.4, solid_capstyle="butt", zorder=5
        )
        axis.plot(
            [x0, x0 + 500], [y0, y0], color="black", linewidth=0.75, solid_capstyle="butt", zorder=6
        )
        axis.text(
            x0 + 250,
            y0 + 0.025 * (ymax - ymin),
            "500 m",
            ha="center",
            va="bottom",
            fontsize=6.0,
            color="black",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.74, "pad": 0.8},
            zorder=6,
        )

    image = draw_units(
        cairn_axis,
        cairn,
        "bng_x",
        "bng_y",
        "canopy_surface_cv",
        0,
        10000.0,
        point_size=3.0,
    )
    draw_units(
        savelsbos_axis,
        savelsbos,
        "rd_x",
        "rd_y",
        "ahn4_height_cv",
        1,
        1000.0,
        point_size=10.0,
    )
    draw_satellite_units(cairn_detail, cairn, "bng_x", "bng_y", cairn_zoom, "EPSG:27700", 2023, 2)
    draw_satellite_units(
        savelsbos_detail, savelsbos, "rd_x", "rd_y", savelsbos_zoom, "EPSG:28992", 2021, 3
    )

    cairn_lon, cairn_lat = Transformer.from_crs(
        "EPSG:27700", "EPSG:4326", always_xy=True
    ).transform(float(cairn["bng_x"].median()), float(cairn["bng_y"].median()))
    savelsbos_lon, savelsbos_lat = Transformer.from_crs(
        "EPSG:28992", "EPSG:4326", always_xy=True
    ).transform(float(savelsbos["rd_x"].median()), float(savelsbos["rd_y"].median()))
    add_country_locator(
        cairn_axis, world, "United Kingdom", cairn_lon, cairn_lat, (-11.0, 3.0, 49.0, 60.8)
    )
    add_country_locator(
        savelsbos_axis, world, "Netherlands", savelsbos_lon, savelsbos_lat, (3.0, 7.7, 49.9, 54.0)
    )

    for axis, bounds in [(cairn_axis, cairn_zoom), (savelsbos_axis, savelsbos_zoom)]:
        axis.add_patch(
            Rectangle(
                (bounds[0], bounds[2]),
                bounds[1] - bounds[0],
                bounds[3] - bounds[2],
                fill=False,
                edgecolor=RED,
                linewidth=1.0,
                zorder=5,
            )
        )
    for axis, bounds in [(cairn_detail, cairn_zoom), (savelsbos_detail, savelsbos_zoom)]:
        axis.add_patch(
            Rectangle(
                (bounds[0], bounds[2]),
                bounds[1] - bounds[0],
                bounds[3] - bounds[2],
                fill=False,
                edgecolor=RED,
                linewidth=0.9,
                zorder=5,
            )
        )

    for full_axis, detail_axis, bounds in [
        (cairn_axis, cairn_detail, cairn_zoom),
        (savelsbos_axis, savelsbos_detail, savelsbos_zoom),
    ]:
        fig.add_artist(
            ConnectionPatch(
                xyA=(bounds[0], bounds[2]),
                coordsA=full_axis.transData,
                xyB=(bounds[0], bounds[3]),
                coordsB=detail_axis.transData,
                color=RED,
                linewidth=0.65,
                alpha=0.85,
            )
        )
        fig.add_artist(
            ConnectionPatch(
                xyA=(bounds[1], bounds[2]),
                coordsA=full_axis.transData,
                xyB=(bounds[1], bounds[3]),
                coordsB=detail_axis.transData,
                color=RED,
                linewidth=0.65,
                alpha=0.85,
            )
        )

    color_axis = fig.add_axes([0.938, 0.20, 0.014, 0.60])
    colorbar = fig.colorbar(image, cax=color_axis, extend="both")
    colorbar.set_label("LiDAR-observed canopy-height CV")
    fig.subplots_adjust(left=0.025, right=0.91, top=0.985, bottom=0.02, hspace=0.08, wspace=0.08)
    save(fig, "fig01_study_landscapes")


def make_workflow_figure() -> None:
    # Keep the source width close to its intended display width so labels stay legible.
    fig, axis = plt.subplots(figsize=(8.5, 3.0))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    centers = [0.105, 0.300, 0.495, 0.700, 0.905]
    left_edges = [0.015, 0.215, 0.405, 0.600, 0.815]
    headings = [
        "Aligned data",
        "50 m representation",
        "Spatial holdouts",
        "Model families",
        "Targets and evaluation",
    ]
    for index, (x, left, heading) in enumerate(zip(centers, left_edges, headings, strict=True)):
        axis.text(
            left,
            0.850,
            string.ascii_lowercase[index],
            ha="left",
            va="top",
            fontsize=10.5,
            fontweight="bold",
        )
        axis.text(x, 0.965, heading, ha="center", va="top", fontsize=9.0, fontweight="bold")

    for left, right in zip(centers[:-1], centers[1:], strict=True):
        axis.add_patch(
            mpl.patches.FancyArrowPatch(
                (left + 0.084, 0.535),
                (right - 0.084, 0.535),
                arrowstyle="-|>",
                mutation_scale=9,
                linewidth=0.9,
                color="#686d70",
            )
        )

    # Real aligned inputs: a PCA display of one 128-channel embedding patch and
    # a representative three-stratum LiDAR profile from the prepared cohort.
    patch_axis = axis.inset_axes([0.025, 0.405, 0.070, 0.285])
    patch_axis.imshow(actual_embedding_rgb(), interpolation="nearest")
    patch_axis.set_xticks([])
    patch_axis.set_yticks([])
    for spine in patch_axis.spines.values():
        spine.set_visible(True)
        spine.set_color(CHARCOAL)
        spine.set_linewidth(0.7)
    axis.text(0.060, 0.725, "Tessera v2", ha="center", va="bottom", fontsize=7.2, fontweight="bold")
    axis.text(0.060, 0.375, "128 channels\n$5\\times5$ pixels", ha="center", va="top", fontsize=6.7)

    profiles = np.load(PROFILE_VALUES, mmap_mode="r")
    profile_ids = np.load(PROFILE_IDS, mmap_mode="r")
    cohort_ids = pd.read_parquet(CAIRN_TARGETS, columns=["row_id"])["row_id"].to_numpy(
        dtype=np.int64
    )
    matched = np.isin(profile_ids, cohort_ids)
    profile_pool = np.asarray(profiles[matched], dtype=np.float64)
    clipped = np.clip(profile_pool, 1e-9, 1.0)
    representative = profile_pool[np.argmax(-(clipped * np.log(clipped)).sum(axis=1))]
    profile_axis = axis.inset_axes([0.112, 0.405, 0.074, 0.285])
    levels = np.arange(3)
    profile_axis.bar(levels, representative, color=["#b9d8bf", GREEN, "#235640"], width=0.68)
    profile_axis.set_xticks(levels, ["0-1", "1-10", ">10"])
    profile_axis.set_ylim(0, 0.5)
    profile_axis.set_xlabel("height (m)", fontsize=6.1, labelpad=1)
    profile_axis.tick_params(axis="both", labelsize=5.8, length=2)
    profile_axis.spines["top"].set_visible(False)
    profile_axis.spines["right"].set_visible(False)
    axis.text(
        0.149, 0.725, "LiDAR profile", ha="center", va="bottom", fontsize=7.2, fontweight="bold"
    )
    axis.text(
        0.105,
        0.205,
        "Conventional benchmark:\nSentinel-1, Sentinel-2 and terrain",
        ha="center",
        va="center",
        fontsize=6.8,
    )

    # Matched 10 m pixels within one non-overlapping 50 m analysis unit.
    support_values = actual_embedding_rgb().mean(axis=2)
    support_axis = axis.inset_axes([0.246, 0.420, 0.108, 0.300])
    support_axis.imshow(support_values, cmap="Greens", vmin=0, vmax=1, interpolation="nearest")
    support_axis.set_xticks(np.arange(-0.5, 5, 1), minor=True)
    support_axis.set_yticks(np.arange(-0.5, 5, 1), minor=True)
    support_axis.grid(which="minor", color="white", linewidth=0.8)
    support_axis.tick_params(
        which="both", bottom=False, left=False, labelbottom=False, labelleft=False
    )
    for spine in support_axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#235640")
        spine.set_linewidth(1.2)
    support_axis.text(
        2, 2, "50 m", color="white", fontsize=8.0, fontweight="bold", ha="center", va="center"
    )
    axis.text(centers[1], 0.395, "$5\\times5$ pixels at 10 m", ha="center", va="top", fontsize=6.4)
    axis.text(
        centers[1],
        0.365,
        "centre  |  mean  |  SD\nE-W and N-S gradients",
        ha="center",
        va="top",
        fontsize=6.8,
    )
    axis.text(
        centers[1],
        0.215,
        "5 summaries $\\times$ 128\n= 640 MLP predictors",
        ha="center",
        va="center",
        fontsize=6.8,
    )

    # Actual Cairngorms held-out units, coloured by deterministic fold.
    fold_colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#b279a2"]
    cohort = pd.read_parquet(CAIRN_TARGETS).reset_index(drop=True)
    fold_owner = np.full(len(cohort), -1, dtype=np.int8)
    for fold in range(5):
        with np.load(HEIGHT_DIR / f"fold_{fold}.npz") as source:
            fold_owner[source["test_indices"].astype(np.int64)] = fold
    if np.any(fold_owner < 0):
        raise RuntimeError("Every Cairngorms unit must belong to one held-out fold")
    fold_axis = axis.inset_axes([0.418, 0.315, 0.153, 0.475])
    fold_axis.scatter(
        cohort["bng_x"],
        cohort["bng_y"],
        c=fold_owner,
        cmap=mpl.colors.ListedColormap(fold_colors),
        vmin=-0.5,
        vmax=4.5,
        marker="s",
        s=1.5,
        linewidths=0,
        rasterized=True,
    )
    fold_axis.set_aspect("equal")
    fold_axis.axis("off")
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            markersize=4.2,
            markerfacecolor=color,
            markeredgewidth=0,
            label=str(index + 1),
        )
        for index, color in enumerate(fold_colors)
    ]
    fold_axis.legend(
        handles=legend_handles,
        title="fold",
        ncol=5,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        frameon=False,
        handletextpad=0.1,
        columnspacing=0.45,
        borderaxespad=0,
        fontsize=5.8,
        title_fontsize=6.0,
    )
    axis.text(
        centers[2],
        0.235,
        "6 km test blocks\n2 km train-test exclusion",
        ha="center",
        va="center",
        fontsize=6.8,
    )

    # Three model families actually used in the study.
    model_x0, model_x1 = 0.622, 0.790
    row_y = [0.680, 0.505, 0.315]
    for y in row_y:
        axis.plot([model_x0, model_x1], [y - 0.080, y - 0.080], color="#e2e4e3", linewidth=0.7)

    axis.text(model_x0, row_y[0], "Ridge", ha="left", va="center", fontsize=7.4, fontweight="bold")
    axis.text(
        model_x0,
        row_y[0] - 0.042,
        "conventional and transfer baseline",
        ha="left",
        va="center",
        fontsize=6.3,
    )
    beta_x = model_x1 - 0.037
    axis.text(beta_x - 0.036, row_y[0], r"$\mathbf{x}$", ha="center", va="center", fontsize=8.0)
    axis.add_patch(
        mpl.patches.FancyArrowPatch(
            (beta_x - 0.025, row_y[0]),
            (beta_x - 0.006, row_y[0]),
            arrowstyle="->",
            mutation_scale=7,
            linewidth=0.7,
            color=CHARCOAL,
        )
    )
    axis.text(
        beta_x + 0.002,
        row_y[0],
        r"$\hat{\beta}$",
        ha="center",
        va="center",
        fontsize=8.0,
        color=ORANGE,
    )
    axis.add_patch(
        mpl.patches.FancyArrowPatch(
            (beta_x + 0.012, row_y[0]),
            (beta_x + 0.031, row_y[0]),
            arrowstyle="->",
            mutation_scale=7,
            linewidth=0.7,
            color=CHARCOAL,
        )
    )
    axis.text(beta_x + 0.040, row_y[0], r"$\hat{y}$", ha="center", va="center", fontsize=8.0)

    axis.text(model_x0, row_y[1], "MLP", ha="left", va="center", fontsize=7.4, fontweight="bold")
    axis.text(
        model_x0, row_y[1] - 0.042, "primary Tessera model", ha="left", va="center", fontsize=6.3
    )
    node_x = [model_x1 - 0.090, model_x1 - 0.048, model_x1 - 0.008]
    node_y = [
        [row_y[1] - 0.025, row_y[1], row_y[1] + 0.025],
        [row_y[1] - 0.017, row_y[1] + 0.017],
        [row_y[1]],
    ]
    for first_x, first_y, second_x, second_y in zip(
        node_x[:-1], node_y[:-1], node_x[1:], node_y[1:], strict=True
    ):
        for y0 in first_y:
            for y1 in second_y:
                axis.plot([first_x, second_x], [y0, y1], color="#aeb3b1", linewidth=0.35, zorder=1)
    for column, (x, ys) in enumerate(zip(node_x, node_y, strict=True)):
        for y in ys:
            axis.add_patch(
                mpl.patches.Circle(
                    (x, y),
                    0.0060,
                    facecolor=[GREEN, BLUE, PURPLE][column],
                    edgecolor="white",
                    linewidth=0.4,
                    zorder=2,
                )
            )

    axis.text(model_x0, row_y[2], "U-Net", ha="left", va="center", fontsize=7.4, fontweight="bold")
    axis.text(
        model_x0,
        row_y[2] - 0.042,
        "height-only context sensitivity",
        ha="left",
        va="center",
        fontsize=6.3,
    )
    unet_x = [
        model_x1 - 0.098,
        model_x1 - 0.070,
        model_x1 - 0.042,
        model_x1 - 0.014,
        model_x1 + 0.014,
    ]
    unet_y = [row_y[2] + 0.025, row_y[2], row_y[2] - 0.025, row_y[2], row_y[2] + 0.025]
    for index, (x, y) in enumerate(zip(unet_x, unet_y, strict=True)):
        size = 0.020 - 0.002 * min(index, 4 - index)
        axis.add_patch(
            Rectangle(
                (x - size / 2, y - size / 2),
                size,
                size,
                facecolor=[GREEN, BLUE, PURPLE, BLUE, GREEN][index],
                edgecolor="white",
                linewidth=0.5,
            )
        )
        if index:
            axis.plot(
                unet_x[index - 1 : index + 1],
                unet_y[index - 1 : index + 1],
                color=CHARCOAL,
                linewidth=0.65,
                zorder=0,
            )
    axis.plot(
        [unet_x[0], unet_x[4]],
        [unet_y[0] + 0.019, unet_y[4] + 0.019],
        color="#89918d",
        linewidth=0.5,
    )
    axis.plot(
        [unet_x[1], unet_x[3]],
        [unet_y[1] + 0.017, unet_y[3] + 0.017],
        color="#89918d",
        linewidth=0.5,
    )

    # Outcomes are grouped by ecological dimension.
    box_x, box_w = 0.830, 0.150
    axis.add_patch(
        Rectangle(
            (box_x, 0.535), box_w, 0.225, facecolor="#eef5f1", edgecolor="#7d9e90", linewidth=0.8
        )
    )
    axis.text(
        box_x + 0.008,
        0.728,
        "Outer-canopy surface",
        ha="left",
        va="top",
        fontsize=7.2,
        fontweight="bold",
        color="#235640",
    )
    axis.text(box_x + 0.008, 0.685, "p95 height  |  SD  |  CV", ha="left", va="top", fontsize=6.4)
    axis.text(box_x + 0.008, 0.642, "robust CV  |  rumple", ha="left", va="top", fontsize=6.4)
    axis.text(box_x + 0.008, 0.599, "openings  |  kurtosis", ha="left", va="top", fontsize=6.4)

    axis.add_patch(
        Rectangle(
            (box_x, 0.315), box_w, 0.175, facecolor="#eef2f6", edgecolor="#7890a3", linewidth=0.8
        )
    )
    axis.text(
        box_x + 0.008,
        0.458,
        "Return-height distribution",
        ha="left",
        va="top",
        fontsize=7.2,
        fontweight="bold",
        color="#315b7c",
    )
    axis.text(
        box_x + 0.008, 0.414, "CV  |  robust CV  |  RMS height", ha="left", va="top", fontsize=6.2
    )
    axis.text(box_x + 0.008, 0.371, "Shannon entropy", ha="left", va="top", fontsize=6.4)
    axis.text(
        box_x + box_w / 2,
        0.245,
        r"original + height-adjusted",
        ha="center",
        va="center",
        fontsize=6.8,
    )
    axis.text(
        box_x + box_w / 2,
        0.185,
        r"held-out RMSE, $R^2$ and Spearman $\rho$",
        ha="center",
        va="center",
        fontsize=7.0,
        fontweight="bold",
    )

    fig.subplots_adjust(left=0.008, right=0.995, top=0.99, bottom=0.02)
    save(fig, "fig02_study_design")


def make_core_prediction_figure() -> None:
    targets = pd.read_parquet(CAIRN_TARGETS)
    predictions = core_predictions("raw")
    fig = plt.figure(figsize=(11.2, 6.6))
    grid = fig.add_gridspec(2, 3, hspace=0.36, wspace=0.34)
    density = None
    for index, target in enumerate(CORE_TARGETS):
        cell = grid[0, index].subgridspec(1, 2, width_ratios=[1, 0.05], wspace=0.08)
        axis = fig.add_subplot(cell[0, 0])
        color_axis = fig.add_subplot(cell[0, 1])
        color_axis.set_axis_off()
        data = predictions[predictions["target"] == target]
        observed_prediction(axis, data, target, "raw")
        axis.set_xlabel(f"LiDAR observed ({UNITS[target]})")
        axis.set_ylabel(f"Tessera predicted ({UNITS[target]})")
        density = axis._density_artist  # type: ignore[attr-defined]
        panel(axis, index)
        if index == len(CORE_TARGETS) - 1 and density is not None:
            color_axis.set_axis_on()
            color_axis.tick_params(labelsize=7.2)
            color_axis.spines[:].set_visible(True)
            colorbar = fig.colorbar(density, cax=color_axis)
            colorbar.set_label("Observations per hexagon (log scale)")

    block = targets["spatial_block"].value_counts().index[0]
    spatial = targets.loc[
        targets["spatial_block"] == block, ["row_id", "bng_x", "bng_y", "canopy_rumple"]
    ]
    pred = predictions[predictions["target"] == "canopy_rumple"][["row_id", "predicted"]]
    spatial = spatial.merge(pred, on="row_id", validate="one_to_one")
    spatial["residual"] = spatial["predicted"] - spatial["canopy_rumple"]
    low, high = np.quantile(
        np.concatenate([spatial["canopy_rumple"], spatial["predicted"]]), [0.01, 0.99]
    )
    residual_limit = float(np.quantile(np.abs(spatial["residual"]), 0.99))
    specs = [
        ("canopy_rumple", "viridis", low, high, "LiDAR-observed rumple"),
        ("predicted", "viridis", low, high, "Tessera-predicted rumple"),
        ("residual", "RdBu_r", -residual_limit, residual_limit, "Prediction minus observation"),
    ]
    for offset, (column, cmap, lower, upper, label) in enumerate(specs):
        cell = grid[1, offset].subgridspec(1, 2, width_ratios=[1, 0.05], wspace=0.08)
        axis = fig.add_subplot(cell[0, 0])
        color_axis = fig.add_subplot(cell[0, 1])
        image = axis.scatter(
            spatial["bng_x"],
            spatial["bng_y"],
            c=spatial[column],
            cmap=cmap,
            vmin=lower,
            vmax=upper,
            s=8,
            marker="s",
            linewidths=0,
        )
        axis.set_aspect("equal")
        axis.set_xlabel("Easting (km)")
        axis.xaxis.set_major_formatter(lambda value, _: f"{value / 1000:.1f}")
        axis.yaxis.set_major_formatter(lambda value, _: f"{value / 1000:.1f}")
        axis.xaxis.set_major_locator(MaxNLocator(4))
        axis.yaxis.set_major_locator(MaxNLocator(4))
        axis.set_ylabel("Northing (km)" if offset == 0 else "")
        colorbar = fig.colorbar(image, cax=color_axis)
        colorbar.set_label(label)
        panel(axis, offset + 3)
    fig.subplots_adjust(left=0.065, right=0.975, top=0.97, bottom=0.08)
    save(fig, "fig03_cairngorms_core_predictions")


def make_height_adjustment_figure() -> None:
    targets = pd.read_parquet(CAIRN_TARGETS).reset_index(drop=True)
    predictions = core_predictions("height_adjusted")
    diagnostics = pd.read_csv(HEIGHT_DIAGNOSTICS).set_index("target")
    components = height_components(targets, CORE_TARGETS)
    fig = plt.figure(figsize=(11.2, 6.7))
    grid = fig.add_gridspec(2, 3, hspace=0.36, wspace=0.34)
    mean_height = targets["canopy_mean_height_m"].to_numpy(float)
    p95_height = targets["canopy_p95_height_m"].to_numpy(float)
    for index, target in enumerate(CORE_TARGETS):
        cell = grid[0, index].subgridspec(1, 2, width_ratios=[1, 0.05], wspace=0.08)
        axis = fig.add_subplot(cell[0, 0])
        color_axis = fig.add_subplot(cell[0, 1])
        fitted = components[:, index]
        lower, upper = np.quantile(fitted, [0.01, 0.99])
        image = axis.hexbin(
            mean_height,
            p95_height,
            C=fitted,
            reduce_C_function=np.mean,
            gridsize=40,
            mincnt=1,
            cmap="viridis",
            vmin=lower,
            vmax=upper,
            linewidths=0,
        )
        row = diagnostics.loc[target]
        removed = 100.0 * max(0.0, float(row["variance_fraction_removed_by_observed_height"]))
        axis.text(
            0.04,
            0.95,
            (
                f"{LABELS[target]}\n"
                f"height-only $R^2$ = {float(row['height_only_r2']):.2f}\n"
                f"variance removed = {removed:.0f}%"
            ),
            transform=axis.transAxes,
            va="top",
            ha="left",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84, "pad": 2.0},
        )
        axis.set_xlabel("Mean canopy height (m)")
        axis.set_ylabel("P95 canopy height (m)")
        colorbar = fig.colorbar(image, cax=color_axis)
        colorbar.set_label("Fitted height component")
        panel(axis, index)

    density = None
    for index, target in enumerate(CORE_TARGETS):
        cell = grid[1, index].subgridspec(1, 2, width_ratios=[1, 0.05], wspace=0.08)
        axis = fig.add_subplot(cell[0, 0])
        color_axis = fig.add_subplot(cell[0, 1])
        color_axis.set_axis_off()
        data = predictions[predictions["target"] == target]
        observed_prediction(axis, data, target, "height_adjusted")
        axis.set_xlabel("Observed height-adjusted value")
        axis.set_ylabel("Predicted height-adjusted value")
        density = axis._density_artist  # type: ignore[attr-defined]
        panel(axis, index + 3)
        if index == len(CORE_TARGETS) - 1 and density is not None:
            color_axis.set_axis_on()
            color_axis.tick_params(labelsize=7.2)
            color_axis.spines[:].set_visible(True)
            colorbar = fig.colorbar(density, cax=color_axis)
            colorbar.set_label("Observations per hexagon (log scale)")
    fig.subplots_adjust(left=0.065, right=0.975, top=0.97, bottom=0.08)
    save(fig, "fig04_height_adjustment")


def relative_rmse() -> pd.DataFrame:
    metrics = pd.read_csv(DUTCH_METRICS)
    metrics = metrics[metrics["model"] == "ridge"].copy()
    grouped = metrics.groupby(["target_site", "target", "scenario"], as_index=False)["rmse"].mean()
    local = grouped[grouped["scenario"] == "local_cv"].rename(columns={"rmse": "local_rmse"})
    comparison = grouped[grouped["scenario"] != "local_cv"].merge(
        local[["target_site", "target", "local_rmse"]],
        on=["target_site", "target"],
        validate="many_to_one",
    )
    comparison["relative_rmse"] = comparison["rmse"] / comparison["local_rmse"]
    return comparison


def plot_dutch_units(axis: plt.Axes) -> None:
    if not DUTCH_COHORT.exists():
        raise FileNotFoundError(
            f"Exact eligible-unit map requires the frozen cohort: {DUTCH_COHORT}"
        )
    units = pd.read_parquet(DUTCH_COHORT)
    sites = pd.read_csv(DUTCH_SITES).sort_values("selection_order").reset_index(drop=True)
    codes = site_codes(sites)
    group_by_code = sites.set_index("site_code")["forest_group"].to_dict()
    if "forest_group" not in units:
        units["forest_group"] = units["site_code"].map(group_by_code)
    country_outline().plot(ax=axis, facecolor="#f4f4f1", edgecolor="#85857f", linewidth=0.6)
    colors = {"conifer": GREEN, "broadleaf": ORANGE}
    for group in ["conifer", "broadleaf"]:
        subset = units[units["forest_group"] == group]
        axis.scatter(
            subset["rd_x"],
            subset["rd_y"],
            s=0.75,
            marker="s",
            color=colors[group],
            linewidths=0,
            alpha=0.78,
            rasterized=True,
        )
    for row in sites.itertuples(index=False):
        color = colors[str(row.forest_group)]
        code = codes[str(row.site_name)]
        dx, dy, horizontal_alignment = site_label_offset(code)
        axis.scatter(
            row.centroid_x,
            row.centroid_y,
            s=12,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            zorder=4,
        )
        axis.text(
            row.centroid_x + dx,
            row.centroid_y + dy,
            code,
            fontsize=5.6,
            color="#252525",
            ha=horizontal_alignment,
            va="bottom",
            fontweight="bold",
            path_effects=[path_effects.withStroke(linewidth=1.6, foreground="white")],
            zorder=5,
        )
    axis.set_xlim(0, 290000)
    axis.set_ylim(295000, 620000)
    axis.set_aspect("equal")
    axis.set_xlabel("RD easting (km)")
    axis.set_ylabel("RD northing (km)")
    axis.xaxis.set_major_formatter(lambda value, _: f"{value / 1000:.0f}")
    axis.yaxis.set_major_formatter(lambda value, _: f"{value / 1000:.0f}")
    add_scale_bar(axis, 50000, "50 km")
    axis.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="s",
                linestyle="none",
                markersize=5,
                color=GREEN,
                label="Conifer-dominated units",
            ),
            Line2D(
                [],
                [],
                marker="s",
                linestyle="none",
                markersize=5,
                color=ORANGE,
                label="Broadleaf-dominated units",
            ),
        ],
        frameon=True,
        facecolor="white",
        edgecolor="none",
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        ncol=1,
    )


def plot_pairwise_matrix(axis: plt.Axes) -> None:
    sites = pd.read_csv(DUTCH_SITES).sort_values("selection_order")
    pair = pd.read_csv(DUTCH_PAIRWISE)
    pair = pair.groupby(["source_site", "target_site"], as_index=False)["relative_rmse"].mean()
    order = sites["site_code"].astype(str).tolist()
    codes_by_site = {
        row.site_code: ("C" if row.forest_group == "conifer" else "B") + str(index + 1)
        for group in ["conifer", "broadleaf"]
        for index, row in enumerate(sites[sites["forest_group"] == group].itertuples(index=False))
    }
    matrix = pair.pivot(index="source_site", columns="target_site", values="relative_rmse").reindex(
        index=order, columns=order
    )
    values = matrix.to_numpy(dtype=float, copy=True)
    np.fill_diagonal(values, 1.0)
    vmax = float(np.nanquantile(values, 0.95))
    image = axis.imshow(values, cmap="magma", norm=LogNorm(vmin=0.8, vmax=max(vmax, 2.0)))
    axis.set_xticks(range(len(order)), [codes_by_site[value] for value in order], rotation=90)
    axis.set_yticks(range(len(order)), [codes_by_site[value] for value in order])
    axis.tick_params(labelsize=5.6)
    axis.set_xlabel("Target forest")
    axis.set_ylabel("Source forest")
    axis.axvline(9.5, color="white", linewidth=1.2)
    axis.axhline(9.5, color="white", linewidth=1.2)
    colorbar = axis.figure.colorbar(image, ax=axis, fraction=0.045, pad=0.03)
    colorbar.set_label("Transfer RMSE / local-model RMSE")


def plot_dutch_unit_grid(
    fig: plt.Figure, grid: mpl.gridspec.GridSpecFromSubplotSpec
) -> mpl.cm.ScalarMappable:
    """Draw the 20 Dutch cohorts as equally sized local maps."""
    units = pd.read_parquet(DUTCH_COHORT)
    if "ahn4_height_cv" not in units:
        raise RuntimeError("Rebuild the Dutch eligible-unit table before drawing the forest maps")
    sites = pd.read_csv(DUTCH_SITES).sort_values("selection_order").reset_index(drop=True)
    codes = site_codes(sites)
    values = units["ahn4_height_cv"].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    lower, upper = np.quantile(finite, [0.02, 0.98])
    norm = mpl.colors.Normalize(vmin=float(lower), vmax=float(upper), clip=True)

    image = None
    for axis, row in zip(
        (fig.add_subplot(cell) for cell in grid),
        sites.itertuples(index=False),
        strict=True,
    ):
        subset = units[units["site_name"] == row.site_name]
        count = len(subset)
        size = 3.8 if count < 800 else 1.8 if count < 2500 else 0.7
        image = axis.scatter(
            subset["rd_x"],
            subset["rd_y"],
            c=subset["ahn4_height_cv"],
            cmap="viridis",
            norm=norm,
            s=size,
            marker="s",
            linewidths=0,
            rasterized=True,
        )
        xmin, xmax = subset["rd_x"].min(), subset["rd_x"].max()
        ymin, ymax = subset["rd_y"].min(), subset["rd_y"].max()
        width = max(float(xmax - xmin), 500.0)
        height = max(float(ymax - ymin), 500.0)
        span = 1.22 * max(width, height)
        xmid = 0.5 * float(xmin + xmax)
        ymid = 0.5 * float(ymin + ymax)
        axis.set_xlim(xmid - span / 2, xmid + span / 2)
        axis.set_ylim(ymid - 0.56 * span, ymid + 0.44 * span)
        axis.set_aspect("equal")
        axis.set_axis_off()

        scale = small_map_scale(span)
        x0 = xmid - 0.42 * span
        y0 = ymid - 0.485 * span
        axis.plot([x0, x0 + scale], [y0, y0], color="black", linewidth=0.9, solid_capstyle="butt")
        label = f"{scale / 1000:g} km" if scale >= 1000 else f"{scale:g} m"
        axis.text(x0 + scale / 2, y0 - 0.018 * span, label, ha="center", va="top", fontsize=4.8)
        axis.text(
            0.02,
            0.98,
            codes[str(row.site_name)],
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=6.6,
            fontweight="bold",
            path_effects=[path_effects.withStroke(linewidth=1.5, foreground="white")],
        )

    if image is None:
        raise RuntimeError("The Dutch eligible-unit cohort is empty")
    return image


def make_dutch_transfer_figure() -> None:
    relative = relative_rmse()
    fig = plt.figure(figsize=(11.4, 9.2))
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=(1.48, 1.0),
        hspace=0.25,
        left=0.045,
        right=0.93,
        top=0.975,
        bottom=0.075,
    )

    map_grid = outer[0].subgridspec(4, 5, hspace=0.08, wspace=0.06)
    image = plot_dutch_unit_grid(fig, map_grid)
    fig.text(0.018, 0.978, "a", fontsize=10.5, fontweight="bold", ha="left", va="top")
    color_axis = fig.add_axes([0.945, 0.565, 0.012, 0.315])
    colorbar = fig.colorbar(image, cax=color_axis, extend="both")
    colorbar.set_label("AHN4 return-height CV")

    lower = outer[1].subgridspec(1, 2, width_ratios=(0.82, 1.18), wspace=0.30)
    axis = fig.add_subplot(lower[0, 0])
    plot_pairwise_matrix(axis)
    panel(axis, 1)

    axis = fig.add_subplot(lower[0, 1])
    scenarios = ["all_other", "same_group", "different_group"]
    scenario_labels = ["All other forests", "Same group", "Other group"]
    colors = [CHARCOAL, GREEN, ORANGE]
    rng = np.random.default_rng(20260901)
    for position, (scenario, color) in enumerate(zip(scenarios, colors, strict=True)):
        values = (
            relative.loc[relative["scenario"] == scenario, "relative_rmse"].dropna().to_numpy(float)
        )
        violin = axis.violinplot(values, positions=[position], widths=0.72, showextrema=False)
        for body in violin["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor("none")
            body.set_alpha(0.28)
        q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75])
        axis.plot(
            [position, position], [q25, q75], color=color, linewidth=4.5, solid_capstyle="butt"
        )
        axis.scatter(
            position, median, color="white", edgecolor=color, linewidth=1.1, s=30, zorder=4
        )
        sample = rng.choice(values, size=min(70, len(values)), replace=False)
        axis.scatter(
            rng.normal(position, 0.055, len(sample)),
            sample,
            color=color,
            s=6,
            alpha=0.22,
            linewidths=0,
        )
    axis.axhline(1.0, linestyle="--", color="black", linewidth=0.8)
    axis.set_xticks(range(3), scenario_labels)
    axis.set_ylabel("Transfer RMSE / local-model RMSE")
    axis.grid(axis="y", color=LIGHT_GREY, linewidth=0.6)
    panel(axis, 2)
    save(fig, "fig05_dutch_transferability")


def make_dutch_unit_small_multiples() -> None:
    units = pd.read_parquet(DUTCH_COHORT)
    if "ahn4_height_cv" not in units:
        raise RuntimeError("Rebuild the Dutch eligible-unit table before drawing the forest maps")
    sites = pd.read_csv(DUTCH_SITES).sort_values("selection_order").reset_index(drop=True)
    values = units["ahn4_height_cv"].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    lower, upper = np.quantile(finite, [0.02, 0.98])
    norm = mpl.colors.Normalize(vmin=float(lower), vmax=float(upper), clip=True)

    fig, axes = plt.subplots(4, 5, figsize=(11.4, 8.35))
    image = None
    for index, (axis, row) in enumerate(zip(axes.flat, sites.itertuples(index=False), strict=True)):
        subset = units[units["site_name"] == row.site_name]
        count = len(subset)
        size = 4.5 if count < 800 else 2.2 if count < 2500 else 0.9
        image = axis.scatter(
            subset["rd_x"],
            subset["rd_y"],
            c=subset["ahn4_height_cv"],
            cmap="viridis",
            norm=norm,
            s=size,
            marker="s",
            linewidths=0,
            rasterized=True,
        )
        xmin, xmax = subset["rd_x"].min(), subset["rd_x"].max()
        ymin, ymax = subset["rd_y"].min(), subset["rd_y"].max()
        width = max(float(xmax - xmin), 500.0)
        height = max(float(ymax - ymin), 500.0)
        span = 1.28 * max(width, height)
        xmid = 0.5 * float(xmin + xmax)
        ymid = 0.5 * float(ymin + ymax)
        axis.set_xlim(xmid - span / 2, xmid + span / 2)
        # Reserve a consistent lower gutter for the scale bar and its label.
        axis.set_ylim(ymid - 0.56 * span, ymid + 0.44 * span)
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_axis_off()

        scale = small_map_scale(span)
        x0 = xmid - 0.43 * span
        y0 = ymid - 0.485 * span
        axis.plot([x0, x0 + scale], [y0, y0], color="black", linewidth=1.1, solid_capstyle="butt")
        label = f"{scale / 1000:g} km" if scale >= 1000 else f"{scale:g} m"
        axis.text(
            x0 + scale / 2,
            y0 - 0.018 * span,
            label,
            ha="center",
            va="top",
            fontsize=5.4,
        )
        axis.text(
            0.01,
            0.99,
            chr(ord("a") + index),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8.2,
            fontweight="bold",
        )

    if image is not None:
        color_axis = fig.add_axes([0.946, 0.18, 0.012, 0.64])
        colorbar = fig.colorbar(image, cax=color_axis, extend="both")
        colorbar.set_label("AHN4 return-height CV")
    fig.subplots_adjust(left=0.025, right=0.925, top=0.985, bottom=0.02, hspace=0.14, wspace=0.08)
    save(fig, "figS07_dutch_unit_cv")


def make_dutch_rank_reference_figure() -> None:
    rank = pd.read_csv(RANK_FOLD_METRICS)
    rank = rank[(rank["budget"] == 0) & (rank["method"] == "source_ridge")].copy()
    curve = pd.read_csv(REFERENCE_CURVE)
    fig, axes = plt.subplots(1, 2, figsize=(10.9, 4.15))

    axis = axes[0]
    colors = {"conifer": GREEN, "broadleaf": ORANGE}
    for group in ["conifer", "broadleaf"]:
        subset = rank[rank["forest_group"] == group]
        axis.scatter(
            subset["r2"],
            subset["unit_spearman"],
            s=12,
            color=colors[group],
            alpha=0.38,
            linewidths=0,
            rasterized=True,
            label=f"{group.capitalize()} target forest",
        )
    axis.axvline(0, color=CHARCOAL, linewidth=0.8, linestyle="--")
    axis.axhline(0, color=CHARCOAL, linewidth=0.8, linestyle="--")
    axis.set_xscale("symlog", linthresh=0.2, linscale=0.8)
    axis.set_xlim(-6.2, 0.85)
    axis.set_ylim(-0.18, 0.95)
    axis.set_xlabel("Exact-value performance ($R^2$)")
    axis.set_ylabel("Within-forest 50 m rank correlation")
    axis.grid(color=LIGHT_GREY, linewidth=0.55)
    axis.legend(frameon=False, loc="lower right")
    panel(axis, 0)

    axis = axes[1]
    matched = curve[curve["cohort"] == "matched_200_eligible_folds"].copy()
    if matched.empty:
        matched = curve[curve["cohort"] == "all_site_folds_to_100"].copy()
    budgets = [0, 2, 5, 10, 25, 50, 100, 200]
    positions = {budget: index for index, budget in enumerate(budgets)}
    methods = {
        "source_ridge": ("Transferred model", CHARCOAL),
        "source_offset": ("Transfer + local mean correction", GREEN),
        "target_only_ridge": ("Locally fitted model", ORANGE),
    }
    for method, (label, color) in methods.items():
        subset = matched[matched["method"] == method]
        summary = (
            subset.groupby("budget", as_index=False)["r2_median"].median().sort_values("budget")
        )
        if method == "source_ridge":
            axis.axhline(
                float(summary.iloc[0]["r2_median"]), color=color, linewidth=1.5, label=label
            )
        else:
            x = [positions[int(value)] for value in summary["budget"]]
            axis.plot(
                x,
                summary["r2_median"],
                marker="o",
                markersize=3.7,
                linewidth=1.5,
                color=color,
                label=label,
            )
    axis.axhline(0, color="black", linewidth=0.7, linestyle="--")
    axis.set_xticks(range(len(budgets)), [str(value) for value in budgets])
    axis.set_xlabel("Local LiDAR reference units from target forest")
    axis.set_ylabel("Median $R^2$ across six outcomes")
    axis.legend(frameon=False, loc="lower right")
    axis.grid(axis="y", color=LIGHT_GREY, linewidth=0.6)
    panel(axis, 1)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.97, bottom=0.16, wspace=0.27)
    save(fig, "fig06_dutch_rank_reference")


def make_full_prediction_figure(variant: str, targets: list[str], stem: str) -> None:
    predictions = pd.read_parquet(CAIRN_PREDICTIONS)
    selected = predictions[
        (predictions["variant"] == variant)
        & (predictions["version"] == "v2")
        & (predictions["model_family"] == "tessera")
        & predictions["target"].isin(targets)
    ]
    columns = 4 if len(targets) > 6 else 3
    rows = int(np.ceil(len(targets) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(11.3, 3.4 * rows))
    axes = np.atleast_1d(axes).ravel()
    density = None
    for index, target in enumerate(targets):
        axis = axes[index]
        data = selected[selected["target"] == target]
        observed_prediction(axis, data, target, variant)
        axis.set_xlabel("Observed")
        axis.set_ylabel("Predicted")
        density = axis._density_artist  # type: ignore[attr-defined]
        panel(axis, index)
    for axis in axes[len(targets) :]:
        axis.axis("off")
    if density is not None:
        color_axis = fig.add_axes([0.935, 0.35, 0.012, 0.30])
        colorbar = fig.colorbar(density, cax=color_axis)
        colorbar.set_label("Observations per hexagon (log scale)")
    fig.subplots_adjust(left=0.06, right=0.91, top=0.98, bottom=0.08, hspace=0.34, wspace=0.28)
    save(fig, stem)


def make_combined_prediction_figure() -> None:
    predictions = pd.read_parquet(CAIRN_PREDICTIONS)
    predictions = predictions[
        (predictions["version"] == "v2") & (predictions["model_family"] == "tessera")
    ].copy()
    raw_targets = [
        "canopy_p95_height_m",
        "canopy_surface_sd_m",
        "canopy_surface_rcv",
        "canopy_height_kurtosis",
    ]
    adjusted_targets = [
        "canopy_surface_sd_m",
        "canopy_surface_rcv",
        "canopy_height_kurtosis",
    ]
    fig = plt.figure(figsize=(11.3, 6.2))
    grid = fig.add_gridspec(2, 8, hspace=0.42, wspace=0.58)
    raw_axes = [fig.add_subplot(grid[0, 2 * index : 2 * index + 2]) for index in range(4)]
    adjusted_axes = [fig.add_subplot(grid[1, start : start + 2]) for start in (1, 3, 5)]

    raw_density = None
    for index, target in enumerate(raw_targets):
        axis = raw_axes[index]
        data = predictions[(predictions["variant"] == "raw") & (predictions["target"] == target)]
        observed_prediction(axis, data, target, "raw")
        axis.set_xlabel("Observed")
        axis.set_ylabel("Predicted")
        raw_density = axis._density_artist  # type: ignore[attr-defined]
        panel(axis, index)
    adjusted_density = None
    for offset, target in enumerate(adjusted_targets):
        axis = adjusted_axes[offset]
        data = predictions[
            (predictions["variant"] == "height_adjusted") & (predictions["target"] == target)
        ]
        observed_prediction(axis, data, target, "height_adjusted")
        axis.set_xlabel("Observed height-adjusted value")
        axis.set_ylabel("Predicted height-adjusted value")
        adjusted_density = axis._density_artist  # type: ignore[attr-defined]
        panel(axis, offset + len(raw_targets))

    if raw_density is not None:
        raw_color_axis = fig.add_axes([0.936, 0.615, 0.012, 0.245])
        raw_colorbar = fig.colorbar(raw_density, cax=raw_color_axis)
        raw_colorbar.set_label("Observations per hexagon (log scale)")
    if adjusted_density is not None:
        adjusted_color_axis = fig.add_axes([0.936, 0.155, 0.012, 0.245])
        adjusted_colorbar = fig.colorbar(adjusted_density, cax=adjusted_color_axis)
        adjusted_colorbar.set_label("Observations per hexagon (log scale)")
    fig.subplots_adjust(left=0.065, right=0.91, top=0.97, bottom=0.11)
    save(fig, "figS04_all_predictions")


def make_target_distribution_figure() -> None:
    frame = pd.read_parquet(CAIRN_TARGETS)
    fig, axes = plt.subplots(2, 4, figsize=(11.2, 5.8))
    for index, (axis, target) in enumerate(zip(axes.ravel(), DIST_TARGETS, strict=True)):
        values = frame[target].dropna().to_numpy(float)
        lower, upper = np.quantile(values, [0.005, 0.995])
        grid = np.linspace(lower, upper, 300)
        density = gaussian_kde(values)(grid)
        axis.fill_between(grid, density, color=GREEN, alpha=0.25, linewidth=0)
        axis.plot(grid, density, color=GREEN, linewidth=1.2)
        axis.axvline(np.median(values), color=CHARCOAL, linestyle="--", linewidth=0.8)
        axis.set_xlabel(f"{LABELS[target]} ({UNITS[target]})")
        axis.set_ylabel("Density")
        axis.set_yticks([])
        panel(axis, index)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.97, bottom=0.10, hspace=0.42, wspace=0.30)
    save(fig, "figS01_target_distributions")


def make_reference_uncertainty_figure() -> None:
    frame = pd.read_csv(REFERENCE_UNCERTAINTY)
    order = [
        "canopy_p95_height_m",
        "canopy_surface_sd_m",
        "canopy_surface_cv",
        "canopy_surface_rcv",
        "canopy_rumple",
        "canopy_open_fraction",
        "canopy_height_kurtosis",
    ]
    frame = frame.set_index("metric").reindex(order).reset_index()
    names = [LABELS[value] for value in order]
    positions = np.arange(len(frame))
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    axes[0].scatter(frame["median_se_to_between_sd"], positions, color=GREEN, s=34)
    p95_ratio = frame["jackknife_se_p95"] / frame["between_unit_sd"]
    for y, median, p95 in zip(positions, frame["median_se_to_between_sd"], p95_ratio, strict=True):
        axes[0].plot([median, p95], [y, y], color=GREEN, linewidth=1.3)
    axes[0].set_yticks(positions, names)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Jackknife SE / between-unit SD")
    axes[0].grid(axis="x", color=LIGHT_GREY, linewidth=0.6)
    panel(axes[0], 0)

    axes[1].barh(positions, frame["rms_jackknife_se_to_model_rmse"], color=PURPLE, alpha=0.82)
    axes[1].axvline(1.0, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_yticks(positions, names)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("RMS jackknife SE / held-out model RMSE")
    axes[1].grid(axis="x", color=LIGHT_GREY, linewidth=0.6)
    panel(axes[1], 1)
    fig.subplots_adjust(left=0.18, right=0.98, top=0.96, bottom=0.16, wspace=0.48)
    save(fig, "figS02_reference_stability")


def make_version_comparison_figure() -> None:
    frame = pd.read_csv(VERSION_COMPARISON)
    frame = frame[frame["model_family"].eq("tessera")].copy()
    orders = {
        "raw": RAW_TARGETS,
        "height_adjusted": ADJUSTED_TARGETS,
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.9, 3.8))
    for index, (axis, variant) in enumerate(zip(axes, ["raw", "height_adjusted"], strict=True)):
        subset = frame[frame["variant"].eq(variant)].set_index("target").reindex(orders[variant])
        positions = np.arange(len(subset))
        width = 0.36
        axis.bar(positions - width / 2, subset["v1_r2"], width, color=GREEN, label="Tessera v1.0")
        axis.bar(positions + width / 2, subset["v2_r2"], width, color=ORANGE, label="Tessera v2")
        axis.axhline(0, color=CHARCOAL, linewidth=0.7)
        axis.set_xticks(
            positions, [LABELS[value] for value in orders[variant]], rotation=28, ha="right"
        )
        axis.set_ylabel("Spatially held-out $R^2$")
        axis.grid(axis="y", color=LIGHT_GREY, linewidth=0.6)
        panel(axis, index)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02)
    )
    fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.30, wspace=0.24)
    save(fig, "figS06_version_comparison")


def make_pairwise_matrix() -> None:
    sites = pd.read_csv(DUTCH_SITES).sort_values("selection_order")
    pair = pd.read_csv(DUTCH_PAIRWISE)
    pair = pair.groupby(["source_site", "target_site"], as_index=False)["relative_rmse"].mean()
    order = sites["site_code"].astype(str).tolist()
    codes_by_site = {
        row.site_code: ("C" if row.forest_group == "conifer" else "B") + str(index + 1)
        for group in ["conifer", "broadleaf"]
        for index, row in enumerate(sites[sites["forest_group"] == group].itertuples(index=False))
    }
    matrix = pair.pivot(index="source_site", columns="target_site", values="relative_rmse").reindex(
        index=order, columns=order
    )
    values = matrix.to_numpy(dtype=float, copy=True)
    np.fill_diagonal(values, 1.0)
    vmax = float(np.nanquantile(values, 0.95))
    fig, axis = plt.subplots(figsize=(6.9, 6.1))
    image = axis.imshow(values, cmap="magma", norm=LogNorm(vmin=0.8, vmax=max(vmax, 2.0)))
    axis.set_xticks(range(len(order)), [codes_by_site[value] for value in order], rotation=90)
    axis.set_yticks(range(len(order)), [codes_by_site[value] for value in order])
    axis.set_xlabel("Target forest")
    axis.set_ylabel("Source forest")
    axis.axvline(9.5, color="white", linewidth=1.2)
    axis.axhline(9.5, color="white", linewidth=1.2)
    colorbar = fig.colorbar(image, ax=axis, fraction=0.045, pad=0.03)
    colorbar.set_label("Transfer RMSE / local-model RMSE")
    panel(axis, 0, x=-0.11, y=1.02)
    fig.subplots_adjust(left=0.13, right=0.90, top=0.98, bottom=0.13)
    save(fig, "figS03_dutch_pairwise_transfer")


def main() -> None:
    configure_style()
    make_landscape_figure()
    make_core_prediction_figure()
    make_height_adjustment_figure()
    make_dutch_transfer_figure()
    make_dutch_rank_reference_figure()
    make_combined_prediction_figure()
    make_reference_uncertainty_figure()
    make_version_comparison_figure()
    print(f"Wrote revised figures to {OUT}")


if __name__ == "__main__":
    main()
