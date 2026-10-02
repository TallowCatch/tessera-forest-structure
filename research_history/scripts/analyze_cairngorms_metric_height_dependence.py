#!/usr/bin/env python3
"""Diagnose how Cairngorms structural metrics relate to canopy height.

The analysis uses the frozen 50 m Cairngorms cohort, Aland's corrected
point-return raster, and the already-frozen TESSERA v2 model results. It
reports both plot-level associations and associations among 1 km block
medians so spatially clustered observations are not presented as independent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
SURFACE_TARGETS = ROOT / "data/processed/phase22_cairngorms_surface_targets.parquet"
CORRECTED_RASTER = ROOT / "data/raw/scotland_lidar/LiDAR_metrics_3.tif"
CORRECTED_TARGETS = (
    ROOT / "data/processed/phase25_cairngorms_corrected_lidar_targets.parquet"
)
SURFACE_RESULTS = ROOT / "outputs/tables/phase29_cairngorms_v2_surface_macro.csv"
CORRECTED_RESULTS = ROOT / "outputs/tables/phase27_cairngorms_v2_metrics.csv"
TABLE_DIR = ROOT / "outputs/tables"
FIGURE_DIR = ROOT / "outputs/figures"
REPORT_DIR = ROOT / "outputs/reports"

BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 20260816


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    source: str
    structural_dimension: str
    frame_name: str
    value_column: str
    mean_height_column: str
    p95_height_column: str
    result_target: str
    plot_scale: float = 1.0
    plot_offset: float = 0.0
    plot_ylabel: str = ""


METRICS = [
    Metric(
        "surface_sd",
        "Canopy-height SD",
        "1 m canopy-height model",
        "Canopy-surface variation",
        "surface",
        "canopy_surface_sd_m",
        "canopy_mean_height_m",
        "canopy_p95_height_m",
        "canopy_surface_sd_m",
        plot_ylabel="Canopy-height SD (m)",
    ),
    Metric(
        "surface_cv",
        "Canopy-height CV",
        "1 m canopy-height model",
        "Height-scaled surface variation",
        "surface",
        "canopy_surface_cv",
        "canopy_mean_height_m",
        "canopy_p95_height_m",
        "canopy_surface_cv",
        plot_ylabel="Canopy-height CV",
    ),
    Metric(
        "surface_rcv",
        "Robust canopy-height CV",
        "1 m canopy-height model",
        "Robust height-scaled variation",
        "surface",
        "canopy_surface_rcv",
        "canopy_mean_height_m",
        "canopy_p95_height_m",
        "canopy_surface_rcv",
        plot_ylabel="Robust canopy-height CV",
    ),
    Metric(
        "rumple",
        "Canopy rumple",
        "1 m canopy-height model",
        "Canopy-top surface roughness",
        "surface",
        "canopy_rumple",
        "canopy_mean_height_m",
        "canopy_p95_height_m",
        "canopy_rumple",
        plot_ylabel="Canopy rumple (surface-area ratio)",
    ),
    Metric(
        "openings",
        "Canopy openings",
        "1 m canopy-height model",
        "Open space below 2 m",
        "surface",
        "canopy_open_fraction",
        "canopy_mean_height_m",
        "canopy_p95_height_m",
        "canopy_open_fraction",
        plot_ylabel="Canopy opening fraction",
    ),
    Metric(
        "kurtosis",
        "Canopy-height kurtosis",
        "1 m canopy-height model",
        "Canopy-height distribution shape",
        "surface",
        "canopy_height_kurtosis",
        "canopy_mean_height_m",
        "canopy_p95_height_m",
        "canopy_height_kurtosis",
        plot_ylabel="Canopy-height kurtosis",
    ),
    Metric(
        "return_cv",
        "Return-height CV",
        "Corrected point-return metrics (>1.3 m)",
        "Height-scaled return variation",
        "corrected",
        "lidar_height_cv_50m",
        "lidar_mean_height_50m",
        "lidar_p95_height_50m",
        "lidar_height_cv_50m",
        plot_ylabel="Return-height CV",
    ),
    Metric(
        "return_rcv",
        "Robust return-height CV",
        "Corrected point-return metrics (>1.3 m)",
        "Robust height-scaled return variation",
        "corrected",
        "lidar_rcv_50m",
        "lidar_mean_height_50m",
        "lidar_p95_height_50m",
        "lidar_rcv_50m",
        plot_ylabel="Robust return-height CV",
    ),
    Metric(
        "return_rms",
        "Return-height RMS",
        "Corrected point-return metrics (>1.3 m)",
        "Absolute return-height spread",
        "corrected",
        "lidar_rms_50m",
        "lidar_mean_height_50m",
        "lidar_p95_height_50m",
        "lidar_rms_50m",
        plot_ylabel="Return-height RMS (m)",
    ),
    Metric(
        "shannon",
        "Return-height Shannon entropy",
        "Corrected point-return metrics (>1.3 m)",
        "Vertical return distribution",
        "corrected",
        "lidar_canopy_shannon_50m",
        "lidar_mean_height_50m",
        "lidar_p95_height_50m",
        "lidar_canopy_shannon_50m",
        plot_ylabel="Normalized Shannon entropy",
    ),
]


def configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def aggregate_raster_band(
    dataset: rasterio.DatasetReader,
    band_index: int,
    windows: list[tuple[slice, slice]],
) -> tuple[np.ndarray, np.ndarray]:
    values = dataset.read(band_index)
    means = np.full(len(windows), np.nan, dtype=np.float32)
    counts = np.zeros(len(windows), dtype=np.int16)
    for index, window in enumerate(windows):
        local = np.asarray(values[window], dtype=np.float64)
        valid = np.isfinite(local)
        counts[index] = int(valid.sum())
        if counts[index] >= 24:
            means[index] = float(local[valid].mean())
    return means, counts


def corrected_target_frame(surface: pd.DataFrame) -> pd.DataFrame:
    """Recreate the frozen corrected target table if storage cleanup removed it."""
    if CORRECTED_TARGETS.exists():
        frame = pd.read_parquet(CORRECTED_TARGETS)
        if len(frame) != 16_831:
            raise RuntimeError(
                f"Corrected target cache has {len(frame):,} rows; expected 16,831"
            )
        return frame

    source_bands = {
        "lidar_mean_height_50m": "lidar_meanH",
        "lidar_p95_height_50m": "lidar_p_95",
        "lidar_height_cv_50m": "lidar_height_cv",
        "lidar_rcv_50m": "lidar_rcv",
        "lidar_rms_50m": "lidar_rms",
        "lidar_canopy_shannon_50m": "lidar_canopy_shannon",
    }
    frame = surface.copy()
    with rasterio.open(CORRECTED_RASTER) as dataset:
        descriptions = {str(name): i for i, name in enumerate(dataset.descriptions, 1)}
        missing = sorted(set(source_bands.values()) - set(descriptions))
        if missing:
            raise RuntimeError(f"Corrected raster is missing bands: {missing}")
        windows: list[tuple[slice, slice]] = []
        for row in frame.itertuples(index=False):
            window = from_bounds(
                float(row.bng_x) - 25.0,
                float(row.bng_y) - 25.0,
                float(row.bng_x) + 25.0,
                float(row.bng_y) + 25.0,
                dataset.transform,
            ).round_offsets().round_lengths()
            if int(window.height) != 5 or int(window.width) != 5:
                raise RuntimeError("A 50 m unit did not cover exactly 5 x 5 raster cells")
            row_start = int(window.row_off)
            column_start = int(window.col_off)
            windows.append(
                (
                    slice(row_start, row_start + 5),
                    slice(column_start, column_start + 5),
                )
            )
        for output_name, source_name in source_bands.items():
            means, counts = aggregate_raster_band(
                dataset, descriptions[source_name], windows
            )
            frame[output_name] = means
            frame[f"{output_name}_valid_10m_cells"] = counts
            print(f"Aggregated corrected band: {source_name}", flush=True)

    primary = [
        "lidar_height_cv_50m",
        "lidar_rcv_50m",
        "lidar_rms_50m",
        "lidar_canopy_shannon_50m",
    ]
    controls = ["lidar_mean_height_50m", "lidar_p95_height_50m"]
    valid = np.all(np.isfinite(frame[primary + controls].to_numpy()), axis=1)
    ranges = {
        "lidar_height_cv_50m": (0.0, 5.0),
        "lidar_rcv_50m": (0.0, 10.0),
        "lidar_rms_50m": (0.0, 60.0),
        "lidar_canopy_shannon_50m": (0.0, 1.61),
    }
    for column, (low, high) in ranges.items():
        valid &= frame[column].between(low, high, inclusive="both").to_numpy()
    frame = frame.loc[valid].reset_index(drop=True)
    if len(frame) != 16_831:
        raise RuntimeError(
            f"Corrected target reconstruction produced {len(frame):,} rows; "
            "the frozen analysis used 16,831"
        )
    CORRECTED_TARGETS.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(CORRECTED_TARGETS, index=False, compression="zstd")
    return frame


def block_association(
    frame: pd.DataFrame,
    value_column: str,
    height_column: str,
    rng: np.random.Generator,
) -> dict[str, float]:
    local = frame[["spatial_block", value_column, height_column]].dropna()
    plot_rho = float(spearmanr(local[value_column], local[height_column]).statistic)
    block = (
        local.groupby("spatial_block", observed=True)[[value_column, height_column]]
        .median()
        .reset_index(drop=True)
    )
    block_rho = float(spearmanr(block[value_column], block[height_column]).statistic)
    values = block[[value_column, height_column]].to_numpy(dtype=np.float64)
    boot = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for replicate in range(BOOTSTRAP_REPLICATES):
        selected = rng.integers(0, len(values), size=len(values))
        boot[replicate] = float(
            spearmanr(values[selected, 0], values[selected, 1]).statistic
        )
    low, high = np.nanquantile(boot, [0.025, 0.975])
    return {
        "n_50m_units": int(len(local)),
        "n_1km_blocks": int(len(block)),
        "plot_spearman": plot_rho,
        "block_spearman": block_rho,
        "block_ci_low": float(low),
        "block_ci_high": float(high),
    }


def association_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    records: list[dict[str, object]] = []
    for metric in METRICS:
        frame = frames[metric.frame_name]
        for height_name, height_column in (
            ("Mean canopy height", metric.mean_height_column),
            ("P95 canopy height", metric.p95_height_column),
        ):
            result = block_association(frame, metric.value_column, height_column, rng)
            records.append(
                {
                    "metric_key": metric.key,
                    "metric": metric.label,
                    "source": metric.source,
                    "structural_dimension": metric.structural_dimension,
                    "height_measure": height_name,
                    **result,
                }
            )
    return pd.DataFrame.from_records(records)


def performance_table() -> pd.DataFrame:
    surface = pd.read_csv(SURFACE_RESULTS)
    surface = surface[
        (surface["variant"] == "raw")
        & (surface["version"] == "v2")
        & (surface["model_family"] == "tessera")
    ]
    surface_adjusted = pd.read_csv(SURFACE_RESULTS)
    surface_adjusted = surface_adjusted[
        (surface_adjusted["variant"] == "height_adjusted")
        & (surface_adjusted["version"] == "v2")
        & (surface_adjusted["model_family"] == "tessera")
    ]
    corrected = pd.read_csv(CORRECTED_RESULTS)
    corrected = corrected[corrected["model"] == "tessera_v2"]

    records: list[dict[str, object]] = []
    for metric in METRICS:
        if metric.frame_name == "surface":
            raw = surface[surface["target"] == metric.result_target]
            adjusted = surface_adjusted[
                surface_adjusted["target"] == metric.result_target
            ]
        else:
            raw = corrected[corrected["target"] == metric.result_target]
            adjusted = corrected[
                corrected["target"] == f"{metric.result_target}_height_adjusted"
            ]
        if len(raw) != 1 or len(adjusted) != 1:
            raise RuntimeError(f"Missing frozen v2 result for {metric.label}")
        raw_row = raw.iloc[0]
        adjusted_row = adjusted.iloc[0]
        records.append(
            {
                "metric_key": metric.key,
                "metric": metric.label,
                "source": metric.source,
                "structural_dimension": metric.structural_dimension,
                "n": int(raw_row["n"]),
                "blocks_reported_by_evaluation": int(raw_row["blocks"]),
                "raw_v2_rmse": float(raw_row["rmse"]),
                "raw_v2_r2": float(raw_row["r2"]),
                "raw_v2_block_spearman": float(raw_row["block_spearman"]),
                "height_adjusted_v2_rmse": float(adjusted_row["rmse"]),
                "height_adjusted_v2_r2": float(adjusted_row["r2"]),
                "height_adjusted_v2_block_spearman": float(
                    adjusted_row["block_spearman"]
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def evidence_table(
    associations: pd.DataFrame, performance: pd.DataFrame
) -> pd.DataFrame:
    mean = associations[associations["height_measure"] == "Mean canopy height"].copy()
    mean = mean.rename(
        columns={
            "plot_spearman": "plot_spearman_with_mean_height",
            "block_spearman": "block_spearman_with_mean_height",
            "block_ci_low": "mean_height_block_ci_low",
            "block_ci_high": "mean_height_block_ci_high",
        }
    )
    p95 = associations[associations["height_measure"] == "P95 canopy height"].copy()
    p95 = p95.rename(
        columns={
            "plot_spearman": "plot_spearman_with_p95_height",
            "block_spearman": "block_spearman_with_p95_height",
            "block_ci_low": "p95_height_block_ci_low",
            "block_ci_high": "p95_height_block_ci_high",
        }
    )
    keep_mean = [
        "metric_key",
        "plot_spearman_with_mean_height",
        "block_spearman_with_mean_height",
        "mean_height_block_ci_low",
        "mean_height_block_ci_high",
    ]
    keep_p95 = [
        "metric_key",
        "plot_spearman_with_p95_height",
        "block_spearman_with_p95_height",
        "p95_height_block_ci_low",
        "p95_height_block_ci_high",
    ]
    return (
        performance.merge(mean[keep_mean], on="metric_key", validate="one_to_one")
        .merge(p95[keep_p95], on="metric_key", validate="one_to_one")
        .sort_values("metric_key")
        .reset_index(drop=True)
    )


def panel_label(axis: plt.Axes, label: str, title: str) -> None:
    axis.text(
        -0.13,
        1.05,
        label,
        transform=axis.transAxes,
        fontsize=12,
        fontweight="bold",
        va="bottom",
    )
    axis.set_title(title, loc="left", fontweight="bold", pad=9)


def plot_height_diagnostics(evidence: pd.DataFrame) -> None:
    ordered = [metric.key for metric in METRICS]
    local = evidence.set_index("metric_key").loc[ordered].reset_index()
    labels = local["metric"].tolist()
    matrix = local[
        ["block_spearman_with_mean_height", "block_spearman_with_p95_height"]
    ].to_numpy()

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(12.5, 6.5), gridspec_kw={"width_ratios": [0.86, 1.15]}
    )
    image = left.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    left.set_xticks([0, 1], ["Mean height", "P95 height"])
    left.set_yticks(np.arange(len(labels)), labels)
    left.tick_params(length=0)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            color = "white" if abs(value) >= 0.58 else "black"
            left.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=8,
            )
    colorbar = fig.colorbar(
        image, ax=left, orientation="horizontal", fraction=0.05, pad=0.11
    )
    colorbar.set_label("Spearman correlation among 1 km block medians")
    panel_label(left, "a", "Direct association with canopy height")

    y = np.arange(len(local))
    height = 0.34
    right.barh(
        y - height / 2,
        local["raw_v2_r2"],
        height,
        color="#4c78a8",
        label="Original measurement",
    )
    right.barh(
        y + height / 2,
        local["height_adjusted_v2_r2"],
        height,
        color="#2a9d8f",
        label="After accounting for mean and P95 height",
    )
    right.axvline(0, color="#333333", linewidth=0.8)
    right.set_yticks(y, labels)
    right.set_yticklabels([])
    right.tick_params(axis="y", length=0)
    right.invert_yaxis()
    right.set_xlabel(r"Spatially held-out TESSERA v2 $R^2$")
    right.grid(axis="x", color="#dddddd", linewidth=0.6)
    right.set_axisbelow(True)
    right.legend(loc="lower right", frameon=False)
    panel_label(right, "b", "Prediction before and after height adjustment")
    fig.suptitle(
        "Height dependence of Cairngorms forest-structure measurements",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    fig.subplots_adjust(left=0.25, right=0.98, top=0.88, bottom=0.16, wspace=0.33)
    fig.text(
        0.5,
        -0.025,
        "Correlations use 1 km block medians. The two R2 values use different outcomes: "
        "the original measurement and its leakage-safe residual after fitting mean and P95 height in training data only.",
        ha="center",
        va="top",
        fontsize=8,
        color="#444444",
        wrap=True,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURE_DIR / f"cairngorms_metric_height_diagnostics.{suffix}")
    plt.close(fig)


def binned_median(x: np.ndarray, y: np.ndarray, bins: int = 12) -> tuple[np.ndarray, np.ndarray]:
    edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    labels = np.digitize(x, edges[1:-1], right=True)
    x_out: list[float] = []
    y_out: list[float] = []
    for label in range(len(edges) - 1):
        selected = labels == label
        if selected.sum() >= 20:
            x_out.append(float(np.median(x[selected])))
            y_out.append(float(np.median(y[selected])))
    return np.asarray(x_out), np.asarray(y_out)


def plot_examples(
    frames: dict[str, pd.DataFrame], associations: pd.DataFrame
) -> None:
    examples = ["surface_cv", "rumple", "openings", "shannon"]
    metric_lookup = {metric.key: metric for metric in METRICS}
    correlation_lookup = associations[
        associations["height_measure"] == "P95 canopy height"
    ].set_index("metric_key")
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.2), constrained_layout=True)
    for index, (axis, key) in enumerate(zip(axes.flat, examples)):
        metric = metric_lookup[key]
        frame = frames[metric.frame_name]
        local = frame[[metric.p95_height_column, metric.value_column]].dropna()
        x = local[metric.p95_height_column].to_numpy(dtype=np.float64)
        y = (
            local[metric.value_column].to_numpy(dtype=np.float64) + metric.plot_offset
        ) * metric.plot_scale
        artist = axis.hexbin(
            x,
            y,
            gridsize=48,
            bins="log",
            mincnt=1,
            cmap="viridis",
            linewidths=0,
        )
        trend_x, trend_y = binned_median(x, y)
        axis.plot(trend_x, trend_y, color="white", linewidth=3.2, zorder=3)
        axis.plot(trend_x, trend_y, color="#c43c39", linewidth=1.5, zorder=4)
        axis.set_xlabel("LiDAR P95 canopy height (m)")
        axis.set_ylabel(metric.plot_ylabel)
        axis.grid(color="#eeeeee", linewidth=0.5)
        axis.set_axisbelow(True)
        association = correlation_lookup.loc[key]
        axis.text(
            0.03,
            0.95,
            rf"block-median $r_s$ = {association['block_spearman']:.2f}",
            transform=axis.transAxes,
            va="top",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 2},
        )
        panel_label(axis, chr(ord("a") + index), metric.label)
    colorbar = fig.colorbar(artist, ax=axes.ravel().tolist(), shrink=0.72, pad=0.02)
    colorbar.set_label("Observations per hexagon (log scale)")
    fig.suptitle(
        "Examples of structural measurements across the canopy-height gradient",
        fontsize=13,
        fontweight="bold",
    )
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURE_DIR / f"cairngorms_metric_height_examples.{suffix}")
    plt.close(fig)


def write_latex_table(evidence: pd.DataFrame) -> None:
    selected = evidence[
        evidence["metric_key"].isin(
            ["surface_sd", "surface_cv", "rumple", "openings", "shannon"]
        )
    ].copy()
    order = ["surface_sd", "surface_cv", "rumple", "openings", "shannon"]
    selected["metric_key"] = pd.Categorical(selected["metric_key"], order, ordered=True)
    selected = selected.sort_values("metric_key")
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Measurement & $r_s$(mean) & $r_s$(P95) & Raw $R^2$ & Height-adjusted $R^2$ \\",
        r"\midrule",
    ]
    for row in selected.itertuples(index=False):
        label = str(row.metric).replace(">", r"$>$")
        lines.append(
            f"{label} & {row.block_spearman_with_mean_height:.2f} & "
            f"{row.block_spearman_with_p95_height:.2f} & "
            f"{row.raw_v2_r2:.2f} & {row.height_adjusted_v2_r2:.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (TABLE_DIR / "cairngorms_metric_height_diagnostics.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def plot_table(evidence: pd.DataFrame) -> None:
    selected_keys = [
        "surface_sd",
        "surface_cv",
        "surface_rcv",
        "rumple",
        "openings",
        "kurtosis",
        "shannon",
    ]
    local = evidence.set_index("metric_key").loc[selected_keys].reset_index()
    cells = []
    for row in local.itertuples(index=False):
        cells.append(
            [
                row.metric,
                f"{row.block_spearman_with_mean_height:.2f}",
                f"{row.block_spearman_with_p95_height:.2f}",
                f"{row.raw_v2_r2:.2f}",
                f"{row.height_adjusted_v2_r2:.2f}",
            ]
        )
    fig, axis = plt.subplots(figsize=(11.5, 3.6))
    axis.axis("off")
    table = axis.table(
        cellText=cells,
        colLabels=[
            "Measurement",
            "Block $r_s$\nwith mean height",
            "Block $r_s$\nwith P95 height",
            "Original\nv2 $R^2$",
            "Height-adjusted\nv2 $R^2$",
        ],
        colLoc="center",
        cellLoc="center",
        loc="center",
        colWidths=[0.35, 0.16, 0.16, 0.13, 0.16],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#333333")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#eeeeee")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("white")
        if column == 0:
            cell.set_text_props(ha="left")
    axis.set_title(
        "Cairngorms height associations and spatially held-out TESSERA v2 results",
        fontsize=12,
        fontweight="bold",
        pad=14,
    )
    fig.text(
        0.5,
        0.015,
        "Associations use medians within 1 km blocks. Height-adjusted outcomes were constructed within each training fold.",
        ha="center",
        fontsize=8,
        color="#444444",
    )
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURE_DIR / f"cairngorms_metric_height_diagnostics_table.{suffix}")
    plt.close(fig)


def write_summary(evidence: pd.DataFrame) -> None:
    by_key = evidence.set_index("metric_key")
    lines = [
        "# Cairngorms metric-height diagnostics",
        "",
        "This diagnostic uses the frozen 50 m Cairngorms cohort, Aland's corrected "
        "point-return raster and the frozen TESSERA v2 spatial evaluation. Direct "
        "associations are reported at plot scale and among 1 km block medians; the "
        "latter receive spatial-block bootstrap intervals.",
        "",
        "## Main results",
        "",
    ]
    for key in ("surface_cv", "rumple", "openings", "shannon"):
        row = by_key.loc[key]
        lines.append(
            f"- **{row.metric}:** block-level association with P95 height "
            f"$r_s={row.block_spearman_with_p95_height:.2f}$; TESSERA v2 "
            f"$R^2={row.raw_v2_r2:.2f}$ for the original measurement and "
            f"$R^2={row.height_adjusted_v2_r2:.2f}$ after accounting for mean and P95 height."
        )
    lines.extend(
        [
            "",
            "The raw and height-adjusted R2 values describe different response variables, "
            "so their difference is not a formal partition of variance. The adjusted "
            "results instead test whether TESSERA retains predictive information after "
            "the average height relationship is removed using training data only.",
            "",
        ]
    )
    (REPORT_DIR / "cairngorms_metric_height_diagnostics.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def run() -> None:
    configure_plotting()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    surface = pd.read_parquet(SURFACE_TARGETS)
    corrected = corrected_target_frame(surface)
    frames = {"surface": surface, "corrected": corrected}
    associations = association_table(frames)
    performance = performance_table()
    evidence = evidence_table(associations, performance)

    associations.to_csv(
        TABLE_DIR / "cairngorms_metric_height_associations.csv", index=False
    )
    evidence.to_csv(
        TABLE_DIR / "cairngorms_metric_prediction_evidence.csv", index=False
    )
    write_latex_table(evidence)
    plot_height_diagnostics(evidence)
    plot_examples(frames, associations)
    plot_table(evidence)
    write_summary(evidence)
    print(evidence.to_string(index=False), flush=True)


if __name__ == "__main__":
    run()
