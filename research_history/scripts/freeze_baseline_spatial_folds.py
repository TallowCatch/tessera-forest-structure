#!/usr/bin/env python3
"""Estimate FHD variograms and freeze buffered spatial-block folds."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import yaml
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/project.yaml"
DEVELOPMENT_PATH = ROOT / "data/processed/tessera_aligned_development_soap_teak.parquet"
PHASE2_FREEZE_PATH = ROOT / "metadata/tessera_phase2_alignment_freeze.json"
FOLDS_PATH = ROOT / "metadata/phase3_spatial_folds.parquet"
VARIOGRAM_BINS_PATH = ROOT / "metadata/phase3_variogram_bins.csv"
VARIOGRAM_SUMMARY_PATH = ROOT / "metadata/phase3_variogram_summary.csv"
FIGURE_PATH = ROOT / "outputs/figures/phase3_fhd_variograms.png"
FREEZE_PATH = ROOT / "metadata/phase3_spatial_fold_freeze.json"


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


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def spherical_variogram(
    distance: np.ndarray,
    nugget: float,
    partial_sill: float,
    practical_range: float,
) -> np.ndarray:
    ratio = distance / practical_range
    structure = np.where(ratio < 1.0, 1.5 * ratio - 0.5 * ratio**3, 1.0)
    return nugget + partial_sill * structure


def estimate_variogram(
    frame: pd.DataFrame,
    site_id: str,
    bin_width_m: float,
    minimum_pairs: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    coordinates = frame[["x_epsg5070", "y_epsg5070"]].to_numpy(dtype=np.float64)
    values = frame["fhd_normal"].to_numpy(dtype=np.float64)
    distances = pdist(coordinates)
    semivariance = 0.5 * pdist(values[:, None], metric="sqeuclidean")
    maximum_pair_distance = float(distances.max())
    maximum_lag = math.floor((maximum_pair_distance / 2.0) / bin_width_m) * bin_width_m
    edges = np.arange(0.0, maximum_lag + bin_width_m, bin_width_m)
    membership = np.digitize(distances, edges) - 1

    records: list[dict[str, Any]] = []
    for index in range(len(edges) - 1):
        selected = membership == index
        pair_count = int(selected.sum())
        if pair_count < minimum_pairs:
            continue
        records.append(
            {
                "site_id": site_id,
                "lag_bin_start_m": float(edges[index]),
                "lag_bin_end_m": float(edges[index + 1]),
                "mean_lag_m": float(distances[selected].mean()),
                "mean_semivariance": float(semivariance[selected].mean()),
                "pair_count": pair_count,
            }
        )
    bins = pd.DataFrame(records)
    if len(bins) < 8:
        raise RuntimeError(f"Too few populated variogram bins for {site_id}: {len(bins)}")

    sample_variance = float(np.var(values, ddof=1))
    first_semivariance = float(bins.iloc[0]["mean_semivariance"])
    initial = [
        max(0.0, first_semivariance),
        max(sample_variance - first_semivariance, sample_variance * 0.2),
        min(2000.0, maximum_lag),
    ]
    fitted, _ = curve_fit(
        spherical_variogram,
        bins["mean_lag_m"].to_numpy(),
        bins["mean_semivariance"].to_numpy(),
        p0=initial,
        bounds=(
            [0.0, 0.0, bin_width_m],
            [sample_variance * 2.0, sample_variance * 3.0, maximum_lag * 2.0],
        ),
        sigma=1.0 / np.sqrt(bins["pair_count"].to_numpy(dtype=np.float64)),
        absolute_sigma=False,
        maxfev=100_000,
    )
    nugget, partial_sill, practical_range = (float(value) for value in fitted)
    prediction = spherical_variogram(bins["mean_lag_m"].to_numpy(), *fitted)
    weighted_rmse = float(
        np.sqrt(
            np.average(
                (bins["mean_semivariance"].to_numpy() - prediction) ** 2,
                weights=bins["pair_count"].to_numpy(),
            )
        )
    )
    bins["fitted_semivariance"] = prediction
    summary = {
        "site_id": site_id,
        "rows": len(frame),
        "sample_variance": sample_variance,
        "maximum_pair_distance_m": maximum_pair_distance,
        "maximum_fitted_lag_m": maximum_lag,
        "variogram_bin_count": len(bins),
        "nugget": nugget,
        "partial_sill": partial_sill,
        "total_sill": nugget + partial_sill,
        "nugget_fraction": nugget / (nugget + partial_sill),
        "practical_range_m": practical_range,
        "weighted_fit_rmse": weighted_rmse,
    }
    return bins, summary


def greedy_site_assignment(
    blocks: pd.DataFrame,
    number_of_folds: int,
) -> tuple[dict[str, int], list[int]]:
    assignment: dict[str, int] = {}
    fold_counts = [0] * number_of_folds
    ordered = blocks.sort_values(["rows", "spatial_block_id"], ascending=[False, True])
    for position, row in enumerate(ordered.itertuples(index=False)):
        if position < number_of_folds:
            fold_id = position
        else:
            fold_id = min(range(number_of_folds), key=lambda item: (fold_counts[item], item))
        assignment[row.spatial_block_id] = fold_id
        fold_counts[fold_id] += int(row.rows)
    return assignment, fold_counts


def assign_folds(
    frame: pd.DataFrame,
    block_width_m: int,
    number_of_folds: int,
    sites: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    result = frame[["site_id", "shot_number", "x_epsg5070", "y_epsg5070"]].copy()
    result["spatial_block_x"] = np.floor(result["x_epsg5070"] / block_width_m).astype(int)
    result["spatial_block_y"] = np.floor(result["y_epsg5070"] / block_width_m).astype(int)
    result["spatial_block_id"] = (
        result["site_id"]
        + "_"
        + result["spatial_block_x"].astype(str)
        + "_"
        + result["spatial_block_y"].astype(str)
    )
    blocks = (
        result.groupby(["site_id", "spatial_block_id"], as_index=False)
        .size()
        .rename(columns={"size": "rows"})
    )
    if set(blocks["site_id"]) != set(sites):
        raise RuntimeError("Spatial blocks do not contain the fixed development sites")

    assignments: dict[str, dict[str, int]] = {}
    fold_counts: dict[str, list[int]] = {}
    for site in sites:
        site_blocks = blocks[blocks["site_id"].eq(site)]
        if len(site_blocks) < number_of_folds:
            raise RuntimeError(
                f"{site} has {len(site_blocks)} occupied range-sized blocks; "
                f"cannot construct {number_of_folds} site-stratified folds"
            )
        assignments[site], fold_counts[site] = greedy_site_assignment(
            site_blocks, number_of_folds
        )

    reference_site, permuted_site = sites
    reference_counts = np.asarray(fold_counts[reference_site], dtype=int)
    candidate_counts = np.asarray(fold_counts[permuted_site], dtype=int)
    best: tuple[tuple[float, int, tuple[int, ...]], tuple[int, ...]] | None = None
    for permutation in itertools.permutations(range(number_of_folds)):
        remapped = np.zeros(number_of_folds, dtype=int)
        for old_fold, new_fold in enumerate(permutation):
            remapped[new_fold] = candidate_counts[old_fold]
        combined = reference_counts + remapped
        score = (
            float(np.sum((combined - combined.mean()) ** 2)),
            int(combined.max() - combined.min()),
            permutation,
        )
        if best is None or score < best[0]:
            best = (score, permutation)
    if best is None:
        raise RuntimeError("Could not balance site-stratified folds")
    permutation = best[1]
    assignments[permuted_site] = {
        block_id: permutation[fold_id]
        for block_id, fold_id in assignments[permuted_site].items()
    }

    complete_assignment = {
        block_id: fold_id
        for site_assignment in assignments.values()
        for block_id, fold_id in site_assignment.items()
    }
    result["spatial_fold_id"] = result["spatial_block_id"].map(complete_assignment).astype(int)
    block_assignments = blocks.copy()
    block_assignments["spatial_fold_id"] = (
        block_assignments["spatial_block_id"].map(complete_assignment).astype(int)
    )
    diagnostics = (
        result.groupby(["spatial_fold_id", "site_id"], as_index=False)
        .agg(rows=("shot_number", "size"), blocks=("spatial_block_id", "nunique"))
        .sort_values(["spatial_fold_id", "site_id"])
    )
    if diagnostics.groupby("spatial_fold_id")["site_id"].nunique().min() != len(sites):
        raise RuntimeError("At least one fold does not contain every development site")
    assignment_record = {
        "site_order": sites,
        "permuted_site": permuted_site,
        "fold_permutation": list(permutation),
        "block_assignments": block_assignments.sort_values(
            ["site_id", "spatial_block_id"]
        ).to_dict(orient="records"),
    }
    return result, diagnostics, assignment_record


def plot_variograms(
    bins: pd.DataFrame,
    summaries: pd.DataFrame,
    block_width_m: int,
    output: Path,
) -> None:
    sites = summaries["site_id"].tolist()
    figure, axes = plt.subplots(1, len(sites), figsize=(11, 4.2), sharey=False)
    if len(sites) == 1:
        axes = [axes]
    for axis, site in zip(axes, sites, strict=True):
        site_bins = bins[bins["site_id"].eq(site)]
        summary = summaries[summaries["site_id"].eq(site)].iloc[0]
        axis.scatter(
            site_bins["mean_lag_m"] / 1000.0,
            site_bins["mean_semivariance"],
            s=18,
            color="#2f6b3c",
            label="Empirical bins",
        )
        axis.plot(
            site_bins["mean_lag_m"] / 1000.0,
            site_bins["fitted_semivariance"],
            color="#bb3e03",
            linewidth=2,
            label="Spherical fit",
        )
        axis.axvline(
            summary["practical_range_m"] / 1000.0,
            color="#005f73",
            linestyle="--",
            linewidth=1.5,
            label="Practical range",
        )
        axis.set_title(site)
        axis.set_xlabel("Lag distance (km)")
        axis.set_ylabel("Semivariance")
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle(f"GEDI FHD variograms; frozen block width = {block_width_m / 1000:.0f} km")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.png")
    figure.savefig(temporary, dpi=180, bbox_inches="tight")
    plt.close(figure)
    temporary.replace(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Replace the Phase 3 fold freeze")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protected = [FOLDS_PATH, VARIOGRAM_BINS_PATH, VARIOGRAM_SUMMARY_PATH, FREEZE_PATH]
    if any(path.exists() for path in protected) and not args.force:
        existing = [str(path.relative_to(ROOT)) for path in protected if path.exists()]
        raise RuntimeError(f"Phase 3 fold freeze exists; refusing to overwrite: {existing}")

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    phase3 = config["phase3"]
    sites = list(phase3["development_sites"])
    if sites != list(config["sites"]["development"]) or sites != ["SOAP", "TEAK"]:
        raise RuntimeError("Phase 3 development sites are not the fixed SOAP/TEAK partition")
    source = pd.read_parquet(DEVELOPMENT_PATH)
    if set(source["site_id"]) != set(sites) or source["site_id"].eq("BART").any():
        raise RuntimeError("Development input contains an unexpected site")
    if source["tessera_alignment_id"].nunique() != 1:
        raise RuntimeError("Development input mixes TESSERA alignments")

    variogram_config = phase3["variogram"]
    all_bins: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for site in sites:
        bins, summary = estimate_variogram(
            source[source["site_id"].eq(site)],
            site,
            float(variogram_config["bin_width_m"]),
            int(variogram_config["minimum_pairs_per_bin"]),
        )
        all_bins.append(bins)
        summaries.append(summary)
    variogram_bins = pd.concat(all_bins, ignore_index=True)
    variogram_summary = pd.DataFrame(summaries)

    folds_config = phase3["folds"]
    rounding = 1000
    maximum_range = float(variogram_summary["practical_range_m"].max())
    block_width_m = max(
        int(folds_config["minimum_block_width_m"]),
        int(math.ceil(maximum_range / rounding) * rounding),
    )
    folds, fold_summary, assignment = assign_folds(
        source,
        block_width_m,
        int(folds_config["number_of_folds"]),
        sites,
    )

    phase2_freeze = json.loads(PHASE2_FREEZE_PATH.read_text(encoding="utf-8"))
    freeze_basis = {
        "phase2_alignment_id": phase2_freeze["alignment_id"],
        "development_input_sha256": phase2_freeze["outputs"]["development"]["sha256"],
        "project_config_sha256": sha256(CONFIG_PATH),
        "fold_script_sha256": sha256(Path(__file__).resolve()),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "sites": sites,
        "target": phase3["target"],
        "variogram_method": variogram_config,
        "variogram_results": summaries,
        "fold_method": folds_config,
        "block_width_m": block_width_m,
        "assignment": assignment,
    }
    freeze_basis_sha = canonical_hash(freeze_basis)
    freeze_id = f"spatial-folds-{freeze_basis_sha[:12]}"
    folds.insert(0, "spatial_fold_freeze_id", freeze_id)

    write_parquet_atomic(folds, FOLDS_PATH)
    write_csv_atomic(variogram_bins, VARIOGRAM_BINS_PATH)
    write_csv_atomic(variogram_summary, VARIOGRAM_SUMMARY_PATH)
    plot_variograms(variogram_bins, variogram_summary, block_width_m, FIGURE_PATH)

    manifest = {
        "freeze_id": freeze_id,
        "created_at": utc_now(),
        "status": "frozen",
        "freeze_basis_sha256": freeze_basis_sha,
        "freeze_basis": freeze_basis,
        "outputs": {
            "folds": {
                "path": str(FOLDS_PATH.relative_to(ROOT)),
                "rows": len(folds),
                "sha256": sha256(FOLDS_PATH),
            },
            "variogram_bins": {
                "path": str(VARIOGRAM_BINS_PATH.relative_to(ROOT)),
                "rows": len(variogram_bins),
                "sha256": sha256(VARIOGRAM_BINS_PATH),
            },
            "variogram_summary": {
                "path": str(VARIOGRAM_SUMMARY_PATH.relative_to(ROOT)),
                "rows": len(variogram_summary),
                "sha256": sha256(VARIOGRAM_SUMMARY_PATH),
            },
            "figure": {
                "path": str(FIGURE_PATH.relative_to(ROOT)),
                "sha256": sha256(FIGURE_PATH),
            },
        },
        "fold_summary": fold_summary.to_dict(orient="records"),
    }
    json_dump(FREEZE_PATH, manifest)
    print(
        json.dumps(
            {
                "freeze_id": freeze_id,
                "block_width_m": block_width_m,
                "variogram_summary": summaries,
                "fold_summary": manifest["fold_summary"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
