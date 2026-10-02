#!/usr/bin/env python3
"""Relate eight-site LOSO errors to target, terrain, forest, and embedding shifts."""

from __future__ import annotations

import hashlib
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.spatial.distance import jensenshannon
from scipy.stats import rankdata, spearmanr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_multisite_loso as loso  # noqa: E402


PROTOCOL_PATH = ROOT / "metadata/project_config_phase8_paper_analysis_protocol_freeze.yaml"
TESSERA_PATHS = [
    ROOT / "data/processed/tessera_aligned_development_soap_teak.parquet",
    ROOT / "data/processed/tessera_aligned_locked_bart.parquet",
    ROOT / "data/processed/phase7_tessera_aligned_expansion.parquet",
]
CONVENTIONAL_PATH = ROOT / "data/processed/phase8_conventional_predictors_all_sites.parquet"
BASELINE_METRICS_PATH = ROOT / "outputs/tables/phase8_fair_baseline_metrics.csv"
BASELINE_FREEZE_PATH = ROOT / "metadata/phase8_fair_baseline_freeze.json"
SITE_PATH = ROOT / "outputs/tables/phase8_domain_shift_site_diagnostics.csv"
ASSOCIATIONS_PATH = ROOT / "outputs/tables/phase8_domain_shift_associations.csv"
FIGURE_PATH = ROOT / "outputs/figures/phase8_domain_shift_diagnostics.png"
FREEZE_PATH = ROOT / "metadata/phase8_domain_shift_freeze.json"
KEYS = ["site_id", "shot_number"]
TERRAIN = [
    "terrain_elevation_m", "terrain_slope_degrees",
    "terrain_aspect_sin", "terrain_aspect_cos",
]
TESSERA = [f"tessera_area_{index:03d}" for index in range(128)]
MODEL_NAMES = ["terrain_ridge", "sentinel2_topography_ridge", "tessera_area_ridge"]
SHIFT_COLUMNS = [
    "fhd_mean_shift", "terrain_standardized_centroid_distance",
    "tessera_standardized_centroid_distance", "nlcd_forest_composition_distance",
]
OUTCOME_COLUMNS = ["rmse", "r2", "absolute_mean_bias"]
FOREST_CLASSES = [41, 42, 43]
EXPECTED_SITES = {"BART", "HARV", "ORNL", "SOAP", "TALL", "TEAK", "UNDE", "WREF"}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def weighted_standardized_centroid_distance(
    training: pd.DataFrame, target: pd.DataFrame, features: list[str]
) -> float:
    values = training[features].to_numpy(dtype=np.float64)
    weights = loso.equal_site_weights(training["site_id"].to_numpy())
    mean = np.average(values, axis=0, weights=weights)
    variance = np.average((values - mean) ** 2, axis=0, weights=weights)
    scale = np.sqrt(variance)
    usable = scale > 1e-10
    if not usable.any():
        raise RuntimeError("No varying dimensions for standardized centroid distance")
    target_mean = target[features].to_numpy(dtype=np.float64).mean(axis=0)
    standardized = (target_mean[usable] - mean[usable]) / scale[usable]
    return float(np.sqrt(np.mean(standardized**2)))


def forest_composition(frame: pd.DataFrame) -> np.ndarray:
    counts = frame["nlcd_center_class"].value_counts()
    composition = np.asarray([float(counts.get(code, 0)) for code in FOREST_CLASSES])
    if composition.sum() <= 0:
        raise RuntimeError("Site has no expected NLCD forest classes")
    return composition / composition.sum()


def equal_training_composition(training: pd.DataFrame) -> np.ndarray:
    compositions = [
        forest_composition(frame)
        for _, frame in training.groupby("site_id", sort=True)
    ]
    value = np.mean(compositions, axis=0)
    return value / value.sum()


def exact_spearman_permutation_p(x: np.ndarray, y: np.ndarray) -> float:
    x_rank = rankdata(x).astype(np.float64)
    y_rank = rankdata(y).astype(np.float64)
    x_rank -= x_rank.mean()
    y_rank -= y_rank.mean()
    denominator = np.sqrt(np.sum(x_rank**2) * np.sum(y_rank**2))
    if denominator <= 0:
        return float("nan")
    observed = abs(float(np.dot(x_rank, y_rank) / denominator))
    permutations = np.asarray(list(itertools.permutations(range(len(y)))), dtype=np.int16)
    permuted = y_rank[permutations]
    statistics = np.abs(permuted @ x_rank / denominator)
    return float(np.mean(statistics >= observed - 1e-12))


def build_associations(site: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for model in MODEL_NAMES:
        for outcome in OUTCOME_COLUMNS:
            y = site[f"{model}_{outcome}"].to_numpy(dtype=np.float64)
            for shift in SHIFT_COLUMNS:
                x = site[shift].to_numpy(dtype=np.float64)
                records.append(
                    {
                        "model": model,
                        "outcome": outcome,
                        "shift": shift,
                        "sites": len(site),
                        "spearman_r": float(spearmanr(x, y).statistic),
                        "exact_two_sided_permutation_p": exact_spearman_permutation_p(x, y),
                        "exploratory": True,
                        "inference_unit": "site",
                    }
                )
    return pd.DataFrame(records)


def save_figure(site: pd.DataFrame) -> None:
    colours = {
        "terrain_ridge": "#bc6c25",
        "sentinel2_topography_ridge": "#3a86ff",
        "tessera_area_ridge": "#0a9396",
    }
    labels = {
        "terrain_ridge": "terrain",
        "sentinel2_topography_ridge": "Sentinel-2 + terrain",
        "tessera_area_ridge": "TESSERA",
    }
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    for model in MODEL_NAMES:
        axes[0, 0].scatter(
            site["fhd_mean_shift"], site[f"{model}_absolute_mean_bias"],
            s=58, label=labels[model], color=colours[model], alpha=0.8,
        )
    for row in site.itertuples():
        axes[0, 0].annotate(
            row.site_id,
            (row.fhd_mean_shift, row.tessera_area_ridge_absolute_mean_bias),
            xytext=(4, 3), textcoords="offset points", fontsize=8,
        )
    axes[0, 0].set_xlabel("Absolute held-out FHD mean shift")
    axes[0, 0].set_ylabel("Absolute mean prediction bias")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].scatter(
        site["tessera_standardized_centroid_distance"],
        site["tessera_area_ridge_rmse"], s=65, color=colours["tessera_area_ridge"],
    )
    for row in site.itertuples():
        axes[0, 1].annotate(
            row.site_id,
            (row.tessera_standardized_centroid_distance, row.tessera_area_ridge_rmse),
            xytext=(4, 3), textcoords="offset points", fontsize=8,
        )
    axes[0, 1].set_xlabel("TESSERA standardized centroid distance")
    axes[0, 1].set_ylabel("TESSERA LOSO RMSE")

    axes[1, 0].scatter(
        site["terrain_standardized_centroid_distance"],
        site["sentinel2_topography_ridge_rmse"], s=65,
        color=colours["sentinel2_topography_ridge"],
    )
    for row in site.itertuples():
        axes[1, 0].annotate(
            row.site_id,
            (row.terrain_standardized_centroid_distance, row.sentinel2_topography_ridge_rmse),
            xytext=(4, 3), textcoords="offset points", fontsize=8,
        )
    axes[1, 0].set_xlabel("Terrain standardized centroid distance")
    axes[1, 0].set_ylabel("Sentinel + terrain LOSO RMSE")

    sites = site["site_id"].tolist()
    matrix = site[[f"{model}_r2" for model in MODEL_NAMES]].to_numpy().T
    image = axes[1, 1].imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=-1, vmax=1)
    axes[1, 1].set_xticks(np.arange(len(sites)), sites)
    axes[1, 1].set_yticks(np.arange(len(MODEL_NAMES)), [labels[model] for model in MODEL_NAMES])
    axes[1, 1].set_title("Held-out-site R-squared")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axes[1, 1].text(
                column, row, f"{matrix[row, column]:.2f}",
                ha="center", va="center", fontsize=8,
                color="white" if abs(matrix[row, column]) > 0.55 else "black",
            )
    figure.colorbar(image, ax=axes[1, 1], fraction=0.046, pad=0.04)
    for axis in axes.flat[:3]:
        axis.grid(alpha=0.18)
    figure.suptitle("Exploratory geographic domain-shift diagnostics")
    figure.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = FIGURE_PATH.with_suffix(".tmp.png")
    figure.savefig(temporary, dpi=180, bbox_inches="tight")
    plt.close(figure)
    temporary.replace(FIGURE_PATH)


def main() -> int:
    protected = [SITE_PATH, ASSOCIATIONS_PATH, FIGURE_PATH, FREEZE_PATH]
    existing = [str(path.relative_to(ROOT)) for path in protected if path.exists()]
    if existing:
        raise RuntimeError(f"Domain-shift freeze exists; refusing to overwrite: {existing}")
    if not BASELINE_FREEZE_PATH.exists():
        raise RuntimeError("Fair eight-site baseline must be frozen before domain diagnostics")
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))[
        "phase8_paper_analysis"
    ]
    tessera_columns = KEYS + ["fhd_normal", "nlcd_center_class", "nlcd_forest_fraction"] + TESSERA
    tessera = pd.concat(
        [pd.read_parquet(path, columns=tessera_columns) for path in TESSERA_PATHS],
        ignore_index=True,
    )
    conventional = pd.read_parquet(CONVENTIONAL_PATH, columns=KEYS + TERRAIN)
    population = tessera.merge(conventional, on=KEYS, validate="one_to_one")
    if len(population) != 10184 or set(population["site_id"]) != EXPECTED_SITES:
        raise RuntimeError("Unexpected domain-shift population")
    metrics = pd.read_csv(BASELINE_METRICS_PATH)
    site_metrics = metrics[metrics["scope"].eq("site")].copy()
    if set(site_metrics["held_out_site"]) != EXPECTED_SITES:
        raise RuntimeError("Baseline metrics do not cover all eight sites")

    records: list[dict[str, Any]] = []
    sites = sorted(EXPECTED_SITES)
    for held_out in sites:
        training = population[~population["site_id"].eq(held_out)]
        target = population[population["site_id"].eq(held_out)]
        training_site_fhd_means = training.groupby("site_id")["fhd_normal"].mean()
        target_composition = forest_composition(target)
        training_composition = equal_training_composition(training)
        record: dict[str, Any] = {
            "site_id": held_out,
            "rows": len(target),
            "fhd_mean": float(target["fhd_normal"].mean()),
            "fhd_std": float(target["fhd_normal"].std(ddof=1)),
            "training_equal_site_fhd_mean": float(training_site_fhd_means.mean()),
            "fhd_mean_shift": float(
                abs(target["fhd_normal"].mean() - training_site_fhd_means.mean())
            ),
            "elevation_mean_m": float(target["terrain_elevation_m"].mean()),
            "forest_fraction_mean": float(target["nlcd_forest_fraction"].mean()),
            "terrain_standardized_centroid_distance": weighted_standardized_centroid_distance(
                training, target, TERRAIN
            ),
            "tessera_standardized_centroid_distance": weighted_standardized_centroid_distance(
                training, target, TESSERA
            ),
            "nlcd_forest_composition_distance": float(
                jensenshannon(target_composition, training_composition, base=2.0)
            ),
        }
        for code, proportion in zip(FOREST_CLASSES, target_composition, strict=True):
            record[f"nlcd_{code}_proportion"] = float(proportion)
        for model in MODEL_NAMES:
            row = site_metrics[
                site_metrics["held_out_site"].eq(held_out)
                & site_metrics["method"].eq(model)
            ].iloc[0]
            for metric in ["rmse", "r2", "spearman_r", "mean_bias"]:
                record[f"{model}_{metric}"] = float(row[metric])
            record[f"{model}_absolute_mean_bias"] = abs(float(row["mean_bias"]))
        records.append(record)
    site = pd.DataFrame(records).sort_values("site_id").reset_index(drop=True)
    associations = build_associations(site)
    loso.write_csv(site, SITE_PATH)
    loso.write_csv(associations, ASSOCIATIONS_PATH)
    save_figure(site)

    outputs = {
        "site_diagnostics": {"path": str(SITE_PATH.relative_to(ROOT)), "sha256": sha256(SITE_PATH)},
        "associations": {"path": str(ASSOCIATIONS_PATH.relative_to(ROOT)), "sha256": sha256(ASSOCIATIONS_PATH)},
        "figure": {"path": str(FIGURE_PATH.relative_to(ROOT)), "sha256": sha256(FIGURE_PATH)},
    }
    freeze_basis = {
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "script_sha256": sha256(Path(__file__).resolve()),
        "fair_baseline_freeze_id": json.loads(
            BASELINE_FREEZE_PATH.read_text(encoding="utf-8")
        )["freeze_id"],
        "input_hashes": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in TESSERA_PATHS + [CONVENTIONAL_PATH, BASELINE_METRICS_PATH]
        },
        "sites": sites,
        "site_count": len(sites),
        "exploratory": bool(protocol["domain_shift_diagnostic"]["exploratory"]),
        "inference_unit": "site",
        "shift_columns": SHIFT_COLUMNS,
        "outcome_columns": OUTCOME_COLUMNS,
        "exact_permutation_count_per_test": 40320,
        "footprints_treated_as_independent_domain_replicates": False,
    }
    freeze_id = "phase8-domain-shift-" + canonical_hash(
        {"freeze_basis": freeze_basis, "outputs": outputs}
    )[:12]
    write_json(
        FREEZE_PATH,
        {"freeze_id": freeze_id, "created_utc": utc_now(), "freeze_basis": freeze_basis, "outputs": outputs},
    )
    strongest = associations.iloc[
        associations["spearman_r"].abs().sort_values(ascending=False).index[:8]
    ]
    print(json.dumps(
        {
            "freeze_id": freeze_id,
            "site_diagnostics": site[[
                "site_id", "fhd_mean_shift", "terrain_standardized_centroid_distance",
                "tessera_standardized_centroid_distance", "nlcd_forest_composition_distance",
                "tessera_area_ridge_rmse", "tessera_area_ridge_r2",
            ]].to_dict("records"),
            "strongest_exploratory_associations": strongest[
                ["model", "outcome", "shift", "spearman_r", "exact_two_sided_permutation_p"]
            ].to_dict("records"),
        },
        indent=2,
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
