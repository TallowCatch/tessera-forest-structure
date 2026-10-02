#!/usr/bin/env python3
"""Analyze NEON stem-height structure as independent supporting field evidence."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from pyproj import Transformer
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]
FIELD_ROOT = ROOT / "data/interim/neon_field_structure"
ACQUISITION_FREEZE_PATH = ROOT / "metadata/phase5_neon_field_structure_acquisition_freeze.json"
TESSERA_TILE_MANIFEST_PATH = ROOT / "metadata/tessera_v1_2024_tile_manifest.json"
TESSERA_ALIGNMENT_FREEZE_PATH = ROOT / "metadata/tessera_phase2_alignment_freeze.json"
TESSERA_MODEL_PATH = ROOT / "outputs/models/phase3_tessera_ridge.joblib"
CALIBRATION_FREEZE_PATH = ROOT / "metadata/phase5_neon_lidar_calibration_freeze.json"
LIDAR_TILE_ROOT = ROOT / "data/processed/neon_lidar_tile_metrics"
OUTPUT_PATH = ROOT / "data/processed/phase5_neon_field_structure_plots.parquet"
CORRELATIONS_PATH = ROOT / "outputs/tables/phase5_neon_field_structure_correlations.csv"
DISTANCES_PATH = ROOT / "outputs/tables/phase5_neon_field_lidar_distance_summary.csv"
FIGURE_PATH = ROOT / "outputs/figures/phase5_neon_field_structure.png"
FREEZE_PATH = ROOT / "metadata/phase5_neon_field_structure_analysis_freeze.json"
SITE_YEARS = {"SOAP": 2023, "TEAK": 2024, "BART": 2024}
SITE_UTM = {"SOAP": 32611, "TEAK": 32611, "BART": 32619}
TREE_GROWTH_FORMS = {"single bole tree", "multi-bole tree", "small tree", "sapling"}
MINIMUM_MEASURED_HEIGHTS = 5
PRIMARY_FIELD_PROXY = "height_q90_minus_q10_m"
HEIGHT_BINS_M = np.array([0.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0, np.inf])


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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def deduplicate_stem_measurements(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    keys = ["plotID", "individualID", "tempStemID"]
    latest_date = data.groupby(keys, dropna=False)["date"].transform("max")
    latest = data[data["date"].eq(latest_date)].copy()
    ambiguous = latest.duplicated(keys, keep=False)
    ambiguous_keys = int(latest.loc[ambiguous, keys].drop_duplicates().shape[0])
    return latest.loc[~ambiguous].copy(), ambiguous_keys


def height_shannon(heights: np.ndarray) -> float:
    counts, _ = np.histogram(heights, bins=HEIGHT_BINS_M)
    probabilities = counts[counts > 0] / counts.sum()
    return float(-np.sum(probabilities * np.log(probabilities)))


def plot_height_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    heights = frame["height"].to_numpy(dtype=float)
    q10, q25, q50, q75, q90 = np.quantile(heights, [0.10, 0.25, 0.50, 0.75, 0.90])
    mean = float(np.mean(heights))
    standard_deviation = float(np.std(heights, ddof=1)) if len(heights) > 1 else 0.0
    return {
        "measured_live_tree_stems": int(len(heights)),
        "height_mean_m": mean,
        "height_std_m": standard_deviation,
        "height_cv": standard_deviation / mean if mean > 0 else math.nan,
        "height_q10_m": float(q10),
        "height_q25_m": float(q25),
        "height_median_m": float(q50),
        "height_q75_m": float(q75),
        "height_q90_m": float(q90),
        "height_q90_minus_q10_m": float(q90 - q10),
        "height_iqr_m": float(q75 - q25),
        "height_max_m": float(np.max(heights)),
        "height_shannon_5m_classes": height_shannon(heights),
    }


def verify_acquisition() -> dict[str, Any]:
    freeze = json.loads(ACQUISITION_FREEZE_PATH.read_text(encoding="utf-8"))
    for record in freeze["files"]:
        path = ROOT / record["path"]
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise RuntimeError(f"NEON field file is absent or changed: {record['path']}")
    return freeze


def load_field_plots() -> tuple[pd.DataFrame, dict[str, int]]:
    plot_outputs = []
    audit: dict[str, int] = {}
    for site, year in SITE_YEARS.items():
        individual_paths = sorted((FIELD_ROOT / site).rglob("*apparentindividual*.csv"))
        plot_paths = sorted((FIELD_ROOT / site).rglob("*perplotperyear*.csv"))
        individuals = pd.concat([pd.read_csv(path) for path in individual_paths], ignore_index=True)
        plots = pd.concat([pd.read_csv(path) for path in plot_paths], ignore_index=True)
        individuals = individuals[pd.to_datetime(individuals["date"]).dt.year.eq(year)].copy()
        deduplicated, ambiguous_keys = deduplicate_stem_measurements(individuals)
        live_trees = deduplicated[
            deduplicated["plantStatus"].astype(str).str.startswith("Live")
            & deduplicated["growthForm"].isin(TREE_GROWTH_FORMS)
            & pd.to_numeric(deduplicated["height"], errors="coerce").gt(0)
        ].copy()
        live_trees["height"] = pd.to_numeric(live_trees["height"], errors="raise")
        metrics = []
        for plot_id, frame in live_trees.groupby("plotID", sort=True):
            if len(frame) < MINIMUM_MEASURED_HEIGHTS:
                continue
            metrics.append({"plotID": plot_id, **plot_height_metrics(frame)})
        metrics_frame = pd.DataFrame(metrics)
        coordinates = (
            plots.sort_values("date")
            .drop_duplicates("plotID", keep="last")
            [["plotID", "siteID", "date", "decimalLatitude", "decimalLongitude", "coordinateUncertainty"]]
            .rename(columns={
                "siteID": "site_id",
                "date": "plot_event_date",
                "decimalLatitude": "latitude",
                "decimalLongitude": "longitude",
                "coordinateUncertainty": "coordinate_uncertainty_m",
            })
        )
        result = coordinates.merge(metrics_frame, on="plotID", how="inner", validate="one_to_one")
        result["field_measurement_year"] = year
        result["field_proxy_is_measured_fhd"] = False
        plot_outputs.append(result)
        audit[f"{site}_raw_individual_rows"] = len(individuals)
        audit[f"{site}_ambiguous_latest_stem_keys_excluded"] = ambiguous_keys
        audit[f"{site}_live_tree_height_rows"] = len(live_trees)
        audit[f"{site}_eligible_plots"] = len(result)
    combined = pd.concat(plot_outputs, ignore_index=True)
    if combined.duplicated(["site_id", "plotID"]).any():
        raise RuntimeError("Field plot keys are not unique")
    return combined, audit


def load_alignment_module():
    script_path = ROOT / "scripts/build_tessera_alignment.py"
    spec = importlib.util.spec_from_file_location("build_tessera_alignment", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def add_tessera_predictions(plots: pd.DataFrame) -> pd.DataFrame:
    module = load_alignment_module()
    tile_manifest = json.loads(TESSERA_TILE_MANIFEST_PATH.read_text(encoding="utf-8"))
    tiles = module.load_tiles(tile_manifest["tiles"])
    config = yaml.safe_load((ROOT / "configs/project.yaml").read_text(encoding="utf-8"))
    aligned, _ = module.align_targets(plots, tiles, config)
    if not aligned["passes_tessera_alignment"].all():
        raise RuntimeError("At least one NEON field plot failed TESSERA alignment")
    artifact = joblib.load(TESSERA_MODEL_PATH)
    if artifact["model"] != "tessera_ridge":
        raise RuntimeError("Unexpected frozen TESSERA-only model")
    features = artifact["feature_columns"]
    aligned["prediction_frozen_tessera_only_ridge"] = artifact["pipeline"].predict(aligned[features])
    return aligned


def add_nearest_lidar(plots: pd.DataFrame) -> pd.DataFrame:
    calibration = json.loads(CALIBRATION_FREEZE_PATH.read_text(encoding="utf-8"))
    coefficients = calibration["final_development_coefficients"]
    lidar = pd.concat(
        [pd.read_parquet(path) for path in sorted(LIDAR_TILE_ROOT.glob("*.parquet"))],
        ignore_index=True,
    )
    lidar["calibrated_neon_fhd"] = (
        float(coefficients["intercept"]) + float(coefficients["slope"]) * lidar["simulated_fhd"]
    )
    outputs = []
    for site, frame in plots.groupby("site_id", sort=True):
        site_frame = frame.copy()
        transformer = Transformer.from_crs(4326, SITE_UTM[site], always_xy=True)
        x, y = transformer.transform(site_frame["longitude"].to_numpy(), site_frame["latitude"].to_numpy())
        site_lidar = lidar[lidar["site_id"].eq(site)].copy()
        tree = cKDTree(site_lidar[["utm_x", "utm_y"]].to_numpy())
        distances, indices = tree.query(np.column_stack([x, y]), k=1)
        nearest = site_lidar.iloc[indices]
        site_frame["nearest_lidar_distance_m"] = distances
        site_frame["nearest_lidar_shot_number"] = nearest["shot_number"].to_numpy()
        site_frame["nearest_calibrated_neon_fhd"] = nearest["calibrated_neon_fhd"].to_numpy()
        outputs.append(site_frame)
    return pd.concat(outputs, ignore_index=True)


def correlations(data: pd.DataFrame) -> pd.DataFrame:
    targets = [
        PRIMARY_FIELD_PROXY,
        "height_std_m",
        "height_cv",
        "height_max_m",
        "height_shannon_5m_classes",
    ]
    records = []
    scopes = [("pooled_raw", "all_sites", data)]
    scopes.extend(("site", site, frame) for site, frame in data.groupby("site_id", sort=True))
    for scope_type, scope_id, frame in scopes:
        for target in targets:
            for model, predictor in (
                ("frozen_tessera_only_ridge", "prediction_frozen_tessera_only_ridge"),
                ("nearest_calibrated_neon_lidar_within_100m", "nearest_calibrated_neon_fhd"),
                ("nearest_calibrated_neon_lidar_within_250m_sensitivity", "nearest_calibrated_neon_fhd"),
            ):
                selected = frame
                if model.endswith("within_100m"):
                    selected = selected[selected["nearest_lidar_distance_m"].le(100)]
                elif model.endswith("within_250m_sensitivity"):
                    selected = selected[selected["nearest_lidar_distance_m"].le(250)]
                pearson = math.nan
                spearman = math.nan
                if len(selected) >= 5 and selected[target].nunique() > 1 and selected[predictor].nunique() > 1:
                    pearson = float(pearsonr(selected[target], selected[predictor]).statistic)
                    spearman = float(spearmanr(selected[target], selected[predictor]).statistic)
                records.append({
                    "scope_type": scope_type,
                    "scope_id": scope_id,
                    "field_proxy": target,
                    "is_primary_field_proxy": target == PRIMARY_FIELD_PROXY,
                    "model_or_reference": model,
                    "n": len(selected),
                    "pearson_r": pearson,
                    "spearman_r": spearman,
                    "field_proxy_is_measured_fhd": False,
                    "confirmatory_status": (
                        "supporting_external_site_field_evidence"
                        if scope_id == "BART" and model == "frozen_tessera_only_ridge"
                        else "exploratory_or_spatial_sensitivity"
                    ),
                })
    return pd.DataFrame(records)


def distance_summary(data: pd.DataFrame) -> pd.DataFrame:
    records = []
    for site, frame in data.groupby("site_id", sort=True):
        distances = frame["nearest_lidar_distance_m"]
        records.append({
            "site_id": site,
            "field_plots": len(frame),
            "minimum_distance_m": float(distances.min()),
            "median_distance_m": float(distances.median()),
            "plots_within_25m": int(distances.le(25).sum()),
            "plots_within_50m": int(distances.le(50).sum()),
            "plots_within_100m": int(distances.le(100).sum()),
            "plots_within_250m": int(distances.le(250).sum()),
        })
    return pd.DataFrame(records)


def save_figure(data: pd.DataFrame) -> None:
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.8))
    for site, frame in data.groupby("site_id", sort=True):
        axes[0].scatter(
            frame[PRIMARY_FIELD_PROXY], frame["prediction_frozen_tessera_only_ridge"],
            label=site, s=36, alpha=0.75,
        )
        axes[1].scatter(
            frame["nearest_lidar_distance_m"], frame[PRIMARY_FIELD_PROXY],
            label=site, s=36, alpha=0.75,
        )
    axes[0].set_xlabel("Field stem-height 90th-10th percentile spread (m)")
    axes[0].set_ylabel("Frozen TESSERA-only predicted GEDI FHD")
    axes[0].set_title("Independent field structure proxy")
    axes[1].axvline(100, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Distance to nearest simulated lidar footprint (m)")
    axes[1].set_ylabel("Field stem-height spread (m)")
    axes[1].set_title("Sparse footprint overlap")
    axes[0].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(FIGURE_PATH, dpi=200)
    plt.close(figure)


def main() -> int:
    for path in (OUTPUT_PATH, CORRELATIONS_PATH, DISTANCES_PATH, FIGURE_PATH, FREEZE_PATH):
        if path.exists():
            raise RuntimeError(f"Refusing to overwrite field analysis output: {path}")
    acquisition = verify_acquisition()
    plots, audit = load_field_plots()
    plots = add_tessera_predictions(plots)
    plots = add_nearest_lidar(plots)
    correlation_table = correlations(plots)
    distances = distance_summary(plots)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CORRELATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    plots.to_parquet(OUTPUT_PATH, index=False)
    correlation_table.to_csv(CORRELATIONS_PATH, index=False)
    distances.to_csv(DISTANCES_PATH, index=False)
    save_figure(plots)

    alignment_freeze = json.loads(TESSERA_ALIGNMENT_FREEZE_PATH.read_text(encoding="utf-8"))
    calibration_freeze = json.loads(CALIBRATION_FREEZE_PATH.read_text(encoding="utf-8"))
    freeze_basis = {
        "field_acquisition_freeze_id": acquisition["freeze_id"],
        "field_acquisition_freeze_sha256": sha256(ACQUISITION_FREEZE_PATH),
        "field_years": SITE_YEARS,
        "tree_growth_forms": sorted(TREE_GROWTH_FORMS),
        "live_status_rule": "plantStatus_starts_with_Live",
        "minimum_measured_heights_per_plot": MINIMUM_MEASURED_HEIGHTS,
        "ambiguous_latest_stem_measurement_rule": "exclude_all_tied_latest_rows_for_same_plot_individual_stem_key",
        "primary_field_proxy": PRIMARY_FIELD_PROXY,
        "field_proxy_is_measured_fhd": False,
        "height_shannon_bins_m": [0, 2, 5, 10, 20, 30, 40, "infinity"],
        "tessera_alignment_freeze_id": alignment_freeze["alignment_id"],
        "tessera_tile_manifest_sha256": sha256(TESSERA_TILE_MANIFEST_PATH),
        "tessera_model_path": str(TESSERA_MODEL_PATH.relative_to(ROOT)),
        "tessera_model_sha256": sha256(TESSERA_MODEL_PATH),
        "neon_lidar_calibration_freeze_id": calibration_freeze["freeze_id"],
        "nearest_lidar_primary_distance_threshold_m": 100,
        "nearest_lidar_sensitivity_distance_threshold_m": 250,
        "audit": audit,
        "script_sha256": sha256(Path(__file__)),
    }
    freeze_id = "phase5-neon-field-structure-" + canonical_hash(freeze_basis)[:12]
    freeze = {
        "freeze_id": freeze_id,
        "created_at": utc_now(),
        "status": "frozen_supporting_field_structure_analysis_not_FHD_ground_truth",
        "freeze_basis": freeze_basis,
        "freeze_basis_sha256": canonical_hash(freeze_basis),
        "outputs": {
            "plots": {"path": str(OUTPUT_PATH.relative_to(ROOT)), "rows": len(plots), "sha256": sha256(OUTPUT_PATH)},
            "correlations": {"path": str(CORRELATIONS_PATH.relative_to(ROOT)), "rows": len(correlation_table), "sha256": sha256(CORRELATIONS_PATH)},
            "distance_summary": {"path": str(DISTANCES_PATH.relative_to(ROOT)), "rows": len(distances), "sha256": sha256(DISTANCES_PATH)},
            "figure": {"path": str(FIGURE_PATH.relative_to(ROOT)), "sha256": sha256(FIGURE_PATH)},
        },
    }
    write_json(FREEZE_PATH, freeze)
    primary = correlation_table[
        correlation_table["is_primary_field_proxy"]
        & correlation_table["model_or_reference"].eq("frozen_tessera_only_ridge")
    ]
    print(distances.to_string(index=False))
    print("\nPrimary exact-plot TESSERA/field-proxy correlations:")
    print(primary[["scope_id", "n", "pearson_r", "spearman_r", "confirmatory_status"]].to_string(index=False))
    print(f"\nFrozen {freeze_id}: {len(plots)} eligible field plots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
