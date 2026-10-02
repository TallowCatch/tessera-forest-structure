#!/usr/bin/env python3
"""Validate the complete Phase 8 paper-analysis package."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


ROOT = Path(__file__).resolve().parents[1]
SITES = {"BART", "HARV", "ORNL", "SOAP", "TALL", "TEAK", "UNDE", "WREF"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(relative: str) -> dict[str, Any]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def validate_outputs(freeze: dict[str, Any]) -> None:
    for record in freeze["outputs"].values():
        path = ROOT / record["path"]
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise RuntimeError(f"Missing or changed frozen output: {record['path']}")


def reproduce_site_r2(predictions: pd.DataFrame, metrics: pd.DataFrame) -> None:
    site_metrics = metrics[metrics["scope"].eq("site")]
    for (site, method), frame in predictions.groupby(["site_id", "method"]):
        stored = site_metrics[
            site_metrics["held_out_site"].eq(site) & site_metrics["method"].eq(method)
        ]
        if len(stored) != 1:
            raise RuntimeError(f"Missing site metric for {site}/{method}")
        value = float(r2_score(frame["fhd_normal"], frame["prediction"]))
        if not math.isclose(value, float(stored.iloc[0]["r2"]), abs_tol=1e-12):
            raise RuntimeError(f"R2 does not reproduce for {site}/{method}")


def main() -> int:
    conventional = load("metadata/phase8_conventional_predictor_all_sites_freeze.json")
    if conventional["freeze_id"] != "phase8-conventional-all-sites-76e18e223431":
        raise RuntimeError("Unexpected Phase 8 conventional predictor freeze")
    validate_outputs(conventional)
    conventional_basis = conventional["freeze_basis"]
    if conventional_basis["target_columns_read"] or conventional_basis[
        "observation_counts_are_model_features"
    ]:
        raise RuntimeError("Conventional predictor freeze is not target-free")
    if conventional_basis["sentinel_2_and_dem"]["expansion_radiometry"] != (
        "earthsearch_boa_offset_already_applied_cog_dn_times_0.0001"
    ):
        raise RuntimeError("Unexpected expansion Sentinel-2 radiometry")
    predictors = pd.read_parquet(
        ROOT / conventional["outputs"]["predictors"]["path"]
    )
    if len(predictors) != 10184 or set(predictors["site_id"]) != SITES:
        raise RuntimeError("Phase 8 predictor population changed")
    if "fhd_normal" in predictors or predictors.duplicated(["site_id", "shot_number"]).any():
        raise RuntimeError("Predictor table contains a target or duplicate key")
    optical = predictors.groupby("site_id")[[
        "s2_blue_median", "s2_red_median", "s2_nir_median", "s2_ndvi_median"
    ]].median()
    if not optical[["s2_blue_median", "s2_red_median", "s2_nir_median"]].apply(
        lambda values: values.between(-0.05, 1.0)
    ).all().all() or not optical["s2_ndvi_median"].between(-1.0, 1.0).all():
        raise RuntimeError("Sentinel-2 radiometric QA failed")
    s1_features = [
        column for column in predictors
        if column.startswith("s1_") and column != "s1_valid_observation_count"
    ]
    unde = predictors[predictors["site_id"].eq("UNDE")]
    if not unde[s1_features].isna().all().all() or not unde[
        "s1_valid_observation_count"
    ].eq(0).all():
        raise RuntimeError("UNDE radar absence changed")

    fair = load("metadata/phase8_fair_baseline_freeze.json")
    if fair["freeze_id"] != "phase8-fair-baselines-73389d0061c4":
        raise RuntimeError("Unexpected Phase 8 fair baseline freeze")
    validate_outputs(fair)
    fair_basis = fair["freeze_basis"]
    if (
        fair_basis["outer_split"] != "leave_one_complete_site_out"
        or fair_basis["inner_tuning"] != "leave_one_training_site_out"
        or not fair_basis["equal_total_weight_per_training_site"]
        or fair_basis["target_site_fhd_used_for_scaling_tuning_or_training"]
        or fair_basis["observation_counts_used_as_model_features"]
    ):
        raise RuntimeError("Fair baseline leakage controls changed")
    if len(fair_basis["feature_sets"]["sentinel2_topography_ridge"]) != 31:
        raise RuntimeError("Fair Sentinel-2 feature set changed")
    fair_predictions = pd.read_parquet(ROOT / fair["outputs"]["predictions"]["path"])
    fair_metrics = pd.read_csv(ROOT / fair["outputs"]["metrics"]["path"])
    if len(fair_predictions) != 40736 or not (
        fair_predictions["site_id"] == fair_predictions["outer_held_out_site"]
    ).all():
        raise RuntimeError("Fair LOSO prediction population changed")
    reproduce_site_r2(fair_predictions, fair_metrics)
    fair_macro = fair_metrics[fair_metrics["scope"].eq("macro")].set_index("method")
    sentinel = fair_macro.loc["sentinel2_topography_ridge"]
    tessera = fair_macro.loc["tessera_area_ridge"]
    if not (
        sentinel["rmse"] < tessera["rmse"]
        and sentinel["r2"] > tessera["r2"]
        and tessera["spearman_r"] > sentinel["spearman_r"]
    ):
        raise RuntimeError("Fair baseline scientific conclusion changed")
    paired = pd.read_csv(ROOT / fair["outputs"]["paired_differences"]["path"])
    contrast = paired[paired["contrast"].eq(
        "tessera_area_ridge_minus_sentinel2_topography_ridge"
    )].set_index("metric")
    if not (
        contrast.loc["rmse", "ci_low"] < 0 < contrast.loc["rmse", "ci_high"]
        and contrast.loc["spearman_r", "ci_low"] > 0
    ):
        raise RuntimeError("Fair paired uncertainty conclusion changed")

    radar = load("metadata/phase8_radar_sensitivity_freeze.json")
    if radar["freeze_id"] != "phase8-radar-sensitivity-18d0cd03d09a":
        raise RuntimeError("Unexpected Phase 8 radar sensitivity freeze")
    validate_outputs(radar)
    radar_basis = radar["freeze_basis"]
    if radar_basis["excluded_site"] != "UNDE" or radar_basis[
        "observation_counts_used_as_model_features"
    ]:
        raise RuntimeError("Radar sensitivity scope changed")
    radar_predictions = pd.read_parquet(ROOT / radar["outputs"]["predictions"]["path"])
    radar_metrics = pd.read_csv(ROOT / radar["outputs"]["metrics"]["path"])
    if len(radar_predictions) != 27762 or "UNDE" in set(radar_predictions["site_id"]):
        raise RuntimeError("Radar sensitivity population changed")
    reproduce_site_r2(radar_predictions, radar_metrics)
    radar_paired = pd.read_csv(ROOT / radar["outputs"]["paired"]["path"])
    radar_contrast = radar_paired[radar_paired["contrast"].eq(
        "sentinel1_sentinel2_topography_ridge_minus_sentinel2_topography_ridge"
    )]
    if not ((radar_contrast["ci_low"] < 0) & (radar_contrast["ci_high"] > 0)).all():
        raise RuntimeError("Radar no-gain conclusion changed")

    height = load("metadata/phase8_fhd_height_dependence_freeze.json")
    if height["freeze_id"] != "phase8-fhd-height-5c824db80f49":
        raise RuntimeError("Unexpected Phase 8 height freeze")
    validate_outputs(height)
    if height["freeze_basis"]["height_formula"] != (
        "elev_highestreturn - elev_lowestmode"
    ) or height["freeze_basis"]["target_site_fhd_used_for_scaling_tuning_or_training"]:
        raise RuntimeError("Height diagnostic protocol changed")
    height_predictions = pd.read_parquet(ROOT / height["outputs"]["predictions"]["path"])
    height_model = height_predictions[height_predictions["method"].eq("gedi_height_proxy_ridge")]
    if len(height_model) != 10184 or not height_model.groupby("site_id").apply(
        lambda frame: r2_score(frame["fhd_normal"], frame["prediction"]) > 0,
        include_groups=False,
    ).all():
        raise RuntimeError("Height-only positive site-level R2 result changed")

    domain = load("metadata/phase8_domain_shift_freeze.json")
    if domain["freeze_id"] != "phase8-domain-shift-0203c5db7c8d":
        raise RuntimeError("Unexpected Phase 8 domain-shift freeze")
    validate_outputs(domain)
    domain_basis = domain["freeze_basis"]
    if (
        domain_basis["inference_unit"] != "site"
        or domain_basis["site_count"] != 8
        or domain_basis["footprints_treated_as_independent_domain_replicates"]
    ):
        raise RuntimeError("Domain-shift inference scope changed")
    associations = pd.read_csv(ROOT / domain["outputs"]["associations"]["path"])
    target_shift = associations[
        associations["model"].eq("tessera_area_ridge")
        & associations["outcome"].eq("rmse")
        & associations["shift"].eq("fhd_mean_shift")
    ].iloc[0]
    if not (
        target_shift["spearman_r"] > 0.75
        and target_shift["exact_two_sided_permutation_p"] < 0.05
    ):
        raise RuntimeError("Domain-shift target-mean result changed")

    print("Phase 8 validation passed")
    print("10,184 target-free predictors; radiometric and missing-radar QA passed")
    print("Sentinel-2 has lower macro absolute error; TESSERA has stronger rank signal")
    print("FHD is height-dependent; cross-site wall-to-wall mapping gate remains closed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
