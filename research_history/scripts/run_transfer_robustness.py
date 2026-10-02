#!/usr/bin/env python3
"""Retrospective target, model, and source-selection robustness analyses."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_multisite_loso as loso  # noqa: E402


SOURCE_PATHS = [
    ROOT / "data/processed/tessera_aligned_development_soap_teak.parquet",
    ROOT / "data/processed/tessera_aligned_locked_bart.parquet",
    ROOT / "data/processed/phase7_tessera_aligned_expansion.parquet",
]
TARGET_PATH = ROOT / "data/processed/phase9_prospective_tessera_aligned.parquet"
SOURCE_CONVENTIONAL_PATH = ROOT / "data/processed/phase8_conventional_predictors_all_sites.parquet"
TARGET_CONVENTIONAL_PATH = ROOT / "data/processed/phase9_prospective_conventional_predictors.parquet"
HABITAT_DISTANCE_PATH = ROOT / "metadata/phase9_landfire_habitat_distances.csv"
SUBSET_PREDICTIONS_PATH = ROOT / "data/processed/phase9_matched_size_source_subset_predictions.parquet"

PREDICTIONS_PATH = ROOT / "data/processed/robustness_transfer_predictions.parquet"
MODEL_METRICS_PATH = ROOT / "outputs/tables/robustness_model_and_target_metrics.csv"
SOURCE_CV_PATH = ROOT / "outputs/tables/robustness_source_selection_cv.csv"
SOURCE_TEST_PATH = ROOT / "outputs/tables/robustness_source_selection_six_site.csv"
WEIGHT_CV_PATH = ROOT / "outputs/tables/robustness_distance_weight_cv.csv"
PERMUTATION_PATH = ROOT / "outputs/tables/robustness_habitat_permutation_null.csv"
PERMUTATION_SUMMARY_PATH = ROOT / "outputs/tables/robustness_habitat_permutation_summary.csv"
FREEZE_PATH = ROOT / "metadata/robustness_transfer_freeze.json"

KEYS = ["site_id", "shot_number"]
FEATURES = [f"tessera_area_{index:03d}" for index in range(128)]
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
SOURCE_COUNTS = list(range(2, 8))
HGB_CANDIDATES = [
    {"learning_rate": 0.05, "max_leaf_nodes": 15, "l2_regularization": 1.0},
    {"learning_rate": 0.05, "max_leaf_nodes": 31, "l2_regularization": 10.0},
]
ENVIRONMENT_VARIABLES = [
    "terrain_elevation_m", "terrain_slope_degrees",
    "s2_ndvi_p10", "s2_ndvi_median", "s2_ndvi_p90",
    "s2_ndmi_p10", "s2_ndmi_median", "s2_ndmi_p90",
    "s2_nbr_p10", "s2_nbr_median", "s2_nbr_p90",
]
RANDOM_SEED = 26072026


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


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def add_targets(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["rh100_m"] = result["elev_highestreturn"] - result["elev_lowestmode"]
    available_layers = np.maximum(2.0, np.ceil(result["rh100_m"].to_numpy()))
    result["height_normalized_entropy_proxy"] = (
        result["fhd_normal"].to_numpy() / np.log(available_layers)
    )
    return result


def metric_values(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = predicted - observed
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r2": float(r2_score(observed, predicted)),
        "spearman_r": float(spearmanr(observed, predicted).statistic),
        "mean_bias": float(np.mean(residual)),
    }


def equal_site_weights(sites: np.ndarray, site_multipliers: dict[str, float] | None = None) -> np.ndarray:
    unique, counts = np.unique(sites, return_counts=True)
    multipliers = site_multipliers or {str(site): 1.0 for site in unique}
    raw = np.array([
        float(multipliers[str(site)]) / counts[np.where(unique == site)[0][0]]
        for site in sites
    ])
    return raw * (len(raw) / raw.sum())


def fit_ridge(training: pd.DataFrame, target: str, alpha: float, multipliers: dict[str, float] | None = None):
    weights = equal_site_weights(training["site_id"].to_numpy(), multipliers)
    scaler = StandardScaler().fit(training[FEATURES].to_numpy(dtype=np.float64), sample_weight=weights)
    transformed = scaler.transform(training[FEATURES].to_numpy(dtype=np.float64))
    model = Ridge(alpha=alpha).fit(transformed, training[target].to_numpy(), sample_weight=weights)
    return scaler, model


def tune_ridge(training: pd.DataFrame, target: str) -> float:
    sites = sorted(training["site_id"].unique())
    if len(sites) == 1:
        return 100.0
    records = []
    for alpha in ALPHAS:
        for validation_site in sites:
            fit = training[~training["site_id"].eq(validation_site)]
            validation = training[training["site_id"].eq(validation_site)]
            scaler, model = fit_ridge(fit, target, alpha)
            prediction = model.predict(scaler.transform(validation[FEATURES].to_numpy()))
            records.append({
                "alpha": alpha,
                "site": validation_site,
                "rmse": metric_values(validation[target].to_numpy(), prediction)["rmse"],
            })
    ranking = (
        pd.DataFrame(records).groupby("alpha", as_index=False)
        .agg(worst_rmse=("rmse", "max"), mean_rmse=("rmse", "mean"))
        .sort_values(["worst_rmse", "mean_rmse", "alpha"])
    )
    return float(ranking.iloc[0]["alpha"])


def fit_hgb(training: pd.DataFrame, target: str, parameters: dict[str, float | int]):
    weights = equal_site_weights(training["site_id"].to_numpy())
    model = HistGradientBoostingRegressor(
        **parameters,
        max_iter=80,
        min_samples_leaf=30,
        random_state=RANDOM_SEED,
    )
    model.fit(training[FEATURES].to_numpy(dtype=np.float32), training[target].to_numpy(), sample_weight=weights)
    return model


def tune_hgb(training: pd.DataFrame, target: str) -> dict[str, float | int]:
    sites = sorted(training["site_id"].unique())
    if len(sites) == 1:
        return HGB_CANDIDATES[0]
    records = []
    for index, parameters in enumerate(HGB_CANDIDATES):
        for validation_site in sites:
            fit = training[~training["site_id"].eq(validation_site)]
            validation = training[training["site_id"].eq(validation_site)]
            model = fit_hgb(fit, target, parameters)
            prediction = model.predict(validation[FEATURES].to_numpy(dtype=np.float32))
            records.append({
                "candidate": index,
                "site": validation_site,
                "rmse": metric_values(validation[target].to_numpy(), prediction)["rmse"],
            })
    best = int(
        pd.DataFrame(records).groupby("candidate", as_index=False)
        .agg(worst_rmse=("rmse", "max"), mean_rmse=("rmse", "mean"))
        .sort_values(["worst_rmse", "mean_rmse", "candidate"])
        .iloc[0]["candidate"]
    )
    return HGB_CANDIDATES[best]


def environment_profiles(
    source: pd.DataFrame,
    target: pd.DataFrame,
) -> pd.DataFrame:
    conventional = pd.concat([
        pd.read_parquet(SOURCE_CONVENTIONAL_PATH, columns=KEYS + ENVIRONMENT_VARIABLES),
        pd.read_parquet(TARGET_CONVENTIONAL_PATH, columns=KEYS + ENVIRONMENT_VARIABLES),
    ], ignore_index=True)
    locations = pd.concat([
        source[KEYS + ["latitude"]], target[KEYS + ["latitude"]]
    ], ignore_index=True)
    frame = conventional.merge(locations, on=KEYS, validate="one_to_one")
    variables = ["latitude", *ENVIRONMENT_VARIABLES]
    medians = frame.groupby("site_id")[variables].median().add_suffix("_median")
    iqrs = (
        frame.groupby("site_id")[variables].quantile(0.75)
        - frame.groupby("site_id")[variables].quantile(0.25)
    ).add_suffix("_iqr")
    return medians.join(iqrs).reset_index()


def environment_distances(profiles: pd.DataFrame, source_sites: list[str]) -> pd.DataFrame:
    values = profiles.set_index("site_id")
    columns = values.columns.tolist()
    reference = values.loc[source_sites, columns]
    mean = reference.mean()
    scale = reference.std(ddof=1).replace(0, 1.0)
    standardized = (values - mean) / scale
    records = []
    for target_site in standardized.index:
        distances = []
        for source_site in source_sites:
            if target_site == source_site:
                continue
            distance = float(np.linalg.norm(
                standardized.loc[target_site].to_numpy()
                - standardized.loc[source_site].to_numpy()
            ) / math.sqrt(len(columns)))
            distances.append((source_site, distance))
        for rank, (source_site, distance) in enumerate(sorted(distances, key=lambda item: (item[1], item[0])), 1):
            records.append({
                "distance_rule": "environmental_context",
                "target_site": target_site,
                "source_site": source_site,
                "distance": distance,
                "rank": rank,
            })
    return pd.DataFrame(records)


def physical_distances(source_sites: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(HABITAT_DISTANCE_PATH)
    frame = frame[frame["hierarchy"].eq("evt_phys") & frame["source_site"].isin(source_sites)]
    return frame.rename(columns={"source_rank": "rank"}).assign(distance_rule="forest_form")[[
        "distance_rule", "target_site", "source_site", "distance", "rank"
    ]]


def hybrid_distances(physical: pd.DataFrame, environmental: pd.DataFrame) -> pd.DataFrame:
    joined = physical.merge(
        environmental,
        on=["target_site", "source_site"],
        suffixes=("_phys", "_env"),
        validate="one_to_one",
    )
    joined["distance"] = joined.groupby("target_site")["distance_phys"].transform(
        lambda values: values / max(float(values.median()), 1e-12)
    ) + joined.groupby("target_site")["distance_env"].transform(
        lambda values: values / max(float(values.median()), 1e-12)
    )
    joined["rank"] = joined.groupby("target_site")["distance"].rank(method="first").astype(int)
    joined["distance_rule"] = "forest_form_plus_environment"
    return joined[["distance_rule", "target_site", "source_site", "distance", "rank"]]


def nearest_sources(distances: pd.DataFrame, rule: str, target_site: str, candidates: list[str], count: int) -> list[str]:
    scoped = distances[
        distances["distance_rule"].eq(rule)
        & distances["target_site"].eq(target_site)
        & distances["source_site"].isin(candidates)
    ].sort_values(["distance", "source_site"])
    if len(scoped) < count:
        raise RuntimeError(f"Insufficient {rule} sources for {target_site}")
    return scoped.head(count)["source_site"].tolist()


def source_only_selection_cv(source: pd.DataFrame, distances: pd.DataFrame) -> pd.DataFrame:
    sites = sorted(source["site_id"].unique())
    records = []
    for rule in sorted(distances["distance_rule"].unique()):
        for count in SOURCE_COUNTS:
            for held_out in sites:
                candidates = sorted(set(sites) - {held_out})
                selected = nearest_sources(distances, rule, held_out, candidates, count)
                training = source[source["site_id"].isin(selected)]
                validation = source[source["site_id"].eq(held_out)]
                alpha = tune_ridge(training, "fhd_normal")
                scaler, model = fit_ridge(training, "fhd_normal", alpha)
                prediction = model.predict(scaler.transform(validation[FEATURES].to_numpy()))
                records.append({
                    "distance_rule": rule,
                    "source_count": count,
                    "held_out_source_site": held_out,
                    "selected_sources": "+".join(selected),
                    "selected_alpha": alpha,
                    **metric_values(validation["fhd_normal"].to_numpy(), prediction),
                })
    result = pd.DataFrame(records)
    macro = result.groupby(["distance_rule", "source_count"], as_index=False)[
        ["rmse", "r2", "spearman_r", "mean_bias"]
    ].mean().sort_values(["rmse", "distance_rule", "source_count"])
    macro["held_out_source_site"] = "macro_mean"
    macro["selected_sources"] = "source_only_selection_summary"
    macro["selected_alpha"] = np.nan
    return pd.concat([result, macro], ignore_index=True)


def source_selection_target_predictions(
    source: pd.DataFrame,
    target: pd.DataFrame,
    distances: pd.DataFrame,
    source_cv: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_sites = sorted(source["site_id"].unique())
    target_sites = sorted(target["site_id"].unique())
    macro = source_cv[source_cv["held_out_source_site"].eq("macro_mean")]
    selected_designs = (
        macro.sort_values(["rmse", "distance_rule", "source_count"])
        .groupby("distance_rule", as_index=False).first()
    )
    frames = []
    selections = []
    for design in selected_designs.itertuples(index=False):
        for target_site in target_sites:
            selected = nearest_sources(
                distances, design.distance_rule, target_site, source_sites, int(design.source_count)
            )
            training = source[source["site_id"].isin(selected)]
            test = target[target["site_id"].eq(target_site)]
            alpha = tune_ridge(training, "fhd_normal")
            scaler, model = fit_ridge(training, "fhd_normal", alpha)
            frame = test[KEYS + ["fhd_normal"]].copy()
            frame["target_variable"] = "fhd_normal"
            frame["model"] = "ridge"
            frame["strategy"] = f"source_cv_{design.distance_rule}_k{int(design.source_count)}"
            frame["prediction"] = model.predict(scaler.transform(test[FEATURES].to_numpy()))
            frames.append(frame)
            selections.append({
                "distance_rule": design.distance_rule,
                "source_count": int(design.source_count),
                "source_cv_macro_rmse": float(design.rmse),
                "target_site": target_site,
                "strategy": f"source_cv_{design.distance_rule}_k{int(design.source_count)}",
                "selected_sources": "+".join(selected),
                "selected_alpha": alpha,
            })
    return pd.concat(frames, ignore_index=True), pd.DataFrame(selections)


def model_target_predictions(source: pd.DataFrame, target: pd.DataFrame, distances: pd.DataFrame) -> pd.DataFrame:
    source_sites = sorted(source["site_id"].unique())
    frames = []
    cache: dict[tuple[str, str, tuple[str, ...]], Any] = {}
    for target_name in ["fhd_normal", "height_normalized_entropy_proxy"]:
        for target_site in sorted(target["site_id"].unique()):
            test = target[target["site_id"].eq(target_site)]
            strategies = {
                "all_sources": source_sites,
                "locked_nearest5_forest_form": nearest_sources(
                    distances, "forest_form", target_site, source_sites, 5
                ),
            }
            for strategy, selected in strategies.items():
                training = source[source["site_id"].isin(selected)]
                source_key = tuple(sorted(selected))
                ridge_key = ("ridge", target_name, source_key)
                if ridge_key not in cache:
                    alpha = tune_ridge(training, target_name)
                    cache[ridge_key] = (alpha, *fit_ridge(training, target_name, alpha))
                alpha, scaler, ridge = cache[ridge_key]
                ridge_frame = test[KEYS + [target_name]].rename(columns={target_name: "observed"})
                ridge_frame["target_variable"] = target_name
                ridge_frame["model"] = "ridge"
                ridge_frame["strategy"] = strategy
                ridge_frame["prediction"] = ridge.predict(scaler.transform(test[FEATURES].to_numpy()))
                ridge_frame["selected_parameters"] = f"alpha={alpha:g}"
                frames.append(ridge_frame)

                hgb_key = ("hist_gradient_boosting", target_name, source_key)
                if hgb_key not in cache:
                    parameters = tune_hgb(training, target_name)
                    cache[hgb_key] = (parameters, fit_hgb(training, target_name, parameters))
                parameters, hgb = cache[hgb_key]
                hgb_frame = test[KEYS + [target_name]].rename(columns={target_name: "observed"})
                hgb_frame["target_variable"] = target_name
                hgb_frame["model"] = "hist_gradient_boosting"
                hgb_frame["strategy"] = strategy
                hgb_frame["prediction"] = hgb.predict(test[FEATURES].to_numpy(dtype=np.float32))
                hgb_frame["selected_parameters"] = json.dumps(parameters, sort_keys=True)
                frames.append(hgb_frame)
    return pd.concat(frames, ignore_index=True)


def prediction_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    records = []
    groups = ["target_variable", "model", "strategy"]
    for keys, frame in predictions.groupby(groups, sort=True):
        target_name, model, strategy = keys
        site_records = []
        for site, scoped in frame.groupby("site_id", sort=True):
            row = {
                "target_variable": target_name,
                "model": model,
                "strategy": strategy,
                "scope": "site",
                "site_id": site,
                "rows": len(scoped),
                **metric_values(scoped["observed"].to_numpy(), scoped["prediction"].to_numpy()),
            }
            records.append(row)
            site_records.append(row)
        site_frame = pd.DataFrame(site_records)
        records.append({
            "target_variable": target_name,
            "model": model,
            "strategy": strategy,
            "scope": "macro",
            "site_id": "macro_mean",
            "rows": int(site_frame["rows"].sum()),
            **{metric: float(site_frame[metric].mean()) for metric in ["rmse", "r2", "spearman_r", "mean_bias"]},
        })
    return pd.DataFrame(records)


def distance_weight_cv(source: pd.DataFrame, physical: pd.DataFrame) -> pd.DataFrame:
    sites = sorted(source["site_id"].unique())
    bandwidths = [0.05, 0.1, 0.2, 0.4, 0.8]
    fixed_alpha = tune_ridge(source, "fhd_normal")
    records = []
    for bandwidth in bandwidths:
        for held_out in sites:
            candidates = sorted(set(sites) - {held_out})
            distance_rows = physical[
                physical["target_site"].eq(held_out)
                & physical["source_site"].isin(candidates)
            ]
            multipliers = {
                row.source_site: float(np.exp(-row.distance / bandwidth))
                for row in distance_rows.itertuples(index=False)
            }
            training = source[source["site_id"].isin(candidates)]
            test = source[source["site_id"].eq(held_out)]
            scaler, model = fit_ridge(training, "fhd_normal", fixed_alpha, multipliers)
            prediction = model.predict(scaler.transform(test[FEATURES].to_numpy()))
            records.append({
                "bandwidth": bandwidth,
                "held_out_source_site": held_out,
                "alpha": fixed_alpha,
                **metric_values(test["fhd_normal"].to_numpy(), prediction),
            })
    result = pd.DataFrame(records)
    macro = result.groupby("bandwidth", as_index=False)[["rmse", "r2", "spearman_r", "mean_bias"]].mean()
    macro["held_out_source_site"] = "macro_mean"
    macro["alpha"] = fixed_alpha
    return pd.concat([result, macro], ignore_index=True)


def permutation_null(physical: pd.DataFrame, source_sites: list[str], target_sites: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset_predictions = pd.read_parquet(SUBSET_PREDICTIONS_PATH)
    subset_metrics = []
    for (target_site, subset), frame in subset_predictions.groupby(["site_id", "source_subset"]):
        subset_metrics.append({
            "target_site": target_site,
            "source_subset": subset,
            "rmse": metric_values(frame["fhd_normal"].to_numpy(), frame["prediction"].to_numpy())["rmse"],
        })
    lookup = pd.DataFrame(subset_metrics).set_index(["target_site", "source_subset"])["rmse"]
    observed_values = []
    for target_site in target_sites:
        selected = nearest_sources(physical, "forest_form", target_site, source_sites, 5)
        observed_values.append(float(lookup.loc[(target_site, "+".join(sorted(selected)))]))
    observed = float(np.mean(observed_values))

    rng = np.random.default_rng(RANDOM_SEED)
    records = []
    for replicate in range(10000):
        permuted = rng.permutation(source_sites)
        profile_to_source = dict(zip(source_sites, permuted, strict=True))
        target_rmse = []
        for target_site in target_sites:
            scoped = physical[physical["target_site"].eq(target_site)].copy()
            scoped["permuted_source"] = scoped["source_site"].map(profile_to_source)
            selected = scoped.sort_values(["distance", "permuted_source"]).head(5)["permuted_source"].tolist()
            target_rmse.append(float(lookup.loc[(target_site, "+".join(sorted(selected)))]))
        records.append({"replicate": replicate, "macro_rmse": float(np.mean(target_rmse))})
    null = pd.DataFrame(records)
    p_value = float((1 + np.sum(null["macro_rmse"].to_numpy() <= observed)) / (len(null) + 1))
    summary = pd.DataFrame([{
        "observed_locked_forest_form_rmse": observed,
        "null_mean_rmse": float(null["macro_rmse"].mean()),
        "null_2.5_percentile": float(null["macro_rmse"].quantile(0.025)),
        "null_97.5_percentile": float(null["macro_rmse"].quantile(0.975)),
        "one_sided_global_permutation_p": p_value,
        "replicates": len(null),
    }])
    return null, summary


def main() -> int:
    data_columns = KEYS + [
        "latitude", "fhd_normal", "elev_highestreturn", "elev_lowestmode", *FEATURES
    ]
    source = add_targets(pd.concat([
        pd.read_parquet(path, columns=data_columns) for path in SOURCE_PATHS
    ], ignore_index=True))
    target = add_targets(pd.read_parquet(TARGET_PATH, columns=data_columns))
    source_sites = sorted(source["site_id"].unique())
    target_sites = sorted(target["site_id"].unique())

    profiles = environment_profiles(source, target)
    physical = physical_distances(source_sites)
    environmental = environment_distances(profiles, source_sites)
    hybrid = hybrid_distances(physical, environmental)
    distances = pd.concat([physical, environmental, hybrid], ignore_index=True)

    source_cv = source_only_selection_cv(source, distances)
    source_predictions, source_selections = source_selection_target_predictions(
        source, target, distances, source_cv
    )
    source_metrics = []
    for (strategy, site), frame in source_predictions.groupby(["strategy", "site_id"]):
        source_metrics.append({
            "strategy": strategy,
            "site_id": site,
            "rows": len(frame),
            **metric_values(frame["fhd_normal"].to_numpy(), frame["prediction"].to_numpy()),
        })
    source_metrics = pd.DataFrame(source_metrics)

    model_predictions = model_target_predictions(source, target, distances)
    model_metrics = prediction_metrics(model_predictions)
    weight_cv = distance_weight_cv(source, physical)
    null, null_summary = permutation_null(physical, source_sites, target_sites)

    write_parquet(model_predictions, PREDICTIONS_PATH)
    write_csv(model_metrics, MODEL_METRICS_PATH)
    write_csv(source_cv, SOURCE_CV_PATH)
    source_test = source_metrics.merge(
        source_selections.rename(columns={"target_site": "site_id"}),
        on=["site_id", "strategy"], how="left", validate="one_to_one"
    )
    write_csv(source_test, SOURCE_TEST_PATH)
    write_csv(weight_cv, WEIGHT_CV_PATH)
    write_csv(null, PERMUTATION_PATH)
    write_csv(null_summary, PERMUTATION_SUMMARY_PATH)

    freeze = {
        "freeze_id": "retrospective-transfer-robustness-" + hashlib.sha256(
            model_metrics.to_csv(index=False).encode("utf-8")
        ).hexdigest()[:12],
        "created_utc": utc_now(),
        "status": "retrospective_source_only_method_selection_then_known_target_evaluation",
        "height_normalized_target": "fhd_normal/log(max(2,ceil(RH100_m)))",
        "height_normalized_target_is_true_shannon_evenness": False,
        "source_count_candidates": SOURCE_COUNTS,
        "nonlinear_candidates": HGB_CANDIDATES,
        "input_hashes": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [*SOURCE_PATHS, TARGET_PATH, SOURCE_CONVENTIONAL_PATH,
                         TARGET_CONVENTIONAL_PATH, HABITAT_DISTANCE_PATH, SUBSET_PREDICTIONS_PATH]
        },
        "outputs": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [PREDICTIONS_PATH, MODEL_METRICS_PATH, SOURCE_CV_PATH,
                         SOURCE_TEST_PATH, WEIGHT_CV_PATH, PERMUTATION_PATH,
                         PERMUTATION_SUMMARY_PATH]
        },
    }
    FREEZE_PATH.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n")
    print(model_metrics[model_metrics["scope"].eq("macro")].to_string(index=False))
    print("\nSource-only selection")
    print(source_cv[source_cv["held_out_source_site"].eq("macro_mean")].head(12).to_string(index=False))
    print("\nPermutation null")
    print(null_summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
