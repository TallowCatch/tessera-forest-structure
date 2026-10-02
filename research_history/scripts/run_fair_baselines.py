#!/usr/bin/env python3
"""Run fair nested eight-site LOSO comparisons for TESSERA and conventional features."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import yaml


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
CONVENTIONAL_MANIFEST = ROOT / "metadata/phase8_conventional_predictor_all_sites_freeze.json"
PHASE7_METRICS_PATH = ROOT / "outputs/tables/phase7_expanded_loso_metrics.csv"
PREDICTIONS_PATH = ROOT / "data/processed/phase8_fair_baseline_predictions.parquet"
METRICS_PATH = ROOT / "outputs/tables/phase8_fair_baseline_metrics.csv"
TUNING_PATH = ROOT / "outputs/tables/phase8_fair_baseline_inner_tuning.csv"
SELECTION_PATH = ROOT / "outputs/tables/phase8_fair_baseline_selected_alphas.csv"
UNCERTAINTY_PATH = ROOT / "outputs/tables/phase8_fair_baseline_macro_uncertainty.csv"
PAIRED_PATH = ROOT / "outputs/tables/phase8_tessera_vs_conventional_paired_differences.csv"
MODELS_PATH = ROOT / "outputs/models/phase8_fair_baseline_models.joblib"
FIGURE_PATH = ROOT / "outputs/figures/phase8_fair_baselines.png"
FREEZE_PATH = ROOT / "metadata/phase8_fair_baseline_freeze.json"
KEYS = ["site_id", "shot_number"]
TERRAIN = [
    "terrain_elevation_m", "terrain_slope_degrees",
    "terrain_aspect_sin", "terrain_aspect_cos",
]
TESSERA = [f"tessera_area_{index:03d}" for index in range(128)]
MODELS = [
    "training_site_mean", "terrain_ridge",
    "sentinel2_topography_ridge", "tessera_area_ridge",
]
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


def tune_model(
    training: pd.DataFrame,
    training_sites: list[str],
    held_out_site: str,
    model_name: str,
    features: list[str],
    alphas: list[float],
) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    for alpha in alphas:
        for validation_site in training_sites:
            fit = training[~training["site_id"].eq(validation_site)]
            validation = training[training["site_id"].eq(validation_site)]
            weights = loso.equal_site_weights(fit["site_id"].to_numpy())
            scaler, model = loso.fit_weighted_ridge(
                fit[features].to_numpy(dtype=np.float64),
                fit["fhd_normal"].to_numpy(dtype=np.float64),
                alpha,
                weights,
            )
            prediction = model.predict(
                scaler.transform(validation[features].to_numpy(dtype=np.float64))
            )
            records.append(
                {
                    "outer_held_out_site": held_out_site,
                    "model": model_name,
                    "alpha": alpha,
                    "inner_validation_site": validation_site,
                    "inner_training_sites": "+".join(
                        sorted(set(training_sites) - {validation_site})
                    ),
                    "training_rows": len(fit),
                    "validation_rows": len(validation),
                    **loso.metric_values(
                        validation["fhd_normal"].to_numpy(dtype=np.float64), prediction
                    ),
                }
            )
    tuning = pd.DataFrame(records)
    selected, ranking = loso.select_alpha(tuning)
    tuning["selected_alpha"] = selected
    ranking.insert(0, "outer_held_out_site", held_out_site)
    ranking.insert(1, "model", model_name)
    ranking["selected_alpha"] = selected
    return selected, tuning, ranking


def build_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (site, method), frame in predictions.groupby(
        ["outer_held_out_site", "method"], sort=True
    ):
        records.append(
            {
                "scope": "site", "held_out_site": site,
                "method": method, "rows": len(frame),
                **loso.metric_values(
                    frame["fhd_normal"].to_numpy(), frame["prediction"].to_numpy()
                ),
            }
        )
    site_metrics = pd.DataFrame(records)
    for method, frame in site_metrics.groupby("method", sort=True):
        records.append(
            {
                "scope": "macro", "held_out_site": "macro_mean",
                "method": method, "rows": len(frame),
                **{metric: float(frame[metric].mean()) for metric in loso.METRICS},
            }
        )
    for method, frame in predictions.groupby("method", sort=True):
        records.append(
            {
                "scope": "pooled", "held_out_site": "all_sites",
                "method": method, "rows": len(frame),
                **loso.metric_values(
                    frame["fhd_normal"].to_numpy(), frame["prediction"].to_numpy()
                ),
            }
        )
    return pd.DataFrame(records).sort_values(
        ["scope", "held_out_site", "method"]
    ).reset_index(drop=True)


def bootstrap_site_metrics(
    site_metrics: pd.DataFrame, replicates: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    random = np.random.default_rng(seed)
    sites = sorted(site_metrics["held_out_site"].unique())
    lookup = site_metrics.set_index(["held_out_site", "method"])
    distributions: dict[tuple[str, str], list[float]] = {
        (method, metric): [] for method in MODELS for metric in loso.METRICS
    }
    paired_distributions: dict[tuple[str, str], list[float]] = {
        (comparison, metric): []
        for comparison in ["terrain_ridge", "sentinel2_topography_ridge"]
        for metric in ["rmse", "r2", "spearman_r"]
    }
    for _ in range(replicates):
        sampled = random.choice(sites, size=len(sites), replace=True)
        for method in MODELS:
            for metric in loso.METRICS:
                distributions[(method, metric)].append(
                    float(np.mean([lookup.loc[(site, method), metric] for site in sampled]))
                )
        for comparison in ["terrain_ridge", "sentinel2_topography_ridge"]:
            for metric in ["rmse", "r2", "spearman_r"]:
                paired_distributions[(comparison, metric)].append(
                    float(np.mean([
                        lookup.loc[(site, "tessera_area_ridge"), metric]
                        - lookup.loc[(site, comparison), metric]
                        for site in sampled
                    ]))
                )
    uncertainty = []
    for (method, metric), values in distributions.items():
        numeric = np.asarray(values)
        uncertainty.append(
            {
                "method": method, "metric": metric,
                "macro_estimate": float(
                    site_metrics[site_metrics["method"].eq(method)][metric].mean()
                ),
                "ci_low": float(np.nanpercentile(numeric, 2.5)),
                "ci_high": float(np.nanpercentile(numeric, 97.5)),
                "bootstrap_unit": "site", "bootstrap_replicates": replicates,
            }
        )
    paired = []
    for (comparison, metric), values in paired_distributions.items():
        numeric = np.asarray(values)
        observed = float(np.mean([
            lookup.loc[(site, "tessera_area_ridge"), metric]
            - lookup.loc[(site, comparison), metric]
            for site in sites
        ]))
        paired.append(
            {
                "contrast": f"tessera_area_ridge_minus_{comparison}",
                "metric": metric, "mean_paired_difference": observed,
                "ci_low": float(np.nanpercentile(numeric, 2.5)),
                "ci_high": float(np.nanpercentile(numeric, 97.5)),
                "favourable_direction": "negative" if metric == "rmse" else "positive",
                "bootstrap_unit": "site", "bootstrap_replicates": replicates,
            }
        )
    return pd.DataFrame(uncertainty), pd.DataFrame(paired)


def save_figure(metrics: pd.DataFrame) -> None:
    site = metrics[metrics["scope"].eq("site")]
    sites = sorted(site["held_out_site"].unique())
    x = np.arange(len(sites))
    width = 0.2
    colours = {
        "training_site_mean": "#6c757d", "terrain_ridge": "#bc6c25",
        "sentinel2_topography_ridge": "#3a86ff", "tessera_area_ridge": "#0a9396",
    }
    labels = {
        "training_site_mean": "training-site mean", "terrain_ridge": "terrain",
        "sentinel2_topography_ridge": "Sentinel-2 + terrain", "tessera_area_ridge": "TESSERA",
    }
    figure, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    for index, method in enumerate(MODELS):
        frame = site[site["method"].eq(method)].set_index("held_out_site").loc[sites]
        offset = (index - 1.5) * width
        axes[0].bar(x + offset, frame["rmse"], width, color=colours[method], label=labels[method])
        axes[1].bar(x + offset, frame["r2"], width, color=colours[method])
    axes[0].set_ylabel("RMSE")
    axes[1].set_ylabel("R-squared")
    axes[1].set_xlabel("Completely held-out site")
    axes[1].set_xticks(x, sites)
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[0].legend(ncol=4, frameon=False, loc="upper left")
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Fair eight-site GEDI FHD baseline comparison")
    figure.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = FIGURE_PATH.with_suffix(".tmp.png")
    figure.savefig(temporary, dpi=180, bbox_inches="tight")
    plt.close(figure)
    temporary.replace(FIGURE_PATH)


def main() -> int:
    protected = [
        PREDICTIONS_PATH, METRICS_PATH, TUNING_PATH, SELECTION_PATH,
        UNCERTAINTY_PATH, PAIRED_PATH, MODELS_PATH, FIGURE_PATH, FREEZE_PATH,
    ]
    existing = [str(path.relative_to(ROOT)) for path in protected if path.exists()]
    if existing:
        raise RuntimeError(f"Fair-baseline freeze exists; refusing to overwrite: {existing}")
    missing = [
        str(path.relative_to(ROOT))
        for path in [CONVENTIONAL_PATH, CONVENTIONAL_MANIFEST]
        if not path.exists()
    ]
    if missing:
        raise RuntimeError(f"Missing conventional predictor freeze: {missing}")
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))[
        "phase8_paper_analysis"
    ]
    manifest = json.loads(CONVENTIONAL_MANIFEST.read_text(encoding="utf-8"))
    all_conventional = manifest["freeze_basis"]["feature_columns"]
    excluded = sorted(
        feature for feature in all_conventional if feature.endswith("observation_count")
    )
    if excluded != sorted(manifest["freeze_basis"]["observation_count_columns"]):
        raise RuntimeError(f"Unexpected observation-count diagnostics: {excluded}")
    sentinel2_topography = manifest["freeze_basis"]["primary_eight_site_features"]
    if len(sentinel2_topography) != 31 or not set(TERRAIN).issubset(sentinel2_topography):
        raise RuntimeError("Sentinel-2-plus-topography feature set is not 31 columns")
    feature_sets = {
        "terrain_ridge": TERRAIN,
        "sentinel2_topography_ridge": sentinel2_topography,
        "tessera_area_ridge": TESSERA,
    }
    if any(
        feature.endswith("observation_count")
        for features in feature_sets.values() for feature in features
    ):
        raise RuntimeError("Observation-count diagnostic entered a model feature set")

    tessera_frames = [
        pd.read_parquet(path, columns=KEYS + ["fhd_normal"] + TESSERA)
        for path in TESSERA_PATHS
    ]
    population = pd.concat(tessera_frames, ignore_index=True).merge(
        pd.read_parquet(CONVENTIONAL_PATH, columns=KEYS + all_conventional),
        on=KEYS,
        validate="one_to_one",
    )
    if len(population) != 10184 or set(population["site_id"]) != EXPECTED_SITES:
        raise RuntimeError("Unexpected fair-baseline population")
    if population.duplicated(KEYS).any():
        raise RuntimeError("Duplicate fair-baseline keys")
    model_features = sorted(set(sum(feature_sets.values(), [])))
    if not np.isfinite(population[model_features + ["fhd_normal"]].to_numpy()).all():
        raise RuntimeError("Fair-baseline population contains non-finite model data")

    sites = sorted(EXPECTED_SITES)
    alphas = [float(value) for value in protocol["ridge_alpha_grid"]]
    predictions: list[pd.DataFrame] = []
    tuning_frames: list[pd.DataFrame] = []
    ranking_frames: list[pd.DataFrame] = []
    artifacts: dict[str, Any] = {"outer_folds": {}, "feature_sets": feature_sets}
    for held_out in sites:
        training_sites = sorted(set(sites) - {held_out})
        training = population[population["site_id"].isin(training_sites)]
        target_keys = population.loc[population["site_id"].eq(held_out), KEYS]
        baseline = target_keys.copy()
        baseline["outer_held_out_site"] = held_out
        baseline["method"] = "training_site_mean"
        baseline["selected_alpha"] = np.nan
        baseline["prediction"] = float(
            training.groupby("site_id")["fhd_normal"].mean().mean()
        )
        predictions.append(baseline)
        artifacts["outer_folds"][held_out] = {}
        for model_name, features in feature_sets.items():
            target = population.loc[
                population["site_id"].eq(held_out), KEYS + features
            ].copy()
            if "fhd_normal" in target.columns:
                raise RuntimeError("Outer target FHD entered fair-baseline fitting")
            selected, tuning, ranking = tune_model(
                training, training_sites, held_out, model_name, features, alphas
            )
            weights = loso.equal_site_weights(training["site_id"].to_numpy())
            scaler, model = loso.fit_weighted_ridge(
                training[features].to_numpy(dtype=np.float64),
                training["fhd_normal"].to_numpy(dtype=np.float64),
                selected,
                weights,
            )
            frame = target[KEYS].copy()
            frame["outer_held_out_site"] = held_out
            frame["method"] = model_name
            frame["selected_alpha"] = selected
            frame["prediction"] = model.predict(
                scaler.transform(target[features].to_numpy(dtype=np.float64))
            )
            predictions.append(frame)
            tuning_frames.append(tuning)
            ranking_frames.append(ranking)
            artifacts["outer_folds"][held_out][model_name] = {
                "training_sites": training_sites,
                "features": features,
                "selected_alpha": selected,
                "scaler": scaler,
                "model": model,
                "target_columns": target.columns.tolist(),
                "target_outcome_present": False,
            }

    predictions_frame = pd.concat(predictions, ignore_index=True).merge(
        population[KEYS + ["fhd_normal"]], on=KEYS, validate="many_to_one"
    )
    tuning = pd.concat(tuning_frames, ignore_index=True)
    rankings = pd.concat(ranking_frames, ignore_index=True)
    selection = rankings[rankings["alpha"].eq(rankings["selected_alpha"])].copy()
    metrics = build_metrics(predictions_frame)
    site_metrics = metrics[metrics["scope"].eq("site")]
    replicates = int(protocol["uncertainty"]["bootstrap_replicates"])
    seed = int(protocol["uncertainty"]["random_seed"])
    uncertainty, paired = bootstrap_site_metrics(site_metrics, replicates, seed)

    previous = pd.read_csv(PHASE7_METRICS_PATH)
    previous = previous[
        previous["scope"].eq("site")
        & previous["method"].eq("tessera_area_ridge")
    ].set_index("held_out_site")
    current = site_metrics[
        site_metrics["method"].eq("tessera_area_ridge")
    ].set_index("held_out_site")
    reproducibility_delta = float(
        np.max(
            np.abs(
                current.loc[sites, loso.METRICS].to_numpy(dtype=np.float64)
                - previous.loc[sites, loso.METRICS].to_numpy(dtype=np.float64)
            )
        )
    )
    if reproducibility_delta > 1e-10:
        raise RuntimeError(
            f"Phase 8 TESSERA LOSO does not reproduce Phase 7: {reproducibility_delta}"
        )

    loso.write_parquet(predictions_frame, PREDICTIONS_PATH)
    loso.write_csv(metrics, METRICS_PATH)
    loso.write_csv(tuning, TUNING_PATH)
    loso.write_csv(selection, SELECTION_PATH)
    loso.write_csv(uncertainty, UNCERTAINTY_PATH)
    loso.write_csv(paired, PAIRED_PATH)
    loso.write_joblib(artifacts, MODELS_PATH)
    save_figure(metrics)
    outputs = {
        "predictions": {"path": str(PREDICTIONS_PATH.relative_to(ROOT)), "sha256": sha256(PREDICTIONS_PATH)},
        "metrics": {"path": str(METRICS_PATH.relative_to(ROOT)), "sha256": sha256(METRICS_PATH)},
        "tuning": {"path": str(TUNING_PATH.relative_to(ROOT)), "sha256": sha256(TUNING_PATH)},
        "selection": {"path": str(SELECTION_PATH.relative_to(ROOT)), "sha256": sha256(SELECTION_PATH)},
        "uncertainty": {"path": str(UNCERTAINTY_PATH.relative_to(ROOT)), "sha256": sha256(UNCERTAINTY_PATH)},
        "paired_differences": {"path": str(PAIRED_PATH.relative_to(ROOT)), "sha256": sha256(PAIRED_PATH)},
        "models": {"path": str(MODELS_PATH.relative_to(ROOT)), "sha256": sha256(MODELS_PATH)},
        "figure": {"path": str(FIGURE_PATH.relative_to(ROOT)), "sha256": sha256(FIGURE_PATH)},
    }
    freeze_basis = {
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "script_sha256": sha256(Path(__file__).resolve()),
        "input_hashes": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in TESSERA_PATHS + [CONVENTIONAL_PATH, CONVENTIONAL_MANIFEST]
        },
        "sites": sites, "rows": len(population),
        "feature_sets": feature_sets,
        "excluded_observation_count_columns": excluded,
        "observation_counts_used_as_model_features": False,
        "outer_split": protocol["outer_validation"],
        "inner_tuning": protocol["inner_tuning"],
        "equal_total_weight_per_training_site": True,
        "target_site_fhd_used_for_scaling_tuning_or_training": False,
        "alphas": alphas,
        "phase7_tessera_reproduction_max_metric_delta": reproducibility_delta,
        "software": {
            "numpy": np.__version__, "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    freeze_id = "phase8-fair-baselines-" + canonical_hash(
        {"freeze_basis": freeze_basis, "outputs": outputs}
    )[:12]
    write_json(
        FREEZE_PATH,
        {"freeze_id": freeze_id, "created_utc": utc_now(), "freeze_basis": freeze_basis, "outputs": outputs},
    )
    macro = metrics[metrics["scope"].eq("macro")].set_index("method")
    print(json.dumps(
        {
            "freeze_id": freeze_id,
            "macro_metrics": {
                method: {metric: float(macro.loc[method, metric]) for metric in loso.METRICS}
                for method in MODELS
            },
            "positive_r2_sites": {
                method: sorted(
                    site_metrics.loc[
                        site_metrics["method"].eq(method) & site_metrics["r2"].gt(0),
                        "held_out_site",
                    ].tolist()
                )
                for method in MODELS[1:]
            },
            "paired_tessera_minus_sentinel2": paired[
                paired["contrast"].eq("tessera_area_ridge_minus_sentinel2_topography_ridge")
            ][["metric", "mean_paired_difference", "ci_low", "ci_high"]].to_dict("records"),
        },
        indent=2,
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
