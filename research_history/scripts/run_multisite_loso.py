#!/usr/bin/env python3
"""Run nested leave-one-site-out GEDI FHD domain-generalization evaluation."""

from __future__ import annotations

import hashlib
import json
import math
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
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/project.yaml"
PROTOCOL_FREEZE_PATH = ROOT / "metadata/project_config_phase6_multisite_protocol_freeze.yaml"
DEVELOPMENT_PATH = ROOT / "data/processed/tessera_aligned_development_soap_teak.parquet"
BART_PATH = ROOT / "data/processed/tessera_aligned_locked_bart.parquet"
DEVELOPMENT_CONVENTIONAL_PATH = (
    ROOT / "data/processed/phase3_conventional_predictors_development.parquet"
)
BART_CONVENTIONAL_PATH = ROOT / "data/processed/phase4_conventional_predictors_bart.parquet"

PREDICTIONS_PATH = ROOT / "data/processed/phase6_multisite_loso_predictions.parquet"
METRICS_PATH = ROOT / "outputs/tables/phase6_multisite_loso_metrics.csv"
TUNING_PATH = ROOT / "outputs/tables/phase6_multisite_loso_inner_tuning.csv"
SELECTION_PATH = ROOT / "outputs/tables/phase6_multisite_loso_selected_alphas.csv"
FIGURE_PATH = ROOT / "outputs/figures/phase6_multisite_loso.png"
MODELS_PATH = ROOT / "outputs/models/phase6_multisite_loso_models.joblib"
FREEZE_PATH = ROOT / "metadata/phase6_multisite_loso_freeze.json"

KEY_COLUMNS = ["site_id", "shot_number"]
TERRAIN_FEATURES = [
    "terrain_elevation_m",
    "terrain_slope_degrees",
    "terrain_aspect_sin",
    "terrain_aspect_cos",
]
TESSERA_FEATURES = [f"tessera_area_{index:03d}" for index in range(128)]
FEATURE_SETS = {
    "terrain_ridge": TERRAIN_FEATURES,
    "tessera_area_ridge": TESSERA_FEATURES,
    "tessera_area_topography_ridge": TESSERA_FEATURES + TERRAIN_FEATURES,
}
MODELS = ["training_mean", *FEATURE_SETS]
METRICS = ["r2", "rmse", "mae", "pearson_r", "spearman_r", "mean_bias"]


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


def write_joblib(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(value, temporary, compress=3)
    temporary.replace(path)


def metric_values(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = predicted - observed
    pearson = math.nan
    spearman = math.nan
    if np.ptp(observed) > 1e-12 and np.ptp(predicted) > 1e-12:
        pearson = float(np.corrcoef(observed, predicted)[0, 1])
        spearman = float(spearmanr(observed, predicted).statistic)
    return {
        "r2": float(r2_score(observed, predicted)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(mean_absolute_error(observed, predicted)),
        "pearson_r": pearson,
        "spearman_r": spearman,
        "mean_bias": float(np.mean(residual)),
    }


def equal_site_weights(site_ids: np.ndarray) -> np.ndarray:
    """Give every site equal total loss weight while preserving mean sample weight one."""
    site_ids = np.asarray(site_ids)
    sites, counts = np.unique(site_ids, return_counts=True)
    if len(sites) == 0:
        raise ValueError("Cannot weight an empty training population")
    count_lookup = dict(zip(sites, counts, strict=True))
    row_count = len(site_ids)
    site_count = len(sites)
    return np.asarray(
        [row_count / (site_count * count_lookup[site]) for site in site_ids],
        dtype=np.float64,
    )


def fit_weighted_ridge(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
    weights: np.ndarray,
) -> tuple[StandardScaler, Ridge]:
    scaler = StandardScaler().fit(x, sample_weight=weights)
    model = Ridge(alpha=alpha).fit(scaler.transform(x), y, sample_weight=weights)
    return scaler, model


def select_alpha(tuning: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    ranking = (
        tuning.groupby("alpha", as_index=False)
        .agg(
            worst_inner_validation_rmse=("rmse", "max"),
            mean_inner_validation_rmse=("rmse", "mean"),
        )
        .sort_values(
            ["worst_inner_validation_rmse", "mean_inner_validation_rmse", "alpha"]
        )
        .reset_index(drop=True)
    )
    return float(ranking.iloc[0]["alpha"]), ranking


def tune_reciprocal_sites(
    training: pd.DataFrame,
    training_sites: list[str],
    model_name: str,
    features: list[str],
    alphas: list[float],
    outer_held_out_site: str,
) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    if set(training["site_id"].unique()) != set(training_sites):
        raise RuntimeError("Inner tuning population contains the wrong sites")
    records: list[dict[str, Any]] = []
    for alpha in alphas:
        for validation_site in training_sites:
            train_site = next(site for site in training_sites if site != validation_site)
            fit = training[training["site_id"].eq(train_site)]
            validation = training[training["site_id"].eq(validation_site)]
            scaler, model = fit_weighted_ridge(
                fit[features].to_numpy(dtype=np.float64),
                fit["fhd_normal"].to_numpy(dtype=np.float64),
                float(alpha),
                np.ones(len(fit), dtype=np.float64),
            )
            predicted = model.predict(
                scaler.transform(validation[features].to_numpy(dtype=np.float64))
            )
            records.append(
                {
                    "outer_held_out_site": outer_held_out_site,
                    "model": model_name,
                    "alpha": float(alpha),
                    "inner_train_site": train_site,
                    "inner_validation_site": validation_site,
                    "training_rows": len(fit),
                    "validation_rows": len(validation),
                    **metric_values(
                        validation["fhd_normal"].to_numpy(dtype=np.float64), predicted
                    ),
                }
            )
    tuning = pd.DataFrame(records)
    selected, ranking = select_alpha(tuning)
    tuning["selected_alpha"] = selected
    ranking.insert(0, "outer_held_out_site", outer_held_out_site)
    ranking.insert(1, "model", model_name)
    ranking["selected_alpha"] = selected
    return selected, tuning, ranking


def fit_outer_fold(
    training: pd.DataFrame,
    target_predictors: pd.DataFrame,
    training_sites: list[str],
    held_out_site: str,
    alphas: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fit using training outcomes and predict a target table that has no outcome column."""
    if "fhd_normal" in target_predictors.columns:
        raise RuntimeError("Held-out outcomes were passed into the target-free fit stage")
    if set(training["site_id"].unique()) != set(training_sites):
        raise RuntimeError("Outer training population contains the wrong sites")
    if set(target_predictors["site_id"].unique()) != {held_out_site}:
        raise RuntimeError("Outer target predictor population contains the wrong site")

    prediction_frames: list[pd.DataFrame] = []
    tuning_frames: list[pd.DataFrame] = []
    selection_frames: list[pd.DataFrame] = []
    artifacts: dict[str, Any] = {}

    site_means = training.groupby("site_id")["fhd_normal"].mean()
    baseline_value = float(site_means.mean())
    baseline = target_predictors[KEY_COLUMNS].copy()
    baseline["outer_held_out_site"] = held_out_site
    baseline["training_sites"] = "+".join(training_sites)
    baseline["model"] = "training_mean"
    baseline["selected_alpha"] = np.nan
    baseline["prediction"] = baseline_value
    prediction_frames.append(baseline)
    artifacts["training_mean"] = {
        "equal_site_mean": baseline_value,
        "site_means": {site: float(value) for site, value in site_means.items()},
    }

    training_weights = equal_site_weights(training["site_id"].to_numpy())
    for model_name, features in FEATURE_SETS.items():
        selected, tuning, ranking = tune_reciprocal_sites(
            training, training_sites, model_name, features, alphas, held_out_site
        )
        scaler, model = fit_weighted_ridge(
            training[features].to_numpy(dtype=np.float64),
            training["fhd_normal"].to_numpy(dtype=np.float64),
            selected,
            training_weights,
        )
        predicted = model.predict(
            scaler.transform(target_predictors[features].to_numpy(dtype=np.float64))
        )
        output = target_predictors[KEY_COLUMNS].copy()
        output["outer_held_out_site"] = held_out_site
        output["training_sites"] = "+".join(training_sites)
        output["model"] = model_name
        output["selected_alpha"] = selected
        output["prediction"] = predicted
        prediction_frames.append(output)
        tuning_frames.append(tuning)
        selection_frames.append(ranking)
        artifacts[model_name] = {
            "features": features,
            "selected_alpha": selected,
            "scaler": scaler,
            "model": model,
        }

    site_totals = {
        site: float(training_weights[training["site_id"].eq(site).to_numpy()].sum())
        for site in training_sites
    }
    artifacts["metadata"] = {
        "held_out_site": held_out_site,
        "training_sites": training_sites,
        "training_rows": len(training),
        "target_rows": len(target_predictors),
        "target_columns": target_predictors.columns.tolist(),
        "target_outcome_present": False,
        "training_weight_total_by_site": site_totals,
    }
    return (
        pd.concat(prediction_frames, ignore_index=True),
        pd.concat(tuning_frames, ignore_index=True),
        pd.concat(selection_frames, ignore_index=True),
        artifacts,
    )


def load_population() -> pd.DataFrame:
    development = pd.read_parquet(DEVELOPMENT_PATH).merge(
        pd.read_parquet(DEVELOPMENT_CONVENTIONAL_PATH),
        on=KEY_COLUMNS,
        validate="one_to_one",
        suffixes=("", "_conventional"),
    )
    bart = pd.read_parquet(BART_PATH).merge(
        pd.read_parquet(BART_CONVENTIONAL_PATH),
        on=KEY_COLUMNS,
        validate="one_to_one",
        suffixes=("", "_conventional"),
    )
    population = pd.concat([development, bart], ignore_index=True)
    expected_counts = {"SOAP": 927, "TEAK": 1110, "BART": 1026}
    if population.groupby("site_id").size().to_dict() != expected_counts:
        raise RuntimeError("Frozen three-site population changed")
    if population.duplicated(KEY_COLUMNS).any():
        raise RuntimeError("Duplicate site/shot keys in three-site population")
    required = ["fhd_normal", *TESSERA_FEATURES, *TERRAIN_FEATURES]
    if not np.isfinite(population[required].to_numpy(dtype=np.float64)).all():
        raise RuntimeError("Non-finite target or predictor in LOSO population")
    return population


def build_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (site, model), frame in predictions.groupby(["outer_held_out_site", "model"], sort=True):
        records.append(
            {
                "scope": "site",
                "held_out_site": site,
                "model": model,
                "rows": len(frame),
                **metric_values(
                    frame["fhd_normal"].to_numpy(dtype=np.float64),
                    frame["prediction"].to_numpy(dtype=np.float64),
                ),
            }
        )
    site_metrics = pd.DataFrame(records)
    for model, frame in site_metrics.groupby("model", sort=True):
        records.append(
            {
                "scope": "macro",
                "held_out_site": "macro_mean",
                "model": model,
                "rows": len(frame),
                **{metric: float(frame[metric].mean()) for metric in METRICS},
            }
        )
    for model, frame in predictions.groupby("model", sort=True):
        records.append(
            {
                "scope": "pooled",
                "held_out_site": "all_sites",
                "model": model,
                "rows": len(frame),
                **metric_values(
                    frame["fhd_normal"].to_numpy(dtype=np.float64),
                    frame["prediction"].to_numpy(dtype=np.float64),
                ),
            }
        )
    return pd.DataFrame(records).sort_values(
        ["scope", "held_out_site", "model"]
    ).reset_index(drop=True)


def evaluate_gate(metrics: pd.DataFrame, primary_model: str) -> dict[str, Any]:
    site = metrics[metrics["scope"].eq("site")]
    primary = site[site["model"].eq(primary_model)].set_index("held_out_site")
    baseline = site[site["model"].eq("training_mean")].set_index("held_out_site")
    sites = sorted(primary.index)
    checks = {
        held_out: {
            "primary_r2": float(primary.loc[held_out, "r2"]),
            "primary_rmse": float(primary.loc[held_out, "rmse"]),
            "training_mean_rmse": float(baseline.loc[held_out, "rmse"]),
            "r2_above_zero": bool(primary.loc[held_out, "r2"] > 0),
            "rmse_below_training_mean": bool(
                primary.loc[held_out, "rmse"] < baseline.loc[held_out, "rmse"]
            ),
        }
        for held_out in sites
    }
    macro = metrics[metrics["scope"].eq("macro")].set_index("model")
    primary_macro = float(macro.loc[primary_model, "rmse"])
    terrain_macro = float(macro.loc["terrain_ridge", "rmse"])
    all_positive = all(value["r2_above_zero"] for value in checks.values())
    all_below_mean = all(value["rmse_below_training_mean"] for value in checks.values())
    below_terrain = primary_macro < terrain_macro
    return {
        "site_checks": checks,
        "primary_r2_above_zero_on_every_held_out_site": all_positive,
        "primary_rmse_below_training_mean_on_every_held_out_site": all_below_mean,
        "primary_macro_rmse": primary_macro,
        "terrain_macro_rmse": terrain_macro,
        "primary_macro_rmse_below_terrain_macro_rmse": below_terrain,
        "passed": bool(all_positive and all_below_mean and below_terrain),
    }


def save_figure(metrics: pd.DataFrame) -> None:
    site = metrics[metrics["scope"].eq("site")]
    sites = ["SOAP", "TEAK", "BART"]
    colours = {
        "training_mean": "#6c757d",
        "terrain_ridge": "#bb3e03",
        "tessera_area_ridge": "#0a9396",
        "tessera_area_topography_ridge": "#005f73",
    }
    labels = {
        "training_mean": "training-site mean",
        "terrain_ridge": "terrain Ridge",
        "tessera_area_ridge": "TESSERA Ridge",
        "tessera_area_topography_ridge": "TESSERA + terrain Ridge",
    }
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
    x = np.arange(len(sites))
    width = 0.19
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(MODELS))
    for model, offset in zip(MODELS, offsets, strict=True):
        frame = site[site["model"].eq(model)].set_index("held_out_site").loc[sites]
        axes[0].bar(
            x + offset, frame["rmse"], width, color=colours[model], label=labels[model]
        )
        axes[1].bar(x + offset, frame["r2"], width, color=colours[model])
    for axis in axes:
        axis.set_xticks(x, sites)
        axis.set_xlabel("Completely held-out site")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("RMSE")
    axes[1].set_ylabel("R-squared")
    axes[1].axhline(0, color="black", linewidth=1, linestyle="--")
    figure.suptitle("Nested leave-one-site-out GEDI FHD evaluation")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="lower center", ncol=4, frameon=False)
    figure.tight_layout(rect=(0, 0.12, 1, 1))
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = FIGURE_PATH.with_suffix(".tmp.png")
    figure.savefig(temporary, dpi=180, bbox_inches="tight")
    plt.close(figure)
    temporary.replace(FIGURE_PATH)


def output_record(path: Path, rows: int | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }
    if rows is not None:
        record["rows"] = rows
    return record


def main() -> int:
    protected = [
        PREDICTIONS_PATH,
        METRICS_PATH,
        TUNING_PATH,
        SELECTION_PATH,
        FIGURE_PATH,
        MODELS_PATH,
        FREEZE_PATH,
    ]
    existing = [str(path.relative_to(ROOT)) for path in protected if path.exists()]
    if existing:
        raise RuntimeError(f"Multi-site LOSO freeze exists; refusing to overwrite: {existing}")

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    frozen_config = yaml.safe_load(PROTOCOL_FREEZE_PATH.read_text(encoding="utf-8"))
    protocol = config["phase6_multisite_leave_one_site_out"]
    if protocol != frozen_config["phase6_multisite_leave_one_site_out"]:
        raise RuntimeError("Current multi-site protocol differs from its pre-result snapshot")
    if protocol["status"] != "posthoc_protocol_frozen_before_execution":
        raise RuntimeError("Multi-site protocol status is not pre-execution frozen")

    population = load_population()
    alphas = [float(value) for value in protocol["inner_tuning"]["alpha_grid"]]
    all_predictor_columns = list(dict.fromkeys(TESSERA_FEATURES + TERRAIN_FEATURES))
    prediction_frames: list[pd.DataFrame] = []
    tuning_frames: list[pd.DataFrame] = []
    selection_frames: list[pd.DataFrame] = []
    model_artifacts: dict[str, Any] = {
        "protocol_snapshot_sha256": sha256(PROTOCOL_FREEZE_PATH),
        "sklearn_version": sklearn.__version__,
        "outer_folds": {},
    }

    for fold in protocol["outer_folds"]:
        held_out_site = str(fold["test_site"])
        training_sites = [str(site) for site in fold["train_sites"]]
        training = population[population["site_id"].isin(training_sites)].copy()
        target_predictors = population.loc[
            population["site_id"].eq(held_out_site),
            KEY_COLUMNS + all_predictor_columns,
        ].copy()
        predictions, tuning, selections, artifacts = fit_outer_fold(
            training, target_predictors, training_sites, held_out_site, alphas
        )
        prediction_frames.append(predictions)
        tuning_frames.append(tuning)
        selection_frames.append(selections)
        model_artifacts["outer_folds"][held_out_site] = artifacts

    predictions = pd.concat(prediction_frames, ignore_index=True)
    observations = population[KEY_COLUMNS + ["fhd_normal"]]
    predictions = predictions.merge(observations, on=KEY_COLUMNS, validate="many_to_one")
    predictions = predictions[
        KEY_COLUMNS
        + [
            "outer_held_out_site",
            "training_sites",
            "model",
            "selected_alpha",
            "fhd_normal",
            "prediction",
        ]
    ].sort_values(["outer_held_out_site", "model", "shot_number"]).reset_index(drop=True)
    tuning = pd.concat(tuning_frames, ignore_index=True).sort_values(
        ["outer_held_out_site", "model", "alpha", "inner_validation_site"]
    ).reset_index(drop=True)
    selections = pd.concat(selection_frames, ignore_index=True)
    selected_rows = selections[selections["alpha"].eq(selections["selected_alpha"])].copy()
    selected_rows = selected_rows.sort_values(["outer_held_out_site", "model"]).reset_index(drop=True)
    metrics = build_metrics(predictions)
    gate = evaluate_gate(metrics, str(protocol["primary_model"]))

    expected_prediction_rows = len(population) * len(MODELS)
    if len(predictions) != expected_prediction_rows:
        raise RuntimeError("Every footprint was not predicted once per model")
    if predictions.groupby(["site_id", "shot_number", "model"]).size().ne(1).any():
        raise RuntimeError("LOSO predictions are not unique by footprint and model")
    if len(tuning) != 3 * len(FEATURE_SETS) * len(alphas) * 2:
        raise RuntimeError("Nested site tuning grid is incomplete")

    write_parquet(predictions, PREDICTIONS_PATH)
    write_csv(metrics, METRICS_PATH)
    write_csv(tuning, TUNING_PATH)
    write_csv(selected_rows, SELECTION_PATH)
    write_joblib(model_artifacts, MODELS_PATH)
    save_figure(metrics)

    site_metrics = metrics[metrics["scope"].eq("site")]
    primary_metrics = site_metrics[
        site_metrics["model"].eq(protocol["primary_model"])
    ].set_index("held_out_site")
    outputs = {
        "predictions": output_record(PREDICTIONS_PATH, len(predictions)),
        "metrics": output_record(METRICS_PATH, len(metrics)),
        "inner_tuning": output_record(TUNING_PATH, len(tuning)),
        "selected_alphas": output_record(SELECTION_PATH, len(selected_rows)),
        "models": output_record(MODELS_PATH),
        "figure": output_record(FIGURE_PATH),
    }
    freeze_basis = {
        "protocol_status": protocol["status"],
        "scientific_role": protocol["scientific_role"],
        "protocol_snapshot": str(PROTOCOL_FREEZE_PATH.relative_to(ROOT)),
        "protocol_snapshot_sha256": sha256(PROTOCOL_FREEZE_PATH),
        "input_files": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [
                DEVELOPMENT_PATH,
                BART_PATH,
                DEVELOPMENT_CONVENTIONAL_PATH,
                BART_CONVENTIONAL_PATH,
            ]
        },
        "sites": protocol["sites"],
        "site_rows": {
            site: int(count) for site, count in population.groupby("site_id").size().items()
        },
        "outer_split": protocol["outer_split"],
        "outer_folds": protocol["outer_folds"],
        "target_site_labels_used_for_scaling_tuning_or_training": False,
        "target_predictor_columns_contain_fhd_normal": False,
        "inner_tuning": protocol["inner_tuning"],
        "outer_training_weighting": protocol["outer_training_weighting"],
        "feature_sets": {name: features for name, features in FEATURE_SETS.items()},
        "models": MODELS,
        "selected_alphas": {
            site: {
                model: float(value)
                for model, value in group.set_index("model")["selected_alpha"].items()
            }
            for site, group in selected_rows.groupby("outer_held_out_site")
        },
        "gate": gate,
        "primary_site_metrics": {
            site: {metric: float(primary_metrics.loc[site, metric]) for metric in METRICS}
            for site in sorted(primary_metrics.index)
        },
        "software": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    freeze_id = "phase6-multisite-loso-" + canonical_hash(
        {"freeze_basis": freeze_basis, "outputs": outputs}
    )[:12]
    freeze = {
        "freeze_id": freeze_id,
        "created_utc": utc_now(),
        "freeze_basis": freeze_basis,
        "outputs": outputs,
    }
    write_json(FREEZE_PATH, freeze)

    print(json.dumps({
        "freeze_id": freeze_id,
        "site_rows": freeze_basis["site_rows"],
        "selected_alphas": freeze_basis["selected_alphas"],
        "primary_site_metrics": freeze_basis["primary_site_metrics"],
        "gate": gate,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
