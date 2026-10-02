#!/usr/bin/env python3
"""Evaluate regional TESSERA FHD prediction with retained local GEDI passes."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import freeze_track_assisted_splits as split_tools  # noqa: E402
import run_context_height_footprint_height_transfer as base  # noqa: E402


PROTOCOL_PATH = ROOT / "metadata/project_config_phase12_track_assisted_protocol_freeze.yaml"
SPLIT_MANIFEST_PATH = ROOT / "metadata/phase12_track_assisted_split_manifest.json"
PHASE11_FREEZE_PATH = ROOT / "metadata/phase11_context_height_freeze.json"
PREDICTIONS_PATH = ROOT / "data/processed/phase12_track_assisted_predictions.parquet"
FOLD_METRICS_PATH = ROOT / "outputs/tables/phase12_track_assisted_fold_metrics.csv"
SITE_METRICS_PATH = ROOT / "outputs/tables/phase12_track_assisted_site_metrics.csv"
MACRO_METRICS_PATH = ROOT / "outputs/tables/phase12_track_assisted_macro_metrics.csv"
SOURCE_TUNING_PATH = ROOT / "outputs/tables/phase12_track_assisted_source_tuning.csv"
SOURCE_SELECTION_PATH = ROOT / "outputs/tables/phase12_track_assisted_source_selection.csv"
PAIRED_SITE_PATH = ROOT / "outputs/tables/phase12_track_assisted_paired_sites.csv"
BOOTSTRAP_PATH = ROOT / "outputs/tables/phase12_track_assisted_bootstrap.csv"
DISTANCE_PATH = ROOT / "outputs/tables/phase12_track_assisted_distance_metrics.csv"
FIGURE_PATH = ROOT / "outputs/figures/phase12_track_assisted_prediction.png"
FREEZE_PATH = ROOT / "metadata/phase12_track_assisted_prediction_freeze.json"

FEATURES = [f"tessera_area_{index:03d}" for index in range(128)]
READ_COLUMNS = [
    "site_id",
    "shot_number",
    "acquisition_datetime",
    "orbit",
    "reference_ground_track",
    "x_epsg5070",
    "y_epsg5070",
    "fhd_normal",
    *FEATURES,
]
MODEL_NAMES = [
    "source_training_mean",
    "source_tessera_ridge",
    "target_local_mean",
    "source_plus_local_offset",
    "target_local_tessera_ridge",
    "source_plus_local_residual_ridge",
]


def fit_local_ridge(
    features: np.ndarray,
    target: np.ndarray,
    alpha: float,
) -> tuple[StandardScaler, Ridge]:
    features = np.asarray(features, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    scaler = StandardScaler().fit(features)
    model = Ridge(alpha=float(alpha), solver="lsqr", tol=1e-6).fit(
        scaler.transform(features), target
    )
    return scaler, model


def select_source_alpha(tuning: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    ranking = (
        tuning.groupby("alpha", as_index=False)
        .agg(
            worst_inner_site_rmse=("rmse", "max"),
            macro_inner_site_rmse=("rmse", "mean"),
        )
        .sort_values(["worst_inner_site_rmse", "macro_inner_site_rmse", "alpha"])
        .reset_index(drop=True)
    )
    return float(ranking.iloc[0]["alpha"]), ranking


def tune_source_model(
    source: pd.DataFrame,
    target_site: str,
    alphas: list[float],
) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    source_sites = sorted(source["site_id"].unique())
    for validation_site in source_sites:
        fit = source[~source["site_id"].eq(validation_site)]
        validation = source[source["site_id"].eq(validation_site)]
        for alpha in alphas:
            scaler, model = base.fit_ridge(fit, FEATURES, "fhd_normal", alpha)
            prediction = model.predict(
                scaler.transform(validation[FEATURES].to_numpy(dtype=np.float64))
            )
            records.append(
                {
                    "target_site": target_site,
                    "inner_held_out_source_site": validation_site,
                    "alpha": float(alpha),
                    "training_rows": len(fit),
                    "validation_rows": len(validation),
                    **base.metric_values(
                        validation["fhd_normal"].to_numpy(dtype=np.float64), prediction
                    ),
                }
            )
    tuning = pd.DataFrame(records)
    selected, ranking = select_source_alpha(tuning)
    tuning["selected_alpha"] = selected
    ranking.insert(0, "target_site", target_site)
    ranking["selected_alpha"] = selected
    ranking["selected"] = np.isclose(ranking["alpha"], selected)
    return selected, tuning, ranking


def verify_fold(
    site_frame: pd.DataFrame,
    fold: dict[str, Any],
    buffer_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coordinates = site_frame[["x_epsg5070", "y_epsg5070"]].to_numpy(dtype=np.float64)
    identifiers = site_frame["pass_id"].to_numpy(dtype=str)
    training, test, distances = split_tools.buffered_local_split(
        coordinates, identifiers, fold["held_out_pass"], buffer_m
    )
    if len(training) != int(fold["local_training_rows_after_buffer"]):
        raise RuntimeError(f"Local training count changed for {fold['site_id']} {fold['held_out_pass']}")
    if len(test) != int(fold["test_rows"]):
        raise RuntimeError(f"Test count changed for {fold['site_id']} {fold['held_out_pass']}")
    if split_tools.shot_hash(site_frame.iloc[test]["shot_number"].to_numpy()) != fold["test_shot_sha256"]:
        raise RuntimeError("Held-out pass shot set changed")
    if split_tools.shot_hash(site_frame.iloc[training]["shot_number"].to_numpy()) != fold["local_training_shot_sha256"]:
        raise RuntimeError("Retained local training shot set changed")
    return training, test, distances


def build_site_and_macro_metrics(
    fold_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_columns = [
        "r2",
        "rmse",
        "mae",
        "pearson_r",
        "spearman_r",
        "mean_bias",
        "prediction_slope",
        "prediction_sd_ratio",
    ]
    site_records = []
    for (analysis_group, site, model), frame in fold_metrics.groupby(
        ["analysis_group", "site_id", "model"], sort=True
    ):
        site_records.append(
            {
                "analysis_group": analysis_group,
                "site_id": site,
                "model": model,
                "folds": len(frame),
                "test_rows": int(frame["test_rows"].sum()),
                **{column: float(frame[column].mean()) for column in metric_columns},
            }
        )
    site_metrics = pd.DataFrame(site_records)
    macro_records = []
    for (analysis_group, model), frame in site_metrics.groupby(
        ["analysis_group", "model"], sort=True
    ):
        macro_records.append(
            {
                "analysis_group": analysis_group,
                "model": model,
                "sites": len(frame),
                "folds": int(frame["folds"].sum()),
                "test_rows": int(frame["test_rows"].sum()),
                **{column: float(frame[column].mean()) for column in metric_columns},
            }
        )
    return site_metrics, pd.DataFrame(macro_records)


def paired_site_results(site_metrics: pd.DataFrame) -> pd.DataFrame:
    primary = site_metrics[site_metrics["analysis_group"].eq("primary")]
    comparisons = [
        ("source_plus_local_offset", "source_tessera_ridge"),
        ("source_plus_local_offset", "target_local_mean"),
        ("source_plus_local_residual_ridge", "source_plus_local_offset"),
        ("source_plus_local_residual_ridge", "target_local_tessera_ridge"),
        ("source_plus_local_residual_ridge", "target_local_mean"),
        ("target_local_tessera_ridge", "target_local_mean"),
    ]
    metrics = ["rmse", "r2", "spearman_r", "mean_bias"]
    records = []
    for candidate_name, reference_name in comparisons:
        candidate = primary[primary["model"].eq(candidate_name)].set_index("site_id")
        reference = primary[primary["model"].eq(reference_name)].set_index("site_id")
        if set(candidate.index) != set(reference.index):
            raise RuntimeError("Phase 12 paired site populations differ")
        for site in sorted(candidate.index):
            record: dict[str, Any] = {
                "comparison": f"{candidate_name}_minus_{reference_name}",
                "candidate_model": candidate_name,
                "reference_model": reference_name,
                "site_id": site,
            }
            for metric in metrics:
                record[f"candidate_{metric}"] = float(candidate.loc[site, metric])
                record[f"reference_{metric}"] = float(reference.loc[site, metric])
                record[f"delta_{metric}"] = float(
                    candidate.loc[site, metric] - reference.loc[site, metric]
                )
            records.append(record)
    return pd.DataFrame(records)


def paired_bootstrap(
    paired: pd.DataFrame,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    directions = {"rmse": "lower", "r2": "higher", "spearman_r": "higher", "mean_bias": "absolute_lower"}
    records = []
    for comparison, frame in paired.groupby("comparison", sort=True):
        for metric, direction in directions.items():
            if direction == "absolute_lower":
                differences = (
                    np.abs(frame[f"candidate_{metric}"].to_numpy(dtype=np.float64))
                    - np.abs(frame[f"reference_{metric}"].to_numpy(dtype=np.float64))
                )
            else:
                differences = frame[f"delta_{metric}"].to_numpy(dtype=np.float64)
            differences = differences[np.isfinite(differences)]
            if len(differences) == 0:
                continue
            draws = np.mean(
                rng.choice(differences, size=(replicates, len(differences)), replace=True),
                axis=1,
            )
            beneficial = draws < 0 if direction in {"lower", "absolute_lower"} else draws > 0
            site_beneficial = differences < 0 if direction in {"lower", "absolute_lower"} else differences > 0
            records.append(
                {
                    "comparison": comparison,
                    "metric": metric,
                    "beneficial_direction": direction,
                    "sites": len(differences),
                    "sites_improved": int(np.sum(site_beneficial)),
                    "mean_paired_difference": float(np.mean(differences)),
                    "site_bootstrap_95_low": float(np.quantile(draws, 0.025)),
                    "site_bootstrap_95_high": float(np.quantile(draws, 0.975)),
                    "bootstrap_probability_beneficial": float(np.mean(beneficial)),
                    "bootstrap_replicates": replicates,
                    "bootstrap_seed": seed,
                }
            )
    return pd.DataFrame(records)


def distance_metrics(predictions: pd.DataFrame, edges: list[float]) -> pd.DataFrame:
    labels = [
        f"{int(low)}-{int(high)}m" if np.isfinite(high) else f">={int(low)}m"
        for low, high in zip(edges[:-1], edges[1:], strict=True)
    ]
    primary = predictions[predictions["analysis_group"].eq("primary")].copy()
    primary["distance_bin"] = pd.cut(
        primary["nearest_local_training_distance_m"],
        bins=edges,
        labels=labels,
        right=False,
        include_lowest=True,
    )
    records = []
    for (model, distance_bin), frame in primary.groupby(
        ["model", "distance_bin"], observed=True, sort=True
    ):
        site_values = []
        for site, site_frame in frame.groupby("site_id", sort=True):
            values = base.metric_values(
                site_frame["observed_fhd"].to_numpy(dtype=np.float64),
                site_frame["prediction"].to_numpy(dtype=np.float64),
            )
            site_values.append(
                {
                    "site_id": site,
                    "rows": len(site_frame),
                    **values,
                }
            )
        sites = pd.DataFrame(site_values)
        records.append(
            {
                "model": model,
                "distance_bin": str(distance_bin),
                "sites": len(sites),
                "rows": int(sites["rows"].sum()),
                **{
                    metric: float(sites[metric].mean())
                    for metric in ["rmse", "mae", "spearman_r", "mean_bias"]
                },
            }
        )
    return pd.DataFrame(records)


def primary_gate(site_metrics: pd.DataFrame, macro_metrics: pd.DataFrame) -> dict[str, Any]:
    primary_sites = site_metrics[site_metrics["analysis_group"].eq("primary")]
    macro = macro_metrics[macro_metrics["analysis_group"].eq("primary")].set_index("model")
    offset = primary_sites[primary_sites["model"].eq("source_plus_local_offset")].set_index("site_id")
    source = primary_sites[primary_sites["model"].eq("source_tessera_ridge")].set_index("site_id")
    local_mean = primary_sites[primary_sites["model"].eq("target_local_mean")].set_index("site_id")
    improved_source = int((offset["rmse"] < source["rmse"]).sum())
    gate = {
        "source_plus_local_offset_macro_rmse_below_source_tessera_ridge": bool(
            macro.loc["source_plus_local_offset", "rmse"]
            < macro.loc["source_tessera_ridge", "rmse"]
        ),
        "source_plus_local_offset_improves_at_least_6_of_10_primary_sites": bool(
            improved_source >= 6
        ),
        "source_plus_local_offset_macro_rmse_below_target_local_mean": bool(
            macro.loc["source_plus_local_offset", "rmse"]
            < macro.loc["target_local_mean", "rmse"]
        ),
        "source_plus_local_offset_macro_r2_above_zero": bool(
            macro.loc["source_plus_local_offset", "r2"] > 0
        ),
        "sites_improved_vs_source_tessera": improved_source,
        "sites_improved_vs_target_local_mean": int(
            (offset["rmse"] < local_mean["rmse"]).sum()
        ),
    }
    gate["passed"] = bool(all(value for key, value in gate.items() if key not in {
        "sites_improved_vs_source_tessera", "sites_improved_vs_target_local_mean"
    }))
    return gate


def make_figure(site_metrics: pd.DataFrame, distance: pd.DataFrame) -> None:
    primary = site_metrics[site_metrics["analysis_group"].eq("primary")]
    sites = sorted(primary["site_id"].unique())
    x = np.arange(len(sites), dtype=float)
    styles = [
        ("source_tessera_ridge", "Source-only TESSERA", "#6C757D"),
        ("target_local_mean", "Retained-pass mean", "#EE9B00"),
        ("source_plus_local_offset", "TESSERA + local offset", "#0A9396"),
        ("source_plus_local_residual_ridge", "TESSERA + local residual", "#005F73"),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=False)
    width = 0.2
    for index, (model, label, color) in enumerate(styles):
        values = primary[primary["model"].eq(model)].set_index("site_id").loc[sites, "rmse"]
        axes[0].bar(x + (index - 1.5) * width, values, width=width, color=color, label=label)
    axes[0].set_xticks(x, sites, rotation=60)
    axes[0].set(ylabel="Complete-pass RMSE", title="Regional prediction at each forest")

    distance_order = ["500-1000m", "1000-2000m", "2000-5000m", ">=5000m"]
    for model, label, color in styles:
        frame = distance[distance["model"].eq(model)].set_index("distance_bin")
        values = [frame.loc[item, "rmse"] if item in frame.index else np.nan for item in distance_order]
        axes[1].plot(distance_order, values, marker="o", linewidth=1.8, color=color, label=label)
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].set(ylabel="Equal-forest RMSE", title="Error by distance to retained GEDI")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=4, frameon=False)
    figure.subplots_adjust(left=0.07, right=0.99, bottom=0.24, top=0.80, wspace=0.22)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=220)
    plt.close(figure)


def main() -> int:
    protected = [
        PREDICTIONS_PATH,
        FOLD_METRICS_PATH,
        SITE_METRICS_PATH,
        MACRO_METRICS_PATH,
        SOURCE_TUNING_PATH,
        SOURCE_SELECTION_PATH,
        PAIRED_SITE_PATH,
        BOOTSTRAP_PATH,
        DISTANCE_PATH,
        FIGURE_PATH,
        FREEZE_PATH,
    ]
    existing = [str(path.relative_to(ROOT)) for path in protected if path.exists()]
    if existing:
        raise RuntimeError(f"Phase 12 outputs already exist: {existing}")
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))[
        "phase12_track_assisted_prediction"
    ]
    split_manifest = json.loads(SPLIT_MANIFEST_PATH.read_text(encoding="utf-8"))
    if split_manifest["freeze_id"] != "phase12-track-splits-03eb62d46a19":
        raise RuntimeError("Unexpected Phase 12 split freeze")
    phase11 = json.loads(PHASE11_FREEZE_PATH.read_text(encoding="utf-8"))
    if phase11["freeze_id"] != protocol["prerequisite_freezes"]["phase11_context_height"]:
        raise RuntimeError("Unexpected Phase 11 prerequisite")

    cohort = pd.concat(
        [pd.read_parquet(path, columns=READ_COLUMNS) for path in base.INPUT_PATHS],
        ignore_index=True,
    )
    cohort["pass_id"] = split_tools.pass_ids(cohort)
    if len(cohort) != 15_326 or cohort[["site_id", "shot_number"]].duplicated().any():
        raise RuntimeError("Phase 12 cohort changed")
    if not np.isfinite(cohort[["fhd_normal", *FEATURES]].to_numpy(dtype=np.float32)).all():
        raise RuntimeError("Phase 12 cohort contains non-finite target or predictors")

    primary_sites = list(split_manifest["freeze_basis"]["primary_sites"])
    secondary_sites = list(split_manifest["freeze_basis"]["two_pass_sensitivity_sites"])
    target_sites = [*primary_sites, *secondary_sites]
    eligible_folds = {
        (fold["site_id"], fold["held_out_pass"]): fold
        for fold in split_manifest["folds"]
        if fold["eligible_by_row_gates"]
        and fold["site_id"] in target_sites
    }
    expected_folds = int(split_manifest["freeze_basis"]["primary_folds"]) + int(
        split_manifest["freeze_basis"]["two_pass_sensitivity_folds"]
    )
    if len(eligible_folds) != expected_folds:
        raise RuntimeError("Phase 12 eligible fold population changed")

    alphas = [float(value) for value in protocol["source_model"]["alpha_grid"]]
    buffer_m = float(protocol["outer_split"]["exclusion_buffer_m"])
    prediction_frames: list[pd.DataFrame] = []
    fold_metric_records: list[dict[str, Any]] = []
    tuning_frames: list[pd.DataFrame] = []
    selection_frames: list[pd.DataFrame] = []

    for site_index, target_site in enumerate(target_sites, start=1):
        analysis_group = "primary" if target_site in primary_sites else "two_pass_sensitivity"
        source = cohort[~cohort["site_id"].eq(target_site)].copy()
        target = cohort[cohort["site_id"].eq(target_site)].copy().reset_index(drop=True)
        print(f"[{site_index:02d}/{len(target_sites)}] {target_site}: tune source model", flush=True)
        selected_alpha, tuning, selection = tune_source_model(source, target_site, alphas)
        source_scaler, source_model = base.fit_ridge(
            source, FEATURES, "fhd_normal", selected_alpha
        )
        source_target_prediction = source_model.predict(
            source_scaler.transform(target[FEATURES].to_numpy(dtype=np.float64))
        )
        source_mean = base.equal_site_mean(source, "fhd_normal")
        tuning_frames.append(tuning)
        selection_frames.append(selection)
        site_folds = [
            fold
            for (site_id, _), fold in eligible_folds.items()
            if site_id == target_site
        ]
        for fold_index, fold in enumerate(sorted(site_folds, key=lambda item: item["held_out_pass"]), start=1):
            local_index, test_index, distances = verify_fold(target, fold, buffer_m)
            local = target.iloc[local_index]
            test = target.iloc[test_index]
            y_local = local["fhd_normal"].to_numpy(dtype=np.float64)
            y_test = test["fhd_normal"].to_numpy(dtype=np.float64)
            source_local = source_target_prediction[local_index]
            source_test = source_target_prediction[test_index]
            local_offset = float(np.mean(y_local - source_local))

            local_scaler, local_model = fit_local_ridge(
                local[FEATURES].to_numpy(dtype=np.float64), y_local, selected_alpha
            )
            local_prediction = local_model.predict(
                local_scaler.transform(test[FEATURES].to_numpy(dtype=np.float64))
            )
            residual_scaler, residual_model = fit_local_ridge(
                local[FEATURES].to_numpy(dtype=np.float64),
                y_local - source_local,
                selected_alpha,
            )
            residual_prediction = source_test + residual_model.predict(
                residual_scaler.transform(test[FEATURES].to_numpy(dtype=np.float64))
            )
            model_predictions = {
                "source_training_mean": np.full(len(test), source_mean),
                "source_tessera_ridge": source_test,
                "target_local_mean": np.full(len(test), float(np.mean(y_local))),
                "source_plus_local_offset": source_test + local_offset,
                "target_local_tessera_ridge": local_prediction,
                "source_plus_local_residual_ridge": residual_prediction,
            }
            common = test[["site_id", "shot_number", "pass_id"]].copy()
            common["analysis_group"] = analysis_group
            common["held_out_pass"] = fold["held_out_pass"]
            common["observed_fhd"] = y_test
            common["nearest_local_training_distance_m"] = distances
            common["local_training_rows"] = len(local)
            common["local_training_passes"] = int(local["pass_id"].nunique())
            common["selected_source_alpha"] = selected_alpha
            common["local_offset"] = local_offset
            for model_name, values in model_predictions.items():
                prediction = common.copy()
                prediction["model"] = model_name
                prediction["prediction"] = values
                prediction_frames.append(prediction)
                fold_metric_records.append(
                    {
                        "analysis_group": analysis_group,
                        "site_id": target_site,
                        "held_out_pass": fold["held_out_pass"],
                        "model": model_name,
                        "test_rows": len(test),
                        "local_training_rows": len(local),
                        "local_training_passes": int(local["pass_id"].nunique()),
                        "minimum_distance_m": float(np.min(distances)),
                        "median_distance_m": float(np.median(distances)),
                        "maximum_distance_m": float(np.max(distances)),
                        "selected_source_alpha": selected_alpha,
                        "local_offset": local_offset,
                        **base.metric_values(y_test, values),
                    }
                )
            print(
                f"  fold {fold_index}/{len(site_folds)} {fold['held_out_pass']}: "
                f"test={len(test)}, local={len(local)}, nearest={np.min(distances):.0f}m",
                flush=True,
            )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    fold_metrics = pd.DataFrame(fold_metric_records)
    tuning = pd.concat(tuning_frames, ignore_index=True)
    selection = pd.concat(selection_frames, ignore_index=True)
    if predictions[["site_id", "shot_number", "model"]].duplicated().any():
        raise RuntimeError("Phase 12 predictions contain duplicate held-out rows")
    if not np.isfinite(predictions[["observed_fhd", "prediction", "nearest_local_training_distance_m"]]).all().all():
        raise RuntimeError("Phase 12 predictions contain non-finite values")
    site_metrics, macro_metrics = build_site_and_macro_metrics(fold_metrics)
    paired = paired_site_results(site_metrics)
    bootstrap = paired_bootstrap(
        paired,
        int(protocol["uncertainty"]["paired_site_bootstrap_replicates"]),
        int(protocol["uncertainty"]["seed"]),
    )
    edges = [float(value) if value != "infinity" else math.inf for value in protocol["distance_analysis"]["bins_m"]]
    distance = distance_metrics(predictions, edges)
    gate = primary_gate(site_metrics, macro_metrics)

    base.write_parquet(predictions, PREDICTIONS_PATH)
    base.write_csv(fold_metrics, FOLD_METRICS_PATH)
    base.write_csv(site_metrics, SITE_METRICS_PATH)
    base.write_csv(macro_metrics, MACRO_METRICS_PATH)
    base.write_csv(tuning, SOURCE_TUNING_PATH)
    base.write_csv(selection, SOURCE_SELECTION_PATH)
    base.write_csv(paired, PAIRED_SITE_PATH)
    base.write_csv(bootstrap, BOOTSTRAP_PATH)
    base.write_csv(distance, DISTANCE_PATH)
    make_figure(site_metrics, distance)
    freeze_basis = {
        "protocol_sha256": base.sha256(PROTOCOL_PATH),
        "split_freeze_id": split_manifest["freeze_id"],
        "phase11_context_height_freeze_id": phase11["freeze_id"],
        "script_sha256": base.sha256(Path(__file__).resolve()),
        "input_hashes": {
            str(path.relative_to(ROOT)): base.sha256(path) for path in base.INPUT_PATHS
        },
        "rows": len(cohort),
        "primary_sites": primary_sites,
        "primary_folds": int(split_manifest["freeze_basis"]["primary_folds"]),
        "two_pass_sensitivity_sites": secondary_sites,
        "two_pass_sensitivity_folds": int(
            split_manifest["freeze_basis"]["two_pass_sensitivity_folds"]
        ),
        "models": protocol["models"],
        "source_model": protocol["source_model"],
        "test_pass_fhd_used_in_training_or_selection": False,
        "analysis_is_retrospective": True,
        "gate": gate,
        "sklearn_version": sklearn.__version__,
    }
    freeze = {
        "freeze_id": "phase12-track-assisted-" + base.canonical_hash(freeze_basis)[:12],
        "created_utc": base.utc_now(),
        "status": "complete_track_assisted_gate_passed" if gate["passed"] else "complete_track_assisted_gate_failed",
        "freeze_basis": freeze_basis,
        "outputs": {
            str(path.relative_to(ROOT)): base.sha256(path)
            for path in [
                PREDICTIONS_PATH,
                FOLD_METRICS_PATH,
                SITE_METRICS_PATH,
                MACRO_METRICS_PATH,
                SOURCE_TUNING_PATH,
                SOURCE_SELECTION_PATH,
                PAIRED_SITE_PATH,
                BOOTSTRAP_PATH,
                DISTANCE_PATH,
                FIGURE_PATH,
            ]
        },
    }
    base.write_json(FREEZE_PATH, freeze)
    primary_macro = macro_metrics[macro_metrics["analysis_group"].eq("primary")].set_index("model")
    print(
        json.dumps(
            {
                "freeze_id": freeze["freeze_id"],
                "gate": gate,
                "primary_macro": primary_macro[
                    ["r2", "rmse", "mae", "spearman_r", "mean_bias"]
                ].to_dict(orient="index"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
