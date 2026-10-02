#!/usr/bin/env python3
"""Evaluate multi-scale TESSERA context for complete-site GEDI RH100 transfer."""

from __future__ import annotations

import argparse
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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_context_height_footprint_height_transfer as base  # noqa: E402


PROTOCOL_PATH = ROOT / "metadata/project_config_phase11_context_height_model_protocol_freeze.yaml"
CONTEXT_FREEZE_PATH = ROOT / "metadata/phase11_tessera_context_freeze.json"
FOOTPRINT_FREEZE_PATH = ROOT / "metadata/phase11_footprint_height_freeze.json"
CONTEXT_PATH = ROOT / "data/processed/phase11_tessera_context_features.parquet"
PREDICTIONS_PATH = ROOT / "data/processed/phase11_context_height_predictions.parquet"
METRICS_PATH = ROOT / "outputs/tables/phase11_context_height_metrics.csv"
TUNING_PATH = ROOT / "outputs/tables/phase11_context_height_tuning.csv"
SELECTION_PATH = ROOT / "outputs/tables/phase11_context_height_selection.csv"
COMPARISON_PATH = ROOT / "outputs/tables/phase11_context_height_comparison.csv"
FIGURE_PATH = ROOT / "outputs/figures/phase11_context_height_transfer.png"
FREEZE_PATH = ROOT / "metadata/phase11_context_height_freeze.json"

WINDOW_WIDTHS_M = [30, 70, 150]
AREA_FEATURES = [f"tessera_area_{index:03d}" for index in range(128)]
CONTEXT_READ_COLUMNS = base.KEYS + [
    f"tessera_ctx{width:03d}_{summary}_{dimension:03d}"
    for width in WINDOW_WIDTHS_M
    for summary in ["mean", "std", "delta"]
    for dimension in range(128)
]


def feature_columns(width_m: int) -> list[str]:
    """Return the non-redundant 384-feature context representation."""
    if width_m not in WINDOW_WIDTHS_M:
        raise ValueError(f"Unsupported context width: {width_m}")
    return AREA_FEATURES + [
        f"tessera_ctx{width_m:03d}_{summary}_{dimension:03d}"
        for summary in ["std", "delta"]
        for dimension in range(128)
    ]


def fit_scaled_ridge(
    training: pd.DataFrame,
    features: list[str],
    alpha: float,
) -> tuple[StandardScaler, Ridge]:
    x = training[features].to_numpy(dtype=np.float64)
    y = training["rh100_m"].to_numpy(dtype=np.float64)
    weights = base.equal_site_weights(training["site_id"].to_numpy())
    scaler = StandardScaler().fit(x, sample_weight=weights)
    model = Ridge(alpha=float(alpha), solver="lsqr", tol=1e-6).fit(
        scaler.transform(x), y, sample_weight=weights
    )
    return scaler, model


def select_width_alpha(tuning: pd.DataFrame) -> tuple[int, float, pd.DataFrame]:
    ranking = (
        tuning.groupby(["window_width_m", "alpha"], as_index=False)
        .agg(
            worst_inner_site_rmse=("rmse", "max"),
            macro_inner_site_rmse=("rmse", "mean"),
        )
        .sort_values(
            [
                "worst_inner_site_rmse",
                "macro_inner_site_rmse",
                "window_width_m",
                "alpha",
            ]
        )
        .reset_index(drop=True)
    )
    winner = ranking.iloc[0]
    return int(winner["window_width_m"]), float(winner["alpha"]), ranking


def tune_context_ridge(
    training: pd.DataFrame,
    widths: list[int],
    alphas: list[float],
    outer_site: str,
) -> tuple[int, float, pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    sites = sorted(training["site_id"].unique())
    for inner_index, validation_site in enumerate(sites, start=1):
        fit = training[~training["site_id"].eq(validation_site)]
        validation = training[training["site_id"].eq(validation_site)]
        weights = base.equal_site_weights(fit["site_id"].to_numpy())
        y_fit = fit["rh100_m"].to_numpy(dtype=np.float64)
        for width in widths:
            features = feature_columns(width)
            scaler = StandardScaler().fit(
                fit[features].to_numpy(dtype=np.float64), sample_weight=weights
            )
            x_fit = scaler.transform(fit[features].to_numpy(dtype=np.float64))
            x_validation = scaler.transform(
                validation[features].to_numpy(dtype=np.float64)
            )
            for alpha in alphas:
                model = Ridge(alpha=float(alpha), solver="lsqr", tol=1e-6).fit(
                    x_fit, y_fit, sample_weight=weights
                )
                prediction = model.predict(x_validation)
                records.append(
                    {
                        "outer_held_out_site": outer_site,
                        "inner_held_out_site": validation_site,
                        "window_width_m": width,
                        "alpha": float(alpha),
                        "training_rows": len(fit),
                        "validation_rows": len(validation),
                        **base.metric_values(
                            validation["rh100_m"].to_numpy(), prediction
                        ),
                    }
                )
        print(
            f"  {outer_site}: inner site {inner_index:02d}/{len(sites)} {validation_site}",
            flush=True,
        )
    tuning = pd.DataFrame(records)
    selected_width, selected_alpha, ranking = select_width_alpha(tuning)
    tuning["selected_window_width_m"] = selected_width
    tuning["selected_alpha"] = selected_alpha
    ranking.insert(0, "outer_held_out_site", outer_site)
    ranking["selected_window_width_m"] = selected_width
    ranking["selected_alpha"] = selected_alpha
    return selected_width, selected_alpha, tuning, ranking


def fit_fixed_hgb(
    training: pd.DataFrame,
    features: list[str],
    parameters: dict[str, Any],
) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(**parameters)
    model.fit(
        training[features].to_numpy(dtype=np.float64),
        training["rh100_m"].to_numpy(dtype=np.float64),
        sample_weight=base.equal_site_weights(training["site_id"].to_numpy()),
    )
    return model


def verify_context_algebra(cohort: pd.DataFrame) -> float:
    maximum_error = 0.0
    area = cohort[AREA_FEATURES].to_numpy(dtype=np.float32)
    for width in WINDOW_WIDTHS_M:
        means = cohort[
            [f"tessera_ctx{width:03d}_mean_{index:03d}" for index in range(128)]
        ].to_numpy(dtype=np.float32)
        deltas = cohort[
            [f"tessera_ctx{width:03d}_delta_{index:03d}" for index in range(128)]
        ].to_numpy(dtype=np.float32)
        maximum_error = max(
            maximum_error,
            float(np.max(np.abs((area - means) - deltas))),
        )
    if maximum_error > 2e-6:
        raise RuntimeError(
            f"Context delta no longer equals footprint minus window mean: {maximum_error}"
        )
    return maximum_error


def build_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for model_name, model_frame in predictions.groupby("model", sort=True):
        site_records = []
        for site, frame in model_frame.groupby("site_id", sort=True):
            row = {
                "model": model_name,
                "scope": "site",
                "site_id": site,
                "rows": len(frame),
                **base.metric_values(
                    frame["observed_rh100_m"].to_numpy(),
                    frame["prediction"].to_numpy(),
                ),
            }
            records.append(row)
            site_records.append(row)
        sites = pd.DataFrame(site_records)
        records.append(
            {
                "model": model_name,
                "scope": "macro",
                "site_id": "macro_mean",
                "rows": int(sites["rows"].sum()),
                **{
                    metric: float(sites[metric].mean())
                    for metric in [
                        "r2",
                        "rmse",
                        "mae",
                        "pearson_r",
                        "spearman_r",
                        "mean_bias",
                        "prediction_slope",
                        "prediction_sd_ratio",
                    ]
                },
            }
        )
    return pd.DataFrame(records)


def build_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    footprint_metrics = pd.read_csv(base.METRICS_PATH)
    footprint = footprint_metrics[
        footprint_metrics["model"].eq("tessera_area_ridge")
        & footprint_metrics["scope"].eq("site")
    ][["site_id", "r2", "rmse", "mean_bias"]].rename(
        columns={
            "r2": "footprint_r2",
            "rmse": "footprint_rmse",
            "mean_bias": "footprint_mean_bias",
        }
    )
    context = metrics[
        metrics["model"].eq("tessera_context_ridge")
        & metrics["scope"].eq("site")
    ][["site_id", "r2", "rmse", "mean_bias"]].rename(
        columns={
            "r2": "context_r2",
            "rmse": "context_rmse",
            "mean_bias": "context_mean_bias",
        }
    )
    comparison = footprint.merge(context, on="site_id", validate="one_to_one")
    comparison["context_minus_footprint_r2"] = (
        comparison["context_r2"] - comparison["footprint_r2"]
    )
    comparison["context_minus_footprint_rmse"] = (
        comparison["context_rmse"] - comparison["footprint_rmse"]
    )
    return comparison.sort_values("site_id").reset_index(drop=True)


def make_figure(metrics: pd.DataFrame, comparison: pd.DataFrame) -> None:
    site_metrics = metrics[metrics["scope"].eq("site")]
    sites = sorted(site_metrics["site_id"].unique())
    x = np.arange(len(sites), dtype=float)
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), constrained_layout=False)
    styles = [
        ("tessera_context_ridge", "Context Ridge", "#0A9396", -0.2),
        ("tessera_context_hgb", "Fixed nonlinear sensitivity", "#CA6702", 0.2),
    ]
    for model_name, label, color, offset in styles:
        values = (
            site_metrics[site_metrics["model"].eq(model_name)]
            .set_index("site_id")
            .loc[sites, "r2"]
            .to_numpy()
        )
        axes[0].bar(x + offset, values, width=0.38, color=color, label=label)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_xticks(x, sites, rotation=60)
    axes[0].set(ylabel="R2", title="Canopy-height transfer to each held-out forest")
    handles, labels = axes[0].get_legend_handles_labels()

    colors = np.where(
        comparison["context_minus_footprint_rmse"].to_numpy() < 0,
        "#0A9396",
        "#BB3E03",
    )
    axes[1].bar(
        comparison["site_id"],
        comparison["context_minus_footprint_rmse"],
        color=colors,
    )
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].tick_params(axis="x", rotation=60)
    axes[1].set(
        ylabel="Context minus footprint RMSE (m)",
        title="Change from adding spatial context",
    )
    figure.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.07, 0.995),
        ncol=2,
        frameon=False,
    )
    figure.subplots_adjust(left=0.075, right=0.99, bottom=0.23, top=0.82, wspace=0.2)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=220)
    plt.close(figure)


def revise_figure_only() -> int:
    """Regenerate presentation only while preserving frozen model outputs."""
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    metrics = pd.read_csv(METRICS_PATH)
    comparison = pd.read_csv(COMPARISON_PATH)
    predecessor = freeze["freeze_id"]
    for path in [PREDICTIONS_PATH, METRICS_PATH, TUNING_PATH, SELECTION_PATH, COMPARISON_PATH]:
        expected = freeze["outputs"][str(path.relative_to(ROOT))]
        if base.sha256(path) != expected:
            raise RuntimeError(f"Frozen model output changed before figure revision: {path}")
    make_figure(metrics, comparison)
    freeze["freeze_basis"]["script_sha256"] = base.sha256(Path(__file__).resolve())
    freeze["freeze_basis"]["visualization_revision"] = {
        "predecessor_freeze_id": predecessor,
        "model_predictions_metrics_tuning_and_selection_unchanged": True,
        "reason": "move_legend_clear_of_panel_title",
    }
    freeze["outputs"][str(FIGURE_PATH.relative_to(ROOT))] = base.sha256(FIGURE_PATH)
    freeze["freeze_id"] = "phase11-context-height-" + base.canonical_hash(
        freeze["freeze_basis"]
    )[:12]
    freeze["created_utc"] = base.utc_now()
    base.write_json(FREEZE_PATH, freeze)
    print(json.dumps({"freeze_id": freeze["freeze_id"], "predecessor": predecessor}, indent=2))
    return 0


def main() -> int:
    protected = [
        PREDICTIONS_PATH,
        METRICS_PATH,
        TUNING_PATH,
        SELECTION_PATH,
        COMPARISON_PATH,
        FIGURE_PATH,
        FREEZE_PATH,
    ]
    existing = [str(path.relative_to(ROOT)) for path in protected if path.exists()]
    if existing:
        raise RuntimeError(f"Phase 11 context-height outputs already exist: {existing}")

    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))[
        "phase11_context_height_model"
    ]
    context_freeze = json.loads(CONTEXT_FREEZE_PATH.read_text(encoding="utf-8"))
    footprint_freeze = json.loads(FOOTPRINT_FREEZE_PATH.read_text(encoding="utf-8"))
    if context_freeze["freeze_id"] != protocol["prerequisite_freezes"]["target_free_context"]:
        raise RuntimeError("Unexpected Phase 11 context freeze")
    if footprint_freeze["freeze_id"] != protocol["prerequisite_freezes"]["footprint_height"]:
        raise RuntimeError("Unexpected Phase 11 footprint-height freeze")

    outcome_columns = base.READ_COLUMNS
    outcomes = pd.concat(
        [pd.read_parquet(path, columns=outcome_columns) for path in base.INPUT_PATHS],
        ignore_index=True,
    )
    outcomes["rh100_m"] = base.height_target(outcomes)
    context = pd.read_parquet(CONTEXT_PATH, columns=CONTEXT_READ_COLUMNS)
    cohort = outcomes.merge(context, on=base.KEYS, how="inner", validate="one_to_one")
    expected_sites = sorted(protocol["cohort"]["sites"])
    if len(cohort) != int(protocol["cohort"]["expected_rows"]):
        raise RuntimeError("Context-height cohort row count changed")
    if sorted(cohort["site_id"].unique()) != expected_sites or cohort[base.KEYS].duplicated().any():
        raise RuntimeError("Context-height cohort sites or keys changed")
    all_features = sorted({feature for width in WINDOW_WIDTHS_M for feature in feature_columns(width)})
    if not np.isfinite(cohort[all_features + ["rh100_m"]].to_numpy(dtype=np.float32)).all():
        raise RuntimeError("Context-height cohort contains non-finite values")
    algebra_error = verify_context_algebra(cohort)

    widths = [int(value) for value in protocol["context_representations"]["candidate_window_widths_m"]]
    alphas = [float(value) for value in protocol["primary_model"]["alpha_grid"]]
    hgb_parameters = dict(protocol["nonlinear_sensitivity"]["parameters"])
    prediction_frames: list[pd.DataFrame] = []
    tuning_frames: list[pd.DataFrame] = []
    selection_frames: list[pd.DataFrame] = []

    for outer_index, held_out in enumerate(expected_sites, start=1):
        print(f"[{outer_index:02d}/{len(expected_sites)}] outer held-out site {held_out}", flush=True)
        training = cohort[~cohort["site_id"].eq(held_out)].copy()
        target = cohort[cohort["site_id"].eq(held_out)].copy()
        selected_width, selected_alpha, tuning, ranking = tune_context_ridge(
            training, widths, alphas, held_out
        )
        features = feature_columns(selected_width)
        scaler, ridge = fit_scaled_ridge(training, features, selected_alpha)
        ridge_prediction = ridge.predict(
            scaler.transform(target[features].to_numpy(dtype=np.float64))
        )
        print(
            f"  selected {selected_width} m, alpha={selected_alpha:g}; fit fixed nonlinear sensitivity",
            flush=True,
        )
        hgb = fit_fixed_hgb(training, features, hgb_parameters)
        hgb_prediction = hgb.predict(target[features].to_numpy(dtype=np.float64))
        baseline_prediction = np.full(
            len(target), base.equal_site_mean(training, "rh100_m")
        )
        common = target[base.KEYS + ["longitude", "latitude", "rh100_m"]].rename(
            columns={"rh100_m": "observed_rh100_m"}
        )
        for model_name, prediction in [
            ("equal_site_training_mean", baseline_prediction),
            ("tessera_context_ridge", ridge_prediction),
            ("tessera_context_hgb", hgb_prediction),
        ]:
            frame = common.copy()
            frame["model"] = model_name
            frame["prediction"] = prediction
            frame["outer_held_out_site"] = held_out
            frame["selected_window_width_m"] = (
                selected_width if model_name != "equal_site_training_mean" else math.nan
            )
            frame["selected_alpha"] = (
                selected_alpha if model_name == "tessera_context_ridge" else math.nan
            )
            prediction_frames.append(frame)
        tuning_frames.append(tuning)
        ranking["selected"] = (
            ranking["window_width_m"].eq(selected_width)
            & np.isclose(ranking["alpha"], selected_alpha)
        )
        selection_frames.append(ranking)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    tuning = pd.concat(tuning_frames, ignore_index=True)
    selection = pd.concat(selection_frames, ignore_index=True)
    metrics = build_metrics(predictions)
    comparison = build_comparison(metrics)
    primary = metrics[
        metrics["model"].eq("tessera_context_ridge")
        & metrics["scope"].eq("macro")
    ].iloc[0]
    baseline = metrics[
        metrics["model"].eq("equal_site_training_mean")
        & metrics["scope"].eq("macro")
    ].iloc[0]
    gate = {
        "primary_model": "tessera_context_ridge",
        "macro_r2_above_zero": bool(primary["r2"] > 0),
        "macro_rmse_below_equal_site_training_mean": bool(
            primary["rmse"] < baseline["rmse"]
        ),
    }
    gate["passed"] = bool(
        gate["macro_r2_above_zero"]
        and gate["macro_rmse_below_equal_site_training_mean"]
    )

    base.write_parquet(predictions, PREDICTIONS_PATH)
    base.write_csv(metrics, METRICS_PATH)
    base.write_csv(tuning, TUNING_PATH)
    base.write_csv(selection, SELECTION_PATH)
    base.write_csv(comparison, COMPARISON_PATH)
    make_figure(metrics, comparison)
    freeze_basis = {
        "protocol_sha256": base.sha256(PROTOCOL_PATH),
        "context_freeze_id": context_freeze["freeze_id"],
        "footprint_height_freeze_id": footprint_freeze["freeze_id"],
        "script_sha256": base.sha256(Path(__file__).resolve()),
        "input_hashes": {
            str(path.relative_to(ROOT)): base.sha256(path)
            for path in [*base.INPUT_PATHS, CONTEXT_PATH]
        },
        "rows": len(cohort),
        "sites": expected_sites,
        "target": "RH100 = elev_highestreturn - elev_lowestmode",
        "fhd_normal_columns_read": [],
        "fhd_normal_used": False,
        "candidate_window_widths_m": widths,
        "predictor_count_per_candidate": len(feature_columns(widths[0])),
        "context_algebra_max_absolute_error": algebra_error,
        "primary_model": protocol["primary_model"],
        "nonlinear_sensitivity": protocol["nonlinear_sensitivity"],
        "sklearn_version": sklearn.__version__,
        "gate": gate,
    }
    freeze_id = "phase11-context-height-" + base.canonical_hash(freeze_basis)[:12]
    freeze = {
        "freeze_id": freeze_id,
        "created_utc": base.utc_now(),
        "status": "complete_context_height_gate_passed" if gate["passed"] else "complete_context_height_gate_failed",
        "freeze_basis": freeze_basis,
        "outputs": {
            str(path.relative_to(ROOT)): base.sha256(path)
            for path in [
                PREDICTIONS_PATH,
                METRICS_PATH,
                TUNING_PATH,
                SELECTION_PATH,
                COMPARISON_PATH,
                FIGURE_PATH,
            ]
        },
    }
    base.write_json(FREEZE_PATH, freeze)
    macro = metrics[metrics["scope"].eq("macro")].set_index("model")
    print(
        json.dumps(
            {
                "freeze_id": freeze_id,
                "gate": gate,
                "macro_metrics": macro[["r2", "rmse", "mae", "spearman_r", "mean_bias"]].to_dict(orient="index"),
                "selected_window_counts": selection[selection["selected"]]["window_width_m"].value_counts().sort_index().to_dict(),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure-only", action="store_true")
    arguments = parser.parse_args()
    raise SystemExit(revise_figure_only() if arguments.figure_only else main())
