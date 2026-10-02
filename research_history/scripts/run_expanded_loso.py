#!/usr/bin/env python3
"""Run nested leave-one-site-out TESSERA Ridge across all eight viable sites."""

from __future__ import annotations

import hashlib
import json
import math
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


PROTOCOL_PATH = ROOT / "metadata/project_config_phase7_site_expansion_protocol_freeze.yaml"
ORIGINAL_DEVELOPMENT_PATH = ROOT / "data/processed/tessera_aligned_development_soap_teak.parquet"
ORIGINAL_BART_PATH = ROOT / "data/processed/tessera_aligned_locked_bart.parquet"
EXPANSION_PATH = ROOT / "data/processed/phase7_tessera_aligned_expansion.parquet"
PROSPECTIVE_EVALUATION_FREEZE_PATH = ROOT / "metadata/phase7_prospective_evaluation_freeze.json"
PREDICTIONS_PATH = ROOT / "data/processed/phase7_expanded_loso_predictions.parquet"
METRICS_PATH = ROOT / "outputs/tables/phase7_expanded_loso_metrics.csv"
TUNING_PATH = ROOT / "outputs/tables/phase7_expanded_loso_inner_tuning.csv"
SELECTION_PATH = ROOT / "outputs/tables/phase7_expanded_loso_selected_alphas.csv"
MODELS_PATH = ROOT / "outputs/models/phase7_expanded_loso_models.joblib"
FIGURE_PATH = ROOT / "outputs/figures/phase7_expanded_loso.png"
FREEZE_PATH = ROOT / "metadata/phase7_expanded_loso_freeze.json"
KEYS = ["site_id", "shot_number"]
FEATURES = [f"tessera_area_{index:03d}" for index in range(128)]
METHODS = ["training_site_mean", "tessera_area_ridge"]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def tune_inner_sites(
    training: pd.DataFrame,
    training_sites: list[str],
    outer_held_out_site: str,
    alphas: list[float],
) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    for alpha in alphas:
        for validation_site in training_sites:
            inner_training = training[~training["site_id"].eq(validation_site)]
            validation = training[training["site_id"].eq(validation_site)]
            weights = loso.equal_site_weights(inner_training["site_id"].to_numpy())
            scaler, model = loso.fit_weighted_ridge(
                inner_training[FEATURES].to_numpy(dtype=np.float64),
                inner_training["fhd_normal"].to_numpy(dtype=np.float64),
                alpha,
                weights,
            )
            predicted = model.predict(
                scaler.transform(validation[FEATURES].to_numpy(dtype=np.float64))
            )
            records.append({
                "outer_held_out_site": outer_held_out_site,
                "alpha": alpha,
                "inner_validation_site": validation_site,
                "inner_training_sites": "+".join(sorted(set(training_sites) - {validation_site})),
                "training_rows": len(inner_training),
                "validation_rows": len(validation),
                **loso.metric_values(
                    validation["fhd_normal"].to_numpy(dtype=np.float64), predicted
                ),
            })
    tuning = pd.DataFrame(records)
    selected, ranking = loso.select_alpha(tuning)
    tuning["selected_alpha"] = selected
    ranking.insert(0, "outer_held_out_site", outer_held_out_site)
    ranking["selected_alpha"] = selected
    return selected, tuning, ranking


def build_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (site, method), frame in predictions.groupby(["outer_held_out_site", "method"], sort=True):
        records.append({
            "scope": "site", "held_out_site": site, "method": method, "rows": len(frame),
            **loso.metric_values(frame["fhd_normal"].to_numpy(), frame["prediction"].to_numpy()),
        })
    site_metrics = pd.DataFrame(records)
    for method, frame in site_metrics.groupby("method", sort=True):
        records.append({
            "scope": "macro", "held_out_site": "macro_mean", "method": method, "rows": len(frame),
            **{metric: float(frame[metric].mean()) for metric in loso.METRICS},
        })
    for method, frame in predictions.groupby("method", sort=True):
        records.append({
            "scope": "pooled", "held_out_site": "all_sites", "method": method, "rows": len(frame),
            **loso.metric_values(frame["fhd_normal"].to_numpy(), frame["prediction"].to_numpy()),
        })
    return pd.DataFrame(records).sort_values(["scope", "held_out_site", "method"]).reset_index(drop=True)


def save_figure(metrics: pd.DataFrame) -> None:
    site = metrics[metrics["scope"].eq("site")]
    sites = sorted(site["held_out_site"].unique())
    x = np.arange(len(sites))
    width = 0.36
    colours = {"training_site_mean": "#6c757d", "tessera_area_ridge": "#0a9396"}
    labels = {"training_site_mean": "training-site mean", "tessera_area_ridge": "TESSERA Ridge"}
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    for index, method in enumerate(METHODS):
        frame = site[site["method"].eq(method)].set_index("held_out_site").loc[sites]
        offset = (index - 0.5) * width
        axes[0].bar(x + offset, frame["rmse"], width, color=colours[method], label=labels[method])
        axes[1].bar(x + offset, frame["r2"], width, color=colours[method])
    for axis in axes:
        axis.set_xticks(x, sites)
        axis.set_xlabel("Completely held-out site")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("RMSE")
    axes[1].set_ylabel("R-squared")
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    figure.suptitle("Expanded nested leave-one-site-out GEDI FHD evaluation")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="lower center", ncol=2, frameon=False)
    figure.tight_layout(rect=(0, 0.12, 1, 1))
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = FIGURE_PATH.with_suffix(".tmp.png")
    figure.savefig(temporary, dpi=180, bbox_inches="tight")
    plt.close(figure)
    temporary.replace(FIGURE_PATH)


def main() -> int:
    protected = [PREDICTIONS_PATH, METRICS_PATH, TUNING_PATH, SELECTION_PATH, MODELS_PATH, FIGURE_PATH, FREEZE_PATH]
    existing = [str(path.relative_to(ROOT)) for path in protected if path.exists()]
    if existing:
        raise RuntimeError(f"Expanded LOSO freeze exists; refusing to overwrite: {existing}")
    if not PROSPECTIVE_EVALUATION_FREEZE_PATH.exists():
        raise RuntimeError("Prospective external evaluation must be frozen before expanded LOSO")
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))["phase7_site_expansion"]
    population = pd.concat([
        pd.read_parquet(ORIGINAL_DEVELOPMENT_PATH, columns=KEYS + ["fhd_normal"] + FEATURES),
        pd.read_parquet(ORIGINAL_BART_PATH, columns=KEYS + ["fhd_normal"] + FEATURES),
        pd.read_parquet(EXPANSION_PATH, columns=KEYS + ["fhd_normal"] + FEATURES),
    ], ignore_index=True)
    sites = sorted(population["site_id"].unique())
    if len(sites) != 8 or population.duplicated(KEYS).any():
        raise RuntimeError("Expanded LOSO population is not the expected eight-site cohort")
    alphas = [float(value) for value in protocol["prospective_external_evaluation"]["alpha_grid"]]
    prediction_frames = []
    tuning_frames = []
    ranking_frames = []
    artifacts: dict[str, Any] = {"outer_folds": {}, "features": FEATURES}

    for held_out in sites:
        training_sites = sorted(set(sites) - {held_out})
        training = population[population["site_id"].isin(training_sites)].copy()
        target_predictors = population.loc[
            population["site_id"].eq(held_out), KEYS + FEATURES
        ].copy()
        if "fhd_normal" in target_predictors:
            raise RuntimeError("Outer target outcome entered expanded LOSO fitting")
        selected, tuning, ranking = tune_inner_sites(training, training_sites, held_out, alphas)
        weights = loso.equal_site_weights(training["site_id"].to_numpy())
        scaler, model = loso.fit_weighted_ridge(
            training[FEATURES].to_numpy(dtype=np.float64),
            training["fhd_normal"].to_numpy(dtype=np.float64),
            selected,
            weights,
        )
        model_frame = target_predictors[KEYS].copy()
        model_frame["outer_held_out_site"] = held_out
        model_frame["training_sites"] = "+".join(training_sites)
        model_frame["method"] = "tessera_area_ridge"
        model_frame["selected_alpha"] = selected
        model_frame["prediction"] = model.predict(
            scaler.transform(target_predictors[FEATURES].to_numpy(dtype=np.float64))
        )
        baseline = target_predictors[KEYS].copy()
        baseline["outer_held_out_site"] = held_out
        baseline["training_sites"] = "+".join(training_sites)
        baseline["method"] = "training_site_mean"
        baseline["selected_alpha"] = np.nan
        baseline["prediction"] = float(training.groupby("site_id")["fhd_normal"].mean().mean())
        prediction_frames.extend([baseline, model_frame])
        tuning_frames.append(tuning)
        ranking_frames.append(ranking)
        artifacts["outer_folds"][held_out] = {
            "training_sites": training_sites,
            "selected_alpha": selected,
            "scaler": scaler,
            "model": model,
            "target_columns": target_predictors.columns.tolist(),
            "target_outcome_present": False,
            "training_weight_totals": {
                site: float(weights[training["site_id"].eq(site).to_numpy()].sum())
                for site in training_sites
            },
        }

    predictions = pd.concat(prediction_frames, ignore_index=True).merge(
        population[KEYS + ["fhd_normal"]], on=KEYS, validate="many_to_one"
    )
    predictions = predictions.sort_values(["outer_held_out_site", "method", "shot_number"]).reset_index(drop=True)
    tuning = pd.concat(tuning_frames, ignore_index=True)
    rankings = pd.concat(ranking_frames, ignore_index=True)
    selected_rows = rankings[rankings["alpha"].eq(rankings["selected_alpha"])].copy()
    metrics = build_metrics(predictions)
    site_metrics = metrics[metrics["scope"].eq("site")]
    tessera = site_metrics[site_metrics["method"].eq("tessera_area_ridge")].set_index("held_out_site")
    baseline = site_metrics[site_metrics["method"].eq("training_site_mean")].set_index("held_out_site")
    majority = math.floor(len(sites) / 2) + 1
    positive_sites = sorted(tessera.index[tessera["r2"].gt(0)].tolist())
    lower_rmse_sites = sorted(tessera.index[tessera["rmse"].lt(baseline["rmse"])].tolist())
    macro = metrics[metrics["scope"].eq("macro")].set_index("method")
    gate = {
        "site_count": len(sites),
        "majority_required": majority,
        "positive_r2_sites": positive_sites,
        "positive_r2_site_count": len(positive_sites),
        "rmse_below_training_mean_sites": lower_rmse_sites,
        "rmse_below_training_mean_site_count": len(lower_rmse_sites),
        "tessera_macro_rmse": float(macro.loc["tessera_area_ridge", "rmse"]),
        "training_mean_macro_rmse": float(macro.loc["training_site_mean", "rmse"]),
    }
    gate["passed"] = bool(
        len(positive_sites) >= majority
        and len(lower_rmse_sites) >= majority
        and gate["tessera_macro_rmse"] < gate["training_mean_macro_rmse"]
    )

    loso.write_parquet(predictions, PREDICTIONS_PATH)
    loso.write_csv(metrics, METRICS_PATH)
    loso.write_csv(tuning, TUNING_PATH)
    loso.write_csv(selected_rows, SELECTION_PATH)
    loso.write_joblib(artifacts, MODELS_PATH)
    save_figure(metrics)
    outputs = {
        "predictions": {"path": str(PREDICTIONS_PATH.relative_to(ROOT)), "rows": len(predictions), "sha256": sha256(PREDICTIONS_PATH)},
        "metrics": {"path": str(METRICS_PATH.relative_to(ROOT)), "rows": len(metrics), "sha256": sha256(METRICS_PATH)},
        "inner_tuning": {"path": str(TUNING_PATH.relative_to(ROOT)), "rows": len(tuning), "sha256": sha256(TUNING_PATH)},
        "selected_alphas": {"path": str(SELECTION_PATH.relative_to(ROOT)), "rows": len(selected_rows), "sha256": sha256(SELECTION_PATH)},
        "models": {"path": str(MODELS_PATH.relative_to(ROOT)), "sha256": sha256(MODELS_PATH)},
        "figure": {"path": str(FIGURE_PATH.relative_to(ROOT)), "sha256": sha256(FIGURE_PATH)},
    }
    freeze_basis = {
        "protocol_snapshot_sha256": sha256(PROTOCOL_PATH),
        "prospective_evaluation_freeze_id": json.loads(
            PROSPECTIVE_EVALUATION_FREEZE_PATH.read_text(encoding="utf-8")
        )["freeze_id"],
        "timing": protocol["expanded_nested_loso"]["timing"],
        "input_hashes": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [ORIGINAL_DEVELOPMENT_PATH, ORIGINAL_BART_PATH, EXPANSION_PATH]
        },
        "sites": sites,
        "site_rows": {site: int(count) for site, count in population.groupby("site_id").size().items()},
        "target_site_labels_used_for_scaling_tuning_or_training": False,
        "outer_split": "leave_one_complete_site_out",
        "inner_tuning": "leave_one_training_site_out",
        "equal_total_weight_per_training_site": True,
        "features": FEATURES,
        "alpha_grid": alphas,
        "selected_alphas": {
            row.outer_held_out_site: float(row.selected_alpha)
            for row in selected_rows.itertuples()
        },
        "gate": gate,
        "software": {"numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
    }
    freeze_id = "phase7-expanded-loso-" + canonical_hash(
        {"freeze_basis": freeze_basis, "outputs": outputs}
    )[:12]
    write_json(FREEZE_PATH, {
        "freeze_id": freeze_id, "created_utc": utc_now(), "freeze_basis": freeze_basis, "outputs": outputs
    })
    print(json.dumps({
        "freeze_id": freeze_id,
        "sites": sites,
        "rows": len(population),
        "selected_alphas": freeze_basis["selected_alphas"],
        "gate": gate,
        "site_metrics": {
            site: {metric: float(tessera.loc[site, metric]) for metric in loso.METRICS}
            for site in sites
        },
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
