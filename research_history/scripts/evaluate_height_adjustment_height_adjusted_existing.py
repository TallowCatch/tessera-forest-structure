#!/usr/bin/env python3
"""Evaluate height-adjusted FHD on the already-opened Phase 9 target cohort."""

from __future__ import annotations

import hashlib
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
import sklearn
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import freeze_height_adjustment_source_height_model as height_model  # noqa: E402
import run_multisite_loso as loso  # noqa: E402
import run_habitat_transfer_habitat_conditioned_transfer as habitat  # noqa: E402


PROTOCOL_PATH = ROOT / "metadata/project_config_phase10_unseen_replication_protocol_freeze.yaml"
HEIGHT_FREEZE_PATH = ROOT / "metadata/phase10_source_height_model_freeze.json"
SOURCE_PATH = ROOT / "data/processed/phase10_source_height_residuals.parquet"
TARGET_PATH = ROOT / "data/processed/phase9_prospective_tessera_aligned.parquet"
DISTANCE_PATH = ROOT / "metadata/phase9_landfire_habitat_distances.csv"
PREDICTIONS_PATH = ROOT / "data/processed/phase10_height_adjusted_existing_blind_predictions.parquet"
EVALUATED_PATH = ROOT / "data/processed/phase10_height_adjusted_existing_evaluated.parquet"
TUNING_PATH = ROOT / "outputs/tables/phase10_height_adjusted_existing_tuning.csv"
METRICS_PATH = ROOT / "outputs/tables/phase10_height_adjusted_existing_metrics.csv"
PAIRED_PATH = ROOT / "outputs/tables/phase10_height_adjusted_existing_paired.csv"
BOOTSTRAP_PATH = ROOT / "outputs/tables/phase10_height_adjusted_existing_bootstrap.csv"
FIGURE_PATH = ROOT / "outputs/figures/phase10_height_adjusted_existing.png"
FREEZE_PATH = ROOT / "metadata/phase10_height_adjusted_existing_freeze.json"
KEYS = ["site_id", "shot_number"]
FEATURES = [f"tessera_area_{index:03d}" for index in range(128)]
STRATEGIES = ["all_sources", "nearest5_evt_phys"]


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


def tune_residual_alpha(
    training: pd.DataFrame,
    source_sites: list[str],
    target_site: str,
    strategy: str,
    alphas: list[float],
) -> tuple[float, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    for alpha in alphas:
        for validation_site in source_sites:
            fit = training[~training["site_id"].eq(validation_site)]
            validation = training[training["site_id"].eq(validation_site)]
            weights = loso.equal_site_weights(fit["site_id"].to_numpy())
            scaler, model = loso.fit_weighted_ridge(
                fit[FEATURES].to_numpy(dtype=np.float64),
                fit["fhd_height_residual"].to_numpy(dtype=np.float64),
                alpha,
                weights,
            )
            prediction = model.predict(
                scaler.transform(validation[FEATURES].to_numpy(dtype=np.float64))
            )
            records.append({
                "target_site": target_site,
                "strategy": strategy,
                "alpha": alpha,
                "inner_validation_site": validation_site,
                "training_rows": len(fit),
                "validation_rows": len(validation),
                **loso.metric_values(
                    validation["fhd_height_residual"].to_numpy(dtype=np.float64),
                    prediction,
                ),
            })
    tuning = pd.DataFrame(records)
    selected, _ = loso.select_alpha(tuning)
    tuning["selected_alpha"] = selected
    return selected, tuning


def expected_fhd_from_frozen_height(height: np.ndarray, freeze: dict[str, Any]) -> np.ndarray:
    basis = freeze["freeze_basis"]
    transformed = height_model.height_features(height)
    mean = np.asarray(basis["scaler_mean"], dtype=np.float64)
    scale = np.asarray(basis["scaler_scale"], dtype=np.float64)
    coefficients = np.asarray(basis["model_coefficients"], dtype=np.float64)
    return float(basis["model_intercept"]) + ((transformed - mean) / scale) @ coefficients


def metrics_table(evaluated: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    groups = ["target_variable", "method", "strategy"]
    for keys, frame in evaluated.groupby(groups, sort=True, dropna=False):
        target_variable, method, strategy = keys
        for site, scoped in frame.groupby("site_id", sort=True):
            records.append({
                "scope": "site", "site_id": site, "target_variable": target_variable,
                "method": method, "strategy": strategy, "rows": len(scoped),
                **loso.metric_values(
                    scoped["observed"].to_numpy(dtype=np.float64),
                    scoped["prediction"].to_numpy(dtype=np.float64),
                ),
            })
    site = pd.DataFrame(records)
    for keys, frame in site.groupby(groups, sort=True, dropna=False):
        target_variable, method, strategy = keys
        records.append({
            "scope": "macro", "site_id": "macro_mean", "target_variable": target_variable,
            "method": method, "strategy": strategy, "rows": len(frame),
            **{metric: float(frame[metric].mean()) for metric in loso.METRICS},
        })
    return pd.DataFrame(records).sort_values(
        ["target_variable", "scope", "site_id", "method", "strategy"]
    ).reset_index(drop=True)


def paired_summary(metrics: pd.DataFrame, replicates: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    site = metrics[metrics["scope"].eq("site")]
    comparisons = [
        {
            "comparison": "height_plus_matched_tessera_minus_height_only",
            "target_variable": "fhd_normal",
            "left_method": "height_plus_residual_tessera",
            "left_strategy": "nearest5_evt_phys",
            "right_method": "height_only",
            "right_strategy": "none",
        },
        {
            "comparison": "matched_minus_all_source_residual_tessera",
            "target_variable": "fhd_height_residual",
            "left_method": "residual_tessera",
            "left_strategy": "nearest5_evt_phys",
            "right_method": "residual_tessera",
            "right_strategy": "all_sources",
        },
        {
            "comparison": "height_plus_all_source_tessera_minus_height_only",
            "target_variable": "fhd_normal",
            "left_method": "height_plus_residual_tessera",
            "left_strategy": "all_sources",
            "right_method": "height_only",
            "right_strategy": "none",
        },
    ]
    rng = np.random.default_rng(seed)
    summaries: list[dict[str, Any]] = []
    bootstrap: list[pd.DataFrame] = []
    for comparison in comparisons:
        scoped = site[site["target_variable"].eq(comparison["target_variable"])]
        left = scoped[
            scoped["method"].eq(comparison["left_method"])
            & scoped["strategy"].eq(comparison["left_strategy"])
        ].set_index("site_id").sort_index()
        right = scoped[
            scoped["method"].eq(comparison["right_method"])
            & scoped["strategy"].eq(comparison["right_strategy"])
        ].set_index("site_id").loc[left.index]
        delta = (left["rmse"] - right["rmse"]).to_numpy(dtype=np.float64)
        draws = np.mean(delta[rng.integers(0, len(delta), size=(replicates, len(delta)))], axis=1)
        summaries.append({
            **comparison,
            "site_count": len(delta),
            "sites_with_lower_rmse": int(np.sum(delta < 0)),
            "mean_delta_rmse": float(np.mean(delta)),
            "median_delta_rmse": float(np.median(delta)),
            "bootstrap_95_low": float(np.quantile(draws, 0.025)),
            "bootstrap_95_high": float(np.quantile(draws, 0.975)),
            "mean_delta_r2": float((left["r2"] - right["r2"]).mean()),
            "mean_delta_spearman_r": float((left["spearman_r"] - right["spearman_r"]).mean()),
        })
        bootstrap.append(pd.DataFrame({
            "comparison": comparison["comparison"],
            "replicate": np.arange(replicates, dtype=np.int64),
            "mean_delta_rmse": draws,
        }))
    return pd.DataFrame(summaries), pd.concat(bootstrap, ignore_index=True)


def save_figure(metrics: pd.DataFrame, paired: pd.DataFrame) -> None:
    site = metrics[metrics["scope"].eq("site")]
    sites = sorted(site["site_id"].unique())
    x = np.arange(len(sites))
    width = 0.25
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    full = site[site["target_variable"].eq("fhd_normal")]
    specifications = [
        ("height_only", "none", "Height only", "#6c757d"),
        ("height_plus_residual_tessera", "all_sources", "Height + all-source TESSERA", "#457b9d"),
        ("height_plus_residual_tessera", "nearest5_evt_phys", "Height + matched TESSERA", "#0a9396"),
    ]
    for index, (method, strategy, label, colour) in enumerate(specifications):
        frame = full[full["method"].eq(method) & full["strategy"].eq(strategy)].set_index("site_id").loc[sites]
        axes[0].bar(x + (index - 1) * width, frame["rmse"], width, label=label, color=colour)
    axes[0].set_xticks(x, sites, rotation=45, ha="right")
    axes[0].set_ylabel("Full FHD RMSE")
    axes[0].legend(frameon=False, fontsize=8)

    residual = site[site["target_variable"].eq("fhd_height_residual")]
    for index, (strategy, label, colour) in enumerate([
        ("all_sources", "All sources", "#457b9d"),
        ("nearest5_evt_phys", "Habitat matched", "#0a9396"),
    ]):
        frame = residual[residual["method"].eq("residual_tessera") & residual["strategy"].eq(strategy)].set_index("site_id").loc[sites]
        axes[1].bar(x + (index - 0.5) * 0.36, frame["rmse"], 0.36, label=label, color=colour)
    axes[1].set_xticks(x, sites, rotation=45, ha="right")
    axes[1].set_ylabel("Height-adjusted FHD RMSE")
    axes[1].legend(frameon=False, fontsize=8)

    summary = paired.set_index("comparison").loc[[
        "height_plus_matched_tessera_minus_height_only",
        "matched_minus_all_source_residual_tessera",
    ]]
    labels = ["Matched TESSERA\nvs height only", "Matched vs all-source\nresidual TESSERA"]
    axes[2].bar(labels, summary["mean_delta_rmse"], color=["#0a9396", "#457b9d"])
    axes[2].errorbar(
        labels, summary["mean_delta_rmse"],
        yerr=np.vstack([
            summary["mean_delta_rmse"] - summary["bootstrap_95_low"],
            summary["bootstrap_95_high"] - summary["mean_delta_rmse"],
        ]),
        fmt="none", color="black", capsize=3,
    )
    axes[2].axhline(0, color="black", linewidth=1)
    axes[2].set_ylabel("Mean paired RMSE change")
    axes[2].tick_params(axis="x", labelsize=8)
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Height-adjusted FHD analysis on the existing Phase 9 cohort")
    figure.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = FIGURE_PATH.with_suffix(".tmp.png")
    figure.savefig(temporary, dpi=200, bbox_inches="tight")
    plt.close(figure)
    temporary.replace(FIGURE_PATH)


def main() -> int:
    protected = [PREDICTIONS_PATH, EVALUATED_PATH, TUNING_PATH, METRICS_PATH, PAIRED_PATH, BOOTSTRAP_PATH, FIGURE_PATH, FREEZE_PATH]
    existing = [str(path.relative_to(ROOT)) for path in protected if path.exists()]
    if existing:
        raise RuntimeError(f"Phase 10 height-adjusted analysis exists; refusing to overwrite: {existing}")
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))["phase10_unseen_replication"]
    source_sites = sorted(protocol["source_sites"])
    height_freeze = json.loads(HEIGHT_FREEZE_PATH.read_text(encoding="utf-8"))
    source_columns = KEYS + ["fhd_height_residual", *FEATURES]
    sources = pd.read_parquet(SOURCE_PATH, columns=source_columns)
    if sorted(sources["site_id"].unique()) != source_sites:
        raise RuntimeError("Unexpected residual-FHD source population")

    # Construct all target predictions using predictor columns only.
    target_predictors = pd.read_parquet(TARGET_PATH, columns=KEYS + FEATURES)
    target_sites = sorted(target_predictors["site_id"].unique())
    distances = pd.read_csv(DISTANCE_PATH)
    alphas = [float(value) for value in protocol["locked_predictions"]["alpha_grid"]]
    prediction_frames: list[pd.DataFrame] = []
    tuning_frames: list[pd.DataFrame] = []
    for target_site in target_sites:
        target = target_predictors[target_predictors["site_id"].eq(target_site)]
        for strategy in STRATEGIES:
            selected_sources = habitat.source_sites_for_strategy(
                distances, target_site, source_sites, strategy, 5
            )
            training = sources[sources["site_id"].isin(selected_sources)]
            selected_alpha, tuning = tune_residual_alpha(
                training, selected_sources, target_site, strategy, alphas
            )
            tuning_frames.append(tuning)
            weights = loso.equal_site_weights(training["site_id"].to_numpy())
            scaler, model = loso.fit_weighted_ridge(
                training[FEATURES].to_numpy(dtype=np.float64),
                training["fhd_height_residual"].to_numpy(dtype=np.float64),
                selected_alpha,
                weights,
            )
            predicted = target[KEYS].copy()
            predicted["strategy"] = strategy
            predicted["training_sites"] = "+".join(selected_sources)
            predicted["selected_alpha"] = selected_alpha
            predicted["residual_prediction"] = model.predict(
                scaler.transform(target[FEATURES].to_numpy(dtype=np.float64))
            )
            prediction_frames.append(predicted)
    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["site_id", "strategy", "shot_number"]
    ).reset_index(drop=True)
    if "fhd_normal" in predictions or "elev_highestreturn" in predictions:
        raise RuntimeError("Target outcomes entered the predictor-only table")
    write_parquet(predictions, PREDICTIONS_PATH)
    blind_prediction_sha256 = sha256(PREDICTIONS_PATH)
    tuning = pd.concat(tuning_frames, ignore_index=True)
    write_csv(tuning, TUNING_PATH)

    # Outcomes were already opened in Phase 9; read them only after predictor-only output is written.
    outcomes = pd.read_parquet(
        TARGET_PATH, columns=KEYS + ["fhd_normal", "elev_highestreturn", "elev_lowestmode"]
    )
    outcomes["gedi_canopy_height_proxy_m"] = outcomes["elev_highestreturn"] - outcomes["elev_lowestmode"]
    outcomes["height_expected_fhd"] = expected_fhd_from_frozen_height(
        outcomes["gedi_canopy_height_proxy_m"].to_numpy(dtype=np.float64), height_freeze
    )
    outcomes["fhd_height_residual"] = outcomes["fhd_normal"] - outcomes["height_expected_fhd"]
    residual_predictions = predictions.merge(outcomes, on=KEYS, how="left", validate="many_to_one")

    frames: list[pd.DataFrame] = []
    baseline = outcomes[KEYS].copy()
    baseline["target_variable"] = "fhd_normal"
    baseline["method"] = "height_only"
    baseline["strategy"] = "none"
    baseline["observed"] = outcomes["fhd_normal"]
    baseline["prediction"] = outcomes["height_expected_fhd"]
    frames.append(baseline)
    residual_zero = outcomes[KEYS].copy()
    residual_zero["target_variable"] = "fhd_height_residual"
    residual_zero["method"] = "zero_residual"
    residual_zero["strategy"] = "none"
    residual_zero["observed"] = outcomes["fhd_height_residual"]
    residual_zero["prediction"] = 0.0
    frames.append(residual_zero)
    for strategy, frame in residual_predictions.groupby("strategy", sort=True):
        full = frame[KEYS].copy()
        full["target_variable"] = "fhd_normal"
        full["method"] = "height_plus_residual_tessera"
        full["strategy"] = strategy
        full["observed"] = frame["fhd_normal"]
        full["prediction"] = frame["height_expected_fhd"] + frame["residual_prediction"]
        frames.append(full)
        residual = frame[KEYS].copy()
        residual["target_variable"] = "fhd_height_residual"
        residual["method"] = "residual_tessera"
        residual["strategy"] = strategy
        residual["observed"] = frame["fhd_height_residual"]
        residual["prediction"] = frame["residual_prediction"]
        frames.append(residual)
    evaluated = pd.concat(frames, ignore_index=True).sort_values(
        ["target_variable", "method", "strategy", "site_id", "shot_number"]
    ).reset_index(drop=True)
    metrics = metrics_table(evaluated)
    evaluation = protocol["evaluation"]
    paired, bootstrap = paired_summary(
        metrics, int(evaluation["bootstrap_replicates"]), int(evaluation["random_seed"])
    )
    write_parquet(evaluated, EVALUATED_PATH)
    write_csv(metrics, METRICS_PATH)
    write_csv(paired, PAIRED_PATH)
    write_csv(bootstrap, BOOTSTRAP_PATH)
    save_figure(metrics, paired)

    freeze_basis = {
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "source_height_freeze_id": height_freeze["freeze_id"],
        "source_residual_sha256": sha256(SOURCE_PATH),
        "target_table_sha256": sha256(TARGET_PATH),
        "habitat_distances_sha256": sha256(DISTANCE_PATH),
        "script_sha256": sha256(Path(__file__).resolve()),
        "sklearn_version": sklearn.__version__,
        "target_sites": target_sites,
        "target_rows": len(outcomes),
        "target_columns_read_for_prediction": KEYS + FEATURES,
        "target_outcome_columns_read_after_predictor_output": ["fhd_normal", "elev_highestreturn", "elev_lowestmode"],
        "blind_prediction_sha256_before_outcome_read": blind_prediction_sha256,
        "retrospective_due_to_target_outcomes_previously_opened_in_phase9": True,
        "confirmatory_claim_allowed": False,
        "interpretation": "predeclared computational analysis but retrospective scientific evidence",
    }
    freeze_id = "phase10-height-adjusted-existing-" + canonical_hash(freeze_basis)[:12]
    outputs = {}
    for label, path in {
        "predictions": PREDICTIONS_PATH, "evaluated": EVALUATED_PATH, "tuning": TUNING_PATH,
        "metrics": METRICS_PATH, "paired": PAIRED_PATH, "bootstrap": BOOTSTRAP_PATH,
        "figure": FIGURE_PATH,
    }.items():
        outputs[label] = {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
    write_json(FREEZE_PATH, {
        "freeze_id": freeze_id,
        "created_utc": utc_now(),
        "status": "complete_retrospective_secondary_analysis",
        "freeze_basis": freeze_basis,
        "outputs": outputs,
    })
    print(json.dumps({
        "freeze_id": freeze_id,
        "target_sites": target_sites,
        "paired_results": paired[["comparison", "mean_delta_rmse", "bootstrap_95_low", "bootstrap_95_high", "sites_with_lower_rmse"]].to_dict(orient="records"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
