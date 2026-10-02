#!/usr/bin/env python3
"""Evaluate footprint TESSERA transfer to GEDI RH100 across 14 complete sites."""

from __future__ import annotations

import hashlib
import json
import math
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
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/project.yaml"
PROTOCOL_PATH = ROOT / "metadata/project_config_phase11_height_context_protocol_freeze.yaml"
INPUT_PATHS = [
    ROOT / "data/processed/tessera_aligned_development_soap_teak.parquet",
    ROOT / "data/processed/tessera_aligned_locked_bart.parquet",
    ROOT / "data/processed/phase7_tessera_aligned_expansion.parquet",
    ROOT / "data/processed/phase9_prospective_tessera_aligned.parquet",
]
PREDICTIONS_PATH = ROOT / "data/processed/phase11_footprint_height_predictions.parquet"
METRICS_PATH = ROOT / "outputs/tables/phase11_footprint_height_metrics.csv"
TUNING_PATH = ROOT / "outputs/tables/phase11_footprint_height_tuning.csv"
SELECTION_PATH = ROOT / "outputs/tables/phase11_footprint_height_selection.csv"
FIGURE_PATH = ROOT / "outputs/figures/phase11_footprint_height_transfer.png"
FREEZE_PATH = ROOT / "metadata/phase11_footprint_height_freeze.json"

KEYS = ["site_id", "shot_number"]
FEATURES = [f"tessera_area_{index:03d}" for index in range(128)]
READ_COLUMNS = KEYS + [
    "longitude",
    "latitude",
    "elev_highestreturn",
    "elev_lowestmode",
    *FEATURES,
]


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


def height_target(frame: pd.DataFrame) -> np.ndarray:
    return (
        frame["elev_highestreturn"].to_numpy(dtype=np.float64)
        - frame["elev_lowestmode"].to_numpy(dtype=np.float64)
    )


def equal_site_weights(site_ids: np.ndarray) -> np.ndarray:
    site_ids = np.asarray(site_ids)
    sites, counts = np.unique(site_ids, return_counts=True)
    if len(sites) == 0:
        raise ValueError("Cannot weight an empty population")
    lookup = dict(zip(sites, counts, strict=True))
    return np.asarray(
        [len(site_ids) / (len(sites) * lookup[site]) for site in site_ids],
        dtype=np.float64,
    )


def metric_values(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    residual = predicted - observed
    pearson = math.nan
    spearman = math.nan
    slope = math.nan
    if np.ptp(observed) > 1e-12 and np.ptp(predicted) > 1e-12:
        pearson = float(np.corrcoef(observed, predicted)[0, 1])
        spearman = float(spearmanr(observed, predicted).statistic)
        slope = float(np.polyfit(observed, predicted, 1)[0])
    observed_sd = float(np.std(observed, ddof=0))
    return {
        "r2": float(r2_score(observed, predicted)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(mean_absolute_error(observed, predicted)),
        "pearson_r": pearson,
        "spearman_r": spearman,
        "mean_bias": float(np.mean(residual)),
        "prediction_slope": slope,
        "prediction_sd_ratio": float(np.std(predicted, ddof=0) / observed_sd)
        if observed_sd > 0
        else math.nan,
    }


def fit_ridge(
    frame: pd.DataFrame,
    features: list[str],
    target: str,
    alpha: float,
) -> tuple[StandardScaler, Ridge]:
    x = frame[features].to_numpy(dtype=np.float64)
    y = frame[target].to_numpy(dtype=np.float64)
    weights = equal_site_weights(frame["site_id"].to_numpy())
    scaler = StandardScaler().fit(x, sample_weight=weights)
    model = Ridge(alpha=float(alpha), solver="lsqr", tol=1e-6).fit(
        scaler.transform(x), y, sample_weight=weights
    )
    return scaler, model


def equal_site_mean(frame: pd.DataFrame, target: str) -> float:
    return float(frame.groupby("site_id", sort=True)[target].mean().mean())


def select_alpha(tuning: pd.DataFrame) -> tuple[float, pd.DataFrame]:
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


def tune_alpha(
    training: pd.DataFrame,
    features: list[str],
    target: str,
    alphas: list[float],
    outer_site: str,
) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    sites = sorted(training["site_id"].unique())
    for alpha in alphas:
        for validation_site in sites:
            fit = training[~training["site_id"].eq(validation_site)]
            validation = training[training["site_id"].eq(validation_site)]
            scaler, model = fit_ridge(fit, features, target, alpha)
            prediction = model.predict(
                scaler.transform(validation[features].to_numpy(dtype=np.float64))
            )
            records.append(
                {
                    "outer_held_out_site": outer_site,
                    "inner_held_out_site": validation_site,
                    "alpha": float(alpha),
                    "training_rows": len(fit),
                    "validation_rows": len(validation),
                    **metric_values(validation[target].to_numpy(), prediction),
                }
            )
    tuning = pd.DataFrame(records)
    selected, ranking = select_alpha(tuning)
    tuning["selected_alpha"] = selected
    ranking.insert(0, "outer_held_out_site", outer_site)
    ranking["selected_alpha"] = selected
    return selected, tuning, ranking


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
                **metric_values(frame["observed_rh100_m"], frame["prediction"]),
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
                    column: float(sites[column].mean())
                    for column in [
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


def make_figure(predictions: pd.DataFrame, metrics: pd.DataFrame) -> None:
    ridge = predictions[predictions["model"].eq("tessera_area_ridge")]
    site_metrics = metrics[
        metrics["model"].eq("tessera_area_ridge") & metrics["scope"].eq("site")
    ].sort_values("site_id")
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), constrained_layout=True)
    axes[0].scatter(
        ridge["observed_rh100_m"], ridge["prediction"], s=6, alpha=0.22, linewidths=0
    )
    low = float(min(ridge["observed_rh100_m"].min(), ridge["prediction"].min()))
    high = float(max(ridge["observed_rh100_m"].max(), ridge["prediction"].max()))
    axes[0].plot([low, high], [low, high], color="black", linestyle="--", linewidth=1)
    axes[0].set(xlabel="GEDI RH100 (m)", ylabel="Held-out-site prediction (m)", title="Complete-site predictions")
    colors = np.where(site_metrics["r2"].to_numpy() > 0, "#148F77", "#C0392B")
    axes[1].bar(site_metrics["site_id"], site_metrics["r2"], color=colors)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].tick_params(axis="x", rotation=60)
    axes[1].set(ylabel="R2", title="Performance at each unseen forest")
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=220)
    plt.close(figure)


def main() -> int:
    protected = [PREDICTIONS_PATH, METRICS_PATH, TUNING_PATH, SELECTION_PATH, FIGURE_PATH, FREEZE_PATH]
    existing = [str(path.relative_to(ROOT)) for path in protected if path.exists()]
    if existing:
        raise RuntimeError(f"Phase 11 footprint-height outputs already exist: {existing}")
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))[
        "phase11_height_context_transfer"
    ]
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "phase11_height_context_transfer"
    ]
    cohort = pd.concat(
        [pd.read_parquet(path, columns=READ_COLUMNS) for path in INPUT_PATHS],
        ignore_index=True,
    )
    cohort["rh100_m"] = height_target(cohort)
    expected_sites = sorted(protocol["cohort"]["sites"])
    if len(cohort) != int(protocol["cohort"]["expected_rows"]):
        raise RuntimeError("Phase 11 cohort row count changed")
    if sorted(cohort["site_id"].unique()) != expected_sites:
        raise RuntimeError("Phase 11 cohort sites changed")
    if cohort[KEYS].duplicated().any() or not np.isfinite(cohort[FEATURES + ["rh100_m"]]).all().all():
        raise RuntimeError("Phase 11 cohort contains duplicate keys or non-finite values")
    if (cohort["rh100_m"] <= 0).any():
        raise RuntimeError("Phase 11 cohort contains non-positive RH100")

    alphas = [float(value) for value in config["footprint_height_gate"]["alpha_grid"]]
    prediction_frames = []
    tuning_frames = []
    selection_frames = []
    for index, held_out in enumerate(expected_sites, start=1):
        training = cohort[~cohort["site_id"].eq(held_out)].copy()
        target = cohort[cohort["site_id"].eq(held_out)].copy()
        selected, tuning, ranking = tune_alpha(
            training, FEATURES, "rh100_m", alphas, held_out
        )
        scaler, model = fit_ridge(training, FEATURES, "rh100_m", selected)
        ridge_prediction = model.predict(
            scaler.transform(target[FEATURES].to_numpy(dtype=np.float64))
        )
        baseline_prediction = np.full(len(target), equal_site_mean(training, "rh100_m"))
        common = target[KEYS + ["longitude", "latitude", "rh100_m"]].rename(
            columns={"rh100_m": "observed_rh100_m"}
        )
        for model_name, prediction in [
            ("equal_site_training_mean", baseline_prediction),
            ("tessera_area_ridge", ridge_prediction),
        ]:
            frame = common.copy()
            frame["model"] = model_name
            frame["prediction"] = prediction
            frame["outer_held_out_site"] = held_out
            frame["selected_alpha"] = selected if model_name == "tessera_area_ridge" else math.nan
            prediction_frames.append(frame)
        tuning_frames.append(tuning)
        selection_frames.append(ranking)
        print(f"[{index:02d}/{len(expected_sites)}] {held_out}: alpha={selected:g}", flush=True)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    tuning = pd.concat(tuning_frames, ignore_index=True)
    selection = pd.concat(selection_frames, ignore_index=True)
    metrics = build_metrics(predictions)
    macro = metrics[metrics["scope"].eq("macro")].set_index("model")
    gate = {
        "macro_r2_above_zero": bool(macro.loc["tessera_area_ridge", "r2"] > 0),
        "ridge_macro_rmse_below_training_mean_macro_rmse": bool(
            macro.loc["tessera_area_ridge", "rmse"]
            < macro.loc["equal_site_training_mean", "rmse"]
        ),
    }
    gate["passed"] = all(gate.values())

    write_parquet(predictions, PREDICTIONS_PATH)
    write_csv(metrics, METRICS_PATH)
    write_csv(tuning, TUNING_PATH)
    write_csv(selection, SELECTION_PATH)
    make_figure(predictions, metrics)
    freeze_basis = {
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "config_sha256": sha256(CONFIG_PATH),
        "script_sha256": sha256(Path(__file__).resolve()),
        "input_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in INPUT_PATHS},
        "rows": len(cohort),
        "sites": expected_sites,
        "site_rows": {site: int(count) for site, count in cohort.groupby("site_id").size().items()},
        "target": "elev_highestreturn - elev_lowestmode",
        "features": "tessera_area_000_to_127",
        "outer_split": "leave_one_complete_site_out",
        "inner_selection": "leave_one_training_site_out",
        "equal_total_weight_per_training_site": True,
        "alphas": alphas,
        "target_site_height_used_for_training_or_selection": False,
        "software": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "gate": gate,
        "macro_metrics": {
            model_name: {
                key: float(macro.loc[model_name, key])
                for key in ["r2", "rmse", "mae", "spearman_r", "mean_bias", "prediction_sd_ratio"]
            }
            for model_name in macro.index
        },
    }
    freeze = {
        "freeze_id": "phase11-footprint-height-" + canonical_hash(freeze_basis)[:12],
        "created_utc": utc_now(),
        "status": "complete_gate_passed" if gate["passed"] else "complete_gate_failed_stop",
        "freeze_basis": freeze_basis,
        "outputs": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [PREDICTIONS_PATH, METRICS_PATH, TUNING_PATH, SELECTION_PATH, FIGURE_PATH]
        },
    }
    write_json(FREEZE_PATH, freeze)
    print("\n" + metrics[metrics["scope"].eq("macro")].to_string(index=False))
    print("\nGate:", json.dumps(gate, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
