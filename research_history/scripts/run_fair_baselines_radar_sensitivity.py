#!/usr/bin/env python3
"""Run the seven-site OPERA Sentinel-1 sensitivity under nested site-wise LOSO."""

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
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_multisite_loso as loso  # noqa: E402
import run_fair_baselines as fair  # noqa: E402


PROTOCOL_PATH = ROOT / "metadata/project_config_phase8_paper_analysis_protocol_freeze.yaml"
CONVENTIONAL_PATH = ROOT / "data/processed/phase8_conventional_predictors_all_sites.parquet"
CONVENTIONAL_FREEZE_PATH = ROOT / "metadata/phase8_conventional_predictor_all_sites_freeze.json"
TESSERA_PATHS = fair.TESSERA_PATHS
PREDICTIONS_PATH = ROOT / "data/processed/phase8_radar_sensitivity_predictions.parquet"
METRICS_PATH = ROOT / "outputs/tables/phase8_radar_sensitivity_metrics.csv"
TUNING_PATH = ROOT / "outputs/tables/phase8_radar_sensitivity_inner_tuning.csv"
SELECTION_PATH = ROOT / "outputs/tables/phase8_radar_sensitivity_selected_alphas.csv"
PAIRED_PATH = ROOT / "outputs/tables/phase8_radar_sensitivity_paired_differences.csv"
FIGURE_PATH = ROOT / "outputs/figures/phase8_radar_sensitivity.png"
FREEZE_PATH = ROOT / "metadata/phase8_radar_sensitivity_freeze.json"
KEYS = fair.KEYS
TESSERA = fair.TESSERA
SITES = ["BART", "HARV", "ORNL", "SOAP", "TALL", "TEAK", "WREF"]
METHODS = [
    "sentinel2_topography_ridge",
    "sentinel1_sentinel2_topography_ridge",
    "tessera_area_ridge",
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
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def bootstrap_paired(
    site_metrics: pd.DataFrame, replicates: int, seed: int
) -> pd.DataFrame:
    random = np.random.default_rng(seed)
    lookup = site_metrics.set_index(["held_out_site", "method"])
    records = []
    contrasts = [
        ("sentinel1_sentinel2_topography_ridge", "sentinel2_topography_ridge"),
        ("tessera_area_ridge", "sentinel1_sentinel2_topography_ridge"),
    ]
    for left, right in contrasts:
        for metric in ["rmse", "r2", "spearman_r"]:
            observed_by_site = np.asarray(
                [lookup.loc[(site, left), metric] - lookup.loc[(site, right), metric] for site in SITES],
                dtype=np.float64,
            )
            distribution = np.asarray(
                [
                    random.choice(observed_by_site, size=len(SITES), replace=True).mean()
                    for _ in range(replicates)
                ]
            )
            records.append(
                {
                    "contrast": f"{left}_minus_{right}",
                    "metric": metric,
                    "mean_paired_difference": float(observed_by_site.mean()),
                    "ci_low": float(np.percentile(distribution, 2.5)),
                    "ci_high": float(np.percentile(distribution, 97.5)),
                    "favourable_direction": "negative" if metric == "rmse" else "positive",
                    "bootstrap_unit": "site",
                    "bootstrap_replicates": replicates,
                }
            )
    return pd.DataFrame(records)


def save_figure(metrics: pd.DataFrame) -> None:
    site = metrics[metrics["scope"].eq("site")]
    x = np.arange(len(SITES))
    width = 0.25
    colours = ["#3a86ff", "#7b2cbf", "#0a9396"]
    labels = ["Sentinel-2 + terrain", "Sentinel-1/2 + terrain", "TESSERA"]
    figure, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for index, (method, colour, label) in enumerate(zip(METHODS, colours, labels, strict=True)):
        values = site[site["method"].eq(method)].set_index("held_out_site").loc[SITES]
        offset = (index - 1) * width
        axes[0].bar(x + offset, values["rmse"], width, color=colour, label=label)
        axes[1].bar(x + offset, values["r2"], width, color=colour)
    axes[0].set_ylabel("RMSE")
    axes[1].set_ylabel("R-squared")
    axes[1].set_xlabel("Completely held-out site")
    axes[1].set_xticks(x, SITES)
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[0].legend(frameon=False, ncol=3)
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Seven-site OPERA Sentinel-1 sensitivity")
    figure.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = FIGURE_PATH.with_suffix(".tmp.png")
    figure.savefig(temporary, dpi=180, bbox_inches="tight")
    plt.close(figure)
    temporary.replace(FIGURE_PATH)


def main() -> int:
    protected = [
        PREDICTIONS_PATH, METRICS_PATH, TUNING_PATH, SELECTION_PATH,
        PAIRED_PATH, FIGURE_PATH, FREEZE_PATH,
    ]
    if any(path.exists() for path in protected):
        raise RuntimeError("Phase 8 radar sensitivity freeze already exists")
    manifest = json.loads(CONVENTIONAL_FREEZE_PATH.read_text(encoding="utf-8"))
    optical = manifest["freeze_basis"]["primary_eight_site_features"]
    radar = manifest["freeze_basis"]["seven_site_radar_sensitivity_features"]
    feature_sets = {
        "sentinel2_topography_ridge": optical,
        "sentinel1_sentinel2_topography_ridge": radar,
        "tessera_area_ridge": TESSERA,
    }
    if any(column.endswith("observation_count") for values in feature_sets.values() for column in values):
        raise RuntimeError("Observation-count diagnostics entered the radar sensitivity")
    tessera = pd.concat(
        [pd.read_parquet(path, columns=KEYS + ["fhd_normal"] + TESSERA) for path in TESSERA_PATHS],
        ignore_index=True,
    )
    conventional_columns = sorted(set(optical + radar))
    conventional = pd.read_parquet(CONVENTIONAL_PATH, columns=KEYS + conventional_columns)
    population = tessera.merge(conventional, on=KEYS, validate="one_to_one")
    population = population[population["site_id"].isin(SITES)].copy()
    if set(population["site_id"]) != set(SITES) or len(population) != 9254:
        raise RuntimeError("Unexpected seven-site radar sensitivity population")
    if not np.isfinite(
        population[["fhd_normal"] + sorted(set(sum(feature_sets.values(), [])))].to_numpy()
    ).all():
        raise RuntimeError("Radar sensitivity contains non-finite model data")

    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))["phase8_paper_analysis"]
    alphas = [float(value) for value in protocol["ridge_alpha_grid"]]
    predictions = []
    tuning_frames = []
    ranking_frames = []
    for held_out in SITES:
        training_sites = sorted(set(SITES) - {held_out})
        training = population[population["site_id"].isin(training_sites)]
        for method, features in feature_sets.items():
            target = population.loc[population["site_id"].eq(held_out), KEYS + features]
            if "fhd_normal" in target.columns:
                raise RuntimeError("Outer target FHD entered radar sensitivity fitting")
            selected, tuning, ranking = fair.tune_model(
                training, training_sites, held_out, method, features, alphas
            )
            weights = loso.equal_site_weights(training["site_id"].to_numpy())
            scaler, model = loso.fit_weighted_ridge(
                training[features].to_numpy(dtype=np.float64),
                training["fhd_normal"].to_numpy(dtype=np.float64),
                selected,
                weights,
            )
            prediction = target[KEYS].copy()
            prediction["outer_held_out_site"] = held_out
            prediction["method"] = method
            prediction["selected_alpha"] = selected
            prediction["prediction"] = model.predict(
                scaler.transform(target[features].to_numpy(dtype=np.float64))
            )
            predictions.append(prediction)
            tuning_frames.append(tuning)
            ranking_frames.append(ranking)
    predictions_frame = pd.concat(predictions, ignore_index=True).merge(
        population[KEYS + ["fhd_normal"]], on=KEYS, validate="many_to_one"
    )
    tuning = pd.concat(tuning_frames, ignore_index=True)
    rankings = pd.concat(ranking_frames, ignore_index=True)
    selection = rankings[rankings["alpha"].eq(rankings["selected_alpha"])].copy()
    metrics = fair.build_metrics(predictions_frame)
    site_metrics = metrics[metrics["scope"].eq("site")]
    paired = bootstrap_paired(
        site_metrics,
        int(protocol["uncertainty"]["bootstrap_replicates"]),
        int(protocol["uncertainty"]["random_seed"]) + 1,
    )
    loso.write_parquet(predictions_frame, PREDICTIONS_PATH)
    loso.write_csv(metrics, METRICS_PATH)
    loso.write_csv(tuning, TUNING_PATH)
    loso.write_csv(selection, SELECTION_PATH)
    loso.write_csv(paired, PAIRED_PATH)
    save_figure(metrics)
    outputs = {
        name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
        for name, path in {
            "predictions": PREDICTIONS_PATH,
            "metrics": METRICS_PATH,
            "tuning": TUNING_PATH,
            "selection": SELECTION_PATH,
            "paired": PAIRED_PATH,
            "figure": FIGURE_PATH,
        }.items()
    }
    freeze_basis = {
        "script_sha256": sha256(Path(__file__).resolve()),
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "conventional_predictor_freeze_id": manifest["freeze_id"],
        "input_hashes": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in TESSERA_PATHS + [CONVENTIONAL_PATH, CONVENTIONAL_FREEZE_PATH]
        },
        "sites": SITES,
        "excluded_site": "UNDE",
        "exclusion_reason": "no Sentinel-1 acquisition covers UNDE in calendar year 2024",
        "feature_sets": feature_sets,
        "outer_split": "leave_one_complete_site_out",
        "inner_tuning": "leave_one_training_site_out",
        "equal_total_weight_per_training_site": True,
        "target_site_fhd_used_for_scaling_tuning_or_training": False,
        "observation_counts_used_as_model_features": False,
    }
    freeze_id = "phase8-radar-sensitivity-" + canonical_hash(
        {"freeze_basis": freeze_basis, "outputs": outputs}
    )[:12]
    write_json(
        FREEZE_PATH,
        {"freeze_id": freeze_id, "created_utc": utc_now(), "freeze_basis": freeze_basis, "outputs": outputs},
    )
    macro = metrics[metrics["scope"].eq("macro")].set_index("method")
    print(
        json.dumps(
            {
                "freeze_id": freeze_id,
                "macro_metrics": {
                    method: {metric: float(macro.loc[method, metric]) for metric in loso.METRICS}
                    for method in METHODS
                },
                "paired_differences": paired.to_dict("records"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
