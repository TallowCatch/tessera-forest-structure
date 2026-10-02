#!/usr/bin/env python3
"""Quantify how strongly GEDI FHD depends on a GEDI canopy-height proxy."""

from __future__ import annotations

import hashlib
import json
import math
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
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_multisite_loso as loso  # noqa: E402


PROTOCOL_PATH = ROOT / "metadata/project_config_phase8_paper_analysis_protocol_freeze.yaml"
ORIGINAL_PATH = ROOT / "data/processed/gedi_v3_2024_primary_forest.parquet"
EXPANSION_PATH = ROOT / "data/processed/phase7_gedi_v3_primary_forest.parquet"
PREDICTIONS_PATH = ROOT / "data/processed/phase8_fhd_height_loso_predictions.parquet"
ASSOCIATIONS_PATH = ROOT / "outputs/tables/phase8_fhd_height_associations.csv"
QUINTILES_PATH = ROOT / "outputs/tables/phase8_fhd_height_quintiles.csv"
METRICS_PATH = ROOT / "outputs/tables/phase8_fhd_height_loso_metrics.csv"
TUNING_PATH = ROOT / "outputs/tables/phase8_fhd_height_loso_tuning.csv"
SELECTION_PATH = ROOT / "outputs/tables/phase8_fhd_height_loso_selected_alphas.csv"
FIGURE_PATH = ROOT / "outputs/figures/phase8_fhd_height_dependence.png"
FREEZE_PATH = ROOT / "metadata/phase8_fhd_height_dependence_freeze.json"
KEYS = ["site_id", "shot_number"]
HEIGHT = "gedi_canopy_height_proxy_m"
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


def association_record(scope: str, frame: pd.DataFrame) -> dict[str, Any]:
    height = frame[HEIGHT].to_numpy(dtype=np.float64)
    fhd = frame["fhd_normal"].to_numpy(dtype=np.float64)
    slope, intercept = np.polyfit(height, fhd, 1)
    return {
        "scope": scope,
        "rows": len(frame),
        "height_mean_m": float(np.mean(height)),
        "height_std_m": float(np.std(height, ddof=1)),
        "fhd_mean": float(np.mean(fhd)),
        "fhd_std": float(np.std(fhd, ddof=1)),
        "pearson_r": float(np.corrcoef(height, fhd)[0, 1]),
        "spearman_r": float(spearmanr(height, fhd).statistic),
        "ols_slope_fhd_per_m": float(slope),
        "ols_intercept": float(intercept),
    }


def tune_height(
    training: pd.DataFrame,
    training_sites: list[str],
    held_out_site: str,
    alphas: list[float],
) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    for alpha in alphas:
        for validation_site in training_sites:
            fit = training[~training["site_id"].eq(validation_site)]
            validation = training[training["site_id"].eq(validation_site)]
            weights = loso.equal_site_weights(fit["site_id"].to_numpy())
            scaler, model = loso.fit_weighted_ridge(
                fit[[HEIGHT]].to_numpy(dtype=np.float64),
                fit["fhd_normal"].to_numpy(dtype=np.float64),
                alpha,
                weights,
            )
            prediction = model.predict(
                scaler.transform(validation[[HEIGHT]].to_numpy(dtype=np.float64))
            )
            records.append(
                {
                    "outer_held_out_site": held_out_site,
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
    ranking["selected_alpha"] = selected
    return selected, tuning, ranking


def build_loso(
    population: pd.DataFrame, alphas: list[float]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sites = sorted(population["site_id"].unique())
    predictions: list[pd.DataFrame] = []
    tuning_frames: list[pd.DataFrame] = []
    ranking_frames: list[pd.DataFrame] = []
    for held_out in sites:
        training_sites = sorted(set(sites) - {held_out})
        training = population[population["site_id"].isin(training_sites)]
        target = population.loc[population["site_id"].eq(held_out), KEYS + [HEIGHT]].copy()
        if "fhd_normal" in target.columns:
            raise RuntimeError("Outer target FHD entered height-model fitting")
        selected, tuning, ranking = tune_height(
            training, training_sites, held_out, alphas
        )
        weights = loso.equal_site_weights(training["site_id"].to_numpy())
        scaler, model = loso.fit_weighted_ridge(
            training[[HEIGHT]].to_numpy(dtype=np.float64),
            training["fhd_normal"].to_numpy(dtype=np.float64),
            selected,
            weights,
        )
        fitted = target[KEYS].copy()
        fitted["outer_held_out_site"] = held_out
        fitted["method"] = "gedi_height_proxy_ridge"
        fitted["selected_alpha"] = selected
        fitted["prediction"] = model.predict(
            scaler.transform(target[[HEIGHT]].to_numpy(dtype=np.float64))
        )
        baseline = target[KEYS].copy()
        baseline["outer_held_out_site"] = held_out
        baseline["method"] = "training_site_mean"
        baseline["selected_alpha"] = np.nan
        baseline["prediction"] = float(
            training.groupby("site_id")["fhd_normal"].mean().mean()
        )
        predictions.extend([baseline, fitted])
        tuning_frames.append(tuning)
        ranking_frames.append(ranking)
    combined = pd.concat(predictions, ignore_index=True).merge(
        population[KEYS + ["fhd_normal", HEIGHT]], on=KEYS, validate="many_to_one"
    )
    rankings = pd.concat(ranking_frames, ignore_index=True)
    selected = rankings[rankings["alpha"].eq(rankings["selected_alpha"])].copy()
    return combined, pd.concat(tuning_frames, ignore_index=True), selected


def build_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (site, method), frame in predictions.groupby(
        ["outer_held_out_site", "method"], sort=True
    ):
        records.append(
            {
                "scope": "site",
                "held_out_site": site,
                "method": method,
                "rows": len(frame),
                **loso.metric_values(
                    frame["fhd_normal"].to_numpy(), frame["prediction"].to_numpy()
                ),
            }
        )
    site_metrics = pd.DataFrame(records)
    for method, frame in site_metrics.groupby("method", sort=True):
        records.append(
            {
                "scope": "macro",
                "held_out_site": "macro_mean",
                "method": method,
                "rows": len(frame),
                **{metric: float(frame[metric].mean()) for metric in loso.METRICS},
            }
        )
    return pd.DataFrame(records)


def save_figure(
    population: pd.DataFrame,
    associations: pd.DataFrame,
    quintiles: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    colours = plt.get_cmap("tab10")
    sites = sorted(population["site_id"].unique())
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    for index, site in enumerate(sites):
        frame = population[population["site_id"].eq(site)]
        axes[0, 0].scatter(
            frame[HEIGHT], frame["fhd_normal"], s=6, alpha=0.18,
            color=colours(index), label=site,
        )
    axes[0, 0].set_xlabel("GEDI canopy-height proxy (m)")
    axes[0, 0].set_ylabel("GEDI V3 fhd_normal")
    axes[0, 0].legend(ncol=4, fontsize=8, frameon=False)
    site_associations = associations[associations["scope"].isin(sites)].set_index("scope").loc[sites]
    axes[0, 1].bar(
        np.arange(len(sites)), site_associations["spearman_r"],
        color=[colours(index) for index in range(len(sites))],
    )
    axes[0, 1].set_xticks(np.arange(len(sites)), sites)
    axes[0, 1].set_ylabel("Within-site Spearman r")
    axes[0, 1].axhline(0, color="black", linewidth=1)
    for site, frame in quintiles.groupby("site_id", sort=True):
        axes[1, 0].plot(
            frame["height_quintile"], frame["fhd_mean"], marker="o", label=site
        )
    axes[1, 0].set_xlabel("Within-site canopy-height quintile")
    axes[1, 0].set_ylabel("Mean GEDI V3 fhd_normal")
    site_metrics = metrics[metrics["scope"].eq("site")]
    x = np.arange(len(sites))
    width = 0.36
    for offset, (method, label, colour) in enumerate(
        [
            ("training_site_mean", "training-site mean", "#6c757d"),
            ("gedi_height_proxy_ridge", "GEDI height diagnostic", "#bc6c25"),
        ]
    ):
        frame = site_metrics[site_metrics["method"].eq(method)].set_index(
            "held_out_site"
        ).loc[sites]
        axes[1, 1].bar(x + (offset - 0.5) * width, frame["r2"], width, label=label, color=colour)
    axes[1, 1].set_xticks(x, sites)
    axes[1, 1].set_ylabel("Leave-one-site-out R-squared")
    axes[1, 1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1, 1].legend(frameon=False)
    for axis in axes.flat:
        axis.grid(alpha=0.18)
    figure.suptitle("GEDI FHD dependence on canopy height")
    figure.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = FIGURE_PATH.with_suffix(".tmp.png")
    figure.savefig(temporary, dpi=180, bbox_inches="tight")
    plt.close(figure)
    temporary.replace(FIGURE_PATH)


def main() -> int:
    protected = [
        PREDICTIONS_PATH, ASSOCIATIONS_PATH, QUINTILES_PATH, METRICS_PATH,
        TUNING_PATH, SELECTION_PATH, FIGURE_PATH, FREEZE_PATH,
    ]
    existing = [str(path.relative_to(ROOT)) for path in protected if path.exists()]
    if existing:
        raise RuntimeError(f"FHD-height freeze exists; refusing to overwrite: {existing}")
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))[
        "phase8_paper_analysis"
    ]
    columns = KEYS + [
        "fhd_normal", "elev_highestreturn", "elev_lowestmode"
    ]
    population = pd.concat(
        [
            pd.read_parquet(ORIGINAL_PATH, columns=columns),
            pd.read_parquet(EXPANSION_PATH, columns=columns),
        ],
        ignore_index=True,
    )
    population[HEIGHT] = (
        population["elev_highestreturn"] - population["elev_lowestmode"]
    )
    if set(population["site_id"]) != EXPECTED_SITES or len(population) != 10184:
        raise RuntimeError("Unexpected eight-site FHD-height population")
    if population.duplicated(KEYS).any():
        raise RuntimeError("Duplicate GEDI keys in FHD-height population")
    if not np.isfinite(population[["fhd_normal", HEIGHT]].to_numpy()).all():
        raise RuntimeError("Non-finite FHD or canopy-height proxy")
    if (population[HEIGHT] <= 0).any():
        raise RuntimeError("Canopy-height proxy is not strictly positive")

    associations = pd.DataFrame(
        [
            association_record(site, frame)
            for site, frame in population.groupby("site_id", sort=True)
        ]
    )
    associations = pd.concat(
        [associations, pd.DataFrame([association_record("pooled_raw", population)])],
        ignore_index=True,
    )
    centered = population.copy()
    centered[HEIGHT] -= centered.groupby("site_id")[HEIGHT].transform("mean")
    centered["fhd_normal"] -= centered.groupby("site_id")["fhd_normal"].transform("mean")
    associations = pd.concat(
        [associations, pd.DataFrame([association_record("pooled_within_site_centered", centered)])],
        ignore_index=True,
    )

    quintile_frames = []
    for site, frame in population.groupby("site_id", sort=True):
        scoped = frame.copy()
        scoped["height_quintile"] = pd.qcut(
            scoped[HEIGHT], 5, labels=False, duplicates="raise"
        ) + 1
        quintile_frames.append(
            scoped.groupby("height_quintile", as_index=False).agg(
                rows=("shot_number", "size"),
                height_mean_m=(HEIGHT, "mean"),
                height_min_m=(HEIGHT, "min"),
                height_max_m=(HEIGHT, "max"),
                fhd_mean=("fhd_normal", "mean"),
                fhd_std=("fhd_normal", "std"),
            ).assign(site_id=site)
        )
    quintiles = pd.concat(quintile_frames, ignore_index=True)[
        ["site_id", "height_quintile", "rows", "height_mean_m", "height_min_m", "height_max_m", "fhd_mean", "fhd_std"]
    ]

    alphas = [float(value) for value in protocol["ridge_alpha_grid"]]
    predictions, tuning, selection = build_loso(population, alphas)
    metrics = build_metrics(predictions)
    loso.write_parquet(predictions, PREDICTIONS_PATH)
    loso.write_csv(associations, ASSOCIATIONS_PATH)
    loso.write_csv(quintiles, QUINTILES_PATH)
    loso.write_csv(metrics, METRICS_PATH)
    loso.write_csv(tuning, TUNING_PATH)
    loso.write_csv(selection, SELECTION_PATH)
    save_figure(population, associations, quintiles, metrics)

    outputs = {
        "predictions": {"path": str(PREDICTIONS_PATH.relative_to(ROOT)), "sha256": sha256(PREDICTIONS_PATH)},
        "associations": {"path": str(ASSOCIATIONS_PATH.relative_to(ROOT)), "sha256": sha256(ASSOCIATIONS_PATH)},
        "quintiles": {"path": str(QUINTILES_PATH.relative_to(ROOT)), "sha256": sha256(QUINTILES_PATH)},
        "metrics": {"path": str(METRICS_PATH.relative_to(ROOT)), "sha256": sha256(METRICS_PATH)},
        "tuning": {"path": str(TUNING_PATH.relative_to(ROOT)), "sha256": sha256(TUNING_PATH)},
        "selection": {"path": str(SELECTION_PATH.relative_to(ROOT)), "sha256": sha256(SELECTION_PATH)},
        "figure": {"path": str(FIGURE_PATH.relative_to(ROOT)), "sha256": sha256(FIGURE_PATH)},
    }
    freeze_basis = {
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "script_sha256": sha256(Path(__file__).resolve()),
        "input_hashes": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [ORIGINAL_PATH, EXPANSION_PATH]
        },
        "rows": len(population),
        "sites": sorted(EXPECTED_SITES),
        "height_formula": "elev_highestreturn - elev_lowestmode",
        "height_role": "explanatory_GEDI_diagnostic_not_wall_to_wall_predictor",
        "outer_split": protocol["outer_validation"],
        "inner_tuning": protocol["inner_tuning"],
        "target_site_fhd_used_for_scaling_tuning_or_training": False,
        "alphas": alphas,
        "software": {
            "numpy": np.__version__, "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    freeze_id = "phase8-fhd-height-" + canonical_hash(
        {"freeze_basis": freeze_basis, "outputs": outputs}
    )[:12]
    write_json(
        FREEZE_PATH,
        {
            "freeze_id": freeze_id,
            "created_utc": utc_now(),
            "freeze_basis": freeze_basis,
            "outputs": outputs,
        },
    )
    site_metrics = metrics[
        metrics["scope"].eq("site")
        & metrics["method"].eq("gedi_height_proxy_ridge")
    ]
    print(
        json.dumps(
            {
                "freeze_id": freeze_id,
                "within_site_centered_pearson_r": float(
                    associations.set_index("scope").loc[
                        "pooled_within_site_centered", "pearson_r"
                    ]
                ),
                "within_site_centered_spearman_r": float(
                    associations.set_index("scope").loc[
                        "pooled_within_site_centered", "spearman_r"
                    ]
                ),
                "positive_height_loso_r2_sites": sorted(
                    site_metrics.loc[site_metrics["r2"].gt(0), "held_out_site"].tolist()
                ),
                "site_associations": associations[
                    associations["scope"].isin(EXPECTED_SITES)
                ].set_index("scope")["spearman_r"].to_dict(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
