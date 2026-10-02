#!/usr/bin/env python3
"""Run exploratory habitat-conditioned complete-site TESSERA transfer."""

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
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_multisite_loso as loso  # noqa: E402


PROTOCOL_PATH = ROOT / "metadata/project_config_phase9_habitat_transfer_protocol_freeze.yaml"
HABITAT_FREEZE_PATH = ROOT / "metadata/phase9_landfire_habitat_freeze.json"
DISTANCE_PATH = ROOT / "metadata/phase9_landfire_habitat_distances.csv"
POINT_HABITAT_PATH = ROOT / "data/processed/phase9_landfire_evt_footprints.parquet"
CONVENTIONAL_PATH = ROOT / "data/processed/phase8_conventional_predictors_all_sites.parquet"
PHASE7_PREDICTIONS_PATH = ROOT / "data/processed/phase7_expanded_loso_predictions.parquet"
POPULATION_PATHS = [
    ROOT / "data/processed/tessera_aligned_development_soap_teak.parquet",
    ROOT / "data/processed/tessera_aligned_locked_bart.parquet",
    ROOT / "data/processed/phase7_tessera_aligned_expansion.parquet",
]

PREDICTIONS_PATH = ROOT / "data/processed/phase9_habitat_conditioned_predictions.parquet"
METRICS_PATH = ROOT / "outputs/tables/phase9_habitat_conditioned_metrics.csv"
TUNING_PATH = ROOT / "outputs/tables/phase9_habitat_conditioned_inner_tuning.csv"
SELECTION_PATH = ROOT / "outputs/tables/phase9_habitat_conditioned_source_selection.csv"
PAIRED_PATH = ROOT / "outputs/tables/phase9_habitat_conditioned_paired_summary.csv"
SLOPE_PATH = ROOT / "outputs/tables/phase9_habitat_conditioned_slope_metrics.csv"
FIGURE_PATH = ROOT / "outputs/figures/phase9_habitat_conditioned_transfer.png"
FREEZE_PATH = ROOT / "metadata/phase9_habitat_conditioned_transfer_freeze.json"

KEYS = ["site_id", "shot_number"]
FEATURES = [f"tessera_area_{index:03d}" for index in range(128)]
HIERARCHIES = ["evt_phys", "evt_group", "evt_system"]
PRIMARY_ANALYSIS = "all_quality_filtered"
SLOPE_ANALYSIS = "slope_le_30"


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


def source_sites_for_strategy(
    distances: pd.DataFrame,
    target_site: str,
    available_sources: list[str],
    strategy: str,
    nearest_count: int,
) -> list[str]:
    if target_site in available_sources:
        raise ValueError("Target site entered the available source pool")
    if strategy == "all_sources":
        return sorted(available_sources)
    hierarchy = strategy.removeprefix("nearest5_")
    if hierarchy not in HIERARCHIES:
        raise ValueError(f"Unknown habitat strategy: {strategy}")
    scoped = distances[
        distances["hierarchy"].eq(hierarchy)
        & distances["target_site"].eq(target_site)
        & distances["source_site"].isin(available_sources)
    ].sort_values(["distance", "source_site"])
    if len(scoped) < nearest_count:
        raise RuntimeError(f"Insufficient habitat sources for {target_site}/{strategy}")
    return scoped.head(nearest_count)["source_site"].tolist()


def tune_alpha(
    training: pd.DataFrame,
    source_sites: list[str],
    target_site: str,
    strategy: str,
    analysis: str,
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
                fit["fhd_normal"].to_numpy(dtype=np.float64),
                alpha,
                weights,
            )
            predicted = model.predict(
                scaler.transform(validation[FEATURES].to_numpy(dtype=np.float64))
            )
            records.append({
                "analysis": analysis,
                "outer_held_out_site": target_site,
                "strategy": strategy,
                "alpha": alpha,
                "inner_validation_site": validation_site,
                "inner_training_sites": "+".join(sorted(set(source_sites) - {validation_site})),
                "training_rows": len(fit),
                "validation_rows": len(validation),
                **loso.metric_values(
                    validation["fhd_normal"].to_numpy(dtype=np.float64), predicted
                ),
            })
    tuning = pd.DataFrame(records)
    selected, _ = loso.select_alpha(tuning)
    tuning["selected_alpha"] = selected
    return selected, tuning


def fit_strategy(
    population: pd.DataFrame,
    distances: pd.DataFrame,
    target_site: str,
    strategy: str,
    analysis: str,
    alphas: list[float],
    nearest_count: int,
) -> tuple[list[pd.DataFrame], pd.DataFrame, list[dict[str, Any]]]:
    available = sorted(set(population["site_id"]) - {target_site})
    sources = source_sites_for_strategy(distances, target_site, available, strategy, nearest_count)
    training = population[population["site_id"].isin(sources)].copy()
    target_predictors = population.loc[
        population["site_id"].eq(target_site), KEYS + FEATURES + ["terrain_slope_degrees"]
    ].copy()
    if "fhd_normal" in target_predictors.columns:
        raise RuntimeError("Held-out FHD entered the target predictor table")
    if set(training["site_id"].unique()) != set(sources):
        raise RuntimeError("The selected habitat source population is incomplete")

    selected_alpha, tuning = tune_alpha(
        training, sources, target_site, strategy, analysis, alphas
    )
    weights = loso.equal_site_weights(training["site_id"].to_numpy())
    scaler, model = loso.fit_weighted_ridge(
        training[FEATURES].to_numpy(dtype=np.float64),
        training["fhd_normal"].to_numpy(dtype=np.float64),
        selected_alpha,
        weights,
    )
    common = target_predictors[KEYS + ["terrain_slope_degrees"]].copy()
    common["analysis"] = analysis
    common["outer_held_out_site"] = target_site
    common["strategy"] = strategy
    common["training_sites"] = "+".join(sources)

    model_frame = common.copy()
    model_frame["method"] = "tessera_area_ridge"
    model_frame["selected_alpha"] = selected_alpha
    model_frame["prediction"] = model.predict(
        scaler.transform(target_predictors[FEATURES].to_numpy(dtype=np.float64))
    )
    site_means = training.groupby("site_id")["fhd_normal"].mean()
    mean_frame = common.copy()
    mean_frame["method"] = "training_site_mean"
    mean_frame["selected_alpha"] = np.nan
    mean_frame["prediction"] = float(site_means.mean())

    source_records = []
    hierarchy = strategy.removeprefix("nearest5_") if strategy != "all_sources" else None
    for source in sources:
        distance = math.nan
        rank = math.nan
        if hierarchy is not None:
            row = distances[
                distances["hierarchy"].eq(hierarchy)
                & distances["target_site"].eq(target_site)
                & distances["source_site"].eq(source)
            ].iloc[0]
            distance = float(row["distance"])
            rank = int(row["source_rank"])
        source_records.append({
            "analysis": analysis,
            "outer_held_out_site": target_site,
            "strategy": strategy,
            "source_site": source,
            "habitat_hierarchy": hierarchy,
            "habitat_distance": distance,
            "source_rank": rank,
            "source_rows": int(len(training[training["site_id"].eq(source)])),
            "source_total_fit_weight": float(weights[training["site_id"].eq(source)].sum()),
            "selected_alpha": selected_alpha,
        })
    return [mean_frame, model_frame], tuning, source_records


def build_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    group_columns = ["analysis", "outer_held_out_site", "strategy", "method"]
    for keys, frame in predictions.groupby(group_columns, sort=True):
        analysis, site, strategy, method = keys
        records.append({
            "scope": "site",
            "analysis": analysis,
            "held_out_site": site,
            "strategy": strategy,
            "method": method,
            "rows": len(frame),
            **loso.metric_values(frame["fhd_normal"].to_numpy(), frame["prediction"].to_numpy()),
        })
    site_metrics = pd.DataFrame(records)
    for keys, frame in site_metrics.groupby(["analysis", "strategy", "method"], sort=True):
        analysis, strategy, method = keys
        records.append({
            "scope": "macro",
            "analysis": analysis,
            "held_out_site": "macro_mean",
            "strategy": strategy,
            "method": method,
            "rows": len(frame),
            **{metric: float(frame[metric].mean()) for metric in loso.METRICS},
        })
    return pd.DataFrame(records).sort_values(
        ["analysis", "scope", "held_out_site", "strategy", "method"]
    ).reset_index(drop=True)


def paired_summary(
    metrics: pd.DataFrame,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    site = metrics[
        metrics["scope"].eq("site") & metrics["method"].eq("tessera_area_ridge")
    ]
    records = []
    rng = np.random.default_rng(seed)
    for analysis in sorted(site["analysis"].unique()):
        baseline = site[
            site["analysis"].eq(analysis) & site["strategy"].eq("all_sources")
        ].set_index("held_out_site")
        strategies = sorted(set(site[site["analysis"].eq(analysis)]["strategy"]) - {"all_sources"})
        for strategy in strategies:
            matched = site[
                site["analysis"].eq(analysis) & site["strategy"].eq(strategy)
            ].set_index("held_out_site").loc[baseline.index]
            delta = (matched["rmse"] - baseline["rmse"]).to_numpy(dtype=np.float64)
            bootstrap = np.mean(
                delta[rng.integers(0, len(delta), size=(replicates, len(delta)))], axis=1
            )
            improved = int(np.sum(delta < 0))
            records.append({
                "analysis": analysis,
                "strategy": strategy,
                "site_count": len(delta),
                "sites_with_lower_rmse": improved,
                "mean_delta_rmse": float(np.mean(delta)),
                "median_delta_rmse": float(np.median(delta)),
                "bootstrap_95_low": float(np.quantile(bootstrap, 0.025)),
                "bootstrap_95_high": float(np.quantile(bootstrap, 0.975)),
                "two_sided_sign_test_p": float(binomtest(improved, len(delta), 0.5).pvalue),
                "mean_delta_r2": float((matched["r2"] - baseline["r2"]).mean()),
                "mean_delta_spearman_r": float(
                    (matched["spearman_r"] - baseline["spearman_r"]).mean()
                ),
            })
    return pd.DataFrame(records)


def slope_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    primary = predictions[predictions["analysis"].eq(PRIMARY_ANALYSIS)].copy()
    primary["slope_stratum"] = pd.cut(
        primary["terrain_slope_degrees"],
        bins=[-np.inf, 15.0, 30.0, np.inf],
        labels=["lt_15", "15_to_30", "gt_30"],
        right=True,
    )
    scoped = primary[
        primary["strategy"].isin(["all_sources", "nearest5_evt_phys"])
        & primary["method"].eq("tessera_area_ridge")
    ]
    records = []
    for keys, frame in scoped.groupby(["strategy", "slope_stratum"], observed=True, sort=True):
        strategy, stratum = keys
        records.append({
            "strategy": strategy,
            "slope_stratum": str(stratum),
            "rows": len(frame),
            **loso.metric_values(frame["fhd_normal"].to_numpy(), frame["prediction"].to_numpy()),
        })
    return pd.DataFrame(records)


def save_figure(metrics: pd.DataFrame, paired: pd.DataFrame) -> None:
    site = metrics[
        metrics["scope"].eq("site")
        & metrics["analysis"].eq(PRIMARY_ANALYSIS)
        & metrics["method"].eq("tessera_area_ridge")
    ]
    all_source = site[site["strategy"].eq("all_sources")].set_index("held_out_site")
    primary = site[site["strategy"].eq("nearest5_evt_phys")].set_index("held_out_site").loc[
        all_source.index
    ]
    sites = all_source.index.tolist()
    x = np.arange(len(sites))
    delta = primary["rmse"] - all_source["rmse"]
    hierarchy = paired[paired["analysis"].eq(PRIMARY_ANALYSIS)].copy()

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw={"width_ratios": [1.25, 1, 1]})
    axes[0].plot(x, all_source["rmse"], marker="o", color="#3a86ff", label="All seven sources")
    axes[0].plot(x, primary["rmse"], marker="s", color="#0a9396", label="Five habitat-nearest")
    axes[0].set_xticks(x, sites, rotation=45, ha="right")
    axes[0].set_ylabel("Held-out-site RMSE")
    axes[0].set_xlabel("Target site")
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].grid(axis="y", alpha=0.2)

    colours = np.where(delta.to_numpy() < 0, "#0a9396", "#d95f02")
    axes[1].bar(sites, delta, color=colours)
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_xticks(range(len(sites)), sites, rotation=45, ha="right")
    axes[1].set_ylabel("Matched minus all-source RMSE")
    axes[1].set_xlabel("Target site")
    axes[1].grid(axis="y", alpha=0.2)

    labels = hierarchy["strategy"].str.replace("nearest5_", "", regex=False)
    axes[2].bar(labels, hierarchy["mean_delta_rmse"], color="#6c757d")
    axes[2].errorbar(
        labels,
        hierarchy["mean_delta_rmse"],
        yerr=np.vstack([
            hierarchy["mean_delta_rmse"] - hierarchy["bootstrap_95_low"],
            hierarchy["bootstrap_95_high"] - hierarchy["mean_delta_rmse"],
        ]),
        fmt="none",
        color="black",
        capsize=3,
    )
    axes[2].axhline(0, color="black", linewidth=1)
    axes[2].set_ylabel("Mean paired RMSE change")
    axes[2].set_xlabel("LANDFIRE hierarchy")
    axes[2].grid(axis="y", alpha=0.2)
    figure.suptitle("Exploratory habitat-conditioned TESSERA transfer")
    figure.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = FIGURE_PATH.with_suffix(".tmp.png")
    figure.savefig(temporary, dpi=180, bbox_inches="tight")
    plt.close(figure)
    temporary.replace(FIGURE_PATH)


def main() -> int:
    protected = [
        PREDICTIONS_PATH, METRICS_PATH, TUNING_PATH, SELECTION_PATH, PAIRED_PATH,
        SLOPE_PATH, FIGURE_PATH, FREEZE_PATH,
    ]
    existing = [str(path.relative_to(ROOT)) for path in protected if path.exists()]
    if existing:
        raise RuntimeError(f"Phase 9 transfer freeze exists; refusing to overwrite: {existing}")
    habitat_freeze = json.loads(HABITAT_FREEZE_PATH.read_text(encoding="utf-8"))
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))[
        "phase9_habitat_conditioned_transfer"
    ]
    distances = pd.read_csv(DISTANCE_PATH)
    points = pd.read_parquet(POINT_HABITAT_PATH)
    conventional = pd.read_parquet(
        CONVENTIONAL_PATH, columns=KEYS + ["terrain_slope_degrees"]
    )
    population = pd.concat([
        pd.read_parquet(path, columns=KEYS + ["fhd_normal"] + FEATURES)
        for path in POPULATION_PATHS
    ], ignore_index=True)
    population = population.merge(conventional, on=KEYS, validate="one_to_one").merge(
        points[KEYS + [
            "landfire_evt_name", "landfire_evt_lifeform", "landfire_evt_phys",
            "landfire_evt_group",
        ]], on=KEYS, validate="one_to_one"
    )
    sites = sorted(population["site_id"].unique())
    if sites != sorted(protocol["current_sites"]) or population.duplicated(KEYS).any():
        raise RuntimeError("Unexpected habitat-transfer population")
    if not np.isfinite(population[FEATURES + ["fhd_normal", "terrain_slope_degrees"]].to_numpy()).all():
        raise RuntimeError("Non-finite values entered habitat transfer")

    alphas = [float(value) for value in protocol["exploratory_transfer"]["alpha_grid"]]
    nearest_count = 5
    analyses = {
        PRIMARY_ANALYSIS: {
            "population": population,
            "strategies": ["all_sources", *[f"nearest5_{value}" for value in HIERARCHIES]],
        },
        SLOPE_ANALYSIS: {
            "population": population[population["terrain_slope_degrees"].le(30)].copy(),
            "strategies": ["all_sources", "nearest5_evt_phys"],
        },
    }
    prediction_frames: list[pd.DataFrame] = []
    tuning_frames: list[pd.DataFrame] = []
    source_records: list[dict[str, Any]] = []
    for analysis, specification in analyses.items():
        analysis_population = specification["population"]
        for target_site in sites:
            for strategy in specification["strategies"]:
                predictions, tuning, sources = fit_strategy(
                    analysis_population,
                    distances,
                    target_site,
                    strategy,
                    analysis,
                    alphas,
                    nearest_count,
                )
                prediction_frames.extend(predictions)
                tuning_frames.append(tuning)
                source_records.extend(sources)

    predictions = pd.concat(prediction_frames, ignore_index=True).merge(
        population[KEYS + [
            "fhd_normal", "landfire_evt_name", "landfire_evt_lifeform",
            "landfire_evt_phys", "landfire_evt_group",
        ]], on=KEYS, validate="many_to_one"
    ).sort_values(["analysis", "outer_held_out_site", "strategy", "method", "shot_number"])
    tuning = pd.concat(tuning_frames, ignore_index=True)
    selections = pd.DataFrame(source_records)
    metrics = build_metrics(predictions)
    paired = paired_summary(
        metrics,
        int(protocol["exploratory_transfer"]["bootstrap_replicates"]),
        int(protocol["exploratory_transfer"]["random_seed"]),
    )
    slope = slope_metrics(predictions)

    phase7 = pd.read_parquet(PHASE7_PREDICTIONS_PATH)
    expected = phase7[
        phase7["method"].eq("tessera_area_ridge")
    ][KEYS + ["prediction"]].rename(columns={"prediction": "phase7_prediction"})
    reproduced = predictions[
        predictions["analysis"].eq(PRIMARY_ANALYSIS)
        & predictions["strategy"].eq("all_sources")
        & predictions["method"].eq("tessera_area_ridge")
    ][KEYS + ["prediction"]].merge(expected, on=KEYS, validate="one_to_one")
    reproduction_delta = float(
        np.max(np.abs(reproduced["prediction"] - reproduced["phase7_prediction"]))
    )
    if reproduction_delta > 1e-10:
        raise RuntimeError(f"All-source comparator did not reproduce Phase 7: {reproduction_delta}")

    write_parquet(predictions.reset_index(drop=True), PREDICTIONS_PATH)
    write_csv(metrics, METRICS_PATH)
    write_csv(tuning, TUNING_PATH)
    write_csv(selections, SELECTION_PATH)
    write_csv(paired, PAIRED_PATH)
    write_csv(slope, SLOPE_PATH)
    save_figure(metrics, paired)

    outputs = {
        "predictions": {"path": str(PREDICTIONS_PATH.relative_to(ROOT)), "rows": len(predictions), "sha256": sha256(PREDICTIONS_PATH)},
        "metrics": {"path": str(METRICS_PATH.relative_to(ROOT)), "rows": len(metrics), "sha256": sha256(METRICS_PATH)},
        "inner_tuning": {"path": str(TUNING_PATH.relative_to(ROOT)), "rows": len(tuning), "sha256": sha256(TUNING_PATH)},
        "source_selection": {"path": str(SELECTION_PATH.relative_to(ROOT)), "rows": len(selections), "sha256": sha256(SELECTION_PATH)},
        "paired_summary": {"path": str(PAIRED_PATH.relative_to(ROOT)), "rows": len(paired), "sha256": sha256(PAIRED_PATH)},
        "slope_metrics": {"path": str(SLOPE_PATH.relative_to(ROOT)), "rows": len(slope), "sha256": sha256(SLOPE_PATH)},
        "figure": {"path": str(FIGURE_PATH.relative_to(ROOT)), "sha256": sha256(FIGURE_PATH)},
    }
    primary = paired[
        paired["analysis"].eq(PRIMARY_ANALYSIS)
        & paired["strategy"].eq("nearest5_evt_phys")
    ].iloc[0]
    gate = {
        "exploratory_only": True,
        "primary_strategy": "nearest5_evt_phys",
        "mean_delta_rmse": float(primary["mean_delta_rmse"]),
        "sites_with_lower_rmse": int(primary["sites_with_lower_rmse"]),
        "site_count": int(primary["site_count"]),
        "bootstrap_95_low": float(primary["bootstrap_95_low"]),
        "bootstrap_95_high": float(primary["bootstrap_95_high"]),
        "descriptive_pattern_present": bool(
            primary["mean_delta_rmse"] < 0
            and primary["sites_with_lower_rmse"] >= math.floor(len(sites) / 2) + 1
        ),
        "confirmatory_claim_allowed": False,
    }
    freeze_basis = {
        "timing": protocol["timing"]["current_eight_sites"],
        "habitat_freeze_id": habitat_freeze["freeze_id"],
        "sites": sites,
        "site_rows": {site: int(count) for site, count in population.groupby("site_id").size().items()},
        "outer_split": "leave_one_complete_site_out",
        "target_site_fhd_used_for_habitat_matching_scaling_tuning_or_training": False,
        "all_source_phase7_prediction_max_absolute_delta": reproduction_delta,
        "strategies": analyses[PRIMARY_ANALYSIS]["strategies"],
        "slope_sensitivity_strategies": analyses[SLOPE_ANALYSIS]["strategies"],
        "features": FEATURES,
        "alpha_grid": alphas,
        "equal_total_weight_per_training_site": True,
        "gate": gate,
        "input_hashes": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [*POPULATION_PATHS, DISTANCE_PATH, POINT_HABITAT_PATH, CONVENTIONAL_PATH]
        },
        "software": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    freeze_id = "phase9-habitat-transfer-" + canonical_hash(
        {"freeze_basis": freeze_basis, "outputs": outputs}
    )[:12]
    write_json(FREEZE_PATH, {
        "freeze_id": freeze_id,
        "created_utc": utc_now(),
        "freeze_basis": freeze_basis,
        "outputs": outputs,
    })
    macro = metrics[
        metrics["analysis"].eq(PRIMARY_ANALYSIS)
        & metrics["scope"].eq("macro")
        & metrics["method"].eq("tessera_area_ridge")
    ][["strategy", "rmse", "r2", "spearman_r"]]
    print(json.dumps({
        "freeze_id": freeze_id,
        "phase7_reproduction_max_delta": reproduction_delta,
        "macro_metrics": macro.set_index("strategy").to_dict(orient="index"),
        "primary_gate": gate,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
