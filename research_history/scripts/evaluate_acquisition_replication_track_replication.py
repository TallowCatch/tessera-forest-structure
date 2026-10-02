#!/usr/bin/env python3
"""Open sealed Phase 13 FHD outcomes and evaluate the frozen track experiment."""

from __future__ import annotations

import json
import sys
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
import run_context_height_footprint_height_transfer as common  # noqa: E402


PROTOCOL_PATH = ROOT / "metadata/project_config_phase13_track_replication_protocol_freeze.yaml"
TARGET_FREEZE_PATH = ROOT / "metadata/phase13_combined_target_free_freeze.json"
PREDICTION_FREEZE_PATH = ROOT / "metadata/phase13_prediction_freeze.json"
PREDICTIONS_PATH = ROOT / "data/processed/phase13_frozen_source_predictions.parquet"
SAMPLES_PATH = ROOT / "data/processed/phase13_frozen_local_samples.parquet"
SCENARIO_PATH = ROOT / "outputs/tables/phase13_track_replication_scenario_metrics.csv"
FOLD_PATH = ROOT / "outputs/tables/phase13_track_replication_fold_metrics.csv"
SITE_PATH = ROOT / "outputs/tables/phase13_track_replication_site_metrics.csv"
MACRO_PATH = ROOT / "outputs/tables/phase13_track_replication_macro_metrics.csv"
YEAR_MACRO_PATH = ROOT / "outputs/tables/phase13_track_replication_year_metrics.csv"
PAIRED_PATH = ROOT / "outputs/tables/phase13_track_replication_paired_sites.csv"
BOOTSTRAP_PATH = ROOT / "outputs/tables/phase13_track_replication_bootstrap.csv"
FIGURE_PATH = ROOT / "outputs/figures/phase13_track_replication.png"
FREEZE_PATH = ROOT / "metadata/phase13_track_replication_result_freeze.json"

MODELS = [
    "source_only_tessera",
    "sampled_target_local_mean",
    "tessera_plus_sampled_local_offset",
    "source_only_sentinel2_terrain",
    "sentinel2_terrain_plus_sampled_local_offset",
]
METRICS = [
    "r2",
    "rmse",
    "mae",
    "pearson_r",
    "spearman_r",
    "mean_bias",
    "prediction_slope",
    "prediction_sd_ratio",
]


def load_sealed_outcomes(target_freeze: dict[str, Any]) -> pd.DataFrame:
    records = target_freeze["freeze_basis"]["sealed_outcome_files"]
    frames = []
    for record in records:
        path = ROOT / record["path"]
        if common.sha256(path) != record["sha256"]:
            raise RuntimeError(f"Sealed Phase 13 outcome hash changed: {path}")
        frame = pd.read_parquet(path, columns=["site_id", "shot_number", "fhd_normal"])
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise RuntimeError("Sealed Phase 13 outcome files contain no records")
    outcomes = pd.concat(frames, ignore_index=True)
    outcomes["fhd_normal"] = pd.to_numeric(outcomes["fhd_normal"], errors="raise")
    duplicate_values = outcomes[outcomes.duplicated(["site_id", "shot_number"], keep=False)]
    if not duplicate_values.empty:
        spread = duplicate_values.groupby(["site_id", "shot_number"])["fhd_normal"].nunique(
            dropna=False
        )
        if spread.gt(1).any():
            raise RuntimeError("Duplicate sealed FHD records disagree")
    return outcomes.drop_duplicates(["site_id", "shot_number"], keep="first")


def aggregate_metrics(
    scenarios: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold = (
        scenarios.groupby(
            ["target_year", "site_id", "held_out_pass", "budget", "model"],
            as_index=False,
        )
        .agg(
            replicates=("replicate", "nunique"),
            test_rows=("test_rows", "first"),
            **{metric: (metric, "mean") for metric in METRICS},
        )
    )
    site = (
        fold.groupby(["target_year", "site_id", "budget", "model"], as_index=False)
        .agg(
            folds=("held_out_pass", "nunique"),
            test_rows=("test_rows", "sum"),
            **{metric: (metric, "mean") for metric in METRICS},
        )
    )
    macro = (
        site.groupby(["budget", "model"], as_index=False)
        .agg(
            sites=("site_id", "nunique"),
            folds=("folds", "sum"),
            **{metric: (metric, "mean") for metric in METRICS},
        )
    )
    return fold, site, macro


def paired_results(
    site: pd.DataFrame, primary_budget: int, replicates: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scoped = site[site["budget"].eq(primary_budget)]
    comparisons = [
        ("tessera_offset_minus_source_tessera", "tessera_plus_sampled_local_offset", "source_only_tessera"),
        ("tessera_offset_minus_local_mean", "tessera_plus_sampled_local_offset", "sampled_target_local_mean"),
        (
            "sentinel_offset_minus_source_sentinel",
            "sentinel2_terrain_plus_sampled_local_offset",
            "source_only_sentinel2_terrain",
        ),
        (
            "tessera_offset_minus_sentinel_offset",
            "tessera_plus_sampled_local_offset",
            "sentinel2_terrain_plus_sampled_local_offset",
        ),
        ("source_tessera_minus_source_sentinel", "source_only_tessera", "source_only_sentinel2_terrain"),
    ]
    paired_records = []
    bootstrap_records = []
    rng = np.random.default_rng(seed)
    for name, candidate_name, reference_name in comparisons:
        candidate = scoped[scoped["model"].eq(candidate_name)].set_index("site_id")
        reference = scoped[scoped["model"].eq(reference_name)].set_index("site_id")
        sites = sorted(set(candidate.index) & set(reference.index))
        differences = (
            candidate.loc[sites, "rmse"].to_numpy(dtype=np.float64)
            - reference.loc[sites, "rmse"].to_numpy(dtype=np.float64)
        )
        for site_id, difference in zip(sites, differences, strict=True):
            paired_records.append(
                {
                    "comparison": name,
                    "site_id": site_id,
                    "candidate_model": candidate_name,
                    "reference_model": reference_name,
                    "rmse_difference": difference,
                }
            )
        draws = np.mean(
            differences[rng.integers(0, len(differences), size=(replicates, len(differences)))],
            axis=1,
        )
        bootstrap_records.append(
            {
                "comparison": name,
                "sites": len(sites),
                "sites_improved": int(np.sum(differences < 0)),
                "mean_paired_rmse_difference": float(np.mean(differences)),
                "site_bootstrap_95_low": float(np.quantile(draws, 0.025)),
                "site_bootstrap_95_high": float(np.quantile(draws, 0.975)),
                "bootstrap_probability_lower_rmse": float(np.mean(draws < 0)),
                "bootstrap_replicates": replicates,
                "bootstrap_seed": seed,
            }
        )
    return pd.DataFrame(paired_records), pd.DataFrame(bootstrap_records)


def primary_gate(
    macro: pd.DataFrame, paired: pd.DataFrame, bootstrap: pd.DataFrame, budget: int
) -> dict[str, Any]:
    values = macro[macro["budget"].eq(budget)].set_index("model")
    site_delta = paired[paired["comparison"].eq("tessera_offset_minus_source_tessera")]
    boot = bootstrap.set_index("comparison").loc["tessera_offset_minus_source_tessera"]
    conditions = {
        "offset_rmse_below_source_only_tessera": bool(
            values.loc["tessera_plus_sampled_local_offset", "rmse"]
            < values.loc["source_only_tessera", "rmse"]
        ),
        "offset_rmse_below_corresponding_local_mean": bool(
            values.loc["tessera_plus_sampled_local_offset", "rmse"]
            < values.loc["sampled_target_local_mean", "rmse"]
        ),
        "majority_of_sites_improve_over_source_only": bool(
            site_delta["rmse_difference"].lt(0).sum() > len(site_delta) / 2
        ),
        "site_bootstrap_95_upper_bound_below_zero": bool(
            boot["site_bootstrap_95_high"] < 0
        ),
    }
    return {"conditions": conditions, "passed": all(conditions.values())}


def make_figure(macro: pd.DataFrame, paired: pd.DataFrame, primary_budget: int) -> None:
    labels = {
        "source_only_tessera": "TESSERA\nsource only",
        "sampled_target_local_mean": "Local\nmean",
        "tessera_plus_sampled_local_offset": "TESSERA +\nlocal correction",
        "source_only_sentinel2_terrain": "Sentinel-2 + terrain\nsource only",
        "sentinel2_terrain_plus_sampled_local_offset": "Sentinel-2 + terrain\n+ local correction",
    }
    colours = {
        "source_only_tessera": "#6C757D",
        "sampled_target_local_mean": "#D6A756",
        "tessera_plus_sampled_local_offset": "#0A9396",
        "source_only_sentinel2_terrain": "#8AB6F9",
        "sentinel2_terrain_plus_sampled_local_offset": "#3B82F6",
    }
    scoped = macro[macro["budget"].eq(primary_budget)].set_index("model").loc[MODELS]
    delta = paired[paired["comparison"].eq("tessera_offset_minus_source_tessera")].sort_values(
        "site_id"
    )
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.3), constrained_layout=True)
    axes[0].bar(
        np.arange(len(MODELS)),
        scoped["rmse"],
        color=[colours[model] for model in MODELS],
    )
    axes[0].set_xticks(np.arange(len(MODELS)), [labels[model] for model in MODELS])
    axes[0].set_ylabel("Mean RMSE across held-out passes and forests")
    axes[0].set_title(f"Independent replication with {primary_budget} local footprints")
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].bar(
        delta["site_id"],
        delta["rmse_difference"],
        color=np.where(delta["rmse_difference"].lt(0), "#0A9396", "#C44536"),
    )
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_ylabel("TESSERA + local correction minus source-only RMSE")
    axes[1].set_title("Replication effect by forest")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(axis="y", alpha=0.2)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=220)
    plt.close(figure)


def main() -> int:
    protected = [SCENARIO_PATH, FOLD_PATH, SITE_PATH, MACRO_PATH, YEAR_MACRO_PATH, PAIRED_PATH, BOOTSTRAP_PATH, FIGURE_PATH, FREEZE_PATH]
    existing = [str(path.relative_to(ROOT)) for path in protected if path.exists()]
    if existing:
        raise RuntimeError(f"Phase 13 evaluation outputs exist; refusing to overwrite: {existing}")
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))[
        "phase13_track_assisted_replication"
    ]
    target_freeze = json.loads(TARGET_FREEZE_PATH.read_text(encoding="utf-8"))
    prediction_freeze = json.loads(PREDICTION_FREEZE_PATH.read_text(encoding="utf-8"))
    if prediction_freeze["freeze_basis"]["target_fhd_read"]:
        raise RuntimeError("Phase 13 predictions were not outcome blind")
    if prediction_freeze["freeze_basis"]["target_freeze_id"] != target_freeze["freeze_id"]:
        raise RuntimeError("Phase 13 target and prediction freezes disagree")
    predictions = pd.read_parquet(PREDICTIONS_PATH)
    samples = pd.read_parquet(SAMPLES_PATH)
    outcomes = load_sealed_outcomes(target_freeze)
    target = predictions.merge(outcomes, on=["site_id", "shot_number"], validate="one_to_one")
    if len(target) != len(predictions) or not np.isfinite(target["fhd_normal"]).all():
        raise RuntimeError("Phase 13 frozen prediction rows lack valid FHD outcomes")
    if samples.duplicated(["site_id", "held_out_pass", "budget", "replicate", "shot_number"]).any():
        raise RuntimeError("A Phase 13 local sample contains duplicate shots")

    records: list[dict[str, Any]] = []
    scenario_groups = samples.groupby(
        ["site_id", "held_out_pass", "budget", "replicate"], sort=True
    )
    for index, ((site, held_out, budget, replicate), sampled) in enumerate(scenario_groups, start=1):
        site_target = target[target["site_id"].eq(site)].set_index("shot_number", drop=False)
        test = site_target[site_target["pass_id"].eq(held_out)]
        if test.empty or sampled["pass_id"].eq(held_out).any():
            raise RuntimeError("Phase 13 local reference overlaps its held-out test pass")
        local = site_target.loc[sampled["shot_number"].to_numpy()]
        y_local = local["fhd_normal"].to_numpy(dtype=np.float64)
        y_test = test["fhd_normal"].to_numpy(dtype=np.float64)
        target_year = int(test["target_year"].iloc[0])
        if target_year != int(local["target_year"].iloc[0]):
            raise RuntimeError("A Phase 13 scenario mixes target years")
        tessera_local = local["source_tessera_prediction"].to_numpy(dtype=np.float64)
        tessera_test = test["source_tessera_prediction"].to_numpy(dtype=np.float64)
        sentinel_local = local["source_sentinel2_terrain_prediction"].to_numpy(dtype=np.float64)
        sentinel_test = test["source_sentinel2_terrain_prediction"].to_numpy(dtype=np.float64)
        local_mean = float(np.mean(y_local))
        tessera_offset = float(np.mean(y_local - tessera_local))
        sentinel_offset = float(np.mean(y_local - sentinel_local))
        model_predictions = {
            "source_only_tessera": tessera_test,
            "sampled_target_local_mean": np.full(len(test), local_mean),
            "tessera_plus_sampled_local_offset": tessera_test + tessera_offset,
            "source_only_sentinel2_terrain": sentinel_test,
            "sentinel2_terrain_plus_sampled_local_offset": sentinel_test + sentinel_offset,
        }
        for model, values in model_predictions.items():
            records.append(
                {
                    "site_id": site,
                    "target_year": target_year,
                    "held_out_pass": held_out,
                    "budget": int(budget),
                    "replicate": int(replicate),
                    "model": model,
                    "test_rows": len(test),
                    "local_rows": len(local),
                    "local_passes": int(local["pass_id"].nunique()),
                    "local_mean": local_mean,
                    "tessera_offset": tessera_offset,
                    "sentinel_offset": sentinel_offset,
                    **common.metric_values(y_test, values),
                }
            )
        if index % 500 == 0:
            print(f"evaluated {index:,}/{len(scenario_groups):,} frozen scenarios", flush=True)

    scenarios = pd.DataFrame(records)
    fold, site, macro = aggregate_metrics(scenarios)
    year_macro = (
        site.groupby(["target_year", "budget", "model"], as_index=False)
        .agg(
            sites=("site_id", "nunique"),
            folds=("folds", "sum"),
            **{metric: (metric, "mean") for metric in METRICS},
        )
    )
    primary_budget = int(protocol["local_reference"]["primary_budget"])
    uncertainty = protocol["uncertainty"]
    paired, bootstrap = paired_results(
        site,
        primary_budget,
        int(uncertainty["bootstrap_replicates"]),
        int(uncertainty["seed"]),
    )
    gate = primary_gate(macro, paired, bootstrap, primary_budget)
    common.write_csv(scenarios, SCENARIO_PATH)
    common.write_csv(fold, FOLD_PATH)
    common.write_csv(site, SITE_PATH)
    common.write_csv(macro, MACRO_PATH)
    common.write_csv(year_macro, YEAR_MACRO_PATH)
    common.write_csv(paired, PAIRED_PATH)
    common.write_csv(bootstrap, BOOTSTRAP_PATH)
    make_figure(macro, paired, primary_budget)
    freeze_basis = {
        "protocol_sha256": common.sha256(PROTOCOL_PATH),
        "script_sha256": common.sha256(Path(__file__).resolve()),
        "target_freeze_id": target_freeze["freeze_id"],
        "prediction_freeze_id": prediction_freeze["freeze_id"],
        "sealed_target_outcomes_opened_after_prediction_freeze": True,
        "target_test_pass_fhd_used_in_training_selection_or_offset_estimation": False,
        "sites": sorted(target["site_id"].unique()),
        "site_years": target_freeze["freeze_basis"]["site_years"],
        "budgets": sorted(int(value) for value in scenarios["budget"].unique()),
        "frozen_scenarios": int(len(scenario_groups)),
        "primary_gate": gate,
    }
    freeze = {
        "freeze_id": "phase13-track-replication-" + common.canonical_hash(freeze_basis)[:12],
        "created_utc": common.utc_now(),
        "status": "complete_gate_passed" if gate["passed"] else "complete_gate_failed",
        "freeze_basis": freeze_basis,
        "outputs": {
            str(path.relative_to(ROOT)): common.sha256(path)
            for path in [SCENARIO_PATH, FOLD_PATH, SITE_PATH, MACRO_PATH, YEAR_MACRO_PATH, PAIRED_PATH, BOOTSTRAP_PATH, FIGURE_PATH]
        },
    }
    common.write_json(FREEZE_PATH, freeze)
    primary_macro = macro[macro["budget"].eq(primary_budget)].set_index("model")
    print(
        json.dumps(
            {
                "freeze_id": freeze["freeze_id"],
                "gate": gate,
                "primary_macro": primary_macro[["rmse", "r2", "spearman_r", "mean_bias"]].to_dict(
                    orient="index"
                ),
                "paired_bootstrap": bootstrap.to_dict(orient="records"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
