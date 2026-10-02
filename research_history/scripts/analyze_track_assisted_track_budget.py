#!/usr/bin/env python3
"""Quantify one-pass and sampled-shot requirements for Phase 12 assistance."""

from __future__ import annotations

import hashlib
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
import freeze_track_assisted_splits as split_tools  # noqa: E402
import run_context_height_footprint_height_transfer as base  # noqa: E402
import run_track_assisted_prediction as phase12  # noqa: E402


PROTOCOL_PATH = ROOT / "metadata/project_config_phase12_track_budget_protocol_freeze.yaml"
SPLIT_PATH = ROOT / "metadata/phase12_track_assisted_split_manifest.json"
RESULT_FREEZE_PATH = ROOT / "metadata/phase12_track_assisted_prediction_freeze.json"
SOURCE_SELECTION_PATH = ROOT / "outputs/tables/phase12_track_assisted_source_selection.csv"
MAIN_FOLD_METRICS_PATH = ROOT / "outputs/tables/phase12_track_assisted_fold_metrics.csv"
SCENARIOS_PATH = ROOT / "outputs/tables/phase12_track_budget_scenarios.csv"
FOLD_PATH = ROOT / "outputs/tables/phase12_track_budget_fold_metrics.csv"
SITE_PATH = ROOT / "outputs/tables/phase12_track_budget_site_metrics.csv"
MACRO_PATH = ROOT / "outputs/tables/phase12_track_budget_macro_metrics.csv"
BOOTSTRAP_PATH = ROOT / "outputs/tables/phase12_track_budget_bootstrap.csv"
FIGURE_PATH = ROOT / "outputs/figures/phase12_track_budget.png"
FREEZE_PATH = ROOT / "metadata/phase12_track_budget_freeze.json"


def stable_seed(*values: object, base_seed: int) -> int:
    payload = "|".join(str(value) for value in values).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return (int.from_bytes(digest[:8], "big") + int(base_seed)) % (2**32)


def stratified_sample_positions(
    pass_ids: np.ndarray,
    budget: int,
    rng: np.random.Generator,
) -> np.ndarray:
    pass_ids = np.asarray(pass_ids, dtype=str)
    if budget > len(pass_ids) or budget <= 0:
        raise ValueError("Budget must be positive and no larger than the local population")
    groups = sorted(np.unique(pass_ids))
    if budget < len(groups):
        raise ValueError("Budget is too small to represent every retained pass")
    chosen = []
    for group in groups:
        candidates = np.flatnonzero(pass_ids == group)
        chosen.append(int(rng.choice(candidates)))
    remaining = budget - len(chosen)
    if remaining:
        pool = np.setdiff1d(np.arange(len(pass_ids)), np.asarray(chosen), assume_unique=False)
        chosen.extend(rng.choice(pool, size=remaining, replace=False).astype(int).tolist())
    return np.sort(np.asarray(chosen, dtype=int))


def scenario_metrics(
    observed_test: np.ndarray,
    source_test: np.ndarray,
    observed_calibration: np.ndarray,
    source_calibration: np.ndarray,
) -> dict[str, dict[str, float]]:
    offset = float(np.mean(observed_calibration - source_calibration))
    local_mean = float(np.mean(observed_calibration))
    return {
        "offset": base.metric_values(observed_test, source_test + offset),
        "local_mean": base.metric_values(
            observed_test, np.full(len(observed_test), local_mean)
        ),
    }


def aggregate_scenarios(scenarios: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    fold = (
        scenarios.groupby(["site_id", "held_out_pass", "method"], as_index=False)
        .agg(
            scenarios=("scenario_id", "nunique"),
            test_rows=("test_rows", "first"),
            calibration_rows=("calibration_rows", "mean"),
            **{metric: (metric, "mean") for metric in metric_columns},
        )
    )
    site = (
        fold.groupby(["site_id", "method"], as_index=False)
        .agg(
            folds=("held_out_pass", "nunique"),
            scenarios=("scenarios", "sum"),
            test_rows=("test_rows", "sum"),
            calibration_rows=("calibration_rows", "mean"),
            **{metric: (metric, "mean") for metric in metric_columns},
        )
    )
    macro = (
        site.groupby("method", as_index=False)
        .agg(
            sites=("site_id", "nunique"),
            folds=("folds", "sum"),
            scenarios=("scenarios", "sum"),
            calibration_rows=("calibration_rows", "mean"),
            **{metric: (metric, "mean") for metric in metric_columns},
        )
    )
    return fold, site, macro


def add_main_references(
    site: pd.DataFrame,
    macro: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    main = pd.read_csv(MAIN_FOLD_METRICS_PATH)
    main = main[
        main["analysis_group"].eq("primary")
        & main["model"].isin(["source_tessera_ridge", "source_plus_local_offset"])
    ].copy()
    main["method"] = main["model"].map(
        {
            "source_tessera_ridge": "source_only_tessera",
            "source_plus_local_offset": "offset_all_retained_passes",
        }
    )
    metric_columns = [
        "r2", "rmse", "mae", "pearson_r", "spearman_r", "mean_bias",
        "prediction_slope", "prediction_sd_ratio",
    ]
    reference_site = (
        main.groupby(["site_id", "method"], as_index=False)
        .agg(
            folds=("held_out_pass", "nunique"),
            scenarios=("held_out_pass", "nunique"),
            test_rows=("test_rows", "sum"),
            calibration_rows=("local_training_rows", "mean"),
            **{metric: (metric, "mean") for metric in metric_columns},
        )
    )
    reference_macro = (
        reference_site.groupby("method", as_index=False)
        .agg(
            sites=("site_id", "nunique"),
            folds=("folds", "sum"),
            scenarios=("scenarios", "sum"),
            calibration_rows=("calibration_rows", "mean"),
            **{metric: (metric, "mean") for metric in metric_columns},
        )
    )
    return (
        pd.concat([site, reference_site], ignore_index=True),
        pd.concat([macro, reference_macro], ignore_index=True),
    )


def bootstrap_comparisons(
    fold: pd.DataFrame,
    main_fold: pd.DataFrame,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    methods = sorted(
        method for method in fold["method"].unique() if method.startswith("offset_")
    )
    records = []
    rng = np.random.default_rng(seed)
    for method in methods:
        candidate_fold = fold[fold["method"].eq(method)]
        candidate = (
            candidate_fold.groupby("site_id", sort=True)["rmse"].mean().to_frame()
        )
        keys = candidate_fold[["site_id", "held_out_pass"]].drop_duplicates()

        matched_references = {}
        for reference_name, model_name in [
            ("source_only_tessera", "source_tessera_ridge"),
            ("offset_all_retained_passes", "source_plus_local_offset"),
        ]:
            reference_folds = main_fold[
                main_fold["analysis_group"].eq("primary")
                & main_fold["model"].eq(model_name)
            ][["site_id", "held_out_pass", "rmse"]]
            matched = keys.merge(
                reference_folds,
                on=["site_id", "held_out_pass"],
                how="left",
                validate="one_to_one",
            )
            if matched["rmse"].isna().any():
                raise RuntimeError(f"Missing matched {reference_name} fold for {method}")
            matched_references[reference_name] = (
                matched.groupby("site_id", sort=True)["rmse"].mean().to_frame()
            )

        local_mean_name = method.replace("offset_", "local_mean_", 1)
        local_mean = (
            fold[fold["method"].eq(local_mean_name)]
            .groupby("site_id", sort=True)["rmse"]
            .mean()
            .to_frame()
        )
        references = [
            ("source_only_tessera", matched_references["source_only_tessera"]),
            (
                "offset_all_retained_passes",
                matched_references["offset_all_retained_passes"],
            ),
        ]
        if not local_mean.empty:
            references.append((local_mean_name, local_mean))
        for reference_name, reference_frame in references:
            common = sorted(set(candidate.index) & set(reference_frame.index))
            differences = (
                candidate.loc[common, "rmse"].to_numpy(dtype=np.float64)
                - reference_frame.loc[common, "rmse"].to_numpy(dtype=np.float64)
            )
            draws = np.mean(
                rng.choice(differences, size=(replicates, len(differences)), replace=True),
                axis=1,
            )
            records.append(
                {
                    "comparison": f"{method}_minus_{reference_name}",
                    "sites": len(common),
                    "sites_improved": int(np.sum(differences < 0)),
                    "mean_paired_rmse_difference": float(np.mean(differences)),
                    "site_bootstrap_95_low": float(np.quantile(draws, 0.025)),
                    "site_bootstrap_95_high": float(np.quantile(draws, 0.975)),
                    "bootstrap_probability_lower_rmse": float(np.mean(draws < 0)),
                    "bootstrap_replicates": replicates,
                    "bootstrap_seed": seed,
                }
            )
    return pd.DataFrame(records)


def make_figure(macro: pd.DataFrame) -> None:
    order = [
        "source_only_tessera",
        "offset_10_shots",
        "offset_25_shots",
        "offset_50_shots",
        "offset_100_shots",
        "offset_one_retained_pass",
        "offset_all_retained_passes",
    ]
    labels = [
        "Source\nonly", "10\nshots", "25\nshots", "50\nshots", "100\nshots",
        "One\npass", "All retained\npasses",
    ]
    values = macro.set_index("method")
    available = [(method, label) for method, label in zip(order, labels, strict=True) if method in values.index]
    figure, axis = plt.subplots(figsize=(9.0, 4.2), constrained_layout=True)
    colors = ["#6C757D"] + ["#0A9396"] * (len(available) - 1)
    axis.bar(
        np.arange(len(available)),
        [values.loc[method, "rmse"] for method, _ in available],
        color=colors,
    )
    axis.set_xticks(np.arange(len(available)), [label for _, label in available])
    axis.set(
        ylabel="Mean RMSE across held-out passes and forests",
        title="How much local GEDI reference data is needed?",
    )
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=220)
    plt.close(figure)


def main() -> int:
    protected = [SCENARIOS_PATH, FOLD_PATH, SITE_PATH, MACRO_PATH, BOOTSTRAP_PATH, FIGURE_PATH, FREEZE_PATH]
    existing = [str(path.relative_to(ROOT)) for path in protected if path.exists()]
    if existing:
        raise RuntimeError(f"Phase 12 budget outputs already exist: {existing}")
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))[
        "phase12_track_assistance_budget"
    ]
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    result = json.loads(RESULT_FREEZE_PATH.read_text(encoding="utf-8"))
    if result["freeze_id"] != protocol["prerequisite_result"]:
        raise RuntimeError("Unexpected Phase 12 result prerequisite")
    cohort = pd.concat(
        [pd.read_parquet(path, columns=phase12.READ_COLUMNS) for path in base.INPUT_PATHS],
        ignore_index=True,
    )
    cohort["pass_id"] = split_tools.pass_ids(cohort)
    selections = pd.read_csv(SOURCE_SELECTION_PATH)
    selections = selections[selections["selected"]].set_index("target_site")
    primary_sites = list(protocol["cohort"]["sites"])
    folds = [
        fold for fold in split["folds"]
        if fold["eligible_by_row_gates"] and fold["site_id"] in primary_sites
    ]
    if len(folds) != int(protocol["cohort"]["folds"]):
        raise RuntimeError("Phase 12 budget fold count changed")
    budgets = [int(value) for value in protocol["shot_budget_diagnostic"]["budgets"]]
    repeats = int(protocol["shot_budget_diagnostic"]["replicates_per_fold"])
    seed = int(protocol["shot_budget_diagnostic"]["seed"])
    scenario_records: list[dict[str, Any]] = []

    for site_index, site in enumerate(primary_sites, start=1):
        source = cohort[~cohort["site_id"].eq(site)].copy()
        target = cohort[cohort["site_id"].eq(site)].copy().reset_index(drop=True)
        alpha = float(selections.loc[site, "alpha"])
        scaler, model = base.fit_ridge(source, phase12.FEATURES, "fhd_normal", alpha)
        source_prediction = model.predict(
            scaler.transform(target[phase12.FEATURES].to_numpy(dtype=np.float64))
        )
        site_folds = sorted(
            [fold for fold in folds if fold["site_id"] == site],
            key=lambda fold: fold["held_out_pass"],
        )
        for fold in site_folds:
            local_index, test_index, _ = phase12.verify_fold(
                target, fold, float(split["freeze_basis"]["buffer_m"])
            )
            local = target.iloc[local_index]
            observed_local = local["fhd_normal"].to_numpy(dtype=np.float64)
            source_local = source_prediction[local_index]
            observed_test = target.iloc[test_index]["fhd_normal"].to_numpy(dtype=np.float64)
            source_test = source_prediction[test_index]

            for calibration_pass, pass_frame in local.groupby("pass_id", sort=True):
                if len(pass_frame) < int(protocol["one_pass_diagnostic"]["minimum_retained_calibration_rows"]):
                    continue
                positions = local.index.get_indexer(pass_frame.index)
                values = scenario_metrics(
                    observed_test,
                    source_test,
                    observed_local[positions],
                    source_local[positions],
                )
                for kind, metrics in values.items():
                    scenario_records.append(
                        {
                            "site_id": site,
                            "held_out_pass": fold["held_out_pass"],
                            "method": f"{'offset' if kind == 'offset' else 'local_mean'}_one_retained_pass",
                            "scenario_id": f"pass:{calibration_pass}",
                            "test_rows": len(test_index),
                            "calibration_rows": len(positions),
                            **metrics,
                        }
                    )

            local_passes = local["pass_id"].to_numpy(dtype=str)
            for budget in budgets:
                if budget > len(local) or budget < len(np.unique(local_passes)):
                    continue
                for repeat in range(repeats):
                    rng = np.random.default_rng(
                        stable_seed(site, fold["held_out_pass"], budget, repeat, base_seed=seed)
                    )
                    positions = stratified_sample_positions(local_passes, budget, rng)
                    values = scenario_metrics(
                        observed_test,
                        source_test,
                        observed_local[positions],
                        source_local[positions],
                    )
                    for kind, metrics in values.items():
                        scenario_records.append(
                            {
                                "site_id": site,
                                "held_out_pass": fold["held_out_pass"],
                                "method": f"{'offset' if kind == 'offset' else 'local_mean'}_{budget}_shots",
                                "scenario_id": f"repeat:{repeat:03d}",
                                "test_rows": len(test_index),
                                "calibration_rows": budget,
                                **metrics,
                            }
                        )
        print(f"[{site_index:02d}/{len(primary_sites)}] {site}: budget scenarios complete", flush=True)

    scenarios = pd.DataFrame(scenario_records)
    fold, site, macro = aggregate_scenarios(scenarios)
    site, macro = add_main_references(site, macro)
    main_fold = pd.read_csv(MAIN_FOLD_METRICS_PATH)
    bootstrap = bootstrap_comparisons(
        fold,
        main_fold,
        int(protocol["uncertainty"]["bootstrap_replicates"]),
        int(protocol["uncertainty"]["seed"]),
    )
    base.write_csv(scenarios, SCENARIOS_PATH)
    base.write_csv(fold, FOLD_PATH)
    base.write_csv(site, SITE_PATH)
    base.write_csv(macro, MACRO_PATH)
    base.write_csv(bootstrap, BOOTSTRAP_PATH)
    make_figure(macro)
    freeze_basis = {
        "protocol_sha256": base.sha256(PROTOCOL_PATH),
        "script_sha256": base.sha256(Path(__file__).resolve()),
        "phase12_result_freeze_id": result["freeze_id"],
        "split_freeze_id": split["freeze_id"],
        "primary_sites": primary_sites,
        "folds": len(folds),
        "one_pass_minimum_rows": int(protocol["one_pass_diagnostic"]["minimum_retained_calibration_rows"]),
        "shot_budgets": budgets,
        "replicates_per_fold": repeats,
        "seed": seed,
        "role": "post_result_diagnostic_not_primary_gate",
        "method_fold_coverage": {
            str(method): int(count)
            for method, count in fold.groupby("method").size().items()
        },
        "figure_excludes_incomplete_250_shot_sensitivity": True,
    }
    freeze = {
        "freeze_id": "phase12-track-budget-" + base.canonical_hash(freeze_basis)[:12],
        "created_utc": base.utc_now(),
        "status": "complete_track_assistance_budget_diagnostic",
        "freeze_basis": freeze_basis,
        "outputs": {
            str(path.relative_to(ROOT)): base.sha256(path)
            for path in [SCENARIOS_PATH, FOLD_PATH, SITE_PATH, MACRO_PATH, BOOTSTRAP_PATH, FIGURE_PATH]
        },
    }
    base.write_json(FREEZE_PATH, freeze)
    print(
        json.dumps(
            {
                "freeze_id": freeze["freeze_id"],
                "offset_macro": macro[macro["method"].str.startswith("offset_")][
                    ["method", "sites", "rmse", "r2", "spearman_r"]
                ].to_dict(orient="records"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
