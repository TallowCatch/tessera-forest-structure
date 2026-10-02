#!/usr/bin/env python3
"""Compare frozen habitat source sets with every five-of-eight source subset."""

from __future__ import annotations

import hashlib
import itertools
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
import freeze_habitat_transfer_prospective_predictions as blind  # noqa: E402


PROTOCOL_PATH = ROOT / "metadata/project_config_phase9_matched_size_control_protocol.yaml"
PREDICTION_FREEZE_PATH = ROOT / "metadata/phase9_prospective_prediction_freeze.json"
EVALUATION_FREEZE_PATH = ROOT / "metadata/phase9_prospective_evaluation_freeze.json"
TARGET_PATH = ROOT / "data/processed/phase9_prospective_tessera_aligned.parquet"

PREDICTIONS_PATH = ROOT / "data/processed/phase9_matched_size_source_subset_predictions.parquet"
METRICS_PATH = ROOT / "outputs/tables/phase9_matched_size_source_subset_metrics.csv"
TUNING_PATH = ROOT / "outputs/tables/phase9_matched_size_source_subset_tuning.csv"
SUMMARY_PATH = ROOT / "outputs/tables/phase9_matched_size_source_control_summary.csv"
NULL_PATH = ROOT / "outputs/tables/phase9_matched_size_source_control_null.csv"
FIGURE_PATH = ROOT / "outputs/figures/phase9_matched_size_source_control.png"
FREEZE_PATH = ROOT / "metadata/phase9_matched_size_source_control_freeze.json"

KEYS = blind.KEYS
FEATURES = blind.FEATURES


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


def subset_id(sites: tuple[str, ...] | list[str]) -> str:
    return "+".join(sorted(sites))


def enumerate_subsets(source_sites: list[str], size: int) -> list[tuple[str, ...]]:
    return list(itertools.combinations(sorted(source_sites), size))


def precompute_inner_tuning(
    sources: pd.DataFrame,
    source_sites: list[str],
    alphas: list[float],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    four_site_combinations = list(itertools.combinations(source_sites, 4))
    for position, fit_sites in enumerate(four_site_combinations, start=1):
        fit = sources[sources["site_id"].isin(fit_sites)]
        weights = loso.equal_site_weights(fit["site_id"].to_numpy())
        validation_sites = sorted(set(source_sites) - set(fit_sites))
        for alpha in alphas:
            scaler, model = loso.fit_weighted_ridge(
                fit[FEATURES].to_numpy(dtype=np.float64),
                fit["fhd_normal"].to_numpy(dtype=np.float64),
                alpha,
                weights,
            )
            for validation_site in validation_sites:
                validation = sources[sources["site_id"].eq(validation_site)]
                predicted = model.predict(
                    scaler.transform(validation[FEATURES].to_numpy(dtype=np.float64))
                )
                source_subset = tuple(sorted((*fit_sites, validation_site)))
                records.append({
                    "source_subset": subset_id(source_subset),
                    "fit_sites": subset_id(fit_sites),
                    "validation_site": validation_site,
                    "alpha": alpha,
                    "training_rows": len(fit),
                    "validation_rows": len(validation),
                    **loso.metric_values(
                        validation["fhd_normal"].to_numpy(dtype=np.float64),
                        predicted,
                    ),
                })
        if position % 10 == 0 or position == len(four_site_combinations):
            print(
                f"inner tuning fits {position}/{len(four_site_combinations)} four-site combinations",
                flush=True,
            )
    return pd.DataFrame(records)


def select_subset_alphas(tuning: pd.DataFrame) -> dict[str, float]:
    selected: dict[str, float] = {}
    for source_subset, frame in tuning.groupby("source_subset", sort=True):
        alpha, _ = loso.select_alpha(frame)
        selected[source_subset] = alpha
    return selected


def build_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (site, source_subset), frame in predictions.groupby(
        ["site_id", "source_subset"], sort=True
    ):
        records.append({
            "site_id": site,
            "source_subset": source_subset,
            "is_frozen_phys_subset": bool(frame["is_frozen_phys_subset"].iloc[0]),
            "selected_alpha": float(frame["selected_alpha"].iloc[0]),
            "rows": len(frame),
            **loso.metric_values(
                frame["fhd_normal"].to_numpy(dtype=np.float64),
                frame["prediction"].to_numpy(dtype=np.float64),
            ),
        })
    return pd.DataFrame(records)


def summarize_controls(
    metrics: pd.DataFrame,
    all_source_metrics: pd.DataFrame,
    frozen_map: dict[str, str],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for site in sorted(frozen_map):
        frame = metrics[metrics["site_id"].eq(site)].sort_values(
            ["rmse", "source_subset"]
        ).reset_index(drop=True)
        frozen = frame[frame["source_subset"].eq(frozen_map[site])].iloc[0]
        all_rmse = float(
            all_source_metrics[
                all_source_metrics["scope"].eq("site")
                & all_source_metrics["site_id"].eq(site)
                & all_source_metrics["strategy"].eq("all_sources")
            ]["rmse"].iloc[0]
        )
        rank = int(frame.index[frame["source_subset"].eq(frozen_map[site])][0]) + 1
        records.append({
            "scope": "site",
            "site_id": site,
            "subset_count": len(frame),
            "frozen_phys_subset": frozen_map[site],
            "frozen_phys_rank_lower_is_better": rank,
            "fraction_subsets_rmse_le_frozen": float(np.mean(frame["rmse"].le(frozen["rmse"]))),
            "frozen_phys_rmse": float(frozen["rmse"]),
            "median_five_source_rmse": float(frame["rmse"].median()),
            "best_five_source_rmse": float(frame["rmse"].min()),
            "worst_five_source_rmse": float(frame["rmse"].max()),
            "frozen_minus_median_five_rmse": float(frozen["rmse"] - frame["rmse"].median()),
            "all_eight_source_rmse": all_rmse,
            "frozen_minus_all_eight_rmse": float(frozen["rmse"] - all_rmse),
        })
    site_summary = pd.DataFrame(records)
    records.append({
        "scope": "macro",
        "site_id": "macro_mean",
        "subset_count": 56,
        "frozen_phys_subset": "target_specific",
        "frozen_phys_rank_lower_is_better": float(site_summary["frozen_phys_rank_lower_is_better"].mean()),
        "fraction_subsets_rmse_le_frozen": float(site_summary["fraction_subsets_rmse_le_frozen"].mean()),
        "frozen_phys_rmse": float(site_summary["frozen_phys_rmse"].mean()),
        "median_five_source_rmse": float(site_summary["median_five_source_rmse"].mean()),
        "best_five_source_rmse": float(site_summary["best_five_source_rmse"].mean()),
        "worst_five_source_rmse": float(site_summary["worst_five_source_rmse"].mean()),
        "frozen_minus_median_five_rmse": float(
            site_summary["frozen_minus_median_five_rmse"].mean()
        ),
        "all_eight_source_rmse": float(site_summary["all_eight_source_rmse"].mean()),
        "frozen_minus_all_eight_rmse": float(
            site_summary["frozen_minus_all_eight_rmse"].mean()
        ),
    })
    return pd.DataFrame(records)


def random_subset_null(
    metrics: pd.DataFrame,
    target_sites: list[str],
    frozen_macro_rmse: float,
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    matrix = np.vstack([
        metrics[metrics["site_id"].eq(site)].sort_values("source_subset")["rmse"].to_numpy()
        for site in target_sites
    ])
    rng = np.random.default_rng(seed)
    choices = rng.integers(0, matrix.shape[1], size=(replicates, len(target_sites)))
    draws = np.mean(
        matrix[np.arange(len(target_sites))[None, :], choices],
        axis=1,
    )
    p_value = float((np.sum(draws <= frozen_macro_rmse) + 1) / (replicates + 1))
    output = pd.DataFrame({
        "replicate": np.arange(replicates, dtype=np.int64),
        "random_five_source_macro_rmse": draws,
    })
    summary = {
        "replicates": replicates,
        "random_seed": seed,
        "frozen_phys_macro_rmse": frozen_macro_rmse,
        "random_macro_rmse_mean": float(np.mean(draws)),
        "random_macro_rmse_median": float(np.median(draws)),
        "random_macro_rmse_2_5_percentile": float(np.quantile(draws, 0.025)),
        "random_macro_rmse_97_5_percentile": float(np.quantile(draws, 0.975)),
        "one_sided_monte_carlo_p": p_value,
    }
    return output, summary


def save_figure(metrics: pd.DataFrame, summary: pd.DataFrame, null: pd.DataFrame) -> None:
    site_summary = summary[summary["scope"].eq("site")].sort_values("site_id")
    sites = site_summary["site_id"].tolist()
    data = [metrics[metrics["site_id"].eq(site)]["rmse"].to_numpy() for site in sites]
    x = np.arange(1, len(sites) + 1)
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.7), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    boxes = axes[0].boxplot(data, positions=x, widths=0.58, patch_artist=True, showfliers=False)
    for box in boxes["boxes"]:
        box.set_facecolor("#d9e2e8")
        box.set_edgecolor("#5c6770")
    axes[0].scatter(x, site_summary["frozen_phys_rmse"], color="#0a9396", marker="D", zorder=3, label="Frozen physiognomy subset")
    axes[0].set_xticks(x, sites, rotation=45, ha="right")
    axes[0].set_ylabel("RMSE across five-source subsets")
    axes[0].set_xlabel("Prospective target site")
    axes[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].bar(
        sites,
        site_summary["fraction_subsets_rmse_le_frozen"],
        color="#457b9d",
    )
    axes[1].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[1].set_xticks(range(len(sites)), sites, rotation=45, ha="right")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Fraction of subsets at least as good")
    axes[1].set_xlabel("Prospective target site")
    axes[1].grid(axis="y", alpha=0.2)

    values = null["random_five_source_macro_rmse"]
    frozen = float(summary[summary["scope"].eq("macro")]["frozen_phys_rmse"].iloc[0])
    axes[2].hist(values, bins=45, color="#adb5bd", edgecolor="white")
    axes[2].axvline(frozen, color="#0a9396", linewidth=2, label="Frozen physiognomy rule")
    axes[2].set_ylabel("Monte Carlo replicates")
    axes[2].set_xlabel("Six-site macro RMSE")
    axes[2].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    axes[2].grid(axis="y", alpha=0.2)
    figure.suptitle("Post-hoc matched-size control for habitat source selection", y=1.02)
    figure.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = FIGURE_PATH.with_suffix(".tmp.png")
    figure.savefig(temporary, dpi=200, bbox_inches="tight")
    plt.close(figure)
    temporary.replace(FIGURE_PATH)


def main() -> int:
    protected = [
        PREDICTIONS_PATH,
        METRICS_PATH,
        TUNING_PATH,
        SUMMARY_PATH,
        NULL_PATH,
        FIGURE_PATH,
        FREEZE_PATH,
    ]
    existing = [str(path.relative_to(ROOT)) for path in protected if path.exists()]
    if existing:
        raise RuntimeError(f"Matched-size control already exists: {existing}")
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))[
        "phase9_matched_size_source_control"
    ]
    prediction_freeze = json.loads(PREDICTION_FREEZE_PATH.read_text(encoding="utf-8"))
    evaluation_freeze = json.loads(EVALUATION_FREEZE_PATH.read_text(encoding="utf-8"))
    source_sites = sorted(protocol["source_sites"])
    target_sites = sorted(protocol["target_sites"])
    subsets = enumerate_subsets(source_sites, int(protocol["source_subset_size"]))
    if len(subsets) != int(protocol["expected_subset_count"]):
        raise RuntimeError("Five-source subset count differs from protocol")

    sources = pd.concat(
        [pd.read_parquet(path, columns=KEYS + ["fhd_normal"] + FEATURES) for path in blind.SOURCE_PATHS],
        ignore_index=True,
    )
    targets = pd.read_parquet(TARGET_PATH, columns=KEYS + ["fhd_normal"] + FEATURES)
    if sorted(sources["site_id"].unique()) != source_sites:
        raise RuntimeError("Matched-size control source population changed")
    if sorted(targets["site_id"].unique()) != target_sites:
        raise RuntimeError("Matched-size control target population changed")

    alphas = [float(value) for value in protocol["alpha_grid"]]
    tuning = precompute_inner_tuning(sources, source_sites, alphas)
    selected_alphas = select_subset_alphas(tuning)
    if len(selected_alphas) != len(subsets):
        raise RuntimeError("Not every source subset received a selected alpha")

    frozen_map = {
        site: subset_id(
            prediction_freeze["freeze_basis"]["source_selection"][
                f"{site}/nearest5_evt_phys"
            ]
        )
        for site in target_sites
    }
    prediction_frames: list[pd.DataFrame] = []
    target_x = targets[FEATURES].to_numpy(dtype=np.float64)
    for position, subset in enumerate(subsets, start=1):
        identifier = subset_id(subset)
        training = sources[sources["site_id"].isin(subset)]
        weights = loso.equal_site_weights(training["site_id"].to_numpy())
        alpha = selected_alphas[identifier]
        scaler, model = loso.fit_weighted_ridge(
            training[FEATURES].to_numpy(dtype=np.float64),
            training["fhd_normal"].to_numpy(dtype=np.float64),
            alpha,
            weights,
        )
        frame = targets[KEYS + ["fhd_normal"]].copy()
        frame["source_subset"] = identifier
        frame["selected_alpha"] = alpha
        frame["prediction"] = model.predict(scaler.transform(target_x))
        frame["is_frozen_phys_subset"] = [
            identifier == frozen_map[site] for site in frame["site_id"]
        ]
        prediction_frames.append(frame)
        if position % 8 == 0 or position == len(subsets):
            print(f"final subset models {position}/{len(subsets)}", flush=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)

    blind_predictions = pd.read_parquet(blind.PREDICTIONS_PATH)
    blind_phys = blind_predictions[
        blind_predictions["strategy"].eq("nearest5_evt_phys")
    ][KEYS + ["prediction"]].rename(columns={"prediction": "frozen_prediction"})
    reproduced = predictions[predictions["is_frozen_phys_subset"]][
        KEYS + ["prediction"]
    ].merge(blind_phys, on=KEYS, validate="one_to_one")
    reproduction_delta = float(
        np.max(np.abs(reproduced["prediction"] - reproduced["frozen_prediction"]))
    )
    if reproduction_delta > 1e-10:
        raise RuntimeError(f"Frozen physiognomy predictions were not reproduced: {reproduction_delta}")

    metrics = build_metrics(predictions)
    all_source_metrics = pd.read_csv(
        ROOT / evaluation_freeze["outputs"]["metrics"]["path"]
    )
    summary = summarize_controls(metrics, all_source_metrics, frozen_map)
    random_config = protocol["random_subset_null"]
    null, null_summary = random_subset_null(
        metrics,
        target_sites,
        float(summary[summary["scope"].eq("macro")]["frozen_phys_rmse"].iloc[0]),
        int(random_config["replicates"]),
        int(random_config["random_seed"]),
    )

    write_parquet(predictions, PREDICTIONS_PATH)
    write_csv(metrics, METRICS_PATH)
    write_csv(tuning, TUNING_PATH)
    write_csv(summary, SUMMARY_PATH)
    write_csv(null, NULL_PATH)
    save_figure(metrics, summary, null)
    outputs = {
        "predictions": {"path": str(PREDICTIONS_PATH.relative_to(ROOT)), "rows": len(predictions), "sha256": sha256(PREDICTIONS_PATH)},
        "metrics": {"path": str(METRICS_PATH.relative_to(ROOT)), "rows": len(metrics), "sha256": sha256(METRICS_PATH)},
        "tuning": {"path": str(TUNING_PATH.relative_to(ROOT)), "rows": len(tuning), "sha256": sha256(TUNING_PATH)},
        "summary": {"path": str(SUMMARY_PATH.relative_to(ROOT)), "rows": len(summary), "sha256": sha256(SUMMARY_PATH)},
        "random_subset_null": {"path": str(NULL_PATH.relative_to(ROOT)), "rows": len(null), "sha256": sha256(NULL_PATH)},
        "figure": {"path": str(FIGURE_PATH.relative_to(ROOT)), "sha256": sha256(FIGURE_PATH)},
    }
    site_summary = summary[summary["scope"].eq("site")]
    freeze_basis = {
        "analysis_timing": protocol["analysis_timing"],
        "confirmatory": False,
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "prospective_evaluation_id": evaluation_freeze["evaluation_id"],
        "prediction_id": prediction_freeze["prediction_id"],
        "analysis_script_sha256": sha256(Path(__file__).resolve()),
        "source_sites": source_sites,
        "target_sites": target_sites,
        "source_subset_size": 5,
        "source_subset_count": len(subsets),
        "frozen_prediction_max_absolute_delta": reproduction_delta,
        "sites_frozen_better_than_median_five_subset": int(
            site_summary["frozen_minus_median_five_rmse"].lt(0).sum()
        ),
        "mean_frozen_minus_median_five_rmse": float(
            site_summary["frozen_minus_median_five_rmse"].mean()
        ),
        "mean_frozen_subset_rank": float(
            site_summary["frozen_phys_rank_lower_is_better"].mean()
        ),
        "random_subset_null": null_summary,
        "interpretation_limit": "post_hoc_mechanism_control_after_target_outcomes_opened",
    }
    freeze_id = "phase9-matched-size-control-" + canonical_hash(freeze_basis)[:12]
    write_json(FREEZE_PATH, {
        "freeze_id": freeze_id,
        "created_utc": utc_now(),
        "status": "frozen_complete_post_hoc",
        "freeze_basis": freeze_basis,
        "outputs": outputs,
    })
    print(json.dumps({
        "freeze_id": freeze_id,
        "site_summary": site_summary.to_dict(orient="records"),
        "null_summary": null_summary,
        "sites_frozen_better_than_median_five_subset": freeze_basis[
            "sites_frozen_better_than_median_five_subset"
        ],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
