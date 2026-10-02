#!/usr/bin/env python3
"""Compare TESSERA with Sentinel-2 plus terrain on prospective target sites."""

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
import build_fair_baselines_conventional_predictors as phase8  # noqa: E402
import freeze_habitat_transfer_prospective_predictions as blind  # noqa: E402


PROTOCOL_PATH = ROOT / "metadata/project_config_phase9_prospective_conventional_protocol.yaml"
CONVENTIONAL_FREEZE_PATH = ROOT / "metadata/phase9_prospective_conventional_freeze.json"
PREDICTION_FREEZE_PATH = ROOT / "metadata/phase9_prospective_prediction_freeze.json"
EVALUATION_FREEZE_PATH = ROOT / "metadata/phase9_prospective_evaluation_freeze.json"
SOURCE_CONVENTIONAL_PATH = ROOT / "data/processed/phase8_conventional_predictors_all_sites.parquet"
TARGET_CONVENTIONAL_PATH = ROOT / "data/processed/phase9_prospective_conventional_predictors.parquet"
TARGET_OUTCOME_PATH = ROOT / "data/processed/phase9_prospective_tessera_aligned.parquet"
TESSERA_EVALUATED_PATH = ROOT / "data/processed/phase9_prospective_evaluated_predictions.parquet"

PREDICTIONS_PATH = ROOT / "data/processed/phase9_prospective_conventional_comparison_predictions.parquet"
METRICS_PATH = ROOT / "outputs/tables/phase9_prospective_conventional_comparison_metrics.csv"
TUNING_PATH = ROOT / "outputs/tables/phase9_prospective_conventional_comparison_tuning.csv"
PAIRED_PATH = ROOT / "outputs/tables/phase9_prospective_conventional_comparison_paired.csv"
BOOTSTRAP_PATH = ROOT / "outputs/tables/phase9_prospective_conventional_comparison_bootstrap.csv"
FIGURE_PATH = ROOT / "outputs/figures/phase9_prospective_conventional_comparison.png"
FREEZE_PATH = ROOT / "metadata/phase9_prospective_conventional_comparison_freeze.json"

KEYS = blind.KEYS
CONVENTIONAL_FEATURES = phase8.TERRAIN_FEATURES + phase8.S2_FEATURES
METHODS = ["sentinel2_topography_ridge", "tessera_area_ridge"]
SOURCE_RULES = ["all_sources", "nearest5_evt_phys"]


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


def tune_alpha(
    training: pd.DataFrame,
    source_sites: list[str],
    target_site: str,
    source_rule: str,
    alphas: list[float],
) -> tuple[float, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    for alpha in alphas:
        for validation_site in source_sites:
            fit = training[~training["site_id"].eq(validation_site)]
            validation = training[training["site_id"].eq(validation_site)]
            weights = loso.equal_site_weights(fit["site_id"].to_numpy())
            scaler, model = loso.fit_weighted_ridge(
                fit[CONVENTIONAL_FEATURES].to_numpy(dtype=np.float64),
                fit["fhd_normal"].to_numpy(dtype=np.float64),
                alpha,
                weights,
            )
            predicted = model.predict(
                scaler.transform(validation[CONVENTIONAL_FEATURES].to_numpy(dtype=np.float64))
            )
            records.append({
                "target_site": target_site,
                "source_rule": source_rule,
                "method": "sentinel2_topography_ridge",
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


def build_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for keys, frame in predictions.groupby(["site_id", "source_rule", "method"], sort=True):
        site, source_rule, method = keys
        records.append({
            "scope": "site",
            "site_id": site,
            "source_rule": source_rule,
            "method": method,
            "rows": len(frame),
            **loso.metric_values(
                frame["fhd_normal"].to_numpy(dtype=np.float64),
                frame["prediction"].to_numpy(dtype=np.float64),
            ),
        })
    site = pd.DataFrame(records)
    for (source_rule, method), frame in site.groupby(["source_rule", "method"], sort=True):
        records.append({
            "scope": "macro",
            "site_id": "macro_mean",
            "source_rule": source_rule,
            "method": method,
            "rows": len(frame),
            **{metric: float(frame[metric].mean()) for metric in loso.METRICS},
        })
    for (source_rule, method), frame in predictions.groupby(["source_rule", "method"], sort=True):
        records.append({
            "scope": "pooled",
            "site_id": "all_sites",
            "source_rule": source_rule,
            "method": method,
            "rows": len(frame),
            **loso.metric_values(
                frame["fhd_normal"].to_numpy(dtype=np.float64),
                frame["prediction"].to_numpy(dtype=np.float64),
            ),
        })
    return pd.DataFrame(records).sort_values(
        ["scope", "site_id", "source_rule", "method"]
    ).reset_index(drop=True)


def paired_comparisons(
    metrics: pd.DataFrame,
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    site = metrics[metrics["scope"].eq("site")]
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    draws_output: list[pd.DataFrame] = []
    for source_rule in SOURCE_RULES:
        scoped = site[site["source_rule"].eq(source_rule)]
        sentinel = scoped[scoped["method"].eq("sentinel2_topography_ridge")].set_index(
            "site_id"
        ).sort_index()
        tessera = scoped[scoped["method"].eq("tessera_area_ridge")].set_index(
            "site_id"
        ).loc[sentinel.index]
        delta = (tessera["rmse"] - sentinel["rmse"]).to_numpy(dtype=np.float64)
        draws = np.mean(
            delta[rng.integers(0, len(delta), size=(replicates, len(delta)))], axis=1
        )
        records.append({
            "source_rule": source_rule,
            "contrast": "tessera_minus_sentinel2_topography",
            "site_count": len(delta),
            "sites_tessera_lower_rmse": int(np.sum(delta < 0)),
            "mean_delta_rmse": float(np.mean(delta)),
            "median_delta_rmse": float(np.median(delta)),
            "bootstrap_95_low": float(np.quantile(draws, 0.025)),
            "bootstrap_95_high": float(np.quantile(draws, 0.975)),
            "mean_delta_r2": float((tessera["r2"] - sentinel["r2"]).mean()),
            "mean_delta_spearman_r": float(
                (tessera["spearman_r"] - sentinel["spearman_r"]).mean()
            ),
        })
        draws_output.append(pd.DataFrame({
            "source_rule": source_rule,
            "replicate": np.arange(replicates, dtype=np.int64),
            "mean_delta_rmse": draws,
        }))
    return pd.DataFrame(records), pd.concat(draws_output, ignore_index=True)


def save_figure(metrics: pd.DataFrame, paired: pd.DataFrame) -> None:
    site = metrics[metrics["scope"].eq("site")]
    sites = sorted(site["site_id"].unique())
    x = np.arange(len(sites))
    width = 0.36
    colours = {
        "sentinel2_topography_ridge": "#3a86ff",
        "tessera_area_ridge": "#0a9396",
    }
    labels = {
        "sentinel2_topography_ridge": "Sentinel-2 + terrain",
        "tessera_area_ridge": "TESSERA",
    }
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.7), gridspec_kw={"width_ratios": [1.2, 1.2, 0.9]})
    for axis, source_rule, title in zip(
        axes[:2], SOURCE_RULES, ["All eight sources", "Five physiognomy-nearest sources"], strict=True
    ):
        scoped = site[site["source_rule"].eq(source_rule)]
        for index, method in enumerate(METHODS):
            frame = scoped[scoped["method"].eq(method)].set_index("site_id").loc[sites]
            axis.bar(
                x + (index - 0.5) * width,
                frame["rmse"],
                width,
                color=colours[method],
                label=labels[method],
            )
        axis.set_xticks(x, sites, rotation=45, ha="right")
        axis.set_ylabel("RMSE")
        axis.set_xlabel("Prospective target site")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.42, 0.99), ncol=2, frameon=False)

    ordered = paired.set_index("source_rule").loc[SOURCE_RULES]
    names = ["All eight", "Physiognomy-nearest"]
    axes[2].bar(names, ordered["mean_delta_rmse"], color="#6c757d")
    axes[2].errorbar(
        names,
        ordered["mean_delta_rmse"],
        yerr=np.vstack([
            ordered["mean_delta_rmse"] - ordered["bootstrap_95_low"],
            ordered["bootstrap_95_high"] - ordered["mean_delta_rmse"],
        ]),
        fmt="none",
        color="black",
        capsize=3,
    )
    axes[2].axhline(0, color="black", linewidth=1)
    axes[2].set_ylabel("TESSERA minus Sentinel-2 RMSE")
    axes[2].set_xlabel("Source rule")
    axes[2].tick_params(axis="x", rotation=20)
    axes[2].grid(axis="y", alpha=0.2)
    figure.suptitle("Post-hoc prospective comparison with a conventional baseline", y=1.04)
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = FIGURE_PATH.with_suffix(".tmp.png")
    figure.savefig(temporary, dpi=200, bbox_inches="tight")
    plt.close(figure)
    temporary.replace(FIGURE_PATH)


def main() -> int:
    protected = [
        PREDICTIONS_PATH, METRICS_PATH, TUNING_PATH, PAIRED_PATH,
        BOOTSTRAP_PATH, FIGURE_PATH, FREEZE_PATH,
    ]
    existing = [str(path.relative_to(ROOT)) for path in protected if path.exists()]
    if existing:
        raise RuntimeError(f"Prospective conventional comparison already exists: {existing}")
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))[
        "phase9_prospective_conventional_comparison"
    ]
    conventional_freeze = json.loads(CONVENTIONAL_FREEZE_PATH.read_text(encoding="utf-8"))
    prediction_freeze = json.loads(PREDICTION_FREEZE_PATH.read_text(encoding="utf-8"))
    evaluation_freeze = json.loads(EVALUATION_FREEZE_PATH.read_text(encoding="utf-8"))
    source_sites = sorted(protocol["source_sites"])
    target_sites = sorted(protocol["target_sites"])

    source_outcomes = pd.concat(
        [pd.read_parquet(path, columns=KEYS + ["fhd_normal"]) for path in blind.SOURCE_PATHS],
        ignore_index=True,
    )
    source_conventional = pd.read_parquet(
        SOURCE_CONVENTIONAL_PATH, columns=KEYS + CONVENTIONAL_FEATURES
    )
    sources = source_outcomes.merge(source_conventional, on=KEYS, validate="one_to_one")
    target_conventional = pd.read_parquet(
        TARGET_CONVENTIONAL_PATH, columns=KEYS + CONVENTIONAL_FEATURES
    )
    target_outcomes = pd.read_parquet(TARGET_OUTCOME_PATH, columns=KEYS + ["fhd_normal"])
    targets = target_outcomes.merge(target_conventional, on=KEYS, validate="one_to_one")
    if sorted(sources["site_id"].unique()) != source_sites:
        raise RuntimeError("Conventional comparison source population changed")
    if sorted(targets["site_id"].unique()) != target_sites:
        raise RuntimeError("Conventional comparison target population changed")
    if not np.isfinite(sources[["fhd_normal", *CONVENTIONAL_FEATURES]].to_numpy()).all():
        raise RuntimeError("Non-finite source conventional values")
    if not np.isfinite(targets[["fhd_normal", *CONVENTIONAL_FEATURES]].to_numpy()).all():
        raise RuntimeError("Non-finite target conventional values")

    alphas = [float(value) for value in protocol["models"]["alpha_grid"]]
    prediction_frames: list[pd.DataFrame] = []
    tuning_frames: list[pd.DataFrame] = []
    for target_site in target_sites:
        target = targets[targets["site_id"].eq(target_site)]
        for source_rule in SOURCE_RULES:
            selected_sources = source_sites if source_rule == "all_sources" else sorted(
                prediction_freeze["freeze_basis"]["source_selection"][
                    f"{target_site}/nearest5_evt_phys"
                ]
            )
            training = sources[sources["site_id"].isin(selected_sources)]
            alpha, tuning = tune_alpha(
                training, selected_sources, target_site, source_rule, alphas
            )
            tuning_frames.append(tuning)
            weights = loso.equal_site_weights(training["site_id"].to_numpy())
            scaler, model = loso.fit_weighted_ridge(
                training[CONVENTIONAL_FEATURES].to_numpy(dtype=np.float64),
                training["fhd_normal"].to_numpy(dtype=np.float64),
                alpha,
                weights,
            )
            frame = target[KEYS + ["fhd_normal"]].copy()
            frame["source_rule"] = source_rule
            frame["method"] = "sentinel2_topography_ridge"
            frame["training_sites"] = "+".join(selected_sources)
            frame["selected_alpha"] = alpha
            frame["prediction"] = model.predict(
                scaler.transform(target[CONVENTIONAL_FEATURES].to_numpy(dtype=np.float64))
            )
            prediction_frames.append(frame)
            print(f"{target_site} {source_rule}: Sentinel alpha={alpha:g}", flush=True)

    sentinel_predictions = pd.concat(prediction_frames, ignore_index=True)
    tessera = pd.read_parquet(TESSERA_EVALUATED_PATH)
    tessera = tessera[tessera["strategy"].isin(SOURCE_RULES)].rename(
        columns={"strategy": "source_rule"}
    )
    tessera = tessera[
        KEYS + ["fhd_normal", "source_rule", "method", "training_sites", "selected_alpha", "prediction"]
    ]
    if set(tessera["method"]) != {"tessera_area_ridge"}:
        raise RuntimeError("Frozen TESSERA comparator changed")
    predictions = pd.concat([sentinel_predictions, tessera], ignore_index=True).sort_values(
        ["site_id", "source_rule", "method", "shot_number"]
    )
    metrics = build_metrics(predictions)
    evaluation = protocol["evaluation"]
    paired, bootstrap = paired_comparisons(
        metrics,
        int(evaluation["bootstrap_replicates"]),
        int(evaluation["random_seed"]),
    )

    write_parquet(predictions, PREDICTIONS_PATH)
    write_csv(metrics, METRICS_PATH)
    write_csv(pd.concat(tuning_frames, ignore_index=True), TUNING_PATH)
    write_csv(paired, PAIRED_PATH)
    write_csv(bootstrap, BOOTSTRAP_PATH)
    save_figure(metrics, paired)
    outputs = {
        "predictions": {"path": str(PREDICTIONS_PATH.relative_to(ROOT)), "rows": len(predictions), "sha256": sha256(PREDICTIONS_PATH)},
        "metrics": {"path": str(METRICS_PATH.relative_to(ROOT)), "rows": len(metrics), "sha256": sha256(METRICS_PATH)},
        "tuning": {"path": str(TUNING_PATH.relative_to(ROOT)), "rows": sum(len(frame) for frame in tuning_frames), "sha256": sha256(TUNING_PATH)},
        "paired": {"path": str(PAIRED_PATH.relative_to(ROOT)), "rows": len(paired), "sha256": sha256(PAIRED_PATH)},
        "bootstrap": {"path": str(BOOTSTRAP_PATH.relative_to(ROOT)), "rows": len(bootstrap), "sha256": sha256(BOOTSTRAP_PATH)},
        "figure": {"path": str(FIGURE_PATH.relative_to(ROOT)), "sha256": sha256(FIGURE_PATH)},
    }
    macro = metrics[metrics["scope"].eq("macro")].set_index(["source_rule", "method"])
    freeze_basis = {
        "analysis_timing": protocol["analysis_timing"],
        "confirmatory": False,
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "conventional_freeze_id": conventional_freeze["freeze_id"],
        "prediction_id": prediction_freeze["prediction_id"],
        "prospective_evaluation_id": evaluation_freeze["evaluation_id"],
        "analysis_script_sha256": sha256(Path(__file__).resolve()),
        "source_sites": source_sites,
        "target_sites": target_sites,
        "methods": METHODS,
        "source_rules": SOURCE_RULES,
        "conventional_features": CONVENTIONAL_FEATURES,
        "observation_counts_used_as_features": False,
        "bootstrap_unit": evaluation["bootstrap_unit"],
        "bootstrap_replicates": int(evaluation["bootstrap_replicates"]),
        "random_seed": int(evaluation["random_seed"]),
        "macro_metrics": {
            f"{rule}/{method}": {
                metric: float(macro.loc[(rule, method), metric])
                for metric in ["rmse", "r2", "mae", "spearman_r", "mean_bias"]
            }
            for rule in SOURCE_RULES
            for method in METHODS
        },
        "paired_results": paired.to_dict(orient="records"),
        "interpretation_limit": "post_hoc_baseline_after_target_outcomes_opened",
    }
    freeze_id = "phase9-prospective-conventional-comparison-" + canonical_hash(freeze_basis)[:12]
    write_json(FREEZE_PATH, {
        "freeze_id": freeze_id,
        "created_utc": utc_now(),
        "status": "frozen_complete_post_hoc",
        "freeze_basis": freeze_basis,
        "outputs": outputs,
    })
    print(json.dumps({
        "freeze_id": freeze_id,
        "macro_metrics": freeze_basis["macro_metrics"],
        "paired_results": freeze_basis["paired_results"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
