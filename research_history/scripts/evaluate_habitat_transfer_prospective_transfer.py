#!/usr/bin/env python3
"""Lock, open, and evaluate the prospective habitat-conditioned transfer once."""

from __future__ import annotations

import argparse
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
import yaml
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_multisite_loso as loso  # noqa: E402


PROTOCOL_PATH = ROOT / "metadata/project_config_phase9_prospective_protocol_freeze.yaml"
PREDICTION_FREEZE_PATH = ROOT / "metadata/phase9_prospective_prediction_freeze.json"
TARGET_PATH = ROOT / "data/processed/phase9_prospective_tessera_aligned.parquet"
PREDICTIONS_PATH = ROOT / "data/processed/phase9_prospective_predictions_blind.parquet"
EXECUTION_LOCK_PATH = ROOT / "metadata/phase9_prospective_evaluation_execution_lock.json"
OPEN_EVENT_PATH = ROOT / "metadata/phase9_prospective_outcome_open_event.json"

EVALUATED_PATH = ROOT / "data/processed/phase9_prospective_evaluated_predictions.parquet"
METRICS_PATH = ROOT / "outputs/tables/phase9_prospective_evaluation_metrics.csv"
PAIRED_PATH = ROOT / "outputs/tables/phase9_prospective_evaluation_paired.csv"
BOOTSTRAP_PATH = ROOT / "outputs/tables/phase9_prospective_evaluation_site_bootstrap.csv"
FIGURE_PATH = ROOT / "outputs/figures/phase9_prospective_habitat_evaluation.png"
FREEZE_PATH = ROOT / "metadata/phase9_prospective_evaluation_freeze.json"

KEYS = ["site_id", "shot_number"]
STRATEGIES = ["all_sources", "nearest5_evt_phys", "nearest5_evt_group"]
PRIMARY_STRATEGY = "nearest5_evt_phys"
SECONDARY_STRATEGY = "nearest5_evt_group"
METRICS = ["rmse", "r2", "mae", "spearman_r", "mean_bias"]


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


def validate_outputs(freeze: dict[str, Any]) -> None:
    for output in freeze["outputs"].values():
        path = ROOT / output["path"]
        if not path.exists() or sha256(path) != output["sha256"]:
            raise RuntimeError(f"Frozen input hash mismatch: {path}")


def metric_table(evaluated: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (site, strategy), frame in evaluated.groupby(["site_id", "strategy"], sort=True):
        records.append({
            "scope": "site",
            "site_id": site,
            "strategy": strategy,
            "rows": len(frame),
            **loso.metric_values(
                frame["fhd_normal"].to_numpy(dtype=np.float64),
                frame["prediction"].to_numpy(dtype=np.float64),
            ),
        })
    site = pd.DataFrame(records)
    for strategy, frame in site.groupby("strategy", sort=True):
        records.append({
            "scope": "macro",
            "site_id": "macro_mean",
            "strategy": strategy,
            "rows": int(len(frame)),
            **{metric: float(frame[metric].mean()) for metric in loso.METRICS},
        })
    for strategy, frame in evaluated.groupby("strategy", sort=True):
        records.append({
            "scope": "pooled",
            "site_id": "all_sites",
            "strategy": strategy,
            "rows": len(frame),
            **loso.metric_values(
                frame["fhd_normal"].to_numpy(dtype=np.float64),
                frame["prediction"].to_numpy(dtype=np.float64),
            ),
        })
    return pd.DataFrame(records).sort_values(["scope", "site_id", "strategy"]).reset_index(drop=True)


def paired_results(
    metrics: pd.DataFrame,
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    site = metrics[metrics["scope"].eq("site")]
    baseline = site[site["strategy"].eq("all_sources")].set_index("site_id").sort_index()
    rng = np.random.default_rng(seed)
    summaries: list[dict[str, Any]] = []
    bootstrap_frames: list[pd.DataFrame] = []
    for strategy in [PRIMARY_STRATEGY, SECONDARY_STRATEGY]:
        matched = site[site["strategy"].eq(strategy)].set_index("site_id").loc[baseline.index]
        delta = (matched["rmse"] - baseline["rmse"]).to_numpy(dtype=np.float64)
        draws = np.mean(
            delta[rng.integers(0, len(delta), size=(replicates, len(delta)))],
            axis=1,
        )
        improved = int(np.sum(delta < 0))
        summaries.append({
            "strategy": strategy,
            "site_count": len(delta),
            "sites_with_lower_rmse": improved,
            "mean_delta_rmse": float(np.mean(delta)),
            "median_delta_rmse": float(np.median(delta)),
            "bootstrap_95_low": float(np.quantile(draws, 0.025)),
            "bootstrap_95_high": float(np.quantile(draws, 0.975)),
            "two_sided_sign_test_p": float(binomtest(improved, len(delta), 0.5).pvalue),
            "mean_delta_r2": float((matched["r2"] - baseline["r2"]).mean()),
            "mean_delta_mae": float((matched["mae"] - baseline["mae"]).mean()),
            "mean_delta_spearman_r": float(
                (matched["spearman_r"] - baseline["spearman_r"]).mean()
            ),
            "mean_delta_mean_bias": float(
                (matched["mean_bias"] - baseline["mean_bias"]).mean()
            ),
        })
        bootstrap_frames.append(pd.DataFrame({
            "strategy": strategy,
            "replicate": np.arange(replicates, dtype=np.int64),
            "mean_delta_rmse": draws,
        }))
    return pd.DataFrame(summaries), pd.concat(bootstrap_frames, ignore_index=True)


def gate_result(paired: pd.DataFrame) -> dict[str, Any]:
    primary = paired[paired["strategy"].eq(PRIMARY_STRATEGY)].iloc[0]
    site_count = int(primary["site_count"])
    required_majority = math.floor(site_count / 2) + 1
    checks = {
        "mean_delta_rmse_below_zero": bool(primary["mean_delta_rmse"] < 0),
        "lower_rmse_on_majority_of_sites": bool(
            primary["sites_with_lower_rmse"] >= required_majority
        ),
        "site_bootstrap_95_upper_below_zero": bool(primary["bootstrap_95_high"] < 0),
    }
    return {
        "primary_strategy": PRIMARY_STRATEGY,
        "comparator": "all_sources",
        "site_count": site_count,
        "required_sites_for_majority": required_majority,
        "mean_delta_rmse": float(primary["mean_delta_rmse"]),
        "sites_with_lower_rmse": int(primary["sites_with_lower_rmse"]),
        "bootstrap_95_low": float(primary["bootstrap_95_low"]),
        "bootstrap_95_high": float(primary["bootstrap_95_high"]),
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def save_figure(metrics: pd.DataFrame, paired: pd.DataFrame) -> None:
    site = metrics[metrics["scope"].eq("site")]
    baseline = site[site["strategy"].eq("all_sources")].set_index("site_id").sort_index()
    primary = site[site["strategy"].eq(PRIMARY_STRATEGY)].set_index("site_id").loc[baseline.index]
    sites = baseline.index.tolist()
    x = np.arange(len(sites))
    width = 0.36
    delta = primary["rmse"] - baseline["rmse"]

    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.7), gridspec_kw={"width_ratios": [1.25, 1, 0.9]})
    axes[0].bar(x - width / 2, baseline["rmse"], width, color="#6c757d", label="All eight sources")
    axes[0].bar(x + width / 2, primary["rmse"], width, color="#0a9396", label="Five habitat-nearest")
    axes[0].set_xticks(x, sites, rotation=45, ha="right")
    axes[0].set_ylabel("RMSE")
    axes[0].set_xlabel("Prospective target site")
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].grid(axis="y", alpha=0.2)

    colours = np.where(delta.to_numpy() < 0, "#0a9396", "#d95f02")
    axes[1].bar(x, delta, color=colours)
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_xticks(x, sites, rotation=45, ha="right")
    axes[1].set_ylabel("Habitat-nearest minus all-source RMSE")
    axes[1].set_xlabel("Prospective target site")
    axes[1].grid(axis="y", alpha=0.2)

    order = [PRIMARY_STRATEGY, SECONDARY_STRATEGY]
    summary = paired.set_index("strategy").loc[order]
    labels = ["Physiognomy", "Vegetation group"]
    axes[2].bar(labels, summary["mean_delta_rmse"], color=["#0a9396", "#457b9d"])
    axes[2].errorbar(
        labels,
        summary["mean_delta_rmse"],
        yerr=np.vstack([
            summary["mean_delta_rmse"] - summary["bootstrap_95_low"],
            summary["bootstrap_95_high"] - summary["mean_delta_rmse"],
        ]),
        fmt="none",
        color="black",
        capsize=3,
    )
    axes[2].axhline(0, color="black", linewidth=1)
    axes[2].set_ylabel("Mean paired RMSE change")
    axes[2].set_xlabel("Source-matching rule")
    axes[2].grid(axis="y", alpha=0.2)

    figure.suptitle("Locked prospective habitat-conditioned transfer evaluation")
    figure.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = FIGURE_PATH.with_suffix(".tmp.png")
    figure.savefig(temporary, dpi=200, bbox_inches="tight")
    plt.close(figure)
    temporary.replace(FIGURE_PATH)


def lock_execution() -> int:
    protected = [
        EXECUTION_LOCK_PATH,
        OPEN_EVENT_PATH,
        EVALUATED_PATH,
        METRICS_PATH,
        PAIRED_PATH,
        BOOTSTRAP_PATH,
        FIGURE_PATH,
        FREEZE_PATH,
    ]
    existing = [str(path.relative_to(ROOT)) for path in protected if path.exists()]
    if existing:
        raise RuntimeError(f"Prospective evaluation already locked or started: {existing}")
    prediction_freeze = json.loads(PREDICTION_FREEZE_PATH.read_text(encoding="utf-8"))
    validate_outputs(prediction_freeze)
    if prediction_freeze["status"] != "frozen_before_target_evaluation":
        raise RuntimeError("Prospective predictions were not frozen before evaluation")
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))[
        "phase9_prospective_habitat_transfer"
    ]
    evaluation = protocol["evaluation"]
    basis = {
        "protocol_snapshot_sha256": sha256(PROTOCOL_PATH),
        "prediction_freeze_sha256": sha256(PREDICTION_FREEZE_PATH),
        "prediction_id": prediction_freeze["prediction_id"],
        "blind_predictions_sha256": sha256(PREDICTIONS_PATH),
        "target_table_sha256": sha256(TARGET_PATH),
        "evaluation_script_sha256": sha256(Path(__file__).resolve()),
        "target_columns_to_open": KEYS + ["fhd_normal"],
        "strategies": STRATEGIES,
        "primary_metric": evaluation["primary_metric"],
        "reported_metrics": evaluation["report"],
        "bootstrap_unit": evaluation["bootstrap_unit"],
        "bootstrap_replicates": int(evaluation["bootstrap_replicates"]),
        "random_seed": int(evaluation["random_seed"]),
        "success_gate": evaluation["success_gate"],
    }
    lock_id = "phase9-prospective-eval-lock-" + canonical_hash(basis)[:12]
    write_json(EXECUTION_LOCK_PATH, {
        "lock_id": lock_id,
        "created_utc": utc_now(),
        "status": "locked_before_target_outcome_open",
        "target_outcomes_opened": False,
        "freeze_basis": basis,
    })
    print(json.dumps({"lock_id": lock_id, "target_outcomes_opened": False}, indent=2))
    return 0


def evaluate() -> int:
    if FREEZE_PATH.exists():
        raise RuntimeError("Prospective evaluation freeze already exists; refusing to rerun")
    if not EXECUTION_LOCK_PATH.exists():
        raise RuntimeError("Run --lock before opening prospective outcomes")
    lock = json.loads(EXECUTION_LOCK_PATH.read_text(encoding="utf-8"))
    basis = lock["freeze_basis"]
    expected = {
        "protocol_snapshot_sha256": sha256(PROTOCOL_PATH),
        "prediction_freeze_sha256": sha256(PREDICTION_FREEZE_PATH),
        "blind_predictions_sha256": sha256(PREDICTIONS_PATH),
        "target_table_sha256": sha256(TARGET_PATH),
        "evaluation_script_sha256": sha256(Path(__file__).resolve()),
    }
    for key, value in expected.items():
        if basis[key] != value:
            raise RuntimeError(f"Evaluation lock hash mismatch: {key}")

    prediction_freeze = json.loads(PREDICTION_FREEZE_PATH.read_text(encoding="utf-8"))
    validate_outputs(prediction_freeze)
    predictions = pd.read_parquet(PREDICTIONS_PATH)
    if "fhd_normal" in predictions.columns:
        raise RuntimeError("Blind prediction file already contains target outcomes")
    if set(predictions["strategy"]) != set(STRATEGIES):
        raise RuntimeError("Blind prediction strategy set changed")

    if not OPEN_EVENT_PATH.exists():
        write_json(OPEN_EVENT_PATH, {
            "lock_id": lock["lock_id"],
            "first_open_utc": utc_now(),
            "status": "evaluation_started",
            "authorized_in_current_session": True,
            "target_outcomes_opened": True,
        })
    event = json.loads(OPEN_EVENT_PATH.read_text(encoding="utf-8"))
    if event["lock_id"] != lock["lock_id"]:
        raise RuntimeError("Outcome-open event belongs to another execution lock")

    outcomes = pd.read_parquet(TARGET_PATH, columns=KEYS + ["fhd_normal"])
    if outcomes.duplicated(KEYS).any() or not np.isfinite(outcomes["fhd_normal"]).all():
        raise RuntimeError("Prospective outcome table is invalid")
    evaluated = predictions.merge(outcomes, on=KEYS, how="left", validate="many_to_one")
    if len(evaluated) != len(predictions) or evaluated["fhd_normal"].isna().any():
        raise RuntimeError("Prospective outcome join is incomplete")

    metrics = metric_table(evaluated)
    paired, bootstrap = paired_results(
        metrics,
        int(basis["bootstrap_replicates"]),
        int(basis["random_seed"]),
    )
    gate = gate_result(paired)
    write_parquet(evaluated, EVALUATED_PATH)
    write_csv(metrics, METRICS_PATH)
    write_csv(paired, PAIRED_PATH)
    write_csv(bootstrap, BOOTSTRAP_PATH)
    save_figure(metrics, paired)

    event["completed_utc"] = utc_now()
    event["status"] = "evaluation_completed"
    write_json(OPEN_EVENT_PATH, event)
    outputs = {
        "evaluated_predictions": {
            "path": str(EVALUATED_PATH.relative_to(ROOT)),
            "rows": len(evaluated),
            "sha256": sha256(EVALUATED_PATH),
        },
        "metrics": {
            "path": str(METRICS_PATH.relative_to(ROOT)),
            "rows": len(metrics),
            "sha256": sha256(METRICS_PATH),
        },
        "paired_summary": {
            "path": str(PAIRED_PATH.relative_to(ROOT)),
            "rows": len(paired),
            "sha256": sha256(PAIRED_PATH),
        },
        "site_bootstrap": {
            "path": str(BOOTSTRAP_PATH.relative_to(ROOT)),
            "rows": len(bootstrap),
            "sha256": sha256(BOOTSTRAP_PATH),
        },
        "figure": {
            "path": str(FIGURE_PATH.relative_to(ROOT)),
            "sha256": sha256(FIGURE_PATH),
        },
        "outcome_open_event": {
            "path": str(OPEN_EVENT_PATH.relative_to(ROOT)),
            "sha256": sha256(OPEN_EVENT_PATH),
        },
    }
    macro = metrics[metrics["scope"].eq("macro")].set_index("strategy")
    freeze_basis = {
        "execution_lock_id": lock["lock_id"],
        "execution_lock_sha256": sha256(EXECUTION_LOCK_PATH),
        "prediction_id": prediction_freeze["prediction_id"],
        "target_outcomes_opened": True,
        "first_open_utc": event["first_open_utc"],
        "site_count": int(outcomes["site_id"].nunique()),
        "target_rows": len(outcomes),
        "strategy_count": len(STRATEGIES),
        "metrics": METRICS,
        "bootstrap_unit": basis["bootstrap_unit"],
        "bootstrap_replicates": int(basis["bootstrap_replicates"]),
        "random_seed": int(basis["random_seed"]),
        "primary_gate": gate,
        "macro_metrics": {
            strategy: {metric: float(macro.loc[strategy, metric]) for metric in METRICS}
            for strategy in STRATEGIES
        },
    }
    evaluation_id = "phase9-prospective-evaluation-" + canonical_hash(freeze_basis)[:12]
    freeze = {
        "evaluation_id": evaluation_id,
        "created_utc": utc_now(),
        "status": "frozen_complete",
        "freeze_basis": freeze_basis,
        "outputs": outputs,
    }
    write_json(FREEZE_PATH, freeze)
    print(json.dumps({
        "evaluation_id": evaluation_id,
        "target_sites": sorted(outcomes["site_id"].unique()),
        "target_rows": len(outcomes),
        "macro_metrics": freeze_basis["macro_metrics"],
        "primary_gate": gate,
    }, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--lock", action="store_true")
    group.add_argument("--evaluate", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    raise SystemExit(lock_execution() if arguments.lock else evaluate())
