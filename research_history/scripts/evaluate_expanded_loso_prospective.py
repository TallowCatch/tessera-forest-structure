#!/usr/bin/env python3
"""Evaluate frozen target-free predictions on the untouched Phase 7 expansion labels."""

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
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_multisite_loso as loso  # noqa: E402


PROTOCOL_PATH = ROOT / "metadata/project_config_phase7_site_expansion_protocol_freeze.yaml"
PREDICTION_FREEZE_PATH = ROOT / "metadata/phase7_prospective_prediction_freeze.json"
PREDICTIONS_PATH = ROOT / "data/processed/phase7_prospective_target_free_predictions.parquet"
TARGET_PATH = ROOT / "data/processed/phase7_tessera_aligned_expansion.parquet"
EVALUATION_PATH = ROOT / "data/processed/phase7_prospective_evaluation.parquet"
METRICS_PATH = ROOT / "outputs/tables/phase7_prospective_metrics.csv"
FIGURE_PATH = ROOT / "outputs/figures/phase7_prospective_external_evaluation.png"
FREEZE_PATH = ROOT / "metadata/phase7_prospective_evaluation_freeze.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_metrics(evaluation: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (site, method), frame in evaluation.groupby(["site_id", "method"], sort=True):
        records.append({
            "scope": "site", "site_id": site, "method": method, "rows": len(frame),
            **loso.metric_values(frame["fhd_normal"].to_numpy(), frame["prediction"].to_numpy()),
        })
    site_metrics = pd.DataFrame(records)
    for method, frame in site_metrics.groupby("method", sort=True):
        records.append({
            "scope": "macro", "site_id": "macro_mean", "method": method, "rows": len(frame),
            **{metric: float(frame[metric].mean()) for metric in loso.METRICS},
        })
    for method, frame in evaluation.groupby("method", sort=True):
        records.append({
            "scope": "pooled", "site_id": "all_expansion_sites", "method": method, "rows": len(frame),
            **loso.metric_values(frame["fhd_normal"].to_numpy(), frame["prediction"].to_numpy()),
        })
    return pd.DataFrame(records).sort_values(["scope", "site_id", "method"]).reset_index(drop=True)


def save_figure(metrics: pd.DataFrame) -> None:
    site = metrics[metrics["scope"].eq("site")]
    sites = sorted(site["site_id"].unique())
    methods = ["original_site_mean", "tessera_area_ridge"]
    colours = ["#6c757d", "#0a9396"]
    labels = ["original-site mean", "TESSERA Ridge"]
    x = np.arange(len(sites))
    width = 0.36
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    for index, (method, colour, label) in enumerate(zip(methods, colours, labels, strict=True)):
        frame = site[site["method"].eq(method)].set_index("site_id").loc[sites]
        offset = (index - 0.5) * width
        axes[0].bar(x + offset, frame["rmse"], width, color=colour, label=label)
        axes[1].bar(x + offset, frame["r2"], width, color=colour)
    for axis in axes:
        axis.set_xticks(x, sites)
        axis.set_xlabel("Untouched expansion site")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("RMSE")
    axes[1].set_ylabel("R-squared")
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    figure.suptitle("Prospective external evaluation on outcome-blind forest sites")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="lower center", ncol=2, frameon=False)
    figure.tight_layout(rect=(0, 0.12, 1, 1))
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = FIGURE_PATH.with_suffix(".tmp.png")
    figure.savefig(temporary, dpi=180, bbox_inches="tight")
    plt.close(figure)
    temporary.replace(FIGURE_PATH)


def main() -> int:
    protected = [EVALUATION_PATH, METRICS_PATH, FIGURE_PATH, FREEZE_PATH]
    existing = [str(path.relative_to(ROOT)) for path in protected if path.exists()]
    if existing:
        raise RuntimeError(f"Prospective evaluation exists; refusing to overwrite: {existing}")
    prediction_freeze = json.loads(PREDICTION_FREEZE_PATH.read_text(encoding="utf-8"))
    basis = prediction_freeze["freeze_basis"]
    if basis["expansion_columns_contain_fhd_normal"] or basis[
        "expansion_target_labels_used_for_tuning_training_or_selection"
    ]:
        raise RuntimeError("Prospective prediction freeze reports expansion-label access")
    predictions = pd.read_parquet(PREDICTIONS_PATH)
    targets = pd.read_parquet(TARGET_PATH, columns=["site_id", "shot_number", "fhd_normal"])
    evaluation = predictions.merge(targets, on=["site_id", "shot_number"], validate="many_to_one")
    if len(evaluation) != len(predictions):
        raise RuntimeError("Prospective target join changed prediction population")
    metrics = build_metrics(evaluation)
    site = metrics[metrics["scope"].eq("site")]
    tessera = site[site["method"].eq("tessera_area_ridge")].set_index("site_id")
    baseline = site[site["method"].eq("original_site_mean")].set_index("site_id")
    site_count = len(tessera)
    majority = math.floor(site_count / 2) + 1
    positive_sites = sorted(tessera.index[tessera["r2"].gt(0)].tolist())
    lower_rmse_sites = sorted(tessera.index[tessera["rmse"].lt(baseline["rmse"])].tolist())
    macro = metrics[metrics["scope"].eq("macro")].set_index("method")
    gate = {
        "viable_expansion_sites": site_count,
        "majority_required": majority,
        "positive_r2_sites": positive_sites,
        "positive_r2_site_count": len(positive_sites),
        "rmse_below_training_mean_sites": lower_rmse_sites,
        "rmse_below_training_mean_site_count": len(lower_rmse_sites),
        "tessera_macro_rmse": float(macro.loc["tessera_area_ridge", "rmse"]),
        "training_mean_macro_rmse": float(macro.loc["original_site_mean", "rmse"]),
    }
    gate["passed"] = bool(
        len(positive_sites) >= majority
        and len(lower_rmse_sites) >= majority
        and gate["tessera_macro_rmse"] < gate["training_mean_macro_rmse"]
    )
    loso.write_parquet(evaluation, EVALUATION_PATH)
    loso.write_csv(metrics, METRICS_PATH)
    save_figure(metrics)
    outputs = {
        "evaluation": {"path": str(EVALUATION_PATH.relative_to(ROOT)), "rows": len(evaluation), "sha256": sha256(EVALUATION_PATH)},
        "metrics": {"path": str(METRICS_PATH.relative_to(ROOT)), "rows": len(metrics), "sha256": sha256(METRICS_PATH)},
        "figure": {"path": str(FIGURE_PATH.relative_to(ROOT)), "sha256": sha256(FIGURE_PATH)},
    }
    freeze_basis = {
        "prediction_freeze_id": prediction_freeze["freeze_id"],
        "prediction_freeze_sha256": sha256(PREDICTION_FREEZE_PATH),
        "target_sha256": sha256(TARGET_PATH),
        "target_columns_opened_at_evaluation": ["site_id", "shot_number", "fhd_normal"],
        "site_count": site_count,
        "gate": gate,
    }
    freeze_id = "phase7-prospective-evaluation-" + canonical_hash(
        {"freeze_basis": freeze_basis, "outputs": outputs}
    )[:12]
    write_json(FREEZE_PATH, {
        "freeze_id": freeze_id, "created_utc": utc_now(), "freeze_basis": freeze_basis, "outputs": outputs
    })
    print(json.dumps({"freeze_id": freeze_id, "gate": gate, "site_metrics": {
        site: {metric: float(tessera.loc[site, metric]) for metric in loso.METRICS}
        for site in sorted(tessera.index)
    }}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
