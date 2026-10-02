#!/usr/bin/env python3
"""Aggregate the frozen Cairngorms v1.0 versus experimental v2 comparison."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from aggregate_cairngorms_height_adjusted import (
    bootstrap_difference,
    weighted_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/tessera_v2.yaml"


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "phase27_tessera_v2"
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp" + path.suffix)
    writer(temporary)
    temporary.replace(path)


def run() -> None:
    config = load_config()
    site = config["cairngorms"]
    evaluation = config["cairngorms_evaluation"]
    paths = {
        key: ROOT / str(site[key])
        for key in [
            "results", "v1_predictions", "predictions", "metrics", "bootstrap",
            "comparison", "report", "result_freeze",
        ]
    }
    if paths["result_freeze"].exists():
        raise RuntimeError("Phase 27 Cairngorms result is already frozen")
    array_dir = ROOT / str(site["folds"])
    row_ids = np.load(array_dir / "row_id.npy")
    blocks = np.load(array_dir / "spatial_block.npy")
    targets = [str(value) for value in evaluation["targets"]]
    suffix = str(evaluation["adjusted_suffix"])
    target_names = targets + [f"{target}{suffix}" for target in targets]
    seeds = [int(value) for value in evaluation["seeds"]]
    v2_parts: list[pd.DataFrame] = []
    for fold in range(int(evaluation["folds"])):
        for base_model in [str(value) for value in evaluation["models"]]:
            test = None
            observed = None
            seed_predictions = []
            for seed in seeds:
                result = paths["results"] / f"{base_model}_fold_{fold}_seed_{seed}.npz"
                if not result.exists():
                    raise RuntimeError(f"Missing v2 result: {result}")
                with np.load(result) as values:
                    current_test = values["test_indices"].astype(np.int64)
                    if test is not None and not np.array_equal(test, current_test):
                        raise RuntimeError(f"Test rows changed in {result}")
                    test = current_test
                    observed = values["observed"].astype(np.float32)
                    seed_predictions.append(values["predictions"].astype(np.float32))
            assert test is not None and observed is not None
            predicted = np.mean(seed_predictions, axis=0)
            for column, target in enumerate(target_names):
                v2_parts.append(
                    pd.DataFrame(
                        {
                            "row_id": row_ids[test],
                            "fold": fold,
                            "model": f"{base_model}_v2",
                            "target": target,
                            "observed": observed[:, column],
                            "predicted": predicted[:, column],
                            "spatial_block": blocks[test],
                        }
                    )
                )
    v2_predictions = pd.concat(v2_parts, ignore_index=True)
    v1_predictions = pd.read_parquet(paths["v1_predictions"])
    v1_predictions = v1_predictions[
        v1_predictions["model"].isin(["tessera", "fused", "conventional"])
    ].copy()
    v1_predictions["model"] = v1_predictions["model"].map(
        {
            "tessera": "tessera_v1",
            "fused": "fused_v1",
            "conventional": "conventional",
        }
    )
    predictions = pd.concat([v1_predictions, v2_predictions], ignore_index=True)
    metric_rows: list[dict[str, Any]] = []
    for (fold, model, target), group in predictions.groupby(
        ["fold", "model", "target"], sort=True
    ):
        metric_rows.append(
            {
                "fold": int(fold), "model": model, "target": target,
                **weighted_metrics(
                    group["observed"].to_numpy(),
                    group["predicted"].to_numpy(),
                    group["spatial_block"].to_numpy(),
                ),
            }
        )
    metrics = pd.DataFrame(metric_rows)
    macro = metrics.groupby(["model", "target"], as_index=False).agg(
        folds=("fold", "nunique"),
        n=("n", "sum"),
        blocks=("blocks", "sum"),
        rmse=("rmse", "mean"),
        r2=("r2", "mean"),
        block_spearman=("block_spearman", "mean"),
        positive_r2_folds=("r2", lambda values: int((values > 0).sum())),
    )
    rng = np.random.default_rng(int(evaluation["bootstrap_seed"]))
    bootstrap_rows = []
    comparisons = [
        ("tessera_v2", "tessera_v1"),
        ("fused_v2", "fused_v1"),
        ("tessera_v2", "conventional"),
    ]
    for target in target_names:
        local = predictions[predictions["target"] == target]
        for first, second in comparisons:
            bootstrap_rows.append(
                {
                    "target": target,
                    "first_model": first,
                    "second_model": second,
                    **bootstrap_difference(
                        local,
                        first,
                        second,
                        int(evaluation["bootstrap_replicates"]),
                        rng,
                    ),
                }
            )
    bootstrap = pd.DataFrame(bootstrap_rows)
    comparison = bootstrap[bootstrap["second_model"].isin(["tessera_v1", "fused_v1"])]
    atomic(paths["predictions"], lambda p: predictions.to_parquet(p, index=False))
    atomic(paths["metrics"], lambda p: macro.to_csv(p, index=False))
    atomic(paths["bootstrap"], lambda p: bootstrap.to_csv(p, index=False))
    atomic(paths["comparison"], lambda p: comparison.to_csv(p, index=False))
    lines = [
        "# Cairngorms experimental TESSERA v2 comparison", "",
        "The corrected greater-than-1.3 m LiDAR targets, dispersed spatial folds, 2 km buffers, MLP settings and seeds were retained. Only the TESSERA representation changed.",
        "", "## Mean held-out metrics", "", "```text",
        macro.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "```", "", "## Paired v2 minus v1 RMSE", "", "```text",
        comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "```", "",
    ]
    atomic(paths["report"], lambda p: p.write_text("\n".join(lines), encoding="utf-8"))
    artifacts = {
        str(path.relative_to(ROOT)): sha256(path)
        for key, path in paths.items()
        if key in {"predictions", "metrics", "bootstrap", "comparison", "report"}
    }
    paths["result_freeze"].parent.mkdir(parents=True, exist_ok=True)
    paths["result_freeze"].write_text(
        json.dumps(
            {"created_utc": datetime.now(timezone.utc).isoformat(), "artifacts": artifacts},
            indent=2, sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    print("phase27 Cairngorms v2 evaluation complete", flush=True)


if __name__ == "__main__":
    run()
