#!/usr/bin/env python3
"""Audit the frozen Cairngorms TESSERA v2 U-Net implementation and outputs."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree

import cairngorms_dense_10m_worker as dense


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/cairngorms_v2_unet.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def membership(row_count: int, indices: np.ndarray) -> np.ndarray:
    selected = np.zeros(row_count, dtype=bool)
    selected[indices] = True
    return selected


def chip_coverage(
    index_grid: np.ndarray,
    indices: np.ndarray,
    size: int,
    stride: int,
    minimum: int,
    row_count: int,
) -> tuple[np.ndarray, int]:
    selected = membership(row_count, indices)
    chips = dense.select_chips(index_grid, selected, size, stride, minimum)
    coverage = np.zeros(row_count, dtype=np.int16)
    for row, column in chips:
        ids = np.asarray(index_grid[row : row + size, column : column + size])
        ids = ids[ids >= 0]
        ids = ids[selected[ids]]
        np.add.at(coverage, ids, 1)
    return coverage[indices], len(chips)


def distribution(values: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(values, return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(unique, counts)}


def run() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "phase28_cairngorms_v2_unet"
    ]
    outputs = config["outputs"]
    model = config["model"]
    target_names = [str(value) for value in config["targets"]["names"]]
    array_dir = ROOT / str(outputs["array_directory"])
    result_dir = ROOT / str(outputs["result_directory"])
    preparation = json.loads(
        (ROOT / str(outputs["preparation_status"])).read_text(encoding="utf-8")
    )
    cohort = pd.read_parquet(
        ROOT / str(config["frozen_inputs"]["corrected_cohort"])
    ).sort_values("row_id")
    row_ids = np.load(array_dir / "row_id.npy")
    if not np.array_equal(row_ids, cohort["row_id"].to_numpy(dtype=np.int64)):
        raise RuntimeError("Audit failed: cohort and array row IDs differ")
    coordinates = cohort[["bng_x", "bng_y"]].to_numpy(dtype=np.float64)
    index_grid = np.load(array_dir / "index_grid.npy", mmap_mode="r")
    row_count = len(row_ids)

    fold_checks: list[dict[str, Any]] = []
    for fold_number in range(int(config["evaluation"]["folds"])):
        with np.load(array_dir / "folds" / f"fold_{fold_number}.npz") as values:
            train = values["train_indices"].astype(np.int64)
            subtrain = values["subtrain_indices"].astype(np.int64)
            validation = values["validation_indices"].astype(np.int64)
            test = values["test_indices"].astype(np.int64)
            tuning_targets = values["tuning_targets"]
            final_targets = values["final_targets"]
        train_coverage, train_chips = chip_coverage(
            index_grid,
            train,
            int(model["chip_cells"]),
            int(model["chip_stride_cells"]),
            int(model["minimum_training_targets_per_chip"]),
            row_count,
        )
        test_coverage, test_chips = chip_coverage(
            index_grid,
            test,
            int(model["chip_cells"]),
            int(model["chip_stride_cells"]),
            1,
            row_count,
        )
        minimum_distance = float(
            cKDTree(coordinates[train]).query(coordinates[test], k=1)[0].min()
        )
        fold_checks.append(
            {
                "fold": fold_number,
                "counts": {
                    "train": int(len(train)),
                    "subtrain": int(len(subtrain)),
                    "validation": int(len(validation)),
                    "test": int(len(test)),
                },
                "train_test_overlap": int(len(np.intersect1d(train, test))),
                "subtrain_validation_overlap": int(
                    len(np.intersect1d(subtrain, validation))
                ),
                "subtrain_validation_reconstruct_train": bool(
                    np.array_equal(np.sort(np.concatenate([subtrain, validation])), np.sort(train))
                ),
                "minimum_train_test_distance_m": minimum_distance,
                "tuning_targets_finite": bool(
                    np.isfinite(tuning_targets[np.concatenate([subtrain, validation])]).all()
                ),
                "final_targets_finite": bool(
                    np.isfinite(final_targets[np.concatenate([train, test])]).all()
                ),
                "training_chips": train_chips,
                "training_target_coverage": distribution(train_coverage),
                "training_targets_uncovered": int((train_coverage == 0).sum()),
                "test_chips": test_chips,
                "test_target_coverage": distribution(test_coverage),
                "test_targets_uncovered": int((test_coverage == 0).sum()),
            }
        )

    metadata_paths = sorted(result_dir.glob("tessera_v2_unet_fold_*_seed_*.json"))
    run_metadata = [json.loads(path.read_text(encoding="utf-8")) for path in metadata_paths]
    best_epochs = [int(value["best_epoch"]) for value in run_metadata]
    parameter_counts = sorted({int(value["parameters"]) for value in run_metadata})

    macro = pd.read_csv(ROOT / str(outputs["macro_metrics"]))
    comparison = pd.read_csv(ROOT / str(outputs["comparison"]))
    result_rows = macro[
        macro["target"].isin(
            [
                "lidar_canopy_shannon_50m",
                "lidar_canopy_shannon_50m_height_adjusted",
                "lidar_height_cv_50m",
                "lidar_height_cv_50m_height_adjusted",
            ]
        )
    ][["model", "target", "n", "blocks", "rmse", "r2", "block_spearman"]]

    evidence_files = [
        CONFIG_PATH,
        ROOT / "scripts/prepare_cairngorms_v2_unet.py",
        ROOT / "scripts/cairngorms_v2_unet_worker.py",
        ROOT / "scripts/cairngorms_dense_10m_worker.py",
        ROOT / "scripts/aggregate_cairngorms_v2_unet.py",
        ROOT / "tests/test_cairngorms_v2_unet.py",
        ROOT / str(outputs["preparation_status"]),
        ROOT / str(outputs["result_freeze"]),
        ROOT / str(outputs["macro_metrics"]),
        ROOT / str(outputs["comparison"]),
    ]
    audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "scope": "Frozen Cairngorms TESSERA v2 U-Net implementation and outputs",
        "evidence_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in evidence_files
        },
        "configuration": {
            "input_channels": int(model["input_channels"]),
            "output_channels": int(model["output_channels"]),
            "base_channels": int(model["base_channels"]),
            "chip_cells": int(model["chip_cells"]),
            "chip_stride_cells": int(model["chip_stride_cells"]),
            "minimum_training_targets_per_chip": int(
                model["minimum_training_targets_per_chip"]
            ),
            "maximum_epochs": int(model["maximum_epochs"]),
            "minimum_epochs": int(model["minimum_epochs"]),
            "patience": int(model["early_stopping_patience"]),
            "targets": target_names,
        },
        "alignment": preparation["centre_alignment"],
        "rows": int(preparation["rows"]),
        "dense_shape": preparation["dense_shape"],
        "dense_valid_cells": int(preparation["dense_valid_cells"]),
        "fold_checks": fold_checks,
        "runs": {
            "metadata_files": len(metadata_paths),
            "expected": int(config["evaluation"]["folds"])
            * len(config["evaluation"]["seeds"]),
            "parameter_counts": parameter_counts,
            "best_epochs": sorted(best_epochs),
            "selected_before_configured_minimum": int(
                sum(epoch < int(model["minimum_epochs"]) for epoch in best_epochs)
            ),
        },
        "scope_checks": {
            "final_v2_targets_include_p95_height": any(
                "p95" in target.lower() or "height_m" in target.lower()
                for target in target_names
            ),
            "final_v2_targets_are_corrected_point_return_metrics": True,
        },
        "results": result_rows.to_dict(orient="records"),
        "paired_comparison": comparison.to_dict(orient="records"),
        "audit_findings": {
            "train_test_overlap_detected": any(
                value["train_test_overlap"] for value in fold_checks
            ),
            "subtrain_validation_overlap_detected": any(
                value["subtrain_validation_overlap"] for value in fold_checks
            ),
            "minimum_two_km_separation_pass": all(
                value["minimum_train_test_distance_m"] >= 2000.0
                for value in fold_checks
            ),
            "all_test_targets_covered": all(
                value["test_targets_uncovered"] == 0 for value in fold_checks
            ),
            "all_training_targets_covered": all(
                value["training_targets_uncovered"] == 0 for value in fold_checks
            ),
            "training_targets_have_equal_chip_multiplicity": all(
                len(value["training_target_coverage"]) == 1 for value in fold_checks
            ),
            "configured_minimum_epoch_enforced": all(
                epoch >= int(model["minimum_epochs"]) for epoch in best_epochs
            ),
        },
    }
    output = ROOT / "metadata/phase28_cairngorms_v2_unet_internal_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit["audit_findings"], indent=2, sort_keys=True))
    print(f"wrote {output}")


if __name__ == "__main__":
    run()
