"""Deterministic worked example for cross-forest transfer evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import (
    coefficient_of_determination,
    root_mean_squared_error,
    spearman_rank_correlation,
)
from .modeling import (
    fit_standardized_ridge,
    offset_from_references,
    predict_standardized_ridge,
)


def load_demo_data(path: Path) -> dict[str, np.ndarray]:
    """Load the labelled synthetic data distributed with the repository."""
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    return {
        "site": np.asarray([row["site"] for row in rows]),
        "partition": np.asarray([row["partition"] for row in rows]),
        "features": np.asarray(
            [[float(row["feature_1"]), float(row["feature_2"])] for row in rows]
        ),
        "target": np.asarray([float(row["lidar_metric"]) for row in rows]),
    }


def run_demo(path: Path) -> dict[str, Any]:
    """Fit on one site, transfer to another, then apply a local offset."""
    data = load_demo_data(path)
    source = data["site"] == "source_forest"
    reference = (data["site"] == "target_forest") & (data["partition"] == "reference")
    test = (data["site"] == "target_forest") & (data["partition"] == "test")

    model = fit_standardized_ridge(data["features"][source], data["target"][source])
    reference_prediction = predict_standardized_ridge(model, data["features"][reference])
    test_prediction = predict_standardized_ridge(model, data["features"][test])
    offset = offset_from_references(data["target"][reference], reference_prediction)
    corrected = test_prediction + offset

    def metrics(predicted: np.ndarray) -> dict[str, float]:
        observed = data["target"][test]
        return {
            "rmse": root_mean_squared_error(observed, predicted),
            "r2": coefficient_of_determination(observed, predicted),
            "spearman": spearman_rank_correlation(observed, predicted),
        }

    return {
        "description": "Synthetic scale-shift example; not a study result.",
        "source_training_units": int(source.sum()),
        "target_reference_units": int(reference.sum()),
        "target_test_units": int(test.sum()),
        "estimated_offset": offset,
        "zero_shot": metrics(test_prediction),
        "reference_assisted": metrics(corrected),
    }


def write_demo_result(result: dict[str, Any], output: Path | None = None) -> None:
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if output is None:
        print(payload, end="")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload, encoding="utf-8")
    print(f"wrote {output}")
