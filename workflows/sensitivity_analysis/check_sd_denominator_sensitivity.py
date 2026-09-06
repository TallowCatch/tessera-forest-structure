#!/usr/bin/env python3
"""Quantify the effect of using N instead of N-1 for CHM height SD."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TARGETS = ROOT / "data/processed/cairngorms_surface_targets.parquet"
PREDICTIONS = ROOT / "data/processed/cairngorms_surface_predictions.parquet"
OUTPUT = ROOT / "results/tables/sd_denominator_sensitivity.csv"


def equal_fold_metrics(frame: pd.DataFrame, observed: str, predicted: str) -> tuple[float, float]:
    fold_r2: list[float] = []
    fold_rmse: list[float] = []
    for _, fold in frame.groupby("fold", sort=True):
        y = fold[observed].to_numpy(float)
        y_hat = fold[predicted].to_numpy(float)
        residual = y - y_hat
        fold_rmse.append(float(np.sqrt(np.mean(residual**2))))
        denominator = float(np.sum((y - y.mean()) ** 2))
        fold_r2.append(float(1.0 - np.sum(residual**2) / denominator))
    return float(np.mean(fold_rmse)), float(np.mean(fold_r2))


def main() -> None:
    targets = pd.read_parquet(TARGETS)[["row_id", "chm_valid_pixels"]]
    predictions = pd.read_parquet(PREDICTIONS)
    selected = predictions[
        (predictions["variant"] == "raw")
        & (predictions["version"] == "v2")
        & (predictions["model_family"] == "tessera")
        & predictions["target"].isin(["canopy_surface_sd_m", "canopy_surface_cv"])
    ].merge(targets, on="row_id", validate="many_to_one")

    n = selected["chm_valid_pixels"].to_numpy(float)
    correction = np.sqrt((n - 1.0) / n)
    selected["population_observed"] = selected["observed"].to_numpy(float) * correction
    selected["population_predicted"] = selected["predicted"].to_numpy(float) * correction

    rows: list[dict[str, float | str]] = []
    for target, frame in selected.groupby("target", sort=True):
        sample_rmse, sample_r2 = equal_fold_metrics(frame, "observed", "predicted")
        population_rmse, population_r2 = equal_fold_metrics(
            frame, "population_observed", "population_predicted"
        )
        relative_change = np.abs(
            frame["population_observed"].to_numpy(float) / frame["observed"].to_numpy(float) - 1.0
        )
        rows.append(
            {
                "target": target,
                "units": int(len(frame)),
                "valid_pixels_minimum": int(frame["chm_valid_pixels"].min()),
                "valid_pixels_maximum": int(frame["chm_valid_pixels"].max()),
                "maximum_target_relative_change_percent": float(relative_change.max() * 100.0),
                "sample_rmse": sample_rmse,
                "population_scaled_rmse": population_rmse,
                "rmse_difference": population_rmse - sample_rmse,
                "sample_r2": sample_r2,
                "population_scaled_r2": population_r2,
                "r2_difference": population_r2 - sample_r2,
            }
        )

    output = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT, index=False)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
