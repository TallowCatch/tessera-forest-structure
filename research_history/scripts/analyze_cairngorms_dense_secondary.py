#!/usr/bin/env python3
"""Run exploratory paired block comparisons for every dense target."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "dense_aggregate", ROOT / "scripts/aggregate_cairngorms_dense_10m.py"
)
assert SPEC is not None and SPEC.loader is not None
aggregate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(aggregate)


def main() -> None:
    config = aggregate.load_config()
    predictions = pd.read_parquet(
        ROOT / str(config["outputs"]["prediction_table"])
    )
    pooled = pd.read_csv(
        ROOT / str(config["outputs"]["pooled_metrics_table"])
    )
    targets = [str(value) for value in config["targets"]["names"]]
    pairs = [
        ("residual_patch_cnn", "summary_mlp"),
        ("compact_unet", "summary_mlp"),
        ("compact_unet", "residual_patch_cnn"),
    ]
    records: list[dict[str, object]] = []
    for candidate, reference in pairs:
        candidate_frame = predictions[
            predictions["model"] == candidate
        ].sort_values("row_index")
        reference_frame = predictions[
            predictions["model"] == reference
        ].sort_values("row_index")
        if not candidate_frame["row_index"].reset_index(drop=True).equals(
            reference_frame["row_index"].reset_index(drop=True)
        ):
            raise RuntimeError(f"OOF rows differ for {candidate} and {reference}")
        for target_index, target in enumerate(targets):
            observed = reference_frame[f"observed_{target}"].to_numpy(
                dtype="float64"
            )
            candidate_prediction = candidate_frame[
                f"predicted_{target}"
            ].to_numpy(dtype="float64")
            reference_prediction = reference_frame[
                f"predicted_{target}"
            ].to_numpy(dtype="float64")
            delta, lower, upper = aggregate.block_bootstrap_comparison(
                observed,
                candidate_prediction,
                reference_prediction,
                reference_frame["spatial_block"].to_numpy(dtype="int64"),
                int(config["uncertainty"]["block_bootstrap_replicates"]),
                int(config["uncertainty"]["seed"]) + target_index,
            )
            candidate_rmse = float(
                pooled[
                    (pooled["model"] == candidate)
                    & (pooled["target"] == target)
                ]["rmse"].iloc[0]
            )
            reference_rmse = float(
                pooled[
                    (pooled["model"] == reference)
                    & (pooled["target"] == target)
                ]["rmse"].iloc[0]
            )
            reduction = (reference_rmse - candidate_rmse) / reference_rmse
            records.append(
                {
                    "analysis": "exploratory_secondary_target_comparison",
                    "candidate": candidate,
                    "reference": reference,
                    "target": target,
                    "candidate_rmse": candidate_rmse,
                    "reference_rmse": reference_rmse,
                    "rmse_reduction_fraction": reduction,
                    "paired_rmse_delta": delta,
                    "paired_rmse_delta_ci_lower": lower,
                    "paired_rmse_delta_ci_upper": upper,
                    "interval_excludes_zero": upper < 0 or lower > 0,
                    "five_percent_and_supported": reduction >= 0.05
                    and upper < 0,
                }
            )
    output = pd.DataFrame(records)
    path = (
        ROOT
        / "outputs/tables/cairngorms_dense_10m_secondary_comparisons.csv"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
