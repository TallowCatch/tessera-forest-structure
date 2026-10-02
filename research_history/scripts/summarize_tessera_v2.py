#!/usr/bin/env python3
"""Create a concise, cross-site interpretation of the frozen v1-v2 results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_TABLE = ROOT / "outputs/tables/phase27_v1_v2_summary.csv"
OUTPUT_REPORT = ROOT / "outputs/reports/phase27_v1_v2_interpretation.md"


def savelsbos_rows() -> list[dict[str, object]]:
    metrics = pd.read_csv(ROOT / "outputs/tables/phase27_savelsbos_v2_metrics.csv")
    metrics = metrics[metrics["scope"] == "pooled"]
    comparison = pd.read_csv(
        ROOT / "outputs/tables/phase27_savelsbos_v1_v2_comparison.csv"
    )
    records: list[dict[str, object]] = []
    for row in comparison.itertuples(index=False):
        family = "fused" if row.candidate == "fused_v2" else "tessera"
        values = metrics[
            (metrics["target"] == row.target)
            & (metrics["target_variant"] == row.target_variant)
            & metrics["model"].isin([row.reference, row.candidate])
        ].set_index("model")
        records.append(
            {
                "site": "Savelsbos",
                "target": row.target,
                "target_variant": row.target_variant,
                "model_family": family,
                "v1_r2": values.loc[row.reference, "r2"],
                "v2_r2": values.loc[row.candidate, "r2"],
                "v1_rmse": values.loc[row.reference, "rmse"],
                "v2_rmse": values.loc[row.candidate, "rmse"],
                "v2_minus_v1_rmse": row.delta_rmse,
                "ci_low": row.ci_low,
                "ci_high": row.ci_high,
                "blocks": row.evaluation_blocks,
            }
        )
    return records


def cairngorms_rows() -> list[dict[str, object]]:
    metrics = pd.read_csv(ROOT / "outputs/tables/phase27_cairngorms_v2_metrics.csv")
    comparison = pd.read_csv(
        ROOT / "outputs/tables/phase27_cairngorms_v1_v2_comparison.csv"
    )
    records: list[dict[str, object]] = []
    suffix = "_height_adjusted"
    for row in comparison.itertuples(index=False):
        family = "fused" if row.first_model == "fused_v2" else "tessera"
        variant = "height_adjusted" if row.target.endswith(suffix) else "raw"
        target = row.target.removesuffix(suffix)
        values = metrics[
            (metrics["target"] == row.target)
            & metrics["model"].isin([row.second_model, row.first_model])
        ].set_index("model")
        records.append(
            {
                "site": "Cairngorms",
                "target": target,
                "target_variant": variant,
                "model_family": family,
                "v1_r2": values.loc[row.second_model, "r2"],
                "v2_r2": values.loc[row.first_model, "r2"],
                "v1_rmse": values.loc[row.second_model, "rmse"],
                "v2_rmse": values.loc[row.first_model, "rmse"],
                "v2_minus_v1_rmse": row.rmse_difference,
                "ci_low": row.ci_low,
                "ci_high": row.ci_high,
                "blocks": row.blocks,
            }
        )
    return records


def run() -> None:
    summary = pd.DataFrame(cairngorms_rows() + savelsbos_rows())
    summary["clear_v2_improvement"] = summary["ci_high"] < 0
    summary["clear_v2_degradation"] = summary["ci_low"] > 0
    summary["r2_change"] = summary["v2_r2"] - summary["v1_r2"]
    summary = summary.sort_values(
        ["site", "target_variant", "target", "model_family"]
    ).reset_index(drop=True)
    OUTPUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_TABLE, index=False)

    tessera = summary[summary["model_family"] == "tessera"]
    clear = summary[summary["clear_v2_improvement"]]
    worse = summary[summary["clear_v2_degradation"]]
    lines = [
        "# Experimental TESSERA v2 interpretation",
        "",
        "The comparison changes only the 128-dimensional TESSERA representation. Observation years, LiDAR targets, 50 m supports, spatial folds, exclusion buffers, models and random seeds are unchanged.",
        "",
        "## Main findings",
        "",
        "- In the Cairngorms, v2 clearly improved corrected canopy Shannon entropy before and after accounting for mean and p95 height. TESSERA-only R2 increased from 0.145 to 0.196 for raw Shannon and from 0.052 to 0.140 for height-adjusted Shannon.",
        "- Cairngorms raw CV, robust CV and RMS improved slightly with v2, but their TESSERA-only paired intervals crossed zero. The evidence does not establish improvements for those targets.",
        "- In Savelsbos, v2 raised pooled R2 for raw entropy from 0.362 to 0.400 and raw CV from 0.493 to 0.530. The 36-block paired intervals crossed zero, so these remain suggestive rather than confirmed improvements.",
        "- Savelsbos pulse penetration worsened. Its TESSERA-only R2 changed from 0.024 to -0.110, and the height-adjusted fused model showed a clear increase in error.",
        "- V2 does not produce a universal gain. Its clearest contribution is stronger information about how vegetation returns are distributed among height layers in the corrected Cairngorms data.",
        "",
        "## Confirmed paired changes",
        "",
        "A change is called clear only when the spatial-block bootstrap 95% interval lies entirely below or above zero.",
        "",
        "```text",
        clear[[
            "site", "target", "target_variant", "model_family", "v1_r2", "v2_r2",
            "v2_minus_v1_rmse", "ci_low", "ci_high",
        ]].to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "```",
        "",
    ]
    if len(worse):
        lines.extend(
            [
                "## Confirmed degradations",
                "",
                "```text",
                worse[[
                    "site", "target", "target_variant", "model_family", "v1_r2",
                    "v2_r2", "v2_minus_v1_rmse", "ci_low", "ci_high",
                ]].to_string(index=False, float_format=lambda value: f"{value:.4f}"),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## Scientific interpretation",
            "",
            "The replicated direction for entropy and CV supports the broader finding that TESSERA represents forest heterogeneity more reliably than absolute upper-canopy height or below-canopy penetration. The substantial v2 gain for height-adjusted Shannon in the Cairngorms is particularly relevant because it remains after removing the average relationship with canopy height. Savelsbos is smaller and supplies less precise version comparisons, so it supports the direction of the result without proving a second significant v2 gain.",
            "",
            "The next model experiment should use v2 for the selected Cairngorms entropy and surface-heterogeneity analysis. A dense U-Net rerun is justified for Shannon because v2 improved that target under the locked scalar test. Pulse penetration should remain a negative-control outcome rather than a headline result.",
            "",
            f"TESSERA-only comparisons represented in this summary: {len(tessera)}.",
        ]
    )
    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT_TABLE.relative_to(ROOT)}")
    print(f"wrote {OUTPUT_REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    run()
