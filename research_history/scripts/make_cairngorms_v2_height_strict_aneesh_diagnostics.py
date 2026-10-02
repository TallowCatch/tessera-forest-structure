"""Create compact diagnostics for independent review of the Phase 31 U-Net."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = (
    ROOT
    / "data/processed/phase31_cairngorms_v2_height_strict_predictions.parquet"
)
OUTPUT_STEM = ROOT / "outputs/figures/phase31_aneesh_height_diagnostics"

MODELS = {
    "tessera_v2_mlp_5x5": "5 x 5 MLP",
    "tessera_v2_unet_strict": "Corrected U-Net",
}
COLORS = {
    "tessera_v2_mlp_5x5": "#39798a",
    "tessera_v2_unet_strict": "#c55233",
}


def rmse(values: pd.DataFrame) -> float:
    return float(np.sqrt(np.mean(np.square(values["predicted"] - values["observed"]))))


def main() -> None:
    frame = pd.read_parquet(PREDICTIONS)
    frame = frame.loc[
        frame["model"].isin(MODELS)
        & frame["target"].eq("canopy_p95_height_m")
    ].copy()

    # Each row appears once for each random seed. Assign height classes from the
    # observed distribution so model comparisons use identical observations.
    unique_observations = frame[["row_id", "observed"]].drop_duplicates("row_id")
    labels = ["Lowest 20%", "20-40%", "40-60%", "60-80%", "Highest 20%"]
    unique_observations["height_group"] = pd.qcut(
        unique_observations["observed"], q=5, labels=labels
    )
    frame = frame.merge(unique_observations[["row_id", "height_group"]], on="row_id")

    height_summary = (
        frame.groupby(["model", "height_group"], observed=True)
        .apply(rmse, include_groups=False)
        .rename("rmse")
        .reset_index()
    )
    fold_summary = (
        frame.groupby(["model", "fold"], observed=True)
        .apply(rmse, include_groups=False)
        .rename("rmse")
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.25), constrained_layout=True)
    width = 0.36

    x_height = np.arange(len(labels))
    for offset, model in zip((-width / 2, width / 2), MODELS):
        values = (
            height_summary.loc[height_summary["model"].eq(model)]
            .set_index("height_group")
            .reindex(labels)["rmse"]
        )
        axes[0].bar(
            x_height + offset,
            values,
            width,
            color=COLORS[model],
            label=MODELS[model],
        )
    axes[0].set_xticks(x_height, labels, rotation=25, ha="right")
    axes[0].set_ylabel("P95 height RMSE (m)")
    axes[0].set_title("a  Error across observed-height ranges", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, ncol=2, loc="upper left")

    folds = sorted(frame["fold"].unique())
    x_fold = np.arange(len(folds))
    for offset, model in zip((-width / 2, width / 2), MODELS):
        values = (
            fold_summary.loc[fold_summary["model"].eq(model)]
            .set_index("fold")
            .reindex(folds)["rmse"]
        )
        axes[1].bar(
            x_fold + offset,
            values,
            width,
            color=COLORS[model],
            label=MODELS[model],
        )
    axes[1].set_xticks(x_fold, [f"Fold {fold + 1}" for fold in folds])
    axes[1].set_ylabel("P95 height RMSE (m)")
    axes[1].set_title("b  Error across geographic test folds", loc="left", fontweight="bold")
    axes[1].legend(frameon=False, ncol=2, loc="upper left")

    for axis in axes:
        axis.grid(axis="y", color="#d7dce0", linewidth=0.8, alpha=0.75)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)

    OUTPUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight")

    print(height_summary.to_string(index=False))
    print(fold_summary.to_string(index=False))


if __name__ == "__main__":
    main()
