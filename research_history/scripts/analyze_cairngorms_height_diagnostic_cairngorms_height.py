#!/usr/bin/env python3
"""Diagnose weak Cairngorms canopy-height prediction from frozen outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/cairngorms_height_diagnostic.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return yaml.safe_load(handle)["phase24_cairngorms_height_diagnostic"]


def absolute(path: str) -> Path:
    return ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def target_summary(broad: pd.DataFrame, mature: pd.DataFrame) -> pd.DataFrame:
    series = {
        "10 m broad sample: mean height": broad["lidar_meanH"],
        "10 m broad sample: p95 height": broad["lidar_p_95"],
        "50 m mature forest: mean height": mature["canopy_mean_height_m"],
        "50 m mature forest: p95 height": mature["canopy_p95_height_m"],
    }
    rows = []
    for name, values in series.items():
        values = values.dropna().astype(float)
        quantiles = values.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
        rows.append(
            {
                "cohort_target": name,
                "n": len(values),
                "mean_m": values.mean(),
                "sd_m": values.std(),
                "q01_m": quantiles.loc[0.01],
                "q05_m": quantiles.loc[0.05],
                "q25_m": quantiles.loc[0.25],
                "median_m": quantiles.loc[0.50],
                "q75_m": quantiles.loc[0.75],
                "q95_m": quantiles.loc[0.95],
                "q99_m": quantiles.loc[0.99],
            }
        )
    return pd.DataFrame(rows)


def target_agreement(mature: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("canopy_mean_height_m", "mean_height_50m"),
        ("canopy_p95_height_m", "mean_p95_height_50m"),
    ]
    rows = []
    for recalculated, supplied in pairs:
        values = mature[[recalculated, supplied]].dropna()
        difference = values[recalculated] - values[supplied]
        rows.append(
            {
                "recalculated_target": recalculated,
                "supplied_target": supplied,
                "n": len(values),
                "pearson_r": values.corr().iloc[0, 1],
                "mean_difference_m": difference.mean(),
                "rmse_difference_m": np.sqrt(np.mean(np.square(difference))),
            }
        )
    return pd.DataFrame(rows)


def prediction_summary(predictions: pd.DataFrame, scheme: str) -> pd.DataFrame:
    selected = predictions[
        (predictions["scheme"] == scheme)
        & (predictions["target"] == "canopy_p95_height_m")
    ]
    rows = []
    for model, values in selected.groupby("model"):
        blocks = values.groupby("spatial_block")[["observed", "predicted"]].mean()
        rows.append(
            {
                "model": model,
                "n": len(values),
                "observed_mean_m": values["observed"].mean(),
                "observed_sd_m": values["observed"].std(),
                "predicted_mean_m": values["predicted"].mean(),
                "predicted_sd_m": values["predicted"].std(),
                "prediction_sd_ratio": values["predicted"].std() / values["observed"].std(),
                "pooled_r2": r2_score(values["observed"], values["predicted"]),
                "pooled_rmse_m": mean_squared_error(values["observed"], values["predicted"]) ** 0.5,
                "point_spearman": spearmanr(values["observed"], values["predicted"]).statistic,
                "block_spearman": spearmanr(blocks["observed"], blocks["predicted"]).statistic,
                "mean_bias_m": (values["predicted"] - values["observed"]).mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)


def quintile_bias(predictions: pd.DataFrame, scheme: str) -> pd.DataFrame:
    selected = predictions[
        (predictions["scheme"] == scheme)
        & (predictions["target"] == "canopy_p95_height_m")
    ].copy()
    selected["observed_quintile"] = pd.qcut(
        selected["observed"], 5, labels=False, duplicates="drop"
    )
    grouped = (
        selected.groupby(["model", "observed_quintile"], as_index=False)
        .agg(n=("observed", "size"), observed_mean_m=("observed", "mean"), predicted_mean_m=("predicted", "mean"))
    )
    grouped["bias_m"] = grouped["predicted_mean_m"] - grouped["observed_mean_m"]
    return grouped


def fold_summary(mature: pd.DataFrame, fold_directory: Path, folds: int) -> pd.DataFrame:
    rows = []
    for fold in range(folds):
        values = np.load(fold_directory / f"balanced_fold_{fold}.npz")
        for split in ["train", "test"]:
            target = mature.iloc[values[f"{split}_indices"]]["canopy_p95_height_m"]
            rows.append(
                {
                    "fold": fold,
                    "split": split,
                    "n": len(target),
                    "mean_m": target.mean(),
                    "sd_m": target.std(),
                    "q05_m": target.quantile(0.05),
                    "q95_m": target.quantile(0.95),
                }
            )
    return pd.DataFrame(rows)


def load_features(config: dict[str, Any], mature: pd.DataFrame) -> np.ndarray:
    inputs = config["inputs"]
    row_ids = np.load(absolute(inputs["tessera_row_ids"]))
    summary = np.load(absolute(inputs["tessera_summary"]), mmap_mode="r")
    lookup = {int(row_id): position for position, row_id in enumerate(row_ids)}
    positions = np.array([lookup[int(row_id)] for row_id in mature["row_id"]], dtype=np.int64)
    return np.asarray(summary[positions], dtype=np.float32)


def fit_ridge(
    features: np.ndarray,
    target: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    alphas: np.ndarray,
) -> dict[str, float]:
    scaler = StandardScaler().fit(features[train])
    model = RidgeCV(alphas=alphas).fit(scaler.transform(features[train]), target[train])
    predicted = model.predict(scaler.transform(features[test]))
    return {
        "r2": r2_score(target[test], predicted),
        "rmse_m": mean_squared_error(target[test], predicted) ** 0.5,
        "alpha": float(model.alpha_),
    }


def ridge_diagnostic(
    features: np.ndarray,
    mature: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    settings = config["evaluation"]
    folds = int(settings["folds"])
    fold_directory = absolute(config["inputs"]["fold_directory"])
    spatial_splits = []
    for fold in range(folds):
        values = np.load(fold_directory / f"balanced_fold_{fold}.npz")
        spatial_splits.append((values["train_indices"], values["test_indices"]))
    random_splits = list(
        KFold(folds, shuffle=True, random_state=int(settings["random_seed"])).split(features)
    )
    alphas = np.asarray(settings["ridge_alphas"], dtype=float)
    rows = []
    for target_name in settings["height_targets"]:
        target = mature[target_name].to_numpy(dtype=np.float64)
        for scheme, splits in [("buffered_spatial", spatial_splits), ("random", random_splits)]:
            for fold, (train, test) in enumerate(splits):
                metrics = fit_ridge(features, target, np.asarray(train), np.asarray(test), alphas)
                rows.append(
                    {
                        "target": target_name,
                        "scheme": scheme,
                        "fold": fold,
                        "train_n": len(train),
                        "test_n": len(test),
                        "target_sd_m": target[test].std(ddof=1),
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def make_figure(
    broad: pd.DataFrame,
    mature: pd.DataFrame,
    predictions: pd.DataFrame,
    bias: pd.DataFrame,
    ridge: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    output = absolute(config["outputs"]["figure"])
    output.parent.mkdir(parents=True, exist_ok=True)
    model = config["evaluation"]["primary_model"]
    scheme = config["evaluation"]["scheme"]
    selected = predictions[
        (predictions["scheme"] == scheme)
        & (predictions["target"] == "canopy_p95_height_m")
        & (predictions["model"] == model)
    ]
    selected_bias = bias[bias["model"] == model]
    macro = ridge.groupby(["target", "scheme"], as_index=False)["r2"].mean()

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9})
    figure, axes = plt.subplots(2, 2, figsize=(8.0, 6.4), constrained_layout=True)

    axes[0, 0].hist(broad["lidar_meanH"], bins=50, density=True, alpha=0.55, label="10 m broad sample")
    axes[0, 0].hist(mature["canopy_mean_height_m"], bins=40, density=True, alpha=0.65, label="50 m mature forest")
    axes[0, 0].set(xlabel="Mean canopy height (m)", ylabel="Density", title="a  Height range changes with the cohort")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].hexbin(selected["observed"], selected["predicted"], gridsize=42, mincnt=1, cmap="viridis")
    low = min(selected["observed"].min(), selected["predicted"].min())
    high = max(selected["observed"].max(), selected["predicted"].max())
    axes[0, 1].plot([low, high], [low, high], color="black", linewidth=1, linestyle="--")
    axes[0, 1].set(xlabel="LiDAR p95 height (m)", ylabel="Predicted p95 height (m)", title="b  Upper-canopy predictions contract")

    axes[1, 0].bar(selected_bias["observed_quintile"] + 1, selected_bias["bias_m"], color="#167f88")
    axes[1, 0].axhline(0, color="black", linewidth=0.8)
    axes[1, 0].set(xlabel="Observed p95-height quintile", ylabel="Mean prediction bias (m)", title="c  Short stands are high; tall stands are low")

    labels = ["Recalculated mean", "Recalculated p95"]
    targets = ["canopy_mean_height_m", "canopy_p95_height_m"]
    x = np.arange(len(targets))
    width = 0.34
    for offset, scheme_name, colour in [(-width / 2, "buffered_spatial", "#167f88"), (width / 2, "random", "#8a8f98")]:
        values = [float(macro[(macro.target == target) & (macro.scheme == scheme_name)]["r2"].iloc[0]) for target in targets]
        axes[1, 1].bar(x + offset, values, width, label=scheme_name.replace("_", " "), color=colour)
    axes[1, 1].axhline(0, color="black", linewidth=0.8)
    axes[1, 1].set_xticks(x, labels)
    axes[1, 1].set(ylabel=r"Mean fold $R^2$", title="d  Spatial separation gives the honest test")
    axes[1, 1].legend(frameon=False)

    figure.savefig(output, dpi=240)
    plt.close(figure)


def write_report(
    config: dict[str, Any],
    summaries: dict[str, pd.DataFrame],
) -> None:
    target = summaries["target_summary"].set_index("cohort_target")
    models = summaries["prediction_summary"].set_index("model")
    ridge = summaries["ridge"]
    ridge_macro = ridge.groupby(["target", "scheme"])["r2"].mean()
    primary = config["evaluation"]["primary_model"]
    text = f"""# Cairngorms canopy-height diagnostic

## Question

Why is upper-canopy height substantially harder to predict than canopy-surface variation in the frozen 2023 Cairngorms experiment?

## Evidence

- The earlier 10 m sample covered a broad vegetation gradient: mean height SD was {target.loc['10 m broad sample: mean height', 'sd_m']:.2f} m. The 50 m mature-forest cohort has a narrower mean-height SD of {target.loc['50 m mature forest: mean height', 'sd_m']:.2f} m and p95-height SD of {target.loc['50 m mature forest: p95 height', 'sd_m']:.2f} m.
- Recalculated 50 m p95 height agrees with the supplied 50 m product at Pearson r = {summaries['target_agreement'].iloc[1]['pearson_r']:.3f}. The weak result is therefore not explained by a gross target-extraction error.
- The {primary} p95 predictions retain only {models.loc[primary, 'prediction_sd_ratio']:.1%} of the observed standard deviation. They overpredict short stands and underpredict tall stands.
- A ridge model using the same TESSERA features reaches mean fold R2 = {ridge_macro.loc[('canopy_mean_height_m', 'buffered_spatial')]:.3f} for recalculated mean height but {ridge_macro.loc[('canopy_p95_height_m', 'buffered_spatial')]:.3f} for p95 height. Mean height is strongly affected by canopy openings and cover; p95 more directly tests the upper canopy.
- Random folds give p95 R2 = {ridge_macro.loc[('canopy_p95_height_m', 'random')]:.3f}, compared with {ridge_macro.loc[('canopy_p95_height_m', 'buffered_spatial')]:.3f} under buffered spatial folds. Nearby training observations therefore make height prediction look substantially easier.
- The conventional MLP, TESSERA MLP, fused model, CNN and compact U-Net all show similar contraction. Architecture changes made so far do not remove the main limitation.
- All retained 50 m areas have complete CHM support and zero flagged pixels under the supplied terrain-quality masks. Known CHM terrain sensitivity remains a general limitation, but the available QA evidence does not identify it as the main cause here.

## Interpretation

TESSERA captures broad canopy state, cover and openings within the Cairngorms. It is less sensitive to the local upper-tail height differences that separate already-mature stands. The current result should be reported as moderate spatial prediction of p95 canopy height within one landscape, alongside stronger prediction of canopy-surface heterogeneity.

## Next checks

1. Ask Aland how the 1 m CHM was constructed: return selection, DTM/DSM method, handling of canopy gaps, vertical accuracy and whether any acquisition blocks differ.
2. Retain both mean height and p95 height in the report. State explicitly that mean height partly measures canopy openness, while p95 is the stricter upper-canopy test.
3. Map p95 residuals against LiDAR acquisition blocks, slope, forest type and stand-management information when those covariates become available.
4. Apply the frozen cohort, folds, targets and model settings to TESSERA v2. Do not change the method after opening the v2 results.
"""
    output = absolute(config["outputs"]["report"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)


def run() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    inputs = config["inputs"]
    broad = pd.read_parquet(absolute(inputs["broad_10m_sample"]))
    mature = pd.read_parquet(absolute(inputs["mature_50m_targets"])).reset_index(drop=True)
    predictions = pd.read_parquet(absolute(inputs["phase22_predictions"]))
    features = load_features(config, mature)

    summaries = {
        "target_summary": target_summary(broad, mature),
        "target_agreement": target_agreement(mature),
        "prediction_summary": prediction_summary(predictions, config["evaluation"]["scheme"]),
        "quintile_bias": quintile_bias(predictions, config["evaluation"]["scheme"]),
        "fold_summary": fold_summary(mature, absolute(inputs["fold_directory"]), int(config["evaluation"]["folds"])),
        "ridge": ridge_diagnostic(features, mature, config),
    }
    output_directory = absolute(config["outputs"]["directory"])
    output_directory.mkdir(parents=True, exist_ok=True)
    for name, frame in summaries.items():
        frame.to_csv(output_directory / f"{name}.csv", index=False)

    make_figure(broad, mature, predictions, summaries["quintile_bias"], summaries["ridge"], config)
    write_report(config, summaries)
    freeze = {
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "input_sha256": {
            key: sha256(absolute(value))
            for key, value in inputs.items()
            if key != "fold_directory"
        },
        "rows": {"broad_10m": len(broad), "mature_50m": len(mature)},
    }
    freeze_path = absolute(config["outputs"]["freeze"])
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    freeze_path.write_text(json.dumps(freeze, indent=2) + "\n")
    print(f"Wrote {config['outputs']['report']}")
    print(f"Wrote {config['outputs']['figure']}")


if __name__ == "__main__":
    run()
