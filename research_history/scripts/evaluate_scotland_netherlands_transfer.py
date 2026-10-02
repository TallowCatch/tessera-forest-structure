#!/usr/bin/env python3
"""Evaluate direct TESSERA v2 transfer between Cairngorms and Savelsbos."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/scotland_netherlands_transfer.yaml"


@dataclass
class SiteData:
    name: str
    frame: pd.DataFrame
    features: dict[str, np.ndarray]
    targets: dict[str, np.ndarray]
    folds: list[tuple[np.ndarray, np.ndarray]]
    training_blocks: np.ndarray
    evaluation_blocks: np.ndarray


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "phase32_scotland_netherlands_transfer"
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


def equal_block_weights(blocks: np.ndarray) -> np.ndarray:
    values = pd.Series(blocks)
    counts = values.value_counts()
    weights = values.map(1.0 / counts).to_numpy(dtype=np.float64)
    return weights * len(weights) / weights.sum()


def metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    blocks: np.ndarray,
) -> dict[str, float]:
    weights = equal_block_weights(blocks)
    error = predicted - observed
    block_frame = pd.DataFrame(
        {"block": blocks, "observed": observed, "predicted": predicted}
    ).groupby("block", as_index=False).mean(numeric_only=True)
    if (
        len(block_frame) >= 3
        and np.ptp(block_frame["observed"]) > 0
        and np.ptp(block_frame["predicted"]) > 0
    ):
        rank = float(
            spearmanr(block_frame["observed"], block_frame["predicted"]).statistic
        )
    else:
        rank = float("nan")
    return {
        "rmse": float(np.sqrt(np.average(np.square(error), weights=weights))),
        "mae": float(np.average(np.abs(error), weights=weights)),
        "r2": float(r2_score(observed, predicted, sample_weight=weights)),
        "bias": float(np.average(error, weights=weights)),
        "block_spearman": rank,
        "evaluation_blocks": int(len(block_frame)),
    }


def fit_ridge(
    features: np.ndarray,
    target: np.ndarray,
    blocks: np.ndarray,
    alpha: float,
) -> tuple[StandardScaler, Ridge]:
    weights = equal_block_weights(blocks)
    scaler = StandardScaler().fit(features, sample_weight=weights)
    model = Ridge(alpha=alpha).fit(
        scaler.transform(features), target, sample_weight=weights
    )
    return scaler, model


def feature_panels(summary: np.ndarray) -> dict[str, np.ndarray]:
    if summary.ndim != 2 or summary.shape[1] != 640:
        raise RuntimeError(f"Expected a 640-column summary, received {summary.shape}")
    return {
        "mean": np.asarray(summary[:, 128:256], dtype=np.float64),
        "context": np.asarray(summary[:, 128:640], dtype=np.float64),
    }


def savelsbos_feature_panels(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    groups: dict[str, list[str]] = {}
    for prefix in ["mean", "std", "dx", "dy"]:
        columns = sorted(
            column
            for column in frame
            if column.startswith(f"tessera_{prefix}_")
        )
        if len(columns) != 128:
            raise RuntimeError(f"Savelsbos {prefix} exposes {len(columns)} columns")
        groups[prefix] = columns
    return {
        "mean": frame[groups["mean"]].to_numpy(dtype=np.float64),
        "context": frame[
            groups["mean"] + groups["std"] + groups["dx"] + groups["dy"]
        ].to_numpy(dtype=np.float64),
    }


def fold_row_ids(
    cohort: pd.DataFrame,
    directory: Path,
    pattern: str,
    count: int,
) -> list[tuple[set[int], set[int]]]:
    result: list[tuple[set[int], set[int]]] = []
    for fold in range(count):
        with np.load(directory / pattern.format(fold=fold)) as values:
            train = set(
                cohort.iloc[values["train_indices"].astype(np.int64)]["row_id"]
                .astype(int)
                .tolist()
            )
            test = set(
                cohort.iloc[values["test_indices"].astype(np.int64)]["row_id"]
                .astype(int)
                .tolist()
            )
        if train & test:
            raise RuntimeError(f"Fold {fold} has overlapping train and test rows")
        result.append((train, test))
    return result


def index_folds(
    row_ids: np.ndarray,
    memberships: list[tuple[set[int], set[int]]],
) -> list[tuple[np.ndarray, np.ndarray]]:
    folds = []
    for train_ids, test_ids in memberships:
        train = np.flatnonzero(np.isin(row_ids, list(train_ids)))
        test = np.flatnonzero(np.isin(row_ids, list(test_ids)))
        if len(train) == 0 or len(test) == 0:
            raise RuntimeError("A spatial fold became empty after target harmonization")
        folds.append((train, test))
    return folds


def load_cairngorms(config: dict[str, Any]) -> SiteData:
    settings = config["cairngorms"]
    surface_path = ROOT / str(settings["surface_cohort"])
    corrected_path = ROOT / str(settings["corrected_cohort"])
    surface = pd.read_parquet(surface_path).sort_values("row_id").reset_index(drop=True)
    corrected = pd.read_parquet(corrected_path).sort_values("row_id").reset_index(drop=True)
    features = np.load(ROOT / str(settings["features"]), mmap_mode="r")
    feature_ids = np.load(ROOT / str(settings["row_ids"])).astype(np.int64)
    if not np.array_equal(surface["row_id"].to_numpy(dtype=np.int64), feature_ids):
        raise RuntimeError("Cairngorms feature rows do not match the frozen cohort")
    feature_frame = pd.DataFrame({"row_id": feature_ids, "feature_row": np.arange(len(feature_ids))})
    frame = corrected.merge(feature_frame, on="row_id", validate="one_to_one")
    mappings = config["harmonized_targets"]
    target_columns = [str(value["cairngorms"]) for value in mappings.values()]
    valid = np.isfinite(frame[target_columns].to_numpy(dtype=np.float64)).all(axis=1)
    frame = frame.loc[valid].reset_index(drop=True)
    selected = frame["feature_row"].to_numpy(dtype=np.int64)
    panels = feature_panels(np.asarray(features[selected]))
    if not all(np.isfinite(value).all() for value in panels.values()):
        raise RuntimeError("Cairngorms common TESSERA features contain non-finite values")
    memberships = fold_row_ids(
        surface,
        ROOT / str(settings["folds"]),
        str(settings["fold_pattern"]),
        int(config["model"]["folds"]),
    )
    targets = {
        name: frame[str(mapping["cairngorms"])].to_numpy(dtype=np.float64)
        for name, mapping in mappings.items()
    }
    return SiteData(
        name="cairngorms",
        frame=frame,
        features=panels,
        targets=targets,
        folds=index_folds(frame["row_id"].to_numpy(dtype=np.int64), memberships),
        training_blocks=frame[str(settings["training_block"])].to_numpy(),
        evaluation_blocks=frame[str(settings["evaluation_block"])].to_numpy(),
    )


def load_savelsbos(config: dict[str, Any]) -> SiteData:
    settings = config["savelsbos"]
    cohort_path = ROOT / str(settings["cohort"])
    cohort = pd.read_parquet(cohort_path).sort_values("row_id").reset_index(drop=True)
    features = pd.read_parquet(ROOT / str(settings["features"]))
    frame = cohort.merge(features, on="row_id", validate="one_to_one")
    mappings = config["harmonized_targets"]
    target_columns = [str(value["savelsbos"]) for value in mappings.values()]
    valid = (
        frame["tessera_context_valid"].astype(bool)
        & np.isfinite(frame[target_columns].to_numpy(dtype=np.float64)).all(axis=1)
    )
    frame = frame.loc[valid].reset_index(drop=True)
    panels = savelsbos_feature_panels(frame)
    if not all(np.isfinite(value).all() for value in panels.values()):
        raise RuntimeError("Savelsbos common TESSERA features contain non-finite values")
    memberships = fold_row_ids(
        cohort,
        ROOT / str(settings["folds"]),
        str(settings["fold_pattern"]),
        int(config["model"]["folds"]),
    )
    targets = {
        name: frame[str(mapping["savelsbos"])].to_numpy(dtype=np.float64)
        for name, mapping in mappings.items()
    }
    return SiteData(
        name="savelsbos",
        frame=frame,
        features=panels,
        targets=targets,
        folds=index_folds(frame["row_id"].to_numpy(dtype=np.int64), memberships),
        training_blocks=frame[str(settings["training_block"])].to_numpy(),
        evaluation_blocks=frame[str(settings["evaluation_block"])].to_numpy(),
    )


def local_predictions(
    site: SiteData,
    panel: str,
    target_name: str,
    alpha: float,
) -> pd.DataFrame:
    parts = []
    for fold, (train, test) in enumerate(site.folds):
        scaler, model = fit_ridge(
            site.features[panel][train],
            site.targets[target_name][train],
            site.training_blocks[train],
            alpha,
        )
        predicted = model.predict(scaler.transform(site.features[panel][test]))
        parts.append(
            pd.DataFrame(
                {
                    "row_id": site.frame.iloc[test]["row_id"].to_numpy(),
                    "fold": fold,
                    "training_site": site.name,
                    "evaluation_site": site.name,
                    "feature_panel": panel,
                    "target": target_name,
                    "model": "local_cv",
                    "observed": site.targets[target_name][test],
                    "predicted": predicted,
                    "evaluation_block": site.evaluation_blocks[test],
                }
            )
        )
    result = pd.concat(parts, ignore_index=True)
    if result["row_id"].duplicated().any() or len(result) != len(site.frame):
        raise RuntimeError(f"Local predictions do not cover {site.name} exactly once")
    return result


def transfer_predictions(
    source: SiteData,
    target: SiteData,
    panel: str,
    target_name: str,
    alpha: float,
) -> pd.DataFrame:
    scaler, model = fit_ridge(
        source.features[panel],
        source.targets[target_name],
        source.training_blocks,
        alpha,
    )
    predicted = model.predict(scaler.transform(target.features[panel]))
    source_weights = equal_block_weights(source.training_blocks)
    source_mean = float(np.average(source.targets[target_name], weights=source_weights))
    base = {
        "row_id": target.frame["row_id"].to_numpy(),
        "fold": -1,
        "training_site": source.name,
        "evaluation_site": target.name,
        "feature_panel": panel,
        "target": target_name,
        "observed": target.targets[target_name],
        "evaluation_block": target.evaluation_blocks,
    }
    transfer = pd.DataFrame(
        {**base, "model": "transfer", "predicted": predicted}
    )
    baseline = pd.DataFrame(
        {
            **base,
            "model": "source_mean",
            "predicted": np.full(len(target.frame), source_mean, dtype=np.float64),
        }
    )
    return pd.concat([transfer, baseline], ignore_index=True)


def summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = [
        "training_site", "evaluation_site", "feature_panel", "target", "model"
    ]
    for keys, group in predictions.groupby(group_columns, sort=True):
        values = metrics(
            group["observed"].to_numpy(dtype=np.float64),
            group["predicted"].to_numpy(dtype=np.float64),
            group["evaluation_block"].to_numpy(),
        )
        rows.append(
            dict(zip(group_columns, keys), rows=len(group), **values)
        )
    return pd.DataFrame(rows)


def bootstrap_transfer_vs_local(
    predictions: pd.DataFrame,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for (site, panel, target), scoped in predictions.groupby(
        ["evaluation_site", "feature_panel", "target"], sort=True
    ):
        local = scoped[
            (scoped["training_site"] == site) & (scoped["model"] == "local_cv")
        ][["row_id", "observed", "predicted", "evaluation_block"]].rename(
            columns={"predicted": "local"}
        )
        transfer = scoped[
            (scoped["training_site"] != site) & (scoped["model"] == "transfer")
        ][["row_id", "predicted"]].rename(columns={"predicted": "transfer"})
        paired = local.merge(transfer, on="row_id", validate="one_to_one")
        blocks = (
            paired.assign(
                local_error=np.square(paired["local"] - paired["observed"]),
                transfer_error=np.square(paired["transfer"] - paired["observed"]),
            )
            .groupby("evaluation_block", as_index=False)[
                ["local_error", "transfer_error"]
            ]
            .mean()
        )
        local_error = blocks["local_error"].to_numpy()
        transfer_error = blocks["transfer_error"].to_numpy()
        draws = np.empty(replicates, dtype=np.float64)
        for index in range(replicates):
            selected = rng.integers(0, len(blocks), size=len(blocks))
            draws[index] = np.sqrt(transfer_error[selected].mean()) - np.sqrt(
                local_error[selected].mean()
            )
        rows.append(
            {
                "evaluation_site": site,
                "feature_panel": panel,
                "target": target,
                "delta_rmse_transfer_minus_local": float(
                    np.sqrt(transfer_error.mean()) - np.sqrt(local_error.mean())
                ),
                "ci_low": float(np.quantile(draws, 0.025)),
                "ci_high": float(np.quantile(draws, 0.975)),
                "evaluation_blocks": len(blocks),
            }
        )
    return pd.DataFrame(rows)


def target_distributions(sites: list[SiteData]) -> pd.DataFrame:
    rows = []
    for site in sites:
        weights = equal_block_weights(site.evaluation_blocks)
        for name, values in site.targets.items():
            mean = float(np.average(values, weights=weights))
            sd = float(np.sqrt(np.average(np.square(values - mean), weights=weights)))
            rows.append(
                {
                    "site": site.name,
                    "target": name,
                    "rows": len(values),
                    "blocks": len(np.unique(site.evaluation_blocks)),
                    "mean": mean,
                    "sd": sd,
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                }
            )
    return pd.DataFrame(rows)


def make_score_figure(metrics_frame: pd.DataFrame, path: Path) -> None:
    labels = {
        "mean_height_m": "Mean height",
        "p95_height_m": "P95 height",
        "within_cell_height_sd_m": "Height SD",
        "height_cv": "Height CV",
    }
    targets = list(labels)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3), sharey=True)
    colors = {"local_cv": "#3b82c4", "transfer": "#d95f02", "source_mean": "#777777"}
    for axis, site in zip(axes, ["savelsbos", "cairngorms"]):
        scoped = metrics_frame[
            (metrics_frame["evaluation_site"] == site)
            & (metrics_frame["feature_panel"] == "context")
        ]
        positions = np.arange(len(targets))
        width = 0.24
        local_values = {
            target: float(
                scoped[
                    (scoped["target"] == target)
                    & (scoped["model"] == "local_cv")
                    & (scoped["training_site"] == site)
                ]["rmse"].iloc[0]
            )
            for target in targets
        }
        for offset, model in zip([-1, 0, 1], ["local_cv", "transfer", "source_mean"]):
            values = []
            for target in targets:
                selected = scoped[
                    (scoped["target"] == target) & (scoped["model"] == model)
                ]
                if model == "local_cv":
                    selected = selected[selected["training_site"] == site]
                else:
                    selected = selected[selected["training_site"] != site]
                values.append(float(selected["rmse"].iloc[0]) / local_values[target])
            axis.bar(
                positions + offset * width,
                values,
                width,
                label={"local_cv": "Local spatial CV", "transfer": "Cross-landscape transfer", "source_mean": "Source mean"}[model],
                color=colors[model],
            )
        axis.axhline(1, color="black", linewidth=0.8, linestyle="--")
        axis.set_xticks(positions, [labels[target] for target in targets], rotation=25, ha="right")
        axis.set_title(
            "Scotland to Savelsbos" if site == "savelsbos" else "Savelsbos to Scotland",
            fontweight="bold",
        )
        axis.grid(axis="y", color="#dddddd", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.tick_params(axis="y", labelleft=True)
        axis.set_ylabel("Relative RMSE\n(local spatial CV = 1)")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=3, frameon=False)
    fig.subplots_adjust(top=0.80, bottom=0.25, left=0.08, right=0.98, wspace=0.29)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_scatter_figure(predictions: pd.DataFrame, metrics_frame: pd.DataFrame, path: Path) -> None:
    labels = {
        "mean_height_m": "Mean height",
        "p95_height_m": "P95 height",
        "within_cell_height_sd_m": "Height SD",
        "height_cv": "Height CV",
    }
    fig, axes = plt.subplots(2, 4, figsize=(13.4, 6.4))
    for row, site in enumerate(["savelsbos", "cairngorms"]):
        for column, target in enumerate(labels):
            axis = axes[row, column]
            scoped = predictions[
                (predictions["evaluation_site"] == site)
                & (predictions["training_site"] != site)
                & (predictions["feature_panel"] == "context")
                & (predictions["target"] == target)
                & (predictions["model"] == "transfer")
            ]
            observed = scoped["observed"].to_numpy()
            predicted = scoped["predicted"].to_numpy()
            if len(scoped) > 2000:
                axis.hexbin(observed, predicted, gridsize=40, bins="log", mincnt=1, cmap="viridis")
            else:
                axis.scatter(observed, predicted, s=11, alpha=0.55, color="#2878b5", edgecolors="none")
            lower = float(min(observed.min(), predicted.min()))
            upper = float(max(observed.max(), predicted.max()))
            axis.plot([lower, upper], [lower, upper], "--", color="#cc3333", linewidth=1)
            value = metrics_frame[
                (metrics_frame["evaluation_site"] == site)
                & (metrics_frame["training_site"] != site)
                & (metrics_frame["feature_panel"] == "context")
                & (metrics_frame["target"] == target)
                & (metrics_frame["model"] == "transfer")
            ].iloc[0]
            axis.text(
                0.04,
                0.96,
                rf"$R^2$={value.r2:.3f}" + "\n" + rf"RMSE={value.rmse:.3f}" + "\n" + rf"$r_s$={value.block_spearman:.3f}",
                transform=axis.transAxes,
                va="top",
                fontsize=8.5,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2.5},
            )
            if row == 0:
                axis.set_title(labels[target], fontweight="bold")
            if column == 0:
                axis.set_ylabel(
                    ("Scotland to Savelsbos\n" if row == 0 else "Savelsbos to Scotland\n")
                    + "Predicted"
                )
            if row == 1:
                axis.set_xlabel("Observed")
            axis.grid(color="#e5e5e5", linewidth=0.6)
            axis.set_axisbelow(True)
    fig.subplots_adjust(top=0.93, bottom=0.10, left=0.08, right=0.99, hspace=0.30, wspace=0.26)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(
    path: Path,
    metrics_frame: pd.DataFrame,
    bootstrap: pd.DataFrame,
    distributions: pd.DataFrame,
) -> None:
    primary = metrics_frame[metrics_frame["feature_panel"] == "context"].copy()
    lines = [
        "# Direct Scotland-Netherlands TESSERA v2 transfer", "",
        "This experiment trains a Ridge model in one landscape and applies it to the other without using target-landscape outcomes for fitting, scaling or model selection. It uses the common 512-dimensional v2 summary and four approximately harmonized 50-m LiDAR outcomes.",
        "", "## Target distributions", "", "```text",
        distributions.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "```", "", "## Primary context-model results", "", "```text",
        primary.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "```", "", "## Transfer minus local RMSE", "", "```text",
        bootstrap[bootstrap["feature_panel"] == "context"].to_string(
            index=False, float_format=lambda value: f"{value:.4f}"
        ),
        "```", "",
        "The local spatial-CV rows measure within-landscape performance. The transfer rows are complete-landscape holdouts. Negative transfer R2 means the imported model is less accurate than predicting the target-landscape mean. Block Spearman reports whether spatial ordering is retained despite an absolute offset or scale error.", "",
        "Target definitions are closely matched but originate from different LiDAR processing systems. Results therefore combine ecological domain shift with differences in acquisition and product construction.", "",
    ]
    atomic(path, lambda p: p.write_text("\n".join(lines), encoding="utf-8"))


def run() -> None:
    config = load_config()
    outputs = {key: ROOT / str(value) for key, value in config["outputs"].items()}
    if outputs["result_freeze"].exists():
        raise RuntimeError("Phase 32 result is already frozen")
    cairngorms = load_cairngorms(config)
    savelsbos = load_savelsbos(config)
    sites = [cairngorms, savelsbos]
    alpha = float(config["model"]["alpha"])
    parts = []
    for panel in config["features"]["panels"]:
        for target_name in config["harmonized_targets"]:
            for site in sites:
                parts.append(local_predictions(site, panel, target_name, alpha))
            parts.append(
                transfer_predictions(cairngorms, savelsbos, panel, target_name, alpha)
            )
            parts.append(
                transfer_predictions(savelsbos, cairngorms, panel, target_name, alpha)
            )
    predictions = pd.concat(parts, ignore_index=True)
    metric_frame = summarize_predictions(predictions)
    bootstrap = bootstrap_transfer_vs_local(
        predictions,
        int(config["model"]["bootstrap_replicates"]),
        int(config["model"]["bootstrap_seed"]),
    )
    distributions = target_distributions(sites)
    atomic(outputs["predictions"], lambda p: predictions.to_parquet(p, index=False))
    atomic(outputs["metrics"], lambda p: metric_frame.to_csv(p, index=False))
    atomic(outputs["bootstrap"], lambda p: bootstrap.to_csv(p, index=False))
    atomic(outputs["distributions"], lambda p: distributions.to_csv(p, index=False))
    make_score_figure(metric_frame, outputs["figure_scores"])
    make_scatter_figure(predictions, metric_frame, outputs["figure_scatter"])
    write_report(outputs["report"], metric_frame, bootstrap, distributions)
    artifacts = {
        str(path.relative_to(ROOT)): sha256(path)
        for path in outputs.values()
        if path.exists() and path != outputs["result_freeze"]
    }
    inputs = [
        CONFIG_PATH,
        ROOT / str(config["cairngorms"]["surface_cohort"]),
        ROOT / str(config["cairngorms"]["corrected_cohort"]),
        ROOT / str(config["cairngorms"]["features"]),
        ROOT / str(config["cairngorms"]["row_ids"]),
        ROOT / str(config["savelsbos"]["cohort"]),
        ROOT / str(config["savelsbos"]["features"]),
    ]
    freeze = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation": config["study"]["interpretation"],
        "inputs": {str(path.relative_to(ROOT)): sha256(path) for path in inputs},
        "artifacts": artifacts,
    }
    outputs["result_freeze"].parent.mkdir(parents=True, exist_ok=True)
    outputs["result_freeze"].write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("Phase 32 Scotland-Netherlands transfer complete")
    primary = metric_frame[
        (metric_frame["feature_panel"] == config["features"]["primary_panel"])
        & (metric_frame["model"].isin(["local_cv", "transfer"]))
    ]
    print(primary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    run()
