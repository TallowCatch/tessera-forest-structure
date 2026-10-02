#!/usr/bin/env python3
"""Strengthen the Scotland airborne-LiDAR evidence with frozen follow-ups."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import yaml
from openpyxl import load_workbook
from pyproj import Transformer
from rasterio.transform import rowcol
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_tessera_alignment as tessera  # noqa: E402
import arran_lidar_core as core  # noqa: E402
import run_scotland_lidar as phase14  # noqa: E402
import run_arran_lidar as phase15  # noqa: E402


CONFIG_PATH = ROOT / "configs/scotland_evidence.yaml"
PROTOCOL_PATH = (
    ROOT / "metadata/project_config_phase16_scotland_evidence_protocol_freeze.yaml"
)
STATUS_PATH = ROOT / "outputs/reports/phase16_scotland_evidence_status.json"
AUDIT_PATH = ROOT / "metadata/phase16_lidar_metric_comparability_audit.json"
AUDIT_REPORT_PATH = ROOT / "outputs/reports/phase16_lidar_metric_comparability.md"
RECIPROCAL_METRICS_PATH = (
    ROOT / "outputs/tables/phase16_arran_to_cairngorms_metrics.csv"
)
RECIPROCAL_PREDICTIONS_PATH = (
    ROOT / "data/processed/phase16_arran_to_cairngorms_predictions.parquet"
)
RECIPROCAL_FREEZE_PATH = (
    ROOT / "metadata/phase16_arran_to_cairngorms_freeze.json"
)
BOOTSTRAP_PATH = ROOT / "outputs/tables/phase16_spatial_block_bootstrap.csv"
BOOTSTRAP_SUMMARY_PATH = (
    ROOT / "outputs/tables/phase16_spatial_block_bootstrap_summary.csv"
)
DIAGNOSTIC_CORRELATION_PATH = (
    ROOT / "outputs/tables/phase16_arran_residual_correlations.csv"
)
DIAGNOSTIC_BIN_PATH = (
    ROOT / "outputs/tables/phase16_arran_residual_quantiles.csv"
)
POPULATION_INDEX_PATH = (
    ROOT / "data/interim/phase16_arran_population_index.parquet"
)
MAP_CHUNK_DIR = ROOT / "data/interim/phase16_arran_map_chunks"
MAP_STREAM_DIR = ROOT / "data/interim/phase16_arran_tessera_stream"
MAP_PATH = ROOT / "data/processed/phase16_arran_cross_fitted_map.parquet"
MAP_METRICS_PATH = (
    ROOT / "outputs/tables/phase16_arran_full_population_metrics.csv"
)
MAP_MANIFEST_PATH = ROOT / "metadata/phase16_arran_map_manifest.json"
FIGURE_PATH = ROOT / "outputs/figures/phase16_scotland_evidence.png"
REPORT_PATH = ROOT / "outputs/reports/phase16_scotland_evidence.md"
RESULT_FREEZE_PATH = ROOT / "metadata/phase16_scotland_evidence_freeze.json"
TESSERA_FEATURES = [f"tessera_{index:03d}" for index in range(128)]
STAGES = ["audit", "reciprocal", "bootstrap", "diagnostics", "full_map", "report"]


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def load_config() -> dict[str, Any]:
    if CONFIG_PATH.read_bytes() != PROTOCOL_PATH.read_bytes():
        raise RuntimeError("The active Phase 16 config differs from its protocol freeze")
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "phase16_scotland_evidence"
    ]


def update_status(stage: str, state: str, detail: dict[str, Any] | None = None) -> None:
    value = {
        "updated_utc": utc_now(),
        "stage": stage,
        "state": state,
        "stages": STAGES,
    }
    if detail is not None:
        value["detail"] = detail
    core.atomic_json(STATUS_PATH, value)


def input_path(config: dict[str, Any], name: str) -> Path:
    value = Path(config["inputs"][name])
    return value if value.is_absolute() else ROOT / value


def validate_phase15_freeze(config: dict[str, Any]) -> dict[str, Any]:
    freeze_path = input_path(config, "phase15_result_freeze")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    for relative, expected_hash in freeze["files"].items():
        path = ROOT / relative
        if not path.exists() or sha256(path) != expected_hash:
            raise RuntimeError(f"Frozen Phase 15 input changed: {relative}")
    return freeze


def run_audit(config: dict[str, Any]) -> None:
    if AUDIT_PATH.exists() and AUDIT_REPORT_PATH.exists():
        print("audit: validated existing comparability audit", flush=True)
        return
    validate_phase15_freeze(config)
    workbook_path = input_path(config, "cairngorms_workbook")
    workbook = load_workbook(workbook_path, read_only=False, data_only=True)
    worksheet = workbook["LiDAR data"]
    workbook_record: dict[str, Any] | None = None
    headers = [cell.value for cell in worksheet[1]]
    for row in worksheet.iter_rows(min_row=2, values_only=True):
        record = dict(zip(headers, row, strict=False))
        if record.get("Code") == config["target"]["name"]:
            workbook_record = {
                key: value
                for key, value in record.items()
                if value is not None and key is not None
            }
            break
    if workbook_record is None:
        raise RuntimeError("The Cairngorms workbook does not list the primary target")
    phase14_inventory = json.loads(
        (ROOT / "metadata/phase14_scotland_lidar_inventory.json").read_text()
    )
    source_code = Path("/tmp/lidarSHM/R/LiDAR_metrics.R")
    helper_code = Path("/tmp/lidarSHM/R/roughness_metrics_f.R")
    code_available = source_code.exists() and helper_code.exists()
    audit = {
        "created_utc": utc_now(),
        "target": config["target"]["name"],
        "verified_matches": {
            "metric_name": config["target"]["name"]
            in phase14_inventory["profile"]["bands"],
            "spatial_resolution_m": phase14_inventory["profile"]["resolution_m"]
            == [10.0, 10.0],
            "height_breaks_m": config["target"]["lidarshm_height_breaks_m"],
            "entropy_formula": "negative_sum_p_log_p",
        },
        "cairngorms_workbook": {
            "path": str(workbook_path),
            "sha256": sha256(workbook_path),
            "record": workbook_record,
        },
        "inspected_lidarshm": {
            "commit": config["comparability"]["lidarshm_commit"],
            "source_available_during_audit": code_available,
            "LiDAR_metrics_R_sha256": sha256(source_code)
            if source_code.exists()
            else None,
            "roughness_metrics_f_R_sha256": sha256(helper_code)
            if helper_code.exists()
            else None,
            "implementation_observation": config["comparability"][
                "lidarshm_implementation_observation"
            ],
        },
        "unresolved": [
            (
                "The workbook says the Shannon metric uses canopy points above "
                "1.3 m, while the inspected lidarSHM commit passes the full LAS "
                "object to roughness_metrics_f."
            ),
            (
                "The repository metadata does not identify the exact lidarSHM "
                "commit and point-filtering path used to create LiDAR_metrics_2.tif."
            ),
            (
                "Cairngorms heights use a supplied precomputed ground-normalized "
                "product; Arran heights use the paired 2025 50 cm terrain raster."
            ),
        ],
        "status": "partially_verified_cross_landscape_target",
        "claim_boundary": (
            "Within-landscape evaluations are unaffected. Cross-landscape "
            "transfer remains provisional until the Cairngorms production "
            "parameters or raw point clouds are confirmed."
        ),
    }
    core.atomic_json(AUDIT_PATH, audit)
    AUDIT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_REPORT_PATH.write_text(
        "\n".join(
            [
                "# LiDAR target comparability audit",
                "",
                "The Arran and Cairngorms products share the target name, 10 m "
                "resolution, height classes and entropy formula.",
                "",
                "The audit found a documentation-versus-code ambiguity. The "
                "Cairngorms workbook describes canopy points above 1.3 m, while "
                "the inspected lidarSHM implementation passes the full point "
                "cloud to the Shannon helper. The exact production commit for "
                "the Cairngorms raster is not recorded locally.",
                "",
                "Consequently, within-Arran results can be interpreted directly. "
                "Transfer between Cairngorms and Arran is retained as provisional "
                "evidence and must disclose the unresolved reference-processing "
                "difference.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print("audit: recorded partial target comparability", flush=True)


def run_reciprocal(config: dict[str, Any]) -> None:
    if (
        RECIPROCAL_METRICS_PATH.exists()
        and RECIPROCAL_PREDICTIONS_PATH.exists()
        and RECIPROCAL_FREEZE_PATH.exists()
    ):
        print("reciprocal: validated existing outputs", flush=True)
        return
    phase15_config = phase15.load_config()
    source = phase15._joined_landscape(
        input_path(config, "arran_sample"),
        input_path(config, "arran_tessera"),
        input_path(config, "arran_conventional"),
    )
    target = phase15._joined_landscape(
        input_path(config, "cairngorms_sample"),
        input_path(config, "cairngorms_tessera"),
        input_path(config, "cairngorms_conventional"),
    )
    terrain, sentinel = phase15._conventional_features(source)
    if phase15._conventional_features(target) != (terrain, sentinel):
        raise RuntimeError("Predictor schemas differ between landscapes")
    target_name = config["target"]["name"]
    source = source.loc[
        phase15._common_valid_mask(
            source, target_name, terrain, sentinel, phase15_config
        )
    ].copy()
    target = target.loc[
        phase15._common_valid_mask(
            target, target_name, terrain, sentinel, phase15_config
        )
    ].copy()
    weights = phase14.spatial_weights(source["spatial_block"])
    predictions = phase15._model_predictions(
        train=source,
        test=target,
        train_y=source[target_name].to_numpy(dtype=np.float64),
        weights=weights,
        terrain_features=terrain,
        sentinel_features=sentinel,
        config=phase15_config,
        nonlinear=True,
    )
    selected_models = config["reciprocal_transfer"]["models"]
    output = target[
        ["row_id", "bng_x", "bng_y", "spatial_region", "spatial_block"]
    ].copy()
    output["observed"] = target[target_name].to_numpy(dtype=np.float64)
    for model in selected_models:
        output[model] = predictions[model]
    records: list[dict[str, Any]] = []
    scopes = [("all_cairngorms", np.ones(len(output), dtype=bool))]
    scopes.extend(
        (
            f"cairngorms_region_{region}",
            output["spatial_region"].to_numpy() == region,
        )
        for region in sorted(output["spatial_region"].unique())
    )
    for scope, mask in scopes:
        for model in selected_models:
            records.append(
                {
                    "experiment": "arran_2025_to_cairngorms_2023",
                    "target": target_name,
                    "model": model,
                    "scope": scope,
                    "train_rows": len(source),
                    "test_rows": int(mask.sum()),
                    **phase14.metric_values(
                        output.loc[mask, "observed"].to_numpy(),
                        output.loc[mask, model].to_numpy(),
                    ),
                }
            )
    write_csv(pd.DataFrame(records), RECIPROCAL_METRICS_PATH)
    core.atomic_parquet(output, RECIPROCAL_PREDICTIONS_PATH)
    core.atomic_json(
        RECIPROCAL_FREEZE_PATH,
        {
            "created_utc": utc_now(),
            "interpretation": config["reciprocal_transfer"]["interpretation"],
            "source_rows": len(source),
            "target_rows": len(target),
            "input_hashes": {
                name: sha256(input_path(config, name))
                for name in [
                    "arran_sample",
                    "arran_tessera",
                    "arran_conventional",
                    "cairngorms_sample",
                    "cairngorms_tessera",
                    "cairngorms_conventional",
                ]
            },
            "metrics_sha256": sha256(RECIPROCAL_METRICS_PATH),
            "predictions_sha256": sha256(RECIPROCAL_PREDICTIONS_PATH),
        },
    )
    print(
        f"reciprocal: evaluated {len(source):,} Arran cells against "
        f"{len(target):,} Cairngorms cells",
        flush=True,
    )


def block_bootstrap_metrics(
    frame: pd.DataFrame,
    models: list[str],
    *,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    """Bootstrap complete spatial blocks using additive metric sufficient statistics."""
    records = []
    for block, group in frame.groupby("spatial_block", sort=True):
        observed = group["observed"].to_numpy(dtype=np.float64)
        record: dict[str, Any] = {
            "spatial_block": block,
            "n": len(group),
            "sum_y": observed.sum(),
            "sum_y2": np.square(observed).sum(),
        }
        for model in models:
            predicted = group[model].to_numpy(dtype=np.float64)
            residual = predicted - observed
            record[f"{model}__sse"] = np.square(residual).sum()
            record[f"{model}__sae"] = np.abs(residual).sum()
            record[f"{model}__sum_p"] = predicted.sum()
        records.append(record)
    blocks = pd.DataFrame(records)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(blocks), size=(replicates, len(blocks)))
    output = []
    n = blocks["n"].to_numpy(dtype=np.float64)[draws].sum(axis=1)
    sum_y = blocks["sum_y"].to_numpy(dtype=np.float64)[draws].sum(axis=1)
    sum_y2 = blocks["sum_y2"].to_numpy(dtype=np.float64)[draws].sum(axis=1)
    sst = sum_y2 - np.square(sum_y) / n
    for model in models:
        sse = blocks[f"{model}__sse"].to_numpy(dtype=np.float64)[draws].sum(axis=1)
        sae = blocks[f"{model}__sae"].to_numpy(dtype=np.float64)[draws].sum(axis=1)
        sum_p = blocks[f"{model}__sum_p"].to_numpy(dtype=np.float64)[draws].sum(axis=1)
        output.append(
            pd.DataFrame(
                {
                    "replicate": np.arange(replicates, dtype=np.int32),
                    "model": model,
                    "rmse": np.sqrt(sse / n),
                    "mae": sae / n,
                    "r2": 1.0 - sse / sst,
                    "bias": (sum_p - sum_y) / n,
                }
            )
        )
    return pd.concat(output, ignore_index=True)


def run_bootstrap(config: dict[str, Any]) -> None:
    if BOOTSTRAP_PATH.exists() and BOOTSTRAP_SUMMARY_PATH.exists():
        print("bootstrap: validated existing outputs", flush=True)
        return
    datasets = {
        "arran_buffered_spatial": pd.read_parquet(
            input_path(config, "arran_oof_predictions")
        ),
        "cairngorms_to_arran": pd.read_parquet(
            input_path(config, "cairngorms_to_arran_predictions")
        ),
        "arran_to_cairngorms": pd.read_parquet(
            RECIPROCAL_PREDICTIONS_PATH
        ),
    }
    models = [
        "training_mean",
        "sentinel2_terrain_nonlinear",
        "tessera_ridge",
        "tessera_nonlinear",
    ]
    replicates = int(config["uncertainty"]["replicates"])
    seed = int(config["uncertainty"]["seed"])
    outputs = []
    for position, (experiment, frame) in enumerate(datasets.items()):
        available = [model for model in models if model in frame]
        result = block_bootstrap_metrics(
            frame,
            available,
            replicates=replicates,
            seed=seed + position,
        )
        result.insert(0, "experiment", experiment)
        outputs.append(result)
        print(
            f"bootstrap: completed {replicates:,} block replicates for {experiment}",
            flush=True,
        )
    bootstrap = pd.concat(outputs, ignore_index=True)
    write_csv(bootstrap, BOOTSTRAP_PATH)
    interval = float(config["uncertainty"]["confidence_interval_percent"])
    tail = (100.0 - interval) / 2.0
    summaries = []
    for (experiment, model), group in bootstrap.groupby(
        ["experiment", "model"], sort=True
    ):
        for metric in ["rmse", "mae", "r2", "bias"]:
            summaries.append(
                {
                    "experiment": experiment,
                    "model": model,
                    "metric": metric,
                    "estimate": float(group[metric].mean()),
                    "ci_low": float(np.percentile(group[metric], tail)),
                    "ci_high": float(np.percentile(group[metric], 100 - tail)),
                }
            )
    comparison = config["uncertainty"]["primary_comparison"]
    for experiment, group in bootstrap.groupby("experiment", sort=True):
        left = group[group["model"].eq(comparison["model"])].set_index("replicate")
        right = group[group["model"].eq(comparison["reference"])].set_index(
            "replicate"
        )
        if left.empty or right.empty:
            continue
        delta = left["rmse"] - right["rmse"]
        summaries.append(
            {
                "experiment": experiment,
                "model": (
                    f"{comparison['model']}_minus_{comparison['reference']}"
                ),
                "metric": "paired_rmse_difference",
                "estimate": float(delta.mean()),
                "ci_low": float(np.percentile(delta, tail)),
                "ci_high": float(np.percentile(delta, 100 - tail)),
                "probability_below_zero": float((delta < 0).mean()),
            }
        )
    write_csv(pd.DataFrame(summaries), BOOTSTRAP_SUMMARY_PATH)


def run_diagnostics(config: dict[str, Any]) -> None:
    if DIAGNOSTIC_CORRELATION_PATH.exists() and DIAGNOSTIC_BIN_PATH.exists():
        print("diagnostics: validated existing outputs", flush=True)
        return
    sample = pd.read_parquet(input_path(config, "arran_sample"))
    conventional = pd.read_parquet(input_path(config, "arran_conventional"))
    predictions = pd.read_parquet(input_path(config, "arran_oof_predictions"))
    frame = predictions.merge(
        sample, on=["row_id", "bng_x", "bng_y", "spatial_block"], validate="one_to_one"
    ).merge(
        conventional, on="row_id", validate="one_to_one"
    )
    model = config["diagnostics"]["model"]
    frame["residual"] = frame[model] - frame["observed"]
    frame["absolute_residual"] = frame["residual"].abs()
    correlation_records = []
    bin_records = []
    for variable in config["diagnostics"]["variables"]:
        for response in ["residual", "absolute_residual"]:
            valid = frame[[variable, response]].dropna()
            correlation_records.append(
                {
                    "variable": variable,
                    "response": response,
                    "spearman_r": float(
                        spearmanr(valid[variable], valid[response]).statistic
                    ),
                    "rows": len(valid),
                }
            )
        groups = pd.qcut(
            frame[variable],
            q=int(config["diagnostics"]["quantile_groups"]),
            labels=False,
            duplicates="drop",
        )
        for quantile, scoped in frame.groupby(groups, observed=True):
            metrics = phase14.metric_values(
                scoped["observed"].to_numpy(), scoped[model].to_numpy()
            )
            bin_records.append(
                {
                    "variable": variable,
                    "quantile": int(quantile) + 1,
                    "minimum": float(scoped[variable].min()),
                    "maximum": float(scoped[variable].max()),
                    "rows": len(scoped),
                    **metrics,
                }
            )
    write_csv(pd.DataFrame(correlation_records), DIAGNOSTIC_CORRELATION_PATH)
    write_csv(pd.DataFrame(bin_records), DIAGNOSTIC_BIN_PATH)
    print("diagnostics: wrote residual associations and quantile errors", flush=True)


def derive_spatial_region_rule(
    sample: pd.DataFrame,
    *,
    block_size: int,
    region_count: int,
) -> dict[str, np.ndarray]:
    blocks = (
        sample.groupby(["block_x", "block_y"], as_index=False)
        .size()
        .rename(columns={"size": "sample_rows"})
    )
    centres = np.column_stack(
        [
            (blocks["block_x"].to_numpy() + 0.5) * block_size,
            (blocks["block_y"].to_numpy() + 0.5) * block_size,
        ]
    )
    weights = blocks["sample_rows"].to_numpy(dtype=np.float64)
    centre = np.average(centres, axis=0, weights=weights)
    centred = centres - centre
    covariance = np.cov(centred.T, aweights=weights)
    _, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, -1]
    if axis[0] < 0:
        axis *= -1
    score = centred @ axis
    boundaries = np.quantile(
        np.repeat(score, weights.astype(np.int64)),
        np.arange(1, region_count) / region_count,
    )
    predicted = np.searchsorted(boundaries, score, side="right")
    expected = (
        sample.groupby(["block_x", "block_y"])["spatial_region"]
        .first()
        .reindex(pd.MultiIndex.from_frame(blocks[["block_x", "block_y"]]))
        .to_numpy()
    )
    if not np.array_equal(predicted, expected):
        raise RuntimeError("Could not reconstruct the frozen spatial-region rule")
    return {"centre": centre, "axis": axis, "boundaries": boundaries}


def apply_spatial_region_rule(
    frame: pd.DataFrame,
    rule: dict[str, np.ndarray],
    *,
    block_size: int,
) -> np.ndarray:
    centres = np.column_stack(
        [
            (frame["block_x"].to_numpy() + 0.5) * block_size,
            (frame["block_y"].to_numpy() + 0.5) * block_size,
        ]
    )
    score = (centres - rule["centre"]) @ rule["axis"]
    return np.searchsorted(rule["boundaries"], score, side="right").astype(
        np.int16
    )


def prepare_population(config: dict[str, Any]) -> pd.DataFrame:
    if POPULATION_INDEX_PATH.exists():
        population = pd.read_parquet(POPULATION_INDEX_PATH)
        if len(population) == int(config["full_map"]["expected_population_rows"]):
            return population
        raise RuntimeError("The population index has an unexpected row count")
    tile_root = ROOT / config["full_map"]["population_metric_tiles"]
    columns = [
        "source_tile",
        "bng_x",
        "bng_y",
        config["full_map"]["target"],
        "lidar_meanH",
        "lidar_maxH",
        "lidar_Cov",
    ]
    frames = []
    for path in sorted(tile_root.glob("*.parquet")):
        frame = pd.read_parquet(path)
        missing = set(columns) - set(frame)
        if missing:
            raise RuntimeError(f"{path.name} lacks {sorted(missing)}")
        frames.append(frame[columns])
    population = pd.concat(frames, ignore_index=True).sort_values(
        ["bng_x", "bng_y"]
    )
    population = population.reset_index(drop=True)
    if len(population) != int(config["full_map"]["expected_population_rows"]):
        raise RuntimeError("The eligible Arran population count changed")
    if population.duplicated(["bng_x", "bng_y"]).any():
        raise RuntimeError("Duplicate cells occur in the Arran population")
    population.insert(0, "population_id", np.arange(len(population), dtype=np.int64))
    block_size = int(phase15.load_config()["spatial_evaluation"]["block_size_m"])
    population["block_x"] = np.floor(population["bng_x"] / block_size).astype(
        np.int32
    )
    population["block_y"] = np.floor(population["bng_y"] / block_size).astype(
        np.int32
    )
    population["spatial_block"] = (
        population["block_x"].astype(str)
        + "_"
        + population["block_y"].astype(str)
    )
    sample = pd.read_parquet(input_path(config, "arran_sample"))
    rule = derive_spatial_region_rule(
        sample,
        block_size=block_size,
        region_count=int(
            phase15.load_config()["spatial_evaluation"]["region_count"]
        ),
    )
    unique_blocks = population[
        ["block_x", "block_y", "spatial_block"]
    ].drop_duplicates()
    unique_blocks["spatial_region"] = apply_spatial_region_rule(
        unique_blocks, rule, block_size=block_size
    )
    population = population.merge(
        unique_blocks[["spatial_block", "spatial_region"]],
        on="spatial_block",
        validate="many_to_one",
    )
    transformer = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
    longitude, latitude = transformer.transform(
        population["bng_x"].to_numpy(),
        population["bng_y"].to_numpy(),
    )
    population["longitude"] = longitude
    population["latitude"] = latitude
    population["tessera_tile"] = [
        tessera.tile_name(tessera.tile_from_world(float(lon), float(lat)))
        for lon, lat in zip(longitude, latitude, strict=True)
    ]
    core.atomic_parquet(population, POPULATION_INDEX_PATH)
    return population


def train_cross_fitted_models(
    config: dict[str, Any],
) -> dict[int, Any]:
    phase15_config = phase15.load_config()
    frame = phase15._joined_landscape(
        input_path(config, "arran_sample"),
        input_path(config, "arran_tessera"),
        input_path(config, "arran_conventional"),
    )
    terrain, sentinel = phase15._conventional_features(frame)
    target = config["full_map"]["target"]
    valid = phase15._common_valid_mask(
        frame, target, terrain, sentinel, phase15_config
    )
    models = {}
    for fold in range(
        int(phase15_config["spatial_evaluation"]["region_count"])
    ):
        train_mask, _, _ = phase14.buffered_fold_masks(
            frame, fold, phase15_config
        )
        train = frame.loc[train_mask & valid]
        weights = phase14.spatial_weights(train["spatial_block"])
        models[fold] = phase14.fit_histogram_model(
            train[TESSERA_FEATURES].to_numpy(dtype=np.float32),
            train[target].to_numpy(dtype=np.float64),
            weights,
            phase15_config,
        )
        print(
            f"full_map: trained frozen fold model {fold + 1}/5 "
            f"on {len(train):,} cells",
            flush=True,
        )
    return models


def extract_tile_embeddings(
    scoped: pd.DataFrame,
    quantized: np.ndarray,
    scales: np.ndarray,
    landmask: np.ndarray,
    transform: Any,
    destination_crs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    transformer = Transformer.from_crs(
        "EPSG:27700", destination_crs, always_xy=True
    )
    x, y = transformer.transform(
        scoped["bng_x"].to_numpy(), scoped["bng_y"].to_numpy()
    )
    rows, columns = rowcol(transform, x, y)
    rows = np.asarray(rows, dtype=np.int64)
    columns = np.asarray(columns, dtype=np.int64)
    inside = (
        (rows >= 0)
        & (columns >= 0)
        & (rows < quantized.shape[0])
        & (columns < quantized.shape[1])
    )
    embeddings = np.full((len(scoped), 128), np.nan, dtype=np.float32)
    valid = np.zeros(len(scoped), dtype=bool)
    indices = np.flatnonzero(inside)
    if len(indices):
        local_scales = scales[rows[indices], columns[indices]].astype(np.float32)
        usable = (
            np.isfinite(local_scales)
            & (local_scales > 0)
            & (landmask[rows[indices], columns[indices]] > 0)
        )
        valid_indices = indices[usable]
        embeddings[valid_indices] = (
            quantized[rows[valid_indices], columns[valid_indices]].astype(
                np.float32
            )
            * scales[rows[valid_indices], columns[valid_indices]]
            .astype(np.float32)[:, None]
        )
        valid[valid_indices] = np.isfinite(
            embeddings[valid_indices]
        ).all(axis=1)
    return embeddings, valid


def run_full_map(config: dict[str, Any], smoke_tile: str | None = None) -> None:
    if MAP_PATH.exists() and MAP_MANIFEST_PATH.exists() and smoke_tile is None:
        manifest = json.loads(MAP_MANIFEST_PATH.read_text())
        if (
            manifest["output_sha256"] == sha256(MAP_PATH)
            and manifest["population_sha256"] == sha256(POPULATION_INDEX_PATH)
        ):
            print("full_map: validated complete map checkpoint", flush=True)
            return
        raise RuntimeError("The map checkpoint differs from its manifest")
    population = prepare_population(config)
    models = train_cross_fitted_models(config)
    tile_names = sorted(population["tessera_tile"].unique())
    if len(tile_names) != int(config["full_map"]["expected_tessera_tiles"]):
        raise RuntimeError("The full Arran map requires an unexpected tile count")
    if smoke_tile is not None:
        if smoke_tile not in tile_names:
            raise RuntimeError(f"Unknown map smoke tile: {smoke_tile}")
        tile_names = [smoke_tile]
    version = str(config["full_map"]["tessera_dataset_version"])
    variant = str(config["full_map"]["tessera_dataset_variant"])
    year = int(config["full_map"]["tessera_year"])
    tiles = [phase15._parse_tessera_tile(name) for name in tile_names]
    registry_path, landmask_registry_path = tessera.ensure_registry(version)
    rows = tessera.lookup_embedding_rows(
        registry_path, tiles, year, version, variant
    )
    masks = tessera.lookup_landmask_rows(landmask_registry_path, tiles)
    tessera.assert_complete_inventory(rows, masks, tiles)
    rows_by_tile = {
        tessera.tile_name((float(row.lon), float(row.lat))): row
        for row in rows.itertuples(index=False)
    }
    masks_by_tile = {
        tessera.tile_name((float(row.lon), float(row.lat))): row
        for row in masks.itertuples(index=False)
    }
    MAP_CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    MAP_STREAM_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for position, name in enumerate(tile_names, start=1):
        scoped = population[population["tessera_tile"].eq(name)].copy()
        checkpoint = MAP_CHUNK_DIR / f"{name}.parquet"
        if checkpoint.exists():
            chunk = pd.read_parquet(checkpoint)
            if len(chunk) != len(scoped):
                raise RuntimeError(f"Stale map chunk: {name}")
            records.append(
                {
                    "tile": name,
                    "rows": len(chunk),
                    "valid_rows": int(chunk["tessera_valid"].sum()),
                    "sha256": sha256(checkpoint),
                    "resumed": True,
                }
            )
            print(f"full_map [{position}/{len(tile_names)}]: resumed {name}", flush=True)
            continue
        row = rows_by_tile[name]
        mask_row = masks_by_tile[name]
        tile = phase15._parse_tessera_tile(name)
        paths = tessera.local_tile_paths(MAP_STREAM_DIR, year, tile)
        sources = {
            "embedding": str(row.grid_path),
            "scales": str(row.scales_path),
            "landmask": str(mask_row.key),
        }
        sizes = {
            "embedding": int(row.grid_size),
            "scales": int(row.scales_size),
            "landmask": int(mask_row.file_size),
        }
        update_status(
            "full_map",
            "running",
            {
                "tile": name,
                "tile_position": position,
                "tile_count": len(tile_names),
                "population_rows": len(population),
            },
        )
        print(
            f"full_map [{position}/{len(tile_names)}]: acquire {name} "
            f"for {len(scoped):,} cells",
            flush=True,
        )
        try:
            for kind in ["embedding", "scales", "landmask"]:
                phase15._download_checked(
                    tessera.s3_url(sources[kind]),
                    paths[kind],
                    sizes[kind],
                    phase15.load_config(),
                )
            quantized = np.load(paths["embedding"], mmap_mode="r")
            scales = np.load(paths["scales"], mmap_mode="r")
            with rasterio.open(paths["landmask"]) as source:
                landmask = source.read(1)
                destination_crs = source.crs
                transform = source.transform
            embeddings, valid = extract_tile_embeddings(
                scoped,
                quantized,
                scales,
                landmask,
                transform,
                destination_crs,
            )
            predicted = np.full(len(scoped), np.nan, dtype=np.float32)
            for fold, model in models.items():
                selected = valid & (
                    scoped["spatial_region"].to_numpy(dtype=np.int16) == fold
                )
                if selected.any():
                    predicted[selected] = model.predict(
                        embeddings[selected]
                    ).astype(np.float32)
            output = scoped[
                [
                    "population_id",
                    "source_tile",
                    "bng_x",
                    "bng_y",
                    "spatial_block",
                    "spatial_region",
                    config["full_map"]["target"],
                    "lidar_meanH",
                    "lidar_maxH",
                    "lidar_Cov",
                ]
            ].copy()
            output["tessera_valid"] = valid
            output["prediction"] = predicted
            output["residual"] = (
                output["prediction"] - output[config["full_map"]["target"]]
            )
            core.atomic_parquet(output, checkpoint)
            records.append(
                {
                    "tile": name,
                    "rows": len(output),
                    "valid_rows": int(valid.sum()),
                    "sha256": sha256(checkpoint),
                    "resumed": False,
                }
            )
        finally:
            for path in paths.values():
                path.unlink(missing_ok=True)
            gc.collect()
        print(
            f"full_map [{position}/{len(tile_names)}]: checkpointed {name}",
            flush=True,
        )
    if smoke_tile is not None:
        return
    output = pd.concat(
        [
            pd.read_parquet(MAP_CHUNK_DIR / f"{name}.parquet")
            for name in sorted(population["tessera_tile"].unique())
        ],
        ignore_index=True,
    ).sort_values("population_id")
    if len(output) != len(population):
        raise RuntimeError("The complete map does not match the population")
    valid_fraction = float(output["tessera_valid"].mean())
    if valid_fraction < float(config["full_map"]["minimum_valid_fraction"]):
        raise RuntimeError("Too few population cells have valid TESSERA embeddings")
    core.atomic_parquet(output, MAP_PATH)
    metric_records = []
    scopes = [("all_arran", output["tessera_valid"].to_numpy(dtype=bool))]
    scopes.extend(
        (
            f"arran_region_{region}",
            output["tessera_valid"].to_numpy(dtype=bool)
            & (output["spatial_region"].to_numpy() == region),
        )
        for region in sorted(output["spatial_region"].unique())
    )
    for scope, selected in scopes:
        metric_records.append(
            {
                "scope": scope,
                "rows": int(selected.sum()),
                **phase14.metric_values(
                    output.loc[selected, config["full_map"]["target"]].to_numpy(),
                    output.loc[selected, "prediction"].to_numpy(),
                ),
            }
        )
    write_csv(pd.DataFrame(metric_records), MAP_METRICS_PATH)
    core.atomic_json(
        MAP_MANIFEST_PATH,
        {
            "created_utc": utc_now(),
            "map_type": config["full_map"]["map_type"],
            "population_rows": len(output),
            "valid_rows": int(output["tessera_valid"].sum()),
            "valid_fraction": valid_fraction,
            "population_sha256": sha256(POPULATION_INDEX_PATH),
            "chunks": records,
            "output_sha256": sha256(MAP_PATH),
            "metrics_sha256": sha256(MAP_METRICS_PATH),
        },
    )
    print(
        f"full_map: wrote {int(output['tessera_valid'].sum()):,} "
        f"cross-fitted predictions",
        flush=True,
    )


def run_report(config: dict[str, Any]) -> None:
    bootstrap = pd.read_csv(BOOTSTRAP_SUMMARY_PATH)
    reciprocal = pd.read_csv(RECIPROCAL_METRICS_PATH)
    diagnostics = pd.read_csv(DIAGNOSTIC_CORRELATION_PATH)
    map_metrics = pd.read_csv(MAP_METRICS_PATH)
    original_transfer = pd.read_csv(
        ROOT / "outputs/tables/phase15_cairngorms_to_arran_metrics.csv"
    )
    within = pd.read_csv(ROOT / "outputs/tables/phase15_arran_spatial_metrics.csv")
    primary = config["target"]["name"]
    within_primary = (
        within[
            within["experiment"].eq("arran_buffered_spatial")
            & within["target"].eq(primary)
        ]
        .groupby("model", as_index=False)[["rmse", "r2", "spearman_r"]]
        .mean()
    )
    transfer_forward = original_transfer[
        original_transfer["fold"].eq("all_arran")
    ]
    transfer_reverse = reciprocal[reciprocal["scope"].eq("all_cairngorms")]

    figure, axes = plt.subplots(2, 2, figsize=(10, 8))
    models = ["sentinel2_terrain_nonlinear", "tessera_nonlinear"]
    table = within_primary.set_index("model")
    axes[0, 0].bar(
        ["Sentinel-2 + terrain", "TESSERA"],
        [table.loc[model, "rmse"] for model in models],
        color=["#4477AA", "#009988"],
    )
    axes[0, 0].set_ylabel("Mean fold RMSE")
    axes[0, 0].set_title("a  Within-Arran spatial holdout")

    directions = []
    for label, frame, scope_column, scope in [
        ("Cairngorms to Arran", transfer_forward, "fold", "all_arran"),
        ("Arran to Cairngorms", transfer_reverse, "scope", "all_cairngorms"),
    ]:
        scoped = frame[frame[scope_column].eq(scope)].set_index("model")
        for model in models:
            directions.append(
                {
                    "direction": label,
                    "model": model,
                    "r2": scoped.loc[model, "r2"],
                }
            )
    direction_frame = pd.DataFrame(directions)
    x = np.arange(2)
    width = 0.36
    for offset, model, label, color in [
        (-width / 2, models[0], "Sentinel-2 + terrain", "#4477AA"),
        (width / 2, models[1], "TESSERA", "#009988"),
    ]:
        scoped = direction_frame[direction_frame["model"].eq(model)]
        axes[0, 1].bar(x + offset, scoped["r2"], width, label=label, color=color)
    axes[0, 1].set_xticks(x, direction_frame["direction"].unique(), rotation=12)
    axes[0, 1].set_ylabel("R2")
    axes[0, 1].set_title("b  Reciprocal transfer")
    axes[0, 1].legend(frameon=False, loc="best")
    axes[0, 1].axhline(0, color="black", linewidth=0.7)

    paired = bootstrap[
        bootstrap["metric"].eq("paired_rmse_difference")
    ].copy()
    axes[1, 0].errorbar(
        np.arange(len(paired)),
        paired["estimate"],
        yerr=[
            paired["estimate"] - paired["ci_low"],
            paired["ci_high"] - paired["estimate"],
        ],
        fmt="o",
        color="#009988",
        capsize=4,
    )
    axes[1, 0].set_xticks(
        np.arange(len(paired)),
        paired["experiment"].str.replace("_", " "),
        rotation=18,
        ha="right",
    )
    axes[1, 0].axhline(0, color="black", linewidth=0.7)
    axes[1, 0].set_ylabel("TESSERA minus Sentinel-2 RMSE")
    axes[1, 0].set_title("c  Spatial block-bootstrap comparison")

    map_frame = pd.read_parquet(
        MAP_PATH, columns=["bng_x", "bng_y", "residual", "tessera_valid"]
    )
    map_frame = map_frame[map_frame["tessera_valid"]]
    if len(map_frame) > 100000:
        map_frame = map_frame.sample(100000, random_state=20260727)
    points = axes[1, 1].scatter(
        map_frame["bng_x"],
        map_frame["bng_y"],
        c=map_frame["residual"],
        s=1,
        cmap="RdBu_r",
        vmin=-0.6,
        vmax=0.6,
        rasterized=True,
    )
    figure.colorbar(points, ax=axes[1, 1], label="Prediction residual")
    axes[1, 1].set_aspect("equal")
    axes[1, 1].set_title("d  Cross-fitted Arran residuals")
    axes[1, 1].set_xlabel("British National Grid easting")
    axes[1, 1].set_ylabel("British National Grid northing")
    figure.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=240, bbox_inches="tight")
    plt.close(figure)

    map_overall = map_metrics[map_metrics["scope"].eq("all_arran")].iloc[0]
    strongest = diagnostics[
        diagnostics["response"].eq("absolute_residual")
    ].iloc[
        diagnostics[diagnostics["response"].eq("absolute_residual")][
            "spearman_r"
        ].abs().argmax()
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            [
                "# Strengthened Scotland airborne-LiDAR evidence",
                "",
                "## Evidence added",
                "",
                "- Spatial block-bootstrap uncertainty for the frozen Arran evaluation.",
                "- Reciprocal Arran-to-Cairngorms transfer using identical predictors.",
                "- Residual diagnostics against forest structure and terrain.",
                "- Cross-fitted predictions for the complete eligible Arran population.",
                "",
                "## Full-population result",
                "",
                (
                    f"The complete cross-fitted Arran map contains "
                    f"{int(map_overall.rows):,} valid 10 m predictions. Its RMSE "
                    f"is {map_overall.rmse:.3f}, R2 is {map_overall.r2:.3f}, and "
                    f"Spearman correlation is {map_overall.spearman_r:.3f}."
                ),
                "",
                "## Interpretation boundary",
                "",
                (
                    "The within-Arran result uses a consistent reference target. "
                    "Cross-landscape results remain provisional because the "
                    "Cairngorms workbook and inspected lidarSHM implementation "
                    "differ in their description of the point filter used for "
                    "canopy Shannon diversity."
                ),
                "",
                (
                    f"The strongest monotonic association with absolute residual "
                    f"was `{strongest.variable}` (Spearman "
                    f"{strongest.spearman_r:.3f})."
                ),
                "",
                f"![Strengthened evidence](../figures/{FIGURE_PATH.name})",
                "",
            ]
        ),
        encoding="utf-8",
    )
    paths = [
        CONFIG_PATH,
        PROTOCOL_PATH,
        AUDIT_PATH,
        RECIPROCAL_METRICS_PATH,
        RECIPROCAL_PREDICTIONS_PATH,
        BOOTSTRAP_PATH,
        BOOTSTRAP_SUMMARY_PATH,
        DIAGNOSTIC_CORRELATION_PATH,
        DIAGNOSTIC_BIN_PATH,
        MAP_PATH,
        MAP_METRICS_PATH,
        MAP_MANIFEST_PATH,
        FIGURE_PATH,
        REPORT_PATH,
    ]
    core.atomic_json(
        RESULT_FREEZE_PATH,
        {
            "created_utc": utc_now(),
            "protocol": "phase16_scotland_evidence",
            "files": {
                str(path.relative_to(ROOT)): sha256(path) for path in paths
            },
            "comparability_status": json.loads(AUDIT_PATH.read_text())["status"],
            "full_population_metrics": map_overall.to_dict(),
        },
    )
    print(f"report: wrote {REPORT_PATH.relative_to(ROOT)}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--start-at", choices=STAGES)
    parser.add_argument("--stop-after", choices=STAGES)
    parser.add_argument("--smoke-tile")
    return parser.parse_args()


def selected_stages(args: argparse.Namespace) -> list[str]:
    if args.stage:
        return [args.stage]
    start = STAGES.index(args.start_at) if args.start_at else 0
    stop = STAGES.index(args.stop_after) + 1 if args.stop_after else len(STAGES)
    if stop <= start:
        raise RuntimeError("--stop-after precedes --start-at")
    return STAGES[start:stop]


def main() -> int:
    args = parse_args()
    if args.smoke_tile and args.stage != "full_map":
        raise RuntimeError("--smoke-tile requires --stage full_map")
    config = load_config()
    functions = {
        "audit": run_audit,
        "reciprocal": run_reciprocal,
        "bootstrap": run_bootstrap,
        "diagnostics": run_diagnostics,
        "full_map": lambda value: run_full_map(
            value, smoke_tile=args.smoke_tile
        ),
        "report": run_report,
    }
    stages = selected_stages(args)
    for stage in stages:
        update_status(stage, "running")
        print(f"\n=== {stage} ===", flush=True)
        try:
            functions[stage](config)
        except Exception as error:
            update_status(
                stage,
                "failed",
                {"error_type": type(error).__name__, "error": str(error)},
            )
            raise
        update_status(stage, "complete")
    update_status(stages[-1], "complete", {"completed_stages": stages})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
