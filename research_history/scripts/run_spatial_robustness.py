#!/usr/bin/env python3
"""Retrospective SOAP/TEAK spatial-separation robustness analyses."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import train_baselines as phase3  # noqa: E402


DATA_PATH = ROOT / "data/processed/tessera_aligned_development_soap_teak.parquet"
FOLDS_PATH = ROOT / "metadata/phase3_spatial_folds.parquet"
ORIGINAL_PREDICTIONS_PATH = ROOT / "data/processed/phase3_oof_predictions.parquet"
PREDICTIONS_PATH = ROOT / "data/processed/robustness_spatial_predictions.parquet"
METRICS_PATH = ROOT / "outputs/tables/robustness_spatial_metrics.csv"
DIAGNOSTICS_PATH = ROOT / "outputs/tables/robustness_spatial_diagnostics.csv"
FREEZE_PATH = ROOT / "metadata/robustness_spatial_freeze.json"
FEATURES = [f"tessera_area_{index:03d}" for index in range(128)]
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
BUFFERS_M = [500.0, 1000.0, 3000.0, 6000.0]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def fit_ridge(training: pd.DataFrame, alpha: float) -> Pipeline:
    model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha))])
    model.fit(
        training[FEATURES].to_numpy(dtype=np.float32),
        training["fhd_normal"].to_numpy(dtype=np.float64),
    )
    return model


def select_buffer_alpha(
    frame: pd.DataFrame,
    outer_train: np.ndarray,
    outer_fold: int,
    buffer_m: float,
) -> float:
    del outer_train, buffer_m
    values = frame.loc[
        frame["spatial_fold_id"].eq(outer_fold), "original_selected_alpha"
    ].unique()
    if len(values) != 1:
        raise RuntimeError(f"Original fold {outer_fold} has a non-unique Ridge penalty")
    return float(values[0])


def buffered_predictions(frame: pd.DataFrame) -> tuple[list[pd.DataFrame], list[dict[str, Any]]]:
    outputs: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    folds = sorted(frame["spatial_fold_id"].unique())
    for buffer_m in BUFFERS_M:
        for fold in folds:
            train_idx, test_idx, removed, minimum_distance = phase3.buffered_split(
                frame, int(fold), buffer_m
            )
            alpha = select_buffer_alpha(frame, train_idx, int(fold), buffer_m)
            model = fit_ridge(frame.iloc[train_idx], alpha)
            test = frame.iloc[test_idx].copy()
            test["prediction"] = model.predict(test[FEATURES].to_numpy(dtype=np.float32))
            test["analysis"] = f"spatial_buffer_{int(buffer_m)}m"
            test["outer_group"] = str(int(fold))
            test["selected_alpha"] = alpha
            outputs.append(test[[
                "site_id", "shot_number", "fhd_normal", "analysis", "outer_group",
                "selected_alpha", "prediction",
            ]])
            diagnostics.append({
                "analysis": f"spatial_buffer_{int(buffer_m)}m",
                "outer_group": str(int(fold)),
                "training_rows": len(train_idx),
                "test_rows": len(test_idx),
                "buffer_removed_rows": removed,
                "minimum_train_test_distance_m": minimum_distance,
                "selected_alpha": alpha,
            })
    return outputs, diagnostics


def granule_inner_alpha(training: pd.DataFrame) -> float:
    groups = sorted(training["granule_id"].unique())
    records = []
    for alpha in ALPHAS:
        for group in groups:
            validation = training[training["granule_id"].eq(group)]
            fit = training[~training["granule_id"].eq(group)]
            if len(fit) == 0 or len(validation) < 10:
                continue
            model = fit_ridge(fit, alpha)
            prediction = model.predict(validation[FEATURES].to_numpy(dtype=np.float32))
            records.append({
                "alpha": alpha,
                "group": group,
                "rmse": phase3.metric_values(
                    validation["fhd_normal"].to_numpy(), prediction
                )["rmse"],
            })
    table = pd.DataFrame(records)
    if table.empty:
        raise RuntimeError("No valid inner granule folds")
    return float(
        table.groupby("alpha", as_index=False)["rmse"].mean()
        .sort_values(["rmse", "alpha"]).iloc[0]["alpha"]
    )


def granule_predictions(frame: pd.DataFrame) -> tuple[list[pd.DataFrame], list[dict[str, Any]]]:
    outputs: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    for granule in sorted(frame["granule_id"].unique()):
        test = frame[frame["granule_id"].eq(granule)].copy()
        training = frame[~frame["granule_id"].eq(granule)].copy()
        alpha = granule_inner_alpha(training)
        model = fit_ridge(training, alpha)
        test["prediction"] = model.predict(test[FEATURES].to_numpy(dtype=np.float32))
        test["analysis"] = "leave_one_gedi_pass_out"
        test["outer_group"] = granule
        test["selected_alpha"] = alpha
        outputs.append(test[[
            "site_id", "shot_number", "fhd_normal", "analysis", "outer_group",
            "selected_alpha", "prediction",
        ]])
        diagnostics.append({
            "analysis": "leave_one_gedi_pass_out",
            "outer_group": granule,
            "test_site": "+".join(sorted(test["site_id"].unique())),
            "training_rows": len(training),
            "test_rows": len(test),
            "buffer_removed_rows": 0,
            "minimum_train_test_distance_m": math.nan,
            "selected_alpha": alpha,
        })
    return outputs, diagnostics


def metric_table(predictions: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for analysis, frame in predictions.groupby("analysis", sort=True):
        for site, scoped in frame.groupby("site_id", sort=True):
            records.append({
                "analysis": analysis,
                "scope": site,
                "rows": len(scoped),
                **phase3.metric_values(
                    scoped["fhd_normal"].to_numpy(), scoped["prediction"].to_numpy()
                ),
            })
        records.append({
            "analysis": analysis,
            "scope": "pooled",
            "rows": len(frame),
            **phase3.metric_values(
                frame["fhd_normal"].to_numpy(), frame["prediction"].to_numpy()
            ),
        })
    return pd.DataFrame(records)


def main() -> int:
    columns = [
        "site_id", "shot_number", "granule_id", "orbit", "reference_ground_track",
        "x_epsg5070", "y_epsg5070", "fhd_normal", *FEATURES,
    ]
    frame = pd.read_parquet(DATA_PATH, columns=columns).merge(
        pd.read_parquet(FOLDS_PATH)[
            ["site_id", "shot_number", "spatial_block_id", "spatial_fold_id"]
        ],
        on=["site_id", "shot_number"],
        validate="one_to_one",
    )
    original_alphas = pd.read_parquet(ORIGINAL_PREDICTIONS_PATH)[
        ["site_id", "shot_number", "outer_selected_ridge_alpha"]
    ].rename(columns={"outer_selected_ridge_alpha": "original_selected_alpha"})
    frame = frame.merge(original_alphas, on=["site_id", "shot_number"], validate="one_to_one")
    buffered, buffered_diagnostics = buffered_predictions(frame)
    passes, pass_diagnostics = granule_predictions(frame)
    predictions = pd.concat([*buffered, *passes], ignore_index=True)
    metrics = metric_table(predictions)
    diagnostics = pd.DataFrame([*buffered_diagnostics, *pass_diagnostics])

    write_parquet(predictions, PREDICTIONS_PATH)
    write_csv(metrics, METRICS_PATH)
    write_csv(diagnostics, DIAGNOSTICS_PATH)
    freeze = {
        "freeze_id": "retrospective-spatial-robustness-" + hashlib.sha256(
            metrics.to_csv(index=False).encode("utf-8")
        ).hexdigest()[:12],
        "created_utc": utc_now(),
        "status": "retrospective_robustness_not_prospective",
        "buffer_distances_m": BUFFERS_M,
        "pass_group": "complete_GEDI_L2B_granule",
        "input_hashes": {
            str(DATA_PATH.relative_to(ROOT)): sha256(DATA_PATH),
            str(FOLDS_PATH.relative_to(ROOT)): sha256(FOLDS_PATH),
            str(ORIGINAL_PREDICTIONS_PATH.relative_to(ROOT)): sha256(ORIGINAL_PREDICTIONS_PATH),
        },
        "outputs": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [PREDICTIONS_PATH, METRICS_PATH, DIAGNOSTICS_PATH]
        },
    }
    FREEZE_PATH.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n")
    print(metrics.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
