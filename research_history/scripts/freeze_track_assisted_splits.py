#!/usr/bin/env python3
"""Freeze target-free complete-pass splits for Phase 12 track assistance."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_context_height_footprint_height_transfer as base  # noqa: E402


PROTOCOL_PATH = ROOT / "metadata/project_config_phase12_track_assisted_protocol_freeze.yaml"
MANIFEST_PATH = ROOT / "metadata/phase12_track_assisted_split_manifest.json"
INPUT_PATHS = base.INPUT_PATHS
READ_COLUMNS = [
    "site_id",
    "shot_number",
    "acquisition_datetime",
    "orbit",
    "reference_ground_track",
    "x_epsg5070",
    "y_epsg5070",
]


def pass_ids(frame: pd.DataFrame) -> pd.Series:
    dates = pd.to_datetime(frame["acquisition_datetime"], utc=True).dt.strftime("%Y-%m-%d")
    return (
        frame["orbit"].astype(str)
        + "_"
        + frame["reference_ground_track"].astype(str)
        + "_"
        + dates
    )


def shot_hash(values: np.ndarray) -> str:
    normalized = "\n".join(str(int(value)) for value in np.sort(values.astype(np.uint64)))
    return hashlib.sha256(normalized.encode("ascii")).hexdigest()


def buffered_local_split(
    coordinates: np.ndarray,
    identifiers: np.ndarray,
    held_out_pass: str,
    buffer_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    test = np.flatnonzero(identifiers == held_out_pass)
    candidates = np.flatnonzero(identifiers != held_out_pass)
    if len(test) == 0 or len(candidates) == 0:
        return candidates[:0], test, np.asarray([], dtype=np.float64)
    candidate_distance, _ = cKDTree(coordinates[test]).query(coordinates[candidates], k=1)
    training = candidates[candidate_distance >= buffer_m]
    if len(training) == 0:
        return training, test, np.full(len(test), np.nan)
    test_distance, _ = cKDTree(coordinates[training]).query(coordinates[test], k=1)
    return training, test, np.asarray(test_distance, dtype=np.float64)


def main() -> int:
    if MANIFEST_PATH.exists():
        raise RuntimeError(f"Phase 12 split manifest exists: {MANIFEST_PATH}")
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))[
        "phase12_track_assisted_prediction"
    ]
    frame = pd.concat(
        [pd.read_parquet(path, columns=READ_COLUMNS) for path in INPUT_PATHS],
        ignore_index=True,
    )
    if len(frame) != 15_326 or frame[["site_id", "shot_number"]].duplicated().any():
        raise RuntimeError("Phase 12 split population changed")
    frame["pass_id"] = pass_ids(frame)
    folds: list[dict[str, Any]] = []
    site_records: list[dict[str, Any]] = []
    buffer_m = float(protocol["outer_split"]["exclusion_buffer_m"])
    min_test = int(protocol["outer_split"]["minimum_test_rows"])
    min_train = int(protocol["outer_split"]["minimum_retained_local_training_rows"])
    for site, site_frame in frame.groupby("site_id", sort=True):
        site_frame = site_frame.reset_index(drop=True)
        coordinates = site_frame[["x_epsg5070", "y_epsg5070"]].to_numpy(dtype=np.float64)
        identifiers = site_frame["pass_id"].to_numpy(dtype=str)
        passes = sorted(site_frame["pass_id"].unique())
        eligible_count = 0
        for held_out in passes:
            training, test, distances = buffered_local_split(
                coordinates, identifiers, held_out, buffer_m
            )
            reasons = []
            if len(test) < min_test:
                reasons.append("test_rows_below_minimum")
            if len(training) < min_train:
                reasons.append("retained_local_training_rows_below_minimum")
            eligible = len(reasons) == 0
            eligible_count += int(eligible)
            folds.append(
                {
                    "site_id": site,
                    "held_out_pass": held_out,
                    "total_site_passes": len(passes),
                    "test_rows": len(test),
                    "local_training_rows_before_buffer": int(len(site_frame) - len(test)),
                    "local_training_rows_after_buffer": len(training),
                    "buffer_removed_rows": int(len(site_frame) - len(test) - len(training)),
                    "minimum_test_to_local_training_distance_m": float(np.nanmin(distances))
                    if len(distances) and np.isfinite(distances).any()
                    else None,
                    "median_test_to_local_training_distance_m": float(np.nanmedian(distances))
                    if len(distances) and np.isfinite(distances).any()
                    else None,
                    "maximum_test_to_local_training_distance_m": float(np.nanmax(distances))
                    if len(distances) and np.isfinite(distances).any()
                    else None,
                    "test_shot_sha256": shot_hash(site_frame.iloc[test]["shot_number"].to_numpy()),
                    "local_training_shot_sha256": shot_hash(
                        site_frame.iloc[training]["shot_number"].to_numpy()
                    ),
                    "eligible_by_row_gates": eligible,
                    "ineligibility_reasons": reasons,
                }
            )
        primary = (
            len(passes) >= int(protocol["primary_site_eligibility"]["minimum_total_passes"])
            and eligible_count
            >= int(protocol["primary_site_eligibility"]["minimum_eligible_held_out_passes"])
        )
        two_pass = len(passes) == 2 and eligible_count == 2
        site_records.append(
            {
                "site_id": site,
                "rows": len(site_frame),
                "total_passes": len(passes),
                "eligible_folds": eligible_count,
                "primary_site": primary,
                "two_pass_sensitivity_site": two_pass,
            }
        )

    sites = pd.DataFrame(site_records)
    primary_sites = sorted(sites.loc[sites["primary_site"], "site_id"])
    secondary_sites = sorted(
        sites.loc[sites["two_pass_sensitivity_site"], "site_id"]
    )
    expected_primary = sorted(protocol["primary_site_eligibility"]["expected_sites"])
    expected_secondary = sorted(protocol["two_pass_sensitivity"]["expected_sites"])
    if primary_sites != expected_primary:
        raise RuntimeError(f"Primary Phase 12 sites changed: {primary_sites}")
    if secondary_sites != expected_secondary:
        raise RuntimeError(f"Two-pass Phase 12 sites changed: {secondary_sites}")
    fold_frame = pd.DataFrame(folds)
    primary_folds = fold_frame[
        fold_frame["site_id"].isin(primary_sites) & fold_frame["eligible_by_row_gates"]
    ]
    secondary_folds = fold_frame[
        fold_frame["site_id"].isin(secondary_sites) & fold_frame["eligible_by_row_gates"]
    ]
    if len(primary_folds) != int(protocol["primary_site_eligibility"]["expected_folds"]):
        raise RuntimeError("Primary Phase 12 fold count changed")
    if len(secondary_folds) != int(protocol["two_pass_sensitivity"]["expected_folds"]):
        raise RuntimeError("Two-pass Phase 12 fold count changed")

    freeze_basis = {
        "protocol_sha256": base.sha256(PROTOCOL_PATH),
        "script_sha256": base.sha256(Path(__file__).resolve()),
        "input_hashes": {
            str(path.relative_to(ROOT)): base.sha256(path) for path in INPUT_PATHS
        },
        "columns_read": READ_COLUMNS,
        "fhd_normal_read": False,
        "rows": len(frame),
        "sites": sorted(frame["site_id"].unique()),
        "passes": int(frame["pass_id"].nunique()),
        "buffer_m": buffer_m,
        "minimum_test_rows": min_test,
        "minimum_retained_local_training_rows": min_train,
        "primary_sites": primary_sites,
        "primary_folds": len(primary_folds),
        "two_pass_sensitivity_sites": secondary_sites,
        "two_pass_sensitivity_folds": len(secondary_folds),
    }
    manifest = {
        "freeze_id": "phase12-track-splits-" + base.canonical_hash(freeze_basis)[:12],
        "created_utc": base.utc_now(),
        "status": "frozen_target_free_track_assisted_splits",
        "freeze_basis": freeze_basis,
        "site_summary": site_records,
        "folds": folds,
    }
    base.write_json(MANIFEST_PATH, manifest)
    print(
        json.dumps(
            {
                "freeze_id": manifest["freeze_id"],
                "primary_sites": primary_sites,
                "primary_folds": len(primary_folds),
                "two_pass_sites": secondary_sites,
                "two_pass_folds": len(secondary_folds),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
