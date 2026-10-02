#!/usr/bin/env python3
"""Freeze the combined 2024 plus 2025 target-free Phase 13 cohort."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "metadata/project_config_phase13_temporal_extension_protocol_freeze.yaml"
FREEZE_2024_PATH = ROOT / "metadata/phase13_target_free_gedi_freeze.json"
FREEZE_2025_PATH = ROOT / "metadata/phase13_2025_target_free_gedi_freeze.json"
PRIMARY_2024_PATH = ROOT / "data/processed/phase13_target_free_primary_forest_metadata.parquet"
PRIMARY_2025_PATH = ROOT / "data/processed/phase13_2025_target_free_primary_forest_metadata.parquet"
FOLDS_2024_PATH = ROOT / "metadata/phase13_target_free_fold_summary.csv"
FOLDS_2025_PATH = ROOT / "metadata/phase13_2025_target_free_fold_summary.csv"
SITES_2024_PATH = ROOT / "metadata/phase13_target_free_site_summary.csv"
SITES_2025_PATH = ROOT / "metadata/phase13_2025_target_free_site_summary.csv"
OUTPUT_PATH = ROOT / "data/processed/phase13_combined_target_free_primary_metadata.parquet"
FOLD_PATH = ROOT / "metadata/phase13_combined_target_free_fold_summary.csv"
SITE_PATH = ROOT / "metadata/phase13_combined_target_free_site_summary.csv"
FREEZE_PATH = ROOT / "metadata/phase13_combined_target_free_freeze.json"


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    protected = [OUTPUT_PATH, FOLD_PATH, SITE_PATH, FREEZE_PATH]
    existing = [str(path.relative_to(ROOT)) for path in protected if path.exists()]
    if existing:
        raise RuntimeError(f"Phase 13 combined cohort exists; refusing to overwrite: {existing}")
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))[
        "phase13_temporal_extension"
    ]
    freeze_2024 = json.loads(FREEZE_2024_PATH.read_text(encoding="utf-8"))
    freeze_2025 = json.loads(FREEZE_2025_PATH.read_text(encoding="utf-8"))
    if freeze_2024["freeze_id"] != protocol["prerequisite_2024_target_free_freeze"]:
        raise RuntimeError("Unexpected Phase 13 2024 prerequisite")
    if freeze_2024["freeze_basis"]["gate"]["passed"]:
        raise RuntimeError("Phase 13 combined extension is unnecessary because 2024 passed")
    if not freeze_2025["freeze_basis"]["gate"]["passed"]:
        raise RuntimeError("Phase 13 2025 extension gate failed")
    sites_2024 = sorted(freeze_2024["freeze_basis"]["viable_sites"])
    sites_2025 = sorted(freeze_2025["freeze_basis"]["viable_2025_sites"])
    overlap = sorted(set(sites_2024) & set(sites_2025))
    if overlap:
        raise RuntimeError(f"A forest appears in both Phase 13 target years: {overlap}")
    combined_sites = sorted([*sites_2024, *sites_2025])
    minimum = int(protocol["combined_replication_gate"]["minimum_unique_forests"])
    if len(combined_sites) < minimum:
        raise RuntimeError("Phase 13 combined unique-forest gate failed")

    frame_2024 = pd.read_parquet(PRIMARY_2024_PATH)
    frame_2024 = frame_2024[frame_2024["site_id"].isin(sites_2024)].copy()
    frame_2024["target_year"] = 2024
    frame_2025 = pd.read_parquet(PRIMARY_2025_PATH)
    frame_2025 = frame_2025[frame_2025["site_id"].isin(sites_2025)].copy()
    frame_2025["target_year"] = 2025
    common_columns = sorted(set(frame_2024.columns) & set(frame_2025.columns))
    combined = pd.concat(
        [frame_2024[common_columns], frame_2025[common_columns]], ignore_index=True
    ).sort_values(["target_year", "site_id", "acquisition_datetime", "shot_number"])
    if combined.duplicated(["site_id", "shot_number"]).any():
        raise RuntimeError("Duplicate site/shot keys entered the combined Phase 13 cohort")
    if "fhd_normal" in combined.columns:
        raise RuntimeError("FHD entered the combined target-free cohort")

    folds_2024 = pd.read_csv(FOLDS_2024_PATH)
    folds_2024 = folds_2024[
        folds_2024["site_id"].isin(sites_2024)
        & folds_2024["eligible"].astype(str).str.lower().eq("true")
    ].copy()
    folds_2024["target_year"] = 2024
    folds_2025 = pd.read_csv(FOLDS_2025_PATH)
    folds_2025 = folds_2025[
        folds_2025["site_id"].isin(sites_2025)
        & folds_2025["eligible"].astype(str).str.lower().eq("true")
    ].copy()
    folds_2025["target_year"] = 2025
    folds = pd.concat([folds_2024, folds_2025], ignore_index=True).sort_values(
        ["target_year", "site_id", "held_out_pass"]
    )
    if folds.groupby("site_id").size().lt(2).any():
        raise RuntimeError("A combined Phase 13 forest has fewer than two eligible folds")

    summary_2024 = pd.read_csv(SITES_2024_PATH)
    summary_2024 = summary_2024[summary_2024["site_id"].isin(sites_2024)].copy()
    summary_2024["target_year"] = 2024
    summary_2025 = pd.read_csv(SITES_2025_PATH)
    summary_2025 = summary_2025[summary_2025["site_id"].isin(sites_2025)].copy()
    summary_2025["target_year"] = 2025
    summary = pd.concat([summary_2024, summary_2025], ignore_index=True).sort_values(
        ["target_year", "site_id"]
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUTPUT_PATH, index=False, compression="zstd")
    folds.to_csv(FOLD_PATH, index=False)
    summary.to_csv(SITE_PATH, index=False)
    sealed = [
        {"target_year": 2024, **record}
        for record in freeze_2024["freeze_basis"]["sealed_outcome_files"]
    ] + [
        {"target_year": 2025, **record}
        for record in freeze_2025["freeze_basis"]["sealed_outcome_files"]
    ]
    freeze_basis = {
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "script_sha256": sha256(Path(__file__).resolve()),
        "freeze_2024_id": freeze_2024["freeze_id"],
        "freeze_2025_id": freeze_2025["freeze_id"],
        "target_fhd_read": False,
        "sites_2024": sites_2024,
        "sites_2025": sites_2025,
        "combined_unique_sites": combined_sites,
        "site_years": {
            site: int(year)
            for site, year in summary[["site_id", "target_year"]].itertuples(index=False)
        },
        "site_rows": {
            site: int(count) for site, count in combined.groupby("site_id").size().items()
        },
        "eligible_folds": len(folds),
        "sealed_outcome_files": sealed,
        "gate": {
            "minimum_unique_sites": minimum,
            "observed_unique_sites": len(combined_sites),
            "passed": True,
        },
    }
    freeze = {
        "freeze_id": "phase13-combined-target-free-" + canonical_hash(freeze_basis)[:12],
        "created_utc": pd.Timestamp.now(tz="UTC").floor("s").isoformat().replace("+00:00", "Z"),
        "status": "frozen_combined_gate_passed",
        "freeze_basis": freeze_basis,
        "outputs": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [OUTPUT_PATH, FOLD_PATH, SITE_PATH]
        },
    }
    write_json(FREEZE_PATH, freeze)
    print(
        json.dumps(
            {
                "freeze_id": freeze["freeze_id"],
                "sites_2024": sites_2024,
                "sites_2025": sites_2025,
                "eligible_folds": len(folds),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
