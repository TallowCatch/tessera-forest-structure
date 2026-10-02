#!/usr/bin/env python3
"""Download and freeze the authenticated NEON vegetation-structure manifest."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "metadata/phase5_neon_field_structure_manifest.csv"
OUTPUT_ROOT = ROOT / "data/interim/neon_field_structure"
FREEZE_PATH = ROOT / "metadata/phase5_neon_field_structure_acquisition_freeze.json"
DISK_RESERVE_BYTES = 2_000_000_000


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def token_from_environment() -> str | None:
    return os.getenv("NEON_TOKEN") or os.getenv("NEON_PAT")


def destination_for(row: Any) -> Path:
    file_name = Path(str(row.file_name)).name
    if file_name != str(row.file_name):
        raise RuntimeError(f"Unsafe NEON manifest filename: {row.file_name}")
    return OUTPUT_ROOT / str(row.site_id) / str(row.month) / file_name


def download_file(session: requests.Session, token: str, url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with session.get(url, headers={"X-API-TOKEN": token}, stream=True, timeout=180) as response:
        if response.status_code == 403:
            raise RuntimeError("NEON rejected the API token or the token has expired")
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    temporary.replace(destination)


def main() -> int:
    if FREEZE_PATH.exists():
        raise RuntimeError(f"Acquisition freeze exists; refusing to overwrite: {FREEZE_PATH.relative_to(ROOT)}")
    token = token_from_environment()
    if not token:
        raise RuntimeError("NEON_TOKEN is required")
    manifest = pd.read_csv(MANIFEST_PATH)
    required_columns = {"site_id", "month", "file_name", "size_bytes", "url"}
    if not required_columns.issubset(manifest.columns):
        raise RuntimeError(f"Field manifest is missing columns: {sorted(required_columns - set(manifest.columns))}")
    total_size = int(manifest["size_bytes"].sum())
    free_bytes = shutil.disk_usage(ROOT).free
    if free_bytes - total_size < DISK_RESERVE_BYTES:
        raise RuntimeError("Field-table download would violate the 2 GB disk reserve")

    session = requests.Session()
    session.headers.update({"Accept": "text/csv,*/*", "User-Agent": "tessera-gedi-fhd/0.1"})
    records: list[dict[str, Any]] = []
    for row in manifest.itertuples(index=False):
        destination = destination_for(row)
        expected_size = int(row.size_bytes)
        if not destination.exists() or destination.stat().st_size != expected_size:
            download_file(session, token, str(row.url), destination)
        actual_size = destination.stat().st_size
        if actual_size != expected_size:
            destination.unlink(missing_ok=True)
            raise RuntimeError(
                f"NEON size mismatch for {row.file_name}: expected {expected_size}, received {actual_size}"
            )
        records.append({
            "site_id": str(row.site_id),
            "month": str(row.month),
            "file_name": str(row.file_name),
            "path": str(destination.relative_to(ROOT)),
            "size_bytes": actual_size,
            "sha256": sha256(destination),
        })
        print(f"verified {row.site_id} {row.month} {row.file_name}")

    freeze_basis = {
        "source_manifest_path": str(MANIFEST_PATH.relative_to(ROOT)),
        "source_manifest_sha256": sha256(MANIFEST_PATH),
        "source_rows": len(manifest),
        "expected_total_size_bytes": total_size,
        "disk_reserve_bytes": DISK_RESERVE_BYTES,
        "token_value_recorded": False,
        "script_sha256": sha256(Path(__file__)),
    }
    freeze_id = "phase5-neon-field-acquisition-" + canonical_hash(freeze_basis)[:12]
    freeze = {
        "freeze_id": freeze_id,
        "created_at": utc_now(),
        "status": "frozen_verified_neon_field_structure_files",
        "freeze_basis": freeze_basis,
        "freeze_basis_sha256": canonical_hash(freeze_basis),
        "files": records,
        "total_size_bytes": sum(record["size_bytes"] for record in records),
    }
    write_json(FREEZE_PATH, freeze)
    print(f"Frozen {freeze_id}: {len(records)} files, {freeze['total_size_bytes'] / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
