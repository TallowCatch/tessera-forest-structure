#!/usr/bin/env python3
"""Record checksums for every curated workflow output."""

from __future__ import annotations

import json

from tessera_forest_structure.repository import ROOT, expected_artifact_paths, sha256


def build_manifest() -> dict[str, object]:
    paths = expected_artifact_paths()
    records = []
    for relative in sorted(paths):
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        records.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return {"schema_version": 1, "hash": "sha256", "artifacts": records}


def main() -> None:
    target = ROOT / "provenance" / "artifacts.json"
    target.write_text(
        json.dumps(build_manifest(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
