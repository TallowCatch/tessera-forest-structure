from __future__ import annotations

import hashlib
from pathlib import Path

from tessera_forest_structure.repository import (
    ROOT,
    load_registry,
    required_project_paths,
    verify_artifacts,
)


def test_registry_has_each_supported_scientific_workflow() -> None:
    workflows = load_registry()["workflows"]
    identifiers = [item["id"] for item in workflows]
    assert identifiers == [
        "cairngorms-local",
        "savelsbos-local",
        "cross-landscape",
        "dutch-multiforest",
        "reference-assisted",
        "reference-uncertainty",
    ]
    assert len(identifiers) == len(set(identifiers))


def test_registered_paths_exist() -> None:
    assert all(path.is_file() for path in required_project_paths())


def test_curated_artifact_checksums_match() -> None:
    assert verify_artifacts() == []


def test_main_figures_are_not_duplicated() -> None:
    figures = [path for path in (ROOT / "results" / "figures" / "main").iterdir() if path.is_file()]
    for suffix in {path.suffix for path in figures}:
        matching = [path for path in figures if path.suffix == suffix]
        hashes = {hashlib.sha256(path.read_bytes()).digest() for path in matching}
        assert len(hashes) == len(matching)


def test_release_builder_excludes_local_and_private_material() -> None:
    from tools.build_reproducibility_package import release_files

    relative = {path.relative_to(ROOT) for path in release_files()}
    assert Path(".env") not in relative
    assert not any(path.parts[0] in {"output", "tmp"} for path in relative)
    assert not any("__pycache__" in path.parts for path in relative)
