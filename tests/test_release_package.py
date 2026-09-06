from __future__ import annotations

import hashlib
import zipfile

from tools.build_reproducibility_package import build_archive


def test_release_archive_is_clean_and_self_contained(tmp_path) -> None:
    output = build_archive(tmp_path / "release.zip")
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
    assert any(name.endswith("/README.md") for name in names)
    assert any(name.endswith("/provenance/artifacts.json") for name in names)
    assert any(name.endswith("/results/tables/cairngorms_performance.csv") for name in names)
    assert not any(name.endswith("/.env") for name in names)
    assert not any("__pycache__" in name or name.endswith(".pyc") for name in names)


def test_release_archive_is_deterministic(tmp_path) -> None:
    first = build_archive(tmp_path / "first.zip")
    second = build_archive(tmp_path / "second.zip")
    assert (
        hashlib.sha256(first.read_bytes()).digest() == hashlib.sha256(second.read_bytes()).digest()
    )
