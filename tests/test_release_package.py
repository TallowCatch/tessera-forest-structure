from __future__ import annotations

import hashlib
import re
import zipfile

from tools.build_reproducibility_package import ROOT, build_archive


def test_release_archive_is_clean_and_self_contained(tmp_path) -> None:
    output = build_archive(tmp_path / "release.zip")
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
    assert any(name.endswith("/README.md") for name in names)
    assert any(name.endswith("/provenance/artifacts.json") for name in names)
    assert any(name.endswith("/results/tables/cairngorms_performance.csv") for name in names)
    dutch_script = "/research_history/scripts/evaluate_dutch_expanded.py"
    assert any(name.endswith(dutch_script) for name in names)
    assert any(name.endswith("/research_history/EXPERIMENT_INDEX.md") for name in names)
    historical_scripts = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "research_history" / "scripts").rglob("*")
        if path.is_file() and path.suffix in {".py", ".sh", ".sbatch"}
    }
    assert len(historical_scripts) >= 319
    assert not any(re.search(r"phase_?\d+", path) for path in historical_scripts)
    assert all(any(name.endswith("/" + path) for name in names) for path in historical_scripts)
    assert not any(name.endswith("/.env") for name in names)
    assert not any("__pycache__" in name or name.endswith(".pyc") for name in names)


def test_release_archive_is_deterministic(tmp_path) -> None:
    first = build_archive(tmp_path / "first.zip")
    second = build_archive(tmp_path / "second.zip")
    assert (
        hashlib.sha256(first.read_bytes()).digest() == hashlib.sha256(second.read_bytes()).digest()
    )
