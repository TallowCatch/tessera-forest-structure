#!/usr/bin/env python3
"""Build a deterministic, cache-free reproducibility archive."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "output" / "release" / "tessera_forest_structure_reproducibility.zip"
PROJECT_FILES = {
    ".env.example",
    ".gitignore",
    "CHANGELOG.md",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "LICENSE",
    "Makefile",
    "README.md",
    "environment.yml",
    "pyproject.toml",
}
PROJECT_DIRECTORIES = {
    ".github",
    "data",
    "docs",
    "examples",
    "provenance",
    "results",
    "src",
    "tests",
    "tools",
    "workflows",
}
EXCLUDED_PARTS = {
    ".DS_Store",
    ".env",
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "raw",
    "interim",
    "processed",
    "external",
}
EXCLUDED_SUFFIXES = {".aux", ".bbl", ".blg", ".log", ".out", ".pyc", ".pyo"}


def include_file(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if any(part in EXCLUDED_PARTS for part in relative.parts):
        return False
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return False
    if relative.as_posix() in PROJECT_FILES:
        return True
    return relative.parts[0] in PROJECT_DIRECTORIES


def release_files() -> list[Path]:
    return sorted(
        (path for path in ROOT.rglob("*") if path.is_file() and include_file(path)),
        key=lambda path: path.relative_to(ROOT).as_posix(),
    )


def build_archive(output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    prefix = "tessera-forest-structure"
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in release_files():
            relative = path.relative_to(ROOT)
            info = zipfile.ZipInfo(f"{prefix}/{relative.as_posix()}")
            info.date_time = (2026, 9, 6, 0, 0, 0)
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = build_archive(args.output)
    print(f"wrote {output} ({output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
