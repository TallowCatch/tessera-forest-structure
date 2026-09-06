"""Repository metadata, integrity checks, and analysis plans."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "provenance" / "workflows.json"
ARTIFACTS = ROOT / "provenance" / "artifacts.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_registry() -> dict[str, Any]:
    payload = read_json(REGISTRY)
    if payload.get("schema_version") != 3:
        raise RuntimeError("Unsupported workflow registry schema")
    return payload


def workflows_by_id() -> dict[str, dict[str, Any]]:
    return {item["id"]: item for item in load_registry()["workflows"]}


def workflow(identifier: str) -> dict[str, Any]:
    workflows = workflows_by_id()
    if identifier not in workflows:
        choices = ", ".join(sorted(workflows))
        raise SystemExit(f"Unknown workflow '{identifier}'. Choose from: {choices}")
    return workflows[identifier]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def required_project_paths() -> list[Path]:
    paths = [
        ROOT / "README.md",
        ROOT / "environment.yml",
        ROOT / "examples" / "demo_data" / "reference_transfer.csv",
    ]
    for item in workflows_by_id().values():
        paths.extend(ROOT / item[key] for key in ("readme", "parameters"))
        paths.extend(ROOT / path for path in item["outputs"])
    return paths


def expected_artifact_paths() -> set[str]:
    paths = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "results").rglob("*")
        if path.is_file() and path.name != "README.md"
    }
    paths.add("examples/demo_data/reference_transfer.csv")
    return paths


def verify_artifacts() -> list[str]:
    if not ARTIFACTS.exists():
        return ["missing provenance/artifacts.json"]
    errors: list[str] = []
    records = read_json(ARTIFACTS)["artifacts"]
    recorded = [artifact["path"] for artifact in records]
    if len(recorded) != len(set(recorded)):
        errors.append("duplicate paths in provenance/artifacts.json")
    missing_records = expected_artifact_paths() - set(recorded)
    extra_records = set(recorded) - expected_artifact_paths()
    errors.extend(f"artifact missing from manifest: {path}" for path in sorted(missing_records))
    errors.extend(f"stale artifact manifest entry: {path}" for path in sorted(extra_records))
    for artifact in records:
        path = ROOT / artifact["path"]
        if not path.is_file():
            errors.append(f"missing artifact: {artifact['path']}")
        elif sha256(path) != artifact["sha256"]:
            errors.append(f"checksum mismatch: {artifact['path']}")
    return errors


def verify_repository() -> tuple[int, list[str]]:
    paths = required_project_paths()
    errors = [f"missing file: {path.relative_to(ROOT)}" for path in paths if not path.is_file()]
    errors.extend(verify_artifacts())
    return len(paths), errors


def input_status(identifier: str) -> list[dict[str, str | bool]]:
    item = workflow(identifier)
    status = []
    for record in item["inputs"]:
        path = ROOT / record["path"]
        status.append({**record, "available": path.exists()})
    return status
