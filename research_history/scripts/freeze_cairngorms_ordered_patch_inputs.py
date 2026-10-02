#!/usr/bin/env python3
"""Validate and freeze the inputs for the ordered-patch experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/cairngorms_ordered_patch.yaml"
OUTPUT_PATH = ROOT / "metadata/cairngorms_ordered_patch_input_freeze.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "cairngorms_ordered_patch"
    ]
    source = config["source"]
    array_directory = ROOT / source["array_directory"]
    patch_manifest_path = ROOT / source["patch_manifest"]
    patch_manifest = json.loads(
        patch_manifest_path.read_text(encoding="utf-8")
    )
    expected_shape = (
        int(source["expected_rows"]),
        *[int(value) for value in source["expected_patch_shape"]],
    )
    patch = np.load(array_directory / "patch_quantized.npy", mmap_mode="r")
    if patch.shape != expected_shape:
        raise RuntimeError(
            f"Expected ordered patches {expected_shape}, received {patch.shape}"
        )

    required = [
        "patch_quantized.npy",
        "patch_scales.npy",
        "patch_valid.npy",
        "row_id.npy",
        "manifest.json",
        *[f"fold_{fold}.npz" for fold in config["evaluation"]["folds"]],
    ]
    hashes = {
        name: sha256(array_directory / name)
        for name in required
        if (array_directory / name).exists()
    }
    missing = sorted(set(required) - set(hashes))
    if missing:
        raise RuntimeError(f"Missing ordered-patch inputs: {missing}")
    for name, expected in patch_manifest["output_sha256"].items():
        if hashes[name] != expected:
            raise RuntimeError(f"Patch checksum mismatch: {name}")

    baseline_files = {
        name: ROOT / path
        for name, path in {
            "baseline_predictions": source["baseline_predictions"],
            "baseline_metrics": source["baseline_metrics"],
            "target_table": source["target_table"],
            "phase19_result_freeze": "metadata/phase19_cairngorms_result_freeze.json",
        }.items()
    }
    freeze = {
        "config_sha256": sha256(CONFIG_PATH),
        "expected_patch_shape": expected_shape,
        "patch_manifest_sha256": sha256(patch_manifest_path),
        "array_hashes": hashes,
        "baseline_hashes": {
            name: sha256(path) for name, path in baseline_files.items()
        },
        "models": config["models"],
        "folds": config["evaluation"]["folds"],
        "seeds": config["evaluation"]["seeds"],
        "status": "frozen_before_ordered_patch_outcomes",
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Validated {len(required)} ordered-patch files")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
