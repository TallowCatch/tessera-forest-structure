#!/usr/bin/env python3
"""Train one reflection-augmented Cairngorms summary MLP."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

import cairngorms_spatial_unet_spatial_unet_worker as raw_base
from reflection_augmentation_flip_mlp_common import (
    configure_threads,
    fit_selected_model,
    predict_original_scale,
    refit_fixed_epochs,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/reflection_augmentation_cairngorms_flip_augmentation.yaml"


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "phase33_cairngorms_flip_augmentation"
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def raw_inputs(config: dict[str, Any], fold_number: int):
    directory = ROOT / str(config["frozen_inputs"]["raw_array_directory"])
    wrapper = {
        "outputs": {"array_directory": str(directory.relative_to(ROOT))}
    }
    arrays = raw_base.load_arrays(wrapper)
    with np.load(directory / "folds" / f"balanced_fold_{fold_number}.npz") as source:
        fold = {name: source[name] for name in source.files}
    features = raw_base.build_tabular_features(
        "tessera_mlp_5x5", arrays, fold["input_mean"], fold["input_sd"]
    )
    weights = np.ones(len(arrays["targets"]), dtype=np.float32)
    subtrain = fold["subtrain_indices"].astype(np.int64)
    validation = fold["validation_indices"].astype(np.int64)
    weights[subtrain] = raw_base.block_weights(arrays["block"], subtrain)
    weights[validation] = raw_base.block_weights(arrays["block"], validation)
    return (
        features,
        np.asarray(arrays["targets"], dtype=np.float32),
        fold,
        weights,
        np.asarray(fold["target_mean"], dtype=np.float32),
        np.asarray(fold["target_sd"], dtype=np.float32),
    )


def adjusted_inputs(config: dict[str, Any], fold_number: int):
    directory = ROOT / str(config["frozen_inputs"]["adjusted_array_directory"])
    features = np.load(directory / "tessera.npy", mmap_mode="r")
    with np.load(directory / f"fold_{fold_number}.npz") as source:
        fold = {name: source[name] for name in source.files}
    return features, fold


def run(variant: str, fold_number: int, seed: int) -> None:
    config = load_config()
    settings = config["model"]
    result_directory = ROOT / str(config["outputs"]["result_directory"])
    result_directory.mkdir(parents=True, exist_ok=True)
    result_path = result_directory / f"{variant}_fold{fold_number}_seed{seed}.npz"
    metadata_path = result_path.with_suffix(".json")
    if result_path.exists() and metadata_path.exists():
        print(f"resumed {result_path.name}", flush=True)
        return
    configure_threads()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if variant == "raw":
        features, targets, fold, weights, y_mean, y_sd = raw_inputs(
            config, fold_number
        )
        subtrain = fold["subtrain_indices"].astype(np.int64)
        validation = fold["validation_indices"].astype(np.int64)
        test = fold["test_indices"].astype(np.int64)
        model, x_mean, x_sd, y_mean, y_sd, best_epoch, history = (
            fit_selected_model(
                features,
                targets,
                subtrain,
                validation,
                weights,
                settings,
                seed,
                device,
                y_mean,
                y_sd,
            )
        )
    else:
        features, fold = adjusted_inputs(config, fold_number)
        train = fold["train_indices"].astype(np.int64)
        subtrain = fold["subtrain_indices"].astype(np.int64)
        validation = fold["validation_indices"].astype(np.int64)
        test = fold["test_indices"].astype(np.int64)
        weights = fold["weights"].astype(np.float32)
        tuning_targets = fold["tuning_targets"].astype(np.float32)
        _, _, _, _, _, best_epoch, history = fit_selected_model(
            features,
            tuning_targets,
            subtrain,
            validation,
            weights,
            settings,
            seed,
            device,
        )
        targets = fold["final_targets"].astype(np.float32)
        model, x_mean, x_sd, y_mean, y_sd = refit_fixed_epochs(
            features,
            targets,
            train,
            weights,
            settings,
            seed,
            best_epoch,
            device,
        )
    predictions = predict_original_scale(
        model,
        features,
        test,
        x_mean,
        x_sd,
        y_mean,
        y_sd,
        settings,
        device,
    )
    temporary = result_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        test_indices=test,
        observed=np.asarray(targets[test], dtype=np.float32),
        predictions=predictions,
        best_epoch=np.asarray([best_epoch], dtype=np.int16),
    )
    temporary.replace(result_path)
    atomic_json(
        metadata_path,
        {
            "variant": variant,
            "fold": fold_number,
            "seed": seed,
            "device": str(device),
            "best_epoch": best_epoch,
            "augmentation": "random reflections in training; four-reflection mean in validation and testing",
            "config_sha256": sha256(CONFIG_PATH),
            "history": history,
        },
    )
    print(f"completed {result_path.name}; best_epoch={best_epoch}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["raw", "height_adjusted"], required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.variant, arguments.fold, arguments.seed)

