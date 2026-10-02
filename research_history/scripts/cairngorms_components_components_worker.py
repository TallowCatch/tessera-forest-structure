#!/usr/bin/env python3
"""Fit one Phase 20 predictor, spatial fold, and random seed."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/cairngorms_components.yaml"
ARRAY_DIR = ROOT / "data/interim/phase20_cairngorms_components"
RESULT_DIR = ROOT / "data/interim/phase20_cairngorms_component_results"


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "phase20_cairngorms_components"
    ]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


class MultiTargetMLP(nn.Module):
    def __init__(self, input_dimensions: int, output_dimensions: int, hidden: list[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = input_dimensions
        for width in hidden:
            layers.extend([nn.Linear(current, width), nn.GELU(), nn.Dropout(dropout)])
            current = width
        layers.append(nn.Linear(current, output_dimensions))
        self.network = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def feature_paths(config: dict[str, Any], model: str) -> list[Path]:
    frozen = config["frozen_inputs"]
    paths = {
        "tessera": [ROOT / str(frozen["tessera_features_path"])],
        "conventional": [ROOT / str(frozen["conventional_features_path"])],
        "fused": [
            ROOT / str(frozen["tessera_features_path"]),
            ROOT / str(frozen["conventional_features_path"]),
        ],
    }
    if model not in paths:
        raise RuntimeError(f"Unknown model: {model}")
    return paths[model]


def load_features(config: dict[str, Any], model: str) -> np.ndarray:
    arrays = [np.load(path, mmap_mode="r") for path in feature_paths(config, model)]
    if len({array.shape[0] for array in arrays}) != 1:
        raise RuntimeError("Feature arrays do not share the same rows")
    if len(arrays) == 1:
        return arrays[0]
    return np.concatenate([np.asarray(array, dtype=np.float32) for array in arrays], axis=1)


def statistics(values: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    subset = np.asarray(values[indices], dtype=np.float32)
    mean = subset.mean(axis=0, dtype=np.float64).astype(np.float32)
    sd = subset.std(axis=0, dtype=np.float64).astype(np.float32)
    sd[sd < 1e-6] = 1.0
    return mean, sd


def train_epoch(
    model: nn.Module,
    inputs: np.ndarray,
    targets: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    x_mean: np.ndarray,
    x_sd: np.ndarray,
    y_mean: np.ndarray,
    y_sd: np.ndarray,
    batch_size: int,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    rng: np.random.Generator | None,
) -> float:
    training = optimizer is not None
    model.train(training)
    order = rng.permutation(indices) if training and rng is not None else indices
    total = 0.0
    total_weight = 0.0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for start in range(0, len(order), batch_size):
            selected = order[start : start + batch_size]
            x = (np.asarray(inputs[selected], dtype=np.float32) - x_mean) / x_sd
            y = (np.asarray(targets[selected], dtype=np.float32) - y_mean) / y_sd
            batch_weights = np.asarray(weights[selected], dtype=np.float32)
            x_tensor = torch.from_numpy(x).to(device)
            y_tensor = torch.from_numpy(y).to(device)
            weight_tensor = torch.from_numpy(batch_weights).to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            per_row = torch.mean(torch.square(model(x_tensor) - y_tensor), dim=1)
            loss = torch.sum(per_row * weight_tensor) / torch.sum(weight_tensor)
            if training:
                loss.backward()
                optimizer.step()
            weight_sum = float(batch_weights.sum())
            total += float(loss.detach().cpu()) * weight_sum
            total_weight += weight_sum
    return total / total_weight


def fit_model(
    inputs: np.ndarray,
    targets: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray | None,
    weights: np.ndarray,
    epochs: int,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[nn.Module, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, list[dict[str, float]]]:
    settings = config["neural"]
    x_mean, x_sd = statistics(inputs, train_indices)
    y_mean, y_sd = statistics(targets, train_indices)
    model = MultiTargetMLP(
        inputs.shape[1], targets.shape[1],
        [int(value) for value in settings["hidden_dimensions"]],
        float(settings["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    rng = np.random.default_rng(torch.initial_seed() % (2**32))
    best_loss = math.inf
    best_epoch = epochs
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        training_loss = train_epoch(
            model, inputs, targets, train_indices, weights, x_mean, x_sd,
            y_mean, y_sd, int(settings["batch_size"]), device, optimizer, rng,
        )
        validation_loss = training_loss
        if validation_indices is not None:
            validation_loss = train_epoch(
                model, inputs, targets, validation_indices, weights, x_mean, x_sd,
                y_mean, y_sd, int(settings["batch_size"]), device, None, None,
            )
        history.append({"epoch": epoch, "training_loss": training_loss, "validation_loss": validation_loss})
        print(f"epoch {epoch:03d}: train={training_loss:.6f} validation={validation_loss:.6f}", flush=True)
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if validation_indices is not None and epoch >= int(settings["minimum_epochs"]) and stale >= int(settings["early_stopping_patience"]):
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, x_mean, x_sd, y_mean, y_sd, best_epoch, history


def predict(
    model: nn.Module,
    inputs: np.ndarray,
    indices: np.ndarray,
    x_mean: np.ndarray,
    x_sd: np.ndarray,
    y_mean: np.ndarray,
    y_sd: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            x = (np.asarray(inputs[selected], dtype=np.float32) - x_mean) / x_sd
            output = model(torch.from_numpy(x).to(device)).cpu().numpy()
            rows.append(output * y_sd + y_mean)
    return np.concatenate(rows).astype(np.float32)


def run(model_name: str, fold: int, seed: int) -> None:
    config = load_config()
    output_path = RESULT_DIR / f"{model_name}_fold_{fold}_seed_{seed}.npz"
    manifest_path = output_path.with_suffix(".json")
    if output_path.exists() and manifest_path.exists():
        print(f"resumed {model_name} fold {fold} seed {seed}", flush=True)
        return
    fold_path = ARRAY_DIR / f"fold_{fold}.npz"
    if not fold_path.exists():
        raise RuntimeError("Prepared Phase 20 fold is missing")
    inputs = load_features(config, model_name)
    with np.load(fold_path) as values:
        weights = values["weights"].astype(np.float32)
        train = values["train_indices"].astype(np.int64)
        subtrain = values["subtrain_indices"].astype(np.int64)
        validation = values["validation_indices"].astype(np.int64)
        test = values["test_indices"].astype(np.int64)
        tuning_targets = values["tuning_targets"].astype(np.float32)
        final_targets = values["final_targets"].astype(np.float32)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"{model_name} fold {fold} seed {seed}: device={device}; "
        f"subtrain={len(subtrain):,}; validation={len(validation):,}; test={len(test):,}",
        flush=True,
    )
    _, _, _, _, _, best_epoch, history = fit_model(
        inputs, tuning_targets, subtrain, validation, weights,
        int(config["neural"]["maximum_epochs"]), config, device,
    )
    set_seed(seed)
    final_model, x_mean, x_sd, y_mean, y_sd, _, _ = fit_model(
        inputs, final_targets, train, None, weights, best_epoch, config, device,
    )
    predictions = predict(
        final_model, inputs, test, x_mean, x_sd, y_mean, y_sd,
        int(config["neural"]["batch_size"]), device,
    )
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        test_indices=test,
        observed=final_targets[test],
        predictions=predictions,
        best_epoch=np.asarray([best_epoch], dtype=np.int16),
    )
    temporary.replace(output_path)
    atomic_json(
        manifest_path,
        {"model": model_name, "fold": fold, "seed": seed, "device": str(device), "best_epoch": best_epoch, "history": history},
    )
    print(f"completed {model_name} fold {fold} seed {seed}; best_epoch={best_epoch}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.model, arguments.fold, arguments.seed)
