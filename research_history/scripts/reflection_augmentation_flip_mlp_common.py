#!/usr/bin/env python3
"""Shared reflection-aware MLP training for the Phase 33 sensitivity test."""

from __future__ import annotations

import copy
import os
import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn


CHANNELS = 128
SUMMARY_WIDTH = CHANNELS * 5
DX = slice(CHANNELS * 3, CHANNELS * 4)
DY = slice(CHANNELS * 4, CHANNELS * 5)
REFLECTIONS = ((1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0))


class MultiTargetMLP(nn.Module):
    def __init__(self, outputs: int, hidden: list[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = SUMMARY_WIDTH
        for next_width in hidden:
            layers.extend(
                [nn.Linear(width, next_width), nn.GELU(), nn.Dropout(dropout)]
            )
            width = next_width
        layers.append(nn.Linear(width, outputs))
        self.network = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def configure_threads() -> None:
    threads = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)


def reflect_summary(
    values: np.ndarray,
    horizontal: float | np.ndarray,
    vertical: float | np.ndarray,
) -> np.ndarray:
    """Reflect 5x5 summary features without reconstructing the ordered patch."""
    if values.ndim != 2 or values.shape[1] != SUMMARY_WIDTH:
        raise ValueError(f"Expected (n, {SUMMARY_WIDTH}) summaries, got {values.shape}")
    reflected = np.asarray(values, dtype=np.float32).copy()
    horizontal_values = np.asarray(horizontal, dtype=np.float32)
    vertical_values = np.asarray(vertical, dtype=np.float32)
    if horizontal_values.ndim == 0:
        reflected[:, DX] *= float(horizontal_values)
    else:
        reflected[:, DX] *= horizontal_values[:, None]
    if vertical_values.ndim == 0:
        reflected[:, DY] *= float(vertical_values)
    else:
        reflected[:, DY] *= vertical_values[:, None]
    return reflected


def reflection_statistics(
    features: np.ndarray, indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Exact feature moments over all four reflected copies of the training rows."""
    selected = np.asarray(features[indices], dtype=np.float32)
    if selected.shape[1] != SUMMARY_WIDTH or not np.isfinite(selected).all():
        raise RuntimeError("TESSERA summary features are invalid")
    mean = selected.mean(axis=0, dtype=np.float64).astype(np.float32)
    sd = selected.std(axis=0, dtype=np.float64).astype(np.float32)
    mean[DX] = 0.0
    mean[DY] = 0.0
    sd[DX] = np.sqrt(
        np.mean(np.square(selected[:, DX]), axis=0, dtype=np.float64)
    ).astype(np.float32)
    sd[DY] = np.sqrt(
        np.mean(np.square(selected[:, DY]), axis=0, dtype=np.float64)
    ).astype(np.float32)
    sd[sd < 1e-6] = 1.0
    return mean, sd


def target_statistics(
    targets: np.ndarray, indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    selected = np.asarray(targets[indices], dtype=np.float32)
    mean = selected.mean(axis=0, dtype=np.float64).astype(np.float32)
    sd = selected.std(axis=0, dtype=np.float64).astype(np.float32)
    sd[sd < 1e-6] = 1.0
    return mean, sd


def random_reflected_batch(
    features: np.ndarray,
    indices: np.ndarray,
    mean: np.ndarray,
    sd: np.ndarray,
    rng: np.random.Generator,
) -> torch.Tensor:
    horizontal = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), len(indices))
    vertical = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), len(indices))
    values = reflect_summary(features[indices], horizontal, vertical)
    values = (values - mean) / sd
    return torch.from_numpy(np.ascontiguousarray(values))


def tta_standardized_prediction(
    model: nn.Module,
    features: np.ndarray,
    indices: np.ndarray,
    mean: np.ndarray,
    sd: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            total = None
            for horizontal, vertical in REFLECTIONS:
                values = reflect_summary(features[selected], horizontal, vertical)
                values = (values - mean) / sd
                prediction = model(torch.from_numpy(values).to(device)).cpu().numpy()
                total = prediction if total is None else total + prediction
            assert total is not None
            rows.append(total / len(REFLECTIONS))
    return np.concatenate(rows).astype(np.float32)


def validation_loss(
    model: nn.Module,
    features: np.ndarray,
    targets: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    x_mean: np.ndarray,
    x_sd: np.ndarray,
    y_mean: np.ndarray,
    y_sd: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> float:
    prediction = tta_standardized_prediction(
        model, features, indices, x_mean, x_sd, batch_size, device
    )
    observed = (np.asarray(targets[indices], dtype=np.float32) - y_mean) / y_sd
    row_loss = np.square(prediction - observed).mean(axis=1)
    return float(np.average(row_loss, weights=weights[indices]))


def train_epochs(
    model: nn.Module,
    features: np.ndarray,
    targets: np.ndarray,
    train: np.ndarray,
    weights: np.ndarray,
    x_mean: np.ndarray,
    x_sd: np.ndarray,
    y_mean: np.ndarray,
    y_sd: np.ndarray,
    settings: dict[str, Any],
    epochs: int,
    rng: np.random.Generator,
    device: torch.device,
    validation: np.ndarray | None = None,
) -> tuple[int, float, dict[str, torch.Tensor], list[dict[str, float]]]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    batch_size = int(settings["batch_size"])
    best_epoch = epochs
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(train)
        total = 0.0
        total_weight = 0.0
        for start in range(0, len(order), batch_size):
            selected = order[start : start + batch_size]
            x_tensor = random_reflected_batch(
                features, selected, x_mean, x_sd, rng
            ).to(device)
            observed = (
                np.asarray(targets[selected], dtype=np.float32) - y_mean
            ) / y_sd
            y_tensor = torch.from_numpy(observed).to(device)
            local_weights = np.asarray(weights[selected], dtype=np.float32)
            weight_tensor = torch.from_numpy(local_weights).to(device)
            optimizer.zero_grad(set_to_none=True)
            row_loss = torch.square(model(x_tensor) - y_tensor).mean(dim=1)
            loss = torch.sum(row_loss * weight_tensor) / torch.sum(weight_tensor)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * float(local_weights.sum())
            total_weight += float(local_weights.sum())
        training_loss = total / total_weight
        current_loss = training_loss
        if validation is not None:
            current_loss = validation_loss(
                model,
                features,
                targets,
                validation,
                weights,
                x_mean,
                x_sd,
                y_mean,
                y_sd,
                batch_size,
                device,
            )
        history.append(
            {
                "epoch": epoch,
                "training_loss": training_loss,
                "validation_loss": current_loss,
            }
        )
        print(
            f"epoch {epoch:03d}: train={training_loss:.6f} "
            f"validation={current_loss:.6f}",
            flush=True,
        )
        if current_loss < best_loss - 1e-6:
            best_loss = current_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if (
            validation is not None
            and epoch >= int(settings["minimum_epochs"])
            and stale >= int(settings["early_stopping_patience"])
        ):
            break
    if best_state is None:
        raise RuntimeError("No model checkpoint was retained")
    return best_epoch, best_loss, best_state, history


def new_model(outputs: int, settings: dict[str, Any], device: torch.device) -> nn.Module:
    return MultiTargetMLP(
        outputs,
        [int(value) for value in settings["hidden_dimensions"]],
        float(settings["dropout"]),
    ).to(device)


def fit_selected_model(
    features: np.ndarray,
    targets: np.ndarray,
    subtrain: np.ndarray,
    validation: np.ndarray,
    weights: np.ndarray,
    settings: dict[str, Any],
    seed: int,
    device: torch.device,
    y_mean: np.ndarray | None = None,
    y_sd: np.ndarray | None = None,
) -> tuple[nn.Module, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, list[dict[str, float]]]:
    set_seed(seed)
    x_mean, x_sd = reflection_statistics(features, subtrain)
    if y_mean is None or y_sd is None:
        y_mean, y_sd = target_statistics(targets, subtrain)
    model = new_model(targets.shape[1], settings, device)
    best_epoch, _, best_state, history = train_epochs(
        model,
        features,
        targets,
        subtrain,
        weights,
        x_mean,
        x_sd,
        y_mean,
        y_sd,
        settings,
        int(settings["maximum_epochs"]),
        np.random.default_rng(seed),
        device,
        validation,
    )
    model.load_state_dict(best_state)
    return model, x_mean, x_sd, y_mean, y_sd, best_epoch, history


def refit_fixed_epochs(
    features: np.ndarray,
    targets: np.ndarray,
    train: np.ndarray,
    weights: np.ndarray,
    settings: dict[str, Any],
    seed: int,
    epochs: int,
    device: torch.device,
) -> tuple[nn.Module, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    set_seed(seed)
    x_mean, x_sd = reflection_statistics(features, train)
    y_mean, y_sd = target_statistics(targets, train)
    model = new_model(targets.shape[1], settings, device)
    train_epochs(
        model,
        features,
        targets,
        train,
        weights,
        x_mean,
        x_sd,
        y_mean,
        y_sd,
        settings,
        epochs,
        np.random.default_rng(seed),
        device,
        None,
    )
    return model, x_mean, x_sd, y_mean, y_sd


def predict_original_scale(
    model: nn.Module,
    features: np.ndarray,
    indices: np.ndarray,
    x_mean: np.ndarray,
    x_sd: np.ndarray,
    y_mean: np.ndarray,
    y_sd: np.ndarray,
    settings: dict[str, Any],
    device: torch.device,
) -> np.ndarray:
    standardized = tta_standardized_prediction(
        model,
        features,
        indices,
        x_mean,
        x_sd,
        int(settings["batch_size"]),
        device,
    )
    return (standardized * y_sd + y_mean).astype(np.float32)

