#!/usr/bin/env python3
"""Train one dense Cairngorms model, geographic fold, and seed."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/cairngorms_dense_10m.yaml"


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "cairngorms_dense_10m"
    ]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def configure_threads() -> None:
    threads = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)


def paths(config: dict[str, Any]) -> dict[str, Path]:
    dense = ROOT / str(config["outputs"]["dense_directory"])
    return {
        "dense": dense,
        "embedding": dense / "embedding_int8.npy",
        "scale": dense / "embedding_scale.npy",
        "valid": dense / "tessera_valid.npy",
        "row": dense / "row.npy",
        "column": dense / "column.npy",
        "block": dense / "spatial_block.npy",
        "targets": dense / "target_rows.npy",
        "index_grid": dense / "index_grid.npy",
        "folds": dense / "folds",
        "results": ROOT / str(config["outputs"]["result_directory"]),
    }


def load_arrays(config: dict[str, Any]) -> dict[str, np.ndarray]:
    p = paths(config)
    return {
        "embedding": np.load(p["embedding"], mmap_mode="r"),
        "scale": np.load(p["scale"], mmap_mode="r"),
        "valid": np.load(p["valid"], mmap_mode="r"),
        "row": np.load(p["row"], mmap_mode="r"),
        "column": np.load(p["column"], mmap_mode="r"),
        "block": np.load(p["block"], mmap_mode="r"),
        "targets": np.load(p["targets"], mmap_mode="r"),
        "index_grid": np.load(p["index_grid"], mmap_mode="r"),
    }


class SummaryMLP(nn.Module):
    def __init__(self, output_dimensions: int, hidden: list[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = 128 * 5
        for width in hidden:
            layers.extend([nn.Linear(current, width), nn.GELU(), nn.Dropout(dropout)])
            current = width
        layers.append(nn.Linear(current, output_dimensions))
        self.network = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(min(8, channels), channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(min(8, channels), channels),
        )
        self.activation = nn.GELU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.activation(values + self.network(values))


class ResidualPatchCNN(nn.Module):
    def __init__(
        self,
        output_dimensions: int,
        channels: int,
        blocks: int,
        head: list[int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(129, channels, 1),
            nn.GroupNorm(min(8, channels), channels),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(channels, dropout) for _ in range(blocks)])
        layers: list[nn.Module] = []
        current = channels * 5 * 5
        for width in head:
            layers.extend([nn.Linear(current, width), nn.GELU(), nn.Dropout(dropout)])
            current = width
        layers.append(nn.Linear(current, output_dimensions))
        self.head = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.stem(values)).flatten(start_dim=1))


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        groups = min(8, output_channels)
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.GroupNorm(groups, output_channels),
            nn.GELU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.GroupNorm(groups, output_channels),
            nn.GELU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class CompactUNet(nn.Module):
    def __init__(self, output_dimensions: int, base: int) -> None:
        super().__init__()
        self.input_projection = nn.Conv2d(129, base, 1)
        self.encoder1 = ConvBlock(base, base)
        self.encoder2 = ConvBlock(base, base * 2)
        self.bottleneck = ConvBlock(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.decoder2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.decoder1 = ConvBlock(base * 2, base)
        self.output = nn.Conv2d(base, output_dimensions, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        first = self.encoder1(self.input_projection(values))
        second = self.encoder2(self.pool(first))
        centre = self.bottleneck(self.pool(second))
        second_up = self.up2(centre)
        second_decoded = self.decoder2(torch.cat([second_up, second], dim=1))
        first_up = self.up1(second_decoded)
        return self.output(self.decoder1(torch.cat([first_up, first], dim=1)))


def build_model(model_name: str, config: dict[str, Any], outputs: int) -> nn.Module:
    settings = config["models"]
    if model_name == "summary_mlp":
        return SummaryMLP(
            outputs,
            [int(value) for value in settings["summary_hidden_dimensions"]],
            float(settings["dropout"]),
        )
    if model_name == "residual_patch_cnn":
        return ResidualPatchCNN(
            outputs,
            int(settings["residual_cnn_channels"]),
            int(settings["residual_cnn_blocks"]),
            [int(value) for value in settings["residual_cnn_head"]],
            float(settings["dropout"]),
        )
    if model_name == "compact_unet":
        return CompactUNet(outputs, int(settings["unet_base_channels"]))
    raise ValueError(f"Unknown model: {model_name}")


def patch_batch(
    indices: np.ndarray,
    arrays: dict[str, np.ndarray],
    mean: np.ndarray,
    sd: np.ndarray,
) -> np.ndarray:
    rows = np.asarray(arrays["row"][indices], dtype=np.int64)
    columns = np.asarray(arrays["column"][indices], dtype=np.int64)
    values = np.zeros((len(indices), 128, 5, 5), dtype=np.float32)
    mask = np.zeros((len(indices), 5, 5), dtype=np.float32)
    for local_row, row_offset in enumerate(range(-2, 3)):
        for local_column, column_offset in enumerate(range(-2, 3)):
            valid = np.asarray(
                arrays["valid"][rows + row_offset, columns + column_offset], dtype=bool
            )
            if not valid.any():
                continue
            local = arrays["embedding"][
                rows[valid] + row_offset, columns[valid] + column_offset
            ].astype(np.float32)
            local *= arrays["scale"][
                rows[valid] + row_offset, columns[valid] + column_offset
            ].astype(np.float32)[:, None]
            local = (local - mean[None, :]) / sd[None, :]
            values[valid, :, local_row, local_column] = local
            mask[valid, local_row, local_column] = 1.0
    return np.concatenate([values, mask[:, None]], axis=1)


def summarize_patch(ordered: np.ndarray) -> np.ndarray:
    values = ordered[:, :128]
    mask = ordered[:, 128].astype(bool)
    count = np.maximum(mask.sum(axis=(1, 2)), 1).astype(np.float32)
    total = np.where(mask[:, None], values, 0.0).sum(axis=(2, 3))
    total_square = np.where(mask[:, None], values * values, 0.0).sum(axis=(2, 3))
    mean = total / count[:, None]
    standard_deviation = np.sqrt(
        np.maximum(total_square / count[:, None] - np.square(mean), 0.0)
    )
    offset = np.arange(5, dtype=np.float32) - 2
    x_grid = np.broadcast_to(offset[None, :], (5, 5))
    y_grid = np.broadcast_to(-offset[:, None], (5, 5))
    sum_x = np.where(mask, x_grid[None], 0.0).sum(axis=(1, 2))
    sum_y = np.where(mask, y_grid[None], 0.0).sum(axis=(1, 2))
    sum_x2 = np.where(mask, x_grid[None] ** 2, 0.0).sum(axis=(1, 2))
    sum_y2 = np.where(mask, y_grid[None] ** 2, 0.0).sum(axis=(1, 2))
    sum_x_value = np.where(
        mask[:, None], values * x_grid[None, None], 0.0
    ).sum(axis=(2, 3))
    sum_y_value = np.where(
        mask[:, None], values * y_grid[None, None], 0.0
    ).sum(axis=(2, 3))
    denominator_x = sum_x2 - np.square(sum_x) / count
    denominator_y = sum_y2 - np.square(sum_y) / count
    dx = np.zeros_like(mean)
    dy = np.zeros_like(mean)
    usable_x = denominator_x > 0
    usable_y = denominator_y > 0
    dx[usable_x] = (
        sum_x_value[usable_x]
        - sum_x[usable_x, None] * total[usable_x] / count[usable_x, None]
    ) / denominator_x[usable_x, None]
    dy[usable_y] = (
        sum_y_value[usable_y]
        - sum_y[usable_y, None] * total[usable_y] / count[usable_y, None]
    ) / denominator_y[usable_y, None]
    centre = values[:, :, 2, 2]
    return np.concatenate([centre, mean, standard_deviation, dx, dy], axis=1).astype(
        np.float32
    )


def block_weights(block: np.ndarray, indices: np.ndarray) -> np.ndarray:
    selected = np.asarray(block[indices], dtype=np.int64)
    unique, inverse, counts = np.unique(selected, return_inverse=True, return_counts=True)
    del unique
    weights = 1.0 / counts[inverse].astype(np.float64)
    return (weights * len(weights) / weights.sum()).astype(np.float32)


def weighted_vector_loss(
    predicted: torch.Tensor, observed: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    row_loss = torch.square(predicted - observed).mean(dim=1)
    return torch.sum(row_loss * weights) / torch.sum(weights)


def scalar_input(
    model_name: str,
    indices: np.ndarray,
    arrays: dict[str, np.ndarray],
    input_mean: np.ndarray,
    input_sd: np.ndarray,
) -> torch.Tensor:
    ordered = patch_batch(indices, arrays, input_mean, input_sd)
    if model_name == "summary_mlp":
        ordered = summarize_patch(ordered)
    return torch.from_numpy(np.ascontiguousarray(ordered))


def scalar_loss(
    model: nn.Module,
    model_name: str,
    indices: np.ndarray,
    weights: np.ndarray,
    arrays: dict[str, np.ndarray],
    input_mean: np.ndarray,
    input_sd: np.ndarray,
    target_mean: np.ndarray,
    target_sd: np.ndarray,
    batch_size: int,
) -> float:
    model.eval()
    total = 0.0
    total_weight = 0.0
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            local_weights = weights[start : start + batch_size]
            x = scalar_input(model_name, selected, arrays, input_mean, input_sd)
            y = (
                np.asarray(arrays["targets"][selected], dtype=np.float32)
                - target_mean[None]
            ) / target_sd[None]
            prediction = model(x)
            row_loss = torch.square(prediction - torch.from_numpy(y)).mean(dim=1)
            total += float(torch.sum(row_loss * torch.from_numpy(local_weights)).item())
            total_weight += float(local_weights.sum())
    return total / total_weight


def train_scalar(
    model_name: str,
    model: nn.Module,
    fold: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
    config: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    subtrain = fold["subtrain_indices"].astype(np.int64)
    validation = fold["validation_indices"].astype(np.int64)
    test = fold["test_indices"].astype(np.int64)
    input_mean = fold["input_mean"].astype(np.float32)
    input_sd = fold["input_sd"].astype(np.float32)
    target_mean = fold["target_mean"].astype(np.float32)
    target_sd = fold["target_sd"].astype(np.float32)
    train_weights = block_weights(arrays["block"], subtrain)
    validation_weights = block_weights(arrays["block"], validation)
    batch_size = int(config["models"]["batch_size"][model_name])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["models"]["learning_rate"]),
        weight_decay=float(config["models"]["weight_decay"]),
    )
    rng = np.random.default_rng(seed)
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []
    maximum = int(config["models"]["maximum_epochs"][model_name])
    minimum = int(config["models"]["minimum_epochs"])
    patience = int(config["models"]["early_stopping_patience"])
    for epoch in range(1, maximum + 1):
        order = rng.permutation(len(subtrain))
        model.train()
        for start in range(0, len(order), batch_size):
            positions = order[start : start + batch_size]
            selected = subtrain[positions]
            x = scalar_input(model_name, selected, arrays, input_mean, input_sd)
            y = (
                np.asarray(arrays["targets"][selected], dtype=np.float32)
                - target_mean[None]
            ) / target_sd[None]
            weights = torch.from_numpy(train_weights[positions])
            optimizer.zero_grad(set_to_none=True)
            loss = weighted_vector_loss(model(x), torch.from_numpy(y), weights)
            loss.backward()
            optimizer.step()
        validation_loss = scalar_loss(
            model,
            model_name,
            validation,
            validation_weights,
            arrays,
            input_mean,
            input_sd,
            target_mean,
            target_sd,
            batch_size,
        )
        history.append({"epoch": epoch, "validation_loss": validation_loss})
        print(f"epoch {epoch:03d}: validation={validation_loss:.6f}", flush=True)
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if epoch >= minimum and stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("No scalar checkpoint was retained")
    model.load_state_dict(best_state)
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(test), batch_size):
            selected = test[start : start + batch_size]
            standardized = model(
                scalar_input(model_name, selected, arrays, input_mean, input_sd)
            ).numpy()
            predictions.append(standardized * target_sd[None] + target_mean[None])
    return np.concatenate(predictions).astype(np.float32), {
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "history": history,
    }


def chip_starts(length: int, size: int, stride: int) -> list[int]:
    starts = list(range(0, max(length - size + 1, 1), stride))
    final = max(0, length - size)
    if not starts or starts[-1] != final:
        starts.append(final)
    return starts


def select_chips(
    index_grid: np.ndarray,
    membership: np.ndarray,
    size: int,
    stride: int,
    minimum: int,
) -> list[tuple[int, int]]:
    selected: list[tuple[int, int]] = []
    for row in chip_starts(index_grid.shape[0], size, stride):
        for column in chip_starts(index_grid.shape[1], size, stride):
            ids = np.asarray(index_grid[row : row + size, column : column + size])
            valid = ids >= 0
            count = int(membership[ids[valid]].sum()) if valid.any() else 0
            if count >= minimum:
                selected.append((row, column))
    return selected


def unet_batch(
    starts: list[tuple[int, int]],
    membership: np.ndarray,
    row_weights: np.ndarray,
    arrays: dict[str, np.ndarray],
    input_mean: np.ndarray,
    input_sd: np.ndarray,
    target_mean: np.ndarray,
    target_sd: np.ndarray,
    size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = np.zeros((len(starts), 129, size, size), dtype=np.float32)
    y = np.zeros((len(starts), len(target_mean), size, size), dtype=np.float32)
    weight = np.zeros((len(starts), size, size), dtype=np.float32)
    for batch_index, (row, column) in enumerate(starts):
        valid = np.asarray(arrays["valid"][row : row + size, column : column + size], dtype=bool)
        values = arrays["embedding"][row : row + size, column : column + size].astype(np.float32)
        values *= arrays["scale"][row : row + size, column : column + size].astype(np.float32)[:, :, None]
        values = (values - input_mean[None, None]) / input_sd[None, None]
        values[~valid] = 0.0
        x[batch_index, :128] = values.transpose(2, 0, 1)
        x[batch_index, 128] = valid.astype(np.float32)
        ids = np.asarray(arrays["index_grid"][row : row + size, column : column + size], dtype=np.int64)
        usable = ids >= 0
        usable &= np.where(usable, membership[np.maximum(ids, 0)], False)
        if usable.any():
            selected_ids = ids[usable]
            standardized = (
                np.asarray(arrays["targets"][selected_ids], dtype=np.float32)
                - target_mean[None]
            ) / target_sd[None]
            target_view = y[batch_index]
            target_view[:, usable] = standardized.T
            weight_view = weight[batch_index]
            weight_view[usable] = row_weights[selected_ids]
    return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(weight)


def weighted_map_loss(
    prediction: torch.Tensor, observed: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    pixel_loss = torch.square(prediction - observed).mean(dim=1)
    return torch.sum(pixel_loss * weights) / torch.sum(weights)


def unet_validation_loss(
    model: nn.Module,
    chips: list[tuple[int, int]],
    membership: np.ndarray,
    row_weights: np.ndarray,
    arrays: dict[str, np.ndarray],
    fold: dict[str, np.ndarray],
    config: dict[str, Any],
) -> float:
    size = int(config["models"]["unet_chip_cells"])
    batch_size = int(config["models"]["batch_size"]["compact_unet"])
    total = 0.0
    total_weight = 0.0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(chips), batch_size):
            x, y, weight = unet_batch(
                chips[start : start + batch_size],
                membership,
                row_weights,
                arrays,
                fold["input_mean"],
                fold["input_sd"],
                fold["target_mean"],
                fold["target_sd"],
                size,
            )
            loss = torch.square(model(x) - y).mean(dim=1)
            total += float(torch.sum(loss * weight).item())
            total_weight += float(weight.sum().item())
    return total / total_weight


def train_unet(
    model: nn.Module,
    fold: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
    config: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    row_count = len(arrays["row"])
    subtrain_membership = np.zeros(row_count, dtype=bool)
    validation_membership = np.zeros(row_count, dtype=bool)
    test_membership = np.zeros(row_count, dtype=bool)
    subtrain_membership[fold["subtrain_indices"]] = True
    validation_membership[fold["validation_indices"]] = True
    test_membership[fold["test_indices"]] = True
    subtrain_weights = np.zeros(row_count, dtype=np.float32)
    validation_weights = np.zeros(row_count, dtype=np.float32)
    subtrain_weights[fold["subtrain_indices"]] = block_weights(
        arrays["block"], fold["subtrain_indices"]
    )
    validation_weights[fold["validation_indices"]] = block_weights(
        arrays["block"], fold["validation_indices"]
    )
    size = int(config["models"]["unet_chip_cells"])
    stride = int(config["models"]["unet_chip_stride_cells"])
    train_chips = select_chips(
        arrays["index_grid"],
        subtrain_membership,
        size,
        stride,
        int(config["models"]["unet_minimum_training_targets_per_chip"]),
    )
    validation_chips = select_chips(
        arrays["index_grid"], validation_membership, size, stride, 1
    )
    test_chips = select_chips(arrays["index_grid"], test_membership, size, stride, 1)
    print(
        f"chips: train={len(train_chips)} validation={len(validation_chips)} test={len(test_chips)}",
        flush=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["models"]["learning_rate"]),
        weight_decay=float(config["models"]["weight_decay"]),
    )
    rng = np.random.default_rng(seed)
    batch_size = int(config["models"]["batch_size"]["compact_unet"])
    maximum = int(config["models"]["maximum_epochs"]["compact_unet"])
    minimum = int(config["models"]["minimum_epochs"])
    patience = int(config["models"]["early_stopping_patience"])
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, maximum + 1):
        order = rng.permutation(len(train_chips))
        model.train()
        for start in range(0, len(order), batch_size):
            local_starts = [train_chips[i] for i in order[start : start + batch_size]]
            x, y, weights = unet_batch(
                local_starts,
                subtrain_membership,
                subtrain_weights,
                arrays,
                fold["input_mean"],
                fold["input_sd"],
                fold["target_mean"],
                fold["target_sd"],
                size,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = weighted_map_loss(model(x), y, weights)
            loss.backward()
            optimizer.step()
        validation_loss = unet_validation_loss(
            model,
            validation_chips,
            validation_membership,
            validation_weights,
            arrays,
            fold,
            config,
        )
        history.append({"epoch": epoch, "validation_loss": validation_loss})
        print(f"epoch {epoch:03d}: validation={validation_loss:.6f}", flush=True)
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if epoch >= minimum and stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("No U-Net checkpoint was retained")
    model.load_state_dict(best_state)
    model.eval()

    test_indices = fold["test_indices"].astype(np.int64)
    test_lookup = np.full(row_count, -1, dtype=np.int32)
    test_lookup[test_indices] = np.arange(len(test_indices), dtype=np.int32)
    prediction_sum = np.zeros((len(test_indices), arrays["targets"].shape[1]), dtype=np.float64)
    prediction_count = np.zeros(len(test_indices), dtype=np.int16)
    neutral_weights = np.ones(row_count, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(test_chips), batch_size):
            local_starts = test_chips[start : start + batch_size]
            x, _, _ = unet_batch(
                local_starts,
                test_membership,
                neutral_weights,
                arrays,
                fold["input_mean"],
                fold["input_sd"],
                fold["target_mean"],
                fold["target_sd"],
                size,
            )
            standardized = model(x).numpy()
            physical = (
                standardized * fold["target_sd"][None, :, None, None]
                + fold["target_mean"][None, :, None, None]
            )
            for batch_index, (row, column) in enumerate(local_starts):
                ids = np.asarray(
                    arrays["index_grid"][row : row + size, column : column + size],
                    dtype=np.int64,
                )
                usable = ids >= 0
                usable &= np.where(usable, test_lookup[np.maximum(ids, 0)] >= 0, False)
                local = test_lookup[ids[usable]]
                prediction_view = physical[batch_index]
                np.add.at(prediction_sum, local, prediction_view[:, usable].T)
                np.add.at(prediction_count, local, 1)
    if (prediction_count == 0).any():
        raise RuntimeError("U-Net inference did not cover every test location")
    prediction = prediction_sum / prediction_count[:, None]
    return prediction.astype(np.float32), {
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "history": history,
        "train_chips": len(train_chips),
        "validation_chips": len(validation_chips),
        "test_chips": len(test_chips),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    config = load_config()
    if args.model not in config["models"]["names"]:
        raise ValueError(f"Unknown model: {args.model}")
    set_seed(args.seed)
    configure_threads()
    p = paths(config)
    p["results"].mkdir(parents=True, exist_ok=True)
    result_path = p["results"] / f"{args.model}_fold{args.fold}_seed{args.seed}.npz"
    metadata_path = result_path.with_suffix(".json")
    if result_path.exists() and metadata_path.exists():
        print(f"resumed {result_path.name}", flush=True)
        return
    arrays = load_arrays(config)
    with np.load(p["folds"] / f"fold_{args.fold}.npz") as source:
        fold = {name: source[name] for name in source.files}
    model = build_model(args.model, config, arrays["targets"].shape[1])
    print(
        f"model={args.model} fold={args.fold} seed={args.seed} parameters={sum(v.numel() for v in model.parameters()):,}",
        flush=True,
    )
    if args.model == "compact_unet":
        prediction, training = train_unet(model, fold, arrays, config, args.seed)
    else:
        prediction, training = train_scalar(
            args.model, model, fold, arrays, config, args.seed
        )
    test = fold["test_indices"].astype(np.int64)
    temporary = result_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        test_indices=test,
        observed=np.asarray(arrays["targets"][test], dtype=np.float32),
        predicted=prediction,
    )
    temporary.replace(result_path)
    metadata_path.write_text(
        json.dumps(
            {
                "model": args.model,
                "fold": args.fold,
                "seed": args.seed,
                "test_rows": len(test),
                "parameters": sum(v.numel() for v in model.parameters()),
                **training,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"completed {result_path.name}", flush=True)


if __name__ == "__main__":
    main()
