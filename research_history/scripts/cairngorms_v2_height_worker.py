#!/usr/bin/env python3
"""Fit one audited TESSERA v2 canopy-height model, fold, and seed."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml

import cairngorms_dense_10m_worker as dense


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/cairngorms_v2_height_unet.yaml"


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "phase30_cairngorms_v2_height_unet"
    ]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def configure_threads() -> None:
    threads = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)


class HeightMLP(nn.Module):
    def __init__(
        self, input_dimensions: int, output_dimensions: int, hidden: list[int], dropout: float
    ) -> None:
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


def load_arrays(config: dict[str, Any]) -> dict[str, np.ndarray]:
    directory = ROOT / str(config["outputs"]["array_directory"])
    return {
        "summary": np.load(directory / "tessera_summary.npy", mmap_mode="r"),
        "embedding": np.load(directory / "embedding_int8.npy", mmap_mode="r"),
        "scale": np.load(directory / "embedding_scale.npy", mmap_mode="r"),
        "valid": np.load(directory / "tessera_valid.npy", mmap_mode="r"),
        "row": np.load(directory / "row.npy", mmap_mode="r"),
        "column": np.load(directory / "column.npy", mmap_mode="r"),
        "block": np.load(directory / "spatial_block.npy", mmap_mode="r"),
        "index_grid": np.load(directory / "index_grid.npy", mmap_mode="r"),
        "targets": np.load(directory / "targets.npy", mmap_mode="r"),
    }


def statistics(values: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    local = np.asarray(values[indices], dtype=np.float32)
    mean = local.mean(axis=0, dtype=np.float64).astype(np.float32)
    sd = local.std(axis=0, dtype=np.float64).astype(np.float32)
    sd[sd < 1e-6] = 1.0
    return mean, sd


def dense_input_statistics(
    arrays: dict[str, np.ndarray], indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray(arrays["row"][indices], dtype=np.int64)
    columns = np.asarray(arrays["column"][indices], dtype=np.int64)
    values = np.asarray(arrays["embedding"][rows, columns], dtype=np.float32)
    values *= np.asarray(arrays["scale"][rows, columns], dtype=np.float32)[:, None]
    return statistics(values, np.arange(len(values), dtype=np.int64))


def block_weights(blocks: np.ndarray, indices: np.ndarray, count: int) -> np.ndarray:
    output = np.zeros(count, dtype=np.float32)
    selected = np.asarray(blocks[indices])
    _, inverse, counts = np.unique(selected, return_inverse=True, return_counts=True)
    local = 1.0 / counts[inverse].astype(np.float32)
    local *= len(local) / local.sum()
    output[indices] = local
    return output


def membership(count: int, indices: np.ndarray) -> np.ndarray:
    selected = np.zeros(count, dtype=bool)
    selected[indices] = True
    return selected


def coverage_counts(
    index_grid: np.ndarray,
    selected: np.ndarray,
    chips: list[tuple[int, int]],
    size: int,
) -> np.ndarray:
    counts = np.zeros(len(selected), dtype=np.int16)
    for row, column in chips:
        ids = np.asarray(index_grid[row : row + size, column : column + size], dtype=np.int64)
        usable = ids >= 0
        usable &= np.where(usable, selected[np.maximum(ids, 0)], False)
        if usable.any():
            np.add.at(counts, ids[usable], 1)
    return counts


def coverage_adjusted_weights(
    base_weights: np.ndarray, coverage: np.ndarray, indices: np.ndarray
) -> np.ndarray:
    adjusted = np.zeros_like(base_weights, dtype=np.float32)
    local_coverage = coverage[indices]
    if np.any(local_coverage <= 0):
        raise RuntimeError("At least one selected U-Net target is not covered by a chip")
    adjusted[indices] = base_weights[indices] / local_coverage.astype(np.float32)
    return adjusted


def selected_epoch(history: list[dict[str, float]], minimum: int) -> int:
    eligible = [row for row in history if int(row["epoch"]) >= minimum]
    if not eligible:
        raise RuntimeError("Training ended before the minimum selectable epoch")
    return int(min(eligible, key=lambda row: (row["validation_loss"], row["epoch"]))["epoch"])


def mlp_loss(
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
) -> float:
    model.eval()
    total = 0.0
    total_weight = 0.0
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            x = (np.asarray(features[selected], dtype=np.float32) - x_mean) / x_sd
            y = (np.asarray(targets[selected], dtype=np.float32) - y_mean) / y_sd
            local_weights = torch.from_numpy(weights[selected])
            row_loss = torch.square(model(torch.from_numpy(x)) - torch.from_numpy(y)).mean(dim=1)
            total += float(torch.sum(row_loss * local_weights))
            total_weight += float(local_weights.sum())
    if total_weight <= 0:
        raise RuntimeError("MLP evaluation has no positive row weight")
    return total / total_weight


def fit_mlp(
    arrays: dict[str, np.ndarray],
    train: np.ndarray,
    validation: np.ndarray | None,
    epochs: int,
    config: dict[str, Any],
    seed: int,
) -> tuple[nn.Module, dict[str, np.ndarray], int, list[dict[str, float]]]:
    settings = config["model"]
    features = arrays["summary"]
    targets = arrays["targets"]
    count = len(targets)
    x_mean, x_sd = statistics(features, train)
    y_mean, y_sd = statistics(targets, train)
    model = HeightMLP(
        features.shape[1],
        targets.shape[1],
        [int(value) for value in settings["mlp_hidden_dimensions"]],
        float(settings["dropout"]),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    weights = block_weights(arrays["block"], train, count)
    if validation is not None:
        weights += block_weights(arrays["block"], validation, count)
    batch_size = int(settings["batch_size"]["tessera_v2_mlp_5x5"])
    rng = np.random.default_rng(seed)
    minimum = int(settings["minimum_epochs"])
    patience = int(settings["early_stopping_patience"])
    best_loss = math.inf
    best_epoch = epochs
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(train)
        training_total = 0.0
        training_weight = 0.0
        for start in range(0, len(order), batch_size):
            selected = order[start : start + batch_size]
            x = (np.asarray(features[selected], dtype=np.float32) - x_mean) / x_sd
            y = (np.asarray(targets[selected], dtype=np.float32) - y_mean) / y_sd
            local_weights = torch.from_numpy(weights[selected])
            optimizer.zero_grad(set_to_none=True)
            row_loss = torch.square(model(torch.from_numpy(x)) - torch.from_numpy(y)).mean(dim=1)
            loss = torch.sum(row_loss * local_weights) / torch.sum(local_weights)
            loss.backward()
            optimizer.step()
            weight_sum = float(local_weights.sum())
            training_total += float(loss.detach()) * weight_sum
            training_weight += weight_sum
        training_loss = training_total / training_weight
        validation_loss = training_loss
        if validation is not None:
            validation_loss = mlp_loss(
                model, features, targets, validation, weights,
                x_mean, x_sd, y_mean, y_sd, batch_size,
            )
        history.append(
            {"epoch": epoch, "training_loss": training_loss, "validation_loss": validation_loss}
        )
        print(
            f"epoch {epoch:03d}: train={training_loss:.6f} validation={validation_loss:.6f}",
            flush=True,
        )
        if validation is not None and epoch >= minimum:
            if validation_loss < best_loss - 1e-6:
                best_loss = validation_loss
                best_epoch = epoch
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                break
    if validation is not None:
        best_epoch = selected_epoch(history, minimum)
    stats = {"input_mean": x_mean, "input_sd": x_sd, "target_mean": y_mean, "target_sd": y_sd}
    return model, stats, best_epoch, history


def predict_mlp(
    model: nn.Module,
    arrays: dict[str, np.ndarray],
    test: np.ndarray,
    stats: dict[str, np.ndarray],
    config: dict[str, Any],
) -> np.ndarray:
    batch_size = int(config["model"]["batch_size"]["tessera_v2_mlp_5x5"])
    predictions: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(test), batch_size):
            selected = test[start : start + batch_size]
            x = (
                np.asarray(arrays["summary"][selected], dtype=np.float32)
                - stats["input_mean"]
            ) / stats["input_sd"]
            standardized = model(torch.from_numpy(x)).numpy()
            predictions.append(
                standardized * stats["target_sd"][None] + stats["target_mean"][None]
            )
    return np.concatenate(predictions).astype(np.float32)


def make_unet_batch(
    chips: list[tuple[int, int]],
    selected: np.ndarray,
    weights: np.ndarray,
    arrays: dict[str, np.ndarray],
    stats: dict[str, np.ndarray],
    size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return dense.unet_batch(
        chips,
        selected,
        weights,
        arrays,
        stats["input_mean"],
        stats["input_sd"],
        stats["target_mean"],
        stats["target_sd"],
        size,
    )


def unet_loss(
    model: nn.Module,
    chips: list[tuple[int, int]],
    selected: np.ndarray,
    weights: np.ndarray,
    arrays: dict[str, np.ndarray],
    stats: dict[str, np.ndarray],
    config: dict[str, Any],
) -> float:
    size = int(config["model"]["chip_cells"])
    batch_size = int(config["model"]["batch_size"]["tessera_v2_unet_audited"])
    total = 0.0
    total_weight = 0.0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(chips), batch_size):
            x, y, local_weights = make_unet_batch(
                chips[start : start + batch_size], selected, weights, arrays, stats, size
            )
            pixel_loss = torch.square(model(x) - y).mean(dim=1)
            total += float(torch.sum(pixel_loss * local_weights))
            total_weight += float(local_weights.sum())
    if total_weight <= 0:
        raise RuntimeError("U-Net evaluation has no positive target weight")
    return total / total_weight


def unet_partition(
    arrays: dict[str, np.ndarray],
    indices: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, list[tuple[int, int]], np.ndarray]:
    settings = config["model"]
    selected = membership(len(arrays["targets"]), indices)
    chips = dense.select_chips(
        arrays["index_grid"],
        selected,
        int(settings["chip_cells"]),
        int(settings["chip_stride_cells"]),
        int(settings["training_chip_minimum_targets"]),
    )
    if not chips:
        raise RuntimeError("No U-Net chips cover the selected rows")
    coverage = coverage_counts(
        arrays["index_grid"], selected, chips, int(settings["chip_cells"])
    )
    if np.any(coverage[indices] <= 0):
        raise RuntimeError("The fixed U-Net grid does not cover every selected row")
    return selected, chips, coverage


def fit_unet(
    arrays: dict[str, np.ndarray],
    train: np.ndarray,
    validation: np.ndarray | None,
    epochs: int,
    config: dict[str, Any],
    seed: int,
) -> tuple[nn.Module, dict[str, np.ndarray], int, list[dict[str, float]], dict[str, Any]]:
    settings = config["model"]
    count = len(arrays["targets"])
    input_mean, input_sd = dense_input_statistics(arrays, train)
    target_mean, target_sd = statistics(arrays["targets"], train)
    stats = {
        "input_mean": input_mean,
        "input_sd": input_sd,
        "target_mean": target_mean,
        "target_sd": target_sd,
    }
    train_selected, train_chips, train_coverage = unet_partition(arrays, train, config)
    train_base = block_weights(arrays["block"], train, count)
    train_weights = coverage_adjusted_weights(train_base, train_coverage, train)
    validation_selected = None
    validation_chips: list[tuple[int, int]] = []
    validation_coverage = np.zeros(count, dtype=np.int16)
    validation_weights = np.zeros(count, dtype=np.float32)
    if validation is not None:
        validation_selected, validation_chips, validation_coverage = unet_partition(
            arrays, validation, config
        )
        validation_base = block_weights(arrays["block"], validation, count)
        validation_weights = coverage_adjusted_weights(
            validation_base, validation_coverage, validation
        )

    model = dense.CompactUNet(arrays["targets"].shape[1], int(settings["unet_base_channels"]))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    rng = np.random.default_rng(seed)
    batch_size = int(settings["batch_size"]["tessera_v2_unet_audited"])
    size = int(settings["chip_cells"])
    minimum = int(settings["minimum_epochs"])
    patience = int(settings["early_stopping_patience"])
    best_loss = math.inf
    best_epoch = epochs
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train_chips))
        training_total = 0.0
        training_weight = 0.0
        for start in range(0, len(order), batch_size):
            chips = [train_chips[index] for index in order[start : start + batch_size]]
            x, y, local_weights = make_unet_batch(
                chips, train_selected, train_weights, arrays, stats, size
            )
            optimizer.zero_grad(set_to_none=True)
            pixel_loss = torch.square(model(x) - y).mean(dim=1)
            loss = torch.sum(pixel_loss * local_weights) / torch.sum(local_weights)
            loss.backward()
            optimizer.step()
            weight_sum = float(local_weights.sum())
            training_total += float(loss.detach()) * weight_sum
            training_weight += weight_sum
        training_loss = training_total / training_weight
        validation_loss = training_loss
        if validation is not None:
            assert validation_selected is not None
            validation_loss = unet_loss(
                model, validation_chips, validation_selected, validation_weights,
                arrays, stats, config,
            )
        history.append(
            {"epoch": epoch, "training_loss": training_loss, "validation_loss": validation_loss}
        )
        print(
            f"epoch {epoch:03d}: train={training_loss:.6f} validation={validation_loss:.6f}",
            flush=True,
        )
        if validation is not None and epoch >= minimum:
            if validation_loss < best_loss - 1e-6:
                best_loss = validation_loss
                best_epoch = epoch
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                break
    if validation is not None:
        best_epoch = selected_epoch(history, minimum)
    diagnostics = {
        "train_chips": len(train_chips),
        "train_coverage_minimum": int(train_coverage[train].min()),
        "train_coverage_maximum": int(train_coverage[train].max()),
        "train_rows_with_multiple_chips": int((train_coverage[train] > 1).sum()),
        "validation_chips": len(validation_chips),
        "validation_coverage_minimum": (
            int(validation_coverage[validation].min()) if validation is not None else None
        ),
        "validation_coverage_maximum": (
            int(validation_coverage[validation].max()) if validation is not None else None
        ),
    }
    return model, stats, best_epoch, history, diagnostics


def predict_unet(
    model: nn.Module,
    arrays: dict[str, np.ndarray],
    test: np.ndarray,
    stats: dict[str, np.ndarray],
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, int]]:
    selected, chips, coverage = unet_partition(arrays, test, config)
    count = len(arrays["targets"])
    lookup = np.full(count, -1, dtype=np.int32)
    lookup[test] = np.arange(len(test), dtype=np.int32)
    prediction_sum = np.zeros((len(test), arrays["targets"].shape[1]), dtype=np.float64)
    prediction_count = np.zeros(len(test), dtype=np.int16)
    neutral_weights = np.ones(count, dtype=np.float32)
    size = int(config["model"]["chip_cells"])
    batch_size = int(config["model"]["batch_size"]["tessera_v2_unet_audited"])
    model.eval()
    with torch.no_grad():
        for start in range(0, len(chips), batch_size):
            local_chips = chips[start : start + batch_size]
            x, _, _ = make_unet_batch(
                local_chips, selected, neutral_weights, arrays, stats, size
            )
            standardized = model(x).numpy()
            physical = (
                standardized * stats["target_sd"][None, :, None, None]
                + stats["target_mean"][None, :, None, None]
            )
            for batch_index, (row, column) in enumerate(local_chips):
                ids = np.asarray(
                    arrays["index_grid"][row : row + size, column : column + size],
                    dtype=np.int64,
                )
                usable = ids >= 0
                usable &= np.where(usable, selected[np.maximum(ids, 0)], False)
                local = lookup[ids[usable]]
                np.add.at(prediction_sum, local, physical[batch_index][:, usable].T)
                np.add.at(prediction_count, local, 1)
    if np.any(prediction_count == 0):
        raise RuntimeError("U-Net inference did not cover every test row")
    if not np.array_equal(prediction_count, coverage[test]):
        raise RuntimeError("Prediction multiplicity differs from audited chip coverage")
    return (prediction_sum / prediction_count[:, None]).astype(np.float32), {
        "test_chips": len(chips),
        "test_coverage_minimum": int(prediction_count.min()),
        "test_coverage_maximum": int(prediction_count.max()),
    }


def run(model_name: str, fold: int, seed: int) -> None:
    config = load_config()
    if model_name not in config["evaluation"]["models"]:
        raise ValueError(f"Unknown Phase 30 model: {model_name}")
    arrays = load_arrays(config)
    output_dir = ROOT / str(config["outputs"]["result_directory"])
    output_path = output_dir / f"{model_name}_fold_{fold}_seed_{seed}.npz"
    metadata_path = output_path.with_suffix(".json")
    if output_path.exists() and metadata_path.exists():
        print(f"resumed {output_path.name}", flush=True)
        return
    fold_path = (
        ROOT / str(config["outputs"]["array_directory"]) / "folds" / f"fold_{fold}.npz"
    )
    with np.load(fold_path) as values:
        train = values["train_indices"].astype(np.int64)
        subtrain = values["subtrain_indices"].astype(np.int64)
        validation = values["validation_indices"].astype(np.int64)
        test = values["test_indices"].astype(np.int64)
    maximum = int(config["model"]["maximum_epochs"])
    print(
        f"{model_name} fold={fold} seed={seed}; train={len(train):,}; "
        f"subtrain={len(subtrain):,}; validation={len(validation):,}; test={len(test):,}",
        flush=True,
    )
    set_seed(seed)
    if model_name == "tessera_v2_mlp_5x5":
        _, _, best_epoch, history = fit_mlp(
            arrays, subtrain, validation, maximum, config, seed
        )
        set_seed(seed)
        final_model, stats, _, _ = fit_mlp(
            arrays, train, None, best_epoch, config, seed
        )
        predictions = predict_mlp(final_model, arrays, test, stats, config)
        diagnostics: dict[str, Any] = {}
    else:
        _, _, best_epoch, history, tuning_diagnostics = fit_unet(
            arrays, subtrain, validation, maximum, config, seed
        )
        set_seed(seed)
        final_model, stats, _, _, final_diagnostics = fit_unet(
            arrays, train, None, best_epoch, config, seed
        )
        predictions, test_diagnostics = predict_unet(
            final_model, arrays, test, stats, config
        )
        diagnostics = {
            "tuning": tuning_diagnostics,
            "final": final_diagnostics,
            "test": test_diagnostics,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        test_indices=test,
        observed=np.asarray(arrays["targets"][test], dtype=np.float32),
        predictions=predictions,
        best_epoch=np.asarray([best_epoch], dtype=np.int16),
    )
    temporary.replace(output_path)
    atomic_json(
        metadata_path,
        {
            "model": model_name,
            "fold": fold,
            "seed": seed,
            "best_epoch": best_epoch,
            "parameters": int(sum(value.numel() for value in final_model.parameters())),
            "target_names": [str(value) for value in config["targets"]["names"]],
            "tuning_history": history,
            "coverage_audit": diagnostics,
        },
    )
    print(
        f"completed {model_name} fold={fold} seed={seed}; best_epoch={best_epoch}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    configure_threads()
    arguments = parse_args()
    run(arguments.model, arguments.fold, arguments.seed)
