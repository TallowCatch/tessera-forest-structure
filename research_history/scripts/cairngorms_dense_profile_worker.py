#!/usr/bin/env python3
"""Train one direct-profile model, geographic fold, and seed."""

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

import cairngorms_dense_10m_worker as dense_worker


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/cairngorms_dense_profile.yaml"


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "cairngorms_dense_profile"
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
    dense = ROOT / str(config["source"]["dense_directory"])
    profile = ROOT / str(config["outputs"]["profile_directory"])
    return {
        "dense": dense,
        "embedding": dense / "embedding_int8.npy",
        "scale": dense / "embedding_scale.npy",
        "valid": dense / "tessera_valid.npy",
        "row": dense / "row.npy",
        "column": dense / "column.npy",
        "block": dense / "spatial_block.npy",
        "index_grid": dense / "index_grid.npy",
        "folds": dense / "folds",
        "targets": profile / "profile_targets.npy",
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
        "index_grid": np.load(p["index_grid"], mmap_mode="r"),
        "targets": np.load(p["targets"], mmap_mode="r"),
    }


def profile_probabilities(logits: torch.Tensor, channel_dimension: int) -> torch.Tensor:
    return torch.softmax(logits, dim=channel_dimension)


def profile_loss_rows(
    logits: torch.Tensor,
    observed: torch.Tensor,
    channel_dimension: int,
    config: dict[str, Any],
) -> torch.Tensor:
    settings = config["models"]["loss"]
    epsilon = float(settings["numerical_epsilon"])
    predicted = profile_probabilities(logits, channel_dimension)
    observed = torch.clamp(observed, min=0.0)
    observed = observed / torch.clamp(
        observed.sum(dim=channel_dimension, keepdim=True), min=epsilon
    )
    mixture = 0.5 * (predicted + observed)
    predicted_term = torch.where(
        predicted > 0,
        predicted * (torch.log(torch.clamp(predicted, min=epsilon)) - torch.log(torch.clamp(mixture, min=epsilon))),
        torch.zeros_like(predicted),
    )
    observed_term = torch.where(
        observed > 0,
        observed * (torch.log(torch.clamp(observed, min=epsilon)) - torch.log(torch.clamp(mixture, min=epsilon))),
        torch.zeros_like(observed),
    )
    divergence = 0.5 * (predicted_term + observed_term).sum(dim=channel_dimension)
    squared_error = torch.square(predicted - observed).mean(dim=channel_dimension)
    return (
        float(settings["jensen_shannon_weight"]) * divergence
        + float(settings["proportion_mse_weight"]) * squared_error
    )


def weighted_loss(losses: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.sum(losses * weights) / torch.sum(weights)


def scalar_validation_loss(
    model: nn.Module,
    model_name: str,
    indices: np.ndarray,
    weights: np.ndarray,
    arrays: dict[str, np.ndarray],
    fold: dict[str, np.ndarray],
    config: dict[str, Any],
) -> float:
    batch_size = int(config["models"]["batch_size"][model_name])
    total = 0.0
    total_weight = 0.0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            local_weights = torch.from_numpy(weights[start : start + batch_size])
            x = dense_worker.scalar_input(
                model_name,
                selected,
                arrays,
                fold["input_mean"].astype(np.float32),
                fold["input_sd"].astype(np.float32),
            )
            observed = torch.from_numpy(
                np.asarray(arrays["targets"][selected], dtype=np.float32)
            )
            losses = profile_loss_rows(model(x), observed, 1, config)
            total += float(torch.sum(losses * local_weights).item())
            total_weight += float(local_weights.sum().item())
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
    train_weights = dense_worker.block_weights(arrays["block"], subtrain)
    validation_weights = dense_worker.block_weights(arrays["block"], validation)
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
            x = dense_worker.scalar_input(
                model_name,
                selected,
                arrays,
                fold["input_mean"].astype(np.float32),
                fold["input_sd"].astype(np.float32),
            )
            observed = torch.from_numpy(
                np.asarray(arrays["targets"][selected], dtype=np.float32)
            )
            weights = torch.from_numpy(train_weights[positions])
            optimizer.zero_grad(set_to_none=True)
            loss = weighted_loss(profile_loss_rows(model(x), observed, 1, config), weights)
            loss.backward()
            optimizer.step()
        validation_loss = scalar_validation_loss(
            model,
            model_name,
            validation,
            validation_weights,
            arrays,
            fold,
            config,
        )
        history.append({"epoch": epoch, "validation_loss": validation_loss})
        print(f"epoch {epoch:03d}: validation={validation_loss:.6f}", flush=True)
        if validation_loss < best_loss - 1e-7:
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
            logits = model(
                dense_worker.scalar_input(
                    model_name,
                    selected,
                    arrays,
                    fold["input_mean"].astype(np.float32),
                    fold["input_sd"].astype(np.float32),
                )
            )
            predictions.append(profile_probabilities(logits, 1).numpy())
    return np.concatenate(predictions).astype(np.float32), {
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "history": history,
    }


def unet_batch(
    starts: list[tuple[int, int]],
    membership: np.ndarray,
    row_weights: np.ndarray,
    arrays: dict[str, np.ndarray],
    fold: dict[str, np.ndarray],
    size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = np.zeros((len(starts), 129, size, size), dtype=np.float32)
    y = np.zeros((len(starts), 3, size, size), dtype=np.float32)
    weight = np.zeros((len(starts), size, size), dtype=np.float32)
    input_mean = fold["input_mean"].astype(np.float32)
    input_sd = fold["input_sd"].astype(np.float32)
    for batch_index, (row, column) in enumerate(starts):
        valid = np.asarray(
            arrays["valid"][row : row + size, column : column + size], dtype=bool
        )
        values = arrays["embedding"][row : row + size, column : column + size].astype(
            np.float32
        )
        values *= arrays["scale"][row : row + size, column : column + size].astype(
            np.float32
        )[:, :, None]
        values = (values - input_mean[None, None]) / input_sd[None, None]
        values[~valid] = 0.0
        x[batch_index, :128] = values.transpose(2, 0, 1)
        x[batch_index, 128] = valid.astype(np.float32)
        ids = np.asarray(
            arrays["index_grid"][row : row + size, column : column + size],
            dtype=np.int64,
        )
        usable = ids >= 0
        usable &= np.where(usable, membership[np.maximum(ids, 0)], False)
        if usable.any():
            selected = ids[usable]
            target_view = y[batch_index]
            target_view[:, usable] = np.asarray(
                arrays["targets"][selected], dtype=np.float32
            ).T
            weight_view = weight[batch_index]
            weight_view[usable] = row_weights[selected]
    return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(weight)


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
            x, y, weights = unet_batch(
                chips[start : start + batch_size],
                membership,
                row_weights,
                arrays,
                fold,
                size,
            )
            losses = profile_loss_rows(model(x), y, 1, config)
            total += float(torch.sum(losses * weights).item())
            total_weight += float(weights.sum().item())
    return total / total_weight


def train_unet(
    model: nn.Module,
    fold: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
    config: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    row_count = len(arrays["row"])
    memberships = {
        name: np.zeros(row_count, dtype=bool)
        for name in ["subtrain", "validation", "test"]
    }
    memberships["subtrain"][fold["subtrain_indices"]] = True
    memberships["validation"][fold["validation_indices"]] = True
    memberships["test"][fold["test_indices"]] = True
    subtrain_weights = np.zeros(row_count, dtype=np.float32)
    validation_weights = np.zeros(row_count, dtype=np.float32)
    subtrain_weights[fold["subtrain_indices"]] = dense_worker.block_weights(
        arrays["block"], fold["subtrain_indices"]
    )
    validation_weights[fold["validation_indices"]] = dense_worker.block_weights(
        arrays["block"], fold["validation_indices"]
    )
    size = int(config["models"]["unet_chip_cells"])
    stride = int(config["models"]["unet_chip_stride_cells"])
    train_chips = dense_worker.select_chips(
        arrays["index_grid"],
        memberships["subtrain"],
        size,
        stride,
        int(config["models"]["unet_minimum_training_targets_per_chip"]),
    )
    validation_chips = dense_worker.select_chips(
        arrays["index_grid"], memberships["validation"], size, stride, 1
    )
    test_chips = dense_worker.select_chips(
        arrays["index_grid"], memberships["test"], size, stride, 1
    )
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
            local_starts = [train_chips[index] for index in order[start : start + batch_size]]
            x, y, weights = unet_batch(
                local_starts,
                memberships["subtrain"],
                subtrain_weights,
                arrays,
                fold,
                size,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = weighted_loss(profile_loss_rows(model(x), y, 1, config), weights)
            loss.backward()
            optimizer.step()
        validation_loss = unet_validation_loss(
            model,
            validation_chips,
            memberships["validation"],
            validation_weights,
            arrays,
            fold,
            config,
        )
        history.append({"epoch": epoch, "validation_loss": validation_loss})
        print(f"epoch {epoch:03d}: validation={validation_loss:.6f}", flush=True)
        if validation_loss < best_loss - 1e-7:
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
    prediction_sum = np.zeros((len(test_indices), 3), dtype=np.float64)
    prediction_count = np.zeros(len(test_indices), dtype=np.int16)
    neutral_weights = np.ones(row_count, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(test_chips), batch_size):
            local_starts = test_chips[start : start + batch_size]
            x, _, _ = unet_batch(
                local_starts,
                memberships["test"],
                neutral_weights,
                arrays,
                fold,
                size,
            )
            probabilities = profile_probabilities(model(x), 1).numpy()
            for batch_index, (row, column) in enumerate(local_starts):
                ids = np.asarray(
                    arrays["index_grid"][row : row + size, column : column + size],
                    dtype=np.int64,
                )
                usable = ids >= 0
                usable &= np.where(
                    usable, test_lookup[np.maximum(ids, 0)] >= 0, False
                )
                local = test_lookup[ids[usable]]
                np.add.at(
                    prediction_sum,
                    local,
                    probabilities[batch_index][:, usable].T,
                )
                np.add.at(prediction_count, local, 1)
    if (prediction_count == 0).any():
        raise RuntimeError("U-Net inference did not cover every test location")
    prediction = prediction_sum / prediction_count[:, None]
    prediction /= prediction.sum(axis=1, keepdims=True)
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
    model = dense_worker.build_model(args.model, config, 3)
    print(
        f"model={args.model} fold={args.fold} seed={args.seed} parameters={sum(value.numel() for value in model.parameters()):,}",
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
                "parameters": sum(value.numel() for value in model.parameters()),
                "output_constraint": "softmax simplex",
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
