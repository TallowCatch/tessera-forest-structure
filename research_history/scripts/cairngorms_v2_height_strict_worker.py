#!/usr/bin/env python3
"""Fit one strict-ownership TESSERA v2 canopy-height U-Net."""

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
import yaml

import cairngorms_dense_10m_worker as dense
import cairngorms_v2_height_worker as phase30


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/cairngorms_v2_height_strict_unet.yaml"


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "phase31_cairngorms_v2_height_strict_unet"
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


def load_arrays(config: dict[str, Any]) -> dict[str, np.ndarray]:
    directory = ROOT / str(config["frozen_inputs"]["phase30_array_directory"])
    return {
        "embedding": np.load(directory / "embedding_int8.npy", mmap_mode="r"),
        "scale": np.load(directory / "embedding_scale.npy", mmap_mode="r"),
        "valid": np.load(directory / "tessera_valid.npy", mmap_mode="r"),
        "row": np.load(directory / "row.npy", mmap_mode="r"),
        "column": np.load(directory / "column.npy", mmap_mode="r"),
        "block": np.load(directory / "spatial_block.npy", mmap_mode="r"),
        "index_grid": np.load(directory / "index_grid.npy", mmap_mode="r"),
        "targets": np.load(directory / "targets.npy", mmap_mode="r"),
    }


def assign_target_chips(
    index_grid: np.ndarray,
    selected: np.ndarray,
    chips: list[tuple[int, int]],
    size: int,
) -> np.ndarray:
    """Assign each selected target to its most centrally positioned containing chip."""
    owner = np.full(len(selected), -1, dtype=np.int32)
    best_distance = np.full(len(selected), np.inf, dtype=np.float32)
    centre = (size - 1) / 2.0
    for chip_index, (row, column) in enumerate(chips):
        ids = np.asarray(index_grid[row : row + size, column : column + size], dtype=np.int64)
        usable = ids >= 0
        usable &= np.where(usable, selected[np.maximum(ids, 0)], False)
        if not usable.any():
            continue
        local_rows, local_columns = np.nonzero(usable)
        local_ids = ids[usable]
        distance = np.square(local_rows - centre) + np.square(local_columns - centre)
        better = distance < best_distance[local_ids]
        if better.any():
            chosen = local_ids[better]
            owner[chosen] = chip_index
            best_distance[chosen] = distance[better]
    selected_ids = np.flatnonzero(selected)
    if np.any(owner[selected_ids] < 0):
        raise RuntimeError("At least one selected target has no containing U-Net chip")
    return owner


def select_partition(
    arrays: dict[str, np.ndarray], indices: np.ndarray, config: dict[str, Any]
) -> tuple[np.ndarray, list[tuple[int, int]], np.ndarray, np.ndarray]:
    settings = config["model"]
    selected = phase30.membership(len(arrays["targets"]), indices)
    chips = dense.select_chips(
        arrays["index_grid"],
        selected,
        int(settings["chip_cells"]),
        int(settings["chip_stride_cells"]),
        int(settings["training_chip_minimum_targets"]),
    )
    if not chips:
        raise RuntimeError("No U-Net chips cover the selected partition")
    owner = assign_target_chips(
        arrays["index_grid"], selected, chips, int(settings["chip_cells"])
    )
    active = np.unique(owner[indices])
    return selected, chips, owner, active


def image_batch(
    chip_indices: np.ndarray,
    chips: list[tuple[int, int]],
    arrays: dict[str, np.ndarray],
    input_mean: np.ndarray,
    input_sd: np.ndarray,
    size: int,
) -> torch.Tensor:
    x = np.zeros((len(chip_indices), 129, size, size), dtype=np.float32)
    for batch_index, chip_index in enumerate(chip_indices):
        row, column = chips[int(chip_index)]
        valid = np.asarray(
            arrays["valid"][row : row + size, column : column + size], dtype=bool
        )
        values = np.asarray(
            arrays["embedding"][row : row + size, column : column + size],
            dtype=np.float32,
        )
        values *= np.asarray(
            arrays["scale"][row : row + size, column : column + size],
            dtype=np.float32,
        )[:, :, None]
        values = (values - input_mean[None, None]) / input_sd[None, None]
        values[~valid] = 0.0
        x[batch_index, :128] = values.transpose(2, 0, 1)
        x[batch_index, 128] = valid.astype(np.float32)
    return torch.from_numpy(x)


def strict_batch(
    chip_indices: np.ndarray,
    chips: list[tuple[int, int]],
    owner: np.ndarray,
    row_weights: np.ndarray,
    arrays: dict[str, np.ndarray],
    stats: dict[str, np.ndarray],
    size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = image_batch(
        chip_indices, chips, arrays, stats["input_mean"], stats["input_sd"], size
    )
    y = np.zeros(
        (len(chip_indices), arrays["targets"].shape[1], size, size), dtype=np.float32
    )
    weights = np.zeros((len(chip_indices), size, size), dtype=np.float32)
    for batch_index, chip_index in enumerate(chip_indices):
        row, column = chips[int(chip_index)]
        ids = np.asarray(
            arrays["index_grid"][row : row + size, column : column + size],
            dtype=np.int64,
        )
        usable = ids >= 0
        usable &= np.where(usable, owner[np.maximum(ids, 0)] == chip_index, False)
        if usable.any():
            selected_ids = ids[usable]
            standardized = (
                np.asarray(arrays["targets"][selected_ids], dtype=np.float32)
                - stats["target_mean"][None]
            ) / stats["target_sd"][None]
            y[batch_index][:, usable] = standardized.T
            weights[batch_index][usable] = row_weights[selected_ids]
    return x, torch.from_numpy(y), torch.from_numpy(weights)


def optimizer_denominator(total_weight: float, batches: int) -> float:
    if total_weight <= 0 or batches <= 0:
        raise ValueError("Positive total weight and batch count are required")
    return total_weight / batches


def partition_loss(
    model: torch.nn.Module,
    active_chips: np.ndarray,
    chips: list[tuple[int, int]],
    owner: np.ndarray,
    weights: np.ndarray,
    arrays: dict[str, np.ndarray],
    stats: dict[str, np.ndarray],
    config: dict[str, Any],
) -> float:
    batch_size = int(config["model"]["batch_size"])
    size = int(config["model"]["chip_cells"])
    numerator = 0.0
    denominator = 0.0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(active_chips), batch_size):
            x, y, local_weights = strict_batch(
                active_chips[start : start + batch_size],
                chips, owner, weights, arrays, stats, size,
            )
            pixel_loss = torch.square(model(x) - y).mean(dim=1)
            numerator += float(torch.sum(pixel_loss * local_weights))
            denominator += float(local_weights.sum())
    if denominator <= 0:
        raise RuntimeError("Strict U-Net partition contains no supervised weight")
    return numerator / denominator


def fit(
    arrays: dict[str, np.ndarray],
    train: np.ndarray,
    validation: np.ndarray | None,
    epochs: int,
    config: dict[str, Any],
    seed: int,
) -> tuple[torch.nn.Module, dict[str, np.ndarray], int, list[dict[str, float]], dict[str, Any]]:
    settings = config["model"]
    input_mean, input_sd = phase30.dense_input_statistics(arrays, train)
    target_mean, target_sd = phase30.statistics(arrays["targets"], train)
    stats = {
        "input_mean": input_mean,
        "input_sd": input_sd,
        "target_mean": target_mean,
        "target_sd": target_sd,
    }
    train_selected, train_chips, train_owner, train_active = select_partition(
        arrays, train, config
    )
    del train_selected
    train_weights = phase30.block_weights(arrays["block"], train, len(arrays["targets"]))
    validation_chips: list[tuple[int, int]] = []
    validation_owner = np.full(len(arrays["targets"]), -1, dtype=np.int32)
    validation_active = np.asarray([], dtype=np.int32)
    validation_weights = np.zeros(len(arrays["targets"]), dtype=np.float32)
    if validation is not None:
        _, validation_chips, validation_owner, validation_active = select_partition(
            arrays, validation, config
        )
        validation_weights = phase30.block_weights(
            arrays["block"], validation, len(arrays["targets"])
        )

    model = dense.CompactUNet(arrays["targets"].shape[1], int(settings["unet_base_channels"]))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    rng = np.random.default_rng(seed)
    batch_size = int(settings["batch_size"])
    size = int(settings["chip_cells"])
    total_train_weight = float(train_weights[train].sum())
    batch_count = int(math.ceil(len(train_active) / batch_size))
    step_denominator = optimizer_denominator(total_train_weight, batch_count)
    minimum = int(settings["minimum_epochs"])
    patience = int(settings["early_stopping_patience"])
    best_loss = math.inf
    best_epoch = epochs
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(train_active)
        training_numerator = 0.0
        training_weight = 0.0
        for start in range(0, len(order), batch_size):
            x, y, local_weights = strict_batch(
                order[start : start + batch_size],
                train_chips, train_owner, train_weights, arrays, stats, size,
            )
            optimizer.zero_grad(set_to_none=True)
            pixel_loss = torch.square(model(x) - y).mean(dim=1)
            numerator = torch.sum(pixel_loss * local_weights)
            loss = numerator / step_denominator
            loss.backward()
            optimizer.step()
            training_numerator += float(numerator.detach())
            training_weight += float(local_weights.sum())
        if not np.isclose(training_weight, total_train_weight, rtol=1e-5, atol=1e-4):
            raise RuntimeError("Targets were not supervised exactly once in this epoch")
        training_loss = training_numerator / training_weight
        validation_loss = training_loss
        if validation is not None:
            validation_loss = partition_loss(
                model, validation_active, validation_chips, validation_owner,
                validation_weights, arrays, stats, config,
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
        best_epoch = phase30.selected_epoch(history, minimum)
    diagnostics = {
        "selected_rows": int(len(train)),
        "candidate_chips": int(len(train_chips)),
        "active_owned_chips": int(len(train_active)),
        "targets_supervised_per_epoch": int((train_owner >= 0).sum()),
        "maximum_target_multiplicity": 1,
        "complete_partition_weight": total_train_weight,
        "optimizer_denominator": step_denominator,
    }
    return model, stats, best_epoch, history, diagnostics


def predict(
    model: torch.nn.Module,
    arrays: dict[str, np.ndarray],
    test: np.ndarray,
    stats: dict[str, np.ndarray],
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, int]]:
    settings = config["model"]
    selected = phase30.membership(len(arrays["targets"]), test)
    chips = dense.select_chips(
        arrays["index_grid"], selected,
        int(settings["chip_cells"]), int(settings["chip_stride_cells"]), 1,
    )
    coverage = phase30.coverage_counts(
        arrays["index_grid"], selected, chips, int(settings["chip_cells"])
    )
    if np.any(coverage[test] <= 0):
        raise RuntimeError("Strict U-Net inference does not cover every test target")
    lookup = np.full(len(arrays["targets"]), -1, dtype=np.int32)
    lookup[test] = np.arange(len(test), dtype=np.int32)
    prediction_sum = np.zeros((len(test), arrays["targets"].shape[1]), dtype=np.float64)
    prediction_count = np.zeros(len(test), dtype=np.int16)
    batch_size = int(settings["batch_size"])
    size = int(settings["chip_cells"])
    chip_indices = np.arange(len(chips), dtype=np.int32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(chip_indices), batch_size):
            local_indices = chip_indices[start : start + batch_size]
            x = image_batch(
                local_indices, chips, arrays,
                stats["input_mean"], stats["input_sd"], size,
            )
            standardized = model(x).numpy()
            physical = (
                standardized * stats["target_sd"][None, :, None, None]
                + stats["target_mean"][None, :, None, None]
            )
            for batch_index, chip_index in enumerate(local_indices):
                row, column = chips[int(chip_index)]
                ids = np.asarray(
                    arrays["index_grid"][row : row + size, column : column + size],
                    dtype=np.int64,
                )
                usable = ids >= 0
                usable &= np.where(usable, selected[np.maximum(ids, 0)], False)
                local = lookup[ids[usable]]
                np.add.at(prediction_sum, local, physical[batch_index][:, usable].T)
                np.add.at(prediction_count, local, 1)
    if not np.array_equal(prediction_count, coverage[test]):
        raise RuntimeError("Strict U-Net inference multiplicity audit failed")
    return (prediction_sum / prediction_count[:, None]).astype(np.float32), {
        "test_chips": int(len(chips)),
        "test_coverage_minimum": int(prediction_count.min()),
        "test_coverage_maximum": int(prediction_count.max()),
    }


def run(fold: int, seed: int) -> None:
    config = load_config()
    arrays = load_arrays(config)
    result_dir = ROOT / str(config["outputs"]["result_directory"])
    result_path = result_dir / f"tessera_v2_unet_strict_fold_{fold}_seed_{seed}.npz"
    metadata_path = result_path.with_suffix(".json")
    if result_path.exists() and metadata_path.exists():
        print(f"resumed {result_path.name}", flush=True)
        return
    fold_path = (
        ROOT / str(config["frozen_inputs"]["phase30_array_directory"])
        / "folds" / f"fold_{fold}.npz"
    )
    with np.load(fold_path) as values:
        train = values["train_indices"].astype(np.int64)
        subtrain = values["subtrain_indices"].astype(np.int64)
        validation = values["validation_indices"].astype(np.int64)
        test = values["test_indices"].astype(np.int64)
    print(
        f"strict V2 U-Net fold={fold} seed={seed}; train={len(train):,}; "
        f"subtrain={len(subtrain):,}; validation={len(validation):,}; test={len(test):,}",
        flush=True,
    )
    set_seed(seed)
    _, _, best_epoch, history, tuning_audit = fit(
        arrays, subtrain, validation, int(config["model"]["maximum_epochs"]), config, seed
    )
    set_seed(seed)
    final_model, stats, _, _, final_audit = fit(
        arrays, train, None, best_epoch, config, seed
    )
    predictions, test_audit = predict(final_model, arrays, test, stats, config)
    result_dir.mkdir(parents=True, exist_ok=True)
    temporary = result_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        test_indices=test,
        observed=np.asarray(arrays["targets"][test], dtype=np.float32),
        predictions=predictions,
        best_epoch=np.asarray([best_epoch], dtype=np.int16),
    )
    temporary.replace(result_path)
    atomic_json(
        metadata_path,
        {
            "model": str(config["evaluation"]["rerun_model"]),
            "fold": fold,
            "seed": seed,
            "best_epoch": best_epoch,
            "parameters": int(sum(value.numel() for value in final_model.parameters())),
            "tuning_history": history,
            "ownership_audit": {
                "tuning": tuning_audit,
                "final": final_audit,
                "test": test_audit,
            },
        },
    )
    print(
        f"completed strict V2 U-Net fold={fold} seed={seed}; best_epoch={best_epoch}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    configure_threads()
    run(args.fold, args.seed)


if __name__ == "__main__":
    main()
