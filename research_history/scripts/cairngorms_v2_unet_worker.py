#!/usr/bin/env python3
"""Fit one frozen TESSERA v2 U-Net fold and seed."""

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


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/cairngorms_v2_unet.yaml"


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "phase28_cairngorms_v2_unet"
    ]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def configure_threads() -> None:
    threads = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)


def load_arrays(config: dict[str, Any]) -> dict[str, np.ndarray]:
    directory = ROOT / str(config["outputs"]["array_directory"])
    return {
        "embedding": np.load(directory / "embedding_int8.npy", mmap_mode="r"),
        "scale": np.load(directory / "embedding_scale.npy", mmap_mode="r"),
        "valid": np.load(directory / "tessera_valid.npy", mmap_mode="r"),
        "row": np.load(directory / "row.npy", mmap_mode="r"),
        "column": np.load(directory / "column.npy", mmap_mode="r"),
        "block": np.load(directory / "spatial_block.npy", mmap_mode="r"),
        "index_grid": np.load(directory / "index_grid.npy", mmap_mode="r"),
    }


def statistics(
    arrays: dict[str, np.ndarray], targets: np.ndarray, indices: np.ndarray
) -> dict[str, np.ndarray]:
    rows = np.asarray(arrays["row"][indices], dtype=np.int64)
    columns = np.asarray(arrays["column"][indices], dtype=np.int64)
    centre = np.asarray(arrays["embedding"][rows, columns], dtype=np.float32)
    centre *= np.asarray(arrays["scale"][rows, columns], dtype=np.float32)[:, None]
    input_mean = centre.mean(axis=0, dtype=np.float64).astype(np.float32)
    input_sd = centre.std(axis=0, dtype=np.float64).astype(np.float32)
    input_sd[input_sd < 1e-6] = 1.0
    local_targets = np.asarray(targets[indices], dtype=np.float32)
    target_mean = local_targets.mean(axis=0, dtype=np.float64).astype(np.float32)
    target_sd = local_targets.std(axis=0, dtype=np.float64).astype(np.float32)
    target_sd[target_sd < 1e-6] = 1.0
    return {
        "input_mean": input_mean,
        "input_sd": input_sd,
        "target_mean": target_mean,
        "target_sd": target_sd,
    }


def membership(count: int, indices: np.ndarray) -> np.ndarray:
    output = np.zeros(count, dtype=bool)
    output[indices] = True
    return output


def select(
    arrays: dict[str, np.ndarray],
    selected: np.ndarray,
    config: dict[str, Any],
    minimum: int,
) -> list[tuple[int, int]]:
    settings = config["model"]
    return dense.select_chips(
        arrays["index_grid"],
        selected,
        int(settings["chip_cells"]),
        int(settings["chip_stride_cells"]),
        minimum,
    )


def make_batch(
    chips: list[tuple[int, int]],
    selected: np.ndarray,
    weights: np.ndarray,
    arrays: dict[str, np.ndarray],
    targets: np.ndarray,
    stats: dict[str, np.ndarray],
    size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    local_arrays = {**arrays, "targets": targets}
    return dense.unet_batch(
        chips,
        selected,
        weights,
        local_arrays,
        stats["input_mean"],
        stats["input_sd"],
        stats["target_mean"],
        stats["target_sd"],
        size,
    )


def validation_loss(
    model: torch.nn.Module,
    chips: list[tuple[int, int]],
    selected: np.ndarray,
    weights: np.ndarray,
    arrays: dict[str, np.ndarray],
    targets: np.ndarray,
    stats: dict[str, np.ndarray],
    config: dict[str, Any],
) -> float:
    settings = config["model"]
    size = int(settings["chip_cells"])
    batch_size = int(settings["batch_size"])
    total = 0.0
    total_weight = 0.0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(chips), batch_size):
            x, y, local_weights = make_batch(
                chips[start : start + batch_size],
                selected,
                weights,
                arrays,
                targets,
                stats,
                size,
            )
            loss = torch.square(model(x) - y).mean(dim=1)
            total += float(torch.sum(loss * local_weights).item())
            total_weight += float(local_weights.sum().item())
    if total_weight <= 0:
        raise RuntimeError("Validation chips contain no supervised rows")
    return total / total_weight


def fit(
    arrays: dict[str, np.ndarray],
    targets: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray | None,
    row_weights: np.ndarray,
    epochs: int,
    config: dict[str, Any],
    seed: int,
) -> tuple[torch.nn.Module, dict[str, np.ndarray], int, list[dict[str, float]], int]:
    settings = config["model"]
    count = len(arrays["row"])
    selected = membership(count, train_indices)
    stats = statistics(arrays, targets, train_indices)
    minimum = int(settings["minimum_training_targets_per_chip"])
    train_chips = select(arrays, selected, config, minimum)
    if not train_chips:
        raise RuntimeError("No U-Net training chips passed the frozen threshold")
    validation_selected = None
    validation_chips: list[tuple[int, int]] = []
    if validation_indices is not None:
        validation_selected = membership(count, validation_indices)
        validation_chips = select(arrays, validation_selected, config, 1)
        if not validation_chips:
            raise RuntimeError("No U-Net validation chips cover held-out rows")

    model = dense.CompactUNet(len(config["targets"]["names"]), int(settings["base_channels"]))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    rng = np.random.default_rng(seed)
    best_loss = math.inf
    best_epoch = epochs
    stale = 0
    history: list[dict[str, float]] = []
    batch_size = int(settings["batch_size"])
    size = int(settings["chip_cells"])
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(train_chips))
        model.train()
        training_total = 0.0
        training_weight = 0.0
        for start in range(0, len(order), batch_size):
            local = [train_chips[index] for index in order[start : start + batch_size]]
            x, y, weights = make_batch(
                local, selected, row_weights, arrays, targets, stats, size
            )
            optimizer.zero_grad(set_to_none=True)
            loss = dense.weighted_map_loss(model(x), y, weights)
            loss.backward()
            optimizer.step()
            weight_sum = float(weights.sum().item())
            training_total += float(loss.detach().item()) * weight_sum
            training_weight += weight_sum
        training_loss = training_total / training_weight
        current = training_loss
        if validation_indices is not None:
            assert validation_selected is not None
            current = validation_loss(
                model,
                validation_chips,
                validation_selected,
                row_weights,
                arrays,
                targets,
                stats,
                config,
            )
        history.append(
            {"epoch": epoch, "training_loss": training_loss, "validation_loss": current}
        )
        print(
            f"epoch {epoch:03d}: train={training_loss:.6f} validation={current:.6f}",
            flush=True,
        )
        if current < best_loss - 1e-6:
            best_loss = current
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if (
            validation_indices is not None
            and epoch >= int(settings["minimum_epochs"])
            and stale >= int(settings["early_stopping_patience"])
        ):
            break
    return model, stats, best_epoch, history, len(train_chips)


def predict(
    model: torch.nn.Module,
    arrays: dict[str, np.ndarray],
    targets: np.ndarray,
    test_indices: np.ndarray,
    stats: dict[str, np.ndarray],
    config: dict[str, Any],
) -> tuple[np.ndarray, int]:
    settings = config["model"]
    count = len(arrays["row"])
    selected = membership(count, test_indices)
    chips = select(arrays, selected, config, 1)
    lookup = np.full(count, -1, dtype=np.int32)
    lookup[test_indices] = np.arange(len(test_indices), dtype=np.int32)
    prediction_sum = np.zeros((len(test_indices), targets.shape[1]), dtype=np.float64)
    prediction_count = np.zeros(len(test_indices), dtype=np.int16)
    neutral_weights = np.ones(count, dtype=np.float32)
    size = int(settings["chip_cells"])
    batch_size = int(settings["batch_size"])
    model.eval()
    with torch.no_grad():
        for start in range(0, len(chips), batch_size):
            local_chips = chips[start : start + batch_size]
            x, _, _ = make_batch(
                local_chips,
                selected,
                neutral_weights,
                arrays,
                targets,
                stats,
                size,
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
                usable &= np.where(usable, lookup[np.maximum(ids, 0)] >= 0, False)
                local = lookup[ids[usable]]
                np.add.at(prediction_sum, local, physical[batch_index][:, usable].T)
                np.add.at(prediction_count, local, 1)
    if (prediction_count == 0).any():
        raise RuntimeError("U-Net inference did not cover every test location")
    return (prediction_sum / prediction_count[:, None]).astype(np.float32), len(chips)


def run(fold_number: int, seed: int) -> None:
    config = load_config()
    outputs = config["outputs"]
    result_dir = ROOT / str(outputs["result_directory"])
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"tessera_v2_unet_fold_{fold_number}_seed_{seed}.npz"
    metadata_path = result_path.with_suffix(".json")
    if result_path.exists() and metadata_path.exists():
        print(f"resumed {result_path.name}", flush=True)
        return
    arrays = load_arrays(config)
    fold_path = ROOT / str(outputs["array_directory"]) / "folds" / f"fold_{fold_number}.npz"
    with np.load(fold_path) as source:
        row_weights = source["weights"].astype(np.float32)
        train = source["train_indices"].astype(np.int64)
        subtrain = source["subtrain_indices"].astype(np.int64)
        validation = source["validation_indices"].astype(np.int64)
        test = source["test_indices"].astype(np.int64)
        tuning_targets = source["tuning_targets"].astype(np.float32)
        final_targets = source["final_targets"].astype(np.float32)
    print(
        f"v2 U-Net fold={fold_number} seed={seed}; train={len(train):,}; "
        f"subtrain={len(subtrain):,}; validation={len(validation):,}; test={len(test):,}",
        flush=True,
    )
    set_seed(seed)
    _, _, best_epoch, history, tuning_chips = fit(
        arrays,
        tuning_targets,
        subtrain,
        validation,
        row_weights,
        int(config["model"]["maximum_epochs"]),
        config,
        seed,
    )
    set_seed(seed)
    final_model, final_stats, _, _, final_chips = fit(
        arrays,
        final_targets,
        train,
        None,
        row_weights,
        best_epoch,
        config,
        seed,
    )
    predictions, test_chips = predict(
        final_model, arrays, final_targets, test, final_stats, config
    )
    temporary = result_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        test_indices=test,
        observed=final_targets[test],
        predictions=predictions,
        best_epoch=np.asarray([best_epoch], dtype=np.int16),
    )
    temporary.replace(result_path)
    metadata_path.write_text(
        json.dumps(
            {
                "fold": fold_number,
                "seed": seed,
                "best_epoch": best_epoch,
                "parameters": sum(value.numel() for value in final_model.parameters()),
                "tuning_train_chips": tuning_chips,
                "final_train_chips": final_chips,
                "test_chips": test_chips,
                "tuning_history": history,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"completed v2 U-Net fold={fold_number} seed={seed}; best_epoch={best_epoch}",
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
