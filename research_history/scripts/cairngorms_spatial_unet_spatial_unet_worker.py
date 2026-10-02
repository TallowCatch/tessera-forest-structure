#!/usr/bin/env python3
"""Train one frozen Phase 22 model, spatial scheme, fold, and seed."""

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

import cairngorms_dense_10m_worker as dense


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/cairngorms_spatial_unet.yaml"


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "phase22_cairngorms_spatial_unet"
    ]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def configure_threads() -> None:
    threads = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)


class MLP(nn.Module):
    def __init__(self, inputs: int, outputs: int, hidden: list[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = inputs
        for width in hidden:
            layers.extend([nn.Linear(current, width), nn.GELU(), nn.Dropout(dropout)])
            current = width
        layers.append(nn.Linear(current, outputs))
        self.network = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


def load_arrays(config: dict[str, Any]) -> dict[str, np.ndarray]:
    directory = ROOT / str(config["outputs"]["array_directory"])
    return {
        "patch": np.load(directory / "patch_quantized.npy", mmap_mode="r"),
        "patch_scale": np.load(directory / "patch_scales.npy", mmap_mode="r"),
        "patch_valid": np.load(directory / "patch_valid.npy", mmap_mode="r"),
        "conventional": np.load(directory / "conventional.npy", mmap_mode="r"),
        "embedding": np.load(directory / "embedding_int8.npy", mmap_mode="r"),
        "scale": np.load(directory / "embedding_scale.npy", mmap_mode="r"),
        "valid": np.load(directory / "tessera_valid.npy", mmap_mode="r"),
        "row": np.load(directory / "row.npy", mmap_mode="r"),
        "column": np.load(directory / "column.npy", mmap_mode="r"),
        "block": np.load(directory / "spatial_block.npy", mmap_mode="r"),
        "targets": np.load(directory / "target_rows.npy", mmap_mode="r"),
        "index_grid": np.load(directory / "index_grid.npy", mmap_mode="r"),
    }


def ordered_patch(
    arrays: dict[str, np.ndarray], indices: np.ndarray, channel_mean: np.ndarray, channel_sd: np.ndarray
) -> np.ndarray:
    mask = np.asarray(arrays["patch_valid"][indices], dtype=np.float32)
    values = np.asarray(arrays["patch"][indices], dtype=np.float32)
    values *= np.asarray(arrays["patch_scale"][indices], dtype=np.float32)[:, None]
    values = (values - channel_mean[None, :, None, None]) / channel_sd[None, :, None, None]
    values *= mask[:, None]
    return np.concatenate([values, mask[:, None]], axis=1).astype(np.float32)


def summarize(values: np.ndarray, cells: int) -> np.ndarray:
    if cells not in {3, 5}:
        raise ValueError("Summary context must be 3 or 5 cells")
    margin = (5 - cells) // 2
    scoped = values[:, :, margin : 5 - margin, margin : 5 - margin]
    embeddings = scoped[:, :128]
    mask = scoped[:, 128] > 0
    count = np.maximum(mask.sum(axis=(1, 2)), 1).astype(np.float32)
    total = np.where(mask[:, None], embeddings, 0).sum(axis=(2, 3))
    square = np.where(mask[:, None], np.square(embeddings), 0).sum(axis=(2, 3))
    mean = total / count[:, None]
    sd = np.sqrt(np.maximum(square / count[:, None] - np.square(mean), 0))
    offsets = np.arange(cells, dtype=np.float32) - (cells - 1) / 2
    xx = np.broadcast_to(offsets[None], (cells, cells))
    yy = np.broadcast_to(-offsets[:, None], (cells, cells))
    sx = np.where(mask, xx[None], 0).sum(axis=(1, 2))
    sy = np.where(mask, yy[None], 0).sum(axis=(1, 2))
    sx2 = np.where(mask, np.square(xx)[None], 0).sum(axis=(1, 2))
    sy2 = np.where(mask, np.square(yy)[None], 0).sum(axis=(1, 2))
    sxv = np.where(mask[:, None], embeddings * xx[None, None], 0).sum(axis=(2, 3))
    syv = np.where(mask[:, None], embeddings * yy[None, None], 0).sum(axis=(2, 3))
    dx_den = sx2 - np.square(sx) / count
    dy_den = sy2 - np.square(sy) / count
    dx = np.zeros_like(mean)
    dy = np.zeros_like(mean)
    good_x = dx_den > 0
    good_y = dy_den > 0
    dx[good_x] = (sxv[good_x] - sx[good_x, None] * total[good_x] / count[good_x, None]) / dx_den[good_x, None]
    dy[good_y] = (syv[good_y] - sy[good_y, None] * total[good_y] / count[good_y, None]) / dy_den[good_y, None]
    centre = embeddings[:, :, cells // 2, cells // 2]
    return np.concatenate([centre, mean, sd, dx, dy], axis=1).astype(np.float32)


def build_tabular_features(
    model_name: str,
    arrays: dict[str, np.ndarray],
    channel_mean: np.ndarray,
    channel_sd: np.ndarray,
) -> np.ndarray:
    if model_name == "conventional_mlp":
        return np.asarray(arrays["conventional"], dtype=np.float32)
    patches = ordered_patch(arrays, np.arange(len(arrays["targets"])), channel_mean, channel_sd)
    cells = 3 if model_name == "tessera_mlp_3x3" else 5
    summary = summarize(patches, cells)
    if model_name == "fused_mlp_5x5":
        return np.concatenate([summary, np.asarray(arrays["conventional"], dtype=np.float32)], axis=1)
    return summary


def block_weights(block: np.ndarray, indices: np.ndarray) -> np.ndarray:
    selected = np.asarray(block[indices], dtype=np.int64)
    _, inverse, counts = np.unique(selected, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse].astype(np.float32)
    return weights * len(weights) / weights.sum()


def fit_scalar(
    model_name: str,
    arrays: dict[str, np.ndarray],
    fold: dict[str, np.ndarray],
    config: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    settings = config["models"]
    subtrain = fold["subtrain_indices"].astype(np.int64)
    validation = fold["validation_indices"].astype(np.int64)
    test = fold["test_indices"].astype(np.int64)
    y_mean = fold["target_mean"].astype(np.float32)
    y_sd = fold["target_sd"].astype(np.float32)
    if model_name == "tessera_cnn_5x5":
        features = None
        model: nn.Module = dense.ResidualPatchCNN(
            arrays["targets"].shape[1],
            int(settings["cnn_channels"]),
            int(settings["cnn_blocks"]),
            [int(value) for value in settings["cnn_head"]],
            float(settings["dropout"]),
        )
        x_mean = x_sd = None
    else:
        features = build_tabular_features(model_name, arrays, fold["input_mean"], fold["input_sd"])
        x_mean = features[subtrain].mean(axis=0, dtype=np.float64).astype(np.float32)
        x_sd = features[subtrain].std(axis=0, dtype=np.float64).astype(np.float32)
        x_sd[x_sd < 1e-6] = 1.0
        model = MLP(
            features.shape[1], arrays["targets"].shape[1],
            [int(value) for value in settings["hidden_dimensions"]], float(settings["dropout"]),
        )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(settings["learning_rate"]), weight_decay=float(settings["weight_decay"])
    )
    batch_size = int(settings["batch_size"][model_name])
    rng = np.random.default_rng(seed)
    weights = np.ones(len(arrays["targets"]), dtype=np.float32)
    weights[subtrain] = block_weights(arrays["block"], subtrain)
    weights[validation] = block_weights(arrays["block"], validation)

    def batch_input(indices: np.ndarray) -> torch.Tensor:
        if model_name == "tessera_cnn_5x5":
            values = ordered_patch(arrays, indices, fold["input_mean"], fold["input_sd"])
        else:
            assert features is not None and x_mean is not None and x_sd is not None
            values = (features[indices] - x_mean) / x_sd
        return torch.from_numpy(np.ascontiguousarray(values))

    def evaluate(indices: np.ndarray) -> float:
        model.eval()
        total = 0.0
        total_weight = 0.0
        with torch.no_grad():
            for start in range(0, len(indices), batch_size):
                selected = indices[start : start + batch_size]
                observed = (np.asarray(arrays["targets"][selected]) - y_mean) / y_sd
                row_loss = torch.square(model(batch_input(selected)) - torch.from_numpy(observed)).mean(dim=1)
                local = torch.from_numpy(weights[selected])
                total += float(torch.sum(row_loss * local))
                total_weight += float(local.sum())
        return total / total_weight

    best_state = None
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []
    maximum = int(settings["maximum_epochs"][model_name])
    for epoch in range(1, maximum + 1):
        model.train()
        order = rng.permutation(subtrain)
        for start in range(0, len(order), batch_size):
            selected = order[start : start + batch_size]
            observed = (np.asarray(arrays["targets"][selected]) - y_mean) / y_sd
            optimizer.zero_grad(set_to_none=True)
            row_loss = torch.square(model(batch_input(selected)) - torch.from_numpy(observed)).mean(dim=1)
            local = torch.from_numpy(weights[selected])
            loss = torch.sum(row_loss * local) / torch.sum(local)
            loss.backward()
            optimizer.step()
        validation_loss = evaluate(validation)
        history.append({"epoch": epoch, "validation_loss": validation_loss})
        print(f"epoch {epoch:03d}: validation={validation_loss:.6f}", flush=True)
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if epoch >= int(settings["minimum_epochs"]) and stale >= int(settings["early_stopping_patience"]):
            break
    if best_state is None:
        raise RuntimeError("No checkpoint retained")
    model.load_state_dict(best_state)
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(test), batch_size):
            selected = test[start : start + batch_size]
            standardized = model(batch_input(selected)).numpy()
            predictions.append(standardized * y_sd + y_mean)
    return np.concatenate(predictions).astype(np.float32), {
        "best_epoch": best_epoch, "best_validation_loss": best_loss, "history": history
    }


def run(model_name: str, scheme: str, fold_number: int, seed: int) -> None:
    config = load_config()
    set_seed(seed)
    configure_threads()
    array_dir = ROOT / str(config["outputs"]["array_directory"])
    result_dir = ROOT / str(config["outputs"]["result_directory"])
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"{scheme}_{model_name}_fold{fold_number}_seed{seed}.npz"
    metadata_path = result_path.with_suffix(".json")
    if result_path.exists() and metadata_path.exists():
        print(f"resumed {result_path.name}", flush=True)
        return
    arrays = load_arrays(config)
    with np.load(array_dir / "folds" / f"{scheme}_fold_{fold_number}.npz") as source:
        fold = {name: source[name] for name in source.files}
    print(
        f"model={model_name} scheme={scheme} fold={fold_number} seed={seed}; "
        f"subtrain={len(fold['subtrain_indices']):,} validation={len(fold['validation_indices']):,} test={len(fold['test_indices']):,}",
        flush=True,
    )
    if model_name == "tessera_unet":
        dense_config = {"models": dict(config["models"])}
        dense_config["models"]["batch_size"] = dict(config["models"]["batch_size"])
        dense_config["models"]["maximum_epochs"] = dict(config["models"]["maximum_epochs"])
        model = dense.CompactUNet(arrays["targets"].shape[1], int(config["models"]["unet_base_channels"]))
        predictions, training = dense.train_unet(model, fold, arrays, dense_config, seed)
    else:
        predictions, training = fit_scalar(model_name, arrays, fold, config, seed)
    test = fold["test_indices"].astype(np.int64)
    temporary = result_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        test_indices=test,
        observed=np.asarray(arrays["targets"][test], dtype=np.float32),
        predictions=predictions,
    )
    temporary.replace(result_path)
    metadata_path.write_text(json.dumps(training, indent=2) + "\n")
    print(f"completed {result_path.name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--scheme", choices=["balanced", "regional"], required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    run(args.model, args.scheme, args.fold, args.seed)


if __name__ == "__main__":
    main()
