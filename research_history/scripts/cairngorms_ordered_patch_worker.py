#!/usr/bin/env python3
"""Train one ordered-patch model, spatial fold, and seed."""

from __future__ import annotations

import argparse
import hashlib
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


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/cairngorms_ordered_patch.yaml"
RESULT_DIR = ROOT / "data/interim/cairngorms_ordered_patch_results"


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "cairngorms_ordered_patch"
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class OrderedMLP(nn.Module):
    def __init__(
        self,
        input_dimensions: int,
        output_dimensions: int,
        hidden: list[int],
        dropout: float,
        linear_skip: bool,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = input_dimensions
        for width in hidden:
            layers.extend(
                [nn.Linear(current, width), nn.GELU(), nn.Dropout(dropout)]
            )
            current = width
        self.trunk = nn.Sequential(*layers)
        self.nonlinear_head = nn.Linear(current, output_dimensions)
        self.linear_head = (
            nn.Linear(input_dimensions, output_dimensions)
            if linear_skip
            else None
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        prediction = self.nonlinear_head(self.trunk(values))
        if self.linear_head is not None:
            prediction = prediction + self.linear_head(values)
        return prediction


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        groups = min(8, channels)
        self.network = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, channels),
        )
        self.activation = nn.GELU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.activation(values + self.network(values))


class ResidualPatchCNN(nn.Module):
    def __init__(
        self,
        output_dimensions: int,
        channels: int,
        block_count: int,
        head: list[int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(129, channels, kernel_size=1),
            nn.GroupNorm(min(8, channels), channels),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(channels, dropout) for _ in range(block_count)]
        )
        layers: list[nn.Module] = []
        current = channels * 5 * 5
        for width in head:
            layers.extend(
                [nn.Linear(current, width), nn.GELU(), nn.Dropout(dropout)]
            )
            current = width
        layers.append(nn.Linear(current, output_dimensions))
        self.head = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        features = self.blocks(self.stem(values))
        return self.head(features.flatten(start_dim=1))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def configure_threads() -> None:
    threads = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)


def patch_statistics(
    quantized: np.ndarray,
    scales: np.ndarray,
    valid: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    channels = quantized.shape[1]
    total = np.zeros(channels, dtype=np.float64)
    total_square = np.zeros(channels, dtype=np.float64)
    count = 0
    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        mask = np.asarray(valid[selected], dtype=bool)
        values = np.asarray(quantized[selected], dtype=np.float32)
        values *= np.asarray(scales[selected], dtype=np.float32)[:, None, :, :]
        masked = np.where(mask[:, None, :, :], values, 0.0)
        total += masked.sum(axis=(0, 2, 3), dtype=np.float64)
        total_square += np.square(masked).sum(
            axis=(0, 2, 3), dtype=np.float64
        )
        count += int(mask.sum())
    if count == 0:
        raise RuntimeError("No valid patch pixels were available")
    mean = total / count
    variance = np.maximum(total_square / count - np.square(mean), 1e-12)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def target_statistics(
    targets: np.ndarray, indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    selected = np.asarray(targets[indices], dtype=np.float32)
    mean = selected.mean(axis=0, dtype=np.float64).astype(np.float32)
    standard_deviation = selected.std(
        axis=0, dtype=np.float64
    ).astype(np.float32)
    standard_deviation[standard_deviation < 1e-6] = 1.0
    return mean, standard_deviation


def input_batch(
    model_name: str,
    indices: np.ndarray,
    arrays: dict[str, np.ndarray],
    mean: np.ndarray,
    standard_deviation: np.ndarray,
) -> torch.Tensor:
    mask = np.asarray(arrays["patch_valid"][indices], dtype=np.float32)
    values = np.asarray(arrays["patch_quantized"][indices], dtype=np.float32)
    values *= np.asarray(
        arrays["patch_scales"][indices], dtype=np.float32
    )[:, None, :, :]
    values = (values - mean[None, :, None, None]) / standard_deviation[
        None, :, None, None
    ]
    values *= mask[:, None, :, :]
    ordered = np.concatenate([values, mask[:, None, :, :]], axis=1)
    if model_name in {"ordered_mlp", "ordered_skip_mlp"}:
        ordered = ordered.reshape(len(indices), -1)
    return torch.from_numpy(np.ascontiguousarray(ordered))


def build_model(
    model_name: str,
    config: dict[str, Any],
    output_dimensions: int,
) -> nn.Module:
    settings = config["neural"]
    if model_name in {"ordered_mlp", "ordered_skip_mlp"}:
        return OrderedMLP(
            129 * 5 * 5,
            output_dimensions,
            [int(value) for value in settings["ordered_mlp_hidden"]],
            float(settings["dropout"]),
            linear_skip=model_name == "ordered_skip_mlp",
        )
    if model_name == "residual_patch_cnn":
        return ResidualPatchCNN(
            output_dimensions,
            int(settings["residual_cnn_channels"]),
            int(settings["residual_cnn_blocks"]),
            [int(value) for value in settings["residual_cnn_head"]],
            float(settings["dropout"]),
        )
    raise ValueError(f"Unknown model: {model_name}")


def weighted_loss(
    predicted: torch.Tensor,
    observed: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    per_row = torch.mean(torch.square(predicted - observed), dim=1)
    return torch.sum(per_row * weights) / torch.sum(weights)


def evaluate_loss(
    model: nn.Module,
    model_name: str,
    indices: np.ndarray,
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
            x = input_batch(
                model_name, selected, arrays, input_mean, input_sd
            )
            y = (
                np.asarray(arrays["targets"][selected], dtype=np.float32)
                - target_mean
            ) / target_sd
            weights = np.asarray(
                arrays["weights"][selected], dtype=np.float32
            )
            loss = weighted_loss(
                model(x), torch.from_numpy(y), torch.from_numpy(weights)
            )
            batch_weight = float(weights.sum())
            total += float(loss) * batch_weight
            total_weight += batch_weight
    return total / total_weight


def train(
    model: nn.Module,
    model_name: str,
    train_indices: np.ndarray,
    validation_indices: np.ndarray | None,
    arrays: dict[str, np.ndarray],
    input_mean: np.ndarray,
    input_sd: np.ndarray,
    target_mean: np.ndarray,
    target_sd: np.ndarray,
    maximum_epochs: int,
    config: dict[str, Any],
) -> tuple[nn.Module, list[dict[str, float]], int]:
    settings = config["neural"]
    batch_size = int(settings["batch_size"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    minimum_epochs = int(settings["minimum_epochs"])
    patience = int(settings["early_stopping_patience"])
    rng = np.random.default_rng(torch.initial_seed() % (2**32))
    history: list[dict[str, float]] = []
    best_loss = math.inf
    best_epoch = 1
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0

    for epoch in range(1, maximum_epochs + 1):
        model.train()
        shuffled = rng.permutation(train_indices)
        total = 0.0
        total_weight = 0.0
        for start in range(0, len(shuffled), batch_size):
            selected = shuffled[start : start + batch_size]
            x = input_batch(
                model_name, selected, arrays, input_mean, input_sd
            )
            y = (
                np.asarray(arrays["targets"][selected], dtype=np.float32)
                - target_mean
            ) / target_sd
            weights = np.asarray(
                arrays["weights"][selected], dtype=np.float32
            )
            optimizer.zero_grad(set_to_none=True)
            loss = weighted_loss(
                model(x), torch.from_numpy(y), torch.from_numpy(weights)
            )
            loss.backward()
            optimizer.step()
            batch_weight = float(weights.sum())
            total += float(loss.detach()) * batch_weight
            total_weight += batch_weight
        training_loss = total / total_weight
        validation_loss = training_loss
        if validation_indices is not None:
            validation_loss = evaluate_loss(
                model,
                model_name,
                validation_indices,
                arrays,
                input_mean,
                input_sd,
                target_mean,
                target_sd,
                batch_size,
            )
        history.append(
            {
                "epoch": float(epoch),
                "training_loss": training_loss,
                "validation_loss": validation_loss,
            }
        )
        print(
            f"epoch {epoch:03d}: train={training_loss:.6f} "
            f"validation={validation_loss:.6f}",
            flush=True,
        )
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if (
            validation_indices is not None
            and epoch >= minimum_epochs
            and stale >= patience
        ):
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_epoch


def predict(
    model: nn.Module,
    model_name: str,
    indices: np.ndarray,
    arrays: dict[str, np.ndarray],
    input_mean: np.ndarray,
    input_sd: np.ndarray,
    target_mean: np.ndarray,
    target_sd: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            x = input_batch(
                model_name, selected, arrays, input_mean, input_sd
            )
            standardized = model(x).numpy()
            rows.append(standardized * target_sd + target_mean)
    return np.concatenate(rows).astype(np.float32)


def run(model_name: str, fold: int, seed: int) -> None:
    configure_threads()
    config = load_config()
    source = config["source"]
    array_directory = ROOT / source["array_directory"]
    array_manifest = array_directory / "manifest.json"
    fold_path = array_directory / f"fold_{fold}.npz"
    if not array_manifest.exists() or not fold_path.exists():
        raise RuntimeError("Ordered-patch inputs are incomplete")

    output_path = RESULT_DIR / f"fold_{fold}_{model_name}_seed_{seed}.npz"
    output_manifest = output_path.with_suffix(".json")
    identity = {
        "config_sha256": sha256(CONFIG_PATH),
        "array_manifest_sha256": sha256(array_manifest),
        "fold_sha256": sha256(fold_path),
        "model": model_name,
        "fold": fold,
        "seed": seed,
    }
    if output_path.exists() and output_manifest.exists():
        existing = json.loads(output_manifest.read_text(encoding="utf-8"))
        if (
            existing.get("identity") == identity
            and existing.get("output_sha256") == sha256(output_path)
        ):
            print(f"resumed {model_name} fold {fold} seed {seed}")
            return
        raise RuntimeError("A stale ordered-patch result already exists")

    arrays: dict[str, np.ndarray] = {
        "patch_quantized": np.load(
            array_directory / "patch_quantized.npy", mmap_mode="r"
        ),
        "patch_scales": np.load(
            array_directory / "patch_scales.npy", mmap_mode="r"
        ),
        "patch_valid": np.load(
            array_directory / "patch_valid.npy", mmap_mode="r"
        ),
    }
    expected_shape = tuple(int(value) for value in source["expected_patch_shape"])
    if arrays["patch_quantized"].shape != (
        int(source["expected_rows"]),
        *expected_shape,
    ):
        raise RuntimeError("The ordered patch array has an unexpected shape")

    with np.load(fold_path) as values:
        arrays["targets"] = values["targets"].astype(np.float32)
        arrays["weights"] = values["weights"].astype(np.float32)
        train_indices = values["train_indices"].astype(np.int64)
        subtrain_indices = values["subtrain_indices"].astype(np.int64)
        validation_indices = values["validation_indices"].astype(np.int64)
        test_indices = values["test_indices"].astype(np.int64)

    settings = config["neural"]
    batch_size = int(settings["batch_size"])
    set_seed(seed)
    input_mean, input_sd = patch_statistics(
        arrays["patch_quantized"],
        arrays["patch_scales"],
        arrays["patch_valid"],
        subtrain_indices,
        batch_size,
    )
    target_mean, target_sd = target_statistics(
        arrays["targets"], subtrain_indices
    )
    model = build_model(model_name, config, arrays["targets"].shape[1])
    parameter_count = sum(value.numel() for value in model.parameters())
    print(
        f"{model_name} fold {fold} seed {seed}: "
        f"parameters={parameter_count:,}; subtrain={len(subtrain_indices):,}; "
        f"validation={len(validation_indices):,}; test={len(test_indices):,}",
        flush=True,
    )
    _, history, best_epoch = train(
        model,
        model_name,
        subtrain_indices,
        validation_indices,
        arrays,
        input_mean,
        input_sd,
        target_mean,
        target_sd,
        int(settings["maximum_epochs"]),
        config,
    )

    set_seed(seed)
    full_input_mean, full_input_sd = patch_statistics(
        arrays["patch_quantized"],
        arrays["patch_scales"],
        arrays["patch_valid"],
        train_indices,
        batch_size,
    )
    full_target_mean, full_target_sd = target_statistics(
        arrays["targets"], train_indices
    )
    final_model = build_model(
        model_name, config, arrays["targets"].shape[1]
    )
    final_model, _, _ = train(
        final_model,
        model_name,
        train_indices,
        None,
        arrays,
        full_input_mean,
        full_input_sd,
        full_target_mean,
        full_target_sd,
        best_epoch,
        config,
    )
    predictions = predict(
        final_model,
        model_name,
        test_indices,
        arrays,
        full_input_mean,
        full_input_sd,
        full_target_mean,
        full_target_sd,
        batch_size,
    )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        test_indices=test_indices,
        predictions=predictions,
        best_epoch=np.asarray([best_epoch], dtype=np.int16),
    )
    temporary.replace(output_path)
    atomic_json(
        output_manifest,
        {
            "identity": identity,
            "best_epoch": best_epoch,
            "parameter_count": parameter_count,
            "history": history,
            "target_names": config["evaluation"]["target_names"],
            "output_sha256": sha256(output_path),
        },
    )
    print(
        f"completed {model_name} fold {fold} seed {seed}; "
        f"best_epoch={best_epoch}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=["ordered_mlp", "ordered_skip_mlp", "residual_patch_cnn"],
        required=True,
    )
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.model, arguments.fold, arguments.seed)
