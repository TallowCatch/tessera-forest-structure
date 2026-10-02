#!/usr/bin/env python3
"""Fit one resumable Phase 18 neural model, fold, and random seed."""

from __future__ import annotations

import argparse
import hashlib
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
CONFIG_PATH = ROOT / "configs/cairngorms_spatial_heads.yaml"
ARRAY_DIR = ROOT / "data/interim/phase18_cairngorms_neural"
RESULT_DIR = ROOT / "data/interim/phase18_cairngorms_neural_results"
ARRAY_MANIFEST_PATH = ARRAY_DIR / "manifest.json"


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "phase18_cairngorms_spatial_heads"
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
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


class SummaryMLP(nn.Module):
    def __init__(
        self,
        input_dimensions: int,
        output_dimensions: int,
        hidden: list[int],
        dropout: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = input_dimensions
        for width in hidden:
            layers.extend(
                [
                    nn.Linear(current, width),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            current = width
        layers.append(nn.Linear(current, output_dimensions))
        self.network = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class PatchCNN(nn.Module):
    def __init__(
        self,
        output_dimensions: int,
        channels: list[int],
        dropout: float,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        current = 129
        for index, width in enumerate(channels):
            kernel = 1 if index == 0 else 3
            blocks.extend(
                [
                    nn.Conv2d(
                        current,
                        width,
                        kernel_size=kernel,
                        padding=0 if kernel == 1 else 1,
                    ),
                    nn.GroupNorm(min(8, width), width),
                    nn.GELU(),
                ]
            )
            current = width
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Linear(current * 2, current),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(current, output_dimensions),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        features = self.features(values)
        average = features.mean(dim=(2, 3))
        maximum = features.amax(dim=(2, 3))
        return self.head(torch.cat([average, maximum], dim=1))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def summary_statistics(
    values: np.ndarray, indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    subset = np.asarray(values[indices], dtype=np.float32)
    mean = subset.mean(axis=0, dtype=np.float64).astype(np.float32)
    standard_deviation = subset.std(axis=0, dtype=np.float64).astype(np.float32)
    standard_deviation[standard_deviation < 1e-6] = 1.0
    return mean, standard_deviation


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
        scoped = indices[start : start + batch_size]
        mask = np.asarray(valid[scoped], dtype=bool)
        values = np.asarray(quantized[scoped], dtype=np.float32)
        values *= np.asarray(scales[scoped], dtype=np.float32)[:, None, :, :]
        total += np.where(mask[:, None, :, :], values, 0.0).sum(
            axis=(0, 2, 3), dtype=np.float64
        )
        total_square += np.where(
            mask[:, None, :, :], values * values, 0.0
        ).sum(axis=(0, 2, 3), dtype=np.float64)
        count += int(mask.sum())
    if count == 0:
        raise RuntimeError("No valid TESSERA patch pixels in neural training data")
    mean = (total / count).astype(np.float32)
    variance = np.maximum(total_square / count - np.square(mean), 1e-12)
    return mean, np.sqrt(variance).astype(np.float32)


def target_statistics(
    targets: np.ndarray, indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    subset = np.asarray(targets[indices], dtype=np.float32)
    mean = subset.mean(axis=0, dtype=np.float64).astype(np.float32)
    standard_deviation = subset.std(axis=0, dtype=np.float64).astype(np.float32)
    standard_deviation[standard_deviation < 1e-6] = 1.0
    return mean, standard_deviation


def build_model(
    model_name: str,
    config: dict[str, Any],
    input_dimensions: int,
    output_dimensions: int,
) -> nn.Module:
    settings = config["neural"]
    if model_name == "mlp":
        return SummaryMLP(
            input_dimensions,
            output_dimensions,
            [int(value) for value in settings["mlp_hidden_dimensions"]],
            float(settings["dropout"]),
        )
    if model_name == "cnn":
        return PatchCNN(
            output_dimensions,
            [int(value) for value in settings["cnn_channels"]],
            float(settings["dropout"]),
        )
    raise ValueError(f"Unknown model: {model_name}")


def input_batch(
    model_name: str,
    indices: np.ndarray,
    arrays: dict[str, np.ndarray],
    mean: np.ndarray,
    standard_deviation: np.ndarray,
    compute_device: torch.device,
) -> torch.Tensor:
    if model_name == "mlp":
        values = np.asarray(arrays["summary"][indices], dtype=np.float32)
        values = (values - mean) / standard_deviation
        return torch.from_numpy(values).to(compute_device)

    mask = np.asarray(arrays["patch_valid"][indices], dtype=np.float32)
    values = np.asarray(arrays["patch_quantized"][indices], dtype=np.float32)
    values *= np.asarray(
        arrays["patch_scales"][indices], dtype=np.float32
    )[:, None, :, :]
    values = (values - mean[None, :, None, None]) / standard_deviation[
        None, :, None, None
    ]
    values *= mask[:, None, :, :]
    values = np.concatenate([values, mask[:, None, :, :]], axis=1)
    return torch.from_numpy(values).to(compute_device)


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
    input_standard_deviation: np.ndarray,
    target_mean: np.ndarray,
    target_standard_deviation: np.ndarray,
    batch_size: int,
    compute_device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    total_weight = 0.0
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            scoped = indices[start : start + batch_size]
            x = input_batch(
                model_name,
                scoped,
                arrays,
                input_mean,
                input_standard_deviation,
                compute_device,
            )
            y = (
                np.asarray(arrays["targets"][scoped], dtype=np.float32)
                - target_mean
            ) / target_standard_deviation
            weights = np.asarray(arrays["weights"][scoped], dtype=np.float32)
            y_tensor = torch.from_numpy(y).to(compute_device)
            weight_tensor = torch.from_numpy(weights).to(compute_device)
            loss = weighted_loss(model(x), y_tensor, weight_tensor)
            batch_weight = float(weights.sum())
            total += float(loss.detach().cpu()) * batch_weight
            total_weight += batch_weight
    return total / total_weight


def train_epochs(
    model: nn.Module,
    model_name: str,
    train_indices: np.ndarray,
    arrays: dict[str, np.ndarray],
    input_mean: np.ndarray,
    input_standard_deviation: np.ndarray,
    target_mean: np.ndarray,
    target_standard_deviation: np.ndarray,
    epochs: int,
    config: dict[str, Any],
    compute_device: torch.device,
    validation_indices: np.ndarray | None = None,
) -> tuple[nn.Module, list[dict[str, float]], int]:
    settings = config["neural"]
    batch_size = int(settings["batch_size"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    patience = int(settings["early_stopping_patience"])
    minimum_epochs = int(settings["minimum_epochs"])
    history: list[dict[str, float]] = []
    best_epoch = epochs
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    rng = np.random.default_rng(torch.initial_seed() % (2**32))

    for epoch in range(1, epochs + 1):
        model.train()
        shuffled = rng.permutation(train_indices)
        total = 0.0
        total_weight = 0.0
        for start in range(0, len(shuffled), batch_size):
            scoped = shuffled[start : start + batch_size]
            x = input_batch(
                model_name,
                scoped,
                arrays,
                input_mean,
                input_standard_deviation,
                compute_device,
            )
            y = (
                np.asarray(arrays["targets"][scoped], dtype=np.float32)
                - target_mean
            ) / target_standard_deviation
            weights = np.asarray(arrays["weights"][scoped], dtype=np.float32)
            y_tensor = torch.from_numpy(y).to(compute_device)
            weight_tensor = torch.from_numpy(weights).to(compute_device)
            optimizer.zero_grad(set_to_none=True)
            loss = weighted_loss(model(x), y_tensor, weight_tensor)
            loss.backward()
            optimizer.step()
            batch_weight = float(weights.sum())
            total += float(loss.detach().cpu()) * batch_weight
            total_weight += batch_weight
        train_loss = total / total_weight
        validation_loss = train_loss
        if validation_indices is not None:
            validation_loss = evaluate_loss(
                model,
                model_name,
                validation_indices,
                arrays,
                input_mean,
                input_standard_deviation,
                target_mean,
                target_standard_deviation,
                batch_size,
                compute_device,
            )
        history.append(
            {
                "epoch": float(epoch),
                "training_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        print(
            f"epoch {epoch:03d}: train={train_loss:.6f} "
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
    input_standard_deviation: np.ndarray,
    target_mean: np.ndarray,
    target_standard_deviation: np.ndarray,
    batch_size: int,
    compute_device: torch.device,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            scoped = indices[start : start + batch_size]
            x = input_batch(
                model_name,
                scoped,
                arrays,
                input_mean,
                input_standard_deviation,
                compute_device,
            )
            values = model(x).detach().cpu().numpy()
            rows.append(values * target_standard_deviation + target_mean)
    return np.concatenate(rows).astype(np.float32)


def run(model_name: str, fold: int, seed: int) -> None:
    if not ARRAY_MANIFEST_PATH.exists():
        raise RuntimeError("Phase 18 neural array manifest is missing")
    config = load_config()
    manifest = json.loads(ARRAY_MANIFEST_PATH.read_text(encoding="utf-8"))
    fold_path = ARRAY_DIR / f"fold_{fold}.npz"
    output_path = RESULT_DIR / f"fold_{fold}_{model_name}_seed_{seed}.npz"
    output_manifest_path = output_path.with_suffix(".json")
    identity = {
        "array_manifest_sha256": sha256(ARRAY_MANIFEST_PATH),
        "fold_sha256": sha256(fold_path),
        "model": model_name,
        "fold": fold,
        "seed": seed,
    }
    if output_path.exists() and output_manifest_path.exists():
        existing = json.loads(output_manifest_path.read_text(encoding="utf-8"))
        if (
            existing["identity"] == identity
            and existing["output_sha256"] == sha256(output_path)
        ):
            print(f"resumed {model_name} fold {fold} seed {seed}", flush=True)
            return
        raise RuntimeError("A stale Phase 18 neural result already exists")

    arrays: dict[str, np.ndarray] = {
        "summary": np.load(ARRAY_DIR / "summary.npy", mmap_mode="r"),
        "patch_quantized": np.load(
            ARRAY_DIR / "patch_quantized.npy", mmap_mode="r"
        ),
        "patch_scales": np.load(
            ARRAY_DIR / "patch_scales.npy", mmap_mode="r"
        ),
        "patch_valid": np.load(ARRAY_DIR / "patch_valid.npy", mmap_mode="r"),
    }
    with np.load(fold_path) as fold_values:
        arrays["targets"] = fold_values["targets"].astype(np.float32)
        arrays["weights"] = fold_values["weights"].astype(np.float32)
        train_indices = fold_values["train_indices"].astype(np.int64)
        subtrain_indices = fold_values["subtrain_indices"].astype(np.int64)
        validation_indices = fold_values["validation_indices"].astype(np.int64)
        test_indices = fold_values["test_indices"].astype(np.int64)

    set_seed(seed)
    compute_device = device()
    settings = config["neural"]
    batch_size = int(settings["batch_size"])
    print(
        f"{model_name} fold {fold} seed {seed}: device={compute_device}; "
        f"subtrain={len(subtrain_indices):,}; validation="
        f"{len(validation_indices):,}; test={len(test_indices):,}",
        flush=True,
    )
    if model_name == "mlp":
        input_mean, input_sd = summary_statistics(
            arrays["summary"], subtrain_indices
        )
    else:
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
    model = build_model(
        model_name,
        config,
        arrays["summary"].shape[1],
        arrays["targets"].shape[1],
    ).to(compute_device)
    _, history, best_epoch = train_epochs(
        model,
        model_name,
        subtrain_indices,
        arrays,
        input_mean,
        input_sd,
        target_mean,
        target_sd,
        int(settings["maximum_epochs"]),
        config,
        compute_device,
        validation_indices,
    )

    set_seed(seed)
    if model_name == "mlp":
        full_input_mean, full_input_sd = summary_statistics(
            arrays["summary"], train_indices
        )
    else:
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
        model_name,
        config,
        arrays["summary"].shape[1],
        arrays["targets"].shape[1],
    ).to(compute_device)
    final_model, _, _ = train_epochs(
        final_model,
        model_name,
        train_indices,
        arrays,
        full_input_mean,
        full_input_sd,
        full_target_mean,
        full_target_sd,
        best_epoch,
        config,
        compute_device,
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
        compute_device,
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
        output_manifest_path,
        {
            "identity": identity,
            "device": str(compute_device),
            "best_epoch": best_epoch,
            "history": history,
            "target_names": manifest["target_names"],
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
    parser.add_argument("--model", choices=["mlp", "cnn"], required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.model, arguments.fold, arguments.seed)
