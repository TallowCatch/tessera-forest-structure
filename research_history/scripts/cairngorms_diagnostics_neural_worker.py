#!/usr/bin/env python3
"""Fit one frozen Phase 19 tabular neural model, fold, and seed."""

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
import torch.nn.functional as F
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/cairngorms_diagnostics.yaml"
ARRAY_DIR = ROOT / "data/interim/phase19_cairngorms_arrays"
SUMMARY_DIR = ROOT / "data/interim/phase19_cairngorms_context_summaries"
RESULT_DIR = ROOT / "data/interim/phase19_cairngorms_neural_results"
ARRAY_MANIFEST_PATH = ARRAY_DIR / "manifest.json"


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))[
        "phase19_cairngorms_diagnostics"
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


class RegressionMLP(nn.Module):
    def __init__(
        self,
        input_dimensions: int,
        output_dimensions: int,
        hidden: list[int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.trunk, final_width = build_trunk(
            input_dimensions, hidden, dropout
        )
        self.regression = nn.Linear(final_width, output_dimensions)

    def forward(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"regression": self.regression(self.trunk(values))}


class ProfileAuxMLP(nn.Module):
    def __init__(
        self,
        input_dimensions: int,
        profile_dimensions: int,
        hidden: list[int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.trunk, final_width = build_trunk(
            input_dimensions, hidden, dropout
        )
        self.regression = nn.Linear(final_width, 1)
        self.profile_logits = nn.Linear(final_width, profile_dimensions)

    def forward(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        representation = self.trunk(values)
        return {
            "regression": self.regression(representation),
            "profile_logits": self.profile_logits(representation),
        }


def build_trunk(
    input_dimensions: int, hidden: list[int], dropout: float
) -> tuple[nn.Sequential, int]:
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
    return nn.Sequential(*layers), current


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def compute_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def statistics(
    values: np.ndarray, indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    subset = np.asarray(values[indices], dtype=np.float32)
    mean = subset.mean(axis=0, dtype=np.float64).astype(np.float32)
    standard_deviation = subset.std(
        axis=0, dtype=np.float64
    ).astype(np.float32)
    standard_deviation[standard_deviation < 1e-6] = 1.0
    return mean, standard_deviation


def model_spec(config: dict[str, Any], name: str) -> dict[str, str]:
    matches = [
        value for value in config["model_specs"] if value["name"] == name
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Unknown or duplicated model specification: {name}")
    return {str(key): str(value) for key, value in matches[0].items()}


def input_path(name: str) -> Path:
    paths = {
        "tessera_1": SUMMARY_DIR / "tessera_1.npy",
        "tessera_3": SUMMARY_DIR / "tessera_3.npy",
        "tessera_5": (
            ROOT / "data/interim/phase18_cairngorms_neural/summary.npy"
        ),
        "tessera_9": SUMMARY_DIR / "tessera_9.npy",
        "conventional_centre": (
            ROOT / "data/interim/phase18_cairngorms_neural/conventional.npy"
        ),
        "conventional_5": ARRAY_DIR / "conventional_5.npy",
    }
    if name not in paths:
        raise RuntimeError(f"Unknown Phase 19 neural input: {name}")
    return paths[name]


def regression_targets(
    objective: str, target_matrix: np.ndarray
) -> np.ndarray:
    if objective in {"single", "profile_aux"}:
        return target_matrix[:, :1]
    if objective == "multi":
        return target_matrix
    raise RuntimeError(f"Unknown objective: {objective}")


def build_model(
    objective: str,
    input_dimensions: int,
    output_dimensions: int,
    profile_dimensions: int,
    config: dict[str, Any],
) -> nn.Module:
    settings = config["neural"]
    hidden = [int(value) for value in settings["hidden_dimensions"]]
    dropout = float(settings["dropout"])
    if objective == "profile_aux":
        return ProfileAuxMLP(
            input_dimensions, profile_dimensions, hidden, dropout
        )
    return RegressionMLP(
        input_dimensions, output_dimensions, hidden, dropout
    )


def input_tensor(
    inputs: np.ndarray,
    indices: np.ndarray,
    mean: np.ndarray,
    standard_deviation: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    values = np.asarray(inputs[indices], dtype=np.float32)
    values = (values - mean) / standard_deviation
    return torch.from_numpy(values).to(device)


def weighted_objective(
    outputs: dict[str, torch.Tensor],
    observed_regression: torch.Tensor,
    observed_profile: torch.Tensor,
    weights: torch.Tensor,
    objective: str,
    profile_loss_weight: float,
) -> torch.Tensor:
    regression_loss = torch.mean(
        torch.square(outputs["regression"] - observed_regression), dim=1
    )
    per_row = regression_loss
    if objective == "profile_aux":
        log_probabilities = F.log_softmax(outputs["profile_logits"], dim=1)
        profile_loss = -torch.sum(
            observed_profile * log_probabilities, dim=1
        )
        per_row = per_row + profile_loss_weight * profile_loss
    return torch.sum(per_row * weights) / torch.sum(weights)


def run_epoch(
    model: nn.Module,
    objective: str,
    indices: np.ndarray,
    inputs: np.ndarray,
    targets: np.ndarray,
    profiles: np.ndarray,
    weights: np.ndarray,
    input_mean: np.ndarray,
    input_sd: np.ndarray,
    target_mean: np.ndarray,
    target_sd: np.ndarray,
    batch_size: int,
    device: torch.device,
    profile_loss_weight: float,
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
            x = input_tensor(
                inputs, selected, input_mean, input_sd, device
            )
            observed = (
                np.asarray(targets[selected], dtype=np.float32) - target_mean
            ) / target_sd
            observed_tensor = torch.from_numpy(observed).to(device)
            profile_tensor = torch.from_numpy(
                np.asarray(profiles[selected], dtype=np.float32)
            ).to(device)
            batch_weights = np.asarray(
                weights[selected], dtype=np.float32
            )
            weight_tensor = torch.from_numpy(batch_weights).to(device)
            if training:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
            loss = weighted_objective(
                model(x),
                observed_tensor,
                profile_tensor,
                weight_tensor,
                objective,
                profile_loss_weight,
            )
            if training:
                loss.backward()
                optimizer.step()
            weight_sum = float(batch_weights.sum())
            total += float(loss.detach().cpu()) * weight_sum
            total_weight += weight_sum
    return total / total_weight


def train(
    model: nn.Module,
    objective: str,
    train_indices: np.ndarray,
    validation_indices: np.ndarray | None,
    arrays: dict[str, np.ndarray],
    input_mean: np.ndarray,
    input_sd: np.ndarray,
    target_mean: np.ndarray,
    target_sd: np.ndarray,
    epochs: int,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[nn.Module, list[dict[str, float]], int]:
    settings = config["neural"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    batch_size = int(settings["batch_size"])
    profile_weight = float(settings["profile_loss_weight"])
    patience = int(settings["early_stopping_patience"])
    minimum_epochs = int(settings["minimum_epochs"])
    rng = np.random.default_rng(torch.initial_seed() % (2**32))
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = math.inf
    best_epoch = epochs
    stale = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        training_loss = run_epoch(
            model,
            objective,
            train_indices,
            arrays["inputs"],
            arrays["targets"],
            arrays["profiles"],
            arrays["weights"],
            input_mean,
            input_sd,
            target_mean,
            target_sd,
            batch_size,
            device,
            profile_weight,
            optimizer,
            rng,
        )
        validation_loss = training_loss
        if validation_indices is not None:
            validation_loss = run_epoch(
                model,
                objective,
                validation_indices,
                arrays["inputs"],
                arrays["targets"],
                arrays["profiles"],
                arrays["weights"],
                input_mean,
                input_sd,
                target_mean,
                target_sd,
                batch_size,
                device,
                profile_weight,
                None,
                None,
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
    objective: str,
    indices: np.ndarray,
    arrays: dict[str, np.ndarray],
    input_mean: np.ndarray,
    input_sd: np.ndarray,
    target_mean: np.ndarray,
    target_sd: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray | None]:
    regression_rows: list[np.ndarray] = []
    profile_rows: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            outputs = model(
                input_tensor(
                    arrays["inputs"],
                    selected,
                    input_mean,
                    input_sd,
                    device,
                )
            )
            regression = outputs["regression"].detach().cpu().numpy()
            regression_rows.append(regression * target_sd + target_mean)
            if objective == "profile_aux":
                profile_rows.append(
                    F.softmax(outputs["profile_logits"], dim=1)
                    .detach()
                    .cpu()
                    .numpy()
                )
    profiles = (
        np.concatenate(profile_rows).astype(np.float32)
        if profile_rows
        else None
    )
    return np.concatenate(regression_rows).astype(np.float32), profiles


def run(model_name: str, fold: int, seed: int) -> None:
    if not ARRAY_MANIFEST_PATH.exists():
        raise RuntimeError("Phase 19 array manifest is missing")
    config = load_config()
    spec = model_spec(config, model_name)
    fold_path = ARRAY_DIR / f"fold_{fold}.npz"
    output_path = RESULT_DIR / f"fold_{fold}_{model_name}_seed_{seed}.npz"
    output_manifest_path = output_path.with_suffix(".json")
    identity = {
        "array_manifest_sha256": sha256(ARRAY_MANIFEST_PATH),
        "fold_sha256": sha256(fold_path),
        "model": model_name,
        "input": spec["input"],
        "objective": spec["objective"],
        "fold": fold,
        "seed": seed,
    }
    if output_path.exists() and output_manifest_path.exists():
        existing = json.loads(
            output_manifest_path.read_text(encoding="utf-8")
        )
        if (
            existing["identity"] == identity
            and existing["output_sha256"] == sha256(output_path)
        ):
            print(
                f"resumed {model_name} fold {fold} seed {seed}", flush=True
            )
            return
        raise RuntimeError("A stale Phase 19 neural result already exists")

    inputs = np.load(input_path(spec["input"]), mmap_mode="r")
    with np.load(fold_path) as values:
        target_matrix = values["targets"].astype(np.float32)
        profiles = values["profiles"].astype(np.float32)
        weights = values["weights"].astype(np.float32)
        train_indices = values["train_indices"].astype(np.int64)
        subtrain_indices = values["subtrain_indices"].astype(np.int64)
        validation_indices = values["validation_indices"].astype(np.int64)
        test_indices = values["test_indices"].astype(np.int64)
    targets = regression_targets(spec["objective"], target_matrix)
    arrays = {
        "inputs": inputs,
        "targets": targets,
        "profiles": profiles,
        "weights": weights,
    }

    set_seed(seed)
    device = compute_device()
    settings = config["neural"]
    input_mean, input_sd = statistics(inputs, subtrain_indices)
    target_mean, target_sd = statistics(targets, subtrain_indices)
    model = build_model(
        spec["objective"],
        inputs.shape[1],
        targets.shape[1],
        profiles.shape[1],
        config,
    ).to(device)
    print(
        f"{model_name} fold {fold} seed {seed}: device={device}; "
        f"subtrain={len(subtrain_indices):,}; "
        f"validation={len(validation_indices):,}; "
        f"test={len(test_indices):,}",
        flush=True,
    )
    _, history, best_epoch = train(
        model,
        spec["objective"],
        subtrain_indices,
        validation_indices,
        arrays,
        input_mean,
        input_sd,
        target_mean,
        target_sd,
        int(settings["maximum_epochs"]),
        config,
        device,
    )

    set_seed(seed)
    full_input_mean, full_input_sd = statistics(inputs, train_indices)
    full_target_mean, full_target_sd = statistics(targets, train_indices)
    final_model = build_model(
        spec["objective"],
        inputs.shape[1],
        targets.shape[1],
        profiles.shape[1],
        config,
    ).to(device)
    final_model, _, _ = train(
        final_model,
        spec["objective"],
        train_indices,
        None,
        arrays,
        full_input_mean,
        full_input_sd,
        full_target_mean,
        full_target_sd,
        best_epoch,
        config,
        device,
    )
    regression_predictions, profile_predictions = predict(
        final_model,
        spec["objective"],
        test_indices,
        arrays,
        full_input_mean,
        full_input_sd,
        full_target_mean,
        full_target_sd,
        int(settings["batch_size"]),
        device,
    )
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.npz")
    payload: dict[str, np.ndarray] = {
        "test_indices": test_indices,
        "regression_predictions": regression_predictions,
        "best_epoch": np.asarray([best_epoch], dtype=np.int16),
    }
    if profile_predictions is not None:
        payload["profile_predictions"] = profile_predictions
    np.savez_compressed(temporary, **payload)
    temporary.replace(output_path)
    atomic_json(
        output_manifest_path,
        {
            "identity": identity,
            "device": str(device),
            "best_epoch": best_epoch,
            "history": history,
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
    parser.add_argument("--model", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.model, arguments.fold, arguments.seed)
