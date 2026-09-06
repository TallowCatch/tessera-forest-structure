"""Small, dependency-light model helpers for demonstration and tests."""

from __future__ import annotations

import numpy as np


def fit_standardized_ridge(
    features: np.ndarray,
    target: np.ndarray,
    alpha: float = 0.01,
) -> dict[str, np.ndarray | float]:
    """Fit a ridge model after standardising predictors on training data only."""
    features = np.asarray(features, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if features.ndim != 2 or target.ndim != 1 or len(features) != len(target):
        raise ValueError("features must be 2-D and aligned with a 1-D target")
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale == 0] = 1.0
    standardized = (features - mean) / scale
    design = np.column_stack([np.ones(len(standardized)), standardized])
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    return {
        "mean": mean,
        "scale": scale,
        "coefficients": coefficients,
        "alpha": float(alpha),
    }


def predict_standardized_ridge(
    model: dict[str, np.ndarray | float], features: np.ndarray
) -> np.ndarray:
    """Predict with a model returned by :func:`fit_standardized_ridge`."""
    features = np.asarray(features, dtype=np.float64)
    mean = np.asarray(model["mean"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    coefficients = np.asarray(model["coefficients"], dtype=np.float64)
    design = np.column_stack([np.ones(len(features)), (features - mean) / scale])
    return design @ coefficients


def offset_from_references(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Estimate a constant target-site correction from labelled reference units."""
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if observed.shape != predicted.shape or observed.size == 0:
        raise ValueError("reference arrays must be non-empty and aligned")
    return float(np.mean(observed - predicted))
