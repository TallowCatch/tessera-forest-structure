"""Metrics used by the lightweight reproducibility example."""

from __future__ import annotations

import numpy as np


def root_mean_squared_error(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Return unweighted root mean squared error."""
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if observed.shape != predicted.shape or observed.size == 0:
        raise ValueError("observed and predicted must be non-empty arrays of equal shape")
    return float(np.sqrt(np.mean(np.square(predicted - observed))))


def coefficient_of_determination(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Return R-squared using the observed mean as the null prediction."""
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if observed.shape != predicted.shape or observed.size == 0:
        raise ValueError("observed and predicted must be non-empty arrays of equal shape")
    denominator = float(np.sum(np.square(observed - np.mean(observed))))
    if denominator == 0:
        return float("nan")
    return float(1.0 - np.sum(np.square(predicted - observed)) / denominator)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def spearman_rank_correlation(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Return Spearman correlation, including average ranks for ties."""
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if observed.shape != predicted.shape or observed.size < 2:
        raise ValueError("observed and predicted must contain at least two values")
    observed_rank = _average_ranks(observed)
    predicted_rank = _average_ranks(predicted)
    if np.ptp(observed_rank) == 0 or np.ptp(predicted_rank) == 0:
        return float("nan")
    return float(np.corrcoef(observed_rank, predicted_rank)[0, 1])
