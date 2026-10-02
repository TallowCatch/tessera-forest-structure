from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis

from scripts.analyze_reference_uncertainty_cairngorms_reference_uncertainty import (
    block_labels,
    jackknife_metrics,
    metric_values,
)


def test_metric_kurtosis_matches_scipy_definition() -> None:
    grid = np.arange(2500, dtype=np.float64).reshape(50, 50) / 100.0
    values = metric_values(grid, 2.0)
    expected = kurtosis(grid.ravel(), fisher=True, bias=False)
    assert np.isclose(values["canopy_height_kurtosis"], expected)


def test_block_labels_make_25_equal_ten_metre_blocks() -> None:
    labels = block_labels(50, 10)
    unique, counts = np.unique(labels, return_counts=True)
    assert len(unique) == 25
    assert np.all(counts == 100)


def test_jackknife_is_zero_for_spatially_repeated_blocks() -> None:
    block = np.arange(100, dtype=np.float64).reshape(10, 10) / 10.0
    grid = np.tile(block, (5, 5))
    _, standard_errors, blocks = jackknife_metrics(grid, 10, 2.0)
    assert blocks == 25
    for metric in [
        "canopy_mean_height_m",
        "canopy_surface_sd_m",
        "canopy_surface_cv",
        "canopy_open_fraction",
        "canopy_height_kurtosis",
    ]:
        assert standard_errors[metric] < 1e-10
