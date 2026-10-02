from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_dutch_multisite_dutch_transfer import source_weights  # noqa: E402
from prepare_dutch_multisite import (  # noqa: E402
    mode_year,
    spatially_balanced_cap,
)


def test_design_has_three_sites_per_forest_group() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/dutch_multisite_transfer.yaml").read_text()
    )["phase36_dutch_multisite_transfer"]
    groups = pd.Series([site["group"] for site in config["study"]["sites"]])
    assert groups.value_counts().to_dict() == {"conifer": 3, "broadleaf": 3}
    assert len({site["code"] for site in config["study"]["sites"]}) == 6


def test_mode_year_is_calculated_within_each_50m_unit() -> None:
    dates = np.full((10, 10), 20210101, dtype=np.int32)
    dates[:5, :5] = 20220303
    usable = np.ones_like(dates, dtype=bool)
    years, purity = mode_year(dates, usable, cells=5)
    assert years.tolist() == [[2022, 2021], [2021, 2021]]
    assert np.allclose(purity, 1.0)


def test_spatial_cap_does_not_only_sample_the_largest_block() -> None:
    frame = pd.DataFrame(
        {"sampling_block": ["large"] * 20 + ["small_a"] * 3 + ["small_b"] * 2}
    )
    retained = spatially_balanced_cap(frame, maximum=9, seed=3)
    assert len(retained) == 9
    assert set(retained["sampling_block"]) == {"large", "small_a", "small_b"}


def test_source_weights_give_sites_equal_total_influence() -> None:
    frame = pd.DataFrame(
        {
            "site_code": ["large"] * 8 + ["small"] * 2,
            "fold_block": ["a"] * 4 + ["b"] * 4 + ["c"] * 2,
        }
    )
    weights = source_weights(frame)
    assert np.isclose(weights.sum(), len(frame))
    assert np.isclose(weights[:8].sum(), weights[8:].sum())

