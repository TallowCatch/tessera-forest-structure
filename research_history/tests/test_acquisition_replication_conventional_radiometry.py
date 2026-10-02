from types import SimpleNamespace

import numpy as np

from scripts import build_fair_baselines_conventional_predictors as phase8


def sentinel_item(item_id: str, offset_applied: bool) -> SimpleNamespace:
    assets = {
        key: SimpleNamespace(
            extra_fields={"raster:bands": [{"scale": 0.0001, "offset": -0.1}]}
        )
        for key in [*phase8.EARTH_S2_BANDS, "scl"]
    }
    return SimpleNamespace(
        id=item_id,
        properties={
            "datetime": "2025-02-10T18:00:00Z",
            "grid:code": "MGRS-12TUK",
            "s2:processing_baseline": "05.11",
            "earthsearch:boa_offset_applied": offset_applied,
        },
        assets=assets,
    )


def test_phase13_accepts_both_earth_search_boa_offset_conventions(monkeypatch):
    def sample_asset(item, asset_key, longitude, latitude):
        value = 4 if asset_key == "scl" else 2000
        return np.full(len(longitude), value, dtype=np.float64)

    monkeypatch.setattr(phase8.base, "sample_asset", sample_asset)
    config = {
        "valid_scl_classes": [4],
        "reflectance_scale": 0.0001,
    }
    longitude = np.array([-112.5])
    latitude = np.array([40.2])

    applied, applied_inventory = phase8.extract_sentinel2(
        [sentinel_item("applied", True)], longitude, latitude, config
    )
    retained, retained_inventory = phase8.extract_sentinel2(
        [sentinel_item("retained", False)],
        longitude,
        latitude,
        config,
        allow_unapplied_boa_offset=True,
    )

    assert np.isclose(applied["s2_blue_median"][0], 0.2)
    assert np.isclose(retained["s2_blue_median"][0], 0.1)
    assert applied_inventory[0]["sentinel_2_radiometry"].endswith("already_applied")
    assert retained_inventory[0]["sentinel_2_radiometry"].endswith("applied_locally")

