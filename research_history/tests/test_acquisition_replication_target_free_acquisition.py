from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/acquire_acquisition_replication_target_free_gedi.py"


def load_module():
    specification = importlib.util.spec_from_file_location("phase13_target_free", SCRIPT)
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    specification.loader.exec_module(module)
    return module


def test_broad_candidate_population_is_reproducible_and_outcome_blind() -> None:
    module = load_module()
    protocol = module.load_protocol()
    assert module.broad_candidates(protocol) == [
        "BLAN", "CLBJ", "DELA", "DSNY", "KONZ", "MOAB", "ONAQ", "RMNP", "SJER", "UKFS", "YELL"
    ]
    serialized = module.PROTOCOL_PATH.read_text(encoding="utf-8")
    assert "target_values_may_affect_site_or_fold_selection: false" in serialized
    assert (
        "target_predictors_source_only_predictions_folds_and_local_shot_ids_frozen_before_fhd_values_opened: true"
        in serialized
    )


def test_target_free_filter_uses_availability_not_fhd_value() -> None:
    module = load_module()
    protocol = module.load_protocol()
    frame = pd.DataFrame(
        {
            "target_available": [True, False],
            "l2b_quality_flag_rel3": [1, 1],
            "degrade_flag": [0, 0],
            "beam_power": ["full", "full"],
            "nlcd_valid_fraction": [1.0, 1.0],
            "nlcd_forest_fraction": [0.9, 0.9],
        }
    )
    filtered = module.target_free_filters(frame, protocol)
    assert filtered["passes_primary_filters"].tolist() == [True, False]
    assert "fhd_normal" not in filtered.columns


def test_complete_pass_gate_uses_coordinates_and_pass_metadata_only() -> None:
    module = load_module()
    protocol = module.load_protocol()
    records = []
    shot = 0
    for pass_index, (orbit, track, x_offset) in enumerate(
        [(1, 11, 0.0), (2, 22, 2000.0), (3, 33, 4000.0)]
    ):
        for row in range(60):
            shot += 1
            records.append(
                {
                    "site_id": "TEST",
                    "shot_number": shot,
                    "acquisition_datetime": pd.Timestamp(f"2024-0{pass_index + 1}-01", tz="UTC"),
                    "orbit": orbit,
                    "reference_ground_track": track,
                    "x_epsg5070": x_offset + row,
                    "y_epsg5070": float(row),
                }
            )
    sites, folds, viable = module.summarize_passes(pd.DataFrame(records), protocol)
    assert viable == ["TEST"]
    assert int(folds["eligible"].sum()) == 3
    assert bool(sites.iloc[0]["viable_replication_site"])
    assert np.isfinite(sites.iloc[0]["occupied_1km_cells"])


def test_2025_extension_preserves_the_failed_2024_gate() -> None:
    protocol = (ROOT / "metadata/project_config_phase13_temporal_extension_protocol_freeze.yaml").read_text(
        encoding="utf-8"
    )
    script = (ROOT / "scripts/acquire_acquisition_replication_2025_extension.py").read_text(encoding="utf-8")
    assert "prohibit_changes_to_2024_gate_or_site_status: true" in protocol
    assert "exclude_2024_viable_sites: [CLBJ, KONZ, RMNP, YELL]" in protocol
    assert "fhd_values_read_by_selection_or_summarization" in script
    assert "viable_2025_sites" in script
