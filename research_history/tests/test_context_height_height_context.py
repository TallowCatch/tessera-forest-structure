import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_context_height_footprint_height_transfer.py"
CONTEXT_SCRIPT = ROOT / "scripts/build_context_height_tessera_context.py"
CONTEXT_MODEL_SCRIPT = ROOT / "scripts/run_context_height_transfer.py"
CONTEXT_DIAGNOSTIC_SCRIPT = ROOT / "scripts/analyze_context_height_result.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_context_height_footprint_height_transfer", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_context_module():
    spec = importlib.util.spec_from_file_location("build_context_height_tessera_context", CONTEXT_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_context_model_module():
    spec = importlib.util.spec_from_file_location(
        "run_context_height_transfer", CONTEXT_MODEL_SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_context_diagnostic_module():
    spec = importlib.util.spec_from_file_location(
        "analyze_context_height_result", CONTEXT_DIAGNOSTIC_SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_height_target_uses_gedi_rh100_definition() -> None:
    module = load_module()
    frame = pd.DataFrame({"elev_highestreturn": [135.0, 204.5], "elev_lowestmode": [100.0, 180.0]})

    assert np.allclose(module.height_target(frame), [35.0, 24.5])


def test_equal_site_weights_give_each_site_equal_total_weight() -> None:
    module = load_module()
    site_ids = np.asarray(["A", "A", "A", "B"])

    weights = module.equal_site_weights(site_ids)

    assert np.isclose(weights[site_ids == "A"].sum(), weights[site_ids == "B"].sum())
    assert np.isclose(weights.mean(), 1.0)


def test_alpha_selection_prioritizes_worst_site_then_macro_rmse() -> None:
    module = load_module()
    tuning = pd.DataFrame(
        {
            "alpha": [0.1, 0.1, 1.0, 1.0, 10.0, 10.0],
            "rmse": [1.0, 4.0, 2.1, 2.2, 2.0, 2.2],
        }
    )

    selected, ranking = module.select_alpha(tuning)

    assert selected == 10.0
    assert ranking.iloc[0]["worst_inner_site_rmse"] == 2.2


def test_metric_values_exposes_prediction_compression() -> None:
    module = load_module()
    observed = np.asarray([0.0, 1.0, 2.0, 3.0])
    predicted = observed * 0.5

    metrics = module.metric_values(observed, predicted)

    assert np.isclose(metrics["prediction_slope"], 0.5)
    assert np.isclose(metrics["prediction_sd_ratio"], 0.5)


def test_context_feature_contract_has_three_scales_and_summaries() -> None:
    module = load_context_module()

    columns = module.context_feature_columns()

    assert len(columns) == 3 * 3 * 128
    assert columns[0] == "tessera_ctx030_mean_000"
    assert columns[-1] == "tessera_ctx150_delta_127"


def test_context_summaries_preserve_mean_spread_and_footprint_contrast() -> None:
    module = load_context_module()
    matrix = np.stack([np.zeros(128), np.full(128, 2.0)]).astype(np.float32)
    footprint = np.full(128, 3.0, dtype=np.float32)

    mean, std, delta = module.summarize_context(matrix, footprint)

    assert np.allclose(mean, 1.0)
    assert np.allclose(std, 1.0)
    assert np.allclose(delta, 2.0)


def test_context_acquisition_read_contract_excludes_gedi_targets() -> None:
    module = load_context_module()

    assert "fhd_normal" not in module.TARGET_READ_COLUMNS
    assert "elev_highestreturn" not in module.TARGET_READ_COLUMNS
    assert "elev_lowestmode" not in module.TARGET_READ_COLUMNS


def test_context_height_model_uses_nonredundant_384_predictors() -> None:
    module = load_context_model_module()

    features = module.feature_columns(70)

    assert len(features) == 384
    assert "tessera_area_000" in features
    assert "tessera_ctx070_std_000" in features
    assert "tessera_ctx070_delta_127" in features
    assert not any("_mean_" in feature for feature in features)


def test_context_height_model_read_contract_excludes_fhd() -> None:
    module = load_context_model_module()

    assert "fhd_normal" not in module.base.READ_COLUMNS
    assert "fhd_normal" not in module.CONTEXT_READ_COLUMNS


def test_context_selection_prioritizes_worst_site_then_macro_error() -> None:
    module = load_context_model_module()
    tuning = pd.DataFrame(
        {
            "window_width_m": [30, 30, 70, 70, 150, 150],
            "alpha": [1.0] * 6,
            "rmse": [1.0, 5.0, 3.0, 4.0, 3.8, 4.0],
        }
    )

    width, alpha, ranking = module.select_width_alpha(tuning)

    assert width == 70
    assert alpha == 1.0
    assert ranking.iloc[0]["worst_inner_site_rmse"] == 4.0
    assert ranking.iloc[0]["macro_inner_site_rmse"] == 3.5


def test_oracle_mean_correction_removes_only_the_group_mean_error() -> None:
    module = load_context_diagnostic_module()
    observed = np.asarray([1.0, 2.0, 4.0])
    predicted = np.asarray([6.0, 7.0, 8.0])

    corrected = module.offset_corrected_predictions(observed, predicted)

    assert np.isclose(np.mean(corrected - observed), 0.0)
    assert np.allclose(np.diff(corrected), np.diff(predicted))
