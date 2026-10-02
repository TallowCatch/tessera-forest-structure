import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts/prepare_neon_local_adaptation_manifest.py"


def load_manifest_module():
    spec = importlib.util.spec_from_file_location("prepare_neon_local_adaptation_manifest", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_lidar_pattern_accepts_official_colorized_tile_name() -> None:
    module = load_manifest_module()
    match = module.LIDAR_PATTERN.search(
        "NEON_D17_SOAP_DP1_300000_4103000_classified_point_cloud_colorized.laz"
    )

    assert match is not None
    assert match.groups() == ("300000", "4103000")


def test_lidar_pattern_rejects_non_tile_point_clouds() -> None:
    module = load_manifest_module()

    assert module.LIDAR_PATTERN.search(
        "NEON_D17_SOAP_DP1_L016-1_2024061115_unclassified_point_cloud.laz"
    ) is None
    assert module.LIDAR_PATTERN.search(
        "NEON_D17_SOAP_DPQA_L014-1_2024061115_horizontal_uncertainty.laz"
    ) is None


def test_storage_gate_records_failure_without_lowering_reserve() -> None:
    module = load_manifest_module()
    maximum_tile_bytes = 800_000_000
    gate = module.build_storage_gate(
        free_bytes=5_000_000_000,
        maximum_tile_bytes=maximum_tile_bytes,
    )

    assert gate["passed"] is False
    assert gate["estimated_maximum_working_bytes"] == 4_800_000_000
    assert gate["disk_reserve_bytes"] == 2_000_000_000
    assert gate["required_free_bytes"] == 6_800_000_000
