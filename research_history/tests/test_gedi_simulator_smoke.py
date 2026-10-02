import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts/smoke_test_gedi_simulator.py"


def load_smoke_module():
    spec = importlib.util.spec_from_file_location("smoke_test_gedi_simulator", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_fhd_uses_numbered_metric_header(tmp_path: Path) -> None:
    module = load_smoke_module()
    metric = tmp_path / "test.metric.txt"
    metric.write_text(
        "# 1 wave ID, 2 true ground, 3 FHD, 4 other,\n"
        "wave 100.0 3.25 9.0\n",
        encoding="utf-8",
    )

    assert module.parse_fhd(metric) == 3.25
