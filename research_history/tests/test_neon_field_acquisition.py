import importlib.util
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts/acquire_neon_field_structure.py"


def load_acquisition_module():
    spec = importlib.util.spec_from_file_location("acquire_neon_field_structure", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_destination_is_scoped_by_site_and_month() -> None:
    module = load_acquisition_module()
    row = SimpleNamespace(site_id="SOAP", month="2023-09", file_name="table.csv")

    assert module.destination_for(row) == (
        module.OUTPUT_ROOT / "SOAP" / "2023-09" / "table.csv"
    )


def test_destination_rejects_path_traversal() -> None:
    module = load_acquisition_module()
    row = SimpleNamespace(site_id="SOAP", month="2023-09", file_name="../token.csv")

    try:
        module.destination_for(row)
    except RuntimeError as error:
        assert "Unsafe" in str(error)
    else:
        raise AssertionError("Path traversal was accepted")
