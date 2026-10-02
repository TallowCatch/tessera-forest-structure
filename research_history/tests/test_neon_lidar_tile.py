import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts/process_neon_lidar_tile.py"


def load_tile_module():
    spec = importlib.util.spec_from_file_location("process_neon_lidar_tile", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_metric_column_uses_numbered_header(tmp_path: Path) -> None:
    module = load_tile_module()
    metric = tmp_path / "metric.txt"
    metric.write_text(
        "# 1 wave ID, 2 true ground, 3 FHD, 4 other,\n"
        "wave 99.0 2.75 5.0\n",
        encoding="utf-8",
    )

    assert module.metric_column(metric, "FHD") == 2.75


def test_selected_gedi_rows_respects_edge_buffer() -> None:
    module = load_tile_module()
    frame = pd.DataFrame({
        "site_id": ["SOAP", "SOAP", "SOAP"],
        "longitude": [-117.0, -117.0, -117.0],
        "latitude": [37.0, 37.0, 37.0],
        "shot_number": [1, 2, 3],
    })
    transformer = module.Transformer.from_crs(4326, module.SITE_UTM["SOAP"], always_xy=True)
    x, y = transformer.transform(frame["longitude"].iloc[0], frame["latitude"].iloc[0])
    easting = int(x // 1000 * 1000)
    northing = int(y // 1000 * 1000)

    selected = module.selected_gedi_rows(frame, "SOAP", easting, northing)

    expected = module.EDGE_BUFFER_M <= x - easting <= 1000 - module.EDGE_BUFFER_M
    expected &= module.EDGE_BUFFER_M <= y - northing <= 1000 - module.EDGE_BUFFER_M
    assert len(selected) == (3 if expected else 0)
