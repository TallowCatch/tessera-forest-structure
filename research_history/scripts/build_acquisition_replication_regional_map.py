#!/usr/bin/env python3
"""Build a conditional 1 km Phase 13 forest map from the replicated track model."""

from __future__ import annotations

import gc
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import yaml
from pyproj import Transformer
from rasterio.transform import from_origin
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_tessera_alignment as tessera  # noqa: E402
import evaluate_acquisition_replication_track_replication as evaluation  # noqa: E402
import freeze_acquisition_replication_track_predictions as prediction_tools  # noqa: E402
import run_context_height_footprint_height_transfer as common  # noqa: E402


PROTOCOL_PATH = ROOT / "metadata/project_config_phase13_track_replication_protocol_freeze.yaml"
RESULT_FREEZE_PATH = ROOT / "metadata/phase13_track_replication_result_freeze.json"
TARGET_FREEZE_PATH = ROOT / "metadata/phase13_combined_target_free_freeze.json"
PREDICTION_FREEZE_PATH = ROOT / "metadata/phase13_prediction_freeze.json"
PREDICTIONS_PATH = ROOT / "data/processed/phase13_frozen_source_predictions.parquet"
SAMPLES_PATH = ROOT / "data/processed/phase13_frozen_local_samples.parquet"
FOLD_PATH = ROOT / "metadata/phase13_combined_target_free_fold_summary.csv"
SELECTION_PATH = ROOT / "outputs/tables/phase13_source_model_selection.csv"
RASTER_PATH = ROOT / "outputs/reports/phase13_regional_track_assisted_map.tif"
POINTS_PATH = ROOT / "data/processed/phase13_regional_map_reference_points.parquet"
FIGURE_PATH = ROOT / "outputs/figures/phase13_regional_track_assisted_map.png"
FREEZE_PATH = ROOT / "metadata/phase13_regional_map_freeze.json"
STREAM_ROOT = ROOT / "data/interim/phase13_map_tessera_stream"


def select_region(predictions: pd.DataFrame, folds: pd.DataFrame) -> tuple[str, str, int, int]:
    extents = (
        predictions.groupby("site_id")
        .agg(
            minx=("x_epsg5070", "min"),
            maxx=("x_epsg5070", "max"),
            miny=("y_epsg5070", "min"),
            maxy=("y_epsg5070", "max"),
        )
    )
    extents["bbox_area"] = (extents["maxx"] - extents["minx"]) * (
        extents["maxy"] - extents["miny"]
    )
    site = str(extents.sort_values(["bbox_area"], ascending=False).index[0])
    eligible_mask = folds["eligible"].astype(str).str.lower().eq("true")
    eligible = folds[folds["site_id"].eq(site) & eligible_mask].copy()
    eligible = eligible.sort_values(["test_rows", "held_out_pass"], ascending=[False, True])
    if eligible.empty:
        raise RuntimeError("Selected Phase 13 map site has no eligible fold")
    held_out = str(eligible.iloc[0]["held_out_pass"])
    test = predictions[
        predictions["site_id"].eq(site) & predictions["pass_id"].eq(held_out)
    ].copy()
    test["cell_x"] = np.floor(test["x_epsg5070"] / 1000).astype(int)
    test["cell_y"] = np.floor(test["y_epsg5070"] / 1000).astype(int)
    cell = (
        test.groupby(["cell_x", "cell_y"], as_index=False)
        .size()
        .sort_values(["size", "cell_x", "cell_y"], ascending=[False, True, True])
        .iloc[0]
    )
    return site, held_out, int(cell["cell_x"]), int(cell["cell_y"])


def source_training() -> pd.DataFrame:
    tessera_frames = [
        pd.read_parquet(
            path,
            columns=prediction_tools.KEYS
            + ["fhd_normal", *prediction_tools.TESSERA_FEATURES],
        )
        for path in prediction_tools.SOURCE_TESSERA_PATHS
    ]
    return pd.concat(tessera_frames, ignore_index=True)


def main() -> int:
    protected = [RASTER_PATH, POINTS_PATH, FIGURE_PATH, FREEZE_PATH]
    existing = [str(path.relative_to(ROOT)) for path in protected if path.exists()]
    if existing:
        raise RuntimeError(f"Phase 13 regional map exists; refusing to overwrite: {existing}")
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))[
        "phase13_track_assisted_replication"
    ]
    result = json.loads(RESULT_FREEZE_PATH.read_text(encoding="utf-8"))
    if not result["freeze_basis"]["primary_gate"]["passed"]:
        raise RuntimeError("Phase 13 replication gate failed; regional map is prohibited")
    target_freeze = json.loads(TARGET_FREEZE_PATH.read_text(encoding="utf-8"))
    prediction_freeze = json.loads(PREDICTION_FREEZE_PATH.read_text(encoding="utf-8"))
    predictions = pd.read_parquet(PREDICTIONS_PATH)
    folds = pd.read_csv(FOLD_PATH)
    site, held_out, cell_x, cell_y = select_region(predictions, folds)
    target_years = predictions.loc[predictions["site_id"].eq(site), "target_year"].unique()
    if len(target_years) != 1:
        raise RuntimeError("Selected Phase 13 map site has an ambiguous target year")
    target_year = int(target_years[0])
    x_min, y_min = cell_x * 1000.0, cell_y * 1000.0
    x = x_min + np.arange(100, dtype=np.float64) * 10.0 + 5.0
    y = y_min + np.arange(100, dtype=np.float64) * 10.0 + 5.0
    grid_x, grid_y = np.meshgrid(x, y[::-1])

    nlcd_path = ROOT / f"data/external/nlcd/Annual_NLCD_Land_Cover_2024_{site}.tif"
    with rasterio.open(nlcd_path) as nlcd:
        classes = np.asarray(
            [value[0] for value in nlcd.sample(zip(grid_x.ravel(), grid_y.ravel(), strict=True))],
            dtype=np.int16,
        )
    forest = np.isin(classes, [41, 42, 43])
    if forest.sum() < 100:
        raise RuntimeError("Selected Phase 13 map cell has fewer than 100 forest pixels")
    to_wgs84 = Transformer.from_crs(5070, 4326, always_xy=True)
    longitude, latitude = to_wgs84.transform(grid_x.ravel()[forest], grid_y.ravel()[forest])
    map_targets = pd.DataFrame(
        {
            "site_id": "MAP",
            "shot_number": np.arange(int(forest.sum()), dtype=np.int64),
            "longitude": longitude,
            "latitude": latitude,
        }
    )

    config = yaml.safe_load((ROOT / "configs/project.yaml").read_text(encoding="utf-8"))
    manifest, landmasks = tessera.ensure_registry("1.0")
    tessera.EXPECTED_SITES = {"MAP"}
    tiles, _ = tessera.required_tiles(map_targets, 12.5)
    rows = tessera.lookup_embedding_rows(manifest, tiles, target_year, "1.0", "vultr")
    masks = tessera.lookup_landmask_rows(landmasks, tiles)
    tessera.assert_complete_inventory(rows, masks, tiles)
    if STREAM_ROOT.exists():
        shutil.rmtree(STREAM_ROOT)
    try:
        tessera.check_storage(STREAM_ROOT, rows, masks, target_year)
        records = tessera.download_tiles(STREAM_ROOT, rows, masks, target_year, config)
        loaded = tessera.load_tiles(records)
        aligned, _ = tessera.align_targets(map_targets, loaded, config)
        aligned = aligned[aligned["passes_tessera_alignment"]].copy()
        del loaded
        gc.collect()
    finally:
        if STREAM_ROOT.exists():
            shutil.rmtree(STREAM_ROOT)
    if len(aligned) < 100:
        raise RuntimeError("Too few valid TESSERA forest pixels in the selected map cell")

    source = source_training()
    selected = pd.read_csv(SELECTION_PATH)
    selected_mask = selected["selected"].astype(str).str.lower().eq("true")
    alpha_rows = selected[selected_mask & selected["model"].eq("tessera")]
    if len(alpha_rows) != 1:
        raise RuntimeError("Phase 13 selected TESSERA alpha is ambiguous")
    alpha = float(alpha_rows.iloc[0]["alpha"])
    scaler, model = common.fit_ridge(
        source, prediction_tools.TESSERA_FEATURES, "fhd_normal", alpha
    )
    map_x = aligned[prediction_tools.TESSERA_FEATURES].to_numpy(dtype=np.float64)
    standardized_map = scaler.transform(map_x)
    source_prediction = model.predict(standardized_map)
    support = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(
        scaler.transform(source[prediction_tools.TESSERA_FEATURES].to_numpy(dtype=np.float64))
    )
    support_distance = support.kneighbors(standardized_map, return_distance=True)[0][:, 0]

    outcomes = evaluation.load_sealed_outcomes(target_freeze)
    target = predictions.merge(outcomes, on=["site_id", "shot_number"], validate="one_to_one")
    primary_budget = int(protocol["local_reference"]["primary_budget"])
    samples = pd.read_parquet(SAMPLES_PATH)
    local_ids = samples[
        samples["site_id"].eq(site)
        & samples["held_out_pass"].eq(held_out)
        & samples["budget"].eq(primary_budget)
        & samples["replicate"].eq(0)
    ]["shot_number"]
    site_target = target[target["site_id"].eq(site)].set_index("shot_number", drop=False)
    local = site_target.loc[local_ids.to_numpy()]
    local_offset = float(
        np.mean(local["fhd_normal"] - local["source_tessera_prediction"])
    )
    adjusted_prediction = source_prediction + local_offset

    output = np.full((3, 100, 100), np.nan, dtype=np.float32)
    valid_flat_indices = np.flatnonzero(forest)[aligned["shot_number"].to_numpy(dtype=int)]
    for band, values in enumerate(
        [source_prediction, adjusted_prediction, support_distance], start=0
    ):
        output[band].ravel()[valid_flat_indices] = values.astype(np.float32)
    RASTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        RASTER_PATH,
        "w",
        driver="GTiff",
        height=100,
        width=100,
        count=3,
        dtype="float32",
        crs="EPSG:5070",
        transform=from_origin(x_min, y_min + 1000.0, 10.0, 10.0),
        nodata=np.nan,
        compress="deflate",
    ) as dataset:
        dataset.write(output)
        dataset.set_band_description(1, "source_only_tessera_fhd")
        dataset.set_band_description(2, "tessera_plus_local_offset_fhd")
        dataset.set_band_description(3, "standardized_source_support_distance")

    reference = target[
        target["site_id"].eq(site)
        & (
            target["pass_id"].eq(held_out)
            | target["shot_number"].isin(local_ids)
        )
    ][
        [
            "site_id",
            "shot_number",
            "pass_id",
            "x_epsg5070",
            "y_epsg5070",
            "fhd_normal",
        ]
    ].copy()
    reference["role"] = np.where(
        reference["pass_id"].eq(held_out), "held_out_reference", "local_reference"
    )
    common.write_parquet(reference, POINTS_PATH)

    figure, axes = plt.subplots(1, 3, figsize=(12.2, 4.2), constrained_layout=True)
    panels = [
        (output[0], "Source-only TESSERA prediction", "viridis", "Predicted FHD"),
        (output[1], f"Prediction after {primary_budget}-footprint local correction", "viridis", "Predicted FHD"),
        (output[2], "Distance from source predictor support", "magma", "Standardized distance"),
    ]
    for axis, (values, title, cmap, label) in zip(axes, panels, strict=True):
        image = axis.imshow(values, extent=[0, 1, 0, 1], origin="upper", cmap=cmap)
        scoped = reference[
            reference["x_epsg5070"].between(x_min, x_min + 1000)
            & reference["y_epsg5070"].between(y_min, y_min + 1000)
        ]
        for role, marker, colour in [
            ("local_reference", "o", "white"),
            ("held_out_reference", "x", "#E63946"),
        ]:
            points = scoped[scoped["role"].eq(role)]
            axis.scatter(
                (points["x_epsg5070"] - x_min) / 1000,
                (points["y_epsg5070"] - y_min) / 1000,
                marker=marker,
                c=colour,
                s=18,
                linewidths=0.8,
                label=role.replace("_", " "),
            )
        axis.set(title=title, xlabel="Easting within 1 km cell", ylabel="Northing within 1 km cell")
        figure.colorbar(image, ax=axis, shrink=0.78, label=label)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    figure.suptitle(f"Conditional Phase 13 mapping demonstration at {site}")
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=220)
    plt.close(figure)

    freeze_basis = {
        "protocol_sha256": common.sha256(PROTOCOL_PATH),
        "script_sha256": common.sha256(Path(__file__).resolve()),
        "result_freeze_id": result["freeze_id"],
        "prediction_freeze_id": prediction_freeze["freeze_id"],
        "selection_uses_target_outcome_performance": False,
        "site_selection": "largest_predictor_complete_target_bbox",
        "fold_selection": "largest_eligible_test_pass",
        "cell_selection": "one_km_cell_with_most_held_out_reference_footprints",
        "site_id": site,
        "target_year": target_year,
        "held_out_pass": held_out,
        "cell_epsg5070": [cell_x, cell_y],
        "local_budget": primary_budget,
        "local_replicate": 0,
        "local_offset": local_offset,
        "valid_forest_pixels": len(aligned),
        "interpretation": "prediction_demonstration_between_tracks_not_unsampled_ground_truth",
    }
    freeze = {
        "freeze_id": "phase13-regional-map-" + common.canonical_hash(freeze_basis)[:12],
        "created_utc": common.utc_now(),
        "status": "complete_after_replication_gate_passed",
        "freeze_basis": freeze_basis,
        "outputs": {
            str(path.relative_to(ROOT)): common.sha256(path)
            for path in [RASTER_PATH, POINTS_PATH, FIGURE_PATH]
        },
    }
    common.write_json(FREEZE_PATH, freeze)
    print(json.dumps({"freeze_id": freeze["freeze_id"], **freeze_basis}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
