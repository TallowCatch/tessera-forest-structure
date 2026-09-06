# Figure generation

The scripts in this directory build the tracked figures from curated tables and
local analysis products. Outputs are written to `results/figures/main/` or
`results/figures/diagnostics/`.

Run from the repository root in the full geospatial environment:

```bash
python workflows/figure_generation/make_figures.py
python workflows/figure_generation/make_study_workflow_figure.py
python workflows/figure_generation/make_point_cloud_diagnostic.py
python workflows/figure_generation/make_height_model_diagnostic.py
python workflows/figure_generation/make_target_distributions.py
```

The multi-panel figures use the tracked tables under `results/tables/`. Maps,
target distributions, and unit-level observed-versus-predicted panels also need
the following untracked local products:

```text
data/processed/cairngorms_surface_targets.parquet
data/processed/cairngorms_surface_predictions.parquet
data/processed/cairngorms_point_cloud_predictions.parquet
data/processed/cairngorms_height_predictions.parquet
data/processed/cairngorms_unet_predictions.parquet
data/processed/dutch_eligible_units_map.parquet
data/processed/savelsbos_targets.parquet
data/processed/savelsbos_predictions.parquet
data/processed/savelsbos_site.gpkg
data/interim/cairngorms_embedding_patches/
data/interim/cairngorms_height_adjusted/
data/interim/cairngorms_return_profiles/
```

`rebuild_dutch_eligible_units.py` and `enrich_dutch_unit_cv.py` reconstruct the
mapped Dutch unit table from public source rasters when those files are present.
All path names describe their scientific content rather than execution order.
