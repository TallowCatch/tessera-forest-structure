# Cairngorms local modelling

This workflow compares Tessera embeddings, conventional Sentinel/terrain
predictors, and fused models on 17,316 non-overlapping 50 m forest units. Five
dispersed spatial folds and 2 km exclusion buffers separate training from test
units. Height-adjusted targets are calculated using training data only.

The parameters are in [`parameters.yaml`](parameters.yaml). Check
which local inputs are available and inspect the frozen execution plan with:

```bash
tessera-study status cairngorms-local
tessera-study plan cairngorms-local
```

Curated performance values are in
[`cairngorms_performance.csv`](../../results/tables/cairngorms_performance.csv)
and model comparisons are in
[`cairngorms_model_comparison.csv`](../../results/tables/cairngorms_model_comparison.csv).
The corresponding figures are `fig03_cairngorms_core_predictions` and
`fig04_height_adjustment` under `results/figures/main/`.
