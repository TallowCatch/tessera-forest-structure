# Curated result dictionary

All files in `results/tables/` are comma-separated, use one header row, and do
not contain index columns. Target and model identifiers are shared with the
workflow configurations.

## Local analyses

- `cairngorms_performance.csv`: equal-fold aggregate performance by outcome,
  target variant, Tessera version, and model family.
- `cairngorms_model_comparison.csv`: paired spatial-block bootstrap contrasts;
  `delta_rmse` is candidate minus reference, so negative values favour the
  candidate.
- `cairngorms_point_cloud_performance.csv`: fold and aggregate performance for
  the four point-return outcomes and their height-adjusted variants.
- `savelsbos_local_performance.csv`: fold and aggregate Savelsbos metrics by
  outcome, original or height-adjusted target, and model.

## Transfer analyses

- `cross_landscape_transfer.csv`: local and direction-specific Scotland-
  Netherlands predictions for each harmonised outcome.
- `joint_landscape_model.csv`: local versus jointly trained two-landscape MLP
  performance.
- `dutch_forest_sites.csv`: identities, broad forest groups, retained unit
  counts, spatial centroids, and selection provenance for the 20 Dutch sites.
- `dutch_pairwise_transfer.csv`: every source-target forest pair, distance,
  forest-group match, error, rank correlation, and error relative to the target
  forest's local model.
- `dutch_transfer_fold_metrics.csv`: fold-level local, same-group,
  opposite-group, and all-other-site results.
- `dutch_transfer_uncertainty.csv`: spatial-block bootstrap contrasts among
  transfer scenarios.
- `dutch_distance_association.csv`: coefficients from the descriptive
  distance/forest-group diagnostic.

## Reference assistance and uncertainty

- `rank_transfer.csv`: zero-reference absolute and rank performance across the
  20 Dutch forests.
- `dutch_rank_transfer_by_fold.csv`: zero-reference fold-level values used in
  the exact-value versus rank comparison.
- `local_reference_curve.csv`: exact-value and rank metrics for reference
  budgets from 0 to 200, including the matched subset that supports 200 units.
- `lidar_reference_uncertainty.csv`: across-fold block-jackknife summary for
  eight Cairngorms LiDAR-derived measurements.
- `lidar_reference_uncertainty_by_fold.csv`: corresponding fold-level values.
- `sd_denominator_sensitivity.csv`: analytical sensitivity to using `N` rather
  than `N - 1` in canopy-height standard deviation.
- `diagnostics/cairngorms_height_adjustment.csv`: target-wise relationship with
  canopy height and the resulting residual variance.
- `diagnostics/cairngorms_embedding_version_comparison.csv`: matched comparison
  of the two embedding versions used in the sensitivity figure.

## Reconstruction diagnostics

- `diagnostics/dutch_eligible_unit_reconstruction.csv`: site-level comparison
  of the reconstructed map cohort with the frozen eligible-unit counts.
- `diagnostics/dutch_height_cv_reconstruction.csv`: agreement between height CV
  reconstructed for the map and the independently frozen AHN4 cohort.

## Common columns

- `rmse`, `mae`, `bias`, and `r2` quantify exact numerical agreement.
- `block_spearman` is Spearman correlation after averaging observations within
  independent evaluation blocks.
- `unit_spearman` is Spearman correlation among individual 50 m units.
- `relative_rmse` and `pooled_over_local_rmse` divide candidate RMSE by the
  corresponding local-model RMSE; values above one indicate worse performance
  than local training.
- `rows`, `n`, and `units` count 50 m observations unless stated otherwise.
- `blocks`, `evaluation_blocks`, and `site_folds` describe independent spatial
  support used for aggregation or uncertainty summaries.
