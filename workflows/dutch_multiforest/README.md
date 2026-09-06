# Twenty-forest Dutch transfer

This workflow evaluates 38,297 eligible 50 m units in ten conifer-dominated and
ten broadleaf-dominated Dutch forests under one national LiDAR product. For each
held-out target forest, it compares local training with training on forests in
the same broad group, the opposite group, and all other sites. A separate
diagnostic tests whether geographic distance explains transfer loss.

The site-selection thresholds, target groups, folds, buffers, models, and seeds
are in [`parameters.yaml`](parameters.yaml).

```bash
tessera-study status dutch-multiforest
tessera-study plan dutch-multiforest
```

The curated tables are:

- [`dutch_forest_sites.csv`](../../results/tables/dutch_forest_sites.csv)
- [`dutch_pairwise_transfer.csv`](../../results/tables/dutch_pairwise_transfer.csv)
- [`dutch_transfer_fold_metrics.csv`](../../results/tables/dutch_transfer_fold_metrics.csv)
- [`dutch_transfer_uncertainty.csv`](../../results/tables/dutch_transfer_uncertainty.csv)
- [`dutch_distance_association.csv`](../../results/tables/dutch_distance_association.csv)

The principal figure is `results/figures/main/fig05_dutch_transferability.pdf`.
The full pairwise matrix remains available in the source table. Each target
forest is independent and can be evaluated sequentially in a compatible
environment.
