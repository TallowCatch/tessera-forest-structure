# LiDAR reference uncertainty

This workflow quantifies finite-support and within-window spatial-composition
instability for eight Cairngorms canopy measurements. Each 50 m unit is divided
into 25 non-overlapping 10 m blocks. Each block is omitted in turn, all metrics
are recalculated, and the resulting block-jackknife standard errors are compared
with held-out Tessera residuals.

The scope and exclusions are explicit in
[`parameters.yaml`](parameters.yaml). This is not a complete LiDAR acquisition
error model.

```bash
tessera-study status reference-uncertainty
tessera-study plan reference-uncertainty
```

Summary and fold-level values are in
[`lidar_reference_uncertainty.csv`](../../results/tables/lidar_reference_uncertainty.csv)
and
[`lidar_reference_uncertainty_by_fold.csv`](../../results/tables/lidar_reference_uncertainty_by_fold.csv).
The corresponding figure is `results/figures/main/figS02_reference_stability.pdf`.
