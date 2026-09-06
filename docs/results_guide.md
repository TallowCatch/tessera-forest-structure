# Results guide

The `results/` directory contains compact, checked outputs for inspecting the
analyses without rerunning data-intensive models.

| Scientific question | Main figure | Source table |
|---|---|---|
| Study landscapes | `main/fig01_study_landscapes.pdf` | mapped unit inputs |
| Analysis design | `main/fig02_study_design.pdf` | workflow configurations |
| Cairngorms local prediction | `main/fig03_cairngorms_core_predictions.pdf` | `cairngorms_performance.csv` |
| Height-adjusted structure | `main/fig04_height_adjustment.pdf` | `cairngorms_performance.csv` |
| Twenty-forest Dutch transfer | `main/fig05_dutch_transferability.pdf` | `dutch_transfer_fold_metrics.csv` |
| Rank and local-reference transfer | `main/fig06_dutch_rank_reference.pdf` | `rank_transfer.csv`, `local_reference_curve.csv` |
| LiDAR reference instability | `main/figS02_reference_stability.pdf` | `lidar_reference_uncertainty.csv` |
| Direct two-landscape transfer | `diagnostics/cross_landscape_transfer.png` | `cross_landscape_transfer.csv` |
| Joint two-landscape training | `diagnostics/joint_landscape_model.png` | `joint_landscape_model.csv` |

Paths in the table are relative to `results/figures/`; source tables are under
`results/tables/`. Main figures combine related panels, while diagnostics keep
focused checks available without duplicating those panels.

See [`data_dictionary.md`](data_dictionary.md) for table columns and metric
directionality.
