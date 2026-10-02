# Experiment source index

The supported workflow descriptions are in [`../workflows/`](../workflows/).
The paths below identify original analysis code, not a new execution order.
Read each script's matching configuration and output freeze before attributing
a result to a run. Earlier alternatives remain in `scripts/` for audit.

| Analysis | Original source to inspect | Frozen settings |
|---|---|---|
| Cairngorms local, Tessera v2 and model comparison | `scripts/tessera_v2_cairngorms_v2_worker.py`, `scripts/aggregate_tessera_v2_cairngorms_v2.py`, `scripts/cairngorms_v2_unet_worker.py`, `scripts/aggregate_cairngorms_v2_unet.py`, `scripts/cairngorms_v2_surface_v2_surface_raw_worker.py`, `scripts/cairngorms_v2_surface_v2_surface_adjusted_worker.py`, `scripts/aggregate_cairngorms_v2_surface.py` | `configs/tessera_v2.yaml`, `configs/cairngorms_v2_unet.yaml`, `configs/cairngorms_v2_surface.yaml` |
| Cairngorms height adjustment and point-cloud outcomes | `scripts/prepare_cairngorms_height_adjusted.py`, `scripts/cairngorms_height_adjusted_height_adjusted_worker.py`, `scripts/aggregate_cairngorms_height_adjusted.py`, `scripts/prepare_cairngorms_point_cloud_cairngorms_corrected_lidar.py`, `scripts/cairngorms_point_cloud_corrected_lidar_worker.py`, `scripts/aggregate_cairngorms_point_cloud_cairngorms_corrected_lidar.py` | `configs/cairngorms_height_adjusted.yaml`, `configs/cairngorms_point_cloud_cairngorms_corrected_lidar.yaml` |
| Savelsbos local evaluation | `scripts/evaluate_ahn4_replication_ahn4.py`, `scripts/evaluate_tessera_v2_savelsbos_v2.py` | `configs/ahn4_replication_ahn4_deciduous.yaml`, `configs/tessera_v2.yaml` |
| Scotland-Netherlands transfer and pooled fitting | `scripts/evaluate_scotland_netherlands_transfer.py`, `scripts/evaluate_pooled_transfer_scotland_netherlands_pooled.py`, `scripts/evaluate_pooled_mlp_scotland_netherlands_pooled_mlp.py` | `configs/scotland_netherlands_transfer.yaml`, `configs/pooled_transfer_scotland_netherlands_pooled.yaml`, `configs/pooled_mlp_scotland_netherlands_pooled_mlp.yaml` |
| Dutch multi-forest transfer | `scripts/prepare_dutch_multisite.py`, `scripts/evaluate_dutch_multisite_dutch_transfer.py`, `scripts/prepare_dutch_expanded.py`, `scripts/evaluate_dutch_expanded.py` | `configs/dutch_multisite_transfer.yaml`, `configs/dutch_expanded_transfer.yaml` |
| Dutch reference assistance and spatial ranking | `scripts/evaluate_dutch_reference_assisted.py`, `scripts/evaluate_dutch_rank_transfer.py` | `configs/dutch_reference_assisted.yaml`, `configs/dutch_rank_transfer.yaml` |
| Cairngorms finite-support uncertainty | `scripts/analyze_reference_uncertainty_cairngorms_reference_uncertainty.py` | `configs/reference_uncertainty_cairngorms_reference_uncertainty.yaml` |
| Dutch AHN3-AHN4 temporal transfer (later analysis) | `../workflows/dutch_temporal_transfer/run.py` | `../workflows/dutch_temporal_transfer/parameters.yaml` |

Each analysis may also have preparation, acquisition, aggregation, validation,
JASMIN batch, deployment, collection, and plotting scripts with matching study
names under `scripts/` and `scripts/jasmin/`. All are included here.
This table highlights entry points, not a complete dependency graph or a
guarantee that each file generated a particular final reported table. Verify
result provenance against the frozen outputs.
