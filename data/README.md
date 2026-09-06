# Data directory

Large geospatial inputs and regenerated arrays are deliberately excluded from
Git. The analysis expects the following local layout:

```text
data/
  raw/          Source LiDAR and other downloaded products
  external/     External masks, boundaries, and cached embedding tiles
  interim/      Fold definitions and model-ready arrays
  processed/    Cohorts, predictions, and harmonised analysis tables
```

The public repository does not redistribute restricted or collaborator-supplied
LiDAR data. Savelsbos AHN4 products are available from Zenodo record 8422129.
The Cairngorms analysis uses the collaborator-supplied 2023 airborne-LiDAR
products described in [`docs/data_sources.md`](../docs/data_sources.md). Tessera
arrays must be obtained from the corresponding authorised dataset release.

Use the semantic paths reported by `tessera-study status <workflow>`. Curated
outputs and their checksums are tracked under `results/` and `provenance/`.
