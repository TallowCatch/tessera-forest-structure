# Original experiment code

This directory restores the original research scripts, frozen configurations,
and scientific tests that preceded the streamlined `workflows/` interface.
Script filenames have descriptive names; `filename_map.csv` records their
original names for provenance. Frozen output identifiers and YAML keys retain
their original values so result records remain interpretable. Not every script
belongs to the final manuscript; many are earlier pilots, alternatives, or
diagnostics.

Start with the [experiment index](EXPERIMENT_INDEX.md) for the code behind the
reported analyses. `scripts/` contains the full chronological record,
`configs/` the frozen settings, and `tests/` the original scientific tests.
`scripts/legacy_tracked_versions/` contains prior versions, not current runners.

Most original scripts assume the old project-root layout (`scripts/`,
`configs/`, `data/`, `outputs/`, and `metadata/`). Some batch scripts contain
historical, user-specific JASMIN paths. This is an auditable source archive,
not a claim that every experiment runs directly from the public checkout.
Consult the [data guide](../docs/data_sources.md) before attempting a run.
The public repository does not include restricted LiDAR or experimental Tessera
arrays. Review paths, resource use, and source-data permissions before running
deployment or acquisition scripts.
