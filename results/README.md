# Curated results

This directory contains the source tables and final figure files needed to
inspect the reported analyses without searching the development history.

```text
tables/               machine-readable reported values and diagnostics
figures/main/         main multi-panel and supplementary analysis figures
figures/diagnostics/  focused standalone diagnostic plots
```

Every registered artifact has a SHA-256 checksum in
`provenance/artifacts.json`. Run `tessera-study verify` after moving or sharing
the repository. Table meanings are documented in
[`docs/data_dictionary.md`](../docs/data_dictionary.md).

These are frozen copies. Do not edit them manually; regenerate an analysis in a
fresh run directory, validate it, then deliberately update the curated copy and
artifact manifest.
