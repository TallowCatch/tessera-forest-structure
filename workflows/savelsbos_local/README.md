# Savelsbos local replication

This workflow repeats spatially held-out local prediction in the broadleaf
Savelsbos forest using 2021 AHN4 point-cloud-derived measurements and matching
Tessera embeddings. It evaluates six return-height outcomes at 50 m support and
also records training-only height-adjusted variants.

The parameters are in [`parameters.yaml`](parameters.yaml). Use:

```bash
tessera-study status savelsbos-local
tessera-study plan savelsbos-local
```

Fold-level and aggregate results are in
[`savelsbos_local_performance.csv`](../../results/tables/savelsbos_local_performance.csv).

AHN4 is public, but the matching Tessera features must be restored separately.
