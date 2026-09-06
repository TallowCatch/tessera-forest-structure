# Reference-assisted transfer

This workflow separates two transfer questions across the 20 Dutch forests:
whether a source-trained model preserves relative ordering within a new forest,
and how absolute and rank performance change when local LiDAR references are
added. Reference budgets are 0, 2, 5, 10, 25, 50, 100, and 200 units. Reference
units always come from the training side of a target forest's spatial fold;
test labels are never used for fitting.

The methods and evaluation measures are in
[`parameters.yaml`](parameters.yaml). A small synthetic version can be run
without the restricted study inputs:

```bash
tessera-study demo
tessera-study status reference-assisted
tessera-study plan reference-assisted
```

Curated outputs are
[`rank_transfer.csv`](../../results/tables/rank_transfer.csv),
[`local_reference_curve.csv`](../../results/tables/local_reference_curve.csv),
and `fig06_dutch_rank_reference.pdf` under `results/figures/main/`.
