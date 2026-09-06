# Worked example

`demo_data/reference_transfer.csv` is a small synthetic dataset, not an
observational result. It reproduces one core transfer diagnostic: a
model can preserve the ordering of high and low values in a new forest while
missing its absolute LiDAR scale.

Run it with:

```bash
tessera-study demo
```

The source-forest relationship is transferred to the target forest. Four
spatially distributed target-reference units then estimate a constant
correction using reference labels only. The untouched target test units are
used for both evaluations. The correction improves absolute error while leaving
rank correlation unchanged by construction.
