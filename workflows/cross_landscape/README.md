# Cross-landscape evaluation

This workflow asks whether relationships fitted in Scotland transfer directly
to Savelsbos, whether the reverse direction works, and whether joint training
improves over separate local models. It uses four harmonised 50 m outcomes and
retains each landscape's held-out spatial folds.

The direct-transfer model is ridge regression. The joint analysis uses a
multilayer perceptron with equal total training weight for each landscape. Full
parameters are in [`parameters.yaml`](parameters.yaml).

```bash
tessera-study status cross-landscape
tessera-study plan cross-landscape
```

Results are in
[`cross_landscape_transfer.csv`](../../results/tables/cross_landscape_transfer.csv)
and [`joint_landscape_model.csv`](../../results/tables/joint_landscape_model.csv).
Supporting plots are under `results/figures/diagnostics/`.

The underlying Scottish LiDAR is collaborator supplied, so the complete run
requires authorised inputs.
