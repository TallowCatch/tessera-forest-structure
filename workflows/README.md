# Scientific workflows

Supported workflows are organised by scientific question, not development
order.

| Workflow | Purpose |
|---|---|
| [Cairngorms local modelling](cairngorms_local/README.md) | Predict canopy structure under buffered within-landscape holdout |
| [Savelsbos local replication](savelsbos_local/README.md) | Repeat local evaluation in a contrasting broadleaf forest |
| [Cross-landscape evaluation](cross_landscape/README.md) | Test direct transfer and joint Scotland-Netherlands training |
| [Twenty-forest Dutch transfer](dutch_multiforest/README.md) | Compare local, within-group, and between-group transfer across 20 forests |
| [Reference-assisted transfer](reference_assisted/README.md) | Test rank preservation and sparse target-reference budgets |
| [Reference uncertainty](reference_uncertainty/README.md) | Quantify finite-support instability in Cairngorms LiDAR metrics |
| [Sensitivity analysis](sensitivity_analysis/) | Check targeted methodological sensitivities |
| [Figure generation](figure_generation/README.md) | Rebuild figures from curated results and local inputs |

For any registered workflow, use `tessera-study status <workflow>` to see which
inputs are present and `tessera-study plan <workflow>` to inspect its configured
scope and outputs. `tessera-study demo` is the quick runnable example.
