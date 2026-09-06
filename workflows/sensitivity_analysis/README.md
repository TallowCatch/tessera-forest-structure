# Sensitivity analyses

This directory contains narrowly scoped methodological checks that do not
require a separate modelling workflow.

`check_sd_denominator_sensitivity.py` compares the stored sample-standard-
deviation convention (`N - 1`) with population scaling (`N`) and writes the
descriptive result to `results/tables/sd_denominator_sensitivity.csv`.

Run it only after restoring the Cairngorms target and prediction tables:

```bash
python workflows/sensitivity_analysis/check_sd_denominator_sensitivity.py
```
