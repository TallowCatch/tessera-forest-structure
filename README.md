# Mapping Forest Canopy Structural Heterogeneity with Tessera

Code, configurations, curated results, and reproducibility checks for evaluating
whether annual Tessera Earth-observation embeddings predict LiDAR-derived forest
structure in Scotland and the Netherlands.

## Scientific scope

The analyses cover:

- local spatial prediction in the Cairngorms and Savelsbos;
- separation of outer-canopy and return-profile measurements;
- training-only adjustment for canopy height;
- direct transfer between Scotland and the Netherlands;
- transfer among 20 Dutch forests;
- relative spatial ranking and sparse local-reference assistance; and
- finite-support uncertainty in LiDAR-derived measurements.

Models are evaluated on spatially held-out 50 m units with explicit exclusion
buffers and spatial weighting. The results support local mapping of defined
canopy-structure measurements; they do not establish direct biodiversity
prediction or a universal relationship across forests.

## Quick start

The lightweight interface requires Python 3.10 or later:

```bash
python -m pip install -e ".[dev]"
tessera-study list
tessera-study verify
tessera-study demo
python -m pytest -q
```

The deterministic demo uses labelled synthetic data and completes in seconds.
It illustrates how a transferred model can preserve spatial ordering while
missing the target forest's absolute LiDAR scale.

Inspect any registered analysis with:

```bash
tessera-study status dutch-multiforest
tessera-study plan dutch-multiforest
```

`status` reports which inputs are present locally. `plan` prints the configured
analysis scope and expected outputs without modifying existing results.

## Repository map

```text
src/                 reusable metrics, modelling helpers, and command-line tools
workflows/           analysis configurations grouped by scientific question
results/             curated tables, figures, and reconstruction diagnostics
examples/            deterministic synthetic worked example
provenance/          workflow registry and SHA-256 artifact manifest
tests/               unit and repository-integrity tests
docs/                data, result, and reproducibility documentation
data/                documented local input layout; large inputs are not tracked
```

See the [repository layout](docs/repository_layout.md),
[workflow index](workflows/README.md), [result dictionary](docs/data_dictionary.md),
and [reproducibility guide](docs/reproducibility.md).

## Full environment and data

```bash
conda env create -f environment.yml
conda activate tessera-forest-structure
python -m pip install -e .
```

AHN4 products and several mapping layers are public. The Cairngorms LiDAR and
the experimental Tessera representation used in the local analyses have access
restrictions and are not stored in Git. Their expected paths and access
conditions are documented in [the data guide](docs/data_sources.md).

The original large runs used JASMIN LOTUS. JASMIN access is not needed to
inspect the code, verify curated outputs, or run the demonstration. Re-executing
the complete analyses requires equivalent storage and compute resources plus
the listed source data.

## Reproducibility archive

```bash
make package
```

This creates one deterministic archive at
`output/release/tessera_forest_structure_reproducibility.zip`. Credentials,
caches, untracked source data, and generated build products are excluded.

Original code is released under the MIT License. Data and third-party products
retain their original licences and access conditions. The archived release is
available at [doi:10.5281/zenodo.22426672](https://doi.org/10.5281/zenodo.22426672).
