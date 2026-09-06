# Reproducibility

## Immediate checks

These commands use only tracked files and do not require geospatial libraries,
LiDAR data, or cluster access:

```bash
python -m pip install -e ".[dev]"
tessera-study verify
tessera-study demo
python -m pytest -q
```

`verify` checks registered workflow paths and SHA-256 hashes for every curated
result table and figure. `demo` fits a small ridge model to synthetic
source-forest data, transfers it to untouched target units, and estimates a
constant correction from separate target-reference units.

## Curated outputs

Source tables are stored under `results/tables/`, described in
`docs/data_dictionary.md`, and recorded in `provenance/artifacts.json`. Main
multi-panel figures are under `results/figures/main/`; focused diagnostics are
under `results/figures/diagnostics/`.

Figure regeneration requires the full environment and, for maps and unit-level
panels, the ignored prediction and geospatial inputs listed in
`workflows/figure_generation/README.md`.

## Full environment

```bash
conda env create -f environment.yml
conda activate tessera-forest-structure
python -m pip install -e .
```

Inspect each analysis before running it:

```bash
tessera-study status <workflow>
tessera-study plan <workflow>
```

The original large runs used JASMIN LOTUS. The current credentials are not
active, so those runs were not repeated during repository cleanup. Equivalent
storage and compute resources can be used after restoring the inputs listed by
`status`. Independent forest or spatial-chunk jobs can also be run sequentially
when runtime permits.

## Limits

- Collaborator-supplied Cairngorms LiDAR cannot be redistributed here.
- The experimental Tessera representation requires access to its authorised
  release; public Tessera v1 is used for the 20-forest Dutch experiment.
- Exact neural-network equality is not guaranteed across PyTorch versions,
  hardware, and numerical libraries.
- The synthetic example validates evaluation logic but is not ecological
  evidence.

## Archive

```bash
make package
```

This produces one deterministic ZIP containing tracked code, configurations,
documentation, curated results, figures, tests, and the example dataset. It
excludes credentials, caches, untracked source data, and generated build files.
