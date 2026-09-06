# Repository layout

The project is organised by scientific purpose:

```text
README.md             project summary and entry point
environment.yml       full geospatial and modelling environment
src/                  reusable evaluation and validation code
tests/                fast unit and repository-integrity tests
examples/             synthetic worked example
workflows/            configurations grouped by scientific question
results/              curated tables and figures
provenance/           workflow registry and artifact checksums
data/                 local input layout; large files are ignored
output/               generated reproducibility archives; ignored
```

New analysis work belongs in a descriptive workflow directory. Machine-specific
logs, temporary experiments, credentials, and document-production files belong
outside this repository.

Use short, descriptive commit messages, for example:

```text
analysis: add Dutch rank-transfer sensitivity
figures: revise forest-location panel
docs: clarify LiDAR access requirements
fix: preserve held-out folds during calibration
```
