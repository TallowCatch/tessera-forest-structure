# Contributing

## Before changing an analysis

1. Identify the scientific question and update the matching directory under
   `workflows/`.
2. Keep train, reference, validation, and test observations explicitly
   separated.
3. Record seeds, spatial support, fold construction, exclusion buffers,
   weighting, model settings, and all input versions in `parameters.yaml`.
4. Add or update a focused test for any changed scientific logic.
5. Write generated tables and figures to `results/` only after the run is
   frozen and verified.

Use descriptive names based on the scientific question or operation. Keep
temporary experiments and machine-specific run records outside the repository.

## Checks

```bash
make verify
make test
make demo
```

The full analysis additionally requires the environment in `environment.yml`
and the inputs listed by `tessera-study status <workflow>`.

## Commit messages

Use `analysis:`, `data:`, `figures:`, `docs:`, `fix:`, `tests:`, or `chore:`
followed by a concrete description. Keep one scientific or maintenance purpose
per commit. Avoid execution-order labels.
