# Dutch temporal transfer

This workflow tests whether a model fitted to matching-year TESSERA embeddings
and AHN3 forest-structure measurements transfers to AHN4 observations of the
same 50 m units. It uses the six outcome definitions shared by the two national
LiDAR products.

The cohort is defined using AHN3 measurements. AHN4 target values are not used
to select mature forest units, so canopy loss between campaigns is not silently
removed. Only units whose AHN3 acquisition year is 2017--2019 are eligible,
because public European TESSERA coverage begins in 2017.

For each site, outcome and buffered spatial fold, the workflow compares:

- an AHN3 model evaluated on held-out AHN3 observations;
- that unchanged model applied to matching-year AHN4 embeddings and outcomes;
- a contemporaneous AHN4 model using the same training and test units; and
- the no-change prediction that the AHN4 value equals the AHN3 observation.

It also compares observed AHN4--AHN3 changes with changes predicted by the
unchanged AHN3 model. Sensor, pulse-density and acquisition-season differences
remain part of the measured temporal-transfer penalty and must be discussed as
a limitation.

Run the resumable workflow with:

```bash
python -u workflows/dutch_temporal_transfer/run.py
```

National metric rasters are downloaded one at a time, reduced to the selected
forest units, and removed after a checkpoint has been written. TESSERA tiles
are handled in the same way. The process can therefore run on a machine without
space for the complete national archives and can be restarted after interruption.
