# Data sources

## TESSERA

The study uses annual 128-channel Tessera Earth-observation embeddings at 10 m
resolution. The local analyses use the experimental v2 representation supplied
to the project; the 20-forest Dutch analysis uses the public v1 representation.

## Cairngorms LiDAR

The Scottish analysis uses the 2023 Cairngorms airborne-LiDAR products supplied
by collaborators. Surface outcomes are calculated from a 1 m canopy-height
model over non-overlapping 50 m units. Updated point-return products apply the
1.3 m vegetation-height threshold consistently. Access and reuse remain subject
to the source data terms.

## Savelsbos LiDAR

The Dutch analysis uses AHN4 raster products for the 2021 survey of Savelsbos.
The public products and documentation are available from
[Zenodo record 8422129](https://zenodo.org/records/8422129). Site selection and
mask provenance are recorded in the workflow configuration.

## Conventional predictors

Annual Sentinel-1 and Sentinel-2 summaries and terrain variables provide a
conventional Earth-observation comparison. Expected collections, bands, and
aggregation choices are documented with the relevant workflow.

Large source files are not committed. See [`data/README.md`](../data/README.md)
for the local layout and `provenance/` for tracked output checksums.
