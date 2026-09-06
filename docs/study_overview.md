# Study overview

## Question

Which dimensions of airborne-LiDAR-observed forest structure are predictable
from Tessera Earth-observation embeddings, and do those fitted relationships
transfer between contrasting forest landscapes?

## Landscapes

- **Cairngorms, Scotland:** predominantly coniferous and mixed managed forest,
  evaluated against 2023 airborne LiDAR.
- **Savelsbos, the Netherlands:** temperate broadleaf forest, evaluated against
  the 2021 AHN4 LiDAR campaign.
- **Twenty Dutch forests:** ten conifer-dominated and ten broadleaf-dominated
  sites evaluated under a common national LiDAR product.

## Design

The primary units are non-overlapping 50 m forest areas. Tessera embeddings are
represented at 10 m and summarised over spatial context matching each LiDAR
unit. Complete geographic blocks are held out, with exclusion buffers between
training and testing. Models are compared on identical observations and folds.

The analyses separate:

1. local spatial prediction within each landscape;
2. prediction after accounting for mean and upper-canopy height;
3. direct and jointly trained cross-landscape models;
4. transfer among forests of the same or different broad type;
5. sparse target-reference assistance and relative-rank transfer; and
6. instability in LiDAR-derived reference measurements.

The [workflow index](../workflows/README.md) links each analysis to its
configuration, required inputs, and curated outputs.
