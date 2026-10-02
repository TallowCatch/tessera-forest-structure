#!/usr/bin/env python3
"""Audit public GEDI and NEON catalog availability without downloading science data."""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


CMR_COLLECTIONS_URL = "https://cmr.earthdata.nasa.gov/search/collections.json"
CMR_GRANULES_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
NEON_PRODUCT_URL = "https://data.neonscience.org/api/v0/products/{product}"
GEDI_V3_CONCEPT_ID = "C3974616135-LPCLOUD"

SITE_BBOXES = {
    "SOAP": (-119.2919464457, 37.0142750706, -119.1767316705, 37.1042875314),
    "TEAK": (-119.0962033206, 36.9514910125, -118.9715140645, 37.0427798538),
    "BART": (-71.3342560121, 43.9919475560, -71.2094969977, 44.0819645961),
}

NEON_PRODUCTS = {
    "discrete_lidar": "DP1.30003.001",
    "waveform_lidar": "DP1.30001.001",
    "ecosystem_structure": "DP3.30015.001",
}


def get_json(url: str) -> tuple[dict, dict[str, str]]:
    request = Request(url, headers={"Accept": "application/json", "User-Agent": "tessera-gedi-fhd/0.1"})
    with urlopen(request, timeout=60) as response:
        headers = {key.lower(): value for key, value in response.headers.items()}
        return json.load(response), headers


def gedi_collections() -> list[dict]:
    query = urlencode({"short_name": "GEDI02_B", "page_size": 20})
    payload, _ = get_json(f"{CMR_COLLECTIONS_URL}?{query}")
    return [
        {
            "version": item.get("version_id"),
            "concept_id": item.get("id"),
            "time_start": item.get("time_start"),
            "time_end": item.get("time_end"),
        }
        for item in payload["feed"]["entry"]
    ]


def gedi_granule_count(site: str, year: int) -> int:
    query = urlencode(
        {
            "collection_concept_id": GEDI_V3_CONCEPT_ID,
            "bounding_box": ",".join(map(str, SITE_BBOXES[site])),
            "temporal": f"{year}-01-01T00:00:00Z,{year}-12-31T23:59:59Z",
            "page_size": 0,
        }
    )
    _, headers = get_json(f"{CMR_GRANULES_URL}?{query}")
    return int(headers["cmr-hits"])


def neon_months(product: str, site: str, year: int) -> list[str]:
    payload, _ = get_json(NEON_PRODUCT_URL.format(product=product))
    for item in payload["data"]["siteCodes"]:
        if item["siteCode"] == site:
            return [month for month in item["availableMonths"] if month.startswith(f"{year}-")]
    return []


def build_audit(year: int) -> dict:
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "year": year,
        "gedi_collections": gedi_collections(),
        "gedi_v3_granules_by_site": {
            site: gedi_granule_count(site, year) for site in SITE_BBOXES
        },
        "neon_months_by_product_and_site": {
            name: {
                site: neon_months(product, site, year) for site in SITE_BBOXES
            }
            for name, product in NEON_PRODUCTS.items()
        },
        "notes": [
            "CMR counts granules intersecting each public NEON Priority-1 bounding box, not quality-filtered shots.",
            "NEON product metadata are public, but science-data downloads require an account/API token.",
            "Exact science counts must be computed from GEDI02_B.003 after spatial and quality filtering.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.year < 2019 or args.year > date.today().year:
        raise SystemExit("GEDI audit year must be between 2019 and the current year")
    audit = build_audit(args.year)
    rendered = json.dumps(audit, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
