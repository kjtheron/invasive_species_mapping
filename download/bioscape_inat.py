#!/usr/bin/env python3
"""Download the BioSCape Invasive Alien Tree Mapping Project observations from iNaturalist.

Source : iNaturalist project "bioscape-invasive-alien-tree-mapping-project"
         (training data for the spring-2023 CFR invasive-alien-tree classification)
URL    : https://www.inaturalist.org/projects/bioscape-invasive-alien-tree-mapping-project
Target species: Acacia mearnsii, A. saligna, A. longifolia, Pinus pinaster,
                P. radiata, Eucalyptus spp., + other woody IAPs.
License: per-observation (mostly CC-BY-NC / CC0); recorded in each record's
         `license_code` field so the adapter can gate on it later.

Uses the public iNaturalist API v1 (no key). Paginates by id_above (the
stable way past the 10k page-offset cap). Writes one observation per line
(NDJSON) to data/labels/bioscape_inat/observations.ndjson — full raw records,
so the adapter can later pick coords, taxon, positional_accuracy, license, date.

Run:  python3 download/bioscape_inat.py
Re-running overwrites the file with a fresh full pull.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

PROJECT = "bioscape-invasive-alien-tree-mapping-project"
API = "https://api.inaturalist.org/v1/observations"
PER_PAGE = 200
OUT = Path("data/labels/bioscape_inat")
DEST = OUT / "observations.ndjson"
UA = "catchment-mrv/0.1 (invasive-species-mapping; contact jurietheron@gmail.com)"


def _get(id_above: int) -> dict:
    q = urllib.parse.urlencode(
        {
            "project_id": PROJECT,
            "per_page": PER_PAGE,
            "order_by": "id",
            "order": "asc",
            "id_above": id_above,
        }
    )
    req = urllib.request.Request(f"{API}?{q}", headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    total = None
    n = 0
    id_above = 0
    with DEST.open("w") as out:
        while True:
            page = _get(id_above)
            if total is None:
                total = page.get("total_results")
                print(f"project total_results: {total}")
            results = page.get("results", [])
            if not results:
                break
            for obs in results:
                out.write(json.dumps(obs) + "\n")
            n += len(results)
            id_above = results[-1]["id"]
            print(f"  fetched {n}/{total} (id_above={id_above})")
            time.sleep(1.0)  # iNat asks <60 req/min; 1s/page is polite
    print(f"\ndone -> {DEST} ({n} observations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
