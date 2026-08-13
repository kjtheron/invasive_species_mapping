#!/usr/bin/env python3
"""Verify (and, with a session, fetch) the NIAPS 2023 national IAP survey.

NIAPS = **National Invasive Alien Plant Survey**, DFFE / Working for Water.
Aerial-survey polygons for 14 alien taxa across all of South Africa, each carrying
a ``gridcode`` = **percent density** (1-100, confirmed against the A0 map legend).
Underlies Kotze et al. (2025).

**The download is manual.** The SharePoint href below answers ``403 Access denied``
to any unauthenticated client — it is a personal OneDrive share that requires an
interactive browser login, so no script can fetch it. Same situation the BioSCape
adapter had with Earthdata. What this script *can* do is tell you whether the copy
on disk is the real one: a truncated 606 MB GeoPackage opens fine and silently
returns fewer polygons.

Run:  python3 download/niaps.py           # verify the files on disk
      python3 download/niaps.py --sha     # also check sha256 (slow, ~600 MB)
Exit code is non-zero if anything is missing or the wrong size.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

DATASET = "niaps_2023"
RAW_DIR = Path("data/labels/raw") / DATASET

# Where a human goes to get it. The Google Sites page is the index ("Abundance"
# section); the file itself lives on DFFE's SharePoint, not Google Drive.
INDEX_PAGE = "https://sites.google.com/site/wfwplanning/assessment"
SHAREPOINT_DIR = (
    "https://environmentza-my.sharepoint.com/personal/"
    "awannenburgh_environment_gov_za/_layouts/15/onedrive.aspx"
    "?id=%2Fpersonal%2Fawannenburgh%5Fenvironment%5Fgov%5Fza%2FDocuments"
    "%2FDocuments%2FArcGIS%2F2023%20NIAPS%20GeoPackage%2Egpkg"
)

# name -> (bytes, sha256) as delivered 2026-08-11.
EXPECTED: dict[str, tuple[int, str]] = {
    "2023 NIAPS GeoPackage.gpkg": (
        635_224_064,
        "e5673f750a44163d0a18fe9ae40ec6198a74d077d9f7b99f4e098ef59457db12",
    ),
    "A0 2023 NIAPS Topo.pdf": (
        47_856_605,
        "4f5e8f50e5d02fdf1ffe11e1ec3eda8b5d8688e5266cee6abae2b90362c315ca",
    ),
}
# Not checksummed: the .aux.xml sidecar (ArcGIS writes it) and NIAPS_2023.txt
# (the provenance/permission email, added by hand).


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(check_sha: bool = False) -> list[str]:
    """Return a list of problems; empty means the raw folder is good."""
    problems = []
    for name, (size, sha) in EXPECTED.items():
        p = RAW_DIR / name
        if not p.exists():
            problems.append(f"MISSING  {name}")
            continue
        actual = p.stat().st_size
        if actual != size:
            problems.append(f"SIZE     {name}: {actual:,} bytes, expected {size:,}")
            continue
        if check_sha and (got := _sha256(p)) != sha:
            problems.append(f"SHA256   {name}: {got[:16]}… != {sha[:16]}…")
            continue
        print(f"ok       {name} ({actual / 1e6:.0f} MB)")
    return problems


def main(argv: list[str]) -> int:
    problems = verify(check_sha="--sha" in argv)
    if not problems:
        print(f"\n{DATASET}: raw folder complete. Next: uv run cmrv labels-niaps-ingest")
        return 0
    print("\n".join(problems), file=sys.stderr)
    print(
        f"\nDownload manually into {RAW_DIR}/ — the SharePoint link needs a browser login:\n"
        f"  index page : {INDEX_PAGE}   (under 'Abundance')\n"
        f"  direct     : {SHAREPOINT_DIR}\n",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
