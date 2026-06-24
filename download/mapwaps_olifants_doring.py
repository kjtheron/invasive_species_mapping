#!/usr/bin/env python3
"""Download MapWAPS Olifants-Doring invasive-alien-plant dataset from SUNScholar.

Source : Cogill L., Skosana T., Rebelo A.J. (2025)
         "MapWAPS Invasive Alien Plant map for the Olifants-Doring Catchments"
DOI    : 10.25413/sun.29958053   (SUNScholar / figshare, public, CC-BY 4.0)
Target species: Pinus spp., Acacia spp., Eucalyptus spp., Populus spp.,
                Solanum mauritianum

Pulls every file in the figshare article (10m RF map .tif, field training .shp,
metadata) into data/labels/mapwaps_olifants_doring/. No API key needed — the
figshare public article endpoint is unauthenticated. md5 is verified per file.

Run:  python3 download/mapwaps_olifants_doring.py
Idempotent: files whose md5 already matches are skipped.
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

ARTICLE_ID = 29958053
API = f"https://api.figshare.com/v2/articles/{ARTICLE_ID}"
OUT = Path("data/labels/mapwaps_olifants_doring")


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(API, timeout=60) as r:
        meta = json.load(r)

    print(f"article : {meta['title']}")
    print(f"doi     : {meta.get('doi')}")
    print(f"license : {meta.get('license', {}).get('name')}")
    files = meta.get("files", [])
    print(f"files   : {len(files)}\n")

    # Persist the raw figshare metadata for provenance (license, doi, md5s).
    (OUT / "_figshare_metadata.json").write_text(json.dumps(meta, indent=2))

    for f in files:
        name, url, want = f["name"], f["download_url"], f.get("computed_md5")
        dest = OUT / name
        if dest.exists() and want and _md5(dest) == want:
            print(f"skip  {name} (md5 ok)")
            continue
        print(f"get   {name} ({f.get('size')} bytes) ...", end=" ", flush=True)
        urllib.request.urlretrieve(url, dest)
        got = _md5(dest)
        if want and got != want:
            print(f"\nMD5 MISMATCH {name}: want {want} got {got}", file=sys.stderr)
            return 1
        print("ok")

        if dest.suffix.lower() == ".zip":
            print(f"      unzip {name} ...", end=" ", flush=True)
            with zipfile.ZipFile(dest) as z:
                members = [m for m in z.namelist() if not m.startswith("__MACOSX/")]
                z.extractall(OUT, members=members)
            dest.unlink()
            print("ok (zip deleted)")

    print(f"\ndone -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
