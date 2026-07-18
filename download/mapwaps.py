#!/usr/bin/env python3
"""Download MapWAPS invasive-alien-plant field training data (SA catchments).

Source : Cogill L., Skosana T., Rebelo A.J. et al. (2024–2025), Stellenbosch
         University / figshare (SUNScholar). All catchments CC-BY 4.0.

Each figshare article carries three things: a **field TrainingData** shapefile
(the labelled points we train on), an **AlienMap** raster (the RF prediction map —
*not* a training label; sampling a model's own output is not ground truth), and a
metadata PDF. This script fetches only TrainingData + metadata; the raster is
skipped (large, and not training signal). Raw lands in
``data/labels/raw/<dataset>/`` where the per-catchment adapter picks it up.

Run:  python3 download/mapwaps.py                       # every catchment
      python3 download/mapwaps.py mapwaps_tugela ...    # named subset
Idempotent: files whose md5 already matches are skipped. Stdlib only (no venv).
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

# dataset dir  ->  (figshare article id, aoi_admin1 province)
CATCHMENTS: dict[str, tuple[int, str]] = {
    "mapwaps_olifants_doring": (29958053, "western_cape"),
    "mapwaps_tugela": (25066151, "kwazulu_natal"),
    "mapwaps_umzimvubu": (25050401, "eastern_cape"),
    "mapwaps_luvuvhu": (25050314, "limpopo"),
    "mapwaps_sabie_crocodile": (25050368, "mpumalanga"),
}
RAW = Path("data/labels/raw")
# fetch only files whose name contains one of these (case-insensitive) — skips
# the big AlienMap_*.zip rasters (the RF prediction map, not a training label).
WANT = ("trainingdata", "metadata")


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(dataset: str, article_id: int) -> None:
    out = RAW / dataset
    out.mkdir(parents=True, exist_ok=True)
    api = f"https://api.figshare.com/v2/articles/{article_id}"
    with urllib.request.urlopen(api, timeout=60) as r:
        meta = json.load(r)

    print(f"\n=== {dataset} ===")
    print(f"title  : {meta.get('title')}")
    print(f"doi    : {meta.get('doi')}  license: {(meta.get('license') or {}).get('name')}")
    (out / "_figshare_metadata.json").write_text(json.dumps(meta, indent=2))

    for f in meta.get("files", []):
        name, url, want_md5 = f["name"], f["download_url"], f.get("computed_md5")
        if not any(w in name.lower() for w in WANT):
            print(f"skip  {name} (not training/metadata)")
            continue
        dest = out / name
        if dest.exists() and want_md5 and _md5(dest) == want_md5:
            print(f"skip  {name} (md5 ok)")
        else:
            print(f"get   {name} ({f.get('size')} bytes) ...", end=" ", flush=True)
            urllib.request.urlretrieve(url, dest)
            got = _md5(dest)
            if want_md5 and got != want_md5:
                raise SystemExit(f"MD5 MISMATCH {name}: want {want_md5} got {got}")
            print("ok")
        if dest.suffix.lower() == ".zip":
            with zipfile.ZipFile(dest) as z:
                members = [m for m in z.namelist() if not m.startswith("__MACOSX/")]
                z.extractall(out, members=members)
            dest.unlink()
            print(f"      unzipped {name} → {out} (zip deleted)")


def main(argv: list[str]) -> int:
    wanted = argv or list(CATCHMENTS)
    unknown = [c for c in wanted if c not in CATCHMENTS]
    if unknown:
        raise SystemExit(f"unknown catchment(s) {unknown}; choose from {list(CATCHMENTS)}")
    for dataset in wanted:
        article_id, _province = CATCHMENTS[dataset]
        fetch(dataset, article_id)
    print("\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
