"""Reconcile and the incremental skip, on the one-file-per-obs layout.

Two sharp edges live here:

* ``_reconcile_manifest`` prunes every manifest obs outside the canonical thinned
  set. On a ``--species`` run that set is only the requested species, so an
  unguarded reconcile deletes every other species' chips off disk.
* A regrouped obs (moved to another rainfall zone) has a chip on the WRONG month
  calendar. A count-only "has 3 of 3 months" test would skip it forever.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from cmrv.ingest.chips import _reconcile_manifest, extract_training_chips

WINTER = [
    {"start": "0000-02-01", "end": "0000-02-28", "label": "feb"},
    {"start": "0000-05-01", "end": "0000-05-31", "label": "may"},
    {"start": "0000-09-01", "end": "0000-09-30", "label": "sep"},
]
SUMMER = [
    {"start": "0000-07-01", "end": "0000-07-31", "label": "jul"},
    {"start": "0000-09-01", "end": "0000-09-30", "label": "sep"},
    {"start": "0000-12-01", "end": "0000-12-31", "label": "dec"},
]
BY_ZONE = {"winter_rainfall": WINTER, "summer_rainfall": SUMMER}


def _chip_set(tmp_path: Path, obs_ids: list[str], months: list[str] | None = None) -> pd.DataFrame:
    """Write one fake chip file per obs and return the matching manifest."""
    months = months or ["feb", "may", "sep"]
    rows = []
    for oid in obs_ids:
        d = tmp_path / oid
        d.mkdir(parents=True, exist_ok=True)
        uri = d / "2023.tif"
        uri.write_bytes(b"not-a-real-tif")
        rows.append(
            {
                "obs_id": oid,
                "species": "pinus",
                "year": 2023,
                "block_id": 0,
                "lon": 18.5,
                "lat": -33.9,
                "chip_uri": str(uri),
                "months": ",".join(months),
                "n_months": len(months),
                "valid_frac": 1.0,
                "utm_epsg": 32734,
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_parquet(tmp_path / "manifest.parquet")
    return manifest


def _labels(obs_ids: list[str], zone: str = "winter_rainfall") -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "obs_id": obs_ids,
            "species_normalized": ["pinus"] * len(obs_ids),
            "block_id": [0] * len(obs_ids),
            "event_date": ["2023-06-01"] * len(obs_ids),
            "_zone": [zone] * len(obs_ids),
        },
        geometry=[Point(18.5 + i / 100, -33.9) for i in range(len(obs_ids))],
        crs="EPSG:4326",
    )


def _run(tmp, zone, monkeypatch, **kw):
    """Run extraction with the chip build stubbed out; return (manifest, attempts)."""
    tried = []

    def fake(row, months_cfg, bands, out_prefix, opt):
        tried.append({w["label"] for w in months_cfg})
        return "stubbed"

    monkeypatch.setattr("cmrv.ingest.chips._process_obs", fake)
    out = extract_training_chips(
        labels=_labels(["x1"], zone=zone),
        months_by_zone=BY_ZONE,
        bands=["B02"],
        out_prefix=str(tmp),
        max_workers=1,
        **kw,
    )
    return out, tried


def test_reconcile_deletes_obs_outside_the_canonical_set(tmp_path):
    """The sharp edge itself — documented so the guard's purpose stays legible."""
    manifest = _chip_set(tmp_path, ["euc_1", "pine_1"])
    kept = _reconcile_manifest(manifest, {"euc_1"}, str(tmp_path / "manifest.parquet"))
    assert set(kept["obs_id"]) == {"euc_1"}
    assert not (tmp_path / "pine_1" / "2023.tif").exists()
    assert not (tmp_path / "pine_1").exists()  # emptied obs dir removed


def test_species_subset_run_leaves_other_species_chips_alone(tmp_path):
    """The regression: reconcile=False (what --species passes) must not touch pine."""
    _chip_set(tmp_path, ["euc_1", "pine_1"])
    out = extract_training_chips(
        labels=_labels(["euc_1"]),
        months_by_zone=BY_ZONE,
        bands=["B02"],
        out_prefix=str(tmp_path),
        reconcile=False,
    )
    assert (tmp_path / "pine_1" / "2023.tif").exists(), "subset run deleted pine's chips"
    assert set(out["obs_id"]) == {"euc_1", "pine_1"}


def test_full_run_still_prunes_stale_chips(tmp_path):
    """The guard must not disable reconcile for whole-store runs."""
    _chip_set(tmp_path, ["euc_1", "pine_1"])
    out = extract_training_chips(
        labels=_labels(["euc_1"]),
        months_by_zone=BY_ZONE,
        bands=["B02"],
        out_prefix=str(tmp_path),
        reconcile=True,
    )
    assert not (tmp_path / "pine_1" / "2023.tif").exists()
    assert set(out["obs_id"]) == {"euc_1"}


def test_regrouped_obs_is_re_chipped_not_skipped(tmp_path, monkeypatch):
    """The trap: a count-only 'has 3 of 3' test skips this obs forever."""
    _chip_set(tmp_path, ["x1"], ["feb", "may", "sep"])  # chipped on the WINTER calendar
    _, tried = _run(tmp_path, "summer_rainfall", monkeypatch)  # ...now summer-rainfall
    assert tried, "regrouped obs was skipped as 'already chipped'"
    assert tried[0] == {"jul", "sep", "dec"}, "must fetch against the NEW calendar"


def test_regrouped_chip_on_the_old_calendar_is_deleted(tmp_path, monkeypatch):
    """One file per obs holds ALL its months, so a zone change invalidates the
    whole chip. There is no per-month salvage any more — and no per-month bug."""
    _chip_set(tmp_path, ["x1"], ["feb", "may", "sep"])
    out, _ = _run(tmp_path, "summer_rainfall", monkeypatch)
    assert not (tmp_path / "x1" / "2023.tif").exists(), "stale winter chip kept"
    assert out.empty or "x1" not in set(out["obs_id"])


def test_unchanged_zone_is_a_no_op(tmp_path, monkeypatch):
    """The common case must not re-download anything on a plain re-run."""
    _chip_set(tmp_path, ["x1"], ["feb", "may", "sep"])
    out, tried = _run(tmp_path, "winter_rainfall", monkeypatch)
    assert not tried, "re-run refetched an already-complete obs"
    assert (tmp_path / "x1" / "2023.tif").exists()
    assert len(out) == 1


def test_manifest_row_without_its_chip_file_is_dropped(tmp_path, monkeypatch):
    """A crash between the chip write and the manifest flush must not leave the
    skip set claiming a chip that is not on disk."""
    _chip_set(tmp_path, ["x1"], ["feb", "may", "sep"])
    (tmp_path / "x1" / "2023.tif").unlink()
    _, tried = _run(tmp_path, "winter_rainfall", monkeypatch)
    assert tried, "skipped an obs whose chip file is gone"
