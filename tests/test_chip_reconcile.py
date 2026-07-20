"""Reconcile must not delete chips a subset run never knew about.

``_reconcile_manifest`` prunes every manifest obs outside the canonical thinned
set. On a ``--species`` run that set is only the requested species, so an
unguarded reconcile deletes every other species' chips off disk.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from cmrv.ingest.chips import _reconcile_manifest, extract_training_chips

MONTHS = [{"start": "2023-02-01", "end": "2023-02-28", "label": "feb"}]
WINTER = [
    {"start": "2023-02-01", "end": "2023-02-28", "label": "feb"},
    {"start": "2023-05-01", "end": "2023-05-31", "label": "may"},
    {"start": "2023-09-01", "end": "2023-09-30", "label": "sep"},
]
SUMMER = [
    {"start": "2023-07-01", "end": "2023-07-31", "label": "jul"},
    {"start": "2023-09-01", "end": "2023-09-30", "label": "sep"},
    {"start": "2023-12-01", "end": "2023-12-31", "label": "dec"},
]
BY_ZONE = {"winter_rainfall": WINTER, "summer_rainfall": SUMMER}


def _chip_set(
    tmp_path: Path, obs_ids: list[str], months: list[str] | None = None
) -> pd.DataFrame:
    """Write one fake chip file per (obs, month) and return the matching manifest."""
    rows = []
    for oid in obs_ids:
        d = tmp_path / oid / "2023"
        d.mkdir(parents=True, exist_ok=True)
        for label in months or ["feb"]:
            uri = d / f"{label}.tif"
            uri.write_bytes(b"not-a-real-tif")
            rows.append(
                {
                    "obs_id": oid,
                    "month_label": label,
                    "year": 2023,
                    "chip_uri": str(uri),
                    "block_id": 0,
                }
            )
    manifest = pd.DataFrame(rows)
    manifest.to_parquet(tmp_path / "manifest.parquet")
    return manifest


def _labels(obs_ids: list[str], zone: str | None = None) -> gpd.GeoDataFrame:
    cols = {
        "obs_id": obs_ids,
        "block_id": [0] * len(obs_ids),
        "event_date": ["2023-06-01"] * len(obs_ids),
    }
    if zone:
        cols["_zone"] = [zone] * len(obs_ids)
    return gpd.GeoDataFrame(
        cols,
        geometry=[Point(18.5 + i / 100, -33.9) for i in range(len(obs_ids))],
        crs="EPSG:4326",
    )


def _blocks():
    return gpd.GeoDataFrame({"block_id": [0]}, geometry=[Point(18.5, -33.9).buffer(1)])


def test_reconcile_deletes_obs_outside_the_canonical_set(tmp_path):
    """The sharp edge itself — documented so the guard's purpose stays legible."""
    manifest = _chip_set(tmp_path, ["euc_1", "pine_1"])
    kept = _reconcile_manifest(manifest, {"euc_1"}, str(tmp_path / "manifest.parquet"))
    assert set(kept["obs_id"]) == {"euc_1"}
    assert not (tmp_path / "pine_1" / "2023" / "feb.tif").exists()


def test_species_subset_run_leaves_other_species_chips_alone(tmp_path):
    """The regression: reconcile=False (what --species passes) must not touch pine."""
    _chip_set(tmp_path, ["euc_1", "pine_1"])
    # Only euc is passed in — and it's already fully chipped, so the run early-returns
    # via the "nothing to do" path, which is where the unguarded reconcile fired.
    out = extract_training_chips(
        labels=_labels(["euc_1"]),
        blocks=_blocks(),
        months_cfg=MONTHS,
        bands=["B02"],
        out_prefix=str(tmp_path),
        reconcile=False,
    )
    assert (tmp_path / "pine_1" / "2023" / "feb.tif").exists(), "subset run deleted pine's chips"
    assert set(out["obs_id"]) == {"euc_1", "pine_1"}


def test_full_run_still_prunes_stale_chips(tmp_path):
    """The guard must not disable reconcile for whole-store runs."""
    _chip_set(tmp_path, ["euc_1", "pine_1"])
    out = extract_training_chips(
        labels=_labels(["euc_1"]),
        blocks=_blocks(),
        months_cfg=MONTHS,
        bands=["B02"],
        out_prefix=str(tmp_path),
        reconcile=True,
    )
    assert not (tmp_path / "pine_1" / "2023" / "feb.tif").exists()
    assert set(out["obs_id"]) == {"euc_1"}


# --- regrouping: moving a label to another zone re-chips only what differs ---


def _run(tmp, zone, monkeypatch, **kw):
    """Run extraction with the STAC path stubbed out; return the groups it tried."""
    tried = []

    def _fake_group(bid, year, grp, months_cfg, *a, **k):
        tried.append({w["label"] for w in months_cfg})
        return []

    monkeypatch.setattr("cmrv.ingest.chips._process_group", _fake_group)
    out = extract_training_chips(
        labels=_labels(["x1"], zone=zone),
        blocks=_blocks(),
        months_cfg=WINTER,
        months_by_zone=BY_ZONE,
        bands=["B02"],
        out_prefix=str(tmp),
        max_workers=1,
        **kw,
    )
    return out, tried


def test_regrouped_obs_is_re_chipped_not_skipped(tmp_path, monkeypatch):
    """The trap: a count-only 'has 3 of 3' test skips this obs forever."""
    _chip_set(tmp_path, ["x1"], ["feb", "may", "sep"])  # chipped on the WINTER calendar
    _, tried = _run(tmp_path, "summer_rainfall", monkeypatch)  # ...now summer-rainfall
    assert tried, "regrouped obs was skipped as 'fully chipped'"
    assert tried[0] == {"jul", "sep", "dec"}, "must fetch against the NEW calendar"


def test_regroup_prunes_stale_months_but_keeps_the_shared_one(tmp_path, monkeypatch):
    """sep is in both calendars — re-fetching it would be wasted download."""
    _chip_set(tmp_path, ["x1"], ["feb", "may", "sep"])
    out, _ = _run(tmp_path, "summer_rainfall", monkeypatch)
    assert not (tmp_path / "x1" / "2023" / "feb.tif").exists(), "stale winter chip kept"
    assert not (tmp_path / "x1" / "2023" / "may.tif").exists(), "stale winter chip kept"
    assert (tmp_path / "x1" / "2023" / "sep.tif").exists(), "shared month should survive"
    assert set(out["month_label"]) == {"sep"}


def test_unchanged_zone_is_a_no_op(tmp_path, monkeypatch):
    """The common case must not re-download anything on a plain re-run."""
    _chip_set(tmp_path, ["x1"], ["feb", "may", "sep"])
    out, tried = _run(tmp_path, "winter_rainfall", monkeypatch)
    assert not tried, "re-run refetched an already-complete obs"
    for m in ["feb", "may", "sep"]:
        assert (tmp_path / "x1" / "2023" / f"{m}.tif").exists()
    assert len(out) == 3
