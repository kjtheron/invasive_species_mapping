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


# --- stopping a run, and picking it up again ---------------------------------


def _real_chip(path: Path, months: list[str], n_bands: int = 10, cloud_px: int = 0):
    """Write a chip the way _write_chip does: uint16, nodata 0, MONTHS tag."""
    import numpy as np
    import rasterio

    path.parent.mkdir(parents=True, exist_ok=True)
    a = np.full((len(months) * n_bands, 8, 8), 5000, dtype="uint16")
    if cloud_px:
        a[:, 0, :cloud_px] = 0  # masked pixels are 0 — the product's own no-data
    with rasterio.open(
        path, "w", driver="GTiff", height=8, width=8, count=a.shape[0],
        dtype="uint16", nodata=0, crs="EPSG:32734",
        transform=rasterio.transform.from_origin(230000, 6240000, 10, 10),
    ) as d:
        d.write(a)
        d.update_tags(MONTHS=",".join(months), YEAR=path.stem, SCALE_FACTOR="10000")


def test_orphan_chip_on_disk_is_adopted_not_redownloaded(tmp_path, monkeypatch):
    """A kill between a chip write and the next manifest flush leaves the file
    with no row. Re-fetching it costs ~29 MB over the wire; reading it costs
    ~700 KB off disk. It must be read."""
    _real_chip(tmp_path / "x1" / "2023.tif", ["feb", "may", "sep"])
    out, tried = _run(tmp_path, "winter_rainfall", monkeypatch)
    assert not tried, "re-downloaded a chip that was already on disk"
    assert set(out["obs_id"]) == {"x1"}
    assert out.iloc[0]["months"] == "feb,may,sep"
    assert out.iloc[0]["n_months"] == 3
    assert out.iloc[0]["utm_epsg"] == 32734


def test_adopted_chip_gets_its_real_valid_frac(tmp_path, monkeypatch):
    """valid_frac is recomputed from the pixels, not trusted from a tag, so a
    chip written by any earlier version can still be adopted."""
    _real_chip(tmp_path / "x1" / "2023.tif", ["feb", "may", "sep"], cloud_px=4)
    out, _ = _run(tmp_path, "winter_rainfall", monkeypatch)
    # 4 of 64 pixels zeroed in every band -> 60/64
    assert out.iloc[0]["valid_frac"] == 60 / 64


def test_orphan_on_the_wrong_calendar_is_not_adopted(tmp_path, monkeypatch):
    """A regrouped label's old chip must be re-cut, not silently kept."""
    _real_chip(tmp_path / "x1" / "2023.tif", ["feb", "may", "sep"])
    _, tried = _run(tmp_path, "summer_rainfall", monkeypatch)
    assert tried, "adopted a chip cut on the old calendar"
    assert all(t == {"jul", "sep", "dec"} for t in tried), "re-cut on the wrong calendar"


def test_unreadable_orphan_is_re_chipped(tmp_path, monkeypatch):
    """A truncated file must not be adopted on the strength of its name."""
    (tmp_path / "x1").mkdir(parents=True)
    (tmp_path / "x1" / "2023.tif").write_bytes(b"truncated")
    _, tried = _run(tmp_path, "winter_rainfall", monkeypatch)
    assert tried, "adopted an unreadable chip"


def test_interrupt_cancels_queued_work_and_banks_finished_rows(tmp_path, monkeypatch):
    """One Ctrl+C must stop the run AND keep what already finished.

    `with ThreadPoolExecutor(...)` calls shutdown(wait=True) on exit, which
    drains every queued future — all 93k of them on a full run — so a single
    Ctrl+C looked like a hang and kept chipping. A KeyboardInterrupt raised in a
    worker is a BaseException, so it passes through `run`'s `except Exception`
    and re-raises in the main thread, which is the path this exercises.
    """
    import pandas as pd

    from cmrv.ingest import chips as C

    import threading
    import time as _t

    n = {"i": 0}
    lk = threading.Lock()

    def flaky(row, months_cfg, bands, out_prefix, opt):
        with lk:
            n["i"] += 1
            i = n["i"]
        if i == 3:
            raise KeyboardInterrupt
        _t.sleep(0.05)  # slow enough that queued work is still queued at interrupt
        return {
            "obs_id": row.obs_id, "species": "pinus", "year": 2023, "block_id": 0,
            "lon": 18.5, "lat": -33.9, "chip_uri": f"{out_prefix}/{row.obs_id}/2023.tif",
            "months": "feb,may,sep", "n_months": 3, "valid_frac": 1.0, "utm_epsg": 32734,
        }

    monkeypatch.setattr(C, "_process_obs", flaky)
    labels = _labels([f"x{i}" for i in range(40)])
    try:
        C.extract_training_chips(
            labels=labels, months_by_zone=BY_ZONE, bands=["B02"],
            out_prefix=str(tmp_path), max_workers=1,
        )
        raise AssertionError("KeyboardInterrupt did not propagate — Ctrl+C would not stop the run")
    except KeyboardInterrupt:
        pass

    # shutdown(wait=False) is the point of the fix, so a straggler may still be
    # in flight. Give it a moment, or it logs after pytest closes the stream.
    _t.sleep(0.3)
    assert n["i"] < 40, f"kept working after the interrupt ({n['i']} of 40 started)"
    banked = pd.read_parquet(tmp_path / "manifest.parquet")
    assert len(banked) >= 2, "rows finished before the interrupt were not saved"


# --- losing as little as possible when a run is killed -----------------------


def _stub_process(monkeypatch, on_call=None):
    """Replace _process_obs with a fast stub that returns valid manifest rows."""
    from cmrv.ingest import chips as C

    n = {"i": 0}

    def fake(row, months_cfg, bands, out_prefix, opt):
        n["i"] += 1
        if on_call:
            on_call(n["i"])
        return {
            "obs_id": row.obs_id, "species": "pinus", "year": 2023, "block_id": 0,
            "lon": 18.5, "lat": -33.9, "chip_uri": f"{out_prefix}/{row.obs_id}/2023.tif",
            "months": "feb,may,sep", "n_months": 3, "valid_frac": 1.0, "utm_epsg": 32734,
        }

    monkeypatch.setattr(C, "_process_obs", fake)
    return n


def test_sigterm_saves_its_rows_like_ctrl_c(tmp_path, monkeypatch):
    """`pkill` must not be a trap.

    Python's default SIGTERM handler ends the process outright — no `finally`,
    so every manifest row since the last flush is lost. That is where the 487
    orphaned chips of 2026-09-06 came from: the advice was to stop the run with
    `pkill`. SIGTERM now raises KeyboardInterrupt so it reaches the same save
    path Ctrl+C does.
    """
    import os
    import signal
    import time as _t

    import pandas as pd

    from cmrv.ingest import chips as C

    def maybe_term(i):
        if i == 4:
            os.kill(os.getpid(), signal.SIGTERM)
        _t.sleep(0.02)

    _stub_process(monkeypatch, on_call=maybe_term)
    try:
        C.extract_training_chips(
            labels=_labels([f"s{i}" for i in range(40)]), months_by_zone=BY_ZONE,
            bands=["B02"], out_prefix=str(tmp_path), max_workers=1,
        )
        raise AssertionError("SIGTERM did not reach the save path — pkill still loses rows")
    except KeyboardInterrupt:
        pass
    _t.sleep(0.3)
    banked = pd.read_parquet(tmp_path / "manifest.parquet")
    # Rows are collected per completed square, so the exact count depends on how
    # far the run got. What matters is that finished work was banked at all —
    # under the old behaviour the manifest did not exist.
    assert len(banked) >= 1, "rows finished before the SIGTERM were not saved"
    assert len(banked) < 40, "SIGTERM did not stop the run"


def test_sigterm_handler_is_restored_afterwards(tmp_path, monkeypatch):
    """The guard must not leak: a library that changes global signal state and
    does not put it back is a trap for whatever runs next."""
    import signal

    from cmrv.ingest import chips as C

    before = signal.getsignal(signal.SIGTERM)
    _stub_process(monkeypatch)
    C.extract_training_chips(
        labels=_labels(["a", "b"]), months_by_zone=BY_ZONE, bands=["B02"],
        out_prefix=str(tmp_path), max_workers=1,
    )
    assert signal.getsignal(signal.SIGTERM) is before


def test_flush_is_bounded_by_the_clock_not_only_the_count(tmp_path, monkeypatch):
    """At ~0.15 chips/s a count of 500 alone leaves a 55-minute window in which a
    hard stop loses every row. The clock is what bounds it."""
    import pandas as pd

    from cmrv.ingest import chips as C

    monkeypatch.setattr(C, "FLUSH_EVERY", 10_000)  # count bound can never trip
    monkeypatch.setattr(C, "FLUSH_SECONDS", 0.0)  # clock bound trips every row

    writes = {"n": 0}
    real = C._flush

    def counting(existing, rows, uri):
        writes["n"] += 1
        return real(existing, rows, uri)

    monkeypatch.setattr(C, "_flush", counting)
    _stub_process(monkeypatch)
    C.extract_training_chips(
        labels=_labels([f"f{i}" for i in range(6)]), months_by_zone=BY_ZONE,
        bands=["B02"], out_prefix=str(tmp_path), max_workers=1,
    )
    assert writes["n"] > 2, "manifest was only written at the end — clock bound not firing"
    assert len(pd.read_parquet(tmp_path / "manifest.parquet")) == 6


def test_a_stop_does_not_start_new_chips_inside_a_running_square(tmp_path, monkeypatch):
    """Cancelling queued futures does not stop a RUNNING one, and a square holds
    many chips — a dense one holds hundreds.

    Without a stop flag the workers kept starting new chips for ~50 s after the
    interrupt, failing with "cannot schedule new futures after interpreter
    shutdown" and writing chips nobody banked. Observed on a live stop: 19 chips
    on disk, 0 manifest rows.
    """
    import threading
    import time as _t

    import pandas as pd

    from cmrv.ingest import chips as C

    n = {"i": 0}
    lk = threading.Lock()

    def fake(row, months_cfg, bands, out_prefix, opt):
        with lk:
            n["i"] += 1
            i = n["i"]
        if i == 3:
            raise KeyboardInterrupt
        _t.sleep(0.05)
        return {
            "obs_id": row.obs_id, "species": "pinus", "year": 2023, "block_id": 0,
            "lon": 18.5, "lat": -33.9, "chip_uri": f"{out_prefix}/{row.obs_id}/2023.tif",
            "months": "feb,may,sep", "n_months": 3, "valid_frac": 1.0, "utm_epsg": 32734,
        }

    monkeypatch.setattr(C, "_process_obs", fake)
    monkeypatch.setattr(C, "DRAIN_SECONDS", 2.0)
    # All 60 labels at one spot, so they land in ONE square and one worker takes
    # the lot — the case the stop flag exists for.
    labels = gpd.GeoDataFrame(
        {
            "obs_id": [f"q{i}" for i in range(60)],
            "species_normalized": ["pinus"] * 60,
            "block_id": [0] * 60,
            "event_date": ["2023-06-01"] * 60,
            "_zone": ["winter_rainfall"] * 60,
        },
        geometry=[Point(18.5, -33.9)] * 60,
        crs="EPSG:4326",
    )
    try:
        C.extract_training_chips(
            labels=labels, months_by_zone=BY_ZONE, bands=["B02"],
            out_prefix=str(tmp_path), max_workers=2,
        )
        raise AssertionError("KeyboardInterrupt did not propagate")
    except KeyboardInterrupt:
        pass
    _t.sleep(0.5)
    assert n["i"] < 60, f"kept starting chips after the stop ({n['i']} of 60)"
    banked = pd.read_parquet(tmp_path / "manifest.parquet")
    assert len(banked) >= 1, "the drain window banked nothing"
