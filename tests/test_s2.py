"""The S2 access layer: geometry screening, chip grids, validity gates, caching.

These guard the four things that decide whether a chip is correct and whether it
was cheap: the footprint screen, the snapped grid, the two validity gates, and
the search cache's expiry.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest
from shapely.geometry import box

from cmrv.ingest import s2


def _item(day, geom, cloud=10.0):
    return SimpleNamespace(
        datetime=datetime(2023, 6, day),
        geometry=geom.__geo_interface__,
        properties={"eo:cloud_cover": cloud},
    )


# ---------------------------------------------------------------------------
# covering_items — a bbox is not a footprint
# ---------------------------------------------------------------------------


class TestCoveringItems:
    def test_sliver_scene_is_dropped(self):
        """Both BBOXES contain the chip; only one FOOTPRINT does.

        This is the screen the old code never ran. It searched with `intersects`
        and then found out the scene missed by computing a median and getting NaN.
        """
        chip = box(0, 0, 1, 1)
        full = _item(1, box(-1, -1, 2, 2))
        sliver = _item(8, box(0.9, -1, 3, 2))
        assert s2.covering_items([full, sliver], chip, 0.98) == [full]

    def test_two_frames_of_one_pass_count_together(self):
        """A chip on a tile boundary is covered by two frames of the same pass.

        Screening per ITEM would drop both. Screening per DATE, over the union,
        keeps them — and groupby="solar_day" then mosaics them into one timestep.
        """
        chip = box(0, 0, 1, 1)
        a = _item(20, box(-1, -1, 0.6, 2))
        b = _item(20, box(0.4, -1, 2, 2))
        assert s2.covering_items([a], chip, 0.98) == []
        assert len(s2.covering_items([a, b], chip, 0.98)) == 2

    def test_empty_input_is_empty_output(self):
        assert s2.covering_items([], box(0, 0, 1, 1), 0.98) == []


def test_by_solar_day_groups_and_orders():
    a, b, c = _item(3, box(0, 0, 1, 1)), _item(1, box(0, 0, 1, 1)), _item(3, box(0, 0, 1, 1))
    groups = s2.by_solar_day([a, b, c])
    assert [len(g) for g in groups] == [1, 2]  # day 1 then day 3
    assert groups[0][0] is b


# ---------------------------------------------------------------------------
# Chip grid
# ---------------------------------------------------------------------------


class TestPointGeobox:
    def test_shape_and_snapping(self):
        g = s2.point_geobox(230_123.4, 6_239_987.6, 32734, 128, 10)
        assert tuple(g.shape) == (128, 128)
        # Snapped to the 10 m grid, so the chip lands on S2's own pixel edges and
        # the resampling is a no-op in the common case.
        assert g.boundingbox.left % 10 == 0
        assert g.boundingbox.bottom % 10 == 0

    def test_is_deterministic(self):
        """The same label must always yield the same pixel edges, so a re-chip is
        comparable and a resume cannot silently shift the grid."""
        a = s2.point_geobox(230_123.4, 6_239_987.6, 32734, 128, 10)
        b = s2.point_geobox(230_123.4, 6_239_987.6, 32734, 128, 10)
        assert a.boundingbox == b.boundingbox

    def test_point_stays_inside_the_chip(self):
        x, y = 230_123.4, 6_239_987.6
        bb = s2.point_geobox(x, y, 32734, 128, 10).boundingbox
        assert bb.left <= x <= bb.right and bb.bottom <= y <= bb.top

    def test_odd_chip_size_still_centres(self):
        bb = s2.point_geobox(100_000.0, 200_000.0, 32734, 64, 10).boundingbox
        assert bb.right - bb.left == 640


def test_bbox_geobox_covers_the_bounds():
    g = s2.bbox_geobox((100_001.0, 200_001.0, 100_099.0, 200_099.0), 32734, 10)
    bb = g.boundingbox
    assert bb.left <= 100_001.0 and bb.right >= 100_099.0
    assert bb.bottom <= 200_001.0 and bb.top >= 200_099.0


# ---------------------------------------------------------------------------
# Validity gates
# ---------------------------------------------------------------------------


def test_valid_fraction_needs_every_band():
    arr = np.ones((3, 10, 10), dtype="float32")
    arr[0, 0, :] = np.nan  # one band, one row
    assert s2.valid_fraction(arr) == pytest.approx(0.9)


def test_center_is_valid_guards_the_trained_pixel():
    """The head pools the CENTRE token, so a cloudy centre trains on a filled zero.

    With one date per month there is no median to repair it, so this is checked
    rather than assumed — a chip can be 95% clear and still have cloud dead centre.
    """
    arr = np.ones((3, 8, 8), dtype="float32")
    assert s2.center_is_valid(arr)
    arr[1, 4, 4] = np.nan
    assert not s2.center_is_valid(arr)
    assert s2.valid_fraction(arr) > 0.98  # ...and the fraction gate would have passed it


# ---------------------------------------------------------------------------
# Search cache
# ---------------------------------------------------------------------------


def test_cell_of_is_stable_within_a_cell():
    assert s2.cell_of(19.01, -33.01) == s2.cell_of(19.40, -33.40)
    assert s2.cell_of(19.01, -33.01) != s2.cell_of(19.60, -33.01)


class TestSearchCache:
    """The cache must dedupe AND expire.

    Planetary Computer signs asset hrefs with a SAS token that lives about an
    hour. A cache that never expired would hand a worker a dead URL, which GDAL
    reports as "not recognized as being in a supported file format" — the failure
    the old code answered by downloading whole tiles to disk.
    """

    def _stub(self, monkeypatch, counter):
        def fake(call, what, attempts=4):
            counter["n"] += 1
            return [SimpleNamespace(id=f"item{counter['n']}")]

        monkeypatch.setattr(s2, "stac_retry", fake)
        s2._SEARCH_CACHE.clear()

    def test_second_call_is_a_cache_hit(self, monkeypatch):
        n = {"n": 0}
        self._stub(monkeypatch, n)
        a = s2.search_cell((0, 0), "2023-06-01", "2023-06-30")
        b = s2.search_cell((0, 0), "2023-06-01", "2023-06-30")
        assert n["n"] == 1 and a == b

    def test_entry_expires(self, monkeypatch):
        n = {"n": 0}
        self._stub(monkeypatch, n)
        s2.search_cell((0, 0), "2023-06-01", "2023-06-30")
        key = ((0, 0), "2023-06-01", "2023-06-30")
        stamp, items = s2._SEARCH_CACHE[key]
        s2._SEARCH_CACHE[key] = (stamp - s2.SEARCH_TTL_S - 1, items)
        s2.search_cell((0, 0), "2023-06-01", "2023-06-30")
        assert n["n"] == 2, "a stale entry must be re-searched, which re-signs it"

    def test_drop_cell_forces_a_refetch(self, monkeypatch):
        n = {"n": 0}
        self._stub(monkeypatch, n)
        s2.search_cell((0, 0), "2023-06-01", "2023-06-30")
        s2.drop_cell((0, 0), "2023-06-01", "2023-06-30")
        s2.search_cell((0, 0), "2023-06-01", "2023-06-30")
        assert n["n"] == 2

    def test_different_windows_do_not_collide(self, monkeypatch):
        n = {"n": 0}
        self._stub(monkeypatch, n)
        s2.search_cell((0, 0), "2023-06-01", "2023-06-30")
        s2.search_cell((0, 0), "2023-07-01", "2023-07-31")
        assert n["n"] == 2


# ---------------------------------------------------------------------------
# Retry
# ---------------------------------------------------------------------------


class TestStacRetry:
    def test_recovers_from_a_transient_rate_limit(self, monkeypatch):
        monkeypatch.setattr(time, "sleep", lambda s: None)
        n = {"n": 0}

        def flaky():
            n["n"] += 1
            if n["n"] < 3:
                raise RuntimeError("You have exceeded a rate limit.")
            return "ok"

        assert s2.stac_retry(flaky, "test") == "ok"
        assert n["n"] == 3

    def test_reraises_after_the_last_attempt(self, monkeypatch):
        monkeypatch.setattr(time, "sleep", lambda s: None)
        n = {"n": 0}

        def dead():
            n["n"] += 1
            raise RuntimeError("nope")

        with pytest.raises(RuntimeError):
            s2.stac_retry(dead, "test")
        assert n["n"] == s2.STAC_ATTEMPTS

    def test_backoff_is_jittered(self, monkeypatch):
        """Every worker trips a rate limit at the same moment. Without jitter they
        all come back at the same moment too, and trip it again."""
        waits = []
        monkeypatch.setattr(time, "sleep", waits.append)
        for _ in range(2):
            with pytest.raises(RuntimeError):
                s2.stac_retry(lambda: (_ for _ in ()).throw(RuntimeError("x")), "t")
        first_attempt_waits = {waits[0], waits[s2.STAC_ATTEMPTS - 1]}
        assert len(first_attempt_waits) == 2, "identical backoffs — jitter is missing"


def test_bad_scl_matches_the_documented_set():
    """Changing this silently changes what counts as a valid pixel everywhere."""
    assert s2.BAD_SCL == frozenset({0, 1, 3, 8, 9, 10})
    # 2 (dark/topographic shadow) stays valid — over-masking risk on fynbos slopes.
    # 11 (snow) stays valid — rare in SA.
    assert 2 not in s2.BAD_SCL and 11 not in s2.BAD_SCL


def test_search_cache_is_thread_safe():
    """Workers share the cache; a torn dict would be a heisenbug at 3am."""
    s2._SEARCH_CACHE.clear()
    seen = []

    def worker(i):
        with s2._SEARCH_LOCK:
            s2._SEARCH_CACHE[((i, i), "a", "b")] = (time.monotonic(), (i,))
        seen.append(i)

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(s2._SEARCH_CACHE) == 50 and len(seen) == 50
    s2._SEARCH_CACHE.clear()


# ---------------------------------------------------------------------------
# Read failures must RAISE, or the retry above them is dead code
# ---------------------------------------------------------------------------


class TestReadFailuresAreNotSwallowed:
    """The 2026-09-06 regression.

    With ``fail_on_error=False`` odc-loader catches a failed read, logs
    "Ignoring read failure while reading", and fills the array with nodata.
    Nodata is 0 and 0 is in BAD_SCL, so every date scored 0.0 clear and the
    month was dropped as if it were cloudy — while the real cause, an expired
    SAS token, never reached the retry in ``_month_chip``. About 27 % of chips
    were lost that way, silently.
    """

    def _kwargs(self, monkeypatch, fn, gbox):
        seen = {}

        def fake(items, **kw):
            seen.update(kw)
            raise RuntimeError("stop here — only the kwargs matter")

        monkeypatch.setattr(s2, "odc_load", fake)
        try:
            fn(gbox)
        except RuntimeError:
            pass
        return seen

    def test_screening_raises_on_a_failed_read(self, monkeypatch):
        g = s2.point_geobox(230_000.0, 6_240_000.0, 32734, 128, 10)
        it = SimpleNamespace(datetime=datetime(2023, 6, 1), properties={})
        kw = self._kwargs(
            monkeypatch, lambda gb: s2.clear_fraction_per_date([it], gb), g
        )
        assert kw["fail_on_error"] is True, (
            "fail_on_error=False makes an expired token look like 100% cloud"
        )

    def test_load_raises_on_a_failed_read(self, monkeypatch):
        g = s2.point_geobox(230_000.0, 6_240_000.0, 32734, 128, 10)
        kw = self._kwargs(
            monkeypatch, lambda gb: s2.load_composite(["i"], gb, ["B02"]), g
        )
        assert kw["fail_on_error"] is True


def test_drop_cell_also_clears_the_sas_token_cache(monkeypatch):
    """Re-searching alone cannot fix an expired token.

    ``planetary_computer.get_token`` refreshes only when under 60 s remain, and
    it checks that at SIGN time. A token with minutes left is judged fine, baked
    into a freshly searched item's href, and dead by the time the read happens.
    Only clearing TOKEN_CACHE forces a real refresh.
    """
    from planetary_computer import sas

    s2._SEARCH_CACHE.clear()
    with s2._SEARCH_LOCK:
        s2._SEARCH_CACHE[((0, 0), "a", "b")] = (time.monotonic(), ("item",))
    sas.TOKEN_CACHE["https://example/token/acct/container"] = "stale"

    s2.drop_cell((0, 0), "a", "b")

    assert ((0, 0), "a", "b") not in s2._SEARCH_CACHE, "cached search survived"
    assert not sas.TOKEN_CACHE, "stale SAS token survived — the retry will fail again"
