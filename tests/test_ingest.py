"""Chip extraction: month selection, the drop reasons, and the split.

The S2 access layer has its own file (test_s2.py); this covers the layer above it
— what a chip run does with what that layer returns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cmrv.ingest import chips
from cmrv.ingest.chips import _month_chip, _reconcile_manifest, temporal_windows

WIN = {"start": "2023-01-17", "end": "2023-03-15", "label": "feb"}
YEAR = 2023
OPT = dict(
    max_dates=1,
    min_coverage=0.98,
    min_valid_frac=0.85,
    screen_max_dates=24,
    read_pool=1,
    date_cell_m=2000.0,
)


class _FakeGbox:
    """Enough geobox for _month_chip: a WGS84 extent, plus bounds and CRS so the
    date-cache key can be built."""

    def __init__(self, left=230_000.0, bottom=6_240_000.0):
        self.boundingbox = type("_BB", (), {"left": left, "bottom": bottom})()
        self.crs = type("_C", (), {"epsg": 32734})()

    @property
    def extent(self):
        from shapely.geometry import box

        g = box(19.0, -33.1, 19.02, -33.08)
        return type("_E", (), {"to_crs": lambda _s, _c: type("_G", (), {"geom": g})()})()


def _chip(gbox=None, **over):
    """Call _month_chip with the test defaults."""
    return _month_chip(
        19.01, -33.09, gbox or _FakeGbox(), WIN, ["B02", "B03"], YEAR, **{**OPT, **over}
    )


@pytest.fixture(autouse=True)
def _clear_date_cache():
    """The remembered-date cache is module state — never let it leak between tests."""
    chips._DATE_CACHE.clear()
    yield
    chips._DATE_CACHE.clear()


class _FakeItem:
    """A STAC item as far as the screening path is concerned."""

    _day = 0

    def __init__(self, name, cloud=10.0):
        import datetime as _dt

        type(self)._day += 1
        self.name = name
        self.properties = {"eo:cloud_cover": cloud}
        # A distinct acquisition date per item, so the remembered-date logic has
        # something real to key on.
        self.datetime = _dt.datetime(2023, 2, 1 + (type(self)._day % 27))

    def __repr__(self):
        return self.name


def _stub(monkeypatch, *, items=("i",), scored=None, arr=None, raises=None):
    """Wire the S2 layer to fixed answers so the decision logic is what is tested.

    Returns a dict the test can read back: ``read["best"]`` is the item list that
    reached ``load_composite``, ``read["screened"]`` how many reached the screen.
    """
    objs = [i if hasattr(i, "properties") else _FakeItem(str(i)) for i in items]
    by_name = {o.name: o for o in objs}
    # `scored` is written with names for readability; swap in the real objects so
    # the code under test sees items, exactly as odc-stac would return them.
    scored_objs = [
        (f, [by_name.get(str(n), n) for n in grp]) for f, grp in (scored or [])
    ]
    read: dict = {}
    monkeypatch.setattr(chips.s2, "search_cell", lambda *a: tuple(objs))
    monkeypatch.setattr(chips.s2, "covering_items", lambda it, g, c: list(it))
    monkeypatch.setattr(chips.s2, "by_solar_day", lambda it: [[i] for i in it])

    def screen(cov, gbox, **k):
        read["screened"] = len(cov)
        read["screens"] = read.get("screens", 0) + 1
        return list(scored_objs)

    def load(best, gbox, bands, **k):
        read["best"] = best
        if raises:
            raise raises
        return type("_A", (), {"values": arr})()

    monkeypatch.setattr(chips.s2, "clear_fraction_per_date", screen)
    monkeypatch.setattr(chips.s2, "load_composite", load)
    return read


# ---------------------------------------------------------------------------
# temporal_windows
# ---------------------------------------------------------------------------


class TestTemporalWindows:
    def test_padding_widens_both_ends(self):
        """Padding is what gives the SCL screen more than one date to choose from."""
        (w,) = temporal_windows(
            2023, [{"start": "0000-02-01", "end": "0000-02-28", "label": "feb"}], padding_days=15
        )
        assert w["start"] == "2023-01-17" and w["end"] == "2023-03-15"

    def test_year_comes_from_the_label_not_the_template(self):
        (w,) = temporal_windows(
            2019, [{"start": "0000-07-01", "end": "0000-07-31", "label": "jul"}], padding_days=0
        )
        assert w["start"].startswith("2019") and w["label"] == "jul"


# ---------------------------------------------------------------------------
# _month_chip — every branch names WHY a chip was lost
# ---------------------------------------------------------------------------


class TestMonthChip:
    def test_clearest_date_wins(self, monkeypatch):
        """The whole point of the screen: pick by clear fraction over THIS chip,
        not by the scene-level cloud property over a 110 km tile.

        The cloud property here says "a" is the cleanest scene; SCL over the chip
        says "b" is. The chip must be built from "b".
        """
        arr = np.ones((2, 8, 8), dtype="float32")
        read = _stub(
            monkeypatch,
            items=(_FakeItem("a", cloud=1.0), _FakeItem("b", cloud=60.0), _FakeItem("c", 30.0)),
            scored=[(0.20, ["a"]), (0.99, ["b"]), (0.60, ["c"])],
            arr=arr,
        )
        res = _chip()
        assert not isinstance(res, str), res
        assert res[1] == pytest.approx(1.0)
        assert [i.name for i in read["best"]] == ["b"], "did not read the clearest date"

    def test_no_items_is_reported_not_raised(self, monkeypatch):
        _stub(monkeypatch, items=())
        assert _chip() == "no_items"

    def test_no_covering_date_is_reported(self, monkeypatch):
        _stub(monkeypatch, items=("a",))
        monkeypatch.setattr(chips.s2, "covering_items", lambda *a: [])
        assert _chip() == "no_coverage"

    def test_too_cloudy_is_rejected_before_reading_bands(self, monkeypatch):
        """The saving: a month that cannot pass must cost SCL only, never 10 bands."""
        loaded = []
        _stub(monkeypatch, items=("a",), scored=[(0.10, ["a"])])
        monkeypatch.setattr(chips.s2, "load_composite", lambda *a, **k: loaded.append(1))
        res = _chip()
        assert res.startswith("screen_clear=")
        assert loaded == [], "read the bands for a month the screen already rejected"

    def test_cloudy_centre_is_rejected(self, monkeypatch):
        arr = np.ones((2, 8, 8), dtype="float32")
        arr[0, 4, 4] = np.nan
        _stub(monkeypatch, items=("a",), scored=[(0.99, ["a"])], arr=arr)
        assert _chip() == "center_cloudy"

    def test_low_valid_fraction_is_rejected(self, monkeypatch):
        arr = np.full((2, 8, 8), np.nan, dtype="float32")
        arr[:, 4, 4] = 1.0  # centre fine, everything else cloud
        _stub(monkeypatch, items=("a",), scored=[(0.99, ["a"])], arr=arr)
        res = _chip()
        assert res.startswith("low_valid=")

    def test_read_failure_drops_the_cached_search_and_retries_once(self, monkeypatch):
        """An expired SAS token is the likely cause, and only a fresh, freshly
        signed search fixes it. The old code answered this by downloading whole
        tiles to disk instead."""
        dropped = []
        _stub(monkeypatch, items=("a",), scored=[(0.99, ["a"])],
              raises=OSError("HTTP 403"))
        monkeypatch.setattr(chips.s2, "drop_cell", lambda *a: dropped.append(a))
        assert _chip() == "read_error"
        # One re-sign, then give up. Pass 0 finds nothing remembered and falls
        # through without dropping; pass 1 fails and drops; pass 2 fails and
        # returns. Retrying past that would just burn bandwidth on a dead scene.
        assert len(dropped) == 1, "must re-sign exactly once before giving up"

    def test_screen_pool_is_capped(self, monkeypatch):
        """A pathological window must not put 60 dates through the SCL screen."""
        read = _stub(
            monkeypatch,
            items=tuple(_FakeItem(f"i{i}", cloud=float(i)) for i in range(60)),
            scored=[(0.99, ["i0"])],
            arr=np.ones((1, 8, 8), dtype="float32"),
        )
        _chip(screen_max_dates=5)
        assert read["screened"] == 5


# ---------------------------------------------------------------------------
# Remembered dates — the byte saving, and its safety net
# ---------------------------------------------------------------------------


class TestDateCache:
    """Screening is roughly half of what a chip costs to download, and every
    label in a square asks the same question. Measured on the real 93,348
    thinned labels: at 2 km, 69 % of labels have a neighbour that already
    answered it."""

    def test_neighbour_skips_the_screen_entirely(self, monkeypatch):
        arr = np.ones((2, 8, 8), dtype="float32")
        read = _stub(
            monkeypatch,
            items=(_FakeItem("a", 50.0), _FakeItem("b", 5.0)),
            scored=[(0.30, ["a"]), (0.99, ["b"])],
            arr=arr,
        )
        g = _FakeGbox()
        assert not isinstance(_chip(g), str)
        assert read["screens"] == 1, "first chip in a square must screen"

        # A second chip 100 m away — same square, so the answer is already known.
        near = _FakeGbox(left=g.boundingbox.left + 100, bottom=g.boundingbox.bottom + 100)
        assert not isinstance(_chip(near), str)
        assert read["screens"] == 1, "neighbour re-screened instead of reusing the date"
        assert [i.name for i in read["best"]] == ["b"], "reused the wrong date"

    def test_a_chip_in_another_square_screens_for_itself(self, monkeypatch):
        arr = np.ones((2, 8, 8), dtype="float32")
        read = _stub(monkeypatch, items=(_FakeItem("a"),), scored=[(0.99, ["a"])], arr=arr)
        g = _FakeGbox()
        _chip(g)
        far = _FakeGbox(left=g.boundingbox.left + 50_000, bottom=g.boundingbox.bottom)
        _chip(far)
        assert read["screens"] == 2, "a distant chip must not inherit the answer"

    def test_a_wrong_remembered_date_falls_back_to_a_full_screen(self, monkeypatch):
        """A cloud edge inside the square makes the remembered date wrong here.

        That must cost one wasted read, never a bad chip: the finished chip is
        still checked, and a failure drops through to a proper screen.
        """
        good = np.ones((2, 8, 8), dtype="float32")
        bad = np.full((2, 8, 8), np.nan, dtype="float32")
        chips._DATE_CACHE.clear()

        # Seed the square with a date whose chip will come back unusable here.
        read = _stub(
            monkeypatch,
            items=(_FakeItem("a"), _FakeItem("b")),
            scored=[(0.99, ["a"]), (0.95, ["b"])],
            arr=good,
        )
        g = _FakeGbox()
        _chip(g)
        assert read["screens"] == 1

        # Now the neighbour: the remembered date returns cloud, the screen does not.
        calls = {"n": 0}

        def load(best, gbox, bands, **k):
            calls["n"] += 1
            read["best"] = best
            return type("_A", (), {"values": bad if calls["n"] == 1 else good})()

        monkeypatch.setattr(chips.s2, "load_composite", load)
        near = _FakeGbox(left=g.boundingbox.left + 100, bottom=g.boundingbox.bottom)
        res = _chip(near)
        assert not isinstance(res, str), "fell back to nothing instead of screening"
        assert read["screens"] == 2, "did not re-screen after the remembered date failed"
        assert calls["n"] == 2, "expected one wasted read, then a screened one"

    def test_zero_cell_size_disables_sharing_without_dividing_by_zero(self, monkeypatch):
        """`date_cell_m: 0` is the documented off switch. It used to raise inside
        a worker, and the blanket `except` turned that into a silent zero-chip
        run."""
        arr = np.ones((2, 8, 8), dtype="float32")
        read = _stub(monkeypatch, items=(_FakeItem("a"),), scored=[(0.99, ["a"])], arr=arr)
        g = _FakeGbox()
        assert not isinstance(_chip(g, date_cell_m=0.0), str)
        assert not isinstance(_chip(g, date_cell_m=0.0), str)
        assert read["screens"] == 2, "sharing was not actually off"
        assert not chips._DATE_CACHE, "remembered a date while disabled"

    def test_the_key_separates_months_years_and_utm_zones(self):
        g = _FakeGbox()
        base = chips._date_key(g, "feb", 2023, 2000.0)
        assert chips._date_key(g, "may", 2023, 2000.0) != base
        assert chips._date_key(g, "feb", 2022, 2000.0) != base
        other_zone = _FakeGbox()
        other_zone.crs = type("_C", (), {"epsg": 32735})()
        assert chips._date_key(other_zone, "feb", 2023, 2000.0) != base, (
            "two chips in different UTM zones must never share an answer"
        )


def test_thin_labels_order_independent_and_stable() -> None:
    """Thinning survivor is the smallest obs_id per cell — independent of input order."""
    import geopandas as gpd
    from shapely.geometry import Point

    from cmrv.ingest.chips import thin_labels

    pts = gpd.GeoDataFrame(
        {"obs_id": [f"obs{i}" for i in range(8)], "species_normalized": ["Pinus"] * 8},
        geometry=[Point(19.0 + 1e-5 * i, -32.0) for i in range(8)],
        crs="EPSG:4326",
    )
    a = set(thin_labels(pts, thin_m=20.0)["obs_id"])
    b = set(thin_labels(pts.iloc[::-1].reset_index(drop=True), thin_m=20.0)["obs_id"])
    assert a == b  # independent of input row order
    assert "obs0" in a  # survivor = smallest obs_id in the cell
    assert len(a) < 8  # near-duplicates collapsed


def test_reconcile_manifest_prunes_thinned_out_obs(tmp_path):
    """Stale obs (not in the thinned set) lose their chips + manifest rows."""
    (tmp_path / "keep").mkdir()
    (tmp_path / "drop").mkdir()
    ck = tmp_path / "keep" / "2023.tif"
    cd = tmp_path / "drop" / "2023.tif"
    ck.write_bytes(b"x")
    cd.write_bytes(b"x")
    man = pd.DataFrame(
        {"obs_id": ["keep", "drop"], "chip_uri": [str(ck), str(cd)], "months": ["feb", "feb"]}
    )
    muri = str(tmp_path / "manifest.parquet")

    kept = _reconcile_manifest(man, {"keep"}, muri)

    assert set(kept["obs_id"]) == {"keep"}
    assert ck.exists() and not cd.exists()  # stale chip file deleted
    assert not (tmp_path / "drop").exists()  # emptied obs dir removed
    assert set(pd.read_parquet(muri)["obs_id"]) == {"keep"}  # manifest rewritten


def test_iterative_stratification_spreads_class_across_folds():
    """A class spanning many blocks must reach all 3 folds, not pile into train."""
    import geopandas as gpd
    from shapely.geometry import Point, box

    from cmrv.ingest.chips import stratified_spatial_split

    blocks = gpd.GeoDataFrame(
        {"block_id": range(20)},
        geometry=[box(i, 0, i + 1, 1) for i in range(20)],
        crs="EPSG:4326",
    )
    rows = [{"obs_id": f"c{i}", "class_id": 0, "geometry": Point(i + 0.5, 0.5)} for i in range(20)]
    rows += [{"obs_id": f"r{i}", "class_id": 1, "geometry": Point(i + 0.5, 0.5)} for i in range(3)]
    labels = gpd.GeoDataFrame(rows, crs="EPSG:4326")

    out, b2f = stratified_spatial_split(labels, blocks, species_col="class_id", seed=0)

    assert set(out.loc[out["class_id"] == 0, "fold"]) == {"train", "val", "test"}
    assert set(b2f.values()) <= {"train", "val", "test"}


def test_cover_gate_drops_only_sparse_or_unrecorded_alien_obs():
    """The pure-pixel gate removes low- or no-cover alien points; natives always pass."""
    import numpy as np
    import pandas as pd

    from cmrv.ingest.chips import cover_gate

    labels = pd.DataFrame(
        {
            "obs_id": ["dense", "edge", "sparse", "unrecorded", "native", "grass", "no_rank"],
            "taxon_rank": ["genus", "species", "species", "genus", "biome", "landcover", None],
            "cover_pct": [100.0, 50.0, 1.0, np.nan, np.nan, np.nan, 5.0],
        }
    )
    obs = pd.DataFrame({"obs_id": [*labels["obs_id"], "not_in_store"]})
    kept = cover_gate(obs, labels, 50)["obs_id"].tolist()
    assert kept == ["dense", "edge", "native", "grass", "no_rank", "not_in_store"]
