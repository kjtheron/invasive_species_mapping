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
OPT = dict(
    max_dates=1,
    min_coverage=0.98,
    min_valid_frac=0.85,
    screen_max_dates=24,
    read_pool=1,
)


class _FakeGbox:
    """Enough geobox for _month_chip: it only needs an extent in WGS84."""

    class _Ext:
        def to_crs(self, _crs):
            from shapely.geometry import box

            return type("_G", (), {"geom": box(19.0, -33.1, 19.02, -33.08)})()

    extent = _Ext()


class _FakeItem:
    """A STAC item as far as the screening path is concerned."""

    def __init__(self, name, cloud=10.0):
        self.name = name
        self.properties = {"eo:cloud_cover": cloud}

    def __repr__(self):
        return self.name


def _stub(monkeypatch, *, items=("i",), scored=None, arr=None, raises=None):
    """Wire the S2 layer to fixed answers so the decision logic is what is tested.

    Returns a dict the test can read back: ``read["best"]`` is the item list that
    reached ``load_composite``, ``read["screened"]`` how many reached the screen.
    """
    objs = [i if hasattr(i, "properties") else _FakeItem(str(i)) for i in items]
    read: dict = {}
    monkeypatch.setattr(chips.s2, "search_cell", lambda *a: tuple(objs))
    monkeypatch.setattr(chips.s2, "covering_items", lambda it, g, c: list(it))
    monkeypatch.setattr(chips.s2, "by_solar_day", lambda it: [[i] for i in it])

    def screen(cov, gbox, **k):
        read["screened"] = len(cov)
        return list(scored or [])

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
        res = _month_chip(19.01, -33.09, _FakeGbox(), WIN, ["B02", "B03"], **OPT)
        assert not isinstance(res, str), res
        assert res[1] == pytest.approx(1.0)
        assert read["best"] == ["b"], "did not read the clearest date"

    def test_no_items_is_reported_not_raised(self, monkeypatch):
        _stub(monkeypatch, items=())
        assert _month_chip(19.0, -33.0, _FakeGbox(), WIN, ["B02"], **OPT) == "no_items"

    def test_no_covering_date_is_reported(self, monkeypatch):
        _stub(monkeypatch, items=("a",))
        monkeypatch.setattr(chips.s2, "covering_items", lambda *a: [])
        assert _month_chip(19.0, -33.0, _FakeGbox(), WIN, ["B02"], **OPT) == "no_coverage"

    def test_too_cloudy_is_rejected_before_reading_bands(self, monkeypatch):
        """The saving: a month that cannot pass must cost SCL only, never 10 bands."""
        loaded = []
        _stub(monkeypatch, items=("a",), scored=[(0.10, ["a"])])
        monkeypatch.setattr(chips.s2, "load_composite", lambda *a, **k: loaded.append(1))
        res = _month_chip(19.0, -33.0, _FakeGbox(), WIN, ["B02"], **OPT)
        assert res.startswith("screen_clear=")
        assert loaded == [], "read the bands for a month the screen already rejected"

    def test_cloudy_centre_is_rejected(self, monkeypatch):
        arr = np.ones((2, 8, 8), dtype="float32")
        arr[0, 4, 4] = np.nan
        _stub(monkeypatch, items=("a",), scored=[(0.99, ["a"])], arr=arr)
        assert _month_chip(19.0, -33.0, _FakeGbox(), WIN, ["B02", "B03"], **OPT) == "center_cloudy"

    def test_low_valid_fraction_is_rejected(self, monkeypatch):
        arr = np.full((2, 8, 8), np.nan, dtype="float32")
        arr[:, 4, 4] = 1.0  # centre fine, everything else cloud
        _stub(monkeypatch, items=("a",), scored=[(0.99, ["a"])], arr=arr)
        res = _month_chip(19.0, -33.0, _FakeGbox(), WIN, ["B02", "B03"], **OPT)
        assert res.startswith("low_valid=")

    def test_read_failure_drops_the_cached_search_and_retries_once(self, monkeypatch):
        """An expired SAS token is the likely cause, and only a fresh, freshly
        signed search fixes it. The old code answered this by downloading whole
        tiles to disk instead."""
        dropped = []
        _stub(monkeypatch, items=("a",), scored=[(0.99, ["a"])],
              raises=OSError("HTTP 403"))
        monkeypatch.setattr(chips.s2, "drop_cell", lambda *a: dropped.append(a))
        assert _month_chip(19.0, -33.0, _FakeGbox(), WIN, ["B02"], **OPT) == "read_error"
        assert len(dropped) == 1, "must invalidate the cached search exactly once"

    def test_screen_pool_is_capped(self, monkeypatch):
        """A pathological window must not put 60 dates through the SCL screen."""
        read = _stub(
            monkeypatch,
            items=tuple(_FakeItem(f"i{i}", cloud=float(i)) for i in range(60)),
            scored=[(0.99, ["i0"])],
            arr=np.ones((1, 8, 8), dtype="float32"),
        )
        _month_chip(19.0, -33.0, _FakeGbox(), WIN, ["B02"], **{**OPT, "screen_max_dates": 5})
        assert read["screened"] == 5


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
