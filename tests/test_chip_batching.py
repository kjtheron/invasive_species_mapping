"""Memory-bounding of chip extraction: spatial sub-batching + scene cap.

The regression these guard: peak RSS used to scale with labels-per-block, so a
dense survey (MapWAPS) OOM'd a run that a sparse one (BioSCape) sailed through.
"""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import box

from cmrv.ingest.chips import SUBCELL_M, _points_bbox_wgs84, _query_items, _subcell_batches


def test_batches_partition_every_point_exactly_once():
    pts = [(float(x), float(y)) for x in range(0, 12000, 500) for y in range(0, 12000, 500)]
    batches = _subcell_batches(pts)
    seen = sorted(i for b in batches for i in b)
    assert seen == list(range(len(pts))), "every point must land in exactly one batch"


def test_batch_extent_is_bounded_by_cell_regardless_of_density():
    """The point of the fix: bbox is set by SUBCELL_M, not by how many labels there are."""
    dense = [(np.random.uniform(0, 40000), np.random.uniform(0, 40000)) for _ in range(5000)]
    for idx in _subcell_batches(dense):
        xs = [dense[i][0] for i in idx]
        ys = [dense[i][1] for i in idx]
        assert max(xs) - min(xs) <= SUBCELL_M
        assert max(ys) - min(ys) <= SUBCELL_M


def test_single_point_batch_still_works():
    assert _subcell_batches([(1000.0, 2000.0)]) == [[0]]


def test_points_bbox_is_padded_and_in_wgs84():
    bbox = _points_bbox_wgs84([(300000.0, 6200000.0)], 32734)
    minx, miny, maxx, maxy = bbox.bounds
    assert -180 <= minx < maxx <= 180
    assert 0 < maxy - miny < 0.01  # ~640 m of padding, not degrees of it


class _FakeItem:
    def __init__(self, cloud):
        self.properties = {"eo:cloud_cover": cloud}
        self.assets = {}


def _fake_client(clouds):
    items = [_FakeItem(c) for c in clouds]
    search = type("_Search", (), {"item_collection": lambda self: items})()
    return type("_Client", (), {"search": lambda self, **kw: search})()


@pytest.mark.parametrize(
    "max_scenes,expected",
    [
        (3, [1.0, 5.0, 40.0]),  # keeps the N least cloudy, in that order
        (None, [90.0, 5.0, 40.0, 1.0, 70.0]),  # no cap → untouched
        (99, [90.0, 5.0, 40.0, 1.0, 70.0]),  # cap above the pool → untouched
    ],
)
def test_scene_cap(monkeypatch, max_scenes, expected):
    monkeypatch.setattr("cmrv.ingest.chips.pc.sign_inplace", lambda item: item)
    kept = _query_items(
        _fake_client([90.0, 5.0, 40.0, 1.0, 70.0]),
        box(18, -34, 19, -33),
        "2023-02-01",
        "2023-02-28",
        max_scenes=max_scenes,
    )
    assert [i.properties["eo:cloud_cover"] for i in kept] == expected


# --- SAS token refresh -------------------------------------------------------


def _signed_item(token: str):
    """A pystac Item whose asset href already carries a SAS token."""
    import pystac

    item = pystac.Item("i", None, None, __import__("datetime").datetime(2023, 2, 1), {})
    item.assets["B02"] = pystac.Asset(
        href=f"https://x.blob.core.windows.net/c/b.tif?st=2026-01-01T00%3A00%3A00Z&se=2026-01-02T00%3A00%3A00Z&sp=rl&sig={token}"
    )
    return item


def test_resign_replaces_an_expired_token(monkeypatch):
    """The bug: sign_url short-circuits on an already-signed href, so without the
    strip the stale token survives and reads fail once it expires."""
    import datetime as dt

    from planetary_computer.sas import SASToken

    fresh = SASToken(
        token="sig=NEW", expiry=dt.datetime(2099, 1, 1, tzinfo=dt.UTC)
    )
    monkeypatch.setattr("planetary_computer.sas.get_token", lambda *a, **k: fresh)

    from cmrv.ingest.chips import _resign

    item = _signed_item("OLD")
    _resign([item])
    href = item.assets["B02"].href
    assert "OLD" not in href, "stale token survived — strip-before-sign is broken"
    assert "NEW" in href


def test_resign_leaves_local_fallback_hrefs_alone():
    """The download fallback rewrites hrefs to local paths — signing must no-op."""
    import pystac

    from cmrv.ingest.chips import _resign

    item = pystac.Item("i", None, None, __import__("datetime").datetime(2023, 2, 1), {})
    item.assets["B02"] = pystac.Asset(href="/tmp/s2dl_x/scene_B02.tif")
    _resign([item])
    assert item.assets["B02"].href == "/tmp/s2dl_x/scene_B02.tif"
