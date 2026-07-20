"""Memory-bounding of chip extraction: spatial sub-batching + scene cap.

The regression these guard: peak RSS used to scale with labels-per-block, so a
dense survey (MapWAPS) OOM'd a run that a sparse one (BioSCape) sailed through.
"""

from __future__ import annotations

import numpy as np

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


def test_scene_cap_keeps_the_least_cloudy(monkeypatch):
    clouds = [90.0, 5.0, 40.0, 1.0, 70.0]

    class _Search:
        def item_collection(self):
            return [_FakeItem(c) for c in clouds]

    class _Client:
        def search(self, **kw):
            return _Search()

    monkeypatch.setattr("cmrv.ingest.chips.pc.sign_inplace", lambda item: item)
    from shapely.geometry import box

    kept = _query_items(_Client(), box(18, -34, 19, -33), "2023-02-01", "2023-02-28", max_scenes=3)
    assert [i.properties["eo:cloud_cover"] for i in kept] == [1.0, 5.0, 40.0]


def test_no_cap_keeps_everything(monkeypatch):
    class _Search:
        def item_collection(self):
            return [_FakeItem(c) for c in (90.0, 5.0)]

    class _Client:
        def search(self, **kw):
            return _Search()

    monkeypatch.setattr("cmrv.ingest.chips.pc.sign_inplace", lambda item: item)
    from shapely.geometry import box

    assert len(_query_items(_Client(), box(18, -34, 19, -33), "a", "b", max_scenes=None)) == 2
