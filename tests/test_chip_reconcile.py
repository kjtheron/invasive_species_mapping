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


def _chip_set(tmp_path: Path, obs_ids: list[str]) -> pd.DataFrame:
    """Write one fake chip file per obs and return the matching manifest."""
    rows = []
    for oid in obs_ids:
        d = tmp_path / oid / "2023"
        d.mkdir(parents=True)
        uri = d / "feb.tif"
        uri.write_bytes(b"not-a-real-tif")
        rows.append(
            {
                "obs_id": oid,
                "month_label": "feb",
                "year": 2023,
                "chip_uri": str(uri),
                "block_id": 0,
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_parquet(tmp_path / "manifest.parquet")
    return manifest


def _labels(obs_ids: list[str]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "obs_id": obs_ids,
            "block_id": [0] * len(obs_ids),
            "event_date": ["2023-06-01"] * len(obs_ids),
        },
        geometry=[Point(18.5 + i / 100, -33.9) for i in range(len(obs_ids))],
        crs="EPSG:4326",
    )


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
        blocks=gpd.GeoDataFrame({"block_id": [0]}, geometry=[Point(18.5, -33.9).buffer(1)]),
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
        blocks=gpd.GeoDataFrame({"block_id": [0]}, geometry=[Point(18.5, -33.9).buffer(1)]),
        months_cfg=MONTHS,
        bands=["B02"],
        out_prefix=str(tmp_path),
        reconcile=True,
    )
    assert not (tmp_path / "pine_1" / "2023" / "feb.tif").exists()
    assert set(out["obs_id"]) == {"euc_1"}
