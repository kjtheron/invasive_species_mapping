"""Smoke test for cmrv.ingest.stats — manifest summary printer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cmrv.ingest.stats import chip_stats


def _write_manifest(tmp_path: Path) -> str:
    rows = []
    for obs_id, sp in enumerate(
        [
            *(["acacia mearnsii"] * 3),
            *(["pinus radiata"] * 2),
            *(["hakea sericea"] * 1),
        ]
    ):
        for month in ("jan", "apr", "aug", "oct"):
            rows.append(
                {
                    "obs_id": obs_id,
                    "species": sp,
                    "month_label": month,
                    "block_id": obs_id % 2,
                    "lon": 25.0 + obs_id * 0.001,
                    "lat": -30.0 - obs_id * 0.001,
                    "year": 2023,
                }
            )
    df = pd.DataFrame(rows)
    uri = str(tmp_path / "manifest.parquet")
    df.to_parquet(uri, index=False)
    return uri


def test_chip_stats_runs(tmp_path: Path, capsys):
    uri = _write_manifest(tmp_path)
    chip_stats(manifest_uri=uri, top_species=5, top_blocks=2)
    out = capsys.readouterr().out
    assert "Manifest overview" in out
    assert "acacia mearnsii" in out
    assert "Top 5 species" in out
    assert "Month-completeness" in out
    assert "4 month(s)" in out  # all obs are fully covered
