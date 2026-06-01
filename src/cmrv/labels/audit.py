"""Viz-side raster sanity check for the optional ``labels-fuse`` output.

The training pipeline does NOT consume fused label rasters — chips +
manifest are the source of truth (see ``cmrv.chips.stats``).  This module
exists only as an opt-in sanity check on the viz COGs produced by
``cmrv labels-fuse`` (QGIS / Streamlit overlays).

If you don't run ``labels-fuse``, you don't need this module.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from cmrv.io import open_raster, write_parquet_df
from cmrv.labels.classmap import build_lookup

NODATA: int = 255


def make_audit_run_id(when: dt.datetime | None = None) -> str:
    when = when or dt.datetime.now(tz=dt.UTC)
    return f"audit_{when.strftime('%Y%m%dT%H%M%SZ')}"


def _list_label_rasters(raster_prefix: str) -> list[str]:
    """Discover ``tile_id=*/label.tif`` under ``raster_prefix``."""
    if raster_prefix.startswith("gs://"):
        import fsspec

        fs = fsspec.filesystem("gs")
        path = raster_prefix[len("gs://") :].rstrip("/")
        hits = fs.glob(f"{path}/tile_id=*/label.tif")
        return [f"gs://{h}" for h in hits]
    base = Path(raster_prefix)
    return sorted(str(p) for p in base.glob("tile_id=*/label.tif"))


def audit_post_fuse(
    raster_prefix: str = "gs://ism-data/labels",
    schema_path: str | Path = "configs/labels_schema.yaml",
    class_map_name: str = "upper_berg_12",
    out_dir: str | None = None,
    nodata: int = NODATA,
) -> dict[str, pd.DataFrame]:
    """Histogram class_ids per fused-tile COG; flag empty + unknown classes.

    Optional viz sanity check — only meaningful if ``cmrv labels-fuse`` has
    been run.  Anomalies surfaced:
      * ``unknown_class_id`` — raster pixel value not in the active class_map
        (corruption).
      * ``empty_class`` — class declared in the class_map has zero pixels in
        a tile (silent gap if you were doing dense training).

    Returns ``{"hist": ..., "anomalies": ...}``.
    """
    classmap = build_lookup(schema_path, class_map_name)
    expected = classmap.class_ids

    rasters = _list_label_rasters(raster_prefix)
    if not rasters:
        raise FileNotFoundError(
            f"no tile_id=*/label.tif rasters under {raster_prefix}"
        )

    rows: list[dict[str, Any]] = []
    anomalies: list[dict[str, Any]] = []

    for uri in rasters:
        try:
            tile_id = int(uri.rsplit("tile_id=", 1)[1].split("/", 1)[0])
        except (IndexError, ValueError):
            tile_id = -1

        try:
            with open_raster(uri) as src:
                arr = src.read(1)
        except Exception as e:
            logger.warning("could not read {}: {}", uri, e)
            anomalies.append(
                {"tile_id": tile_id, "uri": uri, "kind": "read_error", "detail": str(e)}
            )
            continue

        labeled = arr[arr != nodata]
        unique, counts = np.unique(labeled, return_counts=True)
        present = {int(u): int(c) for u, c in zip(unique, counts, strict=True)}

        for cid, n in present.items():
            rows.append({"tile_id": tile_id, "class_id": cid, "n_pixels": n})

        for cid in set(present) - expected:
            anomalies.append(
                {
                    "tile_id": tile_id,
                    "uri": uri,
                    "kind": "unknown_class_id",
                    "class_id": cid,
                    "n_pixels": present[cid],
                }
            )
        for cid in expected - set(present):
            anomalies.append(
                {
                    "tile_id": tile_id,
                    "uri": uri,
                    "kind": "empty_class",
                    "class_id": cid,
                    "n_pixels": 0,
                }
            )

    hist = pd.DataFrame(rows)
    anomaly_df = pd.DataFrame(anomalies)

    if anomaly_df.empty:
        n_unknown = 0
        n_empty = 0
    else:
        n_unknown = int((anomaly_df["kind"] == "unknown_class_id").sum())
        n_empty = int((anomaly_df["kind"] == "empty_class").sum())
    logger.info(
        "post-fuse audit ({}): {} tiles, {} unknown-class, {} empty-class",
        class_map_name,
        len(rasters),
        n_unknown,
        n_empty,
    )

    if out_dir:
        write_parquet_df(hist, f"{out_dir}/post_fuse_class_hist.parquet")
        write_parquet_df(anomaly_df, f"{out_dir}/post_fuse_anomalies.parquet")
        logger.success("post-fuse audit written → {}", out_dir)

    return {"hist": hist, "anomalies": anomaly_df}
