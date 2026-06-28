"""Wall-to-wall inference — apply the frozen head densely → per-pixel class COG.

A lon/lat box → 3-month S2 composite → UniverSat **dense** token grid (per 64×64
window) → frozen head per token → per-pixel ``class_id`` → COG. Same encoder and
per-token representation as training (the center token the head learned), just
applied to *every* token. ``ponytail:`` non-overlapping windows — adjacent windows
may show faint seams; add overlap + blend if it matters.
"""

from __future__ import annotations

import numpy as np
from loguru import logger
from shapely.geometry import box as shp_box

from cmrv.embeddings.base import MONTH_DOY
from cmrv.embeddings.head import load_head, predict
from cmrv.embeddings.universat import UniverSatEmbedder
from cmrv.ingest.chips import _query_items, _stac_client, _stack_items
from cmrv.ingest.composite import _transform_from_da, monthly_median
from cmrv.io import load_config, write_cog

CHIP_PX = 64
RESOLUTION_M = 10
EPSG = 32734
NODATA = 255


def _composite_box(geom_wgs84, year, months_cfg, bands, cloud_cover_max):
    """3-month median composite for the box → ``(T, C, H, W)`` float32 + Affine transform."""
    client = _stac_client()
    arrs, transform = [], None
    for m in months_cfg:
        start, end = f"{year}-{m['start'][5:]}", f"{year}-{m['end'][5:]}"
        items = _query_items(client, geom_wgs84, start, end, cloud_cover_max=cloud_cover_max)
        if not items:
            raise ValueError(f"no S2 scenes for {start}/{end} — try another year/box")
        med = monthly_median(
            _stack_items(items, geom_wgs84, bands, resolution_m=RESOLUTION_M, epsg=EPSG)
        ).compute()  # (band, y, x)
        arrs.append(med.values.astype("float32"))
        transform = transform or _transform_from_da(med)
    return np.stack(arrs), transform


def infer_box(
    bbox: tuple[float, float, float, float],
    ckpt_path: str,
    out_uri: str,
    *,
    year: int = 2023,
    pipeline: str = "configs/pipeline.yaml",
    device: str = "cpu",
) -> str:
    """``(minlon, minlat, maxlon, maxlat)`` → per-pixel class-id COG."""
    cfg = load_config(pipeline)
    months, bands = cfg["months"], cfg["s2_bands"]
    cloud = cfg.get("cloud_cover_max", 95)

    stack, transform = _composite_box(shp_box(*bbox), year, months, bands, cloud)
    t, c, h, w = stack.shape
    logger.info("box composite: {} months × {} bands × {}×{} px", t, c, h, w)

    model, mu, sd, classes = load_head(ckpt_path)
    enc = UniverSatEmbedder(pool="center", output_grid=CHIP_PX, device=device, batch=4)
    dvec = np.array([[MONTH_DOY[m["label"]] for m in months]])

    out = np.full((h, w), NODATA, dtype=np.uint8)
    for r0 in range(0, h, CHIP_PX):
        for c0 in range(0, w, CHIP_PX):
            r1, c1 = min(r0 + CHIP_PX, h), min(c0 + CHIP_PX, w)
            win = np.zeros((t, c, CHIP_PX, CHIP_PX), dtype="float32")  # zero-pad edge windows
            win[:, :, : r1 - r0, : c1 - c0] = stack[:, :, r0:r1, c0:c1]
            win = np.nan_to_num(win / 10000.0, nan=0.0)[None]  # scale + fill, (1,T,C,64,64)
            grid = enc.embed_dense(win, dvec)[0]  # (64, 64, 768)
            cls = predict(model, mu, sd, classes, grid.reshape(-1, grid.shape[-1]))
            out[r0:r1, c0:c1] = cls.reshape(CHIP_PX, CHIP_PX)[: r1 - r0, : c1 - c0].astype(np.uint8)

    write_cog(out, transform, f"EPSG:{EPSG}", out_uri, dtype="uint8", nodata=NODATA)
    logger.success("wrote class map ({}×{}, {} classes) → {}", h, w, len(classes), out_uri)
    return out_uri
