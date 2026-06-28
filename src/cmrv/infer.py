"""Wall-to-wall inference — apply the frozen head densely → class / confidence / OOD COG.

A lon/lat box → 3-month S2 composite → UniverSat **dense** token grid (per 64×64
window) → frozen head per token → per-pixel ``class_id`` + confidence + Mahalanobis
OOD score → 3-band georeferenced COG. Same encoder and per-token representation as
training (the center token the head learned), applied to *every* token.

CRS is **not** hardcoded: it's taken from the Sentinel-2 data's native MGRS UTM zone
(``proj:epsg``), so each tile/box comes out in its own correct projection — mosaic to
a common CRS downstream. ``ponytail:`` non-overlapping windows; add overlap+blend for
seams if it matters.
"""

from __future__ import annotations

import numpy as np
from loguru import logger
from shapely.geometry import box as shp_box

from cmrv.embeddings.base import MONTH_DOY
from cmrv.embeddings.head import load_head, predict_dense
from cmrv.embeddings.universat import UniverSatEmbedder
from cmrv.ingest.chips import _query_items, _stac_client, _stack_items
from cmrv.ingest.composite import _transform_from_da, monthly_median
from cmrv.io import load_config, write_cog

CHIP_PX = 64
RESOLUTION_M = 10
NODATA = 255


def utm_epsg(lon: float, lat: float) -> int:
    """UTM EPSG for a lon/lat — matches Sentinel-2's native MGRS zone (fallback)."""
    return (32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1


def _composite_box(geom_wgs84, year, months_cfg, bands, cloud_cover_max):
    """3-month median composite → ``(T, C, H, W)``, transform, epsg (from the S2 data)."""
    client = _stac_client()
    epsg, arrs, transform = None, [], None
    for m in months_cfg:
        start, end = f"{year}-{m['start'][5:]}", f"{year}-{m['end'][5:]}"
        items = _query_items(client, geom_wgs84, start, end, cloud_cover_max=cloud_cover_max)
        if not items:
            raise ValueError(f"no S2 scenes for {start}/{end} — try another year/box")
        if epsg is None:  # native Sentinel-2 CRS, not hardcoded
            cx, cy = geom_wgs84.centroid.coords[0]
            epsg = int(items[0].properties.get("proj:epsg") or utm_epsg(cx, cy))
        med = monthly_median(
            _stack_items(items, geom_wgs84, bands, resolution_m=RESOLUTION_M, epsg=epsg)
        ).compute()  # (band, y, x)
        arrs.append(med.values.astype("float32"))
        transform = transform or _transform_from_da(med)
    return np.stack(arrs), transform, epsg


def infer_box(
    bbox: tuple[float, float, float, float],
    ckpt_path: str,
    out_uri: str,
    *,
    year: int = 2023,
    pipeline: str = "configs/pipeline.yaml",
    device: str = "cpu",
) -> str:
    """``(minlon, minlat, maxlon, maxlat)`` → 3-band (class, confidence, OOD) COG."""
    cfg = load_config(pipeline)
    months, bands = cfg["months"], cfg["s2_bands"]

    stack, transform, epsg = _composite_box(
        shp_box(*bbox), year, months, bands, cfg.get("cloud_cover_max", 95)
    )
    t, c, h, w = stack.shape
    logger.info("box composite: {} months × {} bands × {}×{} px @ EPSG:{}", t, c, h, w, epsg)

    model, mu, sd, classes, ood = load_head(ckpt_path)
    enc = UniverSatEmbedder(pool="center", output_grid=CHIP_PX, device=device, batch=4)
    dvec = np.array([[MONTH_DOY[m["label"]] for m in months]])

    cls_map = np.full((h, w), NODATA, dtype=np.uint8)
    conf_map = np.zeros((h, w), dtype=np.uint8)
    ood_map = np.zeros((h, w), dtype=np.uint8)
    for r0 in range(0, h, CHIP_PX):
        for c0 in range(0, w, CHIP_PX):
            r1, c1 = min(r0 + CHIP_PX, h), min(c0 + CHIP_PX, w)
            win = np.zeros((t, c, CHIP_PX, CHIP_PX), dtype="float32")  # zero-pad edge windows
            win[:, :, : r1 - r0, : c1 - c0] = stack[:, :, r0:r1, c0:c1]
            win = np.nan_to_num(win / 10000.0, nan=0.0)[None]  # scale + fill, (1,T,C,64,64)
            grid = enc.embed_dense(win, dvec)[0]  # (64, 64, D)
            cl, conf, oods = predict_dense(
                model, mu, sd, classes, ood, grid.reshape(-1, grid.shape[-1])
            )
            sl = (slice(r0, r1), slice(c0, c1))
            ch, cw = r1 - r0, c1 - c0
            cls_map[sl] = cl.reshape(CHIP_PX, CHIP_PX)[:ch, :cw]
            conf_map[sl] = (conf.reshape(CHIP_PX, CHIP_PX)[:ch, :cw] * 100).astype(np.uint8)
            ood_map[sl] = (oods.reshape(CHIP_PX, CHIP_PX)[:ch, :cw] * 100).astype(np.uint8)

    write_cog(
        np.stack([cls_map, conf_map, ood_map]),
        transform,
        f"EPSG:{epsg}",
        out_uri,
        dtype="uint8",
        nodata=NODATA,
    )
    logger.success(
        "wrote class/confidence/OOD COG ({}×{}, {} classes) @ EPSG:{} → {}",
        h,
        w,
        len(classes),
        epsg,
        out_uri,
    )
    return out_uri
