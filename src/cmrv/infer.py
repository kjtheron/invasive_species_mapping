"""Wall-to-wall inference — apply the frozen head densely → class / confidence / OOD COG.

A lon/lat box → 3-month S2 composite → UniverSat **dense** token grid (per 64×64
window) → frozen head per token → per-pixel ``class_id`` + confidence + Mahalanobis
OOD score → 3-band georeferenced COG. Same encoder and per-token representation as
training (the center token the head learned), applied to *every* token.

Windows overlap 50% and are blended with a Hann edge-taper, so edge tokens (truncated
context) don't leave 640 m seams. CRS is **not** hardcoded: it's taken from the
Sentinel-2 data's native MGRS UTM zone (``proj:epsg``), so each box comes out in its
own correct projection — mosaic to a common CRS downstream.
"""

from __future__ import annotations

import numpy as np
from loguru import logger
from shapely.geometry import box as shp_box

from cmrv.embeddings.base import MONTH_DOY
from cmrv.embeddings.head import load_head, predict_probs
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


def _starts(n: int, win: int, stride: int) -> list[int]:
    """Window start positions tiling ``[0, n)`` with overlap; last flush to the edge."""
    s = list(range(0, max(n - win, 0) + 1, stride))
    if n > win and s[-1] != n - win:
        s.append(n - win)
    return s


def _d4_ops(tta: bool):
    """Dihedral (flip × 90° rotation) augmentations as ``(fwd, inv)`` pairs.

    ``fwd`` transforms an input's spatial last-2 axes; ``inv`` applies the inverse to a
    prediction's spatial first-2 axes, bringing probs back to the original frame. Land
    cover is flip/rotation-invariant, so averaging the 8 views de-noises the prediction.
    ``tta=False`` → identity only.
    """
    if not tta:
        return [(lambda x: x, lambda p: p)]
    ops = []
    for flip in (False, True):
        for k in range(4):

            def fwd(x, k=k, flip=flip):
                return np.rot90(np.flip(x, axis=-1) if flip else x, k, axes=(-2, -1))

            def inv(p, k=k, flip=flip):
                p = np.rot90(p, -k, axes=(0, 1))
                return np.flip(p, axis=1) if flip else p

            ops.append((fwd, inv))
    return ops


def infer_box(
    bbox: tuple[float, float, float, float],
    ckpt_path: str,
    out_uri: str,
    *,
    year: int = 2023,
    pipeline: str = "configs/pipeline.yaml",
    device: str = "cpu",
    tta: bool = False,
) -> str:
    """``(minlon, minlat, maxlon, maxlat)`` → 3-band (class, confidence, OOD) COG.

    Overlapping 64 px windows with a Hann edge-taper: each output pixel is the blended
    average of every window covering it, weighted toward window centres — so edge tokens
    (truncated receptive field) contribute ~nothing and there are no 640 m window seams.
    ``tta`` averages 8 dihedral (flip/rotation) views per window — more robust, 8× slower.
    """
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
    k = len(classes)

    # reflect-pad so every output pixel can sit at a window centre; 50% overlap.
    pad = stride = CHIP_PX // 2
    padded = np.nan_to_num(
        np.pad(stack, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="reflect") / 10000.0, nan=0.0
    )
    taper = np.outer(np.hanning(CHIP_PX), np.hanning(CHIP_PX)).astype("float32")  # ~0 at edges
    ops = _d4_ops(tta)
    prob_acc = np.zeros((h, w, k), "float32")
    ood_acc = np.zeros((h, w), "float32")
    wsum = np.zeros((h, w), "float32")

    for r0 in _starts(h + 2 * pad, CHIP_PX, stride):
        for c0 in _starts(w + 2 * pad, CHIP_PX, stride):
            win = padded[None, :, :, r0 : r0 + CHIP_PX, c0 : c0 + CHIP_PX]
            probs = np.zeros((CHIP_PX, CHIP_PX, k), "float32")
            oods = np.zeros((CHIP_PX, CHIP_PX), "float32")
            for fwd, inv in ops:  # test-time augmentation: average dihedral views
                grid = enc.embed_dense(np.ascontiguousarray(fwd(win)), dvec)[0]
                pr, od = predict_probs(model, mu, sd, ood, grid.reshape(-1, grid.shape[-1]))
                probs += inv(pr.reshape(CHIP_PX, CHIP_PX, k))
                oods += inv(od.reshape(CHIP_PX, CHIP_PX, 1))[..., 0]
            probs /= len(ops)
            oods /= len(ops)
            ar0, ac0 = r0 - pad, c0 - pad  # window footprint in (unpadded) output coords
            rr0, rr1 = max(ar0, 0), min(ar0 + CHIP_PX, h)
            cc0, cc1 = max(ac0, 0), min(ac0 + CHIP_PX, w)
            wr, wc, hh, ww = rr0 - ar0, cc0 - ac0, rr1 - rr0, cc1 - cc0
            tw = taper[wr : wr + hh, wc : wc + ww]
            prob_acc[rr0:rr1, cc0:cc1] += probs[wr : wr + hh, wc : wc + ww] * tw[..., None]
            ood_acc[rr0:rr1, cc0:cc1] += oods[wr : wr + hh, wc : wc + ww] * tw
            wsum[rr0:rr1, cc0:cc1] += tw

    wsum = np.maximum(wsum, 1e-6)
    prob = prob_acc / wsum[..., None]
    cls_map = classes[prob.argmax(2)].astype(np.uint8)
    conf_map = (prob.max(2) * 100).astype(np.uint8)
    ood_map = (np.clip(ood_acc / wsum, 0, 1) * 100).astype(np.uint8)

    write_cog(
        np.stack([cls_map, conf_map, ood_map]),
        transform,
        f"EPSG:{epsg}",
        out_uri,
        dtype="uint8",
        nodata=NODATA,
    )
    logger.success(
        "wrote class/confidence/OOD COG ({}×{}, {} classes, overlap-blended) @ EPSG:{} → {}",
        h,
        w,
        k,
        epsg,
        out_uri,
    )
    return out_uri
