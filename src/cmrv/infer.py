"""Wall-to-wall inference — apply the frozen head densely → class / confidence / OOD COG.

A lon/lat box → 3-month S2 composite → UniverSat **dense** token grid (per 64×64
window) → frozen head per token → per-pixel ``class_id`` + confidence + Mahalanobis
OOD score → 3-band georeferenced COG. Same encoder and per-token representation as
training (the center token the head learned), applied to *every* token.

Windows overlap 25% and blend with a raised-cosine (constant-overlap-add) taper, so
edge tokens (truncated context) don't leave 640 m seams. Each window's TTA views run in
one batched forward. CRS is **not** hardcoded: it's taken from the Sentinel-2 data's
native MGRS UTM zone (``proj:epsg``), so each box comes out in its own correct
projection — mosaic to a common CRS downstream.
"""

from __future__ import annotations

import numpy as np
from loguru import logger
from rasterio.crs import CRS as RioCRS
from rasterio.transform import array_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject
from shapely.geometry import box as shp_box

from cmrv.aoi import SA_ALBERS, utm_epsg
from cmrv.embeddings.base import MONTH_DOY
from cmrv.embeddings.head import load_head, predict_probs
from cmrv.embeddings.universat import UniverSatEmbedder
from cmrv.ingest.chips import _query_items, _stac_client, _stack_items
from cmrv.ingest.composite import _transform_from_da, monthly_median
from cmrv.io import load_config, write_cog

CHIP_PX = 64
RESOLUTION_M = 10
NODATA = 255


def _composite_box(geom_wgs84, year, months_cfg, bands, cloud_cover_max, max_scenes=None):
    """3-month median composite → ``(T, C, H, W)``, transform, epsg (from the S2 data)."""
    client = _stac_client()
    epsg, arrs, transform = None, [], None
    for m in months_cfg:
        start, end = f"{year}-{m['start'][5:]}", f"{year}-{m['end'][5:]}"
        items = _query_items(
            client, geom_wgs84, start, end, cloud_cover_max=cloud_cover_max, max_scenes=max_scenes
        )
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


def _d4_ops(n_views: int):
    """The first ``n_views`` dihedral augmentations as ``(fwd, inv)`` pairs.

    Ordered so **1 = identity** (no TTA), **4 = the four 90° rotations**, **8 = the full
    D4 group** (rotations × flips). ``fwd`` transforms an input's spatial last-2 axes;
    ``inv`` applies the inverse to a prediction's first-2 axes, bringing probs back to the
    original frame. Land cover is flip/rotation-invariant, so averaging the views
    (soft-voting) de-noises the prediction.
    """
    ops = []
    for flip in (False, True):
        for k in range(4):

            def fwd(x, k=k, flip=flip):
                return np.rot90(np.flip(x, axis=-1) if flip else x, k, axes=(-2, -1))

            def inv(p, k=k, flip=flip):
                p = np.rot90(p, -k, axes=(0, 1))
                return np.flip(p, axis=1) if flip else p

            ops.append((fwd, inv))
    return ops[: max(1, min(n_views, 8))]


def _reproject_triplet(arr, src_transform, src_epsg: int, dst_crs: str):
    """Warp a ``(3, H, W)`` uint8 class/conf/OOD stack from its native S2 UTM zone to
    ``dst_crs`` (nearest-neighbour — the class band is categorical). Returns
    ``(arr, transform, crs)``. Keeps embedding in the native zone; only the map is warped."""
    src = RioCRS.from_epsg(src_epsg)
    dst = RioCRS.from_user_input(dst_crs)
    h, w = arr.shape[1], arr.shape[2]
    bounds = array_bounds(h, w, src_transform)
    dst_transform, dw, dh = calculate_default_transform(src, dst, w, h, *bounds)
    out = np.full((arr.shape[0], dh, dw), NODATA, dtype=arr.dtype)
    for b in range(arr.shape[0]):
        reproject(
            source=arr[b],
            destination=out[b],
            src_transform=src_transform,
            src_crs=src,
            dst_transform=dst_transform,
            dst_crs=dst,
            src_nodata=NODATA,
            dst_nodata=NODATA,
            resampling=Resampling.nearest,
        )
    return out, dst_transform, dst


def _taper(n: int, ramp: int) -> np.ndarray:
    """2-D raised-cosine blend window: flat 1.0 centre, cosine ramp over ``ramp`` px/edge.

    With ``ramp = overlap`` the adjacent windows' ramps sum to 1 (constant-overlap-add),
    so overlapping predictions blend seamlessly at any stride.
    """
    w = np.ones(n, dtype="float32")
    if ramp > 0:
        r = 0.5 * (1 - np.cos(np.pi * (np.arange(ramp) + 0.5) / ramp))
        w[:ramp], w[-ramp:] = r, r[::-1]
    return np.outer(w, w).astype("float32")


def infer_box(
    bbox: tuple[float, float, float, float],
    ckpt_path: str,
    out_uri: str = "data/outputs/infer.tif",
    *,
    year: int = 2023,
    pipeline: str = "configs/pipeline.yaml",
    device: str = "cpu",
    tta_views: int = 1,
    out_crs: str | None = SA_ALBERS,
):
    """``(minlon, minlat, maxlon, maxlat)`` → writes a COG, returns ``(bands, transform, crs)``.

    Bands are class_id / confidence / OOD. The S2 composite + embedding run in the box's
    **native** S2 UTM zone (no cross-zone resampling — same convention as training chips);
    the output map is then warped to ``out_crs`` (default national :data:`SA_ALBERS`, so
    tiles mosaic into one grid) — pass ``None`` to keep the native zone. Always saves the
    COG to ``out_uri``. Overlapping 64 px windows (25%) blend with a constant-overlap-add
    taper; ``tta_views`` soft-averages augmented views per window (1/4/8), one batched forward.
    """
    cfg = load_config(pipeline)
    months, bands = cfg["months"], cfg["s2_bands"]
    stack, transform, epsg = _composite_box(
        shp_box(*bbox),
        year,
        months,
        bands,
        cfg.get("cloud_cover_max", 95),
        cfg.get("max_scenes_per_composite"),
    )
    t, c, h, w = stack.shape
    logger.info("box composite: {} months × {} bands × {}×{} px @ EPSG:{}", t, c, h, w, epsg)

    model, mu, sd, classes, ood = load_head(ckpt_path)
    enc = UniverSatEmbedder(pool="center", output_grid=CHIP_PX, device=device, batch=4)
    dvec = np.array([[MONTH_DOY[m["label"]] for m in months]])
    k = len(classes)

    # reflect-pad so every output pixel can sit at a window centre; 25% overlap.
    pad = CHIP_PX // 2
    stride = CHIP_PX * 3 // 4
    padded = np.nan_to_num(
        np.pad(stack, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="reflect") / 10000.0, nan=0.0
    )
    taper = _taper(CHIP_PX, CHIP_PX - stride)  # ramp = overlap → constant-overlap-add blend
    ops = _d4_ops(tta_views)
    dates = np.repeat(dvec, len(ops), axis=0)  # (n_views, T)
    prob_acc = np.zeros((h, w, k), "float32")
    ood_acc = np.zeros((h, w), "float32")
    wsum = np.zeros((h, w), "float32")

    for r0 in _starts(h + 2 * pad, CHIP_PX, stride):
        for c0 in _starts(w + 2 * pad, CHIP_PX, stride):
            win = padded[None, :, :, r0 : r0 + CHIP_PX, c0 : c0 + CHIP_PX]
            # all TTA views for this window in ONE batched forward (was 1 view/call)
            aug = np.concatenate([np.ascontiguousarray(fwd(win)) for fwd, _ in ops])
            grids = enc.embed_dense(aug, dates)  # (n_views, 64, 64, D)
            pr, od = predict_probs(model, mu, sd, ood, grids.reshape(-1, grids.shape[-1]))
            pr = pr.reshape(len(ops), CHIP_PX, CHIP_PX, k)
            od = od.reshape(len(ops), CHIP_PX, CHIP_PX)
            probs = np.mean([inv(pr[i]) for i, (_, inv) in enumerate(ops)], axis=0)
            oods = np.mean(
                [inv(od[i][..., None])[..., 0] for i, (_, inv) in enumerate(ops)], axis=0
            )
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
    out = np.stack([cls_map, conf_map, ood_map])

    crs = f"EPSG:{epsg}"
    if out_crs:  # warp native-zone map → national CRS so tiles mosaic into one grid
        out, transform, dst = _reproject_triplet(out, transform, epsg, out_crs)
        crs = dst.to_string()
    write_cog(out, transform, crs, out_uri, dtype="uint8", nodata=NODATA)
    logger.success(
        "wrote class/confidence/OOD COG ({}×{}, {} classes) @ {} → {}",
        out.shape[1],
        out.shape[2],
        k,
        crs,
        out_uri,
    )
    return out, transform, crs
