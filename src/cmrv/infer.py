"""Wall-to-wall inference — apply the frozen head densely → class / confidence / OOD COG.

A lon/lat box → 3-month S2 composite → UniverSat **dense** token grid (per 64x64
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

import geopandas as gpd  # type: ignore
import numpy as np
from loguru import logger  # type: ignore
from rasterio.crs import CRS as RioCRS  # type: ignore
from rasterio.transform import array_bounds  # type: ignore
from rasterio.warp import Resampling, calculate_default_transform, reproject  # type: ignore
from shapely.geometry import box as shp_box  # type: ignore

from cmrv.aoi import SA_ALBERS, months_for_geom, utm_epsg
from cmrv.embeddings.constants import MONTH_DOY
from cmrv.embeddings.head import load_head, predict_probs
from cmrv.embeddings.universat import UniverSatEmbedder
from cmrv.ingest import s2
from cmrv.ingest.chips import CHIP_PX, WINDOW_PADDING_DAYS, temporal_windows
from cmrv.io import load_config, write_cog

RESOLUTION_M = 10
NODATA = 255
# Dask block size for a wall-to-wall box. Unlike a chip (one block, loaded
# eagerly), a box is far too large to hold, so these loads stay lazy.
CHUNK = 1024
# Acquisition dates medianed per month over a box. A chip takes one date, because
# the SCL screen can find a date that is clear over 1.3 km. No single date is
# clear over a whole inference box, so here the median does real work.
INFER_MAX_DATES = 6


def _composite_box(geom_wgs84, year, months_cfg, bands, max_dates=INFER_MAX_DATES):
    """3-month median composite -> ``(T, C, H, W)``, transform, epsg (from the S2 data).

    Same screening as a training chip, at box scale: search, screen footprints per
    date, then rank the dates by the clear fraction **over this box** measured on
    SCL alone at 1/8 resolution. Ranking on ``eo:cloud_cover`` instead would rank
    on cloud over a 110 km MGRS tile, which is not the same question.
    """
    # The grid is fixed before any network call, from the box centroid's UTM zone
    # — the same rule the training chips use, so a chip and a map pixel land on
    # the same 10 m grid. Deriving it from the first item's `proj:epsg` instead
    # would make the output grid depend on which scene the search happened to
    # return first.
    cx, cy = geom_wgs84.centroid.coords[0]
    epsg = utm_epsg(cx, cy)
    box_utm = gpd.GeoSeries([geom_wgs84], crs=4326).to_crs(epsg).iloc[0]
    gbox = s2.bbox_geobox(box_utm.bounds, epsg, RESOLUTION_M)
    scan_gbox = gbox.zoom_out(8)  # ranking months needs 80 m pixels, not 10 m

    arrs = []
    for win in temporal_windows(year, months_cfg, WINDOW_PADDING_DAYS):
        items = s2.search_bbox(geom_wgs84.bounds, win["start"], win["end"])
        if not items:
            raise ValueError(f"no S2 scenes for {win['start']}/{win['end']} — try another year/box")
        # No footprint screen here. A chip demands one date that covers it whole;
        # a box is legitimately mosaicked from many partial frames, so per-date
        # full coverage is the wrong question at this scale.
        scored = s2.clear_fraction_per_date(items, scan_gbox)
        scored.sort(key=lambda p: -p[0])
        keep = [i for _f, grp in scored[:max_dates] for i in grp] or items
        logger.info(
            "  {} {}: {} dates -> keeping best {} (clearest {:.0%})",
            win["label"], year, len(scored), min(max_dates, len(scored)),
            scored[0][0] if scored else 0.0,
        )
        med = s2.load_composite(keep, gbox, bands, chunks={"x": CHUNK, "y": CHUNK}).compute()
        arrs.append(med.values.astype("float32"))
    return np.stack(arrs), gbox.transform, epsg


def _starts(n: int, win: int, stride: int) -> list[int]:
    """Window start positions tiling ``[0, n)`` with overlap; last flush to the edge."""
    s = list(range(0, max(n - win, 0) + 1, stride))
    if n > win and s[-1] != n - win:
        s.append(n - win)
    return s


def _d4_ops(n_views: int):
    """The first ``n_views`` dihedral augmentations as ``(fwd, inv)`` pairs.

    Ordered so **1 = identity** (no TTA), **4 = the four 90° rotations**, **8 = the full
    D4 group** (rotations x flips). ``fwd`` transforms an input's spatial last-2 axes;
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
    geom = shp_box(*bbox)
    bands = cfg["s2_bands"]
    _zone, months = months_for_geom(geom, cfg)
    stack, transform, epsg = _composite_box(
        geom, year, months, bands, cfg.get("infer_max_dates", INFER_MAX_DATES)
    )
    t, c, h, w = stack.shape
    logger.info("box composite: {} months x {} bands x {}x{} px @ EPSG:{}", t, c, h, w, epsg)

    model, mu, sd, classes, ood = load_head(ckpt_path)
    enc = UniverSatEmbedder(pool="center", output_grid=CHIP_PX, device=device, batch=1)
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
        out, transform, dst = _reproject_triplet(out, transform, epsg, out_crs)  # type: ignore
        crs = dst.to_string()
    write_cog(out, transform, crs, out_uri, dtype="uint8", nodata=NODATA)
    logger.success(
        "wrote class/confidence/OOD COG ({}x{}, {} classes) @ {} → {}",
        out.shape[1],
        out.shape[2],
        k,
        crs,
        out_uri,
    )
    return out, transform, crs
