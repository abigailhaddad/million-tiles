"""CONUS Albers projection + land/water masks for the national tiling.

A trimmed, standalone copy of the projection helpers (no green-map site builder): an Albers
Equal-Area Conic for the lower 48 and a panel abstraction that rasterizes each state's cached
land / water / parks polygons into a shared canvas. Reused by tools/pop_mosaic_us.py.
"""
import json
from math import cos, radians, sin, sqrt

import numpy as np

import state_data as sd

CONUS = [c for c in sorted(sd.STATE_NAMES) if c not in ("AK", "HI")]   # lower 48 + DC


def _albers(lon0=-96.0, lat0=23.0, lat1=29.5, lat2=45.5):
    """Albers Equal-Area Conic (CONUS standard parallels) -> planar (X, Y), unit sphere."""
    l0, l1, l2 = radians(lat0), radians(lat1), radians(lat2)
    n = (sin(l1) + sin(l2)) / 2
    C = cos(l1) ** 2 + 2 * n * sin(l1)
    rho0 = sqrt(C - 2 * n * sin(l0)) / n
    lon0r = radians(lon0)

    def proj(lons, lats):
        lonr = np.radians(np.asarray(lons, float)); latr = np.radians(np.asarray(lats, float))
        theta = n * (lonr - lon0r)
        rho = np.sqrt(np.maximum(C - 2 * n * np.sin(latr), 0.0)) / n
        return rho * np.sin(theta), rho0 - rho * np.cos(theta)

    return proj


def _feats_of(codes):
    """Load cached outline features per state (skips states with no cache)."""
    feats, allsf = {}, []
    for ST in codes:
        sp = sd.CACHE / f"{ST}_state.json"
        if not sp.exists():
            continue
        sf = [f for f in json.load(open(sp))["features"] if f.get("geometry")]
        if sf:
            feats[ST] = sf; allsf += sf
    return feats, allsf


def _build_panel(codes, S, mode, clip_bbox=None):
    """A panel: states `codes` projected into a (W, H) canvas via `to_px`. mode='albers' (CONUS)
    or 'equirect'. Returns dict(codes, feats, to_px, W, H), or None if no geometry is cached."""
    feats, allsf = _feats_of(codes)
    if not feats:
        return None
    if mode == "albers":
        proj = _albers()
        xs, ys = [], []
        for f in allsf:
            for poly in sd._iter_polys(f.get("geometry")):
                for ring in poly:
                    a = np.asarray(ring, float); X, Y = proj(a[:, 0], a[:, 1])
                    xs.append(X); ys.append(Y)
        X = np.concatenate(xs); Y = np.concatenate(ys)
        x0, x1, y0, y1 = X.min(), X.max(), Y.min(), Y.max()
        scale = S / (y1 - y0); W = max(1, int(round((x1 - x0) * scale)))

        def to_px(lons, lats):
            Xp, Yp = proj(lons, lats)
            return (Xp - x0) * scale, (y1 - Yp) * scale
        H = S
    else:                                                # equirectangular
        norm, _ = sd._lon_norm(allsf)
        bbox = clip_bbox if clip_bbox else sd._bbox(allsf, norm)
        W, H, _ = sd._canvas(bbox, S)
        lonmin, latmin, lonmax, latmax = bbox
        dlon, dlat = lonmax - lonmin, latmax - latmin

        def to_px(lons, lats):
            return ((norm(np.asarray(lons, float)) - lonmin) / dlon * W,
                    (latmax - np.asarray(lats, float)) / dlat * H)
    return dict(codes=list(feats), feats=feats, to_px=to_px, W=W, H=H)


def _panel_masks(panel):
    """Rasterize a panel's land / water / park masks at its own resolution."""
    W, H, to_px = panel["W"], panel["H"], panel["to_px"]
    state = np.zeros((H, W), bool); water = state.copy(); parks = state.copy()
    for ST in panel["codes"]:
        state |= sd.rasterize_px(panel["feats"][ST], to_px, W, H)
        wp, pp = sd.CACHE / f"{ST}_water.json", sd.CACHE / f"{ST}_parks.json"
        if wp.exists():
            water |= sd.rasterize_px(sd._keep_water(json.load(open(wp))["features"]), to_px, W, H)
        if pp.exists():
            parks |= sd.rasterize_px(json.load(open(pp))["features"], to_px, W, H)
    return state, water & state, parks & state
