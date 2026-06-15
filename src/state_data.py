"""Fetch + cache + rasterize any US state into a region map for the mosaic tool.

One public function, build_region(state, S), returns an int class map at height S:
    0 outside · 1 land · 2 parks · 3 water · 4 capital disc

Data (all via stdlib urllib, no new deps; raw GeoJSON cached under data/cache/):
  state outline  Census TIGERweb States layer   (legal boundary, includes state water)
  water polygons Census TIGERweb Areal Hydrography (rivers/lakes/bays carve the water out)
  parks          USGS PAD-US Management Areas    (parks/forests/refuges, easements dropped)

The legal outline already contains the state's water area (MD owns the Chesapeake and its
half of the Potomac; MI owns its Great-Lakes halves); the hydro polygons then cut that
water back out of the land — which is exactly the green-map look.
"""
import json
import time
import urllib.parse
import urllib.request
from math import cos, radians
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"

# ST -> lowercase canonical name (also gives us the full set of valid codes)
STATE_NAMES = {
    "AL": "alabama", "AK": "alaska", "AZ": "arizona", "AR": "arkansas",
    "CA": "california", "CO": "colorado", "CT": "connecticut", "DE": "delaware",
    "FL": "florida", "GA": "georgia", "HI": "hawaii", "ID": "idaho",
    "IL": "illinois", "IN": "indiana", "IA": "iowa", "KS": "kansas",
    "KY": "kentucky", "LA": "louisiana", "ME": "maine", "MD": "maryland",
    "MA": "massachusetts", "MI": "michigan", "MN": "minnesota", "MS": "mississippi",
    "MO": "missouri", "MT": "montana", "NE": "nebraska", "NV": "nevada",
    "NH": "new hampshire", "NJ": "new jersey", "NM": "new mexico", "NY": "new york",
    "NC": "north carolina", "ND": "north dakota", "OH": "ohio", "OK": "oklahoma",
    "OR": "oregon", "PA": "pennsylvania", "RI": "rhode island", "SC": "south carolina",
    "SD": "south dakota", "TN": "tennessee", "TX": "texas", "UT": "utah",
    "VT": "vermont", "VA": "virginia", "WA": "washington", "WV": "west virginia",
    "WI": "wisconsin", "WY": "wyoming", "DC": "district of columbia",
}
NAME_TO_ST = {name: st for st, name in STATE_NAMES.items()}

CAPITALS = {  # ST: (lat, lon) — approximate city centers, fine at map scale
    "AL": (32.3777, -86.3000), "AK": (58.3019, -134.4197), "AZ": (33.4484, -112.0964),
    "AR": (34.7465, -92.2896), "CA": (38.5767, -121.4934), "CO": (39.7392, -104.9847),
    "CT": (41.7640, -72.6822), "DE": (39.1582, -75.5244), "FL": (30.4383, -84.2807),
    "GA": (33.7490, -84.3880), "HI": (21.3070, -157.8584), "ID": (43.6178, -116.1996),
    "IL": (39.7983, -89.6544), "IN": (39.7684, -86.1581), "IA": (41.5912, -93.6038),
    "KS": (39.0473, -95.6752), "KY": (38.1867, -84.8753), "LA": (30.4571, -91.1874),
    "ME": (44.3072, -69.7817), "MD": (38.9784, -76.4922), "MA": (42.3601, -71.0589),
    "MI": (42.7325, -84.5555), "MN": (44.9537, -93.0900), "MS": (32.2988, -90.1848),
    "MO": (38.5767, -92.1735), "MT": (46.5891, -112.0391), "NE": (40.8136, -96.7026),
    "NV": (39.1638, -119.7674), "NH": (43.2081, -71.5376), "NJ": (40.2206, -74.7597),
    "NM": (35.6870, -105.9378), "NY": (42.6526, -73.7562), "NC": (35.7796, -78.6382),
    "ND": (46.8083, -100.7837), "OH": (39.9612, -82.9988), "OK": (35.4676, -97.5164),
    "OR": (44.9429, -123.0351), "PA": (40.2732, -76.8867), "RI": (41.8240, -71.4128),
    "SC": (34.0007, -81.0348), "SD": (44.3683, -100.3510), "TN": (36.1627, -86.7816),
    "TX": (30.2672, -97.7431), "UT": (40.7608, -111.8910), "VT": (44.2601, -72.5754),
    "VA": (37.5407, -77.4360), "WA": (47.0379, -122.9007), "WV": (38.3498, -81.6326),
    "WI": (43.0731, -89.4012), "WY": (41.1400, -104.8202), "DC": (38.8899, -77.0091),
}


def resolve_state(state):
    """'Maryland' / 'maryland' / 'md' / 'MD' -> 'MD'. Raises on anything unknown."""
    s = str(state).strip()
    if len(s) == 2 and s.upper() in STATE_NAMES:
        return s.upper()
    st = NAME_TO_ST.get(s.lower())
    if st is None:
        raise ValueError(f"unknown state: {state!r} (try a full name or 2-letter code)")
    return st


# ---------------------------------------------------------------------------
# fetch (stdlib urllib; POST form-encoded to ArcGIS /query) + on-disk cache
# ---------------------------------------------------------------------------
def _post(url, params, timeout=180, retries=4):
    body = urllib.parse.urlencode(params).encode()
    req = urllib.request.Request(url, data=body, headers={"User-Agent": "mosaic/1.0"})
    last = None
    for attempt in range(retries):                    # transient 5xx/timeout backoff (busy ArcGIS)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:                        # noqa: BLE001 — retry any failure
            last = e
            if attempt < retries - 1:
                time.sleep(3 * (attempt + 1))
    raise last


def _post_paged(url, params, page_size):
    """ArcGIS pagination: loop resultOffset until exceededTransferLimit clears.
    The flag shows up top-level OR under properties depending on the service."""
    feats = []
    offset = 0
    while True:
        resp = _post(url, dict(params, resultOffset=offset, resultRecordCount=page_size))
        page = resp.get("features", [])
        feats.extend(page)
        exceeded = resp.get("exceededTransferLimit") or \
            resp.get("properties", {}).get("exceededTransferLimit")
        if not exceeded or not page:
            break
        offset += len(page)
    return {"type": "FeatureCollection", "features": feats}


def _cached(ST, kind, fetch_fn):
    """Return cached GeoJSON if present, else fetch (one progress line) and cache it."""
    path = CACHE / f"{ST}_{kind}.json"
    if path.exists():
        return json.load(open(path))
    print(f"  fetching {kind} for {ST} (can take ~20s) ...", flush=True)
    gj = fetch_fn()
    CACHE.mkdir(parents=True, exist_ok=True)
    json.dump(gj, open(path, "w"))
    return gj


def _fetch_state(ST):
    # TIGERweb States layer — the only one that returns real geometry (the Generalized_*
    # services hand back geometry:null). One feature, the LEGAL boundary (water included).
    url = ("https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/"
           "State_County/MapServer/0/query")
    return _post(url, dict(where=f"STUSAB='{ST}'", outFields="GEOID,BASENAME",
                           returnGeometry="true", geometryPrecision="4", f="geojson"))


def _fetch_water(envelopes):
    """Fetch hydro for one or more lon/lat envelopes (>1 only for antimeridian states,
    whose bbox is split at +-180), merging the features into one collection."""
    url = ("https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/"
           "Hydro/MapServer/1/query")
    feats = []
    for lonmin, latmin, lonmax, latmax in envelopes:
        # AREAWATER is a STRING field -> can't go in `where` (400); we filter it client-side.
        gj = _post_paged(url, dict(
            where="1=1", geometry=f"{lonmin},{latmin},{lonmax},{latmax}",
            geometryType="esriGeometryEnvelope", inSR="4326",
            spatialRel="esriSpatialRelIntersects", outFields="MTFCC,AREAWATER",
            returnGeometry="true", geometryPrecision="4", maxAllowableOffset="0.0005",
            f="geojson"), page_size=100_000)
        feats.extend(gj["features"])
    return {"type": "FeatureCollection", "features": feats}


def _fetch_parks(ST):
    url = ("https://services.arcgis.com/v01gqwM5QqNysAAi/arcgis/rest/services/"
           "PADUS_Management_Areas/FeatureServer/0/query")
    des = ("'NP','SP','LP','NF','SF','NWR','SCA','LCA','NCA','SREC','LREC',"
           "'NRA','WA','WSA','NM','NLS','SW'")                 # park/forest/refuge-like only
    # State_Nm holds 2-LETTER codes; maxRecordCount is only 2,000 so we page.
    # NB: maxAllowableOffset nulls the geometry on THIS server (unlike TIGERweb), so we
    # rely on geometryPrecision=4 alone to keep the payload trim.
    return _post_paged(url, dict(
        where=f"State_Nm='{ST}' AND Des_Tp IN ({des})", outFields="Des_Tp,Unit_Nm",
        returnGeometry="true", outSR="4326", geometryPrecision="4",
        f="geojson"), page_size=2_000)


# ---------------------------------------------------------------------------
# rasterize (equirectangular, PIL ImageDraw — same spirit as dc_build)
# ---------------------------------------------------------------------------
def _iter_polys(geom):
    """Yield each polygon (a list of rings) from a Polygon or MultiPolygon geometry."""
    if not geom:
        return
    if geom["type"] == "Polygon":
        yield geom["coordinates"]
    elif geom["type"] == "MultiPolygon":
        yield from geom["coordinates"]


def _lon_norm(features):
    """Detect an antimeridian crossing (lon span > 180°, e.g. Alaska's Aleutians wrap
    +180->-180) and return (norm, crossing): norm maps lon<0 -> lon+360 so the geometry
    is contiguous. Non-crossing states get the identity."""
    lons = np.concatenate([np.asarray(r, float)[:, 0]
                           for f in features for poly in _iter_polys(f.get("geometry"))
                           for r in poly])
    if float(lons.max() - lons.min()) > 180.0:
        return (lambda L: np.where(np.asarray(L) < 0, np.asarray(L) + 360.0, L)), True
    return (lambda L: L), False


def _bbox(features, norm=lambda L: L):
    lons, lats = [], []
    for f in features:
        for poly in _iter_polys(f.get("geometry")):
            for ring in poly:
                a = np.asarray(ring, float)
                lons.append(norm(a[:, 0])); lats.append(a[:, 1])
    lon = np.concatenate(lons); lat = np.concatenate(lats)
    return float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max())


def _canvas(bbox, S):
    lonmin, latmin, lonmax, latmax = bbox
    dlon, dlat = lonmax - lonmin, latmax - latmin
    cosm = cos(radians((latmin + latmax) / 2))
    H = S
    W = max(1, int(round(S * (dlon * cosm) / dlat)))
    return W, H, cosm


def _poly_px(ring, to_px):
    a = np.asarray(ring, float)
    x, y = to_px(a[:, 0], a[:, 1])
    return list(zip(x.tolist(), y.tolist()))


def rasterize_px(features, to_px, W, H):
    """Burn polygon features into a (H,W) bool mask. to_px(lon_arr, lat_arr) -> (x_arr, y_arr)
    in pixels (any projection). Hole-safe: simple (no-hole) polygons share one fast draw pass,
    while holed polygons are drawn in isolation and OR-ed in, so one feature's hole can never
    erase a neighbouring feature's fill."""
    img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(img)
    holed = []
    for f in features:
        for poly in _iter_polys(f.get("geometry")):
            if len(poly[0]) < 3:
                continue
            if len(poly) > 1:                            # has holes -> defer to isolated pass
                holed.append(poly)
            else:
                d.polygon(_poly_px(poly[0], to_px), fill=1)
    arr = np.asarray(img, bool)
    if holed:
        arr = arr.copy()
        for poly in holed:
            sub = Image.new("L", (W, H), 0)
            sd = ImageDraw.Draw(sub)
            for i, ring in enumerate(poly):
                if len(ring) >= 3:
                    sd.polygon(_poly_px(ring, to_px), fill=(1 if i == 0 else 0))
            arr |= np.asarray(sub, bool)
    return arr


def _rasterize(features, bbox, S, norm=lambda L: L):
    """Equirectangular rasterize at the canvas for `bbox` (cos-lat aspect baked into W).
    `norm` maps lon into the bbox's space (antimeridian shift)."""
    lonmin, latmin, lonmax, latmax = bbox
    dlon, dlat = lonmax - lonmin, latmax - latmin
    W, H, _ = _canvas(bbox, S)

    def to_px(lons, lats):
        return (norm(lons) - lonmin) / dlon * W, (latmax - lats) / dlat * H

    return rasterize_px(features, to_px, W, H)


def _keep_water(features):
    """Keep lakes/bays/reservoirs >= 50,000 m²; always keep rivers (H3010) so threads survive."""
    out = []
    for f in features:
        p = f.get("properties", {})
        if p.get("MTFCC") == "H3010":
            out.append(f); continue
        try:
            if float(p.get("AREAWATER")) >= 50_000:
                out.append(f)
        except (TypeError, ValueError):
            pass
    return out


def _despeckle(mask, min_px):
    """Drop connected components smaller than min_px (mirrors dc_build.build_region)."""
    lab, n = ndimage.label(mask)
    if not n:
        return mask
    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    keep = np.nonzero(sizes >= min_px)[0] + 1
    return np.isin(lab, keep)


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------
def build_region(state, S, capital=True):
    """-> int (S-high) region map: 0 outside, 1 land, 2 parks, 3 water, 4 capital."""
    ST = resolve_state(state)
    state_gj = _cached(ST, "state", lambda: _fetch_state(ST))
    sfeat = [f for f in state_gj["features"] if f.get("geometry")]
    if not sfeat:
        raise RuntimeError(f"no state geometry returned for {ST}")
    norm, crossing = _lon_norm(sfeat)                # Alaska wraps the antimeridian
    bbox = _bbox(sfeat, norm)                         # bbox in normalized (possibly shifted) lon
    lonmin, latmin, lonmax, latmax = bbox
    pad = 0.02
    if crossing:                                     # split the hydro envelope back at +-180
        envs = [(lonmin - pad, latmin - pad, 180.0, latmax + pad),
                (-180.0, latmin - pad, lonmax - 360.0 + pad, latmax + pad)]
    else:
        envs = [(lonmin - pad, latmin - pad, lonmax + pad, latmax + pad)]

    water_gj = _cached(ST, "water", lambda: _fetch_water(envs))
    parks_gj = _cached(ST, "parks", lambda: _fetch_parks(ST))

    # all three layers share the STATE bbox/canvas so they register pixel-for-pixel
    state_mask = _rasterize(sfeat, bbox, S, norm)
    water_mask = _rasterize(_keep_water(water_gj["features"]), bbox, S, norm) & state_mask
    parks_mask = _rasterize(parks_gj["features"], bbox, S, norm) & state_mask
    H, W = state_mask.shape

    # speckle filters: drop only true specks so small parks/ponds survive (proportional grout
    # lets the resulting small tiles still read in colour)
    parks_mask = _despeckle(parks_mask, 0.0004 * state_mask.sum())
    water_mask = _despeckle(water_mask, (0.35 * S / 130.0) ** 2)

    region = state_mask.astype(int)                  # 1 = land
    region[parks_mask] = 2
    region[water_mask] = 3                            # water LAST: rivers cut through parks

    if capital and ST in CAPITALS:
        clat, clon = CAPITALS[ST]
        cx = (float(norm(clon)) - lonmin) / (lonmax - lonmin) * W
        cy = (latmax - clat) / (latmax - latmin) * H
        yy, xx = np.mgrid[0:H, 0:W]
        disc = ((xx - cx) ** 2 + (yy - cy) ** 2) < (0.018 * H) ** 2
        region[disc & (region > 0)] = 4

    return region
