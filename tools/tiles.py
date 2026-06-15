"""Equal-population tiling: tile a state so every tessera holds ~k real residents.

The 2D analog of a pavement plot. A pavement plot slices a 1D range into equal-share
boxes, so box WIDTH shrinks where data is dense. Here we slice a 2D map into equal-share
tiles, so tile AREA shrinks where PEOPLE are dense -- downtown shatters into tiny tesserae,
empty countryside stays big slabs. Tile area is proportional to 1/density, and every tile
is "worth" the same k people.

Accuracy: seeds come from the Census 2020 block-group CENTERS OF POPULATION (population-
weighted centroids -- where people actually are inside each block group). A block group of
P people emits round(P/k) seeds, so total population is conserved and each Voronoi cell ends
up holding ~k residents. Randomness only perturbs a tile's shape, never the density.

    python tools/tiles.py Maryland --k 2000 --height 1600 --family quads
"""
import argparse
import base64
import csv
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import families as F                       # noqa: E402
import state_data as sd                    # noqa: E402

BG = ROOT / "data" / "cenpop2020_bg.txt"
COUNTYCSV = ROOT / "data" / "county_pop_2023.csv"


def state_fips(ST):
    """2-letter code -> 2-digit state FIPS, read from the county CSV (STNAME/STATE)."""
    name = sd.STATE_NAMES[ST]
    with open(COUNTYCSV, encoding="latin-1") as f:
        for row in csv.DictReader(f):
            if row["STNAME"].lower() == name:
                return row["STATE"]
    raise ValueError(f"no FIPS for {ST}")


def load_blockgroups(fips):
    """-> (lon, lat, pop) arrays for one state's block-group centers of population."""
    lon, lat, pop = [], [], []
    with open(BG, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["STATEFP"] != fips:
                continue
            p = int(row["POPULATION"])
            if p <= 0:
                continue
            lon.append(float(row["LONGITUDE"])); lat.append(float(row["LATITUDE"])); pop.append(p)
    return np.array(lon), np.array(lat), np.array(pop)


def load_bg_pop(fips):
    """-> dict 12-digit block-group GEOID -> population."""
    out = {}
    with open(BG, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["STATEFP"] != fips:
                continue
            geoid = r["STATEFP"] + r["COUNTYFP"] + r["TRACTCE"] + r["BLKGRPCE"]
            out[geoid] = int(r["POPULATION"])
    return out


def fetch_bg_polys(ST, fips):
    """Real 2020 block-group polygons for the state (cached). TIGERweb layer 10."""
    path = sd.CACHE / f"{ST}_bgpolys.json"
    if path.exists():
        return json.load(open(path))
    url = ("https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/"
           "tigerWMS_Current/MapServer/10/query")
    print(f"  fetching block-group polygons for {ST} ...", flush=True)
    gj = sd._post_paged(url, dict(where=f"STATE='{fips}'", outFields="GEOID",
                                  returnGeometry="true", outSR="4326",
                                  geometryPrecision="5", f="geojson"), page_size=2000)
    sd.CACHE.mkdir(parents=True, exist_ok=True)
    json.dump(gj, open(path, "w"))
    return gj


def fetch_neighborhoods():
    """DC Neighborhood Clusters (they tile DC completely). -> GeoJSON, cached."""
    path = sd.CACHE / "DC_neighborhood_clusters.json"
    if path.exists():
        return json.load(open(path))
    url = ("https://maps2.dcgis.dc.gov/dcgis/rest/services/DCGIS_DATA/"
           "Administrative_Other_Boundaries_WebMercator/MapServer/17/query")
    gj = sd._post(url, dict(where="1=1", outFields="NBH_NAMES,NAME",
                            returnGeometry="true", outSR="4326", f="geojson"))
    sd.CACHE.mkdir(parents=True, exist_ok=True)
    json.dump(gj, open(path, "w"))
    return gj


def state_mask(ST, S):
    """LAND mask (state minus water) + the matching projection. Clipping to land drops the
    Chesapeake/coastal water so coastal block groups don't sprawl across the Bay, and density
    is measured per land area. Uses the green-map pipeline's water/parks layers (cached)."""
    gj = sd._cached(ST, "state", lambda: sd._fetch_state(ST))
    feats = [f for f in gj["features"] if f.get("geometry")]
    norm, _ = sd._lon_norm(feats)
    bbox = sd._bbox(feats, norm)
    W, H, _ = sd._canvas(bbox, S)
    lonmin, latmin, lonmax, latmax = bbox

    def to_px(lons, lats):
        return (norm(lons) - lonmin) / (lonmax - lonmin) * W, (latmax - lats) / (latmax - latmin) * H

    from scipy import ndimage
    region = sd.build_region(ST, S, capital=False)
    # void = only WIDE water (Bay, big rivers): an opening removes thin tidal creeks so they
    # don't render as black dendrites inside coastal tiles. Thin creeks fold back into land.
    wide_water = ndimage.binary_opening(region == 3, iterations=max(2, round(S / 400)))
    land = (region > 0) & ~wide_water
    return land, wide_water, to_px, bbox


def density_ramp(t):
    """t in 0..1 -> RGB (0..1). Cool slate (sparse) -> warm sand -> bright gold (dense)."""
    stops = np.array([
        [0.13, 0.16, 0.26],   # deep slate-indigo  (very sparse)
        [0.20, 0.33, 0.40],   # teal-slate
        [0.78, 0.66, 0.45],   # warm sand
        [0.93, 0.78, 0.30],   # gold
        [0.98, 0.90, 0.62],   # pale gold        (very dense)
    ])
    x = np.clip(t, 0, 1) * (len(stops) - 1)
    i = np.minimum(np.floor(x).astype(int), len(stops) - 2)
    f = (x - i)[..., None]
    return stops[i] * (1 - f) + stops[i + 1] * f


# curated palette for proper colouring -- warm/earth tones only (NO blue/teal, which would
# read as water); mutually distinct & harmonious since any pair may touch.
PALETTE = np.array([
    [0.86, 0.69, 0.28],   # gold
    [0.78, 0.39, 0.24],   # terracotta
    [0.55, 0.60, 0.36],   # sage / olive
    [0.92, 0.87, 0.73],   # cream
    [0.55, 0.33, 0.42],   # plum / mauve
    [0.45, 0.34, 0.23],   # warm brown
])

WATER = np.array([0.24, 0.40, 0.55])               # muted river blue for actual water


def adjacency_lists(big, nbig):
    """Neighbour tile-ids for every tile (4-connectivity across shared borders)."""
    edges = []
    for A, B in ((big[:, :-1], big[:, 1:]), (big[:-1, :], big[1:, :])):
        m = (A != B) & (A > 0) & (B > 0)
        edges.append(np.stack([A[m], B[m]], 1))
    e = np.sort(np.concatenate(edges), 1)
    e = np.unique(e, axis=0)
    nb = [[] for _ in range(nbig + 1)]
    for a, b in e:
        nb[a].append(b); nb[b].append(a)
    return nb


def _dsatur(nb, deg, n, jit):
    """One DSATUR pass: repeatedly colour the most-saturated uncoloured tile (most distinct
    neighbour colours), tie-broken by degree then a random jitter. Hits the chromatic number
    far more often than plain greedy. -> colour index per tile."""
    col = np.full(n + 1, -1)
    sat = [set() for _ in range(n + 1)]
    remaining = set(range(1, n + 1))
    while remaining:
        v = max(remaining, key=lambda i: (len(sat[i]), deg[i], jit[i]))
        used = sat[v]
        c = 0
        while c in used:
            c += 1
        col[v] = c
        remaining.discard(v)
        for j in nb[v]:
            if col[j] < 0:
                sat[j].add(c)
    return col


def proper_colors(big, nbig, seed):
    """Minimal proper graph colouring: no two adjacent tiles share a colour, using as few
    colours as possible (planar maps need <=4 -- the four-colour theorem). DSATUR over a few
    randomised tie-breaks; keep the fewest-colour result. -> (index per tile, n_colors)."""
    nb = adjacency_lists(big, nbig)
    deg = np.array([len(nb[i]) for i in range(nbig + 1)])
    tries = 12 if nbig <= 1200 else 3
    best = None
    for s in range(tries):
        jit = np.random.default_rng(seed + s).random(nbig + 1)
        col = _dsatur(nb, deg, nbig, jit)
        ncol = int(col[1:].max()) + 1
        if best is None or ncol < best[1]:
            best = (col, ncol)
            if ncol <= 4:
                break
    return best


def tile_centroids(big, nbig):
    """Pixel centroid (x, y) of every tile label -> array indexed by label."""
    ys, xs = np.nonzero(big > 0)
    lab = big[ys, xs]
    cnt = np.maximum(np.bincount(lab, minlength=nbig + 1), 1)
    cx = np.bincount(lab, xs, minlength=nbig + 1) / cnt
    cy = np.bincount(lab, ys, minlength=nbig + 1) / cnt
    return np.column_stack([cx, cy])


def bridge_components(nb, cent, nbig):
    """Connect disconnected pieces of the adjacency graph (Chesapeake islands have no land
    bridge) to the nearest mainland block group, so islands can be absorbed into tiles instead
    of surviving as tiny ones. Mutates and returns nb."""
    comp = np.zeros(nbig + 1, int)
    cid = 0
    for s in range(1, nbig + 1):
        if comp[s]:
            continue
        cid += 1
        comp[s] = cid; stack = [s]
        while stack:
            u = stack.pop()
            for v in nb[u]:
                if not comp[v]:
                    comp[v] = cid; stack.append(v)
    if cid == 1:
        return nb
    main = np.bincount(comp)[1:].argmax() + 1
    main_ids = np.where(comp == main)[0]
    tree = cKDTree(cent[main_ids])
    for c in range(1, cid + 1):
        if c == main:
            continue
        ids = np.where(comp == c)[0]
        dist, idx = tree.query(cent[ids])
        k = int(dist.argmin())
        a, b = int(ids[k]), int(main_ids[idx[k]])
        nb[a].append(b); nb[b].append(a)
    return nb


def _two_split(members, nb, cent, pop):
    """Split a tile's block-group set into two connected, ~equal-population halves by growing
    from its two most-distant block groups, always feeding the lighter side. -> (setA, setB)."""
    ms = list(members)
    if len(ms) < 2:
        return None
    far = lambda s: max(ms, key=lambda j: (cent[j, 0] - cent[s, 0]) ** 2 + (cent[j, 1] - cent[s, 1]) ** 2)
    a = far(ms[0]); b = far(a)
    if a == b:
        return None
    side = {a: 0, b: 1}
    p = [pop[a], pop[b]]
    fr = [{x for x in nb[a] if x in members and x not in side},
          {x for x in nb[b] if x in members and x not in side}]
    while fr[0] or fr[1]:
        s = 0 if (fr[0] and (p[0] <= p[1] or not fr[1])) else 1
        x = fr[s].pop()
        if x in side:
            continue
        side[x] = s; p[s] += pop[x]
        for y in nb[x]:
            if y in members and y not in side:
                fr[s].add(y)
    A = {x for x in members if side.get(x, 0) == 0}
    B = members - A
    return (A, B) if A and B else None


def enforce_count(cl, nb, cent, pop, K):
    """Make the number of tiles exactly round(total/K): split the biggest tiles if there are
    too few, merge the smallest into a neighbour if there are too many. Centres the band."""
    M0 = max(1, int(round(pop[1:].sum() / K)))
    nextid = int(cl.max())
    for _ in range(4 * M0):                                    # split the largest until enough
        mm = {}
        for i in range(1, len(cl)):
            if cl[i] > 0:
                mm.setdefault(cl[i], set()).add(i)
        if len(mm) >= M0:
            break
        c = max(mm, key=lambda k: sum(pop[i] for i in mm[k]))
        res = _two_split(mm[c], nb, cent, pop)
        if not res:
            break
        nextid += 1
        for i in res[1]:
            cl[i] = nextid
    for _ in range(4 * M0):                                    # merge the smallest until few enough
        mm = {}
        for i in range(1, len(cl)):
            if cl[i] > 0:
                mm.setdefault(cl[i], set()).add(i)
        if len(mm) <= M0:
            break
        cpop = {k: sum(pop[i] for i in s) for k, s in mm.items()}
        c = min(cpop, key=cpop.get)
        nbrs = {cl[j] for i in mm[c] for j in nb[i] if cl[j] != c and cl[j] > 0}
        if not nbrs:
            break
        tgt = min(nbrs, key=lambda t: cpop.get(t, 1e18))
        for i in mm[c]:
            cl[i] = tgt
    return cl


def regionalize(nb, pop, K, seed, mode, cent):
    """Grow connected clusters of block groups, each accreting neighbours until its headcount
    reaches ~K, so every super-tile holds about the same number of people but with a shape WE
    choose. mode: compact (round honeycomb) / organic (blobby) / tendril (stringy).
    -> (cluster_id per bg label, n_clusters)."""
    n = len(pop) - 1
    rng = np.random.default_rng(seed)
    cl = np.zeros(n + 1, int)
    order = rng.permutation(np.arange(1, n + 1))
    cid = 0
    for s in order:
        if cl[s]:
            continue
        cid += 1
        cl[s] = cid
        cx, cy, cpop = cent[s, 0], cent[s, 1], pop[s]
        frontier = {j for j in nb[s] if not cl[j]}
        while frontier:
            if mode == "compact":
                nxt = min(frontier, key=lambda j: (cent[j, 0] - cx) ** 2 + (cent[j, 1] - cy) ** 2)
            elif mode == "tendril":
                nxt = max(frontier, key=lambda j: (cent[j, 0] - cx) ** 2 + (cent[j, 1] - cy) ** 2)
            else:                                    # organic: random frontier pick
                nxt = list(frontier)[rng.integers(len(frontier))]
            # stop at the point CLOSEST to target: if adding overshoots more than stopping
            # undershoots, don't add (keeps tiles centred on K, not all >= K)
            if cpop >= K or (cpop + pop[nxt] - K) >= (K - cpop):
                break
            cl[nxt] = cid
            np_ = cpop + pop[nxt]
            cx = (cx * cpop + cent[nxt, 0] * pop[nxt]) / np_       # pop-weighted running centroid
            cy = (cy * cpop + cent[nxt, 1] * pop[nxt]) / np_
            cpop = np_
            frontier.discard(nxt)
            frontier |= {j for j in nb[nxt] if not cl[j]}
    # absorb every undersized cluster (incl. 0-pop all-water ones) into a neighbour, smallest
    # first, until none remain below threshold -- kills the tiny-tile tail in the band
    thresh = 0.6 * K
    while True:
        cpop = np.bincount(cl, weights=pop, minlength=cid + 1)
        cadj = {}
        for i in range(1, n + 1):
            ci = cl[i]
            for j in nb[i]:
                if cl[j] != ci:
                    cadj.setdefault(ci, set()).add(cl[j])
        cand = [c for c in range(1, cid + 1) if (cpop[c] < thresh) and cadj.get(c) and (cl == c).any()]
        if not cand:
            break
        c = min(cand, key=lambda x: cpop[x])
        tgt = min(cadj[c], key=lambda t: cpop[t])
        cl[cl == c] = tgt
    # force the tile count to round(total/K) so the band centres on the target
    cl = enforce_count(cl, nb, cent, pop, K)
    # renumber clusters to 1..M contiguous
    uniq = np.unique(cl[cl > 0])
    relab = np.zeros(cl.max() + 1, int)
    relab[uniq] = np.arange(1, len(uniq) + 1)
    cl = relab[cl]
    return cl, len(uniq)


def _connected(S, nb):
    """Is the block-group set S connected under adjacency nb?"""
    if len(S) <= 1:
        return True
    start = next(iter(S))
    seen = {start}; stack = [start]
    while stack:
        u = stack.pop()
        for v in nb[u]:
            if v in S and v not in seen:
                seen.add(v); stack.append(v)
    return len(seen) == len(S)


def balance(cl, nb, pop, K, iters=20, seed=0):
    """Tighten the population band: move boundary block groups from over-target tiles to
    adjacent under-target tiles, greedily reducing total |tile_pop - K|, as long as the donor
    tile stays connected. Distorts shapes (tiles reach for people) but equalises headcounts."""
    M = int(cl.max())
    cpop = np.bincount(cl, weights=pop, minlength=M + 1)
    members = [set() for _ in range(M + 1)]
    for i in range(1, len(cl)):
        if cl[i] > 0:
            members[cl[i]].add(i)
    bgs = np.arange(1, len(cl))
    rng = np.random.default_rng(seed)
    for _ in range(iters):
        improved = False
        rng.shuffle(bgs)
        for i in bgs:
            c = cl[i]
            if len(members[c]) <= 1:
                continue
            best = None
            for d in {cl[j] for j in nb[i] if cl[j] != c and cl[j] > 0}:
                gain = ((abs(cpop[c] - K) + abs(cpop[d] - K))
                        - (abs(cpop[c] - pop[i] - K) + abs(cpop[d] + pop[i] - K)))
                if gain > 0 and (best is None or gain > best[1]):
                    best = (d, gain)
            if best and _connected(members[c] - {i}, nb):
                d = best[0]
                cl[i] = d; cpop[c] -= pop[i]; cpop[d] += pop[i]
                members[c].discard(i); members[d].add(i)
                improved = True
        if not improved:
            break
    return cl


def compactify(cl, nb, pop, K, cent, iters=8, tol=0.12, seed=0):
    """Pull 'gerrymandered' tendrils in: reassign each border unit to the adjacent tile whose
    (pop-weighted) centroid it is actually nearest to -- as long as BOTH tiles stay within tol
    of K and the donor stays connected. Rounder tiles; population still within +-tol of target."""
    M = int(cl.max())
    cpop = np.bincount(cl, weights=pop, minlength=M + 1)
    cx = np.bincount(cl, weights=pop * cent[:, 0], minlength=M + 1) / np.maximum(cpop, 1)
    cy = np.bincount(cl, weights=pop * cent[:, 1], minlength=M + 1) / np.maximum(cpop, 1)
    members = [set() for _ in range(M + 1)]
    for i in range(1, len(cl)):
        if cl[i] > 0:
            members[cl[i]].add(i)
    lo, hi = K * (1 - tol), K * (1 + tol)
    units = np.arange(1, len(cl))
    rng = np.random.default_rng(seed)
    for _ in range(iters):
        moved = 0
        rng.shuffle(units)
        for i in units:
            c = cl[i]
            if len(members[c]) <= 1 or cpop[c] - pop[i] < lo:
                continue
            dc = (cent[i, 0] - cx[c]) ** 2 + (cent[i, 1] - cy[c]) ** 2
            best = None
            for d in {cl[j] for j in nb[i] if cl[j] != c and cl[j] > 0}:
                if cpop[d] + pop[i] > hi:
                    continue
                dd = (cent[i, 0] - cx[d]) ** 2 + (cent[i, 1] - cy[d]) ** 2
                if dd < dc and (best is None or dd < best[1]):
                    best = (d, dd)
            if best and _connected(members[c] - {i}, nb):
                d, p = best[0], pop[i]
                nc, nd = cpop[c] - p, cpop[d] + p           # new tile populations
                cx[c] = (cx[c] * cpop[c] - cent[i, 0] * p) / max(nc, 1)
                cy[c] = (cy[c] * cpop[c] - cent[i, 1] * p) / max(nc, 1)
                cx[d] = (cx[d] * cpop[d] + cent[i, 0] * p) / max(nd, 1)
                cy[d] = (cy[d] * cpop[d] + cent[i, 1] * p) / max(nd, 1)
                cpop[c], cpop[d] = nc, nd
                cl[i] = d; members[c].discard(i); members[d].add(i); moved += 1
        if not moved:
            break
    return cl


def _font(size, bold=False):
    names = (["/System/Library/Fonts/Supplemental/Georgia Bold.ttf", "/System/Library/Fonts/HelveticaNeue.ttc"]
             if bold else ["/System/Library/Fonts/Supplemental/Georgia.ttf", "/System/Library/Fonts/Helvetica.ttc"])
    for p in names:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            pass
    return ImageFont.load_default()


def add_legend(img, title, held, dlo, dhi, literal=False, palette=None, merged=0):
    """Bake a caption band: title, per-tile population range, and either a density ramp or
    (when palette is given) decorative swatches. Returns a new taller image."""
    W, H = img.size
    bh = int(W * 0.135)                                        # band height
    band = Image.new("RGB", (W, bh), (25, 20, 15))
    d = ImageDraw.Draw(band)
    pad = int(W * 0.03)
    ink, mut = (232, 224, 210), (154, 143, 124)

    f_title = _font(int(bh * 0.30), bold=True)
    f_sub = _font(int(bh * 0.135))
    f_tick = _font(int(bh * 0.11))
    d.text((pad, int(bh * 0.16)), title, font=f_title, fill=ink)
    p10, p50, p90 = (int(round(v / 10.0) * 10) for v in np.percentile(held, [10, 50, 90]))
    if merged:
        line1 = (f"each tile groups Census block groups into about {p50:,} people  "
                 f"(most {p10:,} to {p90:,})")
        line2 = (f"{len(palette)} colours, no two neighbours alike - colour encodes nothing"
                 if palette is not None else
                 "equal-population super-tiles  -  smaller tile = denser")
    elif palette is not None:
        line1 = (f"each tile is one real 2020 Census block group  -  "
                 f"about {p50:,} people (most {p10:,} to {p90:,})")
        line2 = "colours are an algorithmic adjacency pattern - they encode nothing"
    elif literal:
        line1 = (f"each tile is one real 2020 Census block group  -  "
                 f"about {p50:,} people (most {p10:,} to {p90:,})")
        line2 = "colour = population density  -  real block-group boundaries, undistorted"
    else:
        line1 = (f"each tile holds about {p50:,} people  (most {p10:,} to {p90:,})  -  "
                 "smaller tile = denser")
        line2 = "tile area scales as 1 / density  -  2020 Census block-group centers of population"
    d.text((pad, int(bh * 0.56)), line1, font=f_sub, fill=mut)
    d.text((pad, int(bh * 0.78)), line2, font=f_tick, fill=mut)

    bx0, bx1 = int(W * 0.60), int(W - pad)
    by0, by1 = int(bh * 0.30), int(bh * 0.46)
    if palette is not None:
        # palette swatches (no encoding)
        n = len(palette); sw = (bx1 - bx0) / n
        for i, c in enumerate(palette):
            x0 = int(bx0 + i * sw)
            d.rectangle([x0, by0, int(x0 + sw) - 2, by1], fill=tuple((c * 255).astype(int)))
        d.text((bx0, by0 - int(bh * 0.16)), "palette  (size = real block groups, colour = decoration)",
               font=f_tick, fill=mut, anchor="la")
    else:
        # density colour-ramp bar
        for i, x in enumerate(range(bx0, bx1)):
            t = i / max(1, bx1 - bx0 - 1)
            c = (density_ramp(np.array(t)) * 255).astype(int)
            d.line([(x, by0), (x, by1)], fill=tuple(c))
        d.rectangle([bx0, by0, bx1, by1], outline=(70, 62, 50))
        for frac, lab in [(0.0, f"{dlo:,.0f}"), (1.0, f"{dhi:,.0f}")]:
            x = int(bx0 + frac * (bx1 - bx0))
            d.text((x, by1 + int(bh * 0.04)), f"{lab}", font=f_tick, fill=ink,
                   anchor=("la" if frac == 0 else "ra"))
        d.text(((bx0 + bx1) // 2, by1 + int(bh * 0.04)), "people per sq km", font=f_tick, fill=mut, anchor="ma")
        d.text((bx0, by0 - int(bh * 0.16)), "sparse  to  dense   (big tiles to small tiles)",
               font=f_tick, fill=mut, anchor="la")

    out = Image.new("RGB", (W, H + bh), (25, 20, 15))
    out.paste(img, (0, 0)); out.paste(band, (0, H))
    return out


def pct1(x):
    """Percent as a string: whole number for >=1%, else one significant figure so a tiny
    sliver shows '0.3%' / '0.04%' rather than rounding to a misleading '0%'."""
    x = float(x)
    if x <= 0:
        return "0"
    if round(x) >= 1:
        return str(int(round(x)))
    d = -int(math.floor(math.log10(x)))
    return f"{round(x, d):.{d}f}"


def _png_uri(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def export_interactive(dest, base_rgb, big, nbig, pop_lab, to_px, W, H, title):
    """Write a self-contained hover HTML: the tiling + a pixel-accurate pick-map + per-tile
    data (population, and which neighbourhood clusters it covers and by what %)."""
    # rasterise neighbourhood clusters into the tile canvas, intersect with tiles
    feats = [f for f in fetch_neighborhoods()["features"] if f.get("geometry")]
    tile_area = np.bincount(big.ravel(), minlength=nbig + 1).astype(float)
    overlap = [dict() for _ in range(nbig + 1)]                # tile -> {cluster_idx: px}
    carea = []
    for ci, f in enumerate(feats):
        cm = sd.rasterize_px([f], to_px, W, H) & (big > 0)
        carea.append(int(cm.sum()))
        labs, cnts = np.unique(big[cm], return_counts=True)
        for lab, cnt in zip(labs, cnts):
            overlap[int(lab)][ci] = int(cnt)

    data = {}
    for t in range(1, nbig + 1):
        parts = []
        for ci, px in overlap[t].items():
            pctT = 100 * px / max(tile_area[t], 1)             # % of this tile in the cluster
            pctN = 100 * px / max(carea[ci], 1)                # % of the cluster in this tile
            if pctT >= 3:
                parts.append((pctT, pctN, feats[ci]["properties"].get("NBH_NAMES", "")))
        parts.sort(reverse=True)
        data[t] = {"pop": int(round(pop_lab[t])),
                   "parts": [{"pctT": pct1(p[0]), "pctN": pct1(p[1]), "names": p[2]}
                             for p in parts[:5]]}

    # pick-map: tile id encoded in R + G*256
    idm = big.astype(np.uint32)
    pick = np.dstack([(idm & 255).astype(np.uint8),
                      ((idm >> 8) & 255).astype(np.uint8),
                      np.zeros_like(idm, np.uint8)])
    html = _HTML_TMPL
    for k, v in {"__W__": str(W), "__H__": str(H), "__TITLE__": title,
                 "__TILES__": _png_uri(base_rgb), "__PICK__": _png_uri(pick),
                 "__DATA__": json.dumps(data, separators=(",", ":"))}.items():
        html = html.replace(k, v)
    open(dest, "w").write(html)


_HTML_TMPL = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>__TITLE__</title>
<style>
 :root{--bg:#19140f;--ink:#e8e0d2;--mut:#9a8f7c;--gold:#e8c66a}
 *{box-sizing:border-box} html,body{margin:0;background:var(--bg);color:var(--ink);
  font:16px/1.5 "Iowan Old Style",Georgia,serif}
 .wrap{max-width:900px;margin:0 auto;padding:28px 20px 60px}
 h1{font-size:24px;margin:0 0 2px} .sub{color:var(--mut);font-size:15px;margin:0 0 18px}
 #stage{position:relative;display:inline-block;width:100%;border-radius:8px;overflow:hidden;
  box-shadow:0 14px 50px rgba(0,0,0,.55)}
 #stage img{display:block;width:100%;height:auto}
 #tip{position:fixed;pointer-events:none;z-index:9;max-width:330px;display:none;
  background:rgba(15,12,9,.96);border:1px solid rgba(232,198,106,.5);border-radius:9px;
  padding:11px 13px;font-size:14px;box-shadow:0 8px 28px rgba(0,0,0,.6)}
 #tip .pop{color:var(--gold);font-size:16px;margin-bottom:6px}
 #tip .row{color:var(--ink);margin:3px 0} #tip .pct{color:var(--gold);font-weight:600}
 #tip .nm{color:var(--mut);font-size:13px}
 .note{color:var(--mut);font-size:13px;margin-top:14px}
</style></head><body><div class="wrap">
<h1>__TITLE__</h1>
<p class="sub">Every tile holds about the same number of people. Hover a tile to see its
population and which neighbourhoods it covers.</p>
<div id="stage"><img id="mos" src="__TILES__" alt="population tiling"></div>
<div id="tip"></div>
<p class="note">Tile size = real 2020 Census block groups merged to a target headcount; colours
are a 4-colour pattern and encode nothing. Neighbourhoods: DC Neighborhood Clusters.</p>
</div>
<canvas id="pk" width="__W__" height="__H__" style="display:none"></canvas>
<script>
const DATA=__DATA__, W=__W__, H=__H__;
const pk=document.getElementById('pk'), ctx=pk.getContext('2d',{willReadFrequently:true});
const pim=new Image(); pim.src="__PICK__"; pim.onload=()=>ctx.drawImage(pim,0,0);
const mos=document.getElementById('mos'), tip=document.getElementById('tip');
function idAt(e){
  const r=mos.getBoundingClientRect();
  const x=Math.floor((e.clientX-r.left)/r.width*W), y=Math.floor((e.clientY-r.top)/r.height*H);
  if(x<0||y<0||x>=W||y>=H) return 0;
  const p=ctx.getImageData(x,y,1,1).data; return p[0]+p[1]*256;
}
mos.addEventListener('mousemove',e=>{
  const id=idAt(e), d=DATA[id];
  if(!d){tip.style.display='none';return;}
  let h='<div class="pop">&#8776; '+d.pop.toLocaleString()+' residents</div>';
  if(d.parts.length){ h+='<div class="nm">covers</div>';
    for(const p of d.parts) h+='<div class="row"><span class="pct">'+p.pctN+'%</span> of '+p.names+
      ' <span class="nm">('+p.pctT+'% of this tile)</span></div>'; }
  tip.innerHTML=h; tip.style.display='block';
  let tx=e.clientX+16, ty=e.clientY+16;
  if(tx+340>innerWidth) tx=e.clientX-340; if(ty+200>innerHeight) ty=e.clientY-200;
  tip.style.left=tx+'px'; tip.style.top=ty+'px';
});
mos.addEventListener('mouseleave',()=>tip.style.display='none');
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("state")
    ap.add_argument("--k", type=int, default=2000, help="residents per tile")
    ap.add_argument("--height", type=int, default=1600)
    ap.add_argument("--literal", action="store_true",
                    help="tile with REAL block-group polygons instead of pop-weighted Voronoi")
    ap.add_argument("--color", choices=["density", "adjacency"], default="density",
                    help="density: colour=people/area.  adjacency: algorithmic palette, encodes nothing")
    ap.add_argument("--merge", type=int, default=0,
                    help="group block groups into ~N-person super-tiles (literal mode only)")
    ap.add_argument("--shape", choices=["compact", "organic", "tendril"], default="compact",
                    help="super-tile shape when --merge is set")
    ap.add_argument("--no-balance", action="store_true",
                    help="skip the population-equalising refinement (looser band, tidier shapes)")
    ap.add_argument("--grout", type=float, default=1.4,
                    help="grout half-width in px (thin uniform lines; raise for fatter grout)")
    ap.add_argument("--html", action="store_true",
                    help="also write an interactive hover HTML (DC neighbourhoods; DC only)")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    ST = sd.resolve_state(args.state)
    fips = state_fips(ST)
    S = args.height
    rng = np.random.default_rng(args.seed)

    mask, water, to_px, bbox = state_mask(ST, S)
    H, W = mask.shape
    latm = (bbox[1] + bbox[3]) / 2
    km2_per_px = ((bbox[2] - bbox[0]) / W * 111.32 * np.cos(np.radians(latm))
                  * (bbox[3] - bbox[1]) / H * 110.57)

    if args.literal:
        # ---- LITERAL: every tile is a real 2020 block-group polygon -------------
        popmap = load_bg_pop(fips)
        feats = [f for f in fetch_bg_polys(ST, fips)["features"] if f.get("geometry")]
        print(f"{sd.STATE_NAMES[ST]}: {len(feats):,} real block-group polygons")
        big = np.zeros((H, W), int)
        pop_lab = np.zeros(len(feats) + 1)
        for i, f in enumerate(feats, 1):
            big[sd.rasterize_px([f], to_px, W, H) & mask] = i
            pop_lab[i] = popmap.get(f["properties"]["GEOID"], 0)
        nbig = len(feats)
        # fill rasterization gaps inside the state with the nearest tile (no black speckle)
        from scipy import ndimage
        ind = ndimage.distance_transform_edt(big == 0, return_distances=False, return_indices=True)
        big = np.where((big == 0) & mask, big[tuple(ind)], big)

        if args.merge:
            # group block groups into ~equal-population super-tiles with a shape we choose
            nb = adjacency_lists(big, nbig)
            cent = tile_centroids(big, nbig)
            nb = bridge_components(nb, cent, nbig)             # connect Chesapeake islands
            cl, M = regionalize(nb, pop_lab, args.merge, args.seed, args.shape, cent)
            if not args.no_balance:
                cl = balance(cl, nb, pop_lab, args.merge, seed=args.seed)
            # renumber to contiguous 1..M (balance may have emptied/kept ids)
            uniq = np.unique(cl[cl > 0])
            relab = np.zeros(int(cl.max()) + 1, int); relab[uniq] = np.arange(1, len(uniq) + 1)
            cl = relab[cl]; M = len(uniq)
            cpop = np.bincount(cl, weights=pop_lab, minlength=M + 1)[1:]
            big = cl[big]                                      # remap pixels to cluster ids
            pop_lab = np.concatenate([[0], cpop])
            nbig = M
            print(f"merged {len(cl) - 1:,} block groups -> {M:,} super-tiles "
                  f"(~{args.merge:,} ppl each, shape={args.shape})")
            lo, hi = cpop.min(), cpop.max()
            print(f"population band: {lo:,.0f} - {hi:,.0f}  (p10-p90 "
                  f"{np.percentile(cpop, 10):,.0f}-{np.percentile(cpop, 90):,.0f})")

        area_km2 = np.maximum(np.bincount(big.ravel(), minlength=nbig + 1), 1) * km2_per_px
        dens_lab = pop_lab / area_km2                          # people / km^2 per tile
        held = pop_lab[1:][pop_lab[1:] > 0]                    # real per-tile population spread
    else:
        # ---- STYLISED: pop-weighted Voronoi, each tile ~k people ---------------
        lon, lat, pop = load_blockgroups(fips)
        print(f"{sd.STATE_NAMES[ST]}: {pop.sum():,} people across {len(pop):,} block groups")
        bgx, bgy = to_px(lon, lat)
        nseed = np.maximum(1, np.round(pop / args.k).astype(int))
        jit = 0.5 * np.sqrt(mask.sum() / nseed.sum())          # sub-cell jitter in px
        sx, sy, held = [], [], []
        for x, y, n, p in zip(bgx, bgy, nseed, pop):
            sx.append(x + rng.normal(0, jit, n)); sy.append(y + rng.normal(0, jit, n))
            held.append(np.full(n, p / n))                     # people this seed's tile holds
        sx = np.concatenate(sx); sy = np.concatenate(sy); held = np.concatenate(held)
        ix = np.clip(np.round(sx).astype(int), 0, W - 1)
        iy = np.clip(np.round(sy).astype(int), 0, H - 1)
        inside = mask[iy, ix]
        sx, sy, held = sx[inside], sy[inside], held[inside]
        print(f"{len(sx):,} seeds  (~{args.k:,} people/tile)")
        ys, xs = np.nonzero(mask)
        _, idx = cKDTree(np.column_stack([sx, sy])).query(np.column_stack([xs, ys]), workers=-1)
        big = np.zeros((H, W), int)
        big[ys, xs] = idx + 1
        nbig = len(sx)
        area_km2 = np.maximum(np.bincount(big.ravel(), minlength=nbig + 1), 1) * km2_per_px
        dens_lab = np.zeros(nbig + 1)
        dens_lab[1:] = args.k / area_km2[1:]                   # k people / real tile area

    # ---- colour the tiles -------------------------------------------------------
    if args.color == "adjacency":
        # colour encodes NOTHING: fewest colours so no two adjacent tiles match
        col, ncol = proper_colors(big, nbig, args.seed)
        print(f"proper colouring: {ncol} colours, no two neighbours alike")
        pal = PALETTE[:max(ncol, 1)]
        lut = pal[np.clip(col, 0, len(pal) - 1)].copy()
        lut[col < 0] = 0
        palette = pal
    else:
        # colour = population density
        t = (np.log10(np.maximum(dens_lab, 1e-6)) - 1.0) / (4.0 - 1.0)   # 10..10,000 ppl/km^2
        lut = density_ramp(t)
        palette = None
    lut[0] = 0
    lut = np.clip(lut * rng.uniform(0.9, 1.1, nbig + 1)[:, None], 0, 1)

    region = mask.astype(int)                                  # 1 inside, 0 outside (-> wall)
    out = F.render_tiles(big, nbig, region, lut, forced_grout=None, gold_class=-1,
                          grout_width=args.grout)
    out[water] = WATER                                         # paint actual rivers blue

    base_rgb = (out * 255).astype(np.uint8)               # pre-legend tiling (for hover HTML)
    img = Image.fromarray(base_rgb.copy())
    dlo, dhi = np.percentile(dens_lab[1:][dens_lab[1:] > 0], [2, 98])
    img = add_legend(img, sd.STATE_NAMES[ST].upper(), held, dlo, dhi, args.literal, palette, args.merge)

    suffix = "_literal" if args.literal else ""
    if args.merge:
        suffix += f"_merge{args.merge}_{args.shape}"
    suffix += "_adj" if args.color == "adjacency" else ""
    stem = f"{sd.STATE_NAMES[ST].replace(' ', '_')}_poptiling{suffix}"
    dest = ROOT / "output" / f"{stem}.png"
    img.save(dest)
    hp = np.percentile(held, [10, 50, 90])
    print(f"per-tile people  p10/p50/p90 = {hp[0]:.0f}/{hp[1]:.0f}/{hp[2]:.0f}")
    print(f"density span (p2-p98) = {dlo:.0f}-{dhi:,.0f} ppl/km^2")
    print(f"{nbig:,} tiles  {W}x{H}  ->  {dest.relative_to(ROOT)}")

    if args.html:
        if ST != "DC":
            print("note: --html neighbourhood data is DC-only; skipping")
        else:
            hdest = ROOT / "output" / f"{stem}.html"
            export_interactive(hdest, base_rgb, big, nbig, pop_lab, to_px, W, H,
                               "Washington, DC - one tile, one neighborhood's worth of people")
            print(f"interactive -> {hdest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
