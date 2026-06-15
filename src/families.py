"""Tile shape FAMILIES — the reusable shape vocabulary for tiling maps.

Every family is a generative *process* that leaves tile-shaped debris (the tone of the
piece: nothing from a pattern book). The original three from dc_build.py plus four new
ones, all behind one uniform interface so any family can fill any region class:

    labels, grout = FAMILIES[name]["fn"](mask, base, seed)

  mask   bool (H, W)  — the region to fill
  base   float        — tile size in px (the "across" dimension of a typical tile)
  labels int  (H, W)  — 1..n inside mask, 0 outside; 0 *inside* mask only where the
                        family forces extra grout (see pebbles)
  grout  bool (H, W) or None — pixels the family wants painted as grout (fat, uneven
                        joints — only pebbles uses this so far)

Families:
  quads    hand-cut rectangular tesserae   (deformed quad lattice + cluster-merge)
  shards   interlocking glass shards       (warped variable-density Voronoi + merge)
  flow     cells stretched along the current (anisotropic cells along the EDT flow)
  pebbles  rounded river stones in a fat grout bed (eroded + smoothed Voronoi)
  strata   long slivers along a wandering grain (oriented Chebyshev cells)
  crackle  dried-mud shatter               (recursive jagged cuts, heavy size tail)
  rings    wobbly growth rings hugging the region edge (warped EDT bands, segmented)
  disc     one smooth piece (the locator)

compose() stitches per-class families into one seamless tessellation;
render_tiles() colours and shades it dc_build-style.

Run `python3 src/families.py` -> output/family_swatches.png (a contact sheet).
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
WALL = np.array([0.10, 0.08, 0.06])              # outside the map (dark wall)
GROUT = np.array([0.34, 0.35, 0.29])             # warm grey, measured from the real tiling


# ---------------------------------------------------------------------------
# shared machinery (the first few adapted from src/dc_build.py)
# ---------------------------------------------------------------------------
def _noise(H, W, scale, seed):
    """Smooth noise in [-1, 1] with feature wavelength ~ scale px."""
    rng = np.random.default_rng(seed)
    lo = rng.standard_normal((max(2, H // scale), max(2, W // scale)))
    z = ndimage.zoom(lo, (H / lo.shape[0], W / lo.shape[1]), order=3)[:H, :W]
    z -= z.mean(); m = np.abs(z).max() or 1.0
    return z / m


def _jit_grid(H, W, sp, seed, jit=0.42):
    rng = np.random.default_rng(seed)
    ys = np.arange(sp / 2, H, sp); xs = np.arange(sp / 2, W, sp)
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    gy = gy + rng.uniform(-jit * sp, jit * sp, gy.shape)
    gx = gx + rng.uniform(-jit * sp, jit * sp, gx.shape)
    return np.column_stack([gy.ravel(), gx.ravel()])


def _quad_grid(H, W, sp, seed, jit=0.34):
    """Seamless deformed lattice: 4-cornered quads sharing edges with their neighbours."""
    rng = np.random.default_rng(seed)
    ny, nx = int(H / sp) + 2, int(W / sp) + 2
    vy = np.arange(ny)[:, None] * sp + rng.uniform(-jit * sp, jit * sp, (ny, nx))
    vx = np.arange(nx)[None, :] * sp + rng.uniform(-jit * sp, jit * sp, (ny, nx))
    im = Image.new("I", (W, H), 0); d = ImageDraw.Draw(im)
    lab = 1
    for i in range(ny - 1):
        for j in range(nx - 1):
            d.polygon([(vx[i, j], vy[i, j]), (vx[i, j+1], vy[i, j+1]),
                       (vx[i+1, j+1], vy[i+1, j+1]), (vx[i+1, j], vy[i+1, j])], fill=lab)
            lab += 1
    return np.array(im, dtype=int)


def _var_seeds(H, W, base, seed, var=1.0):
    """Jittered grid with seeds dropped in low-noise clusters => big+small cells."""
    rng = np.random.default_rng(seed)
    s = _jit_grid(H, W, base, seed)
    n = _noise(H, W, max(2, int(base * 3)), seed + 10)
    si = np.clip(s[:, 0].astype(int), 0, H - 1); sj = np.clip(s[:, 1].astype(int), 0, W - 1)
    keepp = 0.4 + 0.6 * (n[si, sj] + 1) / 2
    keep = rng.random(len(s)) < (1 - var) + var * keepp
    s = s[keep]
    return s if len(s) >= 4 else _jit_grid(H, W, base, seed)


def _merge_variance(labels, strength, seed):
    """Merge adjacent tiles into clusters in high-noise regions => heavy-tailed sizes."""
    H, W = labels.shape
    n = int(labels.max())
    if n < 2:
        return labels
    parent = np.arange(n + 1)

    def find(x):
        r = x
        while parent[r] != r:
            r = parent[r]
        while parent[x] != r:
            parent[x], x = r, parent[x]
        return r

    au, av = labels[:, :-1].ravel(), labels[:, 1:].ravel(); mh = au != av
    bu, bv = labels[:-1, :].ravel(), labels[1:, :].ravel(); mv = bu != bv
    pu = np.concatenate([au[mh], bu[mv]]); pv = np.concatenate([av[mh], bv[mv]])
    ok = (pu > 0) & (pv > 0); pu, pv = pu[ok], pv[ok]
    key = np.unique(np.minimum(pu, pv).astype(np.int64) * (n + 1) + np.maximum(pu, pv))
    pa, pb = (key // (n + 1)).astype(int), (key % (n + 1)).astype(int)
    ln = ndimage.mean(_noise(H, W, 45, seed + 1), labels, index=np.arange(1, n + 1))
    rng = np.random.default_rng(seed)
    rnd = rng.random(len(pa))
    for i in rng.permutation(len(pa)):
        a, b = pa[i], pb[i]
        if rnd[i] < strength * (ln[a - 1] + 1) / 2:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([find(i) for i in range(n + 1)])
    _, newids = np.unique(roots, return_inverse=True)
    return newids[labels].astype(int)


def _clip(lab, mask):
    """Clip any full/partial label image to mask, relabel 1..n (0 outside)."""
    out = np.where(mask, lab.astype(np.int64) + 1, 0)
    u, inv = np.unique(out, return_inverse=True)
    res = inv.reshape(out.shape)
    if u[0] != 0:                                # mask covered everything: shift off 0
        res = res + 1
    return res.astype(int)


def _knn_assign(py, px, seeds, metric, k=14, chunk=250_000):
    """Assign each (py,px) to its metric-nearest seed, but only evaluating the k
    euclidean-nearest candidates (so anisotropic metrics stay O(n·k), not O(n·seeds)).
    metric(dy, dx, idx) -> distance, where idx are candidate seed indices."""
    tree = cKDTree(seeds)
    k = min(k, len(seeds))
    out = np.empty(len(py), int)
    for i in range(0, len(py), chunk):
        P = np.column_stack([py[i:i + chunk], px[i:i + chunk]]).astype(float)
        idx = tree.query(P, k=k)[1]
        if idx.ndim == 1:
            idx = idx[:, None]
        dy = P[:, 0, None] - seeds[idx, 0]; dx = P[:, 1, None] - seeds[idx, 1]
        d = metric(dy, dx, idx)
        out[i:i + chunk] = idx[np.arange(len(P)), d.argmin(1)]
    return out


# ---------------------------------------------------------------------------
# the families
# ---------------------------------------------------------------------------
def quads(mask, base, seed):
    """Hand-cut rectangular tesserae (the DC land family; ~44% right-angle corners)."""
    H, W = mask.shape
    lab = _quad_grid(H, W, base, seed, jit=0.24)
    lab = _merge_variance(lab, strength=0.35, seed=seed + 11)
    return _clip(lab, mask), None


def shards(mask, base, seed):
    """Interlocking glass shards (the DC parks family; low solidity, heavy size tail)."""
    H, W = mask.shape
    Y, X = np.mgrid[0:H, 0:W]
    amp = base * 0.95
    wy = amp * _noise(H, W, 7, seed + 1); wx = amp * _noise(H, W, 7, seed + 2)
    gpts = np.column_stack([(Y + wy).ravel(), (X + wx).ravel()])
    lab = cKDTree(_var_seeds(H, W, base * 1.25, seed, var=1.0)).query(gpts)[1].reshape(H, W)
    lab = _merge_variance(lab + 1, strength=0.45, seed=seed + 12)
    return _clip(lab, mask), None


def flow(mask, base, seed, elong=3.0):
    """Cells elongated along the region's current (the DC water family, generalised)."""
    H, W = mask.shape
    if not mask.any():
        return np.zeros((H, W), int), None
    sm = ndimage.gaussian_filter(ndimage.distance_transform_edt(mask).astype(float), 2.5)
    gy, gx = np.gradient(sm)
    ang = np.arctan2(gy, gx) + np.pi / 2            # along the current
    s = _jit_grid(H, W, base * 1.1, seed)
    si = np.clip(s[:, 0].astype(int), 0, H - 1); sj = np.clip(s[:, 1].astype(int), 0, W - 1)
    inm = mask[si, sj]; s, si, sj = s[inm], si[inm], sj[inm]
    if len(s) < 2:
        return mask.astype(int), None
    ca, sa = np.cos(ang[si, sj]), np.sin(ang[si, sj])
    py, px = np.where(mask)

    def metric(dy, dx, idx):
        u = dx * ca[idx] + dy * sa[idx]; v = -dx * sa[idx] + dy * ca[idx]
        return (u / elong) ** 2 + v ** 2

    ids = _knn_assign(py, px, s, metric, k=14)
    lab = np.zeros((H, W), int); lab[py, px] = ids
    return _clip(lab, mask), None


def pebbles(mask, base, seed):
    """Rounded river stones sitting in a fat, uneven grout bed. Voronoi cells are
    eroded (smoothed EDT threshold => rounded corners) with a per-stone radius
    jitter, and whatever is left between the stones becomes visible grout matrix."""
    H, W = mask.shape
    rng = np.random.default_rng(seed)
    Y, X = np.mgrid[0:H, 0:W]
    amp = base * 0.55
    wy = amp * _noise(H, W, 9, seed + 1); wx = amp * _noise(H, W, 9, seed + 2)
    gpts = np.column_stack([(Y + wy).ravel(), (X + wx).ravel()])
    ids = cKDTree(_jit_grid(H, W, base * 1.18, seed, jit=0.36)).query(gpts)[1].reshape(H, W)
    bound = np.zeros((H, W), bool)
    bound[:, :-1] |= ids[:, :-1] != ids[:, 1:]
    bound[:-1, :] |= ids[:-1, :] != ids[1:, :]
    bound |= mask & ~ndimage.binary_erosion(mask)   # region edge is a joint too
    ed = ndimage.distance_transform_edt(~bound)
    per = rng.uniform(0.80, 1.25, ids.max() + 1)    # per-stone erosion radius
    t = np.maximum(1.1, 0.07 * base * per[ids] *
                   (1 + 0.25 * _noise(H, W, max(4, int(base * 4)), seed + 3)))
    core = mask & (ed > t)                          # guaranteed gaps between stones
    sm = ndimage.gaussian_filter(core.astype(float), 0.14 * base)
    pebble = mask & (sm > 0.5) & (ed > 1.0)         # smoothing rounds; ed keeps gaps
    lab, n = ndimage.label(pebble)
    if n:                                           # crumbs sink into the grout bed
        areas = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
        crumb = np.isin(lab, np.nonzero(areas < (0.45 * base) ** 2)[0] + 1)
        pebble &= ~crumb
        lab[~pebble] = 0
    out = _clip(lab, pebble)
    out[~mask] = 0
    return out, mask & ~pebble


def strata(mask, base, seed, aniso=4.0):
    """Long thin slivers laid along a slowly wandering grain — layered slate on edge.
    Oriented Chebyshev cells (=> straight long sides), per-seed length jitter."""
    H, W = mask.shape
    if not mask.any():
        return np.zeros((H, W), int), None
    rng = np.random.default_rng(seed)
    s = _jit_grid(H, W, base * np.sqrt(aniso) * 0.9, seed, jit=0.40)
    grain = 0.9 * _noise(H, W, max(8, int(base * 9)), seed + 3)   # angle field (radians)
    si = np.clip(s[:, 0].astype(int), 0, H - 1); sj = np.clip(s[:, 1].astype(int), 0, W - 1)
    ca, sa = np.cos(grain[si, sj]), np.sin(grain[si, sj])
    an = aniso * rng.uniform(0.7, 1.4, len(s))      # per-sliver length variation
    py, px = np.where(mask)

    def metric(dy, dx, idx):
        u = dx * ca[idx] + dy * sa[idx]; v = -dx * sa[idx] + dy * ca[idx]
        return np.maximum(np.abs(u) / an[idx], np.abs(v))

    ids = _knn_assign(py, px, s, metric, k=18)
    lab = np.zeros((H, W), int); lab[py, px] = ids
    return _clip(lab, mask), None


def _noise1d(b, wavelen, rng):
    bmin, bmax = b.min(), b.max()
    span = max(bmax - bmin, 1.0)
    ncp = max(3, int(span / max(wavelen, 2)) + 2)
    xs = np.linspace(bmin, bmax, ncp)
    return np.interp(b, xs, rng.standard_normal(ncp))


def crackle(mask, base, seed, rough=0.22, freq=0.7, sel_pow=0.9):
    """Dried-mud shatter: recursively split the region with JAGGED cuts (from
    tools/experiments/fracture.py). Heavy-tailed sizes + concave interlocking
    edges — rawer than shards."""
    H, W = mask.shape
    pix0 = np.argwhere(mask)
    if len(pix0) == 0:
        return np.zeros((H, W), int), None
    rng = np.random.default_rng(seed)
    n_pieces = max(1, int(len(pix0) / (base * base * 1.7)))
    coords = {1: pix0}
    nxt, tries = 2, 0
    while len(coords) < n_pieces and tries < 6 * n_pieces:
        tries += 1
        labs = np.fromiter(coords.keys(), int)
        w = np.array([len(coords[l]) for l in labs], float) ** sel_pow
        p = int(rng.choice(labs, p=w / w.sum()))
        pts = coords[p]
        if len(pts) < 24:
            continue
        # cut ACROSS the piece's long axis (PCA) + jitter => no razor-thin slivers
        c = pts - pts.mean(0)
        cov = c.T @ c
        evec = np.linalg.eigh(cov)[1][:, -1]                # long axis (y, x)
        a = np.arctan2(evec[0], evec[1]) + rng.uniform(-0.5, 0.5)
        d = pts[:, 0] * np.sin(a) + pts[:, 1] * np.cos(a)   # across the cut
        b = pts[:, 0] * np.cos(a) - pts[:, 1] * np.sin(a)   # along the cut
        scale = np.sqrt(len(pts))
        thr = np.median(d) + rough * scale * _noise1d(b, freq * scale, rng)
        m = d > thr
        if m.sum() < 0.18 * len(pts) or (~m).sum() < 0.18 * len(pts):
            continue
        coords[p] = pts[~m]; coords[nxt] = pts[m]; nxt += 1
    lab = np.zeros((H, W), int)
    for l, pts in coords.items():
        lab[pts[:, 0], pts[:, 1]] = l
    return _clip(lab, mask), None


def rings(mask, base, seed):
    """Wobbly growth rings hugging the region's boundary (warped distance-transform
    bands chopped into courses of segments) — tree rings, not neat outlines."""
    H, W = mask.shape
    if not mask.any():
        return np.zeros((H, W), int), None
    ed = ndimage.distance_transform_edt(mask)
    warp = ed + 0.6 * base * _noise(H, W, max(4, int(base * 5)), seed + 1)
    band = np.floor(warp / (0.85 * base)).astype(int)
    band -= band.min()
    Y, X = np.mgrid[0:H, 0:W]
    cell = cKDTree(_jit_grid(H, W, base * 2.3, seed, jit=0.45)).query(
        np.column_stack([Y.ravel(), X.ravel()]))[1].reshape(H, W)
    comb = band.astype(np.int64) * (cell.max() + 2) + cell
    return _clip(comb, mask), None


def disc(mask, base, seed):
    """One smooth piece — the locator (rendered with a sheen by the composer)."""
    return mask.astype(int), None


FAMILIES = {
    "quads":   dict(fn=quads,   blurb="hand-cut rectangular tesserae"),
    "shards":  dict(fn=shards,  blurb="interlocking glass shards"),
    "flow":    dict(fn=flow,    blurb="long cells stretched along the region's flow"),
    "pebbles": dict(fn=pebbles, blurb="rounded stones in a fat grout bed"),
    "strata":  dict(fn=strata,  blurb="slate slivers along a wandering grain"),
    "crackle": dict(fn=crackle, blurb="dried-mud shatter, heavy size tail"),
    "rings":   dict(fn=rings,   blurb="wobbly growth rings hugging the edge"),
    "disc":    dict(fn=disc,    blurb="one smooth piece (locator)"),
}


# ---------------------------------------------------------------------------
# composing + rendering (dc_build-style, generalised to any family assignment)
# ---------------------------------------------------------------------------
def compose(region, assignment, bases, seed=3):
    """region: int class map (0 = outside). assignment: {class: family_name}.
    bases: {class: px} or a single float. -> (big, nbig, forced_grout)"""
    H, W = region.shape
    big = np.zeros((H, W), int)
    forced = np.zeros((H, W), bool)
    for cls in sorted(assignment):
        mask = region == cls
        if not mask.any():
            continue
        b = bases[cls] if isinstance(bases, dict) else bases
        lab, g = FAMILIES[assignment[cls]]["fn"](mask, b, seed + 13 * cls)
        if g is not None:
            forced |= g
            lab = lab.copy()
            lab[mask & (lab == 0)] = int(lab.max()) + 1   # grout pool gets its own id
        off = int(big.max())
        big[mask] = lab[mask] + off
    return big, int(big.max()), forced


def color_lut(big, nbig, region, palettes, seed=3):
    """Per-tile colour from its class palette + dc_build's tonal rules.
    palettes: {class: array of RGB rows in 0..1}.

    Each region draws from its OWN seeded stream (not one shared stream walked in tile order),
    so a region's colours depend only on itself — changing the family of one region never
    recolours the others. (Tile shapes are already independent per region.)"""
    counts = np.zeros((nbig + 1, int(region.max()) + 1), int)
    np.add.at(counts, (big.ravel(), region.ravel()), 1)
    tile_region = counts.argmax(1)
    lut = np.tile(WALL, (nbig + 1, 1))
    ids_by_region = {}                               # region -> its tile ids, in stable order
    for i in range(1, nbig + 1):
        r = tile_region[i]
        if r in palettes:
            ids_by_region.setdefault(r, []).append(i)
    for r, ids in ids_by_region.items():
        pal = palettes[r]
        rng = np.random.default_rng(int(seed) * 101 + int(r))   # this region's own stream
        picks = rng.integers(len(pal), size=len(ids))
        jit = rng.uniform(0.84, 1.14, size=len(ids))
        for k, i in enumerate(ids):
            c = np.asarray(pal[picks[k]], float)
            lum = float(np.mean(c))
            if r == 3:                               # water: muted blue ramp
                c = np.clip(np.array([0.33, 0.52, 0.78]) * (0.5 + 0.75 * lum), 0, 1)
            elif r == 1:                             # land: neutral-cool grey
                c = np.clip(0.72 * np.array([lum * 0.97, lum, lum * 1.04]) + 0.28 * c, 0, 1)
            elif r == 2:                             # parks: boost green saturation
                c = np.clip(np.array([c[0] * 0.85, c[1] * 1.05, c[2] * 0.8]), 0, 1)
            lut[i] = np.clip(c * jit[k], 0, 1)
    return lut, tile_region


def render_tiles(big, nbig, region, lut, forced_grout=None, gold_class=4, grout_width=None):
    """Bevel/emboss + grout, exactly dc_build's look, plus forced-grout beds.
    grout_width: if set, a fixed grout half-width in px (thin uniform lines); otherwise the
    default proportional grout (~5% of each tile's diameter)."""
    H, W = big.shape
    areas = ndimage.sum(np.ones_like(big), big, index=range(1, nbig + 1))
    out = lut[big].astype(float)
    bound = np.zeros((H, W), bool)
    bound[:, :-1] |= big[:, :-1] != big[:, 1:]
    bound[:-1, :] |= big[:-1, :] != big[1:, :]
    ed = ndimage.distance_transform_edt(~bound)
    g = ndimage.gaussian_filter(ed, 0.7); gy, gx = np.gradient(g)
    emboss = np.clip(-(gx + gy) / 1.4, -1, 1)
    shade = (1 + 0.14 * emboss) * (0.80 + 0.20 * np.clip(ed / 3.0, 0, 1))
    nz = ndimage.gaussian_filter(np.random.default_rng(2).standard_normal((H, W)), 1.3)
    shade *= 1 + 0.06 * nz
    out *= shade[..., None]
    # Grout as a constant FRACTION of each tile's own diameter (not a global fixed width),
    # so the coloured fill covers a consistent share of every tile — small/thin tiles (narrow
    # water, elongated 'flow'/'strata') keep their colour instead of being eaten by grout, and
    # the same region reads the same size across shape families.
    if grout_width is not None:
        grout_px = ed < grout_width                  # thin uniform grout
    else:
        diam_t = np.zeros(nbig + 1)
        diam_t[1:] = np.sqrt(4.0 * np.maximum(areas, 1.0) / np.pi)
        half = np.clip(0.05 * diam_t[big], 0.7, None)  # grout half-width ~5% of local tile diameter
        grout_px = ed < half
    if forced_grout is not None:
        grout_px |= forced_grout
    out[grout_px] = GROUT * 0.92
    out[region == 0] = WALL
    cap = region == gold_class
    if cap.any():                                    # locator disc: smooth gold sheen
        ed4 = ndimage.distance_transform_edt(cap)
        sheen = 0.78 + 0.42 * ed4 / (ed4.max() or 1.0)
        out[cap] = np.clip(np.array([0.82, 0.64, 0.15]) * sheen[..., None], 0, 1)[cap]
    return np.clip(out, 0, 1)


# ---------------------------------------------------------------------------
# contact sheet: python3 src/families.py -> output/family_swatches.png
# ---------------------------------------------------------------------------
def _blob_mask(N, seed):
    yy, xx = np.mgrid[0:N, 0:N]
    d = np.hypot(yy - N / 2, xx - N / 2)
    f = _noise(N, N, N // 3, seed)
    return d < N * 0.40 * (1 + 0.28 * f)


def main():
    cm = json.load(open(ROOT / "data" / "color_model.json"))
    pal = {1: np.array(cm["gray"]), 2: np.array(cm["green"]), 3: np.array(cm["light"])}
    N, base = 360, 13
    # show each family in the palette class it would most likely fill
    panels = [("quads", 1), ("strata", 1), ("crackle", 2), ("shards", 2),
              ("rings", 2), ("flow", 3), ("pebbles", 3)]
    cols, rows, pad, cap = 4, 2, 14, 26
    sheet = Image.new("RGB", (cols * (N + pad) + pad, rows * (N + pad + cap) + pad),
                      tuple((WALL * 255).astype(int)))
    drw = ImageDraw.Draw(sheet)
    mask = _blob_mask(N, seed=5)
    for k, (name, cls) in enumerate(panels):
        region = np.where(mask, cls, 0)
        big, nbig, forced = compose(region, {cls: name}, base, seed=7)
        lut, _ = color_lut(big, nbig, region, pal, seed=cls + 1)
        out = render_tiles(big, nbig, region, lut, forced)
        im = Image.fromarray((out * 255).astype("uint8"))
        x = pad + (k % cols) * (N + pad); y = pad + (k // cols) * (N + pad + cap)
        sheet.paste(im, (x, y))
        drw.text((x + 4, y + N + 6), f"{name} — {FAMILIES[name]['blurb']}",
                 fill=(200, 195, 180))
    dest = ROOT / "output" / "family_swatches.png"
    dest.parent.mkdir(exist_ok=True)
    sheet.save(dest)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
