"""Seeding-strategy experiment: does HOW we seed the clustering change how
"gerrymandered" the tiles come out?

Three ways to cut the country into ~K-person tiles, run on the SAME tract graph:

  current   bottom-up growth, RANDOM seed order            (what the site does today)
  density   bottom-up growth, seed densest tracts FIRST     (commenter hypothesis 1)
  rcb       top-down recursive bisection (split the biggest
            region along its long axis at the pop-weighted
            median, recurse to round(total/K) tiles)        (Gurvich hypothesis 2)

"Gerrymandered" is scored with Polsby-Popper compactness on the rasterised tile
shapes (4*pi*Area / Perimeter**2; 1.0 = a circle, low = a contorted tendril) --
the standard redistricting compactness measure. We also track the two things the
comments predicted would move: the count of tiny "Vatican city" tiles, and the
population band.

    python tools/seeding_experiment.py --k 1000000 --height 1100

Outputs land in output/seeding/.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools")); sys.path.insert(0, str(ROOT / "src"))
import tiles as pm                                       # noqa: E402
import build_national as bn                              # noqa: E402
import tiles_us as us                                    # noqa: E402

OUT = ROOT / "output" / "seeding"
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# base data (project tracts, land mask, adjacency, density) -- cached
# ---------------------------------------------------------------------------
def build_base(S, k, seed):
    cache = OUT / f"_base_{S}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return (z["points"], z["pop1"], z["cent"], z["land"], int(z["W"]), int(z["H"]),
                z["nb"].tolist(), z["dens"])
    panel = bn._build_panel(bn.CONUS, S, "albers")
    to_px, W, H = panel["to_px"], panel["W"], panel["H"]
    state, water, _ = bn._panel_masks(panel)
    wmin = max(12, round((S / 300.0) ** 2))
    water = us.clean_water(water, wmin, max(1, round(S / 900)))
    land = state & ~water

    lon, lat, pop = us.load_tracts()
    px, py = to_px(lon, lat)
    inb = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    px, py, pop = px[inb], py[inb], pop[inb]
    keep = state[np.round(py).astype(int), np.round(px).astype(int)]
    points = np.column_stack([px, py])[keep]; pop = pop[keep]
    n = len(points)

    nb = us.delaunay_adj(points, n)
    cent = np.vstack([[0, 0], points]).astype(float)
    pop1 = np.concatenate([[0], pop]).astype(float)

    # local density proxy: tracts are ~equal-population, so density ~ 1/area, and
    # a tract's "area" scales like the median squared distance to its graph neighbours.
    dens = np.zeros(n + 1)
    for i in range(1, n + 1):
        if not nb[i]:
            continue
        d2 = [(cent[i, 0] - cent[j, 0]) ** 2 + (cent[i, 1] - cent[j, 1]) ** 2 for j in nb[i]]
        dens[i] = pop1[i] / max(np.median(d2), 1e-6)

    np.savez_compressed(cache, points=points, pop1=pop1, cent=cent, land=land,
                        W=W, H=H, nb=np.array(nb, dtype=object), dens=dens)
    return points, pop1, cent, land, W, H, nb, dens


# ---------------------------------------------------------------------------
# strategy 1 & 2: bottom-up growth with a CHOSEN seed order
# (lifted from tiles.regionalize; the only change is `order` is passed in)
# ---------------------------------------------------------------------------
def grow_ordered(nb, pop, K, order, cent, mode="compact"):
    n = len(pop) - 1
    cl = np.zeros(n + 1, int)
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
            else:
                nxt = list(frontier)[0]
            if cpop >= K or (cpop + pop[nxt] - K) >= (K - cpop):
                break
            cl[nxt] = cid
            np_ = cpop + pop[nxt]
            cx = (cx * cpop + cent[nxt, 0] * pop[nxt]) / np_
            cy = (cy * cpop + cent[nxt, 1] * pop[nxt]) / np_
            cpop = np_
            frontier.discard(nxt)
            frontier |= {j for j in nb[nxt] if not cl[j]}
    # same tail-cleanup as the production regionalize
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
    cl = pm.enforce_count(cl, nb, cent, pop, K)
    uniq = np.unique(cl[cl > 0])
    relab = np.zeros(cl.max() + 1, int); relab[uniq] = np.arange(1, len(uniq) + 1)
    return relab[cl], len(uniq)


# ---------------------------------------------------------------------------
# strategy 3: top-down recursive coordinate bisection
# ---------------------------------------------------------------------------
def rcb(pop1, cent, K):
    """Recursive coordinate bisection -> exactly round(total/K) compact tiles.
    Split the region along its longer extent at the population-weighted point that
    gives floor(m/2) tiles' worth of people to one side."""
    labels = np.arange(1, len(pop1))                       # tract labels 1..n
    cl = np.zeros(len(pop1), int)
    next_id = [0]

    def split(idx):
        P = pop1[idx].sum()
        m = max(1, int(round(P / K)))
        if m <= 1 or len(idx) <= 1:
            next_id[0] += 1
            cl[idx] = next_id[0]
            return
        xs, ys = cent[idx, 0], cent[idx, 1]
        axis = 0 if (xs.max() - xs.min()) >= (ys.max() - ys.min()) else 1
        key = cent[idx, axis]
        o = np.argsort(key, kind="stable")
        idx_s = idx[o]
        cum = np.cumsum(pop1[idx_s])
        target = P * (m // 2) / m                           # people for the left child
        cut = int(np.searchsorted(cum, target)) + 1
        cut = min(max(cut, 1), len(idx_s) - 1)
        split(idx_s[:cut]); split(idx_s[cut:])

    split(labels)
    return cl, next_id[0]


# ---------------------------------------------------------------------------
# rasterise a clustering and score it
# ---------------------------------------------------------------------------
def rasterize(cl, points, land, W, H):
    tree = cKDTree(points)
    ys, xs = np.nonzero(land)
    _, idx = tree.query(np.column_stack([xs, ys]), workers=-1)
    big = np.zeros((H, W), int)
    big[ys, xs] = cl[idx + 1]
    return big


def polsby_popper(big, nbig):
    """Per-tile 4*pi*Area/Perimeter**2 from the raster. Perimeter counts every pixel
    edge where the tile meets a different tile, water, or the map edge (4-connectivity).
    The standard redistricting metric, but raster-staircase-biased, so we lead with MOI."""
    H, W = big.shape
    area = np.bincount(big.ravel(), minlength=nbig + 1).astype(float)
    perim = np.zeros(nbig + 1)
    pad = np.zeros((H + 2, W + 2), int); pad[1:-1, 1:-1] = big
    c = pad[1:-1, 1:-1]
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nbr = pad[1 + dy:H + 1 + dy, 1 + dx:W + 1 + dx]
        edge = (c > 0) & (nbr != c)                         # boundary of THIS tile
        np.add.at(perim, c[edge], 1)
    pp = np.zeros(nbig + 1)
    ok = (perim > 0) & (area > 0)
    pp[ok] = 4 * np.pi * area[ok] / perim[ok] ** 2
    return pp[1:]                                           # drop id 0 (outside)


def moi_compactness(big, nbig):
    """Per-tile moment-of-inertia compactness: (Area**2 / 2pi) / sum_pixels r**2 about the
    tile's own centroid. 1.0 = a perfect disk, lower = more spread out / tendrilly. Uses
    area not perimeter, so it's free of the raster-staircase bias that flattens Polsby-Popper.
    A recognised redistricting compactness measure (the 'moment of inertia' / dispersion score)."""
    H, W = big.shape
    Y, X = np.mgrid[0:H, 0:W]
    flat = big.ravel()
    area = np.bincount(flat, minlength=nbig + 1).astype(float)
    sx = np.bincount(flat, weights=X.ravel().astype(float), minlength=nbig + 1)
    sy = np.bincount(flat, weights=Y.ravel().astype(float), minlength=nbig + 1)
    sxx = np.bincount(flat, weights=(X.ravel().astype(float)) ** 2, minlength=nbig + 1)
    syy = np.bincount(flat, weights=(Y.ravel().astype(float)) ** 2, minlength=nbig + 1)
    a = np.maximum(area, 1)
    moi = (sxx - sx ** 2 / a) + (syy - sy ** 2 / a)         # sum of squared distance to centroid
    comp = np.zeros(nbig + 1)
    ok = (area > 0) & (moi > 0)
    comp[ok] = (area[ok] ** 2 / (2 * np.pi)) / moi[ok]
    return comp[1:], area[1:]                               # drop id 0 (outside)


def summarize(name, cl, big, nbig, pop1, K):
    comp, area = moi_compactness(big, nbig)
    pp = polsby_popper(big, nbig)
    cpop = np.bincount(cl, weights=pop1, minlength=nbig + 1)[1:]
    cpop = cpop[cpop > 0]
    med_area = np.median(area[area > 0])
    tiny = int((area[area > 0] < 0.10 * med_area).sum())
    cc = comp[comp > 0]
    return {
        "name": name, "tiles": int((area > 0).sum()),
        "moi_median": float(np.median(cc)), "moi_mean": float(np.mean(cc)),
        "moi_frac_lt_0.3": float((cc < 0.30).mean()),
        "pp_median": float(np.median(pp[pp > 0])),
        "tiny_tiles": tiny,
        "pop_p10": float(np.percentile(cpop, 10)), "pop_p90": float(np.percentile(cpop, 90)),
        "pop_cv": float(cpop.std() / cpop.mean()),
        "_comp": cc, "_area": area[area > 0], "_big": big, "_nbig": nbig,
    }


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=1_000_000)
    ap.add_argument("--height", type=int, default=1100)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--balance", action="store_true",
                    help="also run balance(10) downstream of each seeding (matches site more)")
    args = ap.parse_args()
    K, S = args.k, args.height
    t0 = time.time()
    tick = lambda m: print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

    tick(f"building base data (height={S}) ...")
    points, pop1, cent, land, W, H, nb, dens = build_base(S, K, args.seed)
    n = len(pop1) - 1
    tick(f"{n:,} tracts, {pop1.sum():,.0f} people, ~{round(pop1.sum()/K)} tiles target")

    rng = np.random.default_rng(args.seed)
    runs = {}

    tick("strategy: current (random seed order) ...")
    cl, M = grow_ordered(nb, pop1, K, rng.permutation(np.arange(1, n + 1)), cent)
    runs["current"] = (cl, M)

    tick("strategy: density-peak descending ...")
    order = np.argsort(-dens[1:]) + 1                       # densest tract first
    cl, M = grow_ordered(nb, pop1, K, order, cent)
    runs["density"] = (cl, M)

    tick("strategy: recursive bisection (RCB) ...")
    cl, M = rcb(pop1, cent, K)
    runs["rcb"] = (cl, M)

    if args.balance:
        for name in list(runs):
            tick(f"  balancing {name} ...")
            cl, M = runs[name]
            cl = pm.balance(cl, nb, pop1, K, iters=10, seed=args.seed)
            uniq = np.unique(cl[cl > 0])
            relab = np.zeros(int(cl.max()) + 1, int); relab[uniq] = np.arange(1, len(uniq) + 1)
            runs[name] = (relab[cl], len(uniq))

    results = []
    for name, (cl, M) in runs.items():
        tick(f"rasterise + score {name} ...")
        big = rasterize(cl, points, land, W, H)
        results.append(summarize(name, cl, big, M, pop1, K))

    # ---- table -----------------------------------------------------------
    print("\n=== gerrymandering by seeding strategy "
          f"(K={K:,}, balance={'on' if args.balance else 'off'}) ===")
    print("MOI = moment-of-inertia compactness (1.0 = disk, higher = rounder); PP = Polsby-Popper")
    hdr = (f"{'strategy':10} {'tiles':>6} {'MOI med':>8} {'MOI mean':>9} {'%MOI<.3':>8} "
           f"{'PP med':>7} {'tiny':>5} {'pop p10-p90':>20} {'pop CV':>7}")
    print(hdr); print("-" * len(hdr))
    for r in results:
        print(f"{r['name']:10} {r['tiles']:6d} {r['moi_median']:8.3f} {r['moi_mean']:9.3f} "
              f"{r['moi_frac_lt_0.3']*100:7.1f}% {r['pp_median']:7.3f} {r['tiny_tiles']:5d} "
              f"{r['pop_p10']:8,.0f}-{r['pop_p90']:8,.0f} {r['pop_cv']:6.3f}")

    save_figs(results, K, args.balance)
    tick(f"figures -> {OUT.relative_to(ROOT)}/")


def save_figs(results, K, balanced):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tag = "balanced" if balanced else "seedonly"
    nice = {"current": "Current\n(random order)", "density": "Density-peak\n(densest first)",
            "rcb": "Recursive\nbisection"}

    # 1) compactness distribution: box + median labels
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    data = [r["_comp"] for r in results]
    labs = [nice.get(r["name"], r["name"]) for r in results]
    bp = ax.boxplot(data, tick_labels=labs, showfliers=False, widths=0.5, patch_artist=True)
    for patch, col in zip(bp["boxes"], ["#c0654a", "#5a8f9a", "#7a9a5a"]):
        patch.set_facecolor(col); patch.set_alpha(0.65)
    for i, r in enumerate(results):
        ax.text(i + 1, r["moi_median"], f"  {r['moi_median']:.2f}", va="center", fontsize=9, color="#222")
    ax.set_ylabel("Moment-of-inertia compactness\n(1.0 = disk, lower = more gerrymandered)")
    ax.set_title(f"Tile compactness by seeding strategy  (K={K:,}, {tag})")
    ax.axhline(0.30, ls="--", lw=0.8, color="#999")
    ax.text(0.02, 0.31, "0.30 'gerrymandered' line", transform=ax.get_yaxis_transform(),
            fontsize=8, color="#777")
    fig.tight_layout(); fig.savefig(OUT / f"compactness_{tag}.png", dpi=140); plt.close(fig)

    # 2) the maps, side by side, random colour per tile
    fig, axes = plt.subplots(1, len(results), figsize=(5.2 * len(results), 4.2))
    if len(results) == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        big, nbig = r["_big"], r["_nbig"]
        rng = np.random.default_rng(0)
        lut = np.vstack([[1, 1, 1], rng.random((nbig, 3)) * 0.7 + 0.15])
        img = lut[np.clip(big, 0, nbig)]
        img[big == 0] = [0.12, 0.12, 0.14]
        ax.imshow(img); ax.set_axis_off()
        ax.set_title(f"{r['name']}  (MOI med {r['moi_median']:.2f}, {r['tiny_tiles']} tiny)", fontsize=10)
    fig.suptitle(f"Equal-population tiles by seeding strategy (K={K:,}, {tag})", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / f"maps_{tag}.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
