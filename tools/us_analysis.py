"""Two national analyses (no rendering needed -- pure clustering, so it's quick):

  1. ACCURACY: how close to a target headcount every tile lands, vs the target. The band
     tightens as targets grow (more tracts averaged per tile), then is bounded below by the
     ~4,000-person tract atom.
  2. UNIQUENESS: is there one way to draw the buckets, or many? Cluster each target from several
     random seeds and measure partition agreement (Rand index). High agreement = the constraints
     pin a near-unique answer; lower = real freedom.

    python tools/us_analysis.py
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import Delaunay

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools")); sys.path.insert(0, str(ROOT / "src"))
import pop_mosaic as pm                                # noqa: E402
import pop_mosaic_us as pu                             # noqa: E402
import build_national as bn                            # noqa: E402
from shape_compare import rand_index                   # noqa: E402

BG, INK, MUT, GOLD, TEAL, GRID = (25, 20, 15), (236, 228, 214), (150, 139, 120), \
    (232, 198, 106), (90, 178, 178), (60, 54, 44)


def graph():
    """All CONUS tract centres -> (adjacency nb, pop1, cent). Albers coords; no land mask."""
    proj = bn._albers()
    lon, lat, pop = pu.load_tracts()
    X, Y = proj(lon, lat)
    pts = np.column_stack([X, Y])
    n = len(pts)
    nb = pu.delaunay_adj(pts, n)
    return nb, np.concatenate([[0], pop]).astype(float), np.vstack([[0, 0], pts]).astype(float)


def cluster(nb, pop, cent, K, seed):
    cl, _ = pm.regionalize(nb, pop, K, seed, "compact", cent)
    cl = pm.balance(cl, nb, pop, K, iters=8, seed=seed)
    uniq = np.unique(cl[cl > 0])
    relab = np.zeros(int(cl.max()) + 1, int); relab[uniq] = np.arange(1, len(uniq) + 1)
    return relab[cl]


def fmt_k(v):
    return f"{v/1e6:.0f}M" if v >= 1e6 else f"{v/1e3:.0f}k"


def _axes(d, L, R, T, B, xs_log, xticks, yticks, ylab, xlab, ylo, yhi, f):
    for v in yticks:
        y = B - (v - ylo) / (yhi - ylo) * (B - T)
        d.line([(L, y), (R, y)], fill=GRID); d.text((L - 14, y), f"{v:g}", font=f, fill=MUT, anchor="rm")
    for t in xticks:
        x = L + (np.log10(t) - xs_log[0]) / (xs_log[1] - xs_log[0]) * (R - L)
        d.text((x, B + 12), fmt_k(t), font=f, fill=MUT, anchor="ma")
    d.text(((L + R) // 2, B + 56), xlab, font=f, fill=INK, anchor="ma")
    d.text((L, T - 34), ylab, font=f, fill=INK)


def main():
    print("building national tract graph ...", flush=True)
    nb, pop, cent = graph()
    print(f"{len(pop)-1:,} tracts, {pop.sum():,.0f} people\n")

    acc_T = [25_000, 60_000, 150_000, 400_000, 1_000_000, 2_500_000, 6_000_000, 15_000_000]
    uni_T = [50_000, 200_000, 800_000, 3_000_000, 12_000_000]
    cache = ROOT / "output" / "_us_analysis.npz"
    if cache.exists():                                         # redraws are then instant
        z = np.load(cache)
        p = {q: z[f"p{q}"] for q in (10, 25, 50, 75, 90)}
        agree, ntiles = z["agree"], z["ntiles"]
    else:
        p = {q: [] for q in (10, 25, 50, 75, 90)}
        for K in acc_T:                                       # 1. accuracy funnel
            cl = cluster(nb, pop, cent, K, 1)
            r = np.bincount(cl, weights=pop, minlength=int(cl.max()) + 1)[1:] / K * 100
            for q in p:
                p[q].append(np.percentile(r, q))
            print(f"  acc  {K:>10,}: {int(cl.max()):>5} tiles  band {p[10][-1]:.0f}-{p[90][-1]:.0f}%")
        agree, ntiles = [], []
        for K in uni_T:                                       # 2. uniqueness vs target
            cls = [cluster(nb, pop, cent, K, s) for s in (1, 2, 3)]
            rs = [rand_index(cls[i], cls[j]) for i in range(3) for j in range(i + 1, 3)]
            agree.append(float(np.mean(rs))); ntiles.append(int(cls[0].max()))
            print(f"  uniq {K:>10,}: {ntiles[-1]:>5} tiles  agreement {agree[-1]:.3f}")
        np.savez(cache, agree=agree, ntiles=ntiles, **{f"p{q}": p[q] for q in p})
    p = {q: np.asarray(v) for q, v in p.items()}

    f = pm._font  # font helper
    # ----- draw accuracy -----
    W, H = 1500, 950
    img = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(img)
    L, R, T, B = 165, W - 60, 215, H - 150
    xl = (np.log10(acc_T[0]), np.log10(acc_T[-1]))
    X = lambda t: L + (np.log10(t) - xl[0]) / (xl[1] - xl[0]) * (R - L)
    YLO, YHI = 92, 114
    Y = lambda v: B - (np.clip(v, YLO, YHI) - YLO) / (YHI - YLO) * (B - T)
    d.text((L, 64), "How equal can the tiles be?", font=f(42, bold=True), fill=INK)
    d.text((L, 124), "Every US tile aims for a target headcount; the band shows where they land.",
           font=f(24), fill=MUT)
    _axes(d, L, R, T, B, xl, [25_000, 100_000, 1_000_000, 15_000_000],
          [95, 100, 105, 110], "tile population, as % of its target",
          "target people per tile  (log scale)", YLO, YHI, f(22))
    def band(lo, hi, c):
        pts = [(X(t), Y(v)) for t, v in zip(acc_T, hi)] + [(X(t), Y(v)) for t, v in zip(acc_T[::-1], lo[::-1])]
        d.polygon(pts, fill=c)
    band(p[10], p[90], (60, 52, 34)); band(p[25], p[75], (110, 92, 44))
    d.line(list(zip([X(t) for t in acc_T], [Y(v) for v in p[50]])), fill=GOLD, width=4, joint="curve")
    y100 = Y(100)
    for x in range(L, R, 16):
        d.line([(x, y100), (x + 8, y100)], fill=TEAL, width=3)
    d.text((R - 6, y100 - 28), "100% = exactly on target", font=f(22), fill=TEAL, anchor="ra")
    d.text((L, H - 44), "Bigger tiles average over more tracts, so the band tightens toward 100%; "
           "the ~4,000-person tract is the atom that sets the floor.", font=f(22), fill=MUT)
    img.save(ROOT / "output" / "us_accuracy.png")

    # ----- draw uniqueness -----
    img2 = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(img2)
    xl = (np.log10(uni_T[0]), np.log10(uni_T[-1]))
    X = lambda t: L + (np.log10(t) - xl[0]) / (xl[1] - xl[0]) * (R - L)
    YLO, YHI = 90, 100
    Y = lambda v: B - (np.clip(v, YLO, YHI) - YLO) / (YHI - YLO) * (B - T)
    d.text((L, 64), "One way to draw them, or many?", font=f(42, bold=True), fill=INK)
    d.text((L, 124), "Agreement between two random seeds at the same target (100% = identical buckets).",
           font=f(24), fill=MUT)
    _axes(d, L, R, T, B, xl, [50_000, 200_000, 800_000, 3_000_000, 12_000_000],
          [90, 92, 94, 96, 98, 100], "partition agreement between seeds (%)",
          "target people per tile  (log scale)", YLO, YHI, f(22))
    ag = [a * 100 for a in agree]
    d.line(list(zip([X(t) for t in uni_T], [Y(v) for v in ag])), fill=GOLD, width=4, joint="curve")
    for i, (t, v, nt) in enumerate(zip(uni_T, ag, ntiles)):
        d.ellipse([X(t) - 6, Y(v) - 6, X(t) + 6, Y(v) + 6], fill=GOLD)
        anc = "la" if i == 0 else ("ra" if i == len(uni_T) - 1 else "ma")
        dx = 12 if i == 0 else (-12 if i == len(uni_T) - 1 else 0)
        d.text((X(t) + dx, Y(v) + 16), f"{nt:,} tiles", font=f(20), fill=MUT, anchor=anc)
    d.text((L, H - 44), "Few big tiles -> many equally-valid bucketings (more disagreement). Many "
           "small tiles -> the constraints pin a near-unique answer.", font=f(22), fill=MUT)
    img2.save(ROOT / "output" / "us_uniqueness.png")
    print("\n-> output/us_accuracy.png, output/us_uniqueness.png")


if __name__ == "__main__":
    main()
