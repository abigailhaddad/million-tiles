"""How much do the shape RULES (compact / organic / tendril) actually change the tiling, vs
the equal-population + contiguity constraints forcing similar results? Builds the graph once,
clusters under each rule (and a 2nd seed of each), then measures:

  - compactness (Polsby-Popper) of the tiles each rule produces
  - partition agreement (Rand index): how often two block groups share a tile under one
    setting AND under another. Cross-RULE agreement vs same-rule-different-SEED agreement
    tells us whether the rule matters more than random chance.

    python tools/shape_compare.py "District of Columbia" --k 9000
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools")); sys.path.insert(0, str(ROOT / "src"))
import pop_mosaic as pm                                # noqa: E402
import pop_sweep as ps                                 # noqa: E402
import state_data as sd                                # noqa: E402


def cluster(nb, pop, cent, K, seed, shape):
    cl, _ = pm.regionalize(nb, pop, K, seed, shape, cent)
    cl = pm.balance(cl, nb, pop, K, seed=seed)
    uniq = np.unique(cl[cl > 0])
    relab = np.zeros(int(cl.max()) + 1, int); relab[uniq] = np.arange(1, len(uniq) + 1)
    return relab[cl]


def rand_index(a, b):
    """Agreement between two partitions of the same items (block groups 1..n)."""
    items = a[1:], b[1:]
    n = len(items[0])
    # contingency counts via dict
    from collections import Counter
    cont = Counter(zip(items[0].tolist(), items[1].tolist()))
    ca = Counter(items[0].tolist()); cb = Counter(items[1].tolist())
    comb2 = lambda x: x * (x - 1) // 2
    tp = sum(comb2(v) for v in cont.values())
    ap = sum(comb2(v) for v in ca.values())
    bp = sum(comb2(v) for v in cb.values())
    total = comb2(n)
    return (total - ap - bp + 2 * tp) / total           # Rand index


def compactness(cl, big):
    """Mean Polsby-Popper (4*pi*area/perimeter^2; 1=circle) over the tiles cl paints in big."""
    tm = cl[big]
    nb = int(tm.max())
    area = np.bincount(tm.ravel(), minlength=nb + 1).astype(float)
    bnd = np.zeros(tm.shape, bool)
    bnd[:, :-1] |= tm[:, :-1] != tm[:, 1:]
    bnd[:-1, :] |= tm[:-1, :] != tm[1:, :]
    per = np.bincount(tm[bnd].ravel(), minlength=nb + 1).astype(float)
    pp = 4 * np.pi * area[1:] / np.maximum(per[1:], 1) ** 2
    return float(np.median(np.clip(pp, 0, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("state"); ap.add_argument("--k", type=int, default=9000)
    ap.add_argument("--height", type=int, default=1200)
    args = ap.parse_args()
    ST = sd.resolve_state(args.state); fips = pm.state_fips(ST)
    print(f"building graph for {sd.STATE_NAMES[ST]} ...", flush=True)
    nb, pop, cent, nbig = ps.build_graph(ST, fips, args.height)

    # need big (the bg label raster) for compactness; rebuild it
    mask, _w, to_px, _ = pm.state_mask(ST, args.height)
    H, W = mask.shape
    feats = [f for f in pm.fetch_bg_polys(ST, fips)["features"] if f.get("geometry")]
    big = np.zeros((H, W), int)
    for i, f in enumerate(feats, 1):
        big[sd.rasterize_px([f], to_px, W, H) & mask] = i
    ind = ndimage.distance_transform_edt(big == 0, return_distances=False, return_indices=True)
    big = np.where((big == 0) & mask, big[tuple(ind)], big)

    shapes = ["compact", "organic", "tendril"]
    parts = {f"{s}#1": cluster(nb, pop, cent, args.k, 1, s) for s in shapes}
    parts.update({f"{s}#2": cluster(nb, pop, cent, args.k, 2, s) for s in shapes})

    print("\ntile compactness (Polsby-Popper median, 1=circle; higher=rounder):")
    for s in shapes:
        print(f"  {s:8} {compactness(parts[s + '#1'], big):.3f}")

    print("\npartition agreement (Rand index; 1.0 = identical grouping):")
    print("  same rule, different seed:")
    for s in shapes:
        print(f"    {s:8} seed1 vs seed2 : {rand_index(parts[s+'#1'], parts[s+'#2']):.3f}")
    print("  different rule (same seed 1):")
    for i in range(len(shapes)):
        for j in range(i + 1, len(shapes)):
            a, b = shapes[i], shapes[j]
            print(f"    {a:8} vs {b:8}: {rand_index(parts[a+'#1'], parts[b+'#1']):.3f}")


if __name__ == "__main__":
    main()
