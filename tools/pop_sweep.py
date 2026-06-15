"""Sensitivity analysis: how tightly can equal-population super-tiles hit a target headcount,
as a function of that target? Builds the block-group graph ONCE, then clusters at many targets
and reports the population band (spread) for each. No image rendering -- population is data, so
this is fast.

    python tools/pop_sweep.py Maryland
    python tools/pop_sweep.py Maryland --targets 2000,5000,10000,20000,40000,80000 --height 1200
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
import pop_mosaic as pm                                # noqa: E402
import state_data as sd                                # noqa: E402


def build_graph(ST, fips, S):
    """Rasterize block groups once -> (block-group adjacency nb, per-bg population)."""
    mask, _water, to_px, _ = pm.state_mask(ST, S)
    H, W = mask.shape
    popmap = pm.load_bg_pop(fips)
    feats = [f for f in pm.fetch_bg_polys(ST, fips)["features"] if f.get("geometry")]
    big = np.zeros((H, W), int)
    pop = np.zeros(len(feats) + 1)
    for i, f in enumerate(feats, 1):
        big[sd.rasterize_px([f], to_px, W, H) & mask] = i
        pop[i] = popmap.get(f["properties"]["GEOID"], 0)
    nbig = len(feats)
    ind = ndimage.distance_transform_edt(big == 0, return_distances=False, return_indices=True)
    big = np.where((big == 0) & mask, big[tuple(ind)], big)
    nb = pm.adjacency_lists(big, nbig)
    cent = pm.tile_centroids(big, nbig)
    nb = pm.bridge_components(nb, cent, nbig)
    return nb, pop, cent, nbig


def cluster_pops(nb, pop, cent, K, seed, shape="compact"):
    """Run the full grouping at target K -> array of per-tile populations."""
    cl, M = pm.regionalize(nb, pop, K, seed, shape, cent)
    cl = pm.balance(cl, nb, pop, K, seed=seed)
    uniq = np.unique(cl[cl > 0])
    relab = np.zeros(int(cl.max()) + 1, int); relab[uniq] = np.arange(1, len(uniq) + 1)
    cl = relab[cl]
    return np.bincount(cl, weights=pop, minlength=len(uniq) + 1)[1:]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("state")
    ap.add_argument("--targets", default="2000,5000,10000,20000,40000,80000,160000")
    ap.add_argument("--height", type=int, default=1200)
    ap.add_argument("--shape", default="compact")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    ST = sd.resolve_state(args.state)
    fips = pm.state_fips(ST)
    targets = [int(t) for t in args.targets.split(",")]
    print(f"building block-group graph for {sd.STATE_NAMES[ST]} ...", flush=True)
    nb, pop, cent, nbig = build_graph(ST, fips, args.height)
    print(f"{nbig:,} block groups, {pop.sum():,.0f} people\n")

    print(f"{'target':>8} {'tiles':>6} {'mean':>8} {'median':>8} "
          f"{'CV%':>6} {'p10-p90/med%':>13} {'within±10%':>11} {'min':>8} {'max':>8}")
    print("-" * 86)
    for K in targets:
        cp = cluster_pops(nb, pop, cent, K, args.seed, args.shape)
        mean, med, std = cp.mean(), np.median(cp), cp.std()
        p10, p90 = np.percentile(cp, [10, 90])
        cv = 100 * std / mean
        spread = 100 * (p90 - p10) / med
        within = 100 * np.mean(np.abs(cp - K) <= 0.10 * K)
        print(f"{K:>8,} {len(cp):>6,} {mean:>8,.0f} {med:>8,.0f} "
              f"{cv:>6.1f} {spread:>13.1f} {within:>10.0f}% {cp.min():>8,.0f} {cp.max():>8,.0f}")


if __name__ == "__main__":
    main()
