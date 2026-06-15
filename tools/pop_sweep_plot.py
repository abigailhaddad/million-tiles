"""Render the sensitivity relationship as a branded (non-map) chart: how tightly equal-
population tiles hit their target headcount, vs the target. Reuses pop_sweep's machinery.

    python tools/pop_sweep_plot.py Maryland
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
import tiles as pm                                # noqa: E402
import pop_sweep as ps                                 # noqa: E402
import state_data as sd                                # noqa: E402

BG_C = (25, 20, 15)
INK = (232, 224, 210)
MUT = (154, 143, 124)
GOLD = (232, 198, 106)
TEAL = (90, 178, 178)
GRID = (60, 54, 44)


def fmt_k(v):
    return f"{v/1000:.0f}k" if v < 1_000_000 else f"{v/1e6:.0f}M"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("state")
    ap.add_argument("--height", type=int, default=1100)
    ap.add_argument("--seeds", type=int, default=6, help="runs averaged per target")
    args = ap.parse_args()
    ST = sd.resolve_state(args.state)
    fips = pm.state_fips(ST)

    print(f"building block-group graph for {sd.STATE_NAMES[ST]} ...", flush=True)
    nb, pop, cent, nbig = ps.build_graph(ST, fips, args.height)

    targets = np.unique(np.round(np.geomspace(2000, 200000, 13) / 500) * 500).astype(int)
    p10, p25, p50, p75, p90 = ([] for _ in range(5))
    for K in targets:
        pool = np.concatenate([ps.cluster_pops(nb, pop, cent, K, s) / K * 100
                               for s in range(1, args.seeds + 1)])         # % of target, pooled
        a, b, c, e, f = np.percentile(pool, [10, 25, 50, 75, 90])
        p10.append(a); p25.append(b); p50.append(c); p75.append(e); p90.append(f)
        print(f"  {K:>7,}: tiles land {a:3.0f}%-{f:3.0f}% of target (median {c:3.0f}%)")
    p10, p25, p50, p75, p90 = map(np.array, (p10, p25, p50, p75, p90))

    # ---- draw: a funnel that closes toward 100% -------------------------------
    W, H = 1600, 1000
    YMIN, YMAX = 60, 150                              # zoom to where the action is
    img = Image.new("RGB", (W, H), BG_C)
    d = ImageDraw.Draw(img)
    L, R, T, B = 175, W - 70, 230, H - 175
    pw, ph = R - L, B - T
    lx0, lx1 = np.log10(targets.min()), np.log10(targets.max())
    X = lambda t: L + (np.log10(t) - lx0) / (lx1 - lx0) * pw
    Y = lambda pct: B - (np.clip(pct, YMIN, YMAX) - YMIN) / (YMAX - YMIN) * ph
    xs = [X(t) for t in targets]

    f_title = pm._font(46, bold=True)
    f_sub = pm._font(25)
    f_ax = pm._font(24)
    f_tick = pm._font(22)
    f_note = pm._font(23)

    d.text((L, 60), "Can every tile hold the same number of people?", font=f_title, fill=INK)
    d.text((L, 122),
           f"{sd.STATE_NAMES[ST].title()}: each tile aims for a target headcount. The band shows "
           "where tiles actually land.", font=f_sub, fill=MUT)

    # y gridlines / ticks (% of target)
    for pct in range(YMIN, YMAX + 1, 10):
        y = Y(pct)
        d.line([(L, y), (R, y)], fill=GRID)
        d.text((L - 16, y), f"{pct}%", font=f_tick, fill=MUT, anchor="rm")
    for t in [2000, 5000, 10000, 20000, 50000, 100000, 200000]:
        if targets.min() <= t <= targets.max():
            d.text((X(t), B + 16), fmt_k(t), font=f_tick, fill=MUT, anchor="ma")

    # shaded bands: 10-90 (light) then 25-75 (darker)
    def band(lo, hi, color):
        pts = [(X(t), Y(v)) for t, v in zip(targets, hi)]
        pts += [(X(t), Y(v)) for t, v in zip(targets[::-1], lo[::-1])]
        d.polygon(pts, fill=color)
    band(p10, p90, (60, 52, 34))
    band(p25, p75, (110, 92, 44))
    # median line
    d.line(list(zip(xs, [Y(v) for v in p50])), fill=GOLD, width=4, joint="curve")
    for x, v in zip(xs, p50):
        d.ellipse([x - 5, Y(v) - 5, x + 5, Y(v) + 5], fill=GOLD)
    # 100% = perfect reference
    y100 = Y(100)
    for x in range(L, R, 16):
        d.line([(x, y100), (x + 8, y100)], fill=TEAL, width=3)
    d.text((R - 6, y100 - 30), "100% = exactly on target", font=f_note, fill=TEAL, anchor="ra")

    d.text((W // 2, B + 58), "target people per tile  (log scale)", font=f_ax, fill=INK, anchor="ma")
    d.text((L, T - 34), "tile's actual population, as % of its target", font=f_ax, fill=INK)
    d.text((L, H - 44),
           f"Each band pools {args.seeds} runs. The band tightens as targets grow - then widens "
           "again once tiles get so big the state holds only a few.", font=f_note, fill=MUT)

    dest = ROOT / "output" / f"{sd.STATE_NAMES[ST].replace(' ', '_')}_pop_sensitivity.png"
    img.save(dest)
    print(f"-> {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
