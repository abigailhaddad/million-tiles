"""The RECTANGULAR 2D pavement of US population (the other way to do it).

Aaron Schumacher's pavement is equal-count bins in 1D; his matplotlib plot2d does the 2D version
as a recursive quantile GRID -- equal-count axis-aligned rectangles. This is that, with population
spread over geography as the density field.

The trick that sidesteps the "8 million blocks is too many to cluster" wall: we never cluster. We
splat every unit's population into a density raster (one cheap pass), then the whole partition is
cumulative sums -- split x into equal-population vertical strips, split each strip into equal-
population cells. Cuts can fall anywhere (a cell edge slices straight through a unit), so the cells
come out holding ~exactly equal population. Cost is set by the raster, not the number of units.

    python tools/rect_pavement.py --source bg --nx 18 --ny 18 --height 1600
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools")); sys.path.insert(0, str(ROOT / "src"))
import tiles as pm                                     # noqa: E402  proper_colors
import tiles_us as pu                                  # noqa: E402  panel/water/art_frame/PALETTES
import build_national as bn                            # noqa: E402
import state_data as sd                                # noqa: E402
import families as F                                   # noqa: E402

BG = ROOT / "data" / "cenpop2020_bg.txt"


def load_bg_density(to_px, W, H):
    """Splat every CONUS block-group centre of population into a people-per-pixel raster."""
    lon, lat, pop = [], [], []
    with open(BG, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["STATEFP"] in ("02", "15", "72"):
                continue
            p = int(r["POPULATION"])
            if p > 0:
                lon.append(float(r["LONGITUDE"])); lat.append(float(r["LATITUDE"])); pop.append(p)
    px, py = to_px(np.array(lon), np.array(lat))
    ix, iy = np.round(px).astype(int), np.round(py).astype(int)
    ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
    dens = np.zeros((H, W))
    np.add.at(dens, (iy[ok], ix[ok]), np.array(pop)[ok])
    return dens, sum(pop)


def qbounds(weights, n):
    """Integer split points so each of n slices holds ~1/n of total weight (a quantile grid)."""
    c = np.cumsum(weights); tot = c[-1]
    b = [0]
    for k in range(1, n):
        b.append(int(np.searchsorted(c, k / n * tot)))
    b.append(len(weights))
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="bg")
    ap.add_argument("--nx", type=int, default=18)
    ap.add_argument("--ny", type=int, default=18)
    ap.add_argument("--height", type=int, default=1600)
    ap.add_argument("--palette", default="dark")
    ap.add_argument("--html", action="store_true")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    S = args.height

    panel = bn._build_panel(bn.CONUS, S, "albers")
    to_px, W, H = panel["to_px"], panel["W"], panel["H"]
    state, water, _ = bn._panel_masks(panel)
    water = pu.clean_water(water, max(12, round((S / 300) ** 2)), max(1, round(S / 900)))
    land = state & ~water
    print(f"canvas {W}x{H}", flush=True)

    dens, total = load_bg_density(to_px, W, H)
    print(f"{total:,} people splatted from block groups", flush=True)
    # recursive quantile rectangles: x-strips of equal pop, each split into equal-pop y-cells
    cellid = np.zeros((H, W), int)
    cpop, cid = [], 0
    xb = qbounds(dens.sum(0), args.nx)
    for i in range(args.nx):
        x0, x1 = xb[i], xb[i + 1]
        yb = qbounds(dens[:, x0:x1].sum(1), args.ny)
        for j in range(args.ny):
            y0, y1 = yb[j], yb[j + 1]
            cid += 1
            cellid[y0:y1, x0:x1] = cid
            cpop.append(float(dens[y0:y1, x0:x1].sum()))
    nbig = cid
    cpop = np.array(cpop)
    print(f"{nbig} cells  band {cpop.min():,.0f}-{cpop.max():,.0f} "
          f"(p10-p90 {np.percentile(cpop,10):,.0f}-{np.percentile(cpop,90):,.0f})", flush=True)

    big = np.where(land, cellid, 0)
    col, ncol = pm.proper_colors(big, nbig, args.seed)
    pal = pu.PALETTES[args.palette][:max(ncol, 1)]
    lut = pal[np.clip(col, 0, len(pal) - 1)].copy(); lut[col < 0] = 0
    rng = np.random.default_rng(args.seed)
    lut = np.clip(lut * rng.uniform(0.94, 1.06, nbig + 1)[:, None], 0, 1)
    out = F.render_tiles(big, nbig, land.astype(int), lut, gold_class=-1, grout_width=1.4)
    out[water] = pu.WATER_BY_PAL.get(args.palette, pu.WATER_C)
    base_rgb = (np.clip(out, 0, 1) * 255).astype(np.uint8)

    k = round(total / nbig)
    stem = f"us_rect_{args.source}_{args.nx}x{args.ny}_{args.palette}"
    pu.art_frame(out, land, water, np.array(cpop), k, ncol).save(ROOT / "output" / f"{stem}.png")
    print(f"-> output/{stem}.png", flush=True)
    if args.html:
        hdest = ROOT / "output" / f"{stem}.html"
        pu.export_interactive_us(hdest, base_rgb, big, nbig, np.array(cpop), to_px, panel["feats"],
                                 "Rectangular pavement - one million per cell")
        print(f"-> {hdest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
