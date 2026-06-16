"""National equal-population tiling (CONUS): bucket Census TRACTS into ~K-person super-tiles.

Tracts (~85k nationally, ~4,000 people each) are the practical finest unit for the whole US
(blocks are ~8M and not in the centers-of-population dataset). Each super-tile holds ~K people,
so a tile is "a million-ish neighbours" regardless of how much land that takes.

Reuses build_national's Albers CONUS projection + land mask, and tiles's clustering. Tract
adjacency comes from a Delaunay triangulation of the projected centres (no polygons needed).

    python tools/tiles_us.py --k 1000000 --height 1300
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import Delaunay, cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools")); sys.path.insert(0, str(ROOT / "src"))
import tiles as pm                                # noqa: E402
import build_national as bn                            # noqa: E402
import state_data as sd                                # noqa: E402
import families as F                                   # noqa: E402

TRACTS = ROOT / "data" / "cenpop2020_tract.txt"

# major cities (name, lat, lon) to orient the map — well spread, kept light to avoid clutter
CITIES = [
    ("New York", 40.71, -74.01), ("Boston", 42.36, -71.06), ("Philadelphia", 39.95, -75.16),
    ("Detroit", 42.33, -83.05), ("Chicago", 41.88, -87.63), ("Minneapolis", 44.98, -93.27),
    ("Atlanta", 33.75, -84.39), ("Miami", 25.76, -80.19), ("Nashville", 36.16, -86.78),
    ("Houston", 29.76, -95.37), ("Dallas", 32.78, -96.80), ("San Antonio", 29.42, -98.49),
    ("Kansas City", 39.10, -94.58), ("Denver", 39.74, -104.99), ("Phoenix", 33.45, -112.07),
    ("Las Vegas", 36.17, -115.14), ("Salt Lake City", 40.76, -111.89), ("Los Angeles", 34.05, -118.24),
    ("San Francisco", 37.77, -122.42), ("Seattle", 47.61, -122.33), ("Portland", 45.52, -122.68),
    ("New Orleans", 29.95, -90.07),
]

WATER_C = np.array([0.27, 0.42, 0.55])             # muted slate-blue for water (default)
WATER_BY_PAL = {                                   # deeper water for dark/moody palettes
    "dark": np.array([0.11, 0.22, 0.32]),
    "sepia": np.array([0.26, 0.30, 0.31]),
}

# named tile palettes (NO blue -- water owns blue); 6 mutually distinct colours each
PALETTES = {
    "ember":      np.array([[0.85, 0.41, 0.29], [0.93, 0.74, 0.31], [0.56, 0.65, 0.44],
                            [0.95, 0.91, 0.81], [0.67, 0.43, 0.51], [0.49, 0.35, 0.27]]),
    "clay":       np.array([[0.80, 0.53, 0.44], [0.87, 0.76, 0.55], [0.68, 0.74, 0.62],
                            [0.93, 0.90, 0.85], [0.70, 0.58, 0.64], [0.58, 0.52, 0.46]]),
    "candy":      np.array([[0.95, 0.45, 0.40], [0.99, 0.81, 0.33], [0.47, 0.73, 0.50],
                            [0.97, 0.95, 0.86], [0.64, 0.48, 0.76], [0.42, 0.57, 0.42]]),
    "jewel":      np.array([[0.78, 0.28, 0.27], [0.92, 0.66, 0.22], [0.45, 0.56, 0.36],
                            [0.91, 0.86, 0.71], [0.47, 0.31, 0.46], [0.80, 0.50, 0.25]]),
    "terracotta": np.array([[0.79, 0.42, 0.30], [0.88, 0.66, 0.38], [0.55, 0.58, 0.40],
                            [0.93, 0.87, 0.74], [0.66, 0.30, 0.24], [0.82, 0.72, 0.52]]),
    "autumn":     np.array([[0.80, 0.36, 0.22], [0.90, 0.69, 0.26], [0.62, 0.20, 0.20],
                            [0.93, 0.84, 0.62], [0.45, 0.33, 0.22], [0.74, 0.52, 0.26]]),
    "dusty":      np.array([[0.78, 0.55, 0.55], [0.66, 0.56, 0.66], [0.62, 0.70, 0.58],
                            [0.91, 0.87, 0.83], [0.52, 0.48, 0.54], [0.84, 0.74, 0.64]]),
    "forest":     np.array([[0.30, 0.45, 0.32], [0.62, 0.69, 0.40], [0.80, 0.73, 0.45],
                            [0.91, 0.89, 0.79], [0.52, 0.40, 0.28], [0.42, 0.57, 0.46]]),
    "retro70":    np.array([[0.62, 0.55, 0.24], [0.87, 0.62, 0.22], [0.74, 0.36, 0.20],
                            [0.91, 0.85, 0.67], [0.42, 0.34, 0.22], [0.56, 0.59, 0.36]]),
    "pastel":     np.array([[0.96, 0.72, 0.62], [0.98, 0.90, 0.62], [0.66, 0.84, 0.66],
                            [0.97, 0.95, 0.90], [0.78, 0.70, 0.88], [0.92, 0.78, 0.80]]),
    "sepia":      np.array([[0.32, 0.23, 0.16], [0.50, 0.37, 0.25], [0.67, 0.53, 0.38],
                            [0.83, 0.71, 0.55], [0.93, 0.86, 0.74], [0.59, 0.45, 0.31]]),
    "bold":       np.array([[0.84, 0.25, 0.25], [0.95, 0.72, 0.18], [0.30, 0.58, 0.40],
                            [0.95, 0.93, 0.86], [0.35, 0.35, 0.62], [0.88, 0.50, 0.20]]),
    "dark":       np.array([[0.20, 0.47, 0.22], [0.55, 0.13, 0.13], [0.42, 0.18, 0.55],
                            [0.66, 0.48, 0.10], [0.72, 0.34, 0.17], [0.62, 0.20, 0.52]]),
}


def load_tracts():
    lon, lat, pop = [], [], []
    with open(TRACTS, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["STATEFP"] in ("02", "15", "72"):     # drop AK / HI / PR (CONUS v1)
                continue
            p = int(r["POPULATION"])
            if p <= 0:
                continue
            lon.append(float(r["LONGITUDE"])); lat.append(float(r["LATITUDE"])); pop.append(p)
    return np.array(lon), np.array(lat), np.array(pop)


def add_us_legend(img, cpop, k, ncol):
    """Bake a description band: what the map is + the per-tile population spread."""
    from PIL import ImageDraw
    W, Ht = img.size
    bh = int(W * 0.105)
    out = Image.new("RGB", (W, Ht + bh), (22, 18, 13))
    out.paste(img, (0, 0))
    d = ImageDraw.Draw(out)
    pad = int(W * 0.028)
    ink, mut, gold = (236, 228, 214), (150, 139, 120), (232, 198, 106)
    ft = pm._font(int(bh * 0.27), bold=True)
    fs = pm._font(int(bh * 0.135))
    fk = pm._font(int(bh * 0.155), bold=True)
    fz = pm._font(int(bh * 0.115))
    mn, mx = cpop.min(), cpop.max()
    p25, p75 = np.percentile(cpop, [25, 75])
    d.text((pad, Ht + int(bh * 0.12)), "THE UNITED STATES IN MILLION-PERSON TILES", font=ft, fill=ink)
    d.text((pad, Ht + int(bh * 0.45)),
           f"{len(cpop)} tiles, each holding about the same number of people. A tile's size is "
           "inverse population density - small where millions pack in, vast where few live.",
           font=fs, fill=mut)
    d.text((pad, Ht + int(bh * 0.62)),
           f"each tile ~ {k:,} residents     middle half (25-75%): {p25:,.0f} to {p75:,.0f}"
           f"     full range: {mn:,.0f} to {mx:,.0f}", font=fk, fill=gold)
    d.text((pad, Ht + int(bh * 0.83)),
           "2020 Census tract centers of population  -  Albers projection, lower 48  -  "
           f"{ncol} colours, no two neighbours alike, colour carries no meaning",
           font=fz, fill=mut)
    return out


def _text_spaced(d, xy, s, font, fill, tracking, anchor_center=True):
    """Draw text with manual letter-spacing (PIL has none), optionally centred on xy."""
    widths = [d.textlength(c, font=font) for c in s]
    total = sum(widths) + tracking * (len(s) - 1)
    x = xy[0] - total / 2 if anchor_center else xy[0]
    for c, w in zip(s, widths):
        d.text((x, xy[1]), c, font=font, fill=fill, anchor="lm")
        x += w + tracking


def art_frame(out, land, water, cpop, k, ncol, seed=7, cities=None):
    """Turn the raw tiling into a framed art print: glazed-tile enrichment + grain, the country
    floated on a warm vignetted ground with a soft shadow, a hairline frame and elegant type.
    cities: optional [(name, px, py)] in map space, drawn as labelled dots."""
    H, W, _ = out.shape
    country = land | water
    rng = np.random.default_rng(seed)

    # 1. enrich tiles: gentle saturation lift, soft top-light, fine grain (glazed ceramic feel)
    rgb = out.astype(float)
    lum = rgb.mean(2, keepdims=True)
    rgb = lum + (rgb - lum) * 1.14
    rgb *= np.linspace(1.06, 0.95, H)[:, None, None]                       # top-lit
    grain = ndimage.gaussian_filter(rng.standard_normal((H, W)), 0.6)
    rgb = np.clip(rgb * (1 + 0.04 * grain[..., None]), 0, 1)

    # 2. warm vignetted ground, larger than the map (poster margins)
    m, top, bot = round(0.05 * W), round(0.12 * W), round(0.085 * W)
    CW, CH = W + 2 * m, H + top + bot
    yy, xx = np.mgrid[0:CH, 0:CW]
    r = np.sqrt(((xx - CW / 2) / (0.62 * CW)) ** 2 + ((yy - 0.46 * CH) / (0.62 * CH)) ** 2)
    vig = np.clip(1 - r, 0, 1)[..., None] ** 1.5
    bg = np.array([0.065, 0.052, 0.040]) + (np.array([0.125, 0.105, 0.078]) -
                                            np.array([0.065, 0.052, 0.040])) * vig

    # 3. soft drop shadow of the country silhouette
    cmask = np.zeros((CH, CW)); cmask[top:top + H, m:m + W] = country
    sh = ndimage.gaussian_filter(cmask, 20)
    o = 16
    shoff = np.zeros_like(sh); shoff[o:, o:] = sh[:-o, :-o]
    bg = bg * (1 - 0.5 * shoff[..., None])

    # 4. composite the country onto the ground
    canvas = bg
    sub = canvas[top:top + H, m:m + W]
    canvas[top:top + H, m:m + W] = np.where(country[..., None], rgb, sub)

    img = Image.fromarray((np.clip(canvas, 0, 1) * 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    ink, mut, gold = (236, 229, 215), (150, 139, 120), (224, 192, 110)

    # 5. hairline frame
    fi = round(0.02 * W)
    d.rectangle([fi, fi, CW - fi - 1, CH - fi - 1], outline=(120, 102, 64), width=2)

    # 6. typography
    ft = pm._font(round(0.030 * W), bold=True)
    fsub = pm._font(round(0.0145 * W))
    _text_spaced(d, (CW / 2, top * 0.40), "ONE MILLION AMERICANS PER TILE", ft, ink, round(0.004 * W))
    d.text((CW / 2, top * 0.66),
           f"the contiguous United States as {len(cpop)} equal-population tiles",
           font=fsub, fill=mut, anchor="mm")

    p25, p75 = np.percentile(cpop, [25, 75])
    fk = pm._font(round(0.0145 * W), bold=True)
    fz = pm._font(round(0.0115 * W))
    d.text((CW / 2, CH - bot * 0.62),
           f"each tile ~ {k:,} residents       middle half  {p25:,.0f} - {p75:,.0f}"
           f"       full range  {cpop.min():,.0f} - {cpop.max():,.0f}", font=fk, fill=gold, anchor="mm")
    d.text((CW / 2, CH - bot * 0.34),
           "Data: 2020 U.S. Census, tract centers of population  ·  Albers Equal-Area projection  ·  "
           "four-colour map, colour carries no meaning", font=fz, fill=mut, anchor="mm")

    if cities:
        _draw_cities(d, cities, W, m, top)
    return img


def _draw_cities(d, cities, W, ox=0, oy=0):
    """Labelled white dots for cities on an ImageDraw, offset by (ox, oy)."""
    fc = pm._font(round(0.0115 * W), bold=True)
    rdot = max(2, round(0.0016 * W))
    for name, px, py in cities:
        x, y = px + ox, py + oy
        d.ellipse([x - rdot, y - rdot, x + rdot, y + rdot], fill=(245, 245, 245),
                  outline=(20, 16, 12), width=2)
        d.text((x + rdot + 5, y), name, font=fc, fill=(248, 246, 242),
               stroke_width=3, stroke_fill=(15, 12, 9), anchor="lm")


def export_interactive_us(dest, base_rgb, big, nbig, cpop, to_px, feats, title):
    """Self-contained hover HTML for the national map: each tile -> population + which states it
    covers and by what %, with a pixel-accurate pick-map (same trick as the DC version)."""
    H, W = big.shape
    codes = list(feats.keys())
    Ns = len(codes) + 1
    stateid = np.zeros((H, W), np.int32)
    for si, ST in enumerate(codes, 1):
        stateid[sd.rasterize_px(feats[ST], to_px, W, H) & (big > 0)] = si
    sarea = np.bincount(stateid.ravel(), minlength=Ns).astype(float)
    tile_area = np.bincount(big.ravel(), minlength=nbig + 1).astype(float)
    # sq miles per pixel (Albers is equal-area, so it's constant): measure a 2x2 deg box
    bx, by = to_px(np.array([-97., -95., -95., -97.]), np.array([38.5, 38.5, 40.5, 40.5]))
    px_area = 0.5 * abs(sum(bx[i] * by[(i + 1) % 4] - bx[(i + 1) % 4] * by[i] for i in range(4)))
    box_sqmi = (2 * 69.0) * (2 * 69.0 * np.cos(np.radians(39.5)))     # ~sq mi of a 2x2 deg box
    sqmi_per_px = box_sqmi / max(px_area, 1)
    valid = (big > 0) & (stateid > 0)
    comb = big[valid].astype(np.int64) * Ns + stateid[valid]
    uc, cnts = np.unique(comb, return_counts=True)
    overlap = {}
    for code, cnt in zip(uc.tolist(), cnts.tolist()):
        overlap.setdefault(code // Ns, []).append((code % Ns, cnt))

    data = {}
    for t in range(1, nbig + 1):
        parts = []
        for s, cnt in overlap.get(t, []):
            pctT = 100 * cnt / max(tile_area[t], 1)
            pctS = 100 * cnt / max(sarea[s], 1)
            if pctT >= 2:
                parts.append((pctT, pctS, sd.STATE_NAMES[codes[s - 1]].title()))
        parts.sort(reverse=True)
        sqmi = tile_area[t] * sqmi_per_px
        data[t] = {"pop": int(round(cpop[t - 1])), "sqmi": int(round(sqmi)),
                   "dens": round(cpop[t - 1] / max(sqmi, 1), 1),
                   "parts": [{"pctT": pm.pct1(p[0]), "pctS": pm.pct1(p[1]), "st": p[2]} for p in parts[:5]]}

    idm = big.astype(np.uint32)
    pick = np.dstack([(idm & 255).astype(np.uint8), ((idm >> 8) & 255).astype(np.uint8),
                      np.zeros_like(idm, np.uint8)])
    html = _HTML_US
    for kk, vv in {"__W__": str(W), "__H__": str(H), "__TITLE__": title,
                   "__TILES__": pm._png_uri(base_rgb), "__PICK__": pm._png_uri(pick),
                   "__DATA__": json.dumps(data, separators=(",", ":"))}.items():
        html = html.replace(kk, vv)
    open(dest, "w").write(html)


_HTML_US = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>__TITLE__</title>
<style>
 :root{--bg:#17120d;--ink:#ece4d6;--mut:#9a8f7c;--gold:#e0c06e}
 *{box-sizing:border-box} html,body{margin:0;background:var(--bg);color:var(--ink);
  font:16px/1.5 "Iowan Old Style",Georgia,serif}
 .wrap{max-width:1100px;margin:0 auto;padding:30px 20px 70px}
 h1{font-size:25px;letter-spacing:.04em;margin:0 0 2px} .sub{color:var(--mut);font-size:15px;margin:0 0 18px}
 #stage{position:relative;display:block;width:100%;overflow:hidden;cursor:grab;touch-action:none}
 #stage.drag{cursor:grabbing}
 #stage img{display:block;width:100%;height:auto;transform-origin:0 0;will-change:transform}
 #tip{position:fixed;pointer-events:none;z-index:9;max-width:320px;display:none;
  background:rgba(12,9,6,.96);border:1px solid rgba(224,192,110,.5);border-radius:9px;
  padding:11px 13px;font-size:14px;box-shadow:0 8px 28px rgba(0,0,0,.6)}
 #tip .pop{color:var(--gold);font-size:16px;margin-bottom:6px}
 #tip .row{margin:3px 0} #tip .pct{color:var(--gold);font-weight:600} #tip .nm{color:var(--mut);font-size:13px}
 .note{color:var(--mut);font-size:13px;margin-top:14px}
</style></head><body><div class="wrap">
<h1>One million Americans per tile</h1>
<p class="sub">The contiguous US split into equal-population tiles. Hover a tile for its population and
states.</p>
<div id="stage"><img id="mos" src="__TILES__" alt="equal-population tile map of the United States"></div>
<div id="tip"></div>
<p class="note">Tiles are groups of 2020 Census tracts holding ~1,000,000 people each; they cross
state lines. Colours are a four-colour map and carry no meaning.</p>
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
  h+='<div class="row"><span class="pct">'+d.sqmi.toLocaleString()+'</span> sq mi'+
     ' <span class="nm">('+d.dens.toLocaleString()+' people / sq mi)</span></div>';
  if(d.parts.length){ h+='<div class="nm">covers</div>';
    for(const p of d.parts) h+='<div class="row"><span class="pct">'+p.pctS+'%</span> of '+p.st+
      ' <span class="nm">('+p.pctT+'% of this tile)</span></div>'; }
  tip.innerHTML=h; tip.style.display='block';
  let tx=e.clientX+16, ty=e.clientY+16;
  if(tx+330>innerWidth) tx=e.clientX-330; if(ty+200>innerHeight) ty=e.clientY-200;
  tip.style.left=tx+'px'; tip.style.top=ty+'px';
});
mos.addEventListener('mouseleave',()=>tip.style.display='none');

// --- zoom + pan (the hover hit-test above reads the live rect, so it keeps working) ---
const st=document.getElementById('stage');
let z=1, tx=0, ty=0;
function apply(){
  const sw=st.clientWidth, sh=mos.clientHeight;
  tx=Math.min(0,Math.max(sw*(1-z),tx)); ty=Math.min(0,Math.max(sh*(1-z),ty));
  mos.style.transform='translate('+tx+'px,'+ty+'px) scale('+z+')';
}
st.addEventListener('wheel',e=>{
  e.preventDefault();
  const r=st.getBoundingClientRect(), mx=e.clientX-r.left, my=e.clientY-r.top;
  const nz=Math.min(9,Math.max(1, z*(e.deltaY<0?1.2:1/1.2)));
  tx=mx-(mx-tx)*nz/z; ty=my-(my-ty)*nz/z; z=nz; apply();
},{passive:false});
let pan=false, lx=0, ly=0;
st.addEventListener('pointerdown',e=>{pan=true;lx=e.clientX;ly=e.clientY;st.classList.add('drag');st.setPointerCapture(e.pointerId);});
st.addEventListener('pointerup',()=>{pan=false;st.classList.remove('drag');});
st.addEventListener('pointermove',e=>{if(pan){tx+=e.clientX-lx;ty+=e.clientY-ly;lx=e.clientX;ly=e.clientY;apply();}});
st.addEventListener('dblclick',()=>{z=1;tx=0;ty=0;apply();});
</script></body></html>"""


def clean_water(water, min_px, thin):
    """National water reads as grout because the thin, branching river NETWORK survives any
    size filter. So first an OPENING drops everything narrower than ~2*thin px (the stream web,
    minor rivers), keeping only WIDE water (Great Lakes, big reservoirs, bays, broad rivers);
    then a size filter removes leftover specks; then a light close so survivors read solid."""
    w = ndimage.binary_opening(water, iterations=thin) if thin else water
    lab, n = ndimage.label(w)
    if not n:
        return w
    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    keep = np.isin(lab, np.nonzero(sizes >= min_px)[0] + 1)
    return ndimage.binary_closing(keep, iterations=1)


def delaunay_adj(points, n):
    """Adjacency lists (1-indexed) from a Delaunay triangulation of the projected points."""
    tri = Delaunay(points)
    nbset = [set() for _ in range(n + 1)]
    s = tri.simplices
    for a, b, c in s:
        for u, v in ((a, b), (b, c), (a, c)):
            nbset[u + 1].add(v + 1); nbset[v + 1].add(u + 1)
    return [list(x) for x in nbset]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=1_000_000, help="people per tile")
    ap.add_argument("--height", type=int, default=1300)
    ap.add_argument("--grout", type=float, default=1.3)
    ap.add_argument("--water-min", type=int, default=0,
                    help="min water-body size in px (0 = auto-scale to height)")
    ap.add_argument("--water-thin", type=int, default=-1,
                    help="opening iterations to drop narrow rivers (-1 = auto; 0 = keep all)")
    ap.add_argument("--round", type=float, default=0.12,
                    help="compactness tolerance: tiles round up while staying within +-this of K "
                         "(0 = off, keep exact-population tendrils)")
    ap.add_argument("--clamp", type=float, default=0.10,
                    help="hard cap on per-tile deviation: chain-repair every tile into +-this of K")
    ap.add_argument("--palette", choices=list(PALETTES), default="ember")
    ap.add_argument("--no-art", action="store_true", help="plain band instead of the framed art print")
    ap.add_argument("--cities", action="store_true", help="label major cities to orient the map")
    ap.add_argument("--html", action="store_true", help="also write an interactive hover HTML")
    ap.add_argument("--no-cache", action="store_true", help="recompute the clustering")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    S = args.height
    t0 = time.time()
    tick = lambda m: print(f"[{time.time()-t0:5.1f}s] {m}", flush=True)

    # the slow part (land mask + cluster + voronoi) is cached so palette swaps are instant
    cache = ROOT / "output" / f"_us_cache3_{S}_{args.k}_{args.round}_{args.clamp}_{args.seed}.npz"
    if cache.exists() and not args.no_cache:
        tick(f"loading cached clustering ({cache.name}) ...")
        z = np.load(cache)
        big, land, water, nbig = z["big"].astype(int), z["land"], z["water"], int(z["nbig"])
        cpop = z["cpop"]
        H, W = big.shape
    else:
        tick("building CONUS Albers panel + land mask ...")
        panel = bn._build_panel(bn.CONUS, S, "albers")
        to_px, W, H = panel["to_px"], panel["W"], panel["H"]
        state, water, _parks = bn._panel_masks(panel)
        wmin = args.water_min or max(12, round((S / 300.0) ** 2))   # drop sub-pond specks
        thin = max(1, round(S / 900)) if args.water_thin < 0 else args.water_thin
        water = clean_water(water, wmin, thin)
        land = state & ~water

        lon, lat, pop = load_tracts()
        px, py = to_px(lon, lat)
        inb = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        px, py, pop = px[inb], py[inb], pop[inb]
        keep = state[np.round(py).astype(int), np.round(px).astype(int)]
        points = np.column_stack([px, py])[keep]; pop = pop[keep]
        n = len(points)
        tick(f"{n:,} tracts kept, {pop.sum():,} people")

        nb = delaunay_adj(points, n)
        cent = np.vstack([[0, 0], points]).astype(float)
        pop1 = np.concatenate([[0], pop]).astype(float)
        tick("clustering ...")
        cl, M = pm.regionalize(nb, pop1, args.k, args.seed, "compact", cent)
        cl = pm.balance(cl, nb, pop1, args.k, iters=10, seed=args.seed)
        if args.round > 0:
            cl = pm.compactify(cl, nb, pop1, args.k, cent, iters=8, tol=args.round, seed=args.seed)
        cl = pm.repair_outliers(cl, nb, pop1, args.k, band=args.clamp)   # kill the boxed-in tails
        uniq = np.unique(cl[cl > 0])
        relab = np.zeros(int(cl.max()) + 1, int); relab[uniq] = np.arange(1, len(uniq) + 1)
        cl = relab[cl]; M = len(uniq)
        cpop = np.bincount(cl, weights=pop1, minlength=M + 1)[1:]
        tick(f"  {M:,} tiles, band {cpop.min():,.0f}-{cpop.max():,.0f} "
             f"(p10-p90 {np.percentile(cpop,10):,.0f}-{np.percentile(cpop,90):,.0f})")
        tree = cKDTree(points)
        ys, xs = np.nonzero(land)
        _, idx = tree.query(np.column_stack([xs, ys]), workers=-1)
        big = np.zeros((H, W), int); big[ys, xs] = cl[idx + 1]; nbig = M
        np.savez_compressed(cache, big=big.astype(np.int32), land=land, water=water,
                            nbig=nbig, cpop=cpop)
        tick("cached clustering (palette swaps now instant)")

    # ---- colour with the chosen palette -------------------------------------
    col, ncol = pm.proper_colors(big, nbig, args.seed)
    pal = PALETTES[args.palette][: max(ncol, 1)]
    lut = pal[np.clip(col, 0, len(pal) - 1)].copy()
    lut[col < 0] = 0
    rng = np.random.default_rng(args.seed)
    lut = np.clip(lut * rng.uniform(0.94, 1.06, nbig + 1)[:, None], 0, 1)
    out = F.render_tiles(big, nbig, land.astype(int), lut, gold_class=-1, grout_width=args.grout)
    out[water] = WATER_BY_PAL.get(args.palette, WATER_C)

    base_rgb = (np.clip(out, 0, 1) * 255).astype(np.uint8)        # raw tiles for the hover layer
    cities = None
    if args.cities:
        tp = bn._build_panel(bn.CONUS, S, "albers")["to_px"]      # cheap: projections only
        cx, cy = tp(np.array([c[2] for c in CITIES]), np.array([c[1] for c in CITIES]))
        cities = [(CITIES[i][0], cx[i], cy[i]) for i in range(len(CITIES))
                  if 0 <= cx[i] < W and 0 <= cy[i] < H]
    if cities:                                                # put labels on the hover map too
        bi = Image.fromarray(base_rgb); _draw_cities(ImageDraw.Draw(bi), cities, W); base_rgb = np.asarray(bi)
    if args.no_art:
        img = add_us_legend(Image.fromarray(base_rgb.copy()), cpop, args.k, ncol)
    else:
        img = art_frame(out, land, water, cpop, args.k, ncol, cities=cities)
    dest = ROOT / "output" / f"us_tiles_{args.k}_{args.palette}.png"
    img.save(dest)
    tick(f"saved -> {dest.relative_to(ROOT)}  ({W}x{H}, {ncol}-colour, palette={args.palette})")

    if args.html:
        tick("building state-overlap hover HTML ...")
        panel2 = bn._build_panel(bn.CONUS, S, "albers")          # cheap: projections only
        hdest = ROOT / "output" / f"us_tiles_{args.k}_{args.palette}.html"
        export_interactive_us(hdest, base_rgb, big, nbig, cpop, panel2["to_px"],
                              panel2["feats"], "One million Americans per tile")
        tick(f"interactive -> {hdest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
