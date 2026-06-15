"""Render the national tiling in every named palette as a labelled contact sheet, so you can
pick a colour family. Reuses the cached clustering (so it's fast).

    python tools/us_palette_sheet.py            # uses the height-1800 cache
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools")); sys.path.insert(0, str(ROOT / "src"))
import pop_mosaic as pm                                # noqa: E402
import pop_mosaic_us as pu                             # noqa: E402
import families as F                                   # noqa: E402

S, K, RND, SEED = 1800, 1_000_000, 0.12, 3
cache = ROOT / "output" / f"_us_cache2_{S}_{K}_{RND}_{SEED}.npz"
if not cache.exists():
    sys.exit(f"no cache {cache.name} - run: python tools/pop_mosaic_us.py --k {K} --height {S}")
z = np.load(cache)
big, land, water, nbig = z["big"].astype(int), z["land"], z["water"], int(z["nbig"])
col, ncol = pm.proper_colors(big, nbig, SEED)          # colouring is fixed; only the palette maps

# crop bbox to the country so thumbnails aren't mostly black
country = land | water
ys, xs = np.where(country)
y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1


def render(name):
    pal = pu.PALETTES[name][: max(ncol, 1)]
    lut = pal[np.clip(col, 0, len(pal) - 1)].copy(); lut[col < 0] = 0
    rng = np.random.default_rng(SEED)
    lut = np.clip(lut * rng.uniform(0.94, 1.06, nbig + 1)[:, None], 0, 1)
    out = F.render_mosaic(big, nbig, land.astype(int), lut, gold_class=-1, grout_width=1.3)
    out[water] = pu.WATER_C
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)).crop((x0, y0, x1, y1))


names = list(pu.PALETTES)
TW = 460                                               # thumb width
thumbs = {}
for n in names:
    im = render(n); thumbs[n] = im.resize((TW, round(im.height * TW / im.width)))
TH = thumbs[names[0]].height
print(f"rendered {len(names)} palettes")

cols, gap, lab, sw = 4, 26, 30, 26
rows = -(-len(names) // cols)
CW = gap + cols * (TW + gap)
CH = 90 + rows * (TH + lab + sw + gap)
sheet = Image.new("RGB", (CW, CH), (22, 18, 13))
d = ImageDraw.Draw(sheet)
ft = pm._font(40, bold=True); fl = pm._font(26, bold=True)
d.text((gap, 28), "PICK A PALETTE", font=ft, fill=(236, 229, 215))
for i, n in enumerate(names):
    cx = gap + (i % cols) * (TW + gap)
    cy = 90 + (i // cols) * (TH + lab + sw + gap)
    sheet.paste(thumbs[n], (cx, cy))
    d.text((cx + 2, cy + TH + 4), n, font=fl, fill=(224, 192, 110))
    for j, c in enumerate(pu.PALETTES[n][:5]):
        x = cx + j * (sw + 4)
        d.rectangle([x, cy + TH + lab, x + sw, cy + TH + lab + sw],
                    fill=tuple((np.clip(c, 0, 1) * 255).astype(int)))

dest = ROOT / "output" / "us_palette_sheet.png"
sheet.save(dest)
print(f"-> {dest.relative_to(ROOT)}  ({CW}x{CH})")
