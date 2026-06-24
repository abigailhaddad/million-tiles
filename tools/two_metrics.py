"""
two_metrics.py — show how adding a second compactness metric narrows what passes.

3×3 grid:
  Row 1  all MOI = 0.30          (different convex-hull scores)
  Row 2  all convex hull = 0.80  (different MOI scores)
  Row 3  both MOI = 0.30 AND convex hull = 0.80

Produces:  output/shapes/two_metrics.png
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from pathlib import Path

OUT = Path("output/shapes")
OUT.mkdir(parents=True, exist_ok=True)
C = 220


# ── metrics ──────────────────────────────────────────────────────────────────

def moi(mask):
    H, W = mask.shape; Y, X = np.mgrid[0:H, 0:W]
    a = mask.astype(float); area = a.sum()
    if area == 0: return 0.0
    cx = (X*a).sum()/area; cy = (Y*a).sum()/area
    m = ((X-cx)**2+(Y-cy)**2)[mask].sum()
    return float(area**2/(2*np.pi)/m) if m > 0 else 0.0

def ch_ratio(mask):
    ys, xs = np.where(mask)
    if len(xs) < 4: return 0.0
    try: return min(1.0, float(mask.sum() / ConvexHull(np.column_stack([xs,ys])).volume))
    except: return 0.0


# ── shapes ───────────────────────────────────────────────────────────────────

def centered_rect(h, w):
    m = np.zeros((C,C),dtype=bool)
    m[max(0,C//2-h//2):min(C,C//2-h//2+h), max(0,C//2-w//2):min(C,C//2-w//2+w)] = True
    return m

def z_shape(bar_h, bar_w, step_h, step_w):
    mask = np.zeros((C,C),dtype=bool)
    mask[C//4-bar_h//2:C//4+bar_h//2, C//2-bar_w//2:C//2+bar_w//2] = True
    mask[3*C//4-bar_h//2:3*C//4+bar_h//2, C//2-bar_w//2:C//2+bar_w//2] = True
    for i in range(80):
        t = i/80
        rc = int(C//4 + t*(C//2)); cc = int(C//2+bar_w//2 - t*bar_w)
        for dr in range(-step_h//2, step_h//2):
            for dc in range(-step_w//2, step_w//2):
                rr,cc2 = rc+dr, cc+dc
                if 0<=rr<C and 0<=cc2<C: mask[rr,cc2]=True
    return mask

def blob_with_tentacle(r_body, tent_h, tent_l):
    mask = np.zeros((C,C),dtype=bool); Y,X = np.mgrid[0:C,0:C]
    cx_body = C//2-r_body-tent_l//2
    mask[(X-cx_body)**2+(Y-C//2)**2<=r_body**2] = True
    mask[C//2-tent_h//2:C//2+tent_h//2, cx_body+r_body:cx_body+r_body+tent_l] = True
    return mask

def l_shape(vh, vw, fh, fw):
    mask = np.zeros((C,C),dtype=bool); c0 = C//2-fw//2
    mask[C//2-vh//2:C//2+vh//2, c0:c0+vw] = True
    mask[C//2+vh//2-fh:C//2+vh//2, c0:c0+fw] = True
    return mask

def horseshoe(oh, ow, nh, nw):
    mask = np.zeros((C,C),dtype=bool)
    r0=C//2-oh//2; c0=C//2-ow//2
    mask[r0:r0+oh, c0:c0+ow] = True
    mask[r0:r0+nh, C//2-nw//2:C//2+nw//2] = False
    return mask

def dogbone(end_sq, bar_h, bar_w):
    mask = np.zeros((C,C),dtype=bool)
    tw = 2*end_sq+bar_w; x0 = C//2-tw//2
    mask[C//2-end_sq//2:C//2+end_sq//2, x0:x0+end_sq] = True
    mask[C//2-bar_h//2:C//2+bar_h//2, x0+end_sq:x0+end_sq+bar_w] = True
    mask[C//2-end_sq//2:C//2+end_sq//2, x0+end_sq+bar_w:x0+2*end_sq+bar_w] = True
    return mask

def notched_rect(h, w, nh, nw):
    mask = np.zeros((C,C),dtype=bool)
    r0=C//2-h//2; c0=C//2-w//2
    mask[r0:r0+h, c0:c0+w] = True
    mask[C//2-nh//2:C//2+nh//2, c0:c0+nw] = False
    return mask


# ── grid definition ───────────────────────────────────────────────────────────

rows = [
    {
        "row_label": "Metric 1 only\n(MOI = 0.30)",
        "highlight": "moi",
        "shapes": [
            (centered_rect(30, 190),          "1 : 6 rectangle"),
            (z_shape(25, 58, 14, 12),         "Z-shape"),
            (blob_with_tentacle(28,12,105),   "circle + tendril"),
        ],
    },
    {
        "row_label": "Metric 2 only\n(convex hull = 0.80)",
        "highlight": "ch",
        "shapes": [
            (l_shape(60, 16, 34, 106),        "L-shape"),
            (horseshoe(60, 108, 34, 42),      "horseshoe"),
            (dogbone(28, 14, 50),             "dogbone"),
        ],
    },
    {
        "row_label": "Both metrics\n(MOI = 0.30\nconvex hull = 0.80)",
        "highlight": "both",
        "shapes": [
            (dogbone(28, 14, 50),                          "dogbone"),
            (l_shape(138, 22, 24, 40),                     "L-shape"),
            (notched_rect(28, 194, 22, 58),                "notched rectangle"),
        ],
    },
]

for row in rows:
    print(f"\n{row['row_label'].replace(chr(10),' / ')}")
    for shape, lbl in row["shapes"]:
        print(f"  {lbl:20s}  moi={moi(shape):.3f}  ch={ch_ratio(shape):.3f}")


# ── figure ────────────────────────────────────────────────────────────────────

BG    = "#15110c"
PANEL = "#1c1710"
INK   = "#ece4d6"
MUT   = "#9a8f7c"
GOLD  = "#e0c06e"
LINE  = "#352c20"
DIM   = "#6a7a6a"
SHAPE = "#7a9a5a"

fig, axes = plt.subplots(3, 3, figsize=(6, 6.2),
                         gridspec_kw={"hspace": 0.32, "wspace": 0.06},
                         facecolor=BG)
fig.patch.set_facecolor(BG)
fig.subplots_adjust(top=0.90, bottom=0.04, left=0.22, right=0.97)

for row_i, row in enumerate(rows):
    hl = row["highlight"]
    for col_i, (shape, lbl) in enumerate(row["shapes"]):
        ax = axes[row_i, col_i]
        ax.set_facecolor(PANEL)

        img = np.zeros((*shape.shape, 4))
        r,g,b = 0x7a/255, 0x9a/255, 0x5a/255
        img[shape] = [r, g, b, 0.85]
        ax.imshow(img, origin="upper")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(LINE); sp.set_linewidth(0.8)

        mo = moi(shape); ch = ch_ratio(shape)

        moi_col = GOLD if hl in ("moi","both") else DIM
        ch_col  = GOLD if hl in ("ch","both")  else DIM

        ax.text(0.5, 0.13, f"MOI {mo:.2f}", transform=ax.transAxes,
                ha="center", va="bottom", color=moi_col, fontsize=8, fontweight="bold")
        ax.text(0.5, 0.03, f"CH  {ch:.2f}", transform=ax.transAxes,
                ha="center", va="bottom", color=ch_col,  fontsize=8, fontweight="bold")
        ax.set_xlabel(lbl, color=INK, fontsize=7, labelpad=2)

    axes[row_i, 0].text(-0.04, 0.5, row["row_label"],
                        transform=axes[row_i, 0].transAxes,
                        ha="right", va="center", rotation=0,
                        color=MUT, fontsize=7.5, linespacing=1.4)

fig.suptitle("Adding a second metric narrows what passes",
             color=INK, fontsize=11, fontweight="bold", y=0.96)

out = OUT / "two_metrics.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"\nSaved → {out}")
