"""
shape_compactness.py — show that many visually different shapes share the same
MOI compactness score. 3×3 grid: each row = 3 shapes with the same score.

Produces:  output/shapes/compactness_pairs.png
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("output/shapes")
OUT.mkdir(parents=True, exist_ok=True)
C = 220


def moi(mask):
    H, W = mask.shape
    Y, X = np.mgrid[0:H, 0:W]
    a = mask.astype(float); area = a.sum()
    if area == 0: return 0.0
    cx = (X*a).sum()/area; cy = (Y*a).sum()/area
    r2 = ((X-cx)**2+(Y-cy)**2)*mask
    m = r2.sum()
    return float(area**2/(2*np.pi)/m) if m > 0 else 0.0


def centered_rect(h, w):
    mask = np.zeros((C,C),dtype=bool)
    r0 = max(0, C//2-h//2); c0 = max(0, C//2-w//2)
    mask[r0:min(C,r0+h), c0:min(C,c0+w)] = True
    return mask


def square_with_arm(sq, arm_h, arm_l):
    mask = np.zeros((C,C),dtype=bool)
    bc0 = C//2 - sq - arm_l//2; bc1 = bc0 + sq
    br0 = C//2 - sq//2; br1 = br0 + sq
    mask[br0:br1, bc0:bc1] = True
    mask[C//2-arm_h//2 : C//2+arm_h//2, bc1:bc1+arm_l] = True
    return mask


def dogbone(end_sq, bar_h, bar_w):
    """Two squares at ends of a thin bar."""
    mask = np.zeros((C,C),dtype=bool)
    total_w = 2*end_sq + bar_w
    x0 = C//2 - total_w//2
    mask[C//2-end_sq//2:C//2+end_sq//2, x0:x0+end_sq] = True
    mask[C//2-bar_h//2:C//2+bar_h//2, x0+end_sq:x0+end_sq+bar_w] = True
    mask[C//2-end_sq//2:C//2+end_sq//2, x0+end_sq+bar_w:x0+2*end_sq+bar_w] = True
    return mask


def blob_with_tentacle(r_body, tent_h, tent_l):
    mask = np.zeros((C,C),dtype=bool)
    Y, X = np.mgrid[0:C, 0:C]
    cx_body = C//2 - r_body - tent_l//2
    mask[(X-cx_body)**2+(Y-C//2)**2 <= r_body**2] = True
    tc0 = cx_body + r_body
    mask[C//2-tent_h//2 : C//2+tent_h//2, tc0:tc0+tent_l] = True
    return mask


def z_shape(bar_h, bar_w, step_h, step_w):
    mask = np.zeros((C,C),dtype=bool)
    mask[C//4-bar_h//2:C//4+bar_h//2, C//2-bar_w//2:C//2+bar_w//2] = True
    mask[3*C//4-bar_h//2:3*C//4+bar_h//2, C//2-bar_w//2:C//2+bar_w//2] = True
    steps = 80
    for i in range(steps):
        t = i/steps
        rc = int(C//4 + t*(3*C//4-C//4))
        cc = int(C//2+bar_w//2 - t*bar_w)
        for dr in range(-step_h//2, step_h//2):
            for dc in range(-step_w//2, step_w//2):
                rr,cc2 = rc+dr, cc+dc
                if 0<=rr<C and 0<=cc2<C: mask[rr,cc2]=True
    return mask


def horseshoe(outer_h, outer_w, notch_h, notch_w):
    mask = np.zeros((C,C),dtype=bool)
    r0 = C//2-outer_h//2; c0 = C//2-outer_w//2
    mask[r0:r0+outer_h, c0:c0+outer_w] = True
    mask[r0:r0+notch_h, C//2-notch_w//2:C//2+notch_w//2] = False
    return mask


def plus_shape(arm_h, arm_w):
    mask = np.zeros((C,C),dtype=bool)
    mask[C//2-arm_h//2:C//2+arm_h//2, C//2-arm_w//2:C//2+arm_w//2] = True
    mask[C//2-arm_w//2:C//2+arm_w//2, C//2-arm_h//2:C//2+arm_h//2] = True
    return mask


# ── three rows, three shapes each ────────────────────────────────────────────
#   tuned so all three shapes in a row share the same MOI score

rows = [
    {
        "label": "MOI ≈ 0.37",
        "shapes": [
            (centered_rect(40, 198),         "1 : 5 rectangle"),
            (dogbone(44, 12, 40),             "dogbone"),
            (square_with_arm(45, 20, 90),    "square + corridor"),
        ],
    },
    {
        "label": "MOI ≈ 0.30  (gerrymandering threshold)",
        "shapes": [
            (centered_rect(30, 190),             "1 : 6 rectangle"),
            (z_shape(25, 58, 14, 12),            "Z-shape"),
            (blob_with_tentacle(28, 12, 105),    "circle + tendril"),
        ],
    },
    {
        "label": "MOI ≈ 0.57",
        "shapes": [
            (centered_rect(60, 180),                        "1 : 3 rectangle"),
            (plus_shape(30, 160),                           "plus / cross"),
            (horseshoe(88, 144, 52, 52),                    "horseshoe"),
        ],
    },
]

for row in rows:
    scores = [f"{moi(s):.3f}" for s, _ in row["shapes"]]
    print(f"  {row['label']:45s}  {scores}")


# ── figure ─────────────────────────────────────────────────────────────────

BG    = "#15110c"
PANEL = "#1c1710"
INK   = "#ece4d6"
MUT   = "#9a8f7c"
GOLD  = "#e0c06e"
LINE  = "#352c20"
SHAPE = "#7a9a5a"   # single neutral colour — no framing

fig, axes = plt.subplots(3, 3, figsize=(5, 4.6),
                         gridspec_kw={"hspace": 0.28, "wspace": 0.06},
                         facecolor=BG)
fig.patch.set_facecolor(BG)
fig.subplots_adjust(top=0.90, bottom=0.07, left=0.14, right=0.97)

for row_i, row in enumerate(rows):
    for col_i, (shape, lbl) in enumerate(row["shapes"]):
        ax = axes[row_i, col_i]
        ax.set_facecolor(PANEL)

        img = np.zeros((*shape.shape, 4))
        r, g, b = 0x7a/255, 0x9a/255, 0x5a/255
        img[shape] = [r, g, b, 0.85]
        ax.imshow(img, origin="upper")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(LINE); sp.set_linewidth(0.8)

        score = moi(shape)
        ax.set_xlabel(lbl, color=INK, fontsize=7.5, labelpad=2)
        ax.text(0.5, 0.05, f"{score:.2f}", transform=ax.transAxes,
                ha="center", va="bottom", color=GOLD,
                fontsize=9, fontweight="bold")

    # row label on left margin
    axes[row_i, 0].text(-0.14, 0.5, row["label"],
                        transform=axes[row_i, 0].transAxes,
                        ha="right", va="center", rotation=90,
                        color=MUT, fontsize=7)

fig.suptitle(
    "Same compactness score — very different shapes",
    color=INK, fontsize=11, fontweight="bold", y=0.96
)

out = OUT / "compactness_pairs.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"\nSaved → {out}")
