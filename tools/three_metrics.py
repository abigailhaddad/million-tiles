"""
three_metrics.py  —  how each compactness metric can be fooled.

4×3 grid; each row = 1 normal compact shape + 2 best "cheat" examples:
  Row 1  MOI ≥ 0.50   blind spots: jagged edges (comb), smooth deep concavity (crescent)
  Row 2  CH  ≥ 0.90   blind spots: elongation (thin rect, CH=1.0!), small notches (crenellated)
  Row 3  PP  ≥ 0.50   blind spots: smooth deep concavity (pac-man, horseshoe)
  Row 4  all three    residual loophole: diagonal orientation (parallelogram)

Produces: output/shapes/three_metrics.png
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from pathlib import Path
import pickle
import sys

OUT = Path("output/shapes")
OUT.mkdir(parents=True, exist_ok=True)
C  = 180
TM, TC, TP = 0.50, 0.90, 0.50


# ── metrics ───────────────────────────────────────────────────────────────────

def moi(mask):
    H,W=mask.shape; Y,X=np.mgrid[0:H,0:W]; a=mask.astype(float); area=a.sum()
    if area==0: return 0.0
    cx=(X*a).sum()/area; cy=(Y*a).sum()/area
    m=((X-cx)**2+(Y-cy)**2)[mask].sum()
    return float(area**2/(2*np.pi)/m) if m>0 else 0.0

def ch(mask):
    ys,xs=np.where(mask)
    if len(xs)<4: return 0.0
    try: return min(1.0, float(mask.sum()/ConvexHull(np.column_stack([xs,ys])).volume))
    except: return 0.0

def pp(mask):
    area=mask.sum()
    pad=np.pad(mask,1,constant_values=False)
    perim=(mask&(~pad[:-2,1:-1]|~pad[2:,1:-1]|~pad[1:-1,:-2]|~pad[1:-1,2:])).sum()
    return float(4*np.pi*area/perim**2) if perim>0 else 0.0

def s3(mask): return moi(mask), ch(mask), pp(mask)


# ── shape builders ────────────────────────────────────────────────────────────

def rect(h, w):
    m=np.zeros((C,C),dtype=bool)
    m[max(0,C//2-h//2):min(C,C//2-h//2+h),
      max(0,C//2-w//2):min(C,C//2-w//2+w)]=True
    return m

def horseshoe(oh, ow, nh, nw):
    m=np.zeros((C,C),dtype=bool)
    r0=max(0,C//2-oh//2); c0=max(0,C//2-ow//2)
    m[r0:min(C,r0+oh), c0:min(C,c0+ow)]=True
    m[r0:min(C,r0+nh), max(0,C//2-nw//2):min(C,C//2+nw//2)]=False
    return m

def lshape(vh, vw, fh, fw):
    m=np.zeros((C,C),dtype=bool); c0=C//2-fw//2
    m[C//2-vh//2:C//2+vh//2, c0:min(C,c0+vw)]=True
    m[C//2+vh//2-fh:C//2+vh//2, c0:min(C,c0+fw)]=True
    return m

def crescent(R_big, R_small, offset_x):
    m=np.zeros((C,C),dtype=bool); Y,X=np.mgrid[0:C,0:C]
    m[(X-C//2)**2+(Y-C//2)**2<=R_big**2]=True
    m[(X-C//2-offset_x)**2+(Y-C//2)**2<=R_small**2]=False
    return m

def comb(h, w, tooth_h, tooth_w, n_teeth):
    m=np.zeros((C,C),dtype=bool)
    r0=C//2-h//2; c0=C//2-w//2
    m[max(0,r0):min(C,r0+h), max(0,c0):min(C,c0+w)]=True
    spacing=(w-n_teeth*tooth_w)//(n_teeth+1)
    for i in range(n_teeth):
        tc=c0+spacing*(i+1)+tooth_w*i
        m[max(0,r0):min(C,r0+tooth_h), max(0,tc):min(C,tc+tooth_w)]=False
    return m

def crenellated(h, w, tooth_h, tooth_w, n_per_side):
    m=np.zeros((C,C),dtype=bool)
    r0=C//2-h//2; c0=C//2-w//2
    m[max(0,r0):min(C,r0+h), max(0,c0):min(C,c0+w)]=True
    spacing=(w-n_per_side*tooth_w)//(n_per_side+1)
    for i in range(n_per_side):
        tc=c0+spacing*(i+1)+tooth_w*i
        m[max(0,r0):min(C,r0+tooth_h), max(0,tc):min(C,tc+tooth_w)]=False
        m[max(0,r0+h-tooth_h):min(C,r0+h), max(0,tc):min(C,tc+tooth_w)]=False
    return m

def pacman(R, angle_deg):
    """Circle with wedge removed — passes PP (smooth curves), fails CH."""
    m=np.zeros((C,C),dtype=bool); Y,X=np.mgrid[0:C,0:C]
    dx=X-C//2; dy=Y-C//2
    dist=np.sqrt(dx**2+dy**2)
    ang=np.arctan2(dy.astype(float),dx.astype(float))
    half=np.radians(angle_deg/2)
    m[(dist<=R)&(np.abs(ang)>half)]=True
    return m

def parallelogram(h, w, shear):
    """Horizontally sheared rectangle — passes all three, diagonal orientation loophole."""
    m=np.zeros((C,C),dtype=bool)
    extra=abs(shear)
    r0=C//2-h//2; c0=C//2-(w+extra)//2
    for dr in range(h):
        shift=int(dr*shear/h)
        cs=c0+shift
        m[max(0,r0+dr), max(0,cs):min(C,cs+w)]=True
    return m

def arc_shape(R_out, R_in, angle_deg):
    """Thick arc (annular sector) — smooth inner + outer circular boundaries, deep concavity."""
    m=np.zeros((C,C),dtype=bool); Y,X=np.mgrid[0:C,0:C]
    dx=X-C//2; dy=Y-C//2
    dist=np.sqrt(dx**2+dy**2)
    ang=np.arctan2(dy.astype(float),dx.astype(float))
    half=np.radians(angle_deg/2)
    m[(dist>=R_in)&(dist<=R_out)&(np.abs(ang)<=half)]=True
    return m


# ── search helpers ────────────────────────────────────────────────────────────

def best_match(candidates, wm, wc, wp, top=1):
    results=[]
    for label,mask in candidates:
        if mask.sum()<200: continue
        mo,c,p=s3(mask)
        err=abs(mo-TM)*wm+abs(c-TC)*wc+abs(p-TP)*wp
        results.append((err,mo,c,p,label,mask))
    results.sort()
    return results[:top]

def best_ch_near(pool, target_moi=None, tol_c=0.06):
    """Find shape with CH closest to TC=0.90, optionally biased toward target_moi."""
    cands=[]
    for lbl,msk in pool:
        if msk.sum()<200: continue
        mo,c,p=s3(msk)
        if abs(c-TC)>=tol_c: continue
        score=abs(c-TC)
        if target_moi is not None: score+=abs(mo-target_moi)*0.3
        cands.append((score,mo,c,p,lbl,msk))
    cands.sort(); return cands[0] if cands else None

def best_for_all3(pool, tol_m=0.10):
    """Find shapes that actually pass all three thresholds (small MOI slack only)."""
    cands=best_match(pool, 1,1,1, top=len(pool))
    return [it for it in cands
            if it[1]>=TM-tol_m and it[2]>=TC and it[3]>=TP]


SHAPE_CACHE = OUT / "three_metrics_shapes.pkl"
REBUILD     = "--rebuild" in sys.argv


def _search_shapes():
    print("Building candidate pools...")
    rects    = [(f"rect{h}x{w}", rect(h,w))
                for h in range(28,90,3) for w in range(60,176,3)]
    crescents= [(f"cr{R},{r},{dx}", crescent(R,r,dx))
                for R in range(40,80,5) for r in range(20,65,4)
                for dx in range(5,50,4) if r<R and dx+r<R+10]
    combs    = [(f"comb{h},{w},{th},{tw},{n}", comb(h,w,th,tw,n))
                for h in range(40,80,5) for w in range(100,175,6)
                for th in range(12,30,3) for tw in range(8,20,3)
                for n in range(4,10,1)]
    crenels  = [(f"cren{h},{w},{th},{tw},{n}", crenellated(h,w,th,tw,n))
                for h in range(40,120,8) for w in range(60,170,8)
                for th in range(4,20,3) for tw in range(4,20,3)
                for n in range(3,9,1)]
    pacmen   = [(f"pac{R},{a}", pacman(R,a))
                for R in range(44,80,4) for a in range(50,280,10)]
    pp50_rects= [(f"rect{h}x{w}", rect(h,w))
                 for h in range(36,52,1) for w in range(140,176,1)]
    pp50_shoes= [(f"sh{oh},{ow},{nh},{nw}", horseshoe(oh,ow,nh,nw))
                 for oh in range(40,160,6) for ow in range(80,170,6)
                 for nh in range(4,20,2) for nw in range(60,140,4)
                 if nh<oh-4 and nw<ow-4]
    shoes    = [(f"sh{oh},{ow},{nh},{nw}", horseshoe(oh,ow,nh,nw))
                for oh in range(40,150,8) for ow in range(40,170,8)
                for nh in range(8,80,8) for nw in range(8,100,8)
                if nh<oh-4 and nw<ow-4]
    lshapes  = [(f"L{vh},{vw},{fh},{fw}", lshape(vh,vw,fh,fw))
                for vh in range(30,130,6) for vw in range(12,50,4)
                for fh in range(8,44,4) for fw in range(30,130,7)]
    paras    = [(f"para{h},{w},{s}", parallelogram(h,w,s))
                for h in range(40,90,6) for w in range(60,150,8)
                for s in range(20,110,8)]
    arcs     = [(f"arc{ro},{ri},{a}", arc_shape(ro,ri,a))
                for ro in range(50,85,5) for ri in range(15,55,5)
                for a in range(100,300,15) if ri<ro-10]

    r1_normal   = best_match(rects,    1,0,0)[0]
    r1_comb     = best_match(combs,    1,0,0)[0]
    r1_crescent = best_match(crescents,1,0,0)[0]
    row1 = [(r1_normal[5],"compact rectangle"),(r1_comb[5],"comb"),(r1_crescent[5],"crescent")]

    r2_compact   = best_ch_near(shoes,         target_moi=0.85)
    r2_elongated = best_ch_near(shoes+lshapes, target_moi=0.50)
    r2_crenel    = best_ch_near(crenels)
    row2 = [
        (r2_compact[5]   if r2_compact   else rect(88,88),   "compact horseshoe"),
        (r2_elongated[5] if r2_elongated else rect(16,170),  "elongated notch"),
        (r2_crenel[5]    if r2_crenel    else crenels[0][1], "crenellated"),
    ]

    r3_normal = best_match(pp50_rects, 0,0,1)[0]
    r3_arc    = best_match(arcs,       0,0,1)[0]
    r3_shoe   = best_match(pp50_shoes, 0,0,1)[0]
    row3 = [(r3_normal[5],"compact rectangle"),(r3_arc[5],"arc (C-shape)"),(r3_shoe[5],"horseshoe")]

    def passing_display(it, moi_min=0.495):
        return it[1]>=moi_min and it[2]>=TC and it[3]>=0.495
    r4_shoe_cands = [it for it in best_match(shoes,   1,1,1, top=len(shoes))   if passing_display(it)]
    r4_l_cands    = [it for it in best_match(lshapes, 1,1,1, top=len(lshapes)) if passing_display(it)]
    r4_shoe = r4_shoe_cands[0] if r4_shoe_cands else best_match(shoes,   1,1,1)[0]
    r4_l    = r4_l_cands[0]    if r4_l_cands    else best_match(lshapes, 1,1,1)[0]
    row4 = [(r4_l[5],"L-shape"),(r4_shoe[5],"horseshoe")]

    return row1, row2, row3, row4


if not REBUILD and SHAPE_CACHE.exists():
    print("Loading cached shapes (pass --rebuild to redo the search)...")
    with open(SHAPE_CACHE, "rb") as f:
        row1, row2, row3, row4 = pickle.load(f)
else:
    row1, row2, row3, row4 = _search_shapes()
    with open(SHAPE_CACHE, "wb") as f:
        pickle.dump((row1, row2, row3, row4), f)
    print(f"Shapes cached to {SHAPE_CACHE}")


# ── Verify ────────────────────────────────────────────────────────────────────

rows_data = [
    ("Moment of Inertia", "m",   row1, [("MOI","m")]),
    ("Convex Hull ratio", "c",   row2, [("CH","c")]),
    ("Polsby-Popper",     "p",   row3, [("PP","p")]),
    ("All three",         "mcp", row4, [("MOI","m"),("CH","c"),("PP","p")]),
]

score_fn = {"m": lambda mo,c,p: mo, "c": lambda mo,c,p: c, "p": lambda mo,c,p: p}

for rname, hl, shapes, score_lines in rows_data:
    print(f"\n{rname.replace(chr(10),' | ')}")
    for shape, lbl in shapes:
        mo,c,p=s3(shape)
        print(f"  {lbl:22s}  moi={mo:.3f}  ch={c:.3f}  pp={p:.3f}")


# ── figure ────────────────────────────────────────────────────────────────────

BG    = "#15110c"
PANEL = "#1c1710"
INK   = "#ece4d6"
MUT   = "#9a8f7c"
GOLD  = "#e0c06e"
LINE  = "#352c20"

score_anchors = {1:[0.10], 2:[0.16,0.05], 3:[0.21,0.12,0.03]}

fig, axes = plt.subplots(4, 3, figsize=(7.5, 7.0),
                         gridspec_kw={"hspace": 0.08, "wspace": 0.06},
                         facecolor=BG)
fig.patch.set_facecolor(BG)
fig.subplots_adjust(top=0.91, bottom=0.02, left=0.25, right=0.97)

for row_i, (rname, hl, shapes, score_lines) in enumerate(rows_data):
    for col_i, (shape, lbl) in enumerate(shapes):
        ax = axes[row_i, col_i]
        ax.set_facecolor(PANEL)

        img=np.zeros((*shape.shape,4))
        img[shape]=[0x7a/255,0x9a/255,0x5a/255,0.85]
        ax.imshow(img, origin="upper")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(LINE); sp.set_linewidth(0.7)

        ax.text(0.5, 0.97, lbl, transform=ax.transAxes,
                ha="center", va="top", color=GOLD, fontsize=6.5, fontweight="bold")

    # Hide unused cells in this row
    for col_i in range(len(shapes), 3):
        axes[row_i, col_i].set_visible(False)

    # Row label: "Metric name = value" all in one color
    mo,c,p = s3(shapes[0][0])
    if len(score_lines) == 1:
        slbl, sflag = score_lines[0]
        val = min(1.0, score_fn[sflag](mo,c,p))
        label_text = f"{rname}\n= {val:.1f}"
    else:
        scores_str = "\n".join(
            f"{slbl} = {min(1.0, score_fn[sflag](mo,c,p)):.1f}"
            for slbl, sflag in score_lines
        )
        label_text = f"{rname}\n{scores_str}"
    axes[row_i, 0].text(-0.05, 0.5, label_text,
                        transform=axes[row_i,0].transAxes,
                        ha="right", va="center", color=GOLD,
                        fontsize=8, fontweight="bold", linespacing=1.6)

fig.suptitle("Three compactness metrics: each row shows three shapes\nwith the same score on that metric",
             color=INK, fontsize=11, fontweight="bold", y=0.97)

out = OUT/"three_metrics.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"\nSaved → {out}")
