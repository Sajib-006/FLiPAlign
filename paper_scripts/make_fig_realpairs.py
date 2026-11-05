"""Figure: the seven real cross-platform pairs after FLiPAlign, scored on hand-placed landmarks.
(a) each pair after FLiPAlign's frozen answer: section B in grey, section A placed on it in its own labels, and the
    landmarks (orange: B as clicked; dark: A's landmarks moved by the answer; line: the error). Tag: certified or
    refused, right or wrong (right: median error below the floor plus 100 um), median error.
(b) median landmark error per pair: FLiPAlign, the best published tool on that pair (best setting chosen with the
    truth, nothing handed over), the other tools, and the floor (best rigid fit to the landmarks).
Reads results/placement/fig_realpairs.npz (runs_placement/fig_realpairs_data.py), lm_flip_pair*.json, lm_given_pair*_*.json.
Run from paper_scripts/ (env RESULTS, OUT): python3 make_fig_realpairs.py -> figures/fig_realpairs.pdf"""
import os
from _paths import PLACE, FIGURES
from figstyle import *
import glob, json
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

R = PLACE
Z = np.load(os.path.join(R, "fig_realpairs.npz"))
FL = {k: json.load(open(f"{R}/lm_flip_pair{k}.json")) for k in range(1, 8)}
NAME = {1: "Breast Rep1", 2: "Breast Rep2", 3: "Colon P2CRC", 4: "Colon P3NAT", 5: "Colon P5NAT", 6: "Lung HD1", 7: "Lung HD2"}
SHORT = {1: "Rep1", 2: "Rep2", 3: "P2CRC", 4: "P3NAT", 5: "P5NAT", 6: "HD1", 7: "HD2"}
PLAT = {1: "Visium → Xenium", 2: "Visium → Xenium", 3: "Visium → Visium HD", 4: "Visium → Visium HD",
        5: "Visium → Visium HD", 6: "Visium HD → Xenium", 7: "Visium HD → Xenium 5K"}
TOOL = {"paste": "PASTE", "paste2": "PASTE2", "slat": "SLAT", "cpd": "CPD", "spateo": "spateo", "cast": "CAST", "stalign": "STalign", "gala": "GALA"}
rows = []
IMG = {}                                    # rigid registration of the two tissue photos (runs_baselines/image_reg.py)
for f in glob.glob(f"{R}/lm_given_pair*_*.json"):
    rr = [r for r in json.load(open(f)) if "landmark_err_median_um" in r]
    if f.endswith("_imagereg.json"):
        for r in rr:
            IMG[r["pair"]] = min(IMG.get(r["pair"], 1e9), r["landmark_err_median_um"])
        continue
    rows += rr
best = {k: {} for k in range(1, 8)}
for r in rows:
    t = r["method"].split("_")[0]; k = r["pair"]
    best[k][t] = min(best[k].get(t, 1e9), r["landmark_err_median_um"])

FH = 3.72
fig = plt.figure(figsize=(TEXTW, FH))
# ---------------------------------------------------------------- (a) seven overlays, 2 rows x 4 (last cell: key)
inch = lambda x, y, w, h: [x / TEXTW, y / FH, w / TEXTW, h / FH]
MW = 0.84                                           # map size (inches)
X0 = [0.10 + i * 1.19 for i in range(4)]
Y0 = [2.68, 1.40]
axes, SHOWN = {}, {}
for i, k in enumerate(range(1, 8)):
    ax = fig.add_axes(inch(X0[i % 4], Y0[i // 4], MW, MW)); axes[k] = ax
    A, B, la = Z[f"p{k}_A"], Z[f"p{k}_B"], Z[f"p{k}_la"]
    LA, LB = Z[f"p{k}_LA"], Z[f"p{k}_LB"]
    ax.scatter(B[:, 0], B[:, 1], s=0.35, c="#CFCFCF", lw=0, rasterized=True, zorder=1)
    ax.scatter(A[:, 0], A[:, 1], s=0.35 * min(3.0, 9000 / len(A)), c=np.array(REGION)[la % len(REGION)], lw=0, alpha=0.75, rasterized=True, zorder=2)
    pts = np.vstack([B, A]); lo, hi = np.percentile(pts, 0.3, 0), np.percentile(pts, 99.7, 0)
    c, half = (lo + hi) / 2, (hi - lo).max() / 2 * 1.08
    inside = np.all(np.abs(LB - c) < half * 0.98, axis=1)     # landmarks outside the analysed window are not drawn
    LA, LB = LA[inside], LB[inside]; SHOWN[k] = inside.sum()
    for p, q in zip(LA, LB):
        ax.plot([p[0], q[0]], [p[1], q[1]], color=INK, lw=0.45, zorder=3, solid_capstyle="round")
    ax.scatter(LB[:, 0], LB[:, 1], s=4, c=ACCENT, edgecolor=INK, lw=0.25, zorder=4, clip_on=False)
    ax.scatter(LA[:, 0], LA[:, 1], s=1.6, c=INK, lw=0, zorder=5)
    ax.set_xlim(c[0] - half, c[0] + half); ax.set_ylim(c[1] + half, c[1] - half)   # image orientation
    ax.set_aspect("equal"); ax.axis("off")
    x1 = c[0] + half * 0.92; y1 = c[1] + half * 0.93                                   # 1 mm scale bar, bottom right
    ax.plot([x1 - 1000, x1], [y1, y1], color=INK, lw=1.0, solid_capstyle="butt", zorder=6)
    ax.set_title(NAME[k], fontsize=FS, pad=4)
    f = FL[k]; e = f["coarse"]["median_um"]; cert = f["flipalign_certified"]; right = f["right_within_floor_plus_100"]["coarse"]
    status = ("certified" if cert else "refused") + (", right" if right else ", wrong")
    col = (RIGHT if right else WRONG) if cert else MUTED
    es = f"{e / 1000:.1f} mm" if e >= 1000 else f"{e:.0f} µm"
    ax.text(0.5, -0.03, PLAT[k], transform=ax.transAxes, ha="center", va="top", fontsize=FS_SMALL, color=MUTED)
    ax.text(0.5, -0.13, f"{status}, {es}", transform=ax.transAxes, ha="center", va="top", fontsize=FS_SMALL,
            color=col, fontweight="bold" if cert else "normal")
letter(fig, axes[1], "a", dx=-0.004, dy=0.022)
# key in the eighth cell
kx = fig.add_axes(inch(X0[3], Y0[1], MW, MW)); kx.axis("off")
h = [Line2D([], [], marker="o", ls="", ms=3.5, mfc="#CFCFCF", mec="none", label="section B"),
     Line2D([], [], marker="o", ls="", ms=3.5, mfc=REGION[1], mec="none", label="section A, placed\n(own labels)"),
     Line2D([], [], marker="o", ls="", ms=3.2, mfc=ACCENT, mec=INK, mew=0.3, label="landmark of B"),
     Line2D([0, 1], [0, 0], color=INK, lw=0.6, marker="o", ms=1.6, markevery=[0], label="landmark of A,\nmoved (line: error)"),
     Line2D([0, 1], [0, 0], color=INK, lw=1.0, label="1 mm")]
kx.legend(handles=h, loc="center left", bbox_to_anchor=(-0.04, 0.45), fontsize=FS_SMALL, labelspacing=0.55, handlelength=1.4)

# ---------------------------------------------------------------- (b) landmark error per pair
bx = fig.add_axes(inch(0.62, 0.20, 2.92, 0.72))
lo_, hi_ = 20, 10000
for i, k in enumerate(range(1, 8)):
    floor = FL[k]["floor_um"]
    bx.add_patch(Rectangle((i - 0.38, lo_), 0.76, floor + 100 - lo_, color=RIGHT, alpha=0.13, lw=0, zorder=0))
    bx.plot([i - 0.38, i + 0.38], [floor, floor], color=RIGHT, lw=0.8, zorder=1)
    bt = min(best[k], key=best[k].get)
    others = [v for t, v in best[k].items() if t != bt]
    bx.scatter(np.full(len(others), i + 0.17) + np.linspace(-0.05, 0.05, len(others)), np.clip(others, lo_, hi_ * 0.97),
               s=5, c=REFUSED, lw=0, zorder=2)
    bx.scatter(i + 0.17, min(best[k][bt], hi_ * 0.97), s=22, marker="^", c=BASE, edgecolor="white", lw=0.5, zorder=3)
    e = FL[k]["coarse"]["median_um"]; cert = FL[k]["flipalign_certified"]
    bx.scatter(i - 0.17, min(e, hi_ * 0.97), s=26, marker="o", c=OURS if cert else "white", edgecolor=OURS, lw=1.0, zorder=4)
    if k in IMG:
        bx.scatter(i + 0.32, min(IMG[k], hi_ * 0.97), s=14, marker="s", c="white", edgecolor=INK, lw=0.7, zorder=3)
    bx.text(i + 0.17, 1.03, TOOL[bt], transform=bx.get_xaxis_transform(), ha="center", va="bottom", fontsize=FS_SMALL - 0.5, color=MUTED)
bx.set_yscale("log"); bx.set_ylim(lo_, hi_); bx.set_xlim(-0.55, 6.55)
bx.set_xticks(range(7)); bx.set_xticklabels([SHORT[k] for k in range(1, 8)])
bx.tick_params(axis="x", length=0)
bx.set_ylabel("median landmark\nerror (µm)")
bx.text(-0.62, 1.03, "best tool", transform=bx.get_xaxis_transform(), ha="right", va="bottom", fontsize=FS_SMALL - 0.5, color=MUTED)
bx.set_yticks([100, 1000, 10000]); bx.set_yticklabels(["100", "1,000", "10,000"])
h2 = [Line2D([], [], marker="o", ls="", ms=4.2, mfc=OURS, mec=OURS, label="FLiPAlign, certified"),
      Line2D([], [], marker="o", ls="", ms=4.2, mfc="white", mec=OURS, mew=1.0, label="FLiPAlign, refused"),
      Line2D([], [], marker="^", ls="", ms=5, mfc=BASE, mec="white", label="best tool"),
      Line2D([], [], marker="o", ls="", ms=2.6, mfc=REFUSED, mec="none", label="other tools"),
      Line2D([], [], marker="s", ls="", ms=3.2, mfc="white", mec=INK, mew=0.7, label="photo registration"),
      Patch(color=RIGHT, alpha=0.18, label="right (below floor\n+ 100 µm)"),
      Line2D([], [], color=RIGHT, lw=0.8, label="floor (best rigid\nfit to landmarks)")]
bx.legend(handles=h2, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=FS_SMALL, labelspacing=0.45)
fig.text(0.004, (0.20 + 0.72 + 0.10) / FH, "b", fontsize=FS_LETTER, fontweight="bold", va="bottom", ha="left")

save(fig, os.path.join(FIGURES, "fig_realpairs"))
for k in range(1, 8):
    bt = min(best[k], key=best[k].get)
    print(k, NAME[k], "landmarks drawn", int(SHOWN[k]), "of", len(Z[f"p{k}_LB"]), "FLiPAlign", FL[k]["coarse"]["median_um"], "cert", FL[k]["flipalign_certified"], "floor", FL[k]["floor_um"],
          "best tool", bt, best[k][bt])
