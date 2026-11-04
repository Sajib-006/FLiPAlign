"""Figure: seven real cross-platform pairs scored on hand-placed landmarks.
One row per pair. x = median landmark error (um, log scale). Shaded band = right (below floor + 100 um).
FLiPAlign: blue circle, filled if certified, hollow if refused. Best published tool on that pair (nothing handed over,
best setting chosen with the truth): orange diamond with its name. Other tools: small grey dots.
Run from paper_scripts/ (env RESULTS, OUT): python3 make_fig_landmarks.py -> figures/landmarks.pdf"""
import os
from _paths import RES, PLACE, STACK, OUT, TABLES, FIGURES
import glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = PLACE
BLUE, ORANGE, GREY, INK, BAND = "#2a78d6", "#eb6834", "#9a9a95", "#222222", "#e6efe6"
plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
NAMES = {1: "Breast Rep1\nVisium $\\rightarrow$ Xenium", 2: "Breast Rep2\nVisium $\\rightarrow$ Xenium", 3: "Colon P2CRC\nVisium $\\rightarrow$ Visium HD",
         4: "Colon P3NAT\nVisium $\\rightarrow$ Visium HD", 5: "Colon P5NAT\nVisium $\\rightarrow$ Visium HD", 6: "Lung HD1\nVisium HD $\\rightarrow$ Xenium",
         7: "Lung HD2\nVisium HD $\\rightarrow$ Xenium 5K"}
TOOL = {"paste": "PASTE", "paste2": "PASTE2", "slat": "SLAT", "cpd": "CPD", "spateo": "spateo", "cast": "CAST", "stalign": "STalign", "gala": "GALA"}
rows = []
for f in glob.glob(f"{R}/lm_given_pair*_*.json"):
    rows += [r for r in json.load(open(f)) if "landmark_err_median_um" in r]
fl = {k: json.load(open(f"{R}/lm_flip_pair{k}.json")) for k in range(1, 8)}
fig, ax = plt.subplots(figsize=(4.8, 3.35))
lo, hi = 12, 12000
for i, k in enumerate(range(1, 8)):
    y = 7 - i; floor = fl[k]["floor_um"]
    ax.add_patch(plt.Rectangle((lo, y - 0.38), floor + 100 - lo, 0.76, color=BAND, lw=0, zorder=0))
    ax.plot([floor, floor], [y - 0.38, y + 0.38], color="#7fae7f", lw=0.8, zorder=1)
    best = {}
    for r in rows:
        if r["pair"] != k: continue
        t = r["method"].split("_")[0]; best[t] = min(best.get(t, 1e9), r["landmark_err_median_um"])
    bt = min(best, key=best.get)
    for t, v in best.items():
        if t != bt: ax.scatter(min(v, hi * 0.95), y - 0.16, s=9, color=GREY, lw=0, zorder=2)
    ax.scatter(min(best[bt], hi * 0.95), y - 0.16, s=26, marker="D", color=ORANGE, edgecolor="white", lw=0.6, zorder=3)
    ax.text(1.02, y, TOOL[bt], transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=8.5, color=INK)   # best tool, right margin
    e = fl[k]["coarse"]["median_um"]; cert = fl[k]["flipalign_certified"]
    ax.scatter(min(e, hi * 0.95), y + 0.16, s=30, marker="o", color=BLUE if cert else "white", edgecolor=BLUE, lw=1.1, zorder=4)
ax.set_xscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(0.4, 7.6)
ax.set_yticks(range(7, 0, -1)); ax.set_yticklabels([NAMES[k] for k in range(1, 8)], fontsize=8.5)
ax.set_xlabel("median landmark error ($\\mu$m, log scale)")
ax.tick_params(axis="y", length=0)
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
h = [Line2D([], [], marker="o", ls="", color=BLUE, ms=5, label="FLiPAlign, certified"),
     Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=BLUE, ms=5, label="FLiPAlign, refused"),
     Line2D([], [], marker="D", ls="", color=ORANGE, ms=4.5, label="best tool on the pair"),
     Line2D([], [], marker="o", ls="", color=GREY, ms=3, label="other tools"),
     Patch(color=BAND, label=r"right ($<$ floor + 100 $\mu$m)"),
     Line2D([], [], color="#7fae7f", lw=0.8, label="floor (best rigid fit)")]
ax.text(1.02, 7.75, "best tool", transform=ax.get_yaxis_transform(), ha="left", va="bottom", fontsize=8.5, color=INK)
ax.legend(handles=h, bbox_to_anchor=(0.45, -0.22), loc="upper center", ncol=3, frameon=False, fontsize=8.5, handletextpad=0.3, columnspacing=1.0)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES, "landmarks.pdf"), bbox_inches="tight", pad_inches=0.02); fig.savefig(os.path.join(FIGURES, "landmarks.png"), dpi=220, bbox_inches="tight", pad_inches=0.02)
print("saved")
