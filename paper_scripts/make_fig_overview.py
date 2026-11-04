"""Figure 1 (overview) and the stack example, from real data (results/stack/fig_data.npz, made by fig_data.py).
Overview: (a) section A as given (turned, mirrored), own labels; (b) section B, own labels (colours do not correspond);
(c) A placed on B by FLiPAlign's certified answer; (d) three serial sections, own labels; (e) the same after split
labelling, one set of labels for the stack.
Stack example: three sections; rows = atlas division, best one-to-one labelling, split labelling. Colours of a
labelling are matched to the atlas divisions by an assignment on overlap, for display only.
Run from paper_scripts/ (env RESULTS, OUT): python3 make_fig_overview.py -> figures/overview2.pdf, figures/stack_example.pdf"""
import os
from _paths import RES, PLACE, STACK, OUT, TABLES, FIGURES
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

z = np.load(os.path.join(STACK, "fig_data.npz"), allow_pickle=True)
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GREY, INK = "#c9c9c4", "#222222"
plt.rcParams.update({"font.size": 7, "pdf.fonttype": 42, "font.family": "serif", "font.serif": ["cmr10"],
                     "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True})   # Computer Modern, as in the paper body


def dots(ax, xy, lab, s=0.6, alpha=0.9, perm=None):
    lab = np.asarray(lab); cols = np.array(PAL + [GREY] * 20)
    idx = lab if perm is None else np.array([perm.get(int(v), 8) for v in lab])
    ax.scatter(xy[:, 0], xy[:, 1], s=s, c=cols[idx], lw=0, alpha=alpha, rasterized=True)
    ax.set_aspect("equal"); ax.axis("off")


def frame(ax, xy, pad=0.04):
    lo, hi = xy.min(0), xy.max(0); d = (hi - lo) * pad
    ax.set_xlim(lo[0] - d[0], hi[0] + d[0]); ax.set_ylim(lo[1] - d[1], hi[1] + d[1])


# ---------------- overview
from matplotlib.patches import FancyArrowPatch
fig = plt.figure(figsize=(4.8, 1.75))
W, H, Y = 0.17, 0.74, 0.02
WS = 0.15                       # same box for every panel (figure fractions)
X = [0.0, 0.19, 0.44, 0.66, 0.83]
ax = [fig.add_axes([x, Y, W if i < 3 else WS, H]) for i, x in enumerate(X)]
rng = np.random.default_rng(3); permB = {i: int(v) for i, v in enumerate(rng.permutation(8))}
def fit_box(a, xy):                               # fill the panel, keep true proportions, sit on the panel bottom
    lo, hi = xy.min(0), xy.max(0); a.set_xlim(lo[0], hi[0]); a.set_ylim(lo[1], hi[1])
    a.set_aspect("equal", adjustable="box", anchor="C")
dots(ax[0], z["p_A"], z["p_la"], s=0.15); fit_box(ax[0], z["p_A"])
dots(ax[1], z["p_B"], z["p_lb"], s=0.15, perm=permB); fit_box(ax[1], z["p_B"])
ax[2].scatter(z["p_B"][:, 0], z["p_B"][:, 1], s=0.15, c="#dcdcd7", lw=0, rasterized=True)
dots(ax[2], z["p_Aplaced"], z["p_la"], s=0.13, alpha=0.55); fit_box(ax[2], np.vstack([z["p_B"], z["p_Aplaced"]]))
h = np.ptp(z["s0_xy"][:, 1]); stack = []
for j in range(3):
    sh = z[f"s{j}_xy"] - z[f"s{j}_xy"].mean(0) + np.array([0, -j * 1.1 * h]); stack.append(sh)
    rp = np.random.default_rng(10 + j).permutation(8)
    dots(ax[3], sh, z[f"s{j}_own"], s=0.08, perm={i: int(v) for i, v in enumerate(rp)})
    dots(ax[4], sh, z[f"s{j}_split"] % 8, s=0.08)
for a_ in ax[3:]: fit_box(a_, np.vstack(stack))
for a_, (l, t) in zip(ax, [("a", "Section A"), ("b", "Section B"), ("c", "A on B"), ("d", "Own labels"), ("e", "One set")]):
    fig.text(X[ax.index(a_)] + 0.005, Y + H + 0.015, l, fontsize=9, ha="left", va="bottom", family="cmb10")
    fig.text(X[ax.index(a_)] + 0.036, Y + H + 0.015, t, fontsize=8.5, ha="left", va="bottom")
from matplotlib.patches import FancyBboxPatch
for x0, x1, n, txt in ((0.0, 0.62, "1", "Placement"), (0.655, 0.99, "2", "Labelling")):
    fig.add_artist(FancyBboxPatch((x0, 0.885), x1 - x0, 0.085, boxstyle="round,pad=0,rounding_size=0.012",
                                  transform=fig.transFigure, fc="#eef2f7", ec="none"))
    fig.text((x0 + x1) / 2, 0.927, f"{n}   {txt}", fontsize=9.5, ha="center", va="center", color=INK, family="cmb10")
for x0, x1, lab in ((0.37, 0.43, "search"), (0.80, 0.835, "split")):
    fig.add_artist(FancyArrowPatch((x0, Y + H / 2), (x1, Y + H / 2), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=8, lw=0.8, color=INK))
    fig.text((x0 + x1) / 2, Y + H / 2 + 0.04, lab, fontsize=8.5, ha="center", va="bottom", color=INK)
fig.savefig(os.path.join(FIGURES, "overview2.pdf"), bbox_inches="tight", pad_inches=0.02); fig.savefig(os.path.join(FIGURES, "overview2.png"), dpi=220, bbox_inches="tight", pad_inches=0.02)

# ---------------- stack example (atlas / one-to-one / split), colours matched to atlas divisions
divs = np.concatenate([z[f"s{j}_div"] for j in range(3)])
u, c = np.unique(divs[divs != "unassigned"], return_counts=True); top = list(u[np.argsort(-c)][:8])
def match(key):
    """each class takes the colour of the atlas division it mostly covers (same rule for every method)"""
    lab = np.concatenate([z[f"s{j}_{key}"] for j in range(3)]); out = {}
    for l in np.unique(lab):
        d, n = np.unique(divs[lab == l], return_counts=True); m = d[np.argmax(n)]
        out[int(l)] = top.index(m) if m in top else 8
    return out
pa = {d: i for i, d in enumerate(top)}
fig, ax = plt.subplots(3, 3, figsize=(4.6, 3.2))
for j in range(3):
    xy = z[f"s{j}_xy"]
    d = z[f"s{j}_div"]; ax[0, j].scatter(xy[:, 0], xy[:, 1], s=0.35, lw=0, rasterized=True,
                                         c=np.array(PAL + [GREY])[[pa.get(v, 8) for v in d]]); ax[0, j].set_aspect("equal"); ax[0, j].axis("off")
    dots(ax[1, j], xy, z[f"s{j}_one"], s=0.35, perm=match("one"))
    dots(ax[2, j], xy, z[f"s{j}_split"], s=0.35, perm=match("split"))
    ax[0, j].set_title(f"section {j + 1}", fontsize=6.5)
for i, t in enumerate(["atlas division", "one-to-one", "split (ours)"]):
    ax[i, 0].text(-0.08, 0.5, t, transform=ax[i, 0].transAxes, rotation=90, va="center", ha="right", fontsize=6.5)
fig.subplots_adjust(wspace=0.02, hspace=0.06, left=0.06, right=0.99, top=0.93, bottom=0.01)
fig.savefig(os.path.join(FIGURES, "stack_example.pdf")); fig.savefig(os.path.join(FIGURES, "stack_example.png"), dpi=220)
print("top divisions", top)
