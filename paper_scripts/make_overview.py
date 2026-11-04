"""Figure 1: the method on a small made-up tissue (scores computed in ../explainer/toy.py).
(a) two sections with their own labels; (b) B turned into smooth label maps; (c) one placement is
scored by the best label pairing; (d) branch-and-bound ceilings per rotation range, then the
decision: every far ceiling is below the best score, so the answer is certified."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

o = json.load(open("../explainer/toy.json")); T = o["patch"]
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
CA = ["#e0612b", "#d44f96", "#c08a00"]; CB = [(42, 120, 214), (22, 156, 108), (138, 99, 210)]
CBh = ["#2a78d6", "#169c6c", "#8a63d2"]
plt.rcParams.update({"font.size": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "savefig.facecolor": SURF})
LO, N = -140.0, 281
G = np.asarray(T["G"])                                      # (3, y, x)
B, lb = np.asarray(T["B"]), np.asarray(T["lb"]); A, la = np.asarray(T["A"]), np.asarray(T["la"])
bc = B.mean(0)
ext = [LO, LO + N - 1, LO, LO + N - 1]


def rgba(ls, alpha):
    g = G[ls]; tot = g.sum(0)
    rgb = sum(g[i][..., None] * np.array(CB[l]) for i, l in enumerate(ls)) / np.maximum(tot, 1e-6)[..., None] / 255
    return np.dstack([rgb, np.clip(tot, 0, 1) * alpha])


fig, ax = plt.subplots(2, 2, figsize=(4.8, 3.3), gridspec_kw={"hspace": 0.45, "wspace": 0.25})
# (a)
a = ax[0, 0]
a.scatter(A[:, 0] + bc[0] - 150, A[:, 1] + bc[1], s=1.2, c=[CA[k] for k in la], lw=0)
a.scatter(B[:, 0] + 70, B[:, 1], s=1.2, c=[CBh[k] for k in lb], lw=0)
a.text(bc[0] - 150, 105, "A: labels 1, 2, 3", ha="center"); a.text(70, 105, "B: labels a, b, c", ha="center")
a.set_title("(a) two sections, own labels", loc="left", fontsize=7.5)
a.set_xlim(-230, 190); a.set_ylim(-100, 120); a.set_aspect("equal"); a.axis("off")
# (b)
b = ax[0, 1]
b.imshow(rgba([0, 1, 2], 0.9), origin="lower", extent=ext)
b.set_xlim(-125, 125); b.set_ylim(-80, 80); b.axis("off")
b.set_title("(b) B as smooth label maps $G_l$", loc="left", fontsize=7.5)
# (c)
c = ax[1, 0]
R = T["right"]; P = np.asarray(R["P"])
c.imshow(rgba([0, 1, 2], 0.5), origin="lower", extent=ext)
c.scatter(P[:, 0], P[:, 1], s=2.2, c=[CA[k] for k in la], lw=0.2, edgecolors=SURF)
c.set_xlim(-125, 125); c.set_ylim(-80, 80); c.axis("off")
C = np.asarray(R["C"]); pr = R["pair"]
ins = c.inset_axes([0.73, 0.02, 0.27, 0.42])
ins.imshow(C, cmap="Greys", vmin=0, vmax=0.5)
for k in range(3):
    ins.add_patch(Rectangle((pr[k] - 0.5, k - 0.5), 1, 1, fill=False, ec="#2a78d6", lw=1.2))
ins.set_xticks(range(3), ["a", "b", "c"], fontsize=5.5); ins.set_yticks(range(3), ["1", "2", "3"], fontsize=5.5)
ins.tick_params(length=0, pad=1)
c.set_title(f"(c) score = best pairing: {R['score']:.2f}", loc="left", fontsize=7.5)
# (d)
d = ax[1, 1]
prof = T["profile"]; best = T["best"]["score"]; bdeg = T["best"]["deg"]
W = 10
for row, fl in enumerate((0, 1)):
    y0 = (1 - row) * 1.0
    for a0 in range(0, 360, W):
        v = max(s for f, dd, s in prof if f == fl and a0 <= dd < a0 + W) + 0.01
        near = fl == T["best"]["flip"] and abs(((a0 + W / 2) - bdeg + 180) % 360 - 180) <= 20
        h = (v - 0.3) / 0.6 * 0.8
        d.add_patch(Rectangle((a0 + 0.8, y0), W - 1.6, h, color="#2a78d6" if near else "#b9bcc2", lw=0))
    d.plot([0, 360], [y0 + (best - 0.3) / 0.6 * 0.8] * 2, color=INK, lw=0.8, ls=(0, (3, 2)))
    d.text(-6, y0 + 0.3, "no flip" if fl == 0 else "flip", ha="right", va="center", fontsize=6.5)
d.set_xlim(0, 360); d.set_ylim(0, 1.85); d.set_yticks([])
d.set_xticks([0, 90, 180, 270, 360], ["0°", "90°", "180°", "270°", "360°"], fontsize=6)
for s_ in ("left", "right", "top"):
    d.spines[s_].set_visible(False)
d.set_title("(d) ceilings per rotation range", loc="left", fontsize=7.5)
fig.savefig("figures/overview.pdf", bbox_inches="tight"); fig.savefig("figures/overview.png", dpi=220, bbox_inches="tight")
print("ok")
