"""Method figure: the two steps of the paper, on real mouse brain data.
Row 1, placement (FLiPAlign): (a) sections A and B with their own labels; (b) label maps of B (local share of three
labels, Gaussian sigma = 20 units); (c) best score per rotation from branch and bound, with upper-bound boxes, the
excluded zone around the answer and the margin eps; (d) the answer, A placed on B.
Row 2, labelling a stack: (e) three serial sections with own labels and overlap tables of neighbours; (f) one-to-one
matching against a split (cap 2); (g) one set of labels for the stack and its proven share V/U.

Data: ../server_results/stack/fig_data.npz (made by release/FLiPAlign/fig_data.py). Panel c is computed here with
release/FLiPAlign/smoothalign.py on a 4000-cell subsample of A (cached in figures/method_bb.npz; delete it to recompute,
takes several minutes). V/U = 0.90 is stack 1, clean, d = 2, cap 2 (server_results/stack/stackbound2_1.json).
Run from paper/ with a numpy 2 python: python make_fig_method.py -> figures/method.pdf, figures/method.png"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
FD = os.path.join(HERE, "../server_results/stack/fig_data.npz")
BB = os.environ.get("METHOD_BB", os.path.join(HERE, "figures/method_bb.npz"))
OUT = os.environ.get("METHOD_OUT", os.path.join(HERE, "figures/method"))
z = np.load(FD, allow_pickle=True)
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GREY, LGREY, INK = "#bdbdb8", "#e4e4e0", "#222222"
FS = 9                                    # every text >= 9 pt
plt.rcParams.update({"font.size": FS, "pdf.fonttype": 42, "font.family": "serif", "font.serif": ["cmr10"],
                     "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True, "axes.unicode_minus": False,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "xtick.major.size": 2, "ytick.major.size": 2, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5})


# ------------------------------------------------------------------ panel c data: real branch and bound
def run_bb():
    """best placement near the reported answer (172.3 deg, flip), then the far test of smoothalign with every
    rotation range it bounds logged; plus a plain curve of the best score every 3 degrees for both flips."""
    sys.path.insert(0, os.path.join(HERE, "../release/FLiPAlign"))
    from smoothalign import SmoothAlign, far_upper_bound
    rng = np.random.default_rng(0)
    ia = rng.choice(len(z["p_A"]), 4000, replace=False)
    al = SmoothAlign(z["p_A"][ia], z["p_la"][ia], z["p_B"], z["p_lb"], h=2.0, sigma=20.0, tpad=296.0, contrast=True)
    fine = np.radians(np.arange(168, 177, 0.1))
    sc = [al.rotation_eval(t, 1, 1e-9)[0] for t in fine]
    i = int(np.argmax(sc)); best, th = sc[i], fine[i]
    log, orig = [], al.rotation_eval
    def wrap(t, flip, delta, **kw):
        r = orig(t, flip, delta, **kw); log.append((t % (2 * np.pi), flip, delta, r[0], r[2])); return r
    al.rotation_eval = wrap
    u = far_upper_bound(al, best, th, 1, exclude_deg=20.0, n0=36, time_limit=900)
    grid = np.radians(np.arange(1.5, 360, 3.0))
    curve = np.array([[orig(t, f, 1e-9)[0] for t in grid] for f in (0, 1)])
    np.savez(BB, log=np.array(log), n1=0, best=best, th=th, fl=1, eps=u["eps"], far_ub=u["far_ub"],
             complete=u["complete"], grid=grid, curve=curve)


if not os.path.exists(BB):
    run_bb()
bb = np.load(BB)


# ------------------------------------------------------------------ helpers
fig = plt.figure(figsize=(4.8, 2.6))
FW, FH = 4.8, 2.6
def axin(x, y, w, h):                     # axes from inches
    return fig.add_axes([x / FW, y / FH, w / FW, h / FH])
def txt(x, y, s, **kw):
    kw.setdefault("fontsize", FS); return fig.text(x / FW, y / FH, s, **kw)
def arrow(x0, x1, y, **kw):
    fig.add_artist(FancyArrowPatch((x0 / FW, y / FH), (x1 / FW, y / FH), transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=7, lw=0.9, color=INK, **kw))
def dots(ax, xy, col, s=0.15, alpha=1.0, z_=1):
    ax.scatter(xy[:, 0], xy[:, 1], s=s, c=col, lw=0, alpha=alpha, rasterized=True, zorder=z_)
def fit(ax, xy, pad=0.02):
    lo, hi = xy.min(0), xy.max(0); d = (hi - lo) * pad
    ax.set_xlim(lo[0] - d[0], hi[0] + d[0]); ax.set_ylim(lo[1] - d[1], hi[1] + d[1])
    ax.set_aspect("equal", adjustable="box", anchor="C"); ax.axis("off")
def title(x, y, letter, s):
    t = txt(x, y, letter, fontsize=FS + 1, family="cmb10", ha="left", va="baseline")
    txt(x + 0.13, y, s, ha="left", va="baseline")
def band(y, text):
    fig.add_artist(FancyBboxPatch((0.0, y / FH), 1.0, 0.17 / FH, boxstyle="round,pad=0,rounding_size=0.01",
                                  transform=fig.transFigure, fc="#eef2f7", ec="none"))
    txt(0.06, y + 0.085, text, family="cmb10", ha="left", va="center", fontsize=FS + 0.5)
cols = np.array(PAL)


# ================================================================== row 1: placement
band(2.43, "1   Placement (FLiPAlign)")
TY1 = 2.27                     # title baseline
P1, P1H = 1.30, 0.90           # plot bottom and height

# (a) sections A and B, own labels (colours of A and B do not correspond)
permA = np.array([0, 1, 2, 3, 4, 5]); permB = np.array([3, 6, 7, 2, 0, 4])
xA, xB = z["p_A"], z["p_B"]
aA = axin(0.0, P1 + 0.2, 0.52, P1H - 0.22); dots(aA, xA, cols[permA[z["p_la"]]], s=0.12); fit(aA, xA)
aB = axin(0.56, P1 + 0.2, 0.52, P1H - 0.22); dots(aB, xB, cols[permB[z["p_lb"]]], s=0.09); fit(aB, xB)
txt(0.26, P1 + 0.06, "$A$", ha="center", va="center"); txt(0.82, P1 + 0.06, "$B$", ha="center", va="center")
title(0.0, TY1, "a", "Two sections")

# (b) label maps G_l of B: local share of label l, Gaussian sigma = 20 units, white-to-label-colour scale
h_, sig = 4.0, 20.0
lo = xB.min(0) - 2 * sig; n_ = np.ceil((xB.max(0) + 2 * sig - lo) / h_).astype(int) + 1
ij = np.floor((xB - lo) / h_ + 0.5).astype(int)
S = np.zeros((int(z["p_lb"].max()) + 1,) + tuple(n_))
for l in range(len(S)):
    m = z["p_lb"] == l; np.add.at(S[l], (ij[m, 0], ij[m, 1]), 1.0)
S = gaussian_filter(S, (0, sig / h_, sig / h_)); tot = S.sum(0)
cfl = 0.05 * np.median(tot[tot > 0.2 * tot.max()]); Gm = S / (tot + cfl)
show = [1, 5, 2]                                           # the three largest labels of B
bx0, bw, bh, off = 1.27, 0.44, 0.37, 0.165
for k in range(len(show))[::-1]:
    l = show[k]
    a = axin(bx0 + k * off, P1 + 0.22 + (len(show) - 1 - k) * off * 0.75, bw, bh)
    cm = LinearSegmentedColormap.from_list("m", ["#ffffff", PAL[permB[l]]])
    a.imshow(Gm[l].T, origin="lower", cmap=cm, vmin=0, vmax=0.7, interpolation="bilinear", aspect="auto")
    a.set_xticks([]); a.set_yticks([])
    for s_ in a.spines.values(): s_.set_color("#777777"); s_.set_linewidth(0.5)
txt(bx0 + 0.37, P1 + 0.06, "one per label", ha="center", va="center")
title(1.26, TY1, "b", "Label maps")

# (c) best score per rotation for both flips, bound boxes of the far test, excluded zone, margin eps
log = bb["log"]
best, thb, flb, eps, far_ub = float(bb["best"]), float(bb["th"]), int(bb["fl"]), float(bb["eps"]), float(bb["far_ub"])
def leaves(L):                       # rotation ranges the far test left unsplit: its final bound boxes
    key = lambda t, f, d: (int(round(np.degrees(t % (2 * np.pi)) * 1e4)), int(f), int(round(np.degrees(d) * 1e4)))
    keys = {key(t, f, d) for t, f, d, _, _ in L}
    return [r for r in L if key(r[0] - r[2] / 2, r[1], r[2] / 2) not in keys and key(r[0] + r[2] / 2, r[1], r[2] / 2) not in keys]
deg = np.degrees
ylo, yhi = 0.18, best + 0.02
ca = axin(2.55, P1 + 0.32, 1.02, P1H - 0.36)
zc = deg(thb) + 360.0
ca.axvspan(zc - 20, zc + 20, color="#d4d4cf", zorder=0, lw=0)
for f in (0, 1):                     # upper bound over each final rotation range, drawn as one step line per run
    L = sorted([(deg(t - d), deg(t + d), u) for t, f_, d, s_, u in leaves(log) if int(f_) == f])
    o_ = 360.0 if f == flb else 0.0
    runs, cur = [], []
    for x0, x1, u in L:
        if cur and x0 - cur[-1][1] > 1e-6: runs.append(cur); cur = []
        cur.append((x0, x1, u))
    runs.append(cur)
    for r_ in runs:
        if not r_: continue
        xs = [x0 + o_ for x0, _, _ in r_] + [r_[-1][1] + o_]; us = [u for _, _, u in r_] + [r_[-1][2]]
        ca.fill_between(xs, ylo, us, step="post", color="#f6dfa6", lw=0, zorder=1)
        ca.step(xs, us, where="post", color="#c98a00", lw=0.4, zorder=2)
for f, o_ in ((1 - flb, 0.0), (flb, 360.0)):              # x: as is 0..360, mirrored 360..720
    ca.plot(deg(bb["grid"]) + o_, bb["curve"][f], color=PAL[0], lw=0.8, zorder=3)
ca.plot([zc], [best], marker="*", ms=6.5, color=PAL[7], zorder=5, mec="none", clip_on=False)
ca.plot([zc, 735], [best, best], color=PAL[7], lw=0.5, ls=(0, (2, 1.5)), zorder=4, clip_on=False)
ca.plot([0, 735], [far_ub, far_ub], color="#a87300", lw=0.5, ls=(0, (2, 1.5)), zorder=4, clip_on=False)
xe = 742
ca.plot([xe, xe], [far_ub, best], color=INK, lw=0.8, clip_on=False, zorder=6)   # bracket: the margin eps
for y_ in (far_ub, best): ca.plot([xe - 7, xe + 7], [y_, y_], color=INK, lw=0.8, clip_on=False, zorder=6)
ca.text(xe + 20, (best + far_ub) / 2, r"$\varepsilon$", ha="left", va="center", fontsize=FS, clip_on=False)
ca.axvline(360, color=INK, lw=0.5, zorder=2)
ca.set_xlim(0, 720); ca.set_ylim(ylo, yhi)
ca.set_xticks([180, 540]); ca.set_xticklabels(["as is", "mirrored"]); ca.tick_params(axis="x", length=0)
ca.set_xticks([0, 360, 720], minor=True); ca.tick_params(axis="x", which="minor", length=2)
ca.set_yticks([0.2, 0.3]); ca.tick_params(labelsize=FS)
for s_ in ("top", "right"): ca.spines[s_].set_visible(False)
ca.set_xlabel("rotation", labelpad=1); ca.set_ylabel("score", labelpad=1)
title(2.22, TY1, "c", "Search with bounds")

# (d) the answer: A placed on B, certified or refused
da = axin(3.93, P1 + 0.34, 0.87, P1H - 0.36)
dots(da, xB, "#d0d0cb", s=0.09); dots(da, z["p_Aplaced"], cols[permA[z["p_la"]]], s=0.08, alpha=0.6, z_=2)
fit(da, np.vstack([xB, z["p_Aplaced"]]))
txt(4.365, P1 + 0.21, r"certified if $\varepsilon>0$", ha="center", va="center", color=PAL[5])
txt(4.365, P1 + 0.05, "else refused", ha="center", va="center", color="#666666")
title(3.93, TY1, "d", "Answer")

for x0, x1 in ((1.08, 1.24), (2.06, 2.20), (3.73, 3.89)):
    arrow(x0, x1, P1 + 0.55)


# ================================================================== row 2: labelling a stack
band(1.07, "2   Labelling a stack")
TY2 = 0.91
P2, P2H = 0.0, 0.84

def table(s, u, step=25.0, sg=50.0):        # C_su(k,l): share of label l of u around cells of s with label k
    xs, ls = z[f"s{s}_xy"], z[f"s{s}_own"]; xu, lu = z[f"s{u}_xy"], z[f"s{u}_own"]
    lo = np.minimum(xs.min(0), xu.min(0)) - 200; hi = np.maximum(xs.max(0), xu.max(0)) + 200
    sh = np.ceil((hi - lo) / step).astype(int) + 1
    S = np.zeros((8,) + tuple(sh)); ij = np.floor((xu - lo) / step + .5).astype(int)
    for l in range(8): np.add.at(S[l], (ij[lu == l, 0], ij[lu == l, 1]), 1)
    S = gaussian_filter(S, (0, sg / step, sg / step)); tot = S.sum(0); c = 0.05 * np.median(tot[tot > 0.2 * tot.max()])
    G = S / (tot + c); q = (xs - lo) / step
    return np.array([[map_coordinates(G[l], q.T, order=1)[ls == k].sum() for l in range(8)] for k in range(8)]) / len(xs)

ownperm = [np.random.default_rng(10 + j).permutation(8) for j in range(3)]   # own colours, unrelated across sections
sec_w, sec_h, gap = 0.50, 0.32, 0.06
ys = P2 + P2H - sec_h
for j in range(3):
    a = axin(j * (sec_w + gap), ys, sec_w, sec_h)
    dots(a, z[f"s{j}_xy"], cols[ownperm[j][z[f"s{j}_own"]]], s=0.05); fit(a, z[f"s{j}_xy"])
    txt(j * (sec_w + gap) + sec_w / 2, ys - 0.07, f"{j + 1}", ha="center", va="center")
tw = 0.30
for j in range(2):
    C = table(j, j + 1); r, c = linear_sum_assignment(-C); C = C[:, c]        # best one-to-one on the diagonal
    x0 = (j + 1) * (sec_w + gap) - gap / 2 - tw / 2
    a = axin(x0, P2 + 0.03, tw, tw)
    a.imshow(C, cmap="Blues", vmin=0, vmax=0.12, interpolation="nearest")
    a.set_xticks([]); a.set_yticks([])
    for s_ in a.spines.values(): s_.set_linewidth(0.5)
    txt(x0 - 0.03, P2 + 0.03 + tw / 2, f"{j + 1}-{j + 2}", ha="right", va="center")
title(0.0, TY2, "e", "Overlap tables")

# (f) one-to-one against split: schematic, three labels per section
ov = fig.add_axes([0, 0, 1, 1], facecolor="none"); ov.set_xlim(0, FW); ov.set_ylim(0, FH); ov.axis("off")
def bip(x0, split):
    yt, yb = P2 + 0.62, P2 + 0.30
    top = [(x0 + i * 0.16, yt) for i in range(3)]; bot = [(x0 + i * 0.16, yb) for i in range(3)]
    for i, k in [(0, 0), (1, 1)] + ([(1, 2)] if split else []):
        ov.plot([top[i][0], bot[k][0]], [yt, yb], color=INK, lw=1.0, zorder=1)
    if not split:
        ov.plot([top[2][0], bot[2][0]], [yt, yb], color=PAL[7], lw=1.0, ls=(0, (2, 1.2)), zorder=1)
    for i, p_ in enumerate(top): ov.plot(*p_, "o", ms=6, color=PAL[[0, 1, 3][i]], mec="white", mew=0.5, zorder=2)
    for i, p_ in enumerate(bot): ov.plot(*p_, "o", ms=6, color=PAL[[6, 2, 4][i]], mec="white", mew=0.5, zorder=2)
fx = 1.98
bip(fx, False); bip(fx + 0.62, True)
txt(fx + 0.16, P2 + 0.12, "one-to-one", ha="center", va="center")
txt(fx + 0.78, P2 + 0.12, "split", ha="center", va="center")
txt(fx - 0.12, P2 + 0.62, "1", ha="center", va="center"); txt(fx - 0.12, P2 + 0.30, "2", ha="center", va="center")
title(1.82, TY2, "f", "Split labels")

# (g) one set of labels for the stack (split classes, cap 2) and the proven share V/U
vu = json.load(open(os.path.join(HERE, "../server_results/stack/stackbound2_1.json")))["runs"]["clean"]["d2"]["ratio_to_pair_bound"]
gx, gw, gh = 3.33, 0.47, 0.30
for j in range(3):
    a = axin(gx + j * (gw + 0.03), ys + 0.02, gw, gh)
    dots(a, z[f"s{j}_xy"], cols[z[f"s{j}_split"] % 8], s=0.05); fit(a, z[f"s{j}_xy"])
txt(gx + 0.735, P2 + 0.32, "proven share", ha="center", va="center")
txt(gx + 0.735, P2 + 0.13, r"$V/U=%.2f$" % vu, ha="center", va="center")
title(gx, TY2, "g", "One set of labels")

arrow(1.68, 1.84, P2 + 0.46); arrow(3.14, 3.30, P2 + 0.46)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT + ".pdf", dpi=600)
fig.savefig(OUT + ".png", dpi=300)
print("eps (subsample, zone 20 deg)", round(eps, 4), "best", round(best, 4), "far_ub", round(far_ub, 4), "V/U", vu)
