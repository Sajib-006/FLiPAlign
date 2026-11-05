"""Fig. 4 of the paper: one set of labels for a stack of sections.

(a) three consecutive sections of stack 1 (Allen MERFISH, clean): each section's own labels, one-to-one labelling
    (forest), split labelling (ours) and the atlas divisions. One-to-one and split classes are coloured by a one-to-one
    (Hungarian) match to atlas divisions over the three sections, so the same colour means the same region.
(b) NMI with atlas divisions, best one-to-one labelling against split labelling, every run of both brains.
(c) cap sweep: NMI and ARI against the cap (class bound 8), mean +- SE over 30 runs.
(d) agreement of matched cells of neighbouring sections without the atlas (Cohen's kappa), mean +- SE.
(e) the two steps together (flip known): largest drift per stack when every link is chained and when the stack is
    cut at refused links.
Reads results/stack/{fig_data.npz, stack_*, stackcap_*, stackb2_*, stackcons_*, stacksweep_*, stackplacefk_*}.
Usage: python3 make_fig_stacks.py -> OUT/figures/fig_stacks.pdf, .png"""
import glob, json, os
import numpy as np
from scipy.optimize import linear_sum_assignment
from figstyle import *
from _paths import STACK, FIGURES

R = STACK
VV = ["clean", "S1_recluster", "S2_crop50", "S3_noise30", "S4_two_noisy"]
load = lambda pat: [json.load(open(f)) for f in sorted(glob.glob(f"{R}/{pat}_[0-9].json"))]
J, C, B2, CO, SW = load("stack"), load("stackcap"), load("stackb2"), load("stackcons"), load("stacksweep")
z = np.load(os.path.join(R, "fig_data.npz"), allow_pickle=True)


def best11(runs, v, key="nmi"):
    return max((runs[v][q] for q in ("chain", "forest_d2", "star")), key=lambda r: r[key])


# ---------------------------------------------------------------- numbers
b1_best = np.array([best11(j["runs"], v)["nmi"] for j in J for v in VV])
b1_split = np.array([c["runs"][v]["cap2_d2"]["nmi"] for c in C for v in VV])
b2_best = np.array([best11(d["runs"], v)["nmi"] for d in B2 for v in VV])
b2_split = np.array([d["runs"][v]["cap2_fromC"]["nmi"] for d in B2 for v in VV])
w1, w2 = int((b1_split > b1_best).sum()), int((b2_split > b2_best).sum())
kap = {b: {m: np.array([(best11(d["runs"], v) if m == "best" else d["runs"][v]["cap2_fromC"])["kappa"] for d in D for v in VV])
           for m in ("best", "split")} for b, D in (("Brain 1", CO), ("Brain 2", B2))}
caps = [1, 2, 3, 4]
sw = {k: np.array([[s["runs"][v][f"cap{c}_G8"][k] for s in SW for v in VV] for c in caps]) for k in ("nmi", "ari")}
sw0 = {k: np.array([s["runs"][v]["start"][k] for s in SW for v in VV]) for k in ("nmi", "ari")}
A = json.load(open(f"{R}/stackplacefk_summary.json")); Cu = json.load(open(f"{R}/stackplacefk_refuse.json"))
drift_all = np.array([max(a["drift_um"]) for a in A]); drift_cut = np.array([max(c["drift_um"]) for c in Cu])
ncert = [sum(c["certified"]) for c in Cu]
print(f"brain1 split>best {w1}/30, brain2 {w2}/30; kappa", {b: {m: round(v.mean(), 2) for m, v in d.items()} for b, d in kap.items()},
      "drift all", drift_all.round(0), "cut", drift_cut.round(0), "certified", ncert)

# ---------------------------------------------------------------- colours for panel a
xy = [z[f"s{i}_xy"] for i in range(3)]
lab = {k: [np.asarray(z[f"s{i}_{k}"]) for i in range(3)] for k in ("own", "one", "split", "div")}
divs, cnt = np.unique(np.concatenate(lab["div"]), return_counts=True)
top = divs[np.argsort(-cnt)][:len(REGION)]                    # largest divisions get the palette, the rest grey
dcol = {d: REGION[i] for i, d in enumerate(top)}


def match_colors(key):
    """colour each class of a shared labelling by a one-to-one match to atlas divisions (all three sections)"""
    a = np.concatenate(lab[key]); d = np.concatenate(lab["div"])
    cl = np.unique(a)
    M = np.array([[np.sum((a == c) & (d == t)) for t in top] for c in cl])
    r, c = linear_sum_assignment(-M)
    col = {k: REFUSED for k in cl}
    spare = [x for x in REGION if x not in [dcol[top[j]] for j in c]]
    for i, j in zip(r, c):
        col[cl[i]] = dcol[top[j]] if M[i, j] > 0 else REFUSED
    for k in cl:
        if col[k] == REFUSED and spare:
            col[k] = spare.pop()
    return col


colmap = {"one": match_colors("one"), "split": match_colors("split")}

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(TEXTW, 3.3))
cols = [("own", "Own labels"), ("one", "One-to-one"), ("split", "Split (ours)"), ("div", "Atlas divisions")]
x0, w, h, gapx = 0.075, 0.215, 0.158, 0.017
ytop = 0.95
axes_a = {}
for j, (k, title) in enumerate(cols):
    for i in range(3):
        ax = fig.add_axes([x0 + j * (w + gapx), ytop - (i + 1) * h - i * 0.006, w, h])
        L = lab[k][i]
        if k == "own":
            shift = [0, 5, 9][i]                               # own labels: different colours per section
            c = np.array(REGION)[(L + shift) % len(REGION)]
        elif k == "div":
            c = np.array([dcol.get(t, LIGHT) for t in L])
        else:
            c = np.array([colmap[k][t] for t in L])
        tissue(ax, xy[i], None, colors=c, s=0.35)
        axes_a[(i, j)] = ax
        if i == 0:
            ax.set_title(title, fontsize=FS, pad=2, color=OURS if k == "split" else INK,
                         fontweight="bold" if k == "split" else "normal")
        if j == 0:
            bb = ax.get_position()
            fig.text(bb.x0 - 0.008, (bb.y0 + bb.y1) / 2, f"Section {i + 1}", rotation=90, ha="right", va="center", fontsize=FS_SMALL, color=MUTED)
letter(fig, axes_a[(0, 0)], "a", dx=-0.035, dy=0.012)

# (b) split vs best one-to-one, every run
yb, hb = 0.08, 0.28
axb = fig.add_axes([0.085, yb, 0.165, hb])
lo, hi = 0.11, 0.36
axb.plot([lo, hi], [lo, hi], color=MUTED, lw=0.6, ls="--", zorder=0)
axb.scatter(b1_best, b1_split, s=9, marker="o", facecolor=OURS, edgecolor="white", lw=0.3, label="Brain 1", zorder=3)
axb.scatter(b2_best, b2_split, s=10, marker="^", facecolor="white", edgecolor=OURS, lw=0.7, label="Brain 2", zorder=3)
axb.set_xlim(lo, hi); axb.set_ylim(lo, hi)
axb.set_xticks([0.15, 0.25, 0.35]); axb.set_yticks([0.15, 0.25, 0.35])
axb.set_xlabel("Best one-to-one NMI"); axb.set_ylabel("Split NMI")
axb.set_title("Split vs one-to-one", pad=3)
axb.legend(loc="lower right", bbox_to_anchor=(1.03, -0.01), fontsize=7.2, handletextpad=0.0, borderaxespad=0.0, labelspacing=0.3)
letter(fig, axb, "b", dx=-0.055, dy=0.02)

# (c) cap sweep
axc = fig.add_axes([0.345, yb, 0.155, hb])
for k, nm, mk, ls in (("nmi", "NMI", "o", "-"), ("ari", "ARI", "s", "--")):
    m = np.r_[sw[k].mean(1)]; se = sw[k].std(1, ddof=1) / np.sqrt(sw[k].shape[1])
    axc.errorbar(caps, m, yerr=se, color=OURS, marker=mk, ms=3.2, lw=1.1, ls=ls, capsize=1.5, elinewidth=0.6)
    axc.text(4.35, m[-1], nm, va="center", ha="left", fontsize=FS_SMALL, color=OURS)
axc.set_xticks(caps); axc.set_xlim(0.6, 5.3); axc.set_ylim(0.17, 0.30); axc.set_yticks([0.18, 0.22, 0.26, 0.30])
axc.set_xlabel("Cap"); axc.set_ylabel("Score")
axc.set_title("Cap sweep", pad=3)
axc.axvspan(0.6, 1.5, color=LIGHT, lw=0, zorder=0)
letter(fig, axc, "c", dx=-0.05, dy=0.02)

# (d) agreement without the atlas
axd = fig.add_axes([0.605, yb, 0.125, hb])
for i, (b, d) in enumerate(kap.items()):
    m = [d["best"].mean(), d["split"].mean()]; se = [d[q].std(ddof=1) / np.sqrt(len(d[q])) for q in ("best", "split")]
    xs = [i - 0.16, i + 0.16]
    axd.plot(xs, m, color=MUTED, lw=0.8, zorder=1)
    axd.errorbar(xs[0], m[0], yerr=se[0], fmt="o", color=BASE, ms=4, capsize=1.5, elinewidth=0.6, zorder=2)
    axd.errorbar(xs[1], m[1], yerr=se[1], fmt="o", color=OURS, ms=4, capsize=1.5, elinewidth=0.6, zorder=2)
axd.set_xticks([0, 1]); axd.set_xticklabels(["1", "2"]); axd.set_xlabel("Brain"); axd.set_xlim(-0.5, 1.5)
axd.set_ylim(0.25, 0.6); axd.set_yticks([0.3, 0.4, 0.5]); axd.set_ylabel("Cohen's kappa")
axd.set_title("Without atlas", pad=3)
axd.plot([], [], "o", color=BASE, ms=4, label="One-to-one"); axd.plot([], [], "o", color=OURS, ms=4, label="Split")
axd.legend(loc="upper left", bbox_to_anchor=(-0.04, 1.02), fontsize=7.2, handletextpad=0.1, borderaxespad=0.0, labelspacing=0.25)
letter(fig, axd, "d", dx=-0.05, dy=0.02)

# (e) drift, every link chained vs cut at refused links
axe = fig.add_axes([0.862, yb, 0.135, hb])
st = np.arange(1, len(A) + 1)
bw = 0.38
axe.bar(st - bw / 2, drift_all, width=bw, bottom=0, color=REFUSED, lw=0, label="All links chained", zorder=2)
axe.bar(st + bw / 2, drift_cut, width=bw, bottom=0, color=OURS, lw=0, label="Cut at refusals", zorder=3)
axe.set_yscale("log"); axe.set_ylim(80, 1.2e5); axe.set_yticks([1e2, 1e3, 1e4]); axe.set_yticklabels(["100", "1,000", "10,000"]); axe.minorticks_off(); axe.set_xticks(st); axe.set_xlim(0.4, len(A) + 0.6)
axe.set_xlabel("Stack"); axe.set_ylabel("Largest drift (µm)", labelpad=1)
axe.set_title("Both steps", pad=3)
axe.legend(loc="upper center", fontsize=7.2, handlelength=0.9, handletextpad=0.3, borderaxespad=0.0, ncol=1)
letter(fig, axe, "e", dx=-0.05, dy=0.02)

save(fig, os.path.join(FIGURES, "fig_stacks"))
print("wrote", os.path.join(FIGURES, "fig_stacks.pdf"))
