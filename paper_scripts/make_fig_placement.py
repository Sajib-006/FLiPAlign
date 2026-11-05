"""Placement figure (main text): FLiPAlign places held-out sections right more often than seven published tools,
and its certified answers can be trusted.

(a) one held-out case (second held-out set, case 6, human tonsil, Xenium cells): the two sections with their own
    labels; (b) section A placed by the truth, by FLiPAlign (flip searched) and by the two strongest tools (true flip
    and a rough angle handed over), on top of section B; (c) right answers per tool on both held-out sets, from four
    starts (same numbers as the tools table); (d) error of every FLiPAlign answer on both held-out sets, certified
    (after the mirror self-check) or refused.
Reads results/placement (fig_case.npz from runs_placement/fig_case_data.py, and the placement results).
Run from paper_scripts/: python3 make_fig_placement.py -> out/figures/fig_placement.pdf, .png"""
import os
import numpy as np
from _paths import PLACE, FIGURES
from figstyle import *
import figstyle as S
from make_results import load, manifest, RIGHT, TOOLS
from make_results2 import selfcheck
from make_results3 import tool_best, GOOD

# ------------------------------------------------------------------ data
SETS = [("gt3", "Held-out 1", ["gt3base", "gt3base3", "gt3slat", "gt3spateo", "gt3cast", "gt3stalign", "rc_gt3stalign"]),
        ("gt4", "Held-out 2", ["gt4base", "gt4spateo", "gt4cast", "gt4stalign", "rc_gt4stalign"])]
START = ["As given", "True flip", "+ rough angle", "+ rough position"]
counts = {}                                    # set -> {tool: [G, F, A, P]}, plus FLiPAlign (searched, known), n
for sn, lab, base in SETS:                     # same calls as make_results3.tools_main (Table 2)
    man = manifest(sn); cs = sorted(man)
    arms = [tool_best(base, cs, GOOD), tool_best(base, cs, GOOD, [f"{sn}fo_base", f"{sn}fo_spateo", f"{sn}fo_cast"]),
            tool_best([f"rs_{sn}base", f"rs_{sn}spateo", f"rs_{sn}cast", f"rs_{sn}stalign", f"rc_rs_{sn}cpd", f"rc_rs_{sn}stalign"], cs, GOOD),
            tool_best([f"rp_{sn}base", f"rp_{sn}spateo", f"rp_{sn}cast", f"rc_rp_{sn}cpd", f"rc_rp_{sn}stalign"], cs, GOOD)]
    O, K = load(f"{sn}h2"), load(f"fk_{sn}")
    counts[sn] = {t: [a[t][0] if a.get(t) else np.nan for a in arms] for t, _ in TOOLS}
    counts[sn]["_ours"] = (sum(GOOD(c, O[c]["tre_um"]) for c in cs if c in O), sum(GOOD(c, K[c]["tre_um"]) for c in cs if c in K))
    counts[sn]["_n"] = len(cs)

err_c, err_r = [], []                          # FLiPAlign, flip searched, certified after the self-check vs refused
for sn, _, _ in SETS:
    O, keep = load(f"{sn}h2"), selfcheck(f"fc_{sn}")
    for c in sorted(manifest(sn)):
        if c in O:
            (err_c if O[c]["certified"] and keep(c) else err_r).append(O[c]["tre_um"])
err_c, err_r = np.array(err_c), np.array(err_r)
z = np.load(os.path.join(PLACE, "fig_case.npz"), allow_pickle=True)

# ------------------------------------------------------------------ figure
fig = plt.figure(figsize=(TEXTW, 3.42))
W, H = fig.get_size_inches()
ax_in = lambda x, y, w, h: fig.add_axes([x / W, y / H, w / W, h / H])

# (a) inputs, (b) placements
allxy = np.vstack([z["B"]] + list(z["placed"]))
lo, hi = allxy.min(0), allxy.max(0); pad = 0.04 * (hi - lo).max()
def frame(ax, xy=None):
    if xy is None:
        ax.set_xlim(lo[0] - pad, hi[0] + pad); ax.set_ylim(lo[1] - pad, hi[1] + pad)
    else:
        l, h = xy.min(0), xy.max(0); c = (l + h) / 2; r = 0.5 * (h - l).max() + pad
        ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r)
    ax.invert_yaxis()

top, mh = 2.24, 0.92
axA = ax_in(0.05, top, 0.72, mh); axB = ax_in(0.82, top, 0.72, mh)
tissue(axA, z["A_obs"], z["la"], s=0.25)
tissue(axB, z["B"], np.asarray(z["lb"]) + 6, s=0.25)             # other colours: label names are not shared
frame(axA, z["A_obs"]); frame(axB)
for a_ in (axA, axB):
    a_.set_aspect("equal", adjustable="datalim")
axA.set_title("Section A", pad=2); axB.set_title("Section B", pad=2)
fig.text((0.05 + 1.54) / 2 / W, (top + 0.13) / H, "own clusters, no shared names", ha="center", va="top", fontsize=FS_SMALL, color=MUTED)

names, errs = list(z["names"]), list(z["err_um"])
from scipy.ndimage import gaussian_filter
gx = np.linspace(lo[0] - pad, hi[0] + pad, 160); gy = np.linspace(lo[1] - pad, hi[1] + pad, 160)
occ, _, _ = np.histogram2d(z["placed"][0][:, 0], z["placed"][0][:, 1], bins=[np.r_[gx, gx[-1] + gx[1] - gx[0]] - (gx[1] - gx[0]) / 2,
                           np.r_[gy, gy[-1] + gy[1] - gy[0]] - (gy[1] - gy[0]) / 2])
occ = gaussian_filter((occ > 0).astype(float), 1.5)
col = {"Truth": ACCENT, "FLiPAlign": OURS}
x0, w_, gap = 1.80, 0.70, 0.04
for i, (nm, P, e) in enumerate(zip(names, z["placed"], errs)):
    ax = ax_in(x0 + i * (w_ + gap), top, w_, mh)
    ax.scatter(z["B"][:, 0], z["B"][:, 1], c=LIGHT, s=0.25, lw=0, rasterized=True)
    ax.scatter(P[:, 0], P[:, 1], c=col.get(nm, BASE), s=0.25, lw=0, alpha=0.9, rasterized=True)
    if nm != "Truth":                                    # outline of the true position of A
        ax.contour(gx, gy, occ.T, levels=[0.5], colors=[ACCENT], linewidths=0.8, zorder=5)
    ax.set_aspect("equal", adjustable="datalim"); ax.axis("off"); frame(ax)
    ax.set_title(nm, pad=2, color=INK)
    if nm != "Truth":
        ok = e < RIGHT
        ax.text(0.5, 0.06, f"{e:,.0f} µm", transform=ax.transAxes, ha="center", va="top", fontsize=FS_SMALL,
                color=S.RIGHT if ok else S.WRONG, fontweight="bold")
    else:
        ax.text(0.5, 0.06, "A on B", transform=ax.transAxes, ha="center", va="top", fontsize=FS_SMALL, color=MUTED)
    if i == 0:
        bx = ax
letter(fig, axA, "a", dx=0.0, dy=0.02); letter(fig, bx, "b", dx=-0.005, dy=0.02)

# (c) right answers per tool, both held-out sets
order = sorted([t for t, _ in TOOLS], key=lambda t: -np.nanmax(counts["gt4"][t] + counts["gt3"][t]))
rowsn = ["FLiPAlign"] + [dict(TOOLS)[t] for t in order]
blues = ["#C6DBEF", "#6BAED6", "#2171B5", "#08306B"]
mk = ["o", "o", "o", "o"]
ys = np.arange(len(rowsn))[::-1]
cw, cy, ch = 1.02, 0.70, 1.12
axes_c = []
for j, (sn, lab, _) in enumerate(SETS):
    ax = ax_in(0.62 + j * (cw + 0.12), cy, cw, ch); axes_c.append(ax)
    n = counts[sn]["_n"]
    ax.axhspan(ys[0] - 0.5, ys[0] + 0.5, color="#EEF1F7", lw=0, zorder=0)
    for y, t in zip(ys[1:], order):
        v = np.array(counts[sn][t], float)
        ok = ~np.isnan(v)
        ax.plot([np.nanmin(v), np.nanmax(v)], [y, y], color="#BDBDBD", lw=0.8, zorder=1)
        for k in np.where(ok)[0]:
            ax.scatter(v[k], y, s=26, color=blues[k], edgecolor="white", lw=0.4, zorder=2 + k)
    s_, k_ = counts[sn]["_ours"]
    ax.plot([s_, k_], [ys[0], ys[0]], color="#9FB0D0", lw=0.8, zorder=1)
    ax.scatter(s_, ys[0], marker="s", s=28, color=OURS, edgecolor="white", lw=0.4, zorder=5)
    ax.scatter(k_, ys[0], marker="s", s=28, facecolor="white", edgecolor=OURS, lw=0.9, zorder=4)
    ax.set_xlim(-1.5, n); ax.set_ylim(-0.6, len(rowsn) - 0.4)
    ax.set_xticks([0, 12, 24, 36, 48]); ax.set_title(f"{lab} ({n} cases)", pad=3)
    ax.set_yticks(ys); ax.set_yticklabels(rowsn if j == 0 else [])
    if j == 0:
        ax.get_yticklabels()[0].set_color(OURS); ax.get_yticklabels()[0].set_fontweight("bold")
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#EEEEEE", lw=0.5, zorder=0); ax.set_axisbelow(True)
fig.text((0.62 + cw + 0.06) / W, (cy - 0.27) / H, "Right answers (error < 200 µm)", ha="center", va="top")
letter(fig, axes_c[0], "c", dx=(0.05 - 0.62) / W, dy=0.02)
# legend for (c)
from matplotlib.lines import Line2D
hs = [Line2D([], [], marker="s", ls="", color=OURS, ms=4.2, mec="white", mew=0.4, label="FLiPAlign, flip searched"),
      Line2D([], [], marker="s", ls="", mfc="white", mec=OURS, ms=4.0, mew=0.9, label="FLiPAlign, flip known")] + \
     [Line2D([], [], marker="o", ls="", color=blues[k], ms=4.2, mec="white", mew=0.4, label="Tools: " + START[k].lower() if k == 0 else START[k]) for k in range(4)]
leg = fig.legend(handles=hs, loc="lower left", bbox_to_anchor=(0.02, 0.0), ncol=3, fontsize=FS_SMALL, columnspacing=0.8,
                 handletextpad=0.2, borderaxespad=0.0)

# (d) certified vs refused, FLiPAlign flip searched, both held-out sets
ax = ax_in(3.62, cy, 1.10, ch)
rg = np.random.default_rng(1)
for x, e in ((0, err_c), (1, err_r)):
    ok = e < RIGHT
    jx = x + rg.uniform(-0.28, 0.28, len(e))
    ax.scatter(jx[ok], e[ok], s=7, color=S.RIGHT, lw=0, alpha=0.85, zorder=3)
    ax.scatter(jx[~ok], e[~ok], s=7, color=S.WRONG, lw=0, alpha=0.85, zorder=3)
    ax.text(x, 2.6e4, f"{ok.sum()} right", ha="center", va="bottom", fontsize=FS_SMALL, color=S.RIGHT, fontweight="bold")
    ax.text(x, 2.5e4, f"{(~ok).sum()} wrong", ha="center", va="top", fontsize=FS_SMALL, color=S.WRONG, fontweight="bold")
ax.axhline(RIGHT, color=MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)
ax.set_yscale("log"); ax.set_ylim(5, 9e4)
ax.set_yticks([10, 200, 1000, 10000]); ax.set_yticklabels(["10", "200", "1,000", "10,000"])
ax.minorticks_off(); ax.set_xlim(-0.6, 1.6)
ax.set_xticks([0, 1]); ax.set_xticklabels([f"Certified\n({len(err_c)})", f"Refused\n({len(err_r)})"], linespacing=1.05)
ax.tick_params(axis="x", length=0)
ax.set_ylabel("Error (µm)")
ax.set_title("FLiPAlign answers", pad=3)
letter(fig, ax, "d", dx=-0.40 / W, dy=0.02)

save(fig, os.path.join(FIGURES, "fig_placement"))
print({sn: {k: v for k, v in counts[sn].items()} for sn in counts}, "certified", len(err_c), int((err_c < RIGHT).sum()),
      "refused", len(err_r), int((err_r < RIGHT).sum()))
