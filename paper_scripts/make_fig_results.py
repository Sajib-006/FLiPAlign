"""Results figure (review 5): what the proof gives.
(a) proven margin eps against the plain grid margin: eps sits between the margin and the margin minus the
    bound slack (Theorem "complete up to lambda"); (b) certified answers, right and wrong, for the certificate
    and for an unproven margin rule whose threshold is tuned on all development cases or on one development
    tissue; (c) right answers against overlap on mouse brain, flip searched and flip known.
Run from paper_scripts/ (env RESULTS, OUT): python3 make_fig_results.py  -> figures/results.pdf"""
import os
from _paths import RES, PLACE, STACK, OUT, TABLES, FIGURES
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from make_results import load, manifest, RIGHT
from make_results2 import load_s, ad

BLUE, ORANGE, GREY, INK = "#00A087", "#E64B35", "#8A8A8A", "#222222"   # figstyle: right green, wrong red
plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42, "font.family": "sans-serif", "font.sans-serif": ["Arial"]})


def gm(g):
    per = g["per_angle"]; b = max(per, key=lambda p: p[2])
    far = [p[2] for p in per if p[1] != b[1] or ad(p[0], b[0]) > 20]
    return b[2] - max(far), (b[2] - max(far)) / max(b[2], 1e-9), b[5]


def rows(setn, op, tissue):
    O, G = load_s(op), load(f"grid_{setn}"); out = []
    for c in sorted(set(O) & set(G)):
        m, rel, gtre = gm(G[c])
        out.append(dict(t=tissue(c), m=m, rel=rel, gright=gtre < RIGHT, cert=bool(O[c]["certified"]),
                        right=O[c]["tre_um"] < RIGHT, eps=O[c]["ladder"][0]["eps"] if O[c]["ladder"] else None))
    return out


m2 = {c: m["tissue"] for c, m in manifest("gt2").items()}
dev = rows("gt1", "gt1h2", lambda c: "breast") + rows("gt2", "gt2h2", lambda c: m2[c])
ho = rows("gt3", "gt3h2", lambda c: "") + (rows("gt4", "gt4h2", lambda c: "") if load("gt4h2") and load("grid_gt4") else [])
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.35), gridspec_kw={"width_ratios": [1.0, 1.45, 0.85]})

# ---- (a) eps against the grid margin
a = ax[0]; pts = [r for r in dev + ho if r["eps"] is not None]
mm = np.array([r["m"] for r in pts]); ee = np.array([r["eps"] for r in pts]); rg = np.array([r["right"] for r in pts])
slack = float(np.median(mm - ee))
lim = [0, 0.11]
a.plot(lim, lim, color=GREY, lw=0.8); a.plot(lim, [lim[0] - slack, lim[1] - slack], color=GREY, lw=0.8, ls="--")
a.axhline(0, color=INK, lw=0.5)
a.scatter(mm[rg], ee[rg], s=9, color=BLUE, lw=0, alpha=0.8, label="answer right"); a.scatter(mm[~rg], ee[~rg], s=11, color=ORANGE, lw=0, marker="^", alpha=0.9, label="answer wrong")
a.set_xlim(0, 0.11); a.set_ylim(-0.025, 0.11); a.set_xlabel("margin on a 1$^\\circ$ grid (no proof)"); a.set_ylabel("proven margin $\\varepsilon$")
a.text(0.050, 0.064, "$\\varepsilon$ = margin", rotation=41, color=GREY, fontsize=7); a.text(0.074, 0.040, "margin $-$ slack", rotation=41, color=GREY, fontsize=7)
a.text(0.108, 0.003, "certified above", ha="right", fontsize=7, color=INK)
a.legend(frameon=False, loc="upper left", handletextpad=0.2, borderpad=0.1, fontsize=7)
a.set_title("(a) Proven margin and grid margin", fontsize=7, loc="left")

# ---- (b) accepted answers on held-out: certificate vs rules tuned on development tissue
b = ax[1]
best_tau = lambda S, key: max(sorted({r[key] for r in S}), key=lambda x: sum((r[key] > x) == r["gright"] for r in S))
bars = [("certif.\n($h=2$)", sum(r["cert"] and r["right"] for r in ho), sum(r["cert"] and not r["right"] for r in ho))]
for lab, S in [("all dev.", dev)] + [(t, [r for r in dev if r["t"] == t]) for t in ("breast", "colon", "mouse brain", "DLPFC")]:
    tau = best_tau(S, "m"); acc = [r for r in ho if r["m"] > tau]
    bars.append((lab, sum(r["gright"] for r in acc), sum(not r["gright"] for r in acc)))
x = np.arange(len(bars)); R_ = [v[1] for v in bars]; W_ = [v[2] for v in bars]
b.bar(x, R_, width=0.62, color=BLUE, label="accepted, right"); b.bar(x, W_, width=0.62, bottom=np.array(R_) + 0.4, color=ORANGE, label="accepted, wrong")
for xi, (r_, w_) in enumerate(zip(R_, W_)):
    b.text(xi, r_ / 2, str(r_), ha="center", va="center", color="white", fontsize=7)
    if w_:
        b.text(xi, r_ + 0.4 + w_ / 2, str(w_), ha="center", va="center", color="white", fontsize=7)
    if r_ + w_ == 0:
        b.text(xi, 1, "0", ha="center", fontsize=7, color=INK)
b.set_xticks(x); b.set_xticklabels([v[0].replace("mouse brain", "brain") for v in bars], fontsize=7)
b.set_ylabel(f"held-out cases accepted (of {len(ho)})"); b.axvline(0.5, color=GREY, lw=0.5, ls=":")
b.text(3.0, -0.25, "margin rule, threshold tuned on", ha="center", transform=b.get_xaxis_transform(), fontsize=7)
b.legend(frameon=False, loc="upper right", fontsize=7, handlelength=1, borderpad=0.1)
b.set_title("(b) Margin rule against the certificate", fontsize=7, loc="left")

# ---- (c) overlap sweep on mouse brain
c = ax[2]; xs, ys, yk = [100, 80, 60], [], []
for d, n_ in (("gt_ov100", "ov100"), ("gt_ov80", "ov80"), ("gt_ov60", "ov60")):
    O, K = load(d), load(f"fk_{n_}")
    ys.append(sum(r["tre_um"] < RIGHT for r in O.values())); yk.append(sum(r["tre_um"] < RIGHT for r in K.values()))
from make_results3 import tool_best, GOOD
yt = []
for ov in xs:
    cs_ = sorted(load(f"gt_ov{ov}")); pats = [f"sw_ov{ov}base", f"sw_ov{ov}spateo", f"sw_ov{ov}cast"]
    g1 = tool_best(pats, cs_, GOOD, [f"sw_fo_ov{ov}base", f"sw_fo_ov{ov}spateo", f"sw_fo_ov{ov}cast"])
    yt.append(max(v[0] for v in g1.values() if v))
c.plot(xs, yk, color="#3C5488", lw=1.6, marker="o", ms=4, mfc="white", label="FLiPAlign, flip known"); c.plot(xs, ys, color="#3C5488", lw=1.6, marker="^", ms=4, label="FLiPAlign, flip searched")
c.plot(xs, yt, color="#4DBBD5", lw=1.4, marker="s", ms=3.5, ls="--", label="best tool, flip given")
c.legend(frameon=False, loc="upper center", fontsize=7, handlelength=1.6, borderpad=0.1)
c.set_xlim(104, 56); c.set_ylim(0, 31); c.set_yticks([0, 5, 10, 15, 20]); c.set_xticks(xs); c.set_xlabel("share of A on B (%)"); c.set_ylabel("brain cases right (of 20)")
c.set_title("(c) Mouse brain, by overlap", fontsize=7, loc="left")
fig.tight_layout(w_pad=1.0)
fig.savefig(os.path.join(FIGURES, "results.pdf")); fig.savefig(os.path.join(FIGURES, "results.png"), dpi=200)
print("slack", round(slack, 4), "bars", bars, "sweep", ys, yk)
