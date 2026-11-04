"""Stack labelling results: table (mean +- SE over 6 stacks) and figure.
Reads results/stack/{stack,stackcap,stackbound2 or stackbound}_*.json.
Run from paper_scripts/ (env RESULTS, OUT): python3 make_stack_results.py -> tables/stack_results.tex, tables/stack_numbers.tex, figures/stack.pdf"""
import os
from _paths import RES, PLACE, STACK, OUT, TABLES, FIGURES
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = STACK
V = [("clean", "Clean"), ("S1_recluster", "One section reclustered"), ("S2_crop50", "Half a section cut away"),
     ("S3_noise30", "30\\% label noise, one section"), ("S4_two_noisy", "30\\% noise, two sections")]
J = {json.load(open(f))["stack"]: json.load(open(f)) for f in glob.glob(f"{R}/stack_[0-9].json")}
C = {json.load(open(f))["stack"]: json.load(open(f)) for f in glob.glob(f"{R}/stackcap_[0-9].json")}
bf = sorted(glob.glob(f"{R}/stackbound2_[0-9].json")) or sorted(glob.glob(f"{R}/stackbound_[0-9].json"))
B = {json.load(open(f))["stack"]: json.load(open(f)) for f in bf}
src = "stackbound2" if "stackbound2" in (bf[0] if bf else "") else "stackbound"
S = sorted(set(J) & set(C))
CT = {json.load(open(f))["stack"]: json.load(open(f)) for f in glob.glob(f"{R}/stackctrl_[0-9].json")}   # cap-1 control (stack_ctrl.py)


def get(v, m, key="ari"):
    out = []
    for s in S:
        if m == "pooled":
            x = J[s]["runs"]["clean"].get("pooled_ceiling") if v == "clean" else None
        elif m in ("chain", "forest_d2", "star"):
            x = J[s]["runs"][v][m]
        elif m == "best11":                                   # best of the three one-to-one methods on this run
            x = max((J[s]["runs"][v][q] for q in ("chain", "forest_d2", "star")), key=lambda r: r[key])
        elif m == "cap1w":                                    # one-to-one under the split method's score w, same search
            x = CT[s]["runs"][v]["cap1_fromC"]
        else:
            x = C[s]["runs"][v][m]
        if x is not None: out.append(x[key])
    return np.array(out)


ms = lambda a: f"{a.mean():.3f} $\\pm$ {a.std(ddof=1) / np.sqrt(len(a)):.3f}"
M = [("chain", "One-to-one, chain (exact)"), ("forest_d2", "One-to-one, forest $d=2$ + ascent"), ("star", "One-to-one, star"),
     ("cap2_d2", "Split, cap 2 (ours)"), ("pooled", "Pooled clustering$^\\dagger$")]
def write_table(key, path):
    """rows: stack versions; columns: methods, then split-vs-forest gain and the certificate (standard layout, no
    vertical rules). Pooled clustering (one value, needs shared genes) and classes used are given in the text."""
    cols = [("chain", "Chain"), ("forest_d2", "Forest"), ("star", "Star"), ("cap2_d2", "Split")]
    pm = lambda a: f"{a.mean():.3f}\\,$\\pm$\\,{a.std(ddof=1) / np.sqrt(len(a)):.3f}"
    lines = ["\\begin{tabular}{@{}l" + "c" * len(cols) + "cc@{}}", "\\toprule",
             " & \\multicolumn{3}{c}{One-to-one} & Ours & Split gain & Proven \\\\",
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-5}",
             "Stack & " + " & ".join(n for _, n in cols) + " & (wins) & share \\\\", "\\midrule"]
    for v, name in V:
        best = max(get(v, m, key).mean() for m, _ in cols)
        cells = []
        for m, _ in cols:
            a = get(v, m, key)
            cells.append("--" if not len(a) else (("\\textbf{" + pm(a) + "}") if abs(a.mean() - best) < 1e-12 else pm(a)))
        g = get(v, "cap2_d2", key) / get(v, "forest_d2", key) - 1
        r = [B[s]["runs"][v]["d2"]["ratio_to_pair_bound"] for s in S if s in B and v in B[s]["runs"]]
        short = {"clean": "Clean", "S1_recluster": "One reclustered", "S2_crop50": "Half cut away", "S3_noise30": "Noise, one", "S4_two_noisy": "Noise, two"}[v]
        lines.append(f"{short} & " + " & ".join(cells) + f" & {100 * np.median(g):.0f}\\% ({int((g > 0).sum())}/{len(g)}) & " + (f"{min(r):.2f}--{max(r):.2f}" if r else "--") + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(path, "w").write("\n".join(lines) + "\n")
write_table("nmi", os.path.join(TABLES, "stack_results_full.tex")); write_table("ari", os.path.join(TABLES, "stack_results_ari.tex"))


def write_main(key, path):
    """main-text table: best one-to-one per run (of chain, forest, star), cap 1 under the same score w, split; gain against the best
    one-to-one (wins), proven share"""
    cols = [("best11", "Best"), ("cap1w", "Cap 1, $w$"), ("cap2_d2", "Split")]
    pm = lambda a: f"{a.mean():.3f}\\,$\\pm$\\,{a.std(ddof=1) / np.sqrt(len(a)):.3f}"
    lines = ["\\begin{tabular}{@{}lccccc@{}}", "\\toprule",
             " & \\multicolumn{2}{c}{One-to-one} & Ours & Gain over & Proven \\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-4}",
             "Stack & " + " & ".join(n for _, n in cols) + " & Best (wins) & share \\\\", "\\midrule"]
    for v, name in V:
        best = max(get(v, m, key).mean() for m, _ in cols)
        cells = [("\\textbf{" + pm(get(v, m, key)) + "}") if abs(get(v, m, key).mean() - best) < 1e-12 else pm(get(v, m, key)) for m, _ in cols]
        g = get(v, "cap2_d2", key) / get(v, "best11", key) - 1
        r = [B[s]["runs"][v]["d2"]["ratio_to_pair_bound"] for s in S if s in B and v in B[s]["runs"]]
        short = {"clean": "Clean", "S1_recluster": "One reclustered", "S2_crop50": "Half cut away", "S3_noise30": "Noise, one", "S4_two_noisy": "Noise, two"}[v]
        lines.append(f"{short} & " + " & ".join(cells) + f" & {100 * np.median(g):.0f}\\% ({int((g > 0).sum())}/{len(g)}) & {min(r):.2f}--{max(r):.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(path, "w").write("\n".join(lines) + "\n")
write_main("nmi", os.path.join(TABLES, "stack_results.tex"))

gn = np.concatenate([get(v, "cap2_d2", "nmi") / get(v, "forest_d2", "nmi") - 1 for v, _ in V])
ga = np.concatenate([get(v, "cap2_d2", "ari") / get(v, "forest_d2", "ari") - 1 for v, _ in V])
gain = np.median(gn)
wins = sum(int(C[s]["runs"][v]["cap2_d2"]["nmi"] > C[s]["runs"][v]["jlp_forest_d2"]["nmi"]) for s in S for v, _ in V)
allr = [B[s]["runs"][v]["d2"]["ratio_to_pair_bound"] for s in B for v in B[s]["runs"]]
cl = [B[s]["runs"]["clean"]["d2"]["ratio_to_pair_bound"] for s in B]
gb = np.concatenate([get(v, "cap2_d2", "nmi") / get(v, "best11", "nmi") - 1 for v, _ in V])
gba = np.concatenate([get(v, "cap2_d2", "ari") / get(v, "best11", "ari") - 1 for v, _ in V])
gc = np.concatenate([get(v, "cap2_d2", "nmi") / get(v, "cap1w", "nmi") - 1 for v, _ in V])
nums = {"STbestWins": int((gb > 0).sum()), "STbestGain": f"{100 * np.median(gb):.0f}", "STbestWinsAri": int((gba > 0).sum()),
        "STbestGainAri": f"{100 * np.median(gba):.0f}", "STcapOneWins": int((gc > 0).sum()), "STcapOneGain": f"{100 * np.median(gc):.0f}",
        "STn": len(S), "STgain": f"{100 * gain:.0f}", "STwins": wins, "STruns": len(S) * len(V),
        "STcertLo": f"{int(100 * min(cl))}", "STcertHi": f"{100 * max(cl):.0f}",
        "STcertAllLo": f"{int(100 * min(allr))}", "STcertAllHi": f"{100 * max(allr):.0f}", "STcertN": len(allr),
        "STgainAri": f"{100 * np.median(ga):.0f}", "STwinsAri": int((ga > 0).sum()), "STwinsNmi": int((gn > 0).sum()),
        "STsplit": f"{get('clean', 'cap2_d2').mean():.3f}", "STone": f"{get('clean', 'forest_d2').mean():.3f}",
        "STpooled": f"{get('clean', 'pooled').mean():.3f}", "STpooledNmi": f"{get('clean', 'pooled', 'nmi').mean():.3f}", "STsplitNmi": f"{get('clean', 'cap2_d2', 'nmi').mean():.3f}", "STchain": f"{get('clean', 'chain').mean():.3f}"}
open(os.path.join(TABLES, "stack_numbers.tex"), "w").write("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in nums.items()))
print(src, nums)

# ---- figure: (a) ARI per variant, (b) proven share of optimum per run
BLUE, ORANGE, GREY, INK = "#2a78d6", "#eb6834", "#9a9a95", "#222222"
plt.rcParams.update({"font.size": 7.5, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True})
fig, ax = plt.subplots(1, 2, figsize=(4.8, 2.0), gridspec_kw={"width_ratios": [1.6, 1]})
a = ax[0]; SHORT = ["clean", "reclust.", "cut", "noise 1", "noise 2"]
for j, (v, n) in enumerate(V):
    g = 100 * (get(v, "cap2_d2", "nmi") / get(v, "forest_d2", "nmi") - 1)
    a.scatter(np.full(len(g), j) + np.linspace(-0.14, 0.14, len(g)), g, s=12, color=BLUE, lw=0, zorder=3)
    a.plot([j - 0.25, j + 0.25], [np.median(g), np.median(g)], color=INK, lw=1.4, zorder=4)
    a.text(j + 0.28, np.median(g), f"{np.median(g):.0f}$\\%$", va="center", fontsize=7, color=INK)
a.axhline(0, color=GREY, lw=0.7)
a.set_xticks(range(len(V))); a.set_xticklabels(SHORT, fontsize=7); a.set_xlim(-0.5, len(V) - 0.3)
a.set_ylabel(r"NMI gain ($\%$)")
a.set_title(r"$\mathbf{a}$  Gain of split labelling per stack", fontsize=7.5, loc="left")
b = ax[1]
for j, (v, n) in enumerate(V):
    r = [B[s]["runs"][v]["d2"]["ratio_to_pair_bound"] for s in S if s in B and v in B[s]["runs"]]
    b.scatter(np.full(len(r), j) + np.linspace(-0.12, 0.12, len(r)), r, s=12, color=BLUE, lw=0)
b.set_xticks(range(len(V))); b.set_xticklabels(["clean", "recl.", "cut", "n1", "n2"], fontsize=7)
b.set_ylim(0.75, 1.0); b.set_ylabel("proven share")
b.set_title(r"$\mathbf{b}$  Certificate per run", fontsize=7.5, loc="left")
fig.tight_layout(w_pad=1.2)
fig.savefig(os.path.join(FIGURES, "stack.pdf")); fig.savefig(os.path.join(FIGURES, "stack.png"), dpi=220)
