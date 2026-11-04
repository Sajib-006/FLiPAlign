"""Supplement tables and numbers for the runs answering the 2026-11-02 review.
Reads results/stack/{stackctrl,stacksweep}_*.json, {stackplace,stackplacefk}_{summary,refuse}.json and
results/boot_sections.json. Writes tables/rev_*.tex and tables/review_numbers.tex.
Run from paper_scripts/ (env RESULTS, OUT): python3 make_review_tables.py"""
import os
from _paths import RES, PLACE, STACK, OUT, TABLES, FIGURES
import glob, json
import numpy as np

R = STACK
V = [("clean", "Clean"), ("S1_recluster", "One reclustered"), ("S2_crop50", "Half cut away"), ("S3_noise30", "Noise, one"),
     ("S4_two_noisy", "Noise, two")]
load = lambda pat: {json.load(open(f))["stack"]: json.load(open(f)) for f in glob.glob(f"{R}/{pat}_[0-9].json")}
CT, SW = load("stackctrl"), load("stacksweep")
pm = lambda a: f"{np.mean(a):.3f}\\,$\\pm$\\,{np.std(a, ddof=1) / np.sqrt(len(a)):.3f}"
N = {}


def save(name, lines):
    open(f"{TABLES}/{name}.tex", "w").write("\n".join(lines) + "\n")


# 1. cap-1 control, both starts
keys = [("start_fromC", "Forest ($C$)"), ("cap1_fromC", "Cap 1"), ("cap2_fromC", "Cap 2"), ("start_fromW", "Forest ($w$)"),
        ("cap1_fromW", "Cap 1"), ("cap2_fromW", "Cap 2")]
L = ["\\begin{tabular}{@{}lcccccc@{}}", "\\toprule",
     " & \\multicolumn{3}{c}{Start: forest under $C$ (as in the paper)} & \\multicolumn{3}{c}{Start: forest under $w$} \\\\",
     "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}", "Stack & " + " & ".join(n for _, n in keys) + " \\\\", "\\midrule"]
for v, name in V:
    L.append(name + " & " + " & ".join(pm([CT[s]["runs"][v][k]["nmi"] for s in sorted(CT)]) for k, _ in keys) + " \\\\")
L.append("\\midrule")
for a, b in (("cap2_fromC", "cap1_fromC"), ("cap2_fromW", "cap1_fromW")):
    g = np.array([CT[s]["runs"][v]["nmi"] if False else CT[s]["runs"][v][a]["nmi"] / CT[s]["runs"][v][b]["nmi"] - 1 for s in CT for v, _ in V])
    N[f"CT{a[-1]}wins"] = int((g > 0).sum()); N[f"CT{a[-1]}gain"] = f"{100 * np.median(g):.1f}"
same = sum(CT[s]["runs"][v]["cap1_fromC"]["value"] == CT[s]["runs"][v]["start_fromC"]["value"] for s in CT for v, _ in V)
N["CTsame"] = same
L.append(f"\\multicolumn{{7}}{{l}}{{Cap 2 over cap 1, same start and score: {N['CTCwins']}/30 runs higher (median {N['CTCgain']}\\%) from the $C$ start, "
         f"{N['CTWwins']}/30 ({N['CTWgain']}\\%) from the $w$ start.}} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
save("rev_capone", L)

# 2. cap and class-count sweep
ks = ["start"] + [f"cap{c}_G{g}" for g in (8, 12) for c in (1, 2, 3, 4)]
L = ["\\begin{tabular}{@{}lrrrr@{}}", "\\toprule", "Setting & NMI & ARI & Classes used & Value of $w$ \\\\", "\\midrule"]
for k in ks:
    g = lambda f: np.mean([SW[s]["runs"][v][k][f] for s in SW for v, _ in V])
    nm = "Forest start" if k == "start" else f"Cap {k[3]}, $G={k.split('G')[1]}$"
    L.append(f"{nm} & {g('nmi'):.3f} & {g('ari'):.3f} & {g('n_classes'):.2f} & {g('value'):.3f} \\\\")
for c in (3, 4):
    r = np.array([SW[s]["runs"][v][f"cap{c}_G8"]["nmi"] / SW[s]["runs"][v]["cap2_G8"]["nmi"] - 1 for s in SW for v, _ in V])
    N[{3: "SWcapThreeWins", 4: "SWcapFourWins"}[c]] = int((r > 0).sum())
L += ["\\bottomrule", "\\end{tabular}"]
save("rev_sweep", L)

# 3. two steps together: every link chained, and the stack cut at refused links
L = ["\\begin{tabular}{@{}llrrrrrrr@{}}", "\\toprule",
     " & & & \\multicolumn{3}{c}{Every link chained} & \\multicolumn{3}{c}{Cut at refused links} \\\\",
     "\\cmidrule(lr){4-6}\\cmidrule(lr){7-9}",
     "Flip & Stack & Certified & Max drift & Forest & Split & Max drift & Forest & Split \\\\",
     " & & links & ($\\mu$m) & NMI & NMI & ($\\mu$m) & NMI & NMI \\\\", "\\midrule"]
for pre, lab in (("stackplacefk", "known"), ("stackplace", "searched")):
    A, C = json.load(open(f"{R}/{pre}_summary.json")), json.load(open(f"{R}/{pre}_refuse.json"))
    for a, c in zip(A, C):
        f_ = lambda x: "--" if x is None else f"{x:.3f}"
        sizes = "+".join(str(len(p)) for p in c["pieces"])
        L.append(f"{lab if a['stack'] == 0 else ''} & {a['stack'] + 1} & {sum(c['certified'])}/7 & {max(a['drift_um']):.0f} & "
                 f"{a['placed']['forest']['nmi']:.3f} & {a['placed']['split']['nmi']:.3f} & "
                 f"{max(c['drift_um']):.0f} & {f_(c['nmi_forest_pieces'])} & {f_(c['nmi_split_pieces'])} \\\\")
    sh = [c["share_in_pieces_ge2"] for c in C]
    N[f"PC{lab}share"] = f"{100 * np.mean(sh):.0f}"
    N[f"PC{lab}maxdrift"] = f"{max(max(c['drift_um']) for c in C):.0f}"
    N[f"PC{lab}alldrift"] = f"{max(max(a['drift_um']) for a in A):.0f}"
    N[f"PC{lab}wins"] = sum((c["nmi_split_pieces"] or 0) > (c["nmi_forest_pieces"] or 0) for c in C)
    N[f"PC{lab}pieces"] = sum(1 for c in C if c["share_in_pieces_ge2"] > 0)
    if lab == "known":
        L.append("\\midrule")
L += ["\\bottomrule", "\\end{tabular}"]
save("rev_pipeline", L)

# 4. bootstrap over source sections
Bt = json.load(open(os.path.join(RES, "boot_sections.json")))
N.update({"BSratio": f"{Bt['both']['ratio']:.1f}", "BSlo": f"{Bt['both']['ratio_ci'][0]:.2f}", "BShi": f"{Bt['both']['ratio_ci'][1]:.2f}",
          "BSp": f"{Bt['both']['p_flip_not_better']:.3f}", "BStwoRatio": f"{Bt['second']['ratio']:.2f}",
          "BStwoLo": f"{Bt['second']['ratio_ci'][0]:.2f}", "BStwoHi": f"{Bt['second']['ratio_ci'][1]:.2f}",
          "BStwoP": f"{Bt['second']['p_flip_not_better']:.2f}"})
open(os.path.join(TABLES, "review_numbers.tex"), "w").write("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in N.items()))
print(N)


# 5. main-text certificate precision table (every mode, both held-out sets, end-to-end links)
import re
from scipy.stats import beta
mac = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", open(os.path.join(TABLES, "res3_numbers.tex")).read()))
def cpi(k, n):
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1); hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return f"{100 * lo:.0f}--{100 * hi:.0f}"
rows = []
for key, lab in (("HB", "Held-out 2 (clean)"), ("HA", "Held-out 1")):
    for m, ml in (("R", "flip searched"), ("S", "\\quad + self-check"), ("K", "flip known")):
        c, r = int(mac[f"{key}{m}cert"]), int(mac[f"{key}{m}cr"])
        rows.append(f"{lab if m == 'R' else ''} & {ml} & {c} & {r} & {c - r} & {cpi(r, c)} \\\\")
    rows.append("\\midrule")
E = [json.load(open(f))[0] for f in sorted(glob.glob(f"{R}/stackplacefk_[0-9]*.json")) if not f.endswith(("summary.json", "refuse.json"))]
c = [e for e in E if e["certified"]]; r = sum(e["err_um"] < 200 for e in c)
rows.append(f"Stack links & flip known & {len(c)} & {r} & {len(c) - r} & {cpi(r, len(c))} \\\\")
N.update({"PKcertRight": r, "PKcertWrong": len(c) - r})
L = ["\\begin{tabular}{@{}llrrrc@{}}", "\\toprule", "Cases & Mode & Certified & Right & Wrong & Right share (\\%, 95\\% CI) \\\\", "\\midrule"] + rows + ["\\bottomrule", "\\end{tabular}"]
save("rev_certprec", L)
open(os.path.join(TABLES, "review_numbers.tex"), "w").write("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in N.items()))
print(open(os.path.join(TABLES, "rev_certprec.tex")).read())


# 6. second brain (Zhuang MERFISH) and atlas-free agreement of neighbouring sections, both brains
VV = [v for v, _ in V]
BR = {"b1": [json.load(open(f)) for f in sorted(glob.glob(f"{R}/stackcons_[0-9].json"))],
      "b2": [json.load(open(f)) for f in sorted(glob.glob(f"{R}/stackb2_[0-9].json"))]}
def pick(d, v, m):
    if m == "best":
        return max((d["runs"][v][q] for q in ("chain", "forest_d2", "star")), key=lambda r: r["nmi"])
    return d["runs"][v][m]
meth = [("chain", "Chain"), ("forest_d2", "Forest"), ("star", "Star"), ("best", "Best one-to-one"), ("cap1_fromC", "Cap 1, $w$"), ("cap2_fromC", "Split")]
L = ["\\begin{tabular}{@{}lrrrrrr@{}}", "\\toprule",
     " & \\multicolumn{3}{c}{Brain 1 (Allen, 500 genes)} & \\multicolumn{3}{c}{Brain 2 (Zhuang, 1{,}122 genes)} \\\\",
     "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}", "Labelling & NMI & Kappa & Classes & NMI & Kappa & Classes \\\\", "\\midrule"]
for m, nm in meth:
    cells = []
    for b in ("b1", "b2"):
        x = [pick(d, v, m) for d in BR[b] for v in VV]
        cells += [pm([r["nmi"] for r in x]), pm([r["kappa"] for r in x]), f"{np.mean([r['n_classes'] for r in x]):.1f}"]
    L.append(f"{nm} & " + " & ".join(cells) + " \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
save("rev_brains", L)
for b, tag in (("b1", "One"), ("b2", "Two")):
    sp = np.array([pick(d, v, "cap2_fromC")["nmi"] for d in BR[b] for v in VV]); bb = np.array([pick(d, v, "best")["nmi"] for d in BR[b] for v in VV])
    c1 = np.array([pick(d, v, "cap1_fromC")["nmi"] for d in BR[b] for v in VV])
    ks = np.mean([pick(d, v, "cap2_fromC")["kappa"] for d in BR[b] for v in VV]); kb = np.mean([pick(d, v, "best")["kappa"] for d in BR[b] for v in VV])
    N.update({f"BR{tag}wins": int((sp > bb).sum()), f"BR{tag}gain": f"{100 * np.median(sp / bb - 1):.0f}",
              f"BR{tag}capwins": int((sp > c1).sum()), f"BR{tag}capgain": f"{100 * np.median(sp / c1 - 1):.0f}",
              f"BR{tag}kapSplit": f"{ks:.2f}", f"BR{tag}kapBest": f"{kb:.2f}"})
cl2 = [(pick(d, "clean", "cap2_fromC")["nmi"], pick(d, "clean", "best")["nmi"]) for d in BR["b2"]]
N["BRTwoClean"] = sum(a > b for a, b in cl2)
open(os.path.join(TABLES, "review_numbers.tex"), "w").write("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in N.items()))
print({k: v for k, v in N.items() if k.startswith("BR")})
