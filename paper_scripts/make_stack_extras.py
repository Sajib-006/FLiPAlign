"""Table 4 (disjoint gene panels) and the end-to-end numbers (placement plus labelling).
Reads results/stack/stackv2_*.json, stackstal_*.json (STAligner), stackplace_summary.json, stackplacefk_summary.json.
Writes tables/stackv2.tex, tables/stackv2_numbers.tex, tables/stackplace_numbers.tex.
Run from paper_scripts/ (env RESULTS, OUT)"""
import glob, json, os
import numpy as np
from scipy.stats import wilcoxon
from _paths import STACK, TABLES

# ---- Table 4: split against forest methods and gene baseline, shared and disjoint gene panels ----
V = {}
for f in glob.glob(os.path.join(STACK, "stackv2_[0-9].json")):
    d = json.load(open(f)); V[d["stack"]] = d["runs"]
V = [V[k] for k in sorted(V)]
n = len(V)
nmi = lambda p, k: np.array([v[p][k]["nmi"] for v in V])
se = lambda a: np.std(a, ddof=1) / np.sqrt(len(a))
cell = lambda a, bold=False: ("\\textbf{%s}" if bold else "%s") % f"{a.mean():.3f}\\,$\\pm$\\,{se(a):.3f}"
gain = lambda p, g: float(np.median(100 * (nmi(p, "split") / nmi(p, g) - 1)))
wins = lambda p, g: int(np.sum(nmi(p, "split") > nmi(p, g)))
rnd = lambda x: f"{int(np.floor(x + 0.5))}"          # round half up
ST = [json.load(open(f))["k8"]["nmi"] for f in sorted(glob.glob(os.path.join(STACK, "stackstal_[0-9].json")))]   # STAligner, 8 classes
ST = np.array(ST)

lines = ["\\begin{tabular}{@{}lccccc@{}}", "\\toprule",
         " & & Forest, merged & Gene-mean & & \\\\",
         "Gene panels & Forest & to split's classes & matching & STAligner & Split \\\\", "\\midrule",
         "Shared & " + " & ".join([cell(nmi("shared_panel", "forest")), cell(nmi("shared_panel", "forest_merged")),
                                   cell(nmi("shared_panel", "gene_chain")), cell(ST), cell(nmi("shared_panel", "split"), True)]) + " \\\\",
         "Disjoint & " + " & ".join([cell(nmi("disjoint_panels", "forest")), cell(nmi("disjoint_panels", "forest_merged")),
                                    "--", "--", cell(nmi("disjoint_panels", "split"), True)]) + " \\\\",
         "\\bottomrule", "\\end{tabular}"]
open(os.path.join(TABLES, "stackv2.tex"), "w").write("\n".join(lines) + "\n")

D = "disjoint_panels"; S = "shared_panel"
pval = wilcoxon(nmi(D, "split"), nmi(D, "forest")).pvalue
N = {"SVmergedGain": rnd(gain(S, "forest_merged")), "SVmergedWins": wins(S, "forest_merged"),
     "SVgeneGain": rnd(gain(S, "gene_chain")), "SVgeneWins": wins(S, "gene_chain"),
     "SVdisGain": rnd(gain(D, "forest")), "SVdisWins": wins(D, "forest"), "SVdisP": f"{pval:.3f}",
     "SVdisSplit": f"{nmi(D, 'split').mean():.3f}", "SVdisForest": f"{nmi(D, 'forest').mean():.3f}",
     "SVdisMergedWins": wins(D, "forest_merged"), "SVgeneMean": f"{nmi(S, 'gene_chain').mean():.3f}"}
open(os.path.join(TABLES, "stackv2_numbers.tex"), "w").write("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in N.items()))

# ---- end to end: place the sections, then label (PS: flip searched, PK: flip known) ----
FAR_UM = 1000          # a link is "far" if its placement error is above 1000 um
P = {}
for pre, tag in (("stackplace", "PS"), ("stackplacefk", "PK")):
    S_ = json.load(open(os.path.join(STACK, f"{pre}_summary.json")))
    e = np.array([x for s in S_ for x in s["pair_err_um"]]); c = np.array([bool(x) for s in S_ for x in s["certified"]])
    far = e > FAR_UM
    P.update({f"{tag}cert": int(c.sum()), f"{tag}certMax": int(round(e[c].max())) if c.any() else 0,
              f"{tag}far": int(far.sum()), f"{tag}farRefused": int((far & ~c).sum()), f"{tag}medErr": int(round(np.median(e))),
              f"{tag}split": f"{np.mean([s['placed']['split']['nmi'] for s in S_]):.3f}",
              f"{tag}forest": f"{np.mean([s['placed']['forest']['nmi'] for s in S_]):.3f}",
              f"{tag}allen": f"{np.mean([s['allen']['split']['nmi'] for s in S_]):.3f}",
              f"{tag}wins": int(sum(s["placed"]["split"]["nmi"] > s["placed"]["forest"]["nmi"] for s in S_))})
open(os.path.join(TABLES, "stackplace_numbers.tex"), "w").write("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in P.items()))
print("wrote stackv2.tex, stackv2_numbers.tex, stackplace_numbers.tex")
