"""Round-5 tables and numbers (2026-10-05): second held-out set, tools under three starts, FLiPAlign as
the start of other tools, certificate against a margin rule tuned on development data, tight-bound
decision, score ablations, negative controls, and the appendix tables.
Reads results/placement; writes tables/res3_*.tex. Missing result files give "--" (rerun later).
Run from paper_scripts/ (env RESULTS, OUT): python3 make_results3.py"""
import os
from _paths import RES, PLACE, STACK, OUT, TABLES, FIGURES
import collections, glob, json, os
import numpy as np
from scipy.stats import beta
from make_results import load, manifest, pct, R, T, RIGHT, TOOLS
from make_results2 import load_s, holds, ad, selfcheck

M = {}
TOOLN = dict(TOOLS)


def cp(k, n):
    if not n:
        return "--"
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return f"{100 * lo:.0f}--{100 * hi:.0f}"


def rows_of(pats):
    out = []
    for p in pats:
        for f in glob.glob(f"{R}/{p}_t*.json"):
            out += [r for r in json.load(open(f)) if "tre_um" in r]
    return out


def tool_best(pats, cases, good, extra=None, key="tre_um"):
    """per tool: the (setting, fit) with most right answers over `cases` -> (right, n, {case: err}, name)"""
    by = collections.defaultdict(dict)
    for r in rows_of(pats):
        if key in r:
            by[(r["method"], r["fit"])][r["case"]] = r[key]
    if extra:                                              # flip-handed runs replace the flipped cases
        fo = collections.defaultdict(dict)
        for r in rows_of(extra):
            fo[(r["method"], r["fit"])][r["case"]] = r[key]
        for k in by:
            by[k].update(fo.get(k, {}))
    out = {}
    for tool, _ in TOOLS:
        best = None
        for (m, fit), v in by.items():
            if m.split("_")[0] != tool:
                continue
            vals = {c: v[c] for c in cases if c in v}
            if len(vals) < 0.8 * len(cases):
                continue
            k = sum(good(c, e) for c, e in vals.items())
            if best is None or k > best[0]:
                best = (k, len(vals), vals, f"{m}/{fit}")
        out[tool] = best
    return out


def gm(g, flip=None):
    per = [p for p in g["per_angle"] if flip is None or p[1] == flip]; b = max(per, key=lambda p: p[2])
    far = [p[2] for p in per if p[1] != b[1] or ad(p[0], b[0]) > 20]
    m = b[2] - max(far)
    return m, m / max(b[2], 1e-9), b[5]


SETS = {  # name: (dir, searched prefix, known prefix, fine (searched), fine (known), flipcheck, grid, good)
    "gt3": ("gt3", "gt3h2", "fk_gt3", "gt3fine", "fkfine_gt3", "fc_gt3", "grid_gt3"),
    "gt4": ("gt4", "gt4h2", "fk_gt4", "gt4fine", "fkfine_gt4", "fc_gt4", "grid_gt4"),
}
GOOD = lambda c, e: e < RIGHT


def cert_stats(D, cases, keep=None, good=GOOD, K=None):
    cert = [c for c in cases if c in D and D[c]["certified"] and (keep is None or keep(c))]
    hold = [holds(D[c], (K or D)[c]) for c in cert]
    return {"n": len([c for c in cases if c in D]), "right": sum(good(c, D[c]["tre_um"]) for c in cases if c in D),
            "cert": len(cert), "cr": sum(good(c, D[c]["tre_um"]) for c in cert),
            "holds": sum(1 for h in hold if h)}


# ============================================================ 1. the two held-out sets
def heldout():
    lines = []; tot = collections.Counter()
    order = {"gt3": ["human lymph node", "human liver", "human kidney", "human heart", "mouse brain (Prime 5K)", "mouse brain (V1)"],
             "gt4": ["human tonsil", "human pancreas", "human lung cancer", "human skin", "mouse pup", "human brain"]}
    for sn, (d, op, kp, fo, fk, fc, gr) in SETS.items():
        man = {c: m["tissue"] for c, m in manifest(d).items()}
        O, K, FK = load(op), load(kp), load(fk)
        if not O or not K:
            continue
        keep = selfcheck(fc) if load(fc) else None
        key = {"gt3": "A", "gt4": "B"}[sn]
        tb = lambda cs: tool_best([f"{sn}base", f"{sn}base3", f"{sn}slat", f"{sn}spateo", f"{sn}cast", f"{sn}stalign", f"rc_{sn}stalign"], cs, GOOD,
                                  [f"{sn}fo_base", f"{sn}fo_spateo", f"{sn}fo_cast"])
        tr = lambda cs: tool_best([f"rs_{sn}base", f"rs_{sn}spateo", f"rs_{sn}cast", f"rs_{sn}stalign", f"rc_rs_{sn}cpd", f"rc_rs_{sn}stalign"], cs, GOOD)
        def row(lab, cs):
            cs = [c for c in cs if c in O and c in K]
            raw, s, k = cert_stats(O, cs, K=K), cert_stats(O, cs, keep, K=K), cert_stats(K, cs)
            fin = [FK[c]["tre2_um"] if c in FK else K[c]["tre_um"] for c in cs if K[c]["certified"]]
            e = f"{np.median(fin):.0f}" if fin else "--"
            b1 = max((v[0] for v in tb(cs).values() if v), default=None); b2 = max((v[0] for v in tr(cs).values() if v), default=None)
            f_ = lambda v: "--" if v is None else str(v)
            return (f"{lab} & {len(cs)} & {raw['right']} & {raw['cert']} & {raw['cr']} & {s['cert']} & {s['cr']} & "
                    f"{k['right']} & {k['cert']} & {k['cr']} & {e} & {f_(b1)} & {f_(b2)} \\\\"), raw, s, k, e, b1, b2
        lines.append(f"\\multicolumn{{13}}{{l}}{{\\emph{{{'First held-out set (method frozen before it was made)' if sn == 'gt3' else 'Second held-out set (every part fixed before it was made)'}}}}} \\\\")
        for t in order[sn]:
            cs = sorted(c for c in man if man[c] == t)
            if cs:
                lines.append(row(t[0].upper() + t[1:], cs)[0])
        txt, raw, s, k, e, b1, b2 = row("All", sorted(man))
        lines += ["\\midrule" if False else "", txt, "\\midrule"]
        n = raw["n"]
        M.update({f"H{key}n": n, f"H{key}right": raw["right"], f"H{key}Rcert": raw["cert"], f"H{key}Rcr": raw["cr"],
                  f"H{key}Rholds": raw["holds"], f"H{key}Scert": s["cert"], f"H{key}Scr": s["cr"], f"H{key}Sholds": s["holds"],
                  f"H{key}Kright": k["right"], f"H{key}Kcert": k["cert"], f"H{key}Kcr": k["cr"], f"H{key}Kholds": k["holds"],
                  f"H{key}Kerr": e, f"H{key}toolFlip": b1 if b1 is not None else "--", f"H{key}toolRough": b2 if b2 is not None else "--",
                  f"H{key}rightCI": cp(raw["right"], n), f"H{key}RcrCI": cp(raw["cr"], raw["cert"]), f"H{key}ScrCI": cp(s["cr"], s["cert"]),
                  f"H{key}KrightCI": cp(k["right"], n), f"H{key}KcrCI": cp(k["cr"], k["cert"]),
                  f"H{key}Rcov": pct(raw["cert"], n), f"H{key}Scov": pct(s["cert"], n), f"H{key}Kcov": pct(k["cert"], n)})
        for a, b in (("right", raw["right"]), ("Rcert", raw["cert"]), ("Rcr", raw["cr"]), ("Scert", s["cert"]), ("Scr", s["cr"]),
                     ("Kright", k["right"]), ("Kcert", k["cert"]), ("Kcr", k["cr"]), ("n", n)):
            tot[a] += b
    if tot["n"]:
        M.update({f"HT{a}": v for a, v in tot.items()})
        M.update({"HTScrCI": cp(tot["Scr"], tot["Scert"]), "HTKcrCI": cp(tot["Kcr"], tot["Kcert"]), "HTRcrCI": cp(tot["Rcr"], tot["Rcert"]),
                  "HTrightCI": cp(tot["right"], tot["n"]), "HTKrightCI": cp(tot["Kright"], tot["n"])})
        M.update({"HTScov": pct(tot["Scert"], tot["n"]), "HTKcov": pct(tot["Kcert"], tot["n"]), "HTRcov": pct(tot["Rcert"], tot["n"])})
    lines = [l for l in lines if l]
    if lines and lines[-1] == "\\midrule":
        lines = lines[:-1]
    tab = ["\\begin{tabular}{lr|rrr|rr|rrr|r|rr}", "\\toprule",
           " & & \\multicolumn{5}{c|}{Flip searched} & \\multicolumn{3}{c|}{Flip known} & & \\multicolumn{2}{c}{Best tool, handed} \\\\",
           " & & & \\multicolumn{2}{c|}{no self-check} & \\multicolumn{2}{c|}{self-check} & & & & Err. & & flip + \\\\",
           "Tissue & $n$ & Right & Cert. & C\\&R & Cert. & C\\&R & Right & Cert. & C\\&R & ($\\mu$m) & flip & angle \\\\", "\\midrule"] + \
          lines + ["\\bottomrule", "\\end{tabular}"]
    open(f"{T}/res3_heldout.tex", "w").write("\n".join(tab) + "\n")


# ============================================================ 2. tools under three starts, and from FLiPAlign's start
def tools():
    lines = []
    cols = []
    for sn, lab, d, good in (("gt3", "Held-out 1", "gt3", None), ("gt4", "Held-out 2", "gt4", None),
                             ("abc", "MERFISH whole", "abc", "floor"), ("abc70", "MERFISH 70\\%", "abc_ov70", "floor")):
        man = manifest(d)
        if not man:
            continue
        cs = sorted(man)
        g = (lambda c, e, man=man: e < man[c]["rigid_floor_um"] + 100) if good else GOOD
        base = {"gt3": ["gt3base", "gt3base3", "gt3slat", "gt3spateo", "gt3cast", "gt3stalign", "rc_gt3stalign"],
                "gt4": ["gt4base", "gt4spateo", "gt4cast", "gt4stalign", "rc_gt4stalign"],
                "abc": ["abcbase", "abcspateo", "abccast", "abcstalign", "rc_abcstalign"],
                "abc70": ["abc_ov70base", "abc_ov70spateo", "abc70cast", "abc70stalign", "rc_abc70stalign"]}[sn]
        fo = [f"{sn}fo_base", f"{sn}fo_spateo", f"{sn}fo_cast"]
        a0 = tool_best(base, cs, g); a1 = tool_best(base, cs, g, fo)
        a2 = tool_best([f"rs_{sn}base", f"rs_{sn}spateo", f"rs_{sn}cast", f"rs_{sn}stalign", f"rc_rs_{sn}cpd", f"rc_rs_{sn}stalign"], cs, g)
        a3 = tool_best([f"in_{sn}base", f"in_{sn}spateo", f"in_{sn}cast", f"rc_in_{sn}cpd", f"rc_in_{sn}stalign"], cs, g)
        a4 = tool_best([f"ik_{sn}base", f"ik_{sn}spateo", f"ik_{sn}cast", f"rc_ik_{sn}cpd", f"rc_ik_{sn}stalign"], cs, g)
        O = load({"gt3": "gt3h2", "gt4": "gt4h2", "abc": "abc", "abc70": "abc_ov70"}[sn]); K = load(f"fk_{sn}")
        cols.append((sn, lab, len(cs), a0, a1, a2, a3, a4,
                     sum(g(c, O[c]["tre_um"]) for c in cs if c in O) if O else None,
                     sum(g(c, K[c]["tre_um"]) for c in cs if c in K) if K else None))
    f_ = lambda v: "--" if v is None else str(v[0])
    head1 = " & " + " & ".join(f"\\multicolumn{{3}}{{c{'|' if i < len(cols) - 1 else ''}}}{{{c[1]} ($n={c[2]}$)}}" for i, c in enumerate(cols)) + " \\\\"
    head2 = "Method & " + " & ".join("as given & flip & flip+angle" for _ in cols) + " \\\\"
    for tool, name in TOOLS:
        lines.append(f"{name} & " + " & ".join(f"{f_(c[3].get(tool))} & {f_(c[4].get(tool))} & {f_(c[5].get(tool))}" for c in cols) + " \\\\")
        for c in cols:
            for arm, a in (("Given", c[3]), ("Flip", c[4]), ("Rough", c[5]), ("In", c[6]), ("Ik", c[7])):
                M[f"T{c[0].replace('70', 'P').replace('3', 'A').replace('4', 'B')}{tool.replace('2', 'two')}{arm}"] = f_(a.get(tool))
    lines.append("\\midrule")
    lines.append("FLiPAlign & " + " & ".join(f"{'--' if c[8] is None else c[8]} & {'--' if c[9] is None else c[9]} & --" for c in cols) + " \\\\")
    tab = ["\\begin{tabular}{l|" + "|".join("rrr" for _ in cols) + "}", "\\toprule", head1, head2, "\\midrule"] + lines + ["\\bottomrule", "\\end{tabular}"]
    open(f"{T}/res3_tools.tex.body", "w").write("\n".join(tab) + "\n")
    for c in cols:
        key = c[0].replace("70", "P").replace("3", "A").replace("4", "B")
        for arm, a in (("Given", c[3]), ("Flip", c[4]), ("Rough", c[5]), ("In", c[6]), ("Ik", c[7])):
            v = [x[0] for x in a.values() if x]
            M[f"T{key}best{arm}"] = max(v) if v else "--"
        M[f"T{key}oursS"] = "--" if c[8] is None else c[8]; M[f"T{key}oursK"] = "--" if c[9] is None else c[9]
    return cols


def downstream():
    """tools started from FLiPAlign's placement: right answers and the tool's own (bent) error"""
    lines = []
    for sn, lab, d, op in (("abc", "MERFISH whole", "abc", "abc"), ("abc70", "MERFISH 70\\%", "abc_ov70", "abc_ov70")):
        man = manifest(d)
        if not man:
            continue
        cs = sorted(man); floor = "rigid_floor_um" in man[cs[0]]
        g = (lambda c, e, man=man: e < man[c]["rigid_floor_um"] + 100) if floor else GOOD
        key = sn.replace("70", "P").replace("3", "A").replace("4", "B")
        for arm, pre, ours in (("searched", "in", load(op)), ("known", "ik", load(f"fk_{sn}"))):
            if not ours:
                continue
            o_right = sum(g(c, ours[c]["tre_um"]) for c in cs if c in ours)
            o_med = float(np.median([ours[c]["tre_um"] for c in cs if c in ours]))
            cells = []
            for tool in ("stalign", "spateo", "cast", "cpd"):
                rig = tool_best([f"{pre}_{sn}base", f"{pre}_{sn}spateo", f"{pre}_{sn}cast", f"rc_{pre}_{sn}cpd", f"rc_{pre}_{sn}stalign"], cs, g).get(tool)
                ben = tool_best([f"{pre}_{sn}base", f"{pre}_{sn}spateo", f"{pre}_{sn}cast", f"rc_{pre}_{sn}cpd", f"rc_{pre}_{sn}stalign"], cs, g, key="tre_nonrigid_um").get(tool)
                alone = tool_best({"gt3": ["gt3base", "gt3base3", "gt3spateo", "gt3cast", "gt3stalign", "rc_gt3stalign"], "gt4": ["gt4base", "gt4spateo", "gt4cast", "gt4stalign", "rc_gt4stalign"],
                                   "abc": ["abcbase", "abcspateo", "abccast", "abcstalign", "rc_abcstalign"],
                                   "abc70": ["abc_ov70base", "abc_ov70spateo", "abc70cast", "abc70stalign", "rc_abc70stalign"]}[sn], cs, g).get(tool)
                r_ = "--" if rig is None else rig[0]; a_ = "--" if alone is None else alone[0]
                bm = "--" if ben is None else f"{np.median(list(ben[2].values())):.0f}"
                bb = "--" if ben is None else ben[0]
                cells.append(f"{a_} & {r_} & {bb}")
                M[f"D{key}{arm[0].upper()}{tool}Alone"] = a_; M[f"D{key}{arm[0].upper()}{tool}Rigid"] = r_
                M[f"D{key}{arm[0].upper()}{tool}Bent"] = bb; M[f"D{key}{arm[0].upper()}{tool}BentMed"] = bm
            M[f"D{key}{arm[0].upper()}ours"] = o_right; M[f"D{key}{arm[0].upper()}oursMed"] = f"{o_med:.0f}"
            lines.append(f"{lab}, flip {arm} & {len(cs)} & {o_right} & " + " & ".join(cells) + " \\\\")
    tab = ["\\begin{tabular}{lr|r|rrr|rrr|rrr|rrr}", "\\toprule",
           " & & FLiP- & \\multicolumn{3}{c|}{STalign} & \\multicolumn{3}{c|}{spateo} & \\multicolumn{3}{c|}{CAST} & \\multicolumn{3}{c}{CPD} \\\\",
           "Pairs & $n$ & Align & alone & +F & +F bent & alone & +F & +F bent & alone & +F & +F bent & alone & +F & +F bent \\\\", "\\midrule"] + \
          lines + ["\\bottomrule", "\\end{tabular}"]
    open(f"{T}/res3_downstream.tex", "w").write("\n".join(tab) + "\n")
    # breast landmark pair
    xx = []
    for f in glob.glob(f"{R}/xxds_*_t*.json"):
        xx += json.load(open(f))
    for r in xx:
        v = None
        for k in ("landmark_um_bent", "landmark_um_plain", "landmark_um_trimmed"):
            if k in r:
                m_ = float(np.median(r[k])); tag = "Bent" if k.endswith("bent") else "Rigid"
                kk = f"XXD{r['start'][:1].upper()}{r['method'].replace('_', '').replace('1', 'one')}{tag}"
                M[kk] = f"{min(m_, float(M.get(kk, 1e9))):.0f}"
        M[f"XXD{r['start'][:1].upper()}start"] = f"{np.median(r['start_landmark_um']):.0f}"
    return xx


# ============================================================ 3. certificate against a tuned margin rule; tight bound
def proof():
    def rows(setn, op, fk, tissue):
        O, G, K = load_s(op), load(f"grid_{setn}"), load(fk); out = []
        for c in sorted(set(O) & set(G)):
            m, rel, gtre = gm(G[c])
            out.append(dict(c=c, t=tissue(c), m=m, rel=rel, right=gtre < RIGHT, cert=bool(O[c]["certified"]), cright=O[c]["tre_um"] < RIGHT))
        return out
    m2 = {c: m["tissue"] for c, m in manifest("gt2").items()}
    dev = rows("gt1", "gt1h2", "fk_gt1", lambda c: "breast") + rows("gt2", "gt2h2", "fk_gt2", lambda c: m2[c])
    best_tau = lambda S, key: max(sorted({r[key] for r in S}), key=lambda x: sum((r[key] > x) == r["right"] for r in S))
    taus = {"m": best_tau(dev, "m"), "rel": best_tau(dev, "rel")}
    M.update({"RTtauM": f"{taus['m']:.4f}", "RTtauRel": f"{taus['rel']:.3f}"})
    per = {k: {t: best_tau([r for r in dev if r["t"] == t], k) for t in sorted({r["t"] for r in dev})} for k in ("m", "rel")}
    M.update({"RTtauMlo": f"{min(per['m'].values()):.4f}", "RTtauMhi": f"{max(per['m'].values()):.3f}",
              "RTtauRello": f"{min(per['rel'].values()):.3f}", "RTtauRelhi": f"{max(per['rel'].values()):.3f}"})
    lines = []
    for sn, key in (("gt3", "A"), ("gt4", "B")):
        if not load(f"grid_{sn}") or not load(f"{sn}h2"):
            continue
        ho = rows(sn, f"{sn}h2", f"fk_{sn}", lambda c: "")
        c_ = [r for r in ho if r["cert"]]
        M.update({f"RT{key}cert": len(c_), f"RT{key}certRight": sum(r["cright"] for r in c_)})
        lines.append(f"Certificate ($\\eps>0$, no tuning) & {len(c_)} & {sum(r['cright'] for r in c_)} & {len(c_) - sum(r['cright'] for r in c_)} \\\\"
                     if False else None)
        for k, nm in (("m", "M"), ("rel", "Rel")):
            acc = [r for r in ho if r[k] > taus[k]]
            M.update({f"RT{key}{nm}acc": len(acc), f"RT{key}{nm}right": sum(r["right"] for r in acc),
                      f"RT{key}{nm}same": sum(r["cert"] for r in acc)})
            lo = hi = None
            for t, tau in per[k].items():
                a = [r for r in ho if r[k] > tau]; w = len(a) - sum(r["right"] for r in a)
                M[f"RT{key}{nm}{t.replace(' ', '').replace('DLPFC', 'Dlpfc').capitalize()}acc"] = len(a)
                M[f"RT{key}{nm}{t.replace(' ', '').replace('DLPFC', 'Dlpfc').capitalize()}wrong"] = w
                lo = len(a) if lo is None else min(lo, len(a)); hi = len(a) if hi is None else max(hi, len(a))
            M.update({f"RT{key}{nm}oneLo": lo, f"RT{key}{nm}oneHi": hi})
            wr = [len([r for r in ho if r[k] > tau]) - sum(r["right"] for r in ho if r[k] > tau) for tau in per[k].values()]
            M.update({f"RT{key}{nm}oneWrongHi": max(wr)})
    both = []
    for sn in ("gt3", "gt4"):
        if load(f"grid_{sn}") and load(f"{sn}h2"):
            both += rows(sn, f"{sn}h2", f"fk_{sn}", lambda c: "")
    if both:
        acc = [len([r for r in both if r["m"] > tau]) for tau in per["m"].values()]
        wr = [sum(not r["right"] for r in both if r["m"] > tau) for tau in per["m"].values()]
        M.update({"RTTMoneLo": min(acc), "RTTMoneHi": max(acc), "RTTMoneWrongHi": max(wr), "RTTn": len(both)})
    # tight-bound decision
    for sn, key in (("gt3", "A"), ("gt4", "B")):
        Tt = load(f"tight_{sn}")
        if not Tt:
            continue
        ct = [c for c in Tt if Tt[c]["eps_tight"] > 0]; cl = [c for c in Tt if Tt[c]["certified_loose"]]
        M.update({f"TB{key}n": len(Tt), f"TB{key}tight": len(ct), f"TB{key}tightRight": sum(Tt[c]["tre_um"] < RIGHT for c in ct),
                  f"TB{key}loose": len(cl), f"TB{key}looseRight": sum(Tt[c]["tre_um"] < RIGHT for c in cl),
                  f"TB{key}open": sum(1 for c in Tt if not Tt[c]["complete"] and Tt[c]["eps_tight"] <= 0),
                  f"TB{key}min": f"{np.median([Tt[c]['far_s'] for c in Tt]) / 60:.0f}"})
    # grid on the second held-out set
    G, O = load("grid_gt4"), load("gt4h2")
    if G and O:
        cs = sorted(set(G) & set(O))
        M.update({"GRIDDn": len(cs), "GRIDDmiss": sum(G[c]["best"] < O[c]["best"] - 1e-3 for c in cs),
                  "GRIDDright": sum(G[c]["tre_um"] < RIGHT for c in cs)})


# ============================================================ 4. score ablations and negative controls
def ablate_neg():
    arms = [("pairing", "One-to-one pairing (FLiPAlign)"), ("rowmax", "Row maximum (many-to-one)"), ("mi", "Mutual information"),
            ("shape", "Shape only ($K=1$)"), ("shared", "Shared labels (joint clustering)")]
    sets_ = [("abl_gt1", "Breast"), ("abl_gt2", "Colon/brain/DLPFC"), ("abl_gt3", "Held-out 1"), ("abl_gt4", "Held-out 2")]
    data = [(lab, load(p)) for p, lab in sets_ if load(p)]
    lines = []
    for arm, name in arms:
        cells = []
        for lab, Dd in data:
            n = len(Dd); a = sum(r[arm]["tre_um"] < RIGHT for r in Dd.values()); b = sum(r[arm + "_known"]["tre_um"] < RIGHT for r in Dd.values())
            cells.append(f"{a} & {b}")
            k = {"Breast": "A", "Colon/brain/DLPFC": "B", "Held-out 1": "C", "Held-out 2": "D"}[lab]
            M[f"AB{k}{arm}S"] = a; M[f"AB{k}{arm}K"] = b; M[f"AB{k}n"] = n
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    if data:
        tab = ["\\begin{tabular}{l|" + "|".join("rr" for _ in data) + "}", "\\toprule",
               "Score (same grid search) & " + " & ".join(f"\\multicolumn{{2}}{{c{'|' if i < len(data) - 1 else ''}}}{{{lab} ({len(Dd)})}}" for i, (lab, Dd) in enumerate(data)) + " \\\\",
               " & " + " & ".join("srch. & known" for _ in data) + " \\\\", "\\midrule"] + lines + ["\\bottomrule", "\\end{tabular}"]
        open(f"{T}/res3_ablate.tex", "w").write("\n".join(tab) + "\n")
    for sn, key, op in (("gt3", "A", "gt3h2"), ("gt4", "B", "gt4h2")):
        N = load(f"neg_{sn}"); O = load(op)
        if not N:
            continue
        ratio = lambda r: (r["S_other"] / max(r["O_other"], 1e-9)) / (r["S_answer"] / max(r["O_answer"], 1e-9))
        cert = [r for r in N.values() if r["certified"]]
        kept = [r for r in cert if "S_other" in r and ratio(r) < 0.95]
        shiftb = [r for r in cert if r["finest_zone"][1] is not None]
        M.update({f"NG{key}n": len(N), f"NG{key}cert": len(cert), f"NG{key}kept": len(kept), f"NG{key}shift": len(shiftb),
                  f"NG{key}score": f"{np.median([r['best'] for r in N.values()]):.3f}",
                  f"NG{key}scoreTrue": f"{np.median([O[c]['best'] for c in O]):.3f}" if O else "--",
                  f"NG{key}scoreRight": f"{np.median([O[c]['best'] for c in O if O[c]['tre_um'] < RIGHT]):.3f}" if O else "--"})


def checks():
    f = glob.glob(f"{R}/chkfft_t*.json")
    if f:
        r = json.load(open(f[0]))
        M.update({"CHKcases": len(r), "CHKrot": 8 * len(r), "CHKdiff": f"{max(x['max_abs_diff_fft_vs_direct'] for x in r):.1e}".replace("e-", "\\times10^{-").replace("{-0", "{-") + "}",
                  "CHKdirect": f"{sum(x['direct_scores'] for x in r):,}".replace(",", "{,}"), "CHKviol": "none" if sum(x["bound_violations"] for x in r) == 0 else str(sum(x["bound_violations"] for x in r))})
        M["CHKdiff"] = "$" + M["CHKdiff"] + "$"
    else:
        M.update({"CHKcases": "--", "CHKrot": "--", "CHKdiff": "--", "CHKdirect": "--", "CHKviol": "--"})


def clusterings():
    """second held-out set: five re-clusterings and the agreement rule, flip known"""
    base = load("fk_gt4"); runs = [base] + [load(t) for t in ("rl4_1_12_6", "rl4_2_12_6", "rl4_3_12_6", "rl4_1_8_4", "rl4_1_16_8")]
    if not all(runs):
        return
    rights = [sum(r["tre_um"] < RIGHT for r in Rr.values()) for Rr in runs[1:]]
    cert = sum(r["certified"] for Rr in runs[1:] for r in Rr.values())
    cr = sum(r["certified"] and r["tre_um"] < RIGHT for Rr in runs[1:] for r in Rr.values())
    hold_ = sum(bool(holds(r, r)) for Rr in runs[1:] for r in Rr.values() if r["certified"])
    M.update({"RLBlo": min(rights), "RLBhi": max(rights), "RLBcert": cert, "RLBcr": cr, "RLBholds": hold_})
    for need, nm in ((4, "Four"), (6, "Six")):
        acc = right_ = 0
        for c in sorted(base):
            ce = [R_[c] for R_ in runs if c in R_ and R_[c]["certified"]]; best_ = []
            for a_ in ce:
                grp = [b_ for b_ in ce if ad(a_["angle"], b_["angle"]) <= 20 and np.hypot(*(np.array(a_["shift_units"]) - b_["shift_units"])) <= 60]
                if len(grp) > len(best_):
                    best_ = grp
            if len(best_) >= need:
                acc += 1; right_ += np.median([b_["tre_um"] for b_ in best_]) < RIGHT
        M.update({f"CONB{nm}cert": acc, f"CONB{nm}right": int(right_)})


# ============================================================ appendix tables
def datasets():
    def sz(d, tissue=None, key="tissue"):
        man = manifest(d)
        v = [m for m in man.values() if tissue is None or m.get(key) == tissue]
        if not v:
            return "--", "--", "--"
        c = [m["nA"] for m in v if m["kind"] == "cell"]; sp = [m["nA"] for m in v if m["kind"] == "spot"]
        f = lambda x: f"{np.median(x) / 1000:.0f}k" if len(x) and np.median(x) >= 1000 else (f"{np.median(x):.0f}" if len(x) else "--")
        return f(c), f(sp), f([m["nB"] for m in v])
    rows = ["\\multicolumn{7}{l}{\\emph{Development (known-answer cases)}} \\\\"]
    def r(name, plat, src, n, d, tissue=None, key="tissue"):
        a, b, c = sz(d, tissue, key)
        rows.append(f"{name} & {plat} & {src} & {n} & {a} & {b} & {c} \\\\")
    r("Human breast cancer, 2 sections", "Xenium", "\\cite{janesick2023xenium}", 24, "gt1")
    r("Mouse brain, 3 coronal sections", "Xenium", "10x Genomics", 24, "gt2", "mouse brain")
    r("Human colon, 3 blocks (16~$\\mu$m bins)", "Visium HD", "\\cite{oliveira2025hd}", 24, "gt2", "colon")
    r("Human DLPFC, 12 sections", "Visium", "\\cite{maynard2021dlpfc}", 24, "gt2", "DLPFC")
    r("Mouse brain overlap sweep, 5 sections", "Xenium", "10x Genomics", 60, "gt_ov80")
    rows += ["\\midrule", "\\multicolumn{7}{l}{\\emph{First held-out set}} \\\\"]
    for t, nm in (("mouse brain (Prime 5K)", "Mouse brain, 5{,}000-gene panel"), ("mouse brain (V1)", "Mouse brain, coronal"),
                  ("human kidney", "Human kidney"), ("human lymph node", "Human lymph node"), ("human liver", "Human liver"),
                  ("human heart", "Human heart")):
        r(nm, "Xenium", "10x Genomics", 8, "gt3", t)
    rows += ["\\midrule", "\\multicolumn{7}{l}{\\emph{Second held-out set}} \\\\"]
    for t, nm in (("human tonsil", "Human tonsil"), ("human pancreas", "Human pancreas"), ("human lung cancer", "Human lung cancer"),
                  ("human skin", "Human skin"), ("mouse pup", "Mouse pup, whole body"), ("human brain", "Human brain")):
        r(nm, "Xenium", "10x Genomics", 8, "gt4", t)
    rows += ["\\midrule", "\\multicolumn{7}{l}{\\emph{Real pairs}} \\\\",
             "Breast, Xenium Rep1 $\\to$ Rep2 (3 landmarks) & Xenium & \\cite{janesick2023xenium,clifton2023stalign} & 1 & 167k & -- & 118k \\\\",
             "Breast, Visium $\\to$ Xenium Rep1, Rep2 & Visium, Xenium & \\cite{janesick2023xenium} & 2 & -- & 5k & 167k, 118k \\\\",
             "Colon, Visium $\\to$ Visium HD, 3 blocks & Visium, Visium HD & \\cite{oliveira2025hd} & 3 & -- & spots & bins \\\\",
             "Mouse brain, 3 serial sections, whole & Xenium & 10x Genomics & 3 & cells & -- & cells \\\\",
             "Human DLPFC, same-donor pairs & Visium & \\cite{maynard2021dlpfc} & 18 & -- & spots & spots \\\\"]
    a, b, c = sz("abc"); rows.append(f"Mouse brain, neighbouring sections, whole & MERFISH & \\cite{{yao2023abc}} & 20 & {a} & -- & {c} \\\\")
    a, b, c = sz("abc_ov70"); rows.append(f"Mouse brain, neighbouring sections, 70\\% overlap & MERFISH & \\cite{{yao2023abc}} & 20 & {a} & -- & {c} \\\\")
    tab = ["\\begin{table}[H]", "\\centering\\scriptsize", "\\begin{tabular}{lllrrrr}", "\\toprule",
           " & & & & \\multicolumn{2}{c}{Points of $A$ (median)} & Points \\\\",
           "Dataset & Platform & Source & Cases & as cells & as spots & of $B$ \\\\", "\\midrule"] + rows + \
          ["\\bottomrule", "\\end{tabular}",
           "\\caption{Datasets. Known-answer cases split one section into two halves, so $A$ and $B$ have about half of its cells each before cropping.}",
           "\\label{tab:data}", "\\end{table}"]
    open(f"{T}/res3_datasets.tex", "w").write("\n".join(tab) + "\n")
    M["XXfloor"] = 18


def tools_appendix(cols):
    """settings, run time per case, and the arms table"""
    secs = collections.defaultdict(list)
    for p in ("gt3base", "gt3spateo", "gt3cast", "gt3stalign", "gt4base", "gt4spateo", "gt4cast", "gt4stalign"):
        for f in glob.glob(f"{R}/{p}_t*.json"):
            for r in json.load(open(f)):
                if "sec" in r and r.get("fit", "plain") == "plain":
                    secs[r["method"].split("_")[0]].append(r["sec"])
    rt = lambda t: f"{np.median(secs[t]) / 60:.1f}" if secs.get(t) else "--"
    O, K = load("gt3h2"), load("fk_gt3")
    ours_t = f"{np.median([r['total_s'] for r in O.values()]) / 60:.0f}" if O else "--"
    M["RTours"] = ours_t
    rows = [
        f"PASTE & 4{{,}}000 points per side; $\\alpha\\in\\{{0.01,0.1,0.5\\}}$; KL cost & no & {rt('paste')} \\\\",
        f"PASTE2 & 4{{,}}000 points; overlap share $s\\in\\{{0.3,0.5,0.7\\}}$; $\\alpha=0.1$; KL cost & no & {rt('paste2')} \\\\",
        f"SLAT & 4{{,}}000 points; $k\\in\\{{10,20,40\\}}$ neighbours; 6 epochs & no$^*$ & {rt('slat')} \\\\",
        f"CPD & 2{{,}}000 points; rigid; outlier weight $w\\in\\{{0,0.2,0.5\\}}$; 8 starts (4 angles $\\times$ 2 flips), best by its own fit & yes & {rt('cpd')} \\\\",
        f"spateo & 4{{,}}000 points; modes SN-S, SN-N, and SN-S with cosine cost; 200 iterations & yes & {rt('spateo')} \\\\",
        f"CAST & 4{{,}}000 points; rigid, affine, and affine + B-spline; 500 iterations & yes & {rt('cast')} \\\\",
        f"STalign & all points as images of 30~$\\mu$m pixels, one channel per cluster; 800 iterations; 16 starts (8 angles $\\times$ 2 flips), best by density overlap; "
        f"matching weight $\\sigma_M=1.5$ (tutorial) and $\\sigma_M=0.05$ with larger steps (best of 13 settings on six near-true starts, chosen with the truth) & yes & {rt('stalign')} \\\\",
        f"FLiPAlign & all points; one setting (Table~\\ref{{tab:const}}); search and decision & -- & {ours_t} \\\\"]
    tab = ["\\begin{table}[H]", "\\centering\\scriptsize", "\\begin{tabular}{lp{0.62\\linewidth}cr}", "\\toprule",
           "Tool & Settings tried & Uses the start & Min.\\ per case \\\\", "\\midrule"] + rows + ["\\bottomrule", "\\end{tabular}",
           "\\caption{Published tools as run here. Minutes per case: median on held-out cases for one setting (4 cores; CAST 8; FLiPAlign 16). "
           "``Uses the start'': PASTE, PASTE2 and SLAT work from distances inside each section and from expression, so their answer does not depend on how $A$ is placed at the start; "
           "for PASTE and PASTE2 the answers were identical to the last digit whatever start was handed over. $^*$SLAT is trained from random weights, so its answers vary a little from run to run (0 or 1 right).}", "\\label{tab:toolset}", "\\end{table}"]
    txt = ("\\noindent Every tool was given the same 300 most variable genes on both sides (all panel genes for breast), which FLiPAlign never sees. "
           "Tools that return matches or moved coordinates were turned into a rigid placement by a plain fit and by a trimmed fit (refit on the best-matching half, ten rounds), since partial overlap pulls a plain fit toward the centres. "
           "For each tool and each set we report the single combination of setting and fit with the most right answers, chosen with the truth, which favours the tools. "
           "PASTE2 was not told the true overlap; its three values bracket the range of the cases. "
           "Four starts were used. \\emph{As given}: the moved section as it is. \\emph{Flip}: the section is mirrored first when the truth is mirrored. "
           "\\emph{Flip and angle}: the section is also turned to within $\\pm20^\\circ$ of the truth (uniform error), with its centre on $B$'s centre. "
           "\\emph{Flip, angle and position}: its centre is also put within $\\pm500$~$\\mu$m of the true position in each direction. "
           "In the last two arms STalign and CPD run from that single start.\n\n")
    body = open(f"{T}/res3_tools.tex.body").read() if os.path.exists(f"{T}/res3_tools.tex.body") else ""
    arms = ["\\begin{table}[H]", "\\centering\\scriptsize\\setlength{\\tabcolsep}{2.5pt}", body.strip(),
            "\\caption{Right answers of each tool under three starts (as given; true flip handed over; true flip and angle within $20^\\circ$ handed over). "
            "FLiPAlign row: flip searched under ``as given'', flip known under ``flip''.}", "\\label{tab:arms}", "\\end{table}"]
    open(f"{T}/res3_tools.tex", "w").write(txt + "\n".join(tab) + "\n\n" + "\n".join(arms) + "\n")


def appendix_tables():
    out = []
    def table(body, cap, lab, pre=""):
        out.append("\n".join(["\\begin{table}[H]", "\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}", pre + body.strip(),
                              f"\\caption{{{cap}}}", f"\\label{{{lab}}}", "\\end{table}"]))
    A = json.load(open(f"{R}/analysis_r5.json")) if os.path.exists(f"{R}/analysis_r5.json") else {}
    # ---- size of the score, the gap and the margin
    if "A" in A:
        f3 = lambda v: "--" if v is None else f"{v[0]:.3f} ({v[1]:.3f}--{v[2]:.3f})"
        rows = [f"{k[0].upper() + k[1:]} & {f3(v['S'])} & {f3(v['g'])} & {100 * v['g_share']:.0f}\\% & {f3(v['eps_first'])} & {f3(v['S_fine'])} & {f3(v['g_fine'])} \\\\"
                for k, v in A["A"].items()]
        table("\n".join(["\\begin{tabular}{lrrrrrr}", "\\toprule",
                         " & \\multicolumn{4}{c}{Coarse stage} & \\multicolumn{2}{c}{Fine stage} \\\\",
                         "Cases & $S(\\hat T)$ & gap $g$ & $g/S$ & margin $\\eps$, first zone & $S_2(\\hat T_2)$ & gap $g_2$ \\\\", "\\midrule"] + rows +
                        ["\\bottomrule", "\\end{tabular}"]),
              "Size of the best score, of the proven gap and of the proven margin: median (smallest--largest). The gap is set by the bound slack at the grid step; "
              "with shift splitting it falls towards $\\tau$.", "tab:gaps")
    # ---- error thresholds
    if "D" in A:
        rows = [f"{k.split(' (')[0].replace('paste2', 'PASTE2').replace('paste', 'PASTE').replace('slat', 'SLAT').replace('cpd', 'CPD').replace('cast', 'CAST').replace('stalign', 'STalign')} & "
                + " & ".join(str(x) for x in v) + " \\\\" for k, v in A["D"].items()]
        table("\n".join(["\\begin{tabular}{lrrrr}", "\\toprule", "Method & $<20~\\mu$m & $<50~\\mu$m & $<100~\\mu$m & $<200~\\mu$m \\\\", "\\midrule"] + rows +
                        ["\\bottomrule", "\\end{tabular}"]),
              "First held-out set (48 cases): number of answers within four error limits. Tools: best setting, true flip handed over.", "tab:thresh")
    if "E" in A:
        nm = {"paste": "PASTE", "paste2": "PASTE2", "slat": "SLAT", "cpd": "CPD", "spateo": "spateo", "cast": "CAST", "stalign": "STalign"}
        rows = [f"{nm.get(k, k)} & " + " & ".join(f"{a}/{b}" for a, b in v) + " \\\\" for k, v in A["E"].items()]
        table("\n".join(["\\begin{tabular}{lrrr}", "\\toprule", "Method & turn $<30^\\circ$ & $30$--$90^\\circ$ & $>90^\\circ$ \\\\", "\\midrule"] + rows +
                        ["\\bottomrule", "\\end{tabular}"]),
              "Right answers by size of the applied turn, development and first held-out sets together (144 cases; tools with the true flip handed over). "
              "The tools do not fail mainly because of large turns.", "tab:turn")
    if "B" in A:
        f3 = lambda v: "--" if v is None else f"{v[0]:.3f} ({v[1]:.3f}--{v[2]:.3f})"
        rows = [f"{k[0].upper() + k[1:]} & {v['n_right']} & {f3(v['right'])} & {v['n_wrong']} & {f3(v['wrong'])} & {v['eps>0.02'][0]} / {v['eps>0.02'][1]} & {v['eps>0.05'][0]} / {v['eps>0.05'][1]} \\\\"
                for k, v in A["B"].items()]
        table("\n".join(["\\begin{tabular}{lrrrrrr}", "\\toprule",
                         " & \\multicolumn{2}{c}{Certified, right} & \\multicolumn{2}{c}{Certified, wrong} & \\multicolumn{2}{c}{Right / wrong left with} \\\\",
                         "Cases & $n$ & $\\eps$ & $n$ & $\\eps$ & $\\eps>0.02$ & $\\eps>0.05$ \\\\", "\\midrule"] + rows + ["\\bottomrule", "\\end{tabular}"]),
              "Proven margin of certified answers that were right and wrong (first zone; median, smallest--largest), and what a required minimum margin would leave. "
              "With the flip searched, wrong certificates are mirror placements with large margins, so no cut on $\\eps$ removes them; with the flip known a cut at 0.02 leaves one.",
              "tab:epsrw")
    for f_, cap, lab in (("res3_heldout.tex", None, None),):
        pass
    for f_, cap, lab in (("res3_ablate.tex", "Score ablations. The search is the same for every row (a 1$^\\circ$ grid over both flips with Fourier scoring of all shifts); only the score changes. "
                          "Entries: right answers with the flip searched and with the flip known.", "tab:ablfull"),
                         ("res3_downstream.tex", "Tools started from FLiPAlign's placement (+F). Alone: the tool as given. +F: right answers of the rigid fit to the tool's output. "
                          "+F bent: right answers when the tool's own bent placement is scored directly. FLiPAlign column: right answers of FLiPAlign itself.", "tab:downfull")):
        if os.path.exists(f"{T}/{f_}"):
            table(open(f"{T}/{f_}").read(), cap, lab, "\\setlength{\\tabcolsep}{2.2pt}")
    # ---- certificates: certified, holds, certified and right, on both held-out sets
    rows = []
    for sn, lab in (("gt3", "First held-out set"), ("gt4", "Second held-out set")):
        O, K = load(f"{sn}h2"), load(f"fk_{sn}")
        if not O or not K:
            continue
        kp = selfcheck(f"fc_{sn}"); cs = sorted(O)
        for nm, D_, keep in (("flip searched, no self-check", O, None), ("flip searched, self-check", O, kp), ("flip known", K, None)):
            st_ = cert_stats(D_, cs, keep, K=K)
            sb_ = sum(1 for c in cs if D_[c]["certified"] and (keep is None or keep(c)) and D_[c]["finest_zone"][1] is not None)
            rows.append(f"{lab} & {nm} & {st_['right']} & {st_['cert']} & {st_['holds']} & {st_['cr']} & {cp(st_['cr'], st_['cert'])} & {sb_} \\\\")
    if rows:
        table("\n".join(["\\begin{tabular}{llrrrrrr}", "\\toprule",
                         "Set (48 cases) & Mode & Right & Certified & Holds & C\\&R & 95\\% interval (\\%) & With shift bound \\\\", "\\midrule"] + rows +
                        ["\\bottomrule", "\\end{tabular}"]),
              "Certificates on the two held-out sets. Holds: the true placement lies in the certified zone, which the certificate implies only when the true placement scores within $\eps$ of the answer, a condition that cannot be checked on new tissue. "
              "C\\&R: the certified answer is right (error under 200~$\\mu$m). A certificate whose zone leaves the shift free can hold while its answer is far off, "
              "so the last column counts certificates that bound the shift. The interval is for the share of certified answers that are right.", "tab:holds")
    for f_, cap, lab in (("res2_confusion.tex", "Certified and refused answers against right and wrong, on the first held-out set and on the 204 cases that existed before the second "
                          "held-out set (96 development, 48 first held-out, 60 overlap sweep), with the number of certificates that bound the shift and that reached a proven fine zone.", "tab:conf"),
                         ("res2_brain.tex", "Mouse brain. Top: known-answer cases from five Xenium datasets at three overlap levels. Bottom: real neighbouring Allen MERFISH sections "
                          "(right: within 100~$\\mu$m of the best possible rigid placement). Flip-searched counts are with the mirror self-check; without it the 60\\% row has "
                          "11 certificates of which 4 hold, and the Allen 70\\% row 6 of which 2 hold.", "tab:brain"),
                         ("res3_real.tex", "Real pairs with landmarks or an outside check. Landmark error: median over the three hand-placed landmarks (their own rigid fit leaves 18~$\\mu$m). "
                          "Gene score: never used by FLiPAlign. The three colon pairs (Visium against Visium HD) had gene scores of 0.67, 0.36 and 0.52, and 0.68, 0.37 and 0.53 after the fine stage; "
                          "no tool was run on them.", "tab:real")):
        if os.path.exists(f"{T}/{f_}"):
            table(open(f"{T}/{f_}").read(), cap, lab)
    # ---- right answers by overlap band, both held-out sets
    band = lambda o: 0 if o < 0.5 else 1 if o < 0.7 else 2
    cnt = collections.defaultdict(lambda: [[0, 0], [0, 0], [0, 0]])
    for sn in ("gt3", "gt4"):
        man_ = manifest(sn); O_, K_ = load(f"{sn}h2"), load(f"fk_{sn}")
        if not man_ or not O_:
            continue
        cs_ = sorted(man_)
        base_ = {"gt3": ["gt3base", "gt3base3", "gt3slat", "gt3spateo", "gt3cast", "gt3stalign", "rc_gt3stalign"], "gt4": ["gt4base", "gt4spateo", "gt4cast", "gt4stalign", "rc_gt4stalign"]}[sn]
        tb_ = tool_best(base_, cs_, GOOD, [f"{sn}fo_base", f"{sn}fo_spateo", f"{sn}fo_cast"])
        for c in cs_:
            b_ = band(man_[c]["overlap"])
            for nm, D_ in (("FLiPAlign, flip searched", O_), ("FLiPAlign, flip known", K_)):
                cnt[nm][b_][0] += D_[c]["tre_um"] < RIGHT; cnt[nm][b_][1] += 1
            for t_, n_ in (("spateo", "spateo, flip handed over"), ("paste2", "PASTE2"), ("cast", "CAST, flip handed over")):
                if tb_[t_] and c in tb_[t_][2]:
                    cnt[n_][b_][0] += tb_[t_][2][c] < RIGHT; cnt[n_][b_][1] += 1
    if cnt:
        rows = [f"{k} & " + " & ".join(f"{a_} of {b_}" for a_, b_ in v) + " \\\\" for k, v in cnt.items()]
        table("\n".join(["\\begin{tabular}{lrrr}", "\\toprule", "Method & under 50\\% of A on B & 50--70\\% & over 70\\% \\\\", "\\midrule"] + rows + ["\\bottomrule", "\\end{tabular}"]),
              "Right answers by overlap, both held-out sets (96 cases). Below one half no method works; above it FLiPAlign leads in both bands.", "tab:overlap")
    rows = []; sec_all = [0, 0]
    for sn, lab in (("gt3", "First"), ("gt4", "Second")):
        man_ = manifest(sn); O_, K_ = load(f"{sn}h2"), load(f"fk_{sn}")
        if not man_ or not O_:
            continue
        kp_ = selfcheck(f"fc_{sn}")
        for t in dict.fromkeys(man_[c]["tissue"] for c in sorted(man_)):
            cs_ = [c for c in man_ if man_[c]["tissue"] == t]
            acc = [c for c in cs_ if O_[c]["certified"] and kp_(c)]; ck = [c for c in cs_ if K_[c]["certified"]]
            rows.append(f"{lab} & {t[0].upper() + t[1:]} & {len(acc)} & {sum(O_[c]['tre_um'] < RIGHT for c in acc)} & {len(ck)} & {sum(K_[c]['tre_um'] < RIGHT for c in ck)} \\\\")
            if acc:
                sec_all[1] += 1; sec_all[0] += all(O_[c]["tre_um"] < RIGHT for c in acc)
    if rows:
        M["SECacc"] = sec_all[1]; M["SECallRight"] = sec_all[0]; M["SECci"] = cp(sec_all[0], sec_all[1])
        table("\n".join(["\\begin{tabular}{llrrrr}", "\\toprule", " & & \\multicolumn{2}{c}{Flip searched, self-check} & \\multicolumn{2}{c}{Flip known} \\\\",
                         "Set & Section (8 cases) & accepted & right & certified & right \\\\", "\\midrule"] + rows + ["\\bottomrule", "\\end{tabular}"]),
              f"Accepted answers by section. The eight cases of a section share its tissue and are not independent. With the self-check, accepted answers come from {sec_all[1]} of the 12 sections, "
              f"and in all {sec_all[0]} every accepted answer was right (exact 95\\% interval for {sec_all[0]} of {sec_all[1]} sections: {cp(sec_all[0], sec_all[1])}\\%). "
              "The intervals in the main text count cases and are narrower than this. On the second set a paired test of FLiPAlign (flip searched) against the best tool "
              f"with the flip handed over gives {M.get('PAIRours', '--')} cases where only FLiPAlign is right and {M.get('PAIRtool', '--')} where only the tool is (exact $p={M.get('PAIRp', '--')}$).", "tab:sections")
    # ---- overlap sweep on mouse brain: every method at 100 / 80 / 60 % overlap
    rows = collections.OrderedDict(); ok_sw = True
    for ov in (100, 80, 60):
        man_ = manifest(f"gt_ov{ov}"); O_, K_ = load(f"gt_ov{ov}"), load(f"fk_ov{ov}")
        cs_ = sorted(O_)
        rows.setdefault("FLiPAlign, flip searched", []).append(str(sum(O_[c]["tre_um"] < RIGHT for c in cs_)))
        rows.setdefault("FLiPAlign, flip known", []).append(str(sum(K_[c]["tre_um"] < RIGHT for c in cs_)))
        pats = [f"sw_ov{ov}base", f"sw_ov{ov}spateo", f"sw_ov{ov}cast"]
        g0 = tool_best(pats, cs_, GOOD); g1 = tool_best(pats, cs_, GOOD, [f"sw_fo_ov{ov}base", f"sw_fo_ov{ov}spateo", f"sw_fo_ov{ov}cast"])
        for tool, name in TOOLS:
            if tool == "stalign":
                continue
            rows.setdefault(name, []).append(f"{'--' if g0.get(tool) is None else g0[tool][0]} / {'--' if g1.get(tool) is None else g1[tool][0]}")
            ok_sw = ok_sw and g0.get(tool) is not None
            M[f"SW{tool.replace('2', 'two')}{ {100: 'F', 80: 'E', 60: 'S'}[ov] }"] = "--" if g1.get(tool) is None else g1[tool][0]
        bt_ = [v[0] for v in g1.values() if v]
        M[f"SWbest{ {100: 'F', 80: 'E', 60: 'S'}[ov] }"] = max(bt_) if bt_ else "--"
    if any(x != "-- / --" for v in rows.values() for x in v):
        table("\n".join(["\\begin{tabular}{lrrr}", "\\toprule", "Method & 100\\% overlap & 80\\% & 60\\% \\\\", "\\midrule"] +
                        [f"{k} & " + " & ".join(v) + " \\\\" for k, v in rows.items()] + ["\\bottomrule", "\\end{tabular}"]),
              "Mouse brain overlap sweep (20 known-answer cases per level, five Xenium datasets): right answers of every method. Tools: as given / with the true flip handed over, "
              "best setting. With whole sections several tools place most cases right once the flip is given; FLiPAlign with the flip known keeps its level as overlap falls.", "tab:sweep")
    for f_, cap, lab in (("res3_landmarks_detail.tex", "Real pairs with clicked landmarks: number of points after removing three misclicks (pair 2, points 24 and 25; pair 5, point 37), the median misfit of the best rigid move (floor), the turn of that move, and FLiPAlign's median landmark error with its first and fine answers.", "tab:lmdetail"),
                         ("res3_landmarks_all.tex", "Median landmark error ($\\mu$m) of each tool on the five real pairs from three starts. For each tool and pair, the best of its settings and two rigid fits, chosen with the truth. $^\\ast$: right (within 100~$\\mu$m of the floor of the pair). Last column: pairs right.", "tab:lmall")):
        if os.path.exists(f"{T}/{f_}"):
            table(open(f"{T}/{f_}").read(), cap, lab)
    # ---- breast pair: tools from three starts
    xx = []
    for f in glob.glob(f"{R}/xxds_*_t*.json"):
        xx += json.load(open(f))
    if xx:
        nm = {"stalign_labels": "STalign, one channel per cluster", "stalign_density": "STalign, cell density (as in its tutorial)", "stalign_labels_c": "STalign, per cluster, $\\sigma_M=0.05$", "stalign_density_c": "STalign, density, $\\sigma_M=0.05$", "spateo_sns": "spateo",
              "cast_rigid": "CAST, rigid", "cast_full": "CAST, with B-spline", "cpd_1": "CPD"}
        rows = []
        for meth in nm:
            cells = []
            for st_ in ("centroid", "flipalign", "landmarks"):
                r = [x for x in xx if x["method"] == meth and x["start"] == st_]
                if not r:
                    cells.append("--"); continue
                r = r[0]; k = "landmark_um_bent" if "landmark_um_bent" in r else "landmark_um_plain"
                cells.append(f"{np.median(r[k]):.0f}" if k in r else "--")
            rows.append(f"{nm[meth]} & " + " & ".join(cells) + " \\\\")
        st0 = {x["start"]: np.median(x["start_landmark_um"]) for x in xx}
        rows = [f"Start itself & {st0.get('centroid', float('nan')):.0f} & {st0.get('flipalign', float('nan')):.0f} & {st0.get('landmarks', float('nan')):.0f} \\\\", "\\midrule"] + rows
        table("\n".join(["\\begin{tabular}{lrrr}", "\\toprule", " & \\multicolumn{3}{c}{Start handed to the tool} \\\\",
                         "Tool & centres matched & FLiPAlign & landmarks \\\\", "\\midrule"] + rows + ["\\bottomrule", "\\end{tabular}"]),
              "Breast Xenium pair: median landmark error ($\\mu$m) of each tool from three starts. STalign uses its own bent map and the settings of its authors' tutorial "
              "($\\sigma_M=1.5$, $\\sigma_B=1.0$, velocity step 100), except $\\sigma_A=1.5$ where the tutorial has 1.1 and 800 iterations where it has 300; "
              "the tutorial also starts from hand-placed landmarks. STalign moves little from any start, so its error is that of the start.", "tab:xxstart")
    # ---- many-to-many scores
    rows = []
    for arm, name, src in (("pairing", "One-to-one pairing (FLiPAlign)", "abl"), ("mi", "Mutual information", "abl"), ("rowmax", "Row maximum", "abl"),
                           ("chi2", "$\\sum_{kl}C_{kl}^2/(p_kq_l)$", "abl2"), ("pos", "$\\sum_{kl}\\max(C_{kl},0)$", "abl2"), ("abs", "$\\sum_{kl}|C_{kl}|$", "abl2")):
        cells = []
        for p in ("gt1", "gt2", "gt3"):
            Dd = load(f"{src}_{p}")
            cells.append(f"{sum(r[arm]['tre_um'] < RIGHT for r in Dd.values())} & {sum(r[arm + '_known']['tre_um'] < RIGHT for r in Dd.values())}" if Dd else "-- & --")
        rows.append(f"{name} & " + " & ".join(cells) + " \\\\")
    table("\n".join(["\\begin{tabular}{l|rr|rr|rr}", "\\toprule",
                     "Score & \\multicolumn{2}{c|}{Breast (24)} & \\multicolumn{2}{c|}{Colon/brain/DLPFC (72)} & \\multicolumn{2}{c}{Held-out 1 (48)} \\\\",
                     " & srch. & known & srch. & known & srch. & known \\\\", "\\midrule"] + rows + ["\\bottomrule", "\\end{tabular}"]),
          "Many-to-many variants of the score, built from the same label table $C(s)$ and searched on the same $1^\\circ$ grid. They were tried after mutual information "
          "looked strong on the first cases, and judged on the development and first held-out sets only. All are worse than the one-to-one pairing.", "tab:m2m")
    # ---- margin rule by tuning tissue
    m2 = {c: m_["tissue"] for c, m_ in manifest("gt2").items()}
    def rows_(setn, op, tissue):
        O_, G_ = load_s(op), load(f"grid_{setn}"); o_ = []
        for c in sorted(set(O_) & set(G_)):
            mm, rel, gtre = gm(G_[c]); o_.append(dict(t=tissue(c), m=mm, rel=rel, right=gtre < RIGHT, cert=bool(O_[c]["certified"]), cright=O_[c]["tre_um"] < RIGHT))
        return o_
    dev = rows_("gt1", "gt1h2", lambda c: "breast") + rows_("gt2", "gt2h2", lambda c: m2[c])
    hos = [rows_(sn, f"{sn}h2", lambda c: "") for sn in ("gt3", "gt4") if load(f"grid_{sn}") and load(f"{sn}h2")]
    bt = lambda S: max(sorted({r["m"] for r in S}), key=lambda x: sum((r["m"] > x) == r["right"] for r in S))
    rows = ["Certificate ($\\eps>0$) & -- & " + " & ".join(f"{sum(r['cert'] for r in h)} & {sum(r['cert'] and r['cright'] for r in h)} & {sum(r['cert'] and not r['cright'] for r in h)}" for h in hos) + " \\\\", "\\midrule"]
    for lab, S in [("all 96 development cases", dev)] + [(t, [r for r in dev if r["t"] == t]) for t in ("breast", "colon", "mouse brain", "DLPFC")]:
        tau = bt(S)
        rows.append(f"Rule tuned on {lab} & {tau:.4f} & " + " & ".join(f"{sum(r['m'] > tau for r in h)} & {sum(r['m'] > tau and r['right'] for r in h)} & {sum(r['m'] > tau and not r['right'] for r in h)}" for h in hos) + " \\\\")
    btr = max(sorted({r["rel"] for r in dev}), key=lambda x: sum((r["rel"] > x) == r["right"] for r in dev))
    rows.append("\\midrule")
    rows.append(f"Relative margin, tuned on all 96 development cases & {btr:.3f} & " + " & ".join(f"{sum(r['rel'] > btr for r in h)} & {sum(r['rel'] > btr and r['right'] for r in h)} & {sum(r['rel'] > btr and not r['right'] for r in h)}" for h in hos) + " \\\\")
    table("\n".join(["\\begin{tabular}{lr|rrr|rrr}", "\\toprule",
                     " & & \\multicolumn{3}{c|}{First held-out set} & \\multicolumn{3}{c}{Second held-out set} \\\\",
                     "Decision & Threshold & accepted & right & wrong & accepted & right & wrong \\\\", "\\midrule"] + rows + ["\\bottomrule", "\\end{tabular}"]),
          "The certificate against an unproven margin rule (accept when the best score on a $1^\\circ$ grid beats every score more than $20^\\circ$ away or flipped by more than a threshold). "
          "The threshold is the one that best separates right from wrong answers on the tuning cases (ties go to the smallest threshold). Flip searched, no self-check, for both. "
          "Last row: the same rule on the margin divided by the best score.", "tab:rule")
    # ---- negative controls, tight bound, window, clusterings: one table of counts
    rows = []
    for sn, lab in (("gt3", "First held-out set"), ("gt4", "Second held-out set")):
        N = load(f"neg_{sn}")
        if not N:
            continue
        ratio = lambda r: (r["S_other"] / max(r["O_other"], 1e-9)) / (r["S_answer"] / max(r["O_answer"], 1e-9))
        for nm, sel in (("different tissues", [r for r in N.values() if r["A_tissue"].split(" (")[0] != r["B_tissue"].split(" (")[0]]),
                        ("same organ, different animal", [r for r in N.values() if r["A_tissue"].split(" (")[0] == r["B_tissue"].split(" (")[0]])):
            if not sel:
                continue
            ce = [r for r in sel if r["certified"]]
            rows.append(f"{lab} & {nm} & {len(sel)} & {len(sel) - len(ce)} & {len(ce)} & {sum(1 for r in ce if ratio(r) < 0.95)} & "
                        f"{sum(1 for r in ce if r['finest_zone'][1] is not None)} & {np.median([r['best'] for r in sel]):.3f} \\\\")
    if rows:
        table("\n".join(["\\begin{tabular}{llrrrrrr}", "\\toprule",
                         " & & & & & \\multicolumn{2}{c}{of the certified} & Median \\\\",
                         "Sections from & Pair & $n$ & Refused & Certified & pass self-check & bound the shift & score \\\\", "\\midrule"] + rows +
                        ["\\bottomrule", "\\end{tabular}"]),
              "Pairs with no true alignment: $A$ of one case with $B$ of a case from another dataset, default mode. The eight brain pairs of the first set join two "
              "different mouse brains, so a placement may exist; they are listed apart. Median score of true pairs: "
              f"{M.get('NGAscoreTrue', '--')} and {M.get('NGBscoreTrue', '--')}.", "tab:neg")
    Tt = load("tight_gt3"); O3 = load("gt3h2")
    if Tt:
        kp = selfcheck("fc_gt3"); rows = []
        for nm, sel in (("Bound at the grid step (the method)", [c for c in Tt if Tt[c]["certified_loose"]]), ("Tight bound (shift splitting)", [c for c in Tt if Tt[c]["eps_tight"] > 0])):
            rows.append(f"{nm} & {len(sel)} & {sum(Tt[c]['tre_um'] < RIGHT for c in sel)} & {sum(Tt[c]['tre_um'] >= RIGHT for c in sel)} & "
                        f"{sum(kp(c) for c in sel)} & {sum(kp(c) and Tt[c]['tre_um'] < RIGHT for c in sel)} \\\\")
        table("\n".join(["\\begin{tabular}{lrrr|rr}", "\\toprule", " & & & & \\multicolumn{2}{c}{with self-check} \\\\",
                         "Decision for the zone ($20^\\circ$, any shift) & certified & right & wrong & certified & right \\\\", "\\midrule"] + rows + ["\\bottomrule", "\\end{tabular}"]),
              f"First held-out set, flip searched: the decision with the bound at the grid step and with the tight bound, where shift cells are split down to an eighth of the grid step. "
              f"The tight search was stopped at 90 minutes in {sum(1 for c in Tt if not Tt[c]['complete'] and Tt[c]['eps_tight'] <= 0)} cases, counted as not certified. "
              "A smaller tolerance accepts smaller margins, and more of those answers are wrong.", "tab:tight")
    W, WK = load("w7_gt4h2"), load("w7_fk_gt4")
    if W and WK and "WWn" in M:
        O4, K4 = load("gt4h2"), load("fk_gt4"); man4 = manifest("gt4"); rows = []
        for t in ("human tonsil", "mouse pup"):
            cs = [c for c in sorted(set(W) & set(WK)) if man4[c]["tissue"] == t]
            rows.append(f"{t[0].upper() + t[1:]} & {len(cs)} & {sum(O4[c]['tre_um'] < RIGHT for c in cs)} & {sum(K4[c]['tre_um'] < RIGHT for c in cs)} & "
                        f"{sum(W[c]['tre_um'] < RIGHT for c in cs)} & {sum(WK[c]['tre_um'] < RIGHT for c in cs)} & "
                        f"{sum(WK[c]['certified'] for c in cs)} & {sum(WK[c]['certified'] and WK[c]['tre_um'] < RIGHT for c in cs)} \\\\")
        table("\n".join(["\\begin{tabular}{lr|rr|rr|rr}", "\\toprule",
                         " & & \\multicolumn{2}{c|}{Window $\\pm2.8$ mm} & \\multicolumn{2}{c|}{Window $\\pm6.6$ mm} & \\multicolumn{2}{c}{$\\pm6.6$ mm, flip known} \\\\",
                         "Tissue & $n$ & searched & known & searched & known & certified & C\\&R \\\\", "\\midrule"] + rows + ["\\bottomrule", "\\end{tabular}"]),
              "Follow-up on the two largest tissues of the second held-out set, made after its results were seen: right answers with the fixed window and with a window of 700 units. "
              "This is not part of the frozen protocol.", "tab:window")
    rows = []
    for sn, lab, tags, basep in (("gt3", "First held-out set", ("rl_1_12_6", "rl_2_12_6", "rl_3_12_6", "rl_1_8_4", "rl_1_16_8"), "fk_gt3"),
                                 ("gt4", "Second held-out set", ("rl4_1_12_6", "rl4_2_12_6", "rl4_3_12_6", "rl4_1_8_4", "rl4_1_16_8"), "fk_gt4")):
        runs = [load(basep)] + [load(t) for t in tags]
        if not all(runs):
            continue
        cells = [f"{sum(r['tre_um'] < RIGHT for r in Rr.values())} / {sum(r['certified'] for r in Rr.values())} / {sum(r['certified'] and r['tre_um'] < RIGHT for r in Rr.values())}" for Rr in runs]
        cons = []
        for need in (4, 6):
            acc = rg = 0
            for c in sorted(runs[0]):
                ce = [R_[c] for R_ in runs if c in R_ and R_[c]["certified"]]; best_ = []
                for a_ in ce:
                    grp = [b_ for b_ in ce if ad(a_["angle"], b_["angle"]) <= 20 and np.hypot(*(np.array(a_["shift_units"]) - b_["shift_units"])) <= 60]
                    if len(grp) > len(best_):
                        best_ = grp
                if len(best_) >= need:
                    acc += 1; rg += np.median([b_["tre_um"] for b_ in best_]) < RIGHT
            cons.append(f"{acc} / {int(rg)}")
        rows.append(f"{lab} & " + " & ".join(cells) + " & " + " & ".join(cons) + " \\\\")
    if rows:
        table("\n".join(["\\begin{tabular}{l|rrrrrr|rr}", "\\toprule",
                         " & \\multicolumn{6}{c|}{Clustering (right / certified / C\\&R)} & \\multicolumn{2}{c}{Agreement (certified / right)} \\\\",
                         "Set & base & seed 1 & seed 2 & seed 3 & $K$=8/4 & $K$=16/8 & 4 of 6 & 6 of 6 \\\\", "\\midrule"] + rows + ["\\bottomrule", "\\end{tabular}"]),
              "Dependence on the clustering, flip known. Each side is re-clustered from the stored genes (three seeds at 12 clusters and 6 regions, then 8 and 4, then 16 and 8). "
              "Agreement: accept a case when at least that many of the six clusterings certify and their answers lie within $20^\\circ$ and 60 units; the rule was designed on the first set.", "tab:clust", "\\setlength{\\tabcolsep}{2.4pt}")
    open(f"{T}/res3_appendix.tex", "w").write("\n\n".join(out) + "\n")


def tools_main():
    """main-text table: right answers of every tool from four starts, on the two held-out sets and the
    real MERFISH pairs. -- = the tool does not use the start (same as the column to its left) or not run."""
    sets_ = [("gt3", "Held-out 1 (48)", "gt3", ["gt3base", "gt3base3", "gt3slat", "gt3spateo", "gt3cast", "gt3stalign", "rc_gt3stalign"], "gt3h2", True),
             ("gt4", "Held-out 2 (48)", "gt4", ["gt4base", "gt4spateo", "gt4cast", "gt4stalign", "rc_gt4stalign"], "gt4h2", True),
             ("abc", "MERFISH whole (20)", "abc", ["abcbase", "abcspateo", "abccast", "abcstalign", "rc_abcstalign"], "abc", False),
             ("abc70", "MERFISH 70\\% (20)", "abc_ov70", ["abc_ov70base", "abc_ov70spateo", "abc70cast", "abc70stalign", "rc_abc70stalign"], "abc_ov70", False)]
    cols = []
    for sn, lab, d, base, op, has_rs in sets_:
        man = manifest(d); cs = sorted(man)
        g = (lambda c, e, man=man: e < man[c]["rigid_floor_um"] + 100) if "rigid_floor_um" in man[cs[0]] else GOOD
        arms = [tool_best(base, cs, g), tool_best(base, cs, g, [f"{sn}fo_base", f"{sn}fo_spateo", f"{sn}fo_cast"])]
        if has_rs:
            arms.append(tool_best([f"rs_{sn}base", f"rs_{sn}spateo", f"rs_{sn}cast", f"rs_{sn}stalign", f"rc_rs_{sn}cpd", f"rc_rs_{sn}stalign"], cs, g))
        arms.append(tool_best([f"rp_{sn}base", f"rp_{sn}spateo", f"rp_{sn}cast", f"rc_rp_{sn}cpd", f"rc_rp_{sn}stalign"], cs, g))
        O, K = load(op), load(f"fk_{sn}")
        cols.append((sn, lab, arms, sum(g(c, O[c]["tre_um"]) for c in cs if c in O), sum(g(c, K[c]["tre_um"]) for c in cs if c in K), has_rs))
    f_ = lambda v: "--" if v is None else str(v[0])
    names = lambda has_rs: ["G", "F", "A", "P"] if has_rs else ["G", "F", "P"]                # codes explained in the caption
    split_ = lambda s: s.split(" ", 1) if s.startswith("MERFISH") else [s[:s.index(" (")], s[s.index(" (") + 1:]]   # two-line group title
    head0 = " & " + " & ".join(f"\\multicolumn{{{len(c[2])}}}{{c}}{{{split_(c[1])[0]}}}" for c in cols) + " \\\\"
    head1 = " & " + " & ".join(f"\\multicolumn{{{len(c[2])}}}{{c}}{{{split_(c[1])[1]}}}" for c in cols) + " \\\\"
    st, rules = 2, []
    for c in cols: rules.append(f"\\cmidrule(lr){{{st}-{st + len(c[2]) - 1}}}"); st += len(c[2])
    head2 = "Tool & " + " & ".join(" & ".join(names(c[5])) for c in cols) + " \\\\"
    lines = []
    for tool, name in TOOLS:
        lines.append(f"{name} & " + " & ".join(" & ".join(f_(a.get(tool)) for a in c[2]) for c in cols) + " \\\\")
    lines.append("\\midrule")
    lines.append("FLiPAlign & " + " & ".join(" & ".join([str(c[3]), str(c[4])] + ["--"] * (len(c[2]) - 2)) for c in cols) + " \\\\")
    tab = ["\\begin{tabular}{@{}l" + "".join("c" * len(c[2]) for c in cols) + "@{}}", "\\toprule", head0, head1, "".join(rules), head2, "\\midrule"] + lines + ["\\bottomrule", "\\end{tabular}"]
    open(f"{T}/res3_toolsmain.tex", "w").write("\n".join(tab) + "\n")
    for sn, lab, arms, o_, k_, has_rs in cols:
        key = {"gt3": "A", "gt4": "B", "abc": "M", "abc70": "P"}[sn]
        bests = [max((v[0] for v in a.values() if v), default=0) for a in arms]
        M[f"BT{key}given"] = bests[0]; M[f"BT{key}flip"] = bests[1]; M[f"BT{key}pose"] = bests[-1]
        M[f"BT{key}any"] = max(bests); M[f"BT{key}oursS"] = o_; M[f"BT{key}oursK"] = k_
        if has_rs:
            M[f"BT{key}angle"] = bests[2]
        for tool in ("spateo", "cast", "paste2", "stalign"):
            for nm, a in zip(["given", "flip"] + (["angle"] if has_rs else []) + ["pose"], arms):
                M[f"BT{key}{tool.replace('2', 'two')}{nm}"] = f_(a.get(tool))


def text_numbers():
    """numbers quoted in the running text"""
    kp = {sn: selfcheck(f"fc_{sn}") for sn in ("gt3", "gt4")}
    tot_in = collections.Counter()
    for sn, key, op in (("gt3", "A", "gt3h2"), ("gt4", "B", "gt4h2")):
        O, K = load(op), load(f"fk_{sn}")
        if not O or not K:
            continue
        out = [c for c in K if max(abs(x) for x in K[c]["true_shift_units"]) > 300]; ins = [c for c in K if c not in out]
        M[f"WIN{key}out"] = len(out); M[f"WIN{key}in"] = len(ins)
        M[f"WIN{key}outRightK"] = sum(K[c]["tre_um"] < RIGHT for c in out); M[f"WIN{key}outCertK"] = sum(K[c]["certified"] for c in out)
        M[f"WIN{key}outCertS"] = sum(O[c]["certified"] and kp[sn](c) for c in out)
        M[f"WIN{key}inRightS"] = sum(O[c]["tre_um"] < RIGHT for c in ins); M[f"WIN{key}inRightK"] = sum(K[c]["tre_um"] < RIGHT for c in ins)
        ck = [c for c in ins if K[c]["certified"]]; M[f"WIN{key}inCertK"] = len(ck); M[f"WIN{key}inCrK"] = sum(K[c]["tre_um"] < RIGHT for c in ck)
        M[f"H{key}undet"] = sum(1 for c in O if O[c]["certified"] and not kp[sn](c))
        F, FK = load(f"{sn}fine"), load(f"fkfine_{sn}")
        M[f"H{key}fineRan"] = len(FK); M[f"H{key}fineCert"] = sum(1 for r in FK.values() if r["fine_zone"] is not None)
        mv = [r["fine_zone_move_um"] for r in FK.values() if r["fine_zone"] is not None]
        M[f"H{key}fineMove"] = f"{np.median(mv):.0f}" if mv else "--"
        N = load(f"neg_{sn}")
        if N:
            ratio = lambda r: (r["S_other"] / max(r["O_other"], 1e-9)) / (r["S_answer"] / max(r["O_answer"], 1e-9))
            tn = [r for r in N.values() if r["A_tissue"].split(" (")[0] != r["B_tissue"].split(" (")[0]]
            so = [r for r in N.values() if r["A_tissue"].split(" (")[0] == r["B_tissue"].split(" (")[0]]
            ce = [r for r in tn if r["certified"]]
            M[f"NG{key}true"] = len(tn); M[f"NG{key}trueCert"] = len(ce); M[f"NG{key}trueKept"] = sum(1 for r in ce if ratio(r) < 0.95)
            M[f"NG{key}same"] = len(so); M[f"NG{key}sameCert"] = sum(r["certified"] for r in so)
            tot_in["tn"] += len(tn); tot_in["cert"] += len(ce); tot_in["kept"] += M[f"NG{key}trueKept"]
            tot_in["eps"] = max(tot_in["eps"], max([r["ladder"][0]["eps"] for r in ce] + [0]))
    M.update({"NGtrue": tot_in["tn"], "NGtrueCert": tot_in["cert"], "NGtrueKept": tot_in["kept"], "NGtrueRefused": tot_in["tn"] - tot_in["cert"],
              "NGmaxEps": f"{tot_in['eps']:.3f}"})
    Tt = load("tight_gt3")
    if Tt:
        ct = [c for c in Tt if Tt[c]["eps_tight"] > 0]
        M.update({"TBAtightSC": sum(kp["gt3"](c) for c in ct), "TBAtightSCright": sum(kp["gt3"](c) and Tt[c]["tre_um"] < RIGHT for c in ct),
                  "TBAtightWrong": len(ct) - sum(Tt[c]["tre_um"] < RIGHT for c in ct)})
    # ablation totals over the four sets
    tot = collections.Counter()
    for p in ("abl_gt1", "abl_gt2", "abl_gt3", "abl_gt4"):
        for r in load(p).values():
            tot["n"] += 1
            for arm in ("pairing", "rowmax", "mi", "shape", "shared"):
                tot[arm + "S"] += r[arm]["tre_um"] < RIGHT; tot[arm + "K"] += r[arm + "_known"]["tre_um"] < RIGHT
    for p in ("abl2_gt1", "abl2_gt2", "abl2_gt3"):
        for r in load(p).values():
            tot["m"] += 1
            for arm in ("pos", "abs", "chi2"):
                tot[arm + "S"] += r[arm]["tre_um"] < RIGHT; tot[arm + "K"] += r[arm + "_known"]["tre_um"] < RIGHT
    M.update({f"ABT{k}": v for k, v in tot.items()})
    # paired comparison on the second held-out set: FLiPAlign (flip searched) against the best tool with the flip
    from scipy.stats import binomtest
    man4 = manifest("gt4"); O4, K4 = load("gt4h2"), load("fk_gt4")
    if man4 and O4:
        cs4 = sorted(man4)
        tb4 = tool_best(["gt4base", "gt4spateo", "gt4cast", "gt4stalign", "rc_gt4stalign"], cs4, GOOD, ["gt4fo_base", "gt4fo_spateo", "gt4fo_cast"])
        bt = max((v for v in tb4.values() if v), key=lambda v: v[0])[2]
        b_ = sum(O4[c]["tre_um"] < RIGHT and bt[c] >= RIGHT for c in cs4); c_ = sum(O4[c]["tre_um"] >= RIGHT and bt[c] < RIGHT for c in cs4)
        out4 = [c for c in cs4 if max(abs(x) for x in K4[c]["true_shift_units"]) > 300]
        M.update({"PAIRp": f"{binomtest(b_, b_ + c_, 0.5).pvalue:.2f}", "PAIRours": b_, "PAIRtool": c_,
                  "PAIRspIn": sum(bt[c] < RIGHT for c in cs4 if c not in out4), "PAIRspOut": sum(bt[c] < RIGHT for c in out4)})
    # size of the tolerance lambda
    lam = list(load("lam_gt3").values()) + list(load("lam_gt4").values())
    M["LAMtext"] = ""
    if lam:
        lu = np.array([r["lambda_leaf_max"] for r in lam]); ub = np.array([r["lambda_ub"] for r in lam])
        Sb = {**{("gt3", c): r["best"] for c, r in load("gt3h2").items()}, **{("gt4", c): r["best"] for c, r in load("gt4h2").items()}}
        sc = np.array([load("gt3h2")[r["case"]]["best"] for r in load("lam_gt3").values()] + [load("gt4h2")[r["case"]]["best"] for r in load("lam_gt4").values()])
        M.update({"LAMmed": f"{np.median(lu):.3f}", "LAMlo": f"{lu.min():.3f}", "LAMhi": f"{lu.max():.3f}", "LAMubMed": f"{np.median(ub):.3f}",
                  "LAMshare": f"{100 * np.median(lu / sc):.0f}", "LAMn": len(lam)})
        M["LAMapp"] = (f"Measured directly on {len(lam)} held-out cases (24 random leaves each), the largest $E(s)$ had a median of {M['LAMmed']} "
                       f"(smallest {M['LAMlo']}, largest {M['LAMhi']}), against a median of {M['LAMubMed']} for the bound $\\tfrac32\\rho\\max_cD(c)$, "
                       f"and was about {M['LAMshare']}\\% of the best score of the case.")
        M["LAMtext"] = (f"on the held-out cases the largest $E(s)$ at sampled leaves had a median of {M['LAMmed']} "
                        f"({M['LAMlo']}--{M['LAMhi']}), about {M['LAMshare']}\\% of the best score.")
    man = manifest("abc"); Kc = load("fk_abc")
    if Kc:
        M["AMFcr"] = sum(Kc[c]["tre_um"] < man[c]["rigid_floor_um"] + 100 for c in Kc if Kc[c]["certified"])
    M["WWtext"] = ""
    # wider window follow-up on the two largest tissues of the second held-out set
    W, WK, WF = load("w7_gt4h2"), load("w7_fk_gt4"), load("w7_fc_gt4")
    if W and WK:
        def keep(c):
            r = WF.get(c)
            return True if r is None else (r["S_other"] / max(r["O_other"], 1e-9)) / (r["S_answer"] / max(r["O_answer"], 1e-9)) < 0.95
        O, K = load("gt4h2"), load("fk_gt4"); cs = sorted(set(W) & set(WK))
        cw = [c for c in cs if W[c]["certified"] and keep(c)]; ck = [c for c in cs if WK[c]["certified"]]
        M.update({"WWn": len(cs), "WWrightS": sum(W[c]["tre_um"] < RIGHT for c in cs), "WWrightK": sum(WK[c]["tre_um"] < RIGHT for c in cs),
                  "WWcertS": len(cw), "WWcrS": sum(W[c]["tre_um"] < RIGHT for c in cw), "WWcertK": len(ck), "WWcrK": sum(WK[c]["tre_um"] < RIGHT for c in ck),
                  "WWoldS": sum(O[c]["tre_um"] < RIGHT for c in cs), "WWoldK": sum(K[c]["tre_um"] < RIGHT for c in cs),
                  "WWallS": sum((W[c] if c in W else O[c])["tre_um"] < RIGHT for c in O), "WWallK": sum((WK[c] if c in WK else K[c])["tre_um"] < RIGHT for c in K)})
        M["WWtext"] = (f"In a follow-up made after seeing this, the {len(cs)} pup and tonsil cases were rerun with the window widened to "
                       f"$\\pm6.6$~mm: {M['WWrightS']} were right with the flip searched and {M['WWrightK']} with it known "
                       f"(before: {M['WWoldS']} and {M['WWoldK']}), and {M['WWcrS']} of {M['WWcertS']} accepted answers were right.")


def real_table():
    """real pairs with truth or an outside check: adds STalign (16 starts) and STalign started from FLiPAlign"""
    body = open(f"{T}/res_real.tex").read().splitlines()
    rb = json.load(open(f"{R}/real_baselines_arc.json")) if os.path.exists(f"{R}/real_baselines_arc.json") else []
    def st(case, key):
        v = [r.get(key) for r in rb if r.get("case") == case and r.get("method", "").startswith("stalign") and r.get(key) is not None]
        return v[0] if v else None
    xx = []
    for f in glob.glob(f"{R}/xxds_*_t*.json"):
        xx += json.load(open(f))
    fa = [np.median(r["landmark_um_bent"]) for r in xx if r["start"] == "flipalign" and r["method"].startswith("stalign") and "landmark_um_bent" in r]
    rows = []
    for l in body:
        if l.startswith("Xenium$\\to$Xenium"):
            v = st("XX", "landmark_error_median_um")
            rows.append(l.replace(" \\\\", f" & {v:.0f} & {min(fa):.0f} \\\\" if v is not None and fa else " & -- & -- \\\\"))
        elif l.startswith("Visium$\\to$Xenium"):
            v = st("VX1" if "Xenium 1" in l else "VX2", "gene_score")
            rows.append(l.replace(" \\\\", f" & {v:.2f} & -- \\\\" if v is not None else " & -- & -- \\\\"))
    tab = ["\\begin{tabular}{llrrrrrrr}", "\\toprule",
           " & & \\multicolumn{2}{c}{FLiPAlign} & & & & \\multicolumn{2}{c}{STalign} \\\\",
           "Real pair & Measure & coarse & + fine & PASTE2 & spateo & CAST & 16 starts & from FLiPAlign \\\\", "\\midrule"] + rows + ["\\bottomrule", "\\end{tabular}"]
    open(f"{T}/res3_real.tex", "w").write("\n".join(tab) + "\n")
    if fa:
        M["XXstFromF"] = f"{min(fa):.0f}"
    for r in xx:
        if r["method"] == "stalign_density" and "landmark_um_bent" in r:
            M[f"XXst{r['start'][:1].upper()}"] = f"{np.median(r['landmark_um_bent']):.0f}"


def landmarks_tables():
    """real pairs with clicked landmarks: main table (tools as given) and appendix table (three starts)"""
    pr = {k: json.load(open(f"{R}/lm_flip_pair{k}.json")) for k in range(1, 8) if os.path.exists(f"{R}/lm_flip_pair{k}.json")}
    if len(pr) < 7:
        return
    rows_ = []
    for f in glob.glob(f"{R}/lm_*_pair*_*.json"):
        if "lm_flip" in f:
            continue
        rows_ += [r for r in json.load(open(f)) if "landmark_err_median_um" in r]
    floor = {k: pr[k]["floor_um"] for k in pr}; tl = lambda m: m.split("_")[0]
    def best(arm, t, k):
        v = [r["landmark_err_median_um"] for r in rows_ if r["arm"] == arm and r["pair"] == k and tl(r["method"]) == t]
        return min(v) if v else None
    def cell(v, k):
        return "--" if v is None else f"{v:.0f}" + ("$^\\ast$" if v < floor[k] + 100 else "")
    names = [("paste", "PASTE"), ("paste2", "PASTE2"), ("slat", "SLAT"), ("cpd", "CPD"), ("spateo", "spateo"), ("cast", "CAST"), ("stalign", "STalign"), ("gala", "GALA")]
    cnt = {}
    for arm in ("given", "rough", "pose"):
        for t, _ in names:
            ran = any(best(arm, t, k) is not None for k in range(1, 8))      # "--" for a start the tool was not run from
            cnt[(arm, t)] = sum(1 for k in range(1, 8) if best(arm, t, k) is not None and best(arm, t, k) < floor[k] + 100) if ran else "--"
    fl = {k: pr[k]["coarse"]["median_um"] for k in pr}; fl_ok = sum(fl[k] < floor[k] + 100 for k in pr)
    M.update({"LMflipRight": fl_ok, "LMfineRight": sum(pr[k]["fine"]["median_um"] < floor[k] + 100 for k in pr if "fine" in pr[k]),
              "LMn": 7, "LMcert": sum(bool(pr[k]["flipalign_certified"]) for k in pr),
              "LMcertRight": sum(bool(pr[k]["flipalign_certified"]) and pr[k]["coarse"]["median_um"] < floor[k] + 100 for k in pr),
              "LMcertWrong": sum(bool(pr[k]["flipalign_certified"]) and pr[k]["coarse"]["median_um"] >= floor[k] + 100 for k in pr),
              "LMfloorLo": f"{min(floor.values()):.0f}", "LMfloorHi": f"{max(floor.values()):.0f}",
              "LMflipLo": f"{min(fl[k] for k in pr if fl[k] < floor[k] + 100):.0f}", "LMflipHi": f"{max(fl[k] for k in pr if fl[k] < floor[k] + 100):.0f}",
              "LMangleMax": f"{max(pr[k]['coarse']['angle_error_deg'] for k in pr if fl[k] < floor[k] + 100):.1f}",
              "LMpoints": sum(pr[k]["n_landmarks"] for k in pr)})
    for arm, nm in (("given", "G"), ("rough", "R"), ("pose", "P")):
        bt = max((cnt[(arm, t)] for t, _ in names if cnt[(arm, t)] != "--")); M[f"LMbest{nm}"] = bt
        M[f"LMbestName{nm}"] = " and ".join(n_ for t, n_ in names if cnt[(arm, t)] == bt)
        for t, n_ in names:
            M[f"LM{nm}{t.replace('2', 'two')}"] = cnt[(arm, t)]
    lines = ["Best rigid move (floor) & " + " & ".join(f"{floor[k]:.0f}" for k in range(1, 8)) + " & & & \\\\", "\\midrule"]
    for t, n_ in names:
        lines.append(f"{n_} & " + " & ".join(cell(best('given', t, k), k) for k in range(1, 8)) + f" & {cnt[('given', t)]} & {cnt[('rough', t)]} & {cnt[('pose', t)]} \\\\")
    lines += ["\\midrule", "FLiPAlign & " + " & ".join(f"{fl[k]:.0f}$^\\ast$" if fl[k] < floor[k] + 100 else f"{fl[k]:.0f}" for k in range(1, 8)) + f" & {fl_ok} & -- & -- \\\\",
              "\\quad certified & " + " & ".join("yes" if pr[k]["flipalign_certified"] else "no" for k in range(1, 8)) + " & & & \\\\"]
    tab = ["\\begin{tabular}{l|rr|rrr|rr|rrr}", "\\toprule",
           " & \\multicolumn{2}{c|}{Breast} & \\multicolumn{3}{c|}{Colon} & \\multicolumn{2}{c|}{Lung} & \\multicolumn{3}{c}{Right of 7, start} \\\\",
           " & Rep1 & Rep2 & P2CRC & P3NAT & P5NAT & HD1 & HD2 & given & angle & pose \\\\", "\\midrule"] + lines + ["\\bottomrule", "\\end{tabular}"]
    open(f"{T}/res3_landmarks.tex", "w").write("\n".join(tab) + "\n")
    # appendix: all three starts, per pair
    out = []
    for arm, ttl in (("given", "nothing handed over"), ("rough", "true angle within $20^\\circ$ handed over"), ("pose", "angle and position within 500~$\\mu$m handed over")):
        ls = [f"{n_} & " + " & ".join(cell(best(arm, t, k), k) for k in range(1, 8)) + f" & {cnt[(arm, t)]} \\\\" for t, n_ in names]
        out.append(f"\\multicolumn{{9}}{{l}}{{\\emph{{{ttl}}}}} \\\\\n" + "\n".join(ls))
    tab2 = ["\\begin{tabular}{l|rr|rrr|rr|r}", "\\toprule", " & Rep1 & Rep2 & P2CRC & P3NAT & P5NAT & HD1 & HD2 & right \\\\", "\\midrule"] + ["\n\\midrule\n".join(out)] + ["\\bottomrule", "\\end{tabular}"]
    open(f"{T}/res3_landmarks_all.tex", "w").write("\n".join(tab2) + "\n")
    # per-pair details
    det = [f"{NAMES_LM[k]} & {pr[k]['n_landmarks']} & {pr[k]['floor_um']:.0f} & {pr[k]['truth_angle']:.1f} & {pr[k]['coarse']['median_um']:.0f} & {fine_s(pr[k])} & {pr[k]['coarse']['angle_error_deg']:.1f} \\\\" for k in range(1, 8)]
    tab3 = ["\\begin{tabular}{lrrrrrr}", "\\toprule", "Pair & points & floor ($\\mu$m) & turn ($^\\circ$) & FLiPAlign ($\\mu$m) & + fine ($\\mu$m) & angle error ($^\\circ$) \\\\", "\\midrule"] + det + ["\\bottomrule", "\\end{tabular}"]
    open(f"{T}/res3_landmarks_detail.tex", "w").write("\n".join(tab3) + "\n")

fine_s = lambda r: ("%.0f" % r["fine"]["median_um"]) if "fine" in r else "--"
NAMES_LM = {1: "Breast Rep1 (Visium to Xenium)", 2: "Breast Rep2 (Visium to Xenium)", 3: "Colon P2CRC (Visium to Visium HD)", 4: "Colon P3NAT (Visium to Visium HD)", 5: "Colon P5NAT (Visium to Visium HD)", 6: "Lung HD1 (Visium HD to Xenium)", 7: "Lung HD2 (Visium HD to Xenium 5K)"}


def app_bib():
    """reference list of the appendix: the entries of main.tex that the appendix (and its tables) cite"""
    import re
    if not (os.path.exists("main.tex") and os.path.exists("appendix.tex")):
        print("skip app_bib: main.tex / appendix.tex (paper source) not present")
        return
    main = open("main.tex").read()
    items = dict(re.findall(r"\\bibitem\{([^}]+)\}(.*?)(?=\\bibitem|\\end\{thebibliography\})", main, flags=re.S))
    txt = open("appendix.tex").read() + "".join(open(f).read() for f in glob.glob(f"{T}/res3_*.tex") if not f.endswith("appbib.tex"))
    keys = []
    for grp in re.findall(r"\\cite\{([^}]+)\}", txt):
        for k in grp.split(","):
            if k.strip() in items and k.strip() not in keys:
                keys.append(k.strip())
    body = "".join(f"\\bibitem{{{k}}}{items[k].rstrip()}\n" for k in keys)
    open(f"{T}/res3_appbib.tex", "w").write("\\begin{thebibliography}{99}\n" + body + "\\end{thebibliography}\n" if keys else "")

def main():
    heldout(); cols = tools(); tools_main(); downstream(); proof(); ablate_neg(); checks(); clusterings(); text_numbers(); real_table(); landmarks_tables()
    datasets(); tools_appendix(cols); appendix_tables(); app_bib()
    safe = lambda k: "".join(ch for ch in k if ch.isalpha())
    M["WWtext"] = M.get("WWtext", ""); M["LAMtext"] = M.get("LAMtext", ""); M["LAMapp"] = M.get("LAMapp", "")
    stc = {}
    for f in glob.glob(f"{R}/xxds_stc_t*.json"):
        for r in json.load(open(f)):
            if r["method"] == "stalign_density_c" and "landmark_um_bent" in r:
                stc[r["start"]] = f"{np.median(r['landmark_um_bent']):.0f}"
    M.update({f"XXstc{k[:1].upper()}": v for k, v in stc.items()})
    M["STtext"] = (f"Used as its authors do, STalign keeps the start it is given: on the breast pair it ended {M.get('XXstC', '--')}, "
                   f"{M.get('XXstF', '--')} and {M.get('XXstL', '--')}~$\\mu$m from the landmarks when started with centres matched, from "
                   f"FLiPAlign's placement, and from the hand-placed landmarks its tutorial uses (their own rigid fit leaves {M.get('XXfloor', '--')}~$\\mu$m). "
                   f"With a stronger matching weight it moves from a poor start ({M.get('XXstC', '--')} to {stc.get('centroid', '--')}~$\\mu$m) but also "
                   f"leaves a good one ({M.get('XXstL', '--')} to {stc.get('landmarks', '--')}~$\\mu$m). So STalign needs a good start, which FLiPAlign "
                   f"supplies without hand-placed landmarks.")
    seen, lines = set(), []
    for k, v in M.items():                       # macro names keep letters only; keep the first of any clash
        if safe(k) not in seen:
            seen.add(safe(k)); lines.append(f"\\newcommand{{\\{safe(k)}}}{{{v}}}\n")
    open(f"{T}/res3_numbers.tex", "w").write("".join(lines))
    return M


if __name__ == "__main__":
    print(json.dumps(main(), indent=0, default=str))
    import fit_tables; fit_tables.main()            # keep supplement tables inside the text width
