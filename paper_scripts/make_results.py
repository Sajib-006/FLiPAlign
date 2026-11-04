"""All tables and numbers of the rewritten paper (2026-10-03), straight from ../server_results/new.
Writes tables/res_*.tex and tables/res_numbers.tex. Rerun after any new result.

Sets: GT1 (24 breast, development), GT2 (72: colon, mouse brain, DLPFC, development), GT3 (48, six new
datasets, held out, frozen in FROZEN_2026-09-30.md), overlap sweep (60 brain cases), real pairs.
"Right" = median cell error under 200 um. Baselines: best of 3 settings x 2 rigid fits, with the true
flip handed over (FLIP_ORACLE runs replace the flipped cases)."""
import collections, glob, json, os
import numpy as np

R = "../server_results/new"; T = "tables"; os.makedirs(T, exist_ok=True)
RIGHT = 200.0
TOOLS = [("paste", "PASTE"), ("paste2", "PASTE2"), ("slat", "SLAT"), ("cpd", "CPD"), ("spateo", "spateo"), ("cast", "CAST"),
         ("stalign", "STalign")]


def load(prefix):
    o = {}
    for f in glob.glob(f"{R}/{prefix}_t*.json"):
        for r in json.load(open(f)):
            o[r.get("case", r.get("pair"))] = r
    return o


def manifest(name):
    f = f"{R}/{name}/manifest.json"
    return {m["case"]: m for m in json.load(open(f))} if os.path.exists(f) else {}


def pct(k, n):
    return f"{100 * k / n:.0f}" if n else "--"


def se(k, n):
    p = k / n if n else 0
    return f"{100 * np.sqrt(p * (1 - p) / n):.0f}" if n else "--"


def baseline_best(setn, cases):
    """per tool: best (setting, fit) by share right, flipped cases taken from the flip-handed run"""
    orig = collections.defaultdict(dict); fo = collections.defaultdict(dict)
    for pat in (f"{setn}base", f"{setn}base3", f"{setn}slat", f"{setn}spateo", f"{setn}cast", f"{setn}stalign"):
        for f in glob.glob(f"{R}/{pat}_t*.json"):
            for r in json.load(open(f)):
                if "tre_um" in r:
                    orig[(r["method"], r["fit"])][r["case"]] = r["tre_um"]
    for pat in (f"{setn}fo_base", f"{setn}fo_spateo", f"{setn}fo_cast"):
        for f in glob.glob(f"{R}/{pat}_t*.json"):
            for r in json.load(open(f)):
                if "tre_um" in r:
                    fo[(r["method"], r["fit"])][r["case"]] = r["tre_um"]
    out = {}
    for tool, _ in TOOLS:
        best = None
        for (m, fit), v in orig.items():
            if m.split("_")[0] != tool:
                continue
            comb = dict(v); comb.update(fo.get((m, fit), {}))
            vals = np.array([comb[c] for c in cases if c in comb])
            if len(vals) < 0.8 * len(cases):
                continue
            k = int(np.sum(vals < RIGHT))
            if best is None or k > best[0] or (k == best[0] and np.median(vals) < best[1]):
                best = (k, float(np.median(vals)), len(vals))
        out[tool] = best
    return out


def ours_stats(O, F, cases):
    n = len(cases)
    right = sum(O[c]["tre_um"] < RIGHT for c in cases)
    cert = [c for c in cases if O[c]["certified"]]
    final = {c: (F[c]["tre2_um"] if c in F else O[c]["tre_um"]) for c in cases}
    wrong_cert = sum(final[c] >= RIGHT for c in cert)
    med_cert = float(np.median([final[c] for c in cert])) if cert else float("nan")
    return n, right, len(cert), wrong_cert, med_cert


def local_right(L, cases):
    k = 0
    for c in cases:
        if c in L:
            b = max(L[c]["starts"], key=lambda s: s["score"]); k += b["tre_um"] < RIGHT
    return k


def main():
    M = {}
    sets = {"GT1": ("gt1", {c: "breast" for c in range(24)}), "GT2": ("gt2", None), "GT3": ("gt3", None)}
    m2 = {c: m["tissue"] for c, m in manifest("gt2").items()}
    m3 = {c: m["tissue"] for c, m in manifest("gt3").items()}
    sets["GT2"] = ("gt2", m2); sets["GT3"] = ("gt3", m3)
    # ---- Table: held-out GT3 by tissue
    O3, F3 = load("gt3h2"), load("gt3fine"); L3 = load("gt3local")
    rows = []
    names = {"human lymph node": "Human lymph node", "human liver": "Human liver", "human kidney": "Human kidney",
             "human heart": "Human heart", "mouse brain (Prime 5K)": "Mouse brain (Prime 5K)",
             "mouse brain (V1)": "Mouse brain (V1)"}
    for t in names:
        cs = sorted(c for c in m3 if m3[c] == t)
        n, right, nc, wc, med = ours_stats(O3, F3, cs)
        bb = baseline_best("gt3", cs); bestk = max((v[0] for v in bb.values() if v), default=0)
        medtxt = f"{med:.0f}" if np.isfinite(med) else "--"
        rows.append(f"{names[t]} & {n} & {right} & {nc} & {wc} & {medtxt} & {local_right(L3, cs)} & {bestk} \\\\")
    cs = sorted(m3); n, right, nc, wc, med = ours_stats(O3, F3, cs)
    bb = baseline_best("gt3", cs); bestk = max((v[0] for v in bb.values() if v), default=0)
    nb = [c for c in cs if "brain" not in m3[c]]; _, rnb, ncnb, wcnb, mednb = ours_stats(O3, F3, nb)
    M.update({"HOn": n, "HOright": right, "HOcert": nc, "HOwrongcert": wc, "HOmed": f"{med:.0f}",
              "HOnonbrainCert": ncnb, "HOnonbrainWrong": wcnb, "HOnonbrainMed": f"{mednb:.0f}",
              "HOlocal": local_right(L3, cs), "HObestBase": bestk})
    tab = ["\\begin{tabular}{lrrrrrrr}", "\\toprule",
           "Tissue & $n$ & Right $\\uparrow$ & Cert. & Cert. wrong $\\downarrow$ & Err. ($\\mu$m) $\\downarrow$ & Local & Best tool \\\\",
           "\\midrule"] + rows + ["\\midrule",
           f"All held-out & {n} & {right} & {nc} & {wc} & {med:.0f} & {local_right(L3, cs)} & {bestk} \\\\",
           "\\bottomrule", "\\end{tabular}"]
    open(f"{T}/res_heldout.tex", "w").write("\n".join(tab) + "\n")
    # ---- Table: all sets x all methods (share right)
    lines = []
    for label, (setn, man) in sets.items():
        O = load(f"{setn}h2"); cs = sorted(O)
        L = load("gt1local500") if setn == "gt1" else load(f"{setn}local")
        bb = baseline_best(setn, cs)
        cells = [f"{pct(sum(O[c]['tre_um'] < RIGHT for c in cs), len(cs))} $\\pm$ {se(sum(O[c]['tre_um'] < RIGHT for c in cs), len(cs))}",
                 pct(local_right(L, cs), len(cs))]
        for tool, _ in TOOLS:
            v = bb.get(tool); cells.append(pct(v[0], v[2]) if v else "--")
        desc = {"GT1": "Breast (dev.)", "GT2": "Colon, brain, DLPFC (dev.)",
                "GT3": "Six new datasets (held out)"}[label]
        lines.append(f"{desc} & " + " & ".join(cells) + " \\\\")
    tab = ["\\begin{tabular}{lrrrrrrrr}", "\\toprule",
           "Set & FLiPAlign & Local & " + " & ".join(n for _, n in TOOLS) + " \\\\", "\\midrule"] + \
          lines + ["\\bottomrule", "\\end{tabular}"]
    open(f"{T}/res_methods.tex", "w").write("\n".join(tab) + "\n")
    # ---- overlap sweep
    for lvl in ("gt_ov100", "gt_ov80", "gt_ov60"):
        O = load(lvl); cs = sorted(O)
        n, right, nc, wc, _ = ours_stats(O, load(f"{lvl}fine"), cs)
        key = {"gt_ov100": "Full", "gt_ov80": "Eighty", "gt_ov60": "Sixty"}[lvl]
        M.update({f"OV{key}n": n, f"OV{key}right": right, f"OV{key}cert": nc, f"OV{key}wrong": wc,
                  f"OV{key}mirror": sum(1 for c in cs if not O[c]["flip_ok"])})
    # ---- local search: chance that best-of-k starts misses the global best (from 500 saved starts)
    rng = np.random.default_rng(0); brow = []
    for label, lp in (("Breast (development)", "gt1local500"), ("Colon, brain, DLPFC (development)", "gt2local"),
                      ("Six new datasets (held out)", "gt3local")):
        L = load(lp); cells = []
        for k in (25, 50, 100, 200):
            miss = []
            for c, r in L.items():
                sc = np.array([q["score"] for q in r["starts"]]); b = sc.max()
                for _ in range(200):
                    idx = rng.choice(len(sc), k, replace=False); miss.append(sc[idx].max() < b - 1e-3)
            cells.append(f"{100 * np.mean(miss):.0f}")
        brow.append(f"{label} & " + " & ".join(cells) + " & 0 \\\\")
        M["Miss" + {"Breast (development)": "Dev", "Colon, brain, DLPFC (development)": "DevB", "Six new datasets (held out)": "HO"}[label]] = cells[2]
    tab = ["\\begin{tabular}{lrrrrr}", "\\toprule",
           "Known-answer set & 25 starts & 50 & 100 & 200 & FLiPAlign \\\\", "\\midrule"] + brow + ["\\bottomrule", "\\end{tabular}"]
    open(f"{T}/res_budget.tex", "w").write("\n".join(tab) + "\n")
    # ---- real pairs
    rf = {json.load(open(f))[0]["pair"]: json.load(open(f))[0] for f in glob.glob(f"{R}/realfine_t*.json")}
    rb = json.load(open(f"{R}/real_baselines_arc.json")) if os.path.exists(f"{R}/real_baselines_arc.json") else []
    old = json.load(open("../server_results/results_real_baselines.json"))
    def bestgene(case, tool):
        v = [r.get("gene_score") for r in rb + old if r.get("case") == case and r.get("method", "").split("_")[0] == tool
             and r.get("gene_score") is not None]
        return max(v) if v else None
    def bestlm(tool):
        v = [r.get("landmark_error_median_um") for r in rb + old if r.get("case") == "XX" and r.get("method", "").split("_")[0] == tool
             and r.get("landmark_error_median_um") is not None]
        return min(v) if v else None
    xx = rf.get("XX", {})
    lmc = np.median(xx.get("landmark_um_coarse", [np.nan])); lmf = np.median(xx.get("landmark_um_fine", [np.nan]))
    rows = [f"Xenium$\\to$Xenium & landmark err. ($\\mu$m) $\\downarrow$ & {lmc:.0f} & {lmf:.0f} & "
            + " & ".join(f"{bestlm(t):.0f}" if bestlm(t) else "--" for t in ("paste2", "spateo", "cast")) + " \\\\"]
    for k, lab in (("VX1", "Visium$\\to$Xenium 1"), ("VX2", "Visium$\\to$Xenium 2")):
        r = rf.get(k, {})
        rows.append(f"{lab} & gene score $\\uparrow$ & {r.get('gene_coarse', float('nan')):.2f} & {r.get('gene_fine', float('nan')):.2f} & "
                    + " & ".join(f"{bestgene(k, t):.2f}" if bestgene(k, t) else "--" for t in ("paste2", "spateo", "cast")) + " \\\\")
    for k in ("colon P2CRC", "colon P3NAT", "colon P5NAT"):
        r = rf.get(k, {})
        gf = r.get("gene_fine"); gf = f"{gf:.2f}" if gf is not None else "--"
        rows.append(f"Colon {k[-5:-3]}, Visium$\\to$HD & gene score $\\uparrow$ & {r.get('gene_coarse', float('nan')):.2f} & {gf} & -- & -- & -- \\\\")
    tab = ["\\begin{tabular}{llrrrrr}", "\\toprule",
           "Real pair & Measure & FLiPAlign & + fine & PASTE2 & spateo & CAST \\\\", "\\midrule"] + rows + ["\\bottomrule", "\\end{tabular}"]
    open(f"{T}/res_real.tex", "w").write("\n".join(tab) + "\n")
    M.update({"XXcoarse": f"{lmc:.0f}", "XXfine": f"{lmf:.0f}"})
    # ---- Allen MERFISH real neighbouring sections (truth = Allen reconstruction; "near-best" =
    #      within 100 um of the best possible rigid placement, since real sections also differ non-rigidly)
    for d, key in (("abc", "ABCfull"), ("abc_ov70", "ABCpart")):
        man = manifest(d); O = load(d)
        if not O:
            continue
        good = lambda c, e: e < man[c]["rigid_floor_um"] + 100
        cs = sorted(O); cert = [c for c in cs if O[c]["certified"]]
        tools = {}; handed = {}
        fo = "abc" if d == "abc" else "abc70"
        for pat in (f"{d}base", f"{d}spateo", f"{fo}cast", f"{fo}stalign"):
            for f in glob.glob(f"{R}/{pat}_t*.json"):
                for r in json.load(open(f)):
                    if "tre_um" in r:
                        tools.setdefault((r["method"].split("_")[0], r["method"], r["fit"]), {})[r["case"]] = r["tre_um"]
        for pat in (f"{fo}fo_base", f"{fo}fo_spateo", f"{fo}fo_cast"):       # true flip handed over
            for f in glob.glob(f"{R}/{pat}_t*.json"):
                for r in json.load(open(f)):
                    if "tre_um" in r:
                        handed.setdefault((r["method"].split("_")[0], r["method"], r["fit"]), {})[r["case"]] = r["tre_um"]
        tb = {}
        for key_, v in tools.items():
            comb = dict(v); comb.update(handed.get(key_, {}))
            if len(comb) >= 0.8 * len(cs):
                tb[key_[0]] = max(tb.get(key_[0], 0), sum(good(c, e) for c, e in comb.items()))
        M.update({f"{key}cast": tb.get("cast", "--"), f"{key}stalign": tb.get("stalign", "--"),
                  f"{key}bestTool": max([v for v in tb.values()] + [0])})
        M.update({f"{key}n": len(cs), f"{key}good": sum(good(c, O[c]["tre_um"]) for c in cs),
                  f"{key}cert": len(cert), f"{key}wrongcert": sum(not good(c, O[c]["tre_um"]) for c in cert),
                  f"{key}mirror": sum(1 for c in cs if not O[c]["flip_ok"]),
                  f"{key}floor": f"{np.median([man[c]['rigid_floor_um'] for c in cs]):.0f}",
                  f"{key}pastetwo": tb.get("paste2", "--"), f"{key}spateo": tb.get("spateo", "--")})
    # ---- extra numbers quoted in the text
    F3 = load("gt3fine")
    fz = [r["fine_zone_move_um"] for r in F3.values() if r["fine_zone"]]
    M.update({"HOfineRan": len(F3), "HOfineCert": len(fz), "HOfineMove": f"{np.median(fz):.0f}",
              "RTcoarse": f"{np.median([r['total_s'] for r in load('gt3h2').values()]) / 60:.0f}",
              "RTfine": f"{np.median([r['total_s'] for r in F3.values()]) / 60:.0f}"})
    hits = []
    for r in load("gt3local").values():
        sc = np.array([q["score"] for q in r["starts"]]); hits.append(np.mean(sc >= sc.max() - 1e-3))
    M["HOrareCases"] = int(np.sum(np.array(hits) < 0.01))
    D = load("dlpfc66"); sd = [r for r in D.values() if r["group"] != "different donor"]
    ok = lambda r: r["flip"] == 0 and min(r["angle"], 360 - r["angle"]) <= 20
    c = [r for r in sd if r["certified"]]; adj = [r["layer_acc"] for r in sd if r["group"] == "adjacent"]
    M.update({"DLn": len(sd), "DLcert": len(c), "DLcertRight": sum(ok(r) for r in c),
              "DLcertMed": f"{np.median([r['layer_acc'] for r in c]):.2f}",
              "DLrefWrong": sum(not ok(r) for r in sd if not r["certified"]),
              "DLadjLo": f"{min(adj):.2f}", "DLadjHi": f"{max(adj):.2f}"})
    chk = [r for r in json.load(open("../server_results/results_check_dlpfc_baselines.json"))
           if "layer_acc" in r and r["group"] == "adjacent"]
    for m_, k_ in (("paste", "DLadjPaste"), ("slat", "DLadjSlat")):
        M[k_] = f"{max(np.median([r['layer_acc'] for r in chk if r['method'] == m_ and r['fit'] == ft]) for ft in ('plain', 'trimmed')):.2f}"
    cnt = [r for f in glob.glob(f"{R}/count_gt3_t*.json") for r in json.load(open(f))]
    if cnt:
        M.update({"EFFvisited": f"{np.median([r['evals'] for r in cnt]):,.0f}".replace(",", "{,}"),
                  "EFFgrid": f"{np.median([r['full_grid_at_leaf'] for r in cnt]):,.0f}".replace(",", "{,}"),
                  "EFFleaf": f"{np.median([r['leaf_halfwidth_deg'] for r in cnt]):.1f}"})
    if bestlm("stalign"):
        M["XXstalign"] = f"{bestlm('stalign'):,.0f}".replace(",", "{,}")
    bl = [bestlm(t) for t in ("paste", "paste2", "slat", "cpd", "spateo", "cast", "stalign") if bestlm(t)]
    M["XXbestTool"] = f"{min(bl):.0f}"
    br = [json.load(open(f))[0] for f in glob.glob(f"{R}/brainreal_t*.json")]
    if br:
        M.update({"BRn": len(br), "BRcert": sum(r["certified"] for r in br),
                  "BRgeneLo": f"{min(r['gene_answer'] for r in br):.2f}", "BRgeneHi": f"{max(r['gene_answer'] for r in br):.2f}",
                  "BRrivalLo": f"{min(r['gene_rival'] for r in br):.2f}", "BRrivalHi": f"{max(r['gene_rival'] for r in br):.2f}",
                  "BRrandMax": f"{max(r['gene_random_max'] for r in br):.2f}"})
    open(f"{T}/res_numbers.tex", "w").write("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in M.items()))
    return M


if __name__ == "__main__":
    print(json.dumps(main(), indent=0))
