"""Write every number table of the paper straight from the result files (../server_results).
No number in the paper is typed by hand. Rerun this after any new result."""
import collections, json, os
import numpy as np

R = "../server_results"
T = "tables"
os.makedirs(T, exist_ok=True)
RIGHT_UM = 200
load = lambda f: json.load(open(os.path.join(R, f)))
NAMES = {"ours": "\\textbf{FLiPAlign}", "local_50": "Local search (our score, 50 starts)",
         "paste2_s0.5": "PASTE2", "paste2_s0.7": "PASTE2", "paste_a0.1": "PASTE", "paste_a0.5": "PASTE",
         "slat_k20": "SLAT", "cpd_8": "CPD"}


def gt_table():
    man = {x["case"]: x for x in load("gt_manifest.json")}
    ours = {r["case"]: r for r in load("results_gt_ours.json")}
    for r in load("results_gt_ours_recap.json"):
        ours[r["case"]] = r
    tab = collections.defaultdict(dict)
    for r in load("results_gt_baselines.json"):
        if "error" not in r:
            tab[(r["method"], r["fit"])][r["case"]] = r["tre_um"]
    tab[("ours", "")] = {c: r["tre_um"] for c, r in ours.items()}
    best = {}
    for (m, f), d in tab.items():
        v = np.array([d[c] for c in sorted(d)])
        name = NAMES.get(m, m)
        row = (np.median(v), np.mean(v < 100), np.mean(v < 200), np.mean(v < 500), f)
        if name not in best or row[0] < best[name][0]:
            best[name] = row
    lines = ["\\begin{tabular}{lrrrr}", "\\toprule",
             "Method & Median error ($\\mu$m) & $<100\\,\\mu$m & $<200\\,\\mu$m & $<500\\,\\mu$m \\\\", "\\midrule"]
    for name, (med, a, b, c, f) in sorted(best.items(), key=lambda kv: kv[1][0]):
        lines.append(f"{name} & {med:.0f} & {a:.0%} & {b:.0%} & {c:.0%} \\\\".replace("%", "\\%"))
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(f"{T}/gt_accuracy.tex", "w").write("\n".join(lines) + "\n")
    # certificate on the ground truth
    cert = [c for c, r in ours.items() if r.get("unique_shift30") or r.get("unique_shift60") or r.get("unique_rotation")]
    # one threshold everywhere (reviewers): right = within 200 um of the truth, wrong = farther
    wrong_cert = [c for c in cert if ours[c]["tre_um"] >= RIGHT_UM]
    right = [c for c, r in ours.items() if r["tre_um"] < RIGHT_UM]
    stats = {"n": len(ours), "right": len(right), "certified": len(cert), "wrong_certified": len(wrong_cert),
             "right_refused": len([c for c in right if c not in cert]),
             "median_tre": float(np.median([r["tre_um"] for r in ours.values()]))}
    json.dump(stats, open(f"{T}/gt_certificate.json", "w"), indent=1)
    return stats


def landmark_table():
    xx = load("results_xx_prereg.json")
    rows = [("\\textbf{FLiPAlign}", xx["landmark_error_median_um"])]
    best = {}
    for r in load("results_real_baselines.json"):
        if r.get("case") != "XX" or "error" in r:
            continue
        name = NAMES.get(r["method"], r["method"])
        v = r["landmark_error_median_um"]
        if name not in best or v < best[name]:
            best[name] = v
    rows += sorted(best.items(), key=lambda kv: kv[1])
    lines = ["\\begin{tabular}{lr}", "\\toprule", "Method & Median landmark error ($\\mu$m) \\\\", "\\midrule"]
    lines += [f"{n} & {v:.0f} \\\\" for n, v in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(f"{T}/landmarks.tex", "w").write("\n".join(lines) + "\n")


def cross_platform_table():
    """Gene score (spot expression vs cells under the spot) on cross-platform pairs: higher is better.
    Visium vs Xenium: ours under the pre-registered rule (h=2 row of the settings run), each baseline
    at its best setting and fit. Colon: ours only, against 20 random placements (baselines do not
    scale to ~500k Visium HD bins)."""
    sen = {r["pair"]: r for r in load("results_sensitivity.json") if r["changed"] == {"h": 2.0}}
    best = {}
    for r in load("results_real_baselines.json"):
        if r.get("case") in ("VX1", "VX2") and r.get("gene_score") is not None:
            n = NAMES.get(r["method"], r["method"])
            best[(r["case"], n)] = max(best.get((r["case"], n), -1), r["gene_score"])
    meths = ["Local search (our score, 50 starts)", "PASTE2", "PASTE", "SLAT", "CPD"]
    short = {"Local search (our score, 50 starts)": "Local"}
    lines = ["\\begin{tabular}{l" + "r" * (2 + len(meths)) + "}", "\\toprule",
             "Pair & \\textbf{FLiPAlign} & Unique & " + " & ".join(short.get(m, m) for m in meths) + " \\\\",
             "\\midrule"]
    for case, pair in (("VX1", "Visium->Rep1"), ("VX2", "Visium->Rep2")):
        o = sen[pair]
        lines.append(f"Visium vs Xenium {case[-1]} & {o['gene_score']:.2f} & "
                     f"{'yes' if o['unique_PROVEN'] else 'no'} & "
                     + " & ".join(f"{best[(case, m)]:.2f}" for m in meths) + " \\\\")
    lines.append("\\midrule")
    names = {"P2CRC": "Colon P2 (cancer)", "P3NAT": "Colon P3 (normal)", "P5NAT": "Colon P5 (normal)"}
    for r in load("results_colon.json"):
        lines.append(f"{names[r['pair']]} & {r['gene_score']:.2f} & "
                     f"{'yes' if r.get('unique_shift30') else 'no'} & "
                     f"\\multicolumn{{{len(meths)}}}{{l}}{{random: mean {r['random_gene_mean']:.2f}, "
                     f"best of 20: {r['random_gene_max']:.2f}}} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(f"{T}/cross_platform.tex", "w").write("\n".join(lines) + "\n")


def dlpfc_certificate():
    d = load("results_dlpfc_decide.json")
    cert = [r for r in d if r.get("eps_rotation", -1) > 0]
    ref = [r for r in d if r.get("certified") and r.get("eps_rotation", -1) <= 0]
    f = lambda L: {"n": len(L), "median_layer": float(np.median([r["layer_acc"] for r in L])),
                   "min_layer": float(min(r["layer_acc"] for r in L)),
                   "bad": int(sum(r["layer_acc"] < 0.4 for r in L))}
    json.dump({"certified": f(cert), "refused": f(ref)}, open(f"{T}/dlpfc_certificate.json", "w"), indent=1)


if __name__ == "__main__":
    print(gt_table()); landmark_table(); dlpfc_certificate(); cross_platform_table()
    for fn in sorted(os.listdir(T)):
        print("wrote", os.path.join(T, fn))


def numbers_tex():
    """Every number quoted in the text, as a LaTeX macro generated from the result files."""
    g = json.load(open(f"{T}/gt_certificate.json")); dl = json.load(open(f"{T}/dlpfc_certificate.json"))
    xx = load("results_xx_prereg.json")
    lm = {}
    for r in load("results_real_baselines.json"):
        if r.get("case") == "XX" and "error" not in r:
            n = NAMES.get(r["method"], r["method"]); lm[n] = min(lm.get(n, 1e9), r["landmark_error_median_um"])
    comp = load("results_gt_competitors.json")
    rr = [r for r in comp if r["answer_tre_um"] < RIGHT_UM]
    mirror = sum(1 for r in rr if r.get("rival_is_mirror"))
    right_refused = len(rr)
    # near-tie: the rival's exact score is within 2% of the answer's; otherwise the refusal came
    # from the looseness of the bound, not from a real rival
    near_tie = sum(1 for r in rr if r["rival_score"] >= 0.98 * r["best"])
    base = {}
    for r in load("results_gt_baselines.json"):
        if "error" in r:
            continue
        base.setdefault((r["method"], r["fit"]), []).append(r["tre_um"])
    p2 = min((np.median(v), np.mean(np.array(v) < 200)) for (m, f), v in base.items() if m.startswith("paste2"))
    M = {
        "GTn": g["n"], "GTright": g["right"], "GTcert": g["certified"], "GTwrongcert": g["wrong_certified"],
        "GTrightrefused": g["right_refused"], "GTmedian": f"{g['median_tre']:.0f}",
        "DLcertN": dl["certified"]["n"], "DLcertMed": f"{dl['certified']['median_layer']:.2f}",
        "DLcertMin": f"{dl['certified']['min_layer']:.2f}", "DLcertBad": dl["certified"]["bad"],
        "DLrefN": dl["refused"]["n"], "DLrefBad": dl["refused"]["bad"],
        "DLrefMed": f"{dl['refused']['median_layer']:.2f}",
        "XXlandmark": f"{xx['landmark_error_median_um']:.0f}", "XXeps": f"{xx.get('eps_shift30', float('nan')):.4f}",
        "XXsolveMin": f"{xx['solve_s'] / 60:.0f}", "XXgap": f"{xx['gap']:.3f}",
        "LMpastetwo": f"{lm.get('PASTE2', float('nan')):.0f}", "LMpaste": f"{lm.get('PASTE', float('nan')):.0f}",
        "MirrorRivals": mirror, "RightRefusedChecked": right_refused, "NearTies": near_tie,
        "SlackRefusals": right_refused - near_tie,
        "PtwoMedian": f"{p2[0]:.0f}", "PtwoUnderTwoHundred": f"{p2[1]:.0%}".replace("%", "\\%"),
    }
    # held-out colon pairs (Visium vs Visium HD)
    col = load("results_colon.json")
    M.update({"COLn": len(col), "COLcert": sum(r["certified"] for r in col),
              "COLunique": sum(r.get("unique_shift30", False) for r in col),
              "COLgeneMin": f"{min(r['gene_score'] for r in col):.2f}",
              "COLgeneMax": f"{max(r['gene_score'] for r in col):.2f}",
              "COLrandMax": f"{max(r['random_gene_max'] for r in col):.2f}",
              "COLmaxMin": f"{max(r['total_s'] for r in col) / 60:.0f}"})
    # fair DLPFC: ours on gene clusters vs PASTE and SLAT (same 18 pairs, same layer score)
    fair = load("results_dlpfc_fair.json")
    chk = [r for r in load("results_check_dlpfc_baselines.json") if "layer_acc" in r]
    for g, key in (("adjacent", "Adj"), ("300 um", "Far")):
        M[f"FAIR{key}Ours"] = f"{np.median([r['layer_acc'] for r in fair if r['group'] == g]):.2f}"
        for m, mk in (("paste", "Paste"), ("slat", "Slat")):
            M[f"FAIR{key}{mk}"] = f"{max(np.median([r['layer_acc'] for r in chk if r['method'] == m and r['fit'] == ft and r['group'] == g]) for ft in ('plain', 'trimmed')):.2f}"
    own = [r for r in chk if r["group"] == "adjacent" and r["fit"] == "plain"]
    M["PasteOwn"] = f"{np.median([r['own_measure_same_layer'] for r in own if r['method'] == 'paste']):.2f}"
    M["SlatOwn"] = f"{np.median([r['own_measure_same_layer'] for r in own if r['method'] == 'slat']):.2f}"
    uq = [r["layer_acc"] for r in fair if r["unique_rotation"]]
    M.update({"FAIRn": len(fair), "FAIRunique": len(uq), "FAIRuniqueMin": f"{min(uq):.2f}",
              "FAIRworst": f"{min(r['layer_acc'] for r in fair):.2f}",
              "FAIRworstUnique": "no" if min(r["layer_acc"] for r in fair) not in uq else "yes"})
    # settings: spread of the answer angle across the nine setting changes, per Visium pair
    sen = load("results_sensitivity.json")
    spread = max(np.ptp([r["angle"] for r in sen if r["pair"] == p]) for p in {r["pair"] for r in sen})
    M.update({"SENn": len({str(r["changed"]) for r in sen if r["changed"] != "none (frozen setting)"}), "SENspread": f"{spread:.1f}",
              "SENflips": sum(r["flip"] != 0 for r in sen)})
    open(f"{T}/numbers.tex", "w").write("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in M.items()))
    return M


if __name__ == "__main__":
    print(numbers_tex())
