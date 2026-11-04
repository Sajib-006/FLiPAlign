"""Tables with BOTH modes: flip searched (default) and flip known from outside (2026-10-04).

Two separate questions per case: is the answer RIGHT (median cell error under 200 um; on the Allen MERFISH
pairs: within 100 um of the best possible rigid placement), and does the certificate HOLD (the true
placement lies inside the certified zone, which is exactly what the certificate claims).
Truth angle / flip / shift per case are read from the flip-known result files (same cases).
Writes tables/res2_heldout.tex, res2_methods.tex, res2_brain.tex and res2_numbers.tex."""
import glob, json, os
import numpy as np
from make_results import load, manifest, baseline_best, local_right, pct, se, TOOLS, R, T, RIGHT


def load_s(prefix):
    """flip-searched results; older runs get the answer's shift from addshift_<set>.json"""
    O = load(prefix)
    f = f"{R}/addshift_{prefix[:3]}.json"
    if os.path.exists(f):
        for c, v in json.load(open(f)).items():
            if int(c) in O and "shift_units" not in O[int(c)]:
                O[int(c)]["shift_units"] = v["shift_units"]
    return O


KAPPA = 0.95


def selfcheck(fc_prefix):
    """mirror self-check: keep a flip-searched certificate only if the other flip's best placement agrees,
    per overlapping point, less than KAPPA times as well as the answer (fc_* files, bench_gt2 flipcheck)"""
    F = load(fc_prefix)
    def keep(c):
        r = F.get(c)
        if r is None:
            return True
        return (r["S_other"] / max(r["O_other"], 1e-9)) / (r["S_answer"] / max(r["O_answer"], 1e-9)) < KAPPA
    return keep


def ad(a, b):
    return abs((a - b + 180) % 360 - 180)


def holds(r, truth):
    """truth inside the finest certified coarse zone"""
    if not r["certified"]:
        return False
    dz, rz = r["finest_zone"]; tdeg, tfl = truth["truth"]
    if "shift_units" not in r:
        return None
    ds = float(np.hypot(*(np.array(r["shift_units"]) - truth["true_shift_units"])))
    return bool(tfl == r["flip"] and ad(r["angle"], tdeg) <= dz and (rz is None or ds <= rz))


def stats(O, K, FK, cases, good=None, keep=None):
    good = good or (lambda c, e: e < RIGHT)
    out = {}
    out["undet"] = sum(1 for c in cases if O[c]["certified"] and keep is not None and not keep(c))
    for tag, D in (("s", O), ("k", K)):
        cert = [c for c in cases if D[c]["certified"] and (tag == "k" or keep is None or keep(c))]
        h = [holds(D[c], K[c]) for c in cert]
        out[tag] = {"right": sum(good(c, D[c]["tre_um"]) for c in cases), "cert": len(cert),
                    "holds": (sum(1 for x in h if x) if all(x is not None for x in h) else None)}
    fin = [FK[c]["tre2_um"] if c in FK else K[c]["tre_um"] for c in cases if K[c]["certified"]]
    out["err"] = float(np.median(fin)) if fin else float("nan")
    return out


def cell(v):
    return "--" if v is None else str(v)


def main():
    M = {}
    m2 = {c: m["tissue"] for c, m in manifest("gt2").items()}
    m3 = {c: m["tissue"] for c, m in manifest("gt3").items()}
    # ---------------- held-out by tissue
    O, K, FK = load("gt3h2"), load("fk_gt3"), load("fkfine_gt3"); kp3 = selfcheck("fc_gt3")
    names = {"human lymph node": "Human lymph node", "human liver": "Human liver", "human kidney": "Human kidney",
             "human heart": "Human heart", "mouse brain (Prime 5K)": "Mouse brain (Prime 5K)",
             "mouse brain (V1)": "Mouse brain (V1)"}
    rows = []
    for t, lab in names.items():
        cs = sorted(c for c in m3 if m3[c] == t); s = stats(O, K, FK, cs, keep=kp3)
        bb = baseline_best("gt3", cs); bk = max((v[0] for v in bb.values() if v), default=0)
        e = f"{s['err']:.0f}" if np.isfinite(s["err"]) else "--"
        rows.append(f"{lab} & {len(cs)} & {s['s']['right']} & {s['s']['cert']} & {cell(s['s']['holds'])} & "
                    f"{s['k']['right']} & {s['k']['cert']} & {cell(s['k']['holds'])} & {e} & {bk} \\\\")
    cs = sorted(m3); s = stats(O, K, FK, cs, keep=kp3); s_raw = stats(O, K, FK, cs)
    M.update({"HORcert": s_raw["s"]["cert"], "HORholds": s_raw["s"]["holds"], "HOundet": s["undet"]})
    bb = baseline_best("gt3", cs); bk = max((v[0] for v in bb.values() if v), default=0)
    rows += ["\\midrule", f"All held-out & {len(cs)} & {s['s']['right']} & {s['s']['cert']} & {cell(s['s']['holds'])} & "
             f"{s['k']['right']} & {s['k']['cert']} & {cell(s['k']['holds'])} & {s['err']:.0f} & {bk} \\\\"]
    M.update({"HOSright": s["s"]["right"], "HOScert": s["s"]["cert"], "HOSholds": s["s"]["holds"],
              "HOKright": s["k"]["right"], "HOKcert": s["k"]["cert"], "HOKholds": s["k"]["holds"],
              "HOKerr": f"{s['err']:.0f}", "HObestTool": bk})
    nb = [c for c in cs if "brain" not in m3[c]]; sn = stats(O, K, FK, nb, keep=kp3)
    M.update({"NBn": len(nb), "NBScert": sn["s"]["cert"], "NBSholds": sn["s"]["holds"], "NBKcert": sn["k"]["cert"],
              "NBKholds": sn["k"]["holds"], "NBKright": sn["k"]["right"], "NBSright": sn["s"]["right"]})
    tab = ["\\begin{tabular}{lr|rrr|rrr|rr}", "\\toprule",
           " & & \\multicolumn{3}{c|}{Flip searched} & \\multicolumn{3}{c|}{Flip known} & & \\\\",
           "Tissue & $n$ & Right & Cert. & Holds & Right & Cert. & Holds & Err. ($\\mu$m) & Best tool \\\\", "\\midrule"] + \
          rows + ["\\bottomrule", "\\end{tabular}"]
    open(f"{T}/res2_heldout.tex", "w").write("\n".join(tab) + "\n")
    # ---------------- all sets x methods, share right
    lines = []
    for label, setn, lp in (("Breast (dev.)", "gt1", "gt1local500"), ("Colon/brain/DLPFC (dev.)", "gt2", "gt2local"),
                            ("Six new (held out)", "gt3", "gt3local")):
        O, K = load_s(f"{setn}h2"), load(f"fk_{setn}"); cs = sorted(O); n = len(cs)
        ks, kk = sum(O[c]["tre_um"] < RIGHT for c in cs), sum(K[c]["tre_um"] < RIGHT for c in cs)
        bb = baseline_best(setn, cs)
        cells = [f"{pct(ks, n)} $\\pm$ {se(ks, n)}", f"{pct(kk, n)} $\\pm$ {se(kk, n)}", pct(local_right(load(lp), cs), n)]
        cells += [pct(bb[t][0], bb[t][2]) if bb.get(t) else "--" for t, _ in TOOLS]
        lines.append(f"{label} & {n} & " + " & ".join(cells) + " \\\\")
        key = {"gt1": "A", "gt2": "B", "gt3": "C"}[setn]
        M.update({f"SET{key}searched": pct(ks, n), f"SET{key}known": pct(kk, n),
                  f"SET{key}bestTool": max(int(pct(bb[t][0], bb[t][2])) for t, _ in TOOLS if bb.get(t))})
    tab = ["\\begin{tabular}{lr|rr|r|rrrrrrr}", "\\toprule",
           " & & \\multicolumn{2}{c|}{FLiPAlign} & & \\multicolumn{7}{c}{Published tools (flip handed over)} \\\\",
           "Set & $n$ & searched & known & Local & " + " & ".join(n_ for _, n_ in TOOLS) + " \\\\", "\\midrule"] + \
          lines + ["\\bottomrule", "\\end{tabular}"]
    open(f"{T}/res2_methods.tex", "w").write("\n".join(tab) + "\n")
    # ---------------- brain: overlap sweep and Allen MERFISH
    rows = []
    for lab, d, n_, key in (("Xenium brain, 100\\% overlap", "gt_ov100", "ov100", "BF"), ("Xenium brain, 80\\%", "gt_ov80", "ov80", "BE"),
                            ("Xenium brain, 60\\%", "gt_ov60", "ov60", "BS")):
        O, K, FK = load(d), load(f"fk_{n_}"), load(f"fkfine_{n_}"); cs = sorted(O); s = stats(O, K, FK, cs, keep=selfcheck(f"fc_{n_}"))
        sr = stats(O, K, FK, cs)
        M.update({f"{key}Rcert": sr["s"]["cert"], f"{key}Rholds": sr["s"]["holds"], f"{key}undet": s["undet"]})
        rows.append(f"{lab} & {len(cs)} & {s['s']['right']} & {s['s']['cert']} & {cell(s['s']['holds'])} & "
                    f"{s['k']['right']} & {s['k']['cert']} & {cell(s['k']['holds'])} \\\\")
        M.update({f"{key}n": len(cs), f"{key}Sright": s["s"]["right"], f"{key}Scert": s["s"]["cert"], f"{key}Sholds": s["s"]["holds"],
                  f"{key}Kright": s["k"]["right"], f"{key}Kcert": s["k"]["cert"], f"{key}Kholds": s["k"]["holds"]})
    rows.append("\\midrule")
    for lab, d, n_, key in (("Allen MERFISH, whole sections", "abc", "abc", "MF"), ("Allen MERFISH, 70\\% overlap", "abc_ov70", "abc70", "MP")):
        man = manifest(d); O, K, FK = load(d), load(f"fk_{n_}"), load(f"fkfine_{n_}"); cs = sorted(O)
        good = lambda c, e, man=man: e < man[c]["rigid_floor_um"] + 100
        s = stats(O, K, FK, cs, good, keep=selfcheck(f"fc_{n_}")); sr = stats(O, K, FK, cs, good)
        M.update({f"{key}Rcert": sr["s"]["cert"], f"{key}Rholds": sr["s"]["holds"], f"{key}undet": s["undet"]})
        rows.append(f"{lab} & {len(cs)} & {s['s']['right']} & {s['s']['cert']} & {cell(s['s']['holds'])} & "
                    f"{s['k']['right']} & {s['k']['cert']} & {cell(s['k']['holds'])} \\\\")
        M.update({f"{key}n": len(cs), f"{key}Sright": s["s"]["right"], f"{key}Scert": s["s"]["cert"], f"{key}Sholds": s["s"]["holds"],
                  f"{key}Kright": s["k"]["right"], f"{key}Kcert": s["k"]["cert"], f"{key}Kholds": s["k"]["holds"],
                  f"{key}floor": f"{np.median([man[c]['rigid_floor_um'] for c in cs]):.0f}"})
    tab = ["\\begin{tabular}{lr|rrr|rrr}", "\\toprule",
           " & & \\multicolumn{3}{c|}{Flip searched} & \\multicolumn{3}{c}{Flip known} \\\\",
           "Mouse brain pairs & $n$ & Right & Cert. & Holds & Right & Cert. & Holds \\\\", "\\midrule"] + rows + ["\\bottomrule", "\\end{tabular}"]
    open(f"{T}/res2_brain.tex", "w").write("\n".join(tab) + "\n")
    # development brain / DLPFC with flip known
    O, K, FK = load_s("gt2h2"), load("fk_gt2"), load("fkfine_gt2")
    for t, key in (("mouse brain", "DB"), ("DLPFC", "DD"), ("colon", "DC")):
        cs = sorted(c for c in m2 if m2[c] == t); s = stats(O, K, FK, cs, keep=selfcheck("fc_gt2"))
        M.update({f"{key}n": len(cs), f"{key}Sright": s["s"]["right"], f"{key}Kright": s["k"]["right"],
                  f"{key}Kcert": s["k"]["cert"], f"{key}Kholds": s["k"]["holds"], f"{key}Scert": s["s"]["cert"],
                  f"{key}Sholds": s["s"]["holds"]})
    # all certificates, both modes, every known-answer set
    tot = {"r": [0, 0], "s": [0, 0], "k": [0, 0]}; und = 0
    for o, k, fc in (("gt1h2", "fk_gt1", "fc_gt1"), ("gt2h2", "fk_gt2", "fc_gt2"), ("gt3h2", "fk_gt3", "fc_gt3"),
                     ("gt_ov100", "fk_ov100", "fc_ov100"), ("gt_ov80", "fk_ov80", "fc_ov80"), ("gt_ov60", "fk_ov60", "fc_ov60")):
        O, K = load_s(o), load(k); s = stats(O, K, {}, sorted(O), keep=selfcheck(fc)); sr = stats(O, K, {}, sorted(O))
        und += s["undet"]
        tot["r"][0] += sr["s"]["cert"]; tot["r"][1] += sr["s"]["holds"] or 0
        for tag in ("s", "k"):
            tot[tag][0] += s[tag]["cert"]; tot[tag][1] += s[tag]["holds"] or 0
    M.update({"ALLRcert": tot["r"][0], "ALLRholds": tot["r"][1], "ALLScert": tot["s"][0], "ALLSholds": tot["s"][1],
              "ALLKcert": tot["k"][0], "ALLKholds": tot["k"][1], "ALLundet": und})
    # ---------------- certified / refused x right / wrong, and how tight the zones are
    rows = []
    for lab, sets_, key in (("Held out (48)", [("gt3h2", "fk_gt3", "gt3fine", "fkfine_gt3")], "CH"),
                            ("All sets (204)", [("gt1h2", "fk_gt1", "gt1fine", "fkfine_gt1"),
                                                            ("gt2h2", "fk_gt2", "gt2fine", "fkfine_gt2"),
                                                            ("gt3h2", "fk_gt3", "gt3fine", "fkfine_gt3"),
                                                            ("gt_ov100", "fk_ov100", "gt_ov100fine", "fkfine_ov100"),
                                                            ("gt_ov80", "fk_ov80", "gt_ov80fine", "fkfine_ov80"),
                                                            ("gt_ov60", "fk_ov60", "gt_ov60fine", "fkfine_ov60")], "CA")):
        for mode, tag in (("searched, no self-check", "R"), ("searched", "S"), ("known", "K")):
            cr = cw = rr = rw = shiftb = fine = 0
            for o, k, fo, fk in sets_:
                D = load_s(o) if tag != "K" else load(k); F = load(fo if tag != "K" else fk)
                kp = selfcheck("fc_" + k[3:]) if tag == "S" else (lambda c: True)
                for c, r in D.items():
                    right = r["tre_um"] < RIGHT
                    if r["certified"] and kp(c):
                        cr += right; cw += (not right)
                        shiftb += r["finest_zone"][1] is not None
                        fine += (c in F and F[c]["fine_zone"] is not None)
                    else:
                        rr += right; rw += (not right)
            rows.append(f"{lab} & {mode} & {cr} & {cw} & {rr} & {rw} & {shiftb} & {fine} \\\\")
            M.update({f"{key}{tag}cr": cr, f"{key}{tag}cw": cw, f"{key}{tag}rr": rr, f"{key}{tag}rw": rw,
                      f"{key}{tag}shift": shiftb, f"{key}{tag}fine": fine})
    tab = ["\\begin{tabular}{llrrrrrr}", "\\toprule",
           " & & \\multicolumn{2}{c}{Certified} & \\multicolumn{2}{c}{Refused} & \\multicolumn{2}{c}{Certificates with} \\\\",
           "Cases & Flip & right & wrong & right & wrong & shift bound & fine zone \\\\", "\\midrule"] + rows + ["\\bottomrule", "\\end{tabular}"]
    open(f"{T}/res2_confusion.tex", "w").write("\n".join(tab) + "\n")
    # ---------------- plain angle grid + FFT shifts on the same score (no proof)
    for setn, op, key in (("gt1", "gt1h2", "A"), ("gt2", "gt2h2", "B"), ("gt3", "gt3h2", "C")):
        G = load(f"grid_{setn}"); O = load(op); cs = sorted(set(G) & set(O))
        for step, nm in ((1, "One"), (10, "Ten")):
            right = miss = 0
            for c in cs:
                per = [q for q in G[c]["per_angle"] if abs(q[0] / step - round(q[0] / step)) < 1e-6]
                b = max(per, key=lambda q: q[2]); right += b[5] < RIGHT; miss += b[2] < O[c]["best"] - 1e-3
            M.update({f"GRID{key}{nm}right": right, f"GRID{key}{nm}miss": miss})
        M[f"GRID{key}min"] = f"{np.median([G[c]['total_s'] for c in cs]) / 60:.0f}"
        M[f"SEARCH{key}min"] = f"{np.median([O[c]['solve_s'] for c in cs]) / 60:.0f}"
    # ---------------- dependence on the clustering (held-out, flip known, five re-clusterings)
    m3_ = {c: m["tissue"] for c, m in manifest("gt3").items()}
    base = load("fk_gt3"); rights = []; cs_, cb_, hs_, hb_ = 0, 0, 0, 0
    for tag in ("rl_1_12_6", "rl_2_12_6", "rl_3_12_6", "rl_1_8_4", "rl_1_16_8"):
        Rr = load(tag)
        if not Rr:
            continue
        rights.append(sum(r["tre_um"] < RIGHT for r in Rr.values()))
        for c, r in Rr.items():
            if r["certified"]:
                h = holds(r, r)
                solid = m3_[c] in ("human kidney", "human liver", "human lymph node")
                cs_ += solid; hs_ += solid and h; cb_ += (not solid); hb_ += (not solid) and h
    if rights:
        M.update({"RLlo": min(rights), "RLhi": max(rights), "RLsolidCert": cs_, "RLsolidHolds": hs_,
                  "RLhardCert": cb_, "RLhardFail": cb_ - hb_})
    # ---------------- proven certificate vs an unproven margin rule on the 1-degree grid
    def gm(g):
        per = g["per_angle"]; b = max(per, key=lambda q: q[2])
        far = [q[2] for q in per if q[1] != b[1] or ad(q[0], b[0]) > 20]
        return b[2] - max(far), b[5] < RIGHT
    rows_ = []
    for setn, op in (("gt1", "gt1h2"), ("gt2", "gt2h2"), ("gt3", "gt3h2")):
        O_ = load(op); G_ = load(f"grid_{setn}")
        for c in sorted(set(O_) & set(G_)):
            e = O_[c]["ladder"][0]["eps"] if O_[c]["ladder"] else -1.0
            m_, gr = gm(G_[c]); rows_.append((setn, e, O_[c]["tre_um"] < RIGHT, m_, gr))
    for key, sel in (("RVH", [r for r in rows_ if r[0] == "gt3"]), ("RVA", rows_)):
        eps_ = np.array([r[1] for r in sel]); m_ = np.array([r[3] for r in sel])
        ro = np.array([r[2] for r in sel]); rg = np.array([r[4] for r in sel])
        k = int((eps_ > 0).sum()); idx = np.argsort(-m_)[:k]
        M.update({f"{key}k": k, f"{key}certRight": int(ro[eps_ > 0].sum()), f"{key}ruleRight": int(rg[idx].sum()),
                  f"{key}same": len(set(idx.tolist()) & set(np.where(eps_ > 0)[0].tolist()))})
    okm = np.array([r[1] for r in rows_]) > -0.5
    M["RVcorr"] = f"{np.corrcoef(np.array([r[1] for r in rows_])[okm], np.array([r[3] for r in rows_])[okm])[0, 1]:.2f}"
    M["RVslack"] = f"{np.median((np.array([r[3] for r in rows_]) - np.array([r[1] for r in rows_]))[okm]):.3f}"
    # ---------------- self-check threshold sweep (flip-searched certificates, all known-answer sets)
    def ratio_(r):
        return (r["S_other"] / max(r["O_other"], 1e-9)) / (r["S_answer"] / max(r["O_answer"], 1e-9))
    sw = []
    for o, k, fc in (("gt1h2", "fk_gt1", "fc_gt1"), ("gt2h2", "fk_gt2", "fc_gt2"), ("gt3h2", "fk_gt3", "fc_gt3"),
                     ("gt_ov100", "fk_ov100", "fc_ov100"), ("gt_ov80", "fk_ov80", "fc_ov80"), ("gt_ov60", "fk_ov60", "fc_ov60")):
        O_, K_, F_ = load_s(o), load(k), load(fc)
        sw += [(ratio_(F_[c]), bool(holds(O_[c], K_[c]))) for c in O_ if O_[c]["certified"] and c in F_]
    for kap, nm in ((0.80, "Lo"), (1.00, "One"), (1.10, "Hi")):
        kept = [d for d in sw if d[0] < kap]
        M.update({f"KS{nm}kept": len(kept), f"KS{nm}holds": sum(d[1] for d in kept)})
    # ---------------- consensus over clusterings (held-out, flip known): base + 5 re-clusterings
    runs_ = [load("fk_gt3")] + [load(t_) for t_ in ("rl_1_12_6", "rl_2_12_6", "rl_3_12_6", "rl_1_8_4", "rl_1_16_8")]
    if all(runs_):
        for m_need, nm in ((4, "Four"), (6, "Six")):
            acc = right_ = 0
            for c in sorted(runs_[0]):
                cert = [R_[c] for R_ in runs_ if R_[c]["certified"]]
                best_ = []
                for a_ in cert:
                    grp = [b_ for b_ in cert if ad(a_["angle"], b_["angle"]) <= 20
                           and np.hypot(*(np.array(a_["shift_units"]) - b_["shift_units"])) <= 60]
                    if len(grp) > len(best_):
                        best_ = grp
                if len(best_) >= m_need:
                    acc += 1; right_ += np.median([b_["tre_um"] for b_ in best_]) < RIGHT
            M.update({f"CON{nm}cert": acc, f"CON{nm}right": int(right_)})
    open(f"{T}/res2_numbers.tex", "w").write("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in M.items()))
    return M


if __name__ == "__main__":
    print(json.dumps(main(), indent=0))
