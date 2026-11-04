"""Runs the published tools on the known-answer cases and the real pairs. Every row saves the FULL transform.

  sanity  known move of the same slide (Xenium Rep2, two disjoint halves, regions K=6,
          all 313 genes): 20 deg, 60 deg, 135 deg + flip, with a shift. Pass = angle within
          5 deg, flip right, TRE < 100 um. Tells which baselines work at all in our set-up.
  real    the 3 real pairs (XX, VX1, VX2), common unit, plus the diagnostic: gene score at
          the method's own placement AND at (its angle, best shift for that angle under
          our objective), to separate angle error from shift error.
  gt      the ground-truth cases in data/gt (TRE in um against the known truth).
Usage: PYTHONPATH=.pylibs_shims:.pylibs_base MPLBACKEND=Agg python3 runs_baselines/bench_baselines.py MODE [methods...]
"""
import os, sys  # noqa: E401
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import repo_paths  # noqa: E402,F401  (puts the code folders on the import path)
import json, os, sys, time
import numpy as np
sys.path.insert(0, ".")
import baselines as BL
from smoothalign import SmoothAlign, rot, FLIP

UNIT = 9.44
METHODS = {
    "paste_a0.1": lambda c: BL.paste(c["A"], c["XA"], c["B"], c["XB"], c["genes"], 0.1),
    "paste_a0.5": lambda c: BL.paste(c["A"], c["XA"], c["B"], c["XB"], c["genes"], 0.5),
    "paste2_s0.5": lambda c: BL.paste2(c["A"], c["XA"], c["B"], c["XB"], c["genes"], 0.5),
    "paste2_s0.7": lambda c: BL.paste2(c["A"], c["XA"], c["B"], c["XB"], c["genes"], 0.7),
    "slat_k20": lambda c: BL.slat(c["A"], c["XA"], c["B"], c["XB"], c["genes"], 20, 6),
    "cpd_8": lambda c: BL.cpd(c["A"], c["B"], 8),
    # extra settings so every baseline has three (three or more settings each)
    "paste_a0.01": lambda c: BL.paste(c["A"], c["XA"], c["B"], c["XB"], c["genes"], 0.01),
    "paste2_s0.3": lambda c: BL.paste2(c["A"], c["XA"], c["B"], c["XB"], c["genes"], 0.3),
    "slat_k10": lambda c: BL.slat(c["A"], c["XA"], c["B"], c["XB"], c["genes"], 10, 6),
    "slat_k40": lambda c: BL.slat(c["A"], c["XA"], c["B"], c["XB"], c["genes"], 40, 6),
    "cpd_8_w0": lambda c: BL.cpd(c["A"], c["B"], 8, w=0.0),
    "cpd_8_w0.5": lambda c: BL.cpd(c["A"], c["B"], 8, w=0.5),
    # newer tool: spateo morpho_align, three settings
    "spateo_sns": lambda c: BL.spateo_align(c["A"], c["XA"], c["B"], c["XB"], c["genes"], "SN-S"),
    "spateo_snn": lambda c: BL.spateo_align(c["A"], c["XA"], c["B"], c["XB"], c["genes"], "SN-N"),
    "spateo_sns_cos": lambda c: BL.spateo_align(c["A"], c["XA"], c["B"], c["XB"], c["genes"], "SN-S", "cos"),
    "cast_rigid": lambda c: BL.cast_align(c["A"], c["XA"], c["B"], c["XB"], c["genes"], "rigid"),
    "cast_affine": lambda c: BL.cast_align(c["A"], c["XA"], c["B"], c["XB"], c["genes"], "affine"),
    "cast_full": lambda c: BL.cast_align(c["A"], c["XA"], c["B"], c["XB"], c["genes"], "full"),
    "local_50": lambda c: BL.local_search(c["al"], 50),
    "stalign_16": lambda c: BL.stalign(c["A"], c["la"], c["B"], c["lb"], UNIT,
                                       [(d, f) for f in (0, 1) for d in np.arange(0, 360, 45.0)], 800),
    # single start (for the arms where A is handed over already roughly placed)
    "stalign_1": lambda c: BL.stalign(c["A"], c["la"], c["B"], c["lb"], UNIT, [(0.0, 0)], 800,
                                      recenter=False),
    "cpd_1": lambda c: BL.cpd(c["A"], c["B"], 1),
    # STalign with a matching weight that lets it move without landmarks . The tutorial weight
    # (sigmaM 1.5) left every start unchanged; sigmaM 0.05 with larger steps was the best of 13 settings on six
    # near-true starts (calib_tools.py, calib2.py), chosen with the truth.
    "stalign_16c": lambda c: BL.stalign(c["A"], c["la"], c["B"], c["lb"], UNIT,
                                        [(d, f) for f in (0, 1) for d in np.arange(0, 360, 45.0)], 800, epL=2e-6,
                                        sigmaM=0.05, epT=2.0),
    "stalign_1c": lambda c: BL.stalign(c["A"], c["la"], c["B"], c["lb"], UNIT, [(0.0, 0)], 800, epL=2e-6,
                                       recenter=False, sigmaM=0.05, epT=2.0),
    # the other two outlier weights for the single-start path (every arm gets all three settings)
    "cpd_1_w0": lambda c: BL.cpd(c["A"], c["B"], 1, w=0.0),
    "cpd_1_w0.5": lambda c: BL.cpd(c["A"], c["B"], 1, w=0.5),
}


def place(A, deg, fl, t, B):
    M = rot(np.radians(deg)) @ (FLIP if fl else np.eye(2))
    return (A - A.mean(0)) @ M.T + B.mean(0) + t


def run_case(tag, c, wanted, out, truth=None, gene_score=None, extra_fn=None):
    res = json.load(open(out)) if os.path.exists(out) else []
    done = {(r["case"], r["method"]) for r in res}
    for m in wanted:
        if (tag, m) in done:                             # resume: skip finished work
            continue
        t0 = time.time()
        try:
            r = METHODS[m](c)
        except Exception as e:
            import traceback
            res.append({"case": tag, "method": m, "error": repr(e)[:300],
                        "where": traceback.format_exc()[-300:]})
            json.dump(res, open(out, "w"), indent=1); print(json.dumps(res[-1]), flush=True)
            continue
        sec = round(time.time() - t0, 1)
        for fit in ("plain", "trimmed"):
            if fit not in r:
                continue
            deg, fl, t = r[fit]; t = np.asarray(t, float)
            if c.get("flip_fix"):
                fl = 1 - int(fl)               # placement of mirrored A == same rotation, flip toggled
            row = {"case": tag, "method": m, "fit": fit, "angle": round(float(deg), 2),
                   "flip": int(fl), "shift": [round(float(v), 2) for v in t], "sec": sec}
            if truth is not None:
                P = place(c.get("A_eval", c["A"]), deg, fl, t, c["B"]) * UNIT
                row["tre_um"] = round(float(np.median(np.linalg.norm(P - truth, axis=1))), 1)
                if "moved" in r and not c.get("flip_fix"):     # the tool's own placement, not refitted
                    idx, Mv = r["moved"]
                    row["tre_nonrigid_um"] = round(float(np.median(np.linalg.norm(
                        np.asarray(Mv) * UNIT - truth[idx], axis=1))), 1)
            for k in ("arm", "start_err_um"):
                if k in c:
                    row[k] = c[k]
            if gene_score is not None:
                row["gene_score"] = round(gene_score(deg, fl, t), 4)
                # diagnostic: same angle/flip, best shift for that angle under our objective
                s_, t_best, _ = c["al"].rotation_eval(np.radians(deg), fl, np.radians(0.25))
                row["gene_at_own_angle_best_shift"] = round(gene_score(deg, fl, np.asarray(t_best, float)), 4)
            if extra_fn is not None:
                row.update(extra_fn(deg, fl, t))
            for k in ("picked_start", "density_corr"):
                if k in r:
                    row[k] = r[k]
            res.append(row)
        json.dump(res, open(out, "w"), indent=1); print(json.dumps(res[-1]), flush=True)


def sanity(wanted):
    import run_visium as RV
    from bench_domains import neighbourhood_domains
    xy, lab = RV.xenium("Rep2"); X, genes = RV.expr("Rep2", RV.cells("Rep2")); X = np.expm1(X)
    B0 = xy / UNIT; reg = neighbourhood_domains(B0, lab, K=6, seed=1)
    r = np.random.default_rng(0); perm = r.permutation(len(B0)); hA, hB = perm[::2], perm[1::2]
    for true, fl0 in ((20.0, 0), (60.0, 0), (135.0, 1)):
        M0 = rot(np.radians(true)) @ (FLIP if fl0 else np.eye(2))
        cA = B0[hA].mean(0)
        A = (B0[hA] - cA) @ M0.T + cA + np.array([25.0, -40.0])
        c = {"A": A, "la": reg[hA], "XA": X[hA], "B": B0[hB], "lb": reg[hB], "XB": X[hB],
             "genes": list(genes)}
        c["al"] = SmoothAlign(A, c["la"], c["B"], c["lb"], h=4.0, sigma=20.0, tpad=300.0, contrast=True)
        run_case(f"sanity_{int(true)}_{fl0}", c, wanted, "results_sanity_baselines.json",
                 truth=B0[hA] * UNIT)


def preplace(c, M0, shift, truth, arm):
    """Hand the tool A already moved: A' = M0 (A - mean A) + mean B + shift. The tool's answer is then
    scored on what it was given, so no mapping back is needed (rows of A keep their order)."""
    A = c["A"]
    c["A"] = (A - A.mean(0)) @ M0.T + c["B"].mean(0) + np.asarray(shift, float)
    c["arm"] = arm
    c["start_err_um"] = round(float(np.median(np.linalg.norm(c["A"] * UNIT - truth, axis=1))), 1)


def gt(wanted, first=0, last=None):
    # GT_DIR / GT_BASE_OUT pick the case set (data/gt = first set, data/gt2 = new tissues)
    d = os.environ.get("GT_DIR", "data/gt")
    last = int(os.environ.get("GT_LAST", "23")) if last is None else last
    for i in range(int(os.environ.get("GT_FIRST", first)), last + 1):
        f = f"{d}/case_{i:02d}.npz"
        if not os.path.exists(f):
            continue
        z = np.load(f)
        c = {"A": z["Aobs"] / UNIT, "la": z["la"], "XA": z["AX"], "B": z["B"] / UNIT,
             "lb": z["lb"], "XB": z["BX"], "genes": list(z["genes"])}
        # ROUGH_START=d: hand every tool the true flip AND the true angle with an error drawn
        # uniformly from [-d, d] degrees (what a glance at the slide gives). The shift is not handed
        # over: A is put with its centre on B's centre (unless ROUGH_SHIFT is set).
        # INIT_FROM=glob: hand the tool FLiPAlign's own placement (angle, flip, shift) as the start.
        if os.environ.get("ROUGH_START"):
            dmax = float(os.environ["ROUGH_START"])
            W = np.linalg.inv(rot(float(z["th0"])) @ (FLIP if int(z["fl0"]) else np.eye(2)))
            rg = np.random.default_rng(31000 + i)
            e = rg.uniform(-dmax, dmax)
            sh, arm = np.zeros(2), f"rough{dmax:g}"
            if os.environ.get("ROUGH_SHIFT"):      # also hand over the position, to within +-ROUGH_SHIFT um per axis
                smax = float(os.environ["ROUGH_SHIFT"])
                sh = z["Atrue"].mean(0) / UNIT - c["B"].mean(0) + rg.uniform(-smax, smax, 2) / UNIT
                arm += f"_shift{smax:g}"
            preplace(c, rot(np.radians(e)) @ W, sh, z["Atrue"], arm)
        elif os.environ.get("INIT_FROM"):
            import glob
            ours = {}
            for f_ in glob.glob(os.environ["INIT_FROM"]):
                for r_ in json.load(open(f_)):
                    ours[r_["case"]] = r_
            if i not in ours or "shift_units" not in ours[i]:
                continue
            o_ = ours[i]
            preplace(c, rot(np.radians(o_["angle"])) @ (FLIP if o_["flip"] else np.eye(2)), o_["shift_units"],
                     z["Atrue"], "init")
        if "local_50" in wanted:
            c["al"] = SmoothAlign(c["A"], c["la"], c["B"], c["lb"], h=4.0, sigma=20.0, tpad=300.0, contrast=True)
        # FLIP_ORACLE=1 (most tools never try a mirror flip): when the true answer is
        # flipped, hand the tool A already mirrored, so it only has to find a rotation and shift;
        # its answer is mapped back (same rotation, flip toggled) and scored on the original A.
        if os.environ.get("FLIP_ORACLE") == "1" and int(z["fl0"]) == 0:
            continue                           # unflipped cases: results are the same as before
        if os.environ.get("FLIP_ORACLE") == "1" and int(z["fl0"]) == 1:
            c["A_eval"] = c["A"]
            c["A"] = c["A"] @ FLIP
            c["flip_fix"] = True
        run_case(i, c, wanted, os.environ.get("GT_BASE_OUT", "results_gt_baselines.json"), truth=z["Atrue"])


def real(wanted):
    import run_compare as RC
    from run_baselines_scale import GeneScore
    for name, A, la, XA, genes, B, lb, XB, rad in RC.pairs():
        c = {"A": A, "la": np.asarray(la), "XA": XA, "B": B, "lb": np.asarray(lb), "XB": XB,
             "genes": list(genes)}
        c["al"] = SmoothAlign(A, c["la"], B, c["lb"], h=4.0, sigma=20.0, tpad=300.0, contrast=True)
        gs = GeneScore(A, XA, B, XB, rad)
        extra = None
        if name == "XX":                     # STalign authors' hand-placed landmarks (um)
            from run_xx_prereg import landmark_error
            extra = lambda deg, fl, t, A=A, B=B: {
                "landmark_error_um": landmark_error(A, B, deg, fl, t),
                "landmark_error_median_um": float(np.median(landmark_error(A, B, deg, fl, t)))}
        run_case(name, c, wanted, "results_real_baselines.json", gene_score=gs, extra_fn=extra)


if __name__ == "__main__":
    mode, wanted = sys.argv[1], (sys.argv[2:] or list(METHODS))
    {"sanity": sanity, "real": real, "gt": gt}[mode](wanted)
    print("DONE", flush=True)
