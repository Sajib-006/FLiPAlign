"""Real pairs with the full method (coarse search + zone ladder + fine stage), 2026-10-01.

k = 0: Xenium Rep1 -> Rep2 (breast; STalign authors' hand-placed landmarks)
k = 1, 2: Visium -> Xenium Rep1 / Rep2 (same block)
k = 3, 4, 5: colon Visium -> Visium HD (P2CRC, P3NAT, P5NAT)
Same inputs and labels as the comparison table (run_compare.pairs, run_colon). Reports landmark
error (k = 0) and the gene score (k >= 1) at the coarse answer and at the fine answer.
Usage: python3 real_fine.py K
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, ".")
from smoothalign import SmoothAlign
from smoothpar import solve_par, far_par, far_par_ext
from bench_gt2 import LADDER, zone_move_um, NP, TL
from finecert import fine_stage
from bench_gt import UNIT
from lung_load import load_lung
from stack_place_load import load_stack_pair
from smoothalign import rot, FLIP

OUT = os.environ.get("GT_OUT", "results_real_fine.json")
# FLIP_KNOWN=1: the section is known not to be mounted face down (true flip 0 on all seven pairs); search and decision with that flip only
FL = (0,) if os.environ.get("FLIP_KNOWN") else (0, 1)


def load(k):
    if k <= 2:
        import run_compare as RC
        name, A, la, XA, genes, B, lb, XB, rad = list(RC.pairs())[k]
        return name, A, np.asarray(la), XA, B, np.asarray(lb), XB, rad
    import run_colon as C
    from bench_domains import neighbourhood_domains
    import scipy.sparse as sp, pandas as pd
    s = ("P2CRC", "P3NAT", "P5NAT")[k - 3]
    Axy, AX, gA, _ = C.load_visium(s); Bxy, BX, gB = C.load_hd(s)
    la = C.cluster(AX, 12, 1); B = Bxy / UNIT
    lb = neighbourhood_domains(B, C.cluster(BX, 12, 2), K=6, seed=3)
    common = sorted(set(gA) & set(gB))
    iA = pd.Series(np.arange(len(gA)), index=gA); iA = iA[~iA.index.duplicated()][common].values
    iB = pd.Series(np.arange(len(gB)), index=gB); iB = iB[~iB.index.duplicated()][common].values
    XAc = AX[:, iA]; XAn = sp.diags(1e3 / np.maximum(np.asarray(XAc.sum(1)).ravel(), 1)) @ XAc
    XAn.data = np.log1p(XAn.data)
    v = np.asarray(XAn.multiply(XAn).mean(0)).ravel() - np.asarray(XAn.mean(0)).ravel() ** 2
    top = np.argsort(v)[::-1][:500]
    return (f"colon {s}", Axy / UNIT, la, XAc[:, top].toarray().astype(float), B, lb,
            BX[:, iB[top]].toarray().astype(np.float32), 27.5 / UNIT * 1.5)


if __name__ == "__main__":
    k = int(sys.argv[-1])
    (name, A, la, XA, B, lb, XB, rad), At = load_stack_pair(k)
    t0 = time.time()
    al = SmoothAlign(A, la, B, lb, h=2.0, sigma=20.0, tpad=300.0, contrast=True)
    o = solve_par(al, n_proc=NP, time_limit=TL, flips=FL)
    th, t, fl = o["transform"]
    r = {"pair": name, "angle": round(float(np.degrees(th)) % 360, 4), "flip": int(fl),
         "shift_units": [round(float(v), 3) for v in t], "best": round(o["best"], 4), "gap": round(o["gap"], 4),
         "certified_search": o["certified"], "solve_s": round(o["secs"], 1), "ladder": []}
    if o["certified"]:
        for deg_z, rd in LADDER[:1] + [(20.0, 150.0), (20.0, 100.0)] + LADDER[1:]:
            if rd is None:
                u = far_par(al, o["best"], th, fl, exclude_deg=deg_z, n_proc=NP, time_limit=TL / 2, flips=FL)
            else:
                u = far_par_ext(al, o["best"], th, fl, t, exclude_deg=deg_z, shift_radius=rd, n_proc=NP,
                                time_limit=TL / 2, flips=FL)
            r["ladder"].append({"zone_deg": deg_z, "zone_shift_units": rd, "eps": round(u["eps"], 4),
                                "complete": u["complete"], "zone_move_um": round(zone_move_um(A * UNIT, deg_z, rd), 1)})
            if u["eps"] <= 0:
                break
    ok = [s for s in r["ladder"] if s["eps"] > 0]
    r["certified"] = bool(ok)
    r["finest_zone"] = [ok[-1]["zone_deg"], ok[-1]["zone_shift_units"]] if ok else None
    deg = float(np.degrees(th)) % 360
    if False:
        from run_xx_prereg import landmark_error
        r["landmark_um_coarse"] = landmark_error(A, B, deg, fl, t)
    elif False:
        from run_baselines_scale import GeneScore
        gs = GeneScore(A, XA, B, XB, rad)
        r["gene_coarse"] = round(gs(deg, fl, np.asarray(t)), 4)
    if r["certified"] and r["finest_zone"][1] is not None:
        f, (th2, t2) = fine_stage(A, la, B, lb, th, fl, t, *r["finest_zone"], n_proc=NP, time_limit=TL)
        r["fine"] = f
        d2 = float(np.degrees(th2)) % 360
        r["fine_angle"] = round(d2, 4); r["fine_shift_units"] = [round(float(v), 3) for v in t2]
        if False:
            r["landmark_um_fine"] = landmark_error(A, B, d2, fl, t2)
        elif False:
            r["gene_fine"] = round(gs(d2, fl, np.asarray(t2)), 4)
    def place(deg, fl_, t_):
        Mv = rot(np.radians(deg)) @ (FLIP if fl_ else np.eye(2))
        return (A - A.mean(0)) @ Mv.T + B.mean(0) + np.asarray(t_, float)
    P0 = place(r["angle"], r["flip"], r["shift_units"]); r["err_um_coarse"] = round(float(np.median(np.linalg.norm(P0 - At, axis=1)) * UNIT), 1)
    P = place(r["fine_angle"], r["flip"], r["fine_shift_units"]) if "fine_angle" in r else P0
    r["err_um"] = round(float(np.median(np.linalg.norm(P - At, axis=1)) * UNIT), 1)
    ca, cp = At.mean(0), P.mean(0); U_, _, Vt = np.linalg.svd((At - ca).T @ (P - cp)); Rm = Vt.T @ U_.T   # placed ~ true @ Rm.T + tm
    r["E_R"] = Rm.round(6).tolist(); r["E_t"] = (cp - ca @ Rm.T).round(4).tolist()
    r["total_s"] = round(time.time() - t0, 1)
    json.dump([r], open(OUT, "w"), indent=1); print(json.dumps(r), flush=True)
    print("DONE", flush=True)
