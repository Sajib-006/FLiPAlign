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

OUT = os.environ.get("GT_OUT", "results_real_fine.json")


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
    name, A, la, XA, B, lb, XB, rad = load(k)
    t0 = time.time()
    al = SmoothAlign(A, la, B, lb, h=2.0, sigma=20.0, tpad=300.0, contrast=True)
    o = solve_par(al, n_proc=NP, time_limit=TL)
    th, t, fl = o["transform"]
    r = {"pair": name, "angle": round(float(np.degrees(th)) % 360, 4), "flip": int(fl),
         "shift_units": [round(float(v), 3) for v in t], "best": round(o["best"], 4), "gap": round(o["gap"], 4),
         "certified_search": o["certified"], "solve_s": round(o["secs"], 1), "ladder": []}
    if o["certified"]:
        for deg_z, rd in LADDER[:1] + [(20.0, 150.0), (20.0, 100.0)] + LADDER[1:]:
            if rd is None:
                u = far_par(al, o["best"], th, fl, exclude_deg=deg_z, n_proc=NP, time_limit=TL / 2)
            else:
                u = far_par_ext(al, o["best"], th, fl, t, exclude_deg=deg_z, shift_radius=rd, n_proc=NP,
                                time_limit=TL / 2)
            r["ladder"].append({"zone_deg": deg_z, "zone_shift_units": rd, "eps": round(u["eps"], 4),
                                "complete": u["complete"], "zone_move_um": round(zone_move_um(A * UNIT, deg_z, rd), 1)})
            if u["eps"] <= 0:
                break
    ok = [s for s in r["ladder"] if s["eps"] > 0]
    r["certified"] = bool(ok)
    r["finest_zone"] = [ok[-1]["zone_deg"], ok[-1]["zone_shift_units"]] if ok else None
    deg = float(np.degrees(th)) % 360
    if k == 0:
        from run_xx_prereg import landmark_error
        r["landmark_um_coarse"] = landmark_error(A, B, deg, fl, t)
    else:
        from run_baselines_scale import GeneScore
        gs = GeneScore(A, XA, B, XB, rad)
        r["gene_coarse"] = round(gs(deg, fl, np.asarray(t)), 4)
    if r["certified"] and r["finest_zone"][1] is not None:
        f, (th2, t2) = fine_stage(A, la, B, lb, th, fl, t, *r["finest_zone"], n_proc=NP, time_limit=TL)
        r["fine"] = f
        d2 = float(np.degrees(th2)) % 360
        r["fine_angle"] = round(d2, 4); r["fine_shift_units"] = [round(float(v), 3) for v in t2]
        if k == 0:
            r["landmark_um_fine"] = landmark_error(A, B, d2, fl, t2)
        else:
            r["gene_fine"] = round(gs(d2, fl, np.asarray(t2)), 4)
    r["total_s"] = round(time.time() - t0, 1)
    json.dump([r], open(OUT, "w"), indent=1); print(json.dumps(r), flush=True)
    print("DONE", flush=True)
