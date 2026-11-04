"""E3 + G1: Xenium Rep1 -> Rep2 under the PRE-REGISTERED rule, common unit, shift saved, and the
landmark error against the STalign authors' hand-placed landmarks.

Rule (fixed 2026-09-26, before this run): contrast on, regions K=6 (seeds 0 / 1 as in
run_compare.py), sigma 20 cells, h 2 cells, window +-300, parallel. Uniqueness: far-shift check
at 30 cells (contains the rotation check); if it fails, 60 cells, then rotation only.
Landmark error = distance (um) between where our placement puts each Rep1 landmark and the
matching Rep2 landmark (3 pairs, landmarks/xenium_rep1_rep2_landmarks.json).
"""
import json, sys, time
import numpy as np
sys.path.insert(0, ".")
from smoothalign import SmoothAlign, rot, FLIP
from smoothpar import solve_par, far_par, far_par_ext
from bench_domains import neighbourhood_domains
import run_visium as RV
from run_baselines_scale import GeneScore

UNIT = 9.44
LM = json.load(open("landmarks/xenium_rep1_rep2_landmarks.json"))


def landmark_error(A, B, deg, fl, t):
    """A, B: all cells (units) used by the method; placement P = R(a - mean A) + mean B + t."""
    M = rot(np.radians(deg)) @ (FLIP if fl else np.eye(2))
    p1 = np.asarray(LM["rep1_xy"]) / UNIT
    P = (p1 - A.mean(0)) @ M.T + B.mean(0) + np.asarray(t)
    return (np.linalg.norm(P * UNIT - np.asarray(LM["rep2_xy"]), axis=1)).round(1).tolist()


if __name__ == "__main__":
    xy1, l1 = RV.xenium("Rep1"); xy2, l2 = RV.xenium("Rep2")
    A, B = xy1 / UNIT, xy2 / UNIT
    d1 = neighbourhood_domains(A, l1, K=6, seed=0); d2 = neighbourhood_domains(B, l2, K=6, seed=1)
    t0 = time.time()
    al = SmoothAlign(A, d1, B, d2, h=2.0, sigma=20.0, tpad=300.0, contrast=True)
    o = solve_par(al, n_proc=14, time_limit=7200)
    th, t, fl = o["transform"]
    deg = float(np.degrees(th)) % 360
    X1, _ = RV.expr("Rep1", RV.cells("Rep1")); X2, _ = RV.expr("Rep2", RV.cells("Rep2"))
    gs = GeneScore(A, np.expm1(X1), B, np.expm1(X2), None)
    r = {"pair": "XX", "rule": "pre-registered h=2", "angle": round(deg, 3), "flip": int(fl),
         "shift": [round(float(v), 2) for v in t], "objective": round(o["best"], 4),
         "gap": round(o["gap"], 4), "certified": o["certified"], "solve_s": round(o["secs"], 1),
         "gene_score": round(gs(deg, fl, np.asarray(t)), 4),
         "landmark_error_um": landmark_error(A, B, deg, fl, t)}
    r["landmark_error_median_um"] = float(np.median(r["landmark_error_um"]))
    json.dump(r, open("results_xx_prereg.json", "w"), indent=1); print(json.dumps(r), flush=True)
    if o["certified"]:
        ue = far_par_ext(al, o["best"], th, fl, t, shift_radius=30.0, n_proc=14, time_limit=5400)
        r.update({"eps_shift30": round(ue["eps"], 4), "unique_shift30": bool(ue["eps"] > 0),
                  "shift30_complete": ue["complete"]})
        if ue["eps"] <= 0:
            ue = far_par_ext(al, o["best"], th, fl, t, shift_radius=60.0, n_proc=14, time_limit=5400)
            r.update({"eps_shift60": round(ue["eps"], 4), "unique_shift60": bool(ue["eps"] > 0)})
            if ue["eps"] <= 0:
                u = far_par(al, o["best"], th, fl, n_proc=14, time_limit=5400)
                r.update({"eps_rotation": round(u["eps"], 4), "unique_rotation": bool(u["eps"] > 0)})
    r["total_s"] = round(time.time() - t0, 1)
    json.dump(r, open("results_xx_prereg.json", "w"), indent=1); print(json.dumps(r), flush=True)
    print("DONE", flush=True)
