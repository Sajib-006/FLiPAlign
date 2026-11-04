"""Published tools on the real pairs, scored on the clicked landmarks. Usage: python3 runs_baselines/bench_landmark_tools.py PAIR ARM methods...
ARM: given (nothing handed over) | rough (true flip, angle within +-20 degrees, centres matched) |
     pose (rough + position within +-500 um). Output: results/lm_<ARM>_pair<PAIR>_<tag>.json (GT_BASE_OUT) with
landmark_err_median_um per method and fit. Env: ROOT, GT_BASE_OUT"""
import os, sys  # noqa: E401
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import repo_paths  # noqa: E402,F401  (puts the code folders on the import path)
import json, os, sys
import numpy as np
sys.path.insert(0, ".")
import bench_baselines as BB
from landmark_common import *
k, arm, wanted = int(sys.argv[1]), sys.argv[2], sys.argv[3:]
root = os.environ.get("ROOT", HOME); out = os.environ.get("GT_BASE_OUT", f"{root}/results/lm_{arm}_pair{k}.json")
c = load_pair(k); A0, B = c["A"], c["B"]; T = truth(k, A0, B, root); LA, LB = landmarks(k, root)
c["A_raw"] = A0
if arm == "given":
    A1, L1 = A0, LA
else:
    rg = np.random.default_rng(41000 + k); e = rg.uniform(-20.0, 20.0)
    M0 = rot(np.radians(e)) @ T["R"]
    shift = np.zeros(2) if arm == "rough" else T["shift_true"] + rg.uniform(-500, 500, 2) / UNIT
    A1 = (A0 - A0.mean(0)) @ M0.T + B.mean(0) + shift; L1 = (LA - A0.mean(0)) @ M0.T + B.mean(0) + shift
    c["A"] = A1
c["arm"] = arm
c["start_err_um"] = round(float(np.median(np.linalg.norm(L1 - LB, axis=1)) * UNIT), 1)
extra = lambda deg, fl, t: {"landmark_err_median_um": round(float(np.median(lm_error(L1, LB, A1.mean(0), B.mean(0), deg, fl, t))), 1),
                            "floor_um": round(T["floor_um"], 1), "pair": k, "arm": arm}
BB.run_case(k, c, wanted, out, extra_fn=extra)
print("DONE", flush=True)
