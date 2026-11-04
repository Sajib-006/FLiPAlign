"""Size of the tolerance lambda of the completeness theorem: the largest value of E(s) at a leaf.
For each case: lambda_ub = (3/2) rho max_c D(c) (the bound in the theorem), and the largest E(s) over all
shifts at 24 random leaf ranges (rotation half-width delta_min, random angle and flip), with the best
score at the same ranges for scale. Usage: python3 checks/lambda_check.py run FIRST LAST   Env: GT_DIR, GT_OUT"""
import os, sys  # noqa: E401
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import repo_paths  # noqa: E402,F401  (puts the code folders on the import path)
import json, os, sys
import numpy as np
from scipy import fft as sfft
sys.path.insert(0, ".")
from smoothalign import SmoothAlign
from bench_gt import UNIT

D = os.environ.get("GT_DIR", "data/gt3"); out = os.environ.get("GT_OUT", "results_lambda.json")
res = json.load(open(out)) if os.path.exists(out) else []
done = {r["case"] for r in res}
for i in range(int(sys.argv[2]), int(sys.argv[3]) + 1):
    f = f"{D}/case_{i:02d}.npz"
    if i in done or not os.path.exists(f):
        continue
    z = np.load(f)
    al = SmoothAlign(z["Aobs"] / UNIT, z["la"], z["B"] / UNIT, z["lb"], h=2.0, sigma=20.0, tpad=300.0, contrast=True,
                     workers=int(os.environ.get("N_PROC", "8")))
    rho = al.h / np.sqrt(2); dmin = rho / (2 * al.rmax)
    rng = np.random.default_rng(700 + i); emax, ebest, sbest = [], [], []
    for _ in range(24):
        th, fl = rng.uniform(0, 2 * np.pi), int(rng.integers(2))
        s, t, ub = al.rotation_eval(th, fl, dmin)
        P = al._points(th, fl); x = (P - al.lo) / al.h; i0 = np.floor(x).astype(int)
        m = 2 * al.rA * np.sin(dmin / 2) + rho
        Dil, FD = al._dilated(float(m.max()))
        W = np.bincount(i0[:, 0] * al.shape[1] + i0[:, 1], m, minlength=al.shape[0] * al.shape[1]).reshape(al.shape)
        E = al._window(al._corr(sfft.rfft2(W, workers=al.workers), FD)).ravel() / al.n
        emax.append(float(E.max())); ebest.append(float(ub - s)); sbest.append(float(s))
    r = {"case": i, "lambda_ub": float(1.5 * rho * al.Dcell.max()), "lambda_leaf_max": max(emax), "lambda_leaf_median": float(np.median(emax)),
         "slack_at_best_shift_median": float(np.median(ebest)), "score_median_random_rotation": float(np.median(sbest)), "D_max": float(al.Dcell.max())}
    res.append(r); json.dump(res, open(out, "w"), indent=1); print(json.dumps(r), flush=True)
print("DONE")
