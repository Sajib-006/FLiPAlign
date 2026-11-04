"""Numerical check (review 5): the Fourier score against a direct sum of the interpolated maps, and the
bound against direct scores inside the range. 12 held-out cases x 8 random rotations.
Usage: python3 check_fft.py   Env: GT_DIR, GT_OUT"""
import json, os, sys
import numpy as np
sys.path.insert(0, ".")
from smoothalign import SmoothAlign
from bench_gt import UNIT
D = os.environ.get("GT_DIR", "data/gt3"); out = os.environ.get("GT_OUT", "results_check_fft.json")
res = []
for i in range(0, 48, 4):
    z = np.load(f"{D}/case_{i:02d}.npz")
    al = SmoothAlign(z["Aobs"] / UNIT, z["la"], z["B"] / UNIT, z["lb"], h=2.0, sigma=20.0, tpad=300.0, contrast=True)
    rng = np.random.default_rng(i); d_fft, viol, n_in = [], 0, 0
    for _ in range(8):
        th, fl, dl = rng.uniform(0, 2 * np.pi), int(rng.integers(2)), np.radians(rng.choice([2.0, 0.5, 0.1]))
        s, t, ub = al.rotation_eval(th, fl, dl)
        d_fft.append(abs(s - al.exact_score(th, fl, t)))
        for _ in range(40):                              # direct scores anywhere in the range never pass the bound
            v = al.exact_score(th + rng.uniform(-dl, dl), fl, rng.uniform(-300, 300, 2)); n_in += 1; viol += v > ub + 1e-12
    r = {"case": i, "max_abs_diff_fft_vs_direct": float(max(d_fft)), "bound_violations": int(viol), "direct_scores": n_in}
    res.append(r); json.dump(res, open(out, "w"), indent=1); print(json.dumps(r), flush=True)
print("DONE")
