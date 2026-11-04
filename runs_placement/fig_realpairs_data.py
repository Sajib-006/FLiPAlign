"""Data for the real-pairs figure: the seven real cross-platform pairs after FLiPAlign.
For each pair: subsampled cells of A (placed by FLiPAlign's frozen coarse answer) and of B, their own labels, and the
hand-placed landmarks (A's mapped by the same answer, B's as clicked). Coordinates in um, frame of B.
Usage (repository root, needs data/): python3 runs_placement/fig_realpairs_data.py -> results/placement/fig_realpairs.npz"""
import os, sys  # noqa: E401
try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import repo_paths  # noqa: E402,F401  (puts the code folders on the import path)
except ImportError:
    sys.path.insert(0, ".")
import json
import numpy as np
from landmark_common import load_pair, landmarks, UNIT
from smoothalign import rot, FLIP

OUT = os.environ.get("OUT_NPZ", "results/placement/fig_realpairs.npz")
RES = os.environ.get("RESULTS", "results")
NMAX = 9000
rng = np.random.default_rng(0)
out = {}
for k in range(1, 8):
    c = load_pair(k); A, B = c["A"], c["B"]
    r = json.load(open(f"{RES}/lm_flip_pair{k}.json")) if os.path.exists(f"{RES}/lm_flip_pair{k}.json") else \
        json.load(open(f"{RES}/placement/lm_flip_pair{k}.json"))
    deg, fl, t = r["coarse"]["angle"], r["coarse"]["flip"], np.asarray(r["coarse"]["shift"], float)
    M = rot(np.radians(deg)) @ (FLIP if fl else np.eye(2))
    Abar, Bbar = A.mean(0), B.mean(0)
    place = lambda P: (P - Abar) @ M.T + Bbar + t
    LA, LB = landmarks(k)
    ia = rng.choice(len(A), min(NMAX, len(A)), replace=False); ib = rng.choice(len(B), min(NMAX, len(B)), replace=False)
    out[f"p{k}_A"] = (place(A[ia]) * UNIT).astype(np.float32); out[f"p{k}_la"] = np.asarray(c["la"])[ia].astype(np.int16)
    out[f"p{k}_B"] = (B[ib] * UNIT).astype(np.float32); out[f"p{k}_lb"] = np.asarray(c["lb"])[ib].astype(np.int16)
    out[f"p{k}_LA"] = (place(LA) * UNIT).astype(np.float32); out[f"p{k}_LB"] = (LB * UNIT).astype(np.float32)
    err = np.linalg.norm(out[f"p{k}_LA"] - out[f"p{k}_LB"], axis=1)
    print(k, r["name"], "median err (um) recomputed", round(float(np.median(err)), 1), "saved", r["coarse"]["median_um"], flush=True)
np.savez_compressed(OUT, **out)
print("wrote", OUT)
