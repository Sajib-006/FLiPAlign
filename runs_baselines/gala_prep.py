"""Dump the inputs of one real pair for GALA (runs in the old base env that has the loaders).
Usage: python3 runs_baselines/gala_prep.py K  -> results_gala/pairK_in.npz
Same inputs as the other tools: A, B coordinates (um), the 300-gene expression matrices as the loaders give them (log-normalised
here as for CAST and spateo), the landmarks (um) and the truth fit."""
import os, sys  # noqa: E401
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import repo_paths  # noqa: E402,F401  (puts the code folders on the import path)
import sys, numpy as np
sys.path.insert(0, ".")
import bench_baselines as BB
from landmark_common import *
k = int(sys.argv[1]); c = load_pair(k); A, B = c["A"], c["B"]
T = truth(k, A, B, HOME); LA, LB = landmarks(k, HOME)
np.savez_compressed(f"results_gala/pair{k}_in.npz", A=A * UNIT, B=B * UNIT, XA=BB.BL.lognorm(c["XA"]).astype(np.float32),
                    XB=BB.BL.lognorm(c["XB"]).astype(np.float32), LA=LA * UNIT, LB=LB * UNIT, floor=T["floor_um"],
                    R=T["R"], tau=T["tau"])
print("OK", k, A.shape, B.shape, c["XA"].shape, c["XB"].shape, T["floor_um"], flush=True)
