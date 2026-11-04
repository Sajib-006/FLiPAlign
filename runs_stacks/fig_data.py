"""Data for the overview and stack figures (2026-10-22). Writes results/fig_data.npz (subsampled cells).
Part 1: held-out case gt3 case 0 (mouse brain, Prime 5K): A as given (turned and mirrored), B, and A placed by
FLiPAlign's certified answer; checked against A's true position.
Part 2: stack 1, clean: three consecutive sections with atlas division, own labels, one-to-one (forest d=2) labels and
split (cap 2) labels."""
import os, sys  # noqa: E401
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import repo_paths  # noqa: E402,F401  (puts the code folders on the import path)
import glob, json, sys
import numpy as np
sys.path.insert(0, ".")
from smoothalign import rot, FLIP
from bench_gt import UNIT
import stack_jlp as J
import stack_cap as SC

rng = np.random.default_rng(0)
out = {}
# ---------- part 1
z = np.load("data/gt3/case_00.npz", allow_pickle=True)
r = None
for f in glob.glob("results/gt3h2_t*.json"):
    for x in json.load(open(f)):
        if x.get("case") == 0: r = x
A, B, At = z["Aobs"] / UNIT, z["B"] / UNIT, z["Atrue"] / UNIT
print("answer", {k: r.get(k) for k in ("angle", "flip", "shift_units", "shift", "tre_um", "certified")})
t = np.asarray(r.get("shift_units", r.get("shift")), float)
M = rot(np.radians(r["angle"])) @ (FLIP if r["flip"] else np.eye(2))
best = None
for cA, cB in ((A.mean(0), B.mean(0)), (A.mean(0), np.zeros(2)), (np.zeros(2), np.zeros(2))):
    P = (A - cA) @ M.T + cB + t
    e = float(np.median(np.linalg.norm(P - At, axis=1)) * UNIT)
    print("convention", cA.round(1), cB.round(1), "median error um", round(e, 1))
    if best is None or e < best[0]: best = (e, P)
print("placed error um", round(best[0], 1), "reported tre", r.get("tre_um"))
ia = rng.choice(len(A), min(12000, len(A)), replace=False); ib = rng.choice(len(B), min(20000, len(B)), replace=False)
out.update(p_A=A[ia], p_la=z["la"][ia], p_B=B[ib], p_lb=z["lb"][ib], p_Aplaced=best[1][ia], p_err=best[0], p_eps=r["ladder"][0]["eps"],
           p_angle=r["angle"], p_flip=r["flip"])
# ---------- part 2
p, secs = J.sections(); stack = secs[1 * J.KS:2 * J.KS]
S = [J.load_section(p, sec, secs.index(sec)) for sec in stack]; del p
C = {(s, u): J.table(S[s], S[u]) for s in range(J.KS) for u in range(s + 1, min(s + 3, J.KS))}
sig, _ = J.forest(C, list(C), J.KS, 2)
z0 = np.concatenate([sig[s] for s in range(J.KS)])
E = [(s, u) for (s, u) in C if u - s <= 2]; W = SC.weights(C, E)
zc = SC.capped_search(W, z0, J.N, cap=2)
for j, s in enumerate((2, 3, 4)):
    sec = S[s]; ii = rng.choice(len(sec["xy"]), min(15000, len(sec["xy"])), replace=False)
    out[f"s{j}_xy"] = sec["xy"][ii]; out[f"s{j}_div"] = sec["div"][ii]; out[f"s{j}_own"] = sec["lab"][ii]
    out[f"s{j}_one"] = sig[s][sec["lab"][ii]]; out[f"s{j}_split"] = zc[s * J.N + sec["lab"][ii]]
np.savez_compressed("results/fig_data.npz", **out)
print("saved")
