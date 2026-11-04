"""Data for panel (a) of the placement figure: one held-out case, placed by the truth, by FLiPAlign and by two
published tools (the setting each tool used in Table 2, true flip and a rough angle handed over).

Usage (from the repository root, with the held-out cases in data/gt4 and the placement results in a folder):
    python3 runs_placement/fig_case_data.py data/gt4 6 results/placement results/placement/fig_case.npz
Writes a small subsample (cells of A and B, labels, and A placed four ways) and checks that every placement
gives the median error stored in the results file (to within 1 um)."""
import glob
import json
import sys

import numpy as np

UNIT = 9.44                                   # um per unit, as in the benchmark scripts
FLIP = np.diag([1.0, -1.0])
SETN = "gt4"
TOOLS = [("spateo", "rs_gt4spateo", "spateo_sns", "plain"), ("PASTE2", "rs_gt4base", "paste2_s0.3", "trimmed")]
N_SUB = 15000


def rot(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s], [s, c]])


def place(A, deg, fl, t, B):
    M = rot(np.radians(deg)) @ (FLIP if fl else np.eye(2))
    return (A - A.mean(0)) @ M.T + B.mean(0) + t


def rows(res, prefix, case):
    out = []
    for f in glob.glob(f"{res}/{prefix}_t*.json"):
        out += [r for r in json.load(open(f)) if r.get("case") == case]
    return out


def main(case_dir, case, res, out):
    z = np.load(f"{case_dir}/case_{case:02d}.npz")
    Aobs, Atrue, la, B, lb = z["Aobs"], z["Atrue"], z["la"], z["B"], z["lb"]
    med = lambda P: float(np.median(np.linalg.norm(P - Atrue, axis=1)))
    placed, err, names = [Atrue], [0.0], ["Truth"]
    o = rows(res, f"{SETN}h2", case)[0]                                  # FLiPAlign, flip searched
    P = place(Aobs, o["angle"], o["flip"], np.asarray(o["shift_units"]) * UNIT, B)
    assert abs(med(P) - o["tre_um"]) < 1.0, (med(P), o["tre_um"])
    placed.append(P); err.append(o["tre_um"]); names.append("FLiPAlign")
    # rough-start arm: the tool was handed A already turned by the true angle (error drawn in [-20, 20] deg)
    # and flipped, centred on B (bench_baselines.preplace with ROUGH_START=20)
    W = np.linalg.inv(rot(float(z["th0"])) @ (FLIP if int(z["fl0"]) else np.eye(2)))
    e = np.random.default_rng(31000 + case).uniform(-20, 20)
    A0, B0 = Aobs / UNIT, B / UNIT
    A1 = (A0 - A0.mean(0)) @ (rot(np.radians(e)) @ W).T + B0.mean(0)
    for name, prefix, method, fit in TOOLS:
        r = [x for x in rows(res, prefix, case) if x["method"] == method and x["fit"] == fit][0]
        P = place(A1, r["angle"], r["flip"], np.asarray(r["shift"]), B0) * UNIT
        assert abs(med(P) - r["tre_um"]) < 1.0, (name, med(P), r["tre_um"])
        placed.append(P); err.append(r["tre_um"]); names.append(name)
    rg = np.random.default_rng(0)
    ia = rg.choice(len(Aobs), min(N_SUB, len(Aobs)), replace=False)
    ib = rg.choice(len(B), min(N_SUB, len(B)), replace=False)
    np.savez_compressed(out, A_obs=Aobs[ia], la=la[ia], B=B[ib], lb=lb[ib], placed=np.stack([p[ia] for p in placed]),
                        err_um=np.array(err), names=np.array(names), case=case, set=SETN)
    print(dict(zip(names, err)))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4])
