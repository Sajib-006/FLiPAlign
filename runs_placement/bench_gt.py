"""Known-answer benchmark, held-out set 1, built from real Xenium slides (known answer).

Each case, from one Xenium slide (Rep1 or Rep2, alternating):
  * cells split into two DISJOINT random halves (sides never share a cell, like serial
    sections);
  * the 313 genes split into two random halves; each side is clustered on ITS OWN half
    (no shared labels, no shared genes);
  * "spot" cases: side A is pooled into pseudo-Visium spots (55 um spots, 100 um hex grid,
    >= 3 cells) and clustered at spot level (like real Visium); "cell" cases: both sides
    are cells, labelled by neighbourhood regions (K=6) of their own clusters;
  * both sides cropped along the SAME random direction from opposite ends (real partial
    overlap, roughly 30-85% of A lies on B);
  * side A moved by a random rotation, mirror flip (50%) and shift (up to 150 cells).
Truth: every A cell/spot's original position. Error = median distance (um) between where a
method puts each A point and where it really is (target registration error, TRE).

Modes:
  make   write the cases to data/gt/case_XX.npz (+ manifest) so every method sees identical
         inputs (full 313-gene expression for both sides is stored for baselines).
  ours   run our method under the rule fixed in advance (contrast, regions K=6 on the cell
         side, sigma 20 cells, h 2 cells, window +-300, parallel), with the rotation
         uniqueness check and the far-shift check at 30 and 60 cells.
Usage: python3 runs_placement/bench_gt.py make|ours [first_case last_case]
"""
import os, sys  # noqa: E401
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import repo_paths  # noqa: E402,F401  (puts the code folders on the import path)
import json, os, sys, time
import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
sys.path.insert(0, ".")
from smoothalign import SmoothAlign, rot, FLIP
from smoothpar import solve_par, far_par, far_par_ext
from bench_domains import neighbourhood_domains

UNIT = 9.44
D = "data/gt"
N_CASES = 24
import os
OUT = os.environ.get("GT_OUT", "results_gt_ours.json")
TL = float(os.environ.get("GT_TIME_LIMIT", "5400"))


def lognorm(X):
    return np.log1p(X / np.maximum(X.sum(1, keepdims=True), 1) * 1e3)


def cluster(X, k, seed):
    Z = PCA(20, random_state=seed).fit_transform(lognorm(X))
    return KMeans(k, n_init=4, random_state=seed).fit_predict(Z)


def pseudo_visium(xy, X, rng):
    lo, hi = xy.min(0), xy.max(0)
    off = rng.uniform(0, 100, 2)
    cs = []
    for j, y in enumerate(np.arange(lo[1] + off[1], hi[1], 100 * np.sqrt(3) / 2)):
        xs = np.arange(lo[0] + off[0] + (50 if j % 2 else 0), hi[0], 100)
        cs.append(np.c_[xs, np.full(len(xs), y)])
    cs = np.vstack(cs)
    members = cKDTree(xy).query_ball_point(cs, 27.5)
    keep = np.array([len(m) >= 3 for m in members])
    S = np.array([X[m].sum(0) for m, k in zip(members, keep) if k])
    return cs[keep], S


def crop_pair(xyA, xyB, rng):
    """Crop both sides along the SAME random direction from opposite ends, so they only
    partly overlap: A keeps the low end up to quantile qA, B keeps the high end from qB."""
    u = rng.normal(size=2); u /= np.linalg.norm(u)
    pA, pB = xyA @ u, xyB @ u
    qA, qB = rng.uniform(0.6, 0.9), rng.uniform(0.1, 0.4)
    return pA <= np.quantile(pA, qA), pB >= np.quantile(pB, qB)


def make():
    import run_visium as RV
    os.makedirs(D, exist_ok=True)
    data = {}
    for R in ("Rep1", "Rep2"):
        xy, _ = RV.xenium(R)
        X, genes = RV.expr(R, RV.cells(R))
        data[R] = (xy, np.expm1(X), genes)                   # per-cell normalised counts
    manifest = []
    for i in range(N_CASES):
        rng = np.random.default_rng(1000 + i)
        R = ("Rep1", "Rep2")[i % 2]
        kind = "spot" if (i // 2) % 2 else "cell"
        xy, X, genes = data[R]
        perm = rng.permutation(len(xy)); hA, hB = perm[: len(xy) // 2], perm[len(xy) // 2:]
        g = rng.permutation(X.shape[1]); gA, gB = g[: len(g) // 2], g[len(g) // 2:]
        mA, mB = crop_pair(xy[hA], xy[hB], rng)
        # side B (cells, fixed): own genes -> clusters -> regions
        Bxy, BX = xy[hB][mB], X[hB][mB]
        lbB = neighbourhood_domains(Bxy / UNIT, cluster(BX[:, gB], 12, i), K=6, seed=i)
        # side A (moved): cells or pseudo-Visium spots, own genes
        Axy, AX = xy[hA][mA], X[hA][mA]
        if kind == "spot":
            Axy, AX = pseudo_visium(Axy, AX, rng)
            laA = cluster(AX[:, gA], 12, i + 7)
        else:
            laA = neighbourhood_domains(Axy / UNIT, cluster(AX[:, gA], 12, i + 7), K=6, seed=i + 7)
        # known move of side A
        th0 = rng.uniform(0, 2 * np.pi); fl0 = int(rng.integers(2))
        s0 = rng.uniform(-150, 150, 2) * UNIT
        M0 = rot(th0) @ (FLIP if fl0 else np.eye(2))
        cA = Axy.mean(0)
        Aobs = (Axy - cA) @ M0.T + cA + s0
        # overlap: share of A's true positions that lie on B tissue
        ov = float(np.mean(cKDTree(Bxy).query(Axy)[0] < 30))
        np.savez_compressed(f"{D}/case_{i:02d}.npz", Aobs=Aobs, Atrue=Axy, la=laA, AX=AX,
                            B=Bxy, lb=lbB, BX=BX, genes=np.asarray(genes), gA=gA, gB=gB,
                            th0=th0, fl0=fl0, s0=s0)
        m = {"case": i, "slide": R, "kind": kind, "nA": int(len(Axy)), "nB": int(len(Bxy)),
             "applied_deg": round(float(np.degrees(th0)), 2), "applied_flip": fl0,
             "overlap": round(ov, 3)}
        manifest.append(m); print(json.dumps(m), flush=True)
    json.dump(manifest, open(f"{D}/manifest.json", "w"), indent=1)


def tre(P, Atrue):
    return float(np.median(np.linalg.norm(P - Atrue, axis=1)))


def place(Aobs, th, fl, t, Bmean):
    """our convention: P = R F (a - mean A) + mean B + t (all in um here)"""
    M = rot(th) @ (FLIP if fl else np.eye(2))
    return (Aobs - Aobs.mean(0)) @ M.T + Bmean + t


def ours(first, last):
    res = json.load(open(OUT)) if os.path.exists(OUT) else []
    done = {r["case"] for r in res}
    for i in range(first, last + 1):
        if i in done:
            continue
        z = np.load(f"{D}/case_{i:02d}.npz")
        A, B = z["Aobs"] / UNIT, z["B"] / UNIT
        t0 = time.time()
        al = SmoothAlign(A, z["la"], B, z["lb"], h=2.0, sigma=20.0, tpad=300.0, contrast=True)
        o = solve_par(al, n_proc=14, time_limit=TL)
        th, t, fl = o["transform"]
        P = place(z["Aobs"], th, fl, np.asarray(t) * UNIT, z["B"].mean(0))
        # true answer: A came from Atrue by P0 = M0 (a - c) + c + s0, so undo it
        M0 = rot(float(z["th0"])) @ (FLIP if int(z["fl0"]) else np.eye(2))
        want = np.linalg.inv(M0)
        want_deg = float(np.degrees(np.arctan2(want[1, 0], want[0, 0])) % 360)
        r = {"case": i, "tre_um": round(tre(P, z["Atrue"]), 1),
             "angle_err": round(float(abs(np.degrees(np.angle(np.exp(1j * (th - np.radians(want_deg))))))), 3),
             "flip_ok": int(fl == int(z["fl0"])), "objective": round(o["best"], 4),
             "gap": round(o["gap"], 4), "certified": o["certified"], "solve_s": round(o["secs"], 1)}
        if o["certified"]:
            # adaptive (faster): the far-shift check's far set CONTAINS the rotation-only far
            # set, so eps_shift30 > 0 also proves rotation uniqueness. Only if it fails, try a
            # larger shift radius, then rotation only, to report how far the answer is pinned.
            ue = far_par_ext(al, o["best"], th, fl, t, shift_radius=30.0, n_proc=14, time_limit=3600)
            r.update({"eps_shift30": round(ue["eps"], 4), "unique_shift30": bool(ue["eps"] > 0),
                      "shift30_complete": ue["complete"]})
            if ue["eps"] <= 0:
                ue = far_par_ext(al, o["best"], th, fl, t, shift_radius=60.0, n_proc=14, time_limit=3600)
                r.update({"eps_shift60": round(ue["eps"], 4), "unique_shift60": bool(ue["eps"] > 0)})
                if ue["eps"] <= 0:
                    u = far_par(al, o["best"], th, fl, n_proc=14, time_limit=3600)
                    r.update({"eps_rotation": round(u["eps"], 4), "unique_rotation": bool(u["eps"] > 0)})
        r["total_s"] = round(time.time() - t0, 1)
        res.append(r); json.dump(res, open(OUT, "w"), indent=1); print(json.dumps(r), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "make":
        make()
    else:
        a = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        b = int(sys.argv[3]) if len(sys.argv) > 3 else N_CASES - 1
        ours(a, b)
    print("DONE", flush=True)
