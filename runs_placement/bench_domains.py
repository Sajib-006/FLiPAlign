"""Does feeding TISSUE REGIONS (not cell types) make the certificate meaningful at scale?

At full Xenium scale the proof gap was 0.8-1.0 (vacuous) because graph clusters are cell
types mixed cell-by-cell. Here each cell is instead labelled by its NEIGHBOURHOOD: the mix of
cell types among its k nearest cells, clustered with k-means (standard cellular-neighbourhood
analysis, Schurch et al. 2020). Regions become large and smooth - the input the method was
built for. Each slice gets its own independent clustering, so labels never correspond.

Part 1 - ground truth: two random halves of Rep1, known 73 deg + mirror flip + shift on B.
Part 2 - the REAL pair: Rep1 vs Rep2 (adjacent sections, different slides).

Reports accuracy, time, proven gap, and the margin between the best rotation and the best
rotation more than 20 deg away (a second mode = ambiguity). Saves after every run.
"""
import os, sys  # noqa: E401
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import repo_paths  # noqa: E402,F401  (puts the code folders on the import path)
import numpy as np, pandas as pd, json, time, sys, os
sys.path.insert(0, ".")
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from certalign import rot, FLIP
from alignfast import ExhAlign

OUT = "results_domains.json"
res = json.load(open(OUT)) if os.path.exists(OUT) else []


def save(r):
    res.append(r)
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(r), flush=True)


def load(R):
    c = pd.read_csv(f"data/xenium/{R}/cells.csv.gz")
    cl = pd.read_csv(f"data/xenium/{R}/analysis/clustering/gene_expression_graphclust/clusters.csv")
    m = c.merge(cl, left_on="cell_id", right_on="Barcode")
    xy = m[["x_centroid", "y_centroid"]].values.astype(float)
    d, _ = cKDTree(xy).query(xy[::20], k=2)
    sp = float(np.median(d[:, 1]))
    return (xy - xy.mean(0)) / sp, m["Cluster"].values - 1


def neighbourhood_domains(xy, lab, k=30, K=8, seed=0):
    """cell -> mix of cell types among its k nearest neighbours -> k-means into K regions"""
    T = lab.max() + 1
    _, nb = cKDTree(xy).query(xy, k=k)
    comp = np.zeros((len(xy), T))
    for j in range(k):
        comp[np.arange(len(xy)), lab[nb[:, j]]] += 1.0
    comp /= k
    return KMeans(K, n_init=4, random_state=seed).fit_predict(comp)


def smoothness(xy, d):
    _, nb = cKDTree(xy).query(xy, k=7)
    return float(np.mean(d[nb[:, 1:]] == d[:, None]))


def angerr(a, b):
    return float(abs(np.degrees(np.angle(np.exp(1j * (a - b))))))


def scan(ex, n_theta, bound_every=16):
    """rotation scan with translation exhausted; keeps the per-rotation profile"""
    prof = []
    eps = 0.0
    k = 0
    for flip in (0, 1):
        for th in np.linspace(0, 2 * np.pi, n_theta, endpoint=False):
            s, t = ex.best_for_rotation(th, flip)
            prof.append((s, th, t, flip))
            if k % bound_every == 0:
                eps = max(eps, ex.eps_for_step(th, flip, np.pi / n_theta))
            k += 1
    best = max(prof, key=lambda z: z[0])
    far = [p[0] for p in prof
           if p[3] != best[3] or angerr(p[1], best[1]) > 20.0]
    return best, eps, float(max(far)) if far else float("nan")


if __name__ == "__main__":
    H = 5.0
    xy1, lab1 = load("Rep1")
    print(f"Rep1: {len(xy1)} cells", flush=True)

    # ---------- part 1: ground truth, full size ----------
    TH, SHIFT, FL = np.radians(73.0), np.array([10.0, -6.0]), 1
    rng = np.random.default_rng(1)
    idx = rng.permutation(len(xy1))
    iA, iB = idx[: len(idx) // 2], idx[len(idx) // 2:]
    for K in (6, 10):
        t0 = time.time()
        dA = neighbourhood_domains(xy1[iA], lab1[iA], K=K, seed=0)
        dB = neighbourhood_domains(xy1[iB], lab1[iB], K=K, seed=1)   # independent run
        tdom = time.time() - t0
        A = xy1[iA]
        B = (rot(TH) @ (FLIP if FL else np.eye(2)) @ xy1[iB].T).T + SHIFT
        ex = ExhAlign(A, dA, B, dB, h=H, tpad=30.0, mask_dist=2.0)
        n_theta = int(np.ceil(2 * np.pi / (H / ex.rA.max())))
        t0 = time.time()
        (s, th, t, fl), eps, far = scan(ex, n_theta)
        save({"part": "ground truth (Rep1 halves)", "labels": f"neighbourhood regions K={K}",
              "cells A/B": [int(len(A)), int(len(B))],
              "region smoothness A": round(smoothness(A, dA), 3),
              "domain_s": round(tdom, 1), "solve_s": round(time.time() - t0, 1),
              "score": round(s, 4), "angle_err": round(angerr(th, TH), 3),
              "flip_ok": int(fl == FL), "proven_gap": round(eps, 4),
              "best_far_score": round(far, 4), "margin": round(s - far, 4)})

    # ---------- part 2: the REAL pair, Rep1 vs Rep2 ----------
    xy2, lab2 = load("Rep2")
    print(f"Rep2: {len(xy2)} cells", flush=True)
    for K in (6, 10):
        d1 = neighbourhood_domains(xy1, lab1, K=K, seed=0)
        d2 = neighbourhood_domains(xy2, lab2, K=K, seed=1)
        ex = ExhAlign(xy1, d1, xy2, d2, h=H, tpad=60.0, mask_dist=2.0)
        n_theta = int(np.ceil(2 * np.pi / (H / ex.rA.max())))
        t0 = time.time()
        (s, th, t, fl), eps, far = scan(ex, n_theta)
        save({"part": "REAL pair Rep1 vs Rep2", "labels": f"neighbourhood regions K={K}",
              "cells": [int(len(xy1)), int(len(xy2))],
              "solve_s": round(time.time() - t0, 1), "score": round(s, 4),
              "angle_deg": round(float(np.degrees(th)) % 360, 2), "flip": int(fl),
              "shift": [round(float(v), 1) for v in (t - ex.tc)],
              "proven_gap": round(eps, 4), "best_far_score": round(far, 4),
              "margin": round(s - far, 4)})
    print("DONE", flush=True)
