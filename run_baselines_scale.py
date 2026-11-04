"""Baselines at full scale, on the two real pairs where we have gene-confirmed answers:

  XX   Xenium Rep1 -> Rep2          (ours: 357.5 deg, no flip, proven unique)
  VX1  Visium -> Xenium Rep1        (ours: ~273.1 deg, no flip)
  VX2  Visium -> Xenium Rep2        (ours: ~271.0 deg, no flip)

Methods:
  local  8 random starts, Nelder-Mead on OUR contrast score (same objective, only the
         search differs). Tests whether the global search is needed at all.
  paste  PASTE-style fused Gromov-Wasserstein (POT) on 4,000 subsampled cells/spots per
         slide, expression PCA as the feature term, rigid transform by weighted Procrustes
         (reflections allowed), as in run_baselines.py. Visium vs Xenium uses the 307 shared
         genes. Balanced OT assumes full overlap, which these pairs do not have (noted).

Every answer is scored by the GENE-ONLY check (never used by any method): local expression
similarity at matched positions, each slide's mean removed. Plus angle error to our
gene-confirmed answer.
Usage: python3 run_baselines_scale.py local|paste
"""
import json, os, sys, time
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.spatial import cKDTree
sys.path.insert(0, ".")
from smoothalign import SmoothAlign, rot, FLIP
from bench_domains import load, neighbourhood_domains
import run_visium as RV

OUT = "results_baselines_scale.json"
REF = {"XX": (357.5, 0), "VX1": (273.1, 0), "VX2": (271.0, 0)}
# our answers with shifts (same convention): XX regions h=2 sigma 20 (proven unique);
# VX1 / VX2 regions sigma 20 h=4 (gene-confirmed, not proven unique)
OURS = {"XX": (357.5, 0, [36.0, -128.0]), "VX1": (273.13, 0, [28.0, 120.0]), "VX2": (270.87, 0, [66.0, -12.0])}
rng = np.random.default_rng(0)


def save(r):
    res = json.load(open(OUT)) if os.path.exists(OUT) else []
    res.append(r); json.dump(res, open(OUT, "w"), indent=1); print(json.dumps(r), flush=True)


def angdiff(a, b):
    return float(abs(np.degrees(np.angle(np.exp(1j * np.radians(a - b))))))


def lognorm(X):
    X = np.asarray(X, float)
    return np.log1p(X / np.maximum(X.sum(1, keepdims=True), 1) * 1e3)


# ---------------- data for the three pairs ----------------
def pairs():
    """yields name, A xy, A labels, A expr (cells x genes), B xy, B labels, B expr, gene radius"""
    xy1, lab1 = load("Rep1"); xy2, lab2 = load("Rep2")
    X1, g1 = RV.expr("Rep1", RV.cells("Rep1")); X2, g2 = RV.expr("Rep2", RV.cells("Rep2"))
    X1, X2 = np.expm1(X1), np.expm1(X2)                   # back to per-cell normalised counts
    d1 = neighbourhood_domains(xy1, lab1, K=6, seed=0)
    d2 = neighbourhood_domains(xy2, lab2, K=6, seed=1)
    yield "XX", xy1, d1, X1, xy2, d2, X2, None
    common = [g for g in g1 if g in set(RV.genesV)]
    iX = pd.Series(np.arange(len(g1)), index=g1)[common].values
    iVs = pd.Series(np.arange(len(RV.genesV)), index=RV.genesV)
    iV = iVs[~iVs.index.duplicated()][common].values
    XV = RV.XV[:, iV].toarray()
    for name, R, xy_c, d_c, X_c in (("VX1", "Rep1", xy1, d1, X1), ("VX2", "Rep2", xy2, d2, X2)):
        xy_um, _ = RV.xenium(R)
        dd, _ = cKDTree(xy_um).query(xy_um[::20], k=2); um = float(np.median(dd[:, 1]))
        # load() gives the same cells centred and in the same units; Visium goes to those units
        A = RV.xyV_um / um
        yield name, A, RV.labV, XV, xy_c, d_c, X_c[:, iX], 27.5 / um * 1.5


# ---------------- gene-only yardstick ----------------
class GeneScore:
    def __init__(self, A, XA, B, XB, rad=None, n=8000):
        self.A, self.cA, self.cB = A, A.mean(0), B.mean(0)
        self.idx = rng.choice(len(A), min(n, len(A)), replace=False)
        self.TB, self.XB, self.rad = cKDTree(B), lognorm(XB), rad
        self.muB = self.XB.mean(0)
        if rad is None:                                    # cell data: 15-NN local profile
            _, nb = cKDTree(A).query(A[self.idx], k=15)
            LA = lognorm(XA)[nb].mean(1)
        else:                                              # spot data: the spot itself
            LA = lognorm(XA[self.idx])
        LA = LA - lognorm(XA).mean(0)
        self.LA = (LA - LA.mean(1, keepdims=True)) / (LA.std(1, keepdims=True) + 1e-9)

    def __call__(self, th_deg, flip, t):
        P = (rot(np.radians(th_deg)) @ (FLIP if flip else np.eye(2)) @
             (self.A[self.idx] - self.cA).T).T + self.cB + t
        if self.rad is None:
            d, nb = self.TB.query(P, k=15); ok = d[:, 0] < 3.0
            LB = self.XB[nb[ok]].mean(1)
        else:
            nbl = self.TB.query_ball_point(P, self.rad)
            ok = np.array([len(x) >= 3 for x in nbl])
            LB = np.array([self.XB[x].mean(0) for x, o in zip(nbl, ok) if o])
        if ok.sum() < 50:
            return float("nan")
        LB = LB - self.muB
        LB = (LB - LB.mean(1, keepdims=True)) / (LB.std(1, keepdims=True) + 1e-9)
        return float((self.LA[ok] * LB).mean(1).mean())


# ---------------- baselines ----------------
def local_search(A, la, B, lb, n_starts=8, tpad=300.0):
    al = SmoothAlign(A, la, B, lb, h=4.0, sigma=20.0, tpad=tpad, contrast=True)
    out = []
    for s in range(n_starts):
        fl = int(rng.integers(2)); x0 = np.array([rng.uniform(0, 360), *rng.uniform(-tpad, tpad, 2)])
        f = lambda x: -al.exact_score(np.radians(x[0]), fl, np.clip(x[1:], -tpad, tpad))
        simplex = np.array([x0, x0 + [15, 0, 0], x0 + [0, 40, 0], x0 + [0, 0, 40]])
        r = minimize(f, x0, method="Nelder-Mead",
                     options={"initial_simplex": simplex, "maxfev": 300, "xatol": 0.5, "fatol": 1e-4})
        out.append((-r.fun, r.x[0] % 360, fl, np.clip(r.x[1:], -tpad, tpad)))
    return out


def paste(A, XA, B, XB, n=4000, alpha=0.1):
    import ot
    from sklearn.decomposition import PCA
    ia = rng.choice(len(A), min(n, len(A)), replace=False)
    ib = rng.choice(len(B), min(n, len(B)), replace=False)
    ZA, ZB = lognorm(XA[ia]), lognorm(XB[ib])
    ZA -= ZA.mean(0); ZB -= ZB.mean(0)                     # platform/batch offset removed
    pca = PCA(30, random_state=0).fit(np.vstack([ZA, ZB]))
    eA, eB = pca.transform(ZA), pca.transform(ZB)
    a, b = A[ia], B[ib]
    DA = ot.dist(a, a); DB = ot.dist(b, b); DA /= DA.max(); DB /= DB.max()
    M = ot.dist(eA, eB); M /= M.max()
    pi = ot.gromov.fused_gromov_wasserstein(M, DA, DB, ot.unif(len(a)), ot.unif(len(b)),
                                            loss_fun="square_loss", alpha=alpha)
    wa, wb = pi.sum(1), pi.sum(0)
    ca = (wa[:, None] * a).sum(0) / wa.sum(); cb = (wb[:, None] * b).sum(0) / wb.sum()
    U, S, Vt = np.linalg.svd((a - ca).T @ pi @ (b - cb))
    R = Vt.T @ U.T
    tvec = cb - R @ ca
    flip = int(np.linalg.det(R) < 0)
    ang = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])) % 360)
    t = R @ A.mean(0) + tvec - B.mean(0)                   # into our shift convention
    return ang, flip, t


if __name__ == "__main__":
    mode = sys.argv[1]
    for name, A, la, XA, B, lb, XB, rad in pairs():
        gs = GeneScore(A, XA, B, XB, rad)
        ref = REF[name]
        t0 = time.time()
        if mode == "local":
            runs = local_search(A, la, B, lb)
            best = max(runs, key=lambda z: z[0])
            near = sum(1 for r in runs if r[2] == ref[1] and angdiff(r[1], ref[0]) < 5)
            save({"pair": name, "method": "local search, 8 starts, our contrast score",
                  "angle": round(best[1], 2), "flip": best[2],
                  "angle_err_vs_ours": round(angdiff(best[1], ref[0]), 2),
                  "flip_matches_ours": int(best[2] == ref[1]), "score": round(best[0], 4),
                  "starts_that_found_it": f"{near}/8", "gene_score": round(gs(best[1], best[2], best[3]), 4),
                  "all_starts (score, angle, flip)": [[round(r[0], 4), round(r[1], 1), r[2]] for r in runs],
                  "sec": round(time.time() - t0, 1)})
        else:
            ang, fl, t = paste(A, XA, B, XB)
            save({"pair": name, "method": "PASTE-style FGW, 4k subsample",
                  "angle": round(ang, 2), "flip": fl,
                  "angle_err_vs_ours": round(angdiff(ang, ref[0]), 2),
                  "flip_matches_ours": int(fl == ref[1]),
                  "gene_score": round(gs(ang, fl, t), 4), "sec": round(time.time() - t0, 1)})
        # yardstick rows: our answer and random placements, same gene score
        if mode == "local":
            oa, of, ot_ = OURS[name]
            save({"pair": name, "method": "yardstick: OUR answer (SmoothAlign, contrast)",
                  "angle": oa, "flip": of, "gene_score": round(gs(oa, of, np.array(ot_)), 4)})
            rnd = [gs(rng.uniform(0, 360), int(rng.integers(2)), rng.uniform(-150, 150, 2)) for _ in range(20)]
            save({"pair": name, "method": "yardstick: random placements (20)",
                  "gene_score_mean": round(float(np.nanmean(rnd)), 4),
                  "gene_score_max": round(float(np.nanmax(rnd)), 4)})
    print("DONE", flush=True)
