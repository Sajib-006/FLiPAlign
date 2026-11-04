"""Full comparison table on the 3 real pairs, every method on IDENTICAL inputs.

Pairs: XX (Xenium Rep1 -> Rep2), VX1 (Visium -> Xenium Rep1), VX2 (Visium -> Xenium Rep2).
Coordinates: one common unit for ALL slides (UNIT um). Earlier runs scaled each Xenium slide
by its own cell spacing (9.44 vs 9.66 um, a 2.3% size mismatch); fixed here.

Methods (one row each; saved as soon as it finishes):
  ours            SmoothAlign, contrast, regions K=6, sigma 20, h 4 (+ proven-uniqueness check)
  ours_grid       same objective, plain 1-deg grid, no bound (no proof)
  local8          8 random starts, Nelder-Mead on our objective
  paste_a{0.1,0.5,0.9}   official PASTE (paste-bio), 4k cells/spots, KL dissimilarity
  paste_a0.1_8k   official PASTE, 8k
  gw              PASTE with alpha=1 (positions only, pure Gromov-Wasserstein)
  paste_refine    PASTE (alpha 0.1) answer, then Nelder-Mead on our objective
  paste2_s{0.5,0.7}      official PASTE2 (partial overlap), 4k, KL
  stalign         STalign (official, image LDDMM), identity start
  stalign8        STalign, 8 starts (4 rotations x 2 flips), best by density overlap (no genes)
Scores for every row: gene match score (0 = random, higher better, never used by any method),
angle off from our answer, flip, seconds.
Usage: PYTHONPATH=.pylibs_base MPLBACKEND=Agg python3 runs_placement/run_compare.py [method ...]
"""
import os, sys  # noqa: E401
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import repo_paths  # noqa: E402,F401  (puts the code folders on the import path)
import json, os, sys, time
import numpy as np, pandas as pd

# STalign draws progress figures with the account's old matplotlib, which breaks under
# NumPy 2 (np.Inf). Figures are not needed: make canvas drawing a no-op (faster too).
import matplotlib
matplotlib.use("Agg")
import matplotlib.backends.backend_agg as _agg
_agg.FigureCanvasAgg.draw = lambda self, *a, **k: None
if not hasattr(np, "Inf"):
    np.Inf = np.inf
from scipy.optimize import minimize
from scipy.spatial import cKDTree
sys.path.insert(0, ".")
from smoothalign import SmoothAlign, far_upper_bound, rot, FLIP
from bench_domains import neighbourhood_domains
import run_visium as RV
from run_baselines_scale import GeneScore, lognorm, angdiff

# --- compatibility shim for the official PASTE code (paste-bio 1.4.0) with POT 0.9.7 ---
# POT >= 0.9.2 calls line_search(cost, G, deltaG, Mi, cost_G, df_G, **kw); PASTE's line
# searches take no df_G. The shim only drops that extra argument; PASTE's math is unchanged.
import ot as _ot
_orig_cg = _ot.optim.cg
def _cg_compat(a, b, M, reg, f, df, G0=None, line_search=None, *args, **kw):
    if line_search is not None:
        _ls = line_search
        def line_search(cost, G, deltaG, Mi, cost_G, *extra, **kwargs):
            return _ls(cost, G, deltaG, Mi, cost_G, **kwargs)
    return _orig_cg(a, b, M, reg, f, df, G0, line_search, *args, **kw)
_ot.optim.cg = _cg_compat


OUT = "results_compare.json"
UNIT = 9.44
rng0 = 0


def save(r):
    res = json.load(open(OUT)) if os.path.exists(OUT) else []
    res.append(r); json.dump(res, open(OUT, "w"), indent=1); print(json.dumps(r), flush=True)


def pairs():
    xy1, l1 = RV.xenium("Rep1"); xy2, l2 = RV.xenium("Rep2")
    X1, g1 = RV.expr("Rep1", RV.cells("Rep1")); X2, _ = RV.expr("Rep2", RV.cells("Rep2"))
    X1, X2 = np.expm1(X1), np.expm1(X2)
    u1, u2 = xy1 / UNIT, xy2 / UNIT
    d1 = neighbourhood_domains(u1, l1, K=6, seed=0)
    d2 = neighbourhood_domains(u2, l2, K=6, seed=1)
    yield "XX", u1, d1, X1, list(g1), u2, d2, X2, None
    common = [g for g in g1 if g in set(RV.genesV)]
    iX = pd.Series(np.arange(len(g1)), index=g1)[common].values
    iVs = pd.Series(np.arange(len(RV.genesV)), index=RV.genesV)
    XV = RV.XV[:, iVs[~iVs.index.duplicated()][common].values].toarray().astype(float)
    uV = RV.xyV_um / UNIT
    rad = 27.5 / UNIT * 1.5
    yield "VX1", uV, RV.labV, XV, common, u1, d1, X1[:, iX], rad
    yield "VX2", uV, RV.labV, XV, common, u2, d2, X2[:, iX], rad


# ---------------- helpers ----------------
def rigid_from_points(src, dst, w=None):
    """weighted Procrustes, reflections allowed -> (angle deg, flip, R, tvec)"""
    w = np.ones(len(src)) if w is None else w
    cs = (w[:, None] * src).sum(0) / w.sum(); cd = (w[:, None] * dst).sum(0) / w.sum()
    U, S, Vt = np.linalg.svd(((src - cs) * w[:, None]).T @ (dst - cd))
    R = Vt.T @ U.T
    return float(np.degrees(np.arctan2(R[1, 0], R[0, 0])) % 360), int(np.linalg.det(R) < 0), R, cd - R @ cs


def rigid_from_coupling(a, b, pi):
    wa, wb = pi.sum(1), pi.sum(0)
    ca = (wa[:, None] * a).sum(0) / wa.sum(); cb = (wb[:, None] * b).sum(0) / wb.sum()
    U, S, Vt = np.linalg.svd((a - ca).T @ pi @ (b - cb))
    R = Vt.T @ U.T
    return float(np.degrees(np.arctan2(R[1, 0], R[0, 0])) % 360), int(np.linalg.det(R) < 0), R, cb - R @ ca


def to_shift(R, tvec, A, B):
    """our convention: P = R (a - mean A) + mean B + t"""
    return R @ A.mean(0) + tvec - B.mean(0)


def adata(xy, X, genes, idx):
    import anndata as ad
    a = ad.AnnData(X=np.asarray(X[idx], dtype=np.float32))
    a.var_names = genes; a.obsm["spatial"] = xy[idx]
    return a


def nm_refine(al, deg, flip, t, tpad=300.0, maxfev=400):
    f = lambda x: -al.exact_score(np.radians(x[0]), flip, np.clip(x[1:], -tpad, tpad))
    x0 = np.array([deg, *t])
    r = minimize(f, x0, method="Nelder-Mead", options={
        "initial_simplex": np.array([x0, x0 + [5, 0, 0], x0 + [0, 15, 0], x0 + [0, 0, 15]]),
        "maxfev": maxfev, "xatol": 0.2, "fatol": 1e-5})
    return r.x[0] % 360, np.clip(r.x[1:], -tpad, tpad), -r.fun


# ---------------- methods: each returns (deg, flip, t, extra) ----------------
def m_ours(ctx):
    al = ctx["al"]; out = al.solve(time_limit=5400)
    th, t, fl = out["transform"]
    ex = {"objective": round(out["best"], 4), "gap": round(out["gap"], 4), "certified": out["certified"]}
    if out["certified"]:
        u = far_upper_bound(al, out["best"], th, fl, time_limit=1800)
        ex.update({"eps_unique": round(u["eps"], 4), "unique_PROVEN": bool(u["eps"] > 0),
                   "far_complete": u["complete"]})
    ctx["ours"] = (np.degrees(th) % 360, fl)
    return np.degrees(th) % 360, fl, t, ex


def m_ours_grid(ctx):
    al = ctx["al"]; best = (-1, None)
    for fl in (0, 1):
        for d in np.arange(0, 360, 1.0):
            s, t, _ = al.rotation_eval(np.radians(d), fl, np.radians(0.5))
            if s > best[0]:
                best = (s, (d, fl, t))
    d, fl, t = best[1]
    return d, fl, t, {"objective": round(best[0], 4), "evals": 720}


def m_local8(ctx):
    al = ctx["al"]; rng = np.random.default_rng(0); runs = []
    for _ in range(8):
        fl = int(rng.integers(2))
        d, t, s = nm_refine(al, rng.uniform(0, 360), fl, rng.uniform(-300, 300, 2), maxfev=300)
        runs.append((s, d, fl, t))
    s, d, fl, t = max(runs, key=lambda z: z[0])
    ctx_ref = ctx.get("ours")
    hit = (sum(1 for r in runs if ctx_ref and r[2] == ctx_ref[1] and angdiff(r[1], ctx_ref[0]) < 5)
           if ctx_ref else None)
    return d, fl, t, {"objective": round(s, 4), "starts_that_found_ours": hit}


def _paste(ctx, alpha, n, key=None):
    import paste
    rng = np.random.default_rng(1)
    ia = rng.choice(len(ctx["A"]), min(n, len(ctx["A"])), replace=False)
    ib = rng.choice(len(ctx["B"]), min(n, len(ctx["B"])), replace=False)
    sA, sB = adata(ctx["A"], ctx["XA"], ctx["genes"], ia), adata(ctx["B"], ctx["XB"], ctx["genes"], ib)
    pi = paste.pairwise_align(sA, sB, alpha=alpha, dissimilarity="kl", numItermax=1000,
                              verbose=False, numItermaxEmd=1_000_000)
    pi = np.asarray(pi)
    d, fl, R, tv = rigid_from_coupling(ctx["A"][ia], ctx["B"][ib], pi)
    t = to_shift(R, tv, ctx["A"], ctx["B"])
    if key:
        ctx[key] = (d, fl, t)
    return d, fl, t, {"cells_used": [len(ia), len(ib)]}


def m_paste_refine(ctx):
    if "paste01" not in ctx:
        _paste(ctx, 0.1, 4000, key="paste01")
    d0, fl, t0 = ctx["paste01"]
    d, t, s = nm_refine(ctx["al"], d0, fl, t0)
    return d, fl, t, {"objective": round(s, 4), "start": [round(d0, 2), fl]}


def _paste2(ctx, s_frac, n=4000):
    from paste2 import PASTE2
    rng = np.random.default_rng(1)
    ia = rng.choice(len(ctx["A"]), min(n, len(ctx["A"])), replace=False)
    ib = rng.choice(len(ctx["B"]), min(n, len(ctx["B"])), replace=False)
    sA, sB = adata(ctx["A"], ctx["XA"], ctx["genes"], ia), adata(ctx["B"], ctx["XB"], ctx["genes"], ib)
    pi = np.asarray(PASTE2.partial_pairwise_align(sA, sB, s=s_frac, alpha=0.1,
                                                   dissimilarity="kl", verbose=False))
    d, fl, R, tv = rigid_from_coupling(ctx["A"][ia], ctx["B"][ib], pi)
    return d, fl, to_shift(R, tv, ctx["A"], ctx["B"]), {"cells_used": [len(ia), len(ib)], "s": s_frac}


def _density_corr(P, B, bins=5.0):
    lo = np.minimum(P.min(0), B.min(0)); hi = np.maximum(P.max(0), B.max(0))
    e = [np.arange(lo[k], hi[k] + bins, bins) for k in (0, 1)]
    h1, _, _ = np.histogram2d(P[:, 0], P[:, 1], e); h2, _, _ = np.histogram2d(B[:, 0], B[:, 1], e)
    return float(np.corrcoef(h1.ravel(), h2.ravel())[0, 1])


def _stalign(ctx, starts, niter):
    import torch
    import STalign.STalign as S
    A, B = ctx["A"] * UNIT, ctx["B"] * UNIT                       # STalign works in um
    XI, YI, I = S.rasterize(A[:, 0], A[:, 1], dx=30.0, draw=0)
    XJ, YJ, J = S.rasterize(B[:, 0], B[:, 1], dx=30.0, draw=0)
    I = np.vstack([I / I.max()] * 3); J = np.vstack([J / J.max()] * 3)
    swap = np.array([[0, 1], [1, 0]])
    best = None
    for deg0, fl0 in starts:
        M = rot(np.radians(deg0)) @ (FLIP if fl0 else np.eye(2))      # acts on (x, y)
        L = swap @ M @ swap                                            # STalign uses (y, x)
        T = B.mean(0)[::-1] - L @ A.mean(0)[::-1]
        out = S.LDDMM([YI, XI], I, [YJ, XJ], J, L=L, T=T, niter=niter, device="cpu",
                      sigmaM=1.5, sigmaB=1.0, sigmaA=1.5, epV=100,
                      muB=torch.tensor([0, 0, 0]), muA=torch.tensor([1, 1, 1]))
        P = S.transform_points_source_to_target(out["xv"], out["v"], out["A"],
                                                np.stack([A[:, 1], A[:, 0]], 1))
        P = np.asarray(P.detach().cpu() if hasattr(P, "detach") else P)[:, ::-1]
        crit = _density_corr(P, B, bins=50.0)
        if best is None or crit > best[0]:
            best = (crit, P, deg0, fl0)
    crit, P, deg0, fl0 = best
    d, fl, R, tv = rigid_from_points(A, P)
    return d, fl, to_shift(R, tv / UNIT, ctx["A"], ctx["B"]), {
        "picked_start": [deg0, fl0], "density_corr": round(crit, 4), "niter": niter}


METHODS = {
    "ours": m_ours,
    "ours_grid": m_ours_grid,
    "local8": m_local8,
    "paste_a0.1": lambda c: _paste(c, 0.1, 4000, key="paste01"),
    "paste_a0.5": lambda c: _paste(c, 0.5, 4000),
    "paste_a0.9": lambda c: _paste(c, 0.9, 4000),
    "paste_a0.1_8k": lambda c: _paste(c, 0.1, 8000),
    "gw": lambda c: _paste(c, 1.0, 4000),
    "paste_refine": m_paste_refine,
    "paste2_s0.5": lambda c: _paste2(c, 0.5),
    "paste2_s0.7": lambda c: _paste2(c, 0.7),
    "stalign": lambda c: _stalign(c, [(0.0, 0)], 2000),
    "stalign8": lambda c: _stalign(c, [(d, f) for f in (0, 1) for d in (0, 90, 180, 270)], 1000),
}

if __name__ == "__main__":
    wanted = sys.argv[1:] or list(METHODS)
    for name, A, la, XA, genes, B, lb, XB, rad in pairs():
        ctx = {"A": A, "XA": XA, "B": B, "XB": XB, "genes": genes,
               "al": SmoothAlign(A, la, B, lb, h=4.0, sigma=20.0, tpad=300.0, contrast=True)}
        gs = GeneScore(A, XA, B, XB, rad)
        rnd = [gs(np.random.default_rng(k).uniform(0, 360), k % 2,
                  np.random.default_rng(k + 99).uniform(-150, 150, 2)) for k in range(20)]
        save({"pair": name, "method": "random placements (20)",
              "gene_mean": round(float(np.nanmean(rnd)), 4), "gene_max": round(float(np.nanmax(rnd)), 4)})
        for m in wanted:
            t0 = time.time()
            try:
                d, fl, t, ex = METHODS[m](ctx)
                ref = ctx.get("ours")
                save({"pair": name, "method": m, "angle": round(float(d), 2), "flip": int(fl),
                      "off_from_ours_deg": round(angdiff(d, ref[0]), 2) if ref else None,
                      "flip_same_as_ours": int(fl == ref[1]) if ref else None,
                      "gene_score": round(gs(d, fl, np.asarray(t, float)), 4),
                      "sec": round(time.time() - t0, 1), **ex})
            except Exception as e:
                save({"pair": name, "method": m, "error": repr(e)[:300], "sec": round(time.time() - t0, 1)})
    print("DONE", flush=True)
