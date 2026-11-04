"""One shared module for the published tools, used for the real pairs AND the ground-truth benchmark, so
every method sees identical inputs and returns a FULL transform.

Convention (same as ours): a placement is (deg, flip, t) meaning
    P = R(deg) F^flip (a - mean A) + mean B + t
Coordinates are in whatever unit the caller uses (the caller converts).

How the tools are run:
  * every method returns the shift, not only the angle;
  * correspondence methods (PASTE, PASTE2, SLAT) get two rigid fits: "plain" (the tool's
    usual weighted Procrustes) and "trimmed" (refit on the best-matching half, repeated),
    because partial overlap pulls a plain fit's shift toward the centroids;
  * STalign gets one image channel per cluster (it fits its own linear map between the two
    slides' channels, so unrelated labels are allowed) and a larger affine step, and is
    checked on known rotations before use.
Shims (math unchanged) are applied on import: POT line_search (PASTE), torch.cholesky
(GPSA), torch_geometric knn_graph (SLAT), matplotlib drawing off (STalign).
"""
import os, sys  # noqa: E401
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import repo_paths  # noqa: E402,F401  (puts the code folders on the import path)
import types
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.backends.backend_agg as _agg
_agg.FigureCanvasAgg.draw = lambda self, *a, **k: None
# STalign plots progress with imshow, which only accepts 1-4 channels; our label images have
# one channel per cluster. Plotting is not needed: make imshow a no-op (math unchanged).
import matplotlib.axes as _maxes
_maxes.Axes.imshow = lambda self, *a, **k: None
if not hasattr(np, "Inf"):
    np.Inf = np.inf
from scipy.optimize import minimize
from smoothalign import rot, FLIP

import ot as _ot
_orig_cg = _ot.optim.cg


def _cg_compat(a, b, M, reg, f, df, G0=None, line_search=None, *args, **kw):
    if line_search is not None:
        _ls = line_search
        def line_search(cost, G, deltaG, Mi, cost_G, *extra, **kwargs):
            return _ls(cost, G, deltaG, Mi, cost_G, **kwargs)
    return _orig_cg(a, b, M, reg, f, df, G0, line_search, *args, **kw)


_ot.optim.cg = _cg_compat


# ---------------- rigid fitting ----------------
def _procrustes(src, dst, w):
    cs = (w[:, None] * src).sum(0) / w.sum(); cd = (w[:, None] * dst).sum(0) / w.sum()
    U, S, Vt = np.linalg.svd(((src - cs) * w[:, None]).T @ (dst - cd))
    R = Vt.T @ U.T
    return R, cd - R @ cs


def fit_rigid(src, dst, w=None, trimmed=False, keep=0.5, iters=10):
    """Rigid (reflection allowed) fit dst ~ R src + tv. trimmed=True: refit on the best-fitting
    `keep` share of pairs (by weight), repeated `iters` times."""
    w = np.ones(len(src)) if w is None else np.asarray(w, float)
    R, tv = _procrustes(src, dst, w)
    if trimmed:
        for _ in range(iters):
            res = np.linalg.norm(src @ R.T + tv - dst, axis=1)
            order = np.argsort(res)
            cw = np.cumsum(w[order]); sel = order[cw <= keep * w.sum() + 1e-12]
            if len(sel) < 10:
                break
            R, tv = _procrustes(src[sel], dst[sel], w[sel])
    return R, tv


def to_placement(R, tv, A, B):
    deg = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])) % 360)
    flip = int(np.linalg.det(R) < 0)
    return deg, flip, R @ A.mean(0) + tv - B.mean(0)


def lognorm(X):
    X = np.asarray(X, float)
    return np.log1p(X / np.maximum(X.sum(1, keepdims=True), 1) * 1e3)


def _adata(xy, X, genes, idx):
    import anndata as ad
    a = ad.AnnData(X=np.asarray(X[idx], dtype=np.float32))
    a.var_names = [str(g) for g in genes]; a.obsm["spatial"] = xy[idx]
    return a


def _coupling_pairs(a, b, pi):
    """barycentric target of each A point under the coupling, weighted by its mass"""
    wa = pi.sum(1); ok = wa > 0
    tgt = (pi[ok] @ b) / wa[ok][:, None]
    return a[ok], tgt, wa[ok]


# ---------------- methods: each returns {"plain": (deg, flip, t), ["trimmed": ...], extra} ----
def paste(A, XA, B, XB, genes, alpha=0.1, n=4000, seed=1):
    import paste as pst
    rng = np.random.default_rng(seed)
    ia = rng.choice(len(A), min(n, len(A)), replace=False)
    ib = rng.choice(len(B), min(n, len(B)), replace=False)
    pi = np.asarray(pst.pairwise_align(_adata(A, XA, genes, ia), _adata(B, XB, genes, ib),
                                       alpha=alpha, dissimilarity="kl", numItermax=1000,
                                       verbose=False, numItermaxEmd=1_000_000))
    s, d, w = _coupling_pairs(A[ia], B[ib], pi)
    out = {}
    for name, tr in (("plain", False), ("trimmed", True)):
        out[name] = to_placement(*fit_rigid(s, d, w, trimmed=tr), A, B)
    return out


def paste2(A, XA, B, XB, genes, s_frac=0.7, n=4000, seed=1):
    from paste2 import PASTE2
    rng = np.random.default_rng(seed)
    ia = rng.choice(len(A), min(n, len(A)), replace=False)
    ib = rng.choice(len(B), min(n, len(B)), replace=False)
    pi = np.asarray(PASTE2.partial_pairwise_align(_adata(A, XA, genes, ia), _adata(B, XB, genes, ib),
                                                  s=s_frac, alpha=0.1, dissimilarity="kl", verbose=False))
    s, d, w = _coupling_pairs(A[ia], B[ib], pi)
    return {name: to_placement(*fit_rigid(s, d, w, trimmed=tr), A, B)
            for name, tr in (("plain", False), ("trimmed", True))}


def _knn_graph(x, k, loop=False, flow="source_to_target", **kw):
    import torch
    from sklearn.neighbors import NearestNeighbors
    X = np.asarray(x); kk = k if loop else k + 1
    idx = NearestNeighbors(n_neighbors=kk).fit(X).kneighbors(X, return_distance=False)
    if not loop:
        idx = idx[:, 1:]
    ei = np.vstack([np.repeat(np.arange(len(X)), idx.shape[1]), idx.ravel()])
    if flow == "source_to_target":
        ei = ei[::-1]
    return torch.as_tensor(ei.copy(), dtype=torch.long)


def slat(A, XA, B, XB, genes, k=20, epochs=6, nA=4000, nB=3999, seed=5):
    import scSLAT.model as M
    import scSLAT.model.preprocess as P
    P.knn_graph = _knn_graph
    rng = np.random.default_rng(seed)
    ia = rng.choice(len(A), min(nA, len(A)), replace=False)
    ib = rng.choice(len(B), min(nB, len(B)), replace=False)
    a1, a2 = _adata(A, XA, genes, ia), _adata(B, XB, genes, ib)
    M.Cal_Spatial_Net(a1, k_cutoff=k, model="KNN", verbose=False)
    M.Cal_Spatial_Net(a2, k_cutoff=k, model="KNN", verbose=False)
    edges, feats = M.load_anndatas([a1, a2], feature="DPCA", check_order=False,
                                   n_top_genes=min(2500, len(genes)))
    e0, e1, _ = M.run_SLAT(feats, edges, epochs=epochs)
    best, index, dist = M.spatial_match([e0, e1], adatas=[a1, a2], reorder=False)
    best = np.asarray(best)
    if len(best) == len(ib):
        src, dst = A[ia][best], B[ib]
    else:
        src, dst = A[ia], B[ib][best]
    return {name: to_placement(*fit_rigid(src, dst, trimmed=tr), A, B)
            for name, tr in (("plain", False), ("trimmed", True))}


def spateo_align(A, XA, B, XB, genes, mode="SN-S", dissimilarity="kl", n=4000, seed=1, max_iter=200):
    """spateo morpho_align (Qiu et al.) on subsamples; both slices move into a shared frame, so A is
    brought back into B's original frame through B's own (rigid-fitted) move, then a rigid fit
    (plain / trimmed) gives the placement."""
    import spateo as st
    rng = np.random.default_rng(seed)
    ia = rng.choice(len(A), min(n, len(A)), replace=False)
    ib = rng.choice(len(B), min(n, len(B)), replace=False)
    aA, aB = _adata(A, lognorm(XA), genes, ia), _adata(B, lognorm(XB), genes, ib)
    models, _ = st.align.morpho_align(models=[aA, aB], spatial_key="spatial", key_added="align_spatial",
                                      mode=mode, dissimilarity=dissimilarity, max_iter=max_iter,
                                      verbose=False, device="cpu")
    PA = np.asarray(models[0].obsm["align_spatial"], float)
    PB = np.asarray(models[1].obsm["align_spatial"], float)
    RB, tB = _procrustes(B[ib], PB, np.ones(len(ib)))
    A_inB = (PA - tB) @ RB                                 # undo B's move (RB orthogonal)
    out = {"moved": (ia, A_inB)}                           # the tool's own (non-rigid) placement
    for name, tr in (("plain", False), ("trimmed", True)):
        R, tv = fit_rigid(A[ia], A_inB, trimmed=tr)
        out[name] = to_placement(R, tv, A, B)
    return out


def cast_align(A, XA, B, XB, genes, mode="rigid", n=4000, seed=1, unit_um=9.44):
    """CAST (Tang et al.): CAST_MARK graph embeddings, then CAST_STACK (pre-location + affine; with
    mode "full" also the B-spline step). mode "rigid": ifrigid=True; "affine": affine only; "full":
    affine + B-spline. A rigid fit (plain / trimmed) to the returned query coordinates gives the
    placement. Coordinates passed in um so CAST's default length settings apply."""
    import tempfile
    import CAST
    from CAST.CAST_Stack import reg_params
    rng = np.random.default_rng(seed)
    ia = rng.choice(len(A), min(n, len(A)), replace=False)
    ib = rng.choice(len(B), min(n, len(B)), replace=False)
    coords = {"A": A[ia] * unit_um, "B": B[ib] * unit_um}
    exp = {"A": lognorm(XA[ia]), "B": lognorm(XB[ib])}
    out = tempfile.mkdtemp(prefix="cast_")
    emb = CAST.CAST_MARK(coords, exp, out, if_plot=False, graph_strategy="delaunay")
    params = reg_params(dataname="A", gpu=-1, iterations=500, dist_penalty1=0, bleeding=500,
                        d_list=[3, 2, 1, 1 / 2, 1 / 3], attention_params=[None, 3, 1, 0],
                        dist_penalty2=[0], alpha_basis_bs=[500], meshsize=[8],
                        iterations_bs=[400 if mode == "full" else 0], attention_params_bs=[[None, 3, 1, 0]],
                        mesh_weight=[None], ifrigid=(mode == "rigid"))
    fin = CAST.CAST_STACK(coords, emb, out, ["A", "B"], params_dist=params, mid_visual=False)
    PA = np.asarray(fin["A"], float) / unit_um
    res = {"moved": (ia, PA)}
    for name, tr in (("plain", False), ("trimmed", True)):
        R, tv = fit_rigid(A[ia], PA, trimmed=tr)
        res[name] = to_placement(R, tv, A, B)
    return res


def cpd(A, B, n_starts=8, n=2000, seed=3, w=0.2):
    from pycpd import RigidRegistration
    rng = np.random.default_rng(seed)
    ia = rng.choice(len(A), min(n, len(A)), replace=False)
    ib = rng.choice(len(B), min(n, len(B)), replace=False)
    if n_starts == 1:                # A handed over already placed: keep its position as the start
        a, b = A[ia] - B.mean(0), B[ib] - B.mean(0)
        reg = RigidRegistration(X=b, Y=a, w=w, max_iterations=150, scale=False)
        TY, _ = reg.register()
        R, tv = fit_rigid(a, TY)
        return {"plain": (float(np.degrees(np.arctan2(R[1, 0], R[0, 0])) % 360), int(np.linalg.det(R) < 0),
                          R @ (A.mean(0) - B.mean(0)) + tv)}
    a, b = A[ia] - A.mean(0), B[ib] - B.mean(0)
    starts = [(0.0, 0)] if n_starts == 1 else [(d, f) for f in (0, 1)
                                                for d in np.arange(0, 360, 720 / n_starts)]
    best = None
    for d0, f0 in starts:
        M0 = rot(np.radians(d0)) @ (FLIP if f0 else np.eye(2))
        reg = RigidRegistration(X=b, Y=a @ M0.T, w=w, max_iterations=150, scale=False)
        TY, _ = reg.register()
        if best is None or reg.sigma2 < best[0]:
            best = (reg.sigma2, TY)
    R, tv = fit_rigid(a, best[1])
    return {"plain": to_placement(R, tv, a, b)}          # a, b centred: same convention


def local_search(al, n_starts, tpad=300.0, seed=1000, maxfev=300):
    """Nelder-Mead on OUR objective (al = a SmoothAlign on this pair) from random starts."""
    rng = np.random.default_rng(seed); runs = []
    for _ in range(n_starts):
        fl = int(rng.integers(2))
        x0 = np.array([rng.uniform(0, 360), *rng.uniform(-tpad, tpad, 2)])
        f = lambda x: -al.exact_score(np.radians(x[0]), fl, np.clip(x[1:], -tpad, tpad))
        r = minimize(f, x0, method="Nelder-Mead", options={
            "initial_simplex": np.array([x0, x0 + [15, 0, 0], x0 + [0, 40, 0], x0 + [0, 0, 40]]),
            "maxfev": maxfev, "xatol": 0.5, "fatol": 1e-4})
        runs.append((-r.fun, r.x[0] % 360, fl, np.clip(r.x[1:], -tpad, tpad)))
    s, d, fl, t = max(runs, key=lambda z: z[0])
    return {"plain": (d, fl, t), "all_starts": [(float(r[0]), float(r[1]), r[2]) for r in runs]}


def stalign(A, la, B, lb, unit_um, starts, niter=1000, epL=2e-7, dx=30.0, recenter=True, extra=None,
            sigmaM=1.5, sigmaB=1.0, sigmaA=1.5, epV=100, epT=None):
    """STalign with one channel per cluster (A: KA channels, B: KB channels)."""
    import torch
    import STalign.STalign as S
    Au, Bu = A * unit_um, B * unit_um
    # rasterize each label on ONE grid per slide
    XI, YI, _ = S.rasterize(Au[:, 0], Au[:, 1], dx=dx, draw=0)
    XJ, YJ, _ = S.rasterize(Bu[:, 0], Bu[:, 1], dx=dx, draw=0)
    def label_stack(xy, lab, Xg, Yg):
        out = []
        for l in np.unique(lab):
            m = lab == l
            H, _, _ = np.histogram2d(xy[m, 1], xy[m, 0], bins=[np.r_[Yg - dx / 2, Yg[-1] + dx / 2],
                                                                 np.r_[Xg - dx / 2, Xg[-1] + dx / 2]])
            out.append(H)
        I = np.stack(out); return I / (I.max() + 1e-9)
    from scipy.ndimage import gaussian_filter
    I = gaussian_filter(label_stack(Au, la, XI, YI), (0, 1, 1))
    J = gaussian_filter(label_stack(Bu, lb, XJ, YJ), (0, 1, 1))
    swap = np.array([[0, 1], [1, 0]])
    best = None
    for d0, f0 in starts:
        M = rot(np.radians(d0)) @ (FLIP if f0 else np.eye(2))
        L = swap @ M @ swap
        # recenter=False: A was handed over already placed, keep its position as the start
        T = Bu.mean(0)[::-1] - L @ Au.mean(0)[::-1] if recenter else np.zeros(2)
        kw = {} if epT is None else {"epT": epT}
        out = S.LDDMM([YI, XI], I, [YJ, XJ], J, L=L, T=T, niter=niter, device="cpu", epL=epL,
                      sigmaM=sigmaM, sigmaB=sigmaB, sigmaA=sigmaA, epV=epV,
                      muB=torch.zeros(J.shape[0]), muA=torch.ones(J.shape[0]), **kw)
        Pp = S.transform_points_source_to_target(out["xv"], out["v"], out["A"],
                                                 np.stack([Au[:, 1], Au[:, 0]], 1))
        Pp = np.asarray(Pp.detach().cpu() if hasattr(Pp, "detach") else Pp)[:, ::-1]
        # pick the start by how well moved A-density overlaps B-density (no genes, no truth)
        lo = np.minimum(Pp.min(0), Bu.min(0)); hi = np.maximum(Pp.max(0), Bu.max(0))
        e = [np.arange(lo[k], hi[k] + 50, 50) for k in (0, 1)]
        h1 = np.histogram2d(Pp[:, 0], Pp[:, 1], e)[0]; h2 = np.histogram2d(Bu[:, 0], Bu[:, 1], e)[0]
        crit = float(np.corrcoef(h1.ravel(), h2.ravel())[0, 1])
        if best is None or crit > best[0]:
            best = (crit, Pp, d0, f0, out)
    crit, Pp, d0, f0, out = best
    R, tv = fit_rigid(A, Pp / unit_um)
    res = {"plain": to_placement(R, tv, A, B), "picked_start": [float(d0), f0],
           "density_corr": round(crit, 4), "moved": (np.arange(len(A)), Pp / unit_um)}
    if extra is not None:                # other points of A's frame (landmarks) through the same bent map
        E = np.asarray(extra, float) * unit_um
        Ep = S.transform_points_source_to_target(out["xv"], out["v"], out["A"], np.stack([E[:, 1], E[:, 0]], 1))
        res["extra_moved"] = np.asarray(Ep.detach().cpu() if hasattr(Ep, "detach") else Ep)[:, ::-1] / unit_um
    return res
