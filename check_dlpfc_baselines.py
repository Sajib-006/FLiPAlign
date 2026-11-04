"""W7: are the PASTE and SLAT wrappers correct? Run them on DLPFC, the data PASTE was published on,
and report each tool's OWN quality measure next to the standard layer metric.

Genes: the 2,000 most variable shared genes (as PASTE tutorials do).
Pairs: the 18 same-donor pairs (6 adjacent 10 um, 12 about 300 um apart).
  PASTE (alpha 0.1): own measure = share of transport mass on same-layer spot pairs
                     (as in the PASTE paper); then our rigid fit (plain / trimmed).
  SLAT (k 6, Visium neighbours): own measure = share of matched spots with the same layer;
                     then our rigid fit (plain / trimmed).
Standard layer metric for every placement (same as dlpfc_layer_accuracy.py): share of A spots
whose nearest B spot (within 1 spot) has the same true layer. The tools SEE all genes; ours sees
only scrambled layer labels (results_dlpfc_decide.json).
If the tools' own measures are high here, the wrappers work, and failures elsewhere are real.
"""
import itertools, json, os, sys, time
import numpy as np, pandas as pd, h5py, scipy.sparse as sp
from scipy.spatial import cKDTree
sys.path.insert(0, ".")
import baselines as BL
import run_step2 as S
from smoothalign import rot, FLIP

OUT = "results_check_dlpfc_baselines.json"
LAY = ["L1", "L2", "L3", "L4", "L5", "L6", "WM"]
lm = pd.read_csv("data/dlpfc/barcode_level_layer_map.tsv", sep="\t", header=None,
                 names=["barcode", "sample", "layer"])


def load(s):
    p = pd.read_csv(f"data/dlpfc/pos_{s}.txt", header=None,
                    names=["barcode", "in_tissue", "row", "col", "py", "px"])
    m = p[p.in_tissue == 1].merge(lm[lm["sample"] == s], on="barcode")
    m = m[m.layer.isin(LAY)]
    with h5py.File(f"data/dlpfc_h5/{s}_filtered_feature_bc_matrix.h5") as f:
        g = f["matrix"]
        M = sp.csc_matrix((g["data"][:], g["indices"][:], g["indptr"][:]), shape=g["shape"][:]).T.tocsr()
        bc = g["barcodes"][:].astype(str); genes = g["features/name"][:].astype(str)
    pos = pd.Series(np.arange(len(bc)), index=bc)
    X = M[pos[m.barcode.values].values].toarray().astype(np.float32)
    xy = np.c_[m.px.values, m.py.values] / 137.0
    return xy - xy.mean(0), np.array([LAY.index(l) for l in m.layer]), X, genes


def layer_acc(A, la, B, lb, deg, fl, t):
    M = rot(np.radians(deg)) @ (FLIP if fl else np.eye(2))
    P = (A - A.mean(0)) @ M.T + B.mean(0) + np.asarray(t)
    d, j = cKDTree(B).query(P)
    ok = d <= 1.0
    return float(np.mean(la[ok] == lb[j[ok]])) if ok.any() else 0.0


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else []
    done = {(r["pair"], r["method"]) for r in res}
    ours = {(r["a"], r["b"]): r for r in json.load(open("results_dlpfc_decide.json"))}
    pairs = [(a, b) for a, b in itertools.combinations(sorted(S.DONOR), 2) if S.DONOR[a] == S.DONOR[b]]
    for a, b in pairs:
        grp = "adjacent" if (a, b) in S.ADJACENT else "300 um"
        A, la, XA, gA = load(a); B, lb, XB, gB = load(b)
        common = sorted(set(gA) & set(gB))
        iA = pd.Series(np.arange(len(gA)), index=gA); iA = iA[~iA.index.duplicated()][common].values
        iB = pd.Series(np.arange(len(gB)), index=gB); iB = iB[~iB.index.duplicated()][common].values
        XA, XB = XA[:, iA], XB[:, iB]
        # 2,000 most variable shared genes (PASTE's KL step over all 33k genes is ~5e11 operations)
        Z = np.log1p(np.vstack([XA / np.maximum(XA.sum(1, keepdims=True), 1),
                                XB / np.maximum(XB.sum(1, keepdims=True), 1)]) * 1e4)
        top = np.argsort(Z.var(0))[::-1][:2000]
        XA, XB, common = XA[:, top], XB[:, top], [common[k] for k in top]
        tag = f"{a} vs {b}"
        o = ours.get((a, b))
        if o and (tag, "ours") not in done:
            res.append({"pair": tag, "group": grp, "method": "ours", "fit": "-",
                        "layer_acc": round(layer_acc(A, la, B, lb, o["angle"], o["flip"], o["shift"]), 3),
                        "unique_rotation": bool(o.get("eps_rotation", -1) > 0)})
        for m in ("paste", "slat"):
            if (tag, m) in done:
                continue
            t0 = time.time()
            try:
                if m == "paste":
                    import paste as pst
                    sa, sb = BL._adata(A, XA, common, np.arange(len(A))), BL._adata(B, XB, common, np.arange(len(B)))
                    pi = np.asarray(pst.pairwise_align(sa, sb, alpha=0.1, dissimilarity="kl", numItermax=1000,
                                                       verbose=False, numItermaxEmd=1_000_000))
                    own = float((pi * (la[:, None] == lb[None, :])).sum() / pi.sum())
                    s_, d_, w_ = BL._coupling_pairs(A, B, pi)
                    fits = {n: BL.to_placement(*BL.fit_rigid(s_, d_, w_, trimmed=tr), A, B)
                            for n, tr in (("plain", False), ("trimmed", True))}
                else:
                    import scSLAT.model as M_, scSLAT.model.preprocess as P
                    P.knn_graph = BL._knn_graph
                    nB = len(B) - 1 if len(B) == len(A) else len(B)         # unequal counts: direction is clear
                    ib = np.arange(nB)
                    a1, a2 = BL._adata(A, XA, common, np.arange(len(A))), BL._adata(B, XB, common, ib)
                    M_.Cal_Spatial_Net(a1, k_cutoff=6, model="KNN", verbose=False)
                    M_.Cal_Spatial_Net(a2, k_cutoff=6, model="KNN", verbose=False)
                    edges, feats = M_.load_anndatas([a1, a2], feature="DPCA", check_order=False)
                    e0, e1, _ = M_.run_SLAT(feats, edges, epochs=6)
                    best, index, dist = M_.spatial_match([e0, e1], adatas=[a1, a2], reorder=False)
                    best = np.asarray(best)
                    if len(best) == len(ib):
                        src, dst, lsrc, ldst = A[best], B[ib], la[best], lb[ib]
                    else:
                        src, dst, lsrc, ldst = A, B[ib][best], la, lb[ib][best]
                    own = float(np.mean(lsrc == ldst))
                    fits = {n: BL.to_placement(*BL.fit_rigid(src, dst, trimmed=tr), A, B)
                            for n, tr in (("plain", False), ("trimmed", True))}
                for n, (deg, fl, t) in fits.items():
                    res.append({"pair": tag, "group": grp, "method": m, "fit": n,
                                "own_measure_same_layer": round(own, 3),
                                "angle": round(deg, 2), "flip": fl,
                                "layer_acc": round(layer_acc(A, la, B, lb, deg, fl, t), 3),
                                "sec": round(time.time() - t0, 1)})
            except Exception as e:
                res.append({"pair": tag, "method": m, "error": repr(e)[:300]})
            json.dump(res, open(OUT, "w"), indent=1); print(json.dumps(res[-1]), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
