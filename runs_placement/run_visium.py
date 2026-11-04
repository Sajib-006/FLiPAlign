"""Real cross-platform pair at scale (breast). Visium CytAssist (4,992 spots of 55 um, 15
graph clusters, whole transcriptome) vs Xenium Rep1/Rep2 (166k / 119k cells, 313 genes) from
the same breast block (10x / Janesick et al. 2023). Different platform, different
resolution, no shared labels, unknown rotation, possible mirror flip, partial overlap.

A = Visium spots (moved), B = Xenium (smoothed maps). Contrast score. Scale is fixed from
the spot grid: 182 px = 100 um -> 0.5494 um/px (the "spot_diameter_fullres" field is a 65 um
spot, not 55 um). Units = Xenium median cell spacing.
Checks: proven uniqueness (far_upper_bound) and a genes-only tie-breaker: correlation of
each spot's expression with the Xenium cells under it (shared genes, each slide's mean
removed), against 30 random placements.
"""
import os, sys  # noqa: E401
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import repo_paths  # noqa: E402,F401  (puts the code folders on the import path)
import json, os, sys
import numpy as np, pandas as pd, h5py, scipy.sparse as sp
from scipy.spatial import cKDTree
sys.path.insert(0, ".")
from smoothalign import SmoothAlign, far_upper_bound, rot, FLIP
from bench_domains import neighbourhood_domains

def expr(R, barcodes):
    with h5py.File(f"data/xenium/{R}/cell_feature_matrix.h5") as f:
        g = f["matrix"]
        M = sp.csc_matrix((g["data"][:], g["indices"][:], g["indptr"][:]), shape=g["shape"][:])
        bc = g["barcodes"][:].astype(str)
        ftype = g["features/feature_type"][:].astype(str)
        genes = g["features/name"][:].astype(str)
    keep = ftype == "Gene Expression"
    X = M[keep].T.tocsr()                                   # cells x genes
    pos = pd.Series(np.arange(len(bc)), index=bc)
    X = X[pos[barcodes].values]
    lib = np.asarray(X.sum(1)).ravel()
    X = np.log1p(X.multiply(1.0 / np.maximum(lib, 1)[:, None]).tocsr() * 100).toarray()
    return X, genes[keep]


def cells(R):
    c = pd.read_csv(f"data/xenium/{R}/cells.csv.gz")
    cl = pd.read_csv(f"data/xenium/{R}/analysis/clustering/gene_expression_graphclust/clusters.csv")
    m = c.merge(cl, left_on="cell_id", right_on="Barcode")
    return m["cell_id"].astype(str).values


OUT = "results_visium.json"
res = json.load(open(OUT)) if os.path.exists(OUT) else []
UMPX = 100.0 / 182.01
V = "data/visium_breast"
rng = np.random.default_rng(0)


def save(r):
    res.append(r); json.dump(res, open(OUT, "w"), indent=1); print(json.dumps(r), flush=True)


def xenium(R):
    c = pd.read_csv(f"data/xenium/{R}/cells.csv.gz")
    cl = pd.read_csv(f"data/xenium/{R}/analysis/clustering/gene_expression_graphclust/clusters.csv")
    m = c.merge(cl, left_on="cell_id", right_on="Barcode")
    return m[["x_centroid", "y_centroid"]].values.astype(float), m["Cluster"].values - 1


def visium():
    p = pd.read_csv(f"{V}/spatial/tissue_positions.csv")
    p = p[p.in_tissue == 1]
    cl = pd.read_csv(f"{V}/analysis/clustering/gene_expression_graphclust/clusters.csv")
    m = p.merge(cl, left_on="barcode", right_on="Barcode")
    xy = m[["pxl_col_in_fullres", "pxl_row_in_fullres"]].values.astype(float) * UMPX
    with h5py.File(f"{V}/CytAssist_FFPE_Human_Breast_Cancer_filtered_feature_bc_matrix.h5") as f:
        g = f["matrix"]
        M = sp.csc_matrix((g["data"][:], g["indices"][:], g["indptr"][:]), shape=g["shape"][:])
        bc = g["barcodes"][:].astype(str); genes = g["features/name"][:].astype(str)
    X = M.T.tocsr()[pd.Series(np.arange(len(bc)), index=bc)[m["barcode"].values].values]
    return xy, m["Cluster"].values - 1, X, genes


xyV_um, labV, XV, genesV = visium()
SP = {}


def gene_check(xyX_u, XX, genesX, th, flip, t, cA, cB, n_rand=30):
    """mean corr of spot vs Xenium cells within ~one spot footprint of its landing point"""
    common = [g for g in genesX if g in set(genesV)]
    iX = pd.Series(np.arange(len(genesX)), index=genesX)[common].values
    iV = pd.Series(np.arange(len(genesV)), index=genesV)
    iV = iV[~iV.index.duplicated()][common].values
    Vs = XV[:, iV].toarray(); Vs = np.log1p(Vs / np.maximum(Vs.sum(1, keepdims=True), 1) * 1e3)
    Vs -= Vs.mean(0)
    Xs = np.expm1(XX[:, iX])                              # XX is log1p(norm*100)
    T = cKDTree(xyX_u); rad = 27.5 / SP["um"] * 1.5       # 1.5x the 55 um spot radius
    A = xyV_um / SP["um"]

    def one(th, flip, t):
        P = (rot(th) @ (FLIP if flip else np.eye(2)) @ (A - cA).T).T + cB + t
        nb = T.query_ball_point(P, rad)
        ok = np.array([len(x) >= 3 for x in nb])
        if ok.sum() < 50:
            return np.nan, float(ok.mean())
        agg = np.array([Xs[x].sum(0) for x, o in zip(nb, ok) if o])
        agg = np.log1p(agg / np.maximum(agg.sum(1, keepdims=True), 1) * 1e3); agg -= agg.mean(0)
        a, b = Vs[ok], agg
        a = (a - a.mean(1, keepdims=True)) / (a.std(1, keepdims=True) + 1e-9)
        b = (b - b.mean(1, keepdims=True)) / (b.std(1, keepdims=True) + 1e-9)
        return float((a * b).mean(1).mean()), float(ok.mean())

    s, cov = one(th, flip, t)
    rnd = [one(rng.uniform(0, 2 * np.pi), int(rng.integers(2)), rng.uniform(-300, 300, 2))[0]
           for _ in range(n_rand)]
    rnd = [x for x in rnd if np.isfinite(x)]
    return {"gene_corr": round(s, 4), "spots_on_xenium": round(cov, 3), "shared_genes": len(common),
            "random_mean": round(float(np.mean(rnd)), 4), "random_max": round(float(np.max(rnd)), 4)}


def run(tag, R, labB, sigma, XX, genesX, xyX_u, h=4.0, tpad=300.0):
    A = xyV_um / SP["um"]
    al = SmoothAlign(A, labV, xyX_u, labB, h=h, sigma=sigma, tpad=tpad, contrast=True)
    out = al.solve(time_limit=5400)
    th, t, fl = out["transform"]
    r = {"test": tag, "xenium": R, "sigma": sigma, "h": h, "tpad": tpad,
         "KA/KB": [al.KA, al.KB], "angle_deg": round(float(np.degrees(th)) % 360, 2),
         "flip": int(fl), "shift": [round(float(v), 1) for v in t],
         "shift_at_window_edge": bool(np.max(np.abs(t)) >= al.r * al.h - 1e-9),
         "best": round(out["best"], 4), "gap": round(out["gap"], 4),
         "certified": out["certified"], "solve_s": round(out["secs"], 1)}
    if out["certified"]:
        u = far_upper_bound(al, out["best"], th, fl, time_limit=3600)
        r.update({"far_exact": round(u["far_exact"], 4), "eps_unique": round(u["eps"], 4),
                  "unique_PROVEN": bool(u["eps"] > 0), "far_complete": u["complete"]})
    r.update(gene_check(xyX_u, XX, genesX, th, fl, t, A.mean(0), xyX_u.mean(0)))
    save(r)


if __name__ == "__main__":
    for R in ("Rep1", "Rep2"):
        xy_um, lab = xenium(R)
        d, _ = cKDTree(xy_um).query(xy_um[::20], k=2)
        SP["um"] = float(np.median(d[:, 1]))
        xyX = xy_um / SP["um"]
        XX, genesX = expr(R, cells(R))
        reg = neighbourhood_domains(xyX, lab, K=6, seed=1)
        print(f"{R}: {len(xyX)} cells, unit {SP['um']:.2f} um; Visium {len(labV)} spots", flush=True)
        if R == "Rep1":
            for sigma in (20.0, 30.0):
                run("Visium vs Xenium, cell types", R, lab, sigma, XX, genesX, xyX)
            run("Visium vs Xenium, regions K=6", R, reg, 20.0, XX, genesX, xyX)
        else:
            run("Visium vs Xenium, regions K=6", R, reg, 20.0, XX, genesX, xyX)
            run("Visium vs Xenium, cell types", R, lab, 20.0, XX, genesX, xyX)
    print("DONE", flush=True)
