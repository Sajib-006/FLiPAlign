"""Lung pairs 6 and 7 (10x human lung adenocarcinoma, one block, serial sections): A = Visium HD 16 um bins, B = Xenium cells.
Pair 6: Visium HD experiment 1 with Xenium v1 (377 genes); pair 7: Visium HD experiment 2 with Xenium Prime 5K.
Same labelling rule as the colon pairs (run_colon.py): A side own clusters (K=12, seed 1); B side own clusters (K=12, seed 2) then
neighbourhood regions (K=6, seed 3). Units 9.44 um. Returns the tuple of real_fine.load."""
import os, sys  # noqa: E401
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import repo_paths  # noqa: E402,F401  (puts the code folders on the import path)
import gzip, io, json, os, sys, time
T0 = time.time()
log = lambda m: print(f'[{time.time() - T0:6.0f}s] {m}', flush=True)
import numpy as np, pandas as pd, scipy.sparse as sp
sys.path.insert(0, ".")
from run_colon import read_h5, cluster
from bench_domains import neighbourhood_domains

UNIT = 9.44
D = os.environ.get("LUNG_DIR", "data/lung")
SPEC = {6: (1, "xv1"), 7: (2, "xp5")}


def load_lung(k):
    e, xk = SPEC[k]
    cf = f"{D}/cache_pair{k}.npz"
    if os.path.exists(cf):                                  # the labels take ~18 minutes; the cache keeps them identical for every run
        z = np.load(cf, allow_pickle=True); return (str(z["name"]), z["A"], z["la"], z["XA"], z["B"], z["lb"], z["XB"], float(z["rad"]))
    r = _load_lung(k, e, xk)
    np.savez(cf + ".tmp.npz", name=r[0], A=r[1], la=r[2], XA=r[3], B=r[4], lb=r[5], XB=r[6], rad=r[7]); os.replace(cf + ".tmp.npz", cf)
    return r


def _load_lung(k, e, xk):
    # ---- B: Xenium cells
    log('start')
    c = pd.read_csv(f"{D}/{xk}_cells.csv.gz"); XB, bc, gB = read_h5(f"{D}/{xk}_cell_feature_matrix.h5")
    pos = pd.Series(np.arange(len(bc)), index=bc); c = c[c.cell_id.astype(str).isin(pos.index)]
    ib = pos[c.cell_id.astype(str).values].values; XB = XB[ib]; B = c[["x_centroid", "y_centroid"]].values.astype(float)
    ok = np.asarray(XB.sum(1)).ravel() >= 10; XB, B = XB[ok], B[ok] / UNIT
    log(f'xenium read: {XB.shape}')
    # ---- A: Visium HD 16 um bins
    base = f"{D}/hd{e}_16/binned_outputs/square_016um"
    XA, bcA, gA = read_h5(f"{base}/filtered_feature_bc_matrix.h5"); p = pd.read_parquet(f"{base}/spatial/tissue_positions.parquet").set_index("barcode")
    sf = json.load(open(f"{base}/spatial/scalefactors_json.json")); p = p[p.in_tissue == 1]
    keep = np.isin(bcA, p.index); XA, bcA = XA[keep], bcA[keep]
    A = p.loc[bcA, ["pxl_col_in_fullres", "pxl_row_in_fullres"]].values.astype(float) * float(sf["microns_per_pixel"]) / UNIT
    okA = np.asarray(XA.sum(1)).ravel() >= 20; XA, A = XA[okA], A[okA]
    log(f'HD bins: {XA.shape}')
    # ---- labels (each side from its own genes)
    la = cluster(XA, 12, 1)
    log('A clustered')
    cb = cluster(XB, 12, 2); log('B clustered')
    lb = neighbourhood_domains(B, cb, K=6, seed=3)
    log('regions done')
    # ---- shared genes for the tools: 500 most variable of the common genes, A side
    common = sorted(set(gA) & set(gB)); iA = pd.Series(np.arange(len(gA)), index=gA); iA = iA[~iA.index.duplicated()][common].values
    iB = pd.Series(np.arange(len(gB)), index=gB); iB = iB[~iB.index.duplicated()][common].values
    XAc = XA[:, iA]; XAn = sp.diags(1e3 / np.maximum(np.asarray(XAc.sum(1)).ravel(), 1)) @ XAc; XAn.data = np.log1p(XAn.data)
    v = np.asarray(XAn.multiply(XAn).mean(0)).ravel() - np.asarray(XAn.mean(0)).ravel() ** 2; top = np.argsort(v)[::-1][:min(500, len(common))]
    return (f"lung HD{e} / {xk}", A, la, XAc[:, top].toarray().astype(np.float32), B, lb, XB[:, iB[top]].toarray().astype(np.float32), 16.0 / UNIT * 0.75)
