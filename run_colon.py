"""W6: held-out second tissue. Colon (GEO GSE280318, Oliveira et al.): standard Visium (55 um spots)
vs Visium HD (8 um bins) on serial sections of the same block, 3 pairs (P2 cancer, P3 and P5 normal).
Nothing here was tuned on colon.

Same frozen rule and same labelling procedure as the ground-truth benchmark (bench_gt.py):
  * Visium side (A, moved): spots clustered on their own (log-normalise, PCA 20, k-means K=12);
  * HD side (B, maps): 8 um bins clustered the same way (K=12), then neighbourhood regions K=6;
  * contrast on, sigma 20 units, h 2 units, window +-300, unit = 9.44 um (as everywhere);
  * uniqueness: far-shift check at 30 units, then 60, then rotation only.
Scale: HD from its own microns_per_pixel (checked: 8 um bins = 29.2 px); Visium from the spot
grid (centre-to-centre 100 um), NOT from spot_diameter_fullres.
Independent check: gene score (spot expression vs the HD bins under the spot, shared genes, each
slide's mean removed) against 20 random placements. Both sides are whole-transcriptome.
Usage: python3 run_colon.py P2CRC P3NAT P5NAT
"""
import gzip, json, os, sys, time
import numpy as np, pandas as pd, h5py, scipy.sparse as sp
from scipy.spatial import cKDTree
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
sys.path.insert(0, ".")
from smoothalign import SmoothAlign
from smoothpar import solve_par, far_par, far_par_ext
from bench_domains import neighbourhood_domains
from run_baselines_scale import GeneScore

UNIT = 9.44
D = "data/crc"
OUT = "results_colon.json"


def read_h5(path):
    with h5py.File(path) as f:
        g = f["matrix"]
        M = sp.csc_matrix((g["data"][:], g["indices"][:], g["indptr"][:]), shape=g["shape"][:])
        return M.T.tocsr(), g["barcodes"][:].astype(str), g["features/name"][:].astype(str)


def cluster(X, k, seed):
    lib = np.asarray(X.sum(1)).ravel()
    Xn = sp.diags(1e3 / np.maximum(lib, 1)) @ X
    Xn.data = np.log1p(Xn.data)
    var = np.asarray(Xn.multiply(Xn).mean(0)).ravel() - np.asarray(Xn.mean(0)).ravel() ** 2
    top = np.argsort(var)[::-1][:2000]
    Z = TruncatedSVD(20, random_state=seed).fit_transform(Xn[:, top])
    return MiniBatchKMeans(k, n_init=3, random_state=seed, batch_size=10000).fit_predict(Z)


def load_visium(s):
    X, bc, genes = read_h5(f"{D}/v2_{s}/{s}_filtered_feature_bc_matrix.h5")
    p = pd.read_csv(f"{D}/v2_{s}/{s}_tissue_positions.csv.gz")
    p = p[p.in_tissue == 1].set_index("barcode")
    keep = np.isin(bc, p.index)
    X, bc = X[keep], bc[keep]
    px = p.loc[bc, ["pxl_col_in_fullres", "pxl_row_in_fullres"]].values.astype(float)
    d, _ = cKDTree(px).query(px, k=2)
    um_per_px = 100.0 / float(np.median(d[:, 1]))           # spot grid: 100 um centre-to-centre
    return px * um_per_px, X, genes, um_per_px


def load_hd(s):
    X, bc, genes = read_h5(f"{D}/hd_{s}/{s}_filtered_feature_bc_matrix.h5")
    import io
    with gzip.open(f"{D}/hd_{s}/{s}_tissue_positions.parquet.gz") as fh:   # gzip-compressed parquet
        p = pd.read_parquet(io.BytesIO(fh.read())).set_index("barcode")
    sfp = f"{D}/hd_{s}/{s}_scalefactors_json.json.gz"
    sf = json.load(gzip.open(sfp)) if sfp.endswith(".gz") else json.load(open(sfp))
    px = p.loc[bc, ["pxl_col_in_fullres", "pxl_row_in_fullres"]].values.astype(float)
    return px * float(sf["microns_per_pixel"]), X, genes


def save(r):
    res = json.load(open(OUT)) if os.path.exists(OUT) else []
    res.append(r); json.dump(res, open(OUT, "w"), indent=1); print(json.dumps(r), flush=True)


if __name__ == "__main__":
    for s in sys.argv[1:]:
        t0 = time.time()
        Axy, AX, gA, umpx = load_visium(s)
        Bxy, BX, gB = load_hd(s)
        common = sorted(set(gA) & set(gB))
        la = cluster(AX, 12, 1)
        B = Bxy / UNIT
        lb = neighbourhood_domains(B, cluster(BX, 12, 2), K=6, seed=3)
        A = Axy / UNIT
        prep = time.time() - t0
        al = SmoothAlign(A, la, B, lb, h=2.0, sigma=20.0, tpad=300.0, contrast=True)
        o = solve_par(al, n_proc=14, time_limit=21600)
        th, t, fl = o["transform"]; deg = float(np.degrees(th)) % 360
        iA = pd.Series(np.arange(len(gA)), index=gA); iA = iA[~iA.index.duplicated()][common].values
        iB = pd.Series(np.arange(len(gB)), index=gB); iB = iB[~iB.index.duplicated()][common].values
        # 500 most variable shared genes (Visium side): dense HD x all genes would need ~78 GB
        XAc = AX[:, iA]; XAn = sp.diags(1e3 / np.maximum(np.asarray(XAc.sum(1)).ravel(), 1)) @ XAc
        XAn.data = np.log1p(XAn.data)
        v = np.asarray(XAn.multiply(XAn).mean(0)).ravel() - np.asarray(XAn.mean(0)).ravel() ** 2
        top = np.argsort(v)[::-1][:500]
        gs = GeneScore(A, XAc[:, top].toarray().astype(float), B,
                       BX[:, iB[top]].toarray().astype(np.float32), 27.5 / UNIT * 1.5)
        rnd = [gs(np.random.default_rng(k).uniform(0, 360), k % 2,
                  np.random.default_rng(k + 99).uniform(-150, 150, 2)) for k in range(20)]
        r = {"pair": s, "n_spots": int(len(A)), "n_hd_bins": int(len(B)), "visium_um_per_px": round(umpx, 4),
             "angle": round(deg, 2), "flip": int(fl), "shift": [round(float(v), 2) for v in t],
             "objective": round(o["best"], 4), "gap": round(o["gap"], 4), "certified": o["certified"],
             "gene_score": round(gs(deg, fl, np.asarray(t)), 4),
             "random_gene_mean": round(float(np.nanmean(rnd)), 4), "random_gene_max": round(float(np.nanmax(rnd)), 4),
             "prep_s": round(prep, 1), "solve_s": round(o["secs"], 1)}
        if o["certified"]:
            ue = far_par_ext(al, o["best"], th, fl, t, shift_radius=30.0, n_proc=14, time_limit=10800)
            r.update({"eps_shift30": round(ue["eps"], 4), "unique_shift30": bool(ue["eps"] > 0)})
            if ue["eps"] <= 0:
                ue = far_par_ext(al, o["best"], th, fl, t, shift_radius=60.0, n_proc=14, time_limit=10800)
                r.update({"eps_shift60": round(ue["eps"], 4), "unique_shift60": bool(ue["eps"] > 0)})
                if ue["eps"] <= 0:
                    u = far_par(al, o["best"], th, fl, n_proc=14, time_limit=10800)
                    r.update({"eps_rotation": round(u["eps"], 4), "unique_rotation": bool(u["eps"] > 0),
                              "rival_far": u["best_far_placement"]})
        r["total_s"] = round(time.time() - t0, 1)
        save(r)
    print("DONE", flush=True)
