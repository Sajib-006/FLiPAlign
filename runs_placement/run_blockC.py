"""Cross-modal test (10x CytAssist Visium Gene+Protein, human tonsil).

Why this design: CytAssist measures RNA and protein on the SAME spots, so the true
cross-modal correspondence is known exactly. We cluster RNA into domains and protein into
domains INDEPENDENTLY (different algorithms see different numbers of domains, and the two
label alphabets are unrelated), give each modality a DIFFERENT random 70% of the spots so
they share no cells, apply a KNOWN rotation + mirror flip to the protein map, and ask
CertAlign to recover it using no shared features whatsoever.

NOTE: warm_start is essential here. Pruning compares bounds against the incumbent's
ABSOLUTE level, and the cross-modal score tops out near 0.5, so a cold start barely prunes
(the first attempt ran >10 min with no output). A cheap 8-restart incumbent fixes it: ~30 s.

Ground truth available, unlike in a serial-section experiment:
  1. the applied transform (exact),
  2. the true RNA-domain <-> protein-domain correspondence, computable on the shared spots.
"""
import os, sys  # noqa: E401
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import repo_paths  # noqa: E402,F401  (puts the code folders on the import path)
import numpy as np, h5py, json, time, sys
from scipy.sparse import csc_matrix
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, ".")
from certalign import CertAlign, modes, rot, FLIP

D = "data/cytassist/"
H5 = D + "CytAssist_FFPE_Protein_Expression_Human_Tonsil_filtered_feature_bc_matrix.h5"
rng = np.random.default_rng(0)

_CACHE = {}
def load(k_rna=8, k_prot=6):
    if _CACHE: return _CACHE["v"]
    f = h5py.File(H5, "r")
    m = f["matrix"]
    X = csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]), shape=tuple(m["shape"][:])).T.tocsr()
    bc = [b.decode() for b in m["barcodes"][:]]
    ft = np.array([x.decode() for x in m["features/feature_type"][:]])
    rna, prot = X[:, ft == "Gene Expression"], np.asarray(X[:, ft == "Antibody Capture"].todense(), float)
    # coordinates
    import csv
    pos = {}
    with open(D + "spatial/tissue_positions.csv") as fh:
        for r in csv.DictReader(fh):
            if int(r["in_tissue"]) == 1:
                pos[r["barcode"]] = (float(r["pxl_col_in_fullres"]), float(r["pxl_row_in_fullres"]))
    keep = np.array([i for i, b in enumerate(bc) if b in pos])
    xy = np.array([pos[bc[i]] for i in keep])
    rna, prot = rna[keep], prot[keep]
    # spot spacing -> work in units of one spot
    from scipy.spatial import cKDTree
    d, _ = cKDTree(xy).query(xy, k=2); sp = float(np.median(d[:, 1]))
    xy = (xy - xy.mean(0)) / sp
    # independent clustering per modality
    r_cpm = rna.multiply(1e4 / np.maximum(rna.sum(1), 1)).log1p()
    e_r = TruncatedSVD(30, random_state=0).fit_transform(r_cpm)
    lab_r = KMeans(k_rna, n_init=10, random_state=0).fit_predict(e_r / (np.linalg.norm(e_r, axis=1, keepdims=True) + 1e-9))
    p = np.log1p(prot / np.maximum(prot.sum(1, keepdims=True), 1) * 1e4)
    e_p = PCA(min(15, p.shape[1] - 1), random_state=0).fit_transform((p - p.mean(0)) / (p.std(0) + 1e-9))
    lab_p = KMeans(k_prot, n_init=10, random_state=0).fit_predict(e_p)
    _CACHE["v"] = (xy, lab_r, lab_p, sp)
    return _CACHE["v"]

def true_correspondence(lab_r, lab_p, KA, KB):
    """the real RNA-domain <-> protein-domain matching, from the shared spots"""
    N = np.zeros((KA, KB))
    np.add.at(N, (lab_r, lab_p), 1)
    r, c = linear_sum_assignment(-N)
    return {int(a): int(b) for a, b in zip(r, c)}, N[r, c].sum() / len(lab_r)

def go(angle_deg, flip, overlap=0.70, eps=0.03):
    xy, lr, lp, sp = load()
    KA, KB = lr.max() + 1, lp.max() + 1
    truth, agree = true_correspondence(lr, lp, KA, KB)
    n = len(xy)
    iA = rng.choice(n, int(overlap * n), replace=False)
    iB = rng.choice(n, int(overlap * n), replace=False)   # different spots: no shared cells
    th = np.radians(angle_deg); t = np.array([5.0, -3.0])
    A, la = xy[iA], lr[iA]
    B = (rot(th) @ (FLIP if flip else np.eye(2)) @ xy[iB].T).T + t
    lb = lp[iB]
    ca = CertAlign(A, la, B, lb, h=0.35, tpad=8.0, arc_samples=8)
    t0 = time.time(); best, T, boxes, cert = ca.solve(min_dth=np.radians(1.0), min_dt=0.3, warm_start=True)
    sec = time.time() - t0
    _, pi = ca.score(T[0], T[1], T[2], return_pi=True)
    got = {int(k): int(v) for k, v in pi.items()}
    shared = [k for k in truth if k in got]
    match = sum(1 for k in shared if got[k] == truth[k])
    t_true = rot(th) @ (FLIP if flip else np.eye(2)) @ ca.c + t
    lv, _ = ca.cover(best, eps, min_dth=np.radians(1.5), min_dt=0.4, cap=300000)
    md = modes(lv)
    return {"applied angle/flip": [angle_deg, flip],
            "n spots A/B (different 70% each)": [len(A), len(B)],
            "domains RNA/protein": [int(KA), int(KB)],
            "best possible RNA-protein domain agreement (same spots)": round(agree, 3),
            "certified score": round(best, 4),
            "RECOVERED angle/flip": [round(float(np.degrees(T[0])), 1), T[2]],
            "angle error (deg)": round(float(abs(np.degrees(np.angle(np.exp(1j * (T[0] - th)))))), 2),
            "shift error (spots)": round(float(np.linalg.norm(T[1] - t_true)), 2),
            "cross-modal domain correspondence recovered": f"{match}/{len(shared)}",
            "modes within 3%": len(md), "boxes": boxes, "sec": round(sec, 1)}

if __name__ == "__main__":
    out = []
    for ang, fl in [(0.0, 0), (73.0, 1), (211.0, 0)]:
        r = go(ang, fl); out.append(r); print(json.dumps(r), flush=True)
    json.dump(out, open("results_blockC_crossmodal.json", "w"), indent=1)
    print("DONE", flush=True)
