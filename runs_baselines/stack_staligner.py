"""STAligner baseline on the 6 Allen MERFISH stacks (8 sections each). Usage: python3 runs_baselines/stack_staligner.py STACK [SEED]
Shared 500-gene panel, log-normalized counts, radius graph per section, train_STAligner_subgraph (1000 epochs),
KMeans on the joint embedding with 8 and 12 clusters, NMI/ARI vs Allen division over the whole stack."""
import os, sys  # noqa: E401
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import repo_paths  # noqa: E402,F401  (puts the code folders on the import path)
import json, sys, time, os
import numpy as np, scipy.sparse as sp, anndata as ad, scanpy as sc, torch
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ARI, normalized_mutual_info_score as NMI
sys.path.insert(0, "."); sys.path.insert(0, "STAligner")
import STAligner
import stack_jlp as J

RAD = 30.0   # um; about 6 neighbours per cell (60 and 40 um ran out of GPU memory on A30 in the pair stage)
T0 = time.time()
i = int(sys.argv[1]); seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
p, secs = J.sections(); stack = secs[i * 8:(i + 1) * 8]
Bl = []
for k, sec in enumerate(stack):
    V = J.load_section(p, sec, secs.index(sec))
    MAXC = int(os.environ.get("MAXC", 0))
    if MAXC and len(V["xy"]) > MAXC:   # memory fallback: random subsample of cells in this section
        keep = np.sort(np.random.default_rng(seed).choice(len(V["xy"]), MAXC, replace=False)); V = {k: V[k][keep] for k in ("X", "xy", "div")}
    a = ad.AnnData(sp.csr_matrix(V["X"].astype(np.float32)))
    a.obs_names = [f"s{k}_{j}" for j in range(a.n_obs)]; a.var_names = [f"g{j}" for j in range(a.n_vars)]
    a.obsm["spatial"] = V["xy"]; a.obs["div"] = V["div"]; a.obs["batch_name"] = str(k)
    sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a); a.X = sp.csr_matrix(a.X)
    pr = cKDTree(V["xy"]).query_pairs(RAD, output_type="ndarray")
    n = a.n_obs; r = np.r_[pr[:, 0], pr[:, 1], np.arange(n)]; c = np.r_[pr[:, 1], pr[:, 0], np.arange(n)]
    a.uns["adj"] = sp.coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n)); a.uns["Spatial_Net"] = None
    print(sec, n, "edges/cell", round(len(r) / n - 1, 2), flush=True); Bl.append(a)
del p
cat = ad.concat(Bl, join="inner", label="slice_name", keys=[str(k) for k in range(8)]); cat.obs["batch_name"] = cat.obs["slice_name"].astype(str)
cat.obs_names_make_unique()
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); print("device", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "", flush=True)
combs = [(k, k + 1) for k in range(7)]
cat = STAligner.train_STAligner_subgraph(cat, verbose=True, knn_neigh=100, n_epochs=1000, iter_comb=combs, Batch_list=Bl, random_seed=666 + seed, device=dev)
Z = cat.obsm["STAligner"]; div = cat.obs["div"].values; res = {"maxc": int(os.environ.get("MAXC", 0)), "stack": i, "seed": seed, "radius_um": RAD, "n_cells": int(cat.n_obs), "train_s": round(time.time() - T0)}
for K in (8, 12):
    lab = KMeans(K, n_init=10, random_state=seed).fit_predict(Z)
    res[f"k{K}"] = dict(nmi=round(NMI(div, lab), 4), ari=round(ARI(div, lab), 4))
    # also embedding without joint training, for reference (STAGATE pretrain output)
    lab = KMeans(K, n_init=10, random_state=seed).fit_predict(cat.obsm["STAGATE"]); res[f"stagate_k{K}"] = dict(nmi=round(NMI(div, lab), 4), ari=round(ARI(div, lab), 4))
res["wall_s"] = round(time.time() - T0); res["torch"] = torch.__version__
print(json.dumps(res), flush=True)
os.makedirs("results", exist_ok=True); json.dump(res, open(f"results/stackstal_{i}" + (f"_s{seed}" if seed else "") + ".json", "w"), indent=1)
