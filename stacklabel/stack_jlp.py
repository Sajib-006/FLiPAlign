"""One-to-one stack labelling (chain, forest, star), the overlap tables and the scores (protocol fixed before any run).

K=8 consecutive Allen MERFISH sections at known (reconstructed) positions, each clustered on its own.
Consistent labelling of the stack: chain (exact on d=1), forest + ascent on the d=2 and d=3 overlap graphs,
star from the middle, and a pooled clustering ceiling. Scored against Allen CCF divisions.
Usage: python3 stacklabel/stack_jlp.py STACK_INDEX      -> results/stack_{i}.json
"""
import os, sys  # noqa: E401
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import repo_paths  # noqa: E402,F401  (puts the code folders on the import path)
import json, os, sys, time
import numpy as np, pandas as pd, h5py, scipy.sparse as sp
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import minimum_spanning_tree, breadth_first_order
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ARI, normalized_mutual_info_score as NMI
sys.path.insert(0, ".")
from bench_gt import UNIT, cluster
from bench_domains import neighbourhood_domains

D = "data/abc"; OUT = "results"; KS = 8; N = 8; GRID = 25.0; SIG = 50.0
T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def sections():
    p = pd.read_csv(f"{D}/parcellation.csv", usecols=["cell_label", "brain_section_label", "x_reconstructed", "y_reconstructed",
                                                     "z_reconstructed", "parcellation_division", "parcellation_structure"])
    p["cell_label"] = p.cell_label.str.replace(r"-1$", "", regex=True); p = p.drop_duplicates("cell_label").set_index("cell_label")
    have = {f.replace("-raw.h5ad", "") for f in os.listdir(D) if f.endswith("-raw.h5ad")}
    z = p.groupby("brain_section_label").z_reconstructed.first().sort_values(); cnt = p.brain_section_label.value_counts()
    secs = [s for s in z.index if s in have and cnt[s] > 20000]
    return p, secs


def load_section(p, sec, idx):
    cf = f"{D}/stackcache_{sec}.npz"
    if os.path.exists(cf):
        z = np.load(cf, allow_pickle=True); return {k: z[k] for k in z.files}
    with h5py.File(f"{D}/{sec}-raw.h5ad") as h:
        X = h["X"]; M = sp.csr_matrix((X["data"][:], X["indices"][:], X["indptr"][:]), shape=tuple(X.attrs["shape"]))
        cells = h["obs"]["cell_label"][:].astype(str)
    keep = np.asarray(pd.Index(cells).isin(p.index)); cells, M = cells[keep], M[keep]
    ok = np.asarray(M.sum(1)).ravel() >= 10; cells, M = cells[ok], M[ok]
    q = p.loc[cells]; xy = q[["x_reconstructed", "y_reconstructed"]].values * 1000.0
    X = np.asarray(M.todense(), np.float32)
    cl = cluster(X, 12, idx); lab = neighbourhood_domains(xy / UNIT, cl, K=N, seed=idx)
    cl6 = cluster(X, 6, idx + 100); lab_s1 = neighbourhood_domains(xy / UNIT, cl6, K=N, seed=idx + 100)   # stress S1
    out = dict(X=X.astype(np.float16), xy=xy, lab=lab, lab_s1=lab_s1, cl=cl, div=q.parcellation_division.astype(str).values,
               struct=q.parcellation_structure.astype(str).values)
    np.savez(cf + ".tmp.npz", **out); os.replace(cf + ".tmp.npz", cf)
    return out


def table(S, U):
    """C[k,l] = mean over cells of S with label k of the local fraction of U's label l (Gaussian, sigma 50 um)"""
    lo = np.minimum(S["xy"].min(0), U["xy"].min(0)) - 4 * SIG; hi = np.maximum(S["xy"].max(0), U["xy"].max(0)) + 4 * SIG
    shape = tuple(((hi - lo) / GRID).astype(int) + 1)
    iu = ((U["xy"] - lo) / GRID).astype(int); dens = np.zeros((N,) + shape)
    np.add.at(dens, (U["lab"], iu[:, 0], iu[:, 1]), 1.0)
    dens = np.stack([gaussian_filter(d, SIG / GRID) for d in dens]); tot = dens.sum(0)
    frac = dens / (tot + 0.05 * tot.max())
    cs = (S["xy"] - lo) / GRID
    v = np.stack([map_coordinates(f, cs.T, order=1) for f in frac], 1)          # cells of S x labels of U
    C = np.zeros((N, N)); np.add.at(C, S["lab"], v)
    return C / len(S["xy"])


def hung(W):
    r, c = linear_sum_assignment(-W); return c, W[r, c].sum()                 # c[k] = partner of k


def phi(sig, C, edges):
    """sig[s][k] = universe label of label k in section s"""
    tot = 0.0
    for s, u in edges:
        inv = np.argsort(sig[u]); tot += sum(C[(s, u)][k, inv[sig[s][k]]] for k in range(N))
    return tot


def ascent(sig, C, edges, K, rounds=50):
    nb = {s: [] for s in range(K)}
    for s, u in edges:
        nb[s].append((u, C[(s, u)])); nb[u].append((s, C[(s, u)].T))
    best = phi(sig, C, edges)
    for _ in range(rounds):
        moved = False
        for s in range(K):
            W = np.zeros((N, N))
            for u, Cs in nb[s]:
                inv = np.argsort(sig[u])                                          # universe g -> label of u
                W += Cs[:, inv]                                                    # W[k, g]
            new, _ = hung(W); old = sig[s]; sig[s] = new; v = phi(sig, C, edges)
            if v > best + 1e-12: best, moved = v, True
            else: sig[s] = old
        if not moved: break
    return sig


def cut_ascent(sig, C, edges, K, rounds=20):
    """block move for a stack: for each cut j, relabel the whole suffix j..K-1 by one permutation chosen by an
    assignment on the edges that cross the cut; never lowers Phi. Single-section moves cannot undo a wrong link
    that the whole suffix inherited, this move can."""
    best = phi(sig, C, edges)
    for _ in range(rounds):
        moved = False
        for j in range(1, K):
            W = np.zeros((N, N))
            for s, u in edges:
                if s < j <= u:
                    inv = np.argsort(sig[u])
                    for k in range(N): W[sig[s][k]] += C[(s, u)][k, inv]
            c, _ = hung(W); pi = np.argsort(c)
            if (pi == np.arange(N)).all(): continue
            old = {u: sig[u] for u in range(j, K)}
            for u in range(j, K): sig[u] = pi[sig[u]]
            v = phi(sig, C, edges)
            if v > best + 1e-12: best, moved = v, True
            else: sig.update(old)
        sig = ascent(sig, C, edges, K)
        nv = phi(sig, C, edges)
        if nv > best + 1e-12: best, moved = nv, True
        if not moved: break
    return sig


def forest(C, edges, K, d):
    E = [(s, u) for s, u in edges if u - s <= d]
    M = {e: hung(C[e])[1] for e in E}
    A = np.zeros((K, K))
    for (s, u), m in M.items(): A[s, u] = -(m + 1e-9)
    T = minimum_spanning_tree(sp.csr_matrix(A)).toarray(); T = (T != 0) | (T.T != 0)
    order, pred = breadth_first_order(sp.csr_matrix(T.astype(float)), 0, directed=False)
    sig = {0: np.arange(N)}
    for v in order[1:]:
        u = pred[v]; e = (min(u, v), max(u, v)); c, _ = hung(C[e])                # c: label of e[0] -> label of e[1]
        if e[0] == u: sig[v] = np.empty(N, int); sig[v][c] = sig[u]
        else: sig[v] = sig[u][c]
    sig = cut_ascent(ascent(sig, C, E, K), C, E, K)
    return sig, phi(sig, C, E) / max(sum(M.values()), 1e-12)


def chain(C, K):
    sig = {0: np.arange(N)}
    for s in range(K - 1):
        c, _ = hung(C[(s, s + 1)]); sig[s + 1] = np.empty(N, int); sig[s + 1][c] = sig[s]
    return sig


def star(C, K, m):
    sig = {m: np.arange(N)}
    for s in range(K):
        if s == m: continue
        e = (min(s, m), max(s, m)); c, _ = hung(C[e])
        if e[0] == m: sig[s] = np.empty(N, int); sig[s][c] = np.arange(N)
        else: sig[s] = c.copy()
    return sig


def pooled(S, seed=0):
    """ceiling (needs shared genes, not available across platforms): one clustering of all K sections together,
    same rule (12 clusters, neighbourhood mixes, 8 regions), so labels are consistent by construction"""
    from scipy.spatial import cKDTree
    from bench_gt import lognorm
    from sklearn.decomposition import PCA
    X = np.vstack([s["X"].astype(np.float32) for s in S])
    cl = KMeans(12, n_init=4, random_state=seed).fit_predict(PCA(20, random_state=seed).fit_transform(lognorm(X)))
    comps, o = [], 0
    for s in S:
        c = cl[o:o + len(s["xy"])]; o += len(s["xy"])
        _, nb = cKDTree(s["xy"] / UNIT).query(s["xy"] / UNIT, k=30); comp = np.zeros((len(c), 12))
        for j in range(30): comp[np.arange(len(c)), c[nb[:, j]]] += 1 / 30
        comps.append(comp)
    lab = KMeans(N, n_init=4, random_state=seed).fit_predict(np.vstack(comps))
    out, o = [], 0
    for s in S:
        out.append(lab[o:o + len(s["xy"])]); o += len(s["xy"])
    return out


def score(S, sig, key="lab"):
    lab = np.concatenate([sig[i][s[key]] for i, s in enumerate(S)]); div = np.concatenate([s["div"] for s in S])
    per = [ARI(s["div"], s[key]) for s in S]
    return dict(ari=round(ARI(div, lab), 4), nmi=round(NMI(div, lab), 4), per_section_ari=round(float(np.mean(per)), 4))


if __name__ == "__main__":
    i = int(sys.argv[1]); p, secs = sections()
    stack = secs[i * KS:(i + 1) * KS]; assert len(stack) == KS, (i, len(secs))
    log(f"stack {i}: {stack}")
    S = [load_section(p, sec, secs.index(sec)) for sec in stack]; del p
    log("sections loaded")
    res = {"stack": i, "sections": stack, "n_cells": [int(len(s["xy"])) for s in S], "runs": {}}
    rng = np.random.default_rng(7000 + i); mid = 3
    variants = {"clean": S}
    s1 = [dict(s) for s in S]; s1[mid] = dict(S[mid], lab=S[mid]["lab_s1"]); variants["S1_recluster"] = s1
    s2 = [dict(s) for s in S]; xs = S[mid]["xy"][:, 0]; keep = xs < np.median(xs)
    s2[mid] = {k: (v[keep] if hasattr(v, "__len__") and len(v) == len(xs) else v) for k, v in S[mid].items()}; variants["S2_crop50"] = s2
    def shuffled(s, frac, seed):
        r = np.random.default_rng(seed); lab = s["lab"].copy(); j = r.random(len(lab)) < frac; lab[j] = r.integers(0, N, j.sum())
        return dict(s, lab=lab)
    s3 = [dict(s) for s in S]; s3[mid] = shuffled(S[mid], 0.3, 1); variants["S3_noise30"] = s3
    s4 = [dict(s) for s in S]; s4[3] = shuffled(S[3], 0.3, 2); s4[5] = shuffled(S[5], 0.3, 3); variants["S4_two_noisy"] = s4
    for name, V in variants.items():
        C = {(s, u): table(V[s], V[u]) for s in range(KS) for u in range(s + 1, min(s + 4, KS))}
        C.update({(min(s, mid), max(s, mid)): table(V[min(s, mid)], V[max(s, mid)]) for s in range(KS) if s != mid and (min(s, mid), max(s, mid)) not in C})
        E2 = [(s, u) for s in range(KS) for u in range(s + 1, min(s + 3, KS))]
        sc = {}
        g = chain(C, KS); sc["chain"] = dict(score(V, g), phi_d2=round(phi(g, C, E2), 4))
        for d in (2, 3):
            g, ratio = forest(C, list(C.keys()), KS, d); sc[f"forest_d{d}"] = dict(score(V, g), phi_d2=round(phi(g, C, E2), 4), bound_ratio=round(ratio, 4))
        g = star(C, KS, mid); sc["star"] = dict(score(V, g), phi_d2=round(phi(g, C, E2), 4))
        sc["sumM_d2"] = round(sum(hung(C[e])[1] for e in E2), 4)
        if name == "clean":
            pl = pooled(V); lab = np.concatenate(pl); div = np.concatenate([s["div"] for s in V])
            sc["pooled_ceiling"] = dict(ari=round(ARI(div, lab), 4), nmi=round(NMI(div, lab), 4),
                                        per_section_ari=round(float(np.mean([ARI(s["div"], l) for s, l in zip(V, pl)])), 4))
        res["runs"][name] = sc; log(f"{name}: " + json.dumps({k: (v["ari"] if isinstance(v, dict) else v) for k, v in sc.items()}))
        json.dump(res, open(f"{OUT}/stack_{i}.json", "w"), indent=1)
    log("DONE")
