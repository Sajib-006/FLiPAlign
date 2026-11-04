"""Stack labelling, second experiment: disjoint gene panels, class-matched control, gene baseline.
Same 6 stacks of 8 Allen MERFISH sections, positions from the Allen reconstruction.
(1) DISJOINT GENE PANELS: the 500 genes are split at random into two halves; even sections are clustered on half A and
    odd sections on half B (same rule: 12 clusters, then 8 regions). No gene is shared between neighbours, as across
    platforms, so pooled clustering and gene-based matching cannot be run on this version.
(2) CLASS-MATCHED CONTROL: the one-to-one (forest) labelling is merged greedily (largest positive weight between two
    classes first) down to the number of classes the split method used, so both use the same number of classes.
(3) GENE BASELINE (shared panel only): match each section's regions to the previous section's by correlation of mean
    expression, Hungarian, chained.
Scores: NMI and ARI against Allen divisions. Usage: python3 runs_stacks/stack_v2.py STACK -> results/stackv2_{i}.json
"""
import os, sys  # noqa: E401
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import repo_paths  # noqa: E402,F401  (puts the code folders on the import path)
import json, sys
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score as ARI, normalized_mutual_info_score as NMI
sys.path.insert(0, ".")
import stack_jlp as J
import stack_cap as SC
from bench_gt import UNIT, cluster
from bench_domains import neighbourhood_domains

KS, N = J.KS, J.N


def score(V, z):
    lab = np.concatenate([z[s * N + V[s]["lab"]] for s in range(KS)]); div = np.concatenate([v["div"] for v in V])
    return dict(nmi=round(NMI(div, lab), 4), ari=round(ARI(div, lab), 4), n_classes=int(len(np.unique(lab))))


def merge_to(W, z, target):
    z = z.copy()
    while len(np.unique(z)) > target:
        cl = np.unique(z); best, pair = -np.inf, None
        for a in cl:
            for b in cl:
                if a < b:
                    v = W[np.ix_(z == a, z == b)].sum()
                    if v > best: best, pair = v, (a, b)
        z[z == pair[1]] = pair[0]
    return np.unique(z, return_inverse=True)[1]


def run(V, name, out, genes=True):
    C = {(s, u): J.table(V[s], V[u]) for s in range(KS) for u in range(s + 1, min(s + 3, KS))}
    sig, _ = J.forest(C, list(C), KS, 2); z0 = np.concatenate([sig[s] for s in range(KS)])
    E = [(s, u) for (s, u) in C if u - s <= 2]; W = SC.weights(C, E)
    zc = SC.capped_search(W, z0, N, cap=2)
    r = {"forest": score(V, z0), "split": score(V, zc)}
    r["forest_merged"] = score(V, merge_to(W, z0, len(np.unique(zc))))
    if genes:                                                     # gene baseline needs a shared panel
        mu = [np.vstack([np.log1p(v["X"].astype(np.float32)[v["lab"] == k]).mean(0) for k in range(N)]) for v in V]
        g = {0: np.arange(N)}
        for s in range(1, KS):
            Cg = np.corrcoef(mu[s], mu[s - 1])[:N, N:]; rr, cc = linear_sum_assignment(-Cg)
            g[s] = np.empty(N, int); g[s][rr] = g[s - 1][cc]
        r["gene_chain"] = score(V, np.concatenate([g[s] for s in range(KS)]))
    out[name] = r; print(name, json.dumps({k: v["nmi"] for k, v in r.items()}), flush=True)


if __name__ == "__main__":
    i = int(sys.argv[1]); p, secs = J.sections()
    stack = secs[i * KS:(i + 1) * KS]
    S = [J.load_section(p, sec, secs.index(sec)) for sec in stack]; del p
    out = {}
    run(S, "shared_panel", out, genes=True)
    rng = np.random.default_rng(500 + i); G = S[0]["X"].shape[1]; perm = rng.permutation(G); half = [perm[: G // 2], perm[G // 2:]]
    D = []
    for s, sec in enumerate(S):
        X = sec["X"].astype(np.float32)[:, half[s % 2]]; idx = secs.index(stack[s])
        lab = neighbourhood_domains(sec["xy"] / UNIT, cluster(X, 12, idx + 300), K=N, seed=idx + 300)
        D.append(dict(sec, lab=lab))
    run(D, "disjoint_panels", out, genes=False)
    json.dump({"stack": i, "runs": out}, open(f"results/stackv2_{i}.json", "w"), indent=1)
    print("DONE", flush=True)
