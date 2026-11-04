"""Split labelling of a stack (Algorithm 1 of the paper): merges and splits with a cap per section.

Same stacks, labels, tables and stress variants as stack_jlp.py. Each label of each section is a node; compared pairs
of sections (d=1 or d=2) give weights w(k,l) = C_su(k,l) - (row_k * col_l)/total (agreement above chance). A stack
labelling is a partition of all nodes into classes (labels of one section may share a class: a split), and its value
is the sum of w over node pairs inside a class (correlation clustering on a K-partite graph). Putting everything in one
class scores 0, so there is no trivial answer. The one-to-one problem of stack_jlp.py is the restricted case with
exactly one label of each section per class.
Solver: start from the one-to-one forest answer, then local search (move one node to another class or a new one;
merge two classes) until no move gains. Scored like stack_jlp.py.
Usage: python3 stack_cc.py STACK_INDEX -> results/stackcap_{i}.json
"""
import os, sys  # noqa: E401
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import repo_paths  # noqa: E402,F401  (puts the code folders on the import path)
import json, sys, time
import numpy as np
from sklearn.metrics import adjusted_rand_score as ARI, normalized_mutual_info_score as NMI
sys.path.insert(0, ".")
import stack_jlp as J

KS, N = J.KS, J.N


def contrast(C):
    r, c, t = C.sum(1), C.sum(0), C.sum()
    return C - np.outer(r, c) / max(t, 1e-12)


def weights(C, edges):
    n = KS * N; W = np.zeros((n, n))
    for s, u in edges:
        B = contrast(C[(s, u)]); W[s * N:(s + 1) * N, u * N:(u + 1) * N] = B; W[u * N:(u + 1) * N, s * N:(s + 1) * N] = B.T
    return W


def value(W, z):
    same = z[:, None] == z[None, :]
    return float((W * same).sum() / 2)


def local_search(W, z, rounds=200):
    n = len(z); z = z.copy()
    for _ in range(rounds):
        moved = False
        for i in range(n):
            cls = np.unique(z); cur = z[i]
            g = {c: W[i, z == c].sum() - (W[i, i] if c == cur else 0) for c in cls}
            base = g[cur]; best_c, best_g = cur, base
            for c, v in g.items():
                if v > best_g + 1e-12: best_c, best_g = c, v
            if 0 > best_g + 1e-12: best_c = z.max() + 1; best_g = 0.0        # a new class of its own
            if best_c != cur: z[i] = best_c; moved = True
        cls = np.unique(z)
        for a in cls:                                                       # merge two classes when their link is positive
            for b in cls:
                if a < b and (z == a).any() and (z == b).any() and W[np.ix_(z == a, z == b)].sum() > 1e-12:
                    z[z == b] = a; moved = True
        if not moved: break
    _, z = np.unique(z, return_inverse=True)
    return z


def capped_search(W, z, G, cap=2, rounds=200):
    """fixed number of classes G; each class holds at most `cap` labels of one section (a split of at most cap);
    moves: send one label to another class within the cap, or swap two labels of one section between classes"""
    z = z.copy(); n = len(z)
    def gain_move(i, c):
        return W[i, z == c].sum() - W[i, z == z[i]].sum() + W[i, i]
    for _ in range(rounds):
        moved = False
        for i in range(n):
            s = i // N; mine = np.arange(s * N, (s + 1) * N)
            best_c, best_g = z[i], 1e-12
            for c in range(G):
                if c == z[i] or (z[mine] == c).sum() >= cap: continue
                g = gain_move(i, c)
                if g > best_g: best_c, best_g = c, g
            if best_c != z[i]: z[i] = best_c; moved = True
            for j in mine:                                                  # swap with a label of the same section
                if j == i or z[j] == z[i]: continue
                ci, cj = z[i], z[j]
                v0 = value(W, z); z[i], z[j] = cj, ci
                if value(W, z) > v0 + 1e-12: moved = True
                else: z[i], z[j] = ci, cj
        if not moved: break
    return z


def score(V, z):
    lab = np.concatenate([z[s * N + V[s]["lab"]] for s in range(KS)]); div = np.concatenate([v["div"] for v in V])
    return dict(ari=round(ARI(div, lab), 4), nmi=round(NMI(div, lab), 4), n_classes=int(len(np.unique(lab))))


if __name__ == "__main__":
    i = int(sys.argv[1]); p, secs = J.sections()
    stack = secs[i * KS:(i + 1) * KS]
    S = [J.load_section(p, sec, secs.index(sec)) for sec in stack]; del p
    mid = 3; variants = {"clean": S}
    s1 = [dict(s) for s in S]; s1[mid] = dict(S[mid], lab=S[mid]["lab_s1"]); variants["S1_recluster"] = s1
    s2 = [dict(s) for s in S]; xs = S[mid]["xy"][:, 0]; keep = xs < np.median(xs)
    s2[mid] = {k: (v[keep] if hasattr(v, "__len__") and len(v) == len(xs) else v) for k, v in S[mid].items()}; variants["S2_crop50"] = s2
    def shuffled(s, frac, seed):
        r = np.random.default_rng(seed); lab = s["lab"].copy(); j = r.random(len(lab)) < frac; lab[j] = r.integers(0, N, j.sum())
        return dict(s, lab=lab)
    s3 = [dict(s) for s in S]; s3[mid] = shuffled(S[mid], 0.3, 1); variants["S3_noise30"] = s3
    s4 = [dict(s) for s in S]; s4[3] = shuffled(S[3], 0.3, 2); s4[5] = shuffled(S[5], 0.3, 3); variants["S4_two_noisy"] = s4
    res = {"stack": i, "sections": stack, "runs": {}}
    for name, V in variants.items():
        C = {(s, u): J.table(V[s], V[u]) for s in range(KS) for u in range(s + 1, min(s + 3, KS))}
        out = {}
        sig, _ = J.forest(C, list(C), KS, 2)
        z0 = np.concatenate([sig[s] for s in range(KS)])                     # one-to-one start: class = universe label
        out["jlp_forest_d2"] = J.score(V, sig)
        for d in (1, 2):
            E = [(s, u) for (s, u) in C if u - s <= d]; W = weights(C, E)
            z = local_search(W, z0)
            out[f"cc_d{d}"] = dict(score(V, z), value=round(value(W, z), 4), value_start=round(value(W, z0), 4),
                                   pos_weight_bound=round(float(np.clip(W, 0, None).sum() / 2), 4))
            zc = capped_search(W, z0, N, cap=2)
            out[f"cap2_d{d}"] = dict(score(V, zc), value=round(value(W, zc), 4), value_start=round(value(W, z0), 4))
        out["jlp_forest_d2"].update(score(V, z0))
        res["runs"][name] = out
        print(name, json.dumps({k: v["ari"] for k, v in out.items()}), flush=True)
        json.dump(res, open(f"results/stackcap_{i}.json", "w"), indent=1)
    print("DONE", flush=True)
