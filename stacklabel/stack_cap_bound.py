"""Certificates for the split labelling of stacks (2026-10-21).

For each stack and variant (same as stack_cap.py): (1) the pair bound U_cap = sum over compared pairs of the exact optimum
of that pair alone (integer program, at most G=8 classes, at most 2 labels of a section per class), valid because the
restriction of any stack labelling to one pair is feasible for the pair; (2) the exact stack optimum by one integer
program with a time limit (its dual bound is also an upper bound). Reports the local-search value of stack_cap.py
against both. Usage: python3 stacklabel/stack_cap_bound.py STACK_INDEX -> results/stackbound2_{i}.json
"""
import os, sys  # noqa: E401
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import repo_paths  # noqa: E402,F401  (puts the code folders on the import path)
import json, sys, time
import numpy as np
import scipy.sparse as sp
from scipy.optimize import milp, LinearConstraint, Bounds
sys.path.insert(0, ".")
import stack_jlp as J
try:
    import stack_cap as SC
except ImportError:
    import stack_cc as SC

KS, N, G, CAP = J.KS, J.N, J.N, 2
import os
FULL_TL = float(os.environ.get("FULL_TL", 900))


def solve(W, secs, edges, time_limit=600):
    """integer program over the labels of the sections in `secs` with weights W (node index s*N+k) on `edges`"""
    nodes = [s * N + k for s in secs for k in range(N)]; ix = {v: i for i, v in enumerate(nodes)}; nv = len(nodes)
    pairs = [(a, b) for (s, u) in edges for a in range(s * N, (s + 1) * N) for b in range(u * N, (u + 1) * N) if abs(W[a, b]) > 1e-12]
    nx = nv * G; nt = len(pairs) * G; n = nx + nt
    X = lambda v, c: ix[v] * G + c; T = lambda p, c: nx + p * G + c
    cost = np.zeros(n)
    for p, (a, b) in enumerate(pairs):
        for c in range(G): cost[T(p, c)] = -W[a, b]
    rows, cols, vals, lo, hi = [], [], [], [], []; r = 0
    def add(entries, l, h):
        nonlocal r
        for j, v in entries: rows.append(r); cols.append(j); vals.append(v)
        lo.append(l); hi.append(h); r += 1
    for v in nodes: add([(X(v, c), 1) for c in range(G)], 1, 1)                     # each label in one class
    for s in secs:
        for c in range(G): add([(X(s * N + k, c), 1) for k in range(N)], 0, CAP)        # cap per section per class
    for p, (a, b) in enumerate(pairs):
        for c in range(G):
            if W[a, b] > 0:
                add([(T(p, c), 1), (X(a, c), -1)], -np.inf, 0); add([(T(p, c), 1), (X(b, c), -1)], -np.inf, 0)
            else:
                add([(T(p, c), 1), (X(a, c), -1), (X(b, c), -1)], -1, np.inf)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(r, n))
    lb = np.zeros(n); ub = np.ones(n); lb[X(nodes[0], 0)] = 1                          # symmetry: first label in class 0
    integ = np.zeros(n); integ[:nx] = 1
    t0 = time.time()
    res = milp(cost, constraints=LinearConstraint(A, lo, hi), integrality=integ, bounds=Bounds(lb, ub),
               options={"time_limit": time_limit, "disp": False, "mip_rel_gap": 1e-9})
    best = -res.fun if res.x is not None else None
    dual = -getattr(res, "mip_dual_bound", np.nan) if hasattr(res, "mip_dual_bound") else None
    return dict(value=None if best is None else round(float(best), 5), dual_bound=None if dual is None else round(float(dual), 5),
                status=int(res.status), seconds=round(time.time() - t0, 1))


if __name__ == "__main__":
    i = int(sys.argv[1]); p, secs = J.sections()
    stack = secs[i * KS:(i + 1) * KS]
    S = [J.load_section(p, sec, secs.index(sec)) for sec in stack]; del p
    prev = json.load(open(f"results/stackcap_{i}.json"))
    mid = 3; variants = {"clean": S}
    s1 = [dict(s) for s in S]; s1[mid] = dict(S[mid], lab=S[mid]["lab_s1"]); variants["S1_recluster"] = s1
    def shuffled(s, frac, seed):
        r = np.random.default_rng(seed); lab = s["lab"].copy(); j = r.random(len(lab)) < frac; lab[j] = r.integers(0, N, j.sum())
        return dict(s, lab=lab)
    s2 = [dict(s) for s in S]; xs = S[mid]["xy"][:, 0]; keep = xs < np.median(xs)
    s2[mid] = {k: (v[keep] if hasattr(v, "__len__") and len(v) == len(xs) else v) for k, v in S[mid].items()}; variants["S2_crop50"] = s2
    s3 = [dict(s) for s in S]; s3[mid] = shuffled(S[mid], 0.3, 1); variants["S3_noise30"] = s3
    s4 = [dict(s) for s in S]; s4[3] = shuffled(S[3], 0.3, 2); s4[5] = shuffled(S[5], 0.3, 3); variants["S4_two_noisy"] = s4
    out = {"stack": i, "runs": {}}
    for name, V in variants.items():
        C = {(s, u): J.table(V[s], V[u]) for s in range(KS) for u in range(s + 1, min(s + 3, KS))}
        rr = {}
        for d in (1, 2):
            E = [(s, u) for (s, u) in C if u - s <= d]; W = SC.weights(C, E)
            pair = [solve(W, [s, u], [(s, u)], time_limit=600) for (s, u) in E]
            if any(x["status"] != 0 or x["dual_bound"] is None for x in pair):
                raise RuntimeError(f"pair not solved to proven optimality: {pair}")
            ucap = sum(x["dual_bound"] for x in pair)                 # proven upper bounds, not best values found
            full = solve(W, list(range(KS)), E, time_limit=FULL_TL)
            ls = prev["runs"][name][f"cap2_d{d}"]["value"]
            rr[f"d{d}"] = dict(local_search=ls, pair_bound=round(ucap, 5), full_program=full, pairs_all_optimal=True,
                               ratio_to_pair_bound=round(ls / ucap, 4),
                               ratio_to_best_found=round(ls / full["value"], 4) if full["value"] else None,
                               ratio_to_full_dual=round(ls / full["dual_bound"], 4) if full["dual_bound"] else None)
            print(name, d, json.dumps(rr[f"d{d}"]), flush=True)
        out["runs"][name] = rr
        json.dump(out, open(f"results/stackbound2_{i}.json", "w"), indent=1)
    print("DONE", flush=True)
