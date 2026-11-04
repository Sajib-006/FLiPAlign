"""B: approximation by forests. Algorithm MSF: weight each edge by its own optimum M_e, take a maximum-weight spanning forest,
solve it exactly by chained Hungarian, use that joint pairing for the whole instance. Claim: value >= sum_e M_e / gamma(H)
>= OPT / gamma(H), gamma = fractional arboricity max_S |E(S)|/(|S|-1) (<= arboricity a). Then coordinate ascent (never lowers).
Checks on small instances with brute-force OPT: the bound holds; worst observed ratios; local optima of plain coordinate
ascent from random starts (no forest start); the tight family on the squared path P_K^2."""
import itertools
import numpy as np
from ideas_common import (brute_opt, hungarian, chain_forest, value, max_spanning_forest, coordinate_ascent,
                          fractional_arboricity, all_perms)

rng = np.random.default_rng(21)


def graphs(K):
    comp = list(itertools.combinations(range(K), 2))
    P2 = [(i, j) for (i, j) in comp if j - i <= 2]
    P3 = [(i, j) for (i, j) in comp if j - i <= 3]
    cyc = [(i, i + 1) for i in range(K - 1)] + [(0, K - 1)]
    return {"complete": comp, "path^2": P2, "path^3": P3, "cycle": cyc}


def rand_tables(edges, N, kind):
    C = {}
    for e in edges:
        if kind == "sparse": T = rng.integers(0, 10, (N, N)) * (rng.random((N, N)) < 0.4)
        elif kind == "diag":                              # real-data like: strong near-diagonal after a hidden relabelling
            T = rng.integers(0, 3, (N, N)) + 10 * np.eye(N, dtype=int)[rng.permutation(N)]
        else: T = rng.integers(0, 10, (N, N))
        C[e] = T.astype(float)
    return C


rows = []
worst = {}
for K, N, reps in ((4, 3, 30), (5, 3, 20), (4, 4, 10), (6, 3, 4)):
    for gname, edges in graphs(K).items():
        if gname == "path^3" and K <= 4: continue
        gam = fractional_arboricity(K, edges)
        for t in range(reps):
            kind = ("sparse", "dense", "diag")[t % 3]
            C = rand_tables(edges, N, kind)
            M = {e: hungarian(C[e])[0] for e in C}; S = sum(M.values())
            opt = brute_opt(C, K, N)
            F = max_spanning_forest(K, M)
            p = chain_forest(C, K, N, F); v = value(C, p)
            _, va = coordinate_ascent(C, K, N, p)
            assert opt <= S + 1e-9
            assert v >= S / gam - 1e-9, (K, N, gname, v, S, gam)
            # plain coordinate ascent from 5 random starts: record worst local optimum
            lo = min(coordinate_ascent(C, K, N, [rng.permutation(N) for _ in range(K)])[1] for _ in range(5))
            key = (gname, K, N)
            w = worst.setdefault(key, [1, 1, 1, 1, gam, 0])
            if opt > 0:
                w[0] = min(w[0], v / opt); w[1] = min(w[1], va / opt); w[2] = min(w[2], lo / opt); w[3] = min(w[3], opt / S)
            w[5] += 1
print("graph      K N  n   gamma  1/gamma  worst MSF/OPT  worst (MSF+ascent)/OPT  worst ascent-local-opt/OPT  worst OPT/sumM")
for (g, K, N), (a, b, c, d, gam, n) in sorted(worst.items()):
    print(f"{g:9s} {K} {N} {n:3d}  {gam:5.2f}  {1/gam:5.3f}     {a:5.3f}           {b:5.3f}                 {c:5.3f}                 {d:5.3f}")

# tight family on P_K^2: C(k,l) = [k != l] + eps [l = k + N/2]; path edges scaled by (1+delta) so MSF = the path.
eps, delta = 0.01, 0.001
for K, N in ((4, 4), (5, 4)):
    edges = graphs(K)["path^2"]
    base = np.array([[float(k != l) + eps * (l == (k + N // 2) % N) for l in range(N)] for k in range(N)])
    C = {e: base * (1 + delta) if e[1] - e[0] == 1 else base.copy() for e in edges}
    M = {e: hungarian(C[e])[0] for e in C}
    F = max_spanning_forest(K, M); v = value(C, chain_forest(C, K, N, F)); opt = brute_opt(C, K, N)
    gam = fractional_arboricity(K, edges)
    print(f"tight family P_{K}^2, N={N}: MSF value {v:.3f}, OPT {opt:.3f}, ratio {v/opt:.3f}, 1/gamma {1/gam:.3f}, (K-1)/(2K-3) {(K-1)/(2*K-3):.3f}")
print("bound MSF >= sum M / gamma >= OPT / gamma held on every instance. PASSED")
