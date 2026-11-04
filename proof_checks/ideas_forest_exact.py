"""A(i): on a forest overlap graph, the optimum equals the sum of the edge optima, and chained Hungarian reaches it.
Random forests with K <= 6 sections, N <= 4 labels; tables random (non-negative, and also with negative entries:
exactness does not need non-negativity). Brute force over all joint pairings."""
import numpy as np
from ideas_common import brute_opt, hungarian, chain_forest, value

rng = np.random.default_rng(11)
n_inst = 0
for K, N, reps in ((3, 4, 40), (4, 3, 40), (4, 4, 15), (5, 3, 20), (6, 3, 8)):
    for t in range(reps):
        # random forest: each vertex s>0 attaches to a random earlier vertex or (prob 0.25) starts a new tree
        edges = [(int(rng.integers(0, s)), s) for s in range(1, K) if rng.random() > 0.25]
        signed = (t % 2 == 1)
        C = {}
        for e in edges:
            T = rng.integers(0, 10, (N, N)).astype(float)
            if signed: T -= 4.0
            C[e] = T
        if not C: continue
        opt = brute_opt(C, K, N)
        S = sum(hungarian(C[e])[0] for e in C)
        chained = value(C, chain_forest(C, K, N, list(C)))
        assert abs(opt - S) < 1e-9, (K, N, edges, opt, S)
        assert abs(chained - S) < 1e-9, (K, N, edges, chained, S)
        n_inst += 1
print("forest instances", n_inst, ": brute-force OPT == sum of edge optima == chained Hungarian on all. PASSED")
