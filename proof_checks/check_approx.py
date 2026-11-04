"""Check (2/K)-approximation of the 'best reference section' algorithm for joint label pairing, small random instances.
Instance: K sections, N labels each, nonnegative tables C[s][u] (N x N) for s<u. OPT = max over bijections (to universe, section 0 fixed)
of sum_{s<u} sum_g C_su(sigma_s^-1 g, sigma_u^-1 g). Algorithm: for each reference r, match every other section to r by Hungarian; take best r."""
import itertools, numpy as np
from scipy.optimize import linear_sum_assignment
rng = np.random.default_rng(3)
def value(C, K, N, perms):                       # perms[s] maps universe element g -> label of section s
    v = 0.0
    for s in range(K):
        for u in range(s + 1, K):
            v += sum(C[(s, u)][perms[s][g], perms[u][g]] for g in range(N))
    return v
worst = 1.0; trials = 0
for K, N in ((3, 4), (3, 5), (4, 3), (4, 4)):
    for t in range(60 if N < 5 else 25):
        C = {(s, u): rng.integers(0, 10, (N, N)).astype(float) * (rng.random((N, N)) < rng.choice([0.2, 0.5, 1.0])) for s in range(K) for u in range(s + 1, K)}
        M = {k: C[k][linear_sum_assignment(-C[k])].sum() for k in C}
        opt = 0.0
        allp = list(itertools.permutations(range(N)))
        for rest in itertools.product(allp, repeat=K - 1):
            opt = max(opt, value(C, K, N, [tuple(range(N))] + list(rest)))
        assert opt <= sum(M.values()) + 1e-9                          # OPT <= sum of pairwise optima
        best = 0.0
        for r in range(K):
            perms = [None] * K; perms[r] = tuple(range(N))
            for s in range(K):
                if s == r: continue
                T = C[(min(r, s), max(r, s))]; T = T if r < s else T.T          # T[a, b]: label a of r, label b of s
                row, col = linear_sum_assignment(-T); p = [0] * N
                for a, b in zip(row, col): p[a] = b                              # universe element a (label a of r) carries label b of s
                perms[s] = tuple(p)
            best = max(best, value(C, K, N, perms))
        ratio = best / opt if opt > 0 else 1.0; worst = min(worst, ratio); trials += 1
        assert best >= 2.0 / K * opt - 1e-9, (K, N, best, opt)
print("trials", trials, "worst ratio", round(worst, 3), "all satisfy >= 2/K")
