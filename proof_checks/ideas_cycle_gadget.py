"""A(ii): 3DM -> JLP on the cycle C_L, for L = 3, 4, 5, 6, with real squares.
Sections v1..vL (0-based 0..L-1). Core tables from the triangle reduction (check_reduction.build):
  edge (v1,v2) = E12, edge (v2,v3) = E23, edge (vL,v1) = E13 transposed; edges (v3,v4),...,(v_{L-1},vL) = W' * identity,
  W' = 3N+1 (identity-forcing). Target tau = (L-3) N W' + 3N.
Geometry: the triangle layout of the paper, with the old section-3 corridor cut into L-2 runs; consecutive runs share a block
of N W' corridor squares that carry label pair (k,k), W' squares per k.
Checks: no square in three sections; each section connected; overlap graph is exactly C_L; tables from squares equal the
designed tables; exact max of Phi (cycle DP over all permutations, N <= 6) is >= tau iff 3DM has a perfect matching.
Also (APX sketch, L = 3): 3(|T| + mu*) <= max Phi <= 2N + |T| + mu*, mu* = largest matching."""
import itertools, random
from collections import Counter
import numpy as np
from ideas_common import build, has_3dm, connected
from ideas_common import all_perms, edge_value_matrix


def max_matching(m, T):
    best = 0
    for r in range(1, min(m, len(T)) + 1):
        for S in itertools.combinations(T, r):
            if len({t[0] for t in S}) == r and len({t[1] for t in S}) == r and len({t[2] for t in S}) == r: best = r
    return best


def cycle_instance(m, T, L):
    N, E12, E13, E23 = build(m, T)
    Wp = 3 * N + 1
    E = {"12": sorted(E12), "13": sorted(E13), "23": sorted(E23)}
    run = N * Wp
    D = 3 * max(len(v) for v in E.values()) + 10 + (L - 3) * (run + 6)
    sec = {i: {} for i in range(L)}
    for j, (k, l) in enumerate(E["12"]): sec[0][(j, 0)] = k; sec[1][(j, 0)] = l
    for j, (k, l) in enumerate(E["23"]): sec[1][(j, D)] = k; sec[2][(j, D)] = l
    for j, (k, l) in enumerate(E["13"]): sec[0][(D + j, 0)] = k; sec[L - 1][(D + j, 0)] = l
    for x in range(len(E["12"]), D): sec[0][(x, 0)] = 0                 # corridor of v1
    for y in range(0, D + 1): sec[1][(-1, y)] = 0                        # corridor of v2
    Q = [(x, D) for x in range(len(E["23"]), D)] + [(D, y) for y in range(D, 0, -1)]   # old section-3 corridor, from A23 to A13
    starts = [5 + i * (run + 6) for i in range(L - 3)]                   # identity block between vertex 2+i and 3+i
    assert not starts or starts[-1] + run + 5 < len(Q)
    for v in range(2, L):
        lo = 0 if v == 2 else starts[v - 3]
        hi = len(Q) if v == L - 1 else starts[v - 2] + run
        for q in Q[lo:hi]: sec[v][q] = 0
    for i, st in enumerate(starts):                                       # identity labels in the shared block
        for t in range(run):
            q = Q[st + t]; sec[2 + i][q] = t % N; sec[3 + i][q] = t % N
    # designed tables
    Cd = {}
    def ind(Es, transpose=False):
        M = np.zeros((N, N))
        for (k, l) in Es:
            if transpose: M[l, k] = 1
            else: M[k, l] = 1
        return M
    Cd[(0, 1)] = ind(E["12"]); Cd[(1, 2)] = ind(E["23"])
    if L == 3: Cd[(0, 2)] = ind(E["13"])
    else:
        Cd[(0, L - 1)] = ind(E["13"])
        for i in range(L - 3): Cd[(2 + i, 3 + i)] = Wp * np.eye(N)
    tau = (L - 3) * N * Wp + 3 * N
    return N, sec, Cd, tau


def check_geometry(N, sec, Cd, L):
    cnt = Counter(c for s in sec for c in sec[s]); assert max(cnt.values()) <= 2, "square in three sections"
    for s in sec: assert connected(sec[s]), ("not connected", s)
    Cs = {}
    for s in range(L):
        for u in range(s + 1, L):
            sh = set(sec[s]) & set(sec[u])
            if sh:
                M = np.zeros((N, N))
                for c in sh: M[sec[s][c], sec[u][c]] += 1
                Cs[(s, u)] = M
    cyc = {tuple(sorted((i, (i + 1) % L))) for i in range(L)}
    assert set(Cs) == cyc, ("overlap graph is not C_L", sorted(Cs))
    for e in Cs: assert (Cs[e] == Cd[e]).all(), ("table differs", e)
    return Cs


def cycle_max(C, L, N, P):
    """exact max of Phi on the cycle 0-1-...-(L-1)-0, section 0 fixed to identity (perm index 0)."""
    V = {e: edge_value_matrix(C[e], P) for e in C}
    f = V[(0, 1)][0].copy()                                               # over perms of section 1
    for v in range(1, L - 1):
        f = (f[:, None] + V[(v, v + 1)]).max(0)
    close = V[(0, L - 1)][0] if L > 3 or (0, L - 1) in V else None
    return (f + close).max()


random.seed(5)
Ps = {n: all_perms(n) for n in range(2, 7)}
stats = Counter(); apx_ok = 0
for trial in range(240):
    m = random.choice([1, 2, 2, 3])
    allT = [(x, y, z) for x in range(m) for y in range(m) for z in range(m)]
    maxT = 6 - m
    T = random.sample(allT, random.randint(min(m, len(allT)), min(len(allT), maxT)))
    L = random.choice([3, 4, 5, 6])
    N, sec, Cd, tau = cycle_instance(m, T, L)
    C = check_geometry(N, sec, Cd, L)
    yes = has_3dm(m, T)
    best = cycle_max(C, L, N, Ps[N])
    assert (best >= tau - 1e-9) == yes, ("reduction fails", m, T, L, best, tau)
    if L == 3:
        mu = max_matching(m, T)
        assert 3 * (len(T) + mu) <= best + 1e-9 and best <= 2 * N + len(T) + mu + 1e-9, ("apx bound", m, T, best, mu)
        apx_ok += 1
    stats[(L, yes)] += 1
print("cycle instances by (L, 3DM yes):", dict(sorted(stats.items())))
print("geometry (connected, no triple square, overlap graph = C_L, tables) and max Phi >= tau <=> 3DM: PASSED on all")
print("APX sandwich 3(|T|+mu*) <= max Phi <= 2N+|T|+mu* checked on", apx_ok, "triangle instances: PASSED")
