"""Shared helpers for the ideas_*.py checks (2026-10-21).
Convention: perms[s][g] = label of section s at universe element g. Edge table C[(s,u)][k,l] = overlap of label k of s with label l of u.
Value of a joint pairing = sum over edges (s,u) of sum_g C[(s,u)][perms[s][g], perms[u][g]]."""
import itertools
import numpy as np
from scipy.optimize import linear_sum_assignment


def all_perms(N):
    return np.array(list(itertools.permutations(range(N))), dtype=int)


def edge_value_matrix(C, P):
    """V[a,b] = sum_g C[P[a,g], P[b,g]] for all pairs of permutations a (first section) and b (second section)."""
    return C[P[:, None, :], P[None, :, :]].sum(-1)


def value(C, perms):
    return sum(C[(s, u)][perms[s], perms[u]].sum() for (s, u) in C)


def brute_opt(C, K, N):
    """exact max over all joint pairings, section 0 fixed to identity (gauge); exhaustive."""
    P = all_perms(N)
    V = {e: edge_value_matrix(C[e], P) for e in C}
    best = -np.inf
    for rest in itertools.product(range(len(P)), repeat=K - 1):
        idx = (0,) + rest                       # permutation 0 is the identity
        v = sum(V[(s, u)][idx[s], idx[u]] for (s, u) in V)
        if v > best: best = v
    return best


def hungarian(T):
    """best bijection for table T (rows: labels of first section, cols: labels of second). returns (value, pi) with pi[k] = partner of k."""
    r, c = linear_sum_assignment(-T)
    pi = np.empty(T.shape[0], dtype=int); pi[r] = c
    return T[r, c].sum(), pi


def table(C, s, u):
    return C[(s, u)] if (s, u) in C else C[(u, s)].T


def chain_forest(C, K, N, forest_edges):
    """exact pairing for the edges in a forest: root each tree, child label at universe g = partner of parent's label at g."""
    adj = {s: [] for s in range(K)}
    for (s, u) in forest_edges: adj[s].append(u); adj[u].append(s)
    perms = [None] * K
    for root in range(K):
        if perms[root] is not None: continue
        perms[root] = np.arange(N); stack = [root]
        while stack:
            s = stack.pop()
            for u in adj[s]:
                if perms[u] is None:
                    _, pi = hungarian(table(C, s, u))
                    perms[u] = pi[perms[s]]; stack.append(u)
    return perms


def max_spanning_forest(K, weights):
    """Kruskal on edge weights (dict edge -> weight), largest first."""
    parent = list(range(K))
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    F = []
    for e in sorted(weights, key=lambda e: -weights[e]):
        a, b = find(e[0]), find(e[1])
        if a != b: parent[a] = b; F.append(e)
    return F


def coordinate_ascent(C, K, N, perms, max_rounds=100):
    """re-optimise one section at a time against all others (assignment), until no gain."""
    perms = [p.copy() for p in perms]; cur = value(C, perms)
    for _ in range(max_rounds):
        improved = False
        for s in range(K):
            # gain matrix G[k, g]: putting label k of s at universe element g
            G = np.zeros((N, N))
            for u in range(K):
                if u == s or ((s, u) not in C and (u, s) not in C): continue
                T = table(C, s, u)                                  # T[k, l]
                G += T[:, perms[u]]                                 # column g: label perms[u][g] of u
            r, c = linear_sum_assignment(-G)
            new = np.empty(N, dtype=int); new[c] = r                # universe element c gets label r
            trial = [p for p in perms]; trial[s] = new
            v = value(C, trial)
            if v > cur + 1e-9: perms = trial; cur = v; improved = True
        if not improved: break
    return perms, cur


def fractional_arboricity(K, edges):
    best = 0.0
    for r in range(2, K + 1):
        for S in itertools.combinations(range(K), r):
            Ss = set(S); m = sum(1 for (a, b) in edges if a in Ss and b in Ss)
            best = max(best, m / (r - 1))
    return best


# copied from check_reduction.py (importing that file runs its 400-instance check)
def build(m, T):
    N = m + len(T)
    # vertex ids per part: elements 0..m-1, private m..N-1 (private of triple index i -> m+i)
    E12, E13, E23 = set(), set(), set()
    for i, (x, y, z) in enumerate(T):
        p = q = r = m + i
        E12.add((p, q)); E13.add((p, r)); E23.add((q, r))      # private triangle
        E12.add((x, q)); E13.add((x, r))                          # A_t = x q_t r_t  (x in V1, q in V2, r in V3); q-r already there
        E12.add((p, y)); E13.add((p, z)); E23.add((y, z))        # B_t = p_t y z
    return N, E12, E13, E23

def triangles(N, E12, E13, E23):
    return {(a, b, c) for a in range(N) for b in range(N) for c in range(N) if (a, b) in E12 and (a, c) in E13 and (b, c) in E23}

def has_3dm(m, T):
    def rec(i, ux, uy, uz):
        if i == m: return True
        for (x, y, z) in T:
            if x == min(set(range(m)) - ux) and y not in uy and z not in uz and rec(i + 1, ux | {x}, uy | {y}, uz | {z}): return True
        return False
    return rec(0, set(), set(), set())

def has_triangle_partition(N, tris):
    tris = list(tris)
    def rec(used1, used2, used3, left):
        if left == 0: return True
        a = min(set(range(N)) - used1)
        for (x, y, z) in tris:
            if x == a and y not in used2 and z not in used3 and rec(used1 | {x}, used2 | {y}, used3 | {z}, left - 1): return True
        return False
    return rec(set(), set(), set(), N)

def connected(cells):
    cells = set(cells); start = next(iter(cells)); seen = {start}; stack = [start]
    while stack:
        x, y = stack.pop()
        for q in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if q in cells and q not in seen: seen.add(q); stack.append(q)
    return len(seen) == len(cells)

