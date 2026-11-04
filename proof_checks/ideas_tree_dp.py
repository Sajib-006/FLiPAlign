"""D: joint placement and labelling. Section s picks a placement p_s from P candidates; edge tables C_e[p_s][p_u] (N x N)
depend on both placements. Claim: on a tree, max over (p, sigma) = max_p sum_e M_e(p_s, p_u), found by a tree DP
(max-sum message passing) over P states, edge weights = Hungarian value at those placements.
Check 1: brute force over all placements and all joint pairings equals the DP, and the DP's (p, chained sigma) reaches it.
Check 2 (B with placements): on P_K^2 split the edges into the path forest and the 'skip one' forest; solve each forest
exactly by the DP; best forest's solution, used on all edges, is >= OPT / 2 (arboricity 2), and the sum of the two
forest optima is >= OPT (a posteriori upper bound)."""
import itertools
import numpy as np
from ideas_common import all_perms, edge_value_matrix, hungarian

rng = np.random.default_rng(31)


def tables(edges, P, N):
    return {e: [[rng.integers(0, 10, (N, N)).astype(float) * (rng.random((N, N)) < 0.6) for _ in range(P)] for _ in range(P)]
            for e in edges}


def brute(C, K, P, N):
    Pm = all_perms(N)
    V = {e: [[edge_value_matrix(C[e][a][b], Pm) for b in range(P)] for a in range(P)] for e in C}
    best = -1
    for pl in itertools.product(range(P), repeat=K):
        for rest in itertools.product(range(len(Pm)), repeat=K - 1):
            idx = (0,) + rest
            v = sum(V[(s, u)][pl[s]][pl[u]][idx[s], idx[u]] for (s, u) in C)
            best = max(best, v)
    return best


def tree_dp(C, K, P, edges):
    """max-sum DP over placements on a forest; returns (value, placements). Weights M_e(a, b) by Hungarian."""
    Mw = {e: np.array([[hungarian(C[e][a][b])[0] for b in range(P)] for a in range(P)]) for e in edges}
    adj = {s: [] for s in range(K)}
    for (s, u) in edges: adj[s].append(u); adj[u].append(s)
    def w(s, u):                     # matrix indexed [p_s, p_u]
        return Mw[(s, u)] if (s, u) in Mw else Mw[(u, s)].T
    seen, total, place = set(), 0.0, [0] * K
    for root in range(K):
        if root in seen: continue
        order, parent, stack = [], {root: None}, [root]; seen.add(root)
        while stack:
            s = stack.pop(); order.append(s)
            for u in adj[s]:
                if u not in seen: seen.add(u); parent[u] = s; stack.append(u)
        msg, arg = {}, {}
        for s in reversed(order):                               # leaves first
            f = np.zeros(P)
            for u in adj[s]:
                if parent.get(u) == s: f += msg[u]
            if parent[s] is None:
                place[s] = int(f.argmax()); total += f.max(); fs = f
            else:
                M = w(parent[s], s) + f[None, :]                   # [p_parent, p_s]
                msg[s] = M.max(1); arg[s] = M.argmax(1)
            if parent[s] is None: break
        for s in order[1:]: place[s] = int(arg[s][place[parent[s]]])
    return total, place


def realise(C, K, N, edges, place):
    """chained Hungarian at the chosen placements; value on ALL edges of C."""
    adj = {s: [] for s in range(K)}
    for (s, u) in edges: adj[s].append(u); adj[u].append(s)
    perms = [None] * K
    for r in range(K):
        if perms[r] is not None: continue
        perms[r] = np.arange(N); st = [r]
        while st:
            s = st.pop()
            for u in adj[s]:
                if perms[u] is None:
                    T = C[(s, u)][place[s]][place[u]] if (s, u) in C else C[(u, s)][place[u]][place[s]].T
                    perms[u] = hungarian(T)[1][perms[s]]; st.append(u)
    return sum(C[(s, u)][place[s]][place[u]][perms[s], perms[u]].sum() for (s, u) in C)


n1 = 0
for K, P, N, reps in ((3, 3, 3, 20), (4, 2, 3, 20), (4, 3, 3, 10), (5, 2, 3, 4)):
    for t in range(reps):
        edges = [(int(rng.integers(0, s)), s) for s in range(1, K)]       # random tree
        C = tables(edges, P, N)
        b = brute(C, K, P, N); d, pl = tree_dp(C, K, P, edges); r = realise(C, K, N, edges, pl)
        assert abs(b - d) < 1e-9 and abs(r - d) < 1e-9, (K, P, N, b, d, r)
        n1 += 1
print("tree instances with placements", n1, ": brute force == tree DP == realised value. PASSED")

worst, n2 = 1.0, 0
for K, P, N, reps in ((4, 2, 3, 15), (5, 2, 3, 4)):
    for t in range(reps):
        edges = [(i, j) for i in range(K) for j in range(i + 1, K) if j - i <= 2]
        C = tables(edges, P, N)
        opt = brute(C, K, P, N)
        F1 = [e for e in edges if e[1] - e[0] == 1]; F2 = [e for e in edges if e[1] - e[0] == 2]
        best, ub = 0.0, 0.0
        for F in (F1, F2):
            CF = {e: C[e] for e in F}
            v, pl = tree_dp(CF, K, P, F); ub += v
            best = max(best, realise(C, K, N, F, pl))
        assert ub >= opt - 1e-9 and best >= opt / 2 - 1e-9, (opt, ub, best)
        worst = min(worst, best / opt); n2 += 1
print(f"P_K^2 with placements: {n2} instances, best-forest value >= OPT/2 and forest-sum >= OPT on all; worst ratio {worst:.3f}. PASSED")
