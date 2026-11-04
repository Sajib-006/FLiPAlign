"""Brute-force checks for odd_cap.md: K_{c,c}-factor of balanced bipartite graphs is NP-complete for every c >= 3,
by a reduction from exact cover by c-sets (XcC). Run with ~/miniconda3/envs/base-3.10/bin/python check_odd_cap.py.

Reduction (see odd_cap.md). Universe U, family S_1..S_m of c-sets.
  * each element u gives u^A in A and u^B in B, with the edge u^A u^B;
  * each set j gets private P[j][i] (c-1 vertices in A) and R[j][i] (c-1 vertices in B) for i = 0..c-1;
    edges: P[j][i] x R[j][i] complete; u_i^A x R[j][i]; u_i^B x P[j][i];
    group edges: P[j][i][g] x R[j][i'][g] for all i, i' and g = 0..c-2.
Tests
  O1  random XcC instances (c = 3, 4, 5): exact cover exists  <=>  K_{c,c}-factor exists (exact search).
  O2  every K_{c,c} subgraph of the built graph is an element block E_{j,i} or lies inside one gadget V_j.
  O3  yes-instances: the labelling built from the cover has value cN under w01 and N/c classes;
      exact OPT by set-partitioning MILP over connected classes (small instances): OPT = cN iff yes.
  O4  Lemma split-value(c) on random small balanced bipartite graphs: exhaustive OPT <= cN, and
      OPT = cN  <=>  K_{c,c}-factor (with G = infinity and G = N/c).
"""
import itertools, random, sys, time, os

import numpy as np
import scipy.sparse as sp
from scipy.optimize import milp, LinearConstraint, Bounds

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_split import kcc_factor, split_opt  # noqa: E402


# ------------------------------------------------------------------ instances
def exact_cover(U, sets):
    """Brute force: is there a sub-family of pairwise disjoint sets covering U?"""
    U = frozenset(U)
    sets = [frozenset(s) for s in sets]

    def rec(left, avail):
        if not left:
            return True
        u = min(left)
        for k in avail:
            s = sets[k]
            if u in s and s <= left:
                if rec(left - s, [a for a in avail if a != k]):
                    return True
        return False

    return rec(U, list(range(len(sets))))


def find_cover(U, sets):
    U = frozenset(U)
    sets = [frozenset(s) for s in sets]

    def rec(left, used):
        if not left:
            return used
        u = min(left)
        for k, s in enumerate(sets):
            if k not in used and u in s and s <= left:
                r = rec(left - s, used + [k])
                if r is not None:
                    return r
        return None

    return rec(U, [])


def build(c, U, sets):
    """Returns L, R, adj (dict of sets), plus labels describing every vertex."""
    L, R = [], []
    adj = {}

    def add(v, side):
        (L if side == 0 else R).append(v)
        adj[v] = set()

    def edge(a, b):
        adj[a].add(b)
        adj[b].add(a)

    for u in U:
        add(("uA", u), 0)
        add(("uB", u), 1)
        edge(("uA", u), ("uB", u))
    for j, S in enumerate(sets):
        S = list(S)
        assert len(S) == c and len(set(S)) == c
        for i in range(c):
            for g in range(c - 1):
                add(("P", j, i, g), 0)
                add(("R", j, i, g), 1)
        for i, u in enumerate(S):
            for g in range(c - 1):
                for g2 in range(c - 1):
                    edge(("P", j, i, g), ("R", j, i, g2))
                edge(("uA", u), ("R", j, i, g))
                edge(("uB", u), ("P", j, i, g))
        for g in range(c - 1):
            for i in range(c):
                for i2 in range(c):
                    edge(("P", j, i, g), ("R", j, i2, g))
    return L, R, adj


def all_kcc(L, R, adj, c):
    """All K_{c,c} subgraphs, as (frozenset A-side, frozenset B-side)."""
    idx = {v: k for k, v in enumerate(L)}
    out = set()
    for v in L:
        for Y in itertools.combinations(sorted(adj[v], key=repr), c):
            common = set.intersection(*[adj[y] for y in Y])
            cand = sorted([a for a in common if idx[a] > idx[v]], key=lambda a: idx[a])
            for rest in itertools.combinations(cand, c - 1):
                out.add((frozenset((v,) + rest), frozenset(Y)))
    return out


def classify(block, sets, c):
    X, Y = block
    elems = [v for v in X | Y if v[0] in ("uA", "uB")]
    if not elems:
        js = {v[1] for v in X | Y}
        return "inside" if len(js) == 1 else "BAD"
    # must be E_{j,i} = {u^A} + P[j][i] ; {u^B} + R[j][i]
    us = {v[1] for v in elems}
    if len(us) != 1 or len(elems) != 2:
        return "BAD"
    u = us.pop()
    priv = [v for v in X | Y if v[0] in ("P", "R")]
    ji = {(v[1], v[2]) for v in priv}
    if len(ji) != 1:
        return "BAD"
    j, i = ji.pop()
    if list(sets[j])[i] != u:
        return "BAD"
    if len(priv) != 2 * (c - 1):
        return "BAD"
    return "E"


def labelling_from_cover(c, U, sets, cover):
    blocks = []
    for j, S in enumerate(sets):
        S = list(S)
        if j in cover:
            for i, u in enumerate(S):
                blocks.append(({("uA", u)} | {("P", j, i, g) for g in range(c - 1)},
                               {("uB", u)} | {("R", j, i, g) for g in range(c - 1)}))
        else:
            for g in range(c - 1):
                blocks.append(({("P", j, i, g) for i in range(c)}, {("R", j, i, g) for i in range(c)}))
    return blocks


def value01(blocks, adj):
    return sum(1 for X, Y in blocks for a in X for b in Y if b in adj[a])


def random_instance(c, q, m, rng, plant):
    U = list(range(c * q))
    sets = []
    if plant:
        perm = U[:]
        rng.shuffle(perm)
        sets += [tuple(sorted(perm[k * c:(k + 1) * c])) for k in range(q)]
    tries = 0
    while len(sets) < m:
        s = tuple(sorted(rng.sample(U, c)))
        tries += 1
        if s not in sets or tries > 200:  # repeats allowed only when few distinct c-sets exist
            sets.append(s)
    rng.shuffle(sets)
    return U, sets


# ------------------------------------------------------------------ exact OPT by MILP over connected classes
def connected_classes(L, R, adj, c):
    """All vertex sets with <= c vertices per side that induce a connected subgraph with at least one edge.
    ESU enumeration (Wernicke 2006): each connected set is produced once, from its smallest vertex."""
    verts = L + R
    vid = {v: k for k, v in enumerate(verts)}
    nL = len(L)
    nb = [set(vid[u] for u in adj[v]) for v in verts]
    out = []

    def extend(sub, cnt, ext, nsub, v):
        ext = set(ext)
        while ext:
            w = ext.pop()
            sd = 0 if w < nL else 1
            if cnt[sd] >= c:
                continue
            cnt2 = list(cnt)
            cnt2[sd] += 1
            sub2 = sub + [w]
            out.append(tuple(sub2))
            excl = {u for u in nb[w] if u > v and u not in nsub and u not in sub}
            nsub2 = nsub | nb[w]
            extend(sub2, cnt2, ext | excl, nsub2, v)

    for v in range(len(verts)):
        cnt = [0, 0]
        cnt[0 if v < nL else 1] = 1
        extend([v], cnt, {u for u in nb[v] if u > v}, set(nb[v]) | {v}, v)
    return [frozenset(verts[i] for i in S) for S in out]


def opt_milp_conn(L, R, adj, c, G=None, time_limit=1800):
    verts = L + R
    vid = {v: k for k, v in enumerate(verts)}
    cls = connected_classes(L, R, adj, c)
    Lset = set(L)
    vals = np.array([sum(1 for a in S if a in Lset for b in adj[a] if b in S) for S in cls], dtype=float)
    # singletons (value 0) so every vertex can be alone
    rows, cols = [], []
    for k, S in enumerate(cls):
        for v in S:
            rows.append(vid[v]); cols.append(k)
    ncl = len(cls)
    for v in verts:
        rows.append(vid[v]); cols.append(ncl + vid[v])
    ncol = ncl + len(verts)
    allv = np.concatenate([vals, np.zeros(len(verts))])
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(verts), ncol))
    cons = [LinearConstraint(A, np.ones(len(verts)), np.ones(len(verts)))]
    if G is not None:
        cons.append(LinearConstraint(sp.csr_matrix(np.ones((1, ncol))), 0, G))
    res = milp(-allv, constraints=cons, integrality=np.ones(ncol), bounds=Bounds(0, 1),
               options={"mip_rel_gap": 0, "time_limit": time_limit, "disp": False})
    if res.status != 0:
        raise RuntimeError(f"milp status {res.status}: {res.message}")
    return int(round(-res.fun)), ncl


# ------------------------------------------------------------------ tests
def O1_O2_O3(report, c, q_list, m_list, trials, rng, milp_max_N=0, struct=True, milp_per_cell=3):
    yes = no = 0
    nblocks = {"E": 0, "inside": 0}
    nmilp = 0
    t0 = time.time()
    for q in q_list:
        for m in m_list:
            for t in range(trials):
                U, sets = random_instance(c, q, m, rng, plant=(t % 2 == 0 and m >= q))
                ec = exact_cover(U, sets)
                L, R, adj = build(c, U, sets)
                N = len(L)
                assert len(R) == N == c * q + c * (c - 1) * m
                fac = kcc_factor(L, R, adj, c)
                if ec != fac:
                    report(f"  FAIL O1 c={c} q={q} m={m} sets={sets}: cover={ec} factor={fac}")
                    return False
                if struct:
                    for b in all_kcc(L, R, adj, c):
                        k = classify(b, sets, c)
                        if k == "BAD":
                            report(f"  FAIL O2 c={c} sets={sets} block={b}")
                            return False
                        nblocks[k] += 1
                if ec:
                    yes += 1
                    cover = find_cover(U, sets)
                    blocks = labelling_from_cover(c, U, sets, cover)
                    covered = [v for X, Y in blocks for v in X | Y]
                    assert len(covered) == 2 * N and set(covered) == set(L) | set(R)
                    assert all(len(X) == c and len(Y) == c for X, Y in blocks)
                    assert len(blocks) == N // c
                    if value01(blocks, adj) != c * N:
                        report(f"  FAIL O3 value {value01(blocks, adj)} != cN={c * N}")
                        return False
                else:
                    no += 1
                if N <= milp_max_N and t < milp_per_cell:
                    opt, ncl = opt_milp_conn(L, R, adj, c)
                    nmilp += 1
                    if (opt == c * N) != ec or opt > c * N:
                        report(f"  FAIL O3 MILP c={c} sets={sets}: OPT={opt} cN={c * N} cover={ec}")
                        return False
                    report(f"    MILP c={c} q={q} m={m} N={N} classes={ncl} OPT={opt} cN={c * N} cover={ec}")
    report(f"  c={c}: {yes} yes and {no} no instances agree (cover <=> factor); "
           f"K_cc subgraphs seen: {nblocks['E']} element blocks, {nblocks['inside']} inside one gadget, 0 other; "
           f"{nmilp} exact MILP OPT checks; {time.time() - t0:.1f}s")
    return True


def random_bip(N, p, rng):
    L = [("a", i) for i in range(N)]
    R = [("b", j) for j in range(N)]
    adj = {v: set() for v in L + R}
    for a in L:
        for b in R:
            if rng.random() < p:
                adj[a].add(b); adj[b].add(a)
    return L, R, adj


def planted_bip(N, c, p, rng):
    L, R, adj = random_bip(N, p, rng)
    perm_a = L[:]; perm_b = R[:]
    rng.shuffle(perm_a); rng.shuffle(perm_b)
    for k in range(N // c):
        for a in perm_a[k * c:(k + 1) * c]:
            for b in perm_b[k * c:(k + 1) * c]:
                adj[a].add(b); adj[b].add(a)
    return L, R, adj


def O4(report, rng):
    t0 = time.time()
    tot = eq = 0
    for c in (2, 3, 4):
        for N in range(c, 7):
            if 2 * N > 12:
                continue
            for t in range(60 if N <= 5 else 25):
                p = rng.choice([0.3, 0.5, 0.7, 0.85])
                if t % 3 == 0 and N % c == 0:
                    L, R, adj = planted_bip(N, c, p, rng)
                else:
                    L, R, adj = random_bip(N, p, rng)
                sec = [0] * N + [1] * N
                W = {(i, N + j): 1.0 for i, a in enumerate(L) for j, b in enumerate(R) if b in adj[a]}
                fac = kcc_factor(L, R, adj, c)
                for G in (10 ** 6, max(1, N // c)):
                    optf = split_opt(sec, W, c, G)
                    tot += 1
                    if optf == float("-inf"):  # no split labelling with <= G classes
                        if fac and N % c == 0 and G >= N // c:
                            report(f"  FAIL O4 infeasible but factor exists c={c} N={N} G={G}")
                            return False
                        continue
                    opt = int(round(optf))
                    ok_bound = opt <= c * N
                    want = fac and (N % c == 0) and G >= N // c
                    if not ok_bound or (opt == c * N) != want:
                        report(f"  FAIL O4 c={c} N={N} G={G} OPT={opt} cN={c * N} factor={fac}")
                        return False
                    eq += (opt == c * N)
    report(f"  O4: {tot} (graph, G) cases, c in 2,3,4, N <= 6: OPT <= cN always, OPT = cN exactly when a K_cc-factor "
           f"exists ({eq} equality cases); {time.time() - t0:.1f}s")
    return True


def main():
    logf = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "check_odd_cap.log"), "w")

    def report(s):
        print(s, flush=True)
        logf.write(s + "\n"); logf.flush()

    rng = random.Random(20261104)
    ok = True
    report("O4: Lemma split-value(c) equality case on random small graphs (exhaustive DP)")
    ok &= O4(report, rng)
    report("O1-O3: reduction XcC -> K_{c,c}-factor")
    ok &= O1_O2_O3(report, 3, [1, 2, 3], [2, 3, 4, 5], 40, rng, milp_max_N=24, milp_per_cell=4)
    ok &= O1_O2_O3(report, 3, [4], [5, 6, 8], 15, rng)
    ok &= O1_O2_O3(report, 4, [1, 2, 3], [2, 3, 4], 20, rng)
    ok &= O1_O2_O3(report, 5, [1, 2], [2, 3, 4], 12, rng)
    report("ALL PASS" if ok else "SOME CHECK FAILED")
    logf.close()


if __name__ == "__main__":
    main()
