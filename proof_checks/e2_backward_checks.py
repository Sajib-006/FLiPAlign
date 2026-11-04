"""Independent backward-direction checks (review 12, E2). Written from the LaTeX, not from the other checkers.

B1  Lemma split-value (b) table: max ones in an s x t 0/1 matrix, no two rows share 3 columns.
B2  Lemma split-value (b) on random K23/K32-free bipartite graphs, c unbounded, +-1 table, exact OPT by subset DP.
B3  Split reduction H: no K23/K32, and C4-factor <=> 3DM perfect matching, on random 3DM instances
    (including degenerate ones: elements in no triple, triples sharing two coordinates).
B4  Split reduction exact OPT for c=2,3,4,unbounded with +-1 table on the smallest instance (N=6).
B5  Cycle theorem (a), abstract tables, L=4,5: max Phi >= tau  <=>  3DM yes. Brute force over all pairings.
B6  Forest cover tightness for general d: a family that reaches 1/gamma for d>=3 (the paper proves only d=2).

Usage: python e2_backward_checks.py [B1 B2 ...]   (no argument = all)
"""
import itertools, random, sys
from functools import lru_cache

random.seed(12)
RUN = set(sys.argv[1:]) or {'B1', 'B2', 'B3', 'B4', 'B5', 'B6'}


def say(x):
    print(x, flush=True)


# ---------------- helpers
def max_rows(s, t):
    cols = list(range(t))
    rowsets = [frozenset(c) for r in range(t, -1, -1) for c in itertools.combinations(cols, r)]
    best = 0
    def rec(chosen, start, total):
        nonlocal best
        if len(chosen) == s:
            best = max(best, total); return
        if total + (s - len(chosen)) * (len(rowsets[start]) if start < len(rowsets) else 0) <= best:
            return
        for i in range(start, len(rowsets)):
            r = rowsets[i]
            if total + len(r) * (s - len(chosen)) <= best:
                return  # rowsets sorted by size, descending
            if all(len(r & c) <= 2 for c in chosen):
                rec(chosen + [r], i, total + len(r))
    rec([], 0, 0)
    return best


def split_opt(NA, NB, w, c):
    n = NA + NB
    cv = {}
    for mask in range(1, 1 << n):
        A = [i for i in range(NA) if mask >> i & 1]
        B = [j for j in range(NB) if mask >> (NA + j) & 1]
        if len(A) <= c and len(B) <= c:
            cv[mask] = sum(w[i][j] for i in A for j in B)
    best = [None] * (1 << n)
    best[0] = 0
    for mask in range(1, 1 << n):
        low = mask & -mask
        rest = mask ^ low
        sub = rest
        b = None
        while True:
            cls = sub | low
            v = cv.get(cls)
            if v is not None:
                r = best[mask ^ cls]
                if r is not None and (b is None or v + r > b):
                    b = v + r
            if sub == 0: break
            sub = (sub - 1) & rest
        best[mask] = b
    return best[(1 << n) - 1]


def has_k23(NA, NB, adj):
    for i, j in itertools.combinations(range(NA), 2):
        if len(adj[i] & adj[j]) >= 3: return True
    badj = [set(i for i in range(NA) if b in adj[i]) for b in range(NB)]
    for a, b in itertools.combinations(range(NB), 2):
        if len(badj[a] & badj[b]) >= 3: return True
    return False


def c4_factor(NA, NB, adj):
    @lru_cache(None)
    def rec(Am, Bm):
        if Am == 0 and Bm == 0: return True
        if Am == 0 or Bm == 0: return False
        a = (Am & -Am).bit_length() - 1
        for a2 in range(a + 1, NA):
            if not Am >> a2 & 1: continue
            common = [b for b in adj[a] & adj[a2] if Bm >> b & 1]
            for b1, b2 in itertools.combinations(common, 2):
                if rec(Am & ~(1 << a) & ~(1 << a2), Bm & ~(1 << b1) & ~(1 << b2)): return True
        return False
    return rec((1 << NA) - 1, (1 << NB) - 1)


def build_H(X, Y, Z, T):
    A, B = [], []
    def va(n): A.append(n); return len(A) - 1
    def vb(n): B.append(n); return len(B) - 1
    xi = {x: va(('x', x)) for x in X}
    yi = {y: vb(('y', y)) for y in Y}
    zA = {z: va(('zA', z)) for z in Z}
    zB = {z: vb(('zB', z)) for z in Z}
    E = set((zA[z], zB[z]) for z in Z)
    for k, (x, y, z) in enumerate(T):
        p, a, a2, u = (va(('p', k)), va(('a', k)), va(('a2', k)), va(('u', k)))
        r, r2, o, v = (vb(('r', k)), vb(('r2', k)), vb(('o', k)), vb(('v', k)))
        for e in [(p, r), (p, r2), (p, o), (a, o), (a, r), (a2, o), (a2, r2), (a2, v), (u, v), (u, r2),
                  (xi[x], r), (xi[x], r2), (a, yi[y]), (a2, yi[y]), (zA[z], v), (u, zB[z])]:
            E.add(e)
    adj = [set() for _ in A]
    for i, j in E: adj[i].add(j)
    return len(A), len(B), adj


def dm(q, T):
    for M in itertools.combinations(T, q):
        if all(len(set(t[i] for t in M)) == q for i in range(3)): return True
    return False


def triangle_graph(X, Y, Z, T):
    V1 = [('x', x) for x in X] + [('p', k) for k in range(len(T))]
    V2 = [('y', y) for y in Y] + [('q', k) for k in range(len(T))]
    V3 = [('z', z) for z in Z] + [('r', k) for k in range(len(T))]
    E12, E13, E23 = set(), set(), set()
    for k, (x, y, z) in enumerate(T):
        E12 |= {(('p', k), ('q', k)), (('x', x), ('q', k)), (('p', k), ('y', y))}
        E13 |= {(('p', k), ('r', k)), (('x', x), ('r', k)), (('p', k), ('z', z))}
        E23 |= {(('q', k), ('r', k)), (('y', y), ('z', z))}
    i1 = {v: i for i, v in enumerate(V1)}; i2 = {v: i for i, v in enumerate(V2)}; i3 = {v: i for i, v in enumerate(V3)}
    N = len(V1)
    M12 = [[0] * N for _ in range(N)]; M13 = [[0] * N for _ in range(N)]; M23 = [[0] * N for _ in range(N)]
    for a, b in E12: M12[i1[a]][i2[b]] = 1
    for a, b in E13: M13[i1[a]][i3[b]] = 1
    for a, b in E23: M23[i2[a]][i3[b]] = 1
    return N, M12, M13, M23


def cycle_max(L, N, M12, M13, M23, Wp):
    # sections v1..vL; edges v1v2 (M12), v2v3 (M23), vLv1 (M13, rows = v1), v_i v_{i+1} (Wp*I), i=3..L-1.
    # sigma_1 fixed to the identity by fact (i); s[i][g] = label of section i+1 at universe element g.
    perms = list(itertools.permutations(range(N)))
    best = -1
    for s2 in perms:
        v12 = sum(M12[g][s2[g]] for g in range(N))
        for rest in itertools.product(perms, repeat=L - 2):
            s = [tuple(range(N)), s2] + list(rest)
            val = v12 + sum(M23[s[1][g]][s[2][g]] for g in range(N)) + sum(M13[g][s[L - 1][g]] for g in range(N))
            for i in range(2, L - 1):
                val += Wp * sum(1 for g in range(N) if s[i][g] == s[i + 1][g])
            if val > best: best = val
    return best


# ---------------- B1
if 'B1' in RUN:
    pairs = [(3, 3), (3, 4), (3, 5), (4, 4), (4, 5), (4, 6), (5, 5), (5, 6), (6, 6)]
    claimed = [7, 9, 10, 12, 13, 15, 16, 18, 21]
    b1 = []
    for (s, t), cl in zip(pairs, claimed):
        e = max_rows(s, t)
        b1.append(f"({s},{t}) e={e} claimed={cl} {'OK' if e == cl else 'MISMATCH'} strict={2*e < s*t+s+t}")
    say("B1 " + "; ".join(b1))

# ---------------- B2
if 'B2' in RUN:
    for N, want in ((4, 150), (5, 120)):
        cnt = bad = nfac = tries = 0
        while cnt < want and tries < 50000:
            tries += 1
            p = random.choice([0.3, 0.45, 0.6, 0.75])
            adj = [set(j for j in range(N) if random.random() < p) for _ in range(N)]
            if has_k23(N, N, adj): continue
            w = [[1 if j in adj[i] else -1 for j in range(N)] for i in range(N)]
            opt = split_opt(N, N, w, 99)
            fac = c4_factor(N, N, adj)
            nfac += fac; cnt += 1
            if opt > 2 * N or ((opt == 2 * N) != fac):
                bad += 1; say(f"B2 FAIL {N} {adj} {opt} {fac}")
        say(f"B2 N={N}: {cnt} random K23/K32-free graphs ({nfac} with a C4-factor), c unbounded, failures={bad}")
    # the 4+4 block itself: cube graph Q3 = K44 minus a perfect matching
    adj = [set(j for j in range(4) if j != i) for i in range(4)]
    w = [[1 if j in adj[i] else -1 for j in range(4)] for i in range(4)]
    say(f"B2 K44-minus-PM: OPT c=inf {split_opt(4,4,w,99)}, c=2 {split_opt(4,4,w,2)}, 2N=8, C4-factor={c4_factor(4,4,adj)}")

# ---------------- B3
if 'B3' in RUN:
    bad = n = yes = 0
    for trial in range(400):
        q = random.choice([1, 2, 3, 3, 4])
        X = Y = Z = list(range(q))
        allt = list(itertools.product(X, Y, Z))
        m = random.randint(1, min(len(allt), 2 * q + 2))
        T = random.sample(allt, m)
        NA, NB, adj = build_H(X, Y, Z, T)
        k23 = has_k23(NA, NB, adj)
        f = c4_factor(NA, NB, adj)
        d = dm(q, T)
        n += 1; yes += d
        if k23 or f != d or NA != NB or NA != 4 * m + 2 * q:
            bad += 1; say(f"B3 FAIL {q} {T} {k23} {f} {d}")
    say(f"B3 {n} random 3DM instances ({yes} yes, q=1..4, elements in no triple allowed): K23-free and C4-factor<=>3DM, failures={bad}")

# ---------------- B4
if 'B4' in RUN:
    NA, NB, adj = build_H([0], [0], [0], [(0, 0, 0)])
    w = [[1 if j in adj[i] else -1 for j in range(NB)] for i in range(NA)]
    w01 = [[1 if j in adj[i] else 0 for j in range(NB)] for i in range(NA)]
    r = [f"+-1 c={c}: OPT={split_opt(NA, NB, w, c)}" for c in (2, 3, 4, 99)]
    r.append(f"0/1 c=2: OPT={split_opt(NA, NB, w01, 2)}")
    say(f"B4 q=1,m=1 yes-instance, N={NA}, 2N={2*NA}: " + "; ".join(r))
    # a no-instance with N small: q=1 is always yes with one triple; use a hand-made H minus one edge
    adj2 = [set(a) for a in adj]; adj2[0].discard(next(iter(adj2[0])))  # break x's gadget
    w2 = [[1 if j in adj2[i] else -1 for j in range(NB)] for i in range(NA)]
    say(f"B4 same graph with one edge at x removed (no C4-factor={not c4_factor(NA,NB,adj2)}): "
        + "; ".join(f"c={c}: OPT={split_opt(NA, NB, w2, c)}" for c in (2, 3, 99)))

# ---------------- B5
if 'B5' in RUN:
    bad = n = yes = 0
    cases = [(1, [(0, 0, 0)]), (2, [(0, 0, 0), (1, 1, 1)]), (2, [(0, 0, 0), (0, 1, 1)]), (2, [(0, 0, 1), (1, 1, 0)]),
             (2, [(0, 1, 0), (1, 0, 1)]), (2, [(0, 0, 0), (1, 1, 0)]), (2, [(0, 0, 0), (1, 0, 1)])]
    for L in (4, 5):
        for q, T in cases:
            X = Y = Z = list(range(q))
            N, M12, M13, M23 = triangle_graph(X, Y, Z, T)
            for Wp in (1, 3 * N + 1):
                tau = (L - 3) * N * Wp + 3 * N
                mx = cycle_max(L, N, M12, M13, M23, Wp)
                d = dm(q, T)
                n += 1; yes += d
                if (mx >= tau) != d:
                    bad += 1; say(f"B5 FAIL L={L} T={T} W'={Wp} max={mx} tau={tau} 3DM={d}")
    say(f"B5 cycle (a), abstract tables, L=4,5, W'=1 and 3N+1: {n} cases ({yes} yes), decision failures={bad}")

# ---------------- B6
if 'B6' in RUN:
    def pd_edges(K, d):
        return [(s, u) for s in range(K) for u in range(s + 1, min(K, s + d + 1))]

    def family(K, d, N, delta=1e-3):
        tab = {}
        for (s, u) in pd_edges(K, d):
            j = u - s
            if j == 1:   # path edge: best map is the shift by 1, unique, slightly heavier
                tab[(s, u)] = [[(1 + delta) * ((l != k) + delta * (l == (k + 1) % N)) for l in range(N)] for k in range(N)]
            else:        # zero exactly on the relative shift j that the path walk produces
                tab[(s, u)] = [[1.0 * (l != (k + j) % N) for l in range(N)] for k in range(N)]
        return tab

    def phi(tab, lab):
        return sum(T_[lab[s][g]][lab[u][g]] for (s, u), T_ in tab.items() for g in range(len(lab[0])))

    def assign_val(M):
        N = len(M)
        return max(sum(M[k][p[k]] for k in range(N)) for p in itertools.permutations(range(N)))

    out = []
    for K, d, N in [(4, 2, 5), (5, 3, 7), (6, 3, 7), (7, 4, 9), (9, 3, 7)]:
        tab = family(K, d, N)
        heavy = None
        if N <= 7:
            Me = {e: assign_val(T_) for e, T_ in tab.items()}
            heavy = all((e[1] - e[0] == 1) == (Me[e] > N + 1e-9) for e in tab)
        alg = phi(tab, [[(g + s) % N for g in range(N)] for s in range(K)])
        good = phi(tab, [[(g + 2 * s) % N for g in range(N)] for s in range(K)])
        gamma = d - d * (d - 1) / (2 * (K - 1))
        out.append(f"K={K},d={d},N={N}: alg/explicit={alg/good:.4f}, 1/gamma={1/gamma:.4f}, path edges strictly heaviest={heavy}")
    say("B6 " + "; ".join(out))
