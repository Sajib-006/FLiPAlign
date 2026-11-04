"""Brute-force checks for the split labelling hardness results (review 11, 2026-10-21).

What is checked (see reviews/review11_split_theory.md and paper/split_theory.tex):
  T1  Value lemma, c = 2: for every bipartite graph H with N labels per side (all graphs for N = 2, 3; random for N = 4),
      with weights 0/1 and with weights +1/-1 (edge / non-edge), and for G = N, G = ceil(N/2) and G unbounded:
      OPT <= 2N, and OPT = 2N exactly when H has a C4-factor (partition into vertex-disjoint 4-cycles).
  T2  Value lemma, c unbounded, weights +1/-1, on graphs with no K_{2,3} and no K_{3,2}: same statement.
      Also the finite table used in the proof: for 3 <= s <= t <= 6, the largest number of ones in an s x t 0/1 matrix
      in which no two rows share 3 columns satisfies 2e < st + s + t.
  T3  The reduction from 3-dimensional matching (3DM): all instances with q = 2 (every set of triples), random instances
      with q = 3 and q = 4. Checks: H is bipartite and balanced, has no K_{2,3} or K_{3,2}, and
      (3DM has a perfect matching) <=> (H has a C4-factor). On small instances it also computes the exact split optimum
      (c = 2, G = N, 0/1 and +1/-1 weights; c unbounded, +1/-1) and checks it equals 2N exactly for yes-instances.
      For c = 4 it checks the blow-up (each vertex -> 2 copies) has a K_{4,4}-factor exactly for yes-instances.
  T4  Stack bound: random stacks (K = 3 sections, N = 3, signed integer weights, all three pairs compared, c = 2,
      G = N and G unbounded): stack OPT <= sum of pair optima. Also pair OPT (G unbounded) <= c * max weight matching.
Run:  ~/miniconda3/envs/base-3.10/bin/python hardness/check_split.py
"""
import itertools, math, random, sys, time
from functools import lru_cache

import numpy as np
from scipy.optimize import linear_sum_assignment

random.seed(11)
INF = float("inf")


# ---------------------------------------------------------------- exact split optimum on a stack (K sections)
def split_opt(sec, W, c, G):
    """sec[v] = section of vertex v; W[(u, v)] = weight for u < v in different compared sections (missing = 0).
    Partition all vertices into at most G classes, each with <= c vertices of every section. Returns the max value."""
    n = len(sec)
    w = [[0.0] * n for _ in range(n)]
    for (u, v), x in W.items():
        w[u][v] = w[v][u] = x
    full = (1 << n) - 1
    cap = n if c is None else c

    @lru_cache(maxsize=None)
    def f(mask, g):
        if mask == 0:
            return 0.0
        if g == 0:
            return -INF
        v = (mask & -mask).bit_length() - 1
        rest = [u for u in range(n) if (mask >> u) & 1 and u != v]
        best = -INF
        if c is None:
            # enumerate all subsets of rest (small n or uncapped)
            m = len(rest)
            for bits in range(1 << m):
                cls = [v] + [rest[i] for i in range(m) if (bits >> i) & 1]
                cnt = {}
                ok = True
                for x in cls:
                    cnt[sec[x]] = cnt.get(sec[x], 0) + 1
                    if cnt[sec[x]] > cap:
                        ok = False
                        break
                if not ok:
                    continue
                val = sum(w[a][b] for a, b in itertools.combinations(cls, 2))
                cm = 0
                for x in cls:
                    cm |= 1 << x
                r = f(mask & ~cm, g - 1)
                if r > -INF and val + r > best:
                    best = val + r
            return best
        # capped enumeration: choose up to cap (cap-1 for v's section) from each section
        bysec = {}
        for u in rest:
            bysec.setdefault(sec[u], []).append(u)
        options = []
        for s, us in bysec.items():
            k = cap - 1 if s == sec[v] else cap
            opts = []
            for r_ in range(0, min(k, len(us)) + 1):
                opts.extend(itertools.combinations(us, r_))
            options.append(opts)
        if sec[v] not in bysec:
            pass
        for choice in itertools.product(*options):
            cls = [v] + [x for part in choice for x in part]
            val = sum(w[a][b] for a, b in itertools.combinations(cls, 2))
            cm = 0
            for x in cls:
                cm |= 1 << x
            r = f(mask & ~cm, g - 1)
            if r > -INF and val + r > best:
                best = val + r
        return best

    res = f(full, min(G, n))
    f.cache_clear()
    return res


def pair_weights(adjA, NA, NB, mode):
    """vertices 0..NA-1 = side A, NA..NA+NB-1 = side B; adjA[i] = set of B indices. mode '01' or 'pm'."""
    W = {}
    for i in range(NA):
        for j in range(NB):
            e = j in adjA[i]
            if mode == "01":
                if e:
                    W[(i, NA + j)] = 1.0
            else:
                W[(i, NA + j)] = 1.0 if e else -1.0
    return W


# ---------------------------------------------------------------- K_{c,c}-factor (c = 2 is the C4-factor)
def kcc_factor(L, R, adj, c=2):
    """adj: dict vertex -> set of neighbours (bipartite, sides L and R). True if V can be split into K_{c,c}'s."""
    if len(L) != len(R) or len(L) % c:
        return False
    side = {v: 0 for v in L}
    side.update({v: 1 for v in R})
    free = set(L) | set(R)

    def rec():
        if not free:
            return True
        # vertex with fewest free neighbours
        v = min(free, key=lambda x: len(adj[x] & free))
        nb = adj[v] & free
        if len(nb) < c:
            return False
        same_side_pool = [u for u in free if side[u] == side[v] and u != v]
        for other in itertools.combinations(sorted(nb), c):
            common = set.intersection(*[adj[o] & free for o in other]) - {v}
            if len(common) < c - 1:
                continue
            for mates in itertools.combinations(sorted(common), c - 1):
                block = set(other) | set(mates) | {v}
                free.difference_update(block)
                if rec():
                    return True
                free.update(block)
        return False

    return rec()


def c4_factor_bip(NA, NB, adjA):
    L = [("a", i) for i in range(NA)]
    R = [("b", j) for j in range(NB)]
    adj = {v: set() for v in L + R}
    for i in range(NA):
        for j in adjA[i]:
            adj[("a", i)].add(("b", j))
            adj[("b", j)].add(("a", i))
    return kcc_factor(L, R, adj, 2)


def has_k23(adj, L, R):
    for side in (L, R):
        for u, v in itertools.combinations(side, 2):
            if len(adj[u] & adj[v]) >= 3:
                return True
    return False


# ---------------------------------------------------------------- the reduction from 3DM
def build_H(q, triples):
    """Elements x_i (L), y_i (R), z_i -> (zL_i in L, zR_i in R) with edge zL-zR.
    Triple t = (x, y, z): L gadget vertices lx, ly1, ly2, a ; R gadget vertices rx1, rx2, ry, b.
    Gadget edges: lx-rx1, lx-rx2, lx-ry, ly1-ry, ly1-rx1, ly2-ry, ly2-rx2, ly2-b, a-b, a-rx2.
    Element edges: x-rx1, x-rx2, y-ly1, y-ly2, zL-b, zR-a."""
    L, R, E = [], [], []
    for i in range(q):
        L += [("x", i), ("zL", i)]
        R += [("y", i), ("zR", i)]
        E.append((("zL", i), ("zR", i)))
    for t, (x, y, z) in enumerate(triples):
        lx, ly1, ly2, a = ("lx", t), ("ly1", t), ("ly2", t), ("a", t)
        rx1, rx2, ry, b = ("rx1", t), ("rx2", t), ("ry", t), ("b", t)
        L += [lx, ly1, ly2, a]
        R += [rx1, rx2, ry, b]
        E += [(lx, rx1), (lx, rx2), (lx, ry), (ly1, ry), (ly1, rx1), (ly2, ry), (ly2, rx2), (ly2, b), (a, b), (a, rx2)]
        E += [(("x", x), rx1), (("x", x), rx2), (ly1, ("y", y)), (ly2, ("y", y)), (("zL", z), b), (a, ("zR", z))]
    adj = {v: set() for v in L + R}
    for u, v in E:
        adj[u].add(v)
        adj[v].add(u)
    return L, R, adj


def dm_exists(q, triples):
    for comb in itertools.combinations(range(len(triples)), q):
        xs = {triples[t][0] for t in comb}
        ys = {triples[t][1] for t in comb}
        zs = {triples[t][2] for t in comb}
        if len(xs) == len(ys) == len(zs) == q:
            return True
    return False


def H_to_pair(L, R, adj):
    iL = {v: i for i, v in enumerate(L)}
    iR = {v: j for j, v in enumerate(R)}
    adjA = [set(iR[u] for u in adj[v]) for v in L]
    return adjA


def blowup(L, R, adj, k):
    L2 = [(v, r) for v in L for r in range(k)]
    R2 = [(v, r) for v in R for r in range(k)]
    adj2 = {x: set() for x in L2 + R2}
    for v in L:
        for u in adj[v]:
            for r in range(k):
                for s in range(k):
                    adj2[(v, r)].add((u, s))
                    adj2[(u, s)].add((v, r))
    return L2, R2, adj2


# ---------------------------------------------------------------- tests
def T1(report):
    for N in (2, 3, 4):
        cells = [(i, j) for i in range(N) for j in range(N)]
        if N <= 3:
            graphs = range(1 << len(cells))
        else:
            graphs = [random.getrandbits(len(cells)) for _ in range(600)]
        cnt = 0; bad = 0; nfac = 0
        for code in graphs:
            adjA = [set() for _ in range(N)]
            for b, (i, j) in enumerate(cells):
                if (code >> b) & 1:
                    adjA[i].add(j)
            fac = c4_factor_bip(N, N, adjA)
            nfac += fac
            sec = [0] * N + [1] * N
            for mode in ("01", "pm"):
                W = pair_weights(adjA, N, N, mode)
                for G in (N, math.ceil(N / 2), 2 * N):
                    opt = split_opt(sec, W, 2, G)
                    cnt += 1
                    ok = opt <= 2 * N + 1e-9 and ((abs(opt - 2 * N) < 1e-9) == fac)
                    if not ok:
                        bad += 1
                        print("T1 FAIL", N, code, mode, G, opt, fac)
        report.append(f"T1 c=2 N={N}: {len(graphs)} graphs ({nfac} with a C4-factor), {cnt} (graph, weights, G) cases, failures={bad}")


def max_rows(s, t):
    """max total ones in s x t 0/1 matrix, no two rows sharing >= 3 columns (rows may repeat only if size <= 2)."""
    subsets = sorted(range(1 << t), key=lambda m: -bin(m).count("1"))
    best = [0]
    pc = [bin(m).count("1") for m in range(1 << t)]

    def rec(rows, start, total):
        if len(rows) == s:
            best[0] = max(best[0], total)
            return
        left = s - len(rows)
        if total + left * t <= best[0]:
            return
        for idx in range(start, len(subsets)):
            m = subsets[idx]
            if total + left * pc[m] <= best[0]:
                break
            if all(pc[m & r] <= 2 for r in rows):
                rows.append(m)
                rec(rows, idx, total + pc[m])  # idx (not idx+1): a row may repeat if allowed by the test
                rows.pop()

    rec([], 0, 0)
    return best[0]


def T2(report):
    # finite table
    lines = []
    allok = True
    for s in range(3, 7):
        for t in range(s, 7):
            e = max_rows(s, t)
            ok = 2 * e < s * t + s + t or (s, t) == (4, 4) and 2 * e == s * t + s + t
            allok &= ok
            lines.append(f"({s},{t}):e={e},2e={2*e} vs {s*t+s+t}")
    report.append("T2 table (no two rows share 3 columns): " + "; ".join(lines) + f"  -> strict except (4,4) equality: {allok}")
    # the (4,4) equality case: every 4x4 0/1 matrix with 12 ones, no two rows and no two columns sharing 3, is K44 minus a perfect matching
    n44 = n44pm = 0
    for code in range(1 << 16):
        if bin(code).count("1") != 12:
            continue
        M = [[(code >> (4 * i + j)) & 1 for j in range(4)] for i in range(4)]
        rows_ok = all(sum(M[i][k] & M[j][k] for k in range(4)) <= 2 for i, j in itertools.combinations(range(4), 2))
        cols_ok = all(sum(M[k][i] & M[k][j] for k in range(4)) <= 2 for i, j in itertools.combinations(range(4), 2))
        if rows_ok and cols_ok:
            n44 += 1
            zeros = [(i, j) for i in range(4) for j in range(4) if not M[i][j]]
            n44pm += len({i for i, _ in zeros}) == 4 and len({j for _, j in zeros}) == 4
    report.append(f"T2 (4,4) equality case: {n44} matrices with 12 ones and no K23/K32, of which {n44pm} are K44 minus a perfect matching (expect 24 = 4!)")
    # c unbounded, +-1 weights, K23-free graphs
    for N in (2, 3, 4):
        cells = [(i, j) for i in range(N) for j in range(N)]
        graphs = range(1 << len(cells)) if N <= 3 else [random.getrandbits(len(cells)) for _ in range(400)]
        cnt = bad = nfac = 0
        for code in graphs:
            adjA = [set() for _ in range(N)]
            for b, (i, j) in enumerate(cells):
                if (code >> b) & 1:
                    adjA[i].add(j)
            L = [("a", i) for i in range(N)]; R = [("b", j) for j in range(N)]
            adj = {v: set() for v in L + R}
            for i in range(N):
                for j in adjA[i]:
                    adj[("a", i)].add(("b", j)); adj[("b", j)].add(("a", i))
            if has_k23(adj, L, R):
                continue
            fac = c4_factor_bip(N, N, adjA); nfac += fac
            opt = split_opt([0] * N + [1] * N, pair_weights(adjA, N, N, "pm"), None, 2 * N)
            cnt += 1
            if not (opt <= 2 * N + 1e-9 and ((abs(opt - 2 * N) < 1e-9) == fac)):
                bad += 1; print("T2 FAIL", N, code, opt, fac)
        report.append(f"T2 c=unbounded +-1 N={N}: {cnt} K23-free graphs ({nfac} with C4-factor), failures={bad}")


def check_instance(q, triples, exact=False, blow=False):
    L, R, adj = build_H(q, triples)
    out = {}
    out["balanced"] = len(L) == len(R)
    out["bip"] = all(not (adj[v] & set(L)) for v in L)
    out["k23free"] = not has_k23(adj, L, R)
    # no 4 same-side vertices pairwise sharing two neighbours (so no K44 minus a perfect matching)
    k4 = False
    for side in (L, R):
        sh = {v: {u for u in side if u != v and len(adj[u] & adj[v]) >= 2} for v in side}
        for v in side:
            for a_, b_, c_ in itertools.combinations(sorted(sh[v]), 3):
                if b_ in sh[a_] and c_ in sh[a_] and c_ in sh[b_]:
                    k4 = True
    out["k23free"] = out["k23free"] and not k4
    dm = dm_exists(q, triples)
    fac = kcc_factor(L, R, adj, 2)
    out["iff"] = dm == fac
    out["dm"] = dm
    if exact:
        N = len(L)
        adjA = H_to_pair(L, R, adj)
        sec = [0] * N + [1] * N
        o1 = split_opt(sec, pair_weights(adjA, N, N, "01"), 2, N)
        o2 = split_opt(sec, pair_weights(adjA, N, N, "pm"), 2, N)
        out["exact"] = (abs(o1 - 2 * N) < 1e-9) == dm and (abs(o2 - 2 * N) < 1e-9) == dm and o1 <= 2 * N and o2 <= 2 * N
    if blow:
        L2, R2, adj2 = blowup(L, R, adj, 2)
        out["blow"] = kcc_factor(L2, R2, adj2, 4) == dm
    return out


def T3(report):
    # q = 2: all nonempty sets of triples
    q = 2
    alltr = list(itertools.product(range(q), repeat=3))
    stats = dict(n=0, yes=0, fail=0, exact=0, blow=0)
    for code in range(1, 1 << len(alltr)):
        triples = [alltr[i] for i in range(len(alltr)) if (code >> i) & 1]
        m = len(triples)
        o = check_instance(q, triples, exact=(m <= 1), blow=(m <= 4))
        stats["n"] += 1; stats["yes"] += o["dm"]
        if "exact" in o: stats["exact"] += 1
        if "blow" in o: stats["blow"] += 1
        if not (o["balanced"] and o["bip"] and o["k23free"] and o["iff"] and o.get("exact", True) and o.get("blow", True)):
            stats["fail"] += 1; print("T3 FAIL", triples, o)
    report.append(f"T3 q=2 all {stats['n']} instances ({stats['yes']} yes): structure + (3DM <=> C4-factor) failures={stats['fail']}; "
                  f"exact split OPT checked on {stats['exact']}; K44 blow-up checked on {stats['blow']}")
    # q = 1 exact (12 vertices) and q = 2, m = 2 exact
    ex = 0; exf = 0
    for triples in ([(0, 0, 0)],):
        o = check_instance(1, triples, exact=True, blow=True); ex += 1; exf += not (o["exact"] and o["iff"] and o["blow"])
    # q = 2, m = 2 (24 labels): about 9 minutes each, so one yes and one no instance
    for pair in ([(0, 0, 0), (1, 1, 1)], [(0, 0, 0), (1, 1, 0)]):
        o = check_instance(2, list(pair), exact=True); ex += 1; exf += not (o["exact"] and o["iff"])
    report.append(f"T3 exact split OPT (c=2, G=N, 0/1 and +-1) on {ex} instances with q<=2, m<=2 (up to 24 labels): failures={exf}")
    # random q = 3, 4
    for q, reps in ((3, 300), (4, 120)):
        alltr = list(itertools.product(range(q), repeat=3))
        n = yes = fail = 0
        for _ in range(reps):
            m = random.randint(q, 2 * q + 2)
            if random.random() < 0.5:
                # plant a matching so that yes-instances are common
                px, py, pz = (random.sample(range(q), q) for _ in range(3))
                planted = [(px[i], py[i], pz[i]) for i in range(q)]
                if random.random() < 0.5:
                    planted = planted[:-1]  # drop one: usually a near miss
                triples = list(set(planted + random.sample(alltr, m - len(planted))))
            else:
                triples = random.sample(alltr, m)
            o = check_instance(q, triples, blow=(q == 3 and len(triples) <= 5))
            n += 1; yes += o["dm"]
            if not (o["balanced"] and o["bip"] and o["k23free"] and o["iff"] and o.get("blow", True)):
                fail += 1; print("T3 FAIL", q, triples, o)
        report.append(f"T3 q={q} random {n} instances ({yes} yes): failures={fail}")


def mwm(Wm):
    """max weight bipartite matching, unmatched allowed."""
    M = np.maximum(Wm, 0)
    r, c = linear_sum_assignment(-M)
    return float(M[r, c].sum())


def T4(report):
    K, N = 3, 3
    pairs = [(0, 1), (1, 2), (0, 2)]
    sec = [s for s in range(K) for _ in range(N)]
    reps = 300
    bad = bad2 = 0; tight = 0
    t0 = time.time()
    for rep in range(reps):
        tables = {p: np.random.RandomState(rep * 7 + hash(p) % 97).randint(-3, 4, size=(N, N)).astype(float) for p in pairs}
        W = {}
        for (s, u), T in tables.items():
            for k in range(N):
                for l in range(N):
                    W[(s * N + k, u * N + l)] = T[k, l]
        for G in (N, K * N):
            stack = split_opt(sec, W, 2, G)
            pairsum = 0.0
            for (s, u), T in tables.items():
                Wp = {(k, N + l): T[k, l] for k in range(N) for l in range(N)}
                po = split_opt([0] * N + [1] * N, Wp, 2, G)
                pairsum += po
                if G == K * N:  # G unbounded for the pair (2N labels <= 9 classes is not binding when N = 3? 2N = 6 <= 9)
                    if po > 2 * mwm(T) + 1e-9:
                        bad2 += 1; print("T4 FAIL mwm", rep, (s, u), po, mwm(T))
            if stack > pairsum + 1e-9:
                bad += 1; print("T4 FAIL", rep, G, stack, pairsum)
            tight += abs(stack - pairsum) < 1e-9
    report.append(f"T4 stacks K=3 N=3 c=2: {reps} random stacks x 2 values of G (N and unbounded): stack OPT <= sum of pair OPT failures={bad} "
                  f"(bound tight in {tight} of {2*reps}); pair OPT <= 2*max weight matching failures={bad2} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    report = []
    for T in (T1, T2, T3, T4):
        t0 = time.time()
        T(report)
        print(report[-1] if report else "", f"[{time.time()-t0:.0f}s]", flush=True)
    print("\n==== SUMMARY ====")
    for r in report:
        print(r)
