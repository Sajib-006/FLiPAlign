"""Exact checks for split labelling with no cap (c = infinity) on zero-sum tables (2026-11-04).

See hardness/no_cap_zero_sum.md. The construction: take an integer core table w0 on A0 x B0 (n x n) whose row sums
rho_k and column sums gamma_l are all <= 0. Put X = -sum_k rho_k = -sum_l gamma_l. Write the A0 row deficits as a list
of X "units" (row k appears -rho_k times) and the B0 column deficits likewise, and pair them by index: unit j joins
k_j in A0 and l_j in B0. Unit j adds four new labels b1_j, b3_j in B and a2_j, a4_j in A and the path
    k_j  +1  b1_j  -1  a2_j  +1  b3_j  -1  a4_j  +1  l_j
(all other new entries 0). Every row and column of the padded table W sums to 0, and W has entries in {-1,0,1}
when w0 does. Claim (padding lemma): OPT(W; inf, inf) = OPT(w0; inf, inf) + 3X.
With w0 = the +-1 table of the 3DM graph H of Lemma split-H: OPT(W) >= 2n + 3X iff the 3DM instance is a yes-instance.

Exact solvers (cap = infinity, no class bound):
  opt_dp        subset DP over all set partitions of the labels (small cases only).
  opt_multicut  integer program over the nonzero pairs: x_e = 1 if pair e is split. A 0/1 vector is a partition
                exactly when no cycle of the support graph has exactly one split edge. We add these cycle
                constraints lazily (shortest kept path in the class) until the solution is a partition, so the
                final answer is the exact optimum. HiGHS (scipy.optimize.milp), gap 0, integer weights.
Tests:
  N0 zero sums; W = chance(W + J) exactly (Lemma zm-same with alpha = 1); entries in {-1,0,1}.
  N1 padding lemma on random small integer cores (n = 2, 3), DP and multicut both, and DP = multicut.
  N2 small K23-free graphs H (n = 4, all degrees <= 2): OPT(W) >= 2n + 3X iff H has a C4-factor.
  N3 the 3DM graph H (q = 2, m = 2, 3; yes and no instances): OPT(W) = 2n + 3X for yes, < for no.
  N4 explicit yes-labelling: value 2n + 3X with N/2 classes, and with n/2 + 1 classes.
  N5 control: a shorter path (k +1 b -1 a +1 l, length 3) breaks the padding lemma on some cores.
Run: ~/miniconda3/envs/base-3.10/bin/python hardness/check_no_cap.py   (writes hardness/check_no_cap.log)
"""
import itertools, os, random, sys, time
from fractions import Fraction as Fr

import numpy as np
import scipy.sparse as sp
from scipy.optimize import milp, LinearConstraint, Bounds

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_split import build_H, dm_exists, H_to_pair, has_k23, c4_factor_bip  # noqa: E402

random.seed(20261104)
LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "check_no_cap.log")
_logf = open(LOG, "w")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _logf.write(s + "\n")
    _logf.flush()


# ------------------------------------------------------------------ construction
def margins(w):
    n1, n2 = len(w), len(w[0])
    return [sum(w[i]) for i in range(n1)], [sum(w[i][j] for i in range(n1)) for j in range(n2)]


def pad(w0, length=5):
    """Return W (list of lists, ints), n, X, units. Labels: A = A0 (0..n-1) then pads; B likewise.
    length=5: the construction of the proof. length=3 (control): k +1 b1 -1 a2 +1 l."""
    n = len(w0)
    rho, gam = margins(w0)
    assert all(x <= 0 for x in rho) and all(x <= 0 for x in gam) and sum(rho) == sum(gam)
    X = -sum(rho)
    ka = [k for k in range(n) for _ in range(-rho[k])]
    lb = [l for l in range(n) for _ in range(-gam[l])]
    per = 2 if length == 5 else 1  # new labels per side per unit
    N = n + per * X
    W = [[0] * N for _ in range(N)]
    for i in range(n):
        for j in range(n):
            W[i][j] = w0[i][j]
    units = []
    for j in range(X):
        k, l = ka[j], lb[j]
        if length == 5:
            b1, b3 = n + 2 * j, n + 2 * j + 1
            a2, a4 = n + 2 * j, n + 2 * j + 1
            W[k][b1] += 1; W[a2][b1] -= 1; W[a2][b3] += 1; W[a4][b3] -= 1; W[a4][l] += 1
            units.append((k, b1, a2, b3, a4, l))
        else:
            b1, a2 = n + j, n + j
            W[k][b1] += 1; W[a2][b1] -= 1; W[a2][l] += 1
            units.append((k, b1, a2, l))
    return W, n, X, units


def zero_sums(W):
    r, c = margins(W)
    return all(x == 0 for x in r) and all(x == 0 for x in c)


def chance(C):
    M1, M2 = len(C), len(C[0])
    r = [sum(row) for row in C]
    cc = [sum(C[i][j] for i in range(M1)) for j in range(M2)]
    T = sum(r)
    return [[C[i][j] - Fr(r[i] * cc[j], T) for j in range(M2)] for i in range(M1)]


def value(W, classes):
    """classes: list of (set of A indices, set of B indices)."""
    return sum(W[i][j] for SA, SB in classes for i in SA for j in SB)


# ------------------------------------------------------------------ exact solvers (cap = infinity, G = infinity)
def opt_dp(W):
    N = len(W)
    L = 2 * N  # label v < N is A-row v, v >= N is B-col v-N
    full = (1 << L) - 1
    # class value for every subset
    cv = np.zeros(1 << L, dtype=np.int64)
    for S in range(1, 1 << L):
        low = (S & -S).bit_length() - 1
        R = S & (S - 1)
        # add label low to class R
        v = 0
        if low < N:
            for j in range(N):
                if (R >> (N + j)) & 1:
                    v += W[low][j]
        else:
            for i in range(N):
                if (R >> i) & 1:
                    v += W[i][low - N]
        cv[S] = cv[R] + v
    best = np.full(1 << L, -10**9, dtype=np.int64)
    best[0] = 0
    for S in range(1, 1 << L):
        low = S & -S
        rest = S ^ low
        b = -10**9
        sub = rest
        while True:
            T = sub | low
            val = cv[T] + best[S ^ T]
            if val > b:
                b = val
            if sub == 0:
                break
            sub = (sub - 1) & rest
        best[S] = b
    return int(best[full])


def opt_multicut(W, init_cycles=(), time_limit=1200):
    """Exact max over all partitions (no cap, no class bound). Returns (opt, classes, rounds)."""
    N = len(W)
    edges = [(i, N + j, W[i][j]) for i in range(N) for j in range(N) if W[i][j] != 0]
    E = len(edges)
    if E == 0:
        return 0, [(set(range(N)), set(range(N)))], 0
    eid = {}
    nbr = {v: [] for v in range(2 * N)}
    for e, (u, v, _) in enumerate(edges):
        eid[(u, v)] = eid[(v, u)] = e
        nbr[u].append(v); nbr[v].append(u)
    wvec = np.array([w for _, _, w in edges], dtype=np.float64)
    rows = []  # each row: (e, [path edges]) meaning x_e - sum x_f <= 0
    seen = set()

    def add_cycle(cyc):  # cyc: list of vertices, closed
        es = [eid[(cyc[i], cyc[(i + 1) % len(cyc)])] for i in range(len(cyc))]
        for e in es:
            key = (e, tuple(sorted(f for f in es if f != e)))
            if key not in seen:
                seen.add(key)
                rows.append((e, [f for f in es if f != e]))

    for cyc in init_cycles:
        add_cycle(cyc)
    rounds = 0
    while True:
        rounds += 1
        # minimise sum w_e x_e  (value = sum w - sum w_e x_e)
        if rows:
            data, ri, ci = [], [], []
            for r, (e, fs) in enumerate(rows):
                ri.append(r); ci.append(e); data.append(1.0)
                for f in fs:
                    ri.append(r); ci.append(f); data.append(-1.0)
            Amat = sp.csr_matrix((data, (ri, ci)), shape=(len(rows), E))
            cons = [LinearConstraint(Amat, -np.inf, 0.0)]
        else:
            cons = []
        res = milp(c=wvec, constraints=cons, integrality=np.ones(E), bounds=Bounds(0, 1),
                   options={"time_limit": time_limit, "mip_rel_gap": 0.0})
        assert res.status == 0, res.message
        x = np.round(res.x).astype(int)
        # components of kept edges
        comp = [-1] * (2 * N)
        cid = 0
        for s in range(2 * N):
            if comp[s] >= 0:
                continue
            stack = [s]; comp[s] = cid
            while stack:
                u = stack.pop()
                for v in nbr[u]:
                    if comp[v] < 0 and x[eid[(u, v)]] == 0:
                        comp[v] = cid; stack.append(v)
            cid += 1
        viol = 0
        for e, (u, v, _) in enumerate(edges):
            if x[e] == 1 and comp[u] == comp[v]:
                # shortest kept path u -> v (BFS)
                prev = {u: None}
                q = [u]
                for a in q:
                    if a == v:
                        break
                    for b in nbr[a]:
                        if b not in prev and x[eid[(a, b)]] == 0:
                            prev[b] = a; q.append(b)
                path = [v]
                while prev[path[-1]] is not None:
                    path.append(prev[path[-1]])
                add_cycle(path)
                viol += 1
        if viol == 0:
            classes = {}
            for v in range(2 * N):
                SA, SB = classes.setdefault(comp[v], (set(), set()))
                (SA if v < N else SB).add(v if v < N else v - N)
            cl = list(classes.values())
            val = value(W, cl)
            assert abs(val - (wvec.sum() - wvec @ x)) < 1e-6
            return val, cl, rounds


def core_c4_cycles(n, full=True):
    """All 4-cycles of K_{n,n} on the core labels (A i, B N+j are given later)."""
    out = []
    for i1, i2 in itertools.combinations(range(n), 2):
        for j1, j2 in itertools.combinations(range(n), 2):
            out.append((i1, j1, i2, j2))
    return out


def init_cycles_for(W, n, units):
    """Chordless cycles when the core is complete (no zero entries): 4-cycles of the core and, per unit,
    the 6-cycle path + core edge k l. (Lazy constraints still run, so this is only a speed-up.)"""
    N = len(W)
    cyc = [(i1, N + j1, i2, N + j2) for (i1, j1, i2, j2) in core_c4_cycles(n)
           if W[i1][j1] and W[i2][j1] and W[i2][j2] and W[i1][j2]]
    for (k, b1, a2, b3, a4, l) in units:
        if W[k][l] != 0:
            cyc.append((k, N + b1, a2, N + b3, a4, N + l))
    return cyc


# ------------------------------------------------------------------ yes labelling
def yes_labelling(n, units, core_classes, merge_pads):
    """core_classes: list of (SA, SB) on the core. Adds b1_j to k_j's class, a4_j to l_j's class,
    and {a2_j, b3_j} as one class per unit (merge_pads=False) or one class for all units (True)."""
    where_A = {}
    where_B = {}
    cls = [(set(SA), set(SB)) for SA, SB in core_classes]
    for idx, (SA, SB) in enumerate(cls):
        for i in SA:
            where_A[i] = idx
        for j in SB:
            where_B[j] = idx
    padc = (set(), set())
    for (k, b1, a2, b3, a4, l) in units:
        cls[where_A[k]][1].add(b1)
        cls[where_B[l]][0].add(a4)
        if merge_pads:
            padc[0].add(a2); padc[1].add(b3)
        else:
            cls.append(({a2}, {b3}))
    if merge_pads and units:
        cls.append(padc)
    return cls


def find_c4_factor(n, adjA):
    """Return a C4-factor as a list of (SA, SB) or None (small n, brute force)."""
    def rec(freeA, freeB):
        if not freeA:
            return []
        i1 = min(freeA)
        for i2 in freeA:
            if i2 <= i1:
                continue
            common = sorted(adjA[i1] & adjA[i2] & freeB)
            for j1, j2 in itertools.combinations(common, 2):
                r = rec(freeA - {i1, i2}, freeB - {j1, j2})
                if r is not None:
                    return [({i1, i2}, {j1, j2})] + r
        return None
    return rec(frozenset(range(n)), frozenset(range(n)))


def pm_table(n, adjA):
    return [[1 if j in adjA[i] else -1 for j in range(n)] for i in range(n)]


# ------------------------------------------------------------------ tests
def N0(rep, cases):
    bad = 0
    for W in cases:
        N = len(W)
        ok = zero_sums(W) and all(x in (-1, 0, 1) for row in W for x in row)
        C = [[W[i][j] + 1 for j in range(N)] for i in range(N)]
        ok = ok and min(min(r) for r in C) >= 0 and chance(C) == [[Fr(x) for x in row] for row in W]
        bad += not ok
    rep.append(f"N0 zero sums, entries in {{-1,0,1}}, W = chance(W + J), C in {{0,1,2}}: {len(cases)} tables, failures={bad}")
    log(rep[-1])


def random_core(n, maxX, entries):
    while True:
        w = [[random.choice(entries) for _ in range(n)] for _ in range(n)]
        r, c = margins(w)
        if all(x <= 0 for x in r) and all(x <= 0 for x in c) and -sum(r) <= maxX:
            return w


def N1(rep, allW):
    t0 = time.time()
    bad = cnt = 0
    tight_dp = 0
    for n, maxX, entries, reps in [(2, 2, (-1, 0, 1), 40), (2, 2, (-1, 1), 20), (3, 2, (-1, 0, 1), 40),
                                   (3, 1, (-2, -1, 0, 1, 2), 30)]:
        for _ in range(reps):
            w0 = random_core(n, maxX, entries)
            W, n_, X, units = pad(w0)
            allW.append(W) if all(abs(x) <= 1 for r in w0 for x in r) else None
            o0 = opt_dp(w0)
            if 2 * len(W) <= 14:
                o1 = opt_dp(W)
                m1, _, _ = opt_multicut(W)
                tight_dp += 1
                ok = (o1 == m1 == o0 + 3 * X)
            else:
                m1, _, _ = opt_multicut(W)
                ok = (m1 == o0 + 3 * X)
            cnt += 1
            if not ok:
                bad += 1
                log("  N1 FAIL", w0, X, o0, m1)
    rep.append(f"N1 padding lemma OPT(W) = OPT(w0) + 3X on {cnt} random cores (n=2,3; entries in {{-1,0,1}}, "
               f"{{-1,1}}, {{-2..2}}; X <= 2), DP and multicut agree on {tight_dp}: failures={bad} [{time.time()-t0:.0f}s]")
    log(rep[-1])


def N2(rep, allW):
    t0 = time.time()
    n = 4
    graphs = []
    for code in range(1 << 16):
        adjA = [set(j for j in range(n) if (code >> (4 * i + j)) & 1) for i in range(n)]
        dA = [len(a) for a in adjA]
        dB = [sum(1 for i in range(n) if j in adjA[i]) for j in range(n)]
        if max(dA + dB) > n // 2:
            continue
        graphs.append(adjA)
    # keep K23-free (automatic: degree <= 2) graphs; sample: all with a C4-factor + random others
    yes = [g for g in graphs if find_c4_factor(n, g) is not None]
    no = [g for g in graphs if find_c4_factor(n, g) is None]
    random.shuffle(no)
    sample = yes + no[:30]
    bad = 0
    gaps = []
    for adjA in sample:
        w0 = pm_table(n, adjA)
        W, _, X, units = pad(w0)
        allW.append(W)
        opt, _, _ = opt_multicut(W, init_cycles_for(W, n, units))
        fac = find_c4_factor(n, adjA) is not None
        thr = 2 * n + 3 * X
        if (opt >= thr) != fac or opt > thr:
            bad += 1
            log("  N2 FAIL", adjA, opt, thr, fac)
        if not fac:
            gaps.append(thr - opt)
    rep.append(f"N2 n=4 graphs with all degrees <= 2 ({len(graphs)} such graphs, {len(yes)} with a C4-factor): "
               f"checked {len(sample)} ({len(yes)} yes, {min(30, len(no))} no), N = 4 + 2X up to "
               f"{4 + 2 * 16}; OPT >= 2n+3X iff C4-factor, OPT <= 2n+3X always: failures={bad}; "
               f"smallest gap on no: {min(gaps) if gaps else '-'} [{time.time()-t0:.0f}s]")
    log(rep[-1])


def threedm_instances():
    out = []
    q = 2
    allt = [(x, y, z) for x in range(q) for y in range(q) for z in range(q)]
    for m in (2, 3):
        yes_done = no_done = 0
        combos = list(itertools.combinations(allt, m))
        random.shuffle(combos)
        for T in combos:
            d = dm_exists(q, list(T))
            if d and yes_done < 2:
                out.append((q, list(T), True)); yes_done += 1
            elif not d and no_done < 2:
                out.append((q, list(T), False)); no_done += 1
            if yes_done == 2 and no_done == 2:
                break
    # q = 3, m = 3: one yes (a perfect matching) and one no (x = 0 used twice)
    out.append((3, [(0, 0, 0), (1, 1, 1), (2, 2, 2)], True))
    out.append((3, [(0, 0, 0), (0, 1, 1), (2, 2, 2)], False))
    return out


def N3(rep, allW, insts):
    bad = 0
    lines = []
    for q, T, d in insts:
        t0 = time.time()
        L, R, adj = build_H(q, T)
        adjA = H_to_pair(L, R, adj)
        n = len(L)
        assert not has_k23(adj, L, R)
        w0 = pm_table(n, adjA)
        rho, gam = margins(w0)
        W, _, X, units = pad(w0)
        allW.append(W)
        thr = 2 * n + 3 * X
        opt, cl, rounds = opt_multicut(W, init_cycles_for(W, n, units))
        ok = (opt == thr) if d else (opt < thr)
        fac = c4_factor_bip(n, n, adjA)
        ok = ok and (fac == d)
        bad += not ok
        lines.append(f"q={q} m={len(T)} {'yes' if d else 'no '} n={n} X={X} N={len(W)} OPT={opt} 2n+3X={thr} "
                     f"rounds={rounds} [{time.time()-t0:.0f}s]")
        log("  N3", lines[-1])
    rep.append(f"N3 3DM graph H (q=2, m=2,3; q=3, m=3), padded, exact multicut: {len(insts)} instances "
               f"({sum(d for *_, d in insts)} yes): failures={bad}")
    rep.extend("   " + s for s in lines)
    log(rep[-1])


def N4(rep):
    bad = cnt = 0
    for q in (2, 3):
        allt = [(x, y, z) for x in range(q) for y in range(q) for z in range(q)]
        for _ in range(20):
            m = random.randint(q, q + 3)
            T = random.sample(allt, m)
            # force a matching in half the cases
            if random.random() < 0.5:
                perm_y, perm_z = random.sample(range(q), q), random.sample(range(q), q)
                for i in range(q):
                    t = (i, perm_y[i], perm_z[i])
                    if t not in T:
                        T[i % len(T)] = t
                T = list(dict.fromkeys(T))
            if not dm_exists(q, T):
                continue
            L, R, adj = build_H(q, T)
            adjA = H_to_pair(L, R, adj)
            n = len(L)
            F = find_c4_factor(n, adjA)
            w0 = pm_table(n, adjA)
            W, _, X, units = pad(w0)
            N = len(W)
            for merge in (False, True):
                cl = yes_labelling(n, units, F, merge)
                labelsA = sorted(i for SA, _ in cl for i in SA)
                labelsB = sorted(j for _, SB in cl for j in SB)
                ok = labelsA == list(range(N)) and labelsB == list(range(N))
                ok = ok and value(W, cl) == 2 * n + 3 * X and zero_sums(W)
                ok = ok and len(cl) == (N // 2 if not merge else n // 2 + 1)
                cnt += 1
                bad += not ok
    rep.append(f"N4 explicit yes-labelling on {cnt // 2} 3DM yes-instances (q=2,3): value 2n+3X with N/2 classes "
               f"and with n/2+1 classes, zero sums: {cnt} labellings, failures={bad}")
    log(rep[-1])


def N5(rep):
    """Control: path of length 3 instead of 5. Expect OPT(W3) < OPT(w0) + 2X on some cores."""
    found = tried = 0
    ex = None
    cores = [[[2, -3], [-3, 1]], [[1, -2], [-2, 1]]]  # paired units join k, l with a +weight core pair
    cores += [random_core(2, 3, (-3, -2, -1, 0, 1, 2)) for _ in range(60)]
    for w0 in cores:
        W3, n, X, units = pad(w0, length=3)
        if 2 * len(W3) > 14:
            continue
        tried += 1
        if opt_dp(W3) != opt_dp(w0) + 2 * X:
            assert opt_dp(W3) < opt_dp(w0) + 2 * X
            found += 1
            ex = ex or (w0, X, opt_dp(W3), opt_dp(w0))
    rep.append(f"N5 control, length-3 path: identity OPT(W3) = OPT(w0) + 2X fails on {found} of {tried} random cores "
               f"(example w0={ex[0]}, X={ex[1]}, OPT(W3)={ex[2]}, OPT(w0)={ex[3]})" if ex else
               f"N5 control: no failure found in {tried} cores")
    log(rep[-1])


if __name__ == "__main__":
    rep = []
    allW = []
    T0 = time.time()
    log("check_no_cap.py started", time.strftime("%Y-%m-%d %H:%M:%S"))
    N1(rep, allW)
    N5(rep)
    N4(rep)
    N2(rep, allW)
    N3(rep, allW, threedm_instances())
    N0(rep, allW)
    log("\n==== SUMMARY ====")
    for s in rep:
        log(s)
    log(f"total time {time.time()-T0:.0f}s")
