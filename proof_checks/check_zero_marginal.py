"""Brute-force checks: split labelling stays strongly NP-hard on zero-marginal (chance-corrected) tables.

Notes: hardness/zero_marginal.md. Reuses build_H, dm_exists, kcc_factor, has_k23, split_opt from check_split.py.

Two padding constructions turn a hard instance on a bipartite graph H (n labels per side) into a table on
M = n + n' labels per side whose rows and columns all sum to zero:
  (A) constant-margin padding (any cap c, used for c = 2): C = 0/1 table of H, plus fill entries in [0, 1) on new
      labels, plus n'/c disjoint K_{c,c} pad blocks; all row and column sums of C equal R; w = C - (R/M) J.
      Claim: OPT(w; c, G) <= c (M - R) for every G, with equality iff H has a K_{c,c}-factor (G >= M/c).
  (B) +-1 padding (every finite cap c >= 2): the +-1 table of H, cross blocks that cancel its margins, and n'/2
      disjoint C4 pad blocks (+1 inside a block, -theta elsewhere). With n' large enough
      (c n (max|row sum| + max|col sum|) / n' < 1), OPT >= 2M iff H has a C4-factor; otherwise OPT < 2M (the proof gives only this;
      the checks also print the gap 2M - OPT, which is larger on every tested case).
Every table is checked in exact rational arithmetic to have zero margins and to equal C - r c^T / T for an explicit
non-negative integer table C. Exact optima: bitmask DP (check_split.split_opt) for <= 16 labels, else an exact
set-partition integer program (every class with <= c labels per side is a column) with integer weights.

Tests:
  Z0  every square zero-marginal table W equals the chance correction of C = W + alpha J (alpha >= -min W), and every
      chance-corrected table of a non-negative C has zero margins (random, exact fractions).
  Z1  construction (A), c = 2: all bipartite graphs with n = 2 and n = 3 per side (small pads), G = M, ceil(M/2), 2M.
  Z2  construction (A), c = 2, on the 3DM graph H of split_theory.tex: q = 1 and q = 2 instances, yes and no, G = M.
  Z3  construction (A), c = 3 and c = 4 (K_{c,c}-factors), random small graphs.
  Z4  construction (B), +-1 padding with the pad size the proof asks for: c = 2 and c = 3, all graphs n = 2;
      c = 2, random K23-free graphs with n = 3; G = M.
  Z5  the pad sizes used in the proof (A): n' = 2cn, R = max(maxdeg, 2c) satisfy all conditions (3DM graphs, c = 2, 4).
Run:  ~/miniconda3/envs/base-3.10/bin/python hardness/check_zero_marginal.py
"""
import itertools, math, os, random, sys, time
from fractions import Fraction as Fr

import numpy as np
import scipy.sparse as sp
from scipy.optimize import milp, LinearConstraint, Bounds

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_split import build_H, dm_exists, kcc_factor, has_k23, split_opt, H_to_pair  # noqa: E402

random.seed(2026)


# ------------------------------------------------------------------ helpers
def chance(C):
    """w = C - r c^T / T, exact."""
    M1, M2 = len(C), len(C[0])
    r = [sum(row) for row in C]
    cc = [sum(C[i][j] for i in range(M1)) for j in range(M2)]
    T = sum(r)
    return [[C[i][j] - Fr(r[i] * cc[j]) / T for j in range(M2)] for i in range(M1)]


def zero_margins(W):
    return all(sum(row) == 0 for row in W) and all(sum(W[i][j] for i in range(len(W))) == 0 for j in range(len(W[0])))


def lcd(W):
    d = 1
    for row in W:
        for x in row:
            d = d * Fr(x).denominator // math.gcd(d, Fr(x).denominator)
    return d


def as_int(W):
    d = lcd(W)
    return [[int(Fr(x) * d) for x in row] for row in W], d


def adj_dict(n, adjA):
    L = [("a", i) for i in range(n)]
    R = [("b", j) for j in range(n)]
    adj = {v: set() for v in L + R}
    for i in range(n):
        for j in adjA[i]:
            adj[("a", i)].add(("b", j)); adj[("b", j)].add(("a", i))
    return L, R, adj


def has_kcc_factor(n, adjA, c):
    L, R, adj = adj_dict(n, adjA)
    return kcc_factor(L, R, adj, c)


def degrees(n, adjA):
    dA = [len(adjA[i]) for i in range(n)]
    dB = [sum(1 for i in range(n) if j in adjA[i]) for j in range(n)]
    return dA, dB


# ------------------------------------------------------------------ exact optimum
def opt_dp(Wint, c, G):
    M = len(Wint)
    sec = [0] * M + [1] * M
    Wd = {(i, M + j): float(Wint[i][j]) for i in range(M) for j in range(M)}
    return int(round(split_opt(sec, Wd, c, G)))


def subsets_upto(M, c):
    out = []
    for k in range(0, c + 1):
        out.extend(itertools.combinations(range(M), k))
    return out


def opt_milp(Wint, c, G, time_limit=600):
    """Exact max of the split labelling value: set partitioning over all classes with <= c labels per side."""
    M = len(Wint)
    Wn = np.array(Wint, dtype=np.float64)
    subs = subsets_upto(M, c)
    S = len(subs)
    IA = np.zeros((S, M))
    for s, sub in enumerate(subs):
        IA[s, list(sub)] = 1
    V = IA @ Wn @ IA.T  # V[a, b] = value of class (subs[a] on side A, subs[b] on side B)
    a_idx, b_idx = np.meshgrid(np.arange(S), np.arange(S), indexing="ij")
    a_idx = a_idx.ravel(); b_idx = b_idx.ravel()
    keep = (a_idx + b_idx) > 0  # drop the empty class (subs[0] is the empty set)
    a_idx = a_idx[keep]; b_idx = b_idx[keep]
    vals = V[a_idx, b_idx]
    ncol = len(vals)
    # build incidence with numpy
    lenA = np.array([len(s) for s in subs]); lenB = lenA
    nnz_per = lenA[a_idx] + lenB[b_idx]
    colid = np.repeat(np.arange(ncol), nnz_per)
    rowid = np.empty(nnz_per.sum(), dtype=np.int64)
    pos = 0
    subsA = [np.array(s, dtype=np.int64) for s in subs]
    for k in range(ncol):
        sa = subsA[a_idx[k]]; sb = subsA[b_idx[k]]
        rowid[pos:pos + len(sa)] = sa; pos += len(sa)
        rowid[pos:pos + len(sb)] = M + sb; pos += len(sb)
    A = sp.csr_matrix((np.ones(len(rowid)), (rowid, colid)), shape=(2 * M, ncol))
    cons = [LinearConstraint(A, np.ones(2 * M), np.ones(2 * M))]
    if G < 2 * M:
        cons.append(LinearConstraint(sp.csr_matrix(np.ones((1, ncol))), 0, G))
    res = milp(-vals, constraints=cons, integrality=np.ones(ncol), bounds=Bounds(0, 1),
               options={"mip_rel_gap": 0, "time_limit": time_limit, "disp": False})
    if res.status != 0:
        raise RuntimeError(f"milp status {res.status}: {res.message}")
    val = -res.fun
    x = res.x > 0.5
    exact = int(round(vals[x].sum()))  # value of the returned partition, in integers
    assert abs(val - exact) < 1e-6 * max(1, abs(exact)) + 1e-6
    return exact


def opt_exact(Wint, c, G):
    M = len(Wint)
    if 2 * M <= 16:
        return opt_dp(Wint, c, G), "dp"
    return opt_milp(Wint, c, G), "ip"


# ------------------------------------------------------------------ exact optimum using the symmetry of the pad labels
def _pieces(beta):
    return [(a, b) for a in range(beta + 1) for b in range(beta + 1) if a + b > 0]


def _multisets(items, maxa, maxb, start=0):
    """all multisets (sorted tuples) of pieces with total a <= maxa and total b <= maxb (including the empty one)."""
    yield ()
    for i in range(start, len(items)):
        a, b = items[i]
        if a <= maxa and b <= maxb:
            for rest in _multisets(items, maxa - a, maxb - b, i):
                yield (items[i],) + rest


def _block_patterns(beta):
    """partitions of a (beta, beta) pad block into pieces, as sorted tuples."""
    out = set()
    pcs = _pieces(beta)
    def rec(a, b, start, cur):
        if a == 0 and b == 0:
            out.add(tuple(cur)); return
        for i in range(start, len(pcs)):
            pa, pb = pcs[i]
            if pa <= a and pb <= b:
                rec(a - pa, b - pb, i, cur + [pcs[i]])
    rec(beta, beta, 0, [])
    return sorted(out)


def opt_agg(core, x, y, beta, nb, vin, vout, c, G):
    """Exact OPT for a table on (A0 + A') x (B0 + B') where: core = n x n table on A0 x B0; W(k, l') = x[k] for every
    l' in B'; W(k', l) = y[l] for every k' in A'; A' and B' are split into nb pad blocks of beta + beta labels, with
    W = vin inside a block and vout across blocks. All entries Fractions. Every class is described by its labels in
    A0 and B0 and the pieces it takes from pad blocks (one piece per block it meets); a block is cut into pieces by a
    'pattern'. Integer program over counts of class types and block patterns. Returns (OPT as Fraction)."""
    n = len(core)
    pcs = _pieces(beta)
    pats = _block_patterns(beta)
    types = []
    subsA = subsets_upto(n, min(c, n))
    for SA in subsA:
        for SB in subsA:
            for ms in _multisets(pcs, c - len(SA), c - len(SB)):
                if not SA and not SB and not ms:
                    continue
                sp_ = sum(a for a, _ in ms); tp = sum(b for _, b in ms)
                same = sum(a * b for a, b in ms)
                v = sum(core[k][l] for k in SA for l in SB)
                v += tp * sum(x[k] for k in SA) + sp_ * sum(y[l] for l in SB)
                v += vin * same + vout * (sp_ * tp - same)
                types.append((SA, SB, ms, v))
    allv = [t[3] for t in types]
    d = 1
    for v in allv:
        d = d * Fr(v).denominator // math.gcd(d, Fr(v).denominator)
    T = len(types); P = len(pats)
    nvar = T + P
    rows = []; lo = []; hi = []
    # H labels covered once
    for k in range(n):
        rows.append([1 if k in t[0] else 0 for t in types] + [0] * P); lo.append(1); hi.append(1)
    for l in range(n):
        rows.append([1 if l in t[1] else 0 for t in types] + [0] * P); lo.append(1); hi.append(1)
    # piece balance: pieces used by classes == pieces produced by block patterns
    for pc in pcs:
        rows.append([t[2].count(pc) for t in types] + [-pat.count(pc) for pat in pats]); lo.append(0); hi.append(0)
    rows.append([0] * T + [1] * P); lo.append(nb); hi.append(nb)
    if G < 2 * (n + beta * nb):
        rows.append([1] * T + [0] * P); lo.append(0); hi.append(G)
    A = sp.csr_matrix(np.array(rows, dtype=float))
    obj = np.array([-float(int(v * d)) for v in allv] + [0.0] * P)
    res = milp(obj, constraints=[LinearConstraint(A, lo, hi)], integrality=np.ones(nvar),
               bounds=Bounds(0, np.inf), options={"mip_rel_gap": 0, "time_limit": 3600})
    if res.status != 0:
        raise RuntimeError(res.message)
    xs = np.round(res.x).astype(int)
    val = sum(Fr(int(xs[i])) * types[i][3] for i in range(T))
    assert abs(float(val * d) + res.fun) < 1e-6 * max(1, abs(float(val * d))) + 1e-6
    return val


def agg_from_W(W, n, beta):
    """read off (core, x, y, vin, vout) from a padded table and check it has the symmetric form."""
    M = len(W); npad = M - n; nb = npad // beta
    blk = lambda i: (i - n) // beta
    core = [row[:n] for row in W[:n]]
    x = [W[k][n] for k in range(n)]; y = [W[n][l] for l in range(n)]
    vin = W[n][n]
    vout = W[n][n + beta] if nb > 1 else Fr(0)
    ok = all(W[k][j] == x[k] for k in range(n) for j in range(n, M))
    ok &= all(W[i][l] == y[l] for i in range(n, M) for l in range(n))
    ok &= all(W[i][j] == (vin if blk(i) == blk(j) else vout) for i in range(n, M) for j in range(n, M))
    assert ok, "table is not of the symmetric padded form"
    return core, x, y, nb, vin, vout


# ------------------------------------------------------------------ construction (A): constant margins
def pad_const(n, adjA, c, npad, R):
    """Return C (M x M Fractions), adjacency of H* (list of sets), or None if a condition fails."""
    R = Fr(R)
    dA, dB = degrees(n, adjA)
    m0 = sum(dA)
    if npad % c or npad < c:
        return None
    mu = (R * n - m0) / npad
    if npad == c:
        if R - c - mu != 0:
            return None
        phi = Fr(0)
    else:
        phi = (R - c - mu) / (npad - c)
    M = n + npad
    ok = (R >= max(dA + dB)) and phi >= 0 and phi < 1 and R < M
    ok = ok and all((R - d) / npad < 1 for d in dA + dB)
    if not ok:
        return None
    C = [[Fr(0)] * M for _ in range(M)]
    adjS = [set() for _ in range(M)]
    for i in range(n):
        for j in range(n):
            if j in adjA[i]:
                C[i][j] = Fr(1); adjS[i].add(j)
        for jp in range(n, M):
            C[i][jp] = (R - dA[i]) / npad
    for ip in range(n, M):
        for j in range(n):
            C[ip][j] = (R - dB[j]) / npad
        blk = (ip - n) // c
        for jp in range(n, M):
            if (jp - n) // c == blk:
                C[ip][jp] = Fr(1); adjS[ip].add(jp)
            else:
                C[ip][jp] = phi
    return C, adjS, R


def check_const(C, adjS, R):
    """conditions used by the proof: C >= 0, entries <= 1, ==1 exactly on edges of H*, margins all R, 0 < R < M."""
    M = len(C)
    ok = all(0 <= C[i][j] <= 1 for i in range(M) for j in range(M))
    ok &= all((C[i][j] == 1) == (j in adjS[i]) for i in range(M) for j in range(M))
    ok &= all(sum(C[i]) == R for i in range(M)) and all(sum(C[i][j] for i in range(M)) == R for j in range(M))
    ok &= 0 < R < M
    return ok


def choose_const(n, adjA, c, maxpad=40):
    """Smallest pad (then smallest integer R) meeting all conditions."""
    dA, dB = degrees(n, adjA)
    for npad in range(c, maxpad + 1, c):
        for R in range(max(dA + dB + [1]), n + npad):
            out = pad_const(n, adjA, c, npad, R)
            if out is not None:
                return out
    return None


# ------------------------------------------------------------------ construction (B): +-1 table with cancelling pads
def pad_pm(n, adjA, npad):
    """W (M x M Fractions), zero margins; the n x n corner is the +-1 table of H; npad even."""
    dA, dB = degrees(n, adjA)
    rho = [2 * d - n for d in dA]   # row sums of the +-1 table
    gam = [2 * d - n for d in dB]   # column sums
    S = sum(rho)
    theta = (2 - Fr(S, npad)) / (npad - 2)
    assert theta >= 0
    M = n + npad
    W = [[Fr(0)] * M for _ in range(M)]
    for i in range(n):
        for j in range(n):
            W[i][j] = Fr(1) if j in adjA[i] else Fr(-1)
        for jp in range(n, M):
            W[i][jp] = Fr(-rho[i], npad)
    for ip in range(n, M):
        for j in range(n):
            W[ip][j] = Fr(-gam[j], npad)
        for jp in range(n, M):
            W[ip][jp] = Fr(1) if (ip - n) // 2 == (jp - n) // 2 else -theta
    return W


def pm_pad_size(n, adjA, c):
    """Smallest even n' >= max(4, n*n/2) with c n (max|rho| + max|gamma|) / n' < 1 (the proof's condition)."""
    dA, dB = degrees(n, adjA)
    b = max(abs(2 * d - n) for d in dA) + max(abs(2 * d - n) for d in dB)
    npad = max(4, math.ceil(n * n / 2))
    while npad % 2 or not (c * n * b < npad):
        npad += 1
    return npad


# ------------------------------------------------------------------ tests
def Z0(report):
    bad = 0; cnt = 0
    for _ in range(300):
        N = random.randint(2, 5)
        W = [[Fr(random.randint(-9, 9), random.randint(1, 4)) for _ in range(N)] for _ in range(N)]
        # make margins zero by double centring
        rs = [sum(r) / N for r in W]; cs = [sum(W[i][j] for i in range(N)) / N for j in range(N)]
        tot = sum(rs) / N
        W = [[W[i][j] - rs[i] - cs[j] + tot for j in range(N)] for i in range(N)]
        assert zero_margins(W)
        alpha = max(Fr(0), -min(min(r) for r in W)) + Fr(random.randint(0, 3))
        if alpha == 0:
            alpha = Fr(1)
        C = [[W[i][j] + alpha for j in range(N)] for i in range(N)]
        cnt += 1
        bad += not (all(x >= 0 for r in C for x in r) and chance(C) == W)
        C2 = [[Fr(random.randint(0, 9)) for _ in range(N)] for _ in range(N)]
        if sum(map(sum, C2)) > 0:
            bad += not zero_margins(chance(C2))
    report.append(f"Z0 identity: zero-marginal W == chance(W + alpha J) and chance(C >= 0) has zero margins, {cnt} random tables, failures={bad}")


def run_const_case(n, adjA, c, Gs, C=None, adjS=None, R=None):
    if C is None:
        out = choose_const(n, adjA, c)
        if out is None:
            return None
        C, adjS, R = out
    M = len(C)
    if not check_const(C, adjS, R):
        return "cond"
    w = chance(C)
    if not zero_margins(w) or w != [[C[i][j] - R / M for j in range(M)] for i in range(M)]:
        return "w"
    Wint, d = as_int(w)
    thr = c * (M - R) * d  # threshold in integer units
    assert thr.denominator == 1
    fac = has_kcc_factor(n, adjA, c)
    res = []
    for G in Gs:
        opt, how = opt_exact(Wint, c, G)
        feasibleG = G >= math.ceil(M / c) if c else True
        good = opt <= thr and ((opt == thr) == (fac and feasibleG))
        res.append((G, opt, int(thr), how, good))
    return dict(M=M, R=R, fac=fac, res=res, ok=all(r[4] for r in res))


def Z1(report):
    for n in (2, 3):
        cells = [(i, j) for i in range(n) for j in range(n)]
        cnt = bad = nfac = skipped = 0; sizes = set(); hows = set()
        t0 = time.time()
        for code in range(1 << len(cells)):
            adjA = [set() for _ in range(n)]
            for b, (i, j) in enumerate(cells):
                if (code >> b) & 1:
                    adjA[i].add(j)
            out = choose_const(n, adjA, 2)
            if out is None:
                skipped += 1; continue
            M = len(out[0])
            Gs = (M, math.ceil(M / 2), 2 * M) if n == 2 else (M,)
            r = run_const_case(n, adjA, 2, Gs, *out)
            cnt += 1; sizes.add(M)
            if not isinstance(r, dict) or not r["ok"]:
                bad += 1; print("Z1 FAIL", n, code, r)
            else:
                nfac += r["fac"]; hows |= {x[3] for x in r["res"]}
        report.append(f"Z1 (A) c=2 n={n}: {cnt} graphs ({nfac} with C4-factor), padded sizes M in {sorted(sizes)}, solver {sorted(hows)}, "
                      f"OPT <= 2(M-R) with equality iff C4-factor: failures={bad}, no valid pad={skipped} [{time.time()-t0:.0f}s]")
    # n = 4: random graphs, half with a planted C4-factor (n = 3 graphs never have one)
    n = 4; cnt = bad = nfac = 0; sizes = set(); t0 = time.time()
    for rep in range(40):
        adjA = [set(j for j in range(n) if random.random() < 0.4) for _ in range(n)]
        if rep % 2 == 0:
            pa, pb = random.sample(range(n), n), random.sample(range(n), n)
            for blk in range(2):
                for i in pa[2 * blk:2 * blk + 2]:
                    adjA[i] |= set(pb[2 * blk:2 * blk + 2])
        out = choose_const(n, adjA, 2)
        if out is None:
            continue
        M = len(out[0])
        r = run_const_case(n, adjA, 2, (M, math.ceil(M / 2)), *out)
        cnt += 1; sizes.add(M)
        if not isinstance(r, dict) or not r["ok"]:
            bad += 1; print("Z1 FAIL", n, adjA, r)
        else:
            nfac += r["fac"]
    report.append(f"Z1 (A) c=2 n=4: {cnt} random graphs ({nfac} with C4-factor), M in {sorted(sizes)}, G=M and ceil(M/2): failures={bad} [{time.time()-t0:.0f}s]")


def Z2(report):
    cases = [(1, [(0, 0, 0)])]
    alltr = list(itertools.product(range(2), repeat=3))
    cases += [(2, [(0, 0, 0), (1, 1, 1)]), (2, [(0, 0, 0), (1, 1, 0)]), (2, [(0, 1, 0), (1, 0, 1)]),
              (2, [(0, 0, 1), (1, 0, 0)])]
    random.seed(7)
    for _ in range(3):
        cases.append((2, random.sample(alltr, 3)))
    bad = 0; lines = []
    for q, triples in cases:
        L, Rr, adj = build_H(q, triples)
        n = len(L)
        adjA = H_to_pair(L, Rr, adj)
        out = choose_const(n, adjA, 2, maxpad=60)
        t0 = time.time()
        r = run_const_case(n, adjA, 2, (len(out[0]),), *out)
        dm = dm_exists(q, triples)
        ok = isinstance(r, dict) and r["ok"] and r["fac"] == dm
        bad += not ok
        lines.append(f"q={q} m={len(triples)} dm={'yes' if dm else 'no'} n={n} M={r['M']} R={r['R']} OPT={r['res'][0][1]}/{r['res'][0][2]} "
                     f"({'ok' if ok else 'FAIL'}, {time.time()-t0:.0f}s)")
        print("  Z2", lines[-1], flush=True)
    report.append(f"Z2 (A) c=2 G=M on 3DM graphs: {len(cases)} instances, failures={bad}: " + "; ".join(lines))


def Z3(report):
    for c, n, reps in [x for x in ((3, 3, 40), (4, 4, 6)) if str(x[0]) in os.environ.get('Z3CAPS', '34')]:
        cnt = bad = nfac = 0; sizes = set(); t0 = time.time()
        for rep in range(reps):
            if rep < reps // 2:
                adjA = [set(range(n)) for _ in range(n)]  # complete: has a K_{c,c}-factor
                i, j = random.randrange(n), random.randrange(n)
                if rep % 2:
                    adjA[i].discard(j)  # one missing pair: no factor
            else:
                adjA = [set(j for j in range(n) if random.random() < 0.7) for _ in range(n)]
            out = choose_const(n, adjA, c, maxpad=24)
            if out is None:
                continue
            t1 = time.time()
            r = run_const_case(n, adjA, c, (len(out[0]),), *out)
            cnt += 1; sizes.add(len(out[0]))
            print(f"  Z3 c={c} case {rep}: M={len(out[0])} ok={isinstance(r, dict) and r['ok']} fac={isinstance(r, dict) and r['fac']} {time.time()-t1:.0f}s", flush=True)
            if not isinstance(r, dict) or not r["ok"]:
                bad += 1; print("Z3 FAIL", c, adjA, r)
            else:
                nfac += r["fac"]
        report.append(f"Z3 (A) c={c} n={n}: {cnt} graphs ({nfac} with K{c}{c}-factor), M in {sorted(sizes)}: "
                      f"OPT <= c(M-R) with equality iff factor, failures={bad} [{time.time()-t0:.0f}s]")


def run_pm_case(n, adjA, c, G=None):
    npad = pm_pad_size(n, adjA, c)
    W = pad_pm(n, adjA, npad)
    M = len(W)
    if not zero_margins(W):
        return "margins"
    Wint, d = as_int(W)
    alpha = -min(min(r) for r in Wint)
    Cint = [[x + alpha for x in r] for r in Wint]
    if not (min(min(r) for r in Cint) >= 0 and chance([[Fr(x) for x in r] for r in Cint]) == [[Fr(x) for x in r] for r in Wint]):
        return "chance"
    fac = has_kcc_factor(n, adjA, 2)
    G = M if G is None else G
    opt, how = opt_exact(Wint, c, G)
    thr = 2 * M * d
    ok = (opt >= thr) == fac and (fac or Fr(opt, d) <= 2 * M - Fr(1, 2))
    return dict(M=M, npad=npad, fac=fac, opt=Fr(opt, d), ok=ok, how=how, cmax=max(max(r) for r in Cint), d=d)


def Z4(report):
    for c in [x for x in (2, 3) if str(x) in os.environ.get('Z4CAPS', '23')]:
        n = 2
        cells = [(i, j) for i in range(n) for j in range(n)]
        cnt = bad = nfac = skipped = 0; t0 = time.time(); sizes = set(); worst = None
        for code in range(1 << len(cells)):
            adjA = [set() for _ in range(n)]
            for b, (i, j) in enumerate(cells):
                if (code >> b) & 1:
                    adjA[i].add(j)
            if n + pm_pad_size(n, adjA, c) > (20 if c == 2 else 16):
                skipped += 1; continue  # c = 3: the set-partition program is too large for M > 16
            t1 = time.time()
            r = run_pm_case(n, adjA, c)
            cnt += 1
            print(f"  Z4 c={c} graph {code}: {r if not isinstance(r, dict) else {k: r[k] for k in ('M', 'fac', 'opt', 'ok')}} {time.time()-t1:.0f}s", flush=True)
            if not isinstance(r, dict) or not r["ok"]:
                bad += 1; print("Z4 FAIL", c, code, r)
                continue
            nfac += r["fac"]; sizes.add(r["M"])
            if not r["fac"]:
                gap = 2 * r["M"] - r["opt"]
                worst = gap if worst is None else min(worst, gap)
        report.append(f"Z4 (B) +-1 padding c={c} n=2: {cnt} graphs ({nfac} with C4-factor), M in {sorted(sizes)}, G=M: "
                      f"OPT >= 2M iff C4-factor, failures={bad}; smallest gap 2M-OPT on no-graphs = {worst}; skipped (too large) = {skipped} [{time.time()-t0:.0f}s]")
    # n = 3, c = 2: random K23-free graphs (3x3 graphs have no C4-factor since n is odd; checks the no side)
    n = 3; c = 2; cnt = bad = 0; t0 = time.time(); worst = None; sizes = set()
    for _ in range(int(os.environ.get('Z4N3', '4'))):
        while True:
            adjA = [set(j for j in range(n) if random.random() < 0.6) for _ in range(n)]
            L, R, adj = adj_dict(n, adjA)
            if not has_k23(adj, L, R):
                break
        t1 = time.time()
        r = run_pm_case(n, adjA, c)
        cnt += 1
        print(f"  Z4 n=3: {r if not isinstance(r, dict) else {k: r[k] for k in ('M', 'fac', 'opt', 'ok')}} {time.time()-t1:.0f}s", flush=True)
        if not isinstance(r, dict) or not r["ok"]:
            bad += 1; print("Z4 FAIL n=3", adjA, r); continue
        sizes.add(r["M"])
        gap = 2 * r["M"] - r["opt"]
        worst = gap if worst is None else min(worst, gap)
    report.append(f"Z4 (B) +-1 padding c=2 n=3: {cnt} random K23-free graphs, M in {sorted(sizes)}, G=M: OPT < 2M, failures={bad}; "
                  f"smallest gap = {worst} [{time.time()-t0:.0f}s]")


def Z5(report):
    bad = 0; cnt = 0; maxent = 0
    alltr = list(itertools.product(range(3), repeat=3))
    for _ in range(30):
        q = 3
        triples = random.sample(alltr, random.randint(3, 8))
        L, Rr, adj = build_H(q, triples)
        n = len(L)
        adjA = H_to_pair(L, Rr, adj)
        dA, dB = degrees(n, adjA)
        for c in (2, 4):
            npad = 2 * c * n
            R = max(max(dA + dB), 2 * c)
            out = pad_const(n, adjA, c, npad, R)
            cnt += 1
            if out is None:
                bad += 1; print("Z5 FAIL", triples, c); continue
            C, adjS, RR = out
            # margins and entry range (cheap checks on the large table)
            M = len(C)
            ok = all(sum(C[i]) == RR for i in range(M)) and all(sum(C[i][j] for i in range(M)) == RR for j in range(M))
            ok &= all(0 <= C[i][j] <= 1 and ((C[i][j] == 1) == (j in adjS[i])) for i in range(M) for j in range(M))
            D = npad * (npad - c)
            ok &= all((C[i][j] * D).denominator == 1 for i in range(M) for j in range(M))
            maxent = max(maxent, D)
            bad += not ok
    report.append(f"Z5 proof pad sizes (n'=2cn, R=max(maxdeg,2c)) on {cnt} (3DM graph q=3, c in 2,4) cases: all conditions hold, "
                  f"C*n'(n'-c) is an integer table; failures={bad}; largest D=n'(n'-c) = {maxent}")


def graph_from_code(n, code):
    cells = [(i, j) for i in range(n) for j in range(n)]
    adjA = [set() for _ in range(n)]
    for b, (i, j) in enumerate(cells):
        if (code >> b) & 1:
            adjA[i].add(j)
    return adjA


def Z6(report):
    """Exact optima with the symmetry solver opt_agg: first cross-checked against the label-level solvers, then used
    for the cases that are too large for them (caps 3 and 4, n = 3 and 4)."""
    t0 = time.time()
    # (1) cross-check against opt_exact (DP or full set-partition program)
    cmp_n = cmp_bad = 0
    for n, codes in ((2, range(16)), (3, random.sample(range(512), 12))):
        for code in codes:
            adjA = graph_from_code(n, code)
            for c in ((2, 3) if n == 3 else (2,)):
                out = choose_const(n, adjA, c)
                if out is None:
                    continue
                C, adjS, R = out
                M = len(C)
                w = [[C[i][j] - R / M for j in range(M)] for i in range(M)]
                core, x, y, nb, vin, vout = agg_from_W(w, n, c)
                for G in (M, math.ceil(M / 2)):
                    if len(subsets_upto(M, c)) ** 2 > 60000 and 2 * M > 16:
                        continue
                    Wint, d = as_int(w)
                    full, _ = opt_exact(Wint, c, G)
                    agg = opt_agg(core, x, y, c, nb, vin, vout, c, G)
                    cmp_n += 1; cmp_bad += Fr(full, d) != agg
                    if Fr(full, d) != agg:
                        print("Z6 CMP FAIL A", n, code, c, G, Fr(full, d), agg, flush=True)
    for code in range(16):  # construction B, c = 2, n = 2 (M up to 20)
        adjA = graph_from_code(2, code)
        npad = pm_pad_size(2, adjA, 2)
        W = pad_pm(2, adjA, npad)
        M = len(W)
        if M > 12:
            continue
        Wint, d = as_int(W)
        full, _ = opt_exact(Wint, 2, M)
        core, x, y, nb, vin, vout = agg_from_W(W, 2, 2)
        agg = opt_agg(core, x, y, 2, nb, vin, vout, 2, M)
        cmp_n += 1; cmp_bad += Fr(full, d) != agg
        if Fr(full, d) != agg:
            print("Z6 CMP FAIL B", code, Fr(full, d), agg, flush=True)
    report.append(f"Z6a symmetry solver vs label-level solver: {cmp_n} instances (constructions A and B, caps 2 and 3), "
                  f"disagreements={cmp_bad} [{time.time()-t0:.0f}s]")
    print(report[-1], flush=True)

    # (2) construction B (+-1 padding, smallest pad meeting the proof's condition c*n*b < n') for caps 2, 3, 4
    def pm_case(n, adjA, c):
        npad = pm_pad_size(n, adjA, c)
        W = pad_pm(n, adjA, npad)
        assert zero_margins(W)
        M = len(W)
        core, x, y, nb, vin, vout = agg_from_W(W, n, 2)
        opt = opt_agg(core, x, y, 2, nb, vin, vout, c, M)
        fac = has_kcc_factor(n, adjA, 2)
        ok = (opt >= 2 * M) == fac and (fac or opt <= 2 * M - Fr(1, 2))
        return M, fac, opt, ok
    for c in (2, 3, 4):
        t1 = time.time(); cnt = bad = nfac = 0; sizes = set(); worst = None
        cases = [(2, graph_from_code(2, code)) for code in range(16)]
        rnd = random.Random(10 + c)
        for rep in range(30):
            n = 3 if rep < 10 else 4
            while True:
                adjA = [set(j for j in range(n) if rnd.random() < 0.5) for _ in range(n)]
                if n == 4 and rep % 2 == 0:
                    pa, pb = rnd.sample(range(n), n), rnd.sample(range(n), n)
                    for blk in range(2):
                        for i in pa[2 * blk:2 * blk + 2]:
                            adjA[i] |= set(pb[2 * blk:2 * blk + 2])
                L, R, adj = adj_dict(n, adjA)
                if not has_k23(adj, L, R):
                    break
            cases.append((n, adjA))
        for n, adjA in cases:
            M, fac, opt, ok = pm_case(n, adjA, c)
            cnt += 1; nfac += fac; sizes.add(M)
            if not ok:
                bad += 1; print("Z6 FAIL B", c, n, adjA, M, fac, opt, flush=True)
            if not fac:
                g = 2 * M - opt
                worst = g if worst is None else min(worst, g)
        report.append(f"Z6b (B) +-1 padding, cap c={c}, smallest even pad n' with c*n*b < n' (the condition the proof uses; the proof's n'=2cn^2+2 is larger), G=M: {cnt} graphs (16 with n=2, 10 K23-free n=3, "
                      f"20 K23-free n=4; {nfac} with C4-factor), M from {min(sizes)} to {max(sizes)}: OPT >= 2M iff C4-factor, "
                      f"failures={bad}; smallest gap 2M-OPT on no-graphs = {worst} [{time.time()-t1:.0f}s]")
        print(report[-1], flush=True)

    # (3) construction A, caps 3 and 4, larger pads
    for c, n, reps in ((3, 3, 30), (4, 4, 20)):
        t1 = time.time(); cnt = bad = nfac = 0; sizes = set()
        rnd = random.Random(20 + c)
        for rep in range(reps):
            if rep % 3 == 0:
                adjA = [set(range(n)) for _ in range(n)]
                if rep % 2:
                    adjA[rnd.randrange(n)].discard(rnd.randrange(n))
            else:
                adjA = [set(j for j in range(n) if rnd.random() < 0.8) for _ in range(n)]
            out = choose_const(n, adjA, c, maxpad=40)
            if out is None:
                continue
            C, adjS, R = out
            assert check_const(C, adjS, R)
            M = len(C)
            w = chance(C)
            assert zero_margins(w)
            core, x, y, nb, vin, vout = agg_from_W(w, n, c)
            opt = opt_agg(core, x, y, c, nb, vin, vout, c, M)
            fac = has_kcc_factor(n, adjA, c)
            ok = opt <= c * (M - R) and ((opt == c * (M - R)) == fac)
            cnt += 1; nfac += fac; sizes.add(M); bad += not ok
            if not ok:
                print("Z6 FAIL A", c, adjA, opt, c * (M - R), fac, flush=True)
        report.append(f"Z6c (A) equal margins, cap c={c}, n={n}, G=M: {cnt} graphs ({nfac} with K{c}{c}-factor), M in {sorted(sizes)}: "
                      f"OPT <= c(M-R) with equality iff factor, failures={bad} [{time.time()-t1:.0f}s]")
        print(report[-1], flush=True)


if __name__ == "__main__":
    which = sys.argv[1:] or ["Z0", "Z1", "Z2", "Z3", "Z4", "Z5", "Z6"]
    report = []
    for name in which:
        t0 = time.time()
        globals()[name](report)
        print(report[-1], f"[{time.time()-t0:.0f}s]", flush=True)
    print("\n==== SUMMARY ====")
    for r in report:
        print(r)
