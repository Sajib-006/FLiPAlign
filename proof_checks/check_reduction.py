"""Computer check of the reduction 3DM -> joint label pairing (JLP) with three sections (2026-10-17).
3DM instance: X, Y, Z of size m (elements 0..m-1), triples T subset of X x Y x Z. Graph (tripartite, parts of size N = m+|T|):
  V1 = X + {p_t}, V2 = Y + {q_t}, V3 = Z + {r_t};  for each triple t=(x,y,z):
  private triangle p_t q_t r_t;  edges x-q_t, x-r_t  (triangle A_t = x q_t r_t);  edges p_t-y, p_t-z, y-z  (triangle B_t = p_t y z).
Geometric realisation: three arenas, each a strip of unit squares, one square per edge of that pair of parts;
  C_su(k, l) = overlap area of label k of section s with label l of section u.
Checks: (1) the triangles of the graph are exactly P_t, A_t, B_t; (2) 3DM yes <=> perfect triangle partition <=>
max over bijections of Phi equals 3N (brute force for small N); (3) C computed from the squares equals the edge indicators."""
import itertools, random
import numpy as np

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

def squares(N, E12, E13, E23):
    """arena per pair: squares in a row; section s and u both own every square of arena (s,u) with labels (k, l)"""
    cells = {}                                   # (arena, index) -> (label in first section, label in second section)
    for name, E in (("12", E12), ("13", E13), ("23", E23)):
        for j, (k, l) in enumerate(sorted(E)): cells[(name, j)] = (k, l)
    C = {n: np.zeros((N, N)) for n in ("12", "13", "23")}
    for (name, j), (k, l) in cells.items(): C[name][k, l] += 1.0   # unit square: overlap area 1 for the two labels it carries
    return C


def layout(N, E12, E13, E23):
    """actual squares: returns, per section, a dict square -> label; checks the text's layout"""
    E = {"12": sorted(E12), "13": sorted(E13), "23": sorted(E23)}
    D = 3 * max(len(v) for v in E.values()) + 10
    sec = {1: {}, 2: {}, 3: {}}
    for j, (k, l) in enumerate(E["12"]): sec[1][(j, 0)] = k; sec[2][(j, 0)] = l
    for j, (k, l) in enumerate(E["13"]): sec[1][(D + j, 0)] = k; sec[3][(D + j, 0)] = l
    for j, (k, l) in enumerate(E["23"]): sec[2][(j, D)] = k; sec[3][(j, D)] = l
    for x in range(len(E["12"]), D): sec[1][(x, 0)] = 0                       # corridor of section 1
    for y in range(0, D + 1): sec[2][(-1, y)] = 0                              # corridor of section 2
    for y in range(1, D + 1): sec[3][(D, y)] = 0                               # column of section 3
    for x in range(len(E["23"]), D): sec[3][(x, D)] = 0                        # row of section 3
    return sec

def connected(cells):
    cells = set(cells); start = next(iter(cells)); seen = {start}; stack = [start]
    while stack:
        x, y = stack.pop()
        for q in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if q in cells and q not in seen: seen.add(q); stack.append(q)
    return len(seen) == len(cells)

def check_layout(N, E12, E13, E23):
    sec = layout(N, E12, E13, E23)
    allc = list(sec[1]) + list(sec[2]) + list(sec[3])
    from collections import Counter
    cnt = Counter(allc); assert max(cnt.values()) <= 2, "square in three sections"
    for s in (1, 2, 3): assert connected(sec[s]), ("section not connected", s)
    C = {n: np.zeros((N, N)) for n in ("12", "13", "23")}
    for name, (s, u) in (("12", (1, 2)), ("13", (1, 3)), ("23", (2, 3))):
        for c in set(sec[s]) & set(sec[u]): C[name][sec[s][c], sec[u][c]] += 1
    return C

def best_phi(N, C):
    best = -1
    perms = list(itertools.permutations(range(N)))
    for s2 in perms:                              # universe element g carries label g in section 1, s2[g] in section 2, s3[g] in section 3
        base = sum(C["12"][g, s2[g]] for g in range(N))
        for s3 in perms:
            v = base + sum(C["13"][g, s3[g]] + C["23"][s2[g], s3[g]] for g in range(N))
            if v > best: best = v
    return best

random.seed(7); checked = brute = 0; yes = no = 0
for trial in range(400):
    m = random.choice([2, 3, 4]); allT = [(x, y, z) for x in range(m) for y in range(m) for z in range(m)]
    T = random.sample(allT, random.randint(m, min(len(allT), {2: 4, 3: 7, 4: 9}[m])))
    N, E12, E13, E23 = build(m, T); tris = triangles(N, E12, E13, E23)
    exp = set()
    for i, (x, y, z) in enumerate(T):
        p = q = r = m + i; exp |= {(p, q, r), (x, q, r), (p, y, z)}
    assert tris == exp, ("triangle set differs", m, T)                                   # (1)
    d, tp = has_3dm(m, T), has_triangle_partition(N, tris); assert d == tp, ("3DM vs triangle partition", m, T)   # (2a)
    C = squares(N, E12, E13, E23)
    C2 = check_layout(N, E12, E13, E23)
    for name in C: assert (C[name] == C2[name]).all(), ('layout tables differ', name)      # real squares, connectivity, overlaps
    for name, E in (("12", E12), ("13", E13), ("23", E23)):                                # (3)
        assert all(C[name][k, l] == ((k, l) in E) for k in range(N) for l in range(N))
    if N <= 6:                                                                              # (2b) brute force over all bijections
        phi = best_phi(N, C); assert (phi == 3 * N) == d, ("Phi", m, T, phi); brute += 1
    checked += 1; yes += d; no += (not d)
print("instances", checked, "yes", yes, "no", no, "brute-force Phi checks", brute, "ALL CHECKS PASSED")
