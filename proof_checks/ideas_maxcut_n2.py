"""Hardness with only N = 2 labels per section (K part of the input): Max-Cut -> JLP, with real squares.
Graph G on n vertices. Section i = vertical bar (columns 4i, 4i+1) plus one horizontal tooth (rows h_e, h_e+1) per edge
e = (i, j), i < j, running right to section j's bar and then up inside it for W rows.
 - where tooth e crosses the bar of a section k (i < k < j): a 2x2 block, labels (row, column) give the pairs
   (0,0),(0,1),(1,0),(1,1): all-ones table, adds the constant 2 whatever the pairing;
 - inside bar j: 2W squares with pairs (a, 1-a): table W*[[0,1],[1,0]], adds 2W if the two sections are paired differently.
So Phi(x) = 2W * cut(x) + 2 * (number of crossing blocks), for every choice x of the two bijections.
Checks: no square in three sections; each section connected; Phi from the squares equals that formula for all 2^n choices."""
import itertools, random
from collections import Counter, defaultdict
import numpy as np
from ideas_common import connected


def layout(n, edges, W):
    sec = {i: {} for i in range(n)}
    gap = W + 4
    top = 3 + len(edges) * gap + W + 2
    for i in range(n):
        for x in (4 * i, 4 * i + 1):
            for y in range(top): sec[i][(x, y)] = 0
    crossings = 0
    for t, (i, j) in enumerate(edges):
        h = 3 + t * gap
        for x in range(4 * i + 2, 4 * j):
            for r in (0, 1):
                k = x // 4
                if x % 4 in (0, 1) and i < k < j:                    # crossing the bar of k: pairs by (row, column)
                    sec[i][(x, h + r)] = r; sec[k][(x, h + r)] = x % 4
                else:
                    sec[i][(x, h + r)] = 0
            if x % 4 == 0 and i < x // 4 < j: crossings += 1
        for x in (4 * j, 4 * j + 1):                                 # gadget block inside bar j
            for y in range(h, h + W):
                a = (x - 4 * j + y) % 2
                sec[j][(x, y)] = a; sec[i][(x, y)] = 1 - a
    return sec, crossings


def tables(sec, n):
    C = {}
    for s in range(n):
        for u in range(s + 1, n):
            sh = set(sec[s]) & set(sec[u])
            if sh:
                M = np.zeros((2, 2))
                for c in sh: M[sec[s][c], sec[u][c]] += 1
                C[(s, u)] = M
    return C


random.seed(3); done = 0
for trial in range(60):
    n = random.randint(3, 7); W = random.choice([2, 4, 6])
    edges = [e for e in itertools.combinations(range(n), 2) if random.random() < 0.5]
    if not edges: continue
    sec, cr = layout(n, edges, W)
    cnt = Counter(c for s in sec for c in sec[s]); assert max(cnt.values()) <= 2
    for s in sec: assert connected(sec[s])
    C = tables(sec, n)
    perm = {0: np.array([0, 1]), 1: np.array([1, 0])}
    for x in itertools.product((0, 1), repeat=n):
        phi = sum(C[(s, u)][perm[x[s]], perm[x[u]]].sum() for (s, u) in C)
        cut = sum(x[i] != x[j] for (i, j) in edges)
        assert phi == 2 * W * cut + 2 * cr, (n, edges, W, x, phi, cut, cr)
    done += 1
print("Max-Cut gadget instances", done, ": geometry ok and Phi = 2W*cut + 2*crossings for every pairing. PASSED")
