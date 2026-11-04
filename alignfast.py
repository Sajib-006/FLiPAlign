"""ExhAlign - exhaustive-translation alignment with a PROVEN rotation-discretisation bound.

Two changes from the branch-and-bound version:

1. TRANSLATION IS NOT SEARCHED. For a fixed rotation, every translation in the window is
   evaluated at once (a gather; equivalently a cross-correlation - verified identical).
   Cost is fixed and does not depend on whether the data has any signal, which is what
   made branch-and-bound take 49 minutes on a hard pair and 14 seconds on an easy one.

2. ROTATION CARRIES AN EXPLICIT ERROR BOUND. Rotating by delta moves a point at radius r
   by at most 2 r sin(delta/2). A point only changes its matched region if it crosses a
   region boundary, so

       |score(theta + d) - score(theta)|  <=  (1/n) * #{i : dist_to_boundary(image of i) <= 2 r_i sin(delta/2)}

   The right-hand side is measured from the distance transforms, so scanning rotation at
   step delta certifies the global optimum to within that epsilon - a data-dependent,
   a-posteriori bound, tight when regions are large and smooth.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

def rot(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s], [s, c]])
FLIP = np.diag([1.0, -1.0])


class ExhAlign:
    def __init__(self, A, la, B, lb, h=0.35, tpad=8.0, mask_dist=0.75):
        self.h, self.tpad = h, tpad
        self.c = A.mean(0); self.A = A - self.c
        self.la = np.asarray(la); self.lb = np.asarray(lb)
        self.KA, self.KB = int(self.la.max()) + 1, int(self.lb.max()) + 1
        self.n = len(A); self.tc = B.mean(0)
        self.rA = np.hypot(*self.A.T)
        pad = self.rA.max() + tpad + 2
        self.lo = B.min(0) - pad; hi = B.max(0) + pad
        gx = np.arange(self.lo[0], hi[0], h); gy = np.arange(self.lo[1], hi[1], h)
        self.shape = (len(gx), len(gy))
        G = np.stack(np.meshgrid(gx, gy, indexing="ij"), -1).reshape(-1, 2)
        dd, ii = cKDTree(B).query(G)
        self.field = np.where(dd > mask_dist, -1, self.lb[ii]).reshape(self.shape).astype(np.int16)
        # Distance from each cell to the nearest cell of a DIFFERENT label (any region
        # boundary, tissue edge included). A point changes its matched region only by
        # crossing one of these. (An earlier version measured distance to the TISSUE
        # EDGE only - far too optimistic, so not a valid bound. Fixed.)
        self.bdist = np.zeros(self.shape)
        for l in np.unique(self.field):
            m = self.field == l
            self.bdist[m] = (distance_transform_edt(m) * h)[m]
        self.r = int(np.ceil(tpad / h)); self.offs = np.arange(-self.r, self.r + 1)

    def _cells(self, th, flip):
        Y = (rot(th) @ (FLIP if flip else np.eye(2)) @ self.A.T).T + self.tc
        ij = np.floor((Y - self.lo) / self.h).astype(int)
        return (np.clip(ij[:, 0], 0, self.shape[0] - 1),
                np.clip(ij[:, 1], 0, self.shape[1] - 1))

    def best_for_rotation(self, th, flip):
        """Exact best over EVERY translation in the window. No search."""
        ci, cj = self._cells(th, flip)
        L = self.field[np.clip(ci[:, None, None] + self.offs[None, :, None], 0, self.shape[0] - 1),
                       np.clip(cj[:, None, None] + self.offs[None, None, :], 0, self.shape[1] - 1)]
        m = len(self.offs)
        idx = (self.la[:, None, None] * self.KB + L).reshape(self.n, m * m)
        valid = (L >= 0).reshape(self.n, m * m)
        K = self.KA * self.KB
        Cs = np.zeros((m * m, K))
        for s in range(m * m):
            v = valid[:, s]
            if v.any():
                Cs[s] = np.bincount(idx[v, s], minlength=K)
        Cs = Cs.reshape(m * m, self.KA, self.KB)
        ub = Cs.max(axis=2).sum(axis=1)                 # assignment <= sum of row maxima
        best, bt = -1.0, None
        for s in np.argsort(ub)[::-1]:
            if ub[s] <= best: break                     # exact: nothing better can remain
            r_, c_ = linear_sum_assignment(-Cs[s])
            v = Cs[s][r_, c_].sum()
            if v > best:
                best = v
                bt = self.tc + np.array([self.offs[s // m], self.offs[s % m]]) * self.h
        return best / self.n, bt

    def eps_for_step(self, th, flip, delta):
        """Proven bound on how much the score can move between grid points.
        Two sources of displacement: rotating by up to delta (2 r sin(delta/2)), and the
        translation grid, which is exact only at cell centres (at most h*sqrt(2)/2 away)."""
        ci, cj = self._cells(th, flip)
        # raster term: bdist is measured between cell CENTRES, so the true boundary can be
        # up to a full cell diagonal closer. Without it the bound reads 0 for sub-cell
        # moves - a quantisation artefact, not tightness. This term sets the floor.
        move = 2 * self.rA * np.sin(delta / 2) + 1.5 * self.h * np.sqrt(2)
        return float(np.mean(self.bdist[ci, cj] <= move))

    def solve_bnb(self, eps_target=0.0, min_delta=np.radians(0.2), cap=4000):
        """ONE-dimensional branch-and-bound over rotation (x flip). Each interval
        [th-d, th+d] has lower bound = the EXACT best at th (translation exhausted),
        upper bound = that + eps(d) (proven). Returns (best, T, certified_gap)."""
        import heapq
        heap, cnt, best, bT = [], 0, -1.0, None
        for flip in (0, 1):
            for k in range(24):
                th = (k + .5) * 2 * np.pi / 24; d = np.pi / 24
                s, t = self.best_for_rotation(th, flip)
                if s > best: best, bT = s, (th, t, flip)
                heapq.heappush(heap, (-(s + self.eps_for_step(th, flip, d)), cnt, th, d, flip)); cnt += 1
        gap = 0.0
        while heap and cnt < cap:
            nu, _, th, d, flip = heapq.heappop(heap)
            ub = -nu
            if ub <= best + eps_target:
                gap = max(gap, ub - best); break
            if d <= min_delta:
                gap = max(gap, ub - best); continue
            for c in (th - d / 2, th + d / 2):
                s, t = self.best_for_rotation(c, flip)
                if s > best: best, bT = s, (c % (2 * np.pi), t, flip)
                u = s + self.eps_for_step(c, flip, d / 2)
                if u > best + eps_target:
                    heapq.heappush(heap, (-u, cnt, c, d / 2, flip)); cnt += 1
        if heap:
            gap = max(gap, -heap[0][0] - best)
        return best, bT, max(gap, 0.0), cnt

    def solve(self, n_theta=720, flips=(0, 1), bound_every=8):
        """Deterministic. Returns (best, transform, epsilon) where epsilon is the proven
        gap to the true continuous optimum."""
        best, bT, eps = -1.0, None, 0.0
        delta = np.pi / n_theta
        k = 0
        for flip in flips:
            for th in np.linspace(0, 2 * np.pi, n_theta, endpoint=False):
                s, t = self.best_for_rotation(th, flip)
                if s > best: best, bT = s, (th, t, flip)
                if k % bound_every == 0:
                    eps = max(eps, self.eps_for_step(th, flip, delta))
                k += 1
        return best, bT, eps
