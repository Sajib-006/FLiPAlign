"""CertAlign - certified cross-modal alignment of tissue sections.

Score(T) = max over label bijections pi of #{i : pi(domA_i) == domB(T a_i)} / n
         = n - partition distance (Gusfield 2002), solved by Hungarian.

Branch-and-bound over SE(2) x {flip}. Bound for a transform box: every image point lies
within r_i = 2|a_i| sin(dth/2) + sqrt(2) dt of its box-centre image, so let point i match
ANY label reachable within that disk. Reachable counts dominate the true contingency table
entrywise and max-weight assignment is monotone in its entries, so the bound is valid.
(Mutual information has no such monotonicity - hence no certified MI registration.)
"""
import os, sys  # noqa: E401
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import repo_paths  # noqa: E402,F401  (puts the code folders on the import path)
import numpy as np, heapq
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import distance_transform_edt, minimum_filter
from scipy.spatial import cKDTree

def rot(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s], [s, c]])

FLIP = np.diag([1.0, -1.0])


class CertAlign:
    def __init__(self, A, la, B, lb, h=0.25, tpad=25.0, mask_dist=0.75, arc_samples=6):
        """A,B: (n,2) coordinates in units of one spot spacing. la,lb: integer domain labels.
        The two label alphabets are treated as unrelated."""
        self.c = A.mean(0)
        self.A = A - self.c
        self.la = np.asarray(la)
        self.KA, self.KB = int(self.la.max()) + 1, int(np.max(lb)) + 1
        self.h, self.tpad, self.n = h, tpad, len(A)
        self.tc = B.mean(0)
        self.rA = np.hypot(self.A[:, 0], self.A[:, 1])
        self.onehot = np.zeros((self.n, self.KA))
        self.onehot[np.arange(self.n), self.la] = 1.0
        pad = self.rA.max() + tpad + 3
        lo, hi = B.min(0) - pad, B.max(0) + pad
        self.lo = lo
        gx, gy = np.arange(lo[0], hi[0], h), np.arange(lo[1], hi[1], h)
        self.shape = (len(gx), len(gy))
        G = np.stack(np.meshgrid(gx, gy, indexing="ij"), -1).reshape(-1, 2)
        dd, ii = cKDTree(B).query(G)
        f = np.asarray(lb)[ii].astype(int)
        f[dd > mask_dist] = -1                      # -1 = off tissue
        self.field = f.reshape(self.shape)
        self.dist = np.stack([distance_transform_edt(self.field != l) * h
                              for l in range(self.KB)]).astype(np.float32)
        self.n_eval = 0
        self.arc_samples = arc_samples
        self._mf = {}

    # ---- geometry -----------------------------------------------------------------
    def image(self, th, t, flip):
        return (rot(th) @ (FLIP if flip else np.eye(2)) @ self.A.T).T + t

    def _cell(self, Y):
        ij = np.clip(np.floor((Y - self.lo) / self.h).astype(int), 0,
                     np.array(self.shape) - 1)
        return ij[:, 0], ij[:, 1]

    # ---- score and bound ----------------------------------------------------------
    def table(self, th, t, flip):
        i, j = self._cell(self.image(th, t, flip))
        l = self.field[i, j]
        v = l >= 0
        if not v.any():
            return np.zeros((self.KA, self.KB))
        return np.bincount(self.la[v] * self.KB + l[v],
                           minlength=self.KA * self.KB).reshape(self.KA, self.KB)

    def score_points(self, Y):
        """score an explicit set of already-transformed coordinates (used by non-rigid refinement)"""
        self.n_eval += 1
        i, j = self._cell(Y)
        l = self.field[i, j]
        v = l >= 0
        if not v.any():
            return 0.0
        N = np.bincount(self.la[v] * self.KB + l[v],
                        minlength=self.KA * self.KB).reshape(self.KA, self.KB)
        r, c = linear_sum_assignment(-N)
        return N[r, c].sum() / self.n

    def score(self, th, t, flip, return_pi=False):
        self.n_eval += 1
        N = self.table(th, t, flip)
        r, c = linear_sum_assignment(-N)
        s = N[r, c].sum() / self.n
        return (s, dict(zip(r, c))) if return_pi else s

    # -- K2: tighter bound -----------------------------------------------------------
    # Slack 1: covering the translation SQUARE by a disk of radius sqrt(2)dt.
    #   Fix, exactly: min-filter each label's distance transform with that square, so
    #   minfilt(x) = min over the square of dist_l, which is the exact translation reach.
    # Slack 2: a rotation box sweeps a long thin ARC, and one disk over-covers it badly.
    #   Fix: sample the arc at M points; the radius term then falls by roughly M.
    def _minfilt(self, dt):
        """distance transforms eroded by the translation square (cached per level)"""
        lvl = max(0, int(np.ceil(dt / self.h)))                 # exact level, no over-covering
        if lvl not in self._mf:
            self._mf[lvl] = (self.dist if lvl == 0 else
                             minimum_filter(self.dist, size=(1, 2 * lvl + 1, 2 * lvl + 1)))
        return self._mf[lvl], lvl * self.h

    def upper(self, th, dth, t, dt, flip, arc=None, tight=True):
        if not tight:                                           # original, looser bound
            i, j = self._cell(self.image(th, t, flip))
            rad = 2 * self.rA * np.sin(min(dth, np.pi) / 2) + np.sqrt(2) * dt + self.h * 0.71
            reach = self.dist[:, i, j].T <= rad[:, None]
            N = self.onehot.T @ reach
            r, c = linear_sum_assignment(-N)
            return N[r, c].sum() / self.n
        M = arc or self.arc_samples
        dth = min(dth, np.pi)
        kk = int(np.ceil(dt / self.h))
        if kk <= 4:                                         # translation exact via min-filter
            D, _ = self._minfilt(dt); tterm = 0.0
        else:                                               # large box: disk term is cheaper
            D, tterm = self.dist, np.sqrt(2) * dt
        rad = 2 * self.rA * np.sin(dth / (2 * M)) + tterm + self.h * 0.71
        reach = np.zeros((self.n, self.KB), bool)
        offs = (np.arange(M) + 0.5) * (2 * dth / M) - dth       # arc sample angles
        for o in offs:
            i, j = self._cell(self.image(th + o, t, flip))
            np.logical_or(reach, D[:, i, j].T <= rad[:, None], out=reach)
        N = self.onehot.T @ reach
        r, c = linear_sum_assignment(-N)
        return N[r, c].sum() / self.n

    # ---- branch and bound ---------------------------------------------------------
    def _seed(self, thr=-1.0, nth=16):
        heap, cnt = [], 0
        for flip in (0, 1):
            for k in range(nth):
                th, dth = (k + .5) * np.pi / nth, np.pi / nth
                u = self.upper(th, dth, self.tc, self.tpad, flip)
                if u > thr:
                    heapq.heappush(heap, (-u, 0, cnt, th, dth, self.tc.copy(), self.tpad, flip))
                    cnt += 1
        return heap, cnt

    def _kids(self, th, dth, t, dt, min_dth):
        if dth * self.rA.max() >= dt * np.sqrt(2) and dth > min_dth:
            return [(th - dth / 2, dth / 2, t, dt), (th + dth / 2, dth / 2, t, dt)]
        return [(th, dth, t + np.array([sx, sy]) * dt / 2, dt / 2)
                for sx in (-1, 1) for sy in (-1, 1)]

    def warm_grid(self, d_ang=4.0, d_t=1.5):
        """Deterministic coarse scan for an incumbent. Preferred over warm(): random
        restarts can return a terrible incumbent (seen: 0.115 when 0.406 was reachable),
        and a bad incumbent cripples pruning, which is how the 211-degree case failed."""
        best, bT = -1.0, None
        ts = np.arange(-self.tpad, self.tpad + 1e-9, d_t)
        for f in (0, 1):
            for a in np.radians(np.arange(0, 360, d_ang)):
                for dx in ts:
                    for dy in ts:
                        t = self.tc + np.array([dx, dy])
                        s = self.score(a, t, f)
                        if s > best:
                            best, bT = s, (a, t, f)
        return best, bT

    def warm(self, n_restart=8, seed=0):
        """Cheap incumbent before branch-and-bound. Pruning compares bounds against the
        incumbent's ABSOLUTE level, so a low-scoring problem prunes slowly from a cold
        start even when the correct answer wins by a wide margin."""
        from scipy.optimize import minimize
        rg = np.random.default_rng(seed); best, bT = -1.0, None
        for _ in range(n_restart):
            f = int(rg.integers(2))
            x0 = np.array([rg.uniform(0, 2 * np.pi), *(self.tc + rg.uniform(-4, 4, 2))])
            sim = np.vstack([x0, x0 + [.6, 0, 0], x0 + [0, 2, 0], x0 + [0, 0, 2]])
            o = minimize(lambda x: -self.score(x[0], x[1:], f), x0, method="Nelder-Mead",
                         options={"initial_simplex": sim, "maxiter": 300,
                                  "xatol": 1e-3, "fatol": 1e-4})
            if -o.fun > best:
                best, bT = -o.fun, (o.x[0] % (2 * np.pi), np.asarray(o.x[1:]), f)
        return best, bT

    def solve(self, min_dth=np.radians(0.5), min_dt=0.15, cap=600_000, warm_start=False,
              grid_warm=True):
        """Global optimum over SE(2) x {flip}.

        Returns (best, bestT, boxes, certified). certified is False when the box budget
        ran out, in which case the answer is only the best found so far - NOT a proof.
        Reporting a truncated run as certified would be the worst possible bug in a
        method whose whole claim is certification."""
        heap, cnt = self._seed()
        best, bestT = -1.0, None
        if grid_warm:
            best, bestT = self.warm_grid()
        elif warm_start:
            best, bestT = self.warm()
        certified = True
        while heap:
            if cnt >= cap:
                certified = False                       # budget ran out: NOT a proof
                break
            nu, dep, _, th, dth, t, dt, flip = heapq.heappop(heap)
            if -nu <= best + 1e-12:
                break                                   # bound proves nothing better remains
            s = self.score(th, t, flip)
            if s > best:
                best, bestT = s, (th % (2 * np.pi), t.copy(), int(flip))
            if dth <= min_dth and dt <= min_dt:
                continue
            for a, b, c, d in self._kids(th, dth, t, dt, min_dth):
                u = self.upper(a, b, c, d, flip)
                if u > best + 1e-12:
                    heapq.heappush(heap, (-u, dep - 1, cnt, a, b, c, d, flip))
                    cnt += 1
        return best, bestT, cnt, certified

    def cover(self, best, eps, min_dth=np.radians(0.5), min_dt=0.2, cap=600_000):
        """Every eps-optimal transform lies in one of the returned boxes.
        Collected by the BOUND, not the centre score - a box pruned by the bound
        provably contains nothing eps-optimal."""
        thr = best - eps
        heap, cnt = self._seed(thr)
        leaves = []
        while heap and cnt < cap:
            nu, dep, _, th, dth, t, dt, flip = heapq.heappop(heap)
            if -nu <= thr:
                continue
            if dth <= min_dth and dt <= min_dt:
                leaves.append((-nu, self.score(th, t, flip), th % (2 * np.pi), t.copy(), int(flip)))
                continue
            for a, b, c, d in self._kids(th, dth, t, dt, min_dth):
                u = self.upper(a, b, c, d, flip)
                if u > thr:
                    heapq.heappush(heap, (-u, dep - 1, cnt, a, b, c, d, flip))
                    cnt += 1
        return leaves, cnt


def modes(leaves, ang_tol=30.0):
    """Collapse a cover into distinct alignment modes, best score first."""
    out = []
    for ub, s, th, t, f in sorted(leaves, key=lambda z: -z[1]):
        d = np.degrees(th)
        if not any(ff == f and (abs((d - g) % 360) < ang_tol or abs((d - g) % 360) > 360 - ang_tol)
                   for g, ff, _, _ in out):
            out.append((d, f, s, t))
    return out
