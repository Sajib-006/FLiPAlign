"""SmoothAlign - certified alignment on SMOOTHED label maps (CertAlign v2).

Why: at 166k Xenium cells the old certificate proved nothing (gap 0.8-1.0), even with
tissue regions as input (gap 0.92-0.99, results_domains.json). The old bound counts every
cell within reach of a label edge as a possible full label change, plus a fixed grid
slack of 1.5*h*sqrt(2). Both stay large no matter how fine the rotation search is.

Here slice B becomes a set of smooth label-COMPOSITION maps:

    G_l(x) = S_l(x) / (S(x) + c),   S_l = Gaussian(sigma) smoothing of B's label-l cells

so G_l(x) is roughly the share of label l around x. The score of a transform T is

    score(T) = max over label matchings pi of (1/n) sum_i  Gb_{pi(la_i)}( T a_i )

where Gb is the bilinear interpolation of G on a grid of spacing h. That is the exact
objective certified here: "certified for composition maps smoothed at sigma, grid h".

Bound. Moving every point i by at most m_i changes any fixed-matching sum by at most
(1/n) sum_i m_i * (largest gradient of any G_l within m_i of the point). A max over
matchings moves by no more than the largest change of any single matching, so the same
number bounds the score. A rotation box of half-width d plus the continuous translation
between grid shifts moves point i by at most

    m_i = 2 |a_i| sin(d/2) + h/sqrt(2)

For a fixed rotation, every grid translation in the window is scored at once by FFT
cross-correlation (score and bound alike), so cost does not depend on the cell count
after the histograms are built.

Search: 1-D branch-and-bound over rotation (x flip). A rotation interval is discarded when
its upper bound (best over translations of exact score + bound) is below the best score
found. Returns the best transform and the proven gap.
"""
import heapq, time
import numpy as np
from scipy import fft as sfft
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.optimize import linear_sum_assignment


def rot(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s], [s, c]])


FLIP = np.diag([1.0, -1.0])


class SmoothAlign:
    def __init__(self, A, la, B, lb, h=4.0, sigma=10.0, tpad=30.0, floor=0.05, workers=-1,
                 contrast=False, pair_bound=False, center=None, min_overlap=None):
        """A, B: (n,2) coordinates in cell-spacing units. la, lb: integer labels,
        treated as unrelated alphabets. tpad: translation window half-width around B's centre.
        contrast=True: subtract each label's slide-wide share from its map (times the tissue
        mask), so a random placement scores ~0 and only EXCESS co-location counts. Labels
        may then stay unmatched (score 0), since a forced negative match only hurts."""
        self.h, self.sigma, self.tpad, self.workers = float(h), float(sigma), float(tpad), workers
        self.A = np.asarray(A, float) - np.mean(A, 0)
        self.la = np.asarray(la).astype(int)
        self.lb = np.asarray(lb).astype(int)
        self.KA, self.KB = int(self.la.max()) + 1, int(self.lb.max()) + 1
        self.n = len(self.A)
        # center: optional offset of the shift window from B's centre (fine stage: the window is
        # put around a coarse answer). Shifts are always reported relative to B's centre.
        self.t0 = np.zeros(2) if center is None else np.asarray(center, float)
        self.tc = np.mean(B, 0) + self.t0
        self.rA = np.hypot(*self.A.T)
        self.rmax = float(self.rA.max())
        self.r = int(np.ceil(tpad / h))                       # window: shifts -r..r nodes
        half = self.rmax + tpad + 4 * sigma + 4 * h
        self.lo = self.tc - half
        nside = int(np.ceil(2 * half / h)) + 2
        self.shape = (nside, nside)

        # --- composition maps of B on the node grid ---
        S = np.zeros((self.KB,) + self.shape)
        ij = np.floor((np.asarray(B, float) - self.lo) / h + 0.5).astype(int)
        keep = (ij >= 0).all(1) & (ij[:, 0] < nside) & (ij[:, 1] < nside)
        for l in range(self.KB):
            m = keep & (self.lb == l)
            np.add.at(S[l], (ij[m, 0], ij[m, 1]), 1.0)
        S = gaussian_filter(S, sigma=(0, sigma / h, sigma / h), mode="constant")
        tot = S.sum(0)
        inside = tot > 0.2 * tot.max()
        self.c = floor * float(np.median(tot[inside]))
        self.G = S / (tot + self.c)                           # (KB, nx, ny), values in [0,1)
        self.contrast = bool(contrast)
        if self.contrast:
            share = np.bincount(self.lb, minlength=self.KB) / len(self.lb)
            mask = tot / (tot + self.c)                       # ~1 inside tissue, ~0 outside
            self.G = self.G - share[:, None, None] * mask[None]

        # --- per-cell gradient bound of the bilinear map, max over labels ---
        G = self.G
        dx = np.abs(G[:, 1:, :] - G[:, :-1, :])               # (KB, nx-1, ny)
        dy = np.abs(G[:, :, 1:] - G[:, :, :-1])               # (KB, nx, ny-1)
        gx = np.maximum(dx[:, :, :-1], dx[:, :, 1:])          # (KB, nx-1, ny-1)
        gy = np.maximum(dy[:, :-1, :], dy[:, 1:, :])
        dlab = np.sqrt(gx ** 2 + gy ** 2) / h                 # (KB, nx-1, ny-1) per label
        dcell = dlab.max(0)                                    # max over labels
        self.Dcell = np.zeros(self.shape)
        self.Dcell[:-1, :-1] = dcell
        self._dil = {}
        # pair_bound=True: each A label k pays only for the slope of the B label l it is
        # paired with. For any fixed pairing p the score moves by at most sum_k E[k, p(k)], so
        # score <= max_p sum_k (C[k,p(k)] + E[k,p(k)]): a tighter bound, one Hungarian away.
        self.pair_bound = bool(pair_bound)
        if self.pair_bound:
            self.Dlab = np.zeros((self.KB,) + self.shape)
            self.Dlab[:, :-1, :-1] = dlab
            self._dil_lab = {}

        self.FG = sfft.rfft2(self.G, axes=(1, 2), workers=workers)   # fixed, computed once
        self.n_rot_evals = 0
        # min_overlap = o (e.g. 0.25): PER-OVERLAP score. The objective becomes S(T) / O(T), where
        # O(T) = mean over A points of the tissue mask M = S/(S+c) of B (the share of A on B's
        # tissue), over placements with O(T) >= o. A placement that covers more tissue no longer
        # wins just by covering more. O has its own slope map, so it is bounded like S.
        self.min_overlap = None if min_overlap is None else float(min_overlap)
        if self.min_overlap is not None:
            M = tot / (tot + self.c)
            self.M = M
            mx = np.abs(M[1:, :] - M[:-1, :]); my = np.abs(M[:, 1:] - M[:, :-1])
            DM = np.zeros(self.shape)
            DM[:-1, :-1] = np.sqrt(np.maximum(mx[:, :-1], mx[:, 1:]) ** 2 +
                                   np.maximum(my[:-1, :], my[1:, :]) ** 2) / h
            self.DcellM = DM
            self.FM = sfft.rfft2(M, workers=workers)
            self._dilM = {}

    # ---------- helpers ----------
    def _assign(self, Cm):
        """max-weight matching; with contrast, a label may stay unmatched (value 0)."""
        if self.contrast:
            Cm = np.hstack([Cm, np.zeros((Cm.shape[0], Cm.shape[0]))])
        r_, c_ = linear_sum_assignment(-Cm)
        return float(Cm[r_, c_].sum()), r_, c_

    def _points(self, th, flip):
        M = rot(th) @ (FLIP if flip else np.eye(2))
        return (M @ self.A.T).T + self.tc

    def _dilated(self, mmax):
        R = int(np.ceil(mmax / self.h)) + 1
        if R not in self._dil:
            self._dil[R] = (maximum_filter(self.Dcell, size=2 * R + 1, mode="constant"), None)
            self._dil[R] = (self._dil[R][0],
                            sfft.rfft2(self._dil[R][0], workers=self.workers))
        return self._dil[R]

    def _dilated_mask(self, mmax):
        R = int(np.ceil(mmax / self.h)) + 1
        if R not in self._dilM:
            self._dilM[R] = sfft.rfft2(maximum_filter(self.DcellM, size=2 * R + 1, mode="constant"),
                                       workers=self.workers)
        return self._dilM[R]

    def _dilated_labels(self, mmax):
        R = int(np.ceil(mmax / self.h)) + 1
        if R not in self._dil_lab:
            D = np.stack([maximum_filter(d, size=2 * R + 1, mode="constant") for d in self.Dlab])
            self._dil_lab[R] = sfft.rfft2(D, axes=(1, 2), workers=self.workers)
        return self._dil_lab[R]

    def _window(self, full):
        """circular correlation output -> (2r+1, 2r+1) window of shifts -r..r"""
        idx = np.arange(-self.r, self.r + 1)
        return full[..., idx[:, None] % full.shape[-2], idx[None, :] % full.shape[-1]]

    def _corr(self, H_fft, F_fft):
        """sum_x H[x] F[x+s] for all s  = irfft( conj(FFT H) * FFT F )"""
        return sfft.irfft2(np.conj(H_fft) * F_fft, s=self.shape, axes=(-2, -1),
                           workers=self.workers)

    # ---------- core: one rotation, all translations ----------
    def rotation_eval(self, th, flip, delta, far_shift=None, offset=None, cell=None):
        """Returns (best_score, best_shift, upper_bound) for the rotation interval
        [th-delta, th+delta] x all continuous translations in the window.
        best_score is exact at (th, best grid shift); upper_bound is proven."""
        self.n_rot_evals += 1
        h = self.h
        # offset / cell (shift splitting): score the shift lattice moved by `offset`, and bound only
        # the square of half-diagonal `cell` around each lattice point (default: the whole grid cell)
        off = np.zeros(2) if offset is None else np.asarray(offset, float)
        cell = h / np.sqrt(2) if cell is None else float(cell)
        P = self._points(th, flip) + off
        x = (P - self.lo) / h
        i0 = np.floor(x).astype(int)
        f = x - i0
        # bilinear splat histograms, one per A label
        nx, ny = self.shape
        flat, wts = [], []
        for (di, dj, w) in ((0, 0, (1 - f[:, 0]) * (1 - f[:, 1])), (1, 0, f[:, 0] * (1 - f[:, 1])),
                            (0, 1, (1 - f[:, 0]) * f[:, 1]), (1, 1, f[:, 0] * f[:, 1])):
            flat.append(self.la * (nx * ny) + (i0[:, 0] + di) * ny + (i0[:, 1] + dj))
            wts.append(w)
        H = np.bincount(np.concatenate(flat), np.concatenate(wts),
                        minlength=self.KA * nx * ny).reshape((self.KA, nx, ny))
        FH = sfft.rfft2(H, axes=(1, 2), workers=self.workers)
        C = np.empty((self.KA, self.KB, 2 * self.r + 1, 2 * self.r + 1))
        for k in range(self.KA):
            C[k] = self._window(self._corr(FH[k][None], self.FG))
        C /= self.n                                           # C[k,l,s]: score contribution
        C = C.reshape(self.KA, self.KB, -1).transpose(2, 0, 1)  # (S, KA, KB)

        # proven movement bound for every shift at once
        m = 2 * self.rA * np.sin(delta / 2) + cell
        Dil, FD = self._dilated(float(m.max()))
        W = np.bincount(i0[:, 0] * ny + i0[:, 1], m,       # floor cell of each point
                        minlength=nx * ny).reshape(self.shape)
        E = self._window(self._corr(sfft.rfft2(W, workers=self.workers), FD)).ravel() / self.n

        rm = C.max(axis=2)
        if self.contrast:
            rm = np.maximum(rm, 0.0)                          # unmatched option
        rowmax = rm.sum(axis=1)                               # >= exact assignment score
        Epair = None
        if self.pair_bound:
            FDl = self._dilated_labels(float(m.max()))
            flatk = self.la * (nx * ny) + i0[:, 0] * ny + i0[:, 1]
            Wk = np.bincount(flatk, m,
                             minlength=self.KA * nx * ny).reshape((self.KA, nx, ny))
            FWk = sfft.rfft2(Wk, axes=(1, 2), workers=self.workers)
            Epair = np.empty((self.KA, self.KB, 2 * self.r + 1, 2 * self.r + 1))
            for k in range(self.KA):
                Epair[k] = self._window(self._corr(FWk[k][None], FDl))
            Epair = (Epair / self.n).reshape(self.KA, self.KB, -1).transpose(2, 0, 1)
        if far_shift is not None:
            # only shifts FAR from a given shift count (uniqueness check at near rotations).
            # A continuous shift t' with |t' - t0| > radius has its nearest grid shift within
            # h/sqrt(2), so keeping grid shifts with distance > radius - h/sqrt(2) covers it.
            t0, radius = far_shift
            m_ = 2 * self.r + 1
            sidx = np.arange(rowmax.size)
            pos = np.stack([sidx // m_ - self.r, sidx % m_ - self.r], 1) * h + self.t0 + off
            keep = np.hypot(*(pos - np.asarray(t0)).T) > radius - cell
            if not keep.any():
                return -np.inf, None, -np.inf
            rowmax = np.where(keep, rowmax, -np.inf)
            E = np.where(keep, E, -np.inf)
        if self.min_overlap is not None:
            return self._ratio_eval(C, rowmax, E, FH, W, float(m.max()), keep if far_shift is not None else None)
        order = np.argsort(rowmax)[::-1]
        best, bs = -1.0, None
        for s in order:                                       # exact best over grid shifts
            if rowmax[s] <= best:
                break
            v = self._assign(C[s])[0]
            if v > best:
                best, bs = v, s
        # upper bound over the whole interval
        if Epair is None:
            # max_s (exact(s) + E(s)); exact <= rowmax
            ubr = rowmax + E
            ub = best + E[bs]
            for s in np.argsort(ubr)[::-1]:
                if ubr[s] <= ub:
                    break
                ub = max(ub, self._assign(C[s])[0] + E[s])
        else:
            # max_s max_p sum_k (C + Epair)[k, p(k)], never looser than the plain bound
            CE = C + Epair
            rmE = CE.max(axis=2)
            if self.contrast:
                rmE = np.maximum(rmE, 0.0)
            ubr = np.minimum(rmE.sum(axis=1), rowmax + E)        # both are valid upper bounds
            if far_shift is not None:
                ubr = np.where(keep, ubr, -np.inf)
            ub = best
            for s in np.argsort(ubr)[::-1]:
                if ubr[s] <= ub:
                    break
                ub = max(ub, min(self._assign(CE[s])[0], self._assign(C[s])[0] + E[s]))
        m_ = 2 * self.r + 1
        shift = np.array([bs // m_ - self.r, bs % m_ - self.r]) * h + self.t0 + off
        return float(best), shift, float(ub)

    def _ratio_eval(self, C, rowmax, E, FH, W, mmax, keep):
        """per-overlap objective R = S / O with O >= min_overlap (see __init__).
        O(s) is exact at each grid shift (bilinear splat of all A points correlated with M);
        EM(s) bounds how much O can change inside the range (same lemma as E, slope of M).
        For T in the range with O(T) >= o: O(T) >= max(O(s) - EM(s), o) =: Olb, and
        S(T) <= Sub := S(s) + E(s). So R(T) <= Sub / Olb if Sub >= 0, else Sub / (O(s) + EM(s))."""
        o = self.min_overlap
        O = self._window(self._corr(FH.sum(0), self.FM)).ravel() / self.n
        EM = self._window(self._corr(sfft.rfft2(W, workers=self.workers),
                                     self._dilated_mask(mmax))).ravel() / self.n
        feas_pt = O >= o                                      # grid shift itself allowed
        feas_rg = O + EM >= o                                 # some placement in the range allowed
        if keep is not None:
            feas_pt &= keep; feas_rg &= keep
        Olb = np.maximum(O - EM, o)
        def bound(Sub, s):
            return Sub / Olb[s] if Sub >= 0 else Sub / (O[s] + EM[s])
        # exact best over allowed grid shifts; rowmax / O is an upper bound on the exact ratio
        qb = np.where(feas_pt, np.where(rowmax >= 0, rowmax / np.maximum(O, 1e-12), rowmax / np.maximum(O, 1e-12)), -np.inf)
        best, bs = -np.inf, None
        for s in np.argsort(qb)[::-1]:
            if not np.isfinite(qb[s]) or qb[s] <= best:
                break
            v = self._assign(C[s])[0] / O[s]
            if v > best:
                best, bs = v, s
        # upper bound over the range, visiting shifts by a cheap bound first
        Sq = rowmax + E
        ubq = np.where(feas_rg, np.where(Sq >= 0, Sq / Olb, Sq / np.maximum(O + EM, 1e-12)), -np.inf)
        ub = best
        for s in np.argsort(ubq)[::-1]:
            if not np.isfinite(ubq[s]) or ubq[s] <= ub:
                break
            ub = max(ub, bound(self._assign(C[s])[0] + E[s], s))
        if bs is None:
            return -np.inf, None, float(ub)
        m_ = 2 * self.r + 1
        shift = np.array([bs // m_ - self.r, bs % m_ - self.r]) * self.h + self.t0
        return float(best), shift, float(ub)

    def exact_score(self, th, flip, t):
        """Direct evaluation of the certified objective at ANY continuous (th, flip, t),
        independent of the FFT path. Used to check the bound."""
        P = self._points(th, flip) + np.asarray(t, float) - self.t0
        x = (P - self.lo) / self.h
        i0 = np.floor(x).astype(int); f = x - i0
        Gs = np.zeros((self.n, self.KB))
        for (di, dj, w) in ((0, 0, (1 - f[:, 0]) * (1 - f[:, 1])), (1, 0, f[:, 0] * (1 - f[:, 1])),
                            (0, 1, (1 - f[:, 0]) * f[:, 1]), (1, 1, f[:, 0] * f[:, 1])):
            Gs += w[:, None] * self.G[:, i0[:, 0] + di, i0[:, 1] + dj].T
        C = np.zeros((self.KA, self.KB))
        np.add.at(C, self.la, Gs)
        if self.min_overlap is not None:
            Om = sum(w * self.M[i0[:, 0] + di, i0[:, 1] + dj] for di, dj, w in (
                (0, 0, (1 - f[:, 0]) * (1 - f[:, 1])), (1, 0, f[:, 0] * (1 - f[:, 1])),
                (0, 1, (1 - f[:, 0]) * f[:, 1]), (1, 1, f[:, 0] * f[:, 1]))).mean()
            return self._assign(C)[0] / self.n / Om if Om >= self.min_overlap else -np.inf
        return self._assign(C)[0] / self.n

    # ---------- search ----------
    def solve(self, n0=90, flips=(0, 1), tol=1e-3, min_delta=None, max_evals=20000,
              time_limit=None):
        """1-D branch-and-bound over rotation. Starts from n0 intervals per flip.
        Stops refining an interval when its half-width reaches min_delta (default: the
        rotation term equals the grid term, i.e. rmax*d ~ h/sqrt(2)).
        Returns dict with best, transform, proven gap, certified flag, evals, profile."""
        t0 = time.time()
        if min_delta is None:
            min_delta = (self.h / np.sqrt(2)) / (2 * self.rmax)
        heap, cnt = [], 0
        best, bT = -1.0, None
        profile = []                                          # (th, flip, score) coarse level
        d0 = np.pi / n0
        for flip in flips:
            for k in range(n0):
                th = (2 * k + 1) * d0
                s, t, ub = self.rotation_eval(th, flip, d0)
                profile.append((th, flip, s))
                if s > best:
                    best, bT = s, (th, t, flip)
                heapq.heappush(heap, (-ub, cnt, th, d0, flip)); cnt += 1
        floor_gap = 0.0                                       # gap left at finest level
        capped = False
        while heap:
            nu, _, th, d, flip = heapq.heappop(heap)
            if -nu <= best + tol:
                heapq.heappush(heap, (nu, _, th, d, flip))
                break
            if d <= min_delta:
                floor_gap = max(floor_gap, -nu - best)
                continue
            if self.n_rot_evals >= max_evals or (time_limit and time.time() - t0 > time_limit):
                heapq.heappush(heap, (nu, _, th, d, flip)); capped = True
                break
            for c in (th - d / 2, th + d / 2):
                s, t, ub = self.rotation_eval(c, flip, d / 2)
                if s > best:
                    best, bT = s, (c % (2 * np.pi), t, flip)
                if ub > best + tol:
                    heapq.heappush(heap, (-ub, cnt, c, d / 2, flip)); cnt += 1
        open_gap = max([0.0] + [-h_[0] - best for h_ in heap])
        gap = max(floor_gap, open_gap, 0.0)
        return {"best": best, "transform": bT, "gap": gap,
                "certified": (not capped), "capped": capped,
                "evals": self.n_rot_evals, "secs": time.time() - t0,
                "profile": profile}


def far_upper_bound(al, best, th_best, flip_best, exclude_deg=20.0, n0=90, max_evals=20000,
                    time_limit=None, min_delta=None):
    """Proven uniqueness test. Branch-and-bound over rotations more than exclude_deg away
    from the best (either flip), driving their UPPER BOUNDS down. An interval counts as far
    if any part of it is more than exclude_deg from th_best (conservative: boundary pieces
    are included). Returns dict:
      far_ub    proven: no far alignment scores above this
      far_exact best far score actually found (a real alternative)
      eps       best - far_ub; if > 0, every far alignment is at least eps worse (proof)
      complete  False if a cap stopped the refinement (far_ub still valid, just looser)"""
    t0 = time.time()
    if min_delta is None:
        min_delta = (al.h / np.sqrt(2)) / (2 * al.rmax)
    ex = np.radians(exclude_deg)

    def dist(th, flip):
        if flip != flip_best:
            return np.inf
        return abs(np.angle(np.exp(1j * (th - th_best))))

    heap, cnt, far_exact, floor_ub, capped = [], 0, -np.inf, -np.inf, False
    d0 = np.pi / n0
    ev0 = al.n_rot_evals
    for flip in (0, 1):
        for k in range(n0):
            th = (2 * k + 1) * d0
            if dist(th, flip) + d0 <= ex:
                continue                                   # wholly inside the excluded zone
            s, _, ub = al.rotation_eval(th, flip, d0)
            if dist(th, flip) > ex:
                far_exact = max(far_exact, s)
            heapq.heappush(heap, (-ub, cnt, th, d0, flip)); cnt += 1
    while heap:
        nu, c_, th, d, flip = heapq.heappop(heap)
        if d <= min_delta:
            floor_ub = max(floor_ub, -nu); continue
        if -nu <= floor_ub:                                # cannot lower the max any further
            heapq.heappush(heap, (nu, c_, th, d, flip)); break
        if (al.n_rot_evals - ev0 >= max_evals) or (time_limit and time.time() - t0 > time_limit):
            heapq.heappush(heap, (nu, c_, th, d, flip)); capped = True; break
        for c in (th - d / 2, th + d / 2):
            if dist(c, flip) + d / 2 <= ex:
                continue
            s, _, ub = al.rotation_eval(c, flip, d / 2)
            if dist(c, flip) > ex:
                far_exact = max(far_exact, s)
            heapq.heappush(heap, (-ub, cnt, c, d / 2, flip)); cnt += 1
    far_ub = max([floor_ub] + [-h_[0] for h_ in heap])
    return {"far_ub": float(far_ub), "far_exact": float(far_exact),
            "eps": float(best - far_ub), "complete": not capped,
            "far_evals": al.n_rot_evals - ev0, "far_secs": time.time() - t0}
