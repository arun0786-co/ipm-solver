"""Normal-equation solves  (A diag(theta) A') dy = r  for the IPM.

Late in an interior point run theta spans twenty orders of magnitude and
A often has dependent rows, so the normal matrix is singular in exact
arithmetic and badly conditioned in floating point. Two things keep the
solve usable:

  - a blocked Cholesky that, like LIPSOL, replaces a pivot that has
    collapsed to rounding noise with a huge number. That zeroes the
    corresponding component of dy instead of failing.
  - a few steps of iterative refinement against the exact operator.

Large sparse systems go through SuperLU instead.
"""

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

HUGE = 1e128


def cholesky_lipsol(M, block=96, tiny=1e-30):
    """Lower Cholesky factor of a symmetric PSD matrix, in place.

    Returns (L, skipped) where skipped counts replaced pivots. Pivots
    below tiny * max(diag) are set to HUGE.
    """
    n = M.shape[0]
    L = M
    dmax = max(float(np.max(np.diag(L), initial=0.0)), 1e-300)
    thresh = tiny * dmax
    skipped = 0
    for k0 in range(0, n, block):
        k1 = min(k0 + block, n)
        D = L[k0:k1, k0:k1]
        for k in range(k1 - k0):
            d = D[k, k]
            if d <= thresh:
                D[k, k] = HUGE
                D[k + 1:, k] = 0.0
                skipped += 1
                continue
            d = np.sqrt(d)
            D[k, k] = d
            col = D[k + 1:, k]
            col /= d
            D[k + 1:, k + 1:] -= np.outer(col, col)
        if k1 < n:
            P = L[k1:, k0:k1]
            # P <- P D^{-T}
            P[:] = la.solve_triangular(D, P.T, lower=True, check_finite=False).T
            L[k1:, k1:] -= P @ P.T
    return np.tril(L), skipped


class NormalEquations:
    def __init__(self, A, dense_limit=9000, refine=3):
        self.A = sp.csr_matrix(A)
        self.At = self.A.T.tocsr()
        m = self.A.shape[0]
        self.m = m
        self.dense = m <= dense_limit
        self.refine = refine
        self.skipped = 0
        self.reg = 0.0

    def matvec(self, v):
        return self.A @ (self.theta * (self.At @ v))

    def factor(self, theta, sparse_reg=1e-12):
        """Factor A diag(theta) A'.

        The dense path adds no regularization. A diagonal shift would
        lift the pivots of dependent rows just above the skip threshold,
        and the solve would then amplify rounding noise in exactly the
        directions the skip is meant to zero (this broke about 1% of
        random test LPs before it was removed). SuperLU cannot skip
        pivots, so the sparse path keeps a small shift.
        """
        self.theta = theta
        M = (self.A @ sp.diags(theta) @ self.At)
        if self.dense:
            M = M.toarray()
            self.L, self.skipped = cholesky_lipsol(M)
        else:
            M = M.tocsc()
            dmax = float(M.diagonal().max(initial=1.0))
            self.reg = max(sparse_reg * dmax, 1e-14)
            M = M + self.reg * sp.identity(self.m, format="csc")
            self.lu = spla.splu(M, permc_spec="MMD_AT_PLUS_A",
                                diag_pivot_thresh=0.0,
                                options=dict(SymmetricMode=True))
            self.skipped = 0

    def _solve(self, r):
        if self.dense:
            t = la.solve_triangular(self.L, r, lower=True, check_finite=False)
            return la.solve_triangular(self.L, t, lower=True, trans="T",
                                       check_finite=False)
        return self.lu.solve(r)

    def solve(self, r):
        x = self._solve(r)
        rn = np.linalg.norm(r)
        if rn == 0:
            return x
        res = r - self.matvec(x)
        best = np.linalg.norm(res)
        for _ in range(self.refine):
            if best <= 1e-14 * rn:
                break
            dx = self._solve(res)
            xn = x + dx
            resn = r - self.matvec(xn)
            nn = np.linalg.norm(resn)
            if nn >= best:
                break
            x, res, best = xn, resn, nn
        return x
