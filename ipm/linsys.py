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


def cholesky_lipsol(M, block=96, tiny=1e-30, rel=0.0):
    """Lower Cholesky factor of a symmetric PSD matrix, in place.

    Returns (L, skipped) where skipped counts replaced pivots. A pivot is
    replaced by HUGE when it falls below tiny * max(diag), or below
    rel * (its own diagonal entry before elimination). The second test
    catches cancellation in a row whose scale is far below the largest
    one, which a single global threshold misses or overdoes.
    """
    n = M.shape[0]
    L = M
    d0 = np.diag(L).copy()
    dmax = max(float(np.max(d0, initial=0.0)), 1e-300)
    thresh = np.maximum(tiny * dmax, rel * d0)
    skipped = 0
    for k0 in range(0, n, block):
        k1 = min(k0 + block, n)
        D = L[k0:k1, k0:k1]
        for k in range(k1 - k0):
            d = D[k, k]
            if d <= thresh[k0 + k]:
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
    def __init__(self, A, dense_limit=9000, refine=3, tiny=1e-30, rel=0.0):
        self.A = sp.csr_matrix(A)
        self.At = self.A.T.tocsr()
        m = self.A.shape[0]
        self.m = m
        self.dense = m <= dense_limit
        self.refine = refine
        self.skipped = 0
        self.reg = 0.0
        self.tiny, self.rel = tiny, rel

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
            self.L, self.skipped = cholesky_lipsol(M, tiny=self.tiny, rel=self.rel)
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


class AugmentedSystem:
    """Solves the quasi-definite KKT system of a convex QP step

        [ -(Q + Theta^{-1})   A' ] [dx]   [r1]
        [        A        delta I] [dy] = [r2]

    With Q present the normal matrix A (Q + Theta^{-1})^{-1} A' is no longer
    a product of sparse factors, so the augmented system is factored
    directly. delta is a tiny dual regularization that keeps the matrix
    nonsingular when A has dependent rows; iterative refinement against
    the unregularized operator removes its effect.
    """

    def __init__(self, A, Q, dense_limit=3000, refine=3, delta=1e-10):
        self.A = sp.csr_matrix(A)
        self.Q = sp.csr_matrix(Q)
        m, n = self.A.shape
        self.m, self.n = m, n
        dense_Q = self.Q.nnz > 0.1 * n * n
        self.dense = (m + n) <= dense_limit or dense_Q
        if self.dense:
            self.Qd = self.Q.toarray()
            self.Ad = self.A.toarray()
        self.refine = refine
        self.delta = delta

    def matvec(self, dx, dy):
        r1 = -(self.Q @ dx) - dx / self.theta + self.A.T @ dy
        r2 = self.A @ dx
        return r1, r2

    def factor(self, theta):
        self.theta = theta
        m, n = self.m, self.n
        if self.dense:
            K = np.zeros((n + m, n + m))
            K[:n, :n] = -self.Qd
            K[np.arange(n), np.arange(n)] -= 1.0 / theta
            K[:n, n:] = self.Ad.T
            K[n:, :n] = self.Ad
            K[np.arange(n, n + m), np.arange(n, n + m)] = self.delta
            self.lu = la.lu_factor(K, check_finite=False)
        else:
            H = -(self.Q + sp.diags(1.0 / theta))
            K = sp.bmat([[H, self.A.T], [self.A, self.delta * sp.identity(m)]],
                        format="csc")
            self.lu = spla.splu(K, permc_spec="MMD_AT_PLUS_A")

    def _solve(self, r):
        if self.dense:
            return la.lu_solve(self.lu, r, check_finite=False)
        return self.lu.solve(r)

    def solve(self, r1, r2):
        n = self.n
        r = np.concatenate([r1, r2])
        s = self._solve(r)
        rn = np.linalg.norm(r)
        for _ in range(self.refine):
            a, b = self.matvec(s[:n], s[n:])
            res = r - np.concatenate([a, b])
            if np.linalg.norm(res) <= 1e-14 * max(rn, 1e-300):
                break
            s = s + self._solve(res)
        return s[:n], s[n:]
