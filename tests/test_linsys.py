import numpy as np
import scipy.sparse as sp

from ipm.linsys import HUGE, NormalEquations, cholesky_lipsol


def test_cholesky_matches_lapack():
    rng = np.random.default_rng(0)
    B = rng.normal(size=(300, 300))
    M = B @ B.T + 300 * np.eye(300)
    L, skipped = cholesky_lipsol(M.copy(), block=64)
    assert skipped == 0
    np.testing.assert_allclose(L, np.linalg.cholesky(M), rtol=1e-10, atol=1e-10)


def test_cholesky_skips_dependent_rows():
    # rank 5 matrix of size 8: three pivots collapse and get replaced
    rng = np.random.default_rng(1)
    B = rng.normal(size=(8, 5))
    M = B @ B.T
    L, skipped = cholesky_lipsol(M.copy(), block=3, tiny=1e-12)
    assert skipped == 3
    assert np.sum(np.diag(L) == HUGE) == 3


def test_normal_equations_dense_and_sparse_agree():
    rng = np.random.default_rng(2)
    A = sp.random(200, 500, density=0.02, random_state=rng, format="csr")
    A = A + sp.eye(200, 500)
    theta = rng.uniform(1e-3, 1e3, 500)
    r = rng.normal(size=200)
    d = NormalEquations(A, dense_limit=10**6)
    d.factor(theta)
    s = NormalEquations(A, dense_limit=0)
    s.factor(theta)
    x1, x2 = d.solve(r), s.solve(r)
    M = (A @ sp.diags(theta) @ A.T).toarray()
    np.testing.assert_allclose(M @ x1, r, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(x1, x2, rtol=1e-6, atol=1e-8)
