import numpy as np
import pytest
import scipy.sparse as sp
from scipy.optimize import minimize

from ipm import LinearProgram, solve


def test_box_qp_closed_form():
    # min 1/2 |x - t|^2 over 0 <= x <= 1  ->  x = clip(t, 0, 1)
    t = np.array([-0.5, 0.3, 0.9, 1.7])
    lp = LinearProgram(c=-t, A=sp.csr_matrix((0, 4)), rl=[], ru=[],
                       l=np.zeros(4), u=np.ones(4), c0=0.5 * t @ t)
    r = solve(lp, Q=sp.identity(4))
    assert r.status == "optimal"
    np.testing.assert_allclose(r.x, np.clip(t, 0, 1), atol=1e-7)


@pytest.mark.parametrize("seed", range(10))
def test_random_qp_against_slsqp(seed):
    rng = np.random.default_rng(seed)
    n, me, mi = 12, 3, 5
    B = rng.normal(size=(n, n))
    Q = B @ B.T / n + 0.1 * np.eye(n)
    c = rng.normal(size=n)
    Ae, Ai = rng.normal(size=(me, n)), rng.normal(size=(mi, n))
    x0 = rng.uniform(0.2, 0.8, n)
    be, bi = Ae @ x0, Ai @ x0 + 0.3
    lp = LinearProgram.from_arrays(c=c, A_ub=Ai, b_ub=bi, A_eq=Ae, b_eq=be,
                                   l=np.zeros(n), u=np.ones(n))
    r = solve(lp, Q=Q)
    assert r.status == "optimal"
    f = lambda x: 0.5 * x @ Q @ x + c @ x
    ref = minimize(f, x0, jac=lambda x: Q @ x + c, method="SLSQP",
                   bounds=[(0, 1)] * n,
                   constraints=[{"type": "eq", "fun": lambda x: Ae @ x - be},
                                {"type": "ineq", "fun": lambda x: bi - Ai @ x}],
                   options=dict(ftol=1e-12, maxiter=500))
    assert abs(r.objective - ref.fun) < 1e-6 * max(1, abs(ref.fun))
    assert max(r.kkt["primal"], r.kkt["dual"], r.kkt["gap"]) < 1e-7
