import numpy as np
import pytest
import scipy.sparse as sp
from scipy.optimize import linprog

from ipm import LinearProgram, solve


def highs(lp):
    eq = np.isfinite(lp.rl) & (lp.rl == lp.ru)
    up = ~eq & np.isfinite(lp.ru)
    lo = ~eq & np.isfinite(lp.rl)
    A = lp.A.toarray()
    A_ub = np.vstack([A[up], -A[lo]])
    b_ub = np.concatenate([lp.ru[up], -lp.rl[lo]])
    bounds = [(None if np.isneginf(a) else a, None if np.isposinf(b) else b)
              for a, b in zip(lp.l, lp.u)]
    r = linprog(lp.c, A_ub=A_ub if len(b_ub) else None, b_ub=b_ub if len(b_ub) else None,
                A_eq=A[eq] if eq.any() else None, b_eq=lp.rl[eq] if eq.any() else None,
                bounds=bounds, method="highs")
    return r


def random_lp(rng, m, n, density=0.4):
    """A feasible, bounded LP with every row and bound type."""
    A = sp.random(m, n, density=density, random_state=rng, format="csr")
    A.data = rng.normal(size=A.nnz)
    # make sure no column is empty so presolve has something to do
    x0 = rng.uniform(-1, 1, n)
    kinds = rng.integers(0, 4, n)   # 0: [l,u], 1: [l,inf), 2: (-inf,u], 3: free
    l = np.where(kinds <= 1, x0 - rng.uniform(0, 2, n), -np.inf)
    u = np.where((kinds == 0) | (kinds == 2), x0 + rng.uniform(0, 2, n), np.inf)
    ax = A @ x0
    rk = rng.integers(0, 4, m)      # 0: eq, 1: <=, 2: >=, 3: ranged
    rl = np.where(rk == 0, ax, np.where(rk == 1, -np.inf, ax - rng.uniform(0, 1, m)))
    ru = np.where(rk == 0, ax, np.where(rk == 2, np.inf, ax + rng.uniform(0, 1, m)))
    # dual feasible by construction: c = A'y + z with matching signs, so the
    # objective is bounded below on the feasible set
    y = rng.normal(size=m)
    y = np.where(rk == 1, -np.abs(y), np.where(rk == 2, np.abs(y), y))
    z = rng.normal(size=n)
    z = np.where(kinds == 1, np.abs(z), np.where(kinds == 2, -np.abs(z),
                 np.where(kinds == 3, 0.0, z)))
    c = A.T @ y + z
    return LinearProgram(c=c, A=A, rl=rl, ru=ru, l=l, u=u)


def test_textbook_lp():
    # max 3x + 5y st x <= 4, 2y <= 12, 3x + 2y <= 18  -> x=2, y=6, obj 36
    lp = LinearProgram.from_arrays(c=[-3, -5], A_ub=[[1, 0], [0, 2], [3, 2]],
                                   b_ub=[4, 12, 18])
    r = solve(lp)
    assert r.status == "optimal"
    np.testing.assert_allclose(r.x, [2, 6], atol=1e-7)
    assert abs(r.objective + 36) < 1e-7
    # duals of the textbook problem: 0, 1.5, 1 (signs follow c - A'y - z = 0)
    np.testing.assert_allclose(r.y, [0, -1.5, -1], atol=1e-6)


@pytest.mark.parametrize("seed", range(25))
@pytest.mark.parametrize("presolve_on", [True, False])
def test_random_against_highs(seed, presolve_on):
    rng = np.random.default_rng(seed)
    m, n = rng.integers(5, 40), rng.integers(5, 60)
    lp = random_lp(rng, m, n)
    ref = highs(lp)
    assert ref.status == 0
    r = solve(lp, presolve_on=presolve_on)
    assert r.status == "optimal"
    assert abs(r.objective - ref.fun) <= 1e-6 * max(1, abs(ref.fun))
    assert r.kkt["primal"] < 1e-7
    assert r.kkt["dual"] < 1e-7
    assert r.kkt["gap"] < 1e-7


def test_presolve_reductions_keep_duals_right():
    # singleton rows, a fixed column and an empty row, all in one problem
    A = sp.csr_matrix(np.array([
        [1.0, 0, 0, 0],     # singleton: x0 >= 1
        [0, 2.0, 0, 0],     # singleton: 2 x1 <= 3
        [1.0, 1.0, 1.0, 1.0],
        [0, 0, 0, 0],       # empty
        [1.0, -1.0, 0, 1.0],
    ]))
    lp = LinearProgram(c=[1, -1, 2, 1], A=A,
                       rl=[1, -np.inf, 4, -1, -np.inf], ru=[np.inf, 3, 4, 1, 2],
                       l=[0, 0, 0.5, 0], u=[10, 10, 0.5, 10])
    r = solve(lp)
    ref = highs(lp)
    assert r.status == "optimal"
    assert abs(r.objective - ref.fun) < 1e-7
    assert r.info["presolve"]["singleton_rows"] >= 2
    assert r.info["presolve"]["fixed_cols"] >= 1
    assert max(r.kkt["primal"], r.kkt["dual"], r.kkt["gap"]) < 1e-7


def test_degenerate_and_rank_deficient():
    # duplicated equality rows make A rank deficient
    rng = np.random.default_rng(7)
    A = rng.normal(size=(6, 12))
    A = np.vstack([A, A[:3]])
    x0 = rng.uniform(0.5, 1.5, 12)
    b = A @ x0
    c = rng.uniform(0, 1, 12)
    lp = LinearProgram.from_arrays(c=c, A_eq=A, b_eq=b)
    r = solve(lp)
    ref = highs(lp)
    assert r.status == "optimal"
    assert abs(r.objective - ref.fun) < 1e-7 * max(1, abs(ref.fun))


def test_presolve_detects_infeasible_bounds():
    lp = LinearProgram(c=[1, 1], A=sp.csr_matrix([[1.0, 0.0], [1.0, 1.0]]),
                       rl=[5, 0], ru=[np.inf, 10], l=[0, 0], u=[3, 1])
    assert solve(lp).status == "infeasible"


def test_presolve_detects_unbounded_empty_column():
    lp = LinearProgram(c=[1, -1], A=sp.csr_matrix([[1.0, 0.0]]),
                       rl=[1], ru=[2], l=[0, 0], u=[np.inf, np.inf])
    assert solve(lp).status == "unbounded"
