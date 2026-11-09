"""Top-level LP solve: presolve -> standard form -> scaling -> IPM -> back."""

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from .lp import solve_standard
from .model import LinearProgram
from .presolve import postsolve, presolve
from .standard import scale, to_standard


@dataclass
class Result:
    status: str
    x: np.ndarray = None
    y: np.ndarray = None          # row duals
    z: np.ndarray = None          # reduced costs, c - A'y
    objective: float = np.nan
    iterations: int = 0
    time: float = 0.0
    kkt: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)

    def __repr__(self):
        return (f"Result(status={self.status!r}, objective={self.objective:.10g}, "
                f"iterations={self.iterations}, time={self.time:.3f}s)")


def solve(lp: LinearProgram, tol=1e-8, presolve_on=True, scaling=True,
          max_iter=200, verbose=False, correctors=2):
    t0 = time.perf_counter()
    info = {"shape": lp.shape, "nnz": lp.A.nnz}

    if presolve_on:
        pre = presolve(lp)
        info["presolve"] = pre.stats
        if pre.status != "reduced":
            return Result(status=pre.status, time=time.perf_counter() - t0,
                          info=info | {"message": pre.message})
        red = pre.lp
    else:
        pre, red = None, lp
    info["reduced_shape"] = red.shape

    sf = to_standard(red)
    info["standard_shape"] = sf.A.shape

    if sf.A.shape[0] == 0 or sf.A.shape[1] == 0:
        # everything was removed in presolve; minimize over the bounds
        xs = np.where(sf.c >= 0, 0.0, sf.u)
        if not np.all(np.isfinite(xs)):
            return Result(status="unbounded", time=time.perf_counter() - t0, info=info)
        ys = np.zeros(sf.A.shape[0])
        status, iters = "optimal", 0
    else:
        if scaling:
            r, s = scale(sf.A)
        else:
            r, s = np.ones(sf.A.shape[0]), np.ones(sf.A.shape[1])
        As = (sp.diags(r) @ sf.A @ sp.diags(s)).tocsr()
        res = solve_standard(As, r * sf.b, s * sf.c, sf.u / s, free=sf.free, tol=tol,
                             max_iter=max_iter, verbose=verbose,
                             correctors=correctors)
        xs = s * res.x
        ys = r * res.y
        status, iters = res.status, res.iterations
        info["ipm"] = dict(relp=res.relp, reld=res.reld, gap=res.gap)
        info["history"] = res.history

    x_red, y_kept = sf.recover(xs, ys)
    y_red = np.zeros(red.shape[0])
    y_red[sf.rows] = y_kept

    if pre is not None:
        x, y, z = postsolve(lp, pre, x_red, y_red)
    else:
        x, y = x_red, y_red
        z = lp.c - lp.A.T @ y

    out = Result(status=status, x=x, y=y, z=z, objective=lp.objective(x),
                 iterations=iters, info=info)
    out.kkt = kkt_check(lp, x, y, z)
    out.time = time.perf_counter() - t0
    return out


def kkt_check(lp: LinearProgram, x, y, z):
    """Relative KKT residuals of (x, y, z) for the original problem.

    primal: largest bound or row violation, relative to the data
    dual:   largest wrong-signed multiplier, relative to |c|
    gap:    |primal objective - dual objective| relative to the objective
    """
    ax = lp.A @ x
    scale_p = 1.0 + max(np.max(np.abs(lp.rl[np.isfinite(lp.rl)]), initial=0),
                        np.max(np.abs(lp.ru[np.isfinite(lp.ru)]), initial=0),
                        np.max(np.abs(lp.l[np.isfinite(lp.l)]), initial=0),
                        np.max(np.abs(lp.u[np.isfinite(lp.u)]), initial=0))
    primal = lp.violation(x) / scale_p

    # split multipliers into the bound they act on
    y_lo, y_hi = np.maximum(y, 0), np.maximum(-y, 0)
    z_lo, z_hi = np.maximum(z, 0), np.maximum(-z, 0)
    # a multiplier on an infinite bound is dual infeasibility
    bad = np.concatenate([y_lo[np.isneginf(lp.rl)], y_hi[np.isposinf(lp.ru)],
                          z_lo[np.isneginf(lp.l)], z_hi[np.isposinf(lp.u)]])
    scale_d = 1.0 + np.max(np.abs(lp.c), initial=0)
    dual = float(np.max(bad, initial=0.0)) / scale_d

    def dot(mult, bound):
        f = np.isfinite(bound)
        return float(mult[f] @ bound[f])

    pobj = lp.objective(x)
    dobj = (dot(y_lo, lp.rl) - dot(y_hi, lp.ru) + dot(z_lo, lp.l)
            - dot(z_hi, lp.u) + lp.c0)
    gap = abs(pobj - dobj) / (1.0 + abs(pobj))
    return dict(primal=primal, dual=dual, gap=gap, pobj=pobj, dobj=dobj)
