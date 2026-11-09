"""Presolve and postsolve for LinearProgram.

Reductions, applied until nothing changes:
  - empty rows are dropped (or the problem is declared infeasible)
  - fixed columns (l == u) are substituted into the row bounds
  - empty columns are set to their best bound (or the problem is unbounded)
  - singleton rows a*x_j in [rl, ru] become bounds on x_j

Postsolve restores x, row duals y and reduced costs z for the original
problem, so the answer can be checked against the original KKT conditions.
The sign convention is c - A'y - z = 0 with y_i >= 0 when row i sits at
its lower bound and z_j >= 0 when x_j sits at its lower bound.
"""

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from .model import LinearProgram

FEAS_TOL = 1e-9


@dataclass
class PresolveResult:
    status: str                     # 'reduced' | 'infeasible' | 'unbounded'
    lp: LinearProgram = None        # reduced problem
    rows: np.ndarray = None         # kept original row indices
    cols: np.ndarray = None         # kept original column indices
    fixed: dict = field(default_factory=dict)   # col -> value
    singletons: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    message: str = ""


def presolve(lp: LinearProgram) -> PresolveResult:
    m, n = lp.shape
    A = lp.A.tocsr()
    Ac = A.tocsc()
    rl, ru = lp.rl.copy(), lp.ru.copy()
    l, u = lp.l.copy(), lp.u.copy()
    c0 = lp.c0
    row_on = np.ones(m, dtype=bool)
    col_on = np.ones(n, dtype=bool)
    fixed = {}
    singletons = []   # (row, col, a, l_before, u_before, l_after, u_after)
    stats = dict(empty_rows=0, fixed_cols=0, empty_cols=0, singleton_rows=0,
                 free_rows=0)
    pattern = A.copy()
    pattern.data = np.ones_like(pattern.data)
    pattern_c = pattern.tocsc()

    def bad(msg, status):
        return PresolveResult(status=status, message=msg, stats=stats)

    # rows with no finite bound constrain nothing
    free = row_on & np.isneginf(rl) & np.isposinf(ru)
    stats["free_rows"] = int(free.sum())
    row_on &= ~free

    changed = True
    while changed:
        changed = False

        # fixed columns
        width = u - l
        fix = col_on & np.isfinite(width) & (width <= 1e-12 * np.maximum(1, np.abs(l)))
        if fix.any():
            idx = np.flatnonzero(fix)
            if np.any(l[idx] > u[idx] + FEAS_TOL):
                return bad("column with l > u", "infeasible")
            vals = l[idx]
            shift = Ac[:, idx] @ vals
            rl -= shift
            ru -= shift
            c0 += float(lp.c[idx] @ vals)
            for j, v in zip(idx, vals):
                fixed[int(j)] = float(v)
            col_on[idx] = False
            stats["fixed_cols"] += idx.size
            changed = True

        row_cnt = pattern @ col_on.astype(float)
        col_cnt = pattern_c.T @ row_on.astype(float)

        # empty rows
        empty = row_on & (row_cnt == 0)
        if empty.any():
            idx = np.flatnonzero(empty)
            if np.any(rl[idx] > FEAS_TOL) or np.any(ru[idx] < -FEAS_TOL):
                return bad("empty row with nonzero bounds", "infeasible")
            row_on[idx] = False
            stats["empty_rows"] += idx.size
            changed = True

        # empty columns
        empty = col_on & (col_cnt == 0)
        if empty.any():
            for j in np.flatnonzero(empty):
                cj = lp.c[j]
                if cj > 0:
                    v = l[j]
                elif cj < 0:
                    v = u[j]
                else:
                    v = min(max(0.0, l[j]), u[j])
                if not np.isfinite(v):
                    return bad(f"column {j} is unbounded", "unbounded")
                fixed[int(j)] = float(v)
                c0 += float(cj * v)
                col_on[j] = False
            stats["empty_cols"] += int(empty.sum())
            changed = True

        # singleton rows
        single = np.flatnonzero(row_on & (row_cnt == 1))
        for i in single:
            if not row_on[i]:
                continue
            lo, hi = A.indptr[i], A.indptr[i + 1]
            cols = A.indices[lo:hi]
            vals = A.data[lo:hi]
            keep = col_on[cols]
            if keep.sum() != 1:
                continue  # another singleton in this pass changed things
            j = int(cols[keep][0])
            a = float(vals[keep][0])
            if a > 0:
                blo, bhi = rl[i] / a, ru[i] / a
            else:
                blo, bhi = ru[i] / a, rl[i] / a
            lb, ub = l[j], u[j]
            nl, nu = max(lb, blo), min(ub, bhi)
            if nl > nu + FEAS_TOL * max(1.0, abs(nl)):
                return bad(f"singleton row {i} conflicts with bounds", "infeasible")
            if np.isfinite(nl) and np.isfinite(nu) and nu - nl <= 1e-10 * max(1.0, abs(nl)):
                nl = nu = 0.5 * (nl + nu)
            l[j], u[j] = nl, nu
            singletons.append((int(i), j, a, lb, ub, nl, nu))
            row_on[i] = False
            stats["singleton_rows"] += 1
            changed = True

    rows = np.flatnonzero(row_on)
    cols = np.flatnonzero(col_on)
    red = LinearProgram(
        c=lp.c[cols], A=A[rows][:, cols], rl=rl[rows], ru=ru[rows],
        l=l[cols], u=u[cols], c0=c0, name=lp.name,
    )
    return PresolveResult(status="reduced", lp=red, rows=rows, cols=cols,
                          fixed=fixed, singletons=singletons, stats=stats)


def postsolve(lp: LinearProgram, pre: PresolveResult, x_red, y_red):
    """Map a solution of the reduced problem back to the original one."""
    m, n = lp.shape
    x = np.zeros(n)
    x[pre.cols] = x_red
    for j, v in pre.fixed.items():
        x[j] = v
    y = np.zeros(m)
    y[pre.rows] = y_red
    Ac = lp.A.tocsc()

    # undo singleton rows last-in first-out; a removed row takes over the
    # reduced cost of its column when its bound is the one that is active
    for i, j, a, lb, ub, nl, nu in reversed(pre.singletons):
        lo, hi = Ac.indptr[j], Ac.indptr[j + 1]
        zj = lp.c[j] - Ac.data[lo:hi] @ y[Ac.indices[lo:hi]]
        took_lower = nl > lb
        took_upper = nu < ub
        if (zj > 0 and took_lower) or (zj < 0 and took_upper):
            y[i] = zj / a
    z = lp.c - lp.A.T @ y
    return x, y, z
