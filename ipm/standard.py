"""Conversion between the bounded form and the standard form used by the IPM.

Standard form:
    minimize    c'x + c0
    subject to  A x = b
                0 <= x,  x_j <= u_j where u_j is finite

Columns of the bounded problem are mapped as
    l finite          x = l + x'           (upper bound u - l if finite)
    l = -inf, u fin.  x = u - x'
    free              x = x'               (flagged free, no sign constraint)

Free columns are kept as they are instead of being split into x+ - x-.
With a split, both halves drift to large values while their difference
stays put, their Theta entries blow up, and they swamp the normal matrix.
and rows get slacks: Ax + s = ru for <= rows, Ax - s = rl for >= rows and
Ax - s = rl with 0 <= s <= ru - rl for ranged rows.
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .model import LinearProgram


@dataclass
class StandardForm:
    A: sp.csr_matrix
    b: np.ndarray
    c: np.ndarray
    u: np.ndarray
    c0: float
    free: np.ndarray      # True for columns with no sign constraint
    # how to get the bounded-form x back
    n_orig: int
    sign: np.ndarray      # +1 or -1
    offset: np.ndarray    # l or u
    rows: np.ndarray      # bounded-form rows kept (free rows are dropped)

    def recover(self, xs, ys):
        return self.offset + self.sign * xs[:self.n_orig], ys


def to_standard(lp: LinearProgram) -> StandardForm:
    m, n = lp.shape
    A = lp.A.tocsc()
    l, u, c = lp.l, lp.u, lp.c
    lfin, ufin = np.isfinite(l), np.isfinite(u)

    kind_lo = lfin
    kind_up = ~lfin & ufin
    kind_free = ~lfin & ~ufin

    sign = np.where(kind_up, -1.0, 1.0)
    offset = np.where(kind_lo, l, np.where(kind_up, u, 0.0))

    cols_A = (A @ sp.diags(sign)).tocsc()
    cs = c * sign
    with np.errstate(invalid="ignore"):
        us = np.where(kind_lo, u - l, np.inf)

    b_shift = A @ offset
    c0 = lp.c0 + float(c @ offset)

    # rows
    rl, ru = lp.rl - b_shift, lp.ru - b_shift
    rlf, ruf = np.isfinite(rl), np.isfinite(ru)
    keep = rlf | ruf
    rows = np.flatnonzero(keep)
    rl, ru, rlf, ruf = rl[keep], ru[keep], rlf[keep], ruf[keep]
    eq = rlf & ruf & (rl == ru)
    le = ~rlf & ruf
    ge = rlf & ~ruf
    rng = rlf & ruf & ~eq
    b = np.where(le, ru, rl)

    slack_rows = np.flatnonzero(le | ge | rng)
    ns = slack_rows.size
    slack_sign = np.where(le[slack_rows], 1.0, -1.0)
    S = sp.csc_matrix((slack_sign, (slack_rows, np.arange(ns))), shape=(rows.size, ns))
    slack_u = np.where(rng[slack_rows], (ru - rl)[slack_rows], np.inf)

    As = sp.hstack([cols_A[rows], S], format="csr")
    cs = np.concatenate([cs, np.zeros(ns)])
    us = np.concatenate([us, slack_u])
    free = np.concatenate([kind_free, np.zeros(ns, dtype=bool)])
    return StandardForm(A=As, b=b, c=cs, u=us, c0=c0, free=free, n_orig=n,
                        sign=sign, offset=offset, rows=rows)


def scale(A, passes=8):
    """Geometric-mean scaling followed by equilibration.

    Returns row factors r and column factors s with entries that are powers
    of two, so that diag(r) A diag(s) has entries close to 1 in magnitude.
    """
    m, n = A.shape
    r = np.ones(m)
    s = np.ones(n)
    B = sp.csr_matrix(abs(A))
    B.eliminate_zeros()
    if B.nnz == 0:
        return r, s

    def row_minmax(M):
        mx = np.asarray(M.max(axis=1).todense()).ravel()
        inv = M.copy()
        inv.data = 1.0 / inv.data
        mn = np.asarray(inv.max(axis=1).todense()).ravel()
        with np.errstate(divide="ignore"):
            mn = np.where(mn > 0, 1.0 / mn, 0.0)
        return mn, mx

    for _ in range(passes):
        M = sp.diags(r) @ B @ sp.diags(s)
        mn, mx = row_minmax(M.tocsr())
        f = np.where(mx > 0, 1.0 / np.sqrt(np.maximum(mn * mx, 1e-300)), 1.0)
        r *= f
        M = (sp.diags(r) @ B @ sp.diags(s)).T.tocsr()
        mn, mx = row_minmax(M)
        f = np.where(mx > 0, 1.0 / np.sqrt(np.maximum(mn * mx, 1e-300)), 1.0)
        s *= f

    # equilibrate so the largest entry in each row and column is about 1
    M = (sp.diags(r) @ B @ sp.diags(s)).tocsr()
    mx = np.asarray(M.max(axis=1).todense()).ravel()
    r *= 1.0 / np.where(mx > 0, mx, 1.0)
    M = (sp.diags(r) @ B @ sp.diags(s)).tocsc()
    mx = np.asarray(M.max(axis=0).todense()).ravel()
    s *= 1.0 / np.where(mx > 0, mx, 1.0)

    r = 2.0 ** np.round(np.log2(r))
    s = 2.0 ** np.round(np.log2(s))
    return r, s
