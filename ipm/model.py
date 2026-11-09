"""General linear program in bounded form.

    minimize    c'x + c0
    subject to  rl <= A x <= ru
                l  <= x   <= u

Infinite bounds are stored as +-inf. An equality row has rl == ru.
"""

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp


@dataclass
class LinearProgram:
    c: np.ndarray
    A: sp.csr_matrix
    rl: np.ndarray
    ru: np.ndarray
    l: np.ndarray
    u: np.ndarray
    c0: float = 0.0
    name: str = ""
    row_names: list = field(default_factory=list)
    col_names: list = field(default_factory=list)

    def __post_init__(self):
        self.c = np.asarray(self.c, dtype=float)
        self.A = sp.csr_matrix(self.A, dtype=float)
        m, n = self.A.shape
        self.rl = _vec(self.rl, m, -np.inf)
        self.ru = _vec(self.ru, m, np.inf)
        self.l = _vec(self.l, n, 0.0)
        self.u = _vec(self.u, n, np.inf)
        if self.c.shape != (n,):
            raise ValueError(f"c has shape {self.c.shape}, expected ({n},)")

    @property
    def shape(self):
        return self.A.shape

    def objective(self, x):
        return float(self.c @ x + self.c0)

    def violation(self, x):
        """Largest bound or row violation of x, in absolute terms."""
        ax = self.A @ x
        v = [
            np.max(self.rl - ax, initial=0.0),
            np.max(ax - self.ru, initial=0.0),
            np.max(self.l - x, initial=0.0),
            np.max(x - self.u, initial=0.0),
        ]
        return float(max(v))

    @classmethod
    def from_arrays(cls, c, A_ub=None, b_ub=None, A_eq=None, b_eq=None,
                    l=None, u=None, c0=0.0, name=""):
        """Build from the scipy.optimize.linprog style of arguments."""
        c = np.asarray(c, dtype=float)
        n = c.size
        blocks, rl, ru = [], [], []
        if A_ub is not None:
            A_ub = sp.csr_matrix(A_ub)
            blocks.append(A_ub)
            rl.append(np.full(A_ub.shape[0], -np.inf))
            ru.append(np.asarray(b_ub, dtype=float))
        if A_eq is not None:
            A_eq = sp.csr_matrix(A_eq)
            blocks.append(A_eq)
            b_eq = np.asarray(b_eq, dtype=float)
            rl.append(b_eq)
            ru.append(b_eq)
        if blocks:
            A = sp.vstack(blocks, format="csr")
            rl, ru = np.concatenate(rl), np.concatenate(ru)
        else:
            A = sp.csr_matrix((0, n))
            rl, ru = np.zeros(0), np.zeros(0)
        return cls(c=c, A=A, rl=rl, ru=ru,
                   l=np.zeros(n) if l is None else l,
                   u=np.full(n, np.inf) if u is None else u,
                   c0=c0, name=name)


def _vec(v, size, default):
    if v is None:
        return np.full(size, default)
    v = np.asarray(v, dtype=float)
    if v.ndim == 0:
        return np.full(size, float(v))
    if v.shape != (size,):
        raise ValueError(f"bound has shape {v.shape}, expected ({size},)")
    return v.copy()
