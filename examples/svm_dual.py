"""Kernel SVM training as a convex QP, checked against libsvm.

The soft-margin SVM dual is

    min  a'Qa/2 - 1'a   s.t.  y'a = 0,  0 <= a <= C,   Q_ij = y_i y_j k(x_i, x_j)

This trains an RBF-kernel classifier on scikit-learn's digits data
(odd vs even digit, 1797 images of 8x8 pixels) with the IPM and with
sklearn's SVC, which wraps libsvm's SMO solver, and compares the two.

usage: python examples/svm_dual.py [--C 1.0]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ipm import LinearProgram, solve  # noqa: E402


def rbf(X, Z, gamma):
    d = (X * X).sum(1)[:, None] + (Z * Z).sum(1)[None, :] - 2 * X @ Z.T
    return np.exp(-gamma * np.maximum(d, 0))


def main():
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    ap = argparse.ArgumentParser()
    ap.add_argument("--C", type=float, default=1.0)
    args = ap.parse_args()
    C = args.C

    X, digit = load_digits(return_X_y=True)
    y = np.where(digit % 2 == 1, 1.0, -1.0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0,
                                          stratify=y)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    gamma = 1.0 / (Xtr.shape[1] * Xtr.var())
    n = len(ytr)

    # ---- IPM on the dual
    K = rbf(Xtr, Xtr, gamma)
    Q = (ytr[:, None] * ytr[None, :]) * K
    lp = LinearProgram(c=-np.ones(n), A=sp.csr_matrix(ytr[None, :]),
                       rl=[0.0], ru=[0.0], l=np.zeros(n), u=np.full(n, C))
    t = time.perf_counter()
    res = solve(lp, Q=Q)
    t_ipm = time.perf_counter() - t
    a = res.x
    sv = a > 1e-6 * C
    free = sv & (a < C * (1 - 1e-6))
    f_nob = K @ (a * ytr)
    b = np.mean(ytr[free] - f_nob[free])
    pred_ipm = np.sign(rbf(Xte, Xtr, gamma) @ (a * ytr) + b)

    # ---- libsvm
    t = time.perf_counter()
    svc = SVC(C=C, kernel="rbf", gamma=gamma, tol=1e-8, shrinking=True).fit(Xtr, ytr)
    t_svc = time.perf_counter() - t
    a_svc = np.zeros(n)
    a_svc[svc.support_] = np.abs(svc.dual_coef_.ravel())
    pred_svc = svc.predict(Xte)

    def dual_obj(a):
        return 0.5 * a @ Q @ a - a.sum()

    print(f"digits odd/even, n_train={n}, n_test={len(yte)}, C={C}, "
          f"gamma={gamma:.4g}")
    print(f"IPM:    {res.status}, {res.iterations} iterations, {t_ipm:.2f}s")
    print(f"libsvm: {t_svc:.2f}s")
    print(f"dual objective   IPM {dual_obj(a):.10f}   libsvm {dual_obj(a_svc):.10f}")
    print(f"max |a_ipm - a_libsvm| = {np.max(np.abs(a - a_svc)):.2e}")
    print(f"support vectors  IPM {sv.sum()}   libsvm {len(svc.support_)}")
    print(f"intercept        IPM {b:.6f}   libsvm {svc.intercept_[0]:.6f}")
    print(f"test accuracy    IPM {np.mean(pred_ipm == yte):.4f}   "
          f"libsvm {np.mean(pred_svc == yte):.4f}")
    print(f"predictions that differ: {int(np.sum(pred_ipm != pred_svc))}")
    print(f"KKT residuals: primal {res.kkt['primal']:.1e} dual {res.kkt['dual']:.1e} "
          f"gap {res.kkt['gap']:.1e}")


if __name__ == "__main__":
    main()
