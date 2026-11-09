"""Mehrotra predictor-corrector interior point method for standard-form LPs.

    primal:  min c'x   s.t.  A x = b,  x_B >= 0,  x_U + w = u,  w >= 0
    dual:    max b'y - u'v   s.t.  A'y + z - E v = c,  z_B >= 0,  v >= 0

B is the set of sign-constrained columns (the rest are free and have no
z), U the set of columns with a finite upper bound, and E places v on U.
Each iteration solves the Newton system of the perturbed KKT conditions
with one factorization of A Theta A', where

    Theta^{-1} = Z X^{-1} + E V W^{-1} E'     on B
    Theta^{-1} = rho                          on free columns

The free-column regularization rho is a proximal term; it keeps Theta
bounded and leaves an O(rho * |dx|) error in the dual equations that the
next iterations remove.

Every column also gets a tiny primal regularization (reg_primal), which
caps Theta at 1e14. Without a cap, Theta spans 40 orders of magnitude
late in a degenerate run and the factorization loses whole rows
(GREENBEA, DEGEN3). With a large cap (1e-10 here, which is common) the
dual residual of columns that hit it cannot fall below about
reg_primal * |dx|, so the run drifts instead of converging (FINNIS,
PILOT). The value 1e-14, together with the per-row pivot test in
linsys.py, was chosen on the hard Netlib problems; see the README.
"""

from dataclasses import dataclass, field

import numpy as np

from .linsys import NormalEquations


@dataclass
class IPMResult:
    status: str
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    v: np.ndarray        # upper-bound multipliers, full length (0 off U)
    iterations: int
    pobj: float
    dobj: float
    relp: float
    reld: float
    gap: float
    history: list = field(default_factory=list)


def _max_step(x, dx):
    neg = dx < 0
    if not neg.any():
        return np.inf
    return float(np.min(-x[neg] / dx[neg]))


def solve_standard(A, b, c, u, free=None, tol=1e-8, max_iter=200,
                   verbose=False, eta=0.9995, reg_primal=1e-14,
                   reg_free=1e-8, dense_limit=9000, correctors=2,
                   piv_tiny=1e-30, piv_rel=1e-14):
    m, n = A.shape
    free = np.zeros(n, dtype=bool) if free is None else np.asarray(free, bool)
    B = ~free
    U = np.flatnonzero(np.isfinite(u))
    if np.any(free[U]):
        raise ValueError("a free column cannot have an upper bound")
    uU = u[U]
    nU = U.size
    N = max(int(B.sum()) + nU, 1)
    At = A.T.tocsr()
    ne = NormalEquations(A, dense_limit=dense_limit, tiny=piv_tiny, rel=piv_rel)
    bnorm = 1.0 + np.linalg.norm(b)
    cnorm = 1.0 + np.linalg.norm(c)
    unorm = 1.0 + (np.linalg.norm(uU) if nU else 0.0)

    # ---- starting point (Mehrotra 1992, upper bounds as in LIPSOL)
    ne.factor(np.ones(n))
    x = At @ ne.solve(b)
    y = ne.solve(A @ c)
    z = c - At @ y
    w = uU - x[U]
    zU = z[U]
    v = np.maximum(-zU, 0.0)
    z[U] = np.maximum(zU, 0.0)
    z[free] = 0.0
    xb, zb = x[B], z[B]
    dp = max(-1.5 * min(xb.min(initial=0.0), w.min(initial=0.0)), 0.0)
    dd = max(-1.5 * min(zb.min(initial=0.0), v.min(initial=0.0)), 0.0)
    floor = 1e-4
    x[B] = np.maximum(xb + dp, floor)
    w = np.maximum(w + dp, floor)
    z[B] = np.maximum(zb + dd, floor)
    v = np.maximum(v + dd, floor)
    xz = x[B] @ z[B] + w @ v
    dp = 0.5 * xz / max(z[B].sum() + v.sum(), 1e-300)
    dd = 0.5 * xz / max(x[B].sum() + w.sum(), 1e-300)
    x[B] += dp
    w += dp
    z[B] += dd
    v += dd

    def residuals(x, w, y, z, v):
        rb = b - A @ x
        rc = c - At @ y - z
        rc[U] += v
        ru = uU - x[U] - w
        return rb, rc, ru

    def measures(x, w, y, z, v):
        rb, rc, ru = residuals(x, w, y, z, v)
        pobj = float(c @ x)
        dobj = float(b @ y - uU @ v)
        relp = np.linalg.norm(rb) / bnorm + (np.linalg.norm(ru) / unorm if nU else 0.0)
        reld = np.linalg.norm(rc) / cnorm
        gap = abs(pobj - dobj) / (1.0 + abs(pobj))
        return rb, rc, ru, pobj, dobj, relp, reld, gap

    history = []
    status = "max_iter"
    small_steps = 0
    best = None
    mu0 = None
    for it in range(max_iter + 1):
        rb, rc, ru, pobj, dobj, relp, reld, gap = measures(x, w, y, z, v)
        mu = (x[B] @ z[B] + w @ v) / N
        mu0 = mu if mu0 is None else mu0
        history.append(dict(it=it, pobj=pobj, dobj=dobj, relp=relp, reld=reld,
                            gap=gap, mu=mu))
        score = max(relp, reld, gap)
        if best is None or score < best[0]:
            best = (score, x.copy(), w.copy(), y.copy(), z.copy(), v.copy())
        if verbose:
            print(f"{it:4d} {pobj:+.10e} {dobj:+.10e} {relp:.1e} {reld:.1e} "
                  f"{gap:.1e} {mu:.1e}")
        if relp < tol and reld < tol and gap < tol:
            status = "optimal"
            break
        if it == max_iter:
            break
        if np.max(np.abs(y), initial=0) > 1e30 or np.max(np.abs(x), initial=0) > 1e30:
            status = "diverged"
            break
        if mu < 1e-30 * max(mu0, 1.0):
            # complementarity is gone but the residuals are not: stuck
            status = "stalled"
            break

        # ---- factorize A Theta A'
        dinv = np.empty(n)
        dinv[B] = z[B] / x[B]
        dinv[free] = reg_free
        dinv[U] += v / w
        theta = 1.0 / (dinv + reg_primal)
        ne.factor(theta)

        def newton(rxz, rwv):
            rt = rc.copy()
            rt[B] -= rxz[B] / x[B]
            rt[U] += (rwv - v * ru) / w
            dy = ne.solve(rb + A @ (theta * rt))
            dx = theta * (At @ dy - rt)
            dz = np.zeros(n)
            dz[B] = (rxz[B] - z[B] * dx[B]) / x[B]
            dw = ru - dx[U]
            dv = (rwv - v * dw) / w
            return dx, dw, dy, dz, dv

        def steps(dx, dw, dz, dv):
            ap = min(_max_step(x[B], dx[B]), _max_step(w, dw), 1.0 / eta)
            ad = min(_max_step(z[B], dz[B]), _max_step(v, dv), 1.0 / eta)
            return ap, ad

        # predictor (affine scaling direction)
        dx, dw, dy, dz, dv = newton(-x * z, -w * v)
        ap, ad = steps(dx, dw, dz, dv)
        ap, ad = min(ap, 1.0), min(ad, 1.0)
        mu_aff = (((x + ap * dx)[B] @ (z + ad * dz)[B])
                  + (w + ap * dw) @ (v + ad * dv)) / N
        sigma = min((mu_aff / mu) ** 3, 1.0)

        # corrector: second-order term plus centering
        rxz = sigma * mu - x * z - dx * dz
        rwv = sigma * mu - w * v - dw * dv
        rxz[free] = 0.0
        dx, dw, dy, dz, dv = newton(rxz, rwv)
        ap, ad = steps(dx, dw, dz, dv)

        # Gondzio centrality correctors. Look at the complementarity
        # products at a longer trial step, push the outliers back into
        # [0.1, 10] * sigma*mu, and keep the corrected direction only if
        # it allows a longer step. The system is linear, so adding the
        # correction term to the right-hand side gives direction + correction.
        target = sigma * mu
        lo, hi = 0.1 * target, 10.0 * target
        for _ in range(correctors):
            ap_t = min(1.0, 1.5 * min(ap, 1.0) + 0.1)
            ad_t = min(1.0, 1.5 * min(ad, 1.0) + 0.1)
            pxz = (x + ap_t * dx) * (z + ad_t * dz)
            pwv = (w + ap_t * dw) * (v + ad_t * dv)
            txz = np.clip(np.clip(pxz, lo, hi) - pxz, -hi, None)
            twv = np.clip(np.clip(pwv, lo, hi) - pwv, -hi, None)
            txz[free] = 0.0
            nrxz, nrwv = rxz + txz, rwv + twv
            cand = newton(nrxz, nrwv)
            nap, nad = steps(cand[0], cand[1], cand[3], cand[4])
            if min(nap, 1.0) + min(nad, 1.0) >= 1.01 * (min(ap, 1.0) + min(ad, 1.0)):
                dx, dw, dy, dz, dv = cand
                ap, ad = nap, nad
                rxz, rwv = nrxz, nrwv
            else:
                break

        ap = min(1.0, eta * ap)
        ad = min(1.0, eta * ad)
        if verbose:
            print(f"     ap={ap:.2e} ad={ad:.2e} sigma={sigma:.2e} "
                  f"skipped_pivots={ne.skipped}")
        if ap < 1e-8 and ad < 1e-8:
            small_steps += 1
            if small_steps >= 5:
                status = "stalled"
                break
        else:
            small_steps = 0

        x = x + ap * dx
        w = w + ap * dw
        y = y + ad * dy
        z = z + ad * dz
        v = v + ad * dv

    if status != "optimal" and best is not None:
        _, x, w, y, z, v = best
        _, _, _, pobj, dobj, relp, reld, gap = measures(x, w, y, z, v)

    vfull = np.zeros(n)
    vfull[U] = v
    return IPMResult(status=status, x=x, y=y, z=z, v=vfull, iterations=it,
                     pobj=pobj, dobj=dobj, relp=relp, reld=reld, gap=gap,
                     history=history)
