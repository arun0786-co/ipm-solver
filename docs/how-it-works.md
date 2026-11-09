# How it works

This document follows one LP through the solver, then covers the QP path and the MDP formulation. The last section lists the numerical problems found on Netlib and what fixed each one, since most of the work went there.

## 1. From the file to standard form

**Bounded form.** `read_mps` produces

    minimize    c'x + c0
    subject to  rl <= A x <= ru
                l  <= x   <= u

Equality rows have `rl == ru`. MPS details that matter on Netlib:
- A RANGES entry on an E row extends it up or down depending on its sign.
- An UP bound with a negative value and a zero lower bound makes the lower bound minus infinity.
- A RHS entry on the objective row is the negated objective constant. E226 has one, and the published Netlib value leaves it out.
- Some files store explicit zero coefficients, which have to be dropped before presolve counts nonzeros.

**Presolve** (`presolve.py`) repeats four reductions until nothing changes:

| reduction | action | postsolve |
|---|---|---|
| empty row | check `rl <= 0 <= ru`, drop | `y_i = 0` |
| fixed column (`l = u`) | move `a_j * l_j` into the row bounds and `c_j * l_j` into c0 | `x_j = l_j` |
| empty column | set `x_j` to the bound its cost prefers (unbounded if that bound is infinite) | `x_j` as chosen |
| singleton row `a x_j in [rl, ru]` | turn into bounds on `x_j`, drop the row | see below |

The singleton-row postsolve is the only subtle one. After the reduced problem is solved, column j has a reduced cost `z_j = c_j - A_j'y` computed with the rows that are back so far. If `x_j` sits on a bound that came from row i (`z_j > 0` on a tightened lower bound, `z_j < 0` on a tightened upper one), then row i is the constraint that is active. It takes over the multiplier: `y_i = z_j / a_ij`, and column j's reduced cost becomes zero. Undoing the reductions in reverse order handles chains, such as a row that becomes a singleton only after another row fixed a column.

Getting the duals back matters for two reasons. Without them the KKT check in `solve.py` could not run on the original problem. And for the MDP example the duals are the answer.

**Standard form** (`standard.py`). The interior point method works on

    minimize c'x  s.t.  A x = b,  x_B >= 0,  x_U <= u

Each original column becomes one standard column:
- `x = l + x'` when l is finite,
- `x = u - x'` when only u is finite,
- or a *free* column when neither is.

Rows get a slack: `+s` for `<=` rows, `-s` for `>=` rows, and `-s` with `0 <= s <= ru - rl` for ranged rows.

**Scaling.** Eight passes of geometric-mean scaling, `r_i = 1/sqrt(max_j |a_ij| * min_j |a_ij|)` and the same for columns, then one pass of equilibration so the largest entry in each row and column is 1. Factors are rounded to powers of two, so scaling adds no rounding error.

## 2. The primal-dual method

With upper bounds written as `x_U + w = u, w >= 0` and multipliers `z >= 0` for `x_B >= 0` and `v >= 0` for `w >= 0`, the optimality conditions are

    A x = b                          (primal feasibility)
    x_U + w = u
    A'y + z - E v = c                (dual feasibility; E puts v on the U columns)
    x_j z_j = 0,  w_j v_j = 0        (complementarity)

An interior point method replaces `x_j z_j = 0` with `x_j z_j = mu` and follows the solutions as mu goes to 0 (the central path). It stays strictly inside `x, z, w, v > 0` the whole time.

**The Newton step.** Linearizing at the current point with residuals `rb = b - Ax`, `ru = u - x_U - w`, `rc = c - A'y - z + Ev` and complementarity targets `rxz`, `rwv` gives

    A dx = rb
    dx_U + dw = ru
    A'dy + dz - E dv = rc
    Z dx + X dz = rxz
    V dw + W dv = rwv

Eliminate dz, dw, dv. With `Theta^{-1} = Z X^{-1} + E V W^{-1} E'` (diagonal) and

    rt = rc - X^{-1} rxz + E W^{-1} (rwv - V ru)

everything reduces to `dx = Theta (A'dy - rt)` and the normal equations

    (A Theta A') dy = rb + A Theta rt

So each iteration costs one factorization of the m x m matrix `A Theta A'`, plus triangular solves.

**Mehrotra predictor-corrector.** Solve once with `rxz = -XZe` (the affine direction, aiming straight at mu = 0). Compute how far it can go and the mu it would reach, `mu_aff`. Then set `sigma = (mu_aff / mu)^3`: a direction that makes good progress needs little centering. Solve again, with the same factorization, using

    rxz = sigma*mu - XZe - dX_aff dZ_aff e

The last term is the second-order correction: the product `dx_j dz_j` that the linearization dropped.

**Gondzio correctors.** Take the corrector direction to a longer trial step. Find the products `x_j z_j` that would land outside `[0.1, 10] * sigma*mu`, and add the amount needed to pull them back into the right-hand side. Keep the new direction only if it allows a longer step. This costs one extra solve per corrector and cut the average iteration count on random LPs from 7.9 to 6.9.

**Step length.** Separate primal and dual steps: 0.9995 times the largest step that keeps everything positive, capped at 1.

**Starting point.** Mehrotra's heuristic. Take the least-squares solutions of `Ax = b` and `A'y + z = c`, shift them positive by 1.5 times their most negative entry, then shift again so that no complementarity product is far from the average.

**Stopping.** Relative primal residual, relative dual residual and relative gap `|c'x - b'y + u'v| / (1 + |c'x|)` all below 1e-8. Separately, `kkt_check` measures the same three things on the original problem after postsolve. That catches errors in presolve, scaling or the standard-form mapping that the inner test cannot see.

## 3. The factorization

Late in a run, Theta has entries near 1e+14 (basic columns, `x_j/z_j` large) and near 1e-14 (nonbasic columns), so `A Theta A'` is badly conditioned. If A has dependent rows, and many Netlib problems do, it is singular. A standard Cholesky fails.

`cholesky_lipsol` is a blocked right-looking Cholesky:
- Within a block of 96 columns it eliminates column by column.
- It then applies the panel with a triangular solve and updates the trailing matrix with one matrix product, so most of the work runs in BLAS.

When a pivot collapses, it is replaced by 1e128 instead of failing. That happens either below 1e-30 times the largest diagonal, or below 1e-14 times that row's own diagonal before elimination. The replaced pivot sets the matching component of dy to zero, which is the right thing for a dependent row. This trick comes from LIPSOL (Zhang, 1996). Three steps of iterative refinement against the exact operator follow each solve.

## 4. Free columns

The textbook way to handle a free variable is `x = x+ - x-`. Here that failed on random LPs. Both halves grew to about 10-18 while their difference stayed small. Their duals went to zero, their Theta entries blew up, and the resulting rank-one term in `A Theta A'` swamped everything else. The primal residual then stalled at 1e-5.

Free columns are now kept as they are. They have no z and no complementarity. `Theta^{-1}` gets a small proximal term rho instead of `z_j/x_j`. This leaves an error of size `rho * |dx_j|` in the dual equation, which later iterations remove.

## 5. Convex QPs

With `x'Qx/2` in the objective, the dual residual gains `+Qx` and the Newton system becomes

    [ -(Q + Theta^{-1})   A' ] [dx]   [rt]
    [        A          0  ] [dy] = [rb]

`A (Q + Theta^{-1})^{-1} A'` is no longer cheap to form, so `AugmentedSystem` factors this matrix directly. It uses dense LU when the problem is small or Q is dense (the SVM case), and SuperLU otherwise. A 1e-10 shift in the (2,2) block keeps it nonsingular when A has dependent rows. Iterative refinement removes the shift's effect. Everything else in the loop is shared with the LP path.

## 6. MDPs as linear programs

Take an MDP with transient states S, actions A, transitions P and expected one-step reward r. Probability mass that enters a terminal state leaves the system, and the terminal payoff is folded into r. Let `rho(s,a)` be the expected discounted number of times action a is taken in state s. Any policy's rho satisfies the flow equations

    sum_a rho(s',a) - gamma * sum_{s,a} P(s'|s,a) rho(s,a) = mu0(s')    for every s'

Conversely, every nonnegative solution comes from the policy `pi(a|s) = rho(s,a) / sum_a rho(s,a)`. The expected return is `r'rho`, so the MDP is an LP over rho.

Its dual is

    minimize mu0'V  s.t.  V(s) >= r(s,a) + gamma * sum_s' P(s'|s,a) V(s')

That is the Bellman inequality, and the smallest V satisfying it is the optimal value function. So the duals of the flow rows are V. On the MDL grid world they match value iteration to 5e-11.

Any expectation that is linear in rho can be constrained. With gamma = 1, `sum rho(s,a) P(lava | s,a)` is the probability of ever entering lava. Bounding it adds one row. An LP with one extra row has a vertex solution that randomizes in at most one state. Value iteration, by contrast, only produces deterministic policies. That is why penalty tuning jumps from 2.8% risk straight to 31.9%.

One detail: an interior point method converges to the centre of the optimal face, not a vertex. Where several actions are exactly tied, for example two equally short ways around an open area, the returned policy splits between them. The return and risk are the same either way. The randomization count reported by `cmdp_lava.py` includes these ties.

## 7. What broke on the way, and why

Each item was found on a specific test and fixed by changing one thing.

1. **Infinite bounds in presolve.** A singleton row produced `l = -inf, u = 1.5`. The check "is `u - l` tiny" evaluated `inf <= 1e-10 * inf`, which is True, and the column was fixed at minus infinity. Two random LPs gave a false "infeasible" and a wrong "optimal". Fix: test for tiny width only when both bounds are finite.
2. **Split free variables.** See section 4. This affected about 1% of random LPs.
3. **Diagonal regularization of the normal matrix.** Adding `1e-12 * max(diag)` to `A Theta A'` looked like cheap insurance. It lifted the pivots of dependent rows just above the skip threshold. The solve then amplified rounding noise in exactly the directions the skip is meant to zero, and the normal-equation residual came out as large as the right-hand side. Removing it took the random-LP failures from 3/300 to 0/300.
4. **Primal regularization of 1e-10.** This caps Theta at 1e10. When a column hits the cap, its dual residual cannot go below about `1e-10 * |dx_j|`. On FINNIS and PILOT the dual residual sat at about 5e-8 while the objective drifted by a constant amount each iteration. Removing the cap entirely broke GREENBEA instead: 340 of 2316 pivots were skipped and the steps collapsed to 1e-50.
5. **Pivot test relative to the largest diagonal only.** With Theta spanning 40 orders of magnitude, rows whose scale is far below the largest one were either never caught or caught wholesale. Adding a test relative to each row's own diagonal (1e-14), together with a cap of 1e14 on Theta (reg 1e-14), solved all ten hard problems in the sweep. Neighbouring values from 3e-15 to 1e-14 behave the same; larger caps lose GREENBEA.

6. **DFL001 is still open.** With the defaults it reaches a 1e-4 gap by iteration 30. It then crawls, skipping 70 to 90 pivots per factorization. A primal regularization of 1e-15 with a pivot tolerance of 1e-16 solves it in 69 iterations but leaves DEGEN3 3e-8 short of tolerance. Using 1e-15 for both keeps DEGEN3 but leaves DFL001 at 1.2e-8 after 200 iterations. A retry with a second setting after a failure would probably reach 91 of 91, but that tunes the solver to this benchmark, so it is left out.

The sweeps behind items 3 to 6 used 300 random LPs with every row and bound type, checked against HiGHS, plus the hard Netlib problems: FINNIS, GREENBEA, GREENBEB, PILOT, DEGEN3, PILOT87, CYCLE, PEROLD, D2Q06C, WOOD1P.

## References

- S. Mehrotra, On the implementation of a primal-dual interior point method, SIAM J. Optim. 2(4), 1992.
- J. Gondzio, Multiple centrality corrections in a primal-dual method for linear programming, Comput. Optim. Appl. 6, 1996.
- Y. Zhang, Solving large-scale linear programs by interior-point methods under the MATLAB environment (LIPSOL), Optim. Methods Softw. 10, 1998.
- S. J. Wright, Primal-Dual Interior-Point Methods, SIAM, 1997.
- E. D. Andersen and K. D. Andersen, Presolving in linear programming, Math. Programming 71, 1995.
- M. L. Puterman, Markov Decision Processes, Wiley, 1994, chapter 6.9 (linear programming).
- E. Altman, Constrained Markov Decision Processes, Chapman & Hall, 1999.
