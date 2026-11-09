"""Chance-constrained navigation: maximize return subject to P(lava) <= delta.

A 12x7 grid. The agent starts in the bottom-left corner, the goal (+1) is
in the bottom-right corner, and a strip of lava (-1, episode ends) sits
one row above the floor. Moves succeed with probability 0.9 and slip to
either side with 0.05 each. Each step costs 0.08.

The unconstrained optimum runs along the floor next to the lava: short,
but it ends in lava about a third of the time. This script

  1. traces the exact return-vs-risk frontier by adding the risk row
     to the occupancy LP and sweeping delta,
  2. compares it with the usual workaround, which is to raise the lava
     penalty and rerun value iteration (only deterministic policies, so
     only the corner points of the frontier are reachable),
  3. checks one constrained policy with Monte Carlo rollouts.

usage: python examples/cmdp_lava.py [--out docs/img] [--episodes 200000]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mdp import ACTIONS, GridWorld  # noqa: E402

W, H = 12, 7
START = (0, 0)
GOAL = (W - 1, 0)


def make_world(lava_reward=-1.0):
    terminals = {(x, 1): lava_reward for x in range(2, W - 2)}
    terminals[GOAL] = 1.0
    return GridWorld(W, H, walls=set(), terminals=terminals,
                     step_reward=-0.08, p=0.9)


def lava_cells(g):
    return [t for t in g.terminals if t != GOAL]


def evaluate(g, pi):
    ev = g.evaluate(pi, 1.0, START)
    risk = sum(v for t, v in ev["terminal"].items() if t != GOAL)
    return ev["value"], risk, ev["steps"]


def randomized_states(g, rho, tol=1e-6):
    """States visited with nonnegligible probability where pi is not deterministic."""
    visits = rho.sum(axis=1)
    pi = g.policy_from_occupancy(rho)
    mixed = (visits > 1e-6) & (pi.max(axis=1) < 1 - tol)
    return [g.states[i] for i in np.flatnonzero(mixed)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/img")
    ap.add_argument("--episodes", type=int, default=200_000)
    args = ap.parse_args()

    g = make_world()
    mu0 = np.zeros(g.n_states)
    mu0[g.index[START]] = 1.0
    risk = g.risk_row(lava_cells(g))

    # unconstrained optimum
    res, rho, V = g.solve_lp(1.0, mu0)
    pi0 = g.policy_from_occupancy(rho)
    ret0, risk0, steps0 = evaluate(g, pi0)
    print(f"unconstrained: return {ret0:.4f}  P(lava) {risk0:.4f}  steps {steps0:.2f}")

    # 1. exact frontier
    deltas = np.linspace(0.0, risk0, 33)
    frontier = []
    t = time.perf_counter()
    for d in deltas:
        res, rho, _ = g.solve_lp(1.0, mu0, extra_rows=[risk], extra_ub=[d])
        assert res.status == "optimal", res
        pi = g.policy_from_occupancy(rho)
        ret, rk, st = evaluate(g, pi)
        frontier.append((d, ret, rk, st, len(randomized_states(g, rho)),
                         res.iterations, res.kkt))
    dt = time.perf_counter() - t
    print(f"\n{len(deltas)} constrained LPs in {dt:.2f}s "
          f"({g.n_states * len(ACTIONS)} variables each)")
    print(" delta   return  P(lava)  steps  mixed-states  iters")
    for d, ret, rk, st, mixed, it, _ in frontier[::4]:
        print(f"{d:6.3f} {ret:8.4f} {rk:8.4f} {st:6.2f} {mixed:6d} {it:10d}")
    worst = max(max(f[6]["primal"], f[6]["dual"], f[6]["gap"]) for f in frontier)
    print(f"largest KKT residual over the sweep: {worst:.1e}")
    print(f"largest |P(lava) - delta| for active budgets: "
          f"{max(abs(f[2] - f[0]) for f in frontier[1:-1]):.1e}")

    # 2. penalty tuning + value iteration
    penalty = []
    for lam in np.concatenate([np.linspace(0, 2, 41), np.linspace(2, 40, 39)]):
        gl = make_world(lava_reward=-1.0 - lam)
        Vl, pol, _ = gl.value_iteration(0.999999, tol=1e-12)
        pi = np.eye(len(ACTIONS))[pol]
        ret, rk, st = evaluate(g, pi)     # score under the original reward
        penalty.append((lam, ret, rk))
    distinct = sorted({(round(r, 6), round(k, 6)) for _, r, k in penalty})
    print(f"\npenalty tuning over {len(penalty)} penalties reached "
          f"{len(distinct)} distinct (return, risk) points:")
    for r, k in distinct:
        print(f"  return {r:8.4f}  P(lava) {k:.4f}")

    # 3. Monte Carlo check at delta = 0.1
    d = 0.10
    res, rho, _ = g.solve_lp(1.0, mu0, extra_rows=[risk], extra_ub=[d])
    pi = g.policy_from_occupancy(rho)
    ret, rk, st = evaluate(g, pi)
    rng = np.random.default_rng(0)
    ends, steps = g.simulate(pi, START, args.episodes, rng)
    p_lava = sum(v for t, v in ends.items() if t != GOAL)
    se = np.sqrt(p_lava * (1 - p_lava) / args.episodes)
    mc_ret = ends[GOAL] * 1.0 - p_lava * 1.0 - 0.08 * steps.mean()
    print(f"\nMonte Carlo, delta = {d}, {args.episodes} episodes:")
    print(f"  P(lava)  LP {rk:.4f}   simulated {p_lava:.4f} +- {1.96 * se:.4f}")
    print(f"  steps    LP {st:.3f}  simulated {steps.mean():.3f}")
    print(f"  return   LP {ret:.4f}  simulated {mc_ret:.4f}")
    print(f"  randomized states: {randomized_states(g, rho)}")

    plot(g, args.out, frontier, penalty, mu0, risk)


def plot(g, out, frontier, penalty, mu0, risk):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    rk = [f[2] for f in frontier]
    ret = [f[1] for f in frontier]
    ax.plot(rk, ret, "-", color="#1f4e79", lw=2, label="occupancy LP (exact frontier)")
    pr = np.array([(k, r) for _, r, k in penalty])
    ax.plot(pr[:, 0], pr[:, 1], "o", ms=7, mfc="none", mec="#c0392b", mew=1.5,
            label="lava penalty + value iteration")
    ax.set_xlabel("probability of ending in lava")
    ax.set_ylabel("expected return")
    ax.set_title("Return vs risk on the 12x7 lava grid")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "cmdp_frontier.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))
    for ax, d in zip(axes, [frontier[-1][0], 0.10, 0.0]):
        if d == frontier[-1][0]:
            res, rho, _ = g.solve_lp(1.0, mu0)
            title = f"unconstrained (P(lava) = {d:.2f})"
        else:
            res, rho, _ = g.solve_lp(1.0, mu0, extra_rows=[risk], extra_ub=[d])
            title = f"P(lava) <= {d:.2f}"
        img = np.full((H, W), np.nan)
        for s, i in g.index.items():
            img[s[1], s[0]] = rho[i].sum()
        ax.imshow(img, origin="lower", cmap="Blues", vmin=0, vmax=1.2)
        for t in g.terminals:
            color = "#2e7d32" if t == GOAL else "#c0392b"
            ax.add_patch(plt.Rectangle((t[0] - 0.5, t[1] - 0.5), 1, 1, color=color))
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Expected visits per cell (red: lava, green: goal)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "cmdp_visits.png", dpi=160)
    plt.close(fig)
    print(f"\nwrote {out / 'cmdp_frontier.png'} and {out / 'cmdp_visits.png'}")


if __name__ == "__main__":
    main()
