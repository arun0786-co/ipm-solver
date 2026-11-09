"""Grid-world MDPs solved as linear programs over occupancy measures.

For an MDP with transient states S, actions A, substochastic transitions
P (probability mass that enters a terminal state leaves the system) and
expected one-step reward r, the occupancy measure rho(s, a) is the
expected (discounted) number of times action a is taken in state s.
Every stationary policy gives a rho that satisfies the flow equations

    sum_a rho(s', a) - gamma * sum_{s,a} P(s' | s, a) rho(s, a) = mu0(s')

and every nonnegative solution of them comes from some policy, namely
pi(a | s) = rho(s, a) / sum_a rho(s, a). The expected return is linear in
rho, so the MDP is the LP

    max  sum r(s,a) rho(s,a)   s.t. flow equations, rho >= 0

and the duals of the flow equations are the optimal values V(s). Any other
expectation that is linear in rho, such as the probability of ending in
a bad state, can be added as one more row. That is a constrained MDP,
which value iteration cannot express.
"""

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from ipm import LinearProgram, solve

MOVES = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}
ACTIONS = list(MOVES)
PERP = {"up": ("left", "right"), "down": ("left", "right"),
        "left": ("up", "down"), "right": ("up", "down")}


@dataclass
class GridWorld:
    width: int
    height: int
    walls: set
    terminals: dict            # cell -> terminal reward
    step_reward: float = -0.04
    p: float = 0.8             # probability of moving as intended
    states: list = field(init=False)

    def __post_init__(self):
        self.states = [(x, y) for x in range(self.width) for y in range(self.height)
                       if (x, y) not in self.walls and (x, y) not in self.terminals]
        self.index = {s: i for i, s in enumerate(self.states)}
        self.term_list = list(self.terminals)
        self.term_index = {t: i for i, t in enumerate(self.term_list)}
        self._build()

    def _move(self, s, a):
        dx, dy = MOVES[a]
        t = (s[0] + dx, s[1] + dy)
        if not (0 <= t[0] < self.width and 0 <= t[1] < self.height) or t in self.walls:
            return s
        return t

    def outcomes(self, s, a):
        slip = (1 - self.p) / 2
        out = [(self._move(s, a), self.p)]
        out += [(self._move(s, b), slip) for b in PERP[a]]
        return [(t, q) for t, q in out if q > 0]

    def _build(self):
        nS, nA, nT = len(self.states), len(ACTIONS), len(self.term_list)
        rows, cols, vals = [], [], []       # P[(s,a), s'] for transient s'
        trows, tcols, tvals = [], [], []    # T[(s,a), t] for terminals
        for s in self.states:
            for k, a in enumerate(ACTIONS):
                sa = self.index[s] * nA + k
                for t, q in self.outcomes(s, a):
                    if t in self.terminals:
                        trows.append(sa); tcols.append(self.term_index[t]); tvals.append(q)
                    else:
                        rows.append(sa); cols.append(self.index[t]); vals.append(q)
        self.P = sp.csr_matrix((vals, (rows, cols)), shape=(nS * nA, nS))
        self.T = sp.csr_matrix((tvals, (trows, tcols)), shape=(nS * nA, nT))
        self.term_reward = np.array([self.terminals[t] for t in self.term_list])

    @property
    def n_states(self):
        return len(self.states)

    def reward(self, gamma):
        """Expected reward of (s, a): step reward plus discounted terminal payoff."""
        return self.step_reward + gamma * (self.T @ self.term_reward)

    # ---------------------------------------------------------------- solvers
    def value_iteration(self, gamma, tol=1e-12, max_iter=100000):
        nS, nA = self.n_states, len(ACTIONS)
        r = self.reward(gamma)
        V = np.zeros(nS)
        for it in range(max_iter):
            Q = (r + gamma * (self.P @ V)).reshape(nS, nA)
            Vn = Q.max(axis=1)
            if np.max(np.abs(Vn - V)) < tol:
                return Vn, Q.argmax(axis=1), it
            V = Vn
        return V, Q.argmax(axis=1), max_iter

    def occupancy_lp(self, gamma, mu0, extra_rows=(), extra_ub=()):
        """LP over rho (maximization written as minimization of -reward).

        extra_rows / extra_ub add constraints  row @ rho <= ub.
        """
        nS, nA = self.n_states, len(ACTIONS)
        E = sp.kron(sp.identity(nS), np.ones((1, nA)), format="csr")  # sum_a
        flow = E - gamma * self.P.T
        c = -self.reward(gamma)
        rows = [flow]
        rl = [mu0]
        ru = [mu0]
        for row, ub in zip(extra_rows, extra_ub):
            rows.append(sp.csr_matrix(row))
            rl.append(np.array([-np.inf]))
            ru.append(np.array([ub]))
        A = sp.vstack(rows, format="csr")
        return LinearProgram(c=c, A=A, rl=np.concatenate(rl), ru=np.concatenate(ru),
                             l=np.zeros(nS * nA), u=np.full(nS * nA, np.inf))

    def solve_lp(self, gamma, mu0, extra_rows=(), extra_ub=(), **kw):
        lp = self.occupancy_lp(gamma, mu0, extra_rows, extra_ub)
        res = solve(lp, **kw)
        nS, nA = self.n_states, len(ACTIONS)
        rho = np.maximum(res.x, 0.0).reshape(nS, nA)
        V = -res.y[:nS]        # duals of the flow rows (sign from max -> min)
        return res, rho, V

    # ------------------------------------------------------------- policies
    @staticmethod
    def policy_from_occupancy(rho, eps=1e-9):
        tot = rho.sum(axis=1, keepdims=True)
        pi = np.where(tot > eps, rho / np.maximum(tot, eps), 1.0 / rho.shape[1])
        return pi

    def risk_row(self, cells, gamma=1.0):
        """Row r with r @ rho = (discounted) probability of entering cells."""
        idx = [self.term_index[c] for c in cells]
        return gamma * np.asarray(self.T[:, idx].sum(axis=1)).ravel()

    def evaluate(self, pi, gamma, start):
        """Exact return, expected steps and terminal probabilities of pi."""
        nS, nA = self.n_states, len(ACTIONS)
        Ppi = sp.csr_matrix(self.P.multiply(pi.reshape(-1, 1)))
        # rows (s,a) -> sum over a with weights pi
        E = sp.kron(sp.identity(nS), np.ones((1, nA)), format="csr")
        Pss = E @ Ppi                             # nS x nS
        Tss = E @ sp.csr_matrix(self.T.multiply(pi.reshape(-1, 1)))
        rs = E @ (self.reward(gamma) * pi.ravel())
        I = sp.identity(nS, format="csc")
        N = sp.linalg.splu((I - gamma * Pss).tocsc())
        V = N.solve(rs)
        e0 = np.zeros(nS)
        e0[self.index[start]] = 1.0
        visits = sp.linalg.splu((I - Pss).T.tocsc()).solve(e0)  # undiscounted
        return dict(value=V[self.index[start]], steps=visits.sum(),
                    terminal=dict(zip(self.term_list, visits @ Tss)))

    def simulate(self, pi, start, episodes, rng, max_steps=100000):
        """Monte Carlo rollouts of pi, all episodes stepped together.

        Returns the fraction of episodes ending in each terminal and the
        array of episode lengths.
        """
        nS, nA = self.n_states, len(ACTIONS)
        # outcome table: next state id (transient index, or nS + terminal id)
        nxt = np.zeros((nS * nA, 3), dtype=np.int64)
        prob = np.zeros((nS * nA, 3))
        for s in self.states:
            for k, a in enumerate(ACTIONS):
                sa = self.index[s] * nA + k
                for j, (t, q) in enumerate(self.outcomes(s, a)):
                    nxt[sa, j] = (nS + self.term_index[t]) if t in self.terminals \
                        else self.index[t]
                    prob[sa, j] = q
        cum_pi = np.cumsum(pi, axis=1)
        cum_pr = np.cumsum(prob, axis=1)
        state = np.full(episodes, self.index[start])
        steps = np.zeros(episodes, dtype=np.int64)
        alive = np.ones(episodes, dtype=bool)
        for _ in range(max_steps):
            idx = np.flatnonzero(alive)
            if idx.size == 0:
                break
            s = state[idx]
            a = (rng.random(idx.size)[:, None] > cum_pi[s]).sum(axis=1)
            a = np.minimum(a, nA - 1)
            sa = s * nA + a
            k = (rng.random(idx.size)[:, None] > cum_pr[sa]).sum(axis=1)
            k = np.minimum(k, 2)
            state[idx] = nxt[sa, k]
            steps[idx] += 1
            alive[idx] = state[idx] < nS
        ends = {t: float(np.mean(state == nS + i)) for i, t in enumerate(self.term_list)}
        return ends, steps
