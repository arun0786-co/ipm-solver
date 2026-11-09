"""Plot residuals per iteration for a few Netlib problems.

usage: python bench/plot_convergence.py [--out docs/img/convergence.png]
"""

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ipm  # noqa: E402


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/netlib")
    ap.add_argument("--out", default="docs/img/convergence.png")
    ap.add_argument("--problems", default="afiro,25fv47,pilot87,greenbea")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    names = args.problems.split(",")
    fig, axes = plt.subplots(1, len(names), figsize=(3.0 * len(names), 2.9), sharey=True)
    for ax, name in zip(axes, names):
        lp = ipm.read_mps(Path(args.data) / f"{name}.mps.gz")
        r = ipm.solve(lp)
        h = r.info["history"]
        it = [e["it"] for e in h]
        ax.semilogy(it, [e["relp"] + 1e-17 for e in h], label="primal residual")
        ax.semilogy(it, [e["reld"] + 1e-17 for e in h], label="dual residual")
        ax.semilogy(it, [e["gap"] + 1e-17 for e in h], label="relative gap")
        ax.axhline(1e-8, color="0.5", lw=0.8, ls="--")
        ax.set_title(f"{name.upper()} ({lp.shape[0]}x{lp.shape[1]})", fontsize=9)
        ax.set_xlabel("iteration")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("relative residual")
    axes[0].legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
