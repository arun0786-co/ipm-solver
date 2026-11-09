"""Solve an MPS file from the command line.

usage: python -m ipm problem.mps[.gz] [--tol 1e-8] [--no-presolve] [-v]
"""

import argparse

from . import read_mps, solve


def main():
    ap = argparse.ArgumentParser(prog="python -m ipm")
    ap.add_argument("path")
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--no-presolve", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    lp = read_mps(args.path)
    m, n = lp.shape
    print(f"{lp.name or args.path}: {m} rows, {n} columns, {lp.A.nnz} nonzeros")
    res = solve(lp, tol=args.tol, presolve_on=not args.no_presolve,
                max_iter=args.max_iter, verbose=args.verbose)
    print(f"status      {res.status}")
    print(f"objective   {res.objective:.12e}")
    print(f"iterations  {res.iterations}")
    print(f"time        {res.time:.3f} s")
    if res.kkt:
        k = res.kkt
        print(f"kkt         primal {k['primal']:.1e}  dual {k['dual']:.1e}  gap {k['gap']:.1e}")


if __name__ == "__main__":
    main()
