"""Solve every Netlib LP and compare against the published optimum and HiGHS.

usage: python bench/run_netlib.py [--data data/netlib] [--only afiro,adlittle]
                                  [--jobs 8] [--no-highs] [--out bench/results]

The published values come from the Netlib readme. Some of them are known
to be off in the last digits (the readme itself lists CPLEX and MINOS
disagreeing), so HiGHS, run on the same parsed model, is the second check.
"""

import argparse
import csv
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MATCH_TOL = 1e-6


def reference_values(readme):
    vals = {}
    inside = False
    for line in Path(readme).read_text(errors="replace").splitlines():
        if "PROBLEM SUMMARY TABLE" in line:
            inside = True
            continue
        if "BOUND-TYPE TABLE" in line:
            break
        if not inside:
            continue
        parts = line.split()
        if len(parts) >= 5 and re.match(r"^[-+]?\d\.\d+E[-+]\d+$", parts[-1]):
            vals[parts[0].lower()] = float(parts[-1])
    return vals


def highs_objective(lp):
    from scipy.optimize import linprog
    import scipy.sparse as sp

    A = lp.A
    eq = np.isfinite(lp.rl) & np.isfinite(lp.ru) & (lp.rl == lp.ru)
    up = ~eq & np.isfinite(lp.ru)
    lo = ~eq & np.isfinite(lp.rl)
    A_ub = sp.vstack([A[up], -A[lo]], format="csr")
    b_ub = np.concatenate([lp.ru[up], -lp.rl[lo]])
    bounds = np.column_stack([lp.l, lp.u])
    bounds = [(None if np.isneginf(a) else a, None if np.isposinf(b) else b)
              for a, b in bounds]
    t = time.perf_counter()
    r = linprog(lp.c, A_ub=A_ub if A_ub.shape[0] else None,
                b_ub=b_ub if A_ub.shape[0] else None,
                A_eq=A[eq] if eq.any() else None,
                b_eq=lp.rl[eq] if eq.any() else None,
                bounds=bounds, method="highs")
    dt = time.perf_counter() - t
    if r.status != 0:
        return np.nan, dt, r.message
    return float(r.fun + lp.c0), dt, ""


def run_one(path, use_highs):
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    import ipm

    name = Path(path).name.replace(".mps.gz", "").replace(".mps", "")
    lp = ipm.read_mps(path)
    res = ipm.solve(lp)
    row = dict(
        name=name, rows=lp.shape[0], cols=lp.shape[1], nnz=lp.A.nnz,
        status=res.status, objective=res.objective, iterations=res.iterations,
        time=res.time, kkt_primal=res.kkt.get("primal", np.nan),
        kkt_dual=res.kkt.get("dual", np.nan), kkt_gap=res.kkt.get("gap", np.nan),
    )
    if use_highs:
        obj, dt, msg = highs_objective(lp)
        row.update(highs_objective=obj, highs_time=dt)
    return row


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/netlib")
    ap.add_argument("--only", default="")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--no-highs", action="store_true")
    ap.add_argument("--out", default="bench/results")
    args = ap.parse_args()

    refs = reference_values(Path(args.data) / "readme.txt")
    files = sorted(Path(args.data).glob("*.mps*"))
    if args.only:
        keep = set(args.only.lower().split(","))
        files = [f for f in files if f.name.split(".mps")[0] in keep]

    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(run_one, str(f), not args.no_highs): f for f in files}
        for fut in as_completed(futs):
            try:
                r = fut.result()
            except Exception as e:  # keep going, record the failure
                f = futs[fut]
                r = dict(name=f.name.split(".mps")[0], status=f"error: {e}")
            ref = refs.get(r["name"].replace("vtpbase", "vtp.base"))
            r["reference"] = ref if ref is not None else np.nan
            obj = r.get("objective", np.nan)
            r["rel_err_ref"] = rel(obj, ref) if ref is not None else np.nan
            h = r.get("highs_objective", np.nan)
            r["rel_err_highs"] = rel(obj, h) if np.isfinite(h) else np.nan
            rows.append(r)
            print(f"{r['name']:10s} {r['status']:9s} it={r.get('iterations', '-')!s:>4} "
                  f"t={r.get('time', np.nan):7.2f}s obj={obj:+.10e} "
                  f"ref_err={r['rel_err_ref']:.1e} highs_err={r['rel_err_highs']:.1e}",
                  flush=True)

    rows.sort(key=lambda r: r["name"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    keys = ["name", "rows", "cols", "nnz", "status", "iterations", "time",
            "objective", "reference", "rel_err_ref", "highs_objective",
            "rel_err_highs", "highs_time", "kkt_primal", "kkt_dual", "kkt_gap"]
    with open(out / "netlib.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    n = len(rows)
    opt = sum(r["status"] == "optimal" for r in rows)
    match = sum((r.get("rel_err_highs", np.nan) < MATCH_TOL) or
                (r.get("rel_err_ref", np.nan) < MATCH_TOL) for r in rows)
    print(f"\n{opt}/{n} reported optimal, {match}/{n} match HiGHS or the "
          f"published value to {MATCH_TOL:g}")


if __name__ == "__main__":
    main()
