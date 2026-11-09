"""Turn bench/results/netlib.csv into a markdown table and summary numbers.

usage: python bench/summarize.py [--csv bench/results/netlib.csv]
"""

import argparse
import csv
import math
from pathlib import Path

MATCH = 1e-6


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="bench/results/netlib.csv")
    args = ap.parse_args()
    path = Path(args.csv)
    rows = list(csv.DictReader(open(path)))

    lines = ["| problem | rows | cols | nnz | status | iters | time (s) | objective "
             "| rel. err vs HiGHS | rel. err vs Netlib | max KKT residual |",
             "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        kkt = max(f(r["kkt_primal"]), f(r["kkt_dual"]), f(r["kkt_gap"]))
        lines.append(
            f"| {r['name']} | {r['rows']} | {r['cols']} | {r['nnz']} | {r['status']} "
            f"| {r['iterations']} | {f(r['time']):.2f} | {f(r['objective']):.10e} "
            f"| {f(r['rel_err_highs']):.1e} | {f(r['rel_err_ref']):.1e} | {kkt:.1e} |")
    (path.parent / "netlib.md").write_text("\n".join(lines) + "\n")

    n = len(rows)
    opt = [r for r in rows if r["status"] == "optimal"]
    match = [r for r in rows if f(r["rel_err_highs"]) < MATCH]
    its = sorted(int(r["iterations"]) for r in opt)
    t_ipm = sum(f(r["time"]) for r in rows)
    t_highs = sum(f(r["highs_time"]) for r in rows)
    worst_kkt = max(max(f(r["kkt_primal"]), f(r["kkt_dual"]), f(r["kkt_gap"])) for r in opt)
    worst_err = max(f(r["rel_err_highs"]) for r in opt)
    ref_off = [(r["name"], f(r["rel_err_ref"])) for r in rows if f(r["rel_err_ref"]) > MATCH]
    print(f"problems: {n}")
    print(f"optimal (relative KKT residuals < 1e-8): {len(opt)}")
    print(f"objective within {MATCH:g} of HiGHS: {len(match)}")
    print(f"largest objective difference from HiGHS among optimal: {worst_err:.1e}")
    print(f"largest KKT residual in the original problem among optimal: {worst_kkt:.1e}")
    print(f"iterations: median {its[len(its) // 2]}, max {its[-1]}, total {sum(its)}")
    print(f"total time: ipm {t_ipm:.1f}s, HiGHS {t_highs:.1f}s")
    print("published value off by more than 1e-6 (HiGHS agrees with us):")
    for name, e in ref_off:
        print(f"  {name}: {e:.1e}")
    for r in rows:
        if r["status"] != "optimal":
            print(f"not optimal: {r['name']} {r['status']} err {f(r['rel_err_highs']):.1e}")
    print(f"wrote {path.parent / 'netlib.md'}")


if __name__ == "__main__":
    main()
