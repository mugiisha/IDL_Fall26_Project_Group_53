# compare rollout fingerprints (rollout_hashes.txt) across runs, iteration by iteration
# usage: python check_rollouts.py RUN RUN [RUN ...] [--first]   (run folders, or runs/ folders holding one run)
# default: every shared iteration must match (exact resume)
# --first: only the earliest shared iteration must match (the first rollout after a switch)
import argparse
import os
import sys

from compare_runs import find_run

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--first", action="store_true", help="check only the earliest shared iteration")
a = ap.parse_args()
if len(a.runs) < 2:
    sys.exit("need at least two runs")

runs = [find_run(p) for p in a.runs]
tables = []
for r in runs:
    path = os.path.join(r, "rollout_hashes.txt")
    if not os.path.exists(path):
        sys.exit(f"no rollout_hashes.txt in {r}")
    t = {}
    with open(path) as f:
        for line in f:
            it, _, h = line.split()
            if int(it) in t:
                sys.exit(f"{r}: iteration {it} appears twice")
            t[int(it)] = h
    tables.append(t)
for i, r in enumerate(runs):
    print(f"  {i}: {os.path.basename(r)} (iterations {min(tables[i])}-{max(tables[i])})")

common = sorted(set.intersection(*(set(t) for t in tables)))
if not common:
    sys.exit("FAIL: no iteration in common")
checked = common[:1] if a.first else common
bad = 0
for it in common:
    n = len({t[it] for t in tables})
    note = "" if it in checked else "  (not checked)"
    print(f"  iteration {it:4d}: " + ("same in all runs" if n == 1 else f"{n} different rollouts") + note)
    bad += n > 1 and it in checked
what = f"first shared iteration ({checked[0]})" if a.first else f"all {len(common)} shared iterations"
print(f"PASS: {what} identical" if not bad else f"FAIL: {bad} checked iteration(s) differ")
sys.exit(1 if bad else 0)
