# where the time goes in a run: totals of the time/ tags
# usage: python time_breakdown.py RUN   (a run folder, or a runs/ folder holding one run)
import sys

from compare_runs import find_run, load

run = find_run(sys.argv[1])
tags = load(run)
parts = {t: sum(v for _, v in vals) for t, vals in tags.items() if t.startswith("time/")}
total = sum(parts.values())
print(run)
for t, v in sorted(parts.items()):
    print(f"  {t:16s} {v:8.1f} s  {100 * v / total:5.1f}%")
print(f"  {'total':16s} {total:8.1f} s")
