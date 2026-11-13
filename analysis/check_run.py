# sanity check of one run: all scalars finite, expected tags once per iteration, lr shown
# usage: python check_run.py RUN [--expect gate1b]   (a run folder, or a runs/ folder holding one run)
import argparse
import math
import sys

from compare_runs import PRESETS, find_run, load

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--expect", choices=sorted(PRESETS), help="require these tags, one value per iteration")
ap.add_argument("--allow-nan", nargs="*", default=["signals/gns_simple"], help="tags where nan is expected")
a = ap.parse_args()

run = find_run(a.run)
tags = load(run)
print(run)
bad = 0
lr = tags.get("charts/learning_rate", [])
steps = [st for st, _ in lr]
if not steps:
    print("  BAD   no charts/learning_rate")
    bad += 1
for tag, vals in sorted(tags.items()):
    nan = sum(math.isnan(v) for _, v in vals)
    inf = sum(math.isinf(v) for _, v in vals)
    if inf or (nan and tag not in a.allow_nan):
        print(f"  BAD   {tag}: {nan} nan, {inf} inf out of {len(vals)}")
        bad += 1
    elif nan:
        print(f"  note  {tag}: {nan} of {len(vals)} nan (allowed)")
if a.expect and steps:
    for tag in PRESETS[a.expect]:
        got = [st for st, _ in tags.get(tag, [])]
        if got != steps:
            print(f"  BAD   {tag}: {len(got)} values, expected {len(steps)} at the iteration steps")
            bad += 1
if steps:
    print(f"  {len(steps)} iterations, steps {steps[0]} to {steps[-1]}")
    print(f"  learning rate: first {lr[0][1]:.4g}, last {lr[-1][1]:.4g}")
print("PASS: all values finite" + (", expected tags present" if a.expect else "") if not bad else f"FAIL: {bad} problem(s)")
sys.exit(1 if bad else 0)
