# pilot summary, exactly as in docs/pilot_preregistration.md
# usage: python pilot_summary.py ROOT [--seeds 1 2 3] [--out-csv FILE]
# ROOT is runs_oracle; expects ROOT/HalfCheetah_s{seed}/source and .../at50_{name}
import argparse
import csv
import os
import sys

import numpy as np

from compare_runs import find_run, load

TARGETS = ["sgd", "rmsprop", "lion", "muon", "adam_reset"]
NUM_STEPS = 2048
SWITCH_ITER = 244
THRESHOLD = 150.0


def returns(run):
    vals = load(run).get("charts/episodic_return", [])
    if not vals:
        sys.exit(f"no episodic returns in {run}")
    return np.array([s for s, _ in vals]), np.array([v for _, v in vals])


def score(steps, vals):
    # same rule as summarize_runs.py: episodes at or after 90% of the last step
    return float(np.nanmean(vals[steps >= steps.max() * 0.9]))


def window(steps, vals, first_iter, last_iter):
    # mean return of episodes that ended inside these iterations
    lo, hi = (first_iter - 1) * NUM_STEPS, last_iter * NUM_STEPS
    sel = vals[(steps > lo) & (steps <= hi)]
    return float(np.mean(sel)) if len(sel) else float("nan")


def label(deltas):
    d = np.array(deltas)
    if np.all(d > 0) and d.mean() >= THRESHOLD:
        return "promising"
    if np.all(d < 0) and d.mean() <= -THRESHOLD:
        return "harmful"
    return "inconclusive"


ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("--env", default="HalfCheetah")
ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
ap.add_argument("--out-csv")
a = ap.parse_args()

rows, problems = [], []
for seed in a.seeds:
    base = os.path.join(a.root, f"{a.env}_s{seed}")
    src = returns(find_run(os.path.join(base, "source", "runs")))
    src_score = score(*src)
    before = window(*src, SWITCH_ITER - 9, SWITCH_ITER)
    scores = {}
    for name in TARGETS + ["adam_carry"]:
        st, vl = returns(find_run(os.path.join(base, f"at50_{name}", "runs")))
        scores[name] = score(st, vl)
        dip = window(st, vl, SWITCH_ITER + 1, SWITCH_ITER + 16) - before
        rows.append(dict(seed=seed, run=name, score=scores[name], source_score=src_score,
                         delta=scores[name] - src_score, dip=dip))
    # exact resume means the carry run has the same final episodes as the source
    if scores["adam_carry"] != src_score:
        problems.append(f"seed {seed}: adam_carry score {scores['adam_carry']:.4f} != source {src_score:.4f}")
    for r in rows:
        if r["seed"] == seed:
            r["delta_vs_reset"] = r["score"] - scores["adam_reset"]

# stop before any effect is shown or written if the carry check failed
if problems:
    print("VALIDITY PROBLEM, no effects shown and no csv written:")
    for p in problems:
        print("  " + p)
    sys.exit(1)

print("per seed (scores; delta = run - source Adam; dip = iters 245-260 minus source 235-244)")
print(f"  {'seed':>4} {'run':<11} {'score':>9} {'delta':>9} {'vs reset':>9} {'dip':>9}")
for r in rows:
    print(f"  {r['seed']:>4} {r['run']:<11} {r['score']:9.1f} {r['delta']:9.1f} {r['delta_vs_reset']:9.1f} {r['dip']:9.1f}")

print("\nscreening (descriptive only, not significance)")
print(f"  {'target':<11} {'mean delta':>10} {'sd':>7} {'mean vs reset':>14}  label")
for t in TARGETS:
    d = [r["delta"] for r in rows if r["run"] == t]
    vr = [r["delta_vs_reset"] for r in rows if r["run"] == t]
    sd = np.std(d, ddof=1) if len(d) > 1 else float("nan")
    print(f"  {t:<11} {np.mean(d):10.1f} {sd:7.1f} {np.mean(vr):14.1f}  {label(d) if len(d) == 3 else 'needs 3 seeds'}")

if a.out_csv:
    with open(a.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwritten {a.out_csv}")

print("\nreminder: run the validity checks (check_run, check_rollouts --first, compare_runs on adam_carry) first")
