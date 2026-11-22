# oracle summary, exactly as in docs/oracle_preregistration.md
# block A: python oracle_summary.py ROOT --block A [--points 10 30 50 70] [--out-csv FILE]
# block B: python oracle_summary.py ROOT --block B [--out-csv FILE]
# ROOT is runs_oracle; expects ROOT/HalfCheetah_s{seed}/source and .../at{point}_{name}
# run only after every validity check for the points or block has passed
import argparse
import csv
import itertools
import os
import sys

import numpy as np

from compare_runs import find_run, load

NUM_STEPS = 2048
THRESHOLD = 150.0
BLOCKS = {
    'A': dict(seeds=[1, 2, 3], points=[10, 30, 50, 70], targets=['sgd', 'rmsprop', 'lion', 'muon', 'adam_reset'],
              carry_seeds=[1, 2, 3]),
    'B': dict(seeds=[6, 7, 8, 9, 10], points=[50], targets=['muon', 'lion', 'adam_reset'], carry_seeds=[6]),
}


def returns(run):
    vals = load(run).get('charts/episodic_return', [])
    if not vals:
        sys.exit(f'no episodic returns in {run}')
    return np.array([s for s, _ in vals]), np.array([v for _, v in vals])


def score(steps, vals):
    # same rule as summarize_runs.py: episodes at or after 90% of the last step
    return float(np.nanmean(vals[steps >= steps.max() * 0.9]))


def window(steps, vals, first_iter, last_iter):
    # mean return of episodes that ended inside these iterations
    lo, hi = (first_iter - 1) * NUM_STEPS, last_iter * NUM_STEPS
    sel = vals[(steps > lo) & (steps <= hi)]
    return float(np.mean(sel)) if len(sel) else float('nan')


def label(d):
    d = np.array(d)
    if np.all(d > 0) and d.mean() >= THRESHOLD:
        return 'promising'
    if np.all(d < 0) and d.mean() <= -THRESHOLD:
        return 'harmful'
    return 'inconclusive'


def bootstrap_ci(d, n=10000, seed=0):
    # 95% percentile interval of the mean, resampling seeds
    rng = np.random.default_rng(seed)
    d = np.array(d)
    means = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


ap = argparse.ArgumentParser()
ap.add_argument('root')
ap.add_argument('--block', choices=sorted(BLOCKS), required=True)
ap.add_argument('--points', type=int, nargs='+', help='block A: only points that fully passed validation')
ap.add_argument('--seeds', type=int, nargs='+', help='only for testing; the design fixes the seeds')
ap.add_argument('--total-steps', type=int, default=1000000, help='only change for a small test')
ap.add_argument('--env', default='HalfCheetah')
ap.add_argument('--out-csv')
a = ap.parse_args()

cfg = BLOCKS[a.block]
seeds = a.seeds or cfg['seeds']
points = a.points or cfg['points']
targets = cfg['targets']
n_iter = a.total_steps // NUM_STEPS

rows, problems = [], []
for pt in points:
    k = int(pt / 100 * n_iter)  # switch iteration, as in ppo.py
    for seed in seeds:
        base = os.path.join(a.root, f'{a.env}_s{seed}')
        src = returns(find_run(os.path.join(base, 'source', 'runs')))
        src_score = score(*src)
        before = window(*src, k - 9, k)
        scores = {}
        # the preregistered carry controls are required, not optional
        need_carry = seed in cfg['carry_seeds']
        if need_carry and not os.path.isdir(os.path.join(base, f'at{pt}_adam_carry')):
            problems.append(f'{pt}% seed {seed}: preregistered adam_carry control missing')
            continue
        names = targets + (['adam_carry'] if need_carry else [])
        gone = [n for n in names if not os.path.isdir(os.path.join(base, f'at{pt}_{n}'))]
        if gone:
            problems.append(f'{pt}% seed {seed}: missing runs {gone}')
            continue
        for name in names:
            st, vl = returns(find_run(os.path.join(base, f'at{pt}_{name}', 'runs')))
            scores[name] = score(st, vl)
            rows.append(dict(point=pt, seed=seed, run=name, score=scores[name], source_score=src_score,
                             delta=scores[name] - src_score, dip=window(st, vl, k + 1, k + 16) - before))
        # exact resume means the carry run has the same final episodes as the source
        if 'adam_carry' in scores and scores['adam_carry'] != src_score:
            problems.append(f'{pt}% seed {seed}: adam_carry {scores["adam_carry"]:.4f} != source {src_score:.4f}')
        for r in rows:
            if r['point'] == pt and r['seed'] == seed:
                r['delta_vs_reset'] = r['score'] - scores['adam_reset']

# stop before any effect is shown or written if the carry check failed
if problems:
    print('VALIDITY PROBLEM, no effects shown and no csv written:')
    for p in problems:
        print('  ' + p)
    sys.exit(1)


def deltas(t, pt, key='delta'):
    return [r[key] for s in seeds for r in rows if r['run'] == t and r['point'] == pt and r['seed'] == s]


print(f'block {a.block}, seeds {seeds}, points {points}')
print('per seed (delta = run - source Adam; dip = 16 iters after switch minus source 10 before)')
print(f'  {"point":>5} {"seed":>4} {"run":<11} {"score":>9} {"delta":>9} {"vs reset":>9} {"dip":>9}')
for r in rows:
    print(f'  {r["point"]:>5} {r["seed"]:>4} {r["run"]:<11} {r["score"]:9.1f} {r["delta"]:9.1f} {r["delta_vs_reset"]:9.1f} {r["dip"]:9.1f}')

if a.block == 'A':
    print('\nscreening per point (descriptive only, not significance)')
    print(f'  {"point":>5} {"target":<11} {"mean delta":>10} {"sd":>7} {"mean vs reset":>14}  label')
    for pt in points:
        for t in targets:
            d = deltas(t, pt)
            sd = np.std(d, ddof=1) if len(d) > 1 else float('nan')
            lab = label(d) if len(d) == 3 else 'needs 3 seeds'
            print(f'  {pt:>5} {t:<11} {np.mean(d):10.1f} {sd:7.1f} {np.mean(deltas(t, pt, "delta_vs_reset")):14.1f}  {lab}')
    print('\nphase rule: Delta(p1) - Delta(p2), paired by seed; a statement needs the same sign on')
    print('all 3 seeds and |mean| >= 150, otherwise inconclusive')
    for t in targets:
        for p1, p2 in itertools.combinations(points, 2):
            diff = np.array(deltas(t, p1)) - np.array(deltas(t, p2))
            ok = len(diff) == 3 and (np.all(diff > 0) or np.all(diff < 0)) and abs(diff.mean()) >= THRESHOLD
            verdict = ('better at %d%%' % (p1 if diff.mean() > 0 else p2)) if ok else 'inconclusive'
            print(f'  {t:<11} {p1:>2}% vs {p2:>2}%: ' + ' '.join(f'{x:+7.1f}' for x in diff) + f'  mean {diff.mean():+7.1f}  {verdict}')
else:
    print('\nconfirmation at 50% (muon primary, lion secondary and exploratory)')
    print(f'  {"target":<11} {"positive":>8} {"mean delta":>10} {"95% CI":>20} {"mean vs reset":>14}  result')
    for t in targets:
        d = deltas(t, 50)
        lo, hi = bootstrap_ci(d)
        pos = sum(x > 0 for x in d)
        if t == 'adam_reset':
            res = 'control'
        elif len(d) < 5:
            res = 'needs 5 seeds'
        else:
            res = ('confirmed' if pos == 5 else 'not confirmed') + (' (exploratory)' if t == 'lion' else '')
        print(f'  {t:<11} {pos:>4}/{len(d):<3} {np.mean(d):10.1f} {f"[{lo:.1f}, {hi:.1f}]":>20} {np.mean(deltas(t, 50, "delta_vs_reset")):14.1f}  {res}')
    print('  rule: Delta > 0 on all 5 seeds (one-sided exact sign test p = 1/32 = 0.031)')

if a.out_csv:
    with open(a.out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'\nwritten {a.out_csv}')
