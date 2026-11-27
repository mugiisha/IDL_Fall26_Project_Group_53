# oracle figures from the committed result CSVs (no new runs, no new analysis rules)
# python analysis/plot_oracle.py                       -> delta vs adam source
# python analysis/plot_oracle.py --metric delta_vs_reset  -> delta vs adam_reset (supplement)
# labels and the 150 threshold are the preregistered screening rule, not a significance test
import argparse
import csv
import os

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

THRESHOLD = 150.0
TASKS = [('HalfCheetah', 'results/oracle_blockA_halfcheetah.csv'),
         ('Hopper', 'results/oracle_blockC_hopper.csv')]
TARGETS = ['sgd', 'rmsprop', 'lion', 'muon', 'adam_reset']
NAMES = {'sgd': 'SGD', 'rmsprop': 'RMSProp', 'lion': 'Lion', 'muon': 'Muon', 'adam_reset': 'Adam (state reset)'}
POINTS = [10, 30, 50, 70]
SEEDS = [1, 2, 3]
# categorical slots 1-3, plus a different marker per seed so colour is not the only cue
SEED_STYLE = {1: ('#2a78d6', 'o'), 2: ('#eb6834', 's'), 3: ('#1baf7a', '^')}
INK, MUTED, GRID = '#1f1f1e', '#6b6a64', '#e4e3de'


def load(path):
    rows = {}
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            rows[(int(r['point']), int(r['seed']), r['run'])] = r
    return rows


def label(d):
    # same rule as oracle_summary.py
    d = np.array(d)
    if np.all(d > 0) and d.mean() >= THRESHOLD:
        return 'promising'
    if np.all(d < 0) and d.mean() <= -THRESHOLD:
        return 'harmful'
    return ''


def check(rows, path):
    # every point x seed x arm present, and adam_carry has delta 0 (sanity, as in the audit)
    for p in POINTS:
        for s in SEEDS:
            for t in TARGETS + ['adam_carry']:
                if (p, s, t) not in rows:
                    raise SystemExit(f'{path}: missing point {p} seed {s} {t}')
            if abs(float(rows[(p, s, 'adam_carry')]['delta'])) > 1e-9:
                raise SystemExit(f'{path}: adam_carry delta not 0 at point {p} seed {s}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--metric', choices=['delta', 'delta_vs_reset'], default='delta')
    ap.add_argument('--out', default='plots/oracle')
    args = ap.parse_args()

    targets = TARGETS if args.metric == 'delta' else TARGETS[:-1]
    ylab = {'delta': 'Δ return vs Adam',
            'delta_vs_reset': 'Δ return vs Adam reset'}[args.metric]

    plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7.5, 'axes.labelsize': 7,
                         'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6.5,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.edgecolor': MUTED,
                         'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.labelcolor': INK,
                         'text.color': INK})
    fig, axes = plt.subplots(len(TASKS), len(targets), figsize=(6.75, 3.7), sharex=True,
                             sharey='row', squeeze=False)
    offsets = {1: -2.2, 2: 0.0, 3: 2.2}
    table = []

    for i, (task, path) in enumerate(TASKS):
        rows = load(path)
        check(rows, path)
        for j, t in enumerate(targets):
            ax = axes[i, j]
            d = np.array([[float(rows[(p, s, t)][args.metric]) for s in SEEDS] for p in POINTS])
            mean, sd = d.mean(axis=1), d.std(axis=1, ddof=1)

            if args.metric == 'delta':
                # the 150 band belongs to the preregistered rule on delta vs adam only
                ax.axhspan(-THRESHOLD, THRESHOLD, color='#f1f0ec', zorder=0, lw=0)
            ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
            ax.grid(axis='y', color=GRID, lw=0.5, zorder=0)
            for k, s in enumerate(SEEDS):
                c, m = SEED_STYLE[s]
                x = np.array(POINTS) + offsets[s]
                ax.plot(x, d[:, k], color=c, lw=0.6, alpha=0.35, zorder=2)
                ax.scatter(x, d[:, k], color=c, marker=m, s=14, zorder=3,
                           edgecolors='white', linewidths=0.5)
            ax.errorbar(POINTS, mean, yerr=sd, color=INK, lw=1.4, marker='D', ms=3.5,
                        capsize=2, elinewidth=0.9, zorder=4)

            # screening labels only for the main metric, as preregistered
            if args.metric == 'delta':
                for p, col in zip(POINTS, d):
                    lab = label(col)
                    if lab:
                        # fixed strip at the top so it never sits on a data point
                        ax.text(p, 0.97, 'P' if lab == 'promising' else 'H', ha='center', va='top',
                                transform=ax.get_xaxis_transform(), fontsize=6.5,
                                fontweight='bold', color=INK)
            for p, mu, s_, col in zip(POINTS, mean, sd, d):
                table.append((task, t, p, mu, s_, label(col) if args.metric == 'delta' else ''))

            ax.set_xticks(POINTS)
            ax.set_xticklabels([f'{p}%' for p in POINTS])
            ax.set_xlim(2, 78)
            ax.spines[['top', 'right']].set_visible(False)
            if i == 0:
                ax.set_title(NAMES[t])
            if j == 0:
                ax.set_ylabel(f'{task}\n{ylab}')
            if i == len(TASKS) - 1:
                ax.set_xlabel('switch point')

    # headroom so the P / H strip stays clear of the data
    for row in axes:
        lo, hi = row[0].get_ylim()
        row[0].set_ylim(lo, hi + 0.14 * (hi - lo))

    handles = [Line2D([], [], color=SEED_STYLE[s][0], marker=SEED_STYLE[s][1], lw=0, ms=4,
                      label=f'seed {s}') for s in SEEDS]
    handles.append(Line2D([], [], color=INK, marker='D', ms=3.5, lw=1.4, label='mean ± 1 sd (n = 3)'))
    if args.metric == 'delta':
        handles.append(plt.Rectangle((0, 0), 1, 1, color='#f1f0ec', label=f'|Δ| < {THRESHOLD:.0f} (screening band)'))
        handles.append(Line2D([], [], lw=0, label='P / H = promising / harmful (screening)'))
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, -0.01), handletextpad=0.4, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=0.6, h_pad=0.8)

    os.makedirs(args.out, exist_ok=True)
    stem = os.path.join(args.out, f'oracle_{args.metric}')
    fig.savefig(stem + '.pdf', bbox_inches='tight')
    fig.savefig(stem + '.png', dpi=300, bbox_inches='tight')
    with open(stem + '_table.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['task', 'target', 'point', 'mean', 'sd', 'screening_label'])
        for task, t, p, mu, s_, lab in table:
            w.writerow([task, t, p, f'{mu:.1f}', f'{s_:.1f}', (lab or 'inconclusive') if args.metric == 'delta' else ''])
    print(f'wrote {stem}.pdf, {stem}.png, {stem}_table.csv')


if __name__ == '__main__':
    main()
