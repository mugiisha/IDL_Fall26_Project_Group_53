"""Run oracle continuations for complete seed blocks, one run at a time. No scoring.

Generalises run_halfcheetah_pilot.py to any switch point (10/30/50/70%).
With --make-sources it also trains missing Adam sources first (block B, new seeds).
Same rules: committed code only, never overwrite, revalidate existing folders first,
every check saved to an evidence folder and must print its PASS line.
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

NUM_STEPS = 2048
BASE_LR = 0.0003  # source Adam lr
PREREG = 'docs/oracle_preregistration.md'
SOURCE_CODE = '526fc8c'  # src/ must be identical to the code that made the sources
LAPTOP_ENV = '2.14.1+cpu 16 0.29.1 3.16.0 1.26.4 True'
ARMS = {
    'adam_carry': ['--optimizer', 'adam', '--learning-rate', '0.0003'],
    'adam_reset': ['--optimizer', 'adam', '--learning-rate', '0.0003', '--reset-optimizer'],
    'sgd': ['--optimizer', 'sgd', '--learning-rate', '0.01'],
    'rmsprop': ['--optimizer', 'rmsprop', '--learning-rate', '0.0003'],
    'lion': ['--optimizer', 'lion', '--learning-rate', '0.0001'],
    'muon': ['--optimizer', 'muon', '--learning-rate', '0.0003', '--muon-lr', '0.005'],
}
IGNORE = ['charts/SPS', 'time/rollout_s', 'time/update_s', 'time/gns_s']


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def capture(args, cwd):
    return subprocess.check_output([str(a) for a in args], cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()


def ok(args, cwd):
    return subprocess.run([str(a) for a in args], cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def git_check(repo, branch, head=None):
    require(capture(['git', 'branch', '--show-current'], repo) == branch, 'Wrong branch')
    require(not capture(['git', 'status', '--short'], repo), 'Git tree is not clean')
    now = capture(['git', 'rev-parse', 'HEAD'], repo)
    require(head is None or now == head, 'HEAD changed while running; stop and review')
    return now


def runinfo(folder, threads):
    paths = list((folder / 'runs').glob('*/run_info.json'))
    require(len(paths) == 1, f'Expected one run_info.json in {folder}')
    data = json.loads(paths[0].read_text(encoding='utf-8'))
    require(data.get('git_dirty') is False and data.get('torch_threads') == threads, f'Metadata mismatch: {folder}')
    return str(data.get('git_commit', ''))


def same_code(repo, commit):
    # runs from an earlier commit are fine if the training code is identical
    return bool(commit) and ok(['git', 'diff', '--quiet', commit, 'HEAD', '--', 'src'], repo)


def log_count(folder, expected):
    log = folder / 'log.txt'
    require(log.is_file(), f'Missing log: {log}')
    n = sum('SPS' in line for line in log.read_text(encoding='utf-8', errors='replace').splitlines())
    require(n == expected, f'Expected {expected} SPS lines, got {n}: {folder}; do not overwrite')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--env', default='HalfCheetah-v4', choices=['HalfCheetah-v4', 'Hopper-v4'])
    p.add_argument('--points', nargs='+', type=int, required=True, choices=[10, 30, 50, 70], help='run in this order')
    p.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3])
    p.add_argument('--branch', default='pilot-halfcheetah-50')
    p.add_argument('--arms', nargs='+', default=list(ARMS), choices=list(ARMS), help='block B uses a subset')
    p.add_argument('--make-sources', action='store_true', help='train a missing Adam source for a seed first')
    p.add_argument('--threads', type=int, default=16, help='must match the sources')
    p.add_argument('--env-check', default=LAPTOP_ENV, help='expected versions line (laptop default)')
    p.add_argument('--total-steps', type=int, default=1000000, help='only change for a small test')
    p.add_argument('--run-after-approval', action='store_true', help='start training; without it this is a dry run')
    args = p.parse_args()

    repo = Path.cwd().resolve()
    root = repo / 'runs_oracle'
    ppo = repo / 'src' / 'optswitch' / 'ppo.py'
    analysis = repo / 'analysis'
    for f in [ppo, repo / PREREG] + [analysis / n for n in ['check_run.py', 'compare_runs.py', 'check_rollouts.py', 'show_lrs.py']]:
        require(f.is_file(), f'Missing {f}; use the repository root')
    head = git_check(repo, args.branch)
    require(ok(['git', 'cat-file', '-e', f'HEAD:{PREREG}'], repo), f'{PREREG} is not committed')
    require(ok(['git', 'diff', '--quiet', SOURCE_CODE, 'HEAD', '--', 'src'], repo), f'src/ differs from {SOURCE_CODE}')
    versions = capture([sys.executable, '-c', 'import torch,gymnasium,mujoco,numpy; print(torch.__version__,torch.get_num_threads(),gymnasium.__version__,mujoco.__version__,numpy.__version__,hasattr(torch.optim,"Muon"))'], repo)
    require(versions == args.env_check, f'Unexpected environment: {versions}')
    print(f'Code {head[:7]}, threads {args.threads}, versions ok', flush=True)

    n_iter = args.total_steps // NUM_STEPS
    # same rounding as ppo.py: checkpoint iteration and folder percentage
    K = {pt: int(pt / 100 * n_iter) for pt in args.points}
    for pt, k in K.items():
        require(round(100 * k / n_iter) == pt, f'Point {pt}% does not map to a checkpoint')

    prefix = args.env.split('-')[0]
    blocks = {s: root / f'{prefix}_s{s}' for s in args.seeds}
    arms = [a for a in ARMS if a in args.arms]
    missing = [s for s, b in blocks.items() if not (b / 'source').exists()]
    require(not missing or args.make_sources, f'No source for seeds {missing}; use --make-sources')
    ckpts = {}

    def source_checks(s, b):
        runs = list((b / 'source' / 'runs').glob('*'))
        require(len(runs) == 1, f'Expected one source run in {b}')
        require(same_code(repo, runinfo(b / 'source', args.threads)), f'Source code differs from HEAD: {b}')
        log_count(b / 'source', n_iter)
        for pt, k in K.items():
            ck = b / 'source' / 'checkpoints' / runs[0].name / f'iter{k:04d}.pt'
            require(ck.is_file(), f'Missing {ck}')
            state = capture([sys.executable, analysis / 'show_lrs.py', ck], repo)
            lr = f'lr {BASE_LR * (1 - (k - 1) / n_iter):.4g}'
            require(args.env in state and f'iteration {k} ' in state and 'optimizer adam' in state and lr in state and 'all finite' in state,
                    f'Checkpoint check failed: {state}')
            ckpts[s, pt] = ck
        print(f'Seed {s}: source and checkpoints {sorted(K.values())} found', flush=True)

    for s, b in blocks.items():
        if s not in missing:
            source_checks(s, b)

    print('\nPlan (existing folders are revalidated, never overwritten):', flush=True)
    for s in missing:
        print(f'  seed {s} Adam source: new', flush=True)
    for pt in args.points:
        for s, b in blocks.items():
            for name in arms:
                folder = b / f'at{pt}_{name}'
                if folder.exists():
                    log_count(folder, n_iter - K[pt])
                    require(same_code(repo, runinfo(folder, args.threads)), f'Training code differs from HEAD: {folder}')
                print(f'  {pt}% seed {s} {name}: {"existing" if folder.exists() else "new"}', flush=True)
    if not args.run_after_approval:
        print('\nDRY RUN ONLY. No training started and no files written.', flush=True)
        return

    evidence = root / ('oracle_validation_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    evidence.mkdir(exist_ok=False)
    (evidence / 'code.txt').write_text(f'{head}\n{versions}\nthreads {args.threads}\n', encoding='utf-8')
    print(f'Validation evidence: {evidence}', flush=True)

    def check(label, script, arguments, expected):
        path = evidence / (label + '.txt')
        print(f'  Checking {label}', flush=True)
        with path.open('x', encoding='utf-8') as f:
            result = subprocess.run([sys.executable, str(analysis / script), *map(str, arguments)], cwd=repo, stdout=f, stderr=subprocess.STDOUT)
        output = path.read_text(encoding='utf-8', errors='replace')
        require(result.returncode == 0 and all(x in output for x in expected), f'Validation failed: {path}')

    for s in missing:
        b = blocks[s]
        git_check(repo, args.branch, head)
        print(f'[{datetime.now():%H:%M:%S}] Training seed {s} Adam source', flush=True)
        (b / 'source').mkdir(parents=True, exist_ok=False)
        cmd = [sys.executable, str(ppo), '--env-id', args.env, '--seed', str(s), '--total-timesteps', str(args.total_steps),
               '--optimizer', 'adam', '--learning-rate', str(BASE_LR), '--checkpoint-at', '0.1', '0.3', '0.5', '0.7', '--checkpoint-final']
        with (b / 'source' / 'log.txt').open('x', encoding='utf-8') as f:
            result = subprocess.run(cmd, cwd=b / 'source', stdout=f, stderr=subprocess.STDOUT)
        require(result.returncode == 0, f'Training failed: {b / "source" / "log.txt"}')
        require(runinfo(b / 'source', args.threads) == head, f'Recorded commit is not HEAD: {b}')
        check(f's{s}_source_signals', 'check_run.py', [b / 'source' / 'runs', '--expect', 'gate1b'],
              [f'{n_iter} iterations, steps {NUM_STEPS} to {n_iter * NUM_STEPS}', 'PASS: all values finite, expected tags present'])
        for it in [int(f * n_iter) for f in (0.1, 0.3, 0.5, 0.7)] + [n_iter]:
            found = list((b / 'source' / 'checkpoints').glob(f'*/iter{it:04d}.pt'))
            require(len(found) == 1, f'Missing checkpoint {it} for seed {s}')
        source_checks(s, b)
        print(f'[{datetime.now():%H:%M:%S}] PASS seed {s} source', flush=True)

    for pt in args.points:
        k = K[pt]
        n_new = n_iter - k
        for s, b in blocks.items():
            source = b / 'source' / 'runs'
            order = sorted(arms, key=lambda name: not (b / f'at{pt}_{name}').exists())
            for name in order:
                git_check(repo, args.branch, head)
                folder = b / f'at{pt}_{name}'
                label = f'p{pt}_s{s}_{name}'
                new = not folder.exists()
                if not new:
                    print(f'[{datetime.now():%H:%M:%S}] Revalidating {label}', flush=True)
                else:
                    print(f'[{datetime.now():%H:%M:%S}] Training {label}', flush=True)
                    folder.mkdir(exist_ok=False)
                    cmd = [sys.executable, str(ppo), '--env-id', args.env, '--seed', str(s), '--total-timesteps', str(args.total_steps),
                           *ARMS[name], '--resume-from', str(ckpts[s, pt])]
                    with (folder / 'log.txt').open('x', encoding='utf-8') as f:
                        result = subprocess.run(cmd, cwd=folder, stdout=f, stderr=subprocess.STDOUT)
                    require(result.returncode == 0, f'Training failed: {folder / "log.txt"}')
                log_count(folder, n_new)
                commit = runinfo(folder, args.threads)
                # new runs must record HEAD; older ones only need identical training code
                require(commit == head if new else same_code(repo, commit), f'Recorded commit not accepted: {folder}')
                runs = folder / 'runs'
                check(label + '_signals', 'check_run.py', [runs, '--expect', 'gate1b'],
                      [f'{n_new} iterations, steps {(k + 1) * NUM_STEPS} to {n_iter * NUM_STEPS}', 'PASS: all values finite, expected tags present'])
                if name == 'adam_carry':
                    check(label + '_metrics', 'compare_runs.py', [source, runs, '--min-step', k * NUM_STEPS, '--expect', 'gate1b', '--ignore', *IGNORE],
                          ['PASS: shared tags identical, new tags finite'])
                    check(label + '_rollouts', 'check_rollouts.py', [source, runs], [f'PASS: all {n_new} shared iterations identical'])
                else:
                    check(label + '_first_rollout', 'check_rollouts.py', [source, runs, '--first'], [f'PASS: first shared iteration ({k + 1}) identical'])
                print(f'[{datetime.now():%H:%M:%S}] PASS {label}', flush=True)
            check(f'p{pt}_s{s}_group_first_rollout', 'check_rollouts.py', [source, *(b / f'at{pt}_{n}' / 'runs' for n in arms), '--first'],
                  [f'PASS: first shared iteration ({k + 1}) identical'])
            print(f'{pt}% seed {s}: all {len(arms)} continuations passed.', flush=True)
    print(f'ALL PASSED for points {args.points}, seeds {args.seeds}. No scores viewed. Evidence: {evidence}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        print(f'STOP: {exc}\nKeep all folders and logs. Do not delete or overwrite failed runs.', file=sys.stderr, flush=True)
        sys.exit(1)
