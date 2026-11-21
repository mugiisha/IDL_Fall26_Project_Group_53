"""Run the existing Group 53 pilot protocol sequentially. No scoring or Git edits."""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

COMMIT = '44d4fef8e3bb5bb94b067c9da13ec3cf70b61555'
SOURCE_COMMITS = {1: '526fc8c', 2: COMMIT, 3: COMMIT}
RUNS = {1: 'HalfCheetah-v4__adam__lr0.0003__s1__1794437120',
        2: 'HalfCheetah-v4__adam__lr0.0003__s2__1794844686',
        3: 'HalfCheetah-v4__adam__lr0.0003__s3__1794846949'}
# Carry first is an execution-order choice, not a change to the pilot design.
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


def git_check(repo):
    require(capture(['git', 'branch', '--show-current'], repo) == 'pilot-halfcheetah-50', 'Wrong branch')
    require(capture(['git', 'rev-parse', 'HEAD'], repo) == COMMIT, 'Unexpected commit; review before running')
    require(not capture(['git', 'status', '--short'], repo), 'Git tree is not clean')


def runinfo(folder, commit):
    paths = list((folder / 'runs').glob('*/run_info.json'))
    require(len(paths) == 1, f'Expected one run_info.json in {folder}')
    data = json.loads(paths[0].read_text(encoding='utf-8'))
    require(str(data.get('git_commit', '')).startswith(commit), f'Unexpected recorded commit: {folder}')
    require(data.get('git_dirty') is False and data.get('torch_threads') == 16, f'Metadata mismatch: {folder}')


def log_count(folder, expected):
    log = folder / 'log.txt'
    require(log.is_file(), f'Missing log: {log}')
    n = sum('SPS' in line for line in log.read_text(encoding='utf-8', errors='replace').splitlines())
    require(n == expected, f'Expected {expected} SPS lines, got {n}: {folder}; do not overwrite')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-after-approval', action='store_true', help='Start training ONLY after team approval or documented course-authorised permission to proceed.')
    args = parser.parse_args()
    repo = Path.cwd().resolve()
    root = repo / 'runs_oracle'
    ppo = repo / 'src' / 'optswitch' / 'ppo.py'
    analysis = repo / 'analysis'
    for p in [ppo] + [analysis / n for n in ['check_run.py', 'compare_runs.py', 'check_rollouts.py', 'show_lrs.py']]:
        require(p.is_file(), f'Missing {p}; use the repository root')
    git_check(repo)
    versions = capture([sys.executable, '-c', 'import torch,gymnasium,mujoco,numpy; print(torch.__version__,torch.get_num_threads(),gymnasium.__version__,mujoco.__version__,numpy.__version__,hasattr(torch.optim,"Muon"))'], repo)
    require(versions == '2.14.1+cpu 16 0.29.1 3.16.0 1.26.4 True', f'Unexpected environment: {versions}')
    blocks = {s: root / f'HalfCheetah_s{s}' for s in RUNS}
    checkpoints = {s: b / 'source' / 'checkpoints' / RUNS[s] / 'iter0244.pt' for s, b in blocks.items()}
    for s, b in blocks.items():
        require(checkpoints[s].is_file(), f'Missing {checkpoints[s]}')
        runinfo(b / 'source', SOURCE_COMMITS[s])
        log_count(b / 'source', 488)
        state = capture([sys.executable, analysis / 'show_lrs.py', checkpoints[s]], repo)
        require('iteration 244' in state and 'optimizer adam' in state and 'lr 0.0001506' in state and 'all finite' in state, f'Checkpoint check failed: {state}')
        print(f'Seed {s}: source and checkpoint found', flush=True)
    print('\nPlan (existing folders will be revalidated, never overwritten):', flush=True)
    for s, b in blocks.items():
        for name in ARMS:
            folder = b / f'at50_{name}'
            if folder.exists():
                log_count(folder, 244)
                runinfo(folder, COMMIT)
            print(f'  seed {s} {name}: {"existing" if folder.exists() else "new"}', flush=True)
    if not args.run_after_approval:
        print('\nDRY RUN ONLY. No training started and no files written.', flush=True)
        return
    evidence = root / ('pilot_validation_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    evidence.mkdir(exist_ok=False)
    print(f'Validation evidence: {evidence}', flush=True)

    def check(label, script, arguments, expected):
        path = evidence / (label + '.txt')
        print(f'  Checking {label}', flush=True)
        with path.open('x', encoding='utf-8') as f:
            result = subprocess.run([sys.executable, str(analysis / script), *map(str, arguments)], cwd=repo, stdout=f, stderr=subprocess.STDOUT)
        output = path.read_text(encoding='utf-8', errors='replace')
        require(result.returncode == 0 and all(x in output for x in expected), f'Validation failed: {path}')

    for s, b in blocks.items():
        source = b / 'source' / 'runs'
        # Check every already-existing continuation before starting more work for this seed.
        order = sorted(ARMS, key=lambda name: not (b / f'at50_{name}').exists())
        for name in order:
            git_check(repo)
            folder = b / f'at50_{name}'
            label = f's{s}_{name}'
            if folder.exists():
                print(f'[{datetime.now():%H:%M:%S}] Revalidating {label}', flush=True)
            else:
                print(f'[{datetime.now():%H:%M:%S}] Training {label}', flush=True)
                folder.mkdir(exist_ok=False)
                cmd = [sys.executable, str(ppo), '--env-id', 'HalfCheetah-v4', '--seed', str(s), *ARMS[name], '--resume-from', str(checkpoints[s])]
                with (folder / 'log.txt').open('x', encoding='utf-8') as f:
                    result = subprocess.run(cmd, cwd=folder, stdout=f, stderr=subprocess.STDOUT)
                require(result.returncode == 0, f'Training failed: {folder / "log.txt"}')
            log_count(folder, 244)
            runinfo(folder, COMMIT)
            runs = folder / 'runs'
            check(label + '_signals', 'check_run.py', [runs, '--expect', 'gate1b'], ['244 iterations, steps 501760 to 999424', 'PASS: all values finite, expected tags present'])
            if name == 'adam_carry':
                check(label + '_metrics', 'compare_runs.py', [source, runs, '--min-step', '499712', '--expect', 'gate1b', '--ignore', *IGNORE], ['PASS: shared tags identical, new tags finite'])
                check(label + '_rollouts', 'check_rollouts.py', [source, runs], ['PASS: all 244 shared iterations identical'])
            else:
                check(label + '_first_rollout', 'check_rollouts.py', [source, runs, '--first'], ['PASS: first shared iteration (245) identical'])
            print(f'[{datetime.now():%H:%M:%S}] PASS {label}', flush=True)
        check(f's{s}_group_first_rollout', 'check_rollouts.py', [source, *(b / f'at50_{name}' / 'runs' for name in ARMS), '--first'], ['PASS: first shared iteration (245) identical'])
        print(f'Seed {s}: all six continuations passed.', flush=True)
    print(f'ALL 18 PASSED. No scores viewed. Evidence: {evidence}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        print(f'STOP: {exc}\nKeep all folders and logs. Do not delete or overwrite failed runs.', file=sys.stderr, flush=True)
        sys.exit(1)
