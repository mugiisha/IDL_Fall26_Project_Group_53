# Machine validation (Amendment 1)

A machine is validated when gates 2b (exact Adam resume, HalfCheetah), 2c (exact Adam resume,
Hopper) and 3 (cold switches to SGD, RMSProp, Lion, Muon, Adam-reset) pass on that machine.
Checks below were re-run by Claude on 22 Nov 2026 from each machine's run files
(TensorBoard events, rollout hashes, run_info.json), using `compare_runs.py`
(timing tags ignored), `check_rollouts.py` and `check_run.py --expect gate1b`.
Gate runs: 20 iterations, switch at iteration 10, seeds 91 (HalfCheetah) and 92 (Hopper).

| | Razan | Amir | Candace |
|---|---|---|---|
| host | laptop | DESKTOP-62RT6MV | vpn-172-31-61-72 |
| OS | Windows | Windows 11 (10.0.26200) | macOS 26.4.1, arm64 |
| CPU | Intel Core Ultra 7 155H | Intel family 6 model 170 (name to confirm) | Apple silicon (model to confirm) |
| python | (see run_info) | 3.10.22 | 3.10.22 |
| torch | 2.14.1+cpu | 2.14.1+cpu | 2.14.1 |
| gymnasium / mujoco / numpy | 0.29.1 / 3.16.0 / 1.26.4 | 0.29.1 / 3.16.0 / 1.26.4 | 0.29.1 / 3.17.0 / 1.26.4 |
| torch threads | 16 | 16 | 4 |
| commit of gate runs | cf6787e (see pilot prereg) | 5eadeb1 | a8b313b |
| git_dirty | false | false | false |
| 2b HalfCheetah resume | PASS | PASS | PASS |
| 2c Hopper resume | PASS | PASS | PASS |
| 3 cold switches | PASS | PASS | PASS |

Details, both Amir and Candace:
- 2b and 2c: carry run identical to the reference on every shared tag after the switch and on
  all 10 shared rollouts.
- 3: all six continuations 10 iterations (steps 22528 to 40960), all values finite, expected
  signal tags present; iteration 11 rollout identical across the reference and all six.
- Arguments match the preregistered arms (sgd 1e-2, rmsprop 3e-4, lion 1e-4, muon 3e-4 with
  muon-lr 5e-3, adam_reset with reset flag).
- `src/` is identical at 526fc8c, 5eadeb1, a8b313b and 4870b9a, so all gate runs used the same
  training code as the oracle sources.

Notes:
- Amir's written report gives commit 44d4fef; his run files record 5eadeb1. The run files are
  authoritative. No effect: same training code.
- Candace's MuJoCo 3.17.0 differs from 3.16.0 elsewhere. Gates passed with 3.17.0, so her
  environment stays as it is for every run she makes (no upgrade or downgrade mid-block).
- Agreement with Amendment 1: Amir pending, Candace pending (as of 22 Nov).
