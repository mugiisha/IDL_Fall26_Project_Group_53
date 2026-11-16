# Pilot pre-registration: HalfCheetah, switch at 50%

Written and committed before any full-length pilot switching result exists. Any change made after
results are seen goes under "Deviations", with the reason.

- Approved by project member on 16 Nov 2026; team review pending (see Amendment 1)
- Code: branch `pilot-halfcheetah-50`, from `main` at `5eadeb1`
  (merge of PR #1, identical to `cf6787e`; gates 1, 1b, 2a, 2b, 2c, 3 and 4 passed
  on the Windows laptop)

## Purpose

1. Validate the switching pipeline end to end on full-length runs.
2. Measure real run times for the compute decision gate.
3. Take a first, descriptive look at RQ1 at one switch point.

The pilot does not answer RQ1 on its own. With 3 seeds only large effects can show.

## Design

- Task: HalfCheetah-v4, CleanRL default hyperparameters, 1M steps (488 iterations).
- Seeds 1, 2, 3. Every run of a seed (source and continuations) on the same machine
  (the Windows laptop), same package versions, torch default threads (16).
- Source runs: Adam 3e-4, checkpoints at iterations 48, 146, 244, 341 and 488.
  Seed 1 is the gate-4 run (bit-identical to the 19 Oct baseline).
- Switch point: the iteration-244 checkpoint (50%).
- Six continuations per seed, to iteration 488, learning rate on the original
  annealed schedule (chosen LR x remaining fraction):

| name | optimizer | chosen LR | optimizer state |
|---|---|---|---|
| sgd | SGD, momentum 0.9 | 1e-2 | fresh |
| rmsprop | RMSProp | 3e-4 | fresh |
| lion | Lion | 1e-4 | fresh |
| muon | Muon on hidden 64x64 matrices + AdamW elsewhere | muon 5e-3, AdamW 3e-4 | fresh |
| adam_reset | Adam (eps 1e-5) | 3e-4 | fresh (control) |
| adam_carry | Adam (eps 1e-5) | 3e-4 | carried (exactness check) |

- Folders: `runs_oracle/HalfCheetah_s{seed}/source/` and
  `runs_oracle/HalfCheetah_s{seed}/at50_{name}/`. Runs must come from committed
  code (`git_dirty: false`).

## Score

Unchanged from stages 0-2 (`analysis/summarize_runs.py`): mean of
`charts/episodic_return` over episodes whose step is at least 90% of the run's
largest logged step. With 1,000-step episodes this is the last 100 episodes
(steps 900,000 to 999,000), all inside the continuation.

## Validity (required before any effect is looked at)

1. Every continuation passes `check_run.py --expect gate1b`.
2. Per seed, the first rollout after the switch (iteration 245) is identical in the
   source run and all six continuations (`check_rollouts.py --first`).
3. Per seed, `adam_carry` is bit-identical to its source run after the switch
   (`compare_runs.py --min-step 499712 --expect gate1b`, timing tags ignored).
4. A seed failing any check is investigated and rerun, never analysed as is.

## Quantities

- Delta(target, seed) = score(target, seed) - score(source Adam, seed), paired.
- Delta vs reset(target, seed) = score(target, seed) - score(adam_reset, seed).
- Descriptive only: post-switch dip = mean episodic return in iterations 245-260
  minus the source run's mean in iterations 235-244; and signal traces around
  the switch.

## Screening labels (descriptive, not statistical significance)

Applied to Delta vs Adam for sgd, rmsprop, lion, muon and adam_reset:

- promising: Delta > 0 on all 3 seeds and mean Delta >= +150
- harmful: Delta < 0 on all 3 seeds and mean Delta <= -150
- inconclusive: anything else

150 is about twice Adam's seed SD on HalfCheetah in stage 0
(scores 1428.9, 1568.8, 1551.4; SD 76.3).

## Go / no-go for the full oracle

- GO if the validity checks pass, whatever the labels. A null pilot is informative.
- NO-GO only for a pipeline problem that cannot be fixed quickly.
- Labels set priorities only (e.g. which targets get held-out confirmation seeds).

## Guards against over-reading

- No claim about RQ1 from the pilot alone.
- A target picked as best on seeds 1-3 is confirmed on fresh seeds 6-10 before any
  claim (winner's curse).
- These runs become the oracle's 50% cell for seeds 1-3; valid because this rule
  is fixed first.
- Claims about phases (RQ1) stay separate from task dependence.

## After the pilot

The COMPUTE DECISION GATE: measured run times decide whether the full oracle runs
locally or on NCSA Delta, and with how many seeds.

## Deviations

### Amendment 1, 16 Nov 2026: machines for the pilot and the full oracle

Status: proposed by Razan on 16 Nov 2026, team approval pending. Recorded before
any full-length switching continuation was run. The only full-length run so far is
the seed-1 Adam source (gate 4).

The pilot itself is unchanged. It runs as written under Design: seeds 1, 2 and 3,
sources and all 18 continuations, on the validated Windows laptop.

Plan for the other machines and later work:

1. While the pilot runs, Candace, Amir and Regis set up their computers and run
   gates 2b (exact Adam resume, HalfCheetah), 2c (exact Adam resume, Hopper) and
   3 (cold switches to SGD, RMSProp, Lion, Muon and Adam-reset) on their own
   machines, and share the outputs with the team.
2. A machine counts as validated only after it passes all three gates on that
   machine. A pass on one machine says nothing about another.
3. After the pilot is validated and the compute decision gate is held, the full
   oracle is divided among all four teammates by complete seed blocks.
4. A seed block is one Adam source run plus every continuation from its
   checkpoints. The whole block runs on one validated machine, with the same code
   commit, package versions, device (CPU) and torch thread count. A checkpoint is
   never moved to another machine to start an official continuation.
5. Thread counts may differ between machines. Versions, threads, host and commit
   are taken from each run's `run_info.json`.
6. Regis may run a Hopper seed-1 Adam source once his machine passes the gates.
   It is preparation for the full oracle, not part of this pilot, and its
   continuations must run on his machine.

What does not change: task, seeds 1-3, switch point (iteration 244), the six
continuations and their learning rates, the score, the quantities, the screening
labels and thresholds, the validity checks and go/no-go.

For the full oracle: differences between seeds on different machines include
machine differences. Effects stay paired within a seed (each seed's own Adam
source and Adam-reset), and this is reported. A source run is expected to match an
old stage-0 baseline bit for bit only on the laptop that produced that baseline.

Clarification of validity check 3 (no rule change): "timing tags ignored" means
`compare_runs.py SOURCE CARRY --min-step 499712 --expect gate1b --ignore charts/SPS
time/rollout_s time/update_s time/gns_s`. Without `--ignore` the wall-clock tags
always differ and the check fails falsely. The same `--ignore` flags apply to the
exact-resume comparisons in gates 2b and 2c.
