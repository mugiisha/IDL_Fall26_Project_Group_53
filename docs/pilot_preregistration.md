# Pilot pre-registration: HalfCheetah, switch at 50%

Written and committed before any full-length pilot switching result exists. Any change made after
results are seen goes under "Deviations", with the reason.

- Approved by project member on 16 Nov 2026; team review pending
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

(none yet)
