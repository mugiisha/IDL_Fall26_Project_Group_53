# Oracle pre-registration: HalfCheetah switch points and held-out confirmation

Written and committed before any oracle run at 10, 30 or 70% and before any held-out seed
(6-10) exists. Any change made after results are seen goes under "Deviations", with the reason.

- Status: proposed by Razan on 22 Nov 2026; team review pending (as is Amendment 1 of
  `pilot_preregistration.md`).
- Already seen when this was written: the pilot (50%, seeds 1-3, all six arms), 21 Nov.
  The switch points 10/30/50/70% and fresh confirmation seeds 6-10 were planned before the
  pilot (project plan; pilot pre-registration, "Guards against over-reading").
- Code: branch `pilot-halfcheetah-50`. `src/` identical to `526fc8c`, the code that made
  the seed 1-3 sources (the runner checks this). Runner: `analysis/run_oracle_block.py`.

Everything not stated here is as in `pilot_preregistration.md`: task, hyperparameters, the six
arms and their learning rates, annealed schedule, score, quantities, folders, Amendment 1
machine rules.

## Block A: switch points on seeds 1-3 (RQ1, phases)

- HalfCheetah-v4, seeds 1, 2, 3, existing Adam sources and checkpoints (Razan's laptop).
- Switch points 10, 30, 70% = checkpoint iterations 48, 146, 341. The 50% cell is the pilot.
- All six arms at every point: sgd, rmsprop, lion, muon, adam_reset, adam_carry.
- Folders `runs_oracle/HalfCheetah_s{seed}/at{point}_{name}/`.
- 16 torch threads, as the sources. Runs may execute one at a time or several at once at
  16 threads each; tested on the laptop 21 Nov: rollouts identical to a single run.
- Order of execution: 30%, 70%, then 10%. Order does not change the runs.

## Block B: held-out confirmation (winner's curse guard)

- HalfCheetah-v4, fresh seeds 6, 7, 8, 9, 10. New Adam 3e-4 sources, 488 iterations,
  checkpoints at 10/30/50/70% and final, on one validated machine per seed block.
- Switch at 50% (iteration 244). Arms: muon (primary), lion (secondary), adam_reset
  (control); adam_carry on seed 6 only, as the exact-resume check for that machine.
- Why these targets: muon, lion and sgd improved on all 3 pilot seeds. Muon and lion had the two largest mean deltas (both labelled inconclusive). Muon had the largest mean delta, so it is the primary target.
- Seeds 1-3 are never pooled into the confirmation.

## Validity (both blocks, before any score is looked at)

As the pilot, with the numbers for switch iteration K and N = 488:
1. Every continuation passes `check_run.py --expect gate1b`
   (N-K iterations, steps (K+1)*2048 to 999424).
2. Per seed and point, the first rollout after the switch (iteration K+1) is identical in
   the source and every continuation (`check_rollouts.py --first`).
3. adam_carry is bit-identical to its source after the switch
   (`compare_runs.py --min-step K*2048 --expect gate1b`, timing tags ignored).
4. A seed or point failing any check is investigated and rerun, never analysed as is.
5. New sources (block B): 488 SPS lines, 5 checkpoints, git_dirty false, all finite.

## Analysis

Scores are looked at only after all validity checks for a whole point (block A) or the whole
block (block B) have passed. The analysis script is committed before it is run.

Block A, per target and point: Delta vs Adam and Delta vs adam_reset, paired by seed; the
pilot labels (promising / harmful / inconclusive, threshold 150) at each point.
Phase profile: per target, mean Delta (and each seed) at 10, 30, 50, 70%. Descriptive.
A phase statement for a target ("better early than late", or the reverse) needs, for some pair
of points, the paired difference Delta(p1) - Delta(p2) to have the same sign on all 3 seeds and
a mean of at least 150 in size. Otherwise the profile is reported as inconclusive.

Block B, muon (primary): confirmed if Delta vs Adam > 0 on all 5 seeds (one-sided exact sign
test, p = 1/32 = 0.031). Also reported: mean Delta, 95% bootstrap CI over seeds (10,000
resamples), Delta vs adam_reset. Lion uses the same rule but is secondary (no multiplicity
correction), reported as exploratory.

Machines: effects stay paired within a seed. If seeds run on different machines, this is
reported and the machine is listed per seed.

## Block C: Hopper (Amendment O1, 22 Nov 2026)

Status: proposed by Razan on 22 Nov 2026, team review pending. Written before any Hopper oracle
run exists.

- Hopper-v4, seeds 1, 2, 3. New Adam 3e-4 sources (488 iterations, checkpoints at 10/30/50/70%
  and final), made on the machine that runs the block; all continuations on that machine.
- Switch points 10, 30, 50, 70%; all six arms with the same learning rates as HalfCheetah.
- Order of execution: 50%, 30%, 70%, 10% (so the most useful points finish first if time runs
  out). Order does not change the runs.
- Validity and analysis exactly as block A (`oracle_summary.py --block A --env Hopper`),
  including the phase rule. A point is analysed only when all 3 seeds pass validation there.
- Task dependence (HalfCheetah vs Hopper) is reported separately from phase claims (RQ1).
  A Hopper result does not confirm or refute a HalfCheetah result, or the reverse.
- Stage-0 Hopper baselines came from another machine; new sources are not expected to match them
  bit for bit, and they are not used in the deltas.

## Not covered here

Any other task gets its own amendment before its first run.

## Deviations

None to the design.

Machines, 22 Nov 2026 (proposal, team review pending). Gate evidence for each machine was
re-checked by Claude from the run files (see `docs/machine_validation.md`).
- Block A: Razan's laptop, 16 threads (as the seed 1-3 sources).
- Block B: Amir's machine (DESKTOP-62RT6MV), all five seeds, 16 threads, same package versions
  as the laptop.
- Block C: Candace's Mac, all three seeds, 4 threads, MuJoCo 3.17.0 (the other machines use
  3.16.0). Allowed under Amendment 1 because the whole block runs on one machine with one
  version set; it is reported.
- On any machine, runs may execute in parallel at that machine's fixed thread count (same
  thread count gives identical rollouts; tested on the laptop 21 Nov).

Note, 22 Nov 2026 (before any oracle score was viewed; 30% and 70% for seeds 1-3 had run and
validated, scores not viewed): added `analysis/oracle_summary.py`, the analysis exactly as in
"Analysis" above (it reproduces `pilot_summary.py` at 50%), and extended
`analysis/run_oracle_block.py` with `--arms` (block B subset) and `--make-sources` (trains and
checks new Adam sources) and `--env` (Hopper). Completed runs from an earlier commit are
accepted when their `src/` is identical to HEAD (new runs must record HEAD), so runs that
record `4870b9a` still revalidate. The summary refuses to report if any preregistered run or
adam_carry control is missing.
