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
- Why these targets: in the pilot only muon and lion improved on all 3 seeds (both labelled
  inconclusive). Muon had the largest mean delta, so it is the primary target.
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

## Not covered here

Hopper (seeds 1-3, all points) and any other task get their own amendment before their first
run.

## Deviations

None yet.
