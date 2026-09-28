# Which Optimizer, and When? Optimizer Switching Across the Phases of Reinforcement Learning

## Overview

This project investigates whether switching optimizers during reinforcement learning (RL) training can improve performance over using a single optimizer throughout. As an agent learns, its data distribution, gradients, and learning targets change. The study asks whether different stages of training benefit from different optimizers and whether simple training signals can guide these choices.

The proposed approach follows three stages: an oracle study of single optimizer switches, analysis of training signals, and a simple switching controller. The core experiments use Proximal Policy Optimization (PPO) on MuJoCo HalfCheetah and Hopper through Gymnasium, with implementations based on CleanRL. Agents generate their training data through simulator interaction.

**Status:** Research in progress. The methods below describe the proposed study; no experimental results are reported yet.

## Research Questions

1. Does the best optimizer change across RL training phases?
2. Can inexpensive training signals predict when an optimizer switch will help?

## Methodology

The candidate optimizer pool is **Adam, SGD with momentum, RMSProp, Lion, and Muon**, with Muon applied only to two-dimensional hidden-layer weights. The PPO loss remains fixed; the study varies optimizer choice, switch time, and optimizer-state handling.

1. **Oracle switching study:** Train with Adam and save checkpoints at **10%, 30%, 50%, and 70%** of the total training budget. From each checkpoint, run a separate continuation with every candidate optimizer through the remaining budget. Compare each final score with uninterrupted Adam and retrospectively select the best continuation. This is a best-case analysis unavailable during normal training.
2. **Training-signal analysis:** Log return slope, policy entropy, KL divergence between successive policies, gradient-noise scale, gradient norm, and critic loss, and test which signals predict positive switching gains.
3. **Switching controller:** Convert informative signals into simple threshold rules, tune them on separate development seeds, and evaluate how much of the oracle gain the controller recovers.

For final score $R(o,t_s)$ after continuing with optimizer $o$ from checkpoint $t_s$, the oracle gain is:

$$
\Delta^{*}=\max_{o,t_s}\left[R(o,t_s)-R(\mathrm{Adam})\right].
$$

This provides a best-case reference for single-switch controllers restricted to the tested optimizers, checkpoint times, and continuation settings.

## Baselines and Controls

Comparisons include Adam throughout training, each other candidate optimizer held fixed, a progress-only switching schedule, and Adaptive Optimizer Switching (AOS) adapted to RL. Methods will receive equal tuning budgets, and the PPO loss will remain fixed.

Cold starts with fresh optimizer state will be compared with warm starts that transfer compatible state. Same-optimizer resets and learning-rate-only changes will help isolate the effect of switching. Signal ablations against the progress-only schedule will assess which measurements contribute to switching decisions.

## Evaluation

Performance will be assessed using **final episodic return**, **area under the learning curve (AUC)**, and the **share of oracle gain recovered by the controller**, $\Delta_{\mathrm{ctrl}}/\Delta^{*}$. Here, $\Delta_{\mathrm{ctrl}}$ is the controller's gain over uninterrupted Adam; the ratio is meaningful when the oracle gain is positive. The optional language-model extension will use **pass@1**. PPO experiments will be completed before attempting GRPO, and transfer to critic-free GRPO will not be assumed.

Final results will use at least **10 independent training seeds per configuration**, reported as **interquartile means with bootstrap uncertainty intervals**. Switching rules will be tuned on separate seeds, and KL divergence will be monitored around transitions to assess instability.

The hypothesis is that well-timed switches improve return or AUC over the best fixed optimizer and that signal-based rules outperform a progress-only schedule. These are hypotheses to test; a null result would also be informative.

## Selected References

- Schulman et al. (2017). Proximal Policy Optimization Algorithms.
- Huang et al. (2022). CleanRL: High-quality single-file implementations of deep reinforcement learning algorithms.
- Ellis et al. (2024). Adam on Local Time: Addressing Nonstationarity in RL with Relative Adam Timesteps.
- Keskar and Socher (2017). Improving Generalization Performance by Switching from Adam to SGD.
- Pandey et al. (2026). AOS: Adaptive Optimizer Switching via Training-State Signals for Faster Convergence and Better Generalization.
- Agarwal et al. (2021). Deep Reinforcement Learning at the Edge of the Statistical Precipice.
