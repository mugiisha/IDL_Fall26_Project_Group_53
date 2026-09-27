# When Should Reinforcement Learning Change Optimisers?

Investigating whether the best optimiser changes during reinforcement learning and whether training signals can guide beneficial switches.

**Status:** Research in progress. This README describes the planned study; implementation instructions and measured results will be added as experiments are completed.

## Research Questions

1. **RQ1:** Does the best-performing optimiser choice change across training stages under the tested configurations?
2. **RQ2:** Can policy entropy, KL divergence, gradient variability, and return trends identify beneficial switching opportunities beyond elapsed training progress?

## Experimental Scope

| Setting | RL algorithm | Environments or datasets | Primary baseline |
| --- | --- | --- | --- |
| Continuous control | PPO | MuJoCo: HalfCheetah, Hopper, Walker2d, Ant | Fixed Adam |
| Sparse-reward exploration | PPO | Selected MiniGrid tasks | Fixed Adam |
| Mathematical reasoning | GRPO | GSM8K and MATH training sets; GSM8K test and MATH-500 evaluation | Fixed AdamW |

The language-model experiments will use Qwen2.5 0.5B and 1.5B. Work begins with PPO on two MuJoCo tasks before expanding to exploration tasks and GRPO.

## Planned Methods

The candidate optimiser pool includes Adam, SGD with momentum, RMSprop, Lion, and Muon, with AdamW as the language-model reference. Muon experiments will document parameter eligibility and any companion optimiser.

We will compare fixed optimisers with single switches at 10%, 30%, 50%, and 70% of the training budget. These experiments will inform a rule-based controller using smoothed training signals. A predefined schedule and an adapted AOS controller will provide additional comparisons.

State-handling experiments will compare fresh optimiser state with compatible momentum transfer. Restarting the original optimiser and changing only its learning rate will help isolate the source of any improvement. Signal ablations will test which measurements contribute to switching decisions.

## Evaluation and Reproducibility

Classic RL evaluation will report final episodic return and area under the evaluation learning curve. Mathematical reasoning evaluation will report pass@1 under consistent answer verification and decoding settings. Both settings will track runtime, steps to target performance, and training instability.

Methods will receive equal tuning budgets and matched interaction or generation budgets. Hyperparameters and switching rules will be selected on development runs and evaluated on separate seeds. The final study targets at least ten independent training seeds per configuration; smaller pilots will be labelled preliminary.

Each reported experiment should include its configuration, code revision, dependencies, seed, raw logs, and evaluation procedure. Results will include individual runs and uncertainty intervals. No performance improvement is assumed in advance.

## Setup

Installation instructions, pinned dependencies, and verified training commands will be added with the initial implementation. PPO baseline reproduction is planned around CleanRL.

## Roadmap

- [ ] Reproduce PPO with Adam on two MuJoCo tasks.
- [ ] Add training-signal logging and fixed-optimiser comparisons.
- [ ] Evaluate predefined switching times and state-handling controls.
- [ ] Test a signal-based controller on held-out seeds.
- [ ] Expand evaluation to MiniGrid and additional MuJoCo tasks.
- [ ] Extend the study to GRPO with Qwen2.5 0.5B, followed by 1.5B.

## Results

No experimental results are reported yet. Baseline reproduction and preliminary switching comparisons will be documented here as they become available.

## Selected References

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [DeepSeekMath](https://arxiv.org/abs/2402.03300)
- [CleanRL](https://jmlr.org/papers/v23/21-1342.html)
- [Adam on Local Time](https://arxiv.org/abs/2412.17113)
- [Improving Generalization Performance by Switching from Adam to SGD](https://arxiv.org/abs/1712.07628)
- [AOS: Adaptive Optimizer Switching](https://arxiv.org/abs/2608.01997)
- [Deep Reinforcement Learning at the Edge of the Statistical Precipice](https://arxiv.org/abs/2108.13264)
