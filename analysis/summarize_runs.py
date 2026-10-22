# reads the tensorboard logs in runs/, writes a summary csv and the report plots
#   python analysis/summarize_runs.py
import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# published cleanrl ppo_continuous_action results (docs.cleanrl.dev)
CLEANRL_REF = {"HalfCheetah-v4": (1442.64, 46.03), "Hopper-v4": (2382.86, 271.74)}

NEW_NAME = re.compile(r"^(?P<env>[^_]+)__(?P<opt>[a-z]+)__lr(?P<lr>[0-9.e-]+)(_mlr(?P<mlr>[0-9.e-]+))?__s(?P<seed>\d+)__\d+$")
OLD_NAME = re.compile(r"^(?P<env>[^_]+)__ppo__(?P<seed>\d+)__\d+$")  # baseline runs, adam lr 3e-4


def parse_name(folder):
    m = NEW_NAME.match(folder)
    if m:
        lr = float(m["mlr"]) if m["mlr"] else float(m["lr"])  # for muon we compare --muon-lr
        return m["env"], m["opt"], lr, int(m["seed"])
    m = OLD_NAME.match(folder)
    if m:
        return m["env"], "adam", 3e-4, int(m["seed"])
    return None  # smoke tests etc.


def load_returns(path):
    ea = EventAccumulator(path, size_guidance={"scalars": 0})
    ea.Reload()
    if "charts/episodic_return" not in ea.Tags()["scalars"]:
        return None
    ev = ea.Scalars("charts/episodic_return")
    return np.array([e.step for e in ev]), np.array([e.value for e in ev])


def smooth_on_grid(steps, values, grid, window=20):
    vals = pd.Series(values).rolling(window, min_periods=1).mean().to_numpy()
    return np.interp(grid, steps, vals)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--out-csv", default="results/run_summary.csv")
    p.add_argument("--plots", default="plots")
    p.add_argument("--last-frac", type=float, default=0.1, help="score = mean return over this last fraction of training")
    args = p.parse_args()

    rows, curves = [], {}
    for folder in sorted(os.listdir(args.runs)):
        info = parse_name(folder)
        if info is None:
            print(f"skipping {folder}")
            continue
        data = load_returns(os.path.join(args.runs, folder))
        if data is None:
            print(f"no returns logged in {folder}, skipping")
            continue
        env, opt, lr, seed = info
        steps, vals = data
        cut = steps.max() * (1 - args.last_frac)
        final = vals[steps >= cut]
        rows.append(dict(env=env, optimizer=opt, lr=lr, seed=seed, final_score=float(np.nanmean(final)),
                         auc=float(np.sum((vals[1:] + vals[:-1]) / 2 * np.diff(steps)) / steps.max()), max_step=int(steps.max()),
                         has_nan=bool(np.isnan(vals).any()), folder=folder))
        curves[folder] = (env, opt, lr, seed, steps, vals)

    if not rows:
        print("no runs found")
        return
    df = pd.DataFrame(rows).sort_values(["env", "optimizer", "lr", "seed"])
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"\nwrote {args.out_csv}\n")

    summary = df.groupby(["env", "optimizer", "lr"]).agg(
        seeds=("seed", "count"), final_mean=("final_score", "mean"), final_std=("final_score", "std"),
        auc_mean=("auc", "mean")).reset_index()
    summary["lr"] = summary["lr"].map(lambda x: f"{x:g}")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.1f}"))

    os.makedirs(args.plots, exist_ok=True)

    # baseline: adam 3e-4, mean over seeds with min-max band, cleanrl reference
    for env in df.env.unique():
        base = [c for c in curves.values() if c[0] == env and c[1] == "adam" and abs(c[2] - 3e-4) < 1e-12]
        if not base:
            continue
        grid = np.linspace(0, min(c[4].max() for c in base), 200)
        ys = np.stack([smooth_on_grid(c[4], c[5], grid) for c in base])
        plt.figure(figsize=(5, 3.5))
        plt.plot(grid, ys.mean(0), label=f"ours, Adam ({len(base)} seeds)")
        plt.fill_between(grid, ys.min(0), ys.max(0), alpha=0.25)
        if env in CLEANRL_REF:
            ref, sd = CLEANRL_REF[env]
            plt.axhline(ref, color="k", ls="--", lw=1, label="CleanRL reported (final)")
            plt.axhspan(ref - sd, ref + sd, color="k", alpha=0.08)
        plt.xlabel("environment steps")
        plt.ylabel("episodic return")
        plt.title(f"PPO baseline on {env}")
        plt.legend(fontsize=8)
        plt.tight_layout()
        out = os.path.join(args.plots, f"baseline_{env}.png")
        plt.savefig(out, dpi=200)
        plt.close()
        print(f"saved {out}")

    # learning-rate search: one panel per optimizer, mean over seeds with each seed shown as a dot
    for env in df.env.unique():
        sub = df[df.env == env]
        opts = [o for o in ["adam", "sgd", "rmsprop", "lion", "muon"] if o in set(sub.optimizer)]
        opts = [o for o in opts if sub[sub.optimizer == o].lr.nunique() > 1]
        if not opts:
            continue
        fig, axes = plt.subplots(1, len(opts), figsize=(2.6 * len(opts), 2.8), sharey=True, squeeze=False)
        for ax, opt in zip(axes[0], opts):
            g = sub[sub.optimizer == opt]
            agg = g.groupby("lr").final_score.agg(["mean", "std", "count"]).reset_index()
            ax.errorbar(agg.lr, agg["mean"], yerr=agg["std"].fillna(0), marker="o", capsize=3)
            ax.scatter(g.lr, g.final_score, s=10, color="gray", alpha=0.6, zorder=3)
            for _, r in agg.iterrows():
                ax.annotate(f"n={int(r['count'])}", (r.lr, r["mean"]), textcoords="offset points",
                            xytext=(0, 6), ha="center", fontsize=6)
            ax.set_xscale("log")
            ax.set_title(opt, fontsize=9)
            ax.set_xlabel("--muon-lr" if opt == "muon" else "learning rate", fontsize=8)
            ax.tick_params(which="both", labelsize=7)
        axes[0][0].set_ylabel(f"mean return, last {int(args.last_frac * 100)}%", fontsize=8)
        fig.suptitle(f"Learning-rate search on {env} (dots = seeds, bars = std)", fontsize=9)
        fig.tight_layout()
        out = os.path.join(args.plots, f"lr_search_{env}.png")
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
