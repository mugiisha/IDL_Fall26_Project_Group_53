# compare the tensorboard scalars of two runs, value by value
# usage: python compare_runs.py REF_RUN NEW_RUN [--ignore TAG ...]
# a run can be a run folder, or a runs/ folder holding exactly one run
import argparse
import math
import os
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# tags each preset must log once per iteration
PRESETS = {
    "gate1b": [
        "signals/approx_kl_mean", "signals/entropy_mean", "signals/grad_clip_frac", "signals/grad_norm_max",
        "signals/grad_norm_mean", "signals/old_approx_kl_mean", "signals/param_norm", "signals/pg_loss_mean",
        "signals/update_norm", "signals/v_loss_mean", "time/rollout_s", "time/update_s",
        "signals/gns_g2", "signals/gns_tr_sigma", "signals/gns_simple", "time/gns_s",
    ],
}


def find_run(path):
    path = os.path.abspath(path)
    if any(f.startswith("events.out.tfevents") for f in os.listdir(path)):
        return path
    subs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    if len(subs) != 1:
        sys.exit(f"{path}: expected one run folder inside, found {len(subs)}")
    return os.path.join(path, subs[0])


def load(path):
    ea = EventAccumulator(path, size_guidance={"scalars": 0})
    ea.Reload()
    return {tag: [(e.step, e.value) for e in ea.Scalars(tag)] for tag in ea.Tags()["scalars"]}


def same(a, b):
    # exact match, except nan counts as equal to nan
    return len(a) == len(b) and all(
        sa == sb and (va == vb or (math.isnan(va) and math.isnan(vb))) for (sa, va), (sb, vb) in zip(a, b)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("new")
    ap.add_argument("--ignore", nargs="*", default=["charts/SPS"], help="tags to skip (timing etc.)")
    ap.add_argument("--allow-nan", nargs="*", default=["signals/gns_simple"], help="new tags where nan is expected")
    ap.add_argument("--expect", choices=sorted(PRESETS), help="also require these tags, one value per iteration")
    ap.add_argument("--step-tag", default="charts/learning_rate", help="ref tag that marks the iteration steps")
    a = ap.parse_args()

    ref_path, new_path = find_run(a.ref), find_run(a.new)
    print("ref:", ref_path)
    print("new:", new_path)
    ref, new = load(ref_path), load(new_path)

    failed = 0
    for tag in sorted(ref):
        if tag in a.ignore:
            print(f"  skip  {tag}")
            continue
        if tag not in new:
            print(f"  MISSING {tag}")
            failed += 1
            continue
        r, n = ref[tag], new[tag]
        if same(r, n):
            print(f"  same  {tag} ({len(r)} values)")
            continue
        failed += 1
        if len(r) != len(n):
            print(f"  DIFF  {tag}: {len(r)} vs {len(n)} values")
        else:
            i = next(i for i in range(len(r)) if not same([r[i]], [n[i]]))
            print(f"  DIFF  {tag}: first at step {r[i][0]}: {r[i][1]!r} vs {n[i][1]!r}")

    extra = sorted(set(new) - set(ref))
    if extra:
        print(f"  new only ({len(extra)}): " + ", ".join(extra))
    # new tags must be finite, apart from the allowed nan ones
    for tag in extra:
        vals = [v for _, v in new[tag]]
        nan = sum(math.isnan(v) for v in vals)
        inf = sum(math.isinf(v) for v in vals)
        if inf or (nan and tag not in a.allow_nan):
            print(f"  BAD   {tag}: {nan} nan, {inf} inf out of {len(vals)}")
            failed += 1
        elif nan:
            print(f"  note  {tag}: {nan} of {len(vals)} nan (allowed)")
    if a.expect:
        steps = [st for st, _ in ref.get(a.step_tag, [])]
        bad = 0
        if not steps:
            print(f"  BAD   no {a.step_tag} in ref, cannot check iteration steps")
            bad += 1
        for tag in PRESETS[a.expect]:
            got = [st for st, _ in new.get(tag, [])]
            if got != steps:
                print(f"  BAD   {tag}: {len(got)} values, expected {len(steps)} at the iteration steps")
                bad += 1
        failed += bad
        if not bad:
            print(f"  ok    all {len(PRESETS[a.expect])} '{a.expect}' tags present, one value per iteration")

    print("PASS: shared tags identical, new tags finite" if failed == 0 else f"FAIL: {failed} problem(s)")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
