# compare the tensorboard scalars of two runs, value by value
# usage: python compare_runs.py REF_RUN NEW_RUN [--ignore TAG ...]
# a run can be a run folder, or a runs/ folder holding exactly one run
import argparse
import os
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("new")
    ap.add_argument("--ignore", nargs="*", default=["charts/SPS"], help="tags to skip (timing etc.)")
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
        if r == n:
            print(f"  same  {tag} ({len(r)} values)")
            continue
        failed += 1
        if len(r) != len(n):
            print(f"  DIFF  {tag}: {len(r)} vs {len(n)} values")
        else:
            i = next(i for i in range(len(r)) if r[i] != n[i])
            print(f"  DIFF  {tag}: first at step {r[i][0]}: {r[i][1]!r} vs {n[i][1]!r}")

    extra = sorted(set(new) - set(ref))
    if extra:
        print(f"  new only ({len(extra)}): " + ", ".join(extra))
    print("PASS: identical on every shared tag" if failed == 0 else f"FAIL: {failed} tag(s) differ")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
