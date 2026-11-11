# compare the training state in two checkpoints, exactly
# usage: python compare_checkpoints.py A.pt B.pt
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "optswitch"))
from ckpt_utils import verify  # noqa: E402

# training state only; names, paths and args are allowed to differ
KEYS = ["iteration", "global_step", "num_iterations", "optimizer_name", "agent", "optimizer",
        "next_obs", "next_done", "torch_rng", "rng", "env"]


def diff(a, b, path, out):
    if hasattr(a, "equal") and hasattr(a, "dtype"):  # tensor
        if not (hasattr(b, "equal") and a.dtype == b.dtype and a.shape == b.shape and a.equal(b)):
            out.append(f"{path}: tensors differ")
    elif isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b):
            out.append(f"{path}: keys differ")
            return
        for k in a:
            diff(a[k], b[k], f"{path}.{k}", out)
    elif isinstance(a, (list, tuple)):
        if not isinstance(b, (list, tuple)) or len(a) != len(b):
            out.append(f"{path}: length differs")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, f"{path}[{i}]", out)
    elif isinstance(a, str) and isinstance(b, str) and a != b and a[:1] in "[{":
        diff(json.loads(a), json.loads(b), path, out)  # json state, so we can say which field
    elif a != b:
        out.append(f"{path}: {a!r} vs {b!r}")


def load(path):
    import torch

    verify(path)
    return torch.load(path, map_location="cpu", weights_only=True)


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: python compare_checkpoints.py A.pt B.pt")
    a, b = load(sys.argv[1]), load(sys.argv[2])
    print(f"A: {a['run_name']} iteration {a['iteration']}")
    print(f"B: {b['run_name']} iteration {b['iteration']}")
    out = []
    for k in KEYS:
        if k not in a or k not in b:
            out.append(f"{k}: missing")
        else:
            diff(a[k], b[k], k, out)
    n_opt = sum(len(s["state"]) for s in (a["optimizer"] if isinstance(a["optimizer"], list) else [a["optimizer"]]))
    print(f"  checked {len(a['agent'])} weight tensors, optimizer state for {n_opt} tensors, rng and env state")
    for line in out[:20]:
        print("  DIFF ", line)
    if len(out) > 20:
        print(f"  ... and {len(out) - 20} more")
    print("PASS: training state identical" if not out else f"FAIL: {len(out)} difference(s)")
    sys.exit(1 if out else 0)


if __name__ == "__main__":
    main()
