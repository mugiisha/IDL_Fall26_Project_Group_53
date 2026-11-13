# print the learning rate of every optimizer group stored in a checkpoint (muon has two parts)
# and check that every float tensor in the weights and optimizer state is finite
# usage: python show_lrs.py CHECKPOINT.pt
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "optswitch"))
from ckpt_utils import verify  # noqa: E402

import torch  # noqa: E402


def non_finite(x, path, out):
    # paths of float tensors holding nan or inf
    if isinstance(x, torch.Tensor):
        if x.is_floating_point() and not bool(torch.isfinite(x).all()):
            out.append(path)
    elif isinstance(x, dict):
        for k, v in x.items():
            non_finite(v, f"{path}.{k}", out)
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            non_finite(v, f"{path}[{i}]", out)

if len(sys.argv) != 2:
    sys.exit("usage: python show_lrs.py CHECKPOINT.pt")
path = sys.argv[1]
verify(path)
c = torch.load(path, map_location="cpu", weights_only=True)
parts = c["optimizer"] if isinstance(c["optimizer"], list) else [c["optimizer"]]
print(f"{c['run_name']}  iteration {c['iteration']}  optimizer {c['optimizer_name']}")
for i, part in enumerate(parts):
    for j, g in enumerate(part["param_groups"]):
        print(f"  part {i} group {j}: lr {g['lr']:.4g} ({len(g['params'])} tensors)")
bad = []
non_finite(c["agent"], "agent", bad)
non_finite(c["optimizer"], "optimizer", bad)
print("  weights and optimizer state: all finite" if not bad else "  NOT FINITE: " + ", ".join(bad))
sys.exit(1 if bad else 0)
