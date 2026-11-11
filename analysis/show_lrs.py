# print the learning rate of every optimizer group stored in a checkpoint (muon has two parts)
# usage: python show_lrs.py CHECKPOINT.pt
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "optswitch"))
from ckpt_utils import verify  # noqa: E402

import torch  # noqa: E402

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
