import torch
import torch.optim as optim

OPTIMIZERS = ["adam", "sgd", "rmsprop", "lion", "muon"]


class CombinedOptimizer:
    # wraps muon + adamw so ppo can treat them as one optimizer
    def __init__(self, optimizers):
        self.optimizers = optimizers

    @property
    def param_groups(self):
        return [g for opt in self.optimizers for g in opt.param_groups]

    def zero_grad(self, set_to_none=True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self):
        for opt in self.optimizers:
            opt.step()

    def state_dict(self):
        return [opt.state_dict() for opt in self.optimizers]

    def load_state_dict(self, states):
        for opt, state in zip(self.optimizers, states):
            opt.load_state_dict(state)


def is_hidden_matrix(name, param):
    # critic / actor_mean layout: 0 = input, 2 = hidden, 4 = output
    return param.ndim == 2 and name.endswith(".2.weight")


def make_optimizer(name, model, lr, muon_lr=0.02):
    name = name.lower()
    if name == "adam":
        return optim.Adam(model.parameters(), lr=lr, eps=1e-5)  # same as cleanrl
    if name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    if name == "rmsprop":
        return optim.RMSprop(model.parameters(), lr=lr, alpha=0.99, eps=1e-5)
    if name == "lion":
        from lion_pytorch import Lion

        return Lion(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0.0)
    if name == "muon":
        # muon only for the hidden weight matrices, adamw for the rest
        hidden = [p for n, p in model.named_parameters() if is_hidden_matrix(n, p)]
        other = [p for n, p in model.named_parameters() if not is_hidden_matrix(n, p)]
        muon = torch.optim.Muon(hidden, lr=muon_lr, weight_decay=0.0)
        adamw = optim.AdamW(other, lr=lr, eps=1e-5, weight_decay=0.0)
        return CombinedOptimizer([muon, adamw])
    raise ValueError(f"unknown optimizer {name!r}, choose from {OPTIMIZERS}")
