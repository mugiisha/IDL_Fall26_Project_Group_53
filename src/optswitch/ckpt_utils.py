# checkpoint helpers that do not need torch, so they can be tested on their own
# written for gymnasium 0.29.1 (SyncVectorEnv resets a finished env inside the same step)
import hashlib
import json
import os
import random

import gymnasium as gym
import numpy as np

# settings that must match the checkpoint for any continuation
TRAIN_KEYS = [
    "env_id", "seed", "total_timesteps", "num_envs", "num_steps", "anneal_lr", "gamma", "gae_lambda",
    "num_minibatches", "update_epochs", "norm_adv", "clip_coef", "clip_vloss", "ent_coef", "vf_coef",
    "max_grad_norm", "target_kl", "torch_deterministic",
]


def check_compatible(saved, args):
    # returns (carry_state, problems). carry only when the same optimizer continues without a reset
    problems = [f"{k}: checkpoint {saved.get(k)!r}, now {args[k]!r}" for k in TRAIN_KEYS if saved.get(k) != args[k]]
    carry = args["optimizer"] == saved.get("optimizer") and not args["reset_optimizer"]
    if carry:
        # exact continuation, so the lr schedule must be the original one
        keys = ["learning_rate"] + (["muon_lr"] if args["optimizer"] == "muon" else [])
        problems += [f"{k}: checkpoint {saved.get(k)!r}, now {args[k]!r} (needed to carry state)" for k in keys if saved.get(k) != args[k]]
    return carry, problems


def _arr(a):
    return {"dtype": str(a.dtype), "data": np.asarray(a).tolist()}


def _num(x):
    # numpy scalars to plain python, same value
    return x.item() if isinstance(x, np.generic) else x


def _unarr(d):
    return np.array(d["data"], dtype=d["dtype"])


def _wrapper(env, cls):
    while env is not None:
        if isinstance(env, cls):
            return env
        env = getattr(env, "env", None)
    raise RuntimeError(f"wrapper {cls.__name__} not found")


def get_env_state(envs):
    # read-only snapshot of everything the next steps depend on, as a json string
    out = []
    for e in envs.envs:
        u, d = e.unwrapped, e.unwrapped.data
        stats = _wrapper(e, gym.wrappers.RecordEpisodeStatistics)
        obs_n = _wrapper(e, gym.wrappers.NormalizeObservation)
        rew_n = _wrapper(e, gym.wrappers.NormalizeReward)
        out.append({
            "qpos": _arr(d.qpos), "qvel": _arr(d.qvel), "act": _arr(d.act), "ctrl": _arr(d.ctrl),
            "qacc_warmstart": _arr(d.qacc_warmstart), "time": _num(d.time),  # warmstart matters for exactness
            "np_random": u.np_random.bit_generator.state,
            "elapsed_steps": _num(_wrapper(e, gym.wrappers.TimeLimit)._elapsed_steps),
            "ep_returns": _arr(stats.episode_returns), "ep_lengths": _arr(stats.episode_lengths),
            "ep_count": _num(stats.episode_count),
            "obs_rms": [_arr(obs_n.obs_rms.mean), _arr(obs_n.obs_rms.var), _num(obs_n.obs_rms.count)],
            "ret_rms": [_arr(rew_n.return_rms.mean), _arr(rew_n.return_rms.var), _num(rew_n.return_rms.count)],
            "returns": _arr(rew_n.returns),
        })
    return json.dumps(out)


def set_env_state(envs, state):
    state = json.loads(state)
    assert len(state) == len(envs.envs), "number of envs differs from the checkpoint"
    for e, s in zip(envs.envs, state):
        u = e.unwrapped
        u.set_state(_unarr(s["qpos"]), _unarr(s["qvel"]))  # also runs mj_forward
        d = u.data
        d.act[:] = _unarr(s["act"])
        d.ctrl[:] = _unarr(s["ctrl"])
        d.qacc_warmstart[:] = _unarr(s["qacc_warmstart"])  # after set_state, so it is not touched again
        d.time = s["time"]
        u.np_random.bit_generator.state = s["np_random"]
        _wrapper(e, gym.wrappers.TimeLimit)._elapsed_steps = s["elapsed_steps"]
        stats = _wrapper(e, gym.wrappers.RecordEpisodeStatistics)
        stats.episode_returns = _unarr(s["ep_returns"])
        stats.episode_lengths = _unarr(s["ep_lengths"])
        stats.episode_count = s["ep_count"]
        rms = _wrapper(e, gym.wrappers.NormalizeObservation).obs_rms
        rms.mean, rms.var, rms.count = _unarr(s["obs_rms"][0]), _unarr(s["obs_rms"][1]), s["obs_rms"][2]
        rew_n = _wrapper(e, gym.wrappers.NormalizeReward)
        rew_n.return_rms.mean, rew_n.return_rms.var = _unarr(s["ret_rms"][0]), _unarr(s["ret_rms"][1])
        rew_n.return_rms.count = s["ret_rms"][2]
        rew_n.returns = _unarr(s["returns"])


def get_rng_state(gns_rng):
    # python, numpy global and the noise-scale generator, as json (torch rng is saved as a tensor)
    np_state = np.random.get_state()
    return json.dumps({
        "python": random.getstate(),
        "numpy": [np_state[0], np_state[1].tolist(), *np_state[2:]],
        "gns": gns_rng.bit_generator.state,
    })


def set_rng_state(state, gns_rng):
    s = json.loads(state)
    v, internal, gauss = s["python"]
    random.setstate((v, tuple(internal), gauss))
    n = s["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    gns_rng.bit_generator.state = s["gns"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_safely(obj, path, save_fn):
    # never overwrite; write to a temp file first, then rename, then record the hash
    if os.path.exists(path):
        raise FileExistsError(f"checkpoint already exists: {path}")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    save_fn(obj, tmp)
    digest = sha256(tmp)
    os.rename(tmp, path)  # fails if path appeared meanwhile
    with open(path + ".sha256", "w") as f:
        f.write(digest + "\n")
    return digest


def verify(path):
    # the file must match the hash written when it was saved
    side = path + ".sha256"
    if not os.path.exists(side):
        raise RuntimeError(f"no {side}; only load checkpoints written by this script")
    with open(side) as f:
        expected = f.read().strip()
    got = sha256(path)
    if got != expected:
        raise RuntimeError(f"hash mismatch for {path}: file changed or corrupted")
    return got
