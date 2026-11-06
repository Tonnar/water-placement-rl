"""Train and evaluate DQN, TD3 or PPO agents on water placement.

Usage::

    python -m wprl.train --config configs/td3.yaml
    python -m wprl.train --config configs/td3.yaml --set train.total_steps=5000 env.noise=1.0

With ``data.synthetic`` set, structures are generated on the fly, so every
algorithm can be run (and tested) without any PDB or CCP4 files. Train and
test structures are always disjoint.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
import yaml

from wprl.agents import DQNAgent, DQNConfig, PPOAgent, PPOConfig, TD3Agent, TD3Config
from wprl.buffers import NStepAccumulator, PrioritizedReplayBuffer, ReplayBuffer, RolloutBuffer
from wprl.data import Structure, load_directory, make_synthetic_structure
from wprl.envs import DiscreteWaterEnv, WaterPlacementEnv
from wprl.utils import Logger, coverage_precision, get_device, linear_schedule, set_seed

DEFAULTS: dict = {
    "algo": "td3",
    "seed": 0,
    "device": "auto",
    "out_dir": "runs/{algo}-{time}",
    "wandb": False,
    "data": {
        # Either `synthetic` or `train_dir` (+ optional `test_dir`).
        "synthetic": {"n_train": 2, "n_test": 1, "n_waters": 12, "grid_size": 24, "voxel_size": 1.0},
        "train_dir": None,
        "test_dir": None,
        "limit": None,
        "grid_size": 48,
        "voxel_size": 1.0,
        "density_threshold": 0.5,
    },
    "env": {},
    "agent": {},
    "train": {
        "total_steps": 20_000,
        "steps_per_structure": 5_000,  # steps before moving to the next training protein
        "buffer_size": 200_000,
        "batch_size": 128,
        "learning_starts": 1_000,
        "train_freq": 1,
        "n_step": 3,
        "prioritized": False,
        "per_alpha": 0.6,
        "per_beta_start": 0.4,
        # DQN exploration
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay_steps": 10_000,
        # TD3 exploration and demonstrations
        "explore_noise": 0.1,
        "expert_prefill_steps": 0,
        # PPO
        "rollout_steps": 1024,
        "log_every": 1_000,
        "eval_episodes": 1,
    },
}


# ----------------------------------------------------------------------------
# configuration helpers
# ----------------------------------------------------------------------------
def deep_update(base: dict, new: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (new or {}).items():
        out[k] = deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply ``a.b.c=value`` overrides; values are parsed as YAML."""
    for item in overrides:
        key, _, raw = item.partition("=")
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(raw)
    return cfg


def load_config(path: str | None, overrides: list[str] | None = None) -> dict:
    user = yaml.safe_load(Path(path).read_text()) if path else {}
    return apply_overrides(deep_update(DEFAULTS, user), overrides or [])


def build_dataclass(cls, values: dict):
    known = {f.name for f in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**values)


# ----------------------------------------------------------------------------
# data and environments
# ----------------------------------------------------------------------------
def build_structures(data: dict, seed: int) -> tuple[list[Structure], list[Structure]]:
    if data.get("train_dir"):
        kw = {k: data[k] for k in ("grid_size", "voxel_size", "density_threshold")}
        train = load_directory(data["train_dir"], limit=data.get("limit"), **kw)
        test = load_directory(data["test_dir"], **kw) if data.get("test_dir") else []
        return train, test
    syn = dict(data["synthetic"])
    n_train, n_test = syn.pop("n_train"), syn.pop("n_test")
    # Distinct seeds guarantee the test structures are never seen in training.
    train = [make_synthetic_structure(seed=seed * 1000 + i, name=f"train{i}", **syn) for i in range(n_train)]
    test = [make_synthetic_structure(seed=seed * 1000 + 500 + i, name=f"test{i}", **syn) for i in range(n_test)]
    return train, test


def make_env(algo: str, structure: Structure, env_cfg: dict):
    if algo == "dqn":
        return DiscreteWaterEnv(structure, **env_cfg)
    kw = dict(env_cfg)
    kw.setdefault("action_type", "spherical" if algo == "td3" else "cartesian")
    return WaterPlacementEnv(structure, **kw)


def structure_schedule(structures: list[Structure], total: int, per: int):
    """Yield (structure, n_steps) visiting the training proteins round-robin."""
    done, i = 0, 0
    while done < total:
        n = min(per, total - done)
        yield structures[i % len(structures)], n
        done += n
        i += 1


# ----------------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(algo: str, agent, structures: list[Structure], env_cfg: dict, episodes: int, seed: int) -> dict:
    """Run the greedy/deterministic policy and report final placement quality."""
    rows = []
    for s in structures:
        env = make_env(algo, s, env_cfg)
        agent.set_structure(s)
        for ep in range(episodes):
            obs, info = env.reset(seed=seed + ep)
            start_mse = info["mse"]
            done = False
            while not done:
                if algo == "dqn":
                    a = agent.act(obs, 0.0, np.random.default_rng(0), env.action_mask())
                elif algo == "td3":
                    a = agent.act(obs)
                else:
                    a = agent.act(obs, deterministic=True)[0]
                obs, _, term, trunc, info = env.step(a)
                done = term or trunc
            cov, prec = coverage_precision(env.positions, env.target, 0.5)
            rows.append({"start_mse": start_mse, "final_mse": info["mse"], "success": float(term),
                         "coverage@0.5": cov, "precision@0.5": prec})
    if not rows:
        return {}
    return {f"eval/{k}": float(np.mean([r[k] for r in rows])) for k in rows[0]}


# ----------------------------------------------------------------------------
# training loops
# ----------------------------------------------------------------------------
class EpisodeTracker:
    def __init__(self):
        self.ret, self.len, self.history = 0.0, 0, []

    def step(self, reward: float, done: bool, info: dict):
        self.ret += reward
        self.len += 1
        if done:
            self.history.append((self.ret, self.len, info["mse"], info.get("reason") == "success"))
            self.ret, self.len = 0.0, 0

    def summary(self) -> dict:
        if not self.history:
            return {}
        r, n, m, s = zip(*self.history[-20:])
        return {"episode_return": np.mean(r), "episode_length": np.mean(n),
                "episode_final_mse": np.mean(m), "episode_success": np.mean(s)}


def _make_replay(cfg: dict, n_atoms: int, action_shape: tuple, discrete: bool, device):
    t = cfg["train"]
    if t["prioritized"]:
        return PrioritizedReplayBuffer(t["buffer_size"], n_atoms, action_shape, discrete, device, alpha=t["per_alpha"])
    return ReplayBuffer(t["buffer_size"], n_atoms, action_shape, discrete, device)


def _sample(buffer, t: dict, rng, step: int):
    if isinstance(buffer, PrioritizedReplayBuffer):
        beta = linear_schedule(t["per_beta_start"], 1.0, t["total_steps"], step)
        return buffer.sample(t["batch_size"], rng, beta=beta)
    return buffer.sample(t["batch_size"], rng)


def train_off_policy(cfg: dict, agent, train: list[Structure], logger: Logger, rng) -> None:
    """Shared loop for DQN and TD3 (n-step returns, optional PER)."""
    algo, t = cfg["algo"], cfg["train"]
    gamma = agent.cfg.gamma
    buffers: dict[str, ReplayBuffer] = {}
    experts: dict[str, ReplayBuffer] = {}
    tracker, step, last = EpisodeTracker(), 0, {}

    for structure, n_steps in structure_schedule(train, t["total_steps"], t["steps_per_structure"]):
        env = make_env(algo, structure, cfg["env"])
        agent.set_structure(structure)
        discrete = algo == "dqn"
        action_shape = () if discrete else env.action_space.shape
        if structure.name not in buffers:
            buffers[structure.name] = _make_replay(cfg, env.n_atoms, action_shape, discrete, agent.device)
            if algo == "td3" and t["expert_prefill_steps"] > 0:
                experts[structure.name] = prefill_with_oracle(env, buffers[structure.name], t, gamma, rng)
        buffer, expert = buffers[structure.name], experts.get(structure.name)
        nstep = NStepAccumulator(t["n_step"], gamma)
        obs, _ = env.reset(seed=int(rng.integers(2**31)))

        for _ in range(n_steps):
            if discrete:
                eps = linear_schedule(t["epsilon_start"], t["epsilon_end"], t["epsilon_decay_steps"], step)
                action = agent.act(obs, eps, rng, env.action_mask())
            elif step < t["learning_starts"] and expert is None:
                action = env.action_space.sample()
            else:
                action = agent.act(obs, t["explore_noise"], rng)
            next_obs, reward, term, trunc, info = env.step(action)
            for tr in nstep.push(obs, action, reward, next_obs, term, trunc):
                buffer.add(*tr)
            tracker.step(reward, term or trunc, info)
            obs = env.reset(seed=int(rng.integers(2**31)))[0] if (term or trunc) else next_obs

            if len(buffer) >= t["learning_starts"] and step % t["train_freq"] == 0:
                batch = _sample(buffer, t, rng, step)
                expert_batch = expert.sample(t["batch_size"], rng) if expert is not None else None
                metrics = agent.update(batch, expert_batch) if algo == "td3" else agent.update(batch)
                buffer.update_priorities(batch.indices, metrics.pop("td_abs"))
                last = metrics
            step += 1
            if step % t["log_every"] == 0:
                extra = {"epsilon": eps} if discrete else {}
                logger.log(step, **last, **tracker.summary(), **extra)


def prefill_with_oracle(env: WaterPlacementEnv, buffer: ReplayBuffer, t: dict, gamma: float, rng) -> ReplayBuffer:
    """Fill the replay buffer with oracle transitions; also keep them as a
    separate demonstration buffer for the behaviour-cloning loss."""
    demos = ReplayBuffer(t["expert_prefill_steps"], env.n_atoms, env.action_space.shape, False, buffer.device)
    nstep = NStepAccumulator(t["n_step"], gamma)
    obs, _ = env.reset(seed=int(rng.integers(2**31)))
    for _ in range(t["expert_prefill_steps"]):
        action = env.oracle_action()
        next_obs, reward, term, trunc, _ = env.step(action)
        for tr in nstep.push(obs, action, reward, next_obs, term, trunc):
            buffer.add(*tr)
            demos.add(*tr)
        obs = env.reset(seed=int(rng.integers(2**31)))[0] if (term or trunc) else next_obs
    return demos


def train_ppo(cfg: dict, agent: PPOAgent, train: list[Structure], logger: Logger, rng) -> None:
    t, c = cfg["train"], agent.cfg
    tracker, step = EpisodeTracker(), 0
    for structure, n_steps in structure_schedule(train, t["total_steps"], t["steps_per_structure"]):
        env = make_env("ppo", structure, cfg["env"])
        agent.set_structure(structure)
        rollout = RolloutBuffer(t["rollout_steps"], env.n_atoms, env.action_space.shape[0], c.gamma, c.gae_lambda)
        obs, _ = env.reset(seed=int(rng.integers(2**31)))
        remaining = n_steps
        while remaining > 0:
            rollout.reset()
            while not rollout.full:
                action, log_prob, value = agent.act(obs)
                next_obs, reward, term, trunc, info = env.step(action)
                tracker.step(reward, term or trunc, info)
                if trunc and not term:
                    # A time limit is not a real ending: bootstrap from V(s').
                    reward += c.gamma * agent.value(next_obs)
                rollout.add(obs, action, log_prob, value, reward, term or trunc)
                obs = env.reset(seed=int(rng.integers(2**31)))[0] if (term or trunc) else next_obs
                step += 1
                remaining -= 1
            rollout.compute_gae(last_value=agent.value(obs))
            metrics = agent.update(rollout, rng)
            logger.log(step, **metrics, **tracker.summary())


# ----------------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------------
def build_agent(cfg: dict, structure: Structure, device):
    algo, a = cfg["algo"], cfg["agent"]
    he, vs = structure.half_extent, structure.voxel_size
    if algo == "dqn":
        return DQNAgent(he, vs, build_dataclass(DQNConfig, a), device)
    action_dim = make_env(algo, structure, cfg["env"]).action_space.shape[0]
    if algo == "td3":
        return TD3Agent(he, vs, action_dim, build_dataclass(TD3Config, a), device)
    if algo == "ppo":
        return PPOAgent(he, vs, action_dim, build_dataclass(PPOConfig, a), device)
    raise ValueError(f"unknown algo {algo!r}")


def run(cfg: dict) -> dict:
    rng = set_seed(cfg["seed"])
    device = get_device(cfg["device"])
    out_dir = Path(cfg["out_dir"].format(algo=cfg["algo"], time=time.strftime("%Y%m%d-%H%M%S")))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    train, test = build_structures(cfg["data"], cfg["seed"])
    if len({s.grid_size for s in train + test}) != 1:
        raise ValueError("all structures must share one grid size")
    for s in train + test:
        if s.n_out_of_bounds():
            print(f"warning: {s.name} has {s.n_out_of_bounds()} target waters outside the grid")

    agent = build_agent(cfg, train[0], device)
    logger = Logger(out_dir, cfg, use_wandb=cfg["wandb"])
    if cfg["algo"] == "ppo":
        train_ppo(cfg, agent, train, logger, rng)
    else:
        train_off_policy(cfg, agent, train, logger, rng)

    results = evaluate(cfg["algo"], agent, test, cfg["env"], cfg["train"]["eval_episodes"], cfg["seed"] + 10_000)
    (out_dir / "eval.json").write_text(json.dumps(results, indent=2))
    nets = {"dqn": ("q",), "td3": ("actor", "critic"), "ppo": ("ac",)}[cfg["algo"]]
    torch.save({n: getattr(agent, n).state_dict() for n in nets}, out_dir / "checkpoint.pt")
    logger.close()
    print(json.dumps({"out_dir": str(out_dir), **results}, indent=2))
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="YAML config file")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    args = parser.parse_args(argv)
    run(load_config(args.config, args.set))


if __name__ == "__main__":
    main()

