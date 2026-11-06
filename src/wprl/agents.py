"""DQN-family, TD3 and PPO agents.

All agents share one convention: call :meth:`set_structure` when the
environment switches to a new protein, so the agent knows which density grid
to feed its networks.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from wprl.buffers import Batch, RolloutBuffer
from wprl.data import Structure
from wprl.networks import Actor, GaussianActorCritic, QNetwork, StateEncoder, TwinCritic
from wprl.utils import polyak_update


class _BaseAgent:
    def __init__(self, device: torch.device):
        self.device = device
        self._density: torch.Tensor | None = None

    def set_structure(self, structure: Structure) -> None:
        self._density = torch.as_tensor(structure.density, device=self.device).unsqueeze(0)

    def density(self, batch_size: int) -> torch.Tensor:
        if self._density is None:
            raise RuntimeError("call set_structure() before acting or updating")
        return self._density.expand(batch_size, -1, -1, -1)

    def _obs_tensors(self, obs: dict):
        waters = torch.as_tensor(obs["waters"], device=self.device).unsqueeze(0)
        focus = torch.as_tensor([obs["focus"]], device=self.device)
        return waters, focus


# ----------------------------------------------------------------------------
# DQN
# ----------------------------------------------------------------------------
@dataclass
class DQNConfig:
    lr: float = 1e-4
    gamma: float = 0.99
    double: bool = True
    dueling: bool = True
    target_update_every: int = 1000
    grad_clip: float = 10.0
    embed_dim: int = 128
    num_heads: int = 4
    local_size: int = 7
    hidden: int = 256


class DQNAgent(_BaseAgent):
    """DQN with optional Double-DQN targets and a dueling head.

    Prioritized replay is handled by the buffer; the agent applies the
    importance-sampling weights it returns and reports TD errors back.
    """

    def __init__(self, half_extent: float, voxel_size: float, cfg: DQNConfig, device: torch.device):
        super().__init__(device)
        self.cfg = cfg
        enc = StateEncoder(half_extent, voxel_size, cfg.embed_dim, cfg.num_heads, cfg.local_size)
        self.q = QNetwork(enc, hidden=cfg.hidden, dueling=cfg.dueling).to(device)
        self.q_target = copy.deepcopy(self.q).requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.updates = 0

    @torch.no_grad()
    def act(self, obs: dict, epsilon: float, rng: np.random.Generator, mask: np.ndarray | None = None) -> int:
        if rng.random() < epsilon:
            valid = np.flatnonzero(mask) if mask is not None else None
            n = self.q.n_moves * len(obs["waters"])
            return int(rng.choice(valid)) if valid is not None and len(valid) else int(rng.integers(n))
        waters, _ = self._obs_tensors(obs)
        q = self.q(self.density(1), waters)[0]
        if mask is not None:
            q[~torch.as_tensor(mask, device=self.device)] = -torch.inf
        return int(q.argmax())

    def update(self, batch: Batch) -> dict:
        b = len(batch.rewards)
        dens = self.density(b)
        q_all = self.q(dens, batch.waters)
        q_pred = q_all.gather(1, batch.actions.view(-1, 1)).squeeze(1)
        with torch.no_grad():
            q_next_target = self.q_target(dens, batch.next_waters)
            if self.cfg.double:
                # Online net picks the action, target net scores it.
                best = self.q(dens, batch.next_waters).argmax(dim=1, keepdim=True)
                q_next = q_next_target.gather(1, best).squeeze(1)
            else:
                q_next = q_next_target.max(dim=1).values
            target = batch.rewards + batch.discounts * q_next
        td = target - q_pred
        loss = (batch.weights * F.smooth_l1_loss(q_pred, target, reduction="none")).mean()

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.grad_clip))
        self.optimizer.step()

        self.updates += 1
        if self.updates % self.cfg.target_update_every == 0:
            self.q_target.load_state_dict(self.q.state_dict())
        return {
            "loss": loss.item(),
            "q_mean": q_pred.mean().item(),
            "target_mean": float(target.mean()),
            "grad_norm": grad_norm,
            "td_abs": td.detach().abs().cpu().numpy(),
        }


# ----------------------------------------------------------------------------
# TD3
# ----------------------------------------------------------------------------
@dataclass
class TD3Config:
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    gamma: float = 0.95
    tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2
    grad_clip: float = 1.0
    bc_weight: float = 0.0  # initial behaviour-cloning weight (0 disables)
    bc_decay_updates: int = 50_000  # updates over which bc_weight decays to 0
    embed_dim: int = 128
    num_heads: int = 4
    local_size: int = 9
    hidden: int = 256


class TD3Agent(_BaseAgent):
    """Twin Delayed DDPG (Fujimoto et al., 2018), with optional behaviour cloning.

    The BC term ``bc_weight * ||pi(s) - a_expert||^2`` on demonstration
    batches pulls the actor toward the oracle early in training; its weight
    decays linearly to zero so the critic eventually drives the policy.
    """

    def __init__(self, half_extent: float, voxel_size: float, action_dim: int, cfg: TD3Config, device: torch.device):
        super().__init__(device)
        self.cfg = cfg
        mk = lambda: StateEncoder(half_extent, voxel_size, cfg.embed_dim, cfg.num_heads, cfg.local_size)  # noqa: E731
        self.actor = Actor(mk(), action_dim, cfg.hidden).to(device)
        self.critic = TwinCritic(mk(), action_dim, cfg.hidden).to(device)
        self.actor_target = copy.deepcopy(self.actor).requires_grad_(False)
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.action_dim = action_dim
        self.updates = 0

    @torch.no_grad()
    def act(self, obs: dict, noise_std: float = 0.0, rng: np.random.Generator | None = None) -> np.ndarray:
        """Policy action plus Gaussian exploration noise, clipped to [-1, 1].

        The returned (noisy) action is what the environment executes and what
        should be stored in the replay buffer.
        """
        waters, focus = self._obs_tensors(obs)
        a = self.actor(self.density(1), waters, focus)[0].cpu().numpy()
        if noise_std > 0:
            rng = rng or np.random.default_rng()
            a = a + rng.normal(0.0, noise_std, size=a.shape)
        return np.clip(a, -1.0, 1.0).astype(np.float32)

    def bc_weight(self) -> float:
        frac = min(1.0, self.updates / max(1, self.cfg.bc_decay_updates))
        return self.cfg.bc_weight * (1.0 - frac)

    def update(self, batch: Batch, expert_batch: Batch | None = None) -> dict:
        cfg, b = self.cfg, len(batch.rewards)
        dens = self.density(b)
        with torch.no_grad():
            noise = (torch.randn_like(batch.actions) * cfg.policy_noise).clamp(-cfg.noise_clip, cfg.noise_clip)
            next_a = (self.actor_target(dens, batch.next_waters, batch.next_focus) + noise).clamp(-1, 1)
            tq1, tq2 = self.critic_target(dens, batch.next_waters, batch.next_focus, next_a)
            y = batch.rewards + batch.discounts * torch.min(tq1, tq2)

        q1, q2 = self.critic(dens, batch.waters, batch.focus, batch.actions)
        per_sample = F.smooth_l1_loss(q1, y, reduction="none") + F.smooth_l1_loss(q2, y, reduction="none")
        critic_loss = (batch.weights * per_sample).mean()
        self.critic_opt.zero_grad()
        critic_loss.backward()
        critic_grad = float(torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.grad_clip))
        self.critic_opt.step()

        with torch.no_grad():
            var_y = torch.var(y)
            resid_var = float(torch.var(y - q1) / var_y) if var_y > 1e-8 else float("nan")
        metrics = {
            "critic_loss": critic_loss.item(),
            "q1_mean": q1.mean().item(),
            "target_mean": float(y.mean()),
            "residual_variance": resid_var,
            "critic_grad_norm": critic_grad,
            "td_abs": (y - q1).detach().abs().cpu().numpy(),
        }

        self.updates += 1
        if self.updates % cfg.policy_delay == 0:
            pi = self.actor(dens, batch.waters, batch.focus)
            actor_loss = -self.critic.q1_only(dens, batch.waters, batch.focus, pi).mean()
            w = self.bc_weight()
            if expert_batch is not None and w > 0:
                ed = self.density(len(expert_batch.rewards))
                pi_e = self.actor(ed, expert_batch.waters, expert_batch.focus)
                bc_loss = F.mse_loss(pi_e, expert_batch.actions)
                actor_loss = actor_loss + w * bc_loss
                metrics["bc_loss"] = bc_loss.item()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            metrics["actor_grad_norm"] = float(torch.nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.grad_clip))
            self.actor_opt.step()
            metrics["actor_loss"] = actor_loss.item()
            polyak_update(self.actor, self.actor_target, cfg.tau)
            polyak_update(self.critic, self.critic_target, cfg.tau)
        return metrics


# ----------------------------------------------------------------------------
# PPO
# ----------------------------------------------------------------------------
@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.0
    epochs: int = 4
    batch_size: int = 64
    target_kl: float | None = 0.03
    grad_clip: float = 0.5
    embed_dim: int = 128
    num_heads: int = 4
    local_size: int = 9
    hidden: int = 256


class PPOAgent(_BaseAgent):
    """Proximal Policy Optimisation (Schulman et al., 2017) with GAE."""

    def __init__(self, half_extent: float, voxel_size: float, action_dim: int, cfg: PPOConfig, device: torch.device):
        super().__init__(device)
        self.cfg = cfg
        enc = StateEncoder(half_extent, voxel_size, cfg.embed_dim, cfg.num_heads, cfg.local_size)
        self.ac = GaussianActorCritic(enc, action_dim, cfg.hidden).to(device)
        self.optimizer = torch.optim.Adam(self.ac.parameters(), lr=cfg.lr)

    @torch.no_grad()
    def act(self, obs: dict, deterministic: bool = False) -> tuple[np.ndarray, float, float]:
        """Return (action, log-prob, value). The raw Gaussian sample is
        returned (and should be stored); the environment clips it."""
        waters, focus = self._obs_tensors(obs)
        dist, value = self.ac(self.density(1), waters, focus)
        a = dist.mean if deterministic else dist.sample()
        return a[0].cpu().numpy(), float(dist.log_prob(a).sum(-1)), float(value)

    @torch.no_grad()
    def value(self, obs: dict) -> float:
        waters, focus = self._obs_tensors(obs)
        return float(self.ac(self.density(1), waters, focus)[1])

    def update(self, rollout: RolloutBuffer, rng: np.random.Generator) -> dict:
        cfg = self.cfg
        stats = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clip_frac": []}
        stop = False
        for _ in range(cfg.epochs):
            for mb in rollout.minibatches(cfg.batch_size, rng, self.device):
                dist, values = self.ac(self.density(len(mb["focus"])), mb["waters"], mb["focus"])
                log_probs = dist.log_prob(mb["actions"]).sum(-1)
                log_ratio = log_probs - mb["log_probs"]
                ratio = log_ratio.exp()
                adv = mb["advantages"]
                if len(adv) > 1:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                policy_loss = -torch.min(ratio * adv, ratio.clamp(1 - cfg.clip_range, 1 + cfg.clip_range) * adv).mean()
                value_loss = F.mse_loss(values, mb["returns"])
                entropy = dist.entropy().sum(-1).mean()
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ac.parameters(), cfg.grad_clip)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = float(((ratio - 1) - log_ratio).mean())  # Schulman's k3 estimator
                    clip_frac = float(((ratio - 1).abs() > cfg.clip_range).float().mean())
                for k, v in zip(stats, (policy_loss.item(), value_loss.item(), entropy.item(), approx_kl, clip_frac)):
                    stats[k].append(v)
                if cfg.target_kl is not None and approx_kl > 1.5 * cfg.target_kl:
                    stop = True
                    break
            if stop:
                break
        out = {k: float(np.mean(v)) for k, v in stats.items()}
        out["log_std"] = self.ac.log_std.mean().item()
        return out
