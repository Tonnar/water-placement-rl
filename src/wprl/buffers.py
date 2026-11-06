"""Replay buffers (off-policy) and a rollout buffer with GAE (on-policy).

Buffers hold one structure's transitions, so water arrays have a fixed shape.
The density grid is not stored; the agent reads it from the environment.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Batch:
    waters: torch.Tensor  # (B, N, 3)
    focus: torch.Tensor  # (B,) long
    actions: torch.Tensor  # (B, A) float or (B,) long
    rewards: torch.Tensor  # (B,)
    next_waters: torch.Tensor
    next_focus: torch.Tensor
    discounts: torch.Tensor  # (B,) gamma**k, or 0 if the episode terminated
    weights: torch.Tensor  # (B,) importance-sampling weights (1 without PER)
    indices: np.ndarray  # buffer slots, for priority updates


class NStepAccumulator:
    """Turn a stream of 1-step transitions into n-step transitions.

    Each environment step emits (at most) one transition whose reward is the
    discounted sum of the next ``n`` rewards and whose ``discount`` is
    ``gamma**k`` for the ``k`` steps actually taken, or 0 if the episode
    terminated inside the window. Truncation (time limit) is not termination:
    those transitions still bootstrap from the last state.
    """

    def __init__(self, n: int, gamma: float):
        self.n, self.gamma = n, gamma
        self.window: deque = deque()

    def push(self, obs, action, reward, next_obs, terminated, truncated) -> list[tuple]:
        self.window.append((obs, action, reward))
        out = []
        if terminated or truncated:
            while self.window:
                out.append(self._emit(next_obs, terminated))
        elif len(self.window) == self.n:
            out.append(self._emit(next_obs, False))
        return out

    def _emit(self, next_obs, terminated: bool) -> tuple:
        ret, discount = 0.0, 1.0
        for _, _, r in self.window:
            ret += discount * r
            discount *= self.gamma
        obs, action, _ = self.window.popleft()
        return obs, action, ret, next_obs, 0.0 if terminated else discount


class ReplayBuffer:
    """Uniform circular replay buffer."""

    def __init__(self, capacity: int, n_atoms: int, action_shape: tuple = (), discrete: bool = False, device="cpu"):
        self.capacity, self.device = capacity, device
        self.waters = np.zeros((capacity, n_atoms, 3), dtype=np.float32)
        self.next_waters = np.zeros_like(self.waters)
        self.focus = np.zeros(capacity, dtype=np.int64)
        self.next_focus = np.zeros(capacity, dtype=np.int64)
        self.actions = np.zeros((capacity, *action_shape), dtype=np.int64 if discrete else np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.discounts = np.zeros(capacity, dtype=np.float32)
        self.pos, self.size = 0, 0

    def __len__(self) -> int:
        return self.size

    def add(self, obs: dict, action, reward: float, next_obs: dict, discount: float) -> int:
        i = self.pos
        # Assigning into preallocated arrays copies the data, so later in-place
        # changes to the caller's arrays cannot corrupt stored transitions.
        self.waters[i] = obs["waters"]
        self.focus[i] = obs["focus"]
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_waters[i] = next_obs["waters"]
        self.next_focus[i] = next_obs["focus"]
        self.discounts[i] = discount
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return i

    def _batch(self, idx: np.ndarray, weights: np.ndarray) -> Batch:
        t = lambda a: torch.as_tensor(a, device=self.device)  # noqa: E731
        return Batch(
            waters=t(self.waters[idx]),
            focus=t(self.focus[idx]),
            actions=t(self.actions[idx]),
            rewards=t(self.rewards[idx]),
            next_waters=t(self.next_waters[idx]),
            next_focus=t(self.next_focus[idx]),
            discounts=t(self.discounts[idx]),
            weights=t(weights.astype(np.float32)),
            indices=idx,
        )

    def sample(self, batch_size: int, rng: np.random.Generator) -> Batch:
        idx = rng.integers(0, self.size, size=batch_size)
        return self._batch(idx, np.ones(batch_size, dtype=np.float32))

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """No-op for uniform replay (keeps the agent code identical)."""


class SumTree:
    """Binary tree whose leaves hold priorities and whose nodes hold sums.

    Supports O(log n) updates and O(log n) sampling proportional to priority.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity, dtype=np.float64)

    @property
    def total(self) -> float:
        return float(self.tree[1])

    def update(self, idx: np.ndarray, priorities: np.ndarray) -> None:
        for i, p in zip(np.atleast_1d(idx), np.atleast_1d(priorities)):
            node = int(i) + self.capacity
            self.tree[node] = p
            node //= 2
            while node >= 1:
                self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]
                node //= 2

    def find(self, values: np.ndarray) -> np.ndarray:
        """Leaf index for each cumulative-priority value in [0, total)."""
        out = np.empty(len(values), dtype=np.int64)
        for k, v in enumerate(values):
            node = 1
            while node < self.capacity:
                left = 2 * node
                if v < self.tree[left]:
                    node = left
                else:
                    v -= self.tree[left]
                    node = left + 1
            out[k] = node - self.capacity
        return out

    def leaf(self, idx: np.ndarray) -> np.ndarray:
        return self.tree[np.asarray(idx) + self.capacity]


class PrioritizedReplayBuffer(ReplayBuffer):
    """Proportional prioritized experience replay (Schaul et al., 2016).

    Transitions are sampled with probability ``p_i**alpha / sum_k p_k**alpha``
    where ``p_i = |TD error| + eps``. The bias this introduces is corrected
    with importance-sampling weights ``(N * P(i))**-beta``, normalised by
    their maximum; ``beta`` is annealed toward 1 by the agent.
    """

    def __init__(self, capacity: int, n_atoms: int, action_shape: tuple = (), discrete: bool = False,
                 device="cpu", alpha: float = 0.6, eps: float = 1e-5):
        super().__init__(capacity, n_atoms, action_shape, discrete, device)
        self.alpha, self.eps = alpha, eps
        self.tree = SumTree(capacity)
        self.max_priority = 1.0

    def add(self, obs, action, reward, next_obs, discount) -> int:
        i = super().add(obs, action, reward, next_obs, discount)
        self.tree.update(np.array([i]), np.array([self.max_priority**self.alpha]))
        return i

    def sample(self, batch_size: int, rng: np.random.Generator, beta: float = 0.4) -> Batch:
        # Stratified sampling: one draw from each of `batch_size` equal segments.
        segment = self.tree.total / batch_size
        values = (np.arange(batch_size) + rng.random(batch_size)) * segment
        idx = np.minimum(self.tree.find(values), self.size - 1)
        probs = self.tree.leaf(idx) / self.tree.total
        weights = (self.size * probs) ** (-beta)
        weights /= weights.max()
        return self._batch(idx, weights)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        p = np.abs(np.asarray(td_errors, dtype=np.float64)) + self.eps
        self.max_priority = max(self.max_priority, float(p.max()))
        self.tree.update(indices, p**self.alpha)


class RolloutBuffer:
    """Fixed-length on-policy rollout storage with GAE(lambda) returns."""

    def __init__(self, size: int, n_atoms: int, action_dim: int, gamma: float, gae_lambda: float):
        self.size, self.gamma, self.lam = size, gamma, gae_lambda
        self.waters = np.zeros((size, n_atoms, 3), dtype=np.float32)
        self.focus = np.zeros(size, dtype=np.int64)
        self.actions = np.zeros((size, action_dim), dtype=np.float32)
        self.log_probs = np.zeros(size, dtype=np.float32)
        self.values = np.zeros(size, dtype=np.float32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)  # 1 if the episode ended after this step
        self.advantages = np.zeros(size, dtype=np.float32)
        self.returns = np.zeros(size, dtype=np.float32)
        self.pos = 0

    @property
    def full(self) -> bool:
        return self.pos == self.size

    def reset(self) -> None:
        self.pos = 0

    def add(self, obs, action, log_prob, value, reward, done) -> None:
        i = self.pos
        self.waters[i], self.focus[i] = obs["waters"], obs["focus"]
        self.actions[i], self.log_probs[i], self.values[i] = action, log_prob, value
        self.rewards[i], self.dones[i] = reward, float(done)
        self.pos += 1

    def compute_gae(self, last_value: float) -> None:
        """GAE over the rollout. ``last_value`` is V(s_T) for the state after
        the final step (ignored if that step ended an episode)."""
        gae = 0.0
        for t in reversed(range(self.size)):
            next_value = last_value if t == self.size - 1 else self.values[t + 1]
            not_done = 1.0 - self.dones[t]
            delta = self.rewards[t] + self.gamma * next_value * not_done - self.values[t]
            gae = delta + self.gamma * self.lam * not_done * gae
            self.advantages[t] = gae
        self.returns = self.advantages + self.values

    def minibatches(self, batch_size: int, rng: np.random.Generator, device="cpu"):
        idx = rng.permutation(self.size)
        t = lambda a: torch.as_tensor(a, device=device)  # noqa: E731
        for start in range(0, self.size, batch_size):
            j = idx[start : start + batch_size]
            yield {
                "waters": t(self.waters[j]),
                "focus": t(self.focus[j]),
                "actions": t(self.actions[j]),
                "log_probs": t(self.log_probs[j]),
                "values": t(self.values[j]),
                "advantages": t(self.advantages[j]),
                "returns": t(self.returns[j]),
            }
