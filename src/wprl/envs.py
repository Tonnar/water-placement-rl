"""Gymnasium environments for water placement.

Each episode starts from the ground-truth waters plus Gaussian noise; the agent
moves one water per step to bring every water back onto its density peak.

* :class:`WaterPlacementEnv` -- continuous actions (TD3, PPO). The agent moves
  the current *focus* water, chosen by the environment, either along a
  direction given by two angles (``action_type="spherical"``) or by a 3D
  displacement (``action_type="cartesian"``).
* :class:`DiscreteWaterEnv` -- discrete actions (DQN family). Action
  ``a = atom * 7 + d`` moves ``atom`` one step along +-x, +-y, +-z, or not at all.

Observations are ``{"waters": (N, 3) float32, "focus": int}``. The density
grid is static within an episode, so it is exposed as ``env.density`` instead
of being copied into every observation (and every replay-buffer entry).
Observations are always fresh copies: the environment mutates its internal
positions in place, and sharing that array with a stored observation would
silently overwrite past states.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from wprl.data import Structure

DISCRETE_MOVES = np.array(
    [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1], [0, 0, 0]],
    dtype=np.float32,
)


def sample_density(density: np.ndarray, positions: np.ndarray, voxel_size: float) -> np.ndarray:
    """Trilinearly interpolate the density grid at (M, 3) positions.

    Voxel (i, j, k) is centred at ``(i + 0.5) * voxel_size - half_extent``,
    matching :func:`wprl.data.voxelize_points`. Positions outside the grid
    get density 0.
    """
    g = density.shape[0]
    half = g * voxel_size / 2.0
    u = (np.atleast_2d(positions) + half) / voxel_size - 0.5
    base = np.floor(u).astype(np.int64)
    frac = u - base
    out = np.zeros(len(u), dtype=np.float64)
    for corner in range(8):
        offset = np.array([(corner >> 2) & 1, (corner >> 1) & 1, corner & 1])
        idx = base + offset
        weight = np.prod(np.where(offset == 1, frac, 1.0 - frac), axis=1)
        valid = np.all((idx >= 0) & (idx < g), axis=1)
        out[valid] += weight[valid] * density[idx[valid, 0], idx[valid, 1], idx[valid, 2]]
    return out.astype(np.float32)


class _WaterEnvBase(gym.Env):
    """State, reset, bookkeeping and info shared by both environments."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        structure: Structure,
        noise: float = 1.5,
        termination_mse: float = 0.5,
        termination_reward: float = 2.0,
        max_episode_steps: int = 1000,
    ):
        super().__init__()
        if structure.n_waters == 0:
            raise ValueError("structure has no waters")
        self.structure = structure
        self.target = structure.waters.astype(np.float32)
        self.density = structure.density.astype(np.float32)
        self.n_atoms = structure.n_waters
        self.half_extent = structure.half_extent
        # `noise` is the RMS length of each water's initial displacement.
        self.noise = noise
        self.termination_mse = termination_mse
        self.termination_reward = termination_reward
        self.max_episode_steps = max_episode_steps

        self.observation_space = spaces.Dict(
            {
                "waters": spaces.Box(-np.inf, np.inf, shape=(self.n_atoms, 3), dtype=np.float32),
                "focus": spaces.Discrete(self.n_atoms),
            }
        )
        self.positions = self.target.copy()
        self.focus = 0
        self.step_count = 0

    # -- helpers ---------------------------------------------------------
    def distances(self) -> np.ndarray:
        return np.linalg.norm(self.positions - self.target, axis=1)

    def mse(self) -> float:
        """Mean over waters of the squared distance to their targets (A^2)."""
        return float(np.mean(self.distances() ** 2))

    def _inside(self, p: np.ndarray) -> bool:
        return bool(np.all(np.abs(p) <= self.half_extent))

    def _obs(self) -> dict:
        return {"waters": self.positions.copy(), "focus": int(self.focus)}

    def _info(self, **extra) -> dict:
        return {"mse": self.mse(), **extra}

    def _initial_positions(self) -> np.ndarray:
        per_axis = self.noise / np.sqrt(3.0)
        return (self.target + self.np_random.normal(0.0, per_axis, self.target.shape)).astype(
            np.float32
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.positions = self._initial_positions()
        self.step_count = 0
        return self._obs(), self._info()

    def _finish_step(self, reward: float, info: dict):
        mse = self.mse()
        terminated = mse < self.termination_mse
        if terminated:
            reward += self.termination_reward
            info["reason"] = "success"
        self.step_count += 1
        truncated = (not terminated) and self.step_count >= self.max_episode_steps
        if truncated:
            info["reason"] = "max_steps"
        info["mse"] = mse
        return reward, terminated, truncated, info


class WaterPlacementEnv(_WaterEnvBase):
    """Continuous control: move the focus water one step per action.

    Rewards (``reward_mode="distance"``): the focus water's progress toward its
    target divided by ``max_movement`` (so a perfect step earns +1), plus
    ``success_reward`` the first time it lands within ``success_threshold``,
    after which it is locked in place. ``reward_mode="hybrid"`` adds
    ``density_weight`` times the change in map density at the water, a signal
    that does not need the ground-truth positions.

    The episode terminates when :meth:`mse` drops below ``termination_mse``.
    Moves that would take a water out of the grid are rejected (zero reward).
    """

    def __init__(
        self,
        structure: Structure,
        action_type: str = "spherical",
        max_movement: float = 0.25,
        noise: float = 1.5,
        success_threshold: float = 0.1,
        success_reward: float = 3.0,
        termination_mse: float = 0.5,
        termination_reward: float = 2.0,
        max_episode_steps: int = 1000,
        focus_mode: str = "round_robin",
        focus_epsilon: float = 0.3,
        reward_mode: str = "distance",
        density_weight: float = 1.0,
    ):
        super().__init__(structure, noise, termination_mse, termination_reward, max_episode_steps)
        if action_type not in ("spherical", "cartesian"):
            raise ValueError("action_type must be 'spherical' or 'cartesian'")
        if focus_mode not in ("round_robin", "lowest_density"):
            raise ValueError("focus_mode must be 'round_robin' or 'lowest_density'")
        if reward_mode not in ("distance", "hybrid"):
            raise ValueError("reward_mode must be 'distance' or 'hybrid'")
        self.action_type = action_type
        self.max_movement = max_movement
        self.success_threshold = success_threshold
        self.success_reward = success_reward
        self.focus_mode = focus_mode
        self.focus_epsilon = focus_epsilon
        self.reward_mode = reward_mode
        self.density_weight = density_weight
        # Actions live in [-1, 1]: exactly the range of a tanh policy, so the
        # action an agent stores is the action the environment executed.
        dim = 2 if action_type == "spherical" else 3
        self.action_space = spaces.Box(-1.0, 1.0, shape=(dim,), dtype=np.float32)
        self.solved = np.zeros(self.n_atoms, dtype=bool)

    def action_to_movement(self, action: np.ndarray) -> np.ndarray:
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        if self.action_type == "spherical":
            theta = (a[0] + 1.0) * 0.5 * np.pi  # [0, pi]
            phi = (a[1] + 1.0) * np.pi  # [0, 2 pi]
            direction = np.array(
                [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)],
                dtype=np.float32,
            )
            return direction * self.max_movement
        return a * self.max_movement

    def movement_to_action(self, movement: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`action_to_movement` (used by the oracle)."""
        m = np.asarray(movement, dtype=np.float64)
        if self.action_type == "cartesian":
            return np.clip(m / self.max_movement, -1, 1).astype(np.float32)
        r = np.linalg.norm(m)
        if r == 0:
            return np.zeros(2, dtype=np.float32)
        theta = np.arccos(np.clip(m[2] / r, -1, 1))
        phi = np.arctan2(m[1], m[0]) % (2 * np.pi)
        return np.array([2 * theta / np.pi - 1, phi / np.pi - 1], dtype=np.float32)

    def _choose_focus(self) -> int:
        if self.focus_mode == "round_robin":
            return (self.focus + 1) % self.n_atoms
        # Epsilon-greedy over the 10% of waters sitting in the least density.
        if self.np_random.random() < self.focus_epsilon:
            return int(self.np_random.integers(self.n_atoms))
        dens = sample_density(self.density, self.positions, self.structure.voxel_size)
        k = max(1, self.n_atoms // 10)
        worst = np.argsort(dens)[:k]
        return int(self.np_random.choice(worst))

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.solved = self.distances() < self.success_threshold
        self.focus = 0 if self.focus_mode == "round_robin" else self._choose_focus()
        return self._obs(), self._info()

    def step(self, action):
        atom = self.focus
        reward, info = 0.0, {}
        if self.solved[atom]:
            info["skipped"] = "solved"
        else:
            old = self.positions[atom].copy()
            new = old + self.action_to_movement(action)
            if self._inside(old) and not self._inside(new):
                info["skipped"] = "out_of_bounds"
            else:
                target = self.target[atom]
                d_old = float(np.linalg.norm(old - target))
                d_new = float(np.linalg.norm(new - target))
                self.positions[atom] = new
                if d_new < self.success_threshold:
                    reward = float(self.success_reward)
                    self.solved[atom] = True
                else:
                    reward = (d_old - d_new) / self.max_movement
                if self.reward_mode == "hybrid":
                    vs = self.structure.voxel_size
                    gain = sample_density(self.density, new, vs) - sample_density(self.density, old, vs)
                    reward += self.density_weight * float(gain[0])
        reward, terminated, truncated, info = self._finish_step(reward, info)
        info["n_solved"] = int(self.solved.sum())
        self.focus = self._choose_focus()
        return self._obs(), reward, terminated, truncated, info

    def oracle_action(self) -> np.ndarray:
        """Heuristic expert: step the focus water straight toward its target.

        Used to prefill the replay buffer and for behaviour cloning. It reads
        the ground truth, so it is a training aid, not a baseline.
        """
        diff = self.target[self.focus] - self.positions[self.focus]
        dist = np.linalg.norm(diff)
        if dist == 0:
            return np.zeros(self.action_space.shape, dtype=np.float32)
        # Spherical actions always move a full step; cartesian ones can stop short.
        step = diff / dist * min(dist, self.max_movement)
        return self.movement_to_action(step)


class DiscreteWaterEnv(_WaterEnvBase):
    """Discrete control: pick any water and one of 7 axis-aligned moves.

    Reward is the decrease in :meth:`mse`, plus one-off ``milestones`` bonuses
    -- a list of ``(mse_threshold, bonus)`` pairs paid the first time the
    episode's best MSE drops below each threshold -- plus the termination bonus.
    """

    def __init__(
        self,
        structure: Structure,
        step_size: float = 0.5,
        noise: float = 1.5,
        milestones: list[tuple[float, float]] | None = None,
        termination_mse: float = 0.5,
        termination_reward: float = 10.0,
        max_episode_steps: int = 2000,
    ):
        super().__init__(structure, noise, termination_mse, termination_reward, max_episode_steps)
        self.step_size = step_size
        self.milestones = sorted(milestones or [], key=lambda m: -m[0])
        self.n_moves = len(DISCRETE_MOVES)
        self.action_space = spaces.Discrete(self.n_atoms * self.n_moves)
        self.best_mse = np.inf

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = super().reset(seed=seed, options=options)
        self.best_mse = self.mse()
        return obs, info

    def action_mask(self) -> np.ndarray:
        """Boolean (N * 7,) mask of moves that keep the water inside the grid."""
        new = self.positions[:, None, :] + DISCRETE_MOVES[None] * self.step_size
        return np.all(np.abs(new) <= self.half_extent, axis=-1).reshape(-1)

    def step(self, action):
        atom, move = divmod(int(action), self.n_moves)
        mse_before = self.mse()
        new = self.positions[atom] + DISCRETE_MOVES[move] * self.step_size
        info = {}
        if self._inside(new):
            self.positions[atom] = new
        else:
            info["skipped"] = "out_of_bounds"
        mse_after = self.mse()
        reward = mse_before - mse_after
        for threshold, bonus in self.milestones:
            if mse_after < threshold <= self.best_mse:
                reward += bonus
        self.best_mse = min(self.best_mse, mse_after)
        reward, terminated, truncated, info = self._finish_step(float(reward), info)
        return self._obs(), reward, terminated, truncated, info
