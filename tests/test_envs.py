import numpy as np
import pytest

from wprl.data import make_synthetic_structure
from wprl.envs import DiscreteWaterEnv, WaterPlacementEnv, sample_density


@pytest.fixture
def structure():
    return make_synthetic_structure(n_waters=6, grid_size=20, seed=0)


def test_observations_are_independent_copies(structure):
    env = WaterPlacementEnv(structure)
    obs, _ = env.reset(seed=0)
    snapshot = obs["waters"].copy()
    env.step(np.array([0.3, -0.2], dtype=np.float32))
    # Stepping must not change an observation that was already returned
    # (in the original notebook, stored states were overwritten this way).
    assert np.array_equal(obs["waters"], snapshot)
    obs["waters"][:] = 99.0
    assert not np.any(env.positions == 99.0)


@pytest.mark.parametrize("action_type", ["spherical", "cartesian"])
def test_action_conversion_round_trip(structure, action_type):
    env = WaterPlacementEnv(structure, action_type=action_type, max_movement=0.5)
    rng = np.random.default_rng(0)
    for _ in range(20):
        a = rng.uniform(-0.95, 0.95, env.action_space.shape).astype(np.float32)
        m = env.action_to_movement(a)
        assert np.allclose(env.action_to_movement(env.movement_to_action(m)), m, atol=1e-5)


@pytest.mark.parametrize("action_type", ["spherical", "cartesian"])
def test_oracle_solves_episode(structure, action_type):
    env = WaterPlacementEnv(structure, action_type=action_type, max_episode_steps=5000)
    env.reset(seed=1)
    for _ in range(5000):
        _, _, term, trunc, info = env.step(env.oracle_action())
        if term or trunc:
            break
    assert term and info["reason"] == "success"


def test_perfect_step_earns_unit_reward(structure):
    env = WaterPlacementEnv(structure, action_type="cartesian", max_movement=0.25, noise=3.0)
    env.reset(seed=2)
    atom = env.focus
    env.solved[:] = False
    action = env.oracle_action()
    _, reward, *_ = env.step(action)
    if not env.solved[atom]:
        assert reward == pytest.approx(1.0, abs=1e-4)


def test_moves_leaving_the_grid_are_rejected(structure):
    env = WaterPlacementEnv(structure, action_type="cartesian", max_movement=1.0)
    env.reset(seed=0)
    env.positions[0] = [env.half_extent - 0.1, 0.0, 0.0]
    env.solved[0] = False
    env.focus = 0
    before = env.positions[0].copy()
    _, reward, _, _, info = env.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert info["skipped"] == "out_of_bounds" and reward == 0.0
    assert np.array_equal(env.positions[0], before)


def test_solved_waters_are_locked(structure):
    env = WaterPlacementEnv(structure, action_type="cartesian")
    env.reset(seed=0)
    env.solved[:] = True
    env.focus = 3
    before = env.positions.copy()
    _, reward, _, _, info = env.step(np.ones(3, dtype=np.float32))
    assert info["skipped"] == "solved" and reward == 0.0
    assert np.array_equal(env.positions, before)


def test_truncation_at_time_limit(structure):
    env = WaterPlacementEnv(structure, max_episode_steps=3, noise=5.0)
    env.reset(seed=0)
    results = [env.step(np.zeros(2, dtype=np.float32)) for _ in range(3)]
    assert [r[3] for r in results] == [False, False, True]
    assert not any(r[2] for r in results)


def test_lowest_density_focus_mode_targets_badly_placed_waters(structure):
    env = WaterPlacementEnv(structure, focus_mode="lowest_density", focus_epsilon=0.0, noise=0.0)
    env.reset(seed=0)
    env.positions[4] += 4.0  # move one water far off its density peak
    assert env._choose_focus() == 4


def test_discrete_env_rewards_milestones_and_mask(structure):
    env = DiscreteWaterEnv(structure, step_size=0.5, milestones=[(100.0, 5.0)], noise=1.0)
    obs, info = env.reset(seed=0)
    env.best_mse = 200.0  # pretend the episode has not yet crossed the milestone
    mse0 = env.mse()
    atom = 0
    direction = np.sign(env.target[atom] - env.positions[atom])
    move = int(np.argmax(np.abs(direction)))
    move_idx = 2 * move + (0 if direction[move] > 0 else 1)
    _, reward, *_ = env.step(atom * 7 + move_idx)
    assert reward == pytest.approx(mse0 - env.mse() + 5.0, abs=1e-5)
    _, reward2, *_ = env.step(atom * 7 + 6)  # "stay": no change, milestone not paid again
    assert reward2 == pytest.approx(0.0, abs=1e-6)

    env.positions[1] = [env.half_extent - 0.1, 0, 0]
    mask = env.action_mask()
    assert mask.shape == (env.n_atoms * 7,)
    assert not mask[1 * 7 + 0] and mask[1 * 7 + 1]


def test_sample_density_matches_voxel_centres(structure):
    vs, half = structure.voxel_size, structure.half_extent
    idx = np.array([[3, 4, 5], [10, 10, 10]])
    centres = (idx + 0.5) * vs - half
    assert np.allclose(sample_density(structure.density, centres, vs), structure.density[tuple(idx.T)], atol=1e-6)
    assert sample_density(structure.density, np.array([[1e3, 0, 0]]), vs)[0] == 0.0
