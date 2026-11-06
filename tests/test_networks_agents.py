import numpy as np
import pytest
import torch

from wprl.agents import DQNAgent, DQNConfig, PPOAgent, PPOConfig, TD3Agent, TD3Config
from wprl.buffers import PrioritizedReplayBuffer, ReplayBuffer, RolloutBuffer
from wprl.data import make_synthetic_structure
from wprl.envs import DiscreteWaterEnv, WaterPlacementEnv, sample_density
from wprl.networks import QNetwork, StateEncoder, local_density_crops

SMALL = dict(embed_dim=32, num_heads=2, local_size=5, hidden=32)
DEVICE = torch.device("cpu")


@pytest.fixture
def structure():
    return make_synthetic_structure(n_waters=5, grid_size=16, seed=3)


def test_local_crops_match_numpy_interpolation(structure):
    dens = torch.as_tensor(structure.density).unsqueeze(0)
    centres = torch.as_tensor(structure.waters[:3] + 0.3).unsqueeze(0)
    crops = local_density_crops(dens, centres, structure.half_extent, structure.voxel_size, size=3)
    centre_values = crops[0, :, 1, 1, 1].numpy()
    expected = sample_density(structure.density, structure.waters[:3] + 0.3, structure.voxel_size)
    assert np.allclose(centre_values, expected, atol=1e-5)
    # An offset of +1 voxel along x in the crop equals sampling 1 A further along x.
    shifted = sample_density(structure.density, structure.waters[:3] + 0.3 + [1.0, 0, 0], structure.voxel_size)
    assert np.allclose(crops[0, :, 2, 1, 1].numpy(), shifted, atol=1e-5)


def test_networks_handle_any_number_of_waters(structure):
    enc = StateEncoder(structure.half_extent, structure.voxel_size, 32, 2, 5)
    q = QNetwork(enc, hidden=32)
    dens = torch.as_tensor(structure.density).expand(2, -1, -1, -1)
    for n in (3, 9):
        waters = torch.randn(2, n, 3)
        assert q(dens, waters).shape == (2, n * 7)
        assert enc(dens, waters, torch.tensor([0, n - 1])).shape == (2, 32)


def _fill(buffer, env, policy, n, rng):
    obs, _ = env.reset(seed=0)
    for _ in range(n):
        a = policy(obs)
        nxt, r, term, trunc, _ = env.step(a)
        buffer.add(obs, a, r, nxt, 0.0 if term else 0.9)
        obs = env.reset()[0] if term or trunc else nxt


def test_dqn_update_changes_weights_and_uses_per(structure):
    rng = np.random.default_rng(0)
    env = DiscreteWaterEnv(structure)
    agent = DQNAgent(structure.half_extent, structure.voxel_size, DQNConfig(target_update_every=2, **SMALL), DEVICE)
    agent.set_structure(structure)
    buf = PrioritizedReplayBuffer(256, env.n_atoms, (), discrete=True)
    _fill(buf, env, lambda o: agent.act(o, 0.5, rng, env.action_mask()), 64, rng)
    before = [p.clone() for p in agent.q.parameters()]
    for _ in range(2):
        batch = buf.sample(16, rng)
        m = agent.update(batch)
        buf.update_priorities(batch.indices, m.pop("td_abs"))
    assert np.isfinite(m["loss"])
    assert any(not torch.equal(a, b) for a, b in zip(before, agent.q.parameters()))
    # Target network was synced after 2 updates.
    for p, tp in zip(agent.q.parameters(), agent.q_target.parameters()):
        assert torch.equal(p, tp)


def test_td3_update_with_behaviour_cloning(structure):
    rng = np.random.default_rng(0)
    env = WaterPlacementEnv(structure)
    cfg = TD3Config(policy_delay=1, bc_weight=1.0, **SMALL)
    agent = TD3Agent(structure.half_extent, structure.voxel_size, 2, cfg, DEVICE)
    agent.set_structure(structure)
    buf = ReplayBuffer(256, env.n_atoms, (2,))
    _fill(buf, env, lambda o: agent.act(o, 0.3, rng), 64, rng)
    demos = ReplayBuffer(256, env.n_atoms, (2,))
    _fill(demos, env, lambda o: env.oracle_action(), 64, rng)
    target_before = [p.clone() for p in agent.actor_target.parameters()]
    m = agent.update(buf.sample(16, rng), demos.sample(16, rng))
    for k in ("critic_loss", "actor_loss", "bc_loss", "residual_variance"):
        assert np.isfinite(m[k]), k
    moved = [not torch.equal(a, b) for a, b in zip(target_before, agent.actor_target.parameters())]
    assert any(moved)
    a = agent.act(env.reset(seed=0)[0], noise_std=0.5, rng=rng)
    assert a.shape == (2,) and np.all(np.abs(a) <= 1)


def test_ppo_update_runs(structure):
    rng = np.random.default_rng(0)
    env = WaterPlacementEnv(structure, action_type="cartesian")
    cfg = PPOConfig(batch_size=16, epochs=2, **SMALL)
    agent = PPOAgent(structure.half_extent, structure.voxel_size, 3, cfg, DEVICE)
    agent.set_structure(structure)
    roll = RolloutBuffer(32, env.n_atoms, 3, 0.99, 0.95)
    obs, _ = env.reset(seed=0)
    while not roll.full:
        a, lp, v = agent.act(obs)
        nxt, r, term, trunc, _ = env.step(a)
        roll.add(obs, a, lp, v, r, term or trunc)
        obs = env.reset()[0] if term or trunc else nxt
    roll.compute_gae(agent.value(obs))
    m = agent.update(roll, rng)
    assert all(np.isfinite(v) for v in m.values())
