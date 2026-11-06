import numpy as np
import pytest

from wprl.buffers import NStepAccumulator, PrioritizedReplayBuffer, ReplayBuffer, RolloutBuffer, SumTree


def _obs(v):
    return {"waters": np.full((2, 3), v, dtype=np.float32), "focus": 0}


def test_nstep_returns_and_discounts():
    acc = NStepAccumulator(n=3, gamma=0.5)
    assert acc.push(_obs(0), 0, 1.0, _obs(1), False, False) == []
    assert acc.push(_obs(1), 1, 2.0, _obs(2), False, False) == []
    (o, a, ret, nxt, disc), = acc.push(_obs(2), 2, 4.0, _obs(3), False, False)
    assert a == 0 and ret == pytest.approx(1 + 0.5 * 2 + 0.25 * 4) and disc == pytest.approx(0.125)
    assert nxt["waters"][0, 0] == 3

    # Termination flushes the window with zero discount (no bootstrapping).
    out = acc.push(_obs(3), 3, 8.0, _obs(4), True, False)
    assert [t[1] for t in out] == [1, 2, 3]
    assert out[0][2] == pytest.approx(2 + 0.5 * 4 + 0.25 * 8)
    assert all(t[4] == 0.0 for t in out)


def test_nstep_truncation_still_bootstraps():
    acc = NStepAccumulator(n=3, gamma=0.5)
    acc.push(_obs(0), 0, 1.0, _obs(1), False, False)
    out = acc.push(_obs(1), 1, 1.0, _obs(2), False, True)
    assert [t[4] for t in out] == [pytest.approx(0.25), pytest.approx(0.5)]


def test_replay_buffer_copies_inputs():
    buf = ReplayBuffer(10, n_atoms=2, action_shape=(2,))
    obs = _obs(1.0)
    buf.add(obs, np.zeros(2), 0.0, _obs(2.0), 0.9)
    obs["waters"][:] = -5.0
    assert buf.waters[0, 0, 0] == 1.0


def test_sum_tree_finds_leaves():
    tree = SumTree(4)
    tree.update(np.arange(4), np.array([1.0, 2.0, 3.0, 4.0]))
    assert tree.total == 10.0
    assert list(tree.find(np.array([0.5, 1.5, 3.5, 9.9]))) == [0, 1, 2, 3]


def test_prioritized_sampling_is_proportional_and_weighted():
    rng = np.random.default_rng(0)
    buf = PrioritizedReplayBuffer(4, n_atoms=2, action_shape=(), discrete=True, alpha=1.0, eps=0.0)
    for i in range(4):
        buf.add(_obs(i), i, 0.0, _obs(i), 1.0)
    buf.update_priorities(np.arange(4), np.array([1.0, 1.0, 1.0, 7.0]))
    counts = np.zeros(4)
    for _ in range(500):
        batch = buf.sample(8, rng, beta=1.0)
        counts += np.bincount(batch.indices, minlength=4)
    assert counts[3] / counts.sum() == pytest.approx(0.7, abs=0.05)
    batch = buf.sample(64, rng, beta=1.0)
    w = batch.weights.numpy()
    assert w.max() == pytest.approx(1.0)
    # The over-sampled transition gets the smallest weight.
    assert w[batch.indices == 3].max() < w[batch.indices != 3].min()


def test_gae_matches_hand_computation():
    buf = RolloutBuffer(3, n_atoms=1, action_dim=1, gamma=0.9, gae_lambda=0.8)
    obs = {"waters": np.zeros((1, 3), np.float32), "focus": 0}
    for r, v, d in [(1.0, 0.5, False), (0.0, 0.4, True), (2.0, 0.3, False)]:
        buf.add(obs, np.zeros(1), 0.0, v, r, d)
    buf.compute_gae(last_value=1.0)
    d2 = 2.0 + 0.9 * 1.0 - 0.3
    d1 = 0.0 - 0.4  # episode ended: no bootstrap
    d0 = 1.0 + 0.9 * 0.4 - 0.5
    expected = [d0 + 0.9 * 0.8 * d1, d1, d2]
    assert np.allclose(buf.advantages, expected)
    assert np.allclose(buf.returns, np.array(expected) + [0.5, 0.4, 0.3])
