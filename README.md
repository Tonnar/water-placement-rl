# water-placement-rl

Deep reinforcement learning agents that place water molecules into
experimental protein electron-density maps.

In X-ray crystallography, water molecules are usually placed by peak-picking
in the density map, and poorly modelled waters are a major source of
disagreement between models and experimental data. This project frames
water placement as a sequential control problem: starting from noisy
positions, an agent moves one water at a time until every water sits on its
density peak. The repository contains the environments, three families of
agents implemented from scratch in PyTorch, and tests for each component.

| Agent | Action space | Ingredients |
|-------|--------------|-------------|
| **DQN** | discrete: pick a water and one of 7 axis moves | Double-DQN targets, dueling head, sum-tree prioritized replay with importance-sampling weights, n-step returns, action masking |
| **TD3** | continuous: 2 angles (direction of a fixed-length step) | twin critics, target-policy smoothing, delayed actor updates, n-step returns, oracle demonstrations with decaying behaviour cloning |
| **PPO** | continuous: 3D displacement | clipped surrogate objective, GAE(λ), correct bootstrapping on time-limit truncation, KL early stopping |

## How it works

**Environment.** Each episode starts from the ground-truth waters plus
Gaussian noise. The continuous environment chooses a *focus* water each step
(round-robin, or epsilon-greedy over the waters sitting in the least density)
and the agent chooses how to move it. Waters that land within 0.1 Å of
their target are locked. Moves that would leave the map are rejected.

**Reward.** The default reward is the focus water's progress toward its
target, scaled so a perfect step earns +1, with bonuses for locking a water
and for finishing the episode. This uses ground-truth positions, which a
deployed agent would not have, so `reward_mode: hybrid` adds a term for the
change in map density at the water. Comparing agents trained on the two
signals is a small, concrete instance of optimising a proxy reward.

**Networks.** All agents share one state encoder. Each water gets a
**local density crop** (trilinearly sampled around it) so the policy can see
which direction the density peak lies in; a global 3D-CNN token and the other
waters provide context through multi-head attention. The encoder is
size-agnostic, so one agent can train across proteins with different numbers
of waters.

## Quickstart

```bash
git clone https://github.com/Tonnar/water-placement-rl
cd water-placement-rl
pip install -e ".[dev]"        # add ".[wandb]" for Weights & Biases logging
pytest -q                      # 31 tests, ~10 s on CPU

# Train on synthetic structures (no data files needed)
python -m wprl.train --config configs/td3.yaml
python -m wprl.train --config configs/dqn.yaml
python -m wprl.train --config configs/ppo.yaml

# Override any config value from the command line
python -m wprl.train --config configs/td3.yaml --set train.total_steps=200000 env.reward_mode=hybrid wandb=true
```

Each run writes `config.yaml`, `metrics.csv`, `eval.json` (placement quality
on held-out structures) and `checkpoint.pt` to `runs/<algo>-<timestamp>/`.

To train on real structures, see [`data/README.md`](data/README.md) for the
expected PDB/CCP4 layout.

## Results

<!-- Replace with your own runs: a learning curve (episode_final_mse vs. step)
     and the eval.json numbers for each agent on the held-out test set. -->

*To be added.*

## Repository layout

```
src/wprl/
  data.py      PDB/CCP4 loading with space-group symmetry expansion; synthetic structures
  envs.py      continuous and discrete Gymnasium environments
  networks.py  state encoder (local density crops + attention), actor, critics, Q-network
  buffers.py   uniform and prioritized replay, n-step accumulator, GAE rollout buffer
  agents.py    DQN, TD3 and PPO agents
  train.py     training loops, evaluation and command-line interface
configs/       one YAML file per algorithm
tests/         unit tests for every module plus end-to-end training runs
docs/          refactor notes
```

## Notes

This code was refactored from research notebooks; the
[refactor notes](docs/refactor-notes.md) list the bugs found along the way
and the tests that now guard against them.

## References

* Mnih et al. (2015), *Human-level control through deep reinforcement learning* (DQN)
* van Hasselt, Guez & Silver (2016), *Deep reinforcement learning with double Q-learning*
* Wang et al. (2016), *Dueling network architectures for deep reinforcement learning*
* Schaul et al. (2016), *Prioritized experience replay*
* Fujimoto, van Hoof & Meger (2018), *Addressing function approximation error in actor-critic methods* (TD3)
* Schulman et al. (2016), *High-dimensional continuous control using generalized advantage estimation*
* Schulman et al. (2017), *Proximal policy optimization algorithms*
