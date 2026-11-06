# Refactor notes

This package was rebuilt from a set of research notebooks (TD3, Double/Dueling
DQN with prioritized replay, PPO). Rebuilding it turned up several bugs that
would have quietly degraded or invalidated training runs. Each is listed with
the fix and, where possible, the test that now guards against it.

## Issues that affected learning

| # | Where | Problem | Fix |
|---|-------|---------|-----|
| 1 | TD3 training loop | The environment returned its internal position tensor in every observation and then mutated it in place. Because transitions were written to the buffer *after* the n-step rollout, the stored "current" state was really the state n steps later, identical to the stored next state. The critic was learning Q(s', a) labelled as Q(s, a). | Observations are copies; buffers copy on write. `test_observations_are_independent_copies`, `test_replay_buffer_copies_inputs`. |
| 2 | TD3 training loop | Exploration noise was added to the action *after* the policy output, but the noise-free action was stored. The critic was trained on actions the agent never took. | The environment's action space is the policy's [-1, 1] range, and the agent returns the noisy, clipped action that is both executed and stored. |
| 3 | Actor/critic networks | `LayerNorm(3)` on raw coordinates standardised x, y and z *against each other* for every water, discarding its absolute position (and one degree of freedom). The same module object was also shared between two sub-networks. | Coordinates are divided by the box half-width. |
| 4 | Actor/critic networks | The density map was globally average-pooled to one vector. The map is fixed within a protein, so this vector was constant for every state of an episode and told the policy nothing about where density lies relative to the water it must move. | Each water gets a local density crop (trilinear sampling) as input, plus a global context token. `test_local_crops_match_numpy_interpolation`. |
| 5 | Density loading | Map grid node *i* was placed at fractional coordinate (i + 0.5)/n instead of i/n, shifting the density half a grid spacing away from the waters. | Grid nodes at i/n. `test_load_structure_aligns_density_with_waters`. |
| 6 | Density loading | Structures and maps were matched by taking the first item from two independent `glob` calls, whose order is not guaranteed. | `pair_structure_files` pairs by file stem. `test_pair_structure_files_matches_by_stem`. |
| 7 | Prioritized replay (DQN) | `update_priorities` was a stub and sampling was uniform, so "prioritized" replay was ordinary replay. There were no importance-sampling weights. | Sum-tree PER with stratified sampling, IS weights and beta annealing. `test_prioritized_sampling_is_proportional_and_weighted`. |
| 8 | PPO notebook | The "test" structures were the first ten *training* files, so test metrics were training metrics. | Disjoint train/test sets for both real and synthetic data. |
| 9 | PPO notebook | Parameter-noise exploration perturbed the weights inside `forward()` whenever the network was in training mode, including during rollouts. Old and new log-probabilities came from different random perturbations, so the PPO ratio was noisy even before any update. | Exploration comes from the Gaussian policy only. |
| 10 | PPO notebook | The environment was never reset inside a rollout, so after an episode ended the agent kept acting in a finished episode. | Reset on termination or truncation; GAE masks across episode boundaries; truncated episodes bootstrap from V(s'). `test_gae_matches_hand_computation`. |

## Smaller issues

* **DQN checkpointing:** the best-run MSE was updated *before* the "is this a
  new best?" check, so the best model was never saved.
* **Expert prefill (TD3):** if an episode ended on the first step of an
  n-step window, the window's start state was never set and a stale
  transition was stored.
* **n-step returns:** only every n-th state started a transition. The new
  `NStepAccumulator` emits one transition per step and treats time-limit
  truncation as non-terminal (`test_nstep_truncation_still_bootstraps`).
* **PPO evaluation:** the test loop ignored termination and always ran 2,500
  steps; the reported mean correlation averaged an empty list (NaN).
* **Data loading speed:** symmetry expansion looped over every grid point in
  Python; it is now vectorised with NumPy.
* **Target networks:** `tau = 1e-5` gives target networks a time constant of
  ~100k updates; the default is now 0.005.

## Design changes

* One package, three agents, one environment interface; configs in YAML
  instead of notebook globals; CSV and optional W&B logging.
* Networks are size-agnostic (attention over waters), so one agent can train
  across proteins with different numbers of waters.
* `reward_mode="hybrid"` adds a density-based term that does not need
  ground-truth water positions, which a deployable agent would not have.
* Synthetic structures make every algorithm runnable and testable on a laptop.
* The GCN policy from the PPO notebook depended on a lab-internal library and
  is not included; the PPO agent here uses the shared encoder.
