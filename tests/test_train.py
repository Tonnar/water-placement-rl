import json
from pathlib import Path

import pytest

from wprl.train import load_config, run

CONFIGS = Path(__file__).resolve().parents[1] / "configs"

TINY = [
    "data.synthetic.n_train=2",
    "data.synthetic.n_test=1",
    "data.synthetic.n_waters=4",
    "data.synthetic.grid_size=12",
    "agent.embed_dim=16",
    "agent.num_heads=2",
    "agent.local_size=3",
    "agent.hidden=16",
    "train.total_steps=240",
    "train.steps_per_structure=120",
    "train.learning_starts=40",
    "train.batch_size=8",
    "train.log_every=60",
    "env.max_episode_steps=60",
]


@pytest.mark.parametrize(
    "algo,extra",
    [
        ("dqn", ["train.prioritized=true"]),
        ("td3", ["train.expert_prefill_steps=50", "agent.bc_weight=1.0"]),
        ("ppo", ["train.rollout_steps=40", "agent.batch_size=20", "agent.epochs=1"]),
    ],
)
def test_end_to_end_training(tmp_path, algo, extra):
    cfg = load_config(str(CONFIGS / f"{algo}.yaml"), TINY + extra + [f"out_dir={tmp_path}/run"])
    results = run(cfg)
    assert "eval/final_mse" in results
    out = tmp_path / "run"
    assert (out / "checkpoint.pt").exists() and (out / "config.yaml").exists()
    assert json.loads((out / "eval.json").read_text()) == results
    assert (out / "metrics.csv").read_text().count("\n") >= 2
