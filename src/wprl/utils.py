"""Seeding, device selection, target-network updates and metric logging."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return np.random.default_rng(seed)


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@torch.no_grad()
def polyak_update(source: torch.nn.Module, target: torch.nn.Module, tau: float) -> None:
    """target <- (1 - tau) * target + tau * source."""
    for p, tp in zip(source.parameters(), target.parameters()):
        tp.mul_(1.0 - tau).add_(p, alpha=tau)


def linear_schedule(start: float, end: float, duration: int, step: int) -> float:
    frac = min(1.0, step / max(1, duration))
    return start + frac * (end - start)


def coverage_precision(pred: np.ndarray, true: np.ndarray, threshold: float = 0.5) -> tuple[float, float]:
    """Fraction of true waters with a prediction within ``threshold`` A
    (coverage / recall), and fraction of predictions within ``threshold`` A
    of a true water (precision)."""
    if len(pred) == 0 or len(true) == 0:
        return 0.0, 0.0
    d = np.linalg.norm(true[:, None, :] - pred[None, :, :], axis=-1)
    return float((d.min(axis=1) <= threshold).mean()), float((d.min(axis=0) <= threshold).mean())


class Logger:
    """Append metrics to a CSV file and, optionally, to Weights & Biases."""

    def __init__(self, out_dir: str | Path, config: dict | None = None, use_wandb: bool = False,
                 project: str = "water-placement-rl"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.out_dir / "metrics.csv"
        self._fields: list[str] | None = None
        self._wandb = None
        if use_wandb:
            import wandb  # optional dependency

            self._wandb = wandb.init(project=project, config=config or {}, dir=str(self.out_dir))

    def log(self, step: int, **metrics: float) -> None:
        row = {"step": step, **{k: float(v) for k, v in metrics.items()}}
        if self._fields is None:
            self._fields = list(row)
            with self.path.open("w", newline="") as f:
                csv.DictWriter(f, fieldnames=self._fields).writeheader()
        for key in row:
            if key not in self._fields:
                # New metric mid-run: rewrite the header so columns stay aligned.
                self._fields.append(key)
                self._rewrite_header()
        with self.path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=self._fields, restval="").writerow(row)
        if self._wandb is not None:
            self._wandb.log(row, step=step)

    def _rewrite_header(self) -> None:
        rows = list(csv.DictReader(self.path.open()))
        with self.path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self._fields, restval="")
            w.writeheader()
            w.writerows(rows)

    def close(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()
