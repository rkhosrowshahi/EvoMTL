"""Minimal trainer adapter so the OpenES-family runners of :mod:`LibMTL.evomtl.moea` run on the toy problems
(:mod:`problems`).

The runners only touch the trainer through a few hooks: the flat weight vector (``theta_0``, ``set_theta``), the
per-task losses of the current weights (``evaluate_multiobjective_losses``, ``forward4loss``), mini-batch sampling and
the logging hooks. Here the "network" is the point ``x`` in R^D, there is no data (one dummy batch) and the logging
hooks are no-ops; ``metrics_snapshot_from_prefetched_batches``, called once per generation on the updated search mean,
records the trajectory.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch
from torch import nn

from problems import make_problem


class ToyPoint(nn.Module):
    def __init__(self, init: torch.Tensor):
        super().__init__()
        self.x = nn.Parameter(init.clone().float())


class ToyTrainer:
    """Trainer interface of the ES runners for the problem ``name`` of :func:`problems.make_problem` (its settings in
    ``problem_kwargs``, e.g. ``scale`` of ``nashmtl``; losses shifted by ``loss_offset``; see ``runner.py``)."""

    def __init__(self, init, name: str = "nashmtl", loss_offset: float = 0.0, **problem_kwargs):
        self.problem = make_problem(name, **problem_kwargs)
        self.loss_offset = float(loss_offset)
        self.model = ToyPoint(torch.as_tensor(init, dtype=torch.float32))
        self.theta_0 = self.model.x.detach().clone()
        self.task_num = self.problem.n_obj
        self.task_name = [f"f{k + 1}" for k in range(self.task_num)]
        self.device = torch.device("cpu")
        self.save_path = None
        self.last_evo = None
        self.trajectory: List[np.ndarray] = [self.theta_0.numpy().copy()]

    # weights
    def set_theta(self, theta) -> None:
        with torch.no_grad():
            self.model.x.copy_(torch.as_tensor(theta, dtype=torch.float32))

    # losses
    def forward4loss(self, model: nn.Module, inp: Any, gts: Any) -> torch.Tensor:
        return self.problem(model.x) + self.loss_offset

    def evaluate_multiobjective_losses(self, dataloaders: Any, prefetched_batches: Any = None) -> np.ndarray:
        with torch.no_grad():
            return self.forward4loss(self.model, None, None).numpy()

    def _sample_batches(self, dataloaders: Any, num_batches: int) -> list:
        return [(None, None)]

    def metrics_snapshot_from_prefetched_batches(self, batches: list) -> Dict[str, Any]:
        self.trajectory.append(self.model.x.detach().numpy().copy())
        loss = self.evaluate_multiobjective_losses(None)
        return {"loss_item": loss, "results": {t: [float("nan")] for t in self.task_name}}

    # logging hooks (no-ops)
    def _print_evo_progress(self, *args: Any, **kwargs: Any) -> None:
        pass

    def _wandb_log_evo_front_metrics(self, *args: Any, **kwargs: Any) -> None:
        pass

    def _wandb_log_evo_objective_plot(self, *args: Any, **kwargs: Any) -> None:
        pass
