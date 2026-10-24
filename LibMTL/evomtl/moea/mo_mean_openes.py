"""Multi-objective OpenES with the mean of the per-objective pseudo-gradients (MO-Mean-OpenES).

The ablation of MO-UPGrad-OpenES (:mod:`.mo_upgrad_openes`) without the dual-cone step: every objective is shaped
separately (one pseudo-gradient ``g_k`` per objective, the rows of ``J``) and the ES direction is their plain average
``d = mean_k g_k``. With raw fitness this equals OpenES on the mean loss up to the shaping; with centered ranks every
objective weighs equally whatever the size of its loss.

References:
- OpenES: https://arxiv.org/abs/1703.03864
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch

from .es_common import shape_fitness
from .es_loop import make_sigma_rule, run_es


class MeanJacobian:
    """Estimator of MO-Mean-OpenES: per-objective pseudo-gradients, averaged (``estimate`` of
    :func:`~.es_loop.run_es`)."""

    aggregator = "mean"

    def estimate(self, F: np.ndarray, eps: torch.Tensor, scale: Any, fitness_shaping: str):
        shaped = shape_fitness(F, fitness_shaping)  # [N, K]
        J = shaped.T @ eps * scale  # [K, D]
        w = self.weights(J)
        return J, w @ J, w, shaped @ (w / w.abs().sum().clamp_min(1e-30))

    def true_direction(self, J_true: torch.Tensor) -> torch.Tensor:
        return self.weights(J_true) @ J_true

    @staticmethod
    def weights(J: torch.Tensor) -> torch.Tensor:
        return torch.full((J.shape[0],), 1.0 / J.shape[0], dtype=J.dtype)


def run_mo_mean_openes(
    trainer: Any,
    train_dataloaders: Any,
    num_iterations: int,
    pop_size: int,
    sigma: float = 0.01,
    sigma_decay: float = 1.0,
    sigma_min: float = 0.0,
    sigma_schedule: str = "constant",
    noise_sampler: Optional[Any] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """MO-Mean-OpenES ask-and-tell on the weights ``theta`` (see module docstring and :func:`~.es_loop.run_es` for
    the shared parameters); ``sigma*`` as in :func:`~.openes.run_openes`."""
    rule = make_sigma_rule(sigma, sigma_decay, sigma_min, sigma_schedule, num_iterations)
    return run_es(trainer, train_dataloaders, "MO-Mean-OpenES", MeanJacobian(), rule, num_iterations, pop_size,
                  sigma=sigma, noise_sampler=noise_sampler, **kwargs)
