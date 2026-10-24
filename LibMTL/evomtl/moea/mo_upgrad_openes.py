"""Multi-objective OpenES with UPGrad Jacobian aggregation (MO-UPGrad-OpenES).

OpenES (:mod:`.openes`) ranks a population by one scalar fitness. Scalarizing several losses into one fitness does
not guarantee Pareto-stationary progress, so here every objective is shaped **separately**, which yields one
pseudo-gradient per objective, i.e. a Jacobian ``J`` of shape ``[K, D]``:

    g_k = shaped_k @ eps / (N * sigma) = shaped_k @ z / (N * sigma**2),   shaped_k = shape(F[:, k])

``J`` is aggregated with UPGrad (Quinton & Rey 2024): every row ``g_k`` is projected onto the dual cone of the rows of
``J`` and the projections are averaged. The result never conflicts with any objective, ``g_k . d >= 0`` for all
``k``. ``theta`` is updated with the optimizer (Adam by default) and ``d`` as its gradient.

References:
- OpenES: https://arxiv.org/abs/1703.03864
- UPGrad: https://arxiv.org/abs/2406.16232
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch

from .es_common import shape_fitness, upgrad_aggregate
from .es_loop import make_sigma_rule, run_es


class UPGradJacobian:
    """Estimator of MO-UPGrad-OpenES: per-objective pseudo-gradients combined by UPGrad (``estimate`` of
    :func:`~.es_loop.run_es`)."""

    aggregator = "upgrad"

    def __init__(self, norm_eps: float = 1e-4, reg_eps: float = 1e-4):
        self.norm_eps, self.reg_eps = float(norm_eps), float(reg_eps)

    def estimate(self, F: np.ndarray, eps: torch.Tensor, scale: Any, fitness_shaping: str):
        shaped = shape_fitness(F, fitness_shaping)  # [N, K]
        J = shaped.T @ eps * scale  # [K, D]
        direction, w = upgrad_aggregate(J, self.norm_eps, self.reg_eps)
        return J, direction, w, shaped @ (w / w.abs().sum().clamp_min(1e-30))

    def true_direction(self, J_true: torch.Tensor) -> torch.Tensor:
        return upgrad_aggregate(J_true, self.norm_eps, self.reg_eps)[0]


def run_mo_upgrad_openes(
    trainer: Any,
    train_dataloaders: Any,
    num_iterations: int,
    pop_size: int,
    sigma: float = 0.01,
    sigma_decay: float = 1.0,
    sigma_min: float = 0.0,
    sigma_schedule: str = "constant",
    upgrad_norm_eps: float = 1e-4,
    upgrad_reg_eps: float = 1e-4,
    noise_sampler: Optional[Any] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """MO-UPGrad-OpenES ask-and-tell on the weights ``theta`` (see module docstring and :func:`~.es_loop.run_es` for
    the shared parameters).

    Parameters
    ----------
    sigma, sigma_decay, sigma_min, sigma_schedule
        Perturbation std and its schedule, as in :func:`~.openes.run_openes`.
    upgrad_norm_eps, upgrad_reg_eps
        UPGrad: epsilon of the Gramian normalization and its diagonal regularization.
    noise_sampler : object, optional
        Replaces the isotropic noise (Guided ES, :func:`~.mo_upgrad_guidedes.run_mo_upgrad_guidedes`).
    """
    rule = make_sigma_rule(sigma, sigma_decay, sigma_min, sigma_schedule, num_iterations)
    return run_es(trainer, train_dataloaders, "MO-UPGrad-OpenES", UPGradJacobian(upgrad_norm_eps, upgrad_reg_eps),
                  rule, num_iterations, pop_size, sigma=sigma, noise_sampler=noise_sampler, **kwargs)
