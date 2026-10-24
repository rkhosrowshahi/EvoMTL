"""OpenES on the sum of the task losses (single objective).

OpenES (Salimans et al. 2017; evosax ``Open_ES``) scalarizes the task losses into one fitness, their equal-weight sum
``f = sum_k L_k``, and estimates its search gradient from the shaped fitness of the population:

    g = shaped @ eps / (N * sigma) = shaped @ z / (N * sigma**2),   shaped = shape(f)

(the same estimator as evosax ``Open_ES._tell``). ``theta`` is updated with the optimizer (Adam by default) and ``g``
as its gradient. This is the scalarized baseline of the multi-objective variants (:mod:`.mo_upgrad_openes`,
:mod:`.mo_mgda_openes`), which estimate one pseudo-gradient per objective instead.

References:
- OpenES: https://arxiv.org/abs/1703.03864
- evosax: evosax/algorithms/distribution_based/open_es.py
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch

from .es_common import shape_fitness
from .es_loop import make_sigma_rule, run_es


class SumOfLosses:
    """Estimator of OpenES: one pseudo-gradient, of the summed loss (``estimate`` of :func:`~.es_loop.run_es`)."""

    aggregator = "ws"

    def estimate(self, F: np.ndarray, eps: torch.Tensor, scale: Any, fitness_shaping: str):
        shaped = shape_fitness(F.sum(axis=1), fitness_shaping)  # [N]
        J = (shaped @ eps * scale).unsqueeze(0)
        return J, J[0], torch.ones(1, dtype=torch.float32), shaped

    def true_direction(self, J_true: torch.Tensor) -> torch.Tensor:
        return J_true.sum(dim=0)


def run_openes(
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
    """OpenES ask-and-tell on the weights ``theta`` (see module docstring and :func:`~.es_loop.run_es` for the
    shared parameters: optimizer, fitness shaping, antithetic sampling, diagnostics).

    Parameters
    ----------
    sigma, sigma_decay, sigma_min
        Perturbation std; after each update ``sigma = max(sigma * sigma_decay, sigma_min)``.
    sigma_schedule : str
        ``"constant"`` (default; ``sigma_decay`` applies) or ``"cosine"``: generation ``t`` (0-based) samples with
        ``sigma_min + (sigma - sigma_min) * (1 + cos(pi * t / num_iterations)) / 2``. Needs ``sigma_decay = 1``.
    noise_sampler : object, optional
        Replaces the isotropic noise (Guided ES, :func:`~.guidedes.run_guidedes`).
    """
    rule = make_sigma_rule(sigma, sigma_decay, sigma_min, sigma_schedule, num_iterations)
    return run_es(trainer, train_dataloaders, "OpenES", SumOfLosses(), rule, num_iterations, pop_size,
                  sigma=sigma, noise_sampler=noise_sampler, **kwargs)
