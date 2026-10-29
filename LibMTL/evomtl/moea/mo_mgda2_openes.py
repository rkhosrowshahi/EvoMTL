"""Multi-objective OpenES with MGDA on the non-negligible pseudo-gradients (MO-MGDA2-OpenES).

As in MO-UPGrad-OpenES (:mod:`.mo_upgrad_openes`), every objective is shaped **separately**, which yields one
pseudo-gradient per objective, the rows of ``J`` (``[K, D]``):

    g_k = shaped_k @ eps / (N * sigma),   shaped_k = shape(F[:, k])

MGDA (Desideri 2012) takes the minimum-norm point of their convex hull, ``d = J^T alpha*`` with ``alpha*`` the
argmin over the simplex of ``||J^T alpha||^2``. When one objective is (nearly) minimized its gradient is (nearly)
zero, the hull contains (nearly) the origin and ``d`` vanishes, although the other objectives can still decrease
without increasing it at first order: MGDA stops at weakly Pareto-optimal points (e.g. on the axes of
``f = (x_1^2, x_2^2)``).

MGDA2 first drops the negligible pseudo-gradients, greedily from the smallest: with the active set ``A`` (all
objectives at first), the smallest ``g_k`` in ``A`` is dropped while

    ||g_k|| < drop_eps * ||d_{A \\ k}||,     d_{A \\ k} = min-norm point of the hull of {g_j : j in A, j != k}

and at least two objectives remain; then ``d`` is the MGDA direction of the active set. Every active objective keeps
``g_j . d >= ||d||^2`` (MGDA's common descent); a dropped one changes by at most ``drop_eps ||d_{A \\ k}|| ||d||``
per unit step at first order, i.e. ``drop_eps`` bounds its sacrifice relative to the progress of the others. At a
Pareto-stationary point where the gradients conflict, ``d_{A \\ k}`` is itself small and nothing is dropped; the test
compares gradients to each other, so it does not depend on the scale of the losses. A dropped objective whose loss
then rises gets a larger gradient and rejoins the hull at the next iteration. ``drop_eps = 0`` is plain MGDA.

``theta`` is updated with the optimizer (Adam by default) and ``d`` as its gradient.

References:
- OpenES: https://arxiv.org/abs/1703.03864
- MGDA: Desideri, Multiple-gradient descent algorithm (MGDA) for multiobjective optimization,
  C. R. Math. 350 (2012), https://doi.org/10.1016/j.crma.2012.03.014
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from .es_common import shape_fitness
from .es_loop import make_sigma_rule, run_es
from .mo_mgda_openes import min_norm_simplex


def mgda2_aggregate(J: torch.Tensor, drop_eps: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    """MGDA after dropping the negligible rows of ``J`` (``[K, D]``); return ``(d, alpha)`` with ``d = J^T alpha``
    and ``alpha = 0`` on the dropped objectives (see module docstring)."""
    M = (J @ J.T).double().numpy()
    norms = np.sqrt(np.clip(np.diag(M), 0.0, None))
    active = list(range(J.shape[0]))
    while len(active) > 1:
        k = min(active, key=lambda j: norms[j])
        rest = [j for j in active if j != k]
        M_rest = M[np.ix_(rest, rest)]
        a = min_norm_simplex(M_rest)
        if norms[k] < drop_eps * np.sqrt(max(float(a @ M_rest @ a), 0.0)):
            active = rest
        else:
            break
    alpha = np.zeros(J.shape[0])
    alpha[active] = min_norm_simplex(M[np.ix_(active, active)])
    w = torch.as_tensor(alpha, dtype=J.dtype)
    return w @ J, w


class MGDA2Jacobian:
    """Estimator of MO-MGDA2-OpenES: per-objective pseudo-gradients combined by MGDA2 (``estimate`` of
    :func:`~.es_loop.run_es`)."""

    aggregator = "mgda2"

    def __init__(self, drop_eps: float = 0.1):
        if drop_eps < 0:
            raise ValueError(f"drop_eps must be >= 0, got {drop_eps}")
        self.drop_eps = float(drop_eps)

    def estimate(self, F: np.ndarray, eps: torch.Tensor, scale: Any, fitness_shaping: str):
        shaped = shape_fitness(F, fitness_shaping)  # [N, K]
        J = shaped.T @ eps * scale  # [K, D]
        direction, w = mgda2_aggregate(J, self.drop_eps)
        return J, direction, w, shaped @ w

    def true_direction(self, J_true: torch.Tensor) -> torch.Tensor:
        return mgda2_aggregate(J_true, self.drop_eps)[0]


def run_mo_mgda2_openes(
    trainer: Any,
    train_dataloaders: Any,
    num_iterations: int,
    pop_size: int,
    sigma: float = 0.01,
    sigma_decay: float = 1.0,
    sigma_min: float = 0.0,
    sigma_schedule: str = "constant",
    drop_eps: float = 0.1,
    noise_sampler: Optional[Any] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """MO-MGDA2-OpenES ask-and-tell on the weights ``theta`` (see module docstring and :func:`~.es_loop.run_es` for
    the shared parameters).

    Parameters
    ----------
    sigma, sigma_decay, sigma_min, sigma_schedule
        Perturbation std and its schedule, as in :func:`~.openes.run_openes`.
    drop_eps : float
        Drop the smallest pseudo-gradient while its norm is below ``drop_eps`` times the norm of the MGDA direction
        of the others (``0``: plain MGDA).
    noise_sampler : object, optional
        Replaces the isotropic noise (Guided ES, :func:`~.mo_mgda2_guidedes.run_mo_mgda2_guidedes`).
    """
    rule = make_sigma_rule(sigma, sigma_decay, sigma_min, sigma_schedule, num_iterations)
    return run_es(trainer, train_dataloaders, "MO-MGDA2-OpenES", MGDA2Jacobian(drop_eps), rule, num_iterations,
                  pop_size, sigma=sigma, noise_sampler=noise_sampler, **kwargs)
