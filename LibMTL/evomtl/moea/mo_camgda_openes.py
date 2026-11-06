"""Multi-objective OpenES with curvature-aware MGDA (MO-CAMGDA-OpenES).

As in MO-UPGrad-OpenES (:mod:`.mo_upgrad_openes`), every objective is shaped **separately**, which yields one
pseudo-gradient per objective, the rows of ``J`` (``[K, D]``):

    g_k = shaped_k @ eps / (N * sigma),   shaped_k = shape(F[:, k])

MGDA (Desideri 2012) takes the minimum-norm point of their convex hull. When one objective is minimized its gradient
vanishes, the hull contains the origin and the direction vanishes too, although the other objectives can still
decrease: MGDA stops at weakly Pareto-optimal points. Dropping small gradients (:mod:`.mo_mgda2_openes`) removes that
stop, but a gradient is also small where its loss is flat (a plateau) or small in scale, and there MGDA's weight on it
is what moves the search on.

CAMGDA tells a minimized objective from a flat one by its curvature along its own gradient, ``u_k = g_k / ||g_k||``,
from a second difference with step ``h`` around the search mean,

    kappa_k = (f_k(theta + h u_k) + f_k(theta - h u_k) - 2 f_k(theta)) / h^2,

and drops objective ``k`` when it is curved upward and its minimum along ``u_k`` lies within the probed interval:

    kappa_k > 0   and   r_k = ||g_k|| / kappa_k < drop_radius     (default drop_radius = h).

``r_k`` is the Newton distance to the minimum of the quadratic model; it does not change when a loss is scaled, and on
a plateau ``kappa_k ~ 0`` keeps the objective. The direction is MGDA on the remaining objectives (all of them if every
one would be dropped). A dropped objective rejoins the hull once it is no longer near its minimum. The probes cost
``1 + 2K`` evaluations per generation (each returns all ``K`` losses), on the population's mini-batch. ``h`` defaults to
``10 sigma``: a second difference divides the rounding error of the losses by ``h^2``.

``theta`` is updated with the optimizer (Adam by default) and ``d`` as its gradient.

References:
- OpenES: https://arxiv.org/abs/1703.03864
- MGDA: Desideri, Multiple-gradient descent algorithm (MGDA) for multiobjective optimization,
  C. R. Math. 350 (2012), https://doi.org/10.1016/j.crma.2012.03.014
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch

from .es_common import shape_fitness
from .es_loop import make_sigma_rule, run_es
from .mo_mgda_openes import min_norm_simplex


def curvature_along_gradients(J: torch.Tensor, theta: torch.Tensor, evaluate: Callable, h: float) -> np.ndarray:
    """``kappa_k``: second difference of objective ``k`` along ``g_k / ||g_k||`` (along the mean of the other rows
    when ``g_k = 0``), from ``1 + 2K`` evaluations."""
    K = J.shape[0]
    U = J.double().clone()
    for k in range(K):
        if float(U[k].norm()) == 0.0:
            U[k] = J.double().sum(dim=0) - J[k].double()
    U = U / U.norm(dim=1, keepdim=True).clamp_min(1e-30)
    centre = theta.double().unsqueeze(0)
    F = evaluate(torch.cat([centre, centre + h * U, centre - h * U]).float())  # [1 + 2K, K]
    idx = np.arange(K)
    return (F[1 + idx, idx] + F[1 + K + idx, idx] - 2.0 * F[0, idx]) / h ** 2


def camgda_aggregate(J: torch.Tensor, kappa: np.ndarray, drop_radius: float) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """MGDA on the objectives that are not curved-and-near their minimum; return ``(d, w, dropped)`` with
    ``d = J^T w`` and ``w = 0`` on the dropped objectives (see module docstring)."""
    norms = J.double().norm(dim=1).numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(kappa > 0, norms / kappa, np.inf)
    dropped = r < drop_radius
    if dropped.all():
        dropped[:] = False
    active = np.flatnonzero(~dropped)
    M = (J[active] @ J[active].T).double().numpy()
    alpha = np.zeros(J.shape[0])
    alpha[active] = min_norm_simplex(M)
    w = torch.as_tensor(alpha, dtype=J.dtype)
    return w @ J, w, dropped


class CAMGDAJacobian:
    """Estimator of MO-CAMGDA-OpenES: per-objective pseudo-gradients, curvature probes, MGDA on the objectives that
    are not at their minimum (``estimate`` and ``set_probe`` of :func:`~.es_loop.run_es`)."""

    aggregator = "camgda"

    def __init__(self, curv_h: float, drop_radius: Optional[float] = None):
        if curv_h <= 0:
            raise ValueError(f"curv_h must be > 0, got {curv_h}")
        self.h = float(curv_h)
        self.drop_radius = self.h if drop_radius is None else float(drop_radius)
        self.theta = self.evaluate = None
        self.dropped = None  # objectives dropped at the last estimate (also used by true_direction)
        self.n_dropped = 0  # generations with at least one objective dropped

    def set_probe(self, theta: torch.Tensor, evaluate: Callable) -> None:
        self.theta, self.evaluate = theta, evaluate

    def estimate(self, F: np.ndarray, eps: torch.Tensor, scale: Any, fitness_shaping: str):
        shaped = shape_fitness(F, fitness_shaping)  # [N, K]
        J = shaped.T @ eps * scale  # [K, D]
        kappa = curvature_along_gradients(J, self.theta, self.evaluate, self.h)
        direction, w, self.dropped = camgda_aggregate(J, kappa, self.drop_radius)
        self.n_dropped += int(self.dropped.any())
        return J, direction, w, shaped @ w

    def true_direction(self, J_true: torch.Tensor) -> torch.Tensor:
        active = np.flatnonzero(~self.dropped) if self.dropped is not None else np.arange(J_true.shape[0])
        alpha = np.zeros(J_true.shape[0])
        alpha[active] = min_norm_simplex((J_true[active] @ J_true[active].T).double().numpy())
        return torch.as_tensor(alpha, dtype=J_true.dtype) @ J_true


def run_mo_camgda_openes(
    trainer: Any,
    train_dataloaders: Any,
    num_iterations: int,
    pop_size: int,
    sigma: float = 0.01,
    sigma_decay: float = 1.0,
    sigma_min: float = 0.0,
    sigma_schedule: str = "constant",
    curv_h: Optional[float] = None,
    drop_radius: Optional[float] = None,
    noise_sampler: Optional[Any] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """MO-CAMGDA-OpenES ask-and-tell on the weights ``theta`` (see module docstring and :func:`~.es_loop.run_es` for
    the shared parameters).

    Parameters
    ----------
    sigma, sigma_decay, sigma_min, sigma_schedule
        Perturbation std and its schedule, as in :func:`~.openes.run_openes`.
    curv_h : float, optional
        Step of the curvature probes (default ``10 * sigma``).
    drop_radius : float, optional
        Drop an objective whose Newton distance to its minimum is below this (default ``curv_h``).
    noise_sampler : object, optional
        Replaces the isotropic noise (Guided ES, :func:`~.mo_camgda_guidedes.run_mo_camgda_guidedes`).
    """
    rule = make_sigma_rule(sigma, sigma_decay, sigma_min, sigma_schedule, num_iterations)
    estimator = CAMGDAJacobian(10.0 * float(sigma) if curv_h is None else curv_h, drop_radius)
    result = run_es(trainer, train_dataloaders, "MO-CAMGDA-OpenES", estimator, rule, num_iterations, pop_size,
                    sigma=sigma, noise_sampler=noise_sampler, **kwargs)
    print(f"[EvoMTL] MO-CAMGDA: an objective was dropped in {estimator.n_dropped} of {int(num_iterations)} "
          f"generations", flush=True)
    return result
