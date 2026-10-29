"""Multi-objective OpenES with the CAGrad conflict-averse direction (MO-CAGrad-OpenES).

As in MO-UPGrad-OpenES (:mod:`.mo_upgrad_openes`), every objective is shaped **separately**, which yields one
pseudo-gradient per objective, the rows of ``J`` (``[K, D]``):

    g_k = shaped_k @ eps / (N * sigma),   shaped_k = shape(F[:, k])

``J`` is aggregated with CAGrad (Liu et al. 2021): the direction ``d`` maximizes the worst first-order decrease over
the objectives within a ball around the average gradient ``g_0 = mean_k g_k``,

    max_d  min_k g_k . d     s.t.  ||d - g_0|| <= c ||g_0||,

solved through its dual on the simplex, with ``g_w = J^T w``,

    w* = argmin_{w in simplex}  g_w . g_0 + c ||g_0|| ||g_w||,     d = g_0 + (c ||g_0|| / ||g_w*||) g_w*.

``c = cagrad_alpha`` interpolates between the mean of the pseudo-gradients (``c = 0``, MO-Mean-OpenES) and a
min-norm, MGDA-like direction (large ``c``). ``cagrad_rescale`` divides ``d`` by ``1 + c^2`` (``1``, default),
``1 + c`` (``2``) or nothing (``0``), as LibMTL's ``CAGrad``. ``theta`` is updated with the optimizer (Adam by default)
and ``d`` as its gradient.

The dual is solved as in LibMTL (SLSQP from the uniform weights) on the Gramian ``J J^T`` divided by its largest
diagonal entry: the minimizer does not depend on that scale, and ES pseudo-gradients can be small enough for the
solver's absolute tolerance to stop it at the start.

References:
- OpenES: https://arxiv.org/abs/1703.03864
- CAGrad: Liu et al., "Conflict-Averse Gradient Descent for Multi-task Learning", NeurIPS 2021,
  https://arxiv.org/abs/2110.14048 (official code: https://github.com/Cranial-XIX/CAGrad)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from scipy.optimize import minimize

from .es_common import shape_fitness
from .es_loop import make_sigma_rule, run_es

RESCALES = (0, 1, 2)


def cagrad_aggregate(J: torch.Tensor, calpha: float = 0.5, rescale: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """CAGrad (Liu et al. 2021) on Jacobian rows ``J`` (``[K, D]``); return ``(d, w)`` with ``d = J^T w`` (see
    module docstring)."""
    K = J.shape[0]
    GG = (J @ J.T).double().numpy()
    A = GG / max(float(GG.diagonal().max()), 1e-30)  # scale-free dual (see module docstring)
    b = np.full(K, 1.0 / K)
    c = calpha * np.sqrt(max(float(A.mean()), 0.0) + 1e-8)  # c ||g_0||, ||g_0||^2 = mean of the Gramian

    def objective(x):
        return x @ A @ b + c * np.sqrt(max(float(x @ A @ x), 0.0) + 1e-8)

    res = minimize(objective, b, bounds=[(0.0, 1.0)] * K, constraints={"type": "eq", "fun": lambda x: 1.0 - x.sum()})
    ww = torch.as_tensor(res.x, dtype=J.dtype)
    g0_norm = float(np.sqrt(max(float(GG.mean()), 0.0)))
    gw_norm = float((ww @ J).norm())
    lmbda = calpha * g0_norm / gw_norm if gw_norm > 0 else 0.0
    w = torch.full((K,), 1.0 / K, dtype=J.dtype) + lmbda * ww  # d = g_0 + lmbda g_w
    w = w / {0: 1.0, 1: 1.0 + calpha ** 2, 2: 1.0 + calpha}[rescale]
    return w @ J, w


class CAGradJacobian:
    """Estimator of MO-CAGrad-OpenES: per-objective pseudo-gradients combined by CAGrad (``estimate`` of
    :func:`~.es_loop.run_es`)."""

    aggregator = "cagrad"

    def __init__(self, calpha: float = 0.5, rescale: int = 1):
        if int(rescale) not in RESCALES:
            raise ValueError(f"Unknown CAGrad rescale {rescale!r}; options: {RESCALES}")
        self.calpha, self.rescale = float(calpha), int(rescale)

    def estimate(self, F: np.ndarray, eps: torch.Tensor, scale: Any, fitness_shaping: str):
        shaped = shape_fitness(F, fitness_shaping)  # [N, K]
        J = shaped.T @ eps * scale  # [K, D]
        direction, w = cagrad_aggregate(J, self.calpha, self.rescale)
        return J, direction, w, shaped @ (w / w.abs().sum().clamp_min(1e-30))

    def true_direction(self, J_true: torch.Tensor) -> torch.Tensor:
        return cagrad_aggregate(J_true, self.calpha, self.rescale)[0]


def run_mo_cagrad_openes(
    trainer: Any,
    train_dataloaders: Any,
    num_iterations: int,
    pop_size: int,
    sigma: float = 0.01,
    sigma_decay: float = 1.0,
    sigma_min: float = 0.0,
    sigma_schedule: str = "constant",
    cagrad_alpha: float = 0.5,
    cagrad_rescale: int = 1,
    noise_sampler: Optional[Any] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """MO-CAGrad-OpenES ask-and-tell on the weights ``theta`` (see module docstring and :func:`~.es_loop.run_es` for
    the shared parameters).

    Parameters
    ----------
    sigma, sigma_decay, sigma_min, sigma_schedule
        Perturbation std and its schedule, as in :func:`~.openes.run_openes`.
    cagrad_alpha : float
        ``c``: radius of the ball around the average pseudo-gradient, relative to its norm (LibMTL ``calpha``).
    cagrad_rescale : int
        ``0``: no rescaling, ``1``: divide the direction by ``1 + c^2`` (default), ``2``: by ``1 + c``.
    noise_sampler : object, optional
        Replaces the isotropic noise (Guided ES, :func:`~.mo_cagrad_guidedes.run_mo_cagrad_guidedes`).
    """
    rule = make_sigma_rule(sigma, sigma_decay, sigma_min, sigma_schedule, num_iterations)
    return run_es(trainer, train_dataloaders, "MO-CAGrad-OpenES", CAGradJacobian(cagrad_alpha, cagrad_rescale),
                  rule, num_iterations, pop_size, sigma=sigma, noise_sampler=noise_sampler, **kwargs)
