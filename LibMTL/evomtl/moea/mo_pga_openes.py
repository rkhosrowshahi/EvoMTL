"""Multi-objective OpenES with principal-gradient aggregation (MO-SVD-PGA-OpenES, MO-OS-NMF-PGA-OpenES).

As in :mod:`.mo_upgrad_openes`, every objective is shaped separately, which yields one pseudo-gradient per objective,
the rows of a Jacobian ``J`` of shape ``[K, D]``:

    g_k = shaped_k @ eps / (N * sigma) = shaped_k @ z / (N * sigma**2),   shaped_k = shape(F[:, k])

``J`` is aggregated by SVD-PGA or OS-NMF-PGA (:mod:`.pga`, ported from TorchJD): the average of the oriented,
unit-norm principal directions of the pseudo-gradients (SVD-PGA), or of the columns of their orthogonal
semi-nonnegative factorization (OS-NMF-PGA). With ``pga_scale = None`` the direction has unit scale whatever the size
of the pseudo-gradients (the learning rate sets the step length); ``"mean_norm"`` multiplies it by their mean norm.
``theta`` is updated with the optimizer and the direction as its gradient.

References:
- OpenES: https://arxiv.org/abs/1703.03864
- SVD-PGA / OS-NMF-PGA: TorchJD (``torchjd.aggregation.SVDPGA``, ``OSNMFPGA``)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch

from .es_common import shape_fitness
from .es_loop import make_sigma_rule, run_es
from .pga import SCALES, os_nmf_pga_aggregate, svd_pga_aggregate

METHODS = {"svd": ("svd_pga", "MO-SVD-PGA-OpenES"), "os_nmf": ("os_nmf_pga", "MO-OS-NMF-PGA-OpenES")}


def _scale(pga_scale: Optional[str]) -> Optional[str]:
    scale = None if pga_scale in (None, "none") else pga_scale
    if scale not in SCALES:
        raise ValueError(f"Unknown pga_scale {pga_scale!r}; options: none, mean_norm")
    return scale


class PGAJacobian:
    """Estimator of MO-SVD-PGA-OpenES / MO-OS-NMF-PGA-OpenES: per-objective pseudo-gradients combined by SVD-PGA or
    OS-NMF-PGA (``estimate`` of :func:`~.es_loop.run_es`)."""

    def __init__(self, method: str = "svd", rank: Optional[int] = 2, scale: Optional[str] = None,
                 eigenvalue_tol: float = 1e-10, max_iters: int = 30):
        if method not in METHODS:
            raise ValueError(f"Unknown PGA method {method!r}; options: {list(METHODS)}")
        self.method, self.aggregator = method, METHODS[method][0]
        self.rank, self.scale = rank, _scale(scale)
        self.eigenvalue_tol, self.max_iters = float(eigenvalue_tol), int(max_iters)

    def _aggregate(self, J: torch.Tensor):
        if self.method == "svd":
            return svd_pga_aggregate(J, rank=self.rank, eigenvalue_tol=self.eigenvalue_tol, scale=self.scale)
        return os_nmf_pga_aggregate(J, rank=self.rank, max_iters=self.max_iters, eigenvalue_tol=self.eigenvalue_tol,
                                    scale=self.scale)

    def estimate(self, F: np.ndarray, eps: torch.Tensor, scale: Any, fitness_shaping: str):
        shaped = shape_fitness(F, fitness_shaping)  # [N, K]
        J = shaped.T @ eps * scale  # [K, D]
        direction, w = self._aggregate(J)
        return J, direction, w, shaped @ (w / w.abs().sum().clamp_min(1e-30))

    def true_direction(self, J_true: torch.Tensor) -> torch.Tensor:
        return self._aggregate(J_true)[0]


def _run(method: str, trainer: Any, train_dataloaders: Any, num_iterations: int, pop_size: int, sigma: float,
         sigma_decay: float, sigma_min: float, sigma_schedule: str, pga_rank: Optional[int], pga_scale: Optional[str],
         pga_eigenvalue_tol: float, pga_max_iters: int, noise_sampler: Optional[Any], **kwargs: Any) -> Dict[str, Any]:
    rule = make_sigma_rule(sigma, sigma_decay, sigma_min, sigma_schedule, num_iterations)
    estimator = PGAJacobian(method, pga_rank, pga_scale, pga_eigenvalue_tol, pga_max_iters)
    return run_es(trainer, train_dataloaders, METHODS[method][1], estimator, rule, num_iterations, pop_size,
                  sigma=sigma, noise_sampler=noise_sampler, **kwargs)


def run_mo_svd_pga_openes(
    trainer: Any,
    train_dataloaders: Any,
    num_iterations: int,
    pop_size: int,
    sigma: float = 0.01,
    sigma_decay: float = 1.0,
    sigma_min: float = 0.0,
    sigma_schedule: str = "constant",
    pga_rank: Optional[int] = 2,
    pga_scale: Optional[str] = None,
    pga_eigenvalue_tol: float = 1e-10,
    noise_sampler: Optional[Any] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """MO-SVD-PGA-OpenES ask-and-tell on the weights ``theta`` (see module docstring and :func:`~.es_loop.run_es`
    for the shared parameters).

    Parameters
    ----------
    sigma, sigma_decay, sigma_min, sigma_schedule
        Perturbation std and its schedule, as in :func:`~.openes.run_openes`.
    pga_rank : int or None
        Maximum number of principal directions averaged (``None``: all with a non-negligible eigenvalue).
    pga_scale : str or None
        ``None`` / ``"none"`` (unit scale) or ``"mean_norm"`` (times the mean norm of the pseudo-gradients).
    pga_eigenvalue_tol : float
        Eigenvalues of the Gramian below this fraction of the largest are treated as zero.
    noise_sampler : object, optional
        Replaces the isotropic noise (Guided ES, :func:`~.mo_pga_guidedes.run_mo_svd_pga_guidedes`).
    """
    return _run("svd", trainer, train_dataloaders, num_iterations, pop_size, sigma, sigma_decay, sigma_min,
                sigma_schedule, pga_rank, pga_scale, pga_eigenvalue_tol, 30, noise_sampler, **kwargs)


def run_mo_os_nmf_pga_openes(
    trainer: Any,
    train_dataloaders: Any,
    num_iterations: int,
    pop_size: int,
    sigma: float = 0.01,
    sigma_decay: float = 1.0,
    sigma_min: float = 0.0,
    sigma_schedule: str = "constant",
    pga_rank: Optional[int] = 2,
    pga_scale: Optional[str] = None,
    pga_eigenvalue_tol: float = 1e-10,
    pga_max_iters: int = 30,
    noise_sampler: Optional[Any] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """MO-OS-NMF-PGA-OpenES ask-and-tell on the weights ``theta``: as :func:`run_mo_svd_pga_openes`, with the
    orthogonal semi-NMF basis (at most ``pga_max_iters`` alternating updates)."""
    return _run("os_nmf", trainer, train_dataloaders, num_iterations, pop_size, sigma, sigma_decay, sigma_min,
                sigma_schedule, pga_rank, pga_scale, pga_eigenvalue_tol, pga_max_iters, noise_sampler, **kwargs)
