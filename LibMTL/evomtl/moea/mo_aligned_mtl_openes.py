"""Multi-objective OpenES with Aligned-MTL Jacobian aggregation (MO-Aligned-MTL-OpenES).

As in :mod:`.mo_upgrad_openes`, every objective is shaped separately, which yields one pseudo-gradient per objective,
the rows of a Jacobian ``J`` of shape ``[K, D]``:

    g_k = shaped_k @ eps / (N * sigma) = shaped_k @ z / (N * sigma**2),   shaped_k = shape(F[:, k])

``J`` is aggregated with Aligned-MTL (Senushkin et al. 2023, Algorithm 1), which aligns the principal components of
the Jacobian so that the update has a condition number of one. With the Gramian ``M = J J^T = V Lambda V^T`` (its
positive eigenvalues only, rank ``r``), the balance transformation

    B = sqrt(lambda_s) V Lambda^{-1/2} V^T

(``lambda_s``: the smallest positive eigenvalue for ``scale_mode = "min"``, their median for ``"median"``, their mean
for ``"rmse"``) gives the weights ``w = B p`` of the preference vector ``p = 1 / K`` and the direction ``d = J^T w``.
This is the ``AlignedMTL`` aggregator of TorchJD (``pref_vector = 1 / K``); LibMTL's ``Aligned_MTL`` weighting uses
``p = 1`` (``K`` times the step). ``theta`` is updated with the optimizer and ``d`` as its gradient.

References:
- OpenES: https://arxiv.org/abs/1703.03864
- Aligned-MTL: Senushkin et al., "Independent Component Alignment for Multi-Task Learning", CVPR 2023
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from .es_common import shape_fitness
from .es_loop import make_sigma_rule, run_es

SCALE_MODES = ("min", "median", "rmse")


def aligned_mtl_aggregate(J: torch.Tensor, scale_mode: str = "min") -> Tuple[torch.Tensor, torch.Tensor]:
    """Aligned-MTL (Senushkin et al. 2023) on Jacobian rows ``J`` (``[K, D]``) with the preference vector ``1 / K``;
    return ``(d, w)`` with ``d = J^T w`` (see module docstring). A zero Gramian gives the mean (``B = I``)."""
    if scale_mode not in SCALE_MODES:
        raise ValueError(f"Unknown Aligned-MTL scale_mode {scale_mode!r}; options: {SCALE_MODES}")
    K = J.shape[0]
    M = J @ J.T
    lam, V = torch.linalg.eigh(M)
    tol = lam.max() * K * torch.finfo(J.dtype).eps  # rank test at the precision of J (as TorchJD)
    rank = int((lam > tol).sum())
    if rank == 0:
        B = torch.eye(K, dtype=J.dtype)
    else:
        order = torch.argsort(lam, descending=True)
        lam, V = lam[order][:rank], V[:, order][:, :rank]
        scale = {"min": lam[-1], "median": torch.median(lam), "rmse": lam.mean()}[scale_mode]
        B = scale.sqrt() * V @ torch.diag(1.0 / lam.sqrt()) @ V.T
    w = B @ torch.full((K,), 1.0 / K, dtype=J.dtype)
    return w @ J, w


class AlignedMTLJacobian:
    """Estimator of MO-Aligned-MTL-OpenES: per-objective pseudo-gradients combined by Aligned-MTL (``estimate`` of
    :func:`~.es_loop.run_es`)."""

    aggregator = "aligned_mtl"

    def __init__(self, scale_mode: str = "min"):
        if scale_mode not in SCALE_MODES:
            raise ValueError(f"Unknown Aligned-MTL scale_mode {scale_mode!r}; options: {SCALE_MODES}")
        self.scale_mode = scale_mode

    def estimate(self, F: np.ndarray, eps: torch.Tensor, scale: Any, fitness_shaping: str):
        shaped = shape_fitness(F, fitness_shaping)  # [N, K]
        J = shaped.T @ eps * scale  # [K, D]
        direction, w = aligned_mtl_aggregate(J, self.scale_mode)
        return J, direction, w, shaped @ (w / w.abs().sum().clamp_min(1e-30))

    def true_direction(self, J_true: torch.Tensor) -> torch.Tensor:
        return aligned_mtl_aggregate(J_true, self.scale_mode)[0]


def run_mo_aligned_mtl_openes(
    trainer: Any,
    train_dataloaders: Any,
    num_iterations: int,
    pop_size: int,
    sigma: float = 0.01,
    sigma_decay: float = 1.0,
    sigma_min: float = 0.0,
    sigma_schedule: str = "constant",
    aligned_mtl_scale_mode: str = "min",
    noise_sampler: Optional[Any] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """MO-Aligned-MTL-OpenES ask-and-tell on the weights ``theta`` (see module docstring and
    :func:`~.es_loop.run_es` for the shared parameters).

    Parameters
    ----------
    sigma, sigma_decay, sigma_min, sigma_schedule
        Perturbation std and its schedule, as in :func:`~.openes.run_openes`.
    aligned_mtl_scale_mode : str
        Eigenvalue that scales the balance transformation: ``min`` (default), ``median`` or ``rmse`` (the mean).
    noise_sampler : object, optional
        Replaces the isotropic noise (Guided ES, :func:`~.mo_aligned_mtl_guidedes.run_mo_aligned_mtl_guidedes`).
    """
    rule = make_sigma_rule(sigma, sigma_decay, sigma_min, sigma_schedule, num_iterations)
    return run_es(trainer, train_dataloaders, "MO-Aligned-MTL-OpenES", AlignedMTLJacobian(aligned_mtl_scale_mode),
                  rule, num_iterations, pop_size, sigma=sigma, noise_sampler=noise_sampler, **kwargs)
