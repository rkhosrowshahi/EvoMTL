"""Multi-objective DUST with the MGDA common descent direction of the per-task gradient estimates (MO-MGDA-DUST).

DUST (:mod:`.dust`) with every task loss rewarded separately: each site gets one output error and one weight-gradient
estimate per task, the rows ``g_k`` of ``J`` (``[K, D]``). They are combined as LibMTL's MGDA weighting combines its
backprop gradients (Sener & Koltun 2018, :class:`~LibMTL.weighting.MGDA.MGDA`):

    g~_k = g_k / n_k,     alpha* = argmin_{alpha in simplex} || sum_k alpha_k g~_k ||^2,     d = sum_k alpha*_k g~_k

with the normalizer ``n_k`` of ``grad_norm`` (LibMTL's ``mgda_gn``): ``none`` (``n_k = 1``, MO-MGDA-DUST), ``loss``
(the task loss on the batch, MO-MGDA-Loss-DUST), ``l2`` (the norm of ``g_k``, MO-MGDA-GN-DUST) or ``loss+`` (both,
MO-MGDA-LossGN-DUST). Without normalization ``d`` is the min-norm point of the convex hull of the estimates, a common
descent direction of the estimated objectives (``g_k . d >= ||d||^2``).

References:
- DUST: :mod:`~LibMTL.evomtl.moea.dust`
- MGDA for MTL: https://arxiv.org/abs/1810.04650
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from ...weighting.MGDA import MGDA
from .dust import run_dust

GRAD_NORMS = {"none": "", "loss": "-Loss", "l2": "-GN", "loss+": "-LossGN"}  # mgda_gn -> part of the name
_LIBMTL_MGDA = MGDA()
_LIBMTL_MGDA.device = torch.device("cpu")  # the estimates live on the CPU


def mgda_aggregate(J: torch.Tensor, losses: torch.Tensor, grad_norm: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """LibMTL MGDA on the gradient rows ``J`` (``[K, D]``) with task losses ``losses`` (``[K]``): return ``(d, alpha)``
    with ``d = alpha @ (J / n)`` (see the module docstring)."""
    G = _LIBMTL_MGDA._gradient_normalizers(J, losses, grad_norm)
    alpha = _LIBMTL_MGDA._find_min_norm_element(G)
    return alpha @ G, alpha


def run_mo_mgda_dust(trainer: Any, train_dataloaders: Any, grad_norm: str = "none", **kwargs: Any) -> Dict[str, Any]:
    """MO-MGDA-DUST: :func:`~.dust.run_dust` (every other argument) with the per-task estimates combined by MGDA;
    ``grad_norm`` (``none``, ``loss``, ``l2``, ``loss+``) normalizes them first (LibMTL ``mgda_gn``)."""
    if grad_norm not in GRAD_NORMS:
        raise ValueError(f"MO-MGDA-DUST grad_norm must be one of {tuple(GRAD_NORMS)}, got {grad_norm!r}")
    return run_dust(trainer, train_dataloaders, method=f"MO-MGDA{GRAD_NORMS[grad_norm]}-DUST",
                    aggregate=lambda J, losses: mgda_aggregate(J, losses, grad_norm), **kwargs)
