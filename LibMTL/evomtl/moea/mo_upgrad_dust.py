"""Multi-objective DUST with UPGrad aggregation of the per-task gradient estimates (MO-UPGrad-DUST).

DUST (:mod:`.dust`) with every task loss rewarded separately: each site gets one output error and one weight-gradient
estimate per task, the rows ``g_k`` of ``J`` (``[K, D]``). They are combined with UPGrad (Quinton & Rey 2024), as
MO-UPGrad-OpenES combines its pseudo-gradients (:func:`~.es_common.upgrad_aggregate`, LibMTL's UPGrad): every row is
projected onto the dual cone of the rows and the projections are averaged, so the direction ``d`` never conflicts
with a task, ``g_k . d >= 0`` for every ``k``.

References:
- DUST: :mod:`~LibMTL.evomtl.moea.dust`
- UPGrad: https://arxiv.org/abs/2406.16232
"""

from __future__ import annotations

from typing import Any, Dict

from .dust import run_dust
from .es_common import upgrad_aggregate


def run_mo_upgrad_dust(trainer: Any, train_dataloaders: Any, upgrad_norm_eps: float = 1e-4,
                       upgrad_reg_eps: float = 1e-4, **kwargs: Any) -> Dict[str, Any]:
    """MO-UPGrad-DUST: :func:`~.dust.run_dust` (every other argument) with the per-task estimates combined by UPGrad;
    ``upgrad_norm_eps`` / ``upgrad_reg_eps`` are the epsilon of its Gramian normalization and its diagonal
    regularization."""
    return run_dust(trainer, train_dataloaders, method="MO-UPGrad-DUST",
                    aggregate=lambda J, losses: upgrad_aggregate(J, upgrad_norm_eps, upgrad_reg_eps), **kwargs)
