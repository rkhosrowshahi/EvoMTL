"""Multi-objective DUST with the sum of the per-task gradient estimates (MO-Sum-DUST).

DUST (:mod:`.dust`) estimates the gradient of one loss, the sum of the task losses: every jitter is rewarded by the
reduction of that sum at its sample. MO-Sum-DUST rewards every task loss separately, so each site gets one output
error and one weight-gradient estimate per task, the rows ``g_k`` of ``J`` (``[K, D]``), and the weights step along
their sum ``d = sum_k g_k``.

With the same noise the two directions agree up to rounding: the chunk centering, the reward-weighted noise and the
local linear maps are all linear in the rewards, so summing the task losses before the estimate (DUST) equals summing
the per-task estimates after it. MO-Sum-DUST keeps K errors per site instead of one, and its per-task rows can be
compared with the per-task backprop gradients.

References:
- DUST: :mod:`~LibMTL.evomtl.moea.dust`
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from .dust import run_dust


def run_mo_sum_dust(trainer: Any, train_dataloaders: Any, **kwargs: Any) -> Dict[str, Any]:
    """MO-Sum-DUST: :func:`~.dust.run_dust` (every argument) with one estimate per task loss, summed."""
    return run_dust(trainer, train_dataloaders, method="MO-Sum-DUST",
                    aggregate=lambda J, losses: (J.sum(0), torch.ones(len(J))), **kwargs)
