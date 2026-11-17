"""RLW-DUST: reverse layer-wise DUST, block-coordinate descent over the layers from the last to the first.

LW-DUST (:mod:`.lw_dust`) in the opposite order: one iteration visits the parameter groups from the last layer the
forward pass calls (the task heads) to the first, and gives group ``l`` its ``T_l`` inner steps before moving to the
layer before it. Each inner step is LW-DUST's: fresh mini-batches, a clean forward, the group's DUST estimate (output
jitter and the exact local linear map for a layer, weight-space ES otherwise) and one optimizer step on that group.

The layers after ``l`` were just trained, so its jitter rewards are measured through the updated downstream network.
The layers before ``l`` have not moved yet in this iteration, so the layers after ``l`` were trained on the input
distribution that ``l`` itself then changes. The inner step counts are given in forward order, as the draw counts are.
Everything else (draws, optimizer, logging, W&B steps per inner step) is LW-DUST's.
"""

from __future__ import annotations

from typing import Any, Dict

from .lw_dust import LWDUST, run_lw_dust


class RLWDUST(LWDUST):
    """:class:`~.lw_dust.LWDUST` with the groups visited from the last to the first."""

    name = "RLW-DUST"
    reverse = True


def run_rlw_dust(trainer: Any, train_dataloaders: Any, **kwargs: Any) -> Dict[str, Any]:
    """RLW-DUST: :func:`~.lw_dust.run_lw_dust` (every argument) from the last layer to the first."""
    return run_lw_dust(trainer, train_dataloaders, method="RLW-DUST", _estimator=RLWDUST, **kwargs)
