"""Antithetic DUST (A-DUST): DUST with antithetic jitter pairs on the layer outputs.

DUST (:mod:`.dust`) jitters one layer's output ``y`` with independent draws ``y + sigma a_n`` and rewards each draw
with the loss reduction at its sample, centered over the ``chunk`` draws of one forward pass. A-DUST draws ``K`` noise
tensors ``a_k ~ N(0, I)`` per layer and evaluates each twice, ``y + sigma a_k`` and ``y - sigma a_k``: same sample,
same targets, same weights, eval mode in both, only the sign of the jitter differs. With the per-sample losses
``l+[k, b]`` and ``l-[k, b]`` of the pair, the reward and the error at the layer's output are

    r[k, b] = (l-[k, b] - l+[k, b]) / 2
    e[b]    = -1 / (K sigma) sum_k r[k, b] a_k[b]  =  1 / (2 K sigma) sum_k (l+[k, b] - l-[k, b]) a_k[b]  ~ dL[b] / dy[b]

averaged over the ``K`` pairs, not the ``2K`` evaluations. The rewards are per sample, as in DUST; nothing is
normalized, ranked, clipped or centered. The pair difference cancels the clean loss and every even-order term of the
loss around ``y`` (the curvature term ``sigma^2 a^T H a / 2`` among them), so the estimate is unbiased up to
``O(sigma^2)`` without a baseline; DUST's chunk mean removes the clean loss but shrinks the estimate by
``1 - 1 / chunk``. Everything after the error is DUST's: the exact local linear map of ``e`` and the cached input
(``sum_b e[b] x[b]^T`` and ``sum_b e[b]`` for a Linear layer, the correlation for a convolution, a scatter-add for an
embedding; divided by the batch size), the two-sided weight-space ES on the parameters outside the weight layers
(normalization affines, free scalars; already antithetic) and the optimizer step on ``theta``. A-DUST is
single-objective: the rewards come from the summed task loss.

Cost. The draw counts (``pop_size``, ``allocation``, ``draws``) still count forward evaluations of the batch, so a
layer with ``d`` draws gets ``d / 2`` pairs and A-DUST costs what DUST costs at the same settings. One forward pass of
the batch repeated ``chunk`` times holds ``chunk / 2`` pairs (``chunk`` even): the first half of the copies gets
``+sigma a_k``, the second half ``-sigma a_k``. The samples of an eval-mode forward do not interact, so the two
members of a pair see the same conditions.

Credit assignment. The reward of a jitter is the loss change of its own sample, the per-sample analogue of DUST's
per-token credit. The future-token credit of the attention internals, the cached-logit head draws and the per-layer
tuned draw counts of the transformer recipe are not part of this DUST (see :mod:`.dust`), so A-DUST has none to carry
over.

References:
- DUST: :mod:`~LibMTL.evomtl.moea.dust`; https://qlabs.sh/research/dust, https://github.com/qlabs-eng/dust
"""

from __future__ import annotations

from typing import Any, Dict

from .dust import run_dust


def run_a_dust(trainer: Any, train_dataloaders: Any, **kwargs: Any) -> Dict[str, Any]:
    """A-DUST: :func:`~.dust.run_dust` (every argument) with antithetic jitter pairs; ``chunk`` must be even."""
    return run_dust(trainer, train_dataloaders, antithetic=True, method="A-DUST", **kwargs)
