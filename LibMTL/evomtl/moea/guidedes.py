"""Guided ES on the sum of the task losses (single objective).

OpenES (:mod:`.openes`: one pseudo-gradient of the summed loss, optimizer on the search mean) with Guided ES sampling
(Maheswaranathan et al., 2019; evosax ``GuidedES``) in place of isotropic noise: a fraction ``alpha`` of the noise
budget stays isotropic and the rest goes along a self-generated guide, the last ``guided_k`` unit ES directions
(FIFO), or with ``guided_ema > 0`` one moving average of them (see :class:`~.es_common.GuidedSampler`). The
pseudo-gradient regresses the fitness on the whole noise ``eps``; no backprop.

References:
- Guided ES: Maheswaranathan et al., "Guided evolutionary strategies", ICML 2019, https://arxiv.org/abs/1806.10230
- OpenES: :mod:`~LibMTL.evomtl.moea.openes`
"""

from __future__ import annotations

from typing import Any, Dict

from .es_common import GuidedSampler
from .openes import run_openes


def run_guidedes(
    trainer: Any,
    train_dataloaders: Any,
    guided_k: int = 1,
    guided_alpha: float = 0.9,
    guided_sigma: float = 0.05,
    guided_ema: float = 0.9,
    **kwargs: Any,
) -> Dict[str, Any]:
    """GuidedES: :func:`~.openes.run_openes` (every other argument) with Guided ES sampling.

    Parameters
    ----------
    guided_k : int
        Number of past ES directions spanning the guide (FIFO); with ``guided_ema > 0`` one moving average.
    guided_alpha : float
        Fraction of the noise budget kept isotropic.
    guided_sigma : float
        If > 0, perturbation size along the guide (weight units); else the total budget matches OpenES.
    guided_ema : float
        If > 0, guide with one moving average of past unit ES directions with this decay (0.9 ~ the last 10
        generations).
    """
    sampler = GuidedSampler(guided_k, guided_alpha, guided_sigma, guided_ema)
    return run_openes(trainer, train_dataloaders, noise_sampler=sampler, **kwargs)
