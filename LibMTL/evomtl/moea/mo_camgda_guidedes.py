"""Multi-objective Guided ES with curvature-aware MGDA (MO-CAMGDA-GuidedES).

MO-CAMGDA-OpenES (:mod:`.mo_camgda_openes`: per-objective pseudo-gradients, the objectives curved upward and near
their minimum dropped, MGDA on the rest, optimizer on the search mean) with Guided ES sampling (Maheswaranathan et
al., 2019; evosax ``GuidedES``) in place of isotropic noise: the guide is the last ``guided_k`` unit CAMGDA directions
(FIFO), or with ``guided_ema > 0`` one moving average of them (see :class:`~.es_common.GuidedSampler`). The
pseudo-gradients regress the fitness on the whole noise ``eps``.

References:
- Guided ES: Maheswaranathan et al., "Guided evolutionary strategies", ICML 2019, https://arxiv.org/abs/1806.10230
- MO-CAMGDA-OpenES: :mod:`~LibMTL.evomtl.moea.mo_camgda_openes`
"""

from __future__ import annotations

from typing import Any, Dict

from .es_common import GuidedSampler
from .mo_camgda_openes import run_mo_camgda_openes


def run_mo_camgda_guidedes(
    trainer: Any,
    train_dataloaders: Any,
    guided_k: int = 1,
    guided_alpha: float = 0.9,
    guided_sigma: float = 0.05,
    guided_ema: float = 0.9,
    **kwargs: Any,
) -> Dict[str, Any]:
    """MO-CAMGDA-GuidedES: :func:`~.mo_camgda_openes.run_mo_camgda_openes` (every other argument) with Guided ES
    sampling; ``guided_*`` as in :func:`~.guidedes.run_guidedes`."""
    sampler = GuidedSampler(guided_k, guided_alpha, guided_sigma, guided_ema)
    return run_mo_camgda_openes(trainer, train_dataloaders, noise_sampler=sampler, **kwargs)
