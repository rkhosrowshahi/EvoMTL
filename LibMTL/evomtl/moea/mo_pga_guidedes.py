"""Multi-objective Guided ES with principal-gradient aggregation (MO-SVD-PGA-GuidedES, MO-OS-NMF-PGA-GuidedES).

MO-SVD-PGA-OpenES / MO-OS-NMF-PGA-OpenES (:mod:`.mo_pga_openes`) with Guided ES sampling (Maheswaranathan et al.,
2019; evosax ``GuidedES``) in place of isotropic noise: the guide is the last ``guided_k`` unit aggregated directions
(FIFO), or with ``guided_ema > 0`` one moving average of them (see :class:`~.es_common.GuidedSampler`).

References:
- Guided ES: Maheswaranathan et al., "Guided evolutionary strategies", ICML 2019, https://arxiv.org/abs/1806.10230
- MO-*-PGA-OpenES: :mod:`~LibMTL.evomtl.moea.mo_pga_openes`
"""

from __future__ import annotations

from typing import Any, Dict

from .es_common import GuidedSampler
from .mo_pga_openes import run_mo_os_nmf_pga_openes, run_mo_svd_pga_openes


def run_mo_svd_pga_guidedes(trainer: Any, train_dataloaders: Any, guided_k: int = 1, guided_alpha: float = 0.9,
                            guided_sigma: float = 0.05, guided_ema: float = 0.9, **kwargs: Any) -> Dict[str, Any]:
    """MO-SVD-PGA-GuidedES: :func:`~.mo_pga_openes.run_mo_svd_pga_openes` (every other argument) with Guided ES
    sampling; ``guided_*`` as in :func:`~.guidedes.run_guidedes`."""
    sampler = GuidedSampler(guided_k, guided_alpha, guided_sigma, guided_ema)
    return run_mo_svd_pga_openes(trainer, train_dataloaders, noise_sampler=sampler, **kwargs)


def run_mo_os_nmf_pga_guidedes(trainer: Any, train_dataloaders: Any, guided_k: int = 1, guided_alpha: float = 0.9,
                               guided_sigma: float = 0.05, guided_ema: float = 0.9, **kwargs: Any) -> Dict[str, Any]:
    """MO-OS-NMF-PGA-GuidedES: :func:`~.mo_pga_openes.run_mo_os_nmf_pga_openes` (every other argument) with Guided ES
    sampling; ``guided_*`` as in :func:`~.guidedes.run_guidedes`."""
    sampler = GuidedSampler(guided_k, guided_alpha, guided_sigma, guided_ema)
    return run_mo_os_nmf_pga_openes(trainer, train_dataloaders, noise_sampler=sampler, **kwargs)
