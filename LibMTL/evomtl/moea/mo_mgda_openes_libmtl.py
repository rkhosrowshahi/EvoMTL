"""Multi-objective OpenES with LibMTL's MGDA (MO-MGDA-OpenES-LibMTL).

MO-MGDA-OpenES (:mod:`.mo_mgda_openes`) in which the min-norm solver and the gradient normalizers are LibMTL's
:class:`~LibMTL.weighting.MGDA.MGDA` (Sener & Koltun 2018, from the official MultiObjectiveOptimization code) instead
of MO-MGDA-OpenES's own:

* ``alpha*`` comes from ``MGDA._find_min_norm_element``: the exact two-vector line search over all pairs, then (for
  ``K >= 3``) Frank-Wolfe steps (at most 250, stop when the change is below 1e-5). For two objectives the pair
  solution is returned with ``gamma`` clamped to ``[0.001, 0.999]`` when one gradient dominates the other, where
  MO-MGDA-OpenES returns the exact 0 or 1.
* ``grad_norm`` (``none``, ``loss``, ``l2``, ``loss+``) is ``MGDA._gradient_normalizers`` (LibMTL's ``mgda_gn``).

The solver takes gradient rows; every Gramian ``M`` (raw, debiased, moving average, true) is passed as a ``K x K``
factor ``F`` with ``F F^T = M``, so the solver sees the same inner products. Everything else (sampling, the robust
ablation, diagnostics, outputs) is that of :mod:`.mo_mgda_openes`.

References:
- MGDA for MTL: https://arxiv.org/abs/1810.04650
- LibMTL MGDA: :mod:`LibMTL.weighting.MGDA`
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch

from ...weighting.MGDA import MGDA
from .mo_mgda_openes import run_mo_mgda_openes

_LIBMTL_MGDA = MGDA()
_LIBMTL_MGDA.device = torch.device("cpu")  # the pseudo-gradients live on the CPU


def libmtl_gradient_normalizers(F: np.ndarray, G: torch.Tensor, grad_norm: str) -> torch.Tensor:
    """Per-objective divisors ``n_k`` with ``g_k / n_k`` = LibMTL ``MGDA._gradient_normalizers(G, L, grad_norm)``;
    the losses ``L_k`` are the population means (an estimate of the smoothed loss at ``theta``)."""
    loss = torch.as_tensor(F.mean(axis=0), dtype=torch.float32)
    normalized = _LIBMTL_MGDA._gradient_normalizers(G.float(), loss, grad_norm)
    return G.norm(dim=1).clamp_min(1e-30) / normalized.norm(dim=1).clamp_min(1e-30)


def libmtl_min_norm(M: np.ndarray) -> np.ndarray:
    """LibMTL ``MGDA._find_min_norm_element`` for a Gramian ``M`` (``[K, K]``), via a factor ``F F^T = M``."""
    if M.shape[0] == 1:
        return np.ones(1)
    vals, vecs = np.linalg.eigh(0.5 * (M + M.T))
    factor = torch.as_tensor(vecs * np.sqrt(np.clip(vals, 0.0, None)), dtype=torch.float32)
    return _LIBMTL_MGDA._find_min_norm_element(factor).double().numpy()


def run_mo_mgda_openes_libmtl(trainer: Any, train_dataloaders: Any, **kwargs: Any) -> Dict[str, Any]:
    """MO-MGDA-OpenES-LibMTL: :func:`~.mo_mgda_openes.run_mo_mgda_openes` (every argument) with LibMTL's min-norm
    solver and gradient normalizers."""
    return run_mo_mgda_openes(trainer, train_dataloaders, min_norm=libmtl_min_norm,
                              normalizers=libmtl_gradient_normalizers, method_suffix="-LibMTL", **kwargs)
