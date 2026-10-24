"""Pareto-safe common descent Guided ES (PCD-GuidedES).

PCD-OpenES (:mod:`.pcd_openes`: per-objective pseudo-gradients, MGDA / MGDA-Loss / MGDA-GN / robust common
direction ``d``, optimizer on the search mean) with Guided ES sampling (Maheswaranathan et al., 2019; evosax
``GuidedES``) in place of isotropic noise. The guide is self-generated, with no backprop: the last ``guided_k`` unit
common directions ``-d`` (FIFO), or with ``guided_ema > 0`` one moving average of them. Each member's noise is

    eps = sqrt(alpha) e + s U e_k,     e ~ N(0, I_D),  e_k ~ N(0, I_k),  U = orthonormal basis of the guide

with ``s = guided_sigma / sigma`` if ``guided_sigma > 0`` (``guided_sigma`` is then the perturbation size along the
guide, in weight units), else ``s = sqrt((1 - alpha) D / k)`` (same total noise budget as isotropic OpenES). The first
generation, before any guide exists, samples isotropically. As in MO-GuidedES (:func:`~.mo_openes.run_mo_openes` with
``guided_k > 0``) the pseudo-gradients regress the fitness on the whole noise ``eps``.

References:
- Guided ES: Maheswaranathan et al., "Guided evolutionary strategies", ICML 2019, https://arxiv.org/abs/1806.10230
- PCD-OpenES: :mod:`~LibMTL.evomtl.moea.pcd_openes`
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from .mo_openes import guided_noise, update_guide
from .pcd_openes import run_pcd_openes


class GuidedSampler:
    """Guided ES noise around the past common directions (the ``noise_sampler`` of :func:`run_pcd_openes`)."""

    name = "GuidedES"

    def __init__(self, guided_k: int, guided_alpha: float, guided_sigma: float, guided_ema: float):
        if int(guided_k) < 1:
            raise ValueError(f"PCD-GuidedES needs guided_k >= 1, got {guided_k}.")
        if not 0.0 < float(guided_alpha) <= 1.0:
            raise ValueError(f"guided_alpha must be in (0, 1], got {guided_alpha}.")
        self.k, self.alpha = int(guided_k), float(guided_alpha)
        self.sigma_sub, self.ema = float(guided_sigma), float(guided_ema)
        self.guide: list = []  # unit guiding directions

    def describe(self) -> str:
        return (f"guided k={self.k}, alpha={self.alpha}, sigma_sub={self.sigma_sub or 'auto'}, ema={self.ema}")

    def sample(self, eps: torch.Tensor, gen: torch.Generator, sigma: float) -> torch.Tensor:
        if not self.guide:  # first generation: no direction yet
            return eps
        return guided_noise(eps, self.guide, gen, sigma, self.alpha, self.sigma_sub)

    def update(self, direction: torch.Tensor) -> None:
        self.guide = update_guide(self.guide, direction, self.k, self.ema)


def run_pcd_guidedes(
    trainer: Any,
    train_dataloaders: Any,
    guided_k: int = 1,
    guided_alpha: float = 0.9,
    guided_sigma: float = 0.05,
    guided_ema: float = 0.9,
    **kwargs: Any,
) -> Dict[str, Any]:
    """PCD-GuidedES: :func:`~.pcd_openes.run_pcd_openes` (every other argument) with Guided ES sampling.

    Parameters
    ----------
    guided_k : int
        Number of past common directions spanning the guide (FIFO); with ``guided_ema > 0`` one moving average.
    guided_alpha : float
        Fraction of the noise budget kept isotropic.
    guided_sigma : float
        If > 0, perturbation size along the guide (weight units); else the total budget matches OpenES.
    guided_ema : float
        If > 0, guide with one moving average of past unit common directions with this decay.
    """
    sampler = GuidedSampler(guided_k, guided_alpha, guided_sigma, guided_ema)
    return run_pcd_openes(trainer, train_dataloaders, noise_sampler=sampler, **kwargs)
