"""Separable NES on the sum of the task losses (SNES, single objective).

OpenES (:mod:`.openes`) samples isotropically with one ``sigma``. SNES (Schaul et al., 2011) keeps one step size
``sigma_i`` per parameter: it samples ``theta + z`` with ``z = sigma * eps``, ``eps ~ N(0, I)``, the gradient on
``theta`` divides by ``sigma_i`` per coordinate, and after each update

    g_sigma = u @ (eps**2 - 1) / N,     sigma *= exp(-lr_sigma / 2 * g_sigma)

where ``u`` is the shaped fitness of the summed loss. ``g_sigma_i > 0`` means large ``|eps_i|`` came with high loss,
so ``sigma_i`` shrinks. Then ``sigma = max(sigma * sigma_decay, sigma_min)``.

References:
- SNES: https://doi.org/10.1145/2001576.2001692
"""

from __future__ import annotations

import math
from typing import Any, Dict

import torch

from .es_loop import run_es
from .openes import SumOfLosses


class SNESSigma:
    """Per-parameter step sizes adapted by the SNES rule (sigma rule of :func:`~.es_loop.run_es`)."""

    def __init__(self, sigma_lr: float = 0.0, sigma_decay: float = 1.0, sigma_min: float = 0.0):
        self.lr, self.decay, self.sigma_min = float(sigma_lr), float(sigma_decay), float(sigma_min)

    def describe(self) -> str:
        return f"snes lr_sigma={self.lr:.4g}, decay={self.decay}, min={self.sigma_min}"

    def init(self, sigma: float, n_var: int) -> torch.Tensor:
        if self.lr <= 0:  # default learning rate of the SNES step sizes
            self.lr = (3.0 + math.log(n_var)) / (5.0 * math.sqrt(n_var))
        return torch.full((n_var,), float(sigma), dtype=torch.float32)

    def next(self, sigma: torch.Tensor, u: torch.Tensor, eps: torch.Tensor, iteration: int) -> torch.Tensor:
        g_sigma = (u @ (eps * eps - 1.0)) / eps.shape[0]
        sigma = sigma * torch.exp(-0.5 * self.lr * g_sigma)
        return (sigma * self.decay).clamp_min(self.sigma_min)


def run_snes(
    trainer: Any,
    train_dataloaders: Any,
    num_iterations: int,
    pop_size: int,
    sigma: float = 0.01,
    sigma_lr: float = 0.0,
    sigma_decay: float = 1.0,
    sigma_min: float = 0.0,
    **kwargs: Any,
) -> Dict[str, Any]:
    """SNES ask-and-tell on the weights ``theta`` (see module docstring and :func:`~.es_loop.run_es` for the shared
    parameters). The final step sizes are printed per parameter tensor and saved as ``<save_path>/sigma_final.pt``.

    Parameters
    ----------
    sigma : float
        Initial value of every ``sigma_i``.
    sigma_lr : float
        Learning rate of the step sizes; ``0`` picks the SNES default ``(3 + ln d) / (5 sqrt d)``.
    sigma_decay, sigma_min
        After each adaptation ``sigma = max(sigma * sigma_decay, sigma_min)`` per parameter.
    """
    return run_es(trainer, train_dataloaders, "SNES", SumOfLosses(), SNESSigma(sigma_lr, sigma_decay, sigma_min),
                  num_iterations, pop_size, sigma=sigma, **kwargs)
