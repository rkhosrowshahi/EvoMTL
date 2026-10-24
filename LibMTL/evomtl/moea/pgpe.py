"""PGPE on the sum of the task losses (single objective).

OpenES (:mod:`.openes`) samples isotropically with one ``sigma``. PGPE (Sehnke et al., 2010) keeps one step size
``sigma_i`` per parameter: it samples ``theta + z`` with ``z = sigma * eps``, ``eps ~ N(0, I)``, the gradient on
``theta`` divides by ``sigma_i`` per coordinate, and after each update

    g_sigma = u @ (eps**2 - 1) / N,     sigma -= lr_sigma * sigma * g_sigma

(the PGPE gradient ``(z**2 - sigma**2) / sigma`` of the perturbation), where ``u`` is the shaped fitness of the summed
loss; each ``sigma_i`` changes by at most a factor ``1 +- sigma_max_change`` per generation (evotorch's
``stdev_max_change``). Then ``sigma = max(sigma * sigma_decay, sigma_min)``.

References:
- PGPE: https://doi.org/10.1016/j.neunet.2009.12.004
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from .es_loop import run_es
from .openes import SumOfLosses


class PGPESigma:
    """Per-parameter step sizes adapted by the PGPE rule (sigma rule of :func:`~.es_loop.run_es`)."""

    def __init__(self, sigma_lr: float = 0.0, sigma_max_change: float = 0.2, sigma_decay: float = 1.0,
                 sigma_min: float = 0.0):
        self.lr = float(sigma_lr) if float(sigma_lr) > 0 else 0.1  # 0: the default learning rate
        self.max_change, self.decay, self.sigma_min = float(sigma_max_change), float(sigma_decay), float(sigma_min)

    def describe(self) -> str:
        return f"pgpe lr_sigma={self.lr:.4g} max_change={self.max_change}, decay={self.decay}, min={self.sigma_min}"

    def init(self, sigma: float, n_var: int) -> torch.Tensor:
        return torch.full((n_var,), float(sigma), dtype=torch.float32)

    def next(self, sigma: torch.Tensor, u: torch.Tensor, eps: torch.Tensor, iteration: int) -> torch.Tensor:
        g_sigma = (u @ (eps * eps - 1.0)) / eps.shape[0]
        new = sigma - self.lr * sigma * g_sigma
        sigma = torch.minimum(torch.maximum(new, sigma * (1.0 - self.max_change)), sigma * (1.0 + self.max_change))
        return (sigma * self.decay).clamp_min(self.sigma_min)


def run_pgpe(
    trainer: Any,
    train_dataloaders: Any,
    num_iterations: int,
    pop_size: int,
    sigma: float = 0.01,
    sigma_lr: float = 0.0,
    sigma_max_change: float = 0.2,
    sigma_decay: float = 1.0,
    sigma_min: float = 0.0,
    **kwargs: Any,
) -> Dict[str, Any]:
    """PGPE ask-and-tell on the weights ``theta`` (see module docstring and :func:`~.es_loop.run_es` for the shared
    parameters). The final step sizes are printed per parameter tensor and saved as ``<save_path>/sigma_final.pt``.

    Parameters
    ----------
    sigma : float
        Initial value of every ``sigma_i``.
    sigma_lr : float
        Learning rate of the step sizes; ``0`` picks the default ``0.1``.
    sigma_max_change : float
        Bound on the relative change of each ``sigma_i`` per generation.
    sigma_decay, sigma_min
        After each adaptation ``sigma = max(sigma * sigma_decay, sigma_min)`` per parameter.
    """
    rule = PGPESigma(sigma_lr, sigma_max_change, sigma_decay, sigma_min)
    return run_es(trainer, train_dataloaders, "PGPE", SumOfLosses(), rule, num_iterations, pop_size,
                  sigma=sigma, **kwargs)
