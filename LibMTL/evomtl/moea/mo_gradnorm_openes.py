"""Multi-objective OpenES with GradNorm loss balancing (MO-GradNorm-OpenES).

As in MO-UPGrad-OpenES (:mod:`.mo_upgrad_openes`), every objective is shaped **separately**, which yields one
pseudo-gradient per objective, the rows of ``J`` (``[K, D]``):

    g_k = shaped_k @ eps / (N * sigma),   shaped_k = shape(F[:, k])

GradNorm (Chen et al. 2018) learns task weights ``w_k = K softmax(a)_k`` so that the objectives train at a common
rate. With ``L_k`` the loss of objective ``k`` (the population mean, an estimate of the smoothed loss at ``theta``),
``L_k(0)`` that of the first generation, the training rates

    r_k = (L_k / L_k(0)) / mean_j (L_j / L_j(0)),

the weighted gradient norms ``G_k = w_k ||g_k||`` and their mean ``G``, the logits ``a`` take one optimizer step per
generation on the GradNorm loss

    L_grad(a) = sum_k | G_k - G r_k^alpha |        (the target ``G r_k^alpha`` held constant),

which raises the weight of an objective that trains slower than the others (``r_k > 1``). ``alpha`` sets the strength
of the pull towards a common rate. The ES direction is ``d = (w / K) @ J`` with the weights before the step (LibMTL's
``GradNorm`` takes both gradients, then steps the model and the weights; its model step is ``w @ J``, ``K`` times this
one). The logits start at zero (equal weights) and are stepped by an optimizer of the same kind and learning rate as
``theta`` (LibMTL: one optimizer on the model and the weights).

GradNorm takes the norms on the last shared layer; here they are those of the pseudo-gradients over all the weights,
so nothing depends on the architecture. The training rates need positive losses (as LibMTL's, they are ratios of the
raw losses).

References:
- OpenES: https://arxiv.org/abs/1703.03864
- GradNorm: Chen et al., "GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks",
  ICML 2018, http://proceedings.mlr.press/v80/chen18a/chen18a.pdf
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch

from .es_common import make_es_optimizer, shape_fitness
from .es_loop import make_sigma_rule, run_es

class GradNormJacobian:
    """Estimator of MO-GradNorm-OpenES: per-objective pseudo-gradients weighted by GradNorm (``estimate`` of
    :func:`~.es_loop.run_es`); ``optimizer`` steps the ``logits``."""

    aggregator = "gradnorm"

    def __init__(self, logits: torch.Tensor, optimizer: torch.optim.Optimizer, alpha: float = 1.5):
        self.logits, self.optimizer, self.alpha = logits, optimizer, float(alpha)
        self.initial_losses = None  # L_k(0)
        self.w = None  # weights of the last direction (sum to one)

    def estimate(self, F: np.ndarray, eps: torch.Tensor, scale: Any, fitness_shaping: str):
        shaped = shape_fitness(F, fitness_shaping)  # [N, K]
        J = shaped.T @ eps * scale  # [K, D]
        K = J.shape[0]
        loss_scale = K * torch.softmax(self.logits, dim=0)  # GradNorm's w
        self.w = loss_scale.detach() / K

        losses = F.mean(axis=0).astype(np.float64)
        if self.initial_losses is None:
            self.initial_losses = losses
        rate = torch.as_tensor(losses / self.initial_losses, dtype=torch.float32)
        G = loss_scale * J.norm(dim=1)
        target = (G.mean() * (rate / rate.mean()) ** self.alpha).detach()
        self.optimizer.zero_grad()
        (G - target).abs().sum().backward()
        self.optimizer.step()
        return J, self.w @ J, self.w, shaped @ self.w

    def true_direction(self, J_true: torch.Tensor) -> torch.Tensor:
        return self.w @ J_true


def run_mo_gradnorm_openes(
    trainer: Any,
    train_dataloaders: Any,
    num_iterations: int,
    pop_size: int,
    sigma: float = 0.01,
    sigma_decay: float = 1.0,
    sigma_min: float = 0.0,
    sigma_schedule: str = "constant",
    gradnorm_alpha: float = 1.5,
    lr: float = 1e-3,
    optim: str = "adam",
    **kwargs: Any,
) -> Dict[str, Any]:
    """MO-GradNorm-OpenES ask-and-tell on the weights ``theta`` (see module docstring and :func:`~.es_loop.run_es` for
    the shared parameters).

    Parameters
    ----------
    sigma, sigma_decay, sigma_min, sigma_schedule
        Perturbation std and its schedule, as in :func:`~.openes.run_openes`.
    gradnorm_alpha : float
        GradNorm's ``alpha``: strength of the pull towards a common training rate (LibMTL's default, 1.5).
    lr, optim
        Optimizer on ``theta``, as in :func:`~.es_loop.run_es`; the GradNorm logits get one of the same kind and
        learning rate (with the same ``adam_*`` / ``rmsprop_*`` / ``sgd_momentum``).
    """
    rule = make_sigma_rule(sigma, sigma_decay, sigma_min, sigma_schedule, num_iterations)
    logits = torch.zeros(int(trainer.task_num), requires_grad=True)
    shared = ("adam_betas", "adam_eps", "rmsprop_alpha", "rmsprop_eps", "sgd_momentum")  # options of the optimizer
    optimizer = make_es_optimizer([logits], str(optim).strip().lower(), lr, **{k: kwargs[k] for k in shared if k in kwargs})
    return run_es(trainer, train_dataloaders, "MO-GradNorm-OpenES", GradNormJacobian(logits, optimizer, gradnorm_alpha),
                  rule, num_iterations, pop_size, lr=lr, sigma=sigma, optim=optim, **kwargs)
