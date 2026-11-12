"""Multi-objective OpenES with MGDA on the non-negligible pseudo-gradients (MO-MGDA2-OpenES).

As in MO-UPGrad-OpenES (:mod:`.mo_upgrad_openes`), every objective is shaped **separately**, which yields one
pseudo-gradient per objective, the rows of ``J`` (``[K, D]``):

    g_k = shaped_k @ eps / (N * sigma),   shaped_k = shape(F[:, k])

MGDA (Desideri 2012) takes the minimum-norm point of their convex hull, ``d = J^T alpha*`` with ``alpha*`` the
argmin over the simplex of ``||J^T alpha||^2``. When one objective is (nearly) minimized its gradient is (nearly)
zero, the hull contains (nearly) the origin and ``d`` vanishes, although the other objectives can still decrease
without increasing it at first order: MGDA stops at weakly Pareto-optimal points (e.g. on the axes of
``f = (x_1^2, x_2^2)``).

MGDA2 first drops the negligible pseudo-gradients, greedily from the smallest: with the active set ``A`` (all
objectives at first), the smallest ``g_k`` in ``A`` is dropped while

    ||g_k|| < drop_eps * ||d_{A \\ k}||,     d_{A \\ k} = min-norm point of the hull of {g_j : j in A, j != k}

and at least two objectives remain; then ``d`` is the MGDA direction of the active set. Every active objective keeps
``g_j . d >= ||d||^2`` (MGDA's common descent); a dropped one changes by at most ``drop_eps ||d_{A \\ k}|| ||d||``
per unit step at first order, i.e. ``drop_eps`` bounds its sacrifice relative to the progress of the others. At a
Pareto-stationary point where the gradients conflict, ``d_{A \\ k}`` is itself small and nothing is dropped; the test
compares gradients to each other, so it does not depend on the scale of the losses. A dropped objective whose loss
then rises gets a larger gradient and rejoins the hull at the next iteration. ``drop_eps = 0`` is plain MGDA.

``drop_rule = "harmless"`` replaces the fixed threshold by an adaptive test with no scale parameter: the smallest
``g_k`` in ``A`` is dropped while the MGDA direction of the others does not increase it at first order, beyond the
estimation noise,

    g_k . d_{A \\ k} >= - drop_z * SE(g_k . d_{A \\ k})

and at least two objectives remain. A dropped objective then keeps MGDA's guarantee (it does not increase to first
order), and the others get the larger step of their own MGDA direction. ``SE`` is the standard error of the ES
estimate of ``g_k . d`` over the i.i.d. sampling units (antithetic pairs, or single members); with exact gradients it
is zero and the test is ``g_k . d_{A \\ k} >= 0``. Where the gradients conflict (near a Pareto-stationary point of
conflicting objectives) nothing is dropped and the step is MGDA's; where one gradient vanishes or is orthogonal to the
others (``f = (x_1^2, x_2^2)``) it is dropped, so MGDA does not stall.

``grad_ema = beta > 0`` runs all of this on a moving average of the pseudo-gradients over the iterations,
``G_bar <- beta G_bar + (1 - beta) G`` (the first iteration uses ``G``), as the gradient tracking of MoCo (Fernando et
al., ICLR 2023): the weights and the step both come from ``G_bar``, whose noise is smaller (variance factor
``(1 - beta) / (1 + beta)`` for independent estimates; the harmless test scales its standard error accordingly), at
the price of a lag of about ``1 / (1 - beta)`` iterations.

``theta`` is updated with the optimizer (Adam by default) and ``d`` as its gradient.

References:
- OpenES: https://arxiv.org/abs/1703.03864
- MGDA: Desideri, Multiple-gradient descent algorithm (MGDA) for multiobjective optimization,
  C. R. Math. 350 (2012), https://doi.org/10.1016/j.crma.2012.03.014
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch

from .es_common import shape_fitness
from .es_loop import make_sigma_rule, run_es
from .mo_mgda_openes import min_norm_simplex

GRAD_NORMS = ("none", "loss2")
DROP_RULES = ("eps", "harmless")


def mgda2_aggregate(J: torch.Tensor, drop_eps: float = 0.1,
                    losses: Optional[np.ndarray] = None, drop_rule: str = "eps", drop_z: float = 0.5,
                    se: Optional[Callable[[int, torch.Tensor], float]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """MGDA after dropping the negligible rows of ``J`` (``[K, D]``); return ``(d, w)`` with ``d = J^T w`` and
    ``w = 0`` on the dropped objectives (see module docstring). ``drop_rule``: ``"eps"`` (the fixed threshold
    ``drop_eps``) or ``"harmless"`` (drop while the others' MGDA direction ``d`` does not increase the objective,
    ``g_k . d >= -drop_z * se(k, d)``; ``se``: standard error of ``J[k] . d``, ``None`` for exact gradients). With
    ``losses`` (MGDA2-LN2) the test and the min-norm problem use ``g_k / |L_k|`` and
    ``w_k = (alpha_k / |L_k|) / sum_j (alpha_j / |L_j|)``."""
    if drop_rule not in DROP_RULES:
        raise ValueError(f"Unknown MGDA2 drop_rule {drop_rule!r}; options: {DROP_RULES}")
    if losses is not None:
        inv = torch.as_tensor(1.0 / np.maximum(np.abs(np.asarray(losses, dtype=np.float64)), 1e-12), dtype=J.dtype)
        se_n = None if se is None else (lambda k, d: float(inv[k]) * se(k, d))  # SE of the normalized rows
        _, alpha = mgda2_aggregate(J * inv.unsqueeze(1), drop_eps, drop_rule=drop_rule, drop_z=drop_z, se=se_n)
        w = alpha * inv / (alpha * inv).sum().clamp_min(1e-30)
        return w @ J, w
    M = (J @ J.T).numpy()
    norms = np.sqrt(np.clip(np.diag(M), 0.0, None))
    active = list(range(J.shape[0]))
    while len(active) > 1:
        k = min(active, key=lambda j: norms[j])
        rest = [j for j in active if j != k]
        M_rest = M[np.ix_(rest, rest)]
        a = min_norm_simplex(M_rest)
        if drop_rule == "eps":
            drop = norms[k] < drop_eps * np.sqrt(max(float(a @ M_rest @ a), 0.0))
        else:  # harmless: the others' MGDA direction does not increase objective k (beyond the noise)
            d_rest = torch.as_tensor(a, dtype=J.dtype) @ J[rest]
            if float(d_rest.norm()) == 0.0:  # the others are stationary: dropping k gains nothing
                break
            tol = 0.0 if se is None else float(drop_z) * float(se(k, d_rest))
            drop = float(J[k] @ d_rest) >= -tol
        if drop:
            active = rest
        else:
            break
    alpha = np.zeros(J.shape[0])
    alpha[active] = min_norm_simplex(M[np.ix_(active, active)])
    w = torch.as_tensor(alpha, dtype=J.dtype)
    return w @ J, w


class MGDA2Jacobian:
    """Estimator of MO-MGDA2-OpenES: per-objective pseudo-gradients combined by MGDA2 (``estimate`` of
    :func:`~.es_loop.run_es`)."""

    aggregator = "mgda2"

    def __init__(self, drop_eps: float = 0.1, grad_norm: str = "none", drop_rule: str = "eps",
                 drop_z: float = 0.5, grad_ema: float = 0.0):
        if drop_eps < 0:
            raise ValueError(f"drop_eps must be >= 0, got {drop_eps}")
        if grad_norm not in GRAD_NORMS:
            raise ValueError(f"Unknown MGDA2 grad_norm {grad_norm!r}; options: {GRAD_NORMS}")
        if drop_rule not in DROP_RULES:
            raise ValueError(f"Unknown MGDA2 drop_rule {drop_rule!r}; options: {DROP_RULES}")
        self.drop_eps, self.grad_norm = float(drop_eps), grad_norm
        if not 0.0 <= grad_ema < 1.0:
            raise ValueError(f"grad_ema must be in [0, 1), got {grad_ema}")
        self.drop_rule, self.drop_z = drop_rule, float(drop_z)
        self.grad_ema, self.J_bar = float(grad_ema), None  # moving average of the pseudo-gradients
        self.losses = None  # population-mean losses of the last estimate (loss2)

    def estimate(self, F: np.ndarray, eps: torch.Tensor, scale: Any, fitness_shaping: str):
        shaped = shape_fitness(F, fitness_shaping)  # [N, K]
        J = shaped.T @ eps * scale  # [K, D]
        self.losses = F.mean(axis=0) if self.grad_norm == "loss2" else None
        se = unit_standard_error(shaped, eps, scale) if self.drop_rule == "harmless" else None
        J_used = J
        if self.grad_ema > 0.0:  # MGDA on the moving average of the pseudo-gradients (gradient tracking)
            b = self.grad_ema
            self.J_bar = J.clone() if self.J_bar is None else b * self.J_bar + (1.0 - b) * J
            J_used = self.J_bar
            if se is not None:
                shrink = float(np.sqrt((1.0 - b) / (1.0 + b)))
                se_raw = se
                se = lambda k, d: shrink * se_raw(k, d)  # noqa: E731
        direction, w = mgda2_aggregate(J_used, self.drop_eps, self.losses, self.drop_rule, self.drop_z, se)
        return J, direction, w, shaped @ w

    def true_direction(self, J_true: torch.Tensor) -> torch.Tensor:
        return mgda2_aggregate(J_true, self.drop_eps, self.losses, self.drop_rule, self.drop_z)[0]


def unit_standard_error(shaped: torch.Tensor, eps: torch.Tensor, scale: float) -> Callable[[int, torch.Tensor], float]:
    """``se(k, d)``: standard error of the ES estimate ``g_k . d`` (``g_k = scale * shaped_k @ eps``) over its i.i.d.
    sampling units: the antithetic pairs when the second half of ``eps`` mirrors the first, else every member."""
    N = eps.shape[0]
    n = N // 2
    antithetic = N % 2 == 0 and n > 0 and torch.equal(eps[n:], -eps[:n])
    if antithetic:  # g_k . d = mean_u (shaped_uk - shaped_(u+n)k) (eps_u . d) * N * scale / 2 over the n pairs
        coef, units, factor = shaped[:n] - shaped[n:], eps[:n], N * scale / 2.0
    else:
        coef, units, factor = shaped, eps, N * scale

    def se(k: int, d: torch.Tensor) -> float:
        v = coef[:, k] * (units @ d.to(units.dtype)) * factor  # per-unit estimates of g_k . d
        return float(v.std(unbiased=True) / np.sqrt(len(v))) if len(v) > 1 else 0.0

    return se


def run_mo_mgda2_openes(
    trainer: Any,
    train_dataloaders: Any,
    num_iterations: int,
    pop_size: int,
    sigma: float = 0.01,
    sigma_decay: float = 1.0,
    sigma_min: float = 0.0,
    sigma_schedule: str = "constant",
    drop_eps: float = 0.1,
    grad_norm: str = "none",
    drop_rule: str = "eps",
    drop_z: float = 0.5,
    grad_ema: float = 0.0,
    noise_sampler: Optional[Any] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """MO-MGDA2-OpenES ask-and-tell on the weights ``theta`` (see module docstring and :func:`~.es_loop.run_es` for
    the shared parameters).

    Parameters
    ----------
    sigma, sigma_decay, sigma_min, sigma_schedule
        Perturbation std and its schedule, as in :func:`~.openes.run_openes`.
    drop_eps : float
        Drop the smallest pseudo-gradient while its norm is below ``drop_eps`` times the norm of the MGDA direction
        of the others (``0``: plain MGDA).
    grad_norm : str
        ``"none"`` (default) or ``"loss2"`` (MGDA2-LN2): the test and the min-norm problem on ``g_k / L_k``
        (``L_k``: population-mean loss, must be positive) and the step with the weights ``alpha_k / L_k``
        normalized to sum to one, as MO-MGDA(LN2).
    drop_rule : str
        ``"eps"`` (default): the fixed threshold ``drop_eps``; ``"harmless"``: drop while the others' MGDA direction
        does not increase the objective beyond ``drop_z`` standard errors of its estimate (adaptive; ``drop_eps``
        unused).
    drop_z : float
        Noise tolerance of ``"harmless"`` in standard errors (default 0.5; at 2, objectives whose conflict is within
        the noise are dropped, which stalls ES near the Pareto set of conflicting objectives).
    grad_ema : float
        ``beta`` in ``[0, 1)``: MGDA (weights and step) on the moving average of the pseudo-gradients (``0``: the
        current estimates).
    noise_sampler : object, optional
        Replaces the isotropic noise (Guided ES, :func:`~.mo_mgda2_guidedes.run_mo_mgda2_guidedes`).
    """
    rule = make_sigma_rule(sigma, sigma_decay, sigma_min, sigma_schedule, num_iterations)
    method = "MO-MGDA2-LN2-OpenES" if grad_norm == "loss2" else "MO-MGDA2-OpenES"
    if drop_rule == "harmless":
        method = method.replace("MGDA2", "MGDA2-Harmless")
    if grad_ema > 0.0:
        method = method.replace("-OpenES", "-EMA-OpenES")
    return run_es(trainer, train_dataloaders, method, MGDA2Jacobian(drop_eps, grad_norm, drop_rule, drop_z, grad_ema),
                  rule, num_iterations, pop_size, sigma=sigma, noise_sampler=noise_sampler, **kwargs)
