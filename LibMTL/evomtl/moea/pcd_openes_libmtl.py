"""Pareto-safe common descent OpenES with LibMTL's MGDA (PCD-OpenES-LibMTL).

A copy of :mod:`.pcd_openes` in which the min-norm solver and the gradient normalizers are LibMTL's
:class:`~LibMTL.weighting.MGDA.MGDA` (Sener & Koltun 2018, from the official MultiObjectiveOptimization code) instead of
PCD-OpenES's own:

* ``alpha*`` comes from ``MGDA._find_min_norm_element``: the exact two-vector line search over all pairs, then (for
  ``K >= 3``) Frank-Wolfe steps (at most 250, stop when the change is below 1e-5). For two objectives the pair
  solution is returned with ``gamma`` clamped to ``[0.001, 0.999]`` when one gradient dominates the other, where
  PCD-OpenES returns the exact 0 or 1.
* ``grad_norm`` (``none``, ``loss``, ``l2``, ``loss+``) is ``MGDA._gradient_normalizers`` (LibMTL's ``mgda_gn``).

The solver takes gradient rows; every Gramian ``M`` here (raw, debiased, moving average, true) is passed as a
``K x K`` factor ``F`` with ``F F^T = M``, so the solver sees the same inner products. Everything else (sampling, the
robust ablation, diagnostics, outputs) is unchanged from :mod:`.pcd_openes`.

The search distribution is over the network weights: member ``i`` is ``theta + z_i`` with ``z_i = sigma * eps_i``,
``eps_i ~ N(0, I)``. As in :mod:`.mo_openes`, every objective is shaped separately, which gives one OpenES
pseudo-gradient per objective, the rows of ``G`` (``[K, D]``):

    g_j = shaped_j @ eps / (N * sigma)

The update direction ``d`` should be a first-order decrease of every objective, ``g_j . d < 0`` for all ``j``.

``direction = "mgda"`` (default): minimum-norm multi-gradient descent (MGDA, Desideri 2012)

    alpha* = argmin_{alpha in simplex} || G^T alpha ||^2,     d = -G^T alpha*

If ``G^T alpha* != 0``, ``g_j . d <= -||d||^2 < 0`` for every ``j``: ``d`` is a common descent direction of the
estimated objectives. Near a Pareto-stationary point ``||G^T alpha*|| -> 0`` and no direction improves every
objective; the generation is flagged ``stationary`` (``||d|| <= stationary_tol * max_j ||g_j||``), which is
where a Pareto exploration mode would take over (not part of this module: the step is still taken).

The Gramian ``M = G G^T`` that sets ``alpha*`` is biased by the estimation noise: ``E[g^_i . g^_j] = g_i . g_j +
tr Cov(g^_i, g^_j)``, and at OpenES's signal-to-noise ratio the noise term dominates, so ``alpha*`` weighs the
objectives by their noise rather than by their gradients. ``gramian = "debiased"`` uses the U-statistic over the
i.i.d. sampling units ``u`` (antithetic pairs, or single members), ``g^_j = mean_u h_uj``,
``h_uj = coef_uj eps_u / sigma``::

    M~_ij = (n^2 g^_i . g^_j - sum_u h_ui . h_uj) / (n (n - 1)),     h_ui . h_uj = coef_ui coef_uj |eps_u|^2 / sigma^2

which is unbiased for ``g_i . g_j`` (the ``u != v`` cross terms only), projected onto the PSD cone. ``gramian_ema > 0``
averages the Gramian over generations before solving for ``alpha*``; the step is still ``-G^T alpha*`` with the
current estimates. The stationarity test then uses the same Gramian, ``alpha*^T M alpha* <= stationary_tol^2 max_j
M_jj`` (and every generation with ``max_j M_jj <= 0``, no detectable gradient, is stationary).

``direction = "robust"`` (ablation): a noise-aware directional program

    max_{d, s}  s - lam / 2 ||d||^2
    s.t.        g_j . d + kappa sqrt(d^T Sigma_j d) <= -s   for all j,     ||d|| <= r

``Sigma_j`` estimates the covariance of ``g_j``: ``g_j`` is the mean over i.i.d. sampling units (antithetic pairs,
or single members) of ``h_uj = c_uj eps_u / sigma``, so ``Sigma_j = Cov_u(h_uj) / n_units``. ``s > 0`` means the
direction is predicted to decrease every objective despite the uncertainty penalty (a statistical surrogate, not a
guarantee); ``s <= 0`` flags the generation ``stationary``. ``d`` is restricted to the span of the rows of ``G``,
``d = G^T beta``, where ``d^T Sigma_j d = beta^T (G Sigma_j G^T) beta`` needs only ``K``-dimensional projections of
the samples, so the program has ``K + 1`` variables (solved with SLSQP). ``r = trust_radius * max_j ||g_j||``.
With ``kappa = 0``, ``lam = 1`` and an inactive trust region the solution is exactly the MGDA direction (its dual
is the min-norm problem), so ``kappa`` is the only change in the ablation.

``theta`` is moved by the optimizer (``sgd``, ``adam`` or ``rmsprop``) with ``-d`` as its gradient. Plain SGD keeps
the common-descent property of ``d``; Adam / RMSprop precondition it per coordinate, so the logged
``applied_descent`` checks whether the step actually taken, ``theta_new - theta``, still has ``g_j . step < 0``
for every ``j``.

References:
- OpenES: https://arxiv.org/abs/1703.03864
- MGDA: Desideri, Multiple-gradient descent algorithm (MGDA) for multiobjective optimization,
  C. R. Math. 350 (2012), https://doi.org/10.1016/j.crma.2012.03.014
- MGDA for MTL: https://arxiv.org/abs/1810.04650
"""

from __future__ import annotations

import csv
import os
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from scipy.optimize import minimize

from ...weighting.MGDA import MGDA
from .common import configure_moea_random_seed, should_run_full_test
from .hv import pareto_front_indices
from .mo_openes import _cosine, _mean_pairwise_cosine, make_es_optimizer, shape_fitness, true_jacobian

DIRECTIONS = ("mgda", "robust")
GRAMIANS = ("raw", "debiased")
GRAD_NORMS = ("none", "loss", "l2", "loss+")


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


def unit_contributions(shaped: torch.Tensor, antithetic: bool) -> torch.Tensor:
    """Fitness coefficients ``coef`` (``[n_units, K]``) of the i.i.d. sampling units: ``g_j = mean_u coef_uj eps_u /
    sigma`` with ``eps_u`` the first half of the noise (antithetic pairs) or every member."""
    if antithetic:
        n = shaped.shape[0] // 2
        return (shaped[:n] - shaped[n:]) / 2.0
    return shaped


def debiased_gramian(G: torch.Tensor, coef: torch.Tensor, eps_units: torch.Tensor, sigma: float) -> np.ndarray:
    """Unbiased estimate of the true Gramian ``g_i . g_j`` (U-statistic, see module docstring); may be indefinite."""
    n = coef.shape[0]
    M = (G @ G.T).double().numpy()
    sq = (eps_units.double() ** 2).sum(dim=1) / sigma ** 2  # |eps_u|^2 / sigma^2
    c = coef.double()
    S = (c.T * sq) @ c  # sum_u h_ui . h_uj
    return (n * n * M - S.numpy()) / (n * (n - 1))


def psd_project(M: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(0.5 * (M + M.T))
    return (vecs * np.clip(vals, 0.0, None)) @ vecs.T


def robust_direction(
    M: np.ndarray,
    C: np.ndarray,
    alpha_mgda: np.ndarray,
    kappa: float,
    lam: float,
    trust_radius: float,
) -> Tuple[np.ndarray, float]:
    """Solve the noise-aware program in the span of ``G``: ``d = G^T beta``; return ``(beta, s)``.

    With ``c^2 = max_j ||g_j||^2`` the constraints are divided by ``c^2`` (``M / c^2``, and ``C / c^4`` inside
    the square root) so the program is dimensionless; the returned ``s`` is in the original units.
    ``trust_radius`` is relative to ``c``.
    """
    c2 = max(float(np.max(np.diag(M))), 1e-30)
    Ms, Cs = M / c2, C / c2 ** 2
    K = M.shape[0]

    def unpack(x):
        return x[:K], x[K]

    def objective(x):
        beta, s = unpack(x)
        return -(s - 0.5 * lam * beta @ Ms @ beta)

    def margin(x):  # >= 0: -(g_j . d) - kappa sqrt(d^T Sigma_j d) - s
        beta, s = unpack(x)
        spread = np.sqrt(np.einsum("i,kij,j->k", beta, Cs, beta) + 1e-18)
        return -(Ms @ beta) - kappa * spread - s

    constraints = [{"type": "ineq", "fun": margin}]
    if np.isfinite(trust_radius):
        constraints.append({"type": "ineq", "fun": lambda x: trust_radius ** 2 - unpack(x)[0] @ Ms @ unpack(x)[0]})
    beta0 = -alpha_mgda
    x0 = np.append(beta0, float(np.min(margin(np.append(beta0, 0.0)))))
    res = minimize(objective, x0, method="SLSQP", constraints=constraints, options={"maxiter": 200, "ftol": 1e-12})
    beta, s = unpack(res.x)
    return beta, float(s) * c2


def run_pcd_openes_libmtl(
    trainer: Any,
    train_dataloaders: Any,
    num_iterations: int,
    pop_size: int,
    lr: float = 1e-3,
    sigma: float = 0.01,
    sigma_decay: float = 1.0,
    sigma_min: float = 0.0,
    fitness_shaping: str = "raw",
    direction: str = "mgda",
    gramian: str = "raw",
    gramian_ema: float = 0.0,
    grad_norm: str = "none",
    kappa: float = 1.0,
    lam: float = 1.0,
    trust_radius: float = 1.0,
    stationary_tol: float = 0.05,
    antithetic: bool = True,
    optim: str = "adam",
    adam_betas: Tuple[float, float] = (0.9, 0.999),
    adam_eps: float = 1e-8,
    rmsprop_alpha: float = 0.99,
    rmsprop_eps: float = 1e-8,
    sgd_momentum: float = 0.0,
    noise_sampler: Optional[Any] = None,
    num_batches: int = 1,
    seed: int = 0,
    hv_ref_point: Optional[np.ndarray] = None,
    wandb_step_offset: int = 0,
    test_dataloaders: Optional[Any] = None,
    evo_eval_freq: int = 1,
    grad_cos_freq: int = 1,
    grad_save_freq: int = 0,
    **_unused: Any,
) -> Dict[str, Any]:
    """PCD-OpenES ask-and-tell on the weights ``theta`` (see module docstring).

    The search distribution is ``N(theta, sigma^2 I)`` with ``theta`` starting at ``theta_0``; ``theta`` is the
    returned solution (``pareto_center_X``). ``pareto_F`` / ``pareto_X`` are the non-dominated members of the last
    population.

    Parameters
    ----------
    lr, optim, adam_betas, adam_eps, rmsprop_alpha, rmsprop_eps
        Optimizer on ``theta`` (see :func:`~.mo_openes.make_es_optimizer`).
    sigma, sigma_decay, sigma_min
        Perturbation std; after each update ``sigma = max(sigma * sigma_decay, sigma_min)``.
    fitness_shaping : str
        ``"raw"`` (default; losses minus the population mean, so ``||g_j||`` carries the size of each gradient,
        which the min-norm problem compares) or ``"centered_ranks"``.
    direction : str
        ``"mgda"`` (default) or ``"robust"`` (noise-aware program; uses ``kappa``, ``lam``, ``trust_radius``).
    gramian : str
        MGDA: ``"raw"`` (default; ``G G^T`` of the estimates) or ``"debiased"`` (unbiased for the true Gramian).
    gramian_ema : float
        MGDA: if > 0, ``alpha*`` comes from an exponential moving average of the Gramian with this decay.
    kappa : float
        Conservativeness: weight of the standard deviation ``sqrt(d^T Sigma_j d)`` in each constraint.
    lam : float
        Penalty ``lam / 2 ||d||^2`` (``1`` gives the MGDA scale at ``kappa = 0``).
    trust_radius : float
        ``||d|| <= trust_radius * max_j ||g_j||``; ``inf`` or ``<= 0`` disables it.
    grad_norm : str
        Divide each pseudo-gradient by a normalizer before the direction is computed, as LibMTL's MGDA
        ``mgda_gn``: ``"none"`` (default), ``"loss"`` (its loss), ``"l2"`` (its norm) or ``"loss+"`` (both). The
        step is ``-sum_k alpha_k g_k / n_k``, so objective k's raw gradient gets the weight ``alpha_k / n_k``.
        The backprop diagnostic is normalized the same way (its own norms for ``l2``, the same losses).
    stationary_tol : float
        MGDA: flag the generation Pareto-stationary when ``||d|| <= stationary_tol * max_j ||g_j||``.
    antithetic : bool
        Mirrored sampling ``eps, -eps``; requires even ``pop_size``. The covariance estimate uses the pairs as
        sampling units.
    grad_cos_freq : int
        Every this many iterations, log cosine(pseudo-gradient_k, backprop gradient_k) and cosine(d, MGDA
        direction of the backprop Jacobian). ``0`` disables. Diagnostic only.
    noise_sampler : object, optional
        Replaces the isotropic noise (used by :func:`~.pcd_guidedes.run_pcd_guidedes`): ``name`` replaces
        ``"OpenES"`` in the method name, ``describe()`` is printed, ``sample(eps, gen, sigma)`` returns the noise of
        the first half of the population from the standard-normal ``eps`` and ``update(direction)`` sees each
        applied ascent direction ``-d``. The pseudo-gradients regress on the returned noise.
    grad_save_freq : int
        Every this many iterations, from iteration 0, store the search mean, the true and pseudo-gradients per
        objective, the descent directions (ES and the same rule on the true Jacobian, as ascent directions
        ``G^T alpha``) and both ``alpha`` in ``<save_path>/gradients.pth.tar`` (the format of
        :func:`~.mo_openes.run_mo_openes`, read by ``scripts/pca_gradients.py``). ``0`` disables.
    """
    if trainer.theta_0 is None:
        raise RuntimeError("prepare_evolution() first.")
    direction = str(direction).strip().lower()
    if direction not in DIRECTIONS:
        raise ValueError(f"Unknown PCD-OpenES direction {direction!r}; options: {DIRECTIONS}")
    pop_size = int(pop_size)
    if antithetic and pop_size % 2 != 0:
        raise ValueError(f"Antithetic sampling needs an even pop_size, got {pop_size}.")
    grad_norm = str(grad_norm).strip().lower()
    if grad_norm not in GRAD_NORMS:
        raise ValueError(f"Unknown PCD-OpenES grad_norm {grad_norm!r}; options: {GRAD_NORMS}")
    gramian = str(gramian).strip().lower()
    if gramian not in GRAMIANS:
        raise ValueError(f"Unknown PCD-OpenES gramian {gramian!r}; options: {GRAMIANS}")
    if direction == "robust" and (gramian != "raw" or gramian_ema > 0):
        raise ValueError("gramian / gramian_ema apply to direction=mgda only.")
    fitness_shaping = str(fitness_shaping).strip().lower()
    trust_radius = float(trust_radius) if float(trust_radius) > 0 else float("inf")

    configure_moea_random_seed(seed)
    gen = torch.Generator().manual_seed(int(seed))

    n_var = int(trainer.theta_0.numel())
    n_obj = int(trainer.task_num)
    theta_0 = trainer.theta_0.detach().cpu().float()
    theta = theta_0.clone()
    theta.requires_grad_(True)
    optim = str(optim).strip().lower()
    optimizer = make_es_optimizer([theta], optim, lr, adam_betas, adam_eps, rmsprop_alpha, rmsprop_eps, sgd_momentum)
    sigma = float(sigma)
    method = "PCD-OpenES-" + ("MGDA" if direction == "mgda" else "Robust") + "-LibMTL"
    if gramian == "debiased":
        method += "-Debiased"
    method += {"none": "", "loss": "-Loss", "l2": "-GN", "loss+": "-LossGN"}[grad_norm]  # e.g. PCD-OpenES-MGDA-GN
    if noise_sampler is not None:  # e.g. PCD-OpenES-X -> PCD-GuidedES-X
        method = method.replace("OpenES", noise_sampler.name)
    M_ema = None
    if grad_save_freq > 0 and not getattr(trainer, "save_path", None):
        print("[EvoMTL] grad_save_freq ignored: needs a save_path.", flush=True)
        grad_save_freq = 0
    grad_snapshots: Dict[str, list] = {k: [] for k in ("generation", "mean", "true", "pseudo", "direction",
                                                      "true_direction", "alpha", "alpha_true")}

    print(
        f"[EvoMTL] {method}: dims={n_var} | pop={pop_size} | optim={optim} lr={lr}"
        + (f" momentum={sgd_momentum}" if optim == "sgd" else "") + f" | sigma={sigma} "
        f"(decay={sigma_decay}, min={sigma_min}) | antithetic={antithetic} | fitness={fitness_shaping} "
        f"| stationary_tol={stationary_tol} | gramian={gramian} ema={gramian_ema} | grad_norm={grad_norm}"
        + (f" | kappa={kappa} lam={lam} trust_radius={trust_radius}" if direction == "robust" else "")
        + (f" | {noise_sampler.describe()}" if noise_sampler is not None else ""),
        flush=True,
    )

    num_iterations = int(num_iterations)
    population_X = np.empty((0, n_var))
    population_F = np.empty((0, n_obj))
    n_stationary = n_applied_descent = 0

    tasks = list(trainer.task_name)
    csv_cols = (
        ["generation", "time_s", "sigma"]
        + [f"pop_loss_mean_{t}" for t in tasks]
        + [f"train_loss_{t}" for t in tasks] + [f"train_acc_{t}" for t in tasks]
        + [f"test_loss_{t}" for t in tasks] + [f"test_acc_{t}" for t in tasks]
        + [f"weight_{t}" for t in tasks] + ["pseudo_grad_cos", "direction_norm_rel", "margin", "stationary",
                                            "applied_descent"]
        + [f"predicted_change_{t}" for t in tasks]
        + [f"true_cos_{t}" for t in tasks] + ["true_cos_direction"] + [f"alpha_true_{t}" for t in tasks]
    )
    csv_file = csv_writer = None
    if getattr(trainer, "save_path", None):
        os.makedirs(trainer.save_path, exist_ok=True)
        csv_file = open(os.path.join(trainer.save_path, "metrics.csv"), "w", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=csv_cols)
        csv_writer.writeheader()

    for iteration in range(num_iterations):
        t0 = time.perf_counter()
        step = wandb_step_offset + iteration
        test_metrics = None

        # ask
        n_draw = pop_size // 2 if antithetic else pop_size
        eps = torch.randn(n_draw, n_var, generator=gen, dtype=torch.float32)
        if noise_sampler is not None:
            eps = noise_sampler.sample(eps, gen, sigma)
        if antithetic:
            eps = torch.cat([eps, -eps])
        population = theta.detach() + sigma * eps

        shared_batches = trainer._sample_batches(train_dataloaders, num_batches)
        F = np.zeros((pop_size, n_obj), dtype=np.float32)
        for i in range(pop_size):
            trainer.set_theta(population[i])
            F[i] = trainer.evaluate_multiobjective_losses(train_dataloaders, prefetched_batches=shared_batches)

        # tell: per-objective pseudo-gradients G [K, D]
        shaped = shape_fitness(F, fitness_shaping)  # [N, K]
        G = shaped.T @ eps / (pop_size * sigma)
        gn = libmtl_gradient_normalizers(F, G, grad_norm)  # [K]; scaling shaped keeps G, coef and the covariances consistent
        shaped, G = shaped / gn, G / gn.unsqueeze(1)
        M = (G @ G.T).double().numpy()
        g_max = float(np.sqrt(np.max(np.diag(M))))
        # sampling units u: mean_u h_uj = g_j, h_uj = coef_uj eps_u / sigma
        coef = unit_contributions(shaped, antithetic)  # [n_units, K]
        if direction == "mgda":
            M_sel = debiased_gramian(G, coef, eps[:n_draw], sigma) if gramian == "debiased" else M
            if gramian_ema > 0:
                M_ema = M_sel if M_ema is None else gramian_ema * M_ema + (1.0 - gramian_ema) * M_sel
                M_sel = M_ema
            M_sel = psd_project(M_sel) if gramian == "debiased" else M_sel
            alpha = libmtl_min_norm(M_sel)
            beta = -alpha
            margin = float(alpha @ M_sel @ alpha)  # s = ||G^T alpha*||^2 = min_j -(g_j . d), on M_sel
            sel_max = float(np.max(np.diag(M_sel)))
            stationary = sel_max <= 0.0 or margin <= float(stationary_tol) ** 2 * sel_max
        else:
            alpha = libmtl_min_norm(M)
            proj = eps[:n_draw] @ G.T / sigma  # [n_units, K]: G eps_u / sigma
            P = coef.T.unsqueeze(-1) * proj.unsqueeze(0)  # [K(obj j), n_units, K]: G h_uj
            P = P - P.mean(dim=1, keepdim=True)
            C = (P.transpose(1, 2) @ P / (P.shape[1] - 1) / P.shape[1]).double().numpy()  # G Sigma_j G^T
            beta, margin = robust_direction(M, C, alpha, float(kappa), float(lam), trust_radius)
            stationary = margin <= 0.0
        d = torch.as_tensor(beta, dtype=torch.float32) @ G  # common descent direction (on the losses)
        d_norm = float(d.norm())
        n_stationary += int(stationary)

        # diagnostic: backprop Jacobian at the pre-update theta, same batch
        true_cos = None
        do_save = grad_save_freq > 0 and iteration % int(grad_save_freq) == 0
        if do_save or (grad_cos_freq > 0 and iteration % int(grad_cos_freq) == 0):
            J_true = true_jacobian(trainer, theta.detach(), shared_batches)
            J_true = J_true / libmtl_gradient_normalizers(F, J_true, grad_norm).unsqueeze(1)
            alpha_true = libmtl_min_norm((J_true @ J_true.T).double().numpy())
            d_true = -torch.as_tensor(alpha_true, dtype=torch.float32) @ J_true
            true_cos = [_cosine(G[k], J_true[k]) for k in range(n_obj)]
            dir_cos = _cosine(d, d_true)
        if do_save:
            for key, val in (("generation", iteration), ("mean", theta.detach()), ("true", J_true), ("pseudo", G),
                             ("direction", -d), ("true_direction", -d_true),
                             ("alpha", torch.as_tensor(alpha)), ("alpha_true", torch.as_tensor(alpha_true))):
                grad_snapshots[key].append(val if key == "generation" else val.detach().float().clone())
            torch.save({
                "generation": torch.tensor(grad_snapshots["generation"]),
                **{k: torch.stack(v) for k, v in grad_snapshots.items() if k != "generation"},
                "tasks": tasks, "aggregator": direction if direction == "robust" else "mgda", "method": method,
                "grad_norm": grad_norm,  # pseudo / true are the normalized gradients g_k / n_k
                "pop_size": pop_size, "lr": float(lr), "sigma": sigma, "dims": n_var,
            }, os.path.join(trainer.save_path, "gradients.pth.tar"))

        old = theta.detach().clone()
        theta.grad = -d
        optimizer.step()
        if noise_sampler is not None:
            noise_sampler.update(-d)
        sigma = max(sigma * float(sigma_decay), float(sigma_min))
        predicted = (G @ (theta.detach() - old)).numpy()  # first-order change of each loss for the applied step
        applied_descent = bool(np.all(predicted < 0))
        n_applied_descent += int(applied_descent)

        population_X = population.numpy()
        population_F = F
        pareto_front = F[pareto_front_indices(F)]
        elapsed = time.perf_counter() - t0

        w = torch.as_tensor(-beta, dtype=torch.float32)  # weights of G^T in -d
        grad_cos = _mean_pairwise_cosine(G) if n_obj > 1 else float("nan")
        extras_wandb: Dict[str, Any] = {
            "evo/pcd/sigma": sigma,
            "evo/pcd/direction_norm": d_norm,
            "evo/pcd/direction_norm_rel": d_norm / max(g_max, 1e-30),
            "evo/pcd/margin": margin,
            "evo/pcd/stationary": int(stationary),
            "evo/pcd/stationary_frac": n_stationary / (iteration + 1),
            "evo/pcd/applied_descent_frac": n_applied_descent / (iteration + 1),
            "evo/pcd/mean_norm": float((theta.detach() - theta_0).norm()),
            "evo/pcd/pop_losses/avg": float(F.mean()),
            "evo/pcd/grad_cos": grad_cos,
        }
        for k, tn in enumerate(tasks):
            extras_wandb[f"evo/pcd/weight/{tn}"] = float(w[k])
            extras_wandb[f"evo/pcd/grad_norm/{tn}"] = float(G[k].norm())
            extras_wandb[f"evo/pcd/predicted_change/{tn}"] = float(predicted[k])
        progress = (f"sigma={sigma:.4g} |d|/|g|max={d_norm / max(g_max, 1e-30):.3f} s={margin:.3g} "
                    f"cos(g)={grad_cos:+.3f} w=[" + ", ".join(f"{float(x):.3f}" for x in w) + "]"
                    + (" STATIONARY" if stationary else "") + ("" if applied_descent else " non-descent-step"))
        if true_cos is not None:
            progress += " true_cos=[" + ", ".join(f"{c:+.4f}" for c in true_cos) + f"] dir_cos={dir_cos:+.4f}"
            for k, tn in enumerate(tasks):
                extras_wandb[f"evo/pcd/true_cos/{tn}"] = true_cos[k]
            extras_wandb["evo/pcd/true_cos/direction"] = dir_cos
            for k, tn in enumerate(tasks):
                extras_wandb[f"evo/pcd/alpha_true/{tn}"] = float(alpha_true[k])

        trainer._print_evo_progress(method, iteration, num_iterations, pareto_front, extras=progress,
                                    elapsed=elapsed, offspring_X=population_X, wandb_log_step=step)
        trainer._wandb_log_evo_front_metrics(iteration, pareto_front, step, extras=extras_wandb)

        trainer.set_theta(theta.detach())
        train_batch_metrics = trainer.metrics_snapshot_from_prefetched_batches(shared_batches)
        run_eval = should_run_full_test(iteration, num_iterations, evo_eval_freq)
        if run_eval:
            trainer._wandb_log_evo_objective_plot(
                population_F=F,
                pareto_F=pareto_front,
                pareto_center_F=np.asarray(train_batch_metrics["loss_item"], dtype=np.float32),
                step=step,
            )
        if test_dataloaders is not None and run_eval:
            test_metrics = trainer.test(test_dataloaders, epoch=None, mode="test", wandb_log_step=step,
                                        suppress_display=True, return_metrics=True)
            trainer.print_compact_train_test_line(train_batch_metrics, test_metrics, epoch_label=iteration + 1)
        else:
            parts = " | ".join(f"{task}: {float(train_batch_metrics['loss_item'][i]):.4f}"
                               for i, task in enumerate(tasks))
            print(f"[EvoMTL] Mean theta TRAIN (curr batch) iteration {iteration + 1:04d} | {parts}", flush=True)

        if csv_writer is not None:
            row: Dict[str, Any] = {"generation": iteration + 1, "time_s": round(elapsed, 4), "sigma": sigma,
                                   "pseudo_grad_cos": grad_cos, "direction_norm_rel": d_norm / max(g_max, 1e-30),
                                   "margin": margin, "stationary": int(stationary),
                                   "applied_descent": int(applied_descent)}
            for k, t in enumerate(tasks):
                row[f"pop_loss_mean_{t}"] = float(F[:, k].mean())
                row[f"train_loss_{t}"] = float(train_batch_metrics["loss_item"][k])
                row[f"train_acc_{t}"] = float(train_batch_metrics["results"][t][0])
                if test_metrics is not None:
                    row[f"test_loss_{t}"] = float(test_metrics["loss_item"][k])
                    row[f"test_acc_{t}"] = float(test_metrics["results"][t][0])
                row[f"weight_{t}"] = float(w[k])
                row[f"predicted_change_{t}"] = float(predicted[k])
                if true_cos is not None:
                    row[f"true_cos_{t}"] = true_cos[k]
            if true_cos is not None:
                row["true_cos_direction"] = dir_cos
                for k, t in enumerate(tasks):
                    row[f"alpha_true_{t}"] = float(alpha_true[k])
            csv_writer.writerow(row)
            csv_file.flush()

    if csv_file is not None:
        csv_file.close()
    print(f"[EvoMTL] {method}: stationary generations {n_stationary}/{num_iterations}, "
          f"applied steps with first-order descent on every objective {n_applied_descent}/{num_iterations}",
          flush=True)

    theta_X = theta.detach().numpy().copy()
    nd_idx = pareto_front_indices(population_F) if len(population_F) else np.arange(0)
    trainer.set_theta(theta_X)
    out = {
        "method": method,
        "pareto_F": population_F[nd_idx],
        "pareto_X": population_X[nd_idx],
        "population_F": population_F,
        "population_X": population_X,
        "hv_weights": None,
        "pareto_center_X": theta_X,
        "hv_ref_point": None if hv_ref_point is None else np.asarray(hv_ref_point, dtype=np.float32),
        "hv_center_aggregation": None,
        "hv_softmax_temperature": None,
        "sigma": sigma,
        "direction": direction,
        "stationary_generations": n_stationary,
        "applied_descent_generations": n_applied_descent,
    }
    trainer.last_evo = out
    return out
