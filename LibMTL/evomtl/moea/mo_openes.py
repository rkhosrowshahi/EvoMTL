"""Multi-objective OpenAI-ES with Jacobian aggregation (MO-OpenES).

The search distribution is over the network weights: member ``i`` is ``theta + z_i`` around the search mean ``theta``,
with perturbation ``z_i = sigma * eps_i`` and standard-normal noise ``eps_i ~ N(0, I)``.

OpenES (Salimans et al. 2017; evosax ``Open_ES``) ranks a population by one scalar
fitness and estimates a search gradient from the centered ranks. Scalarizing several
losses into one fitness does not guarantee Pareto-stationary progress, so here every
objective is ranked **separately**, which yields one pseudo-gradient per objective,
i.e. a Jacobian ``J`` of shape ``[K, D]``:

    g_k = shaped_k @ eps / (N * sigma) = shaped_k @ z / (N * sigma**2),   shaped_k = centered_rank(F[:, k])

(the same estimator as evosax ``Open_ES._tell`` with ``centered_rank_fitness_shaping_fn``).
``J`` is aggregated into one common direction and ``theta`` is updated with Adam:

* ``upgrad``: project every row ``g_k`` onto the dual cone of the rows of ``J`` and
  average the projections (UPGrad, Quinton & Rey 2024). The result never conflicts
  with any objective: ``g_k . d >= 0`` for all ``k``.
* ``mean``: plain average of the rows (ablation: Jacobian without the dual-cone step).
* ``ws``: standard OpenES on the equal-weight sum of losses (scalarized baseline).

Per-parameter step sizes (``sigma_adapt``): OpenES samples isotropically with one ``sigma``. With
``snes`` or ``pgpe`` every coordinate has its own ``sigma_i`` (samples ``theta + z``,
``z = sigma * eps``, ``eps ~ N(0, I)``), the gradient on ``theta`` divides by ``sigma_i`` per coordinate, and after each update

    g_sigma = u @ (eps**2 - 1) / N,   u = shaped fitness of the aggregated objective

where ``u`` is the centered rank of the summed loss (``ws``) or the per-objective centered ranks
combined with the aggregator's weights ``w / sum(w)`` (``upgrad``, ``mean``). ``g_sigma_i > 0`` means
large ``|eps_i|`` came with high loss, so ``sigma_i`` shrinks:

* ``snes`` (Schaul et al., 2011): ``sigma *= exp(-lr_sigma / 2 * g_sigma)``.
* ``pgpe`` (Sehnke et al., 2010): ``sigma -= lr_sigma * sigma * g_sigma`` (the PGPE gradient
  ``(z**2 - sigma**2) / sigma`` of the perturbation ``z = sigma * eps``), each ``sigma_i`` changing by at most a
  factor ``1 +- sigma_max_change`` per generation (evotorch's ``stdev_max_change``).

References:
- OpenES: https://arxiv.org/abs/1703.03864
- evosax: evosax/algorithms/distribution_based/open_es.py
- UPGrad: https://arxiv.org/abs/2406.16232
- SNES: https://doi.org/10.1145/2001576.2001692
- PGPE: https://doi.org/10.1016/j.neunet.2009.12.004
"""

from __future__ import annotations

import csv
import math
import os
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from scipy.stats import rankdata

from ...weighting.UPGrad import _compute_gramian, _normalize, _project_weights, _regularize
from .common import configure_moea_random_seed, should_run_full_test
from .hv import pareto_front_indices

AGGREGATORS = ("upgrad", "mean", "ws")
SIGMA_ADAPT = ("none", "snes", "pgpe")


def centered_ranks(fitness: np.ndarray) -> np.ndarray:
    """Centered ranks in ``[-0.5, 0.5]`` along the population axis (evosax ``centered_rank_fitness_shaping_fn``)."""
    ranks = rankdata(fitness, axis=0) - 1.0
    return ranks / (fitness.shape[0] - 1) - 0.5


FITNESS_SHAPING = ("centered_ranks", "raw")


def shape_fitness(fitness: np.ndarray, shaping: str) -> torch.Tensor:
    """Shaped fitness along the population axis: ``centered_ranks`` or ``raw`` (losses minus the population mean,
    the unshaped OpenAI-ES estimator; with antithetic pairs the mean cancels exactly)."""
    if shaping == "raw":
        return torch.as_tensor(fitness - fitness.mean(axis=0, keepdims=True), dtype=torch.float32)
    return torch.as_tensor(centered_ranks(fitness), dtype=torch.float32)


def aggregate_jacobian(
    J: torch.Tensor,
    aggregator: str,
    norm_eps: float = 1e-4,
    reg_eps: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Combine Jacobian rows ``J`` (``[K, D]``) into one direction ``d = J^T w``; return ``(d, w)``."""
    K = J.shape[0]
    if aggregator == "upgrad":
        U = torch.diag(torch.full((K,), 1.0 / K, dtype=J.dtype))
        G = _regularize(_normalize(_compute_gramian(J), norm_eps), reg_eps)
        w = torch.sum(_project_weights(U, G), dim=0)
    elif aggregator == "mean":
        w = torch.full((K,), 1.0 / K, dtype=J.dtype)
    else:
        raise ValueError(f"aggregate_jacobian: unknown aggregator {aggregator!r}")
    return w @ J, w


def make_es_optimizer(
    params: list,
    optim: str,
    lr: float,
    adam_betas: Tuple[float, float] = (0.9, 0.999),
    adam_eps: float = 1e-8,
    rmsprop_alpha: float = 0.99,
    rmsprop_eps: float = 1e-8,
    sgd_momentum: float = 0.0,
) -> torch.optim.Optimizer:
    """Optimizer on the search mean ``theta``: ``sgd`` (heavy-ball momentum ``sgd_momentum``), ``adam`` or ``rmsprop``."""
    if optim == "adam":
        return torch.optim.Adam(params, lr=float(lr), betas=tuple(adam_betas), eps=float(adam_eps))
    if optim == "sgd":  # plain SGD on theta (evosax OpenES / GuidedES default optimizer)
        return torch.optim.SGD(params, lr=float(lr), momentum=float(sgd_momentum))
    if optim == "rmsprop":
        return torch.optim.RMSprop(params, lr=float(lr), alpha=float(rmsprop_alpha), eps=float(rmsprop_eps))
    raise ValueError(f"Unknown ES optimizer {optim!r}; options: sgd, adam, rmsprop")


def guided_noise(
    eps: torch.Tensor,
    guide: list,
    gen: torch.Generator,
    sigma: float,
    guided_alpha: float,
    guided_sigma: float,
) -> torch.Tensor:
    """Guided ES sampling: keep ``sqrt(alpha)`` of the isotropic noise ``eps`` and add noise along ``guide``."""
    U = torch.linalg.qr(torch.stack(guide, dim=1))[0]  # [D, k'] orthonormal guiding basis
    e_sub = torch.randn(eps.shape[0], U.shape[1], generator=gen, dtype=torch.float32)
    sub_scale = (guided_sigma / sigma) if guided_sigma > 0 else math.sqrt((1.0 - guided_alpha) * eps.shape[1] / U.shape[1])
    return math.sqrt(guided_alpha) * eps + sub_scale * (e_sub @ U.T)


def update_guide(guide: list, direction: torch.Tensor, guided_k: int, guided_ema: float) -> list:
    """Add the unit ES ``direction`` to the guiding directions (FIFO of ``guided_k``, or one EMA)."""
    u = (direction / direction.norm().clamp_min(1e-30)).detach().clone()
    if guided_ema > 0:  # one guiding direction: running average of past directions
        return [u if not guide else guided_ema * guide[0] + (1.0 - guided_ema) * u]
    return (guide + [u])[-int(guided_k):]


def _mean_pairwise_cosine(J: torch.Tensor) -> float:
    Jn = J / J.norm(dim=1, keepdim=True).clamp_min(1e-12)
    C = Jn @ Jn.T
    K = J.shape[0]
    if K < 2:
        return 1.0
    return float((C.sum() - C.diagonal().sum()) / (K * (K - 1)))


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(a @ b / (a.norm() * b.norm()).clamp_min(1e-30))


def _sigma_stats(sigma: Any) -> Tuple[float, float, float]:
    """(mean, min, max) of a scalar or per-parameter sigma."""
    if isinstance(sigma, torch.Tensor):
        return float(sigma.mean()), float(sigma.min()), float(sigma.max())
    return float(sigma), float(sigma), float(sigma)


def adapt_sigma(
    sigma: torch.Tensor,
    u: torch.Tensor,
    eps: torch.Tensor,
    rule: str,
    lr: float,
    max_change: float,
) -> torch.Tensor:
    """One SNES / PGPE step on the per-parameter ``sigma`` [D] from shaped fitness ``u`` [N] (higher = worse)
    and the standard-normal noise ``eps`` [N, D] (see module docstring)."""
    g_sigma = (u @ (eps * eps - 1.0)) / eps.shape[0]
    if rule == "snes":
        return sigma * torch.exp(-0.5 * lr * g_sigma)
    new = sigma - lr * sigma * g_sigma
    return torch.minimum(torch.maximum(new, sigma * (1.0 - max_change)), sigma * (1.0 + max_change))


def true_jacobian(trainer: Any, theta: torch.Tensor, batches: list) -> torch.Tensor:
    """Backprop Jacobian ``[K, D]`` of the per-task losses w.r.t. the weights at ``theta``.

    Same mini-batches and eval-mode forward as the population fitness.
    """
    model = trainer.model
    trainer.set_theta(theta)
    model.eval()
    params = list(model.parameters())
    K = int(trainer.task_num)
    J_theta = torch.zeros(K, sum(p.numel() for p in params), dtype=torch.float32, device=trainer.device)
    with torch.enable_grad():
        for inp, gts in batches:
            losses = trainer.forward4loss(model, inp, gts)
            for k in range(K):
                grads = torch.autograd.grad(losses[k], params, retain_graph=k < K - 1, allow_unused=True)
                J_theta[k] += torch.cat([
                    (g if g is not None else torch.zeros_like(p)).reshape(-1).float()
                    for g, p in zip(grads, params)
                ])
    return (J_theta / len(batches)).cpu()


def run_mo_openes(
    trainer: Any,
    train_dataloaders: Any,
    num_iterations: int,
    pop_size: int,
    lr: float = 1e-3,
    sigma: float = 0.01,
    sigma_decay: float = 1.0,
    sigma_min: float = 0.0,
    sigma_schedule: str = "constant",
    sigma_adapt: str = "none",
    sigma_lr: float = 0.0,
    sigma_max_change: float = 0.2,
    fitness_shaping: str = "raw",
    aggregator: str = "upgrad",
    antithetic: bool = True,
    optim: str = "adam",
    adam_betas: Tuple[float, float] = (0.9, 0.999),
    adam_eps: float = 1e-8,
    rmsprop_alpha: float = 0.99,
    rmsprop_eps: float = 1e-8,
    sgd_momentum: float = 0.0,
    upgrad_norm_eps: float = 1e-4,
    upgrad_reg_eps: float = 1e-4,
    num_batches: int = 1,
    seed: int = 0,
    hv_ref_point: Optional[np.ndarray] = None,
    wandb_step_offset: int = 0,
    test_dataloaders: Optional[Any] = None,
    evo_eval_freq: int = 1,
    grad_cos_freq: int = 1,
    grad_save_freq: int = 0,
    guided_k: int = 0,
    guided_alpha: float = 0.5,
    guided_sigma: float = 0.0,
    guided_ema: float = 0.0,
    **_unused: Any,
) -> Dict[str, Any]:
    """MO-OpenES ask-and-tell on the weights ``theta`` (see module docstring).

    The search distribution is ``N(theta, sigma^2 I)`` with ``theta`` starting at the current
    weights ``theta_0``; ``theta`` is the returned solution (``pareto_center_X``) and the trainer
    ends at it. ``pareto_F`` / ``pareto_X`` are the non-dominated members of the last population
    (for logging and checkpoints).

    Parameters
    ----------
    lr : float
        Learning rate of the optimizer on ``theta``.
    optim, adam_betas, adam_eps, rmsprop_alpha, rmsprop_eps
        Optimizer for ``theta``: ``"sgd"``, ``"adam"`` (default; ``adam_*`` apply) or ``"rmsprop"``
        (``rmsprop_*`` apply; ``alpha`` is the squared-gradient smoothing constant).
    sigma, sigma_decay, sigma_min
        Perturbation std; after each update ``sigma = max(sigma * sigma_decay, sigma_min)``.
        With ``sigma_adapt`` it is the initial value of every ``sigma_i``.
    sigma_schedule : str
        ``"constant"`` (default; ``sigma_decay`` applies) or ``"cosine"``: generation ``t`` (0-based) samples
        with ``sigma_min + (sigma - sigma_min) * (1 + cos(pi * t / num_iterations)) / 2``, i.e. ``sigma`` at
        the first generation, annealed towards ``sigma_min`` at the end. Needs ``sigma_decay = 1`` and no
        ``sigma_adapt``.
    sigma_adapt, sigma_lr, sigma_max_change
        Per-parameter step sizes: ``"none"`` (isotropic OpenES), ``"snes"`` or ``"pgpe"`` (see module
        docstring). ``sigma_lr = 0`` picks the default: ``(3 + ln d) / (5 sqrt d)`` for SNES, ``0.1`` for
        PGPE. ``sigma_max_change`` bounds PGPE's relative change per generation. Not combinable with
        Guided ES sampling.
    fitness_shaping : str
        ``"raw"`` (default): the losses minus the population mean, i.e. the unshaped estimator
        ``F @ eps / (N sigma)`` of the Gaussian-smoothed gradient, whose size Adam's moments can track; or
        ``"centered_ranks"`` (evosax / OpenAI-ES), bounded and robust to outliers but of near-constant size.
    aggregator : str
        ``"upgrad"`` (default), ``"mean"``, or ``"ws"`` (see module docstring).
    antithetic : bool
        Mirrored sampling ``eps, -eps`` (evosax default); requires even ``pop_size``.
    evo_eval_freq : int
        Full test-set evaluation every this many iterations (see :func:`run_nsga2`).
    grad_cos_freq : int
        Every this many iterations, backprop the true per-task gradients at ``theta`` on the
        same mini-batch and log cosine(pseudo-gradient_k, true gradient_k) and
        cosine(ES direction, same aggregator applied to the true Jacobian). ``0`` disables.
        Diagnostic only: the update never uses true gradients.
    grad_save_freq : int
        Every this many iterations, starting at iteration 0 (``theta_0``, before the first
        update), store the search mean ``theta`` and the true and pseudo-gradients per objective plus the
        aggregated ES direction and its true-gradient counterpart in ``<save_path>/gradients.pth.tar``
        (float32, rewritten at each snapshot). ``0`` disables. Needs a ``save_path``.
    guided_k, guided_alpha
        Guided ES sampling (Maheswaranathan et al., 2019; evosax ``GuidedES``). ``guided_k > 0``
        keeps the last ``k`` ES directions (unit vectors, FIFO) as guiding subspace U and draws
        ``eps = sqrt(alpha) * e + sqrt((1 - alpha) * D / k) * U e_k`` with ``e ~ N(0, I_D)``,
        ``e_k ~ N(0, I_k)``: a fraction ``alpha`` of the noise budget stays isotropic, the rest goes
        into the subspace, and the total perturbation size matches plain OpenES (``alpha = 1``).
        Self-guided: no backprop. The rank-based estimator below is unchanged.
    guided_sigma
        If > 0, the perturbation size along each guiding direction (in weight units), set directly
        instead of ``sigma * sqrt((1 - alpha) * D / k)``. Keeps the guided part in the near-linear range.
    guided_ema
        If > 0, guide with one direction: an exponential moving average of past unit ES directions
        with this decay (0.9 ~ the last 10 generations), instead of the FIFO of the last k.
    """
    if trainer.theta_0 is None:
        raise RuntimeError("prepare_evolution() first.")
    aggregator = str(aggregator).strip().lower()
    if aggregator not in AGGREGATORS:
        raise ValueError(f"Unknown MO-OpenES aggregator {aggregator!r}; options: {AGGREGATORS}")
    pop_size = int(pop_size)
    if antithetic and pop_size % 2 != 0:
        raise ValueError(f"Antithetic sampling needs an even pop_size, got {pop_size}.")
    sigma_adapt = str(sigma_adapt).strip().lower()
    if sigma_adapt not in SIGMA_ADAPT:
        raise ValueError(f"Unknown sigma_adapt {sigma_adapt!r}; options: {SIGMA_ADAPT}")
    if sigma_adapt != "none" and guided_k > 0:
        raise ValueError("sigma_adapt (snes / pgpe) is not combinable with Guided ES sampling (guided_k > 0).")
    fitness_shaping = str(fitness_shaping).strip().lower()
    if fitness_shaping not in FITNESS_SHAPING:
        raise ValueError(f"Unknown fitness_shaping {fitness_shaping!r}; options: {FITNESS_SHAPING}")
    sigma_schedule = str(sigma_schedule).strip().lower()
    if sigma_schedule not in ("constant", "cosine"):
        raise ValueError(f"Unknown sigma_schedule {sigma_schedule!r}; options: constant, cosine")
    if sigma_schedule == "cosine" and (sigma_adapt != "none" or float(sigma_decay) != 1.0):
        raise ValueError("sigma_schedule=cosine needs sigma_decay=1 and sigma_adapt=none.")
    sigma_0 = float(sigma)

    configure_moea_random_seed(seed)
    gen = torch.Generator().manual_seed(int(seed))

    n_var = int(trainer.theta_0.numel())
    n_obj = int(trainer.task_num)
    if grad_save_freq > 0 and not getattr(trainer, "save_path", None):
        print("[EvoMTL] grad_save_freq ignored: needs a save_path.", flush=True)
        grad_save_freq = 0
    grad_snapshots: Dict[str, list] = {"generation": [], "mean": [], "true": [], "pseudo": [],
                                       "direction": [], "true_direction": []}
    theta_0 = trainer.theta_0.detach().cpu().float()
    theta = theta_0.clone()
    theta.requires_grad_(True)
    optim = str(optim).strip().lower()
    optimizer = make_es_optimizer([theta], optim, lr, adam_betas, adam_eps, rmsprop_alpha, rmsprop_eps, sgd_momentum)
    sigma = float(sigma)
    if sigma_adapt != "none":  # one step size per parameter, all starting at sigma
        sigma = torch.full((n_var,), sigma, dtype=torch.float32)
        if sigma_lr <= 0:
            sigma_lr = (3.0 + math.log(n_var)) / (5.0 * math.sqrt(n_var)) if sigma_adapt == "snes" else 0.1
    method = f"MO-OpenES-{aggregator.upper()}" if aggregator != "ws" else "OpenES"
    if guided_k > 0:  # Guided ES sampling: OpenES -> GuidedES, MO-OpenES-X -> MO-GuidedES-X
        method = method.replace("OpenES", "GuidedES")
    if sigma_adapt != "none":  # OpenES -> SNES / PGPE, MO-OpenES-X -> MO-SNES-X / MO-PGPE-X
        method = method.replace("OpenES", sigma_adapt.upper())
    guide: list = []  # unit vectors of the last guided_k ES directions

    print(
        f"[EvoMTL] {method}: dims={n_var} | pop={pop_size} | optim={optim} lr={lr} | sigma={_sigma_stats(sigma)[0]} "
        f"(schedule={sigma_schedule}, decay={sigma_decay}, min={sigma_min}) | antithetic={antithetic} "
        f"| fitness={fitness_shaping}"
        + (f" | sigma_adapt={sigma_adapt} lr_sigma={sigma_lr:.4g}"
           + (f" max_change={sigma_max_change}" if sigma_adapt == "pgpe" else "") if sigma_adapt != "none" else "")
        + (f" | guided k={guided_k}, alpha={guided_alpha}, sigma_sub={guided_sigma or 'auto'}, ema={guided_ema}"
           if guided_k > 0 else ""),
        flush=True,
    )

    num_iterations = int(num_iterations)
    population_X = np.empty((0, n_var))
    population_F = np.empty((0, n_obj))

    # metrics.csv next to the checkpoints: one row per generation, flushed as it is written.
    # Test columns are filled only on generations with a test evaluation (evo_eval_freq).
    tasks = list(trainer.task_name)
    csv_cols = (
        ["generation", "time_s", "sigma", "sigma_min", "sigma_max"]
        + [f"pop_loss_mean_{t}" for t in tasks]
        + [f"train_loss_{t}" for t in tasks] + [f"train_acc_{t}" for t in tasks]
        + [f"test_loss_{t}" for t in tasks] + [f"test_acc_{t}" for t in tasks]
        + [f"weight_{t}" for t in tasks] + ["pseudo_grad_cos"]
        + [f"true_cos_{t}" for t in tasks] + ["true_cos_direction"]
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
        if guided_k > 0 and guide:
            eps = guided_noise(eps, guide, gen, sigma, guided_alpha, guided_sigma)
        if antithetic:
            eps = torch.cat([eps, -eps])
        z = sigma * eps  # perturbations: member i is theta + z_i
        population = theta.detach() + z

        # evaluate every member on the same mini-batch(es)
        shared_batches = trainer._sample_batches(train_dataloaders, num_batches)
        F = np.zeros((pop_size, n_obj), dtype=np.float32)
        for i in range(pop_size):
            trainer.set_theta(population[i])
            F[i] = trainer.evaluate_multiobjective_losses(
                train_dataloaders, prefetched_batches=shared_batches
            )

        # tell: per-objective rank-based pseudo-gradients -> Jacobian -> common direction
        scale = 1.0 / (pop_size * sigma)
        if aggregator == "ws":
            shaped = shape_fitness(F.sum(axis=1), fitness_shaping)
            J = (shaped @ eps * scale).unsqueeze(0)
            direction, w = J[0], torch.ones(1, dtype=torch.float32)
        else:
            shaped = shape_fitness(F, fitness_shaping)  # [N, K]
            J = shaped.T @ eps * scale  # [K, D]
            direction, w = aggregate_jacobian(J, aggregator, upgrad_norm_eps, upgrad_reg_eps)

        # diagnostic: compare with backprop gradients at the (pre-update) theta, same batch
        true_cos = None
        do_cos = grad_cos_freq > 0 and iteration % int(grad_cos_freq) == 0
        do_save = grad_save_freq > 0 and iteration % int(grad_save_freq) == 0
        if do_cos or do_save:
            J_true = true_jacobian(trainer, theta.detach(), shared_batches)
            J_obj = shape_fitness(F, fitness_shaping).T @ eps * scale
            if aggregator == "ws":
                d_true = J_true.sum(dim=0)
            else:
                d_true, _ = aggregate_jacobian(J_true, aggregator, upgrad_norm_eps, upgrad_reg_eps)
        if do_cos:
            true_cos = [_cosine(J_obj[k], J_true[k]) for k in range(n_obj)]
            dir_cos = _cosine(direction, d_true)
        if do_save:
            # gradients at the pre-update theta of generation iteration+1 (iteration 0 = initialization)
            for key, val in (("generation", iteration), ("mean", theta.detach()), ("true", J_true), ("pseudo", J_obj),
                             ("direction", direction), ("true_direction", d_true)):
                grad_snapshots[key].append(val if key == "generation" else val.detach().float().clone())
            torch.save({
                "generation": torch.tensor(grad_snapshots["generation"]),
                "mean": torch.stack(grad_snapshots["mean"]),            # [S, D] search mean theta (weights)
                "true": torch.stack(grad_snapshots["true"]),            # [S, K, D] backprop gradients
                "pseudo": torch.stack(grad_snapshots["pseudo"]),        # [S, K, D] per-objective ES estimates
                "direction": torch.stack(grad_snapshots["direction"]),  # [S, D] aggregated ES direction
                "true_direction": torch.stack(grad_snapshots["true_direction"]),  # [S, D] same rule on true J
                "tasks": list(trainer.task_name), "aggregator": aggregator, "method": method,
                "pop_size": pop_size, "lr": float(lr), "sigma": sigma, "dims": n_var,
            }, os.path.join(trainer.save_path, "gradients.pth.tar"))

        theta.grad = direction.clone()
        optimizer.step()
        if guided_k > 0:
            guide = update_guide(guide, direction, guided_k, guided_ema)
        if sigma_adapt != "none":
            # shaped fitness of the aggregated objective: the summed loss, or the ranks weighted like the update
            u = shaped if aggregator == "ws" else shaped @ (w / w.abs().sum().clamp_min(1e-30))
            sigma = adapt_sigma(sigma, u, eps, sigma_adapt, sigma_lr, sigma_max_change)
            sigma = (sigma * float(sigma_decay)).clamp_min(float(sigma_min))
        elif sigma_schedule == "cosine":  # sigma for the next generation (iteration + 1)
            sigma = float(sigma_min) + 0.5 * (sigma_0 - float(sigma_min)) * (
                1.0 + math.cos(math.pi * (iteration + 1) / num_iterations))
        else:
            sigma = max(sigma * float(sigma_decay), float(sigma_min))
        sig_mean, sig_min, sig_max = _sigma_stats(sigma)

        population_X = population.numpy()
        population_F = F
        nd_idx = pareto_front_indices(F)
        pareto_front = F[nd_idx]
        elapsed = time.perf_counter() - t0

        # diagnostics
        grad_cos = _mean_pairwise_cosine(J) if J.shape[0] > 1 else float("nan")
        conflict = (J @ direction).min().item() if J.shape[0] > 1 else float("nan")
        extras_wandb: Dict[str, Any] = {
            "evo/openes/sigma": sig_mean,
            "evo/openes/direction_norm": float(direction.norm()),
            "evo/openes/mean_norm": float((theta.detach() - theta_0).norm()),  # distance from theta_0
            "evo/openes/pop_losses/avg": float(F.mean()),
        }
        if sigma_adapt != "none":
            extras_wandb["evo/openes/sigma_min"], extras_wandb["evo/openes/sigma_max"] = sig_min, sig_max
            progress = f"sigma={sig_mean:.4g} [{sig_min:.3g}, {sig_max:.3g}] |d|={float(direction.norm()):.4g}"
        else:
            progress = f"sigma={sig_mean:.4g} |d|={float(direction.norm()):.4g}"
        if J.shape[0] > 1:
            extras_wandb["evo/openes/grad_cos"] = grad_cos
            extras_wandb["evo/openes/min_gk_dot_d"] = conflict
            for k, tn in enumerate(trainer.task_name):
                extras_wandb[f"evo/openes/weight/{tn}"] = float(w[k])
                extras_wandb[f"evo/openes/grad_norm/{tn}"] = float(J[k].norm())
            progress += f" cos(g)={grad_cos:+.3f} w=[" + ", ".join(f"{float(x):.3f}" for x in w) + "]"
        if true_cos is not None:
            progress += " true_cos=[" + ", ".join(f"{c:+.4f}" for c in true_cos) + f"] dir_cos={dir_cos:+.4f}"
            for k, tn in enumerate(trainer.task_name):
                extras_wandb[f"evo/openes/true_cos/{tn}"] = true_cos[k]
            extras_wandb["evo/openes/true_cos/direction"] = dir_cos

        trainer._print_evo_progress(
            method,
            iteration,
            num_iterations,
            pareto_front,
            extras=progress,
            elapsed=elapsed,
            offspring_X=population_X,
            wandb_log_step=step,
        )
        trainer._wandb_log_evo_front_metrics(iteration, pareto_front, step, extras=extras_wandb)

        # theta is the ES solution: snapshot it on the current batch (and test periodically)
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
            test_metrics = trainer.test(
                test_dataloaders,
                epoch=None,
                mode="test",
                wandb_log_step=step,
                suppress_display=True,
                return_metrics=True,
            )
            trainer.print_compact_train_test_line(
                train_batch_metrics, test_metrics, epoch_label=iteration + 1
            )
        else:
            parts = " | ".join(
                f"{task}: {float(train_batch_metrics['loss_item'][i]):.4f}"
                for i, task in enumerate(trainer.task_name)
            )
            print(
                f"[EvoMTL] Mean theta TRAIN (curr batch) iteration {iteration + 1:04d} | {parts}",
                flush=True,
            )

        if csv_writer is not None:
            row: Dict[str, Any] = {"generation": iteration + 1, "time_s": round(elapsed, 4),
                                   "sigma": sig_mean, "sigma_min": sig_min, "sigma_max": sig_max}
            for k, t in enumerate(tasks):
                row[f"pop_loss_mean_{t}"] = float(F[:, k].mean())
                row[f"train_loss_{t}"] = float(train_batch_metrics["loss_item"][k])
                row[f"train_acc_{t}"] = float(train_batch_metrics["results"][t][0])
                if test_metrics is not None:
                    row[f"test_loss_{t}"] = float(test_metrics["loss_item"][k])
                    row[f"test_acc_{t}"] = float(test_metrics["results"][t][0])
                if len(w) == n_obj:
                    row[f"weight_{t}"] = float(w[k])
                if true_cos is not None:
                    row[f"true_cos_{t}"] = true_cos[k]
            if J.shape[0] > 1:
                row["pseudo_grad_cos"] = grad_cos
            if true_cos is not None:
                row["true_cos_direction"] = dir_cos
            csv_writer.writerow(row)
            csv_file.flush()

    if csv_file is not None:
        csv_file.close()
    if isinstance(sigma, torch.Tensor):
        # final step size per parameter tensor (same order as the flat weight vector theta)
        start = 0
        for name, p in trainer.model.named_parameters():
            s = sigma[start:start + p.numel()]
            print(f"[EvoMTL] sigma {name:45s} mean={float(s.mean()):.4g} "
                  f"min={float(s.min()):.4g} max={float(s.max()):.4g}", flush=True)
            start += p.numel()
        if getattr(trainer, "save_path", None):
            torch.save(sigma.float(), os.path.join(trainer.save_path, "sigma_final.pt"))

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
        "sigma": sigma.numpy() if isinstance(sigma, torch.Tensor) else sigma,
        "aggregator": aggregator,
    }
    trainer.last_evo = out
    return out
