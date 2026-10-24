"""Ask-and-tell loop of the OpenES family on the network weights ``theta``.

The search distribution is over the network weights: member ``i`` is ``theta + z_i`` around the search mean ``theta``,
with perturbation ``z_i = sigma * eps_i`` and standard-normal noise ``eps_i ~ N(0, I)`` (antithetic pairs
``eps, -eps`` by default). Every generation evaluates the population on one shared mini-batch, asks an *estimator*
(the algorithm: :mod:`.openes`, :mod:`.mo_upgrad_openes`, :mod:`.mo_mean_openes`, ...) for the ES direction, steps the
optimizer on ``theta`` with it as the gradient, and asks a *sigma rule* for the next step size (constant / decay,
cosine, or per-parameter SNES / PGPE). An optional *noise sampler* (Guided ES, :class:`~.es_common.GuidedSampler`)
replaces the isotropic noise.

Estimator interface:

* ``aggregator`` -- label saved with the run (``"ws"``, ``"upgrad"``, ``"mean"``).
* ``estimate(F, eps, scale, fitness_shaping) -> (J, direction, w, u)``: the pseudo-gradient rows ``J`` (``[K', D]``),
  the ES (ascent) direction, the weights of the rows in it, and ``u`` (``[N]``), the shaped fitness of the aggregated
  objective (higher = worse) that per-parameter sigma rules adapt to.
* ``true_direction(J_true)``: the same rule applied to the backprop Jacobian (diagnostic).

Sigma rule interface: ``init(sigma, n_var)`` returns the initial step size (a float, or a ``[D]`` tensor for one step
size per parameter), ``next(sigma, u, eps, iteration)`` the step size of generation ``iteration + 1`` and
``describe()`` a short summary for the log.
"""

from __future__ import annotations

import math
import os
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from .common import configure_moea_random_seed
from .es_common import (
    FITNESS_SHAPING,
    GradientSnapshots,
    MetricsCSV,
    _cosine,
    _mean_pairwise_cosine,
    es_result,
    make_es_optimizer,
    metrics_columns,
    report_search_mean,
    shape_fitness,
    true_jacobian,
)
from .hv import pareto_front_indices


class DecaySigma:
    """Isotropic ``sigma``; after each update ``sigma = max(sigma * decay, sigma_min)`` (``decay = 1``: constant)."""

    def __init__(self, decay: float = 1.0, sigma_min: float = 0.0):
        self.decay, self.sigma_min = float(decay), float(sigma_min)

    def describe(self) -> str:
        return f"schedule=constant, decay={self.decay}, min={self.sigma_min}"

    def init(self, sigma: float, n_var: int) -> float:
        return float(sigma)

    def next(self, sigma: float, u: torch.Tensor, eps: torch.Tensor, iteration: int) -> float:
        return max(sigma * self.decay, self.sigma_min)


class CosineSigma:
    """Isotropic ``sigma`` annealed from ``sigma_0`` to ``sigma_min``: generation ``t`` (0-based) samples with
    ``sigma_min + (sigma_0 - sigma_min) * (1 + cos(pi * t / num_iterations)) / 2``."""

    def __init__(self, sigma_0: float, sigma_min: float, num_iterations: int):
        self.sigma_0, self.sigma_min, self.num_iterations = float(sigma_0), float(sigma_min), int(num_iterations)

    def describe(self) -> str:
        return f"schedule=cosine, min={self.sigma_min}"

    def init(self, sigma: float, n_var: int) -> float:
        return float(sigma)

    def next(self, sigma: float, u: torch.Tensor, eps: torch.Tensor, iteration: int) -> float:
        return self.sigma_min + 0.5 * (self.sigma_0 - self.sigma_min) * (
            1.0 + math.cos(math.pi * (iteration + 1) / self.num_iterations))


def make_sigma_rule(sigma: float, sigma_decay: float, sigma_min: float, sigma_schedule: str, num_iterations: int):
    """Isotropic sigma rule from the ``sigma_schedule`` option: ``"constant"`` (with ``sigma_decay``) or ``"cosine"``."""
    sigma_schedule = str(sigma_schedule).strip().lower()
    if sigma_schedule == "constant":
        return DecaySigma(sigma_decay, sigma_min)
    if sigma_schedule == "cosine":
        if float(sigma_decay) != 1.0:
            raise ValueError("sigma_schedule=cosine needs sigma_decay=1.")
        return CosineSigma(sigma, sigma_min, num_iterations)
    raise ValueError(f"Unknown sigma_schedule {sigma_schedule!r}; options: constant, cosine")


def _sigma_stats(sigma: Any) -> Tuple[float, float, float]:
    """(mean, min, max) of a scalar or per-parameter sigma."""
    if isinstance(sigma, torch.Tensor):
        return float(sigma.mean()), float(sigma.min()), float(sigma.max())
    return float(sigma), float(sigma), float(sigma)


def run_es(
    trainer: Any,
    train_dataloaders: Any,
    method: str,
    estimator: Any,
    sigma_rule: Any,
    num_iterations: int,
    pop_size: int,
    lr: float = 1e-3,
    sigma: float = 0.01,
    fitness_shaping: str = "raw",
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
) -> Dict[str, Any]:
    """Run ``num_iterations`` generations of ``estimator`` with ``sigma_rule`` (see module docstring).

    ``method`` is the logged name; with a ``noise_sampler`` its ``"OpenES"`` becomes the sampler's ``name``. The
    search mean ``theta`` starts at ``theta_0`` and is the returned solution (``pareto_center_X``); the trainer ends at
    it. ``pareto_F`` / ``pareto_X`` are the non-dominated members of the last population.

    Parameters
    ----------
    lr : float
        Learning rate of the optimizer on ``theta``.
    optim, adam_betas, adam_eps, rmsprop_alpha, rmsprop_eps, sgd_momentum
        Optimizer for ``theta``: ``"sgd"`` (heavy-ball ``sgd_momentum``), ``"adam"`` (default; ``adam_*`` apply) or
        ``"rmsprop"`` (``rmsprop_*`` apply; ``alpha`` is the squared-gradient smoothing constant).
    sigma : float
        Initial perturbation std (``sigma_rule`` sets the later ones).
    fitness_shaping : str
        ``"raw"`` (default): the losses minus the population mean, i.e. the unshaped estimator
        ``F @ eps / (N sigma)`` of the Gaussian-smoothed gradient, whose size Adam's moments can track; or
        ``"centered_ranks"`` (evosax / OpenAI-ES), bounded and robust to outliers but of near-constant size.
    antithetic : bool
        Mirrored sampling ``eps, -eps`` (evosax default); requires even ``pop_size``.
    noise_sampler : object, optional
        Replaces the isotropic noise: ``name`` replaces ``"OpenES"`` in ``method``, ``describe()`` is printed,
        ``sample(eps, gen, sigma)`` returns the noise of the first half of the population from the standard-normal
        ``eps`` and ``update(direction)`` sees each applied ES direction. The pseudo-gradients regress on the
        returned noise.
    evo_eval_freq : int
        Full test-set evaluation every this many iterations (see :func:`~.common.should_run_full_test`).
    grad_cos_freq : int
        Every this many iterations, backprop the true per-task gradients at ``theta`` on the same mini-batch and
        log cosine(pseudo-gradient_k, true gradient_k) and cosine(ES direction, the estimator's rule applied to the
        true Jacobian). ``0`` disables. Diagnostic only: the update never uses true gradients.
    grad_save_freq : int
        Every this many iterations, starting at iteration 0 (``theta_0``, before the first update), store the search
        mean ``theta``, the true and pseudo-gradients per objective, the ES direction and its true-gradient
        counterpart in ``<save_path>/gradients.pth.tar`` (see :class:`~.es_common.GradientSnapshots`). ``0``
        disables. Needs a ``save_path``.
    """
    if trainer.theta_0 is None:
        raise RuntimeError("prepare_evolution() first.")
    pop_size = int(pop_size)
    if antithetic and pop_size % 2 != 0:
        raise ValueError(f"Antithetic sampling needs an even pop_size, got {pop_size}.")
    fitness_shaping = str(fitness_shaping).strip().lower()
    if fitness_shaping not in FITNESS_SHAPING:
        raise ValueError(f"Unknown fitness_shaping {fitness_shaping!r}; options: {FITNESS_SHAPING}")
    if noise_sampler is not None:  # e.g. OpenES -> GuidedES, MO-UPGrad-OpenES -> MO-UPGrad-GuidedES
        method = method.replace("OpenES", noise_sampler.name)

    configure_moea_random_seed(seed)
    gen = torch.Generator().manual_seed(int(seed))

    n_var = int(trainer.theta_0.numel())
    n_obj = int(trainer.task_num)
    snapshots = GradientSnapshots(trainer, grad_save_freq)
    theta_0 = trainer.theta_0.detach().cpu().float()
    theta = theta_0.clone()
    theta.requires_grad_(True)
    optim = str(optim).strip().lower()
    optimizer = make_es_optimizer([theta], optim, lr, adam_betas, adam_eps, rmsprop_alpha, rmsprop_eps, sgd_momentum)
    sigma = sigma_rule.init(sigma, n_var)
    per_parameter = isinstance(sigma, torch.Tensor)

    print(
        f"[EvoMTL] {method}: dims={n_var} | pop={pop_size} | optim={optim} lr={lr} | sigma={_sigma_stats(sigma)[0]} "
        f"({sigma_rule.describe()}) | antithetic={antithetic} | fitness={fitness_shaping}"
        + (f" | {noise_sampler.describe()}" if noise_sampler is not None else ""),
        flush=True,
    )

    num_iterations = int(num_iterations)
    population_X = np.empty((0, n_var))
    population_F = np.empty((0, n_obj))

    tasks = list(trainer.task_name)
    metrics_csv = MetricsCSV(trainer, metrics_columns(
        tasks, ["sigma", "sigma_min", "sigma_max"],
        [f"weight_{t}" for t in tasks] + ["pseudo_grad_cos"] + [f"true_cos_{t}" for t in tasks]
        + ["true_cos_direction"]))

    for iteration in range(num_iterations):
        t0 = time.perf_counter()
        step = wandb_step_offset + iteration

        # ask
        n_draw = pop_size // 2 if antithetic else pop_size
        eps = torch.randn(n_draw, n_var, generator=gen, dtype=torch.float32)
        if noise_sampler is not None:
            eps = noise_sampler.sample(eps, gen, sigma)
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

        # tell: pseudo-gradients -> ES direction
        scale = 1.0 / (pop_size * sigma)
        J, direction, w, u = estimator.estimate(F, eps, scale, fitness_shaping)

        # diagnostic: compare with backprop gradients at the (pre-update) theta, same batch
        true_cos = None
        do_cos = grad_cos_freq > 0 and iteration % int(grad_cos_freq) == 0
        do_save = snapshots.due(iteration)
        if do_cos or do_save:
            J_true = true_jacobian(trainer, theta.detach(), shared_batches)
            J_obj = shape_fitness(F, fitness_shaping).T @ eps * scale  # per-objective pseudo-gradients
            d_true = estimator.true_direction(J_true)
        if do_cos:
            true_cos = [_cosine(J_obj[k], J_true[k]) for k in range(n_obj)]
            dir_cos = _cosine(direction, d_true)
        if do_save:
            # gradients at the pre-update theta of generation iteration+1 (iteration 0 = initialization)
            snapshots.save(iteration, {
                "mean": theta.detach(),          # [S, D] search mean theta (weights)
                "true": J_true,                  # [S, K, D] backprop gradients
                "pseudo": J_obj,                 # [S, K, D] per-objective ES estimates
                "direction": direction,          # [S, D] ES direction
                "true_direction": d_true,        # [S, D] same rule on the true Jacobian
            }, {"tasks": tasks, "aggregator": estimator.aggregator, "method": method,
                "pop_size": pop_size, "lr": float(lr), "sigma": sigma, "dims": n_var})

        theta.grad = direction.clone()
        optimizer.step()
        if noise_sampler is not None:
            noise_sampler.update(direction)
        sigma = sigma_rule.next(sigma, u, eps, iteration)
        sig_mean, sig_min, sig_max = _sigma_stats(sigma)

        population_X = population.numpy()
        population_F = F
        pareto_front = F[pareto_front_indices(F)]
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
        if per_parameter:
            extras_wandb["evo/openes/sigma_min"], extras_wandb["evo/openes/sigma_max"] = sig_min, sig_max
            progress = f"sigma={sig_mean:.4g} [{sig_min:.3g}, {sig_max:.3g}] |d|={float(direction.norm()):.4g}"
        else:
            progress = f"sigma={sig_mean:.4g} |d|={float(direction.norm()):.4g}"
        if J.shape[0] > 1:
            extras_wandb["evo/openes/grad_cos"] = grad_cos
            extras_wandb["evo/openes/min_gk_dot_d"] = conflict
            for k, tn in enumerate(tasks):
                extras_wandb[f"evo/openes/weight/{tn}"] = float(w[k])
                extras_wandb[f"evo/openes/grad_norm/{tn}"] = float(J[k].norm())
            progress += f" cos(g)={grad_cos:+.3f} w=[" + ", ".join(f"{float(x):.3f}" for x in w) + "]"
        if true_cos is not None:
            progress += " true_cos=[" + ", ".join(f"{c:+.4f}" for c in true_cos) + f"] dir_cos={dir_cos:+.4f}"
            for k, tn in enumerate(tasks):
                extras_wandb[f"evo/openes/true_cos/{tn}"] = true_cos[k]
            extras_wandb["evo/openes/true_cos/direction"] = dir_cos

        trainer._print_evo_progress(method, iteration, num_iterations, pareto_front, extras=progress,
                                    elapsed=elapsed, offspring_X=population_X, wandb_log_step=step)
        trainer._wandb_log_evo_front_metrics(iteration, pareto_front, step, extras=extras_wandb)

        train_batch_metrics, test_metrics = report_search_mean(
            trainer, theta, iteration, num_iterations, F, pareto_front, shared_batches, test_dataloaders,
            evo_eval_freq, step)

        row: Dict[str, Any] = {"sigma": sig_mean, "sigma_min": sig_min, "sigma_max": sig_max}
        if len(w) == n_obj:
            row.update({f"weight_{t}": float(w[k]) for k, t in enumerate(tasks)})
        if J.shape[0] > 1:
            row["pseudo_grad_cos"] = grad_cos
        if true_cos is not None:
            row.update({f"true_cos_{t}": true_cos[k] for k, t in enumerate(tasks)})
            row["true_cos_direction"] = dir_cos
        metrics_csv.write(iteration, elapsed, tasks, F, train_batch_metrics, test_metrics, **row)

    metrics_csv.close()
    if per_parameter:
        # final step size per parameter tensor (same order as the flat weight vector theta)
        start = 0
        for name, p in trainer.model.named_parameters():
            s = sigma[start:start + p.numel()]
            print(f"[EvoMTL] sigma {name:45s} mean={float(s.mean()):.4g} "
                  f"min={float(s.min()):.4g} max={float(s.max()):.4g}", flush=True)
            start += p.numel()
        if getattr(trainer, "save_path", None):
            torch.save(sigma.float(), os.path.join(trainer.save_path, "sigma_final.pt"))

    return es_result(trainer, method, theta, population_X, population_F, hv_ref_point, sigma,
                     aggregator=estimator.aggregator)
