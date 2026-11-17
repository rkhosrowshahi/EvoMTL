"""Hybrid-RLW-DUST: DUST for the first part of the budget, then RLW-DUST for the rest.

DUST (:mod:`.dust`) estimates every layer at one weight snapshot and updates them all together; RLW-DUST
(:mod:`.rlw_dust`) trains one layer at a time, from the last to the first, ``T_l`` inner steps each. While the whole
network is still close to its random initialization, a layer-wise step trains one layer against layers that are not
trained yet, and spends a mini-batch on one layer where DUST updates all of them. Hybrid-RLW-DUST therefore starts
with DUST and switches to RLW-DUST for the remaining, fine-grained training.

The budget is ``num_iterations`` mini-batches (the DUST generations of the same run length):

1. the first ``round(dust_fraction * num_iterations)`` iterations are DUST's: one mini-batch, every layer's jitter
   estimate (and the weight-space ES on the parameters outside the weight layers) at one snapshot, one optimizer step on all
   parameters;
2. the rest of the budget goes to RLW-DUST iterations, as many whole ones as fit: one iteration takes ``sum_l T_l``
   mini-batches, one inner step each, from the last group to the first. A leftover smaller than one iteration is
   unused.

One optimizer runs over the model's parameters through both phases, so its state (Adam moments and step counts)
carries over at the switch; in the RLW-DUST phase only the active group's parameters get a gradient. Each iteration is
logged at the W&B step of its last mini-batch, so both phases share one axis with DUST and LW-DUST. ``evo_eval_freq``
counts mini-batches: a test runs at every multiple of it, and at the first and last iteration. Single-objective: the
rewards come from the summed task loss. The estimator is :class:`~.rlw_dust.RLWDUST` (draws, groups, inner step);
DUST's phase uses its inherited :meth:`~.dust.DUST.estimate`.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .common import configure_moea_random_seed
from .es_common import (
    MetricsCSV,
    _cosine,
    es_result,
    make_es_optimizer,
    metrics_columns,
    report_search_mean,
    require_shuffled_loader,
    true_jacobian,
)
from .rlw_dust import RLWDUST

METHOD = "Hybrid-RLW-DUST"


def run_hybrid_rlw_dust(
    trainer: Any,
    train_dataloaders: Any,
    num_iterations: int,
    pop_size: int,
    dust_fraction: float = 0.5,
    inner_steps: Optional[List[int]] = None,
    lr: float = 1e-3,
    sigma: float = 0.05,
    chunk: int = 8,
    allocation: str = "equal",
    draw_scale: float = 1.0,
    draws: Optional[list] = None,
    es_draws: int = 8,
    es_sigma: float = 0.03,
    optim: str = "adam",
    adam_betas: Tuple[float, float] = (0.9, 0.999),
    adam_eps: float = 1e-8,
    rmsprop_alpha: float = 0.99,
    rmsprop_eps: float = 1e-8,
    sgd_momentum: float = 0.0,
    num_batches: int = 1,
    seed: int = 0,
    hv_ref_point: Optional[np.ndarray] = None,
    wandb_step_offset: int = 0,
    test_dataloaders: Optional[Any] = None,
    evo_eval_freq: Any = 1,
    grad_cos_freq: int = 1,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Hybrid-RLW-DUST training from ``theta_0`` (see module docstring); the trainer ends at the final weights.

    Parameters
    ----------
    num_iterations : int
        Budget in mini-batches (``num_batches`` each), shared by the two phases.
    dust_fraction : float
        Share of the budget for DUST, in ``[0, 1]``; the rest goes to RLW-DUST.
    inner_steps : list of int, optional
        RLW-DUST's ``T_l``: one count for every group, or one per group in forward order. Default ``[10]``.
    pop_size, allocation, draw_scale, draws, sigma, chunk, es_draws, es_sigma
        DUST's per-layer jitter draws and weight-space ES (:func:`~.dust.run_dust`), in both phases.
    lr, optim, adam_betas, adam_eps, rmsprop_alpha, rmsprop_eps, sgd_momentum
        One optimizer on the model's parameters for both phases (:func:`~.es_common.make_es_optimizer`).
    evo_eval_freq : int or set
        Test every this many mini-batches (or at the listed mini-batch indices), and at the first and last iteration.
    grad_cos_freq : int
        Every this many iterations: cosine of the estimate with the backprop gradient (DUST phase: the whole
        direction; RLW-DUST phase: every group at its first inner step, and the concatenated blocks). ``0`` disables.
    """
    if kwargs.get("aggregate") is not None:
        raise ValueError(f"{METHOD} uses the summed task loss; multi-objective aggregation is unsupported.")
    if trainer.theta_0 is None:
        raise RuntimeError("prepare_evolution() first.")
    if not 0.0 <= float(dust_fraction) <= 1.0:
        raise ValueError(f"{METHOD} needs 0 <= dust_fraction <= 1, got {dust_fraction}.")
    if int(chunk) < 1 or (allocation in ("equal", "width") and int(pop_size) < int(chunk)):
        raise ValueError(f"{METHOD} needs 1 <= chunk <= pop_size, got chunk={chunk}, pop_size={pop_size}.")
    require_shuffled_loader(train_dataloaders, METHOD)
    configure_moea_random_seed(seed)
    gen = torch.Generator(trainer.device).manual_seed(int(seed))
    trainer.set_theta(trainer.theta_0.detach().cpu().float())
    params = list(trainer.model.parameters())
    optim = str(optim).strip().lower()
    optimizer = make_es_optimizer(params, optim, lr, adam_betas, adam_eps, rmsprop_alpha, rmsprop_eps, sgd_momentum)
    inner_steps = inner_steps or [10]
    dust = RLWDUST(trainer, inner_steps, pop_size, sigma, chunk, allocation, es_draws, es_sigma, gen,
                   draw_scale=draw_scale, draws=draws)
    dust.name = METHOD
    pop = {"log": f"log rule x{draw_scale}", "custom": f"custom {draws}"}.get(allocation, pop_size)
    print(f"[EvoMTL] {METHOD}: dims={sum(p.numel() for p in params)} | pop={pop} | chunk={chunk} | "
          f"dust_fraction={dust_fraction} | inner_steps={inner_steps} | optim={optim} lr={lr} | sigma={sigma}",
          flush=True)
    offsets, start = {}, 0
    for p in params:
        offsets[p], start = slice(start, start + p.numel()), start + p.numel()

    batches = trainer._sample_batches(train_dataloaders, num_batches)
    dust._capture(*batches[0])  # finds the layers, the groups and the draws
    per_iteration = sum(dust.steps.values())
    joint = round(float(dust_fraction) * int(num_iterations))
    layerwise = (int(num_iterations) - joint) // per_iteration if per_iteration else 0
    rows = joint + layerwise
    print(f"[EvoMTL] {METHOD}: {joint} DUST iterations, then {layerwise} RLW-DUST iterations of {per_iteration} "
          f"mini-batches ({joint + layerwise * per_iteration} of {num_iterations} mini-batches)", flush=True)

    tasks = list(trainer.task_name)
    metrics_csv = MetricsCSV(trainer, metrics_columns(
        tasks, ["sigma"], ["phase", "inner_steps", "total_inner_steps", "true_cos_direction"]))
    F = np.empty((0, len(tasks)))
    theta = trainer.theta_0.detach().cpu().float().clone()
    total = 0  # mini-batches so far: the W&B step
    try:
        for iteration in range(rows):
            t0 = time.perf_counter()
            diagnose = grad_cos_freq > 0 and iteration % int(grad_cos_freq) == 0
            if iteration:
                batches = trainer._sample_batches(train_dataloaders, num_batches)
            J = torch.zeros(start)
            J_true = torch.zeros(start)
            losses, taken, cosines = torch.zeros(len(tasks)), 0, {}
            if iteration < joint:  # DUST: every group at one snapshot, one step on all of them
                phase = "DUST"
                for inputs, gts in batches:
                    J_b, losses_b = dust.estimate(inputs, gts)
                    J += J_b[0].cpu() / len(batches)
                    losses += losses_b.cpu() / len(batches)
                if diagnose:
                    J_true = true_jacobian(trainer, nn.utils.parameters_to_vector(params).detach().cpu(),
                                           batches).sum(0)
                optimizer.zero_grad(set_to_none=True)
                for p in params:
                    p.grad = J[offsets[p]].view_as(p).to(p)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                taken = 1
            else:  # RLW-DUST: the groups from the last to the first, T_l inner steps each
                phase = "RLW-DUST"
                for name in dust.groups:
                    for t in range(dust.steps[name]):
                        if taken:
                            batches = trainer._sample_batches(train_dataloaders, num_batches)
                        if diagnose and t == 0:
                            exact = true_jacobian(trainer, nn.utils.parameters_to_vector(params).detach().cpu(),
                                                  batches).sum(0)
                        block, inner_losses = dust.inner_step(name, batches, optimizer)
                        losses += inner_losses
                        taken += 1
                        if t == 0:  # the group's first estimate, at the weights the iteration reached
                            for p in dust.groups[name]:
                                J[offsets[p]] = block[p].reshape(-1).cpu()
                                if diagnose:
                                    J_true[offsets[p]] = exact[offsets[p]]
                            if diagnose:
                                cosines[name] = _cosine(torch.cat([J[offsets[p]] for p in dust.groups[name]]),
                                                        torch.cat([J_true[offsets[p]] for p in dust.groups[name]]))
            theta = nn.utils.parameters_to_vector(params).detach().cpu().clone()
            previous, total = total, total + taken
            step = wandb_step_offset + max(total - 1, 0)
            if isinstance(evo_eval_freq, (set, frozenset)):
                run_eval = any(b in evo_eval_freq for b in range(previous, total))
            else:
                every = int(evo_eval_freq)
                run_eval = every > 0 and (total // every > previous // every or iteration in (0, rows - 1))
            F = (losses / max(taken, 1)).numpy()[None].astype(np.float32)  # mean pre-step loss over the iteration
            row: Dict[str, Any] = {"sigma": float(sigma), "phase": phase, "inner_steps": taken,
                                   "total_inner_steps": total}
            extras_wandb: Dict[str, Any] = {"evo/dust/direction_norm": float(J.norm()),
                                            "evo/dust/inner_steps": taken,
                                            "evo/dust/total_inner_steps": total,
                                            "evo/dust/phase_rlw": int(phase != "DUST")}
            progress = f"{phase} |d|={float(J.norm()):.4g} steps={taken}"
            if diagnose:
                dir_cos = _cosine(J, J_true)
                if cosines:
                    progress += " cos=[" + ", ".join(f"{n} {c:+.3f}" for n, c in cosines.items()) + "]"
                progress += f" dir_cos={dir_cos:+.4f}"
                row["true_cos_direction"] = dir_cos
                extras_wandb["evo/dust/true_cos/direction"] = dir_cos
                extras_wandb.update({f"evo/dust/true_cos/{n}": c for n, c in cosines.items()})
            extras_wandb["evo/dust/mean_norm"] = float((theta - trainer.theta_0.cpu()).norm())
            elapsed = time.perf_counter() - t0

            trainer._print_evo_progress(METHOD, iteration, rows, F, extras=progress, elapsed=elapsed,
                                        wandb_log_step=step)
            trainer._wandb_log_evo_front_metrics(iteration, F, step, extras=extras_wandb)
            train_batch_metrics, test_metrics = report_search_mean(
                trainer, theta, iteration, rows, F, F, batches, test_dataloaders, int(run_eval), step)
            metrics_csv.write(iteration, elapsed, tasks, F, train_batch_metrics, test_metrics, **row)
    finally:
        metrics_csv.close()
        dust.close()

    return es_result(trainer, METHOD, theta, theta.numpy()[None], F, hv_ref_point, sigma, aggregator=METHOD,
                     wandb_steps=total)
