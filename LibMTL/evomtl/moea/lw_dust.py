"""LW-DUST: layer-wise DUST, block-coordinate descent over the layers from the first to the last.

DUST (:mod:`.dust`) estimates the gradient of every layer on one mini-batch at one weight snapshot and updates all
layers together. LW-DUST visits the parameter groups one at a time, in the order the forward pass calls them (first
layer to last), and gives group ``l`` its own number ``T_l`` of inner steps before moving on to the next. A group is one
site (a module with a weight matrix: its weight and bias) or one module owning parameters outside the sites (a
normalization affine). One inner step on group ``l``:

1. fresh mini-batches (``num_batches``) and a clean forward at the current weights;
2. the group's gradient estimate, as in DUST: for a site, its ``draws_l`` output jitters (``allocation``) and the exact
   local linear map of the reward-weighted noise and the cached input; for any other group, the two-sided weight-space
   ES (``es_draws`` pairs) on its parameters alone;
3. one optimizer step on the group's parameters; the rest of the network is unchanged.

The layers before ``l`` do not change during its inner steps, so its input distribution is fixed while it trains; the
layers after ``l`` were last trained against its previous output. One iteration (one pass over the groups) takes
``sum_l T_l`` optimizer steps on ``sum_l T_l`` fresh mini-batch draws (times ``num_batches``), and
``sum_l T_l draws_l`` jitter evaluations: with every ``T_l = 1``, an iteration spends the jitter evaluations of one DUST
generation over ``L`` mini-batches.

The optimizer runs over all the model's parameters; an inner step sets the gradients of the active group only (the
others ``None``), so the momentum and Adam statistics of a group advance only on its own steps. LW-DUST is
single-objective: the rewards come from the summed task loss. Distinct modules sharing a parameter would split its
gradient over two groups, so they are rejected; one module called several times (a shared decoder) is one site.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .common import configure_moea_random_seed
from .dust import DUST
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


class LWDUST(DUST):
    """DUST's per-site estimator, applied to one parameter group at a time with ``T_l`` inner steps per group."""

    name = "LW-DUST"
    reverse = False  # RLW-DUST (:mod:`.rlw_dust`): visit the groups from the last to the first

    def __init__(self, trainer: Any, inner_steps: List[int], *args: Any, **kwargs: Any):
        super().__init__(trainer, *args, **kwargs)
        self.inner_steps = [int(t) for t in inner_steps]
        if not self.inner_steps or any(t < 0 for t in self.inner_steps):
            raise ValueError(f"{self.name} needs non-negative inner steps, got {inner_steps}.")
        self.order: list = []                 # parameter-owning modules in the order the forward calls them
        self.groups: Dict[str, list] = {}     # group -> its parameters, in update order
        self.steps: Dict[str, int] = {}       # group -> inner steps T_l
        self.hooks += [m.register_forward_hook(self._order_hook(n)) for n, m in self.model.named_modules()
                       if any(True for _ in m.parameters(recurse=False))]

    def _order_hook(self, name: str):
        def hook(module, args, output):
            if self.capturing and name not in self.order:
                self.order.append(name)
        return hook

    def _setup(self) -> None:
        owners: Dict[int, str] = {}
        for name, module in self.model.named_modules():
            for p in module.parameters(recurse=False):
                if id(p) in owners:
                    raise ValueError(f"{self.name}: modules {owners[id(p)]!r} and {name!r} share a parameter.")
                owners[id(p)] = name
        super()._setup()
        free = {id(p) for p in self.free}
        groups = {n: [p for p in m.parameters(recurse=False) if n in self.sites or id(p) in free]
                  for n, m in self.model.named_modules()}
        groups = {n: ps for n, ps in groups.items() if ps}
        # modules never called by the forward (a scalar read directly by its parent) go last
        order = [n for n in self.order if n in groups] + [n for n in groups if n not in self.order]
        steps = self.inner_steps * len(order) if len(self.inner_steps) == 1 else self.inner_steps
        if len(steps) != len(order):
            raise ValueError(f"{self.name}: {len(self.inner_steps)} inner step counts for {len(order)} groups "
                             f"({', '.join(order)}); give one count, or one per group in forward order.")
        steps = dict(zip(order, steps))  # counts are always given in forward order
        if self.reverse:
            order = order[::-1]
        self.groups = {n: groups[n] for n in order}
        self.steps = {n: steps[n] for n in order}
        print(f"[EvoMTL] {self.name}: {sum(steps.values())} inner steps per iteration, "
              f"{'last to first' if self.reverse else 'first to last'}: "
              + ", ".join(f"{n} x{t}" for n, t in self.steps.items()), flush=True)

    @torch.no_grad()
    def inner_step(self, name: str, batches: list, optimizer: torch.optim.Optimizer
                   ) -> Tuple[Dict[nn.Parameter, torch.Tensor], torch.Tensor]:
        """Estimate group ``name``'s gradient of the mean summed loss on ``batches`` (averaged over them) and take one
        optimizer step on that group. Returns the estimate per parameter and the pre-step task losses ``[K]``."""
        active = self.groups[name]
        block = {p: torch.zeros_like(p) for p in active}
        losses = torch.zeros(len(self.trainer.task_name))
        for inputs, gts in batches:
            task_losses = self._capture(inputs, gts)  # [K, B] at the current weights
            losses += task_losses.mean(1).cpu() / len(batches)
            grads: Dict[nn.Parameter, torch.Tensor] = {}
            if name in self.sites:
                n = self.chunk
                x = inputs.repeat(n, *(1,) * (inputs.dim() - 1))
                y = {t: g.repeat(n, *(1,) * (g.dim() - 1)) for t, g in gts.items()}
                self._estimate_site(name, self.W @ task_losses, x, y, grads)
            else:
                self._weight_space(inputs, gts, grads, 1, params=active)
            for p in active:
                block[p] += grads[p][0] / (len(inputs) * len(batches))
        optimizer.zero_grad(set_to_none=True)  # None, not zero: the other groups' optimizer state must not advance
        for p in active:
            p.grad = block[p]
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return block, losses


def run_lw_dust(
    trainer: Any,
    train_dataloaders: Any,
    num_iterations: int,
    pop_size: int,
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
    evo_eval_freq: int = 1,
    grad_cos_freq: int = 1,
    method: str = "LW-DUST",
    _estimator: Callable[..., LWDUST] = LWDUST,
    **kwargs: Any,
) -> Dict[str, Any]:
    """LW-DUST training from ``theta_0`` (see module docstring); the trainer ends at the final weights.

    Parameters
    ----------
    num_iterations : int
        Iterations; one iteration is one pass over the groups (first to last; last to first for RLW-DUST): ``sum_l T_l`` optimizer steps on fresh
        mini-batches.
    inner_steps : list of int, optional
        ``T_l``: one count for every group, or one per group in forward order (``0`` freezes a group). Default ``[1]``.
    pop_size, allocation, draw_scale, draws, sigma, chunk, es_draws, es_sigma
        DUST's per-site jitter draws and weight-space ES (:func:`~.dust.run_dust`), used at every inner step.
    lr, optim, adam_betas, adam_eps, rmsprop_alpha, rmsprop_eps, sgd_momentum
        One optimizer on the model's parameters (:func:`~.es_common.make_es_optimizer`).
    method, _estimator
        Logged name and estimator class (RLW-DUST: :class:`~.rlw_dust.RLWDUST`, the reverse order).
    wandb_step_offset : int
        W&B step of the first inner step; an iteration is logged at the step of its last inner step, so the W&B
        axis counts optimizer steps (mini-batch draws), as DUST's counts generations.
    grad_cos_freq : int
        Every this many iterations: cosine of every group's estimate with its backprop gradient at the first inner step
        (same mini-batches, same weights), and of the concatenated blocks. ``0`` disables. Diagnostic only.
    """
    if kwargs.get("aggregate") is not None:
        raise ValueError(f"{method} uses the summed task loss; multi-objective aggregation is unsupported.")
    if trainer.theta_0 is None:
        raise RuntimeError("prepare_evolution() first.")
    if int(chunk) < 1 or (allocation in ("equal", "width") and int(pop_size) < int(chunk)):
        raise ValueError(f"{method} needs 1 <= chunk <= pop_size, got chunk={chunk}, pop_size={pop_size}.")
    require_shuffled_loader(train_dataloaders, method)
    configure_moea_random_seed(seed)
    gen = torch.Generator(trainer.device).manual_seed(int(seed))
    trainer.set_theta(trainer.theta_0.detach().cpu().float())
    params = list(trainer.model.parameters())
    optim = str(optim).strip().lower()
    optimizer = make_es_optimizer(params, optim, lr, adam_betas, adam_eps, rmsprop_alpha, rmsprop_eps, sgd_momentum)
    dust = _estimator(trainer, inner_steps or [1], pop_size, sigma, chunk, allocation, es_draws, es_sigma, gen,
                  draw_scale=draw_scale, draws=draws)
    pop = {"log": f"log rule x{draw_scale}", "custom": f"custom {draws}"}.get(allocation, pop_size)
    print(f"[EvoMTL] {method}: dims={sum(p.numel() for p in params)} | pop={pop} | chunk={chunk} | "
          f"inner_steps={inner_steps or [1]} | optim={optim} lr={lr} | sigma={sigma}", flush=True)
    offsets, start = {}, 0
    for p in params:
        offsets[p], start = slice(start, start + p.numel()), start + p.numel()

    tasks = list(trainer.task_name)
    metrics_csv = MetricsCSV(trainer, metrics_columns(tasks, ["sigma"], ["inner_steps", "total_inner_steps", "true_cos_direction"]))
    F = np.empty((0, len(tasks)))
    theta = trainer.theta_0.detach().cpu().float().clone()
    total = 0  # inner steps so far: the W&B step, so runs with different T share one axis
    try:
        for iteration in range(int(num_iterations)):
            t0 = time.perf_counter()
            diagnose = grad_cos_freq > 0 and iteration % int(grad_cos_freq) == 0
            batches = trainer._sample_batches(train_dataloaders, num_batches)
            if dust.draws is None:
                dust._capture(*batches[0])  # finds the sites, the groups and the draws
            J = torch.zeros(start)
            J_true = torch.zeros(start)
            losses, taken, cosines = torch.zeros(len(tasks)), 0, {}
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
            total += taken
            step = wandb_step_offset + max(total - 1, 0)
            F = (losses / max(taken, 1)).numpy()[None].astype(np.float32)  # mean pre-step loss over the iteration
            row: Dict[str, Any] = {"sigma": float(sigma), "inner_steps": taken, "total_inner_steps": total}
            extras_wandb: Dict[str, Any] = {"evo/dust/direction_norm": float(J.norm()),
                                            "evo/dust/inner_steps": taken,
                                            "evo/dust/total_inner_steps": total}
            progress = f"|d|={float(J.norm()):.4g} steps={taken}"
            if diagnose:
                dir_cos = _cosine(J, J_true)
                progress += " cos=[" + ", ".join(f"{n} {c:+.3f}" for n, c in cosines.items()) + f"] dir_cos={dir_cos:+.4f}"
                row["true_cos_direction"] = dir_cos
                extras_wandb["evo/dust/true_cos/direction"] = dir_cos
                extras_wandb.update({f"evo/dust/true_cos/{n}": c for n, c in cosines.items()})
            extras_wandb["evo/dust/mean_norm"] = float((theta - trainer.theta_0.cpu()).norm())
            elapsed = time.perf_counter() - t0

            trainer._print_evo_progress(method, iteration, num_iterations, F, extras=progress, elapsed=elapsed,
                                        wandb_log_step=step)
            trainer._wandb_log_evo_front_metrics(iteration, F, step, extras=extras_wandb)
            train_batch_metrics, test_metrics = report_search_mean(
                trainer, theta, iteration, num_iterations, F, F, batches, test_dataloaders, evo_eval_freq, step)
            metrics_csv.write(iteration, elapsed, tasks, F, train_batch_metrics, test_metrics, **row)
    finally:
        metrics_csv.close()
        dust.close()

    return es_result(trainer, method, theta, theta.numpy()[None], F, hv_ref_point, sigma, aggregator=method,
                     wandb_steps=total)
