"""Reverse DUST with its own activation estimator and training loop.

One outer iteration traverses parameter groups in reverse observed forward order. Each group
gets a fresh clean forward, estimates its activation-space DUST gradient using the current
downstream weights, and immediately updates its own parameters. Free parameters use two-sided
weight-space ES in groups owned by their modules. Rewards are per sample and use the summed
task loss.
"""

from __future__ import annotations

import copy
import math
import time
from typing import Any, Callable, Dict, Optional, Tuple

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

ALLOCATIONS = ("equal", "width", "log", "custom")
CONV_WEIGHT = {nn.Conv1d: torch.nn.grad.conv1d_weight, nn.Conv2d: torch.nn.grad.conv2d_weight,
               nn.Conv3d: torch.nn.grad.conv3d_weight}


class RDUST:
    """Activation perturbations and immediate reverse-order parameter updates.

    Groups follow the first completed call of each owning module in a clean forward.
    Repeated calls to one module form one layer. Distinct modules with tied parameters
    cannot receive independent updates and are rejected before an iteration starts.
    ``estimate`` provides fixed-snapshot estimates. Training uses ``step``.
    """

    name = "RDUST"

    def __init__(self, trainer: Any, pop_size: int, sigma: float, chunk: int, allocation: str, es_draws: int,
                 es_sigma: float, gen: torch.Generator, per_task: bool = False, draw_scale: float = 1.0,
                 draws: Optional[list] = None, antithetic: bool = False):
        if per_task:
            raise ValueError("RDUST uses the summed task loss (per_task=False).")
        if trainer.multi_input:
            raise NotImplementedError("RDUST needs one input batch shared by the tasks (multi_input=False).")
        if allocation not in ALLOCATIONS:
            raise ValueError(f"Unknown RDUST allocation {allocation!r}. Options are {ALLOCATIONS}.")
        if antithetic and int(chunk) % 2:
            raise ValueError(f"RDUST antithetic draws need an even chunk, got {chunk}.")
        self.trainer, self.model, self.gen = trainer, trainer.model, gen
        self.pop_size, self.sigma, self.chunk, self.allocation = int(pop_size), float(sigma), int(chunk), allocation
        self.antithetic = bool(antithetic)
        self.draw_scale = float(draw_scale)
        self.custom = [int(n) for n in draws] if draws else None
        if allocation == "custom" and (not self.custom or any(n < self.chunk or n % self.chunk for n in self.custom)):
            raise ValueError(f"RDUST custom allocation needs draws per layer in multiples of chunk={self.chunk}, "
                             f"got {draws}.")
        self.es_draws, self.es_sigma = int(es_draws), float(es_sigma)
        # The objective is the summed task loss at each sample.
        K = len(trainer.task_name)
        self.W = torch.ones(1, K).to(trainer.device)
        # per-sample losses: copies of the task losses without the batch reduction
        self.sample_loss = {}
        for task in trainer.task_name:
            fn = getattr(trainer.meter.losses[task], "loss_fn", None)
            if not hasattr(fn, "reduction"):
                raise ValueError(f"RDUST needs per-sample losses: the loss of task {task!r} has no loss_fn with a "
                                 "reduction.")
            self.sample_loss[task] = copy.copy(fn)
            self.sample_loss[task].reduction = "none"
        # candidate layers; those never called by the forward are dropped at the first clean pass
        self.layers = {n: m for n, m in self.model.named_modules()
                      if isinstance(m, (nn.Linear, nn.Embedding))
                      or (type(m) in CONV_WEIGHT and m.padding_mode == "zeros" and not isinstance(m.padding, str))}
        self.hooks = [m.register_forward_hook(self._hook(n)) for n, m in self.layers.items()]
        self.inputs: Dict[str, list] = {}   # layer -> input of every call (clean pass)
        self.shapes: Dict[str, list] = {}   # layer -> output shape of every call (clean pass)
        self.jitter: Dict[str, list] = {}   # layer -> jitter added to the output of every call (current draw)
        self.calls: Dict[str, int] = {}
        self.capturing = False
        self.draws: Optional[Dict[str, int]] = None
        self.free: list = []
        self.order: list = []
        self.groups: Dict[str, tuple] = {}
        # Include normalization affines and other free parameters in the observed execution order.
        self.hooks.extend(module.register_forward_hook(self._order_hook(name))
                          for name, module in self.model.named_modules()
                          if any(True for _ in module.parameters(recurse=False)))

    def _hook(self, name: str):
        def hook(module, args, output):
            call = self.calls[name] = self.calls.get(name, -1) + 1
            if self.capturing:
                self.inputs.setdefault(name, []).append(args[0])
                self.shapes.setdefault(name, []).append(output.shape)
            if name in self.jitter:
                return output + self.jitter[name][call]
        return hook

    def _order_hook(self, name: str):
        def hook(module, args, output):
            if self.capturing and name not in self.order:
                self.order.append(name)
        return hook

    def close(self) -> None:
        for hook in self.hooks:
            hook.remove()

    def losses(self, inputs: torch.Tensor, gts: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Per-sample task losses ``[K, B]`` of one forward pass (the elementwise loss averaged over all but the batch
        axis, so their mean over samples is the batch loss)."""
        self.calls = {}
        preds = self.trainer.process_preds(self.model(inputs))
        rows = [self.sample_loss[task](preds[task], gts[task]) for task in self.trainer.task_name]
        return torch.stack([r.reshape(len(r), -1).mean(1) for r in rows])

    def _setup(self) -> None:
        # Check every direct owner before filtering free parameters: a layer can also share its weight
        # with a non-layer module. Reject before setting draws, so a failed setup cannot be reused partially.
        owners = {}
        for name, module in self.model.named_modules():
            for p in module.parameters(recurse=False):
                if id(p) in owners:
                    raise ValueError(f"RDUST does not support tied parameters across distinct modules "
                                     f"({owners[id(p)]!r}, {name!r}); reuse one module for a shared layer.")
                owners[id(p)] = name
        self.layers = {n: m for n, m in self.layers.items() if n in self.inputs}
        covered = {id(p) for m in self.layers.values() for p in m.parameters(recurse=False)}
        self.free = [p for p in self.model.parameters() if id(p) not in covered]
        widths = {n: sum(math.prod(s[1:]) for s in self.shapes[n]) for n in self.layers}  # output size per sample
        if self.allocation == "custom":
            if len(self.custom) != len(widths):
                raise ValueError(f"RDUST custom allocation: {len(self.custom)} draw counts for {len(widths)} layers "
                                 f"({', '.join(widths)}).")
            order = list(self.inputs)  # layers in the order the forward pass calls them
            self.draws = {n: dict(zip(order, self.custom))[n] for n in widths}
        elif self.allocation == "log":  # CMA-ES population rule per layer, rounded up to whole chunks
            self.draws = {n: self.chunk * math.ceil(self.draw_scale * (4 + math.floor(3 * math.log(d))) / self.chunk)
                          for n, d in widths.items()}
        else:  # one chunk per layer, the other chunks in proportion to the share (largest remainder: pop_size draws)
            share = {n: (d if self.allocation == "width" else 1.0) for n, d in widths.items()}
            units = self.pop_size // self.chunk - len(share)
            if units < 0:
                raise ValueError(f"RDUST needs pop_size >= chunk * layers = {self.chunk * len(share)}, "
                                 f"got {self.pop_size}.")
            quota = {n: units * s / sum(share.values()) for n, s in share.items()}
            extra = {n: int(q) for n, q in quota.items()}
            for n in sorted(quota, key=lambda n: quota[n] - extra[n], reverse=True)[:units - sum(extra.values())]:
                extra[n] += 1
            self.draws = {n: self.chunk * (1 + extra[n]) for n in share}
        total = sum(self.draws.values())
        print(f"[EvoMTL] {self.name}: {total} direct draws per generation"
              + (f" ({total // 2} antithetic pairs)" if self.antithetic else "") + f" ({self.allocation}): "
              + ", ".join(f"{n}={d}" for n, d in self.draws.items())
              + (f" | weight-space ES on {sum(p.numel() for p in self.free)} other weights: "
                 f"{self.es_draws} pairs per free group"
                 if self.free else ""), flush=True)
        free = {id(p) for p in self.free}
        candidates = {}
        for name, module in self.model.named_modules():
            params = tuple(p for p in module.parameters(recurse=False) if name in self.layers or id(p) in free)
            if not params:
                continue
            candidates[name] = params
        # A free parameter's owner may never be called (e.g. a scalar accessed directly by its parent).
        # Put unobserved owners before observed ones, so their fallback updates happen last in the iteration.
        order = [n for n in candidates if n not in self.order] + [n for n in self.order if n in candidates]
        self.groups = {n: candidates[n] for n in order}

    def _sample_noise(self, name: str, samples: int) -> list:
        return [torch.randn(samples, *s[1:], generator=self.gen, device=self.gen.device) for s in self.shapes[name]]

    def _finish_layer(self, name: str, errors: list) -> list:
        return errors

    def _capture(self, inputs: torch.Tensor, gts: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Refresh clean inputs, output shapes and per-sample task losses at the current weights."""
        self.order = []
        self.model.eval()
        self.inputs, self.shapes, self.capturing = {}, {}, True
        try:
            task_losses = self.losses(inputs, gts)  # [K, B]
        finally:
            self.capturing = False
        if self.draws is None:
            self._setup()
        return task_losses

    def _estimate_layer(self, name: str, clean: torch.Tensor, x: torch.Tensor,
                       y: Dict[str, torch.Tensor], grads: Dict[nn.Parameter, torch.Tensor]) -> None:
        """Estimate one layer's gradient, using a clean baseline and chunk-repeated data at the same weights."""
        K, B = clean.shape
        n = self.chunk
        m = n // 2 if self.antithetic else n
        draws = self.draws[name]
        count = draws // 2 if self.antithetic else draws
        errors = [torch.zeros(K, *s, device=clean.device) for s in self.shapes[name]]
        for _ in range(draws // n):
            noise = self._sample_noise(name, m * B)
            # Pair k of sample b: +a at k*B+b and -a at (m+k)*B+b.
            self.jitter = {name: [self.sigma * (torch.cat([a, -a]) if self.antithetic else a) for a in noise]}
            try:
                loss = (self.W @ self.losses(x, y)).view(K, n, B)
            finally:
                self.jitter = {}
            if self.antithetic:
                reward = (loss[:, m:] - loss[:, :m]) / 2
            else:
                reward = clean.unsqueeze(1) - loss
                if n > 1:
                    reward = reward - reward.mean(1, keepdim=True)
            for e, a in zip(errors, noise):
                e -= torch.einsum("knb,nb...->kb...", reward, a.view(m, B, *a.shape[1:])) / (self.sigma * count)
        self._reassemble(self.layers[name], self._finish_layer(name, errors), self.inputs[name], grads)

    @torch.no_grad()
    def estimate(self, inputs: torch.Tensor, gts: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gradient estimates ``J`` (``[M, D]``, order of ``model.parameters()``) of the M objectives (mean over the
        batch), and the task losses (``[K]``), on one mini-batch."""
        task_losses = self._capture(inputs, gts)
        clean = self.W @ task_losses  # [M, B]
        K, B = clean.shape  # K: number of objectives from here on
        n = self.chunk
        x = inputs.repeat(n, *(1,) * (inputs.dim() - 1))
        y = {t: g.repeat(n, *(1,) * (g.dim() - 1)) for t, g in gts.items()}

        grads: Dict[nn.Parameter, torch.Tensor] = {}
        for name in self.draws:
            self._estimate_layer(name, clean, x, y, grads)
        if self.free:
            self._weight_space(inputs, gts, grads, K)
        J = torch.cat([grads[p].reshape(K, -1) if p in grads else torch.zeros(K, p.numel(), device=clean.device)
                       for p in self.model.parameters()], dim=1)
        return J / B, task_losses.mean(1)

    @staticmethod
    def _reassemble(m: nn.Module, errors: list, inputs: list, grads: Dict[nn.Parameter, torch.Tensor]) -> None:
        """Exact local linear map of the layer ``m``: its weight (and bias) gradients from the output errors ``e``
        (``[K, B, ...]``) and inputs of every call, added to ``grads``."""
        for e, x in zip(errors, inputs):
            K = e.shape[0]
            if isinstance(m, nn.Embedding):
                w = torch.zeros(K, *m.weight.shape, device=e.device)
                w.index_add_(1, x.reshape(-1), e.reshape(K, -1, m.embedding_dim))
                if m.padding_idx is not None:
                    w[:, m.padding_idx] = 0.0
                b = None
            elif isinstance(m, nn.Linear):
                e2 = e.reshape(K, -1, m.out_features)
                w = torch.einsum("kno,ni->koi", e2, x.reshape(-1, m.in_features))
                b = e2.sum(1)
            else:  # convolution: weight gradient = correlation of the input with the output error
                w = torch.stack([CONV_WEIGHT[type(m)](x, m.weight.shape, e_k, m.stride, m.padding, m.dilation,
                                                      m.groups) for e_k in e])
                b = e.sum(dim=(1, *range(3, e.dim())))
            grads[m.weight] = grads.get(m.weight, 0) + w
            bias = getattr(m, "bias", None)
            if bias is not None:
                grads[bias] = grads.get(bias, 0) + b

    def _weight_space(self, inputs: torch.Tensor, gts: Dict[str, torch.Tensor],
                      grads: Dict[nn.Parameter, torch.Tensor], K: int, params: Optional[list] = None) -> None:
        """Two-sided weight-space ES on the parameters outside the weight layers, rewarded by each objective summed over the
        samples (divided by the batch size in :meth:`estimate`, like the layer gradients)."""
        params = self.free if params is None else params
        clean = [p.detach().clone() for p in params]
        est = [torch.zeros(K, *p.shape, device=p.device) for p in params]
        try:
            for _ in range(self.es_draws):
                a = [torch.randn(p.shape, generator=self.gen, device=self.gen.device) for p in params]
                losses = []
                for sign in (1.0, -1.0):
                    for p, c, d in zip(params, clean, a):
                        p.copy_(c + sign * self.es_sigma * d)
                    losses.append((self.W @ self.losses(inputs, gts)).sum(1))  # [K objectives]
                g = (losses[0] - losses[1]) / (2 * self.es_sigma * self.es_draws)
                for e, d in zip(est, a):
                    e += g.view(K, *(1,) * d.dim()) * d
        finally:
            for p, c in zip(params, clean):
                p.copy_(c)
        for p, e in zip(params, est):
            grads[p] = e

    @torch.no_grad()
    def step(self, batches: list, optimizer: torch.optim.Optimizer, diagnose: bool = False
              ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """One reverse pass over the groups (one iteration) on fixed minibatches, updating only the current group after its estimates.

        Returns ``(J, losses, J_true)`` on CPU: concatenated estimated blocks ``[1, D]``, pre-iteration task losses,
        and, with ``diagnose``, exact per-task blocks measured at each matching pre-update state ``[K, D]``.
        Neither concatenation is a gradient at one common parameter snapshot. Multiple batches are averaged
        within each group before its one optimizer step; the same batches are reused for the entire iteration.
        The optimizer must own the model parameters, not a flattened parameter vector.
        """
        if not batches:
            raise ValueError("RDUST needs at least one minibatch per iteration.")
        if self.W.shape[0] != 1:
            raise ValueError("RDUST uses the summed task loss (per_task=False).")
        params = list(self.model.parameters())
        optimizer_ids = {id(p) for group in optimizer.param_groups for p in group['params']}
        if optimizer_ids != {id(p) for p in params}:
            raise ValueError("RDUST's optimizer must own exactly the model parameters.")
        initial = self._capture(*batches[0])
        offsets = {}
        start = 0
        for p in params:
            offsets[p] = slice(start, start + p.numel())
            start += p.numel()
        J = torch.zeros(1, start)
        reference = torch.zeros(len(self.trainer.task_name), start) if diagnose else None
        losses = torch.zeros(len(self.trainer.task_name))
        try:
            for group_index, name in enumerate(reversed(self.groups)):
                active = self.groups[name]
                block = {p: torch.zeros_like(p) for p in active}
                for batch_index, (inputs, gts) in enumerate(batches):
                    task_losses = initial if group_index == batch_index == 0 else self._capture(inputs, gts)
                    if group_index == 0:
                        losses += task_losses.mean(1).cpu() / len(batches)
                    grads: Dict[nn.Parameter, torch.Tensor] = {}
                    if name in self.layers:
                        n = self.chunk
                        x = inputs.repeat(n, *(1,) * (inputs.dim() - 1))
                        y = {t: g.repeat(n, *(1,) * (g.dim() - 1)) for t, g in gts.items()}
                        self._estimate_layer(name, self.W @ task_losses, x, y, grads)
                    else:
                        self._weight_space(inputs, gts, grads, 1, params=list(active))
                    for p in active:
                        if p in grads:
                            block[p] += grads[p][0] / (len(inputs) * len(batches))
                if diagnose:
                    theta = nn.utils.parameters_to_vector(params).detach().cpu().clone()
                    exact = true_jacobian(self.trainer, theta, batches)
                    for p in active:
                        reference[:, offsets[p]] = exact[:, offsets[p]]
                # None is essential: zero gradients would still advance momentum/Adam in other groups.
                optimizer.zero_grad(set_to_none=True)
                for p in active:
                    p.grad = block[p]
                    J[0, offsets[p]] = block[p].reshape(-1).cpu()
                optimizer.step()
        finally:
            optimizer.zero_grad(set_to_none=True)
        return J, losses, reference


def run_rdust(
    trainer: Any,
    train_dataloaders: Any,
    num_iterations: int,
    pop_size: int,
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
    aggregate: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
    antithetic: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Train with one complete reverse pass over the groups per iteration.

    Draw allocation, perturbations and optimizer settings match DUST. Each group's gradient
    is estimated at its current pre-update weights and applied immediately by an optimizer
    on the model parameters. Diagnostics compare estimated and exact blocks at the matching
    pre-update states. The concatenated blocks span the snapshots of the iteration.
    """
    if aggregate is not None:
        raise ValueError("RDUST uses the summed task loss. Multi-objective aggregation is unsupported.")
    if trainer.theta_0 is None:
        raise RuntimeError("prepare_evolution() first.")
    if int(chunk) < 1 or (allocation in ("equal", "width") and int(pop_size) < int(chunk)):
        raise ValueError(f"RDUST needs 1 <= chunk <= pop_size, got chunk={chunk}, pop_size={pop_size}.")
    require_shuffled_loader(train_dataloaders, "RDUST")

    configure_moea_random_seed(seed)
    gen = torch.Generator(trainer.device).manual_seed(int(seed))
    theta = trainer.theta_0.detach().cpu().float().clone()
    trainer.set_theta(theta)
    optim = str(optim).strip().lower()
    optimizer = make_es_optimizer(list(trainer.model.parameters()), optim, lr, adam_betas, adam_eps,
                                  rmsprop_alpha, rmsprop_eps, sgd_momentum)
    dust = RDUST(trainer, pop_size, sigma, chunk, allocation, es_draws, es_sigma, gen,
                 draw_scale=draw_scale, draws=draws, antithetic=antithetic)
    pop = {"log": f"log rule x{draw_scale}", "custom": f"custom {draws}"}.get(allocation, pop_size)
    print(f"[EvoMTL] RDUST: dims={theta.numel()} | pop={pop} | chunk={chunk} | optim={optim} lr={lr} | "
          f"sigma={sigma}", flush=True)

    tasks = list(trainer.task_name)
    metrics_csv = MetricsCSV(trainer, metrics_columns(
        tasks, ["sigma"], [f"weight_{task}" for task in tasks] + [f"true_cos_{task}" for task in tasks]
        + ["true_cos_direction"]))
    F = np.empty((0, len(tasks)))
    try:
        for iteration in range(int(num_iterations)):
            t0 = time.perf_counter()
            step = wandb_step_offset + iteration
            batches = trainer._sample_batches(train_dataloaders, num_batches)
            trainer.set_theta(theta)
            diagnose = grad_cos_freq > 0 and iteration % int(grad_cos_freq) == 0
            J, losses, reference = dust.step(batches, optimizer, diagnose=diagnose)
            theta = nn.utils.parameters_to_vector(trainer.model.parameters()).detach().cpu().clone()
            direction = J[0]
            F = losses.numpy()[None].astype(np.float32)
            row: Dict[str, Any] = {"sigma": float(sigma)}
            direction_norm = float(direction.norm())
            extras_wandb: Dict[str, Any] = {"evo/dust/direction_norm": direction_norm}
            progress = f"|d|={direction_norm:.4g}"
            if diagnose:
                direction_cosine = _cosine(direction, reference.sum(0))
                progress += f" dir_cos={direction_cosine:+.4f}"
                row["true_cos_direction"] = direction_cosine
                extras_wandb["evo/dust/true_cos/direction"] = direction_cosine
            extras_wandb["evo/dust/mean_norm"] = float((theta - trainer.theta_0.cpu()).norm())
            elapsed = time.perf_counter() - t0

            trainer._print_evo_progress("RDUST", iteration, num_iterations, F, extras=progress, elapsed=elapsed,
                                        wandb_log_step=step)
            trainer._wandb_log_evo_front_metrics(iteration, F, step, extras=extras_wandb)
            train_batch_metrics, test_metrics = report_search_mean(
                trainer, theta, iteration, num_iterations, F, F, batches, test_dataloaders, evo_eval_freq, step)
            metrics_csv.write(iteration, elapsed, tasks, F, train_batch_metrics, test_metrics, **row)
    finally:
        metrics_csv.close()
        dust.close()

    return es_result(trainer, "RDUST", theta, theta.numpy()[None], F, hv_ref_point, sigma, aggregator="RDUST")


