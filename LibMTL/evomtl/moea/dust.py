"""DUST on the network weights: forward-only gradient estimates from per-sample activity perturbation.

DUST (Q Labs, 2026) trains a network with forward evaluations only, no backward pass. Every layer with a weight
matrix (Linear, convolution, embedding) is a *site*. One generation, on one mini-batch:

1. A clean forward caches every site's input and the per-sample loss ``L[b] = sum_k L_k[b]``, the equal-weight sum
   of the task losses (the scalarized loss of OpenES).
2. Direct draws jitter one site's output, ``y + sigma a`` with ``a ~ N(0, I)`` drawn independently at every sample,
   rerun the forward and reward each sample's jitter with that sample's loss reduction
   ``r[n, b] = L[b] - L(y + sigma a_n)[b]``, centered over the ``chunk`` draws of one forward pass (the batch
   repeated ``chunk`` times). The reward-weighted noise estimates the error at the site's output:
   ``e[b] = -mean_n(r[n, b] a_n[b]) / sigma ~ dL[b] / dy[b]``.
3. The site's weight gradient is the exact local linear map of ``e`` and the cached input, as in backprop:
   ``sum_b e[b] x[b]^T`` for a Linear layer (bias: ``sum_b e[b]``), the correlation of ``e`` with ``x`` for a
   convolution, a scatter-add for an embedding. Divided by the batch size, it estimates the gradient of the mean loss.
4. Parameters outside the sites (normalization affines, free scalars) get two-sided weight-space ES on the loss
   summed over the samples, as the paper does for its residual mixing scalars.
5. One optimizer step on the weights ``theta`` along the estimate.

DUST is single-objective, as in the paper. The multi-objective variants reward every task loss separately instead:
one gradient estimate per task, the rows ``g_k`` of ``J`` (``[K, D]``), combined by an MTL aggregator into the
direction (``aggregate`` of :func:`run_dust`): their sum (MO-Sum-DUST, :mod:`.mo_sum_dust`; equal to DUST up to
rounding), UPGrad (MO-UPGrad-DUST, :mod:`.mo_upgrad_dust`) or MGDA (MO-MGDA-DUST, :mod:`.mo_mgda_dust`).
A-DUST (:mod:`.a_dust`, ``antithetic``) replaces the independent draws and the chunk centering of step 2 by
antithetic pairs ``y +- sigma a`` rewarded by half their loss difference; the rest is unchanged.
CADUST (:mod:`.cadust`) learns a low-rank covariance for the draws and corrects the error estimate for that covariance.

Each draw is one forward pass of the batch, as one member of an OpenES population of ``pop_size``; the jitter is as
large as the site's activations (for a convolution, far more numbers than its weights), so drawing it adds to the
cost. The ``pop_size`` draws are split over the sites equally (``allocation="equal"``) or in proportion to their output
size per sample (``"width"``), in multiples of ``chunk``. With ``"log"`` each site gets its own count from its output
size ``d`` per sample, the CMA-ES default population ``draw_scale * (4 + floor(3 ln d))`` rounded up to whole chunks,
and ``pop_size`` is unused. With ``"custom"`` the counts are given per site (``draws``, multiples of ``chunk``), in the
order the forward pass calls the sites.

Differences from the transformer recipe of the paper. There, a jitter is credited with the loss change at its own
token, plus a decayed credit over later tokens for the attention internals; here the credit is per sample, which is
exact because the samples of an eval-mode forward do not interact. The cached-logit head draws and the per-layer
tuned draw counts are specific to that transformer and are not used. A module called several times per forward (a
decoder shared by the tasks) is one site: its calls are jittered together and its gradient sums over them.

References:
- DUST: https://qlabs.sh/research/dust, https://github.com/qlabs-eng/dust
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


class DUST:
    """Forward hooks capture each site's input during the clean pass and jitter its output during a draw.

    :meth:`step` estimates gradients and updates the weights. :meth:`estimate` returns the gradient estimate of the
    summed loss, or one estimate per task with ``per_task``.
    """

    name = "DUST"

    def __init__(self, trainer: Any, pop_size: int, sigma: float, chunk: int, allocation: str, es_draws: int,
                 es_sigma: float, gen: torch.Generator, per_task: bool = False, draw_scale: float = 1.0,
                 draws: Optional[list] = None, antithetic: bool = False):
        if trainer.multi_input:
            raise NotImplementedError("DUST needs one input batch shared by the tasks (multi_input=False).")
        if allocation not in ALLOCATIONS:
            raise ValueError(f"Unknown DUST allocation {allocation!r}; options: {ALLOCATIONS}")
        if antithetic and int(chunk) % 2:
            raise ValueError(f"A-DUST needs an even chunk (chunk / 2 antithetic pairs per forward pass), got {chunk}.")
        self.trainer, self.model, self.gen = trainer, trainer.model, gen
        self.pop_size, self.sigma, self.chunk, self.allocation = int(pop_size), float(sigma), int(chunk), allocation
        self.antithetic = bool(antithetic)
        self.draw_scale = float(draw_scale)
        self.custom = [int(n) for n in draws] if draws else None
        if allocation == "custom" and (not self.custom or any(n < self.chunk or n % self.chunk for n in self.custom)):
            raise ValueError(f"DUST custom allocation needs draws per site in multiples of chunk={self.chunk}, "
                             f"got {draws}.")
        self.es_draws, self.es_sigma = int(es_draws), float(es_sigma)
        # objectives = W @ task losses: their sum (DUST) or each task loss (MO-Sum-DUST)
        K = len(trainer.task_name)
        self.W = (torch.eye(K) if per_task else torch.ones(1, K)).to(trainer.device)
        # per-sample losses: copies of the task losses without the batch reduction
        self.sample_loss = {}
        for task in trainer.task_name:
            fn = getattr(trainer.meter.losses[task], "loss_fn", None)
            if not hasattr(fn, "reduction"):
                raise ValueError(f"DUST needs per-sample losses: the loss of task {task!r} has no loss_fn with a "
                                 "reduction.")
            self.sample_loss[task] = copy.copy(fn)
            self.sample_loss[task].reduction = "none"
        # candidate sites; those never called by the forward are dropped at the first clean pass
        self.sites = {n: m for n, m in self.model.named_modules()
                      if isinstance(m, (nn.Linear, nn.Embedding))
                      or (type(m) in CONV_WEIGHT and m.padding_mode == "zeros" and not isinstance(m.padding, str))}
        self.hooks = [m.register_forward_hook(self._hook(n)) for n, m in self.sites.items()]
        self.inputs: Dict[str, list] = {}   # site -> input of every call (clean pass)
        self.shapes: Dict[str, list] = {}   # site -> output shape of every call (clean pass)
        self.jitter: Dict[str, list] = {}   # site -> jitter added to the output of every call (current draw)
        self.calls: Dict[str, int] = {}
        self.capturing = False
        self.draws: Optional[Dict[str, int]] = None
        self.free: list = []

    def _hook(self, name: str):
        def hook(module, args, output):
            call = self.calls[name] = self.calls.get(name, -1) + 1
            if self.capturing:
                self.inputs.setdefault(name, []).append(args[0])
                self.shapes.setdefault(name, []).append(output.shape)
            if name in self.jitter:
                return output + self.jitter[name][call]
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
        """After the first clean pass: keep the sites the forward calls, collect the parameters outside them and split
        the draws over the sites."""
        self.sites = {n: m for n, m in self.sites.items() if n in self.inputs}
        covered = {id(p) for m in self.sites.values() for p in m.parameters(recurse=False)}
        self.free = [p for p in self.model.parameters() if id(p) not in covered]
        widths = {n: sum(math.prod(s[1:]) for s in self.shapes[n]) for n in self.sites}  # output size per sample
        self.draws = self._allocate(widths, list(self.inputs))  # sites in the order the forward pass calls them
        total = sum(self.draws.values())
        print(f"[EvoMTL] {self.name}: {total} direct draws per generation"
              + (f" ({total // 2} antithetic pairs)" if self.antithetic else "") + f" ({self.allocation}): "
              + ", ".join(f"{n}={d}" for n, d in self.draws.items())
              + (f" | weight-space ES on {sum(p.numel() for p in self.free)} other weights: {self.es_draws} pairs"
                 if self.free else ""), flush=True)

    def _allocate(self, widths: Dict[str, int], order: list) -> Dict[str, int]:
        """Draws per unit (a site, or a group of sites jittered together) from its output size per sample
        ``widths``; ``order`` is the forward order of the units, for ``"custom"``."""
        if self.allocation == "custom":
            if len(self.custom) != len(widths):
                raise ValueError(f"DUST custom allocation: {len(self.custom)} draw counts for {len(widths)} sites "
                                 f"({', '.join(widths)}).")
            draws = {n: dict(zip(order, self.custom))[n] for n in widths}
        elif self.allocation == "log":  # CMA-ES population rule per site, rounded up to whole chunks
            draws = {n: self.chunk * math.ceil(self.draw_scale * (4 + math.floor(3 * math.log(d))) / self.chunk)
                     for n, d in widths.items()}
        else:  # one chunk per site, the other chunks in proportion to the share (largest remainder: pop_size draws)
            share = {n: (d if self.allocation == "width" else 1.0) for n, d in widths.items()}
            units = self.pop_size // self.chunk - len(share)
            if units < 0:
                raise ValueError(f"DUST needs pop_size >= chunk * sites = {self.chunk * len(share)}, "
                                 f"got {self.pop_size}.")
            quota = {n: units * s / sum(share.values()) for n, s in share.items()}
            extra = {n: int(q) for n, q in quota.items()}
            for n in sorted(quota, key=lambda n: quota[n] - extra[n], reverse=True)[:units - sum(extra.values())]:
                extra[n] += 1
            return {n: self.chunk * (1 + extra[n]) for n in share}
        return draws

    def _sample_noise(self, name: str, samples: int) -> list:
        return [torch.randn(samples, *s[1:], generator=self.gen, device=self.gen.device) for s in self.shapes[name]]

    def _finish_site(self, name: str, errors: list) -> list:
        return errors

    def _capture(self, inputs: torch.Tensor, gts: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Refresh clean inputs, output shapes and per-sample task losses at the current weights."""
        self.model.eval()
        self.inputs, self.shapes, self.capturing = {}, {}, True
        try:
            task_losses = self.losses(inputs, gts)  # [K, B]
        finally:
            self.capturing = False
        if self.draws is None:
            self._setup()
        return task_losses

    def _estimate_site(self, name: str, clean: torch.Tensor, x: torch.Tensor,
                       y: Dict[str, torch.Tensor], grads: Dict[nn.Parameter, torch.Tensor]) -> None:
        """Estimate one site's gradient, using a clean baseline and chunk-repeated data at the same weights."""
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
        self._reassemble(self.sites[name], self._finish_site(name, errors), self.inputs[name], grads)

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
            self._estimate_site(name, clean, x, y, grads)
        if self.free:
            self._weight_space(inputs, gts, grads, K)
        J = torch.cat([grads[p].reshape(K, -1) if p in grads else torch.zeros(K, p.numel(), device=clean.device)
                       for p in self.model.parameters()], dim=1)
        return J / B, task_losses.mean(1)

    @torch.no_grad()
    def step(
        self,
        batches: list,
        theta: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        *,
        aggregate: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
        diagnose: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Estimate on fixed minibatches, update the flat weights ``theta`` once, and install them in the model.

        The optimizer must own ``theta``. Returns the pre-update estimates, task losses, update direction, optional
        task weights from ``aggregate``, and, with ``diagnose``, the pre-update backprop Jacobian.
        """
        if not batches:
            raise ValueError("DUST needs at least one minibatch per step.")
        if {id(p) for group in optimizer.param_groups for p in group['params']} != {id(theta)}:
            raise ValueError("DUST's optimizer must own the flat weight tensor theta.")
        if self.W.shape[0] > 1 and aggregate is None:
            raise ValueError("DUST needs an aggregator for per-task gradient estimates.")

        self.trainer.set_theta(theta.detach())
        J = torch.zeros(self.W.shape[0], theta.numel())
        losses = torch.zeros(len(self.trainer.task_name))
        for inputs, gts in batches:
            J_b, losses_b = self.estimate(inputs, gts)
            J += J_b.cpu() / len(batches)
            losses += losses_b.cpu() / len(batches)
        direction, weights = aggregate(J, losses) if aggregate is not None else (J[0], None)
        reference = true_jacobian(self.trainer, theta.detach(), batches) if diagnose else None
        theta.grad = direction.to(theta, copy=True)
        optimizer.step()
        self.trainer.set_theta(theta.detach())
        return J, losses, direction, weights, reference

    @staticmethod
    def _reassemble(m: nn.Module, errors: list, inputs: list, grads: Dict[nn.Parameter, torch.Tensor]) -> None:
        """Exact local linear map of the site ``m``: its weight (and bias) gradients from the output errors ``e``
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
        """Two-sided weight-space ES on the parameters outside the sites, rewarded by each objective summed over the
        samples (divided by the batch size in :meth:`estimate`, like the site gradients)."""
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


def run_dust(
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
    method: str = "DUST",
    _estimator: Callable[..., DUST] = DUST,
    **kwargs: Any,
) -> Dict[str, Any]:
    """DUST training of the weights ``theta`` from ``theta_0`` (see module docstring); the trainer ends at ``theta``.

    Parameters
    ----------
    pop_size : int
        Direct draws (forward passes of the batch) per generation, split over the sites by ``allocation``
        (unused with ``"log"``).
    draw_scale : float
        ``allocation="log"``: multiplier of the per-site CMA-ES population ``4 + floor(3 ln d)``.
    draws : list of int, optional
        ``allocation="custom"``: draws per site, in the order the forward pass calls the sites.
    sigma : float
        Std of the jitter on the site outputs (activation units, not weight units).
    chunk : int
        Draws per forward pass (the batch repeated); the rewards are centered over them. With ``antithetic``: even,
        ``chunk / 2`` pairs per forward pass, no centering.
    es_draws, es_sigma
        Antithetic pairs and std of the weight-space ES on the parameters outside the sites (unused without them).
    lr, optim, adam_betas, adam_eps, rmsprop_alpha, rmsprop_eps, sgd_momentum
        Optimizer on ``theta`` (:func:`~.es_common.make_es_optimizer`); the paper uses SGD with momentum.
    grad_cos_freq : int
        Every this many generations, on the same mini-batch: cosine(direction, the same rule on the backprop
        gradients), and with ``aggregate`` the cosine of every task's estimate with its backprop gradient. ``0``
        disables. Diagnostic only: the update never uses true gradients.
    aggregate : callable, optional
        Multi-objective DUST: one estimate per task loss, combined by ``aggregate(J, losses) -> (direction, w)``
        (``J``: ``[K, D]`` task gradients, ``losses``: ``[K]`` task losses on the batch, ``w``: ``[K]`` weights of
        the rows, logged). ``None``: DUST on the summed loss.
    antithetic : bool
        A-DUST (:mod:`.a_dust`): the draws of a site are ``draws / 2`` antithetic pairs; every draw count (``pop_size``,
        ``draws``) still counts forward evaluations, so the cost per generation is that of DUST.
    method : str
        Logged name of the method.
    """
    if trainer.theta_0 is None:
        raise RuntimeError("prepare_evolution() first.")
    if int(chunk) < 1 or (allocation in ("equal", "width") and int(pop_size) < int(chunk)):
        raise ValueError(f"DUST needs 1 <= chunk <= pop_size, got chunk={chunk}, pop_size={pop_size}.")
    require_shuffled_loader(train_dataloaders, method)
    configure_moea_random_seed(seed)
    gen = torch.Generator(trainer.device).manual_seed(int(seed))
    per_task = aggregate is not None
    dust = _estimator(trainer, pop_size, sigma, chunk, allocation, es_draws, es_sigma, gen, per_task, draw_scale,
                      draws, antithetic)

    theta = trainer.theta_0.detach().cpu().float().clone().requires_grad_(True)
    optim = str(optim).strip().lower()
    optimizer = make_es_optimizer([theta], optim, lr, adam_betas, adam_eps, rmsprop_alpha, rmsprop_eps, sgd_momentum)
    pop = {"log": f"log rule x{draw_scale}", "custom": f"custom {draws}"}.get(allocation, pop_size)
    print(f"[EvoMTL] {method}: dims={theta.numel()} | pop={pop} | chunk={chunk} | optim={optim} lr={lr} | "
          f"sigma={sigma}", flush=True)

    tasks = list(trainer.task_name)
    metrics_csv = MetricsCSV(trainer, metrics_columns(
        tasks, ["sigma"], [f"weight_{t}" for t in tasks] + [f"true_cos_{t}" for t in tasks] + ["true_cos_direction"]))
    F = np.empty((0, len(tasks)))
    try:
        for iteration in range(int(num_iterations)):
            t0 = time.perf_counter()
            step = wandb_step_offset + iteration
            batches = trainer._sample_batches(train_dataloaders, num_batches)
            diagnose = grad_cos_freq > 0 and iteration % int(grad_cos_freq) == 0
            J, losses, direction, w, J_true = dust.step(batches, theta, optimizer, aggregate=aggregate,
                                                      diagnose=diagnose)
            F = losses.numpy()[None].astype(np.float32)  # [1, K]: the search mean before the update
            row: Dict[str, Any] = {"sigma": float(sigma)}
            if per_task:
                row.update({f"weight_{t}": float(w[k]) for k, t in enumerate(tasks)})
            extras_wandb: Dict[str, Any] = {"evo/dust/direction_norm": float(direction.norm())}
            progress = f"|d|={float(direction.norm()):.4g}"
            if per_task:
                progress += " w=[" + ", ".join(f"{float(x):.3f}" for x in w) + "]"
                extras_wandb.update({f"evo/dust/weight/{t}": float(w[k]) for k, t in enumerate(tasks)})
            if diagnose:
                dir_cos = _cosine(direction, aggregate(J_true, losses)[0] if per_task else J_true.sum(0))
                if per_task:
                    true_cos = [_cosine(J[k], J_true[k]) for k in range(len(tasks))]
                    progress += " true_cos=[" + ", ".join(f"{c:+.4f}" for c in true_cos) + "]"
                    row.update({f"true_cos_{t}": true_cos[k] for k, t in enumerate(tasks)})
                    extras_wandb.update({f"evo/dust/true_cos/{t}": true_cos[k] for k, t in enumerate(tasks)})
                progress += f" dir_cos={dir_cos:+.4f}"
                row["true_cos_direction"] = dir_cos
                extras_wandb["evo/dust/true_cos/direction"] = dir_cos

            extras_wandb["evo/dust/mean_norm"] = float((theta.detach() - trainer.theta_0.cpu()).norm())
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

    return es_result(trainer, method, theta, theta.detach().numpy()[None], F, hv_ref_point, sigma,
                     aggregator=method)
