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

Each draw is one forward pass of the batch, as one member of an OpenES population of ``pop_size``; the jitter is as
large as the site's activations (for a convolution, far more numbers than its weights), so drawing it adds to the
cost. The draws are split over the sites equally (``allocation="equal"``) or in proportion to their output size per
sample (``"width"``), in multiples of ``chunk``.

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
    true_jacobian,
)

ALLOCATIONS = ("equal", "width")
CONV_WEIGHT = {nn.Conv1d: torch.nn.grad.conv1d_weight, nn.Conv2d: torch.nn.grad.conv2d_weight,
               nn.Conv3d: torch.nn.grad.conv3d_weight}


class DUST:
    """The estimator: forward hooks on every site capture its input during the clean pass and add the current jitter
    to its output during a draw (see the module docstring). :meth:`estimate` returns the gradient of the summed loss,
    or with ``per_task`` one gradient per task loss."""

    def __init__(self, trainer: Any, pop_size: int, sigma: float, chunk: int, allocation: str, es_draws: int,
                 es_sigma: float, gen: torch.Generator, per_task: bool = False):
        if trainer.multi_input:
            raise NotImplementedError("DUST needs one input batch shared by the tasks (multi_input=False).")
        if allocation not in ALLOCATIONS:
            raise ValueError(f"Unknown DUST allocation {allocation!r}; options: {ALLOCATIONS}")
        self.trainer, self.model, self.gen = trainer, trainer.model, gen
        self.pop_size, self.sigma, self.chunk, self.allocation = int(pop_size), float(sigma), int(chunk), allocation
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
        # one chunk per site, the other chunks in proportion to the share (largest remainder: exactly pop_size draws)
        share = {n: (sum(math.prod(s[1:]) for s in self.shapes[n]) if self.allocation == "width" else 1.0)
                 for n in self.sites}
        units = self.pop_size // self.chunk - len(share)
        if units < 0:
            raise ValueError(f"DUST needs pop_size >= chunk * sites = {self.chunk * len(share)}, got {self.pop_size}.")
        quota = {n: units * s / sum(share.values()) for n, s in share.items()}
        extra = {n: int(q) for n, q in quota.items()}
        for n in sorted(quota, key=lambda n: quota[n] - extra[n], reverse=True)[:units - sum(extra.values())]:
            extra[n] += 1
        self.draws = {n: self.chunk * (1 + extra[n]) for n in share}
        print(f"[EvoMTL] DUST: {sum(self.draws.values())} direct draws per generation ({self.allocation}): "
              + ", ".join(f"{n}={d}" for n, d in self.draws.items())
              + (f" | weight-space ES on {sum(p.numel() for p in self.free)} other weights: {self.es_draws} pairs"
                 if self.free else ""), flush=True)

    @torch.no_grad()
    def estimate(self, inputs: torch.Tensor, gts: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gradient estimates ``J`` (``[M, D]``, order of ``model.parameters()``) of the M objectives (mean over the
        batch), and the task losses (``[K]``), on one mini-batch."""
        self.model.eval()
        self.inputs, self.shapes, self.capturing = {}, {}, True
        try:
            task_losses = self.losses(inputs, gts)  # [K, B]
        finally:
            self.capturing = False
        if self.draws is None:
            self._setup()
        clean = self.W @ task_losses  # [M, B]
        K, B = clean.shape  # K: number of objectives from here on
        n = self.chunk
        x = inputs.repeat(n, *(1,) * (inputs.dim() - 1))
        y = {t: g.repeat(n, *(1,) * (g.dim() - 1)) for t, g in gts.items()}

        grads: Dict[nn.Parameter, torch.Tensor] = {}
        for name, draws in self.draws.items():
            shapes = self.shapes[name]
            errors = [torch.zeros(K, *s, device=clean.device) for s in shapes]
            for _ in range(draws // n):
                noise = [torch.randn(n * B, *s[1:], generator=self.gen, device=self.gen.device) for s in shapes]
                self.jitter = {name: [self.sigma * a for a in noise]}
                try:
                    reward = clean.unsqueeze(1) - (self.W @ self.losses(x, y)).view(K, n, B)
                finally:
                    self.jitter = {}
                if n > 1:
                    reward = reward - reward.mean(1, keepdim=True)
                for e, a in zip(errors, noise):
                    e -= torch.einsum("knb,nb...->kb...", reward, a.view(n, B, *a.shape[1:])) / (self.sigma * draws)
            self._reassemble(self.sites[name], errors, self.inputs[name], grads)
        if self.free:
            self._weight_space(inputs, gts, grads, K)
        J = torch.cat([grads[p].reshape(K, -1) if p in grads else torch.zeros(K, p.numel(), device=clean.device)
                       for p in self.model.parameters()], dim=1)
        return J / B, task_losses.mean(1)

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
            if m.bias is not None:
                grads[m.bias] = grads.get(m.bias, 0) + b

    def _weight_space(self, inputs: torch.Tensor, gts: Dict[str, torch.Tensor],
                      grads: Dict[nn.Parameter, torch.Tensor], K: int) -> None:
        """Two-sided weight-space ES on the parameters outside the sites, rewarded by each objective summed over the
        samples (divided by the batch size in :meth:`estimate`, like the site gradients)."""
        clean = [p.detach().clone() for p in self.free]
        est = [torch.zeros(K, *p.shape, device=p.device) for p in self.free]
        try:
            for _ in range(self.es_draws):
                a = [torch.randn(p.shape, generator=self.gen, device=self.gen.device) for p in self.free]
                losses = []
                for sign in (1.0, -1.0):
                    for p, c, d in zip(self.free, clean, a):
                        p.copy_(c + sign * self.es_sigma * d)
                    losses.append((self.W @ self.losses(inputs, gts)).sum(1))  # [K objectives]
                g = (losses[0] - losses[1]) / (2 * self.es_sigma * self.es_draws)
                for e, d in zip(est, a):
                    e += g.view(K, *(1,) * d.dim()) * d
        finally:
            for p, c in zip(self.free, clean):
                p.copy_(c)
        for p, e in zip(self.free, est):
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
    method: str = "DUST",
    **kwargs: Any,
) -> Dict[str, Any]:
    """DUST training of the weights ``theta`` from ``theta_0`` (see module docstring); the trainer ends at ``theta``.

    Parameters
    ----------
    pop_size : int
        Direct draws (forward passes of the batch) per generation, split over the sites by ``allocation``.
    sigma : float
        Std of the jitter on the site outputs (activation units, not weight units).
    chunk : int
        Draws per forward pass (the batch repeated); the rewards are centered over them.
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
    method : str
        Logged name of the method.
    """
    if trainer.theta_0 is None:
        raise RuntimeError("prepare_evolution() first.")
    if int(chunk) < 1 or int(pop_size) < int(chunk):
        raise ValueError(f"DUST needs 1 <= chunk <= pop_size, got chunk={chunk}, pop_size={pop_size}.")
    configure_moea_random_seed(seed)
    gen = torch.Generator(trainer.device).manual_seed(int(seed))
    per_task = aggregate is not None
    dust = DUST(trainer, pop_size, sigma, chunk, allocation, es_draws, es_sigma, gen, per_task)

    theta = trainer.theta_0.detach().cpu().float().clone().requires_grad_(True)
    optim = str(optim).strip().lower()
    optimizer = make_es_optimizer([theta], optim, lr, adam_betas, adam_eps, rmsprop_alpha, rmsprop_eps, sgd_momentum)
    print(f"[EvoMTL] {method}: dims={theta.numel()} | pop={pop_size} | chunk={chunk} | optim={optim} lr={lr} | "
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
            trainer.set_theta(theta.detach())
            J = torch.zeros(dust.W.shape[0], theta.numel())
            losses = torch.zeros(len(tasks))
            for inp, gts in batches:
                J_b, L_b = dust.estimate(inp, gts)
                J += J_b.cpu() / len(batches)
                losses += L_b.cpu() / len(batches)
            F = losses.numpy()[None].astype(np.float32)  # [1, K]: the search mean before the update
            row: Dict[str, Any] = {"sigma": float(sigma)}
            if per_task:
                direction, w = aggregate(J, losses)
                row.update({f"weight_{t}": float(w[k]) for k, t in enumerate(tasks)})
            else:
                direction = J[0]
            extras_wandb: Dict[str, Any] = {"evo/dust/direction_norm": float(direction.norm())}
            progress = f"|d|={float(direction.norm()):.4g}"
            if per_task:
                progress += " w=[" + ", ".join(f"{float(x):.3f}" for x in w) + "]"
                extras_wandb.update({f"evo/dust/weight/{t}": float(w[k]) for k, t in enumerate(tasks)})
            if grad_cos_freq > 0 and iteration % int(grad_cos_freq) == 0:
                J_true = true_jacobian(trainer, theta.detach(), batches)
                dir_cos = _cosine(direction, aggregate(J_true, losses)[0] if per_task else J_true.sum(0))
                if per_task:
                    true_cos = [_cosine(J[k], J_true[k]) for k in range(len(tasks))]
                    progress += " true_cos=[" + ", ".join(f"{c:+.4f}" for c in true_cos) + "]"
                    row.update({f"true_cos_{t}": true_cos[k] for k, t in enumerate(tasks)})
                    extras_wandb.update({f"evo/dust/true_cos/{t}": true_cos[k] for k, t in enumerate(tasks)})
                progress += f" dir_cos={dir_cos:+.4f}"
                row["true_cos_direction"] = dir_cos
                extras_wandb["evo/dust/true_cos/direction"] = dir_cos

            theta.grad = direction.clone()
            optimizer.step()
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
