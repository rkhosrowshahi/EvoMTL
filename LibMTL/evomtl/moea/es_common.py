"""Building blocks shared by the evolution strategies on the network weights (OpenES family, PSL-ES, SC-DE, ...).

Fitness shaping, the optimizer on the search mean, Guided ES sampling, the backprop Jacobian used by the diagnostics,
the UPGrad combination of a Jacobian, and the per-generation reporting (train / test snapshot of the search mean,
``metrics.csv``, ``gradients.pth.tar``, the result dict) of the single-network ES runners.
"""

from __future__ import annotations

import csv
import math
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from scipy.stats import rankdata

from ...weighting.UPGrad import _compute_gramian, _normalize, _project_weights, _regularize
from .common import should_run_full_test
from .hv import pareto_front_indices

FITNESS_SHAPING = ("centered_ranks", "raw")


def centered_ranks(fitness: np.ndarray) -> np.ndarray:
    """Centered ranks in ``[-0.5, 0.5]`` along the population axis (evosax ``centered_rank_fitness_shaping_fn``)."""
    ranks = rankdata(fitness, axis=0) - 1.0
    return ranks / (fitness.shape[0] - 1) - 0.5


def shape_fitness(fitness: np.ndarray, shaping: str) -> torch.Tensor:
    """Shaped fitness along the population axis: ``centered_ranks`` or ``raw`` (losses minus the population mean,
    the unshaped OpenAI-ES estimator; with antithetic pairs the mean cancels exactly)."""
    if shaping == "raw":
        return torch.as_tensor(fitness - fitness.mean(axis=0, keepdims=True), dtype=torch.float32)
    return torch.as_tensor(centered_ranks(fitness), dtype=torch.float32)


def upgrad_aggregate(J: torch.Tensor, norm_eps: float = 1e-4, reg_eps: float = 1e-4) -> Tuple[torch.Tensor, torch.Tensor]:
    """UPGrad (Quinton & Rey 2024) on Jacobian rows ``J`` (``[K, D]``): project every row onto the dual cone of the
    rows and average the projections; return ``(d, w)`` with ``d = J^T w`` and ``g_k . d >= 0`` for every ``k``."""
    K = J.shape[0]
    U = torch.diag(torch.full((K,), 1.0 / K, dtype=J.dtype))
    G = _regularize(_normalize(_compute_gramian(J), norm_eps), reg_eps)
    w = torch.sum(_project_weights(U, G), dim=0)
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


class GuidedSampler:
    """Guided ES sampling (Maheswaranathan et al., 2019; evosax ``GuidedES``) around past ES directions.

    The ``noise_sampler`` of the OpenES-family runners. The guide is self-generated, with no backprop: the last
    ``guided_k`` unit ascent directions (FIFO), or with ``guided_ema > 0`` one moving average of them. Each member's
    noise is ``eps = sqrt(alpha) e + s U e_k`` (``e ~ N(0, I_D)``, ``e_k ~ N(0, I_k)``, ``U`` an orthonormal basis of
    the guide) with ``s = guided_sigma / sigma`` if ``guided_sigma > 0`` (the perturbation size along the guide, in
    weight units), else ``s = sqrt((1 - alpha) D / k)`` (the noise budget of isotropic OpenES). The first generation,
    before any guide exists, samples isotropically.
    """

    name = "GuidedES"

    def __init__(self, guided_k: int, guided_alpha: float, guided_sigma: float, guided_ema: float):
        if int(guided_k) < 1:
            raise ValueError(f"Guided ES needs guided_k >= 1, got {guided_k}.")
        if not 0.0 < float(guided_alpha) <= 1.0:
            raise ValueError(f"guided_alpha must be in (0, 1], got {guided_alpha}.")
        self.k, self.alpha = int(guided_k), float(guided_alpha)
        self.sigma_sub, self.ema = float(guided_sigma), float(guided_ema)
        self.guide: list = []  # unit guiding directions

    def describe(self) -> str:
        return f"guided k={self.k}, alpha={self.alpha}, sigma_sub={self.sigma_sub or 'auto'}, ema={self.ema}"

    def sample(self, eps: torch.Tensor, gen: torch.Generator, sigma: float) -> torch.Tensor:
        if not self.guide:  # first generation: no direction yet
            return eps
        return guided_noise(eps, self.guide, gen, sigma, self.alpha, self.sigma_sub)

    def update(self, direction: torch.Tensor) -> None:
        self.guide = update_guide(self.guide, direction, self.k, self.ema)


def _mean_pairwise_cosine(J: torch.Tensor) -> float:
    Jn = J / J.norm(dim=1, keepdim=True).clamp_min(1e-12)
    C = Jn @ Jn.T
    K = J.shape[0]
    if K < 2:
        return 1.0
    return float((C.sum() - C.diagonal().sum()) / (K * (K - 1)))


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(a @ b / (a.norm() * b.norm()).clamp_min(1e-30))


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


# --------------------------------------------------------------------------- reporting of single-network ES runs

def metrics_columns(tasks: list, sigma_cols: list, extra_cols: list) -> list:
    """Columns of ``metrics.csv``: generation, time, the sigma columns, per-task population / train / test metrics,
    then the runner's own ``extra_cols``."""
    return (
        ["generation", "time_s"] + sigma_cols
        + [f"pop_loss_mean_{t}" for t in tasks]
        + [f"train_loss_{t}" for t in tasks] + [f"train_acc_{t}" for t in tasks]
        + [f"test_loss_{t}" for t in tasks] + [f"test_acc_{t}" for t in tasks]
        + extra_cols
    )


class MetricsCSV:
    """``metrics.csv`` next to the checkpoints: one row per generation, flushed as it is written (no-op without a
    ``save_path``). Test columns are filled only on generations with a test evaluation (``evo_eval_freq``)."""

    def __init__(self, trainer: Any, columns: list):
        self.file = self.writer = None
        if getattr(trainer, "save_path", None):
            os.makedirs(trainer.save_path, exist_ok=True)
            self.file = open(os.path.join(trainer.save_path, "metrics.csv"), "w", newline="")
            self.writer = csv.DictWriter(self.file, fieldnames=columns)
            self.writer.writeheader()

    def write(self, iteration: int, elapsed: float, tasks: list, F: np.ndarray, train_metrics: Dict[str, Any],
              test_metrics: Optional[Dict[str, Any]], **extra: Any) -> None:
        """Row of generation ``iteration + 1``: the shared per-task columns plus ``extra`` (column -> value)."""
        if self.writer is None:
            return
        row: Dict[str, Any] = {"generation": iteration + 1, "time_s": round(elapsed, 4)}
        for k, t in enumerate(tasks):
            row[f"pop_loss_mean_{t}"] = float(F[:, k].mean())
            row[f"train_loss_{t}"] = float(train_metrics["loss_item"][k])
            row[f"train_acc_{t}"] = float(train_metrics["results"][t][0])
            if test_metrics is not None:
                row[f"test_loss_{t}"] = float(test_metrics["loss_item"][k])
                row[f"test_acc_{t}"] = float(test_metrics["results"][t][0])
        row.update(extra)
        self.writer.writerow(row)
        self.file.flush()

    def close(self) -> None:
        if self.file is not None:
            self.file.close()


class GradientSnapshots:
    """Every ``freq`` generations from generation 0 (before the first update), store tensors (search mean, true and
    pseudo-gradients, directions, ...) in ``<save_path>/gradients.pth.tar``, rewritten at each snapshot (float32,
    stacked along a leading snapshot axis; read by ``scripts/pca_gradients.py``). ``freq = 0`` disables."""

    def __init__(self, trainer: Any, freq: int):
        self.path = None
        if freq > 0 and not getattr(trainer, "save_path", None):
            print("[EvoMTL] grad_save_freq ignored: needs a save_path.", flush=True)
            freq = 0
        if freq > 0:
            self.path = os.path.join(trainer.save_path, "gradients.pth.tar")
        self.freq = int(freq)
        self.generation: list = []
        self.tensors: Dict[str, list] = {}

    def due(self, iteration: int) -> bool:
        return self.freq > 0 and iteration % self.freq == 0

    def save(self, iteration: int, tensors: Dict[str, torch.Tensor], meta: Dict[str, Any]) -> None:
        self.generation.append(iteration)
        for key, val in tensors.items():
            self.tensors.setdefault(key, []).append(val.detach().float().clone())
        torch.save({
            "generation": torch.tensor(self.generation),
            **{k: torch.stack(v) for k, v in self.tensors.items()},
            **meta,
        }, self.path)


def report_search_mean(
    trainer: Any,
    theta: torch.Tensor,
    iteration: int,
    num_iterations: int,
    F: np.ndarray,
    pareto_front: np.ndarray,
    shared_batches: list,
    test_dataloaders: Optional[Any],
    evo_eval_freq: Any,
    step: int,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """``theta`` is the ES solution: set it, snapshot it on the generation's mini-batch and, on test generations,
    plot the objective space and evaluate the test set. Return ``(train_metrics, test_metrics or None)``."""
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
    test_metrics = None
    if test_dataloaders is not None and run_eval:
        test_metrics = trainer.test(test_dataloaders, epoch=None, mode="test", wandb_log_step=step,
                                    suppress_display=True, return_metrics=True)
        trainer.print_compact_train_test_line(train_batch_metrics, test_metrics, epoch_label=iteration + 1)
    else:
        parts = " | ".join(f"{task}: {float(train_batch_metrics['loss_item'][i]):.4f}"
                           for i, task in enumerate(trainer.task_name))
        print(f"[EvoMTL] Mean theta TRAIN (curr batch) iteration {iteration + 1:04d} | {parts}", flush=True)
    return train_batch_metrics, test_metrics


def es_result(
    trainer: Any,
    method: str,
    theta: torch.Tensor,
    population_X: np.ndarray,
    population_F: np.ndarray,
    hv_ref_point: Optional[np.ndarray],
    sigma: Any,
    **extra: Any,
) -> Dict[str, Any]:
    """Result dict of a single-network ES run; the trainer ends at the search mean ``theta`` (``pareto_center_X``).

    ``pareto_F`` / ``pareto_X`` are the non-dominated members of the last population (for logging and checkpoints).
    """
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
        **extra,
    }
    trainer.last_evo = out
    return out
