"""Evolutionary multi-objective training on low-dimensional parameter-sharing codes.

Default schedule (``gd_then_moea``, :meth:`EvoMTLTrainer.train_then_evolve`):

1. Gradient-descent warmup (reuse :meth:`Trainer.train`).
2. Snapshot ``theta_0`` and build a parameter-sharing ``adapter`` so that
   ``theta = theta_0 + adapter.forward(z, alpha=evo_adapter_alpha)`` (``alpha`` scales ``delta_theta``
   inside :meth:`~LibMTL.evomtl.parameter_sharing` ``process``; config ``evo_adapter_alpha``
   maps to :attr:`adapter_alpha` on the trainer).
3. Run NSGA-II, MOPSO-CD (pymoo), or COMO-CMA-ES / MO-CMA-ES (comocma) with an ask-and-tell loop.
4. Form a *Pareto-center* latent vector by averaging Pareto-set points weighted by
   marginal hypervolume contribution, apply it, then evaluate train and test (same
   path as :meth:`Trainer.test`) and save.

Alternate schedule (``moea_then_gd``, :meth:`EvoMTLTrainer.evolve_then_train`): run
steps 2–4 on the initial weights (no GD warmup), apply the Pareto-center ``theta``,
then run full-parameter GD for the configured epochs—using MOEA as a structured
initialization when subspace-only fine-tuning stalls.

References:
- pymoo ask-and-tell: https://www.pymoo.org/algorithms/usage.html#nb-algorithms-ask-and-tell
- comocma: https://github.com/CMA-ES/pycomocma
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

from ..trainer import Trainer
from .parameter_sharing import (
    DictLoRA,
    FlattenLoRA,
    LayerwiseRandomBlocking,
    LayerwiseRandomProjection,
    LayerwiseScaledRandomProjection,
    LinearOnlyLoRA,
    ModulationLoRA,
    RandomProjection,
    SpectralAllSVD,
    SpectralLoRA,
    adapt_parameters,
)

EVO_PS_REGISTRY = {
    "random_proj": RandomProjection,
    "layerwise_random_proj": LayerwiseRandomProjection,
    "layerwise_random_blocking": LayerwiseRandomBlocking,
    "layerwise_scaled_random_proj": LayerwiseScaledRandomProjection,
    "flatten_lora": FlattenLoRA,
    "spherical_lora": FlattenLoRA,
    "dict_lora": DictLoRA,
    "linear_lora": LinearOnlyLoRA,
    "modulation_lora": ModulationLoRA,
    "spectral_lora": SpectralLoRA,
    "spectral_all_svd": SpectralAllSVD,
}


def resolve_adapter_class(name: str) -> Type[Any]:
    """Map :mod:`LibMTL.config` ``--evo_adapter`` string to a parameter-sharing adapter class."""
    key = name.strip().lower().replace("-", "_")
    if key not in EVO_PS_REGISTRY:
        raise ValueError(
            f"Unknown evo_adapter {name!r}. Options: {sorted(EVO_PS_REGISTRY)}"
        )
    return EVO_PS_REGISTRY[key]


try:
    from pymoo.algorithms.moo.mopso_cd import MOPSO_CD
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.evaluator import Evaluator
    from pymoo.core.problem import Problem
    from pymoo.core.termination import NoTermination
    from pymoo.indicators.hv import Hypervolume
    from pymoo.problems.static import StaticProblem
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "EvoMTLTrainer requires pymoo. Install with: pip install pymoo"
    ) from e


def _moea_should_run_full_test(step_idx: int, n_steps: int, every: int) -> bool:
    """Whether to run a full test pass this MOEA outer step.

    ``every <= 0`` disables in-loop test evaluation (train-snapshot logging only).
    ``every >= 1`` runs test on steps ``0, every, 2*every, ...`` and always on the
    last step ``n_steps - 1``.
    """
    if every <= 0:
        return False
    n_steps = int(n_steps)
    if n_steps <= 0:
        return False
    step_idx = int(step_idx)
    every = int(every)
    return (step_idx % every == 0) or (step_idx == n_steps - 1)


def _default_hv_ref_point(F: np.ndarray) -> np.ndarray:
    """Reference point for HV (minimization): slightly beyond the nadir of F."""
    F = np.atleast_2d(F)
    return np.max(F, axis=0) + 1e-6


def _normalize_objectives_for_hv(
    F_nd: np.ndarray,
    ref_point: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Min-max each objective on the Pareto set (ideal→0, nadir→1).

    Matches ``hv_normalized_*_center``: ``scale = where(nadir-ideal > 1e-8, ...)``,
    default normalized HV reference ``1.1`` per objective. Custom ``ref_point``
    (raw space) uses the same ``ideal`` / ``scale``.
    """
    F_nd = np.atleast_2d(np.asarray(F_nd, dtype=np.float64))
    n, m = F_nd.shape
    if n == 0:
        return F_nd, np.full(m, 1.1, dtype=np.float64)
    ideal = np.min(F_nd, axis=0)
    nadir = np.max(F_nd, axis=0)
    scale = np.where(nadir - ideal > 1e-8, nadir - ideal, 1.0)
    F_norm = (F_nd - ideal) / scale
    if ref_point is None:
        ref_norm = np.full(m, 1.1, dtype=np.float64)
    else:
        ref = np.asarray(ref_point, dtype=np.float64).reshape(m)
        ref_norm = (ref - ideal) / scale
    return F_norm, ref_norm


def pareto_front_indices(F: np.ndarray) -> np.ndarray:
    """Indices of the first nondominated front (minimization)."""
    F = np.atleast_2d(F)
    nds = NonDominatedSorting()
    return nds.do(F, only_non_dominated_front=True)


def pareto_center_weights_from_marginal_hv(
    contributions: np.ndarray,
    *,
    aggregation: str = "linear",
    softmax_temperature: float = 1.0,
) -> np.ndarray:
    """Turn raw marginal HV contributions into weights for a Pareto-set average.

    * ``linear``: nonnegative contributions, normalized to sum to 1 (same as
      ``hv_normalized_linear_center``).
    * ``softmax``: ``exp((c/temp) - max(c/temp))`` normalized (same as
      ``hv_normalized_softmax_center``).
    """
    contributions = np.asarray(contributions, dtype=np.float64).reshape(-1)
    n = contributions.shape[0]
    if n == 0:
        return np.zeros(0)
    agg = aggregation.strip().lower()
    if agg == "softmax":
        temp = max(float(softmax_temperature), 1e-8)
        scaled = contributions / temp
        scaled = scaled - np.max(scaled)
        expv = np.exp(scaled)
        s = float(expv.sum())
        if s < 1e-12:
            return np.ones(n) / n
        return expv / s
    if agg != "linear":
        raise ValueError(
            "aggregation must be 'linear' or 'softmax', got {!r}".format(aggregation)
        )
    contribs = np.maximum(0.0, contributions)
    s = contribs.sum()
    if s < 1e-12:
        return np.ones(n) / n
    return contribs / s


def marginal_hypervolume_weights(
    F_nd: np.ndarray,
    ref_point: Optional[np.ndarray] = None,
    *,
    normalize_objectives: bool = True,
    aggregation: str = "linear",
    softmax_temperature: float = 1.0,
) -> np.ndarray:
    """Weights for Pareto-center averaging from marginal hypervolume contributions.

    Parameters
    ----------
    F_nd : ndarray, shape (n, m)
        Objectives for points on one Pareto front (nondominated among themselves).
    ref_point : ndarray, shape (m,), optional
        In raw objective space (before normalization). Dominated by all points
        (worse in every objective for minimization). Ignored for the default
        normalized reference when ``normalize_objectives`` is True and this is
        omitted.
    normalize_objectives : bool
        If True (default), min-max normalize each objective on ``F_nd`` before HV
        so Pareto-center weights are not skewed by objective scale.
    aggregation : {'linear', 'softmax'}
        How marginal contributions become weights (see
        :func:`pareto_center_weights_from_marginal_hv`).
    softmax_temperature : float
        Used when ``aggregation`` is ``softmax``; larger values yield softer weights.
    """
    F_nd = np.atleast_2d(F_nd)
    n, m = F_nd.shape
    if normalize_objectives:
        F_nd, ref_point = _normalize_objectives_for_hv(F_nd, ref_point)
    elif ref_point is None:
        ref_point = _default_hv_ref_point(F_nd)
    else:
        ref_point = np.asarray(ref_point, dtype=np.float64).reshape(m)

    hv = Hypervolume(ref_point=ref_point)
    if n == 0:
        return np.zeros(0)
    if n == 1:
        return np.ones(1)

    total = hv.do(F_nd)
    contribs = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        sub = hv.do(F_nd[mask])
        contribs[i] = max(0.0, total - sub)
    return pareto_center_weights_from_marginal_hv(
        contribs,
        aggregation=aggregation,
        softmax_temperature=softmax_temperature,
    )


def pareto_center_from_weights(X_nd: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted average of rows of ``X_nd``."""
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    X_nd = np.atleast_2d(X_nd)
    return np.sum(X_nd * w[:, None], axis=0)


class EvoMTLTrainer(Trainer):
    """Trainer with MOEA phases on parameter-sharing latent vectors ``z``.

    After GD warmup, call :meth:`init_parameter_sharing` (or rely on it from
    :meth:`prepare_evolution`) then :meth:`run_nsga2` or :meth:`run_comocma`.

    Notes
    -----
    * ``comocma`` is only used for **two** objectives in upstream tests; this
      class raises if ``task_num != 2`` for :meth:`run_comocma`.
    * ``SpectralLoRA`` uses SVD modulation on conv weights and LoRA on linear
      weights; ``SpectralAllSVD`` uses SVD on both conv and linear 2D weights.
      There is no separate ``SphericalLoRA`` class — use ``FlattenLoRA``
      (``spherical_lora``) or another strategy from ``LibMTL.evomtl.parameter_sharing``.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.theta_0: Optional[torch.Tensor] = None
        self.adapter: Any = None
        self.adapter_alpha: float = 1.0
        self.last_evo: Dict[str, Any] = {}

    def capture_base_parameters(self) -> None:
        """Store ``theta_0`` as a flat vector on the trainer device."""
        self.theta_0 = nn.utils.parameters_to_vector(self.model.parameters()).detach().clone()

    def init_parameter_sharing(
        self,
        adapter_class: Type[Any],
        adapter_alpha: float = 1.0,
        capture_if_needed: bool = True,
        **adapter_kwargs: Any,
    ) -> None:
        """Instantiate parameter-sharing ``adapter`` on the current model.

        If ``theta_0`` was never captured and ``capture_if_needed`` is True,
        the current weights are stored as ``theta_0`` first.
        """
        if self.theta_0 is None and capture_if_needed:
            self.capture_base_parameters()
        adapter_kwargs = dict(adapter_kwargs)
        adapter_kwargs.setdefault("device", str(self.device))
        self.adapter = adapter_class(self.model, **adapter_kwargs)
        self.adapter_alpha = float(adapter_alpha)

    def _print_and_log_moea_adapter_stats(self) -> None:
        """Print model size and MOEA latent dimensionality; add to W&B run config if enabled."""
        if self.adapter is None:
            return
        num_parameters = int(sum(p.numel() for p in self.model.parameters()))
        num_dimensions = int(self.adapter.num_dims)
        print(
            f"[EvoMTL] MOEA (parameter sharing): num_parameters={num_parameters} | "
            f"num_dimensions={num_dimensions}",
            flush=True,
        )
        if getattr(self, "_wandb_enabled", False):
            wandb.config.update(
                {
                    "evo_num_parameters": num_parameters,
                    "evo_num_dimensions": num_dimensions,
                },
                allow_val_change=True,
            )

    def prepare_evolution(
        self,
        adapter_class: Type[Any],
        adapter_alpha: float = 1.0,
        **adapter_kwargs: Any,
    ) -> None:
        """Capture ``theta_0`` (if missing) and build ``adapter``."""
        self.init_parameter_sharing(
            adapter_class, adapter_alpha=adapter_alpha, **adapter_kwargs
        )
        self._print_and_log_moea_adapter_stats()

    def adapt_parameters(
        self,
        z: Union[np.ndarray, List[float], torch.Tensor],
        alpha_override: Optional[float] = None,
    ) -> None:
        """Set weights to ``theta_0 + phi(z)`` where ``phi`` is ``adapter.forward`` (expand + ``process``).

        Delegates to :func:`~LibMTL.evomtl.parameter_sharing.adapt_parameters`.
        ``alpha_override`` (else :attr:`adapter_alpha` / config ``evo_adapter_alpha``) scales the map inside each strategy's ``process``.
        """
        if self.theta_0 is None:
            raise RuntimeError(
                "Call capture_base_parameters() before adapt_parameters()."
            )
        if self.adapter is None:
            raise RuntimeError(
                "Call init_parameter_sharing() before adapt_parameters()."
            )
        alpha = (
            self.adapter_alpha
            if alpha_override is None
            else float(alpha_override)
        )
        adapt_parameters(
            self.model,
            self.theta_0,
            self.adapter,
            z,
            alpha=alpha,
            device=self.device,
        )

    def restore_base_parameters(self) -> None:
        if self.theta_0 is None:
            raise RuntimeError("No base snapshot; run capture_base_parameters() first.")
        nn.utils.vector_to_parameters(self.theta_0, self.model.parameters())

    def _sample_batches(
        self,
        train_dataloaders: Any,
        n_batches: int,
    ) -> List[Tuple[Any, Any]]:
        """Sample ``n_batches`` mini-batches once; return as a list of ``(inp, gts)`` pairs.

        For ``multi_input=True``, each element is ``({task: inp}, {task: gts})``.
        These are meant to be passed to :meth:`evaluate_multiobjective_losses` so
        that all solutions in one iteration share the exact same data.
        """
        train_loader, train_batch = self._prepare_dataloaders(train_dataloaders)
        n_batches = int(max(1, n_batches))
        batches: List[Tuple[Any, Any]] = []

        if not self.multi_input:
            n = min(n_batches, train_batch)
            for _ in range(n):
                inp, gts = self._process_data(train_loader)
                batches.append((inp, gts))
        else:
            n = min(n_batches, min(train_batch))
            for _ in range(n):
                task_inputs: Dict[str, Any] = {}
                task_gts: Dict[str, Any] = {}
                for task in self.task_name:
                    task_inputs[task], task_gts[task] = self._process_data(
                        train_loader[task]
                    )
                batches.append((task_inputs, task_gts))
        return batches

    @torch.no_grad()
    def evaluate_multiobjective_losses(
        self,
        train_dataloaders: Any,
        n_batches: int = 5,
        prefetched_batches: Optional[List[Tuple[Any, Any]]] = None,
    ) -> np.ndarray:
        """Mean per-task training loss over up to ``n_batches`` mini-batches.

        If ``prefetched_batches`` is provided (a list returned by
        :meth:`_sample_batches`), those exact batches are reused instead of
        drawing new ones.  Pass pre-fetched batches so that all solutions in a
        single MOEA iteration are evaluated on the **same** mini-batch.
        """
        self.model.eval()

        if prefetched_batches is not None:
            acc = torch.zeros(self.task_num, device=self.device)
            for inp, gts in prefetched_batches:
                losses = self.forward4loss(self.model, inp, gts)
                acc += losses.detach()
            return (acc / len(prefetched_batches)).cpu().numpy()

        train_loader, train_batch = self._prepare_dataloaders(train_dataloaders)
        n_batches = int(max(1, n_batches))

        if not self.multi_input:
            n = min(n_batches, train_batch)
            acc = torch.zeros(self.task_num, device=self.device)
            for _ in range(n):
                inp, gts = self._process_data(train_loader)
                losses = self.forward4loss(self.model, inp, gts)
                acc += losses.detach()
            return (acc / n).cpu().numpy()

        n = min(n_batches, min(train_batch))
        acc = torch.zeros(self.task_num, device=self.device)
        for _ in range(n):
            train_inputs, train_gts = {}, {}
            for task in self.task_name:
                train_inputs[task], train_gts[task] = self._process_data(
                    train_loader[task]
                )
            losses = self.forward4loss(self.model, train_inputs, train_gts)
            acc += losses.detach()
        return (acc / n).cpu().numpy()

    @torch.no_grad()
    def metrics_snapshot_from_prefetched_batches(
        self,
        prefetched_batches: List[Tuple[Any, Any]],
    ) -> Dict[str, Any]:
        """Per-task loss and meter metrics (e.g. accuracy) on the same batches as MOEA eval.

        Matches :meth:`Trainer.test` with ``return_metrics=True`` layout: ``loss_item``,
        ``results`` per task, and wall-clock ``elapsed`` for the forward pass.
        """
        self.model.eval()
        self.meter.reinit()
        self.meter.record_time("begin")
        for inp, gts in prefetched_batches:
            _, preds = self.forward4loss(self.model, inp, gts, return_preds=True)
            if not self.multi_input:
                self.meter.update(preds, gts)
            else:
                for task in self.task_name:
                    self.meter.update(preds[task], gts[task], task)
        self.meter.record_time("end")
        self.meter.get_score()
        metrics = {
            "loss_item": np.copy(self.meter.loss_item),
            "results": {
                task: tuple(self.meter.results[task]) for task in self.task_name
            },
            "elapsed": float(self.meter.end_time - self.meter.beg_time),
        }
        self.meter.reinit()
        return metrics

    def finalize_pareto_center(
        self,
        F: np.ndarray,
        X: np.ndarray,
        ref_point: Optional[np.ndarray] = None,
        *,
        hv_center_aggregation: str = "linear",
        hv_softmax_temperature: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Restrict to the Pareto front, compute HV weights, return ``pareto_center_X``."""
        F = np.atleast_2d(F)
        X = np.atleast_2d(X)
        idx = pareto_front_indices(F)
        F_nd = F[idx]
        X_nd = X[idx]
        w = marginal_hypervolume_weights(
            F_nd,
            ref_point=ref_point,
            aggregation=hv_center_aggregation,
            softmax_temperature=hv_softmax_temperature,
        )
        pareto_center_X = pareto_center_from_weights(X_nd, w)
        return pareto_center_X, w, idx

    def _wandb_log_evo_front_metrics(
        self,
        iterations: int,
        pareto_front: np.ndarray,
        step: int,
        extras: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log one MOEA step under ``evo/`` (shared keys for NSGA-II and COMO-CMA-ES)."""
        if not getattr(self, "_wandb_enabled", False):
            return
        pareto_front = np.atleast_2d(np.asarray(pareto_front, dtype=np.float64))
        ideal = np.min(pareto_front, axis=0)
        nadir = np.max(pareto_front, axis=0) if len(pareto_front) else np.max(pareto_front, axis=0)
        log: Dict[str, Any] = {
            "evo/iterations": int(iterations),
            "evo/pareto_front_size": float(len(pareto_front)),
        }
        # Mean task loss per individual, then mean over population (like train losses/avg).
        log["evo/losses/avg"] = float(np.mean(np.mean(pareto_front, axis=1)))
        for ti, tn in enumerate(self.task_name):
            log[f"evo/{tn}/mean"] = float(np.mean(pareto_front[:, ti]))
            log[f"evo/{tn}/ideal"] = float(ideal[ti])
            log[f"evo/{tn}/nadir"] = float(nadir[ti])
        if extras:
            log.update(extras)
        self._wandb_log(log, step=step)

    def _wandb_log_evo_objective_plot(
        self,
        *,
        population_F: np.ndarray,
        pareto_F: np.ndarray,
        center_F: Optional[np.ndarray],
        offspring_F: Optional[np.ndarray] = None,
        step: int,
    ) -> None:
        """Single 2D scatter: population, offspring, Pareto front, Pareto-center snapshot."""
        if not getattr(self, "_wandb_enabled", False):
            return
        n_obj = int(self.task_num)
        if n_obj < 2:
            return
        F_pop = np.atleast_2d(np.asarray(population_F, dtype=np.float64))
        F_pf = np.atleast_2d(np.asarray(pareto_F, dtype=np.float64))
        if offspring_F is not None:
            F_off = np.atleast_2d(np.asarray(offspring_F, dtype=np.float64))
            if F_off.size == 0:
                F_off = np.empty((0, n_obj), dtype=np.float64)
        else:
            F_off = np.empty((0, n_obj), dtype=np.float64)
        if (
            F_pop.size == 0
            and F_pf.size == 0
            and center_F is None
            and F_off.size == 0
        ):
            return
        if F_pop.size and F_pop.shape[1] != n_obj:
            return
        if F_pf.size and F_pf.shape[1] != n_obj:
            return
        if F_off.size and F_off.shape[1] != n_obj:
            return
        if center_F is not None:
            c = np.asarray(center_F, dtype=np.float64).reshape(-1)
            if c.size != n_obj:
                return

        i, j = 0, 1
        names = list(self.task_name)
        fig, ax = plt.subplots(1, 1, figsize=(6.0, 6.0), dpi=120)

        if F_pop.size:
            ax.scatter(
                F_pop[:, i],
                F_pop[:, j],
                c="#94a3b8",
                s=22,
                alpha=0.65,
                linewidths=0,
                label=f"Population ({len(F_pop)})",
                zorder=1,
            )
        if F_off.size:
            ax.scatter(
                F_off[:, i],
                F_off[:, j],
                c="#2563eb",
                s=36,
                alpha=0.85,
                linewidths=0,
                marker="^",
                label=f"Offspring ({len(F_off)})",
                zorder=2,
            )
        if F_pf.size:
            ax.scatter(
                F_pf[:, i],
                F_pf[:, j],
                c="#dc2626",
                s=45,
                alpha=0.9,
                linewidths=0,
                marker="o",
                label=f"Pareto front ({len(F_pf)})",
                zorder=3,
            )
        if center_F is not None:
            c = np.asarray(center_F, dtype=np.float64).reshape(-1)
            ax.scatter(
                c[i],
                c[j],
                c="#16a34a",
                s=120,
                marker="*",
                edgecolors="#14532d",
                linewidths=0.8,
                label="Pareto center",
                zorder=4,
            )
        ax.set_xlabel(names[i])
        ax.set_ylabel(names[j])
        ax.grid(True, alpha=0.35)
        n_leg = (
            int(bool(F_pop.size))
            + int(bool(F_off.size))
            + int(bool(F_pf.size))
            + int(center_F is not None)
        )
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=min(4, max(1, n_leg)),
            frameon=False,
        )
        plt.tight_layout()
        self._wandb_log({"evo/objective_plot": wandb.Image(fig)}, step=step)
        plt.close(fig)

    def _print_evo_progress(
        self,
        phase: str,
        step: int,
        n_steps: int,
        pareto_front: np.ndarray,
        extras: Optional[str] = None,
        elapsed: Optional[float] = None,
        offspring_X: Optional[np.ndarray] = None,
        wandb_log_step: Optional[int] = None,
    ) -> None:
        """Print one MOEA step to stdout (mirrors :meth:`_wandb_log_evo_front_metrics`)."""
        pareto_front_size = len(pareto_front)
        line = (
            f"[EvoMTL] {phase} {step + 1:04d}/{int(n_steps):04d}"
            + " | "
            + f"PF size={pareto_front_size}"
        )

        if offspring_X is not None:
            X = np.atleast_2d(np.asarray(offspring_X, dtype=np.float64))
            if X.size:
                z_min = float(np.min(X))
                z_max = float(np.max(X))
                z_mean = float(np.mean(X))
                z_norm = float(np.linalg.norm(X))
                line += (
                    f" | offspring z min={z_min:.6f} max={z_max:.6f} "
                    f"mean={z_mean:.6f} norm={z_norm:.6f}"
                )
                if getattr(self, "_wandb_enabled", False) and wandb_log_step is not None:
                    prefix = f"evo/offspring"
                    self._wandb_log(
                        {
                            f"{prefix}/min": z_min,
                            f"{prefix}/max": z_max,
                            f"{prefix}/mean": z_mean,
                            f"{prefix}/norm": z_norm,
                        },
                        step=wandb_log_step,
                    )
        if extras:
            line += f" | {extras}"
        if elapsed is not None:
            line += f" | Time: {elapsed:.4f} | "
        parts = []
        for ti, tn in enumerate(self.task_name):
            col = pareto_front[:, ti]
            parts.append(
                f"{tn} mean={float(np.mean(col)):.4f} "
                f"min={float(np.min(col)):.4f} max={float(np.max(col)):.4f}"
            )
        line += " | ".join(parts)
        print(line, flush=True)

    def run_nsga2(
        self,
        train_dataloaders: Any,
        num_iterations: int,
        pop_size: int,
        lb: float = -3.0,
        ub: float = 3.0,
        num_batches: int = 5,
        seed: int = 0,
        hv_ref_point: Optional[np.ndarray] = None,
        hv_center_aggregation: str = "linear",
        hv_softmax_temperature: float = 1.0,
        wandb_step_offset: int = 0,
        test_dataloaders: Optional[Any] = None,
        evo_eval_freq: int = 1,
        crossover_prob: float = 0.9,
        crossover_eta: float = 10.0,
        mutation_prob: float = 0.9,
        mutation_eta: float = 10.0,
    ) -> Dict[str, Any]:
        """NSGA-II (pymoo) with ask-and-tell on latent vectors ``z``.

        Parameters
        ----------
        evo_eval_freq : int
            Full test-set evaluation every this many generations (1 = every gen).
            ``<= 0`` skips test during the MOEA loop (Pareto-center train snapshot only).
        crossover_prob, crossover_eta
            SBX simulated-binary crossover (pymoo ``SBX``).
        mutation_prob, mutation_eta
            Polynomial mutation (pymoo ``PM``).
        """
        if self.adapter is None:
            raise RuntimeError("init_parameter_sharing() first.")

        n_var = int(self.adapter.num_dims)
        n_obj = self.task_num

        xl = np.full(n_var, lb, dtype=np.float64)
        xu = np.full(n_var, ub, dtype=np.float64)
        problem = Problem(
            n_var=n_var,
            n_obj=n_obj,
            n_constr=0,
            xl=xl,
            xu=xu,
        )
        from pymoo.operators.crossover.sbx import SBX
        from pymoo.operators.mutation.pm import PM
        algorithm = NSGA2(
            pop_size=pop_size,
            crossover=SBX(eta=float(crossover_eta), prob=float(crossover_prob)),
            mutation=PM(eta=float(mutation_eta), prob=float(mutation_prob)),
            seed=seed,
            verbose=False,
        )
        algorithm.setup(problem, termination=NoTermination())

        num_iterations = int(num_iterations)
        for iteration in range(num_iterations):
            t0 = time.perf_counter()
            
            shared_batches = self._sample_batches(train_dataloaders, num_batches)
            if algorithm.pop is not None:
                pop_F = algorithm.pop.get("F")
                pop_X = algorithm.pop.get("X")
                for i in range(len(pop_X)):
                    self.adapt_parameters(pop_X[i])
                    pop_F[i] = self.evaluate_multiobjective_losses(
                        train_dataloaders, prefetched_batches=shared_batches
                    )
                algorithm.pop.set("F", pop_F)
            offspring = algorithm.ask()
            offspring_X = offspring.get("X")
            offspring_F = np.zeros((len(offspring_X), self.task_num))
            for i in range(len(offspring_X)):
                self.adapt_parameters(offspring_X[i])
                offspring_F[i] = self.evaluate_multiobjective_losses(
                    train_dataloaders, prefetched_batches=shared_batches
                )
            static = StaticProblem(problem, F=offspring_F)
            Evaluator().eval(static, offspring)
            algorithm.tell(infills=offspring)
            result = algorithm.result()
            pareto_front = result.opt.get("F")
            pareto_latents = result.opt.get("X")

            elapsed = time.perf_counter() - t0
            self._print_evo_progress(
                "NSGA-II iteration",
                iteration,
                num_iterations,
                pareto_front,
                elapsed=elapsed,
                offspring_X=offspring_X,
                wandb_log_step=wandb_step_offset + iteration,
            )
            self._wandb_log_evo_front_metrics(
                iteration, pareto_front, wandb_step_offset + iteration
            )

            # --- Pareto-center evaluation on current batch + test set ---
            hv_weights = marginal_hypervolume_weights(
                pareto_front,
                ref_point=hv_ref_point,
                aggregation=hv_center_aggregation,
                softmax_temperature=hv_softmax_temperature,
            )
            pareto_center_z = pareto_center_from_weights(pareto_latents, hv_weights)
            self.adapt_parameters(pareto_center_z)
            train_batch_metrics = self.metrics_snapshot_from_prefetched_batches(
                shared_batches
            )
            population_F = np.atleast_2d(
                np.asarray(algorithm.pop.get("F"), dtype=np.float64)
            )
            if _moea_should_run_full_test(iteration, num_iterations, evo_eval_freq):
                self._wandb_log_evo_objective_plot(
                    population_F=population_F,
                    pareto_F=np.asarray(pareto_front, dtype=np.float64),
                    center_F=np.asarray(train_batch_metrics["loss_item"], dtype=np.float64),
                    offspring_F=np.asarray(offspring_F, dtype=np.float64),
                    step=wandb_step_offset + iteration,
                )
            run_test = test_dataloaders is not None and _moea_should_run_full_test(
                iteration, num_iterations, evo_eval_freq
            )
            if run_test:
                test_metrics = self.test(
                    test_dataloaders,
                    epoch=None,
                    mode="test",
                    wandb_log_step=wandb_step_offset + iteration,
                    suppress_display=True,
                    return_metrics=True,
                )
                self.print_compact_train_test_line(
                    train_batch_metrics,
                    test_metrics,
                    epoch_label=iteration + 1,
                )
            else:
                parts = " | ".join(
                    f"{task}: {float(train_batch_metrics['loss_item'][i]):.4f}"
                    for i, task in enumerate(self.task_name)
                )
                print(
                    f"[EvoMTL] Pareto z TRAIN (curr batch) iteration {iteration + 1:04d} | {parts}",
                    flush=True,
                )

        pop = algorithm.pop
        F_all = pop.get("F")
        X_all = pop.get("X")
        nd_front_indices = pareto_front_indices(F_all)
        pareto_F = F_all[nd_front_indices]
        pareto_X = X_all[nd_front_indices]

        hv_weights = marginal_hypervolume_weights(
            pareto_F,
            ref_point=hv_ref_point,
            aggregation=hv_center_aggregation,
            softmax_temperature=hv_softmax_temperature,
        )
        if hv_ref_point is None:
            hv_ref_point = _default_hv_ref_point(pareto_F)
        pareto_center_X = pareto_center_from_weights(pareto_X, hv_weights)

        self.adapt_parameters(pareto_center_X)
        out = {
            "method": "NSGA2",
            "algorithm": algorithm,
            "pareto_F": pareto_F,
            "pareto_X": pareto_X,
            "population_F": F_all,
            "population_X": X_all,
            "hv_weights": hv_weights,
            "pareto_center_X": pareto_center_X,
            "hv_ref_point": np.asarray(hv_ref_point, dtype=np.float64),
            "hv_center_aggregation": hv_center_aggregation,
            "hv_softmax_temperature": float(hv_softmax_temperature),
        }
        self.last_evo = out
        return out

    def run_mopso(
        self,
        train_dataloaders: Any,
        num_iterations: int,
        pop_size: int,
        lb: float = -3.0,
        ub: float = 3.0,
        num_batches: int = 5,
        seed: int = 0,
        hv_ref_point: Optional[np.ndarray] = None,
        hv_center_aggregation: str = "linear",
        hv_softmax_temperature: float = 1.0,
        wandb_step_offset: int = 0,
        test_dataloaders: Optional[Any] = None,
        evo_eval_freq: int = 1,
        pso_w: float = 0.6,
        pso_c1: float = 2.0,
        pso_c2: float = 2.0,
        pso_max_velocity_rate: float = 0.5,
        pso_archive_size: int = 200,
    ) -> Dict[str, Any]:
        """MOPSO-CD (pymoo :class:`~pymoo.algorithms.moo.mopso_cd.MOPSO_CD`) ask-and-tell on ``z``.

        Crowding-distance multi-objective PSO; same outer loop contract as :meth:`run_nsga2`.
        """
        if self.adapter is None:
            raise RuntimeError("init_parameter_sharing() first.")

        n_var = int(self.adapter.num_dims)
        n_obj = self.task_num

        xl = np.full(n_var, lb, dtype=np.float64)
        xu = np.full(n_var, ub, dtype=np.float64)
        problem = Problem(
            n_var=n_var,
            n_obj=n_obj,
            n_constr=0,
            xl=xl,
            xu=xu,
        )
        algorithm = MOPSO_CD(
            pop_size=pop_size,
            w=float(pso_w),
            c1=float(pso_c1),
            c2=float(pso_c2),
            max_velocity_rate=float(pso_max_velocity_rate),
            archive_size=int(pso_archive_size),
            seed=seed,
            verbose=False,
        )
        algorithm.setup(problem, termination=NoTermination())

        num_iterations = int(num_iterations)
        for iteration in range(num_iterations):
            t0 = time.perf_counter()

            shared_batches = self._sample_batches(train_dataloaders, num_batches)
            if algorithm.pop is not None:
                pop_F = algorithm.pop.get("F")
                pop_X = algorithm.pop.get("X")
                for i in range(len(pop_X)):
                    self.adapt_parameters(pop_X[i])
                    pop_F[i] = self.evaluate_multiobjective_losses(
                        train_dataloaders, prefetched_batches=shared_batches
                    )
                algorithm.pop.set("F", pop_F)
            offspring = algorithm.ask()
            offspring_X = offspring.get("X")
            offspring_F = np.zeros((len(offspring_X), self.task_num))
            for i in range(len(offspring_X)):
                self.adapt_parameters(offspring_X[i])
                offspring_F[i] = self.evaluate_multiobjective_losses(
                    train_dataloaders, prefetched_batches=shared_batches
                )
            static = StaticProblem(problem, F=offspring_F)
            Evaluator().eval(static, offspring)
            algorithm.tell(infills=offspring)
            result = algorithm.result()
            pareto_front = result.opt.get("F")
            pareto_latents = result.opt.get("X")

            elapsed = time.perf_counter() - t0
            self._print_evo_progress(
                "MOPSO-CD iteration",
                iteration,
                num_iterations,
                pareto_front,
                elapsed=elapsed,
                offspring_X=offspring_X,
                wandb_log_step=wandb_step_offset + iteration,
            )
            self._wandb_log_evo_front_metrics(
                iteration, pareto_front, wandb_step_offset + iteration
            )

            # --- Pareto-center evaluation on current batch + test set ---
            hv_weights = marginal_hypervolume_weights(
                pareto_front,
                ref_point=hv_ref_point,
                aggregation=hv_center_aggregation,
                softmax_temperature=hv_softmax_temperature,
            )
            pareto_center_z = pareto_center_from_weights(pareto_latents, hv_weights)
            self.adapt_parameters(pareto_center_z)
            train_batch_metrics = self.metrics_snapshot_from_prefetched_batches(
                shared_batches
            )
            population_F = np.atleast_2d(
                np.asarray(algorithm.pop.get("F"), dtype=np.float64)
            )
            if _moea_should_run_full_test(iteration, num_iterations, evo_eval_freq):
                self._wandb_log_evo_objective_plot(
                    population_F=population_F,
                    pareto_F=np.asarray(pareto_front, dtype=np.float64),
                    center_F=np.asarray(train_batch_metrics["loss_item"], dtype=np.float64),
                    offspring_F=np.asarray(offspring_F, dtype=np.float64),
                    step=wandb_step_offset + iteration,
                )
            run_test = test_dataloaders is not None and _moea_should_run_full_test(
                iteration, num_iterations, evo_eval_freq
            )
            if run_test:
                test_metrics = self.test(
                    test_dataloaders,
                    epoch=None,
                    mode="test",
                    wandb_log_step=wandb_step_offset + iteration,
                    suppress_display=True,
                    return_metrics=True,
                )
                self.print_compact_train_test_line(
                    train_batch_metrics,
                    test_metrics,
                    epoch_label=iteration + 1,
                )
            else:
                parts = " | ".join(
                    f"{task}: {float(train_batch_metrics['loss_item'][i]):.4f}"
                    for i, task in enumerate(self.task_name)
                )
                print(
                    f"[EvoMTL] Pareto z TRAIN (curr batch) iteration {iteration + 1:04d} | {parts}",
                    flush=True,
                )

        pop = algorithm.pop
        F_all = pop.get("F")
        X_all = pop.get("X")
        nd_front_indices = pareto_front_indices(F_all)
        pareto_F = F_all[nd_front_indices]
        pareto_X = X_all[nd_front_indices]

        hv_weights = marginal_hypervolume_weights(
            pareto_F,
            ref_point=hv_ref_point,
            aggregation=hv_center_aggregation,
            softmax_temperature=hv_softmax_temperature,
        )
        if hv_ref_point is None:
            hv_ref_point = _default_hv_ref_point(pareto_F)
        pareto_center_X = pareto_center_from_weights(pareto_X, hv_weights)

        self.adapt_parameters(pareto_center_X)
        out = {
            "method": "MOPSO-CD",
            "algorithm": algorithm,
            "pareto_F": pareto_F,
            "pareto_X": pareto_X,
            "population_F": F_all,
            "population_X": X_all,
            "hv_weights": hv_weights,
            "pareto_center_X": pareto_center_X,
            "hv_ref_point": np.asarray(hv_ref_point, dtype=np.float64),
            "hv_center_aggregation": hv_center_aggregation,
            "hv_softmax_temperature": float(hv_softmax_temperature),
        }
        self.last_evo = out
        return out

    def run_comocma(
        self,
        train_dataloaders: Any,
        num_iterations: int,
        num_kernels: int = 5,
        sigma0: float = 0.3,
        reference_point: Optional[List[float]] = None,
        num_batches: int = 5,
        hv_ref_point: Optional[np.ndarray] = None,
        hv_center_aggregation: str = "linear",
        hv_softmax_temperature: float = 1.0,
        cma_opts: Optional[Dict[str, Any]] = None,
        lb: float = -3.0,
        ub: float = 3.0,
        seed: int = 0,
        wandb_step_offset: int = 0,
        test_dataloaders: Optional[Any] = None,
        evo_eval_freq: int = 1,
    ) -> Dict[str, Any]:
        """COMO-CMA-ES (comocma) ask-and-tell; **two objectives only**.

        See :meth:`run_nsga2` for ``evo_eval_freq``.
        """
        if self.task_num != 2:
            raise ValueError(
                "run_comocma() is only supported for two tasks in this build "
                "(see comocma / pycomocma)."
            )
        if self.adapter is None:
            raise RuntimeError("init_parameter_sharing() first.")

        try:
            import comocma
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "run_comocma requires comocma and cma. pip install comocma cma"
            ) from e

        np.random.seed(int(seed))

        n_var = int(self.adapter.num_dims)
        z0 = self.adapter.z0
        if hasattr(z0, "tolist"):
            z0_list = z0.tolist()
        else:
            z0_list = list(z0)

        x_starts = [list(z0_list) for _ in range(int(num_kernels))]
        inopts = dict(cma_opts or {})
        # inopts.setdefault("popsize", 24)
        # inopts.setdefault("seed", int(seed))
        lo, hi = float(lb), float(ub)
        inopts.setdefault("bounds", [[lo] * n_var, [hi] * n_var])
        # # Fixed coordinate scaling for cma (1.0 on every latent dim); not configurable.
        # inopts["CMA_stds"] = [1.0] * n_var
        list_of_solvers = comocma.get_cmas(x_starts, sigma0, inopts=inopts)

        if reference_point is None:
            probe = self.evaluate_multiobjective_losses(
                train_dataloaders, n_batches=num_batches
            )
            reference_point = [float(probe[0] * 2 + 1.0), float(probe[1] * 2 + 1.0)]

        moes = comocma.Sofomore(list_of_solvers, reference_point)

        num_iterations = int(num_iterations)
        for iteration in range(num_iterations):
            t0 = time.perf_counter()
            solutions = moes.ask("all")
            solutions_X = np.asarray(solutions, dtype=np.float64)
            shared_batches = self._sample_batches(train_dataloaders, num_batches)
            solution_objectives: List[List[float]] = []
            for x in solutions:
                self.adapt_parameters(x)
                L = self.evaluate_multiobjective_losses(
                    train_dataloaders, prefetched_batches=shared_batches
                )
                solution_objectives.append([float(L[0]), float(L[1])])
            moes.tell(solutions, solution_objectives)
            solutions_F = np.asarray(solution_objectives, dtype=np.float64)
            pareto_front_cut = np.asarray(moes.pareto_front_cut, dtype=np.float64)
            pareto_set_cut = np.asarray(moes.pareto_set_cut, dtype=np.float64)

            elapsed = time.perf_counter() - t0
            wandb_iteration_extras: Optional[Dict[str, Any]] = None
            progress_extras: Optional[str] = None
            if moes.kernels:
                sig = float(np.mean([float(k.sigma) for k in moes.kernels]))
                wandb_iteration_extras = {
                    "evo/comocma/sigma_mean": sig,
                }
                progress_extras = f"sigma_mean={sig:.4f}"
            else:
                progress_extras = None
            self._print_evo_progress(
                "COMO-CMA",
                iteration,
                num_iterations,
                pareto_front_cut,
                extras=progress_extras,
                elapsed=elapsed,
                offspring_X=solutions_X,
                wandb_log_step=wandb_step_offset + iteration,
            )
            
            # if len(pareto_front_cut) > 0:
            #     self.print_pareto_front_objective_stats(
            #         pareto_front_cut,
            #         line_label="Pareto front (ref cut)",
            #     )
            self._wandb_log_evo_front_metrics(
                iteration,
                pareto_front_cut,
                wandb_step_offset + iteration,
                extras=wandb_iteration_extras,
            )

            # --- Pareto-center evaluation on current batch + test set ---
            center_objectives_F: Optional[np.ndarray] = None
            if len(pareto_front_cut) > 0:
                hv_weights = marginal_hypervolume_weights(
                    pareto_front_cut,
                    ref_point=hv_ref_point,
                    aggregation=hv_center_aggregation,
                    softmax_temperature=hv_softmax_temperature,
                )
                pareto_center_z = pareto_center_from_weights(
                    pareto_set_cut, hv_weights
                )
                self.adapt_parameters(pareto_center_z)
                train_batch_metrics = self.metrics_snapshot_from_prefetched_batches(
                    shared_batches
                )
                center_objectives_F = np.asarray(
                    train_batch_metrics["loss_item"], dtype=np.float64
                )
                run_test = test_dataloaders is not None and _moea_should_run_full_test(
                    iteration, num_iterations, evo_eval_freq
                )
                if run_test:
                    test_metrics = self.test(
                        test_dataloaders,
                        epoch=None,
                        mode="test",
                        wandb_log_step=wandb_step_offset + iteration,
                        suppress_display=True,
                        return_metrics=True,
                    )
                    self.print_compact_train_test_line(
                        train_batch_metrics,
                        test_metrics,
                        epoch_label=iteration + 1,
                    )
                else:
                    parts = " | ".join(
                        f"{task}: {float(train_batch_metrics['loss_item'][i]):.4f}"
                        for i, task in enumerate(self.task_name)
                    )
                    print(
                        f"[EvoMTL] Pareto z TRAIN (curr batch) iteration {iteration + 1:04d} | {parts}",
                        flush=True,
                    )

            if _moea_should_run_full_test(iteration, num_iterations, evo_eval_freq):
                self._wandb_log_evo_objective_plot(
                    population_F=solutions_F,
                    pareto_F=pareto_front_cut,
                    center_F=center_objectives_F,
                    offspring_F=None,
                    step=wandb_step_offset + iteration,
                )

        pareto_front_cut = np.asarray(moes.pareto_front_cut, dtype=np.float64)
        pareto_set_cut = np.asarray(moes.pareto_set_cut, dtype=np.float64)
        if len(pareto_front_cut) == 0:
            raise RuntimeError("comocma returned an empty Pareto front.")

        hv_weights = marginal_hypervolume_weights(
            pareto_front_cut,
            ref_point=hv_ref_point,
            aggregation=hv_center_aggregation,
            softmax_temperature=hv_softmax_temperature,
        )
        if hv_ref_point is None:
            hv_ref_point = _default_hv_ref_point(pareto_front_cut)
        pareto_center_X = pareto_center_from_weights(pareto_set_cut, hv_weights)

        self.adapt_parameters(pareto_center_X)
        out = {
            "method": "COMO-CMA-ES",
            "moes": moes,
            "pareto_F": pareto_front_cut,
            "pareto_X": pareto_set_cut,
            "hv_weights": hv_weights,
            "pareto_center_X": pareto_center_X,
            "hv_ref_point": np.asarray(hv_ref_point, dtype=np.float64),
            "hv_center_aggregation": hv_center_aggregation,
            "hv_softmax_temperature": float(hv_softmax_temperature),
            "reference_point": reference_point,
        }
        self.last_evo = out
        return out

    def print_pareto_front_objective_stats(
        self,
        pareto_F: np.ndarray,
        *,
        line_label: str = "Pareto front",
    ) -> None:
        """Print mean / min / max of each MOEA objective (training loss) on the Pareto set."""
        F = np.atleast_2d(np.asarray(pareto_F, dtype=np.float64))
        if len(F) == 0:
            print(f"[EvoMTL] {line_label} | (empty)", flush=True)
            return
        n_obj = F.shape[1]
        if n_obj != self.task_num:
            raise ValueError(
                f"pareto_F has {n_obj} objectives but task_num={self.task_num}"
            )
        parts = []
        for j, name in enumerate(self.task_name):
            col = F[:, j]
            parts.append(
                f"{name}: mean={float(np.mean(col)):.4f} "
                f"min={float(np.min(col)):.4f} max={float(np.max(col)):.4f}"
            )
        print(f"[EvoMTL] {line_label} | " + " | ".join(parts), flush=True)

    def evolve_then_train(
        self,
        train_dataloaders: Any,
        test_dataloaders: Any,
        gd_epochs: int,
        adapter_class: Type[Any],
        moea: str = "nsga2",
        evo_kwargs: Optional[Dict[str, Any]] = None,
        adapter_kwargs: Optional[Dict[str, Any]] = None,
        adapter_alpha: float = 1.0,
        val_dataloaders: Any = None,
        finish_wandb: bool = True,
    ) -> Dict[str, Any]:
        """MOEA on ``z`` from initial ``theta_0`` → Pareto-center weights → full-space GD.

        ``theta_0`` is the model at initialization (after any ``load_path`` restore).
        After MOEA, :meth:`adapt_parameters` leaves the network at the Pareto-center
        ``theta_0 + adapter.forward(pareto_center_X, ...)``; gradient steps then update all parameters freely
        (parameter sharing is not used during GD).

        Checkpoints: ``evo_result.pt`` and ``model_evo.pt`` are written after the MOEA
        phase (Pareto-center state before GD). Standard GD saves ``best.pt`` under
        :attr:`Trainer.save_path` when applicable.
        """
        evo_kwargs = dict(evo_kwargs or {})
        adapter_kwargs = dict(adapter_kwargs or {})
        moea_key = moea.lower()

        w = getattr(self, "weighting", None)
        w_part = f", weighting={w}" if w is not None else ""
        print(
            f"[EvoMTL] MOEA-first schedule: {moea_key} on initial weights, "
            f"then GD for {int(gd_epochs)} epoch(s){w_part}.",
            flush=True,
        )

        self.prepare_evolution(
            adapter_class, adapter_alpha=adapter_alpha, **adapter_kwargs
        )

        n_outer = int(evo_kwargs.get("num_iterations", 0))

        evo_kwargs.setdefault("wandb_step_offset", 1)
        wandb_base = int(evo_kwargs["wandb_step_offset"])
        post_evo_step = wandb_base + max(0, n_outer)

        if moea_key == "nsga2":
            evo_out = self.run_nsga2(
                train_dataloaders, test_dataloaders=test_dataloaders, **evo_kwargs
            )
        elif moea_key in ("mopso", "mopso_cd", "mopso-cd"):
            evo_out = self.run_mopso(
                train_dataloaders, test_dataloaders=test_dataloaders, **evo_kwargs
            )
        elif moea_key in ("comocma", "mocma", "mo-cma-es"):
            evo_out = self.run_comocma(
                train_dataloaders, test_dataloaders=test_dataloaders, **evo_kwargs
            )
        else:
            raise ValueError(f"Unknown moea backend: {moea}")

        if getattr(self, "_wandb_enabled", False):
            zc = np.asarray(evo_out["pareto_center_X"], dtype=np.float64).reshape(-1)
            self._wandb_log(
                {"evo/final_pareto_center_X_l2": float(np.linalg.norm(zc))},
                step=post_evo_step,
            )

        if moea_key not in ("comocma", "mocma", "mo-cma-es"):
            self.print_pareto_front_objective_stats(evo_out["pareto_F"])

        train_eval_metrics = self.test(
            train_dataloaders,
            epoch=None,
            mode="train",
            wandb_log_step=post_evo_step,
            random_minibatch=True,
            suppress_display=True,
            return_metrics=True,
        )
        test_eval_metrics = self.test(
            test_dataloaders,
            epoch=None,
            mode="test",
            wandb_log_step=post_evo_step,
            suppress_display=True,
            return_metrics=True,
        )
        self.print_compact_train_test_line(
            train_eval_metrics,
            test_eval_metrics,
            epoch_label=post_evo_step,
        )

        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
            torch.save(
                {
                    "theta_0": self.theta_0,
                    "pareto_center_X": torch.as_tensor(evo_out["pareto_center_X"]),
                    "pareto_F": evo_out["pareto_F"],
                    "pareto_X": evo_out["pareto_X"],
                    "hv_weights": evo_out["hv_weights"],
                    "hv_ref_point": evo_out["hv_ref_point"],
                    "hv_center_aggregation": evo_out.get("hv_center_aggregation"),
                    "hv_softmax_temperature": evo_out.get("hv_softmax_temperature"),
                    "method": evo_out["method"],
                    "schedule": "moea_then_gd",
                },
                os.path.join(self.save_path, "evo_result.pt"),
            )
            torch.save(
                self.model.state_dict(),
                os.path.join(self.save_path, "model_evo.pt"),
            )

        print(
            f"[EvoMTL] Switching from MOEA to full-parameter GD: "
            f"{int(gd_epochs)} epoch(s){w_part}.",
            flush=True,
        )

        _prev = self.kwargs.get("evo_args")
        self.kwargs["evo_args"] = {"evo_training": False}
        prev_off = getattr(self, "_wandb_step_offset", 0)
        # Post-MOEA train/test above uses wandb_log_step=post_evo_step; GD x-axis continues after it.
        self._wandb_step_offset = int(post_evo_step) + 1
        try:
            self.train(
                train_dataloaders,
                test_dataloaders,
                int(gd_epochs),
                val_dataloaders=val_dataloaders,
                finish_wandb=False,
            )
        finally:
            self._wandb_step_offset = prev_off
            if _prev is None:
                self.kwargs.pop("evo_args", None)
            else:
                self.kwargs["evo_args"] = _prev

        self._wandb_finish_if(finish_wandb)
        return evo_out

    def train_then_evolve(
        self,
        train_dataloaders: Any,
        test_dataloaders: Any,
        gd_epochs: int,
        adapter_class: Type[Any],
        moea: str = "nsga2",
        evo_kwargs: Optional[Dict[str, Any]] = None,
        adapter_kwargs: Optional[Dict[str, Any]] = None,
        adapter_alpha: float = 1.0,
        val_dataloaders: Any = None,
        finish_wandb: bool = True,
    ) -> Dict[str, Any]:
        """GD warmup → snapshot → adapter → MOEA → Pareto-center train/test (same as :meth:`Trainer.test`).

        Evo checkpoints (``evo_result.pt``, ``model_evo.pt``) use :attr:`Trainer.save_path`
        (same directory as ``best.pt``).

        Parameters
        ----------
        moea : str
            ``"nsga2"``, ``"mopso"`` / ``"mopso_cd"``, or ``"comocma"``.
        evo_kwargs : dict
            Passed to :meth:`run_nsga2`, :meth:`run_mopso`, or :meth:`run_comocma`.
        """
        evo_kwargs = dict(evo_kwargs or {})
        adapter_kwargs = dict(adapter_kwargs or {})

        # GD phase only: mask config so Trainer.train does not recurse into MOEA.
        _prev = self.kwargs.get('evo_args')
        self.kwargs['evo_args'] = {'evo_training': False}
        try:
            self.train(
                train_dataloaders,
                test_dataloaders,
                gd_epochs,
                val_dataloaders=val_dataloaders,
                finish_wandb=False,
            )
        finally:
            if _prev is None:
                self.kwargs.pop('evo_args', None)
            else:
                self.kwargs['evo_args'] = _prev

        moea_key = moea.lower()
        w = getattr(self, "weighting", None)
        w_part = f", weighting={w}" if w is not None else ""
        print(
            f"[EvoMTL] Switching from GD warmup to MOEA ({moea_key}): "
            f"finished {gd_epochs} epoch(s){w_part}.",
            flush=True,
        )

        self.prepare_evolution(
            adapter_class, adapter_alpha=adapter_alpha, **adapter_kwargs
        )

        n_outer = int(evo_kwargs.get("num_iterations", 0))

        # After GD, last wandb step is ``gd_epochs`` (1-based epochs); evo starts at ``gd_epochs + 1``.
        evo_kwargs.setdefault("wandb_step_offset", int(gd_epochs) + 1)
        wandb_base = int(evo_kwargs["wandb_step_offset"])
        post_evo_step = wandb_base + max(0, n_outer)

        if moea_key == "nsga2":
            evo_out = self.run_nsga2(train_dataloaders, test_dataloaders=test_dataloaders, **evo_kwargs)
        elif moea_key in ("mopso", "mopso_cd", "mopso-cd"):
            evo_out = self.run_mopso(train_dataloaders, test_dataloaders=test_dataloaders, **evo_kwargs)
        elif moea_key in ("comocma", "mocma", "mo-cma-es"):
            evo_out = self.run_comocma(train_dataloaders, test_dataloaders=test_dataloaders, **evo_kwargs)
        else:
            raise ValueError(f"Unknown moea backend: {moea}")

        if getattr(self, "_wandb_enabled", False):
            zc = np.asarray(evo_out["pareto_center_X"], dtype=np.float64).reshape(-1)
            self._wandb_log(
                {"evo/final_pareto_center_X_l2": float(np.linalg.norm(zc))},
                step=post_evo_step,
            )

        # Pareto-center weights are already applied in run_nsga2 / run_mopso / run_comocma.
        # COMO-CMA prints ref-cut Pareto stats each iteration; pymoo MOEAs only here.
        if moea_key not in ("comocma", "mocma", "mo-cma-es"):
            self.print_pareto_front_objective_stats(evo_out["pareto_F"])

        train_eval_metrics = self.test(
            train_dataloaders,
            epoch=None,
            mode="train",
            wandb_log_step=post_evo_step,
            random_minibatch=True,
            suppress_display=True,
            return_metrics=True,
        )
        test_eval_metrics = self.test(
            test_dataloaders,
            epoch=None,
            mode="test",
            wandb_log_step=post_evo_step,
            suppress_display=True,
            return_metrics=True,
        )
        self.print_compact_train_test_line(
            train_eval_metrics,
            test_eval_metrics,
            epoch_label=post_evo_step,
        )

        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
            torch.save(
                {
                    "theta_0": self.theta_0,
                    "pareto_center_X": torch.as_tensor(evo_out["pareto_center_X"]),
                    "pareto_F": evo_out["pareto_F"],
                    "pareto_X": evo_out["pareto_X"],
                    "hv_weights": evo_out["hv_weights"],
                    "hv_ref_point": evo_out["hv_ref_point"],
                    "hv_center_aggregation": evo_out.get("hv_center_aggregation"),
                    "hv_softmax_temperature": evo_out.get("hv_softmax_temperature"),
                    "method": evo_out["method"],
                },
                os.path.join(self.save_path, "evo_result.pt"),
            )
            torch.save(self.model.state_dict(), os.path.join(self.save_path, "model_evo.pt"))

        self._wandb_finish_if(finish_wandb)
        return evo_out


def pareto_center_solution(
    F: np.ndarray,
    X: np.ndarray,
    ref_point: Optional[np.ndarray] = None,
    *,
    hv_center_aggregation: str = "linear",
    hv_softmax_temperature: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Utility: Pareto front + HV weights + center ``pareto_center_X`` (no model side effects)."""
    F = np.atleast_2d(F)
    X = np.atleast_2d(X)
    idx = pareto_front_indices(F)
    F_nd = F[idx]
    X_nd = X[idx]
    w = marginal_hypervolume_weights(
        F_nd,
        ref_point=ref_point,
        aggregation=hv_center_aggregation,
        softmax_temperature=hv_softmax_temperature,
    )
    pareto_center_X = pareto_center_from_weights(X_nd, w)
    return pareto_center_X, w
