"""Evolutionary multi-objective training on low-dimensional parameter-sharing codes.

Workflow:
1. Gradient-descent warmup (reuse :meth:`Trainer.train`).
2. Snapshot ``theta_0`` and build a parameter-sharing map ``ps`` so that
   ``theta = theta_0 + ps_scale * ps(z)`` (same convention as in ``parameter_sharing``).
3. Run NSGA-II (pymoo) or COMO-CMA-ES / MO-CMA-ES (comocma) with an ask-and-tell loop.
4. Form a *Pareto-center* latent vector by averaging Pareto-set points weighted by
   marginal hypervolume contribution, apply it, then evaluate train and test (same
   path as :meth:`Trainer.test`) and save.

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

from ..trainer import Trainer
from .parameter_sharing import (
    DictLoRAParameterSharing,
    FlattenLoRAParameterSharing,
    LayerwiseRandomProjectionParameterSharing,
    LayerwiseScaledRandomProjectionParameterSharing,
    LinearOnlyLoRA,
    ModulationLoRA,
    RandomProjectionParameterSharing,
    SpectralAllSVD,
    SpectralLoRA,
)

EVO_PS_REGISTRY = {
    "random_proj": RandomProjectionParameterSharing,
    "layerwise_random_proj": LayerwiseRandomProjectionParameterSharing,
    "layerwise_scaled_random_proj": LayerwiseScaledRandomProjectionParameterSharing,
    "flatten_lora": FlattenLoRAParameterSharing,
    "spherical_lora": FlattenLoRAParameterSharing,
    "dict_lora": DictLoRAParameterSharing,
    "linear_lora": LinearOnlyLoRA,
    "modulation_lora": ModulationLoRA,
    "spectral_lora": SpectralLoRA,
    "spectral_all_svd": SpectralAllSVD,
}


def resolve_ps_class(name: str) -> Type[Any]:
    """Map :mod:`LibMTL.config` ``--evo_ps`` string to a parameter-sharing class."""
    key = name.strip().lower().replace("-", "_")
    if key not in EVO_PS_REGISTRY:
        raise ValueError(
            f"Unknown evo_ps {name!r}. Options: {sorted(EVO_PS_REGISTRY)}"
        )
    return EVO_PS_REGISTRY[key]


try:
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
      There is no separate ``SphericalLoRA`` class — use ``FlattenLoRAParameterSharing``
      (``spherical_lora``) or another strategy from ``LibMTL.evomtl.parameter_sharing``.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.theta_0: Optional[torch.Tensor] = None
        self.ps: Any = None
        self.ps_scale: float = 1.0
        self.last_evo: Dict[str, Any] = {}

    def capture_base_parameters(self) -> None:
        """Store ``theta_0`` as a flat vector on the trainer device."""
        self.theta_0 = nn.utils.parameters_to_vector(self.model.parameters()).detach().clone()

    def init_parameter_sharing(
        self,
        ps_class: Type[Any],
        ps_scale: float = 1.0,
        capture_if_needed: bool = True,
        **ps_kwargs: Any,
    ) -> None:
        """Instantiate parameter sharing on the current model.

        If ``theta_0`` was never captured and ``capture_if_needed`` is True,
        the current weights are stored as ``theta_0`` first.
        """
        if self.theta_0 is None and capture_if_needed:
            self.capture_base_parameters()
        ps_kwargs = dict(ps_kwargs)
        ps_kwargs.setdefault("device", str(self.device))
        self.ps = ps_class(self.model, **ps_kwargs)
        self.ps_scale = float(ps_scale)

    def _print_and_log_moea_ps_stats(self) -> None:
        """Print model size and MOEA latent dimensionality; add to W&B run config if enabled."""
        if self.ps is None:
            return
        num_parameters = int(sum(p.numel() for p in self.model.parameters()))
        num_dimensions = int(self.ps.num_dims)
        print(
            f"[EvoMTL] MOEA (parameter sharing): num_parameters={num_parameters} | "
            f"num_dimensions={num_dimensions}",
            flush=True,
        )
        if getattr(self, "_wandb_enabled", False):
            import wandb

            wandb.config.update(
                {
                    "evo_num_parameters": num_parameters,
                    "evo_num_dimensions": num_dimensions,
                },
                allow_val_change=True,
            )

    def prepare_evolution(
        self,
        ps_class: Type[Any],
        ps_scale: float = 1.0,
        **ps_kwargs: Any,
    ) -> None:
        """Capture ``theta_0`` (if missing) and build ``ps``."""
        self.init_parameter_sharing(ps_class, ps_scale=ps_scale, **ps_kwargs)
        self._print_and_log_moea_ps_stats()

    def apply_z(
        self,
        z: Union[np.ndarray, List[float], torch.Tensor],
        ps_scale: Optional[float] = None,
    ) -> None:
        """Set model weights to ``theta_0 + ps_scale * ps(z)``."""
        if self.theta_0 is None:
            raise RuntimeError("Call capture_base_parameters() before apply_z().")
        if self.ps is None:
            raise RuntimeError("Call init_parameter_sharing() before apply_z().")

        scale = self.ps_scale if ps_scale is None else float(ps_scale)
        z_np = np.asarray(z, dtype=np.float64).reshape(-1)
        delta = self.ps.forward(z_np)
        if not isinstance(delta, torch.Tensor):
            delta = torch.as_tensor(
                np.asarray(delta, dtype=np.float32),
                device=self.device,
                dtype=torch.float32,
            )
        else:
            delta = delta.to(self.device).float()
        full = self.theta_0 + scale * delta.reshape_as(self.theta_0)
        nn.utils.vector_to_parameters(full, self.model.parameters())

    def restore_base_parameters(self) -> None:
        if self.theta_0 is None:
            raise RuntimeError("No base snapshot; run capture_base_parameters() first.")
        nn.utils.vector_to_parameters(self.theta_0, self.model.parameters())

    def _sample_eval_batches(
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
        :meth:`_sample_eval_batches`), those exact batches are reused instead of
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
        """Restrict to the Pareto front, compute HV weights, return ``z_center``."""
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
        z_c = pareto_center_from_weights(X_nd, w)
        return z_c, w, idx

    def _wandb_log_evo_front_metrics(
        self,
        iterations: int,
        F: np.ndarray,
        step: int,
        extras: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log one MOEA step under ``evo/`` (shared keys for NSGA-II and COMO-CMA-ES)."""
        if not getattr(self, "_wandb_enabled", False):
            return
        F = np.atleast_2d(np.asarray(F, dtype=np.float64))
        ideal = np.min(F, axis=0)
        idx_pf = pareto_front_indices(F)
        F_pf = F[idx_pf]
        nadir = np.max(F_pf, axis=0) if len(F_pf) else np.max(F, axis=0)
        log: Dict[str, Any] = {
            "evo/iterations": int(iterations),
            "evo/pareto_front_size": float(len(F_pf)),
        }
        if F.size:
            # Mean task loss per individual, then mean over population (like train losses/avg).
            log["evo/losses/avg"] = float(np.mean(np.mean(F, axis=1)))
        for ti, tn in enumerate(self.task_name):
            log[f"evo/{tn}/mean"] = float(np.mean(F[:, ti]))
            log[f"evo/{tn}/ideal"] = float(ideal[ti])
            log[f"evo/{tn}/nadir"] = float(nadir[ti])
        if extras:
            log.update(extras)
        self._wandb_log(log, step=step)

    def _print_evo_progress(
        self,
        phase: str,
        step: int,
        n_steps: int,
        F: np.ndarray,
        extras: Optional[str] = None,
        elapsed: Optional[float] = None,
        pareto_front_size: Optional[int] = None,
    ) -> None:
        """Print one MOEA step to stdout (mirrors :meth:`_wandb_log_evo_front_metrics`)."""
        F = np.atleast_2d(np.asarray(F, dtype=np.float64))
        if pareto_front_size is None:
            pareto_front_size = len(pareto_front_indices(F))
        parts = []
        for ti, tn in enumerate(self.task_name):
            col = F[:, ti]
            parts.append(
                f"{tn} mean={float(np.mean(col)):.4f} "
                f"min={float(np.min(col)):.4f} max={float(np.max(col)):.4f}"
            )
        line = (
            f"[EvoMTL] {phase} {step + 1:04d}/{int(n_steps):04d} | "
            + " | ".join(parts)
            + f" | pareto_front={pareto_front_size}"
        )
        if extras:
            line += f" | {extras}"
        if elapsed is not None:
            line += f" | Time: {elapsed:.4f}"
        print(line, flush=True)

    def run_nsga2(
        self,
        train_dataloaders: Any,
        n_gen: int,
        pop_size: int,
        z_lower: float = -3.0,
        z_upper: float = 3.0,
        n_eval_batches: int = 5,
        seed: int = 0,
        hv_ref_point: Optional[np.ndarray] = None,
        hv_center_aggregation: str = "linear",
        hv_softmax_temperature: float = 1.0,
        wandb_step_offset: int = 0,
        test_dataloaders: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """NSGA-II (pymoo) with ask-and-tell on latent vectors ``z``."""
        if self.ps is None:
            raise RuntimeError("init_parameter_sharing() first.")

        n_var = int(self.ps.num_dims)
        n_obj = self.task_num

        xl = np.full(n_var, z_lower, dtype=np.float64)
        xu = np.full(n_var, z_upper, dtype=np.float64)
        problem = Problem(
            n_var=n_var,
            n_obj=n_obj,
            n_constr=0,
            xl=xl,
            xu=xu,
        )
        algorithm = NSGA2(pop_size=pop_size)
        algorithm.setup(problem, termination=NoTermination(), seed=seed, verbose=False)
        np.random.seed(seed)

        history_F: List[np.ndarray] = []

        n_gen = int(n_gen)
        for gen_idx in range(n_gen):
            t0 = time.perf_counter()
            pop = algorithm.ask()
            X = pop.get("X")
            shared_batches = self._sample_eval_batches(train_dataloaders, n_eval_batches)
            F_list = []
            for i in range(len(X)):
                self.apply_z(X[i])
                F_list.append(
                    self.evaluate_multiobjective_losses(
                        train_dataloaders, prefetched_batches=shared_batches
                    )
                )
            F = np.stack(F_list, axis=0)
            static = StaticProblem(problem, F=F)
            Evaluator().eval(static, pop)
            algorithm.tell(infills=pop)
            history_F.append(F.copy())

            elapsed = time.perf_counter() - t0
            self._print_evo_progress(
                "NSGA-II gen", gen_idx, n_gen, F, elapsed=elapsed
            )
            self._wandb_log_evo_front_metrics(
                gen_idx, F, wandb_step_offset + gen_idx
            )

            # --- Pareto-center evaluation on current batch + test set ---
            _F_pop = algorithm.pop.get("F")
            _X_pop = algorithm.pop.get("X")
            _idx_pf = pareto_front_indices(_F_pop)
            _w = marginal_hypervolume_weights(
                _F_pop[_idx_pf],
                ref_point=hv_ref_point,
                aggregation=hv_center_aggregation,
                softmax_temperature=hv_softmax_temperature,
            )
            _z_c = pareto_center_from_weights(_X_pop[_idx_pf], _w)
            self.apply_z(_z_c)
            _train_snap = self.metrics_snapshot_from_prefetched_batches(shared_batches)
            if test_dataloaders is not None:
                _m_test = self.test(
                    test_dataloaders,
                    epoch=None,
                    mode="test",
                    wandb_log_step=wandb_step_offset + gen_idx,
                    suppress_display=True,
                    return_metrics=True,
                )
                self.print_compact_train_test_line(
                    _train_snap,
                    _m_test,
                    epoch_label=gen_idx + 1,
                )
            else:
                parts = " | ".join(
                    f"{task}: {float(_train_snap['loss_item'][i]):.4f}"
                    for i, task in enumerate(self.task_name)
                )
                print(f"[EvoMTL] Pareto z TRAIN (curr batch) gen {gen_idx + 1:04d} | {parts}", flush=True)

        pop = algorithm.pop
        F_all = pop.get("F")
        X_all = pop.get("X")
        idx = pareto_front_indices(F_all)
        F_nd = F_all[idx]
        X_nd = X_all[idx]

        w = marginal_hypervolume_weights(
            F_nd,
            ref_point=hv_ref_point,
            aggregation=hv_center_aggregation,
            softmax_temperature=hv_softmax_temperature,
        )
        if hv_ref_point is None:
            hv_ref_point = _default_hv_ref_point(F_nd)
        z_center = pareto_center_from_weights(X_nd, w)

        self.apply_z(z_center)
        out = {
            "method": "NSGA2",
            "algorithm": algorithm,
            "pareto_F": F_nd,
            "pareto_X": X_nd,
            "population_F": F_all,
            "population_X": X_all,
            "hv_weights": w,
            "z_center": z_center,
            "hv_ref_point": np.asarray(hv_ref_point, dtype=np.float64),
            "hv_center_aggregation": hv_center_aggregation,
            "hv_softmax_temperature": float(hv_softmax_temperature),
            "history_F": history_F,
        }
        self.last_evo = out
        return out

    def run_comocma(
        self,
        train_dataloaders: Any,
        n_iterations: int,
        num_kernels: int = 5,
        sigma0: float = 0.3,
        reference_point: Optional[List[float]] = None,
        n_eval_batches: int = 5,
        hv_ref_point: Optional[np.ndarray] = None,
        hv_center_aggregation: str = "linear",
        hv_softmax_temperature: float = 1.0,
        cma_opts: Optional[Dict[str, Any]] = None,
        z_lower: float = -3.0,
        z_upper: float = 3.0,
        seed: int = 0,
        wandb_step_offset: int = 0,
        test_dataloaders: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """COMO-CMA-ES (comocma) ask-and-tell; **two objectives only**."""
        if self.task_num != 2:
            raise ValueError(
                "run_comocma() is only supported for two tasks in this build "
                "(see comocma / pycomocma)."
            )
        if self.ps is None:
            raise RuntimeError("init_parameter_sharing() first.")

        try:
            import comocma
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "run_comocma requires comocma and cma. pip install comocma cma"
            ) from e

        np.random.seed(int(seed))

        n_var = int(self.ps.num_dims)
        z0 = self.ps.z0
        if hasattr(z0, "tolist"):
            z0_list = z0.tolist()
        else:
            z0_list = list(z0)

        x_starts = [list(z0_list) for _ in range(int(num_kernels))]
        inopts = dict(cma_opts or {})
        # inopts.setdefault("popsize", 24)
        # inopts.setdefault("seed", int(seed))
        lo, hi = float(z_lower), float(z_upper)
        inopts.setdefault("bounds", [[lo] * n_var, [hi] * n_var])
        # # Fixed coordinate scaling for cma (1.0 on every latent dim); not configurable.
        # inopts["CMA_stds"] = [1.0] * n_var
        list_of_solvers = comocma.get_cmas(x_starts, sigma0, inopts=inopts)

        if reference_point is None:
            probe = self.evaluate_multiobjective_losses(
                train_dataloaders, n_batches=n_eval_batches
            )
            reference_point = [float(probe[0] * 2 + 1.0), float(probe[1] * 2 + 1.0)]

        moes = comocma.Sofomore(list_of_solvers, reference_point)
        history_F: List[np.ndarray] = []

        n_iterations = int(n_iterations)
        for it in range(n_iterations):
            t0 = time.perf_counter()
            solutions = moes.ask("all")
            shared_batches = self._sample_eval_batches(train_dataloaders, n_eval_batches)
            objs = []
            for x in solutions:
                self.apply_z(x)
                L = self.evaluate_multiobjective_losses(
                    train_dataloaders, prefetched_batches=shared_batches
                )
                objs.append([float(L[0]), float(L[1])])
            moes.tell(solutions, objs)
            objs_arr = np.asarray(objs, dtype=np.float64)
            history_F.append(objs_arr.copy())

            elapsed = time.perf_counter() - t0
            extras = None
            extras_print = None
            pf_n = len(moes.pareto_front_cut)
            if moes.kernels:
                sig = float(np.mean([float(k.sigma) for k in moes.kernels]))
                extras = {
                    "evo/comocma/sigma_mean": sig,
                    "evo/comocma/pareto_front_size": float(pf_n),
                }
                extras_print = f"sigma_mean={sig:.4f}"
            else:
                extras = {"evo/comocma/pareto_front_size": float(pf_n)}
                extras_print = None
            self._print_evo_progress(
                "COMO-CMA",
                it,
                n_iterations,
                objs_arr,
                extras=extras_print,
                elapsed=elapsed,
                pareto_front_size=pf_n,
            )
            pf_cut = np.asarray(moes.pareto_front_cut, dtype=np.float64)
            ps_cut = np.asarray(moes.pareto_set_cut, dtype=np.float64)
            if len(pf_cut) > 0:
                self.print_pareto_front_objective_stats(
                    pf_cut,
                    line_label="Pareto front (ref cut)",
                )
            self._wandb_log_evo_front_metrics(
                it, objs_arr, wandb_step_offset + it, extras=extras
            )

            # --- Pareto-center evaluation on current batch + test set ---
            if len(pf_cut) > 0:
                _w = marginal_hypervolume_weights(
                    pf_cut,
                    ref_point=hv_ref_point,
                    aggregation=hv_center_aggregation,
                    softmax_temperature=hv_softmax_temperature,
                )
                _z_c = pareto_center_from_weights(ps_cut, _w)
                self.apply_z(_z_c)
                _train_snap = self.metrics_snapshot_from_prefetched_batches(shared_batches)
                if test_dataloaders is not None:
                    _m_test = self.test(
                        test_dataloaders,
                        epoch=None,
                        mode="test",
                        wandb_log_step=wandb_step_offset + it,
                        suppress_display=True,
                        return_metrics=True,
                    )
                    self.print_compact_train_test_line(
                        _train_snap,
                        _m_test,
                        epoch_label=it + 1,
                    )
                else:
                    parts = " | ".join(
                        f"{task}: {float(_train_snap['loss_item'][i]):.4f}"
                        for i, task in enumerate(self.task_name)
                    )
                    print(f"[EvoMTL] Pareto z TRAIN (curr batch) it {it + 1:04d} | {parts}", flush=True)

        F_nd = np.asarray(moes.pareto_front_cut, dtype=np.float64)
        X_nd = np.asarray(moes.pareto_set_cut, dtype=np.float64)
        if len(F_nd) == 0:
            raise RuntimeError("comocma returned an empty Pareto front.")

        w = marginal_hypervolume_weights(
            F_nd,
            ref_point=hv_ref_point,
            aggregation=hv_center_aggregation,
            softmax_temperature=hv_softmax_temperature,
        )
        if hv_ref_point is None:
            hv_ref_point = _default_hv_ref_point(F_nd)
        z_center = pareto_center_from_weights(X_nd, w)

        self.apply_z(z_center)
        out = {
            "method": "COMO-CMA-ES",
            "moes": moes,
            "pareto_F": F_nd,
            "pareto_X": X_nd,
            "hv_weights": w,
            "z_center": z_center,
            "hv_ref_point": np.asarray(hv_ref_point, dtype=np.float64),
            "hv_center_aggregation": hv_center_aggregation,
            "hv_softmax_temperature": float(hv_softmax_temperature),
            "reference_point": reference_point,
            "history_F": history_F,
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

    def train_then_evolve(
        self,
        train_dataloaders: Any,
        test_dataloaders: Any,
        gd_epochs: int,
        ps_class: Type[Any],
        moea: str = "nsga2",
        evo_kwargs: Optional[Dict[str, Any]] = None,
        ps_kwargs: Optional[Dict[str, Any]] = None,
        ps_scale: float = 1.0,
        val_dataloaders: Any = None,
        finish_wandb: bool = True,
    ) -> Dict[str, Any]:
        """GD warmup → snapshot → PS → MOEA → Pareto-center train/test (same as :meth:`Trainer.test`).

        Evo checkpoints (``evo_result.pt``, ``model_evo.pt``) use :attr:`Trainer.save_path`
        (same directory as ``best.pt``).

        Parameters
        ----------
        moea : str
            ``"nsga2"`` or ``"comocma"``.
        evo_kwargs : dict
            Passed to :meth:`run_nsga2` or :meth:`run_comocma`.
        """
        evo_kwargs = dict(evo_kwargs or {})
        ps_kwargs = dict(ps_kwargs or {})

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

        self.prepare_evolution(ps_class, ps_scale=ps_scale, **ps_kwargs)

        if moea_key == "nsga2":
            n_outer = int(evo_kwargs.get("n_gen", 0))
        elif moea_key in ("comocma", "mocma", "mo-cma-es"):
            n_outer = int(evo_kwargs.get("n_iterations", 0))
        else:
            n_outer = 0

        evo_kwargs.setdefault("wandb_step_offset", int(gd_epochs))
        wandb_base = int(evo_kwargs["wandb_step_offset"])
        post_evo_step = wandb_base + max(0, n_outer)

        if moea_key == "nsga2":
            evo_out = self.run_nsga2(train_dataloaders, test_dataloaders=test_dataloaders, **evo_kwargs)
        elif moea_key in ("comocma", "mocma", "mo-cma-es"):
            evo_out = self.run_comocma(train_dataloaders, test_dataloaders=test_dataloaders, **evo_kwargs)
        else:
            raise ValueError(f"Unknown moea backend: {moea}")

        if getattr(self, "_wandb_enabled", False):
            zc = np.asarray(evo_out["z_center"], dtype=np.float64).reshape(-1)
            self._wandb_log(
                {"evo/final_z_center_l2": float(np.linalg.norm(zc))},
                step=post_evo_step,
            )

        # Pareto-center weights are already applied in run_nsga2 / run_comocma.
        # COMO-CMA prints ref-cut Pareto stats each iteration; NSGA-II only here.
        if moea_key not in ("comocma", "mocma", "mo-cma-es"):
            self.print_pareto_front_objective_stats(evo_out["pareto_F"])

        m_train = self.test(
            train_dataloaders,
            epoch=None,
            mode="train",
            wandb_log_step=post_evo_step,
            random_minibatch=True,
            suppress_display=True,
            return_metrics=True,
        )
        m_test = self.test(
            test_dataloaders,
            epoch=None,
            mode="test",
            wandb_log_step=post_evo_step,
            suppress_display=True,
            return_metrics=True,
        )
        self.print_compact_train_test_line(
            m_train,
            m_test,
            epoch_label=post_evo_step,
        )

        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
            torch.save(
                {
                    "theta_0": self.theta_0,
                    "z_center": torch.as_tensor(evo_out["z_center"]),
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
    """Utility: Pareto front + HV weights + center ``z`` (no model side effects)."""
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
    z_c = pareto_center_from_weights(X_nd, w)
    return z_c, w
