"""Evolutionary multi-objective training of the network weights ``theta``.

Every algorithm searches the flat weight vector ``theta`` directly (order of ``model.parameters()``),
starting from the current weights ``theta_0``.

Two schedules (``evo_schedule``):

* ``evolution`` (:meth:`EvoMTLTrainer.evolve`): multi-objective evolutionary training from the
  initial weights, steps 2–4 below. Either a Pareto front (NSGA-II, MOPSO-CD, COMO-CMA,
  PSL-ES, Tangent PSL-ES, Dual-cone PSL-ES) or a single network from an evolution strategy (OpenES / GuidedES on the
  sum of losses, or MO-UPGrad-OpenES / MO-MGDA-OpenES and their Guided ES variants on the per-objective
  pseudo-gradients).
* ``gd_then_evolution`` (:meth:`EvoMTLTrainer.train_then_evolve`, default): multi-objective
  gradient descent (step 1), then the same evolutionary training as fine-tuning that shapes the
  Pareto front.

1. Gradient descent with the configured weighting (:meth:`Trainer.train`), ``gd_then_evolution`` only.
2. Snapshot the starting weights ``theta_0``.
3. Run NSGA-II, stochastic NSGA-II (SNSGA-II), MOPSO-CD (pymoo), COMO-CMA-ES / MO-CMA-ES (comocma),
   or an OpenES-family ES (e.g. MO-UPGrad-OpenES: per-objective OpenES pseudo-gradients aggregated with UPGrad)
   with an ask-and-tell loop.
4. Form a *Pareto-center* weight vector by averaging Pareto-set points weighted by
   marginal hypervolume contribution, apply it, then evaluate train and test (same
   path as :meth:`Trainer.test`) and save.

References:
- pymoo ask-and-tell: https://www.pymoo.org/algorithms/usage.html#nb-algorithms-ask-and-tell
- comocma: https://github.com/CMA-ES/pycomocma
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

from ..trainer import Trainer
from .moea.common import epoch_fraction_eval_steps
from .moea.comocma import run_comocma
from .moea.hv import (
    marginal_hypervolume_weights,
    pareto_center_from_weights,
    pareto_center_solution,
    pareto_center_weights_from_marginal_hv,
    pareto_front_indices,
)
from .moea.guidedes import run_guidedes
from .moea.multi_center_pareto_openes import run_multi_center_pareto_openes
from .moea.mo_mean_guidedes import run_mo_mean_guidedes
from .moea.mo_cagrad_guidedes import run_mo_cagrad_guidedes
from .moea.mo_cagrad_openes import run_mo_cagrad_openes
from .moea.mo_mean_openes import run_mo_mean_openes
from .moea.mo_mgda2_guidedes import run_mo_mgda2_guidedes
from .moea.mo_mgda2_openes import run_mo_mgda2_openes
from .moea.mo_mgda_guidedes import run_mo_mgda_guidedes
from .moea.mo_mgda_openes import run_mo_mgda_openes
from .moea.mo_mgda_openes_libmtl import run_mo_mgda_openes_libmtl
from .moea.mo_upgrad_guidedes import run_mo_upgrad_guidedes
from .moea.mo_upgrad_openes import run_mo_upgrad_openes
from .moea.mopso import run_mopso
from .moea.openes import run_openes
from .moea.pgpe import run_pgpe
from .moea.psl_es import run_psl_es
from .moea.snes import run_snes
from .moea.tpsl_es import run_tpsl_es
from .moea.dcpsl_es import run_dcpsl_es
from .moea.psl_taskguide import run_psl_taskguide
from .moea.nsga2 import run_nsga2
from .moea.sc_de import run_sc_de

# Single-network evolution strategies, by evo_moea key (hyphens are accepted for underscores)
ES_RUNNERS = {
    "openes": run_openes,
    "guidedes": run_guidedes,
    "snes": run_snes,
    "pgpe": run_pgpe,
    "mo_upgrad_openes": run_mo_upgrad_openes,
    "mo_upgrad_guidedes": run_mo_upgrad_guidedes,
    "mo_mean_openes": run_mo_mean_openes,
    "mo_mean_guidedes": run_mo_mean_guidedes,
    "mo_mgda_openes": run_mo_mgda_openes,
    "mo_mgda_guidedes": run_mo_mgda_guidedes,
    "mo_mgda_openes_libmtl": run_mo_mgda_openes_libmtl,
    "mo_cagrad_openes": run_mo_cagrad_openes,
    "mo_cagrad_guidedes": run_mo_cagrad_guidedes,
    "mo_mgda2_openes": run_mo_mgda2_openes,
    "mo_mgda2_guidedes": run_mo_mgda2_guidedes,
}
PSL_ES_KEYS = ("psl_es", "psl-es")
TPSL_ES_KEYS = ("tpsl_es", "tpsl-es")
DCPSL_ES_KEYS = ("dcpsl_es", "dcpsl-es")
PSL_TASKGUIDE_KEYS = ("psl_taskguide", "psl-taskguide", "psl_splitguide", "psl-splitguide")
SC_DE_KEYS = ("sc_de", "sc-de")


class EvoMTLTrainer(Trainer):
    """Trainer with MOEA phases on the network weights ``theta``.

    After GD warmup, call :meth:`prepare_evolution` (snapshots ``theta_0``) then MOEA runners in
    :mod:`LibMTL.evomtl.moea` (e.g. :func:`~LibMTL.evomtl.moea.nsga2.run_nsga2`), which install
    candidate weights with :meth:`set_theta`.

    Notes
    -----
    * ``comocma`` is only used for **two** objectives in upstream tests; this
      class raises if ``task_num != 2`` for :func:`~LibMTL.evomtl.moea.comocma.run_comocma`.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.theta_0: Optional[torch.Tensor] = None
        self.last_evo: Dict[str, Any] = {}

    def train(
        self,
        train_dataloaders: Any,
        test_dataloaders: Any,
        epochs: int,
        val_dataloaders: Any = None,
        return_weight: bool = False,
        finish_wandb: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Run the configured ``evo_schedule`` (``evo_args`` from :func:`LibMTL.config.prepare_args`).

        * ``evolution``: :meth:`evolve`, multi-objective evolutionary training from the initial
          weights (``epochs`` must be 0);
        * ``gd_then_evolution``: :meth:`train_then_evolve`, ``epochs`` of multi-objective gradient
          descent, then evolutionary fine-tuning of the Pareto front.

        Without ``evo_training`` this is plain :meth:`Trainer.train`.
        """
        evo_args = self.kwargs.get("evo_args") or {}
        if not evo_args.get("evo_training"):
            return super().train(train_dataloaders, test_dataloaders, epochs, val_dataloaders,
                                 return_weight, finish_wandb=finish_wandb, **kwargs)
        schedule = evo_args.get("evo_schedule", "gd_then_evolution")
        self._evo_eval_per_epoch = int(evo_args.get("evo_eval_per_epoch", 0) or 0)
        evo = dict(
            moea=evo_args["moea"],
            evo_kwargs=dict(evo_args["evo_kwargs"]),
            finish_wandb=finish_wandb,
        )
        if schedule == "evolution":
            if int(epochs) != 0:
                raise ValueError(f"evo_schedule evolution runs no gradient descent, but epochs={epochs}; "
                                 "set epochs: 0 or use gd_then_evolution.")
            return self.evolve(train_dataloaders, test_dataloaders, **evo)
        if schedule == "gd_then_evolution":
            return self.train_then_evolve(train_dataloaders, test_dataloaders, gd_epochs=epochs,
                                          val_dataloaders=val_dataloaders, **evo)
        raise ValueError(f"Unknown evo_schedule {schedule!r}; options: evolution, gd_then_evolution")

    def prepare_evolution(self) -> None:
        """Store the current weights as ``theta_0`` (flat vector on the trainer device), the starting point
        of the search; print the number of weights and add it to the W&B run config if enabled."""
        self.theta_0 = nn.utils.parameters_to_vector(self.model.parameters()).detach().clone()
        num_parameters = int(self.theta_0.numel())
        print(f"[EvoMTL] MOEA on the weights theta: num_parameters={num_parameters}", flush=True)
        if getattr(self, "_wandb_enabled", False):
            wandb.config.update({"evo_num_parameters": num_parameters}, allow_val_change=True)

    def set_theta(self, theta: Union[np.ndarray, torch.Tensor]) -> None:
        """Install the flat weight vector ``theta`` (order of ``model.parameters()``) into the model."""
        theta = torch.as_tensor(theta, dtype=torch.float32).reshape(-1).to(self.device)
        nn.utils.vector_to_parameters(theta, self.model.parameters())

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
        record_losses: bool = True,
    ) -> np.ndarray:
        """Mean per-task training loss over up to ``n_batches`` mini-batches.

        If ``prefetched_batches`` is provided (a list returned by
        :meth:`_sample_batches`), those exact batches are reused instead of
        drawing new ones.  Pass pre-fetched batches so that all solutions in a
        single MOEA iteration are evaluated on the **same** mini-batch.
        ``record_losses=False`` excludes disposable probes from meter histories.
        """
        # ES probes must not enter the training meter or its loss history.
        loss_forward = self.forward4loss if record_losses else self._compute_losses_no_record
        self.model.eval()

        if prefetched_batches is not None:
            acc = torch.zeros(self.task_num, device=self.device)
            for inp, gts in prefetched_batches:
                losses = loss_forward(self.model, inp, gts)
                acc += losses.detach()
            return (acc / len(prefetched_batches)).cpu().numpy()

        train_loader, train_batch = self._prepare_dataloaders(train_dataloaders)
        n_batches = int(max(1, n_batches))

        if not self.multi_input:
            n = min(n_batches, train_batch)
            acc = torch.zeros(self.task_num, device=self.device)
            for _ in range(n):
                inp, gts = self._process_data(train_loader)
                losses = loss_forward(self.model, inp, gts)
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
            losses = loss_forward(self.model, train_inputs, train_gts)
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
        pf = np.asarray(pareto_front, dtype=np.float32)
        if pf.size == 0:
            pf = pf.reshape(0, int(self.task_num))
        elif pf.ndim == 1:
            pf = pf[np.newaxis, :]
        pareto_front = pf
        n_pf = len(pareto_front)
        if n_pf == 0:
            return
        log: Dict[str, Any] = {
            "evo/iterations": int(iterations),
            "evo/pareto_front_size": float(n_pf),
        }
        ideal = np.min(pareto_front, axis=0)
        nadir = np.max(pareto_front, axis=0)
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
        pareto_center_F: Optional[np.ndarray],
        offspring_F: Optional[np.ndarray] = None,
        step: int,
    ) -> None:
        """Single 2D scatter: population, offspring, Pareto front, Pareto-center snapshot."""
        if not getattr(self, "_wandb_enabled", False):
            return
        n_obj = int(self.task_num)
        if n_obj < 2:
            return
        F_pop = np.atleast_2d(np.asarray(population_F, dtype=np.float32))
        F_pf = np.atleast_2d(np.asarray(pareto_F, dtype=np.float32))
        if offspring_F is not None:
            F_off = np.atleast_2d(np.asarray(offspring_F, dtype=np.float32))
            if F_off.size == 0:
                F_off = np.empty((0, n_obj), dtype=np.float32)
        else:
            F_off = np.empty((0, n_obj), dtype=np.float32)
        if (
            F_pop.size == 0
            and F_pf.size == 0
            and pareto_center_F is None
            and F_off.size == 0
        ):
            return
        if F_pop.size and F_pop.shape[1] != n_obj:
            return
        if F_pf.size and F_pf.shape[1] != n_obj:
            return
        if F_off.size and F_off.shape[1] != n_obj:
            return
        if pareto_center_F is not None:
            c = np.asarray(pareto_center_F, dtype=np.float32).reshape(-1)
            if c.size != n_obj:
                return
        
        i, j = 0, 1
        names = list(self.task_name)
        fig, ax = plt.subplots(1, 1, figsize=(6.0, 6.0))

        if F_pop.size:
            ax.scatter(
                F_pop[:, i],
                F_pop[:, j],
                s=30,
                alpha=0.9,
                # marker="v",
                facecolors="#94a3b8",
                # edgecolors="black",
                label=f"Population ({len(F_pop)})",
                zorder=1,
            )
        if F_off.size:
            ax.scatter(
                F_off[:, i],
                F_off[:, j],
                c="#94a3b8",
                s=30,
                alpha=0.7,
                linewidths=0,
                # marker="v",
                facecolors="#94a3b8",
                edgecolors="none",
                label=f"Offspring ({len(F_off)})",
                zorder=2,
            )
        if F_pf.size:
            ax.scatter(
                F_pf[:, i],
                F_pf[:, j],
                c="#dc2626",
                s=100,
                alpha=0.7,
                linewidths=0,
                marker="^",
                label=f"Pareto Front ({len(F_pf)})",
                zorder=3,
            )
        if pareto_center_F is not None:
            c = np.asarray(pareto_center_F, dtype=np.float32).reshape(-1)
            ax.scatter(
                c[i],
                c[j],
                c="#16a34a",
                s=150,
                marker="*",
                edgecolors="#14532d",
                linewidths=0.8,
                label="Pareto Center",
                zorder=4,
            )
        ax.set_xlabel(names[i])
        ax.set_ylabel(names[j])
        # ax.set_xscale("log")
        # ax.set_yscale("log")
        # ax.set_xlim(0.0, 10.0)
        # ax.set_ylim(0.0, 10.0)
        ax.grid(True, alpha=0.25)
        # n_leg = (
        #     int(bool(F_pop.size))
        #     + int(bool(F_off.size))
        #     + int(bool(F_pf.size))
        #     + int(pareto_center_F is not None)
        # )
        # ax.legend(
        #     loc="lower center",
        #     bbox_to_anchor=(0.5, 1.02),
        #     ncol=min(4, max(1, n_leg)),
        #     frameon=False,
        # )
        ax.legend(
            loc="upper right",
            frameon=True,
            facecolor="none",
            edgecolor="none",
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
        pf = np.asarray(pareto_front, dtype=np.float32)
        if pf.size == 0:
            pf = pf.reshape(0, int(self.task_num))
        elif pf.ndim == 1:
            pf = pf[np.newaxis, :]
        pareto_front = pf
        pareto_front_size = len(pareto_front)
        line = (
            f"[EvoMTL] {phase} {step + 1:04d}/{int(n_steps):04d}"
            + " | "
            + f"PF size={pareto_front_size}"
        )

        if offspring_X is not None:
            # displacement of the offspring weights from theta_0
            X = np.atleast_2d(np.asarray(offspring_X, dtype=np.float32))
            if X.size:
                X = X - self.theta_0.detach().cpu().float().numpy()
                d_min = float(np.min(X))
                d_max = float(np.max(X))
                d_mean = float(np.mean(X))
                d_norm = float(np.linalg.norm(X))
                line += (
                    f" | offspring theta-theta_0 min={d_min:.6f} max={d_max:.6f} "
                    f"mean={d_mean:.6f} norm={d_norm:.6f}"
                )
                if getattr(self, "_wandb_enabled", False) and wandb_log_step is not None:
                    prefix = f"evo/offspring"
                    self._wandb_log(
                        {
                            f"{prefix}/min": d_min,
                            f"{prefix}/max": d_max,
                            f"{prefix}/mean": d_mean,
                            f"{prefix}/norm": d_norm,
                        },
                        step=wandb_log_step,
                    )
        if extras:
            line += f" | {extras}"
        if elapsed is not None:
            line += f" | Time: {elapsed:.4f} | "
        parts = []
        for ti, tn in enumerate(self.task_name):
            if pareto_front_size == 0:
                parts.append(f"{tn} mean=n/a min=n/a max=n/a")
            else:
                col = pareto_front[:, ti]
                parts.append(
                    f"{tn} mean={float(np.mean(col)):.4f} "
                    f"min={float(np.min(col)):.4f} max={float(np.max(col)):.4f}"
                )
        line += " | ".join(parts)
        print(line, flush=True)

    def print_pareto_front_objective_stats(
        self,
        pareto_F: np.ndarray,
        *,
        line_label: str = "Pareto front",
    ) -> None:
        """Print mean / min / max of each MOEA objective (training loss) on the Pareto set."""
        F = np.atleast_2d(np.asarray(pareto_F, dtype=np.float32))
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

    def _run_moea(
        self,
        moea_key: str,
        train_dataloaders: Any,
        test_dataloaders: Any,
        evo_kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Dispatch to the MOEA / ES backend named by ``moea_key``."""
        if moea_key.replace("-", "_") == "multi_center_pareto_openes":
            runner = run_multi_center_pareto_openes
        elif moea_key == "nsga2":
            runner = run_nsga2
        elif moea_key in ("mopso", "mopso_cd", "mopso-cd"):
            runner = run_mopso
        elif moea_key in ("comocma", "mocma", "mo-cma-es"):
            runner = run_comocma
        elif moea_key.replace("-", "_") in ES_RUNNERS:
            runner = ES_RUNNERS[moea_key.replace("-", "_")]
        elif moea_key in PSL_ES_KEYS:
            runner = run_psl_es
        elif moea_key in TPSL_ES_KEYS:
            runner = run_tpsl_es
        elif moea_key in DCPSL_ES_KEYS:
            runner = run_dcpsl_es
        elif moea_key in PSL_TASKGUIDE_KEYS:
            runner = run_psl_taskguide
        elif moea_key in SC_DE_KEYS:
            runner = run_sc_de
        else:
            raise ValueError(f"Unknown moea backend: {moea_key}")
        return runner(
            self, train_dataloaders, test_dataloaders=test_dataloaders, **evo_kwargs
        )

    def evolve(
        self,
        train_dataloaders: Any,
        test_dataloaders: Any,
        moea: str = "nsga2",
        evo_kwargs: Optional[Dict[str, Any]] = None,
        finish_wandb: bool = True,
    ) -> Dict[str, Any]:
        """Multi-objective evolutionary training from the initial weights (schedule ``evolution``).

        Covers Pareto-front methods (NSGA-II, MOPSO, COMO-CMA, PSL-ES, Tangent PSL-ES, Dual-cone PSL-ES) and single-network ES
        (OpenES, GuidedES, SNES, PGPE, MO-UPGrad-OpenES, MO-MGDA-OpenES, ...). The network ends at the weights
        ``pareto_center_X``; see :meth:`_evolution_phase` for the evaluation and checkpoints.
        """
        moea_key = moea.lower()
        print(f"[EvoMTL] Evolutionary training: {moea_key} from the initial weights.", flush=True)
        self._wandb_log_init(train_dataloaders, test_dataloaders)
        evo_out = self._evolution_phase(train_dataloaders, test_dataloaders, moea_key, evo_kwargs,
                                        wandb_step_offset=1, schedule="evolution")
        self._wandb_finish_if(finish_wandb)
        return evo_out

    def train_then_evolve(
        self,
        train_dataloaders: Any,
        test_dataloaders: Any,
        gd_epochs: int,
        moea: str = "nsga2",
        evo_kwargs: Optional[Dict[str, Any]] = None,
        val_dataloaders: Any = None,
        finish_wandb: bool = True,
    ) -> Dict[str, Any]:
        """Multi-objective gradient descent, then evolutionary fine-tuning (schedule ``gd_then_evolution``).

        ``gd_epochs`` of :meth:`Trainer.train` with the configured weighting, then the same
        evolutionary phase as :meth:`evolve` starting from the trained weights, to shape the
        Pareto front. GD checkpoints (``best.pt``) and the evolution checkpoints share
        :attr:`Trainer.save_path`.

        Parameters
        ----------
        moea : str
            ``"nsga2"``, ``"mopso"`` / ``"mopso_cd"``, ``"comocma"``, ``"psl_es"``, ``"tpsl_es"``, ``"dcpsl_es"``, or a key
            of :data:`ES_RUNNERS` (``"openes"``, ``"mo_upgrad_openes"``, ``"mo_mgda_openes"``, ...).
        evo_kwargs : dict
            Passed to the runner of ``moea`` in :mod:`LibMTL.evomtl.moea` (first arg is this trainer).
        """
        super().train(train_dataloaders, test_dataloaders, gd_epochs,
                      val_dataloaders=val_dataloaders, finish_wandb=False)

        moea_key = moea.lower()
        w = getattr(self, "weighting", None)
        w_part = f", weighting={w}" if w is not None else ""
        print(f"[EvoMTL] Switching from gradient descent ({gd_epochs} epoch(s){w_part}) to evolutionary "
              f"fine-tuning: {moea_key}.", flush=True)
        # wandb steps count iterations: GD ends at step ``gd_epochs * T`` (T iterations per epoch), and
        # generation g (1-based) of the evolution logs at ``gd_epochs * T + g``.
        gd_iterations = int(gd_epochs) * int(getattr(self, "_wandb_iters_per_epoch", None) or 1)
        evo_out = self._evolution_phase(train_dataloaders, test_dataloaders, moea_key, evo_kwargs,
                                        wandb_step_offset=gd_iterations + 1, schedule="gd_then_evolution")
        self._wandb_finish_if(finish_wandb)
        return evo_out

    def _evolution_phase(
        self,
        train_dataloaders: Any,
        test_dataloaders: Any,
        moea_key: str,
        evo_kwargs: Optional[Dict[str, Any]],
        wandb_step_offset: int,
        schedule: str,
    ) -> Dict[str, Any]:
        """Evolutionary phase of both schedules, from the current weights.

        Snapshot ``theta_0``, run the MOEA / ES (wandb steps from ``wandb_step_offset``), apply the
        Pareto-center weights, evaluate them on train (one random minibatch) and test, and write
        ``evo_result.pt`` and ``model_evo.pt`` under :attr:`Trainer.save_path`.
        """
        evo_kwargs = dict(evo_kwargs or {})

        self.prepare_evolution()

        evo_kwargs.setdefault("wandb_step_offset", int(wandb_step_offset))
        per_epoch = int(getattr(self, "_evo_eval_per_epoch", 0) or 0)
        if per_epoch > 0:
            # Test passes at fractions of an epoch of T iterations (GD ends on a multiple of T, so the
            # fractions line up with the global wandb step).
            # len() of the loaders (not _prepare_dataloaders): no iterator, so no draw from the torch RNG
            iters = getattr(self, "_wandb_iters_per_epoch", None) or (
                max(len(d) for d in train_dataloaders.values()) if isinstance(train_dataloaders, dict)
                else len(train_dataloaders))
            iters = int(iters)
            evo_kwargs["evo_eval_freq"] = epoch_fraction_eval_steps(
                int(evo_kwargs.get("num_iterations", 0)), iters, per_epoch)
            print(f"[EvoMTL] Test evaluation {per_epoch}x per epoch (T={iters}): generations "
                  f"{sorted(s + 1 for s in evo_kwargs['evo_eval_freq'])} and the last.", flush=True)
        post_evo_step = int(evo_kwargs["wandb_step_offset"]) + max(0, int(evo_kwargs.get("num_iterations", 0)))

        evo_out = self._run_moea(moea_key, train_dataloaders, test_dataloaders, evo_kwargs)

        if getattr(self, "_wandb_enabled", False):  # distance of the returned weights from theta_0
            dc = np.asarray(evo_out["pareto_center_X"], dtype=np.float32).reshape(-1) \
                - self.theta_0.detach().cpu().float().numpy()
            self._wandb_log({"evo/final_pareto_center_X_l2": float(np.linalg.norm(dc))}, step=post_evo_step)

        # Pareto-center weights are already applied by the runners. COMO-CMA prints ref-cut Pareto
        # stats each iteration; the others only here.
        if moea_key not in ("comocma", "mocma", "mo-cma-es"):
            self.print_pareto_front_objective_stats(
                evo_out["pareto_F"],
                line_label="Persistent centers" if "centers_X" in evo_out else "Pareto front")

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
        self.print_compact_train_test_line(train_eval_metrics, test_eval_metrics, epoch_label=post_evo_step)

        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
            torch.save(
                {
                    **{key: evo_out[key] for key in (
                        "centers_X", "centers_F", "center_statistics", "center_evaluations", "representative_index",
                        "sample_pool", "pool_settings", "pool_diagnostics", "last_pool"
                    ) if key in evo_out},
                    "theta_0": self.theta_0,
                    "search_space": "theta",  # X entries are weights (runs before it was set: theta - theta_0)
                    "pareto_center_X": torch.as_tensor(evo_out["pareto_center_X"], dtype=torch.float32),
                    "pareto_F": evo_out["pareto_F"],
                    "pareto_X": evo_out["pareto_X"],
                    "hv_weights": evo_out["hv_weights"],
                    "hv_ref_point": evo_out["hv_ref_point"],
                    "hv_center_aggregation": evo_out.get("hv_center_aggregation"),
                    "hv_softmax_temperature": evo_out.get("hv_softmax_temperature"),
                    "method": evo_out["method"],
                    "schedule": schedule,
                },
                os.path.join(self.save_path, "evo_result.pt"),
            )
            torch.save(self.model.state_dict(), os.path.join(self.save_path, "model_evo.pt"))
        return evo_out
