"""COMO-CMA-ES (comocma) ask-and-tell; two objectives only in this build."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np

from .common import configure_moea_random_seed, should_run_full_test
from .hv import (
    default_hv_ref_point,
    marginal_hypervolume_weights,
    pareto_center_from_weights,
)


def run_comocma(
    trainer: Any,
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

    See :func:`run_nsga2` for ``evo_eval_freq``.
    """
    if trainer.task_num != 2:
        raise ValueError(
            "run_comocma() is only supported for two tasks in this build "
            "(see comocma / pycomocma)."
        )
    if trainer.adapter is None:
        raise RuntimeError("init_parameter_sharing() first.")

    try:
        import comocma
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "run_comocma requires comocma and cma. pip install comocma cma"
        ) from e

    configure_moea_random_seed(seed)

    n_var = int(trainer.adapter.num_dims)
    z0 = trainer.adapter.z0
    if hasattr(z0, "tolist"):
        z0_list = z0.tolist()
    else:
        z0_list = list(z0)

    x_starts = [list(z0_list) for _ in range(int(num_kernels))]
    inopts = dict(cma_opts or {})
    lo, hi = float(lb), float(ub)
    inopts.setdefault("bounds", [[lo] * n_var, [hi] * n_var])
    list_of_solvers = comocma.get_cmas(x_starts, sigma0, inopts=inopts)

    if reference_point is None:
        probe = trainer.evaluate_multiobjective_losses(
            train_dataloaders, n_batches=num_batches
        )
        reference_point = [float(probe[0] * 2 + 1.0), float(probe[1] * 2 + 1.0)]

    moes = comocma.Sofomore(list_of_solvers, reference_point)

    num_iterations = int(num_iterations)
    for iteration in range(num_iterations):
        t0 = time.perf_counter()
        solutions = moes.ask("all")
        solutions_X = np.asarray(solutions, dtype=np.float64)
        shared_batches = trainer._sample_batches(train_dataloaders, num_batches)
        solution_objectives: List[List[float]] = []
        for x in solutions:
            trainer.adapt_parameters(x)
            L = trainer.evaluate_multiobjective_losses(
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
        trainer._print_evo_progress(
            "COMO-CMA",
            iteration,
            num_iterations,
            pareto_front_cut,
            extras=progress_extras,
            elapsed=elapsed,
            offspring_X=solutions_X,
            wandb_log_step=wandb_step_offset + iteration,
        )
        trainer._wandb_log_evo_front_metrics(
            iteration,
            pareto_front_cut,
            wandb_step_offset + iteration,
            extras=wandb_iteration_extras,
        )

        center_objectives_F: Optional[np.ndarray] = None
        if len(pareto_front_cut) > 0:
            hv_weights = marginal_hypervolume_weights(
                pareto_front_cut,
                ref_point=hv_ref_point,
                aggregation=hv_center_aggregation,
                softmax_temperature=hv_softmax_temperature,
            )
            pareto_center_X = pareto_center_from_weights(
                pareto_set_cut, hv_weights
            )
            trainer.adapt_parameters(pareto_center_X)
            train_batch_metrics = trainer.metrics_snapshot_from_prefetched_batches(
                shared_batches
            )
            center_objectives_F = np.asarray(
                train_batch_metrics["loss_item"], dtype=np.float64
            )
            run_test = test_dataloaders is not None and should_run_full_test(
                iteration, num_iterations, evo_eval_freq
            )
            if run_test:
                test_metrics = trainer.test(
                    test_dataloaders,
                    epoch=None,
                    mode="test",
                    wandb_log_step=wandb_step_offset + iteration,
                    suppress_display=True,
                    return_metrics=True,
                )
                trainer.print_compact_train_test_line(
                    train_batch_metrics,
                    test_metrics,
                    epoch_label=iteration + 1,
                )
            else:
                parts = " | ".join(
                    f"{task}: {float(train_batch_metrics['loss_item'][i]):.4f}"
                    for i, task in enumerate(trainer.task_name)
                )
                print(
                    f"[EvoMTL] Pareto z TRAIN (curr batch) iteration {iteration + 1:04d} | {parts}",
                    flush=True,
                )

        if should_run_full_test(iteration, num_iterations, evo_eval_freq):
            trainer._wandb_log_evo_objective_plot(
                population_F=solutions_F,
                pareto_F=pareto_front_cut,
                pareto_center_F=center_objectives_F,
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
        hv_ref_point = default_hv_ref_point(pareto_front_cut)
    pareto_center_X = pareto_center_from_weights(pareto_set_cut, hv_weights)

    trainer.adapt_parameters(pareto_center_X)
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
    trainer.last_evo = out
    return out
