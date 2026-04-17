"""NSGA-II (pymoo) ask-and-tell on latent vectors ``z``."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np

from .common import configure_moea_random_seed, inject_pareto_center_into_pop, should_run_full_test
from .hv import (
    default_hv_ref_point,
    marginal_hypervolume_weights,
    pareto_center_from_weights,
    pareto_front_indices,
)
from .pymoo_deps import Evaluator, NSGA2, Problem, StaticProblem, NoTermination


def run_nsga2(
    trainer: Any,
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
    if trainer.adapter is None:
        raise RuntimeError("init_parameter_sharing() first.")

    configure_moea_random_seed(seed)

    n_var = int(trainer.adapter.num_dims)
    n_obj = trainer.task_num

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

        shared_batches = trainer._sample_batches(train_dataloaders, num_batches)
        if algorithm.pop is not None:
            pop_F = algorithm.pop.get("F")
            pop_X = algorithm.pop.get("X")
            for i in range(len(pop_X)):
                trainer.adapt_parameters(pop_X[i])
                pop_F[i] = trainer.evaluate_multiobjective_losses(
                    train_dataloaders, prefetched_batches=shared_batches
                )
            algorithm.pop.set("F", pop_F)
        offspring = algorithm.ask()
        offspring_X = offspring.get("X")
        offspring_F = np.zeros((len(offspring_X), trainer.task_num))
        for i in range(len(offspring_X)):
            trainer.adapt_parameters(offspring_X[i])
            offspring_F[i] = trainer.evaluate_multiobjective_losses(
                train_dataloaders, prefetched_batches=shared_batches
            )
        static = StaticProblem(problem, F=offspring_F)
        Evaluator().eval(static, offspring)
        algorithm.tell(infills=offspring)
        result = algorithm.result()
        pareto_front = result.opt.get("F")
        pareto_latents = result.opt.get("X")

        elapsed = time.perf_counter() - t0
        trainer._print_evo_progress(
            "NSGA-II iteration",
            iteration,
            num_iterations,
            pareto_front,
            elapsed=elapsed,
            offspring_X=offspring_X,
            wandb_log_step=wandb_step_offset + iteration,
        )
        trainer._wandb_log_evo_front_metrics(
            iteration, pareto_front, wandb_step_offset + iteration
        )

        hv_weights = marginal_hypervolume_weights(
            pareto_front,
            ref_point=hv_ref_point,
            aggregation=hv_center_aggregation,
            softmax_temperature=hv_softmax_temperature,
        )
        pareto_center_X = pareto_center_from_weights(pareto_latents, hv_weights)
        trainer.adapt_parameters(pareto_center_X)
        train_batch_metrics = trainer.metrics_snapshot_from_prefetched_batches(
            shared_batches
        )
        population_F = np.atleast_2d(
            np.asarray(algorithm.pop.get("F"), dtype=np.float64)
        )
        if should_run_full_test(iteration, num_iterations, evo_eval_freq):
            trainer._wandb_log_evo_objective_plot(
                population_F=population_F,
                pareto_F=np.asarray(pareto_front, dtype=np.float64),
                pareto_center_F=np.asarray(train_batch_metrics["loss_item"], dtype=np.float64),
                step=wandb_step_offset + iteration,
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

        inject_pareto_center_into_pop(
            algorithm=algorithm,
            pareto_center_X=pareto_center_X,
            pareto_center_F=train_batch_metrics["loss_item"],
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
        hv_ref_point = default_hv_ref_point(pareto_F)
    pareto_center_X = pareto_center_from_weights(pareto_X, hv_weights)

    trainer.adapt_parameters(pareto_center_X)
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
    trainer.last_evo = out
    return out
