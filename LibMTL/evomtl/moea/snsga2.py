"""Stochastic NSGA-II: EMA fitness + epsilon-dominance survival (pymoo ask-and-tell).

EMA statistics live on each :class:`~pymoo.core.individual.Individual` as custom fields
``snsga2_f_bar`` and ``snsga2_var`` (via ``Individual.set`` / ``get``), so they move with
surviving solutions across generations.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np

from .common import configure_moea_random_seed, inject_pareto_center_into_pop, should_run_full_test
from .hv import (
    default_hv_ref_point,
    marginal_hypervolume_weights,
    pareto_center_from_weights,
    pareto_front_indices,
)
from .pymoo_deps import (
    Evaluator,
    NSGA2,
    Problem,
    RankAndCrowding,
    NoTermination,
    StaticProblem,
    Survival,
)

# Per-individual EMA state (pymoo ``Individual.data`` via ``set`` / ``get``).
_SNSGA2_KEY_F_BAR = "snsga2_f_bar"
_SNSGA2_KEY_VAR = "snsga2_var"


def _fast_nds_from_domination_matrix(M: np.ndarray) -> List[np.ndarray]:
    """Non-dominated fronts from a pairwise domination matrix (pymoo-compatible)."""
    M = np.asarray(M, dtype=np.float64)
    n = M.shape[0]
    fronts: List[List[int]] = []
    if n == 0:
        return []
    is_dominating: List[List[int]] = [[] for _ in range(n)]
    n_dominated = np.zeros(n, dtype=np.float64)
    current_front: List[int] = []
    n_ranked = 0

    for i in range(n):
        for j in range(i + 1, n):
            rel = M[i, j]
            if rel == 1.0:
                is_dominating[i].append(j)
                n_dominated[j] += 1.0
            elif rel == -1.0:
                is_dominating[j].append(i)
                n_dominated[i] += 1.0
        if n_dominated[i] == 0:
            current_front.append(i)
            n_ranked += 1

    fronts.append(current_front)

    while n_ranked < n:
        next_front: List[int] = []
        for i in current_front:
            for j in is_dominating[i]:
                n_dominated[j] -= 1.0
                if n_dominated[j] == 0:
                    next_front.append(j)
                    n_ranked += 1
        fronts.append(next_front)
        current_front = next_front

    return [np.array(f, dtype=int) for f in fronts]


class _EpsilonPerObjectiveDominator:
    """Vector epsilon-dominance (minimization), matching pymoo's matrix convention."""

    def __init__(self, eps_per_obj: np.ndarray) -> None:
        self.eps = np.asarray(eps_per_obj, dtype=np.float64).reshape(1, -1)

    def calc_domination_matrix(self, F: np.ndarray) -> np.ndarray:
        F = np.asarray(F, dtype=np.float64)
        n = F.shape[0]
        L = np.repeat(F, n, axis=0)
        R = np.tile(F, (n, 1))
        e = self.eps
        smaller = np.reshape(np.any(L + e < R, axis=1), (n, n))
        larger = np.reshape(np.any(L > R + e, axis=1), (n, n))
        return (
            np.logical_and(smaller, np.logical_not(larger)).astype(np.float64)
            - np.logical_and(larger, np.logical_not(smaller)).astype(np.float64)
        )


class _EpsilonNDS:
    """Non-dominated sorting on minimization objectives with per-objective epsilon."""

    def __init__(self, eps_per_obj: np.ndarray) -> None:
        self.eps_per_obj = np.asarray(eps_per_obj, dtype=np.float64).reshape(-1)

    def do(
        self,
        F: np.ndarray,
        n_stop_if_ranked: int = 10**8,
        n_fronts: int = 10**8,
        **kwargs: Any,
    ) -> List[np.ndarray]:
        del kwargs
        F = np.asarray(F, dtype=np.float64)
        dom = _EpsilonPerObjectiveDominator(self.eps_per_obj)
        M = dom.calc_domination_matrix(F)
        fronts = _fast_nds_from_domination_matrix(M)
        _fronts: List[np.ndarray] = []
        n_ranked = 0
        for front in fronts:
            _fronts.append(np.asarray(front, dtype=int))
            n_ranked += len(front)
            if n_ranked >= n_stop_if_ranked:
                break
            if len(_fronts) >= n_fronts:
                break
        return _fronts


def _pooled_noise_eps(
    var_parents: np.ndarray,
    F_raw_parents: np.ndarray,
    eps_kappa: float,
    eps_var_floor: float,
) -> np.ndarray:
    """Per-objective epsilon ~ noise scale from EMA update variance (adaptive)."""
    var_parents = np.asarray(var_parents, dtype=np.float64)
    F_raw_parents = np.asarray(F_raw_parents, dtype=np.float64)
    m = var_parents.shape[1]
    row_mass = np.sum(var_parents, axis=1)
    mask = row_mass > 1e-30
    if np.any(mask):
        med = np.median(var_parents[mask], axis=0)
    else:
        med = np.median(var_parents, axis=0) if var_parents.size else np.zeros(m)
    # spread = np.std(F_raw_parents, axis=0) if F_raw_parents.size else np.zeros(m)
    eps = float(eps_kappa) * np.sqrt(np.maximum(med, 0.0) + float(eps_var_floor))
    # eps = np.where(eps > 1e-30, eps, float(eps_kappa) * (1e-6 + 0.01 * np.abs(spread)))
    return eps


class SNSGA2Survival(Survival):
    """NSGA-II survival on EMA-smoothed objectives with adaptive epsilon-dominance (merged parent+offspring ``pop``)."""

    def __init__(
        self,
        fitness_beta: float,
        eps_kappa: float,
        eps_var_floor: float,
    ) -> None:
        super().__init__(filter_infeasible=True)
        self._fitness_beta = float(fitness_beta)
        self._eps_kappa = float(eps_kappa)
        self._eps_var_floor = float(eps_var_floor)

    def _do(
        self,
        problem: Any,
        pop: Any,
        n_survive: Optional[int] = None,
        random_state: Any = None,
        **kwargs: Any,
    ) -> Any:
        del problem
        algorithm = kwargs.get("algorithm")
        if algorithm is None:
            raise RuntimeError("SNSGA2Survival requires keyword argument algorithm= from GeneticAlgorithm.")

        merged = pop
        parents = algorithm.pop
        parent_ids = frozenset(id(parents[i]) for i in range(len(parents)))
        F_raw = np.asarray(merged.get("F"), dtype=np.float64).copy()
        n_m, n_obj = F_raw.shape
        F_bar = np.zeros_like(F_raw)
        var_row = np.zeros_like(F_raw)
        parent_mask = np.zeros(n_m, dtype=bool)
        beta = self._fitness_beta

        for i in range(n_m):
            ind = merged[i]
            is_parent = id(ind) in parent_ids
            parent_mask[i] = is_parent
            if is_parent:
                prev_f = ind.get(_SNSGA2_KEY_F_BAR)
                prev_v = ind.get(_SNSGA2_KEY_VAR)
            else:
                prev_f, prev_v = None, None
            if prev_f is not None and prev_v is not None:
                f_old = np.asarray(prev_f, dtype=np.float64).reshape(-1)
                v_old = np.asarray(prev_v, dtype=np.float64).reshape(-1)
                res = F_raw[i] - f_old
                F_bar[i] = beta * f_old + (1.0 - beta) * F_raw[i]
                var_row[i] = beta * v_old + (1.0 - beta) * (res ** 2)
            else:
                F_bar[i] = F_raw[i]
                var_row[i] = np.zeros(n_obj, dtype=np.float64)

        eps_vec = _pooled_noise_eps(
            var_row[parent_mask],
            F_raw[parent_mask],
            eps_kappa=self._eps_kappa,
            eps_var_floor=self._eps_var_floor,
        )
        nds = _EpsilonNDS(eps_vec)
        inner = RankAndCrowding(nds=nds)
        merged.set("F", F_bar)
        if n_survive is None:
            n_survive = len(merged)
        surv_idx = inner.do(
            algorithm.problem,
            merged,
            n_survive=int(n_survive),
            random_state=random_state,
            return_indices=True,
        )
        surv_idx_arr = np.asarray(surv_idx, dtype=int)
        new_pop = merged[surv_idx_arr]
        new_pop.set("F", F_raw[surv_idx_arr])

        for si in surv_idx_arr:
            ind = merged[si]
            ind.set(_SNSGA2_KEY_F_BAR, F_bar[si].copy())
            ind.set(_SNSGA2_KEY_VAR, var_row[si].copy())

        return new_pop


def run_snsga2(
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
    fitness_beta: float = 0.9,
    eps_kappa: float = 1.0,
    eps_var_floor: float = 1e-12,
) -> Dict[str, Any]:
    """Stochastic NSGA-II (ask-and-tell) with EMA-smoothed fitness and epsilon-dominance survival.

    Same outer loop as :func:`run_nsga2`, but after initialization each generation merges
    parents and offspring, updates a per-individual EMA :math:`\\bar{f}` of raw batch losses,
    runs non-dominated sorting on :math:`\\bar{f}` with per-objective :math:`\\epsilon`
    (scaled from pooled variance of EMA innovations on parents), and truncates with NSGA-II
    crowding distance on :math:`\\bar{f}`. Offspring and any individual without history use
    :math:`\\bar{f}=f` for that step. After each survival step, :math:`\\bar{f}` and the
    per-objective variance of EMA innovations are stored on each surviving individual as
    ``snsga2_f_bar`` and ``snsga2_var``.

    Parameters
    ----------
    fitness_beta : float
        EMA coefficient :math:`\\beta` in ``[0.7, 0.95]``; larger = more smoothing.
    eps_kappa : float
        Multiplier for adaptive :math:`\\epsilon_j \\propto \\sqrt{\\mathrm{Var}_j}`.
    eps_var_floor : float
        Small constant under the radical for numerical stability.
    """
    if trainer.adapter is None:
        raise RuntimeError("init_parameter_sharing() first.")

    beta = float(fitness_beta)
    if not (0.7 <= beta <= 0.95):
        raise ValueError(f"fitness_beta must be in [0.7, 0.95], got {fitness_beta}")

    configure_moea_random_seed(seed)

    n_var = int(trainer.adapter.num_dims)
    n_obj = int(trainer.task_num)

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

    snsga2_survival = SNSGA2Survival(
        beta,
        float(eps_kappa),
        float(eps_var_floor),
    )
    algorithm = NSGA2(
        pop_size=pop_size,
        crossover=SBX(eta=float(crossover_eta), prob=float(crossover_prob)),
        mutation=PM(eta=float(mutation_eta), prob=float(mutation_prob)),
        survival=snsga2_survival,
        seed=seed,
        verbose=False,
    )
    algorithm.setup(problem, termination=NoTermination())

    def _eval_population_f_on_batches(pop: Any, batches: Any) -> None:
        """Fill ``F`` for a pymoo population (parents, offspring, or initial infills) on fixed batches."""
        pop_X = pop.get("X")
        pop_F = np.zeros((len(pop_X), trainer.task_num))
        for i in range(len(pop_X)):
            trainer.adapt_parameters(pop_X[i])
            pop_F[i] = trainer.evaluate_multiobjective_losses(
                train_dataloaders, prefetched_batches=batches
            )
        static = StaticProblem(problem, F=pop_F)
        Evaluator().eval(static, pop)

    def _post_tell_iteration(iteration: int, t0: float, offspring_X: Any, batches: Any) -> None:
        result = algorithm.result()
        pareto_front = result.opt.get("F")
        pareto_latents = result.opt.get("X")

        elapsed = time.perf_counter() - t0
        trainer._print_evo_progress(
            "SNSGA-II iteration",
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
        train_batch_metrics = trainer.metrics_snapshot_from_prefetched_batches(batches)
        population_F = np.atleast_2d(np.asarray(algorithm.pop.get("F"), dtype=np.float64))
        if should_run_full_test(iteration, num_iterations, evo_eval_freq):
            trainer._wandb_log_evo_objective_plot(
                population_F=population_F,
                pareto_F=np.asarray(pareto_front, dtype=np.float64),
                pareto_center_F=np.asarray(
                    train_batch_metrics["loss_item"], dtype=np.float64
                ),
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

        inj = inject_pareto_center_into_pop(
            algorithm=algorithm,
            pareto_center_X=pareto_center_X,
            pareto_center_F=train_batch_metrics["loss_item"],
        )
        pc_f = np.asarray(train_batch_metrics["loss_item"], dtype=np.float64).reshape(-1)
        if inj is not None:
            algorithm.pop[inj].set(_SNSGA2_KEY_F_BAR, pc_f.copy())
            algorithm.pop[inj].set(_SNSGA2_KEY_VAR, np.zeros(n_obj, dtype=np.float64))

    num_iterations = int(num_iterations)
    if num_iterations < 1:
        raise ValueError(f"num_iterations must be >= 1, got {num_iterations}")

    # Bootstrap: sample minibatch(es), draw unevaluated initial population, score everyone
    # on that same batch, then commit with tell(). Later generations always refresh parent
    # F on the new batch before ask() returns offspring to evaluate.
    t0 = time.perf_counter()
    shared_batches = trainer._sample_batches(train_dataloaders, num_batches)
    initial = algorithm.ask()
    _eval_population_f_on_batches(initial, shared_batches)
    algorithm.tell(infills=initial)
    _post_tell_iteration(0, t0, initial.get("X"), shared_batches)

    for iteration in range(1, num_iterations):
        t0 = time.perf_counter()
        shared_batches = trainer._sample_batches(train_dataloaders, num_batches)
        _eval_population_f_on_batches(algorithm.pop, shared_batches)
        offspring = algorithm.ask()
        offspring_X = offspring.get("X")
        _eval_population_f_on_batches(offspring, shared_batches)
        algorithm.tell(infills=offspring)
        _post_tell_iteration(iteration, t0, offspring_X, shared_batches)

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
        "method": "SNSGA2",
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
        "snsga2_fitness_beta": beta,
        "snsga2_eps_kappa": float(eps_kappa),
    }
    trainer.last_evo = out
    return out
