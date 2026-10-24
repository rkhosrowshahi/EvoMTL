"""SC-DE: Single-Center Differential Evolution on the equal-weight sum of the losses.

The only persistent candidate is the centre ``mu``; each generation draws temporary donor pairs around it, turns
each pair into a *fitness-oriented* difference vector, and moves the centre along it:

A. *Donor pairs*: ``x_a = mu + sigma eps_a``, ``x_b = mu + sigma eps_b``, ``eps ~ N(0, I_D)``, ``K = num_pairs`` pairs
   evaluated on the generation's mini-batch; ``mirrored=True`` uses ``eps_b = -eps_a``, so ``x+ - x- = +-2 sigma eps_a``
   and the curvature term ``sigma^2 (eps_a' H eps_a - eps_b' H eps_b) / 2`` of the loss gap cancels (with ``gap``
   weights and ``CR = 1`` this is OpenES with antithetic sampling).
B. *Oriented differential mutation*: ``x+`` is the better donor of a pair, ``x-`` the worse, and

       Delta = F / K sum_k w_k (x+_k - x-_k),

   with ``w_k = 1`` (``pair_weight="sign"``, the proposal: only the order of ``f_a, f_b`` is used) or
   ``w_k = (f-_k - f+_k) / (2 sigma^2)`` (``"gap"``: weighted by the loss gap, which makes ``Delta`` ``-F`` times the
   two-point gradient estimate ``mean_k (f_a - f_b)(eps_a - eps_b) / (2 sigma)``). A random difference
   ``x_a - x_b`` is ``N(0, 2 sigma^2 I)`` noise; the orientation gives it an expected descent direction.
C. *Binomial crossover*: ``Delta~ = M * Delta``, ``M_j = 1`` if ``r_j < CR`` or ``j = j_rand``.
D. *Greedy centre update*: the optimizer (SGD, Adam or RMSprop with learning rate ``lr``, as OpenES) takes one step on
   ``mu`` with the pseudo-gradient ``-Delta~``, giving the trial ``u`` (plain SGD with ``lr = eta`` is
   ``u = mu + eta Delta~``). ``u`` and ``mu`` are evaluated on the same mini-batch; ``u`` replaces ``mu`` only if it is
   better, otherwise ``mu`` and the optimizer state are rolled back (``greedy=False``: always accepted). ``sigma`` is
   fixed unless ``sigma_increase`` / ``sigma_decrease`` set a success rule.

Each generation costs ``2 K + 2`` evaluations (``K = 1``: the proposal's two donors, the trial and the centre, which
is re-evaluated because the mini-batch changes). The returned solution (``pareto_center_X``) is the final centre;
``pareto_F`` / ``pareto_X`` are the non-dominated donors of the last generation.
"""

from __future__ import annotations

import copy
import csv
import os
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from .common import configure_moea_random_seed, should_run_full_test
from .hv import pareto_front_indices
from .es_common import _cosine, make_es_optimizer, true_jacobian


def run_sc_de(
    trainer: Any,
    train_dataloaders: Any,
    num_iterations: int,
    num_pairs: int = 1,
    lr: float = 1e-3,
    optim: str = "adam",
    adam_betas: Tuple[float, float] = (0.9, 0.999),
    adam_eps: float = 1e-8,
    rmsprop_alpha: float = 0.99,
    rmsprop_eps: float = 1e-8,
    sgd_momentum: float = 0.0,
    sigma: float = 0.01,
    sigma_min: float = 0.0,
    sigma_max: float = 0.0,
    sigma_increase: float = 1.0,
    sigma_decrease: float = 1.0,
    F: float = 0.5,
    CR: float = 0.5,
    pair_weight: str = "sign",
    mirrored: bool = False,
    greedy: bool = True,
    num_batches: int = 1,
    seed: int = 0,
    hv_ref_point: Optional[np.ndarray] = None,
    wandb_step_offset: int = 0,
    test_dataloaders: Optional[Any] = None,
    evo_eval_freq: int = 1,
    grad_cos_freq: int = 1,
    **_unused: Any,
) -> Dict[str, Any]:
    """SC-DE (module docstring) on the weights ``theta`` with fitness = the sum of the losses, from ``theta_0``.

    Parameters
    ----------
    num_pairs : int
        Donor pairs ``K`` per generation (the proposal: 1); their oriented differences are averaged.
    lr, optim, adam_betas, adam_eps, rmsprop_alpha, rmsprop_eps, sgd_momentum
        Optimizer on ``mu`` (see :func:`~.es_common.make_es_optimizer`), stepped with the pseudo-gradient
        ``-Delta~``.
    sigma, sigma_min, sigma_max, sigma_increase, sigma_decrease
        Donor radius; after an accepted / rejected trial it is multiplied by ``sigma_increase`` /
        ``sigma_decrease`` (both ``1``: fixed) and clamped to ``[sigma_min, sigma_max]`` (``sigma_max = 0``: no cap).
    F, CR
        Differential scale and binomial crossover rate.
    pair_weight : str
        ``sign`` (the proposal) or ``gap`` (each oriented difference weighted by its loss gap, see the module
        docstring).
    mirrored : bool
        Mirrored donor pairs ``x_b = mu - sigma eps_a`` instead of two independent draws.
    greedy : bool
        Keep the trial only if it lowers the summed loss on the generation's mini-batch (step D).
    grad_cos_freq : int
        Every this many iterations, log cosine(Delta~, true descent direction of the summed loss) at ``mu`` on the
        same mini-batch. ``0`` disables.
    """
    if trainer.theta_0 is None:
        raise RuntimeError("prepare_evolution() first.")
    K = int(num_pairs)
    if K < 1:
        raise ValueError(f"SC-DE needs num_pairs >= 1, got {K}.")
    pair_weight = str(pair_weight).strip().lower()
    if pair_weight not in ("sign", "gap"):
        raise ValueError(f"SC-DE: unknown pair_weight {pair_weight!r} (sign, gap).")
    F, CR = float(F), float(CR)

    configure_moea_random_seed(seed)
    gen = torch.Generator().manual_seed(int(seed))
    n_var = int(trainer.theta_0.numel())
    n_obj = int(trainer.task_num)
    theta_0 = trainer.theta_0.detach().cpu().float()
    theta = theta_0.clone().requires_grad_(True)  # the centre mu
    optim = str(optim).strip().lower()
    optimizer = make_es_optimizer([theta], optim, lr, adam_betas, adam_eps, rmsprop_alpha, rmsprop_eps, sgd_momentum)
    mu = theta.detach().clone()
    sigma = float(sigma)
    n_accept = 0
    method, tag = "SC-DE", "sc_de"
    print(
        f"[EvoMTL] {method}: dims={n_var} | pairs={K} ({2 * K + 2} evals/gen) | optim={optim} lr={lr}"
        + (f" momentum={sgd_momentum}" if optim == "sgd" else "")
        + f" | F={F} CR={CR} pair_weight={pair_weight} mirrored={mirrored} greedy={greedy} "
        f"| sigma={sigma} (x{sigma_increase} / x{sigma_decrease}, min={sigma_min}, max={sigma_max})",
        flush=True,
    )

    num_iterations = int(num_iterations)
    population_X = np.empty((0, n_var))
    population_F = np.empty((0, n_obj))
    tasks = list(trainer.task_name)
    csv_cols = (
        ["generation", "time_s", "sigma", "accepted", "accept_rate", "center_loss", "trial_loss", "pair_gap_mean",
         "step_norm"]
        + [f"pop_loss_mean_{t}" for t in tasks]
        + [f"train_loss_{t}" for t in tasks] + [f"train_acc_{t}" for t in tasks]
        + [f"test_loss_{t}" for t in tasks] + [f"test_acc_{t}" for t in tasks]
        + ["true_cos_direction"]
    )
    csv_file = csv_writer = None
    if getattr(trainer, "save_path", None):
        os.makedirs(trainer.save_path, exist_ok=True)
        csv_file = open(os.path.join(trainer.save_path, "metrics.csv"), "w", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=csv_cols)
        csv_writer.writeheader()

    def evaluate(x: torch.Tensor, batches: list) -> np.ndarray:
        trainer.set_theta(x)
        return trainer.evaluate_multiobjective_losses(train_dataloaders, prefetched_batches=batches)

    for iteration in range(num_iterations):
        t0 = time.perf_counter()
        step = wandb_step_offset + iteration
        test_metrics = None
        shared_batches = trainer._sample_batches(train_dataloaders, num_batches)

        # A. donor pairs around the centre, all on the same mini-batch
        eps = torch.randn(2, K, n_var, generator=gen)
        if mirrored:
            eps[1] = -eps[0]
        donors = mu + sigma * eps.reshape(2 * K, n_var)  # rows 0..K-1: x_a, K..2K-1: x_b
        F_pop = np.zeros((2 * K, n_obj), dtype=np.float32)
        for i in range(2 * K):
            F_pop[i] = evaluate(donors[i], shared_batches)
        f = torch.as_tensor(F_pop.sum(axis=1), dtype=torch.float32)
        f_a, f_b = f[:K], f[K:]

        # B. fitness-oriented differences x+ - x- = sigma (eps+ - eps-)
        orient = torch.where(f_a <= f_b, 1.0, -1.0)  # +1: x_a is the better donor
        diff = orient.unsqueeze(1) * sigma * (eps[0] - eps[1])
        gap = (f_a - f_b).abs()
        w = torch.ones(K) if pair_weight == "sign" else gap / (2.0 * sigma ** 2)
        delta = F * (w @ diff) / K

        # C. binomial crossover of the step
        mask = torch.rand(n_var, generator=gen) < CR
        mask[torch.randint(n_var, (1,), generator=gen)] = True
        delta = torch.where(mask, delta, torch.zeros_like(delta))

        # D. optimizer step to the trial, greedy selection against the centre on the same mini-batch
        center_loss = float(evaluate(mu, shared_batches).sum())
        dir_cos = None
        if grad_cos_freq > 0 and iteration % int(grad_cos_freq) == 0:
            d_true = true_jacobian(trainer, mu, shared_batches).sum(dim=0)
            dir_cos = _cosine(-delta, d_true)
        optim_state = copy.deepcopy(optimizer.state_dict())
        theta.grad = -delta
        optimizer.step()
        trial = theta.detach().clone()
        trial_loss = float(evaluate(trial, shared_batches).sum())
        accepted = trial_loss < center_loss or not greedy
        step_norm = float((trial - mu).norm())
        if accepted:
            mu = trial
            n_accept += 1
            sigma *= float(sigma_increase)
        else:  # roll back the centre and the optimizer state
            with torch.no_grad():
                theta.copy_(mu)
            optimizer.load_state_dict(optim_state)
            sigma *= float(sigma_decrease)
        sigma = max(sigma, float(sigma_min))
        if sigma_max > 0:
            sigma = min(sigma, float(sigma_max))

        population_X = donors.numpy()
        population_F = F_pop
        pareto_front = F_pop[pareto_front_indices(F_pop)]
        elapsed = time.perf_counter() - t0
        accept_rate = n_accept / (iteration + 1)
        metrics = {
            "accepted": int(accepted), "accept_rate": accept_rate, "center_loss": center_loss,
            "trial_loss": trial_loss, "pair_gap_mean": float(gap.mean()), "step_norm": step_norm,
        }
        extras_wandb: Dict[str, Any] = {
            f"evo/{tag}/sigma": sigma,
            f"evo/{tag}/direction_norm": float(delta.norm()),
            f"evo/{tag}/mean_norm": float((mu - theta_0).norm()),
            f"evo/{tag}/pop_losses/avg": float(F_pop.mean()),
            **{f"evo/{tag}/{key}": val for key, val in metrics.items()},
        }
        progress = (f"sigma={sigma:.4g} acc={int(accepted)} rate={accept_rate:.3f} gap={float(gap.mean()):.4g} "
                    f"|step|={step_norm:.4g}")
        if dir_cos is not None:
            progress += f" dir_cos={dir_cos:+.4f}"
            extras_wandb[f"evo/{tag}/true_cos/direction"] = dir_cos

        trainer._print_evo_progress(method, iteration, num_iterations, pareto_front, extras=progress,
                                    elapsed=elapsed, offspring_X=population_X, wandb_log_step=step)
        trainer._wandb_log_evo_front_metrics(iteration, pareto_front, step, extras=extras_wandb)

        trainer.set_theta(mu)
        train_batch_metrics = trainer.metrics_snapshot_from_prefetched_batches(shared_batches)
        run_eval = should_run_full_test(iteration, num_iterations, evo_eval_freq)
        if run_eval:
            trainer._wandb_log_evo_objective_plot(
                population_F=F_pop,
                pareto_F=pareto_front,
                pareto_center_F=np.asarray(train_batch_metrics["loss_item"], dtype=np.float32),
                step=step,
            )
        if test_dataloaders is not None and run_eval:
            test_metrics = trainer.test(test_dataloaders, epoch=None, mode="test", wandb_log_step=step,
                                        suppress_display=True, return_metrics=True)
            trainer.print_compact_train_test_line(train_batch_metrics, test_metrics, epoch_label=iteration + 1)
        else:
            parts = " | ".join(f"{task}: {float(train_batch_metrics['loss_item'][i]):.4f}"
                               for i, task in enumerate(tasks))
            print(f"[EvoMTL] Mean theta TRAIN (curr batch) iteration {iteration + 1:04d} | {parts}", flush=True)

        if csv_writer is not None:
            row: Dict[str, Any] = {"generation": iteration + 1, "time_s": round(elapsed, 4), "sigma": sigma, **metrics}
            for k, t in enumerate(tasks):
                row[f"pop_loss_mean_{t}"] = float(F_pop[:, k].mean())
                row[f"train_loss_{t}"] = float(train_batch_metrics["loss_item"][k])
                row[f"train_acc_{t}"] = float(train_batch_metrics["results"][t][0])
                if test_metrics is not None:
                    row[f"test_loss_{t}"] = float(test_metrics["loss_item"][k])
                    row[f"test_acc_{t}"] = float(test_metrics["results"][t][0])
            if dir_cos is not None:
                row["true_cos_direction"] = dir_cos
            csv_writer.writerow(row)
            csv_file.flush()

    if csv_file is not None:
        csv_file.close()
    theta_X = mu.numpy().copy()
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
        "sigma": sigma,
        "aggregator": "ws",
        "accept_rate": n_accept / max(num_iterations, 1),
    }
    trainer.last_evo = out
    return out
