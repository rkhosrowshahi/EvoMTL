"""PSL-TaskGuide: Pareto set learning whose Guided ES subspace is built from the per-task pseudo-gradients.

One run learns a line of networks ``theta(lam) = theta + S B psi_{w,u}(phi(lam))``: ``theta`` is the balanced
network (the ES mean, started at ``theta_0``), ``B = B_hat / sqrt(kappa)`` a steering direction that is not searched
(``kappa``: curvature of the summed loss along ``B_hat``) and ``psi_{w,u}`` a monotone Bernstein map with log width
``w``. ES searches ``x = (theta, w, u)`` with the pair differences of the preference scores (Chebyshev or weighted
sum). The population also estimates each task's pseudo-gradient ``g_k = sum_j r_jk eps_theta,j`` (``r_jk``: centered
ranks of task ``k``'s own pair differences). Their moving averages ``m_1``, ``m_2`` (decay ``steering_ema``) and the
moving average ``d`` of the ``theta`` update (decay ``guided_ema``) span a 3-dimensional subspace that contains

* the common descent direction (``~ m_1 + m_2``): what the balanced network ``theta`` needs;
* the trade-off direction ``B_hat = (m_2 - m_1) / |m_2 - m_1|``: what the front needs;
* the recent update.

The noise is Guided ES noise in that subspace (Maheswaranathan et al., ICML 2019):
``eps = sqrt(alpha) eps_iso + s U e``, ``U`` an orthonormal basis of the subspace in the ``theta`` block, ``e ~ N(0, I_3)``,
``s = guided_sigma / sigma`` (or ``sqrt((1 - alpha) n / 3)``). The members spread along ``B_hat`` for the front then
also sample the subspace in which ``theta``'s descent direction lies, so the front's samples are no longer wasted for
``theta``. The per-task estimates ``g_k`` (which build the guide and ``B_hat``) use the isotropic part ``eps_iso`` only, so
they are not biased toward the guide they define; the ``(theta, w, u)`` update uses the full ``eps``, as in Guided ES.
"""

from __future__ import annotations

import csv
import math
import os
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from .common import configure_moea_random_seed, should_run_full_test
from .hv import pareto_front_indices
from .mo_openes import centered_ranks, make_es_optimizer, true_jacobian
from .psl_es import (conditioned_indices, hypervolume, preference_features, preference_grid, sample_preferences,
                     scalarize, simplex_basis)

SCALARIZATIONS = ("tch", "ws")


def run_psl_taskguide(
    trainer: Any,
    train_dataloaders: Any,
    num_iterations: int,
    pop_size: int,
    lr: float = 1e-3,
    sigma: float = 0.01,
    sigma_decay: float = 1.0,
    sigma_min: float = 0.0,
    scalarization: str = "tch",
    psl_params: str = "all",
    eval_points: int = 11,
    fitness_shaping: str = "raw",
    map_degree: int = 3,
    steering_ema: float = 0.9,
    map_init_width: float = 0.02,
    map_sigma: float = 0.1,
    map_lr: float = 0.01,
    map_width_sigma: float = 0.2,
    curvature_step: float = 0.1,
    curvature_ema: float = 0.9,
    grad_cos_freq: int = 0,
    ablation: str = "none",
    antithetic: bool = True,
    optim: str = "adam",
    adam_betas: Tuple[float, float] = (0.9, 0.999),
    adam_eps: float = 1e-8,
    rmsprop_alpha: float = 0.99,
    rmsprop_eps: float = 1e-8,
    num_batches: int = 1,
    seed: int = 0,
    hv_ref_point: Optional[Any] = None,
    wandb_step_offset: int = 0,
    test_dataloaders: Optional[Any] = None,
    evo_eval_freq: int = 1,
    guided_alpha: float = 0.5,
    guided_sigma: float = 0.0,
    guided_ema: float = 0.9,
    **_unused: Any,
) -> Dict[str, Any]:
    """PSL-TaskGuide ask-and-tell on ``x = (theta, w, u)`` (see module docstring).

    Parameters
    ----------
    guided_alpha : float
        Fraction of the noise budget kept isotropic (1: no guide, plain OpenES sampling).
    guided_sigma : float
        Perturbation size along the guiding subspace (0: ``sigma * sqrt((1 - alpha) n / 3)``).
    guided_ema : float
        Decay of the moving average of the ``theta`` update (the third guiding direction).
    ablation : str
        ``"none"`` (default) or ``"balanced"``: every pair at ``lam = 1/K`` (``phi = 0``, so ``psi = 0`` and the map
        has no effect): Guided ES on the balanced Chebyshev score with this method's subspace, to separate the cost
        of preference sampling from the sampling scheme.

    map_degree, map_init_width : int, float
        Degree of the monotone Bernstein preference map (``map_degree`` shape variables ``u``) and its initial
        half-width ``exp(w)``.
    map_sigma, map_width_sigma, map_lr : float
        Noise scale of ``u``, of ``w``, and learning rate of both (0: ``sigma`` / ``sigma`` / ``lr``).
    steering_ema : float
        Decay of the moving averages ``m_1``, ``m_2`` of the per-task pseudo-gradients (guide and ``B_hat``).
    curvature_step, curvature_ema : float
        Finite-difference step for the curvature ``kappa`` along ``B_hat`` (0: ``B = B_hat``) and decay of its
        moving average in log space.

    The other parameters are those of :func:`~LibMTL.evomtl.moea.psl_es.run_psl_es`. Needs two tasks and antithetic
    sampling.
    """
    if trainer.theta_0 is None:
        raise RuntimeError("prepare_evolution() first.")
    if int(trainer.task_num) != 2:
        raise ValueError(f"PSL-TaskGuide is implemented for two tasks, got {trainer.task_num}.")
    scalarization = str(scalarization).strip().lower()
    if scalarization not in SCALARIZATIONS:
        raise ValueError(f"Unknown PSL-TaskGuide scalarization {scalarization!r}; options: {SCALARIZATIONS}")
    fitness_shaping = str(fitness_shaping).strip().lower()
    if fitness_shaping not in ("centered_ranks", "raw"):
        raise ValueError(f"Unknown fitness_shaping {fitness_shaping!r}; options: centered_ranks, raw")
    raw = fitness_shaping == "raw"
    map_degree = int(map_degree)
    if map_degree < 1:
        raise ValueError(f"map_degree must be >= 1, got {map_degree}")
    pop_size = int(pop_size)
    if not antithetic or pop_size % 2 != 0:
        raise ValueError(f"PSL-TaskGuide needs antithetic sampling and an even pop_size (a pair shares one "
                         f"preference), got antithetic={antithetic}, pop_size={pop_size}.")
    guided_alpha = float(guided_alpha)
    use_guide = guided_alpha < 1.0
    ablation = str(ablation).strip().lower()
    if ablation not in ("none", "balanced"):
        raise ValueError(f"PSL-TaskGuide ablation must be none or balanced, got {ablation!r}")

    configure_moea_random_seed(seed)
    gen = torch.Generator().manual_seed(int(seed))

    D = int(trainer.theta_0.numel())
    K = 2
    M = pop_size // 2
    idx = conditioned_indices(trainer.model, psl_params)
    P = int(idx.numel())
    Q = simplex_basis(K)
    n_var = D + 1 + map_degree  # x = (theta, w, u): w is the log half-width of the front, u the map's shape
    x = torch.zeros(n_var, dtype=torch.float32)
    x[:D] = trainer.theta_0.detach().cpu().float().reshape(-1)
    x[D] = math.log(float(map_init_width))  # u = 0: equal increments, linear shape -> psi(phi) = map_init_width * phi
    x.requires_grad_(True)
    steer = torch.zeros(P, dtype=torch.float32)                  # unit steering direction B_hat
    task_avg = [torch.zeros(D, dtype=torch.float32) for _ in range(K)]  # moving averages m_k of the raw g_k
    upd_avg = torch.zeros(D, dtype=torch.float32)                # moving average of the unit theta update
    curv_scale, log_kappa, kappa_now = 1.0, None, float("nan")  # B = B_hat * curv_scale = B_hat / sqrt(kappa)
    bern_coef = torch.tensor([math.comb(map_degree, i) for i in range(map_degree + 1)], dtype=torch.float32)
    bern_pow = torch.arange(map_degree + 1, dtype=torch.float32)

    def bernstein(u: torch.Tensor, ph: float) -> torch.Tensor:
        """Monotone Bernstein map with increments ``softplus(u)``, shifted so that it is 0 at ``phi = 0``."""
        c = torch.cat([torch.zeros(1, dtype=torch.float32), torch.cumsum(torch.nn.functional.softplus(u), 0)])

        def at(t: float) -> torch.Tensor:
            return (c * bern_coef * t ** bern_pow * (1.0 - t) ** (map_degree - bern_pow)).sum()
        return at((ph + 1.0) / 2.0) - at(0.5)

    def pref_map(m: torch.Tensor, ph: float) -> torch.Tensor:
        """Preference map ``psi(phi) = exp(w) * bernstein(u, phi) / half range``, ``m = (w, u)``; ``psi(0) = 0``."""
        u = m[1:]
        half_range = (bernstein(u, 1.0) - bernstein(u, -1.0)) / 2.0
        return torch.exp(m[0]) * bernstein(u, ph) / half_range

    def spread(xv: torch.Tensor) -> float:
        """Half the weight-space distance between the end networks (``|B|`` of the learned line)."""
        return float((pref_map(xv[D:], 1.0) - pref_map(xv[D:], -1.0)) / 2.0 * steer.norm() * curv_scale)

    def theta_at(xv: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """Weights ``theta + S B psi(phi)`` of the network for preference features ``phi`` (``[1]``)."""
        d = xv[:D].clone()
        d.index_add_(0, idx, steer * (curv_scale * pref_map(xv[D:], float(phi[0]))))
        return d

    def guide_basis() -> Optional[torch.Tensor]:
        """Orthonormal basis ``[n_var, k]`` of span{m_1, m_2, d} in the ``theta`` block (None before the first update)."""
        vecs = [v for v in (*task_avg, upd_avg) if float(v.norm()) > 0]
        if not vecs:
            return None
        G = torch.zeros(n_var, len(vecs), dtype=torch.float32)
        for i, v in enumerate(vecs):
            G[:D, i] = v / v.norm()
        return torch.linalg.qr(G)[0]

    optim = str(optim).strip().lower()
    optimizer = make_es_optimizer([x], optim, lr, adam_betas, adam_eps, rmsprop_alpha, rmsprop_eps)
    sigma = float(sigma)
    method = f"PSL-TaskGuide-{scalarization.upper()}"
    grid = preference_grid(K, eval_points)
    grid_phi = preference_features(grid, Q)
    balanced_phi = torch.zeros(K - 1, dtype=torch.float32)  # phi(1/K) = 0
    ref = None if hv_ref_point is None else np.asarray(hv_ref_point, dtype=np.float32).reshape(K)

    def evaluate_front(xv: torch.Tensor, batches: list) -> Dict[str, list]:
        """Metrics of ``theta(lam)`` for every grid preference: current train batch and test set."""
        out: Dict[str, list] = {"train": [], "test": []}
        for g in range(len(grid)):
            trainer.set_theta(theta_at(xv, grid_phi[g]))
            out["train"].append(trainer.metrics_snapshot_from_prefetched_batches(batches))
            if test_dataloaders is not None:
                out["test"].append(trainer.test(test_dataloaders, epoch=None, mode="test", suppress_display=True,
                                                return_metrics=True, log_wandb=False))  # front.csv / test_hv only
        return out

    print(
        f"[EvoMTL] {method}: dims={n_var} | pop={pop_size} | optim={optim} lr={lr} | sigma={sigma} "
        f"(decay={sigma_decay}, min={sigma_min}) | conditioned params={P} (psl_params={psl_params}), theta dims={D}, "
        f"B from task-gradient difference (ema={steering_ema}, curvature step={curvature_step}), Bernstein map "
        f"degree {map_degree} (init width={map_init_width}, sigma={map_sigma or sigma}, lr={map_lr or lr}, "
        f"log width sigma={map_width_sigma}) | guide=span(m_1, m_2, update), alpha={guided_alpha}, "
        f"sigma_sub={guided_sigma or 'auto'}, update ema={guided_ema}"
        f" | scalarization={scalarization} | front grid={len(grid)} preferences | fitness={fitness_shaping}"
        + (f" | ablation={ablation}" if ablation != "none" else ""),
        flush=True,
    )

    num_iterations = int(num_iterations)
    tasks = list(trainer.task_name)
    csv_cols = (
        ["generation", "time_s", "sigma"]
        + [f"pop_loss_mean_{t}" for t in tasks]
        + [f"train_loss_{t}" for t in tasks] + [f"train_acc_{t}" for t in tasks]
        + [f"test_loss_{t}" for t in tasks] + [f"test_acc_{t}" for t in tasks]
        + ["b_norm", "test_hv", "curvature"]
        + [f"true_cos_{t}" for t in tasks] + ["true_cos_direction"]
    )
    front_cols = (["generation", "split"] + [f"lambda_{t}" for t in tasks]
                  + [f"loss_{t}" for t in tasks] + [f"acc_{t}" for t in tasks])
    csv_file = csv_writer = front_file = front_writer = None
    if getattr(trainer, "save_path", None):
        os.makedirs(trainer.save_path, exist_ok=True)
        csv_file = open(os.path.join(trainer.save_path, "metrics.csv"), "w", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=csv_cols)
        csv_writer.writeheader()
        front_file = open(os.path.join(trainer.save_path, "front.csv"), "w", newline="")
        front_writer = csv.DictWriter(front_file, fieldnames=front_cols)
        front_writer.writeheader()

    F = np.zeros((pop_size, K), dtype=np.float32)
    shared_batches: list = []
    for iteration in range(num_iterations):
        t0 = time.perf_counter()
        step = wandb_step_offset + iteration
        test_metrics = None

        # ask: isotropic noise plus Guided ES noise in span{m_1, m_2, d}; one preference per antithetic pair
        eps_iso = torch.randn(M, n_var, generator=gen, dtype=torch.float32)
        eps = eps_iso
        U = guide_basis() if use_guide else None
        if U is not None:
            e_sub = torch.randn(M, U.shape[1], generator=gen, dtype=torch.float32)
            s_sub = (float(guided_sigma) / sigma) if guided_sigma > 0 else math.sqrt(
                (1.0 - guided_alpha) * n_var / U.shape[1])
            eps = math.sqrt(guided_alpha) * eps_iso + s_sub * (e_sub @ U.T)
        lam = sample_preferences(M, K, gen)
        if ablation == "balanced":  # drawn anyway so the random stream matches the full method
            lam = torch.full((M, K), 1.0 / K, dtype=torch.float32)
        phi = preference_features(lam, Q)

        shared_batches = trainer._sample_batches(train_dataloaders, num_batches)
        xd = x.detach()
        sig = torch.full((n_var,), sigma, dtype=torch.float32)  # per-coordinate noise scale
        if map_sigma > 0:
            sig[D:] = float(map_sigma)
        if map_width_sigma > 0:
            sig[D] = float(map_width_sigma)
        if curvature_step > 0 and float(steer.norm()) > 0:
            # curvature of the summed loss along the unit steering direction, on this generation's batch
            h = float(curvature_step)
            L = []
            for sh in (0.0, h, -h):
                dz = xd[:D].clone()
                dz.index_add_(0, idx, sh * steer)
                trainer.set_theta(dz)
                L.append(trainer.evaluate_multiobjective_losses(train_dataloaders, prefetched_batches=shared_batches))
            kappa_now = float(((L[1] + L[2] - 2.0 * L[0]) / h ** 2).sum())
            lk = math.log(max(kappa_now, 1e-4))
            log_kappa = lk if log_kappa is None else curvature_ema * log_kappa + (1.0 - curvature_ema) * lk
            curv_scale = math.exp(-0.5 * log_kappa)

        # evaluate every member on the same mini-batch(es): rows j (x + sig eps_j) and M + j (x - sig eps_j)
        for j in range(M):
            for s, sign in enumerate((1.0, -1.0)):
                trainer.set_theta(theta_at(xd + sign * sig * eps[j], phi[j]))
                F[j + s * M] = trainer.evaluate_multiobjective_losses(
                    train_dataloaders, prefetched_batches=shared_batches
                )

        # tell: pair differences of the preference scores -> Guided ES search gradient on x (full eps)
        lam_np = np.concatenate([lam.numpy(), lam.numpy()])
        s_all = scalarize(F, lam_np, scalarization, ref)
        diff = s_all[:M] - s_all[M:]
        if raw:  # unshaped: 2 * (d_j / 2) = d_j in the estimator below
            ranks = torch.as_tensor(diff / 2.0, dtype=torch.float32)
        else:
            ranks = torch.as_tensor(centered_ranks(np.concatenate([diff, -diff]))[:M], dtype=torch.float32)
        direction = 2.0 * (ranks @ eps) / (pop_size * sig)
        g_task = []  # per-task pseudo-gradients on the isotropic theta noise (unbiased by the guide they define)
        for k in range(K):
            d = F[:M, k] - F[M:, k]  # always ranked: a task with larger losses must not dominate
            rk = torch.as_tensor(centered_ranks(np.concatenate([d, -d]))[:M], dtype=torch.float32)
            g_task.append(rk @ eps_iso[:, :D])
        true_cos = None
        if grad_cos_freq > 0 and iteration % int(grad_cos_freq) == 0:  # diagnostic, at the pre-update balanced net
            J_true = true_jacobian(trainer, theta_at(xd, balanced_phi), shared_batches)  # [K, D]

            def cos(a: torch.Tensor, b: torch.Tensor) -> float:
                return float(a @ b / (a.norm() * b.norm()).clamp_min(1e-30))
            true_cos = [cos(g_task[k], J_true[k]) for k in range(K)] + [cos(direction[:D], J_true.sum(dim=0))]
        x.grad = direction.clone()
        u_old = x.detach()[D:].clone()
        optimizer.step()
        if map_lr > 0:  # the map variables take Adam's step at their own learning rate
            with torch.no_grad():
                x[D:] = u_old + (float(map_lr) / float(lr)) * (x[D:] - u_old)

        # guide and steering: moving averages of the raw per-task estimates and of the unit theta update
        for k in range(K):
            task_avg[k] = g_task[k] if float(task_avg[k].norm()) == 0.0 else (
                steering_ema * task_avg[k] + (1.0 - steering_ema) * g_task[k])
        dz_unit = direction[:D] / direction[:D].norm().clamp_min(1e-30)
        upd_avg = dz_unit if float(upd_avg.norm()) == 0.0 else guided_ema * upd_avg + (1.0 - guided_ema) * dz_unit
        v = (task_avg[1] - task_avg[0])[idx]  # m_2 - m_1: moving along it lowers l_1 and raises l_2
        steer = v / v.norm().clamp_min(1e-30)
        sigma = max(sigma * float(sigma_decay), float(sigma_min))
        elapsed = time.perf_counter() - t0

        xd = x.detach()
        b_norm = spread(xd)
        pareto_front = F[pareto_front_indices(F)]
        trainer._print_evo_progress(
            method,
            iteration,
            num_iterations,
            pareto_front,
            extras=f"sigma={sigma:.4g} |d|={float(direction.norm()):.4g} |B|={b_norm:.4g} kappa={kappa_now:.4g}"
                   + (" true_cos=[" + ", ".join(f"{c:+.4f}" for c in true_cos[:K]) + f"] dir_cos={true_cos[K]:+.4f}"
                      if true_cos is not None else ""),
            elapsed=elapsed,
            wandb_log_step=step,
        )
        trainer._wandb_log_evo_front_metrics(iteration, pareto_front, step, extras={
            "evo/psl/sigma": sigma,
            "evo/psl/direction_norm": float(direction.norm()),
            "evo/psl/b_norm": b_norm,
            "evo/psl/curvature": kappa_now,
            "evo/psl/curvature_scale": curv_scale,
        })

        # the balanced network theta: snapshot on the current batch (and test periodically)
        trainer.set_theta(theta_at(xd, balanced_phi))
        train_batch_metrics = trainer.metrics_snapshot_from_prefetched_batches(shared_batches)
        run_eval = should_run_full_test(iteration, num_iterations, evo_eval_freq)
        test_hv = None
        if test_dataloaders is not None and run_eval:
            test_metrics = trainer.test(
                test_dataloaders,
                epoch=None,
                mode="test",
                wandb_log_step=step,
                suppress_display=True,
                return_metrics=True,
            )
            trainer.print_compact_train_test_line(
                train_batch_metrics, test_metrics, epoch_label=iteration + 1
            )
            front = evaluate_front(xd, shared_batches)
            test_F = np.array([m["loss_item"] for m in front["test"]], dtype=np.float32)
            if ref is not None:
                test_hv = hypervolume(test_F, ref)
            picks = sorted({0, len(grid) // 2, len(grid) - 1})
            print(
                f"[EvoMTL] PSL front TEST iteration {iteration + 1:04d} | "
                + " | ".join("lam=(" + ", ".join(f"{float(v):.2f}" for v in grid[g]) + "): "
                             + "/".join(f"{v:.4f}" for v in test_F[g]) for g in picks)
                + (f" | HV={test_hv:.4f}" if test_hv is not None else ""),
                flush=True,
            )
            if front_writer is not None:
                for split, ms in front.items():
                    for g, m in enumerate(ms):
                        row = {"generation": iteration + 1, "split": split}
                        for k, t in enumerate(tasks):
                            row[f"lambda_{t}"] = float(grid[g, k])
                            row[f"loss_{t}"] = float(m["loss_item"][k])
                            row[f"acc_{t}"] = float(m["results"][t][0])
                        front_writer.writerow(row)
                front_file.flush()
        else:
            parts = " | ".join(
                f"{task}: {float(train_batch_metrics['loss_item'][i]):.4f}"
                for i, task in enumerate(trainer.task_name)
            )
            print(
                f"[EvoMTL] Balanced theta TRAIN (curr batch) iteration {iteration + 1:04d} | {parts}",
                flush=True,
            )

        if csv_writer is not None:
            row: Dict[str, Any] = {"generation": iteration + 1, "time_s": round(elapsed, 4), "sigma": sigma,
                                   "b_norm": b_norm, "curvature": kappa_now}
            if true_cos is not None:
                for k, t in enumerate(tasks):
                    row[f"true_cos_{t}"] = true_cos[k]
                row["true_cos_direction"] = true_cos[K]
            for k, t in enumerate(tasks):
                row[f"pop_loss_mean_{t}"] = float(F[:, k].mean())
                row[f"train_loss_{t}"] = float(train_batch_metrics["loss_item"][k])
                row[f"train_acc_{t}"] = float(train_batch_metrics["results"][t][0])
                if test_metrics is not None:
                    row[f"test_loss_{t}"] = float(test_metrics["loss_item"][k])
                    row[f"test_acc_{t}"] = float(test_metrics["results"][t][0])
            if test_hv is not None:
                row["test_hv"] = test_hv
            csv_writer.writerow(row)
            csv_file.flush()

    for f in (csv_file, front_file):
        if f is not None:
            f.close()

    # final front (the same numbers as the last front.csv rows when evo_eval_freq > 0)
    xd = x.detach()
    front = evaluate_front(xd, shared_batches)
    grid_X = torch.stack([theta_at(xd, grid_phi[g]) for g in range(len(grid))]).numpy()
    train_F = np.array([m["loss_item"] for m in front["train"]], dtype=np.float32)
    test_F = np.array([m["loss_item"] for m in front["test"]], dtype=np.float32).reshape(-1, K)
    test_acc = np.array([[m["results"][t][0] for t in tasks] for m in front["test"]], dtype=np.float32).reshape(-1, K)
    if getattr(trainer, "save_path", None):
        torch.save({
            "theta": xd[:D].clone(),                # the balanced network (lam = 1/K)
            "B_hat": steer.clone(),                 # unit steering direction on the conditioned parameters
            "curvature_scale": curv_scale,          # B = B_hat * curvature_scale
            "map_u": xd[D:].clone(), "map_degree": map_degree, "map_log_width": True,  # (w, u) of the map
            "task_avg": torch.stack(task_avg),      # moving averages m_1, m_2 of the per-task pseudo-gradients
            "index": idx,                           # flat positions of the conditioned parameters
            "Q": Q,                                 # phi(lam) = sqrt(2) Q^T (lam - 1/K)
            "grid": grid, "train_F": train_F, "test_F": test_F, "test_acc": test_acc,
            "tasks": tasks, "psl_params": psl_params, "scalarization": scalarization, "method": method,
            "pop_size": pop_size, "lr": float(lr), "sigma": sigma,
        }, os.path.join(trainer.save_path, "psl_result.pt"))

    center_X = theta_at(xd, balanced_phi).numpy().copy()
    trainer.set_theta(center_X)
    out = {
        "method": method,
        "pareto_F": train_F,       # the grid front on the last mini-batch
        "pareto_X": grid_X,        # weights of the grid networks
        "population_F": F,
        "population_X": np.empty((0, D)),
        "hv_weights": None,
        "pareto_center_X": center_X,
        "hv_ref_point": ref,
        "hv_center_aggregation": None,
        "hv_softmax_temperature": None,
        "sigma": sigma,
        "scalarization": scalarization,
        "front_lambda": grid.numpy(),
        "front_test_F": test_F,
    }
    trainer.last_evo = out
    return out
