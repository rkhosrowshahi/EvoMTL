"""Tangent Pareto set learning with OpenES (Tangent PSL-ES, TPSL): PSL-ES with a computed steering direction.

PSL-ES (:mod:`~LibMTL.evomtl.moea.psl_es`) learns a preference-conditioned family of networks
``theta(lam) = theta + S B phi(lam)`` and searches ``theta`` and the steering matrix ``B`` together
(``D + P (K - 1)`` dimensions). Tangent PSL-ES keeps the family and the training signal (one preference per
antithetic pair, each member scored under its own preference) but does not search ``B``. Its direction is the
Pareto-set tangent estimated from the same population, and only its length ``c`` is learned::

    theta(lam) = theta + c S T^T phi(lam)

so ES searches ``theta`` (``D`` dimensions) and the scalar ``c``. ``lam``, ``phi``, ``Q``, ``S`` and ``P`` are as
in PSL-ES; for two tasks ``phi = lam_1 - lam_2`` in ``[-1, 1]``, the line runs from ``theta - c T`` (task 2 only)
through the balanced network ``theta`` to ``theta + c T`` (task 1 only), and ``c`` is the distance of its ends
from ``theta``.

Why the tangent. A network ``theta*(lam)`` on the Pareto set is stationary for its preference,
``sum_k lam_k g_k = 0`` (``g_k`` the task gradients, the rows of ``G``). Moving the preference along the simplex
and differentiating gives the direction in which the Pareto set continues::

    d theta* / d phi = -H_lam^-1 G^T Q / sqrt(2),        H_lam = sum_k lam_k H_k

With identity curvature (``H_lam = I``) the columns are those of ``-G^T Q``; for two tasks
(``Q = (1, -1) / sqrt(2)``) the single column is ``(g_2 - g_1) / 2``, the gradient difference, pointing to the
task-1 side as ``lam_1`` grows. It is exact when the tasks share an isotropic curvature
(``f_k = |theta - a_k|^2 / 2``: the Pareto set is the segment from ``a_2`` to ``a_1``, and
``g_2 - g_1 = a_1 - a_2`` at every ``theta``) and first order otherwise; away from the Pareto set it is the
direction that trades one task for the other.

One generation, with antithetic pairs ``i = 1..N/2``:

1. draw a preference ``lam_i`` (stratified uniform for two tasks, Dirichlet(1) otherwise) and noise
   ``eps_i ~ N(0, I)`` over ``theta``; the perturbation is ``z_i = sigma eps_i``;
2. evaluate the pair ``theta + c S T^T phi_i +- z_i`` on the shared mini-batch: both members share the point
   on the line, only the noise is mirrored. Score each member under its own preference (Chebyshev, weighted
   sum or hypervolume score, as in PSL-ES);
3. ``d_i = s(+) - s(-)`` and its shaping ``r_i`` (``d_i / 2`` raw, or centered ranks), as in PSL-ES;
4. ``theta``: ``g = 2 sum_i r_i eps_i / (N sigma)`` estimates the gradient of ``E_lam[s(theta(lam), lam)]``
   under Gaussian smoothing, and the optimizer steps on ``theta``;
5. ``c``: ``g_c = 2 sum_i r_i phi_i^T T S^T eps_i / (N sigma)``, the chain rule through each pair's shift
   ``c S T^T phi_i`` applied to the same estimates (no extra evaluations). Adam steps on ``log c``
   (learning rate ``c_lr``; 0 keeps ``c`` at ``c0``). Brake (``brake``): the antithetic means
   ``(F(+) + F(-)) / 2`` estimate the losses at the pairs' points on the line; when, in moving averages over
   generations (decay ``grad_ema``), the points near some corner of the simplex are worse on every task than
   the points near the centre (the regions of the task guide: thirds of ``lam_1`` for two tasks), the line runs
   past the front there, and ``c`` shrinks by one step (``log c -= c_lr``) instead of following ``g_c``;
6. tangent: the per-task estimates ``g_k = 2 sum_i r_ik eps_i / (N sigma)`` (``r_ik``: the shaped pair
   differences of task ``k``'s loss) update moving averages ``G`` (decay ``grad_ema``), and ``T`` becomes the
   rows of ``-Q^T G`` on the conditioned weights, scaled to unit length. ``T`` is used from the next generation
   on; it is 0 before the first estimate, so the first generation evaluates every preference at ``theta``.

The split update (``center_a`` for ``theta``, ``spread_score`` for ``c``), the hypervolume score,
``rank_groups`` and Guided ES sampling work as in PSL-ES, with ``c`` in the place of ``B``; the directions of the
task guide (``psl_guide="tasks"``) are placed in ``theta`` only, ``(K + 1) K`` guide vectors.

Without the brake, the Chebyshev score measured from 0 lets the line grow past the front: near ``lam_1 = 1`` it
scores task 1 alone, so stretching gets no penalty for the damage to task 2. The ideal-point score for ``c``
(``spread_score="ideal"``) attacks the same problem in the score: measured from the population's best losses,
the damaged task dominates the max for most off-centre preferences.

``theta`` is the returned solution (``pareto_center_X``) and fills the train/test columns of ``metrics.csv``,
which also logs ``c``, ``brake`` (1 when the brake shrank ``c``) and ``grad_cos`` (cosine between the averaged
task gradients; negative when the tasks conflict). The front is read out on a grid of preferences: ``front.csv`` at every test evaluation,
``tpsl_result.pt`` at the end.

References:
- PSL-ES: :mod:`~LibMTL.evomtl.moea.psl_es` (PaMaL, Pareto HyperNetworks)
- Pareto-set continuation along the tangent: Hillermeier, "Nonlinear Multiobjective Optimization: A Generalized
  Homotopy Approach", 2001
- OpenES: https://arxiv.org/abs/1703.03864
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
from .es_common import _mean_pairwise_cosine, centered_ranks, guided_noise, make_es_optimizer, update_guide
from .psl_es import (
    PSL_GUIDES,
    SCALARIZATIONS,
    SPREAD_SCORES,
    center_weights,
    conditioned_indices,
    guide_regions,
    hypervolume,
    preference_features,
    preference_grid,
    sample_preferences,
    scalarize,
    simplex_basis,
)


def tangent_rows(G: torch.Tensor, Q: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Pareto-tangent directions ``[K - 1, P]`` on the conditioned weights ``idx``: the rows of ``-Q^T G``
    (``G``: task gradients ``[K, D]``) scaled to unit length; ``unit(g_2 - g_1)`` for two tasks, 0 where ``G`` is 0."""
    rows = -(Q.T @ G)[:, idx]
    return rows / rows.norm(dim=1, keepdim=True).clamp_min(1e-30)


def run_tpsl_es(
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
    center_a: float = 1.0,
    spread_score: str = "same",
    ideal_margin: float = 0.1,
    rank_groups: int = 1,
    fitness_shaping: str = "raw",
    psl_guide: str = "update",
    c0: float = 0.1,
    c_lr: float = 0.05,
    grad_ema: float = 0.9,
    brake: bool = True,
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
    guided_k: int = 0,
    guided_alpha: float = 0.5,
    guided_sigma: float = 0.0,
    guided_ema: float = 0.0,
    **_unused: Any,
) -> Dict[str, Any]:
    """Tangent PSL-ES ask-and-tell on ``theta`` and the length ``c`` (see module docstring).

    Parameters
    ----------
    c0 : float
        Initial length of the line: distance of the end preferences from ``theta`` (two tasks).
    c_lr : float
        Adam learning rate on ``log c``; 0 keeps ``c`` fixed at ``c0``.
    grad_ema : float
        Decay of the moving averages ``G`` of the per-task gradient estimates that give the tangent, and of the
        region losses compared by the brake.
    brake : bool
        Shrink ``c`` while the line runs past the front at some corner (see step 5); default on.

    The other parameters are those of :func:`~LibMTL.evomtl.moea.psl_es.run_psl_es`, with ``c`` in the place of
    ``B`` (``spread_score``). Needs antithetic sampling (both members of a pair share one preference).
    """
    if trainer.theta_0 is None:
        raise RuntimeError("prepare_evolution() first.")
    scalarization = str(scalarization).strip().lower()
    if scalarization not in SCALARIZATIONS:
        raise ValueError(f"Unknown PSL scalarization {scalarization!r}; options: {SCALARIZATIONS}")
    spread_score = str(spread_score).strip().lower()
    if spread_score not in SPREAD_SCORES:
        raise ValueError(f"Unknown PSL spread_score {spread_score!r}; options: {SPREAD_SCORES}")
    center_a = float(center_a)
    if center_a <= 0:
        raise ValueError(f"PSL center_a must be > 0, got {center_a}")
    rank_groups = int(rank_groups)
    uses_hv = "hv" in (scalarization, spread_score)
    if (uses_hv or rank_groups > 1) and int(trainer.task_num) != 2:
        raise ValueError("The PSL hv score and rank_groups > 1 are implemented for two tasks.")
    if uses_hv and hv_ref_point is None:
        raise ValueError("The PSL hv score needs hv_ref_point (evo_hv_ref_point).")
    fitness_shaping = str(fitness_shaping).strip().lower()
    if fitness_shaping not in ("centered_ranks", "raw"):
        raise ValueError(f"Unknown fitness_shaping {fitness_shaping!r}; options: centered_ranks, raw")
    if fitness_shaping == "raw" and rank_groups > 1:
        raise ValueError("fitness_shaping=raw does not rank, so it cannot be combined with rank_groups > 1.")
    raw = fitness_shaping == "raw"
    split = center_a != 1.0 or spread_score != "same"
    psl_guide = str(psl_guide).strip().lower()
    if psl_guide not in PSL_GUIDES:
        raise ValueError(f"Unknown PSL guide {psl_guide!r}; options: {PSL_GUIDES}")
    c0, c_lr, grad_ema = float(c0), float(c_lr), float(grad_ema)
    if c0 <= 0 or c_lr < 0 or not 0.0 <= grad_ema < 1.0:
        raise ValueError(f"Tangent PSL-ES needs c0 > 0, c_lr >= 0 and 0 <= grad_ema < 1, "
                         f"got c0={c0}, c_lr={c_lr}, grad_ema={grad_ema}.")
    pop_size = int(pop_size)
    if not antithetic or pop_size % 2 != 0:
        raise ValueError(f"Tangent PSL-ES needs antithetic sampling and an even pop_size (a pair shares one "
                         f"preference), got antithetic={antithetic}, pop_size={pop_size}.")

    configure_moea_random_seed(seed)
    gen = torch.Generator().manual_seed(int(seed))

    D = int(trainer.theta_0.numel())
    K = int(trainer.task_num)
    M = pop_size // 2
    idx = conditioned_indices(trainer.model, psl_params)
    P, C = int(idx.numel()), K - 1
    Q = simplex_basis(K)
    theta = trainer.theta_0.detach().cpu().float().clone()  # the search mean: the balanced network
    theta.requires_grad_(True)
    optim = str(optim).strip().lower()
    optimizer = make_es_optimizer([theta], optim, lr, adam_betas, adam_eps, rmsprop_alpha, rmsprop_eps)
    c = c0  # length of the line along the tangent
    log_c = torch.tensor(math.log(c0), dtype=torch.float32, requires_grad=True)
    c_optimizer = (torch.optim.Adam([log_c], lr=c_lr, betas=(float(adam_betas[0]), float(adam_betas[1])),
                                    eps=float(adam_eps)) if c_lr > 0 else None)
    G = torch.zeros(K, D, dtype=torch.float32)  # moving averages of the per-task gradient estimates
    T = torch.zeros(C, P, dtype=torch.float32)  # unit tangent rows on the conditioned weights (0 until estimated)
    has_tangent = False
    gaps = np.zeros((K, K))  # brake: moving averages of (corner region - centre region) mean losses [corner, task]
    sigma = float(sigma)
    method = f"TPSL-OpenES-{scalarization.upper()}"
    if guided_k > 0:  # Guided ES sampling: TPSL-OpenES-X -> TPSL-GuidedES-X
        method = method.replace("OpenES", "GuidedES")
    guide: list = []
    task_dirs: Dict[Tuple[int, int], torch.Tensor] = {}  # psl_guide="tasks": (region, task) -> weight-space direction
    grid = preference_grid(K, eval_points)
    grid_phi = preference_features(grid, Q)
    ref = None if hv_ref_point is None else np.asarray(hv_ref_point, dtype=np.float32).reshape(K)

    def theta_at(t: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """Weights ``t + c S T^T phi`` of the network for preference features ``phi`` (``[C]``) around ``t``."""
        w = t.clone()
        w.index_add_(0, idx, c * (T.T @ phi))
        return w

    def evaluate_front(t: torch.Tensor, batches: list) -> Dict[str, list]:
        """Metrics of ``theta(lam)`` for every grid preference: current train batch and test set."""
        out: Dict[str, list] = {"train": [], "test": []}
        for g in range(len(grid)):
            trainer.set_theta(theta_at(t, grid_phi[g]))
            out["train"].append(trainer.metrics_snapshot_from_prefetched_batches(batches))
            if test_dataloaders is not None:
                out["test"].append(trainer.test(test_dataloaders, epoch=None, mode="test",
                                                suppress_display=True, return_metrics=True, log_wandb=False))
        return out

    def pair_ranks(scores: np.ndarray, lam1: np.ndarray) -> torch.Tensor:
        """Shaped pair differences ``r_i`` of the scores (rows ``i`` and ``M + i``): ``d_i / 2`` (raw) or the
        centered ranks of ``{d_i} U {-d_i}``, within ``rank_groups`` groups of neighbouring ``lam_1``."""
        diff = scores[:M] - scores[M:]
        if raw:
            return torch.as_tensor(diff / 2.0, dtype=torch.float32)
        if rank_groups <= 1:
            return torch.as_tensor(centered_ranks(np.concatenate([diff, -diff]))[:M], dtype=torch.float32)
        r = np.zeros(M)
        for grp in np.array_split(np.argsort(lam1), rank_groups):
            r[grp] = centered_ranks(np.concatenate([diff[grp], -diff[grp]]))[: len(grp)]
        return torch.as_tensor(r, dtype=torch.float32)

    def task_ranks(d: np.ndarray) -> torch.Tensor:
        """Shaped per-task pair differences: ``d / 2`` (raw) or the centered ranks of ``{d} U {-d}``."""
        return torch.as_tensor(d / 2.0 if raw else centered_ranks(np.concatenate([d, -d]))[: len(d)],
                               dtype=torch.float32)

    print(
        f"[EvoMTL] {method}: dims={D} | pop={pop_size} | optim={optim} lr={lr} | sigma={sigma} "
        f"(decay={sigma_decay}, min={sigma_min}) | conditioned params={P} (psl_params={psl_params}) | "
        f"tangent line: c0={c0}, c_lr={c_lr}, grad_ema={grad_ema}, brake={'on' if brake else 'off'} | "
        f"scalarization={scalarization} | "
        f"front grid={len(grid)} preferences"
        + (f" | guided k={guided_k}, alpha={guided_alpha}, sigma_sub={guided_sigma or 'auto'}, ema={guided_ema}, "
           f"guide={psl_guide}" + (f" ({(K + 1) * K} vectors)" if psl_guide == "tasks" else "")
           if guided_k > 0 else "")
        + (f" | split update: theta weights Dirichlet(a={center_a}), c scores={spread_score}"
           + (f" (margin {ideal_margin})" if spread_score == "ideal" else "") if split else "")
        + (f" | ranks within {rank_groups} preference groups" if rank_groups > 1 else "")
        + f" | fitness={fitness_shaping}",
        flush=True,
    )

    num_iterations = int(num_iterations)
    tasks = list(trainer.task_name)
    # metrics.csv: the balanced network theta, same columns as PSL-ES plus the line length and task-gradient cosine.
    # front.csv: every grid preference at every test evaluation (split = train batch / test set).
    csv_cols = (
        ["generation", "time_s", "sigma"]
        + [f"pop_loss_mean_{t}" for t in tasks]
        + [f"train_loss_{t}" for t in tasks] + [f"train_acc_{t}" for t in tasks]
        + [f"test_loss_{t}" for t in tasks] + [f"test_acc_{t}" for t in tasks]
        + ["b_norm", "c", "brake", "grad_cos", "test_hv"]
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

        # ask: noise eps over theta and perturbations z = sigma eps, one preference per antithetic pair
        eps = torch.randn(M, D, generator=gen, dtype=torch.float32)
        if guided_k > 0 and guide:
            eps = guided_noise(eps, guide, gen, sigma, guided_alpha, guided_sigma)
        z = sigma * eps
        lam = sample_preferences(M, K, gen)
        phi = preference_features(lam, Q)

        # evaluate every member on the same mini-batch(es): rows i (point + z_i) and M + i (point - z_i), with the
        # pair's point theta + c S T^T phi_i on the tangent line
        shared_batches = trainer._sample_batches(train_dataloaders, num_batches)
        td = theta.detach()
        for i in range(M):
            point = theta_at(td, phi[i])
            for s, sign in enumerate((1.0, -1.0)):
                trainer.set_theta(point + sign * z[i])
                F[i + s * M] = trainer.evaluate_multiobjective_losses(
                    train_dataloaders, prefetched_batches=shared_batches
                )

        # tell: shaped pair differences of the preference scores
        lam_np = np.concatenate([lam.numpy(), lam.numpy()])
        lam1 = lam[:, 0].numpy()
        ranks = pair_ranks(scalarize(F, lam_np, scalarization, ref), lam1)
        ranks_c = ranks  # ranks for the length c (spread_score)
        if spread_score == "ideal":
            ranks_c = pair_ranks(scalarize(F - (F.min(axis=0) - ideal_margin), lam_np, "tch"), lam1)
        elif spread_score == "hv":
            ranks_c = pair_ranks(scalarize(F, lam_np, "hv", ref), lam1)

        # theta: centre-weighted ranks (equal weights for center_a = 1)
        direction = 2.0 * ((center_weights(lam, center_a) * ranks) @ eps) / (pop_size * sigma)
        theta.grad = direction.clone()
        optimizer.step()

        # brake: losses at the pairs' points (antithetic means), corner regions vs the centre region, averaged over
        # generations; the line runs past the front at a corner whose points are worse on every task
        braking = False
        if brake:
            regions = guide_regions(lam).numpy()
            points = 0.5 * (F[:M] + F[M:])
            if (regions == 0).any():
                centre = points[regions == 0].mean(axis=0)
                for r in range(1, K + 1):
                    if (regions == r).any():
                        gaps[r - 1] = grad_ema * gaps[r - 1] + (1.0 - grad_ema) * (points[regions == r].mean(axis=0) - centre)
            braking = has_tangent and bool((gaps > 0).all(axis=1).any())

        # c: chain rule through each pair's shift c S T^T phi_i (this generation's T), on log c; the brake
        # shrinks it by one step instead
        g_c = 2.0 * float(ranks_c @ ((eps[:, idx] @ T.T) * phi).sum(dim=1)) / (pop_size * sigma)
        if c_optimizer is not None and has_tangent:
            if braking:
                with torch.no_grad():
                    log_c -= c_lr
            else:
                log_c.grad = torch.tensor(c * g_c, dtype=torch.float32)
                c_optimizer.step()
            c = float(log_c.detach().exp())

        # tangent for the next generation: per-task gradient estimates -> moving averages G -> unit rows of -Q^T G
        task_r = torch.stack([task_ranks(F[:M, k] - F[M:, k]) for k in range(K)])  # [K, M]
        G = grad_ema * G + (1.0 - grad_ema) * (2.0 * (task_r @ eps) / (pop_size * sigma))
        T = tangent_rows(G, Q, idx)
        has_tangent = True
        grad_cos = _mean_pairwise_cosine(G)

        if guided_k > 0 and psl_guide == "tasks":
            # per-task directions of the pairs within each region of the simplex, as in PSL-ES, on theta only
            regions = guide_regions(lam)
            for r in range(K + 1):
                members = (regions == r).nonzero().flatten().numpy()
                if len(members) < 2:
                    continue
                for k in range(K):
                    v = task_ranks(F[members, k] - F[members + M, k]) @ eps[members]
                    v = v / v.norm().clamp_min(1e-30)
                    old = task_dirs.get((r, k))
                    task_dirs[(r, k)] = v if old is None else guided_ema * old + (1.0 - guided_ema) * v
            guide = list(task_dirs.values())
        elif guided_k > 0:
            guide = update_guide(guide, direction, guided_k, guided_ema)
        sigma = max(sigma * float(sigma_decay), float(sigma_min))
        elapsed = time.perf_counter() - t0

        td = theta.detach()
        b_norm = float(c * T.norm())  # |B| of B = c T^T (c sqrt(K - 1) once the tangent is estimated)
        pareto_front = F[pareto_front_indices(F)]
        trainer._print_evo_progress(
            method,
            iteration,
            num_iterations,
            pareto_front,
            extras=f"sigma={sigma:.4g} |d|={float(direction.norm()):.4g} c={c:.4g}" + (" (brake)" if braking else "")
                   + f" cos(g)={grad_cos:+.3f}",
            elapsed=elapsed,
            wandb_log_step=step,
        )
        trainer._wandb_log_evo_front_metrics(iteration, pareto_front, step, extras={
            "evo/psl/sigma": sigma,
            "evo/psl/direction_norm": float(direction.norm()),
            "evo/psl/b_norm": b_norm,
            "evo/psl/c": c,
            "evo/psl/brake": int(braking),
            "evo/psl/grad_cos": grad_cos,
        })

        # the balanced network theta: snapshot on the current batch (and test periodically)
        trainer.set_theta(td)
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
            front = evaluate_front(td, shared_batches)
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
                for part, ms in front.items():
                    for g, m in enumerate(ms):
                        row = {"generation": iteration + 1, "split": part}
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
                                   "b_norm": b_norm, "c": c, "brake": int(braking), "grad_cos": grad_cos}
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
    td = theta.detach()
    front = evaluate_front(td, shared_batches)
    grid_X = torch.stack([theta_at(td, grid_phi[g]) for g in range(len(grid))]).numpy()
    train_F = np.array([m["loss_item"] for m in front["train"]], dtype=np.float32)
    test_F = np.array([m["loss_item"] for m in front["test"]], dtype=np.float32).reshape(-1, K)
    test_acc = np.array([[m["results"][t][0] for t in tasks] for m in front["test"]], dtype=np.float32).reshape(-1, K)
    if getattr(trainer, "save_path", None):
        torch.save({
            "theta": td.clone(),                    # weights of the balanced network
            "c": c,                                 # length of the line
            "T": T.clone(),                         # unit tangent rows [K - 1, P] on the conditioned parameters
            "B": c * T.T.clone(),                   # the equivalent PSL-ES steering matrix c T^T [P, K - 1]
            "G": G.clone(),                         # moving averages of the per-task gradient estimates [K, D]
            "index": idx,                           # flat positions of the conditioned parameters
            "Q": Q,                                 # phi(lam) = sqrt(2) Q^T (lam - 1/K)
            "grid": grid, "train_F": train_F, "test_F": test_F, "test_acc": test_acc,
            "tasks": tasks, "psl_params": psl_params, "scalarization": scalarization, "method": method,
            "pop_size": pop_size, "lr": float(lr), "sigma": sigma,
        }, os.path.join(trainer.save_path, "tpsl_result.pt"))

    center_X = td.numpy().copy()
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
        "c": c,
        "front_lambda": grid.numpy(),
        "front_test_F": test_F,
    }
    trainer.last_evo = out
    return out
