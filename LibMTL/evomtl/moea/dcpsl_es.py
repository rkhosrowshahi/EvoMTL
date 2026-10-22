"""Dual-cone Pareto set learning with OpenES (DCPSL-ES): a Pareto front moved along non-conflicting directions.

Same family of networks as PSL-ES (:mod:`~LibMTL.evomtl.moea.psl_es`), for M objectives::

    theta(lam) = theta + S B phi(lam),        phi(lam) = sqrt(2) Q^T (lam - 1/M)

``theta`` is the balanced network (``lam = 1/M``). ``B`` (``[P, M - 1]`` on the conditioned parameters, scattered by
``S``) spans a simplex whose M corners ``c_m = theta + S B phi_m``, ``phi_m = sqrt(2) Q^T e_m``, are the networks for
one objective each; for two objectives ``phi = lam_1 - lam_2`` and the front is the segment from ``theta - B`` to
``theta + B``.

PSL-ES moves the family along the gradient of a scalarized score. Here every anchor point moves along a direction
from the dual cone of its gradients (``J``: rows ``g_m``; ``G = J J^T``)::

    C* = {d : J d >= 0}                                   no loss rises along -d, to first order
    pi(v) = argmin_{d in C*} |d - v|^2                     the projection onto C*
    pi(g_m) = J^T w_m,   w_m = argmin_{w >= e_m} w^T G w    an M-dim QP (UPGrad's, ``_project_weights``)

* ``theta`` moves along UPGrad at the centre, ``(1/M) sum_m pi(g_m)``: the balanced, non-conflicting direction
  (``g_m . d >= 0`` for every m). With ``normalize`` (default) each objective's gradient estimate is scaled to unit
  length first. That leaves ``C*`` unchanged but weighs the objectives equally: for two objectives the direction makes
  the same angle with both gradients, so each loss falls at the same fraction of its own steepest-descent rate (to
  first order), instead of the objective with the larger gradient taking the larger share;
* corner ``m`` should move along ``d_m = pi(g_m)`` computed at that corner: the steepest first-order improvement of
  objective m that raises no other loss. ``B`` takes the least-squares fit of the corners' moves relative to the
  centre, ``argmin_dB sum_m |dB phi_m - (d_m - mean_j d_j)|^2 = (1/sqrt(2)) [d_1 ... d_M] Q`` (``Phi = sqrt(2) Q^T``
  gives ``Phi Phi^T = 2 I``, and ``1^T Q = 0``), so corner m moves by ``d_theta + d_m - mean_j d_j``. For two
  objectives ``B`` gets half the difference of the two end moves.

Adam steps on ``x = (theta, B)`` with these directions; ``B`` starts at 0, so every corner starts at ``theta_0``.

Anchor sampling: of the N/2 antithetic pairs, ``center_fraction`` sit at ``theta`` and the rest are split equally
between the M corners. Each anchor's Jacobian comes from its own pairs, so corner samples never enter ``theta``'s
step: the per-objective pair differences ``d_im = f_m(+) - f_m(-)`` (shaped: ``d / 2`` raw, or centered ranks) weight
the isotropic noise ``e_i``, ``g_m = sum_i d_im e_i / (2 n sigma sqrt(alpha))``. With Guided ES sampling
(``guided_k > 0``) the perturbation is ``sigma (sqrt(alpha) e + s U e_k)``; regressing on its isotropic part ``e``
keeps the estimate unbiased, while regressing on the whole perturbation would scale the guide directions by
``(alpha + s^2) / alpha``. The guide is the M dual-cone edges at the centre (unit ``pi(g_m)``, moving averages with
decay ``guided_ema``), so exploration stretches along the non-conflicting directions symmetrically across the
objectives; ``guided_k`` only switches it on.

Every move lowers or keeps every loss to first order, so the corners stay in the region that dominates where they
started (inside the hypervolume box when its reference point is the initial losses). At a Pareto-stationary point
(``sum_m lam_m g_m = 0`` with every ``lam_m > 0``) every projection is 0 and the point stops.

The front is read out on a grid of preferences like PSL-ES (``front.csv`` at every test evaluation,
``dcpsl_result.pt`` at the end). ``theta`` is the returned solution and fills ``metrics.csv``, which also logs
``b_norm``, the centre's UPGrad weights and ``grad_cos`` (mean pairwise cosine of the centre's gradient estimates).

References:
- UPGrad: Quinton and Rey, "Jacobian Descent for Multi-Objective Optimization", 2024, https://arxiv.org/abs/2406.16232
- PSL-ES: :mod:`~LibMTL.evomtl.moea.psl_es`
- Guided ES: Maheswaranathan et al., "Guided evolutionary strategies", ICML 2019
"""

from __future__ import annotations

import csv
import math
import os
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from ...weighting.UPGrad import _compute_gramian, _normalize, _project_weights, _regularize
from .common import configure_moea_random_seed, should_run_full_test
from .hv import pareto_front_indices
from .mo_openes import _mean_pairwise_cosine, aggregate_jacobian, centered_ranks, guided_noise, make_es_optimizer
from .psl_es import conditioned_indices, hypervolume, preference_features, preference_grid, simplex_basis


def dual_cone_weights(J: torch.Tensor, U: torch.Tensor, norm_eps: float = 1e-4, reg_eps: float = 1e-4) -> torch.Tensor:
    """Weights ``W`` (shape of ``U``) such that ``J^T w`` is the projection of ``J^T u`` onto the dual cone
    ``{d : J d >= 0}`` for every row ``u`` of ``U``: UPGrad's QP ``min_{w >= u} w^T G w`` on the normalised,
    regularised Gramian ``G = J J^T``."""
    G = _regularize(_normalize(_compute_gramian(J), norm_eps), reg_eps)
    return _project_weights(U, G)


def run_dcpsl_es(
    trainer: Any,
    train_dataloaders: Any,
    num_iterations: int,
    pop_size: int,
    lr: float = 1e-3,
    sigma: float = 0.01,
    sigma_decay: float = 1.0,
    sigma_min: float = 0.0,
    psl_params: str = "all",
    eval_points: int = 11,
    center_fraction: float = 0.5,
    normalize: bool = True,
    fitness_shaping: str = "raw",
    antithetic: bool = True,
    optim: str = "adam",
    adam_betas: Tuple[float, float] = (0.9, 0.999),
    adam_eps: float = 1e-8,
    rmsprop_alpha: float = 0.99,
    rmsprop_eps: float = 1e-8,
    upgrad_norm_eps: float = 1e-4,
    upgrad_reg_eps: float = 1e-4,
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
    """DCPSL-ES ask-and-tell on ``x = (theta, B)`` (see module docstring).

    Parameters
    ----------
    psl_params, eval_points
        As in :func:`~LibMTL.evomtl.moea.psl_es.run_psl_es`: the parameters ``B`` acts on, and the preferences per
        simplex edge on which the front is read out.
    center_fraction : float
        Fraction of the antithetic pairs evaluated at ``theta``; the rest are split equally between the M corners
        (each corner needs at least 2 pairs).
    normalize : bool
        Scale each objective's gradient estimate to unit length before the dual-cone projections (default), or use
        the estimates as they are (UPGrad as in :func:`~LibMTL.evomtl.moea.mo_openes.run_mo_openes`).
    fitness_shaping : str
        ``"raw"`` (default): per-objective pair differences ``d / 2``; ``"centered_ranks"``: their centered ranks
        within each anchor.
    upgrad_norm_eps, upgrad_reg_eps
        Gramian normalisation and regularisation of the UPGrad QP.
    guided_k, guided_alpha, guided_sigma, guided_ema
        Guided ES sampling (``guided_k > 0`` switches it on): the guide is the M dual-cone edges at the centre,
        moving averages with decay ``guided_ema``; ``guided_alpha`` and ``guided_sigma`` as in MO-OpenES.

    The other parameters are those of :func:`~LibMTL.evomtl.moea.mo_openes.run_mo_openes`. Needs antithetic sampling.
    """
    if trainer.theta_0 is None:
        raise RuntimeError("prepare_evolution() first.")
    fitness_shaping = str(fitness_shaping).strip().lower()
    if fitness_shaping not in ("centered_ranks", "raw"):
        raise ValueError(f"Unknown fitness_shaping {fitness_shaping!r}; options: centered_ranks, raw")
    raw = fitness_shaping == "raw"
    pop_size = int(pop_size)
    if not antithetic or pop_size % 2 != 0:
        raise ValueError(f"DCPSL-ES needs antithetic sampling and an even pop_size, got antithetic={antithetic}, "
                         f"pop_size={pop_size}.")
    K = int(trainer.task_num)
    Mp = pop_size // 2
    center_fraction = float(center_fraction)
    if not 0.0 < center_fraction < 1.0:
        raise ValueError(f"center_fraction must be in (0, 1), got {center_fraction}")
    n_corner = (Mp - int(round(center_fraction * Mp))) // K
    n_center = Mp - K * n_corner
    if n_corner < 2 or n_center < 2:
        raise ValueError(f"DCPSL-ES with {K} objectives and center_fraction={center_fraction} needs more pairs: "
                         f"pop_size={pop_size} gives {n_center} at the centre and {n_corner} per corner (2 needed).")

    configure_moea_random_seed(seed)
    gen = torch.Generator().manual_seed(int(seed))

    D = int(trainer.theta_0.numel())
    idx = conditioned_indices(trainer.model, psl_params)
    P, C = int(idx.numel()), K - 1
    Q = simplex_basis(K)
    corner_phi = preference_features(torch.eye(K, dtype=torch.float32), Q)  # [K, C]: phi_m = sqrt(2) Q^T e_m
    n_var = D + P * C
    x = torch.zeros(n_var, dtype=torch.float32)  # x = (theta, B)
    x[:D] = trainer.theta_0.detach().cpu().float()  # theta starts at theta_0, B at 0
    x.requires_grad_(True)
    optim = str(optim).strip().lower()
    optimizer = make_es_optimizer([x], optim, lr, adam_betas, adam_eps, rmsprop_alpha, rmsprop_eps)
    sigma = float(sigma)
    method = "DCPSL-OpenES-UPGrad"
    if guided_k > 0:  # Guided ES sampling: DCPSL-OpenES-UPGrad -> DCPSL-GuidedES-UPGrad
        method = method.replace("OpenES", "GuidedES")
    # anchor of each pair: 0 = the centre theta, m = corner m (n_center pairs, then n_corner per corner)
    anchor = torch.cat([torch.zeros(n_center, dtype=torch.long),
                        torch.arange(1, K + 1, dtype=torch.long).repeat_interleave(n_corner)])
    guide: list = []
    edges_avg: Optional[torch.Tensor] = None  # moving averages of the centre's unit dual-cone edges [K, D]
    eye = torch.eye(K, dtype=torch.float32)
    grid = preference_grid(K, eval_points)
    grid_phi = preference_features(grid, Q)
    balanced_phi = torch.zeros(C, dtype=torch.float32)  # phi(1/K) = 0
    ref = None if hv_ref_point is None else np.asarray(hv_ref_point, dtype=np.float64).reshape(K)

    def theta_at(xv: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """Weights ``theta + S B phi`` of the network for preference features ``phi`` (``[C]``)."""
        t = xv[:D].clone()
        t.index_add_(0, idx, xv[D:].view(P, C) @ phi)
        return t

    def jacobian(a: int, diff: np.ndarray, e: torch.Tensor, scale: float) -> torch.Tensor:
        """Per-objective gradient estimates ``[K, D]`` at anchor ``a`` from its pairs' loss differences ``diff``
        (``[Mp, K]``) and isotropic noise ``e`` (``[Mp, D]``); rows scaled to unit length with ``normalize``."""
        rows = (anchor == a).nonzero().flatten().numpy()
        d = diff[rows]
        r = d / 2.0 if raw else np.stack([centered_ranks(np.concatenate([d[:, k], -d[:, k]]))[: len(rows)]
                                          for k in range(K)], axis=1)
        J = torch.as_tensor(r.T, dtype=torch.float32) @ e[rows] / (len(rows) * sigma * scale)
        return J / J.norm(dim=1, keepdim=True).clamp_min(1e-30) if normalize else J

    def evaluate_front(xv: torch.Tensor, batches: list) -> Dict[str, list]:
        """Metrics of ``theta(lam)`` for every grid preference: current train batch and test set."""
        out: Dict[str, list] = {"train": [], "test": []}
        for g in range(len(grid)):
            trainer.set_theta(theta_at(xv, grid_phi[g]))
            out["train"].append(trainer.metrics_snapshot_from_prefetched_batches(batches))
            if test_dataloaders is not None:
                out["test"].append(trainer.test(test_dataloaders, epoch=None, mode="test",
                                                suppress_display=True, return_metrics=True))
        return out

    print(
        f"[EvoMTL] {method}: dims={n_var} | pop={pop_size} | optim={optim} lr={lr} | sigma={sigma} "
        f"(decay={sigma_decay}, min={sigma_min}) | conditioned params={P} (psl_params={psl_params}), "
        f"theta dims={D}, B dims={P * C} | anchors: centre {n_center} pairs, {K} corners x {n_corner} pairs | "
        f"normalize={normalize} | front grid={len(grid)} preferences"
        + (f" | guided alpha={guided_alpha}, sigma_sub={guided_sigma or 'auto'}, ema={guided_ema}, "
           f"guide = {K} dual-cone edges" if guided_k > 0 else "")
        + f" | fitness={fitness_shaping}",
        flush=True,
    )

    num_iterations = int(num_iterations)
    tasks = list(trainer.task_name)
    # metrics.csv: the balanced network theta, PSL-ES's columns plus the centre's UPGrad weights and grad_cos.
    # front.csv: every grid preference at every test evaluation (split = train batch / test set).
    csv_cols = (
        ["generation", "time_s", "sigma"]
        + [f"pop_loss_mean_{t}" for t in tasks]
        + [f"train_loss_{t}" for t in tasks] + [f"train_acc_{t}" for t in tasks]
        + [f"test_loss_{t}" for t in tasks] + [f"test_acc_{t}" for t in tasks]
        + [f"weight_{t}" for t in tasks] + ["b_norm", "grad_cos", "test_hv"]
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

    F = np.zeros((pop_size, K), dtype=np.float64)
    shared_batches: list = []
    for iteration in range(num_iterations):
        t0 = time.perf_counter()
        step = wandb_step_offset + iteration
        test_metrics = None

        # ask: isotropic noise e (kept for the estimates), guided noise eps, perturbations z = sigma eps
        e = torch.randn(Mp, D, generator=gen, dtype=torch.float32)
        guided = guided_k > 0 and bool(guide)
        eps = guided_noise(e, guide, gen, sigma, guided_alpha, guided_sigma) if guided else e
        iso = math.sqrt(guided_alpha) if guided else 1.0  # e enters eps as sqrt(alpha) e
        z = sigma * eps

        # evaluate every member on the same mini-batch(es): rows i (anchor + z_i) and Mp + i (anchor - z_i)
        shared_batches = trainer._sample_batches(train_dataloaders, num_batches)
        xd = x.detach()
        points = [xd[:D].clone()] + [theta_at(xd, corner_phi[m]) for m in range(K)]
        for i in range(Mp):
            p = points[int(anchor[i])]
            for s, sign in enumerate((1.0, -1.0)):
                trainer.set_theta(p + sign * z[i])
                F[i + s * Mp] = trainer.evaluate_multiobjective_losses(
                    train_dataloaders, prefetched_batches=shared_batches
                )

        # tell: Jacobian at each anchor -> UPGrad for theta, dual-cone edge per corner -> least-squares B
        diff = F[:Mp] - F[Mp:]
        J_center = jacobian(0, diff, e, iso)
        d_theta, w = aggregate_jacobian(J_center, "upgrad", upgrad_norm_eps, upgrad_reg_eps)
        corner_dirs = []
        for m in range(K):
            J_m = jacobian(m + 1, diff, e, iso)
            corner_dirs.append(dual_cone_weights(J_m, eye[m:m + 1], upgrad_norm_eps, upgrad_reg_eps)[0] @ J_m)
        grad_B = torch.stack(corner_dirs, dim=1)[idx] @ Q / math.sqrt(2.0)  # [P, C]
        direction = torch.cat([d_theta, grad_B.flatten()])
        x.grad = direction.clone()
        optimizer.step()
        grad_cos = _mean_pairwise_cosine(J_center)

        if guided_k > 0:  # the guide: the centre's K dual-cone edges, unit length, moving averages
            edges = dual_cone_weights(J_center, eye, upgrad_norm_eps, upgrad_reg_eps) @ J_center
            edges = edges / edges.norm(dim=1, keepdim=True).clamp_min(1e-30)
            edges_avg = edges if edges_avg is None else guided_ema * edges_avg + (1.0 - guided_ema) * edges
            guide = [v.clone() for v in edges_avg if float(v.norm()) > 1e-6]
        sigma = max(sigma * float(sigma_decay), float(sigma_min))
        elapsed = time.perf_counter() - t0

        xd = x.detach()
        b_norm = float(xd[D:].norm())
        pareto_front = F[pareto_front_indices(F)]
        trainer._print_evo_progress(
            method,
            iteration,
            num_iterations,
            pareto_front,
            extras=f"sigma={sigma:.4g} |d|={float(direction.norm()):.4g} |B|={b_norm:.4g} "
                   f"w=[" + ", ".join(f"{float(v):.3f}" for v in w) + f"] cos(g)={grad_cos:+.3f}",
            elapsed=elapsed,
            wandb_log_step=step,
        )
        trainer._wandb_log_evo_front_metrics(iteration, pareto_front, step, extras={
            "evo/psl/sigma": sigma,
            "evo/psl/direction_norm": float(direction.norm()),
            "evo/psl/b_norm": b_norm,
            "evo/psl/grad_cos": grad_cos,
            **{f"evo/psl/weight/{t}": float(w[k]) for k, t in enumerate(tasks)},
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
            test_F = np.array([m["loss_item"] for m in front["test"]], dtype=np.float64)
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
                                   "b_norm": b_norm, "grad_cos": grad_cos}
            for k, t in enumerate(tasks):
                row[f"pop_loss_mean_{t}"] = float(F[:, k].mean())
                row[f"train_loss_{t}"] = float(train_batch_metrics["loss_item"][k])
                row[f"train_acc_{t}"] = float(train_batch_metrics["results"][t][0])
                row[f"weight_{t}"] = float(w[k])
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
    train_F = np.array([m["loss_item"] for m in front["train"]], dtype=np.float64)
    test_F = np.array([m["loss_item"] for m in front["test"]], dtype=np.float64).reshape(-1, K)
    test_acc = np.array([[m["results"][t][0] for t in tasks] for m in front["test"]], dtype=np.float64).reshape(-1, K)
    if getattr(trainer, "save_path", None):
        torch.save({
            "theta": xd[:D].clone(),                # weights of the balanced network
            "B": xd[D:].view(P, C).clone(),         # steering matrix on the conditioned parameters
            "index": idx,                           # flat positions of the conditioned parameters
            "Q": Q,                                 # phi(lam) = sqrt(2) Q^T (lam - 1/K)
            "corner_phi": corner_phi,               # phi of the K corners (rows)
            "grid": grid, "train_F": train_F, "test_F": test_F, "test_acc": test_acc,
            "tasks": tasks, "psl_params": psl_params, "method": method, "normalize": bool(normalize),
            "center_fraction": center_fraction, "pop_size": pop_size, "lr": float(lr), "sigma": sigma,
        }, os.path.join(trainer.save_path, "dcpsl_result.pt"))

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
        "front_lambda": grid.numpy(),
        "front_test_F": test_F,
    }
    trainer.last_evo = out
    return out
