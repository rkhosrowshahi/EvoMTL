"""Persistent-center vector OpenES with constrained objective-space spreading.

All objectives are minimized. Perturbations are disposable gradient probes; only
centers own moving statistics and survive a step. No loss scalarization/archive.
"""
from __future__ import annotations

import random
from typing import Any, Dict

import numpy as np
import torch
from scipy.optimize import minimize

from .common import should_run_full_test
from .mo_mgda_openes import min_norm_simplex


def antithetic_jacobian(plus, minus, noise, sigma):
    """Raw vector estimator: J = (F+ - F-)^T E / (2 P sigma)."""
    if sigma <= 0 or noise.ndim != 2 or len(noise) == 0:
        raise ValueError("positive sigma and nonempty [pairs, parameters] noise required")
    plus = torch.as_tensor(plus, dtype=noise.dtype, device=noise.device)
    minus = torch.as_tensor(minus, dtype=noise.dtype, device=noise.device)
    if plus.ndim != 2 or plus.shape != minus.shape or len(plus) != len(noise):
        raise ValueError("plus/minus must have matching [pairs, objectives] shapes")
    return (plus - minus).T @ noise / (2 * len(noise) * sigma)


def pooled_sample_proposals(centers, samples, losses, center_losses, scale, repulsion,
                            target_distance=0.02, temperature=0.005):
    """Center-specific kernel recombination of ALL current probe networks.

    Each target moves toward lower vector losses and away from nearby centers.
    Kernel weights select proximity in loss space, never a weighted task loss.
    The weighted perturbation proposals are biased selection directions, NOT estimates of
    the receiving center's task Jacobian. The receiver's own Jacobian must
    constrain them through ``coordinated_direction``.
    """
    centers, samples, losses, center_losses, scale, repulsion = [
        np.asarray(x, dtype=np.float64) for x in
        (centers, samples, losses, center_losses, scale, repulsion)]
    if (centers.ndim != 2 or samples.ndim != 2 or losses.ndim != 2
            or center_losses.ndim != 2 or not len(centers) or losses.shape[1] == 0
            or samples.shape[1] != centers.shape[1]
            or losses.shape[0] != samples.shape[0] or not len(samples)
            or center_losses.shape != (len(centers), losses.shape[1])
            or repulsion.shape != center_losses.shape or scale.shape != (losses.shape[1],)
            or not (scale > 0).all() or target_distance <= 0 or temperature <= 0
            or not np.isfinite([target_distance, temperature]).all()
            or not all(np.isfinite(x).all() for x in
                       (centers, samples, losses, center_losses, scale, repulsion))):
        raise ValueError("invalid shared-pool arrays or kernel settings")
    unit = repulsion / np.maximum(np.linalg.norm(repulsion, axis=1, keepdims=True), 1e-30)
    improvement = np.ones(losses.shape[1]) / np.sqrt(losses.shape[1])
    targets = center_losses / scale + target_distance * (unit - improvement)
    distances = np.sum(((losses[None] / scale - targets[:, None]) / temperature)**2, axis=2)
    log_weights = -0.5 * distances
    log_weights -= np.max(log_weights, axis=1, keepdims=True)
    weights = np.exp(log_weights)
    weights /= weights.sum(axis=1, keepdims=True)
    # The pool is grouped by generating center with the same probe count.
    # Share mutations, not absolute network positions: uniform weights on
    # antithetic pairs must produce zero steering rather than center consensus.
    if len(samples) % len(centers):
        raise ValueError("shared pool requires equal probe counts per generating center")
    sources = np.repeat(np.arange(len(centers)), len(samples) // len(centers))
    proposals = weights @ (samples - centers[sources])
    return proposals, weights, targets


def coordinated_direction(J, target, spread_radius, tradeoff, descent_fraction,
                          desired_direction=None):
    """MGDA descent + bounded steering in the task/proposal span.

    Return parameter directions d, s satisfying ||s|| <= spread_radius and
    J(d+s) <= descent_fraction * J d + tradeoff (up to solver tolerance).
    Solve in the span of task gradients and an optional external parameter
    proposal (at most K+1 dimensions), never a D-dimensional QP.
    """
    J = np.asarray(J, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if (J.ndim != 2 or target.shape != (len(J),) or not np.isfinite(J).all()
            or not np.isfinite(target).all()):
        raise ValueError("finite Jacobian and matching objective target required")
    if spread_radius < 0 or tradeoff < 0 or not 0 <= descent_fraction <= 1:
        raise ValueError("invalid spreading constraints")
    gram = J @ J.T
    alpha = min_norm_simplex(gram)
    d = -alpha @ J
    # Numerical MGDA errors must not introduce an ascent common direction.
    if np.max(J @ d) > 1e-8:
        d = np.zeros_like(d)
    if spread_radius == 0:
        return d, np.zeros_like(d)
    # Adding the external proposal to the basis preserves parameter transfer
    # in directions orthogonal to the receiver's task gradients.
    if desired_direction is None:
        basis_rows = J
    else:
        desired_direction = np.asarray(desired_direction, dtype=np.float64)
        if desired_direction.shape != (J.shape[1],) or not np.isfinite(desired_direction).all():
            raise ValueError("invalid external parameter proposal")
        basis_rows = np.vstack((J, desired_direction))
    vals, vecs = np.linalg.eigh(basis_rows @ basis_rows.T)
    keep = vals > max(1e-14, float(vals.max()) * 1e-10)
    if not keep.any():
        return d, np.zeros_like(d)
    # B has orthonormal rows in parameter space, A = J B^T.
    B = (vecs[:, keep].T @ basis_rows) / np.sqrt(vals[keep])[:, None]
    A = J @ B.T
    desired = A.T @ target if desired_direction is None else B @ desired_direction
    desired *= min(1.0, spread_radius / max(np.linalg.norm(desired), 1e-30))
    bound = tradeoff - (1 - descent_fraction) * (J @ d)
    result = minimize(
        lambda z: 0.5 * np.sum((z - desired) ** 2), np.zeros(len(desired)),
        jac=lambda z: z - desired, method="SLSQP",
        constraints=[
            {"type": "ineq", "fun": lambda z: bound - A @ z,
             "jac": lambda z: -A},
            {"type": "ineq", "fun": lambda z: spread_radius**2 - z @ z,
             "jac": lambda z: -2 * z},
        ], options={"ftol": 1e-10, "maxiter": 200},
    )
    z = result.x
    if (not result.success or not np.isfinite(z).all()
            or np.linalg.norm(z) > spread_radius + 1e-7
            or np.max(A @ z - bound) > 1e-7):
        z = np.zeros_like(desired)  # safe fallback: common descent only
    return d, z @ B


class CenterStatistics:
    """EMA mean and central second moment, per persistent center.

    Variance describes a moving stochastic trajectory, not a stationary standard
    error or a confidence guarantee. Probe losses never enter these statistics.
    """
    def __init__(self, beta):
        self.beta = beta
        self.count = 0
        self.loss_mean = self.loss_var = self.grad_mean = self.grad_var = None

    def update(self, losses, gradients):
        for prefix, value in (("loss", losses), ("grad", gradients)):
            value = value.detach().clone()
            mean = getattr(self, prefix + "_mean")
            if mean is None:
                setattr(self, prefix + "_mean", value)
                setattr(self, prefix + "_var", torch.zeros_like(value))
            else:
                delta = value - mean
                var = getattr(self, prefix + "_var")
                setattr(self, prefix + "_var", self.beta * (var + (1-self.beta) * delta.square()))
                setattr(self, prefix + "_mean", mean + (1-self.beta) * delta)
        self.count += 1

    def state_dict(self):
        return {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in vars(self).items()}


def run_multi_center_pareto_openes(
    trainer: Any, train_dataloaders: Any, num_iterations: int, pop_size: int,
    num_centers: int = 5, sigma: float = 0.01, lr: float = 0.001,
    init_std: float = 0.001, stats_beta: float = 0.9,
    sample_pool: str = "off", pool_target_distance: float = 0.02,
    pool_temperature: float = 0.005,
    spread_radius: float = 0.1, tradeoff: float = 0.01,
    descent_fraction: float = 0.5, max_step_norm: float = 1.0,
    num_batches: int = 1, seed: int = 0, test_dataloaders: Any = None,
    evo_eval_freq: Any = 0, wandb_step_offset: int = 0,
) -> Dict[str, Any]:
    """Run vector OpenES. ``pop_size`` is the even probe budget *per center*.

    ``tradeoff`` bounds each normalized objective's linearized ascent per unit
    learning rate; zero keeps estimated common descent. A positive allowance is
    needed to move along a conflicting front at Pareto stationarity. Plain SGD
    preserves these constraints; adaptive optimizers would change the direction.
    """
    scalars = [sigma, lr, init_std, stats_beta, spread_radius, tradeoff,
               descent_fraction, max_step_norm, pool_target_distance, pool_temperature]
    if (not np.isfinite(scalars).all() or num_iterations < 0 or num_centers < 1
            or pop_size < 2 or pop_size % 2 or num_batches < 1
            or sigma <= 0 or lr <= 0 or init_std < 0 or not 0 <= stats_beta < 1
            or spread_radius < 0 or tradeoff < 0 or not 0 <= descent_fraction <= 1
            or max_step_norm <= 0 or sample_pool not in ("off", "kernel")
            or pool_target_distance <= 0 or pool_temperature <= 0):
        raise ValueError("invalid Multi-Center Pareto-OpenES configuration")
    if trainer.theta_0 is None:
        raise RuntimeError("prepare_evolution() first")
    base = trainer.theta_0.detach().cpu().clone()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    centers = base.repeat(num_centers, 1)
    centers += init_std * torch.randn(centers.shape, generator=generator, dtype=base.dtype)
    centers[0] = base
    stats = CenterStatistics(stats_beta)
    model = trainer.model
    original_modes = [(module, module.training) for module in model.modules()]
    buffers = {name: b.detach().clone() for name, b in model.named_buffers()}
    current_F = None
    center_evaluations = []
    pool_diagnostics = []
    last_pool = None

    def matched_evaluator(batches):
        # Replay stochastic forward draws for all centers and both probe signs.
        py_state, np_state = random.getstate(), np.random.get_state()
        cpu_state = torch.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

        def evaluate(theta):
            with torch.no_grad():
                for name, buf in model.named_buffers():
                    buf.copy_(buffers[name])
            trainer.set_theta(theta)
            random.setstate(py_state)
            np.random.set_state(np_state)
            torch.set_rng_state(cpu_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)
            try:
                value = np.asarray(trainer.evaluate_multiobjective_losses(
                    train_dataloaders, prefetched_batches=batches, record_losses=False), dtype=np.float64)
                if value.shape != (trainer.task_num,) or not np.isfinite(value).all():
                    raise ValueError("objective evaluation must return a finite task vector")
                return value
            finally:
                random.setstate(py_state)
                np.random.set_state(np_state)
                torch.set_rng_state(cpu_state)
                if cuda_states is not None:
                    torch.cuda.set_rng_state_all(cuda_states)
        return evaluate

    def record_test_evaluation(iteration):
        # Evaluation must not change subsequent training-batch RNG draws.
        py_state, np_state = random.getstate(), np.random.get_state()
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        metrics = []
        try:
            with torch.random.fork_rng(devices=devices):
                for center in centers:
                    trainer.set_theta(center)
                    metrics.append(trainer.test(test_dataloaders, suppress_display=True,
                                                return_metrics=True, log_wandb=False))
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
        losses = np.stack([m["loss_item"] for m in metrics])
        parameter_distances = [float((centers[i]-centers[j]).norm())
                               for i in range(num_centers) for j in range(i)]
        objective_distances = [float(np.linalg.norm(losses[i]-losses[j]))
                               for i in range(num_centers) for j in range(i)]
        center_evaluations.append(dict(iteration=iteration, metrics=metrics,
                                       parameter_distances=parameter_distances,
                                       objective_distances=objective_distances))
        print(f"[EvoMTL] Center evaluation iteration={iteration} "
              f"test_losses={losses.tolist()} "
              f"mean_parameter_distance={np.mean(parameter_distances) if parameter_distances else 0:.6g}",
              flush=True)
        if getattr(trainer, "_wandb_enabled", False):
            values = {f"evo/center_{i}/test_loss_{j}": float(loss)
                      for i, row in enumerate(losses) for j, loss in enumerate(row)}
            for i, metric in enumerate(metrics):
                for task, scores in metric["results"].items():
                    for j, score in enumerate(scores):
                        values[f"evo/center_{i}/test/{task}/metric_{j}"] = float(score)
            trainer._wandb_log(values, step=wandb_step_offset + max(0, iteration-1))

    try:
        if test_dataloaders is not None and (evo_eval_freq if isinstance(evo_eval_freq, (set, frozenset))
                                             else evo_eval_freq > 0):
            record_test_evaluation(0)
        for step in range(num_iterations):
            batches = trainer._sample_batches(train_dataloaders, num_batches)
            evaluate = matched_evaluator(batches)
            current_F = np.stack([evaluate(c) for c in centers])
            jacobians = []
            pool_samples, pool_losses = [], []
            # Shared directions across centers also reduce comparison noise.
            noise = torch.randn((pop_size // 2, base.numel()), generator=generator, dtype=base.dtype)
            for center in centers:
                plus = np.stack([evaluate(center + sigma * e) for e in noise])
                minus = np.stack([evaluate(center - sigma * e) for e in noise])
                jacobians.append(antithetic_jacobian(plus, minus, noise, sigma))
                if sample_pool == "kernel":
                    pool_samples.extend((center[None] + sigma * noise, center[None] - sigma * noise))
                    pool_losses.extend((plus, minus))
            stats.update(torch.as_tensor(current_F), torch.stack(jacobians))
            means = stats.loss_mean.numpy()
            # One common objective scale for every center, never probe extrema.
            scale = np.maximum(np.mean(np.abs(means), axis=0), 1e-8)
            normalized = means / scale
            targets = np.zeros_like(normalized)
            for i in range(num_centers):
                for j in range(i):
                    diff = normalized[i] - normalized[j]
                    if np.linalg.norm(diff) < 1e-10:
                        # Deterministic symmetry breaking in objective space.
                        diff = np.cos((i + 1) * np.arange(1, trainer.task_num + 1))
                        diff -= diff.mean()
                    force = diff / max(np.dot(diff, diff), 1e-4)
                    targets[i] += force
                    targets[j] -= force
            proposals = None
            if sample_pool == "kernel":
                probe_losses = np.concatenate(pool_losses)
                proposals, weights, loss_targets = pooled_sample_proposals(
                    centers.numpy(), torch.cat(pool_samples).numpy(), probe_losses,
                    current_F, scale, targets, pool_target_distance, pool_temperature)
                # Source j's mass in receiving center i's kernel recombination.
                source_mass = weights.reshape(num_centers, num_centers, pop_size).sum(axis=2)
                ess = 1 / np.sum(weights**2, axis=1)
                pool_diagnostics.append(dict(iteration=step+1, effective_samples=ess,
                                             source_mass=source_mass,
                                             foreign_mass=1-np.diag(source_mass)))
                last_pool = dict(iteration=step+1, weights=weights, losses=probe_losses,
                                 center_losses=current_F.copy(), normalized_targets=loss_targets,
                                 scale=scale.copy(), source_indices=np.repeat(np.arange(num_centers), pop_size))
            # All directions use the same pre-update center snapshot.
            updates = []
            for i, J in enumerate(stats.grad_mean.numpy()):
                d, s = coordinated_direction(J / scale[:, None], targets[i], spread_radius,
                                              tradeoff, descent_fraction,
                                              desired_direction=None if proposals is None else proposals[i])
                update = lr * (d + s)
                update *= min(1.0, max_step_norm / max(np.linalg.norm(update), 1e-30))
                updates.append(update)
            centers += torch.as_tensor(np.stack(updates), dtype=centers.dtype)
            print(f"[EvoMTL] Multi-Center Pareto-OpenES {step+1}/{num_iterations} "
                  f"centers={num_centers} center_losses={current_F.tolist()}", flush=True)
            if getattr(trainer, "_wandb_enabled", False):
                trainer._wandb_log({f"evo/center_{i}/task_{j}": float(current_F[i, j])
                                   for i in range(num_centers) for j in range(trainer.task_num)},
                                  step=wandb_step_offset + step)
            if test_dataloaders is not None and should_run_full_test(step, num_iterations, evo_eval_freq):
                record_test_evaluation(step + 1)
        # Reevaluate ALL final centers on one fresh shared batch. Never mix stale
        # EMA values with current parameters or rank centers across mini-batches.
        evaluate = matched_evaluator(trainer._sample_batches(train_dataloaders, num_batches))
        current_F = np.stack([evaluate(c) for c in centers])
        X = centers.numpy().copy()
        out = dict(method="Multi-Center Pareto-OpenES" + (" (shared pool)" if sample_pool == "kernel" else ""),
                   sample_pool=sample_pool, pool_diagnostics=pool_diagnostics, last_pool=last_pool,
                   pool_settings=dict(target_distance=pool_target_distance, temperature=pool_temperature),
                   centers_X=X, centers_F=current_F,
                   center_statistics=stats.state_dict(), center_evaluations=center_evaluations,
                   representative_index=0,
                   pareto_X=X, pareto_F=current_F, pareto_center_X=X[0].copy(),
                   hv_weights=None, hv_ref_point=None)
        trainer.set_theta(centers[0])
        trainer.last_evo = out
        return out
    except Exception:
        trainer.set_theta(base)
        raise
    finally:
        with torch.no_grad():
            for name, buf in model.named_buffers():
                buf.copy_(buffers[name])
        for module, training in original_modes:
            module.training = training
