"""Shared MOEA loop utilities (population injection, test cadence)."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


def configure_moea_random_seed(seed: int) -> None:
    """Seed NumPy's legacy global RNG for MOEA runs.

    Pymoo algorithms also take ``seed=`` on construction (their own ``numpy.random.Generator``);
    this aligns any code paths that still draw from ``numpy.random.*`` without an explicit
    generator (e.g. some operators or user code).
    """
    np.random.seed(int(seed))


def should_run_full_test(step_idx: int, n_steps: int, every: int) -> bool:
    """Whether to run a full test pass this MOEA outer step.

    ``every <= 0`` disables in-loop test evaluation (train-snapshot logging only).
    ``every >= 1`` runs test on steps ``0, every, 2*every, ...`` and always on the
    last step ``n_steps - 1``.
    """
    if every <= 0:
        return False
    n_steps = int(n_steps)
    if n_steps <= 0:
        return False
    step_idx = int(step_idx)
    every = int(every)
    return (step_idx % every == 0) or (step_idx == n_steps - 1)


def inject_pareto_center_into_pop(
    *,
    algorithm: Any,
    pareto_center_X: np.ndarray,
    pareto_center_F: np.ndarray,
) -> Optional[int]:
    """Replace one population member (worst by sum of objectives) with the HV Pareto center.

    Keeps population size fixed. For MOPSO-CD, resets that particle's velocity to zero and
    aligns ``pbest`` / ``pbest_f`` with the injected point.

    Returns
    -------
    int or None
        Index of the replaced individual, or ``None`` if injection was skipped.
    """
    pop = algorithm.pop
    if pop is None or len(pop) == 0:
        return None
    pop_X = np.asarray(pop.get("X"), dtype=np.float64, copy=True)
    pop_F = np.asarray(pop.get("F"), dtype=np.float64, copy=True)
    center_x = np.asarray(pareto_center_X, dtype=np.float64).reshape(-1)
    pareto_center_f = np.asarray(pareto_center_F, dtype=np.float64).reshape(-1)
    if center_x.shape[0] != pop_X.shape[1] or pareto_center_f.shape[0] != pop_F.shape[1]:
        return None
    j = int(np.argmax(np.sum(pop_F, axis=1)))
    pop_X[j] = center_x
    pop_F[j] = pareto_center_f
    pop.set("X", pop_X)
    pop.set("F", pop_F)

    velocities = getattr(algorithm, "velocities", None)
    if velocities is not None:
        velocities[j] = 0.0

    pbest = getattr(algorithm, "pbest", None)
    if pbest is not None and len(pbest) == len(pop):
        pX = np.asarray(pbest.get("X"), dtype=np.float64, copy=True)
        pF = np.asarray(pbest.get("F"), dtype=np.float64, copy=True)
        pX[j] = center_x
        pF[j] = pareto_center_f
        pbest.set("X", pX)
        pbest.set("F", pF)

    pbest_f = getattr(algorithm, "pbest_f", None)
    if pbest_f is not None and getattr(pbest_f, "shape", None) == pop_F.shape:
        pbest_f[j] = pareto_center_f

    return j
