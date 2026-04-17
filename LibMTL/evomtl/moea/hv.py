"""Pareto-front indexing, hypervolume weights, and Pareto-center averaging."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .pymoo_deps import Hypervolume, NonDominatedSorting


def default_hv_ref_point(F: np.ndarray) -> np.ndarray:
    """Reference point for HV (minimization): slightly beyond the nadir of F."""
    F = np.atleast_2d(F)
    return np.max(F, axis=0) + 1e-6


def normalize_objectives_for_hv(
    F_nd: np.ndarray,
    ref_point: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Min-max each objective on the Pareto set (ideal→0, nadir→1).

    Matches ``hv_normalized_*_center``: ``scale = where(nadir-ideal > 1e-8, ...)``,
    default normalized HV reference ``1.1`` per objective. Custom ``ref_point``
    (raw space) uses the same ``ideal`` / ``scale``.
    """
    F_nd = np.atleast_2d(np.asarray(F_nd, dtype=np.float64))
    n, m = F_nd.shape
    if n == 0:
        return F_nd, np.full(m, 1.1, dtype=np.float64)
    ideal = np.min(F_nd, axis=0)
    nadir = np.max(F_nd, axis=0)
    scale = np.where(nadir - ideal > 1e-8, nadir - ideal, 1.0)
    F_norm = (F_nd - ideal) / scale
    if ref_point is None:
        ref_norm = np.full(m, 1.1, dtype=np.float64)
    else:
        ref = np.asarray(ref_point, dtype=np.float64).reshape(m)
        ref_norm = (ref - ideal) / scale
    return F_norm, ref_norm


def pareto_front_indices(F: np.ndarray) -> np.ndarray:
    """Indices of the first nondominated front (minimization)."""
    F = np.atleast_2d(F)
    nds = NonDominatedSorting()
    return nds.do(F, only_non_dominated_front=True)


def pareto_center_weights_from_marginal_hv(
    contributions: np.ndarray,
    *,
    aggregation: str = "linear",
    softmax_temperature: float = 1.0,
) -> np.ndarray:
    """Turn raw marginal HV contributions into weights for a Pareto-set average.

    * ``linear``: nonnegative contributions, normalized to sum to 1 (same as
      ``hv_normalized_linear_center``).
    * ``softmax``: ``exp((c/temp) - max(c/temp))`` normalized (same as
      ``hv_normalized_softmax_center``).
    """
    contributions = np.asarray(contributions, dtype=np.float64).reshape(-1)
    n = contributions.shape[0]
    if n == 0:
        return np.zeros(0)
    agg = aggregation.strip().lower()
    if agg == "softmax":
        temp = max(float(softmax_temperature), 1e-8)
        scaled = contributions / temp
        scaled = scaled - np.max(scaled)
        expv = np.exp(scaled)
        s = float(expv.sum())
        if s < 1e-12:
            return np.ones(n) / n
        return expv / s
    if agg != "linear":
        raise ValueError(
            "aggregation must be 'linear' or 'softmax', got {!r}".format(aggregation)
        )
    contribs = np.maximum(0.0, contributions)
    s = contribs.sum()
    if s < 1e-12:
        return np.ones(n) / n
    return contribs / s


def marginal_hypervolume_weights(
    F_nd: np.ndarray,
    ref_point: Optional[np.ndarray] = None,
    *,
    normalize_objectives: bool = True,
    aggregation: str = "linear",
    softmax_temperature: float = 1.0,
) -> np.ndarray:
    """Weights for Pareto-center averaging from marginal hypervolume contributions.

    Parameters
    ----------
    F_nd : ndarray, shape (n, m)
        Objectives for points on one Pareto front (nondominated among themselves).
    ref_point : ndarray, shape (m,), optional
        In raw objective space (before normalization). Dominated by all points
        (worse in every objective for minimization). Ignored for the default
        normalized reference when ``normalize_objectives`` is True and this is
        omitted.
    normalize_objectives : bool
        If True (default), min-max normalize each objective on ``F_nd`` before HV
        so Pareto-center weights are not skewed by objective scale.
    aggregation : {'linear', 'softmax'}
        How marginal contributions become weights (see
        :func:`pareto_center_weights_from_marginal_hv`).
    softmax_temperature : float
        Used when ``aggregation`` is ``softmax``; larger values yield softer weights.
    """
    F_nd = np.atleast_2d(F_nd)
    n, m = F_nd.shape
    if normalize_objectives:
        F_nd, ref_point = normalize_objectives_for_hv(F_nd, ref_point)
    elif ref_point is None:
        ref_point = default_hv_ref_point(F_nd)
    else:
        ref_point = np.asarray(ref_point, dtype=np.float64).reshape(m)

    hv = Hypervolume(ref_point=ref_point)
    if n == 0:
        return np.zeros(0)
    if n == 1:
        return np.ones(1)

    total = hv.do(F_nd)
    contribs = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        sub = hv.do(F_nd[mask])
        contribs[i] = max(0.0, total - sub)
    return pareto_center_weights_from_marginal_hv(
        contribs,
        aggregation=aggregation,
        softmax_temperature=softmax_temperature,
    )


def pareto_center_from_weights(X_nd: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted average of rows of ``X_nd``."""
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    X_nd = np.atleast_2d(X_nd)
    return np.sum(X_nd * w[:, None], axis=0)


def pareto_center_solution(
    F: np.ndarray,
    X: np.ndarray,
    ref_point: Optional[np.ndarray] = None,
    *,
    hv_center_aggregation: str = "linear",
    hv_softmax_temperature: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Utility: Pareto front + HV weights + center ``pareto_center_X`` (no model side effects)."""
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
    return pareto_center_X, w
