"""Synthetic multi-objective problems of the toy benchmark: one class per kind of problem, as in
``tests/trajectories/_objectives.py`` of SimplexLab/TorchJD. The problems themselves (their settings) and their
initial points are in :mod:`constants`.

* :class:`ValleyBowl`: the Valley-Bowl problem, 2-D with two tasks (introduced with CAGrad, Liu et al. 2021, and
  reused by Nash-MTL, Navon et al. 2022, whose ``experiments/toy/problem.py`` it reproduces): a plateau cut by narrow
  log-valleys (x_2 > 0) above two conflicting quadratic bowls (x_2 < 0), with ``scale`` on the first loss.
* :class:`QuadraticForms`: ``f_k(x) = (x - u_k)^T A_k (x - u_k)`` (``A_k`` PSD), the problems of the Jacobian-descent
  trajectory plots of UPGrad (Quinton & Rey 2024; https://github.com/SimplexLab/trajectories). The Pareto set is known
  exactly: for weights ``w > 0``, the minimizer of ``sum_k w_k f_k`` is
  ``x(w) = (sum_k w_k A_k)^{-1} sum_k w_k A_k u_k``. Special cases:

  - :class:`ElementwiseQuadratic`: ``f_k(x) = x_k^2``; orthogonal gradients, Pareto set the origin.
  - :class:`HomogeneousQuadraticForm`: ``A_k = s_k A`` (one matrix, several scales).
  - :class:`Multinorm`: ``f_k(x) = a_k ||x - a_k e_k||^2``, round bowls of different sizes; the Pareto set is
    ``{x >= 0, sum_k x_k / a_k = 1}`` (a simplex with vertices ``a_k e_k``), for any number of objectives.

Every problem gives the losses (``__call__``, ``x`` of shape ``[..., D]`` -> ``[..., K]``), a sampled Pareto set, and
the distances of points to the Pareto set (parameter space) and of losses to the Pareto front (loss space, two
objectives).
"""

from __future__ import annotations

from math import cos, sin

import numpy as np
import torch


class Problem:
    """Interface: ``n_params``, ``n_obj``, ``__call__``, ``pareto_set``, ``global_optimum``."""

    n_params: int
    n_obj: int

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def pareto_set(self, n: int = 2000) -> np.ndarray:
        """``[n, D]`` points of the Pareto set, ordered along it (two objectives)."""
        raise NotImplementedError

    def losses(self, X: np.ndarray) -> np.ndarray:
        """Losses (float32) of the points ``X`` (``[..., D]``)."""
        with torch.no_grad():
            return self(torch.as_tensor(X, dtype=torch.float32)).numpy()

    def pareto_front(self, n: int = 2000) -> np.ndarray:
        return self.losses(self.pareto_set(n))

    def global_optimum(self) -> np.ndarray:
        """``[D]`` minimizer of the average loss (Pareto optimal: the scalarization with equal weights)."""
        raise NotImplementedError

    def set_distance(self, X: np.ndarray) -> np.ndarray:
        """Euclidean distance of the points ``X`` (``[..., D]``) to the Pareto set (sampled)."""
        return _min_distance(X, self.pareto_set())

    def front_distance(self, L: np.ndarray) -> np.ndarray:
        """Euclidean distance of the losses ``L`` (``[..., K]``) to the Pareto front (sampled)."""
        return _min_distance(L, self.pareto_front())


def _min_distance(P: np.ndarray, S: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """Distance of the points ``P`` (``[..., d]``) to the nearest of the samples ``S`` (``[n, d]``)."""
    flat = np.asarray(P, dtype=np.float32).reshape(-1, S.shape[1])
    out = np.empty(len(flat))
    for s in range(0, len(flat), chunk):
        out[s:s + chunk] = np.linalg.norm(flat[s:s + chunk, None] - S[None], axis=-1).min(axis=1)
    return out.reshape(np.shape(P)[:-1])


def _polyline_distance(P: np.ndarray, S: np.ndarray, chunk: int = 512) -> np.ndarray:
    """Distance of the points ``P`` (``[..., d]``) to the polyline through the ordered samples ``S`` (``[n, d]``),
    as ``compute_2d_pf_distance`` of SimplexLab/trajectories (without its clamp below the front); exact between the
    samples, unlike :func:`_min_distance`, whose error is up to half the sample spacing."""
    flat = np.asarray(P, dtype=np.float32).reshape(-1, S.shape[1])
    A, AB = S[:-1], S[1:] - S[:-1]
    len2 = (AB * AB).sum(axis=1)
    out = np.empty(len(flat))
    for s in range(0, len(flat), chunk):
        AP = flat[s:s + chunk, None] - A[None]  # [m, n - 1, d]
        t = np.clip(np.where(len2 > 0, (AP * AB).sum(-1) / np.where(len2 > 0, len2, 1.0), 0.0), 0.0, 1.0)
        out[s:s + chunk] = np.linalg.norm(AP - t[..., None] * AB, axis=-1).min(axis=1)
    return out.reshape(np.shape(P)[:-1])


class ValleyBowl(Problem):
    """The Valley-Bowl problem (``Toy`` of Nash-MTL's ``experiments/toy/problem.py``) and its Pareto set
    (``x_2 = -8.3552``, ``x_1`` in ``[-7, 7]``). For ``x_2 > 0`` each loss is a log-valley, ``log|.| + 6``; for
    ``x_2 < 0`` a quadratic bowl centred at ``(+-7, -8)``; ``tanh(+-x_2 / 2)`` blends the two. ``scale`` multiplies
    the first loss."""

    n_params, n_obj = 2, 2
    LOWER = 0.000005  # floor of the log-valleys' argument
    PARETO_SET_X2 = -8.3552  # Pareto set: x_2 = PARETO_SET_X2 (Nash-MTL's experiments/toy/utils.py)

    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., 0], x[..., 1]

        f1_log = torch.clamp((0.5 * (-x1 - 7) - torch.tanh(-x2)).abs(), self.LOWER).log() + 6
        f2_log = torch.clamp((0.5 * (-x1 + 3) + torch.tanh(-x2) + 2).abs(), self.LOWER).log() + 6
        c_log = torch.clamp(torch.tanh(x2 * 0.5), 0)

        f1_sq = ((-x1 + 7).pow(2) + 0.1 * (-x2 - 8).pow(2)) / 10 - 20
        f2_sq = ((-x1 - 7).pow(2) + 0.1 * (-x2 - 8).pow(2)) / 10 - 20
        c_sq = torch.clamp(torch.tanh(-x2 * 0.5), 0)

        f1 = (f1_log * c_log + f1_sq * c_sq) * self.scale
        f2 = f2_log * c_log + f2_sq * c_sq
        return torch.stack([f1, f2], dim=-1)

    def pareto_set(self, n: int = 2000) -> np.ndarray:
        return np.stack([np.linspace(-7.0, 7.0, n), np.full(n, self.PARETO_SET_X2)], axis=1)

    def global_optimum(self) -> np.ndarray:
        """Grid search over ``[-10, 10]^2``, refined by Nelder-Mead (``(0, -8.355)`` for scale 1)."""
        from scipy.optimize import minimize

        g = np.linspace(-10.0, 10.0, 401)
        X = np.stack(np.meshgrid(g, g), axis=-1).reshape(-1, 2)
        x0 = X[self.losses(X).mean(axis=1).argmin()]
        return minimize(lambda x: self.losses(x[None]).mean(), x0, method="Nelder-Mead",
                        options={"xatol": 1e-8, "fatol": 1e-12}).x


class QuadraticForms(Problem):
    """``f_k(x) = (x - u_k)^T A_k (x - u_k)`` (``A``: ``[K, D, D]``, ``u``: ``[K, D]``)."""

    def __init__(self, A, u):
        self.A = torch.as_tensor(np.asarray(A, dtype=np.float32))
        self.u = torch.as_tensor(np.asarray(u, dtype=np.float32))
        if self.A.shape != (*self.u.shape, self.u.shape[1]):
            raise ValueError(f"A {tuple(self.A.shape)} and u {tuple(self.u.shape)}: expected [K, D, D] and [K, D]")
        self.n_obj, self.n_params = self.u.shape

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        d = x.unsqueeze(-2) - self.u.to(x.dtype)  # [..., K, D]
        return torch.einsum("...kd,kde,...ke->...k", d, self.A.to(x.dtype), d)

    def pareto_point(self, w: np.ndarray) -> np.ndarray:
        """Minimizer of ``sum_k w_k f_k`` for weights ``w > 0`` (the strongly Pareto stationary point of ``w``)."""
        A, u = self.A.numpy(), self.u.numpy()
        G = np.einsum("k,kde->de", w, A)
        b = np.einsum("k,kde,ke->d", w, A, u)
        return np.linalg.lstsq(G, b, rcond=None)[0]

    def global_optimum(self) -> np.ndarray:
        return self.pareto_point(np.full(self.n_obj, 1.0 / self.n_obj))

    def pareto_set(self, n: int = 2000) -> np.ndarray:
        if self.n_obj != 2:
            raise ValueError(f"{type(self).__name__}: sampled Pareto set for two objectives only")
        w = np.linspace(1e-5, 1.0 - 1e-5, n)  # from the minimum of f_2 to that of f_1
        return np.stack([self.pareto_point(np.array([a, 1.0 - a])) for a in w])

    def set_distance(self, X: np.ndarray) -> np.ndarray:
        return _polyline_distance(X, self.pareto_set())

    def front_distance(self, L: np.ndarray) -> np.ndarray:
        return _polyline_distance(L, self.pareto_front())


def rotated_form(angle: float, s) -> np.ndarray:
    """``B B^T`` with ``B = R(angle) diag(s)``: the PSD matrix with eigenvalues ``s^2`` along the axes rotated by
    ``angle`` (2-D)."""
    B = np.array([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]]) @ np.diag(s)
    return B @ B.T


class ElementwiseQuadratic(QuadraticForms):
    """``f_k(x) = x_k^2`` for ``k = 1..n_dim`` (``A_k = e_k e_k^T``, ``u_k = 0``); Pareto set the origin."""

    def __init__(self, n_dim: int):
        super().__init__(A=[np.diag(e) for e in np.eye(n_dim)], u=np.zeros((n_dim, n_dim)))


class HomogeneousQuadraticForm(QuadraticForms):
    """``A_k = s_k A``: one matrix ``A`` (``[D, D]``) scaled by ``scales`` (``[K]``), centred at ``u`` (``[K, D]``)."""

    def __init__(self, A, scales, u):
        base = np.asarray(A, dtype=np.float32)
        super().__init__(A=[s * base for s in np.asarray(scales, dtype=np.float32)], u=u)


class Multinorm(QuadraticForms):
    """``f_k(x) = a_k ||x - a_k e_k||^2``; Pareto set ``{x >= 0, sum_k x_k / a_k = 1}`` (projection, any ``K``)."""

    def __init__(self, a):
        a = np.asarray(a, dtype=np.float32)
        super().__init__(A=a[:, None, None] * np.eye(len(a)), u=np.diag(a))
        self.a = a

    def set_distance(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        return np.linalg.norm(X - _project_weighted_simplex(X, 1.0 / self.a), axis=-1)

    def front_distance(self, L: np.ndarray) -> np.ndarray:
        if self.n_obj != 2:  # no sampled front beyond two objectives: use set_distance
            return np.full(np.shape(L)[:-1], np.nan)
        return super().front_distance(L)


def _project_weighted_simplex(X: np.ndarray, c: np.ndarray, iters: int = 100) -> np.ndarray:
    """Euclidean projection of the rows of ``X`` onto ``{y >= 0, c . y = 1}`` (``c > 0``): ``y = max(x - lam c, 0)``
    with ``lam`` found by bisection."""
    flat = X.reshape(-1, X.shape[-1])
    lo = np.full(len(flat), -np.abs(flat).max() / c.min() - 1.0 / (c @ c))
    hi = np.full(len(flat), np.abs(flat).max() / c.min() + 1.0)
    for _ in range(iters):  # c . max(x - lam c, 0) decreases in lam
        mid = 0.5 * (lo + hi)
        s = (np.maximum(flat - mid[:, None] * c, 0.0) * c).sum(axis=1)
        lo, hi = np.where(s > 1.0, mid, lo), np.where(s > 1.0, hi, mid)
    return np.maximum(flat - 0.5 * (lo + hi)[:, None] * c, 0.0).reshape(X.shape)
