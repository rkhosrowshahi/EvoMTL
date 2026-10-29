"""Synthetic multi-objective problems of the toy benchmark.

* ``nashmtl``: the 2-D two-task problem of Nash-MTL (``problem.py``, Navon et al. 2022; CAGrad's toy), with
  ``scale`` on the first loss.
* ``ewq``, ``cqf``, ``cqf2``, ``hqf``, ``mn2``, ``mn20``: the quadratic problems of the Jacobian-descent trajectory
  plots of UPGrad (Quinton & Rey 2024; https://github.com/SimplexLab/trajectories, ``src/trajectories/constants.py``),
  with their initial points. Every objective is a quadratic form ``f_k(x) = (x - u_k)^T A_k (x - u_k)`` (``A_k`` PSD),
  so the Pareto set is known exactly: for weights ``w > 0``, the minimizer of ``sum_k w_k f_k`` is
  ``x(w) = (sum_k w_k A_k)^{-1} sum_k w_k A_k u_k``.

  - ``ewq`` (element-wise quadratic): ``f = (x_1^2, x_2^2)``; orthogonal gradients, Pareto set the origin.
  - ``cqf`` (convex quadratic form, Eq. 14 of https://arxiv.org/abs/2406.16232): two ill-conditioned bowls
    (condition numbers 100 and 300) rotated by +-pi/16, centred at ``(+-1, 0)``; curved Pareto set.
  - ``cqf2``: the same with explicit matrices (condition numbers 108 and 41).
  - ``hqf`` (homogeneous quadratic form): one matrix, scaled by 1 and 10, centred at ``(1, 0)`` and ``(-10, 0)``;
    the Pareto set is the segment between the centres.
  - ``mn2`` / ``mn20`` (multinorm): ``f_k(x) = a_k ||x - a_k e_k||^2``, ``a = (1, 10)`` / ``(1, ..., 20)``: round
    bowls of very different sizes; the Pareto set is ``{x >= 0, sum_k x_k / a_k = 1}`` (a simplex with vertices
    ``a_k e_k``), with 20 objectives for ``mn20``.

Every problem gives the losses (``__call__``, ``x`` of shape ``[..., D]`` -> ``[..., K]``), its default initial
points, a sampled Pareto set, and the distances of points to the Pareto set (parameter space) and of losses to the
Pareto front (loss space, two objectives).
"""

from __future__ import annotations

from math import cos, pi, sin, sqrt
from typing import List

import numpy as np
import torch

from problem import Toy

NASHMTL_INITS = [[-8.5, 7.5], [0.0, 0.0], [9.0, 9.0], [-7.5, -0.5], [9.0, -1.0]]  # experiments/toy/trainer.py
NASHMTL_PARETO_X2 = -8.3552  # Pareto set: x2 = NASHMTL_PARETO_X2, x1 in [-7, 7] (experiments/toy/utils.py)


class Problem:
    """Interface: ``name``, ``n_params``, ``n_obj``, ``inits``, ``__call__``, ``pareto_set``."""

    name: str
    n_params: int
    n_obj: int
    inits: List[List[float]]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def pareto_set(self, n: int = 2000) -> np.ndarray:
        """``[n, D]`` points of the Pareto set, ordered along it (two objectives)."""
        raise NotImplementedError

    def losses(self, X: np.ndarray) -> np.ndarray:
        """Losses (float64) of the points ``X`` (``[..., D]``)."""
        with torch.no_grad():
            return self(torch.as_tensor(X, dtype=torch.float64)).numpy()

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
    flat = np.asarray(P, dtype=np.float64).reshape(-1, S.shape[1])
    out = np.empty(len(flat))
    for s in range(0, len(flat), chunk):
        out[s:s + chunk] = np.linalg.norm(flat[s:s + chunk, None] - S[None], axis=-1).min(axis=1)
    return out.reshape(np.shape(P)[:-1])


def _polyline_distance(P: np.ndarray, S: np.ndarray, chunk: int = 512) -> np.ndarray:
    """Distance of the points ``P`` (``[..., d]``) to the polyline through the ordered samples ``S`` (``[n, d]``),
    as ``compute_2d_pf_distance`` of SimplexLab/trajectories (without its clamp below the front); exact between the
    samples, unlike :func:`_min_distance`, whose error is up to half the sample spacing."""
    flat = np.asarray(P, dtype=np.float64).reshape(-1, S.shape[1])
    A, AB = S[:-1], S[1:] - S[:-1]
    len2 = (AB * AB).sum(axis=1)
    out = np.empty(len(flat))
    for s in range(0, len(flat), chunk):
        AP = flat[s:s + chunk, None] - A[None]  # [m, n - 1, d]
        t = np.clip(np.where(len2 > 0, (AP * AB).sum(-1) / np.where(len2 > 0, len2, 1.0), 0.0), 0.0, 1.0)
        out[s:s + chunk] = np.linalg.norm(AP - t[..., None] * AB, axis=-1).min(axis=1)
    return out.reshape(np.shape(P)[:-1])


class NashMTLToy(Problem):
    """The Nash-MTL toy (``problem.Toy``), its five initial points and its Pareto set (``x_2 = -8.3552``)."""

    name, n_params, n_obj = "nashmtl", 2, 2

    def __init__(self, scale: float = 0.1):
        self.toy = Toy(scale=scale)
        self.inits = NASHMTL_INITS

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            return self.toy(x)
        return self.toy.batch_forward(x.reshape(-1, 2)).reshape(*x.shape[:-1], 2)

    def pareto_set(self, n: int = 2000) -> np.ndarray:
        return np.stack([np.linspace(-7.0, 7.0, n), np.full(n, NASHMTL_PARETO_X2)], axis=1)

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

    def __init__(self, name: str, A, u, inits: List[List[float]]):
        self.name = name
        self.A = torch.as_tensor(np.asarray(A, dtype=np.float64))
        self.u = torch.as_tensor(np.asarray(u, dtype=np.float64))
        self.n_obj, self.n_params = self.u.shape
        self.inits = inits

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
            raise ValueError(f"{self.name}: sampled Pareto set for two objectives only")
        w = np.linspace(1e-5, 1.0 - 1e-5, n)  # from the minimum of f_2 to that of f_1
        return np.stack([self.pareto_point(np.array([a, 1.0 - a])) for a in w])

    def set_distance(self, X: np.ndarray) -> np.ndarray:
        return _polyline_distance(X, self.pareto_set())

    def front_distance(self, L: np.ndarray) -> np.ndarray:
        return _polyline_distance(L, self.pareto_front())


class Multinorm(QuadraticForms):
    """``f_k(x) = a_k ||x - a_k e_k||^2``; Pareto set ``{x >= 0, sum_k x_k / a_k = 1}`` (projection, any ``K``)."""

    def __init__(self, name: str, a, inits: List[List[float]]):
        a = np.asarray(a, dtype=np.float64)
        super().__init__(name, a[:, None, None] * np.eye(len(a)), np.diag(a), inits)
        self.a = a

    def set_distance(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return np.linalg.norm(X - _project_weighted_simplex(X, 1.0 / self.a), axis=-1)

    def front_distance(self, L: np.ndarray) -> np.ndarray:
        if self.n_obj != 2:  # no sampled front in 20-D: use set_distance
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


def _rotation(a: float) -> np.ndarray:
    return np.array([[cos(a), -sin(a)], [sin(a), cos(a)]])


def make_problem(name: str = "nashmtl", **kwargs) -> Problem:
    """Problem ``name`` (``nashmtl`` takes ``scale``); the settings of SimplexLab/trajectories for the others."""
    name = name.lower()
    if name == "nashmtl":
        return NashMTLToy(**kwargs)
    if kwargs:
        raise ValueError(f"{name}: unexpected settings {sorted(kwargs)}")
    if name == "ewq":
        return QuadraticForms("ewq", [np.diag([1.0, 0.0]), np.diag([0.0, 1.0])], np.zeros((2, 2)),
                              [[3.0, -2.0], [0.0, -3.0], [-4.0, 4.0], [-3.0, 4.0], [-3.5, -0.75]])
    if name == "cqf":
        B1 = _rotation(pi / 16) @ np.diag([1.0, 0.1])
        B2 = _rotation(-pi / 16) @ np.diag([sqrt(3.0), 0.1])
        return QuadraticForms("cqf", [B1 @ B1.T, B2 @ B2.T], [[1.0, 0.0], [-1.0, 0.0]],
                              [[0.5, 0.5], [-1.0, 7.0], [0.0, 0.0], [1.0, 6.0]])
    if name == "cqf2":
        return QuadraticForms("cqf2", [[[1.0, 0.2], [0.2, 0.05]], [[3.0, -0.6], [-0.6, 0.2]]],
                              [[1.0, 0.0], [-1.0, 0.0]], [[0.5, 0.5], [-0.3, 7.0], [0.0, 0.0]])
    if name == "hqf":
        A = np.array([[2.0, -1.0], [-1.0, 2.0]])
        return QuadraticForms("hqf", [A, 10.0 * A], [[1.0, 0.0], [-10.0, 0.0]],
                              [[-6.0, 4.0], [-3.0, -1.5], [1.5, 2.0], [2.5, 5.5]])
    if name == "mn2":
        return Multinorm("mn2", [1.0, 10.0], [[0.0, 0.0], [-5.0, 5.0], [10.0, 5.0], [10.0, 0.0], [20.0, 0.0]])
    if name == "mn20":
        return Multinorm("mn20", np.arange(1.0, 21.0), [[0.0] * 20])
    raise ValueError(f"Unknown problem {name!r}; options: {PROBLEMS}")


PROBLEMS = ("nashmtl", "ewq", "cqf", "cqf2", "hqf", "mn2", "mn20")
