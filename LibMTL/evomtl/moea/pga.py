"""Principal-gradient aggregators SVD-PGA and OS-NMF-PGA, ported from TorchJD (``torchjd.aggregation._pga``).

Both are Gramian weightings: from the Gramian ``M = J J^T`` of the Jacobian rows ``J`` (``[K, D]``) they compute task
weights ``w`` and the direction ``d = J^T w``.

* SVD-PGA averages the principal directions of the gradients (the left singular vectors of ``J^T``), each one unit
  norm and oriented towards the side where most of the gradients lie (most of its squared mass in the gradients'
  coordinates is positive). ``rank`` keeps at most that many directions; directions whose eigenvalue is below
  ``eigenvalue_tol`` times the largest are dropped.
* OS-NMF-PGA approximates the gradients by ``W H`` with orthonormal columns in ``W`` (in the span of the gradients,
  ``W = J^T B``) and nonnegative entries in ``H`` (orthogonal semi-nonnegative matrix factorization), starting from
  the SVD-PGA basis and alternating ``H = max(W^T J^T, 0)`` with the orthonormal ``W`` closest to ``J^T H^T``
  (Procrustes), and averages the columns of ``W``.

``scale``: ``None`` returns the average of the unit-norm directions whatever the size of the gradients (the step
length is set by the learning rate alone); ``"mean_norm"`` multiplies it by the mean norm of the gradients.
Everything is computed on the Gramian divided by its largest diagonal element (scale-free, avoids overflow).
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch

SCALES = (None, "mean_norm")


def _sorted_spectrum(gramian: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Eigenvalues (clamped to be nonnegative) and eigenvectors, by decreasing eigenvalue."""
    eigenvalues, eigenvectors = torch.linalg.eigh(gramian)
    eigenvalues = eigenvalues.clamp_min(0)
    order = eigenvalues.argsort(descending=True)
    return eigenvalues[order], eigenvectors[:, order]


def _normalize(gramian: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """Symmetric Gramian divided by its largest diagonal element, and that element."""
    symmetric = 0.5 * (gramian + gramian.T)
    scale = symmetric.diagonal().max().item()
    if scale <= 0:
        return symmetric, 0.0
    return symmetric / scale, scale


def _select_rank(available_rank: int, rank: Optional[int]) -> int:
    return available_rank if rank is None else min(available_rank, rank)


def _check(rank: Optional[int], eigenvalue_tol: float, scale: Optional[str]) -> None:
    if rank is not None and rank < 1:
        raise ValueError(f"Parameter `rank` should be a positive integer or None. Found {rank=}.")
    if eigenvalue_tol < 0:
        raise ValueError(f"Parameter `eigenvalue_tol` should be nonnegative. Found {eigenvalue_tol=}.")
    if scale not in SCALES:
        raise ValueError(f"Parameter `scale` should be None or 'mean_norm'. Found {scale=}.")


def _step_length(gramian: torch.Tensor, scale: Optional[str]) -> Union[torch.Tensor, float]:
    """1 if ``scale`` is ``None``; the mean norm of the gradients if it is ``"mean_norm"``."""
    if scale is None:
        return 1.0
    return gramian.diagonal().clamp_min(0).sqrt().mean()


def _orient_by_mass(vectors: torch.Tensor, dim: int) -> torch.Tensor:
    """Flips the sign of each vector (along ``dim``) so that most of its squared mass is positive."""
    positive_mass = vectors.clamp_min(0).square().sum(dim=dim, keepdim=True)
    negative_mass = vectors.clamp_max(0).square().sum(dim=dim, keepdim=True)
    return vectors * torch.where(negative_mass > positive_mass, -1.0, 1.0)


def _initial_basis(gramian: torch.Tensor, rank: Optional[int],
                   eigenvalue_tol: float) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Weights of the oriented, unit-norm principal directions (the basis is ``J^T @ basis_weights``) and the
    coordinates of the gradients in the principal basis; ``None`` if no direction is kept."""
    eigenvalues, eigenvectors = _sorted_spectrum(gramian)
    positive = eigenvalues > eigenvalue_tol * eigenvalues[0]
    n_positive = int(positive.sum().item())
    t = _select_rank(n_positive, rank)
    if t == 0:
        return None
    roots = eigenvalues[:n_positive].sqrt()
    vectors = eigenvectors[:, :n_positive]
    basis_weights = _orient_by_mass(vectors[:, :t], dim=0) / roots[:t]
    coordinates = roots.unsqueeze(1) * vectors.T
    return basis_weights, coordinates


def svd_pga_weights(gramian: torch.Tensor, rank: Optional[int] = None, eigenvalue_tol: float = 1e-10,
                    scale: Optional[str] = None) -> torch.Tensor:
    """SVD-PGA weights ``w`` (``[K]``) of the Gramian ``J J^T`` (see module docstring)."""
    _check(rank, eigenvalue_tol, scale)
    gram, s = _normalize(gramian)
    basis = _initial_basis(gram, rank, eigenvalue_tol) if s > 0 else None
    if basis is None:
        return torch.zeros(gramian.shape[0], dtype=gramian.dtype, device=gramian.device)
    basis_weights, _ = basis
    return basis_weights.mean(dim=1) / s ** 0.5 * _step_length(gramian, scale)


def _gram_procrustes_step(gram: torch.Tensor, factor_h: torch.Tensor, coordinates: torch.Tensor,
                          eigenvalue_tol: float) -> torch.Tensor:
    """Weights B of the orthonormal basis ``W = J^T B`` closest to ``J^T H^T``, from the Gramian; completed with a
    reduced SVD of the gradients' coordinates when that small Gramian is ill-conditioned."""
    small_gram = factor_h @ gram @ factor_h.T
    small_gram = 0.5 * (small_gram + small_gram.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(small_gram)
    if eigenvalues[0] <= eigenvalue_tol * eigenvalues[-1]:
        u, _, v_t = torch.linalg.svd(coordinates @ factor_h.T, full_matrices=False)
        return (coordinates / coordinates.square().sum(dim=1, keepdim=True)).T @ (u @ v_t)
    return (factor_h.T @ (eigenvectors / eigenvalues.sqrt())) @ eigenvectors.T


def _gram_loss(gram: torch.Tensor, basis_weights: torch.Tensor, factor_h: torch.Tensor) -> float:
    """``0.5 ||J^T - W H||^2`` for the basis ``W = J^T B``, from the Gramian."""
    gram_b = gram @ basis_weights
    cross = (factor_h * gram_b.T).sum()
    quadratic = (factor_h * ((basis_weights.T @ gram_b) @ factor_h)).sum()
    return (0.5 * (gram.trace() - 2 * cross + quadratic)).item()


def os_nmf_pga_weights(gramian: torch.Tensor, rank: Optional[int] = None, max_iters: int = 30, tol: float = 1e-9,
                       roundoff_scale: float = 16.0, eigenvalue_tol: float = 1e-10,
                       scale: Optional[str] = None) -> torch.Tensor:
    """OS-NMF-PGA weights ``w`` (``[K]``) of the Gramian ``J J^T`` (see module docstring): at most ``max_iters``
    alternating updates, stopped when the reconstruction loss changes by less than ``tol`` times itself plus
    ``roundoff_scale`` machine epsilons times the squared norm of the gradients."""
    _check(rank, eigenvalue_tol, scale)
    if max_iters < 1:
        raise ValueError(f"Parameter `max_iters` should be a positive integer. Found {max_iters=}.")
    if tol <= 0:
        raise ValueError(f"Parameter `tol` should be positive. Found {tol=}.")
    if roundoff_scale < 0:
        raise ValueError(f"Parameter `roundoff_scale` should be nonnegative. Found {roundoff_scale=}.")
    gram, s = _normalize(gramian)
    basis = _initial_basis(gram, rank, eigenvalue_tol) if s > 0 else None
    if basis is None:
        return torch.zeros(gram.shape[0], dtype=gram.dtype, device=gram.device)
    basis_weights, coordinates = basis
    energy = gram.trace().item()
    roundoff = roundoff_scale * torch.finfo(gram.dtype).eps * energy
    factor_h = (basis_weights.T @ gram).clamp_min(0)
    previous = _gram_loss(gram, basis_weights, factor_h)
    for _ in range(max_iters):
        basis_weights = _gram_procrustes_step(gram, factor_h, coordinates, eigenvalue_tol)
        loss = _gram_loss(gram, basis_weights, factor_h)
        factor_h = (basis_weights.T @ gram).clamp_min(0)
        if abs(previous - loss) <= tol * abs(previous) + roundoff:
            break
        previous = loss
    return basis_weights.mean(dim=1) / s ** 0.5 * _step_length(gramian, scale)


def svd_pga_aggregate(J: torch.Tensor, rank: Optional[int] = None, eigenvalue_tol: float = 1e-10,
                      scale: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """SVD-PGA on Jacobian rows ``J`` (``[K, D]``); return ``(d, w)`` with ``d = J^T w``."""
    w = svd_pga_weights(J @ J.T, rank, eigenvalue_tol, scale)
    return w @ J, w


def os_nmf_pga_aggregate(J: torch.Tensor, rank: Optional[int] = None, max_iters: int = 30, tol: float = 1e-9,
                         roundoff_scale: float = 16.0, eigenvalue_tol: float = 1e-10,
                         scale: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """OS-NMF-PGA on Jacobian rows ``J`` (``[K, D]``); return ``(d, w)`` with ``d = J^T w``."""
    w = os_nmf_pga_weights(J @ J.T, rank, max_iters, tol, roundoff_scale, eigenvalue_tol, scale)
    return w @ J, w
