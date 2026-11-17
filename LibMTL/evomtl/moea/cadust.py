"""Curvature-aware DUST (CADUST): adaptive Gaussian probes in activation space.

For each layer, concatenate the flattened outputs of all its calls for one sample. Use the same covariance for
every sample, with independent draws: ``Sigma = alpha I + U diag(lambda) U.T``. The initial covariance is I.
After a batch, update a low-rank, exponentially discounted second moment of the per-sample activation gradients
estimated by DUST. This is an empirical-Fisher-style curvature proxy, built without backprop or extra forwards.

The top eigenvectors of the proxy guide sampling in the next batch. Split a fraction of the total variance among
those directions in proportion to their eigenvalues, keeping ``trace(Sigma) = D`` so ``sigma`` still sets the RMS
jitter per activation. In the small-jitter limit, the raw reward-weighted error estimates ``Sigma @ gradient``.
Apply ``Sigma^-1`` before the local weight update so adapting the probes does not also precondition that update.
As in DUST, chunk centering scales the leading term by ``1 - 1/chunk`` for chunks larger than one (also removed
from the samples used to update the proxy). The covariance stays fixed over all draws for the layer in each batch.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Any, Dict, Optional, Tuple

import torch

from .dust import DUST, run_dust

DEFAULT_RANK = 8
DEFAULT_DECAY = 0.9
DEFAULT_SUBSPACE_FRACTION = 0.5


class _Curvature:
    """Store a thin factor of the gradient second moment and derive the covariance spectrum from it."""

    def __init__(self, width: int, rank: int, decay: float, fraction: float):
        self.width, self.rank = width, min(width, rank)
        self.decay, self.fraction = decay, fraction
        self.basis: Optional[torch.Tensor] = None  # [D, N], orthonormal columns
        self.spectrum: Optional[torch.Tensor] = None  # eigenvalues of the curvature proxy
        self._total: Optional[torch.Tensor] = None

    def sample(self, rows: int, gen: torch.Generator) -> torch.Tensor:
        z = torch.randn(rows, self.width, generator=gen, device=gen.device)
        if self.basis is None:
            return z
        u = torch.randn(rows, self.basis.shape[1], generator=gen, device=gen.device)
        return math.sqrt(1 - self.fraction) * z + (u * self._variances().sqrt()) @ self.basis.T

    def inverse(self, errors: torch.Tensor) -> torch.Tensor:
        if self.basis is None:
            return errors
        alpha = 1 - self.fraction
        variances = self._variances()
        correction = variances / (alpha + variances)
        return (errors - ((errors @ self.basis) * correction) @ self.basis.T) / alpha

    def _variances(self) -> torch.Tensor:
        return (self.spectrum / self._total) * (self.fraction * self.width)

    def update(self, errors: torch.Tensor) -> None:
        # The proxy update is F @ F.T, with F formed from the old thin factor and new gradient samples.
        # Fold the samples in blocks and truncate after each block to bound the factor width.
        if self.rank == 0 or self.fraction == 0:
            return
        old = None if self.basis is None else self.basis * (self.decay * self.spectrum).sqrt()
        scale = math.sqrt((1 - self.decay) / len(errors))
        if not bool(torch.isfinite(errors).all()):
            return
        for block in errors.split(max(self.rank, 32)):
            new = block.T * scale
            factor = new if old is None else torch.cat((old, new), dim=1)
            basis, singular, _ = torch.linalg.svd(factor, full_matrices=False)
            spectrum = singular[:self.rank].square()
            basis = basis[:, :self.rank].contiguous()
            old = basis * singular[:self.rank]
        total = spectrum.sum()
        if bool(torch.isfinite(total) & (total > 0)):
            self.basis, self.spectrum, self._total = basis, spectrum, total
        else:
            self.basis = self.spectrum = self._total = None


class CADUST(DUST):
    """DUST on the summed task loss with a separate learned covariance for each layer."""

    name = "CADUST"

    def __init__(self, *args: Any, rank: int = DEFAULT_RANK, decay: float = DEFAULT_DECAY,
                 subspace_fraction: float = DEFAULT_SUBSPACE_FRACTION, **kwargs: Any):
        if not math.isfinite(rank) or rank < 0 or int(rank) != rank:
            raise ValueError(f"CADUST rank must be a nonnegative integer, got {rank}.")
        if not 0 <= decay < 1:
            raise ValueError(f"CADUST decay must be in [0, 1), got {decay}.")
        if not 0 <= subspace_fraction < 1:
            raise ValueError(f"CADUST subspace_fraction must be in [0, 1), got {subspace_fraction}.")
        self.rank, self.decay, self.subspace_fraction = int(rank), float(decay), float(subspace_fraction)
        self.curvature: Dict[str, Tuple[tuple, _Curvature]] = {}
        super().__init__(*args, **kwargs)

    def _layer_curvature(self, name: str, widths: tuple) -> _Curvature:
        # A change in the number or shape of calls changes the coordinates of the layer.
        layout = tuple(s[1:] for s in self.shapes[name])
        if name not in self.curvature or self.curvature[name][0] != layout:
            self.curvature[name] = (layout, _Curvature(sum(widths), self.rank, self.decay, self.subspace_fraction))
        return self.curvature[name][1]

    def _sample_noise(self, name: str, samples: int) -> list:
        if not self.rank or not self.subspace_fraction:
            return super()._sample_noise(name, samples)
        shapes = self.shapes[name]
        widths = tuple(math.prod(s[1:]) for s in shapes)
        state = self._layer_curvature(name, widths)
        if state.basis is None:
            return super()._sample_noise(name, samples)
        noise = state.sample(samples, self.gen).split(widths, dim=1)
        return [a.reshape(samples, *s[1:]) for a, s in zip(noise, shapes)]

    def _finish_layer(self, name: str, errors: list) -> list:
        if not self.rank or not self.subspace_fraction:
            return errors
        objectives, batch = errors[0].shape[:2]
        flat = [e.reshape(objectives, batch, -1) for e in errors]
        widths = tuple(e.shape[-1] for e in flat)
        state = self._layer_curvature(name, widths)
        corrected = state.inverse(torch.cat(flat, dim=-1))
        samples = corrected.sum(dim=0)
        if self.chunk > 1 and not self.antithetic:
            samples = samples * (self.chunk / (self.chunk - 1))
        state.update(samples)
        return [e.reshape_as(original) for e, original in zip(corrected.split(widths, dim=-1), errors)]


def run_cadust(trainer: Any, train_dataloaders: Any, *, rank: int = DEFAULT_RANK, decay: float = DEFAULT_DECAY,
               subspace_fraction: float = DEFAULT_SUBSPACE_FRACTION, **kwargs: Any) -> Dict[str, Any]:
    """Run CADUST with the DUST training loop and an adaptive activation sampler."""
    estimator = partial(CADUST, rank=rank, decay=decay, subspace_fraction=subspace_fraction)
    return run_dust(trainer, train_dataloaders, method="CADUST", _estimator=estimator, **kwargs)
