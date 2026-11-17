"""Multi-objective DUST with per-layer task weights from cross-chunk Gramians (MO-UPGrad-Split-DUST, MO-MGDA-Split-DUST).

MO-UPGrad-DUST and MO-MGDA-DUST (:mod:`.mo_upgrad_dust`, :mod:`.mo_mgda_dust`) choose the task weights from the Gramian
``J J^T`` of the per-task DUST estimates. Every estimate is the true gradient plus noise of about the same norm, so
that Gramian is biased: each diagonal entry carries the noise variance, and the off-diagonal entries are pulled
towards zero by the nearly orthogonal noise. The solvers then see neither the true norms nor the true conflicts.

MO-Split-DUST keeps DUST's estimates and fixes the Gramian, separately at every weight layer ``l``:

1. Each chunk ``c`` of the layer's draws (one forward pass) gives its own share ``T_c`` (``[K, D_l]``) of the per-task
   weight gradients; the shares sum to the layer's estimate ``J_l`` and their noise is independent across chunks.
   With ``num_batches > 1`` the chunks of all mini-batches are pooled.
2. The cross-chunk Gramian keeps only the products of different chunks, whose noise has zero mean:
   ``G_l = M / (M - 1) * (J_l J_l^T - sum_c T_c T_c^T)`` over the ``M`` chunks. It is an unbiased estimate of the
   Gramian of the expected estimate (its noise-free counterpart), at no extra forward pass.
3. An exponential moving average over iterations (``gram_decay``) smooths ``G_l``; the result is projected onto the
   positive semi-definite matrices (negative eigenvalues clipped to zero), as the solvers require.
4. The solver (``upgrad`` or ``mgda``) turns ``G_l`` into the weights ``w_l`` and the layer steps along ``w_l @ J_l``.
   UPGrad: the average of the projections of the rows onto their dual cone, so ``g_k . d >= 0`` for every task and
   the mean of the rows when they do not conflict. MGDA: the min-norm point of their convex hull.

Tasks whose estimate is zero at a layer (another task's head) are left out of its solve, and the weights of the others
are scaled by ``K' / K`` (``K'`` tasks present), so a layer that only one task reaches steps along ``g_k / K``, as in
the mean of the rows. The parameters outside the weight layers (normalization affines) step along the mean of their
weight-space ES rows. The logged task weights are the average over the layers that every task reaches.

With ``antithetic`` (MO-UPGrad-Split-ADUST, MO-MGDA-Split-ADUST) the draws are A-DUST's antithetic pairs
``y +- sigma a`` rewarded by half their loss difference (:mod:`.a_dust`). Without the chunk centering every share is an
unbiased estimate on its own (the centered one is shrunk by ``(n - 1) / n`` for a chunk of ``n`` draws, the same for
every task and layer), and the cross-chunk Gramian estimates the Gramian of the backprop gradients.

The diagnostic cosine with backprop applies the same per-layer rule to the backprop Jacobian, with its exact Gramian.

References:
- DUST: :mod:`~LibMTL.evomtl.moea.dust`
- UPGrad: https://arxiv.org/abs/2406.16232
- MGDA for MTL: https://arxiv.org/abs/1810.04650
- Double sampling for unbiased stochastic MGDA (SDMGrad): https://arxiv.org/abs/2305.18409
- Tracked gradients for stochastic multi-objective descent (MoCo): https://arxiv.org/abs/2210.12624
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from ...weighting.MGDA import MGDA
from ...weighting.UPGrad import _normalize, _project_weights, _regularize
from .dust import DUST, run_dust

SOLVERS = {"upgrad": "UPGrad", "mgda": "MGDA"}  # solver -> part of the method name
_LIBMTL_MGDA = MGDA()


class SplitDUST(DUST):
    """DUST with one estimate per task that also records, per weight layer, the self-products of its chunk shares
    (see the module docstring); :meth:`aggregate` turns them into the per-layer weights and the direction."""

    def __init__(self, *args: Any, solver: str, gram_decay: float, norm_eps: float, reg_eps: float, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.solver, self.gram_decay, self.norm_eps, self.reg_eps = solver, float(gram_decay), norm_eps, reg_eps
        self.self_products: Dict[str, torch.Tensor] = {}  # layer -> sum_c T_c T_c^T of the current step
        self.chunks: Dict[str, int] = {}                   # layer -> M of the current step
        self.gram: Dict[str, torch.Tensor] = {}            # layer -> smoothed cross-chunk Gramian
        self.scale = 1.0                                   # chunk share -> units of the step's J

    def _setup(self) -> None:
        super()._setup()
        few = [n for n, d in self.draws.items() if d // self.chunk < 2]
        if few:
            raise ValueError(f"{self.name} needs at least 2 chunks of draws per layer for the cross-chunk Gramian; "
                             f"too few at {', '.join(few)} (draws {self.draws}, chunk {self.chunk}).")
        offsets, start = {}, 0
        for p in self.model.parameters():
            offsets[id(p)] = (start, start + p.numel())
            start += p.numel()
        self.columns, owned = {}, set()
        for n, m in self.layers.items():
            params = list(m.parameters(recurse=False))
            if owned & {id(p) for p in params}:
                raise ValueError(f"{self.name}: layer {n} shares a weight with another layer; the per-layer "
                                 "Gramian needs every weight in one layer.")
            owned |= {id(p) for p in params}
            self.columns[n] = torch.cat([torch.arange(*offsets[id(p)]) for p in params])
        self.free_columns = torch.cat([torch.arange(*offsets[id(p)]) for p in self.free] or [torch.zeros(0).long()])

    def _estimate_layer(self, name: str, clean: torch.Tensor, x: torch.Tensor,
                        y: Dict[str, torch.Tensor], grads: Dict[torch.nn.Parameter, torch.Tensor]) -> None:
        m = self.layers[name]
        params = list(m.parameters(recurse=False))
        products = 0
        for errors in self._chunk_errors(name, clean, x, y):
            share: Dict[torch.nn.Parameter, torch.Tensor] = {}
            self._reassemble(m, errors, self.inputs[name], share)
            for p in params:
                grads[p] = grads.get(p, 0) + share[p]
            T = torch.cat([share[p].reshape(len(clean), -1) for p in params], 1).double() * self.scale / clean.shape[1]
            products = products + T @ T.T
            self.chunks[name] = self.chunks.get(name, 0) + 1
        self.self_products[name] = self.self_products.get(name, 0) + products.cpu()

    @torch.no_grad()
    def step(self, batches: list, *args: Any, **kwargs: Any):
        self.self_products, self.chunks, self.scale = {}, {}, 1.0 / max(len(batches), 1)
        return super().step(batches, *args, **kwargs)

    def _solve(self, G: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        """Task weights ``[K]`` from the Gramian ``G`` of one layer; the tasks not ``present`` get zero."""
        K, w = len(G), torch.zeros(len(G), dtype=torch.float64)
        idx = present.nonzero().flatten()
        if len(idx) <= 1:
            w[idx] = 1.0 / K
            return w
        vals, vecs = torch.linalg.eigh(G[idx][:, idx])
        vals = vals.clamp(min=0.0)
        if self.solver == "upgrad":
            Gp = _regularize(_normalize((vecs * vals) @ vecs.T, self.norm_eps), self.reg_eps)
            w_present = _project_weights(torch.eye(len(idx), dtype=G.dtype) / len(idx), Gp).sum(0)
        else:  # MGDA's min-norm solver on a square root of G: its rows have Gramian G
            w_present = _LIBMTL_MGDA._find_min_norm_element((vecs * vals.sqrt()).float())
        w[idx] = w_present.double() * len(idx) / K
        return w

    def aggregate(self, J: torch.Tensor, losses: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Direction and logged task weights from the rows ``J`` (``[K, D]``). Right after an estimate the chunk
        products of that step give the cross-chunk Gramians (consumed here); otherwise (the backprop Jacobian of the
        diagnostic) the exact Gramians ``J_l J_l^T`` without smoothing."""
        estimated, self.self_products = self.self_products, {}
        d = torch.zeros(J.shape[1], dtype=J.dtype)
        shared = []
        for n, cols in self.columns.items():
            Jl = J[:, cols].double()
            norms = Jl.norm(dim=1)
            present = norms > 1e-6 * norms.max().clamp(min=1e-30)
            if n in estimated:
                M = self.chunks[n]
                G = M / (M - 1) * (Jl @ Jl.T - estimated[n])
                G = self.gram[n] = G if n not in self.gram else self.gram_decay * self.gram[n] + (1 - self.gram_decay) * G
            else:
                G = Jl @ Jl.T
            w = self._solve(G, present)
            d[cols] = (w @ Jl).to(J.dtype)
            if bool(present.all()):
                shared.append(w)
        d[self.free_columns] = J[:, self.free_columns].mean(0)
        weights = torch.stack(shared).mean(0) if shared else torch.full((len(J),), 1.0 / len(J), dtype=torch.float64)
        return d, weights.float()


def run_mo_split_dust(trainer: Any, train_dataloaders: Any, solver: str = "upgrad", gram_decay: float = 0.9,
                      upgrad_norm_eps: float = 1e-4, upgrad_reg_eps: float = 1e-4, **kwargs: Any) -> Dict[str, Any]:
    """MO-UPGrad-Split-DUST / MO-MGDA-Split-DUST: :func:`~.dust.run_dust` (every other argument) with per-layer task
    weights from smoothed cross-chunk Gramians (module docstring). ``solver``: ``upgrad`` or ``mgda``;
    ``gram_decay``: EMA factor of the Gramians over iterations (``0``: no smoothing); ``upgrad_norm_eps`` /
    ``upgrad_reg_eps``: UPGrad's Gramian normalization and diagonal regularization; ``antithetic`` (through
    ``kwargs``): A-DUST's antithetic pairs, an even ``chunk``."""
    if solver not in SOLVERS:
        raise ValueError(f"MO-Split-DUST solver must be one of {tuple(SOLVERS)}, got {solver!r}")
    if not 0.0 <= float(gram_decay) < 1.0:
        raise ValueError(f"MO-Split-DUST gram_decay must be in [0, 1), got {gram_decay}")
    method = f"MO-{SOLVERS[solver]}-Split-{'A' if kwargs.get('antithetic') else ''}DUST"
    built = []

    def estimator(*args: Any) -> SplitDUST:
        built.append(SplitDUST(*args, solver=solver, gram_decay=gram_decay, norm_eps=upgrad_norm_eps,
                               reg_eps=upgrad_reg_eps))
        built[-1].name = method
        return built[-1]

    return run_dust(trainer, train_dataloaders, method=method, _estimator=estimator,
                    aggregate=lambda J, losses: built[-1].aggregate(J, losses), **kwargs)
