"""Pareto set learning with OpenES (PSL-OpenES): one ES run learns a whole Pareto front.

MO-OpenES and OpenES search for one network. Here the search distribution is over a
preference-conditioned family of networks (for two tasks a line in weight space, as in PaMaL):

    theta(lam) = theta_0 + z + S B phi(lam)

* ``lam``: preference on the simplex (``lam_k >= 0``, ``sum_k lam_k = 1``);
* ``phi(lam) = sqrt(2) Q^T (lam - 1/K)``: its coordinates in the simplex plane (``Q`` is an
  orthonormal basis of ``{v : sum(v) = 0}``); for two tasks ``phi = lam_1 - lam_2`` in ``[-1, 1]``;
* ``B``: ``[P, K - 1]`` steering matrix on the ``P`` conditioned parameters (``psl_params``),
  scattered into the full parameter vector by ``S``;
* ``z``: shared delta. At the balanced preference ``lam = 1/K``, ``phi = 0`` and ``theta = theta_0 + z``.

ES searches ``x = (z, B)``. One generation, with antithetic pairs ``j = 1..N/2``:

1. draw a preference ``lam_j`` (stratified uniform for two tasks, Dirichlet(1) otherwise) and
   ``eps_j`` over ``x``; both members of a pair use ``lam_j``;
2. evaluate ``theta(lam_j)`` at ``x +- sigma eps_j`` on the shared mini-batch and score it with its
   own preference: Chebyshev ``max_k lam_k l_k`` (``tch``; ideal point 0, since cross-entropy is
   >= 0) or weighted sum ``sum_k lam_k l_k`` (``ws``);
3. ``d_j = s(x + sigma eps_j) - s(x - sigma eps_j)`` cancels the preference-dependent level of the
   score (ranking raw scores would mostly rank the preferences); the centered ranks of
   ``{d_j} U {-d_j}`` take the place of OpenES's fitness ranks;
4. ``g = 2 sum_j r_j eps_j / (N sigma)`` estimates the gradient of ``E_lam[s(theta(lam), lam)]``
   under Gaussian smoothing, and the optimizer steps on ``x``.

Split update (optional; the defaults give the plain estimator above). ``z`` moves every network and
is the balanced network, ``B`` sets the spread, and the two can follow different weightings of the
same pairs:

* ``center_a > 1``: the ``z`` part of ``g`` weights pair ``j`` by ``Dirichlet(a)/Dirichlet(1)`` at
  ``lam_j`` (for two tasks the ``Beta(a, a)`` density of ``lam_1``), normalised to mean 1, so the
  balanced network follows the near-balanced preferences instead of networks far along the line;
* ``spread_score``: the scores ranked for the ``B`` part: ``"same"`` as ``z``; ``"ideal"``, Chebyshev
  measured from an ideal point (each task's lowest loss in the population on this batch, minus
  ``ideal_margin``) instead of from 0, so an off-centre preference keeps a trade-off with its weaker
  task instead of reducing to one task; or ``"hv"`` (below).

Hypervolume score (``scalarization="hv"`` or ``spread_score="hv"``, two tasks): ``lam_1`` maps to the
angle ``t = pi/2 (1 - lam_1)`` and the direction ``u = (cos t, sin t)`` (``lam_1 = 1``: task 1 only,
``lam_1 = 0.5``: the diagonal), and a network is scored by how far its losses sit inside the box of the
hypervolume reference point ``r`` along that direction::

    rho(l, u) = min_k (r_k - l_k) / u_k            (score -rho; negative rho = outside the box)

With angles spread uniformly, ``HV = pi/4 * E_t[max_l rho^2]`` (Golovin & Zhang, ICML 2020), so the line
is trained on the indicator it is evaluated with: a network gains nothing by trading one task past the
reference point. The ``1/u_k`` factor inflates scores near the ends, so the hv score is meant to be
used with ``rank_groups > 1``: the pair differences are then ranked within groups of pairs with
neighbouring ``lam_1`` instead of across all pairs.

Task guide (``psl_guide="tasks"``, with Guided ES sampling): the guiding subspace is built from the task
losses instead of past updates. Pairs are grouped into regions of the preference simplex (the nearest of the
centre and of K points near the corners; for two tasks, thirds of ``lam_1``). For each region R and task k,
the per-task loss differences of the pairs in R are ranked and give a weight-space direction
``sum_j rho_jk (eps_z,j + S eps_B,j phi_j)``, smoothed over generations with decay ``guided_ema``. Every
direction is placed in the ``z`` block and in each column of ``B``, giving ``(K + 1) K K`` guide vectors (12
for two tasks): the PSL gradient lies almost entirely in the span of the task gradients at a few points of the
family.

The front is read out by evaluating ``theta(lam)`` on a grid of preferences: ``front.csv`` at
every test evaluation, ``psl_result.pt`` at the end. The balanced network ``theta_0 + z`` is the
returned solution (``pareto_center_X``) and fills the train/test columns of ``metrics.csv``, like
the ES mean of :func:`~LibMTL.evomtl.moea.mo_openes.run_mo_openes`.

References:
- PaMaL: Dimitriadis, Frossard and Fleuret, "Pareto Manifold Learning: Tackling multiple tasks
  via ensembles of single-task models", ICML 2023
- Pareto HyperNetworks: Navon et al., "Learning the Pareto Front with Hypernetworks", ICLR 2021
- OpenES: https://arxiv.org/abs/1703.03864
"""

from __future__ import annotations

import csv
import itertools
import math
import os
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from .common import configure_moea_random_seed, should_run_full_test
from .hv import pareto_front_indices
from .mo_openes import centered_ranks, guided_noise, make_es_optimizer, update_guide
from .pymoo_deps import Hypervolume

SCALARIZATIONS = ("tch", "ws", "hv")
SPREAD_SCORES = ("same", "ideal", "hv")
PSL_GUIDES = ("update", "tasks")


def conditioned_indices(model: torch.nn.Module, spec: str) -> torch.Tensor:
    """Flat positions (order of ``model.parameters()``) of the parameters whose names start with
    one of the comma-separated prefixes in ``spec``; ``"all"`` selects every parameter."""
    prefixes = [s.strip() for s in str(spec).split(",") if s.strip()]
    idx, offset = [], 0
    for name, p in model.named_parameters():
        if "all" in prefixes or any(name.startswith(pre) for pre in prefixes):
            idx.append(torch.arange(offset, offset + p.numel()))
        offset += p.numel()
    if not idx:
        names = [n for n, _ in model.named_parameters()]
        raise ValueError(f"psl_params={spec!r} matches no parameter; names: {names}")
    return torch.cat(idx)


def simplex_basis(K: int) -> torch.Tensor:
    """Orthonormal basis ``[K, K - 1]`` of ``{v : sum(v) = 0}``; column ``i`` follows ``e_i - 1/K``
    (for two tasks the single column is ``(1, -1) / sqrt(2)``)."""
    A = torch.eye(K, dtype=torch.float32)[:, : K - 1] - 1.0 / K
    Q, R = torch.linalg.qr(A)
    return Q * torch.sign(torch.diagonal(R))


def preference_features(lam: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """``phi(lam) = sqrt(2) Q^T (lam - 1/K)`` for the rows of ``lam`` (``[M, K]``); ``lam_1 - lam_2`` for two tasks."""
    return math.sqrt(2.0) * (lam - 1.0 / lam.shape[1]) @ Q


def sample_preferences(M: int, K: int, gen: torch.Generator) -> torch.Tensor:
    """``M`` preferences ``[M, K]``, uniform on the simplex: stratified for two tasks (one ``lam_1``
    in each bin of width ``1/M``, in random order), Dirichlet(1) from sorted uniforms otherwise."""
    if K == 2:
        lam1 = (torch.randperm(M, generator=gen).float() + torch.rand(M, generator=gen, dtype=torch.float32)) / M
        return torch.stack([lam1, 1.0 - lam1], dim=1)
    u = torch.sort(torch.rand(M, K - 1, generator=gen, dtype=torch.float32), dim=1).values
    zeros, ones = torch.zeros(M, 1, dtype=torch.float32), torch.ones(M, 1, dtype=torch.float32)
    return torch.cat([zeros, u, ones], dim=1).diff(dim=1)


def center_weights(lam: torch.Tensor, a: float) -> torch.Tensor:
    """Importance weights ``Dirichlet(a)/Dirichlet(1)`` of the preferences ``lam`` (``[M, K]``), normalised to
    mean 1: weighting by them follows preferences from ``Dirichlet(a, ..., a)`` (``lam_1 ~ Beta(a, a)`` for two
    tasks), concentrated around the balanced preference for ``a > 1``. ``a = 1`` gives equal weights."""
    if a == 1.0:
        return torch.ones(lam.shape[0], dtype=torch.float32)
    logw = (a - 1.0) * torch.log(lam.clamp_min(1e-12)).sum(dim=1)
    w = torch.exp(logw - logw.max())
    return w / w.mean()


def guide_regions(lam: torch.Tensor) -> torch.Tensor:
    """Region of each preference (rows of ``lam``, ``[M, K]``) for the task guide: index of the nearest of the
    centre ``1/K`` (region 0) and the K points ``1/(3K) + (2/3) e_k`` (regions 1..K); for two tasks these are
    the thirds of ``lam_1``."""
    K = lam.shape[1]
    refs = torch.cat([torch.full((1, K), 1.0 / K, dtype=lam.dtype),
                      torch.full((K, K), 1.0 / (3 * K), dtype=lam.dtype) + (2.0 / 3.0) * torch.eye(K, dtype=lam.dtype)])
    return torch.cdist(lam, refs).argmin(dim=1)


def preference_grid(K: int, n: int) -> torch.Tensor:
    """Simplex lattice with ``n`` points per edge (Das-Dennis); for two tasks ``lam_1 = 0, 1/(n-1), ..., 1``."""
    H = max(int(n) - 1, 1)
    pts = [c for c in itertools.product(range(H + 1), repeat=K) if sum(c) == H]
    return torch.tensor(pts, dtype=torch.float32) / H


def hv_directions(lam: np.ndarray) -> np.ndarray:
    """Hypervolume preference directions ``(cos t, sin t)`` with ``t = pi/2 (1 - lam_1)`` (two tasks): uniform
    ``lam_1`` gives uniform angles, ``lam_1 = 1`` is task 1 only and ``lam_1 = 0.5`` the diagonal."""
    t = 0.5 * math.pi * (1.0 - lam[:, 0])
    return np.stack([np.cos(t), np.sin(t)], axis=1)


def scalarize(F: np.ndarray, lam: np.ndarray, kind: str, ref: Optional[np.ndarray] = None) -> np.ndarray:
    """Score of each row of ``F`` under its preference, lower is better: Chebyshev ``max_k lam_k F_k``, weighted
    sum ``sum_k lam_k F_k``, or ``hv``: ``-rho`` with ``rho = min_k (ref_k - F_k) / u_k`` along the direction
    ``u`` of :func:`hv_directions`."""
    if kind == "tch":
        return (lam * F).max(axis=1)
    if kind == "ws":
        return (lam * F).sum(axis=1)
    return ((F - ref) / np.maximum(hv_directions(lam), 1e-12)).max(axis=1)


def hypervolume(F: np.ndarray, ref: np.ndarray) -> float:
    """Hypervolume of ``F`` (minimization) w.r.t. ``ref``; points not better than ``ref`` add nothing."""
    F = F[np.all(F < ref, axis=1)]
    return float(Hypervolume(ref_point=ref).do(F)) if len(F) else 0.0


def run_psl_es(
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
    psl_guide: str = "update",
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
    """PSL-OpenES ask-and-tell on ``x = (z, B)`` (see module docstring).

    Parameters
    ----------
    scalarization : str
        ``"tch"`` (Chebyshev, default), ``"ws"`` (weighted sum) or ``"hv"`` (hypervolume score from
        ``hv_ref_point``, two tasks): score of a member under its preference.
    psl_params : str
        Parameters that depend on the preference: ``"all"`` (default) or comma-separated name
        prefixes from ``model.named_parameters()``, e.g. ``"encoder.fc,decoders"``. Each
        conditioned parameter adds ``K - 1`` search dimensions.
    eval_points : int
        Preferences per simplex edge on which the front is evaluated (11: ``lam_1 = 0, 0.1, ..., 1``).
    center_a : float
        Concentration of the preference weights in the ``z`` part of the update (``Beta(a, a)`` for two
        tasks); 1 (default) weights all pairs equally.
    spread_score : str
        Scores ranked for the ``B`` part of the update: ``"same"`` (default, the ranks of ``z``), ``"ideal"``
        (Chebyshev from each task's lowest population loss on the batch minus ``ideal_margin``) or ``"hv"``
        (hypervolume score from ``hv_ref_point``).
    rank_groups : int
        Rank the pair differences within this many groups of pairs with neighbouring ``lam_1`` (two tasks)
        instead of across all pairs (1, default).
    psl_guide : str
        Guiding subspace of Guided ES sampling (``guided_k > 0``): ``"update"`` (default, a moving average of
        past update directions) or ``"tasks"`` (per-task, per-region directions along the family; see above).
    hv_ref_point
        If given, the test hypervolume of that front is printed and logged (``test_hv``).

    The other parameters are those of :func:`~LibMTL.evomtl.moea.mo_openes.run_mo_openes`; Guided ES
    sampling (``guided_*``) acts on ``x``. Needs ``evo_adapter=full`` and antithetic sampling
    (both members of a pair share one preference).
    """
    if trainer.adapter is None:
        raise RuntimeError("init_parameter_sharing() first.")
    if type(trainer.adapter).__name__ != "FullParameters":
        raise ValueError("PSL-OpenES conditions model parameters directly and needs evo_adapter=full, "
                         f"got {type(trainer.adapter).__name__}.")
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
    split = center_a != 1.0 or spread_score != "same"
    psl_guide = str(psl_guide).strip().lower()
    if psl_guide not in PSL_GUIDES:
        raise ValueError(f"Unknown PSL guide {psl_guide!r}; options: {PSL_GUIDES}")
    pop_size = int(pop_size)
    if not antithetic or pop_size % 2 != 0:
        raise ValueError(f"PSL-OpenES needs antithetic sampling and an even pop_size (a pair shares one "
                         f"preference), got antithetic={antithetic}, pop_size={pop_size}.")

    configure_moea_random_seed(seed)
    gen = torch.Generator().manual_seed(int(seed))

    D = int(trainer.adapter.num_dims)
    K = int(trainer.task_num)
    M = pop_size // 2
    idx = conditioned_indices(trainer.model, psl_params)
    P, C = int(idx.numel()), K - 1
    Q = simplex_basis(K)
    n_var = D + P * C
    x = torch.zeros(n_var, dtype=torch.float32)
    x[:D] = torch.as_tensor(trainer.adapter.z0, dtype=torch.float32).reshape(-1)  # B starts at 0
    x.requires_grad_(True)
    optim = str(optim).strip().lower()
    optimizer = make_es_optimizer([x], optim, lr, adam_betas, adam_eps, rmsprop_alpha, rmsprop_eps)
    sigma = float(sigma)
    method = f"PSL-OpenES-{scalarization.upper()}"
    if guided_k > 0:  # Guided ES sampling: PSL-OpenES-X -> PSL-GuidedES-X
        method = method.replace("OpenES", "GuidedES")
    guide: list = []
    task_dirs: Dict[Tuple[int, int], torch.Tensor] = {}  # psl_guide="tasks": (region, task) -> weight-space direction
    grid = preference_grid(K, eval_points)
    grid_phi = preference_features(grid, Q)
    balanced_phi = torch.zeros(C, dtype=torch.float32)  # phi(1/K) = 0
    ref = None if hv_ref_point is None else np.asarray(hv_ref_point, dtype=np.float64).reshape(K)

    def delta(xv: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """Parameter delta ``z + S B phi`` of the network for preference features ``phi`` (``[C]``)."""
        d = xv[:D].clone()
        d.index_add_(0, idx, xv[D:].view(P, C) @ phi)
        return d

    def evaluate_front(xv: torch.Tensor, batches: list) -> Dict[str, list]:
        """Metrics of ``theta(lam)`` for every grid preference: current train batch and test set."""
        out: Dict[str, list] = {"train": [], "test": []}
        for g in range(len(grid)):
            trainer.adapt_parameters(delta(xv, grid_phi[g]))
            out["train"].append(trainer.metrics_snapshot_from_prefetched_batches(batches))
            if test_dataloaders is not None:
                out["test"].append(trainer.test(test_dataloaders, epoch=None, mode="test",
                                                suppress_display=True, return_metrics=True))
        return out

    print(
        f"[EvoMTL] {method}: dims={n_var} | pop={pop_size} | optim={optim} lr={lr} | sigma={sigma} "
        f"(decay={sigma_decay}, min={sigma_min}) | conditioned params={P} (psl_params={psl_params}), "
        f"z dims={D}, B dims={P * C} | scalarization={scalarization} | front grid={len(grid)} preferences"
        + (f" | guided k={guided_k}, alpha={guided_alpha}, sigma_sub={guided_sigma or 'auto'}, ema={guided_ema}, "
           f"guide={psl_guide}" + (f" ({(K + 1) * K * K} vectors)" if psl_guide == "tasks" else "")
           if guided_k > 0 else "")
        + (f" | split update: z weights Dirichlet(a={center_a}), B scores={spread_score}"
           + (f" (margin {ideal_margin})" if spread_score == "ideal" else "") if split else "")
        + (f" | ranks within {rank_groups} preference groups" if rank_groups > 1 else ""),
        flush=True,
    )

    num_iterations = int(num_iterations)
    tasks = list(trainer.task_name)
    # metrics.csv: the balanced network, same columns as run_mo_openes for the train/test metrics.
    # front.csv: every grid preference at every test evaluation (split = train batch / test set).
    csv_cols = (
        ["generation", "time_s", "sigma"]
        + [f"pop_loss_mean_{t}" for t in tasks]
        + [f"train_loss_{t}" for t in tasks] + [f"train_acc_{t}" for t in tasks]
        + [f"test_loss_{t}" for t in tasks] + [f"test_acc_{t}" for t in tasks]
        + ["b_norm", "test_hv"]
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

        # ask: eps over x, one preference per antithetic pair
        eps = torch.randn(M, n_var, generator=gen, dtype=torch.float32)
        if guided_k > 0 and guide:
            eps = guided_noise(eps, guide, gen, sigma, guided_alpha, guided_sigma)
        lam = sample_preferences(M, K, gen)
        phi = preference_features(lam, Q)

        # evaluate every member on the same mini-batch(es): rows j (x + sigma eps_j) and M + j (x - sigma eps_j)
        shared_batches = trainer._sample_batches(train_dataloaders, num_batches)
        xd = x.detach()
        for j in range(M):
            for s, sign in enumerate((1.0, -1.0)):
                trainer.adapt_parameters(delta(xd + sign * sigma * eps[j], phi[j]))
                F[j + s * M] = trainer.evaluate_multiobjective_losses(
                    train_dataloaders, prefetched_batches=shared_batches
                )

        # tell: ranks of the pair differences of the preference scores -> search gradient on x
        lam_np = np.concatenate([lam.numpy(), lam.numpy()])

        def pair_ranks(scores: np.ndarray) -> torch.Tensor:
            diff = scores[:M] - scores[M:]
            if rank_groups <= 1:
                return torch.as_tensor(centered_ranks(np.concatenate([diff, -diff]))[:M], dtype=torch.float32)
            r = np.zeros(M)
            for grp in np.array_split(np.argsort(lam_np[:M, 0]), rank_groups):
                r[grp] = centered_ranks(np.concatenate([diff[grp], -diff[grp]]))[: len(grp)]
            return torch.as_tensor(r, dtype=torch.float32)

        ranks = pair_ranks(scalarize(F, lam_np, scalarization, ref))
        if split:  # z: centre-weighted ranks; B: ranks of the scores chosen by spread_score
            ranks_b = ranks
            if spread_score == "ideal":
                ranks_b = pair_ranks(scalarize(F - (F.min(axis=0) - ideal_margin), lam_np, "tch"))
            elif spread_score == "hv":
                ranks_b = pair_ranks(scalarize(F, lam_np, "hv", ref))
            direction = 2.0 * torch.cat([(center_weights(lam, center_a) * ranks) @ eps[:, :D],
                                         ranks_b @ eps[:, D:]]) / (pop_size * sigma)
        else:
            direction = 2.0 * (ranks @ eps) / (pop_size * sigma)
        x.grad = direction.clone()
        optimizer.step()
        if guided_k > 0 and psl_guide == "tasks":
            # weight-space perturbation of each pair (eps_z + S eps_B phi_j) and per-task ranks within regions
            dtheta = eps[:, :D].clone()
            dtheta[:, idx] += (eps[:, D:].view(M, P, C) * phi[:, None, :]).sum(-1)
            regions = guide_regions(lam)
            for r in range(K + 1):
                members = (regions == r).nonzero().flatten().numpy()
                if len(members) < 2:
                    continue
                for k in range(K):
                    d = F[members, k] - F[members + M, k]
                    rk = torch.as_tensor(centered_ranks(np.concatenate([d, -d]))[: len(members)], dtype=torch.float32)
                    v = rk @ dtheta[members]
                    v = v / v.norm().clamp_min(1e-30)
                    old = task_dirs.get((r, k))
                    task_dirs[(r, k)] = v if old is None else guided_ema * old + (1.0 - guided_ema) * v
            guide = []  # each direction in the z block and in every column of B
            for g in task_dirs.values():
                guide.append(torch.cat([g, torch.zeros(P * C, dtype=torch.float32)]))
                for c in range(C):
                    b = torch.zeros(P, C, dtype=torch.float32)
                    b[:, c] = g[idx]
                    guide.append(torch.cat([torch.zeros(D, dtype=torch.float32), b.flatten()]))
        elif guided_k > 0:
            guide = update_guide(guide, direction, guided_k, guided_ema)
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
            extras=f"sigma={sigma:.4g} |d|={float(direction.norm()):.4g} |B|={b_norm:.4g}",
            elapsed=elapsed,
            wandb_log_step=step,
        )
        trainer._wandb_log_evo_front_metrics(iteration, pareto_front, step, extras={
            "evo/psl/sigma": sigma,
            "evo/psl/direction_norm": float(direction.norm()),
            "evo/psl/b_norm": b_norm,
        })

        # the balanced network theta_0 + z: snapshot on the current batch (and test periodically)
        trainer.adapt_parameters(delta(xd, balanced_phi))
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
                for split, ms in front.items():
                    for g, m in enumerate(ms):
                        row = {"generation": iteration + 1, "split": split}
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
                f"[EvoMTL] Balanced z TRAIN (curr batch) iteration {iteration + 1:04d} | {parts}",
                flush=True,
            )

        if csv_writer is not None:
            row: Dict[str, Any] = {"generation": iteration + 1, "time_s": round(elapsed, 4), "sigma": sigma,
                                   "b_norm": b_norm}
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
    xd = x.detach()
    front = evaluate_front(xd, shared_batches)
    grid_X = torch.stack([delta(xd, grid_phi[g]) for g in range(len(grid))]).numpy()
    train_F = np.array([m["loss_item"] for m in front["train"]], dtype=np.float64)
    test_F = np.array([m["loss_item"] for m in front["test"]], dtype=np.float64).reshape(-1, K)
    test_acc = np.array([[m["results"][t][0] for t in tasks] for m in front["test"]], dtype=np.float64).reshape(-1, K)
    if getattr(trainer, "save_path", None):
        torch.save({
            "z": xd[:D].clone(),                    # shared delta: the balanced network is theta_0 + z
            "B": xd[D:].view(P, C).clone(),         # steering matrix on the conditioned parameters
            "index": idx,                           # flat positions of the conditioned parameters
            "Q": Q,                                 # phi(lam) = sqrt(2) Q^T (lam - 1/K)
            "grid": grid, "train_F": train_F, "test_F": test_F, "test_acc": test_acc,
            "tasks": tasks, "psl_params": psl_params, "scalarization": scalarization, "method": method,
            "pop_size": pop_size, "lr": float(lr), "sigma": sigma,
        }, os.path.join(trainer.save_path, "psl_result.pt"))

    center_X = delta(xd, balanced_phi).numpy().copy()
    trainer.adapt_parameters(center_X)
    out = {
        "method": method,
        "pareto_F": train_F,       # the grid front on the last mini-batch
        "pareto_X": grid_X,        # parameter deltas of the grid networks
        "population_F": F,
        "population_X": np.empty((0, D)),
        "hv_weights": None,
        "pareto_center_X": center_X,
        "hv_ref_point": ref,
        "hv_center_aggregation": None,
        "hv_softmax_temperature": None,
        "sigma": sigma,
        "scalarization": scalarization,
        "front_lambda": grid.numpy(),
        "front_test_F": test_F,
    }
    trainer.last_evo = out
    return out
