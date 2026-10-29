"""Figures of the toy runs (``runner.py``, read from ``logs/<config name>/``): objective-space trajectories and convergence to the Pareto front.

* ``<fig-dir>/traj_<method>.pdf``: one panel per method, the Pareto front (grey) and, for each of the five initial
  points, the seed-mean trajectory of the losses (orange early to purple late, the colors of Nash-MTL; start = black
  circle, end = small dot; black star: minimum of the average loss), as in
  Fig. 1 of Nash-MTL. Each axis fits the data, as in the paper (l1 spans ~10x less than l2 with scale 0.1, so equal
  axes would flatten the front); the limits are shared by every panel.
* ``<fig-dir>/convergence_<family>.pdf``: distance of the search mean to the Pareto front (objective space), mean over
  initial points and seeds, per method of the OpenES / GuidedES family.
* ``<fig-dir>/overview.png``: every trajectory panel on one grid (quick look).
* ``<fig-dir>/nashmtl/<method>.pdf``: the figure of ``experiments/toy/trainer.py`` of Nash-MTL
  (``plot_2d_pareto`` of its ``utils.py``, reproduced in :func:`plot_2d_pareto`), one run per initial point
  (``--seed``).
* ``<fig-dir>/3d/<method>.pdf``: the runs (``--seed``) on the surface of the average loss over
  ``(theta_1, theta_2)``, as in Fig. 1 (bottom row) of Aligned-MTL (:func:`plot_3d`).
* ``<fig-dir>/contour/<method>.pdf``: the same runs on the contour map of the average loss (:func:`plot_contour`).
* ``<logs>/summary.tex``: final distance to the front per initial point, mean +- std over seeds.

For the quadratic problems of :mod:`problems` (``logs/<problem>/``) instead: ``<fig-dir>/contour/<method>.pdf``,
``<fig-dir>/loss/<method>.pdf`` and ``<fig-dir>/convergence_<family>.pdf`` (:func:`problem_figures`; default
``<fig-dir>`` = ``figures/<problem>``).

Usage (from the repository root)::

    python examples/toy/utils.py [--logs examples/toy/logs] [--fig-dir examples/toy/figures/scale_<s>] [--seed 1]
    python examples/toy/utils.py --logs examples/toy/logs/cqf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import matplotlib.ticker  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import torch  # noqa: E402
import yaml  # noqa: E402

from problem import Toy  # noqa: E402
from problems import NashMTLToy, make_problem  # noqa: E402
from runner import PARETO_X2, pareto_front  # noqa: E402

FAMILIES = {
    "OpenES": ["openes", "mo_mean_openes", "mo_upgrad_openes", "mo_cagrad_openes", "mo_mgda_openes",
               "mo_mgda2_openes", "mo_mgda_loss_openes", "mo_mgda_loss2_openes"],
    "GuidedES": ["guidedes", "mo_mean_guidedes", "mo_upgrad_guidedes", "mo_cagrad_guidedes", "mo_mgda_guidedes",
                 "mo_mgda2_guidedes", "mo_mgda_loss_guidedes", "mo_mgda_loss2_guidedes"],
}
# convergence curves, by position in the family: baseline, MO-Mean, MO-UPGrad, MO-CAGrad, MO-MGDA, MO-MGDA2,
# MO-MGDA(LN), MO-MGDA(LN2)
COLORS = ["C0", "C4", "C2", "C1", "C3", "C5", "C7", "C6"]
MARKERS = ["o", "s", "^", "P", "D", "X", "v", "<"]
# trajectory panels: the colors of Nash-MTL's plot_2d_pareto for every method
FRONT_COLOR = "#72727A"
TRAJ_CMAP, TRAJ_RANGE = matplotlib.cm.magma_r, (0.1, 0.6)  # orange (start) to purple (end)

def position(key: str) -> int:
    return next(fam.index(key) for fam in FAMILIES.values() if key in fam)


def gradient_line(ax, xy: np.ndarray) -> None:
    seg = np.stack([xy[:-1], xy[1:]], axis=1)
    colors = TRAJ_CMAP(np.linspace(*TRAJ_RANGE, len(seg)))
    ax.add_collection(LineCollection(seg, colors=colors, linewidth=3, zorder=5))


def nashmtl_axes(ax) -> None:
    """Axis style of Nash-MTL's plot_2d_pareto: no top / right spines, large labels at the ends of the axes."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel(r"$\mathcal{L}_1$", size=30)
    ax.set_ylabel(r"$\mathcal{L}_2$", size=30)
    ax.xaxis.set_label_coords(1.015, -0.03)
    ax.yaxis.set_label_coords(-0.01, 1.01)
    ax.tick_params(labelsize=20)


def plot_convergence(keys: list, dist: dict, names: dict, ylabel: str = "Distance to Pareto front", gens=None):
    """Distance to the Pareto front against the generation (``dist[k]``: ``[inits, seeds, T + 1]``, or at the
    generations ``gens``): mean over the initial points; over the seeds, in log scale (the axis): line = geometric
    mean, band = one standard deviation of log10(distance). Nash-MTL-sized fonts."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for k in keys:
        log_d = np.log10(np.maximum(dist[k].mean(axis=0), 1e-12))  # [seeds, T + 1]
        mean, std = log_d.mean(axis=0), log_d.std(axis=0, ddof=1)
        gens = np.arange(len(mean)) if gens is None else gens
        p = position(k)
        ax.plot(gens, 10 ** mean, color=COLORS[p], marker=MARKERS[p], markevery=max(1, len(gens) // 8),
                markersize=9, linewidth=2.5, label=names[k])
        ax.fill_between(gens, 10 ** (mean - std), 10 ** (mean + std), color=COLORS[p], alpha=0.2, linewidth=0)
    ax.set_yscale("log")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Generation", size=20)
    ax.set_ylabel(ylabel, size=20)
    ax.tick_params(labelsize=16)
    ax.xaxis.get_offset_text().set_fontsize(16)
    ax.legend(fontsize=12, frameon=False)
    return ax, fig


def rasterize_dense(fig, n: int = 1000) -> None:
    """Rasterize the artists with more than ``n`` points (trajectories, bands) so PDFs stay small; axes, text and
    sparse markers stay vector. Save with ``dpi=300``."""
    for ax in fig.axes:
        for artist in list(ax.collections) + list(ax.lines):
            if hasattr(artist, "get_xdata"):
                size = len(artist.get_xdata())
            elif hasattr(artist, "_segments3d"):
                size = len(artist._segments3d)
            elif hasattr(artist, "get_segments") and len(artist.get_segments()):
                size = len(artist.get_segments())
            else:
                size = max(len(artist.get_offsets()), sum(len(q.vertices) for q in artist.get_paths()))
            if size > n:
                artist.set_rasterized(True)


def optimum_losses(scale: float) -> np.ndarray:
    """Losses at the minimum of the average loss (marked with a black star in every figure)."""
    problem = NashMTLToy(scale=scale)
    return problem.losses(problem.global_optimum())


def match_ticks(ax) -> None:
    """Same tick step on both axes when they span the same range (scale 1); matplotlib otherwise picks a coarser step
    for the wider x axis. Scale 0.1 keeps independent ticks (L1 spans a tenth of L2)."""
    same = np.allclose(ax.get_xlim(), ax.get_ylim(), rtol=0.05, atol=0.5)
    for axis in (ax.xaxis, ax.yaxis) if same else (ax.yaxis,):  # L2 (y) always spans ~30: step 5
        axis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=7))


def extend_to_tick(ax) -> None:
    """Raise the upper limit of each axis to the next tick, so the last positive tick is shown (the data end just
    below it: L1 at ~0.9 with scale 0.1, L2 at ~9.3)."""
    for axis, get, set_ in ((ax.xaxis, ax.get_xlim, ax.set_xlim), (ax.yaxis, ax.get_ylim, ax.set_ylim)):
        lo, hi = get()
        t = axis.get_ticklocs()
        hi = t[t >= hi].min() if (t >= hi).any() else hi
        set_(lo, hi)
        axis.set_ticks(t[(t >= lo) & (t <= hi)])  # keep the step (a wider range could change it)


def traj_panel(ax, key: str, name: str, L: np.ndarray, front: np.ndarray, xlim, ylim, star: np.ndarray) -> None:
    mean = L.mean(axis=1)  # [inits, T + 1, 2]: seed mean
    ax.plot(front[:, 0], front[:, 1], "-", color=FRONT_COLOR, linewidth=8, label="Pareto front", zorder=1)
    ax.scatter(*star, marker="*", s=300, color="k", zorder=11)
    for i in range(mean.shape[0]):
        gradient_line(ax, mean[i])
        ax.scatter(*mean[i, 0], color="k", s=150, zorder=10)
        ax.scatter(*mean[i, -1], color=TRAJ_CMAP(TRAJ_RANGE[1]), s=30, zorder=10)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    nashmtl_axes(ax)
    match_ticks(ax)


def plot_2d_pareto(trajectories: dict, scale: float):
    """``plot_2d_pareto`` of Nash-MTL's ``experiments/toy/utils.py`` (itself adapted from CAGrad), unchanged except:
    ``sns.despine()`` is written out (no seaborn here) and the per-tick ``tick.label.set_fontsize`` (removed from
    matplotlib) is ``ax.tick_params``. ``trajectories``: ``{i: {"traj": [T, 2] x-space trajectory}}``."""
    fig, ax = plt.subplots(figsize=(6, 5))

    F = Toy(scale=scale)

    losses = []
    for res in trajectories.values():
        losses.append(F.batch_forward(torch.from_numpy(res["traj"])))

    yy = -8.3552
    x = np.linspace(-7, 7, 1000)

    inpt = np.stack((x, [yy] * len(x))).T
    Xs = torch.from_numpy(inpt).double()

    Ys = F.batch_forward(Xs)
    ax.plot(
        Ys.numpy()[:, 0],
        Ys.numpy()[:, 1],
        "-",
        linewidth=8,
        color="#72727A",
        label="Pareto Front",
    )  # Pareto front

    for i, tt in enumerate(losses):
        ax.scatter(
            tt[0, 0],
            tt[0, 1],
            color="k",
            s=150,
            zorder=10,
            label="Initial Point" if i == 0 else None,
        )
        colors = matplotlib.cm.magma_r(np.linspace(0.1, 0.6, tt.shape[0]))
        ax.scatter(tt[:, 0], tt[:, 1], color=colors, s=5, zorder=9)

    ax.scatter(*optimum_losses(scale), marker="*", s=300, color="k", zorder=11)  # minimum of the average (added)

    ax.spines[["top", "right"]].set_visible(False)  # sns.despine()
    ax.set_xlabel(r"$\mathcal{L}_1$", size=30)
    ax.set_ylabel(r"$\mathcal{L}_2$", size=30)
    ax.xaxis.set_label_coords(1.12, -0.03)  # (moved out from (1.015, -0.03): clear of the end tick, extend_to_tick)
    ax.yaxis.set_label_coords(-0.01, 1.09)  # (moved out from (-0.01, 1.01))

    ax.tick_params(labelsize=20)
    match_ticks(ax)  # (added)
    extend_to_tick(ax)  # (added)

    plt.tight_layout()

    legend = ax.legend(
        loc=2, bbox_to_anchor=(-0.15, 1.3), frameon=False, fontsize=20, ncol=2
    )

    return ax, fig, legend


def mean_loss(X: np.ndarray, scale: float) -> np.ndarray:
    """Average of the two losses at the points ``X`` (``[..., 2]``, x-space)."""
    with torch.no_grad():
        L = Toy(scale=scale).batch_forward(torch.from_numpy(X.reshape(-1, 2)).double()).numpy()
    return L.mean(axis=1).reshape(X.shape[:-1])


def plot_3d(trajectories: np.ndarray, scale: float, xl: float = 10.0, n: int = 300):
    """Trajectories on the surface of the average loss over ``(theta_1, theta_2)``, as in Fig. 1 (bottom row) of
    Aligned-MTL (Senushkin et al., CVPR 2023): surface ``viridis`` (dark = low, yellow = high), Pareto set (grey line), minimum
    of the average loss (star), runs from yellow (start, black dot) to dark red. ``trajectories``: ``[inits, T, 2]``
    in x-space."""
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(projection="3d", computed_zorder=False)  # draw in zorder: runs on top of the surface

    g = np.linspace(-xl, xl, n)
    T1, T2 = np.meshgrid(g, g)
    Z = mean_loss(np.stack([T1, T2], axis=-1), scale)
    ax.plot_surface(T1, T2, Z, cmap="viridis", rstride=1, cstride=1, linewidth=0, antialiased=False,
                    rasterized=True, zorder=1)

    ps = np.stack([np.linspace(-7.0, 7.0, 500), np.full(500, PARETO_X2)], axis=1)
    ax.plot(ps[:, 0], ps[:, 1], mean_loss(ps, scale), color="0.5", linewidth=4, zorder=2)
    opt = NashMTLToy(scale=scale).global_optimum()
    ax.scatter(*opt, mean_loss(opt[None], scale)[0], marker="*", s=200, color="k", depthshade=False, zorder=6)

    for traj in trajectories:
        z = mean_loss(traj, scale)
        pts = np.column_stack([traj, z])
        seg = np.stack([pts[:-1], pts[1:]], axis=1)
        colors = matplotlib.cm.inferno_r(np.linspace(0.15, 0.85, len(seg)))
        ax.add_collection3d(Line3DCollection(seg, colors=colors, linewidth=2.5, zorder=4))
        ax.scatter(*pts[0], color="k", s=20, depthshade=False, zorder=5)
        ax.scatter(*pts[-1], color=colors[-1], s=8, depthshade=False, zorder=5)

    ax.set_xlim(-xl, xl)
    ax.set_ylim(-xl, xl)
    ax.set_xticks([-xl, -xl / 2, 0, xl / 2, xl])
    ax.set_yticks([-xl, -xl / 2, 0, xl / 2, xl])
    ax.zaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("none")
    return ax, fig


def plot_contour(trajectories: np.ndarray, scale: float, xl: float = 10.0, n: int = 200):
    """Contour map of the average loss over ``(theta_1, theta_2)`` with the runs, the 2-D counterpart of
    :func:`plot_3d`: contour lines on white from ``viridis``, dark purple = low, yellow = high; grey Pareto set; black
    star at the minimum of the average loss; runs from yellow (start, black dot) to dark red (end, small dot). ``trajectories``:
    ``[inits, T, 2]`` in x-space."""
    ticks = [-xl, -xl / 2, 0, xl / 2, xl]
    return contour_panel(NashMTLToy(scale=scale), trajectories, (-xl, xl), (-xl, xl), ticks=ticks, n=n)


def contour_panel(problem, trajectories: np.ndarray, xlim, ylim, ticks=None, n: int = 200,
                  levels="linear"):
    """:func:`plot_contour` for any 2-D problem of :mod:`problems` over ``xlim`` x ``ylim``: contour lines of the
    average loss, the Pareto set (grey; a grey dot when it is one point), the minimum of the
    average loss (black star) and the runs.
    ``levels``: ``"linear"`` (20 equally spaced) or ``"quantile"`` (at 20 quantiles of the map, for losses that span
    orders of magnitude)."""
    fig, ax = plt.subplots(figsize=(5, 5))
    T1, T2 = np.meshgrid(np.linspace(*xlim, n), np.linspace(*ylim, n))
    Z = problem.losses(np.stack([T1, T2], axis=-1)).mean(axis=-1)
    if levels == "quantile":
        levels = np.unique(np.quantile(Z, np.linspace(0.0, 1.0, 22)[1:-1]))
    else:
        levels = 20
    # contour lines on white, colored by level as the surface of plot_3d (dark purple = low, yellow = high)
    ax.contour(T1, T2, Z, levels=levels, cmap="viridis", linewidths=1.2, zorder=2)

    ps = problem.pareto_set(500)
    if np.ptp(ps, axis=0).max() < 1e-9:  # a single point (ewq)
        ax.scatter(*ps[0], color="0.5", s=250, zorder=3)
    else:
        ax.plot(*ps.T, color="0.5", linewidth=4, zorder=3)
    ax.scatter(*problem.global_optimum(), marker="*", s=300, color="k", zorder=7)
    for traj in trajectories:
        # vector figure kept small (a 300-dpi raster of the map is larger): 400 points per smooth run (below the
        # rasterize_dense threshold), drawn as 40 polylines of one color each
        traj = traj[np.unique(np.linspace(0, len(traj) - 1, 400).astype(int))]
        cuts = np.linspace(0, len(traj) - 1, 41).astype(int)
        seg = [traj[a:b + 1] for a, b in zip(cuts[:-1], cuts[1:])]
        colors = matplotlib.cm.inferno_r(np.linspace(0.15, 0.85, len(seg)))
        ax.plot(*traj.T, color="k", linewidth=4.5, alpha=0.6, zorder=4)  # outline: the yellow start stays visible
        ax.add_collection(LineCollection(seg, colors=colors, linewidth=3, zorder=5))
        ax.scatter(*traj[0], color="k", s=80, zorder=6)
        ax.scatter(*traj[-1], color=colors[-1], s=20, zorder=6)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    if ticks is not None:
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel(r"$\theta_1$", size=26, labelpad=0)
    ax.set_ylabel(r"$\theta_2$", size=26, labelpad=0)
    ax.tick_params(labelsize=16)
    return ax, fig


def loss_panel(problem, losses: np.ndarray, xlim=None, ylim=None):
    """Loss-plane trajectories of a two-objective problem of :mod:`problems` in the style of :func:`plot_2d_pareto`
    (``losses``: ``[inits, T, 2]``, one seed): Pareto front (grey; a grey dot when it is one point), initial points
    (black), every iterate colored from orange (start) to purple (end), end = small dot, losses at the minimum of the
    average loss (black star). Limits fit the runs and the
    front unless given."""
    fig, ax = plt.subplots(figsize=(6, 5))
    front = problem.pareto_front(1000)
    if np.ptp(front, axis=0).max() < 1e-9:
        ax.scatter(*front[0], color=FRONT_COLOR, s=250, zorder=1)
    else:
        ax.plot(*front.T, "-", linewidth=8, color=FRONT_COLOR, zorder=1)
    ax.scatter(*problem.losses(problem.global_optimum()), marker="*", s=300, color="k", zorder=11)
    for L in losses:
        gradient_line(ax, L)
        ax.scatter(*L[0], color="k", s=150, zorder=10)
        ax.scatter(*L[-1], color=TRAJ_CMAP(TRAJ_RANGE[1]), s=30, zorder=10)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.autoscale_view()
    nashmtl_axes(ax)
    ax.xaxis.set_label_coords(1.12, -0.03)  # clear of the end tick (as plot_2d_pareto)
    ax.yaxis.set_label_coords(-0.01, 1.09)
    return ax, fig


def front_distance(L: np.ndarray, front: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """Distance of every point of ``L`` (``[..., 2]``) to the sampled Pareto front."""
    flat = L.reshape(-1, 2)
    out = np.empty(len(flat))
    for s in range(0, len(flat), chunk):
        out[s:s + chunk] = np.linalg.norm(flat[s:s + chunk, None] - front[None], axis=-1).min(axis=1)
    return out.reshape(L.shape[:-1])


def square_limits(problem, pad: float = 0.15):
    """Square window (equal axes) around the initial points and the Pareto set of a 2-D problem."""
    pts = np.concatenate([np.asarray(problem.inits, dtype=float), problem.pareto_set(200)])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    c, h = (lo + hi) / 2, (hi - lo).max() / 2 * (1 + pad)
    return (c[0] - h, c[0] + h), (c[1] - h, c[1] + h)


def problem_figures(problem, logs: Path, fig_dir: Path, seed: int = 1) -> None:
    """Figures of a quadratic problem of :mod:`problems` (``logs/<problem>/<method>/``): per method, the runs
    (``seed``) on the contour map of the average loss (``contour/<method>.pdf``, 2-D problems) and in the loss plane
    (``loss/<method>.pdf``, two objectives); per family, the distance of the search mean to the Pareto set against the
    iteration (``convergence_<family>.pdf``)."""
    keys = [k for fam in FAMILIES.values() for k in fam if (logs / k / "trajectories.npz").exists()]
    names = {k: json.loads((logs / k / "summary.json").read_text())["name"] for k in keys}
    dist = {}
    fig_dir.mkdir(parents=True, exist_ok=True)
    for sub, needed in (("contour", problem.n_params == 2), ("loss", problem.n_obj == 2)):
        if needed:
            (fig_dir / sub).mkdir(exist_ok=True)
    for k in keys:
        data = np.load(logs / k / "trajectories.npz")
        s = list(data["seeds"]).index(seed)
        if problem.n_params == 2:
            ax, fig = contour_panel(problem, data["x"][:, s], *square_limits(problem), levels="quantile")
            fig.savefig(fig_dir / "contour" / f"{k}.pdf", bbox_inches="tight", dpi=300)
            plt.close(fig)
        if problem.n_obj == 2:
            ax, fig = loss_panel(problem, data["losses"][:, s])
            rasterize_dense(fig)
            fig.savefig(fig_dir / "loss" / f"{k}.pdf", bbox_inches="tight", dpi=300)
            plt.close(fig)
        gens = np.unique(np.linspace(0, data["x"].shape[2] - 1, 2001).astype(int))  # at most 2001 iterations
        dist[k] = problem.set_distance(data["x"][:, :, gens])  # [inits, seeds, len(gens)]
    for fam_name, fam in FAMILIES.items():
        present = [k for k in fam if k in dist]
        if present:
            ax, fig = plot_convergence(present, dist, names, ylabel="Distance to Pareto set", gens=gens)
            ax.set_xlabel("Iteration", size=20)
            rasterize_dense(fig)
            fig.savefig(fig_dir / f"convergence_{fam_name.lower()}.pdf", bbox_inches="tight", dpi=300)
            plt.close(fig)
    print(f"{problem.name}: figures -> {fig_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logs", type=Path, default=HERE / "logs")
    parser.add_argument("--fig-dir", type=Path, default=None,
                        help="default: figures/scale_<scale> (Nash-MTL toy) or figures/<problem>")
    parser.add_argument("--seed", type=int, default=1, help="seed of the Nash-MTL-style and 3-D figures")
    args = parser.parse_args()

    keys = [k for fam in FAMILIES.values() for k in fam if (args.logs / k / "trajectories.npz").exists()]
    if not keys:
        raise SystemExit(f"No logs in {args.logs}")
    problem_cfg = yaml.safe_load((args.logs / keys[0] / "config.yaml").read_text())["problem"]
    if problem_cfg.get("name", "nashmtl") != "nashmtl":  # quadratic problems (problems.py)
        problem = make_problem(problem_cfg["name"])
        problem_figures(problem, args.logs, args.fig_dir or HERE / "figures" / problem.name, args.seed)
        return
    losses = {k: np.load(args.logs / k / "trajectories.npz")["losses"] for k in keys}
    summaries = {k: json.loads((args.logs / k / "summary.json").read_text()) for k in keys}
    names = {k: summaries[k]["name"] for k in keys}
    scales = {yaml.safe_load((args.logs / k / "config.yaml").read_text())["problem"]["scale"] for k in keys}
    if len(scales) != 1:
        raise SystemExit(f"Runs with different problem.scale: {sorted(scales)}")
    scale = scales.pop()
    front = pareto_front(scale)
    star = optimum_losses(scale)
    inits = summaries[keys[0]]["inits"]
    if args.fig_dir is None:
        args.fig_dir = HERE / "figures" / f"scale_{scale:g}"
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    (args.fig_dir / "nashmtl").mkdir(exist_ok=True)
    for k in keys:
        data = np.load(args.logs / k / "trajectories.npz")
        s = list(data["seeds"]).index(args.seed)
        # Nash-MTL records x before each of the n_iter steps: drop the point after the last one
        trajectories = {i: {"traj": data["x"][i, s, :-1].astype(np.float64)} for i in range(len(inits))}
        ax, fig, legend = plot_2d_pareto(trajectories, scale)
        rasterize_dense(fig)
        fig.savefig(args.fig_dir / "nashmtl" / f"{k}.pdf", bbox_extra_artists=(legend,), bbox_inches="tight", dpi=300,
                    facecolor="white")
        plt.close(fig)

    (args.fig_dir / "3d").mkdir(exist_ok=True)
    for k in keys:
        data = np.load(args.logs / k / "trajectories.npz")
        s = list(data["seeds"]).index(args.seed)
        ax, fig = plot_3d(data["x"][:, s].astype(np.float64), scale)
        rasterize_dense(fig)
        fig.savefig(args.fig_dir / "3d" / f"{k}.pdf", bbox_inches="tight", dpi=300)
        plt.close(fig)

    (args.fig_dir / "contour").mkdir(exist_ok=True)
    for k in keys:
        data = np.load(args.logs / k / "trajectories.npz")
        s = list(data["seeds"]).index(args.seed)
        # vector: plot_contour thins the runs, rasterize_dense would rasterize the contour lines
        ax, fig = plot_contour(data["x"][:, s].astype(np.float64), scale)
        fig.savefig(args.fig_dir / "contour" / f"{k}.pdf", bbox_inches="tight", dpi=300)
        plt.close(fig)

    every = np.concatenate([front] + [L.mean(axis=1).reshape(-1, 2) for L in losses.values()])
    lo, hi = every.min(axis=0), every.max(axis=0)
    pad = 0.05 * (hi - lo)
    xlim, ylim = (lo[0] - pad[0], hi[0] + pad[0]), (lo[1] - pad[1], hi[1] + pad[1])

    for k in keys:
        fig, ax = plt.subplots(figsize=(6, 5))
        traj_panel(ax, k, names[k], losses[k], front, xlim, ylim, star)
        rasterize_dense(fig)
        fig.savefig(args.fig_dir / f"traj_{k}.pdf", bbox_inches="tight", dpi=300)
        plt.close(fig)

    n_col = max(len(fam) for fam in FAMILIES.values())
    fig, axes = plt.subplots(len(FAMILIES), n_col, figsize=(6 * n_col, 5.5 * len(FAMILIES)), squeeze=False)
    for r, fam in enumerate(FAMILIES.values()):
        for c, k in enumerate(fam):
            if k in losses:
                traj_panel(axes[r, c], k, names[k], losses[k], front, xlim, ylim, star)
            else:
                axes[r, c].set_visible(False)
    fig.savefig(args.fig_dir / "overview.png", bbox_inches="tight", dpi=120)
    plt.close(fig)

    dist = {k: front_distance(losses[k], front) for k in keys}  # [inits, seeds, T + 1]
    for fam_name, fam in FAMILIES.items():
        present = [k for k in fam if k in dist]
        if not present:
            continue
        ax, fig = plot_convergence(present, dist, names)
        rasterize_dense(fig)
        fig.savefig(args.fig_dir / f"convergence_{fam_name.lower()}.pdf", bbox_inches="tight", dpi=300)
        plt.close(fig)

    # final distance per initial point: mean +- std over seeds, best (lowest mean) per column in bold
    final = {k: dist[k][:, :, -1] for k in keys}
    n_init = len(inits)
    cols = [f"$({x:g}, {y:g})$" for x, y in inits]
    lines = [r"\begin{tabular}{l" + "c" * (n_init + 1) + "}", r"\toprule",
             "Method & " + " & ".join(cols) + r" & Mean \\", r"\midrule"]
    for fam in FAMILIES.values():
        present = [k for k in fam if k in final]
        if not present:
            continue
        cells = {k: [final[k][i] for i in range(n_init)] + [final[k].mean(axis=0)] for k in present}
        best = [min(present, key=lambda k: cells[k][j].mean()) for j in range(n_init + 1)]
        for k in present:
            row = []
            for j, v in enumerate(cells[k]):
                s = f"{v.mean():.3f} $\\pm$ {v.std():.3f}"
                row.append(f"\\textbf{{{s}}}" if best[j] == k else s)
            lines.append(names[k] + " & " + " & ".join(row) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    (args.logs / "summary.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
