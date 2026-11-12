"""Comparison figures of the toy runs of ``main.py`` in one results folder (``results/<problem>/evo/`` or
``results/<problem>/grad/``, one subfolder per method, as ``tests/trajectories/results/`` of TorchJD; ``--suffix
_adam``: the ``*_adam`` runs of the folder), saved in ``<fig-dir>``, default ``figures/`` (``figures_adam/``) in the
results folder. The figures of each run, in ``<method>/figures/``, are made by ``main.py`` (:func:`run_figures`).
Limits and ticks of the search and value plots: ``SEARCH_PLOT_LIMS`` / ``VALUE_PLOT_LIMS`` of ``constants.py``.

Quadratic problems (:func:`problem_figures`):

* ``search_plot.pdf``: the runs (``--seed``) on the contour map of the average loss (2-D problems), one subplot per
  method (a row per family);
* ``value_plot.pdf``: the same runs in the loss plane (two objectives), one subplot per method;
* ``ps_dist_convergence_plot_<family>.pdf``: distance of the search mean to the Pareto set against the iteration,
  per method of the family (OpenES, GuidedES, Grad, LibMTL).

Valley-Bowl problem (``vb``, ``vb01``), each a figure with a subplot per method (a row per family):

* ``value_plot.pdf``: the figure of ``experiments/toy/trainer.py`` of Nash-MTL (``plot_2d_pareto`` of its
  ``utils.py``, reproduced in :func:`plot_2d_pareto`), one run per initial point (``--seed``);
* ``value_plot_seed_mean.pdf``: the Pareto front (grey) and, for each initial point, the seed-mean trajectory of the
  losses (orange early to purple late, the colors of Nash-MTL; start = black circle, end = small dot; black star:
  minimum of the average loss), as in Fig. 1 of Nash-MTL;
* ``search_plot_3d.pdf``: the runs (``--seed``) on the surface of the average loss over ``(theta_1, theta_2)``, as in
  Fig. 1 (bottom row) of Aligned-MTL (:func:`plot_3d`);
* ``search_plot.pdf``: the same runs on the contour map of the average loss (:func:`contour_panel`);
* ``pf_dist_convergence_plot_<family>.pdf``: distance of the search mean to the Pareto front (loss space), mean
  over initial points and seeds, per method of the family;
* ``<results>/summary<suffix>.tex``: final distance to the front per initial point, mean +- std over seeds.

Usage (from the repository root)::

    python examples/toy/utils.py --results examples/toy/results/vb01/evo [--fig-dir ...] [--seed 1]
    python examples/toy/utils.py --results examples/toy/results/cqf/grad
    python examples/toy/utils.py --results examples/toy/results/cqf/evo --suffix _adam
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
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))  # first: the toy modules

import torch  # noqa: E402
import yaml  # noqa: E402

from constants import (  # noqa: E402
    COLORS, FAMILIES, FRONT_COLOR, LOSS_PLANE_SPACING, MARKERS, PROBLEMS, SEARCH_PLOT_LIMS, SUFFIXES, TRAJ_CMAP,
    TRAJ_RANGE, VALUE_PLOT_LIMS,
)
from problems import ValleyBowl  # noqa: E402


def set_lims(ax, lims: dict) -> None:
    """Limits and ticks of an entry of ``SEARCH_PLOT_LIMS`` / ``VALUE_PLOT_LIMS`` (:mod:`constants`)."""
    ax.set_xlim(*lims["xlim"])
    ax.set_ylim(*lims["ylim"])
    if "xticks" in lims:
        ax.set_xticks(lims["xticks"])
    if "yticks" in lims:
        ax.set_yticks(lims["yticks"])


def gradient_line(ax, xy: np.ndarray) -> None:
    seg = np.stack([xy[:-1], xy[1:]], axis=1)
    colors = matplotlib.colormaps[TRAJ_CMAP](np.linspace(*TRAJ_RANGE, len(seg)))
    ax.add_collection(LineCollection(seg, colors=colors, linewidth=3, zorder=5))


def loss_plane_axes(ax) -> None:
    """Axis style of Nash-MTL's plot_2d_pareto: no top / right spines, large labels at the ends of the axes."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel(r"$\mathcal{L}_1$", size=30)
    ax.set_ylabel(r"$\mathcal{L}_2$", size=30)
    ax.xaxis.set_label_coords(1.015, -0.03)
    ax.yaxis.set_label_coords(-0.01, 1.01)
    ax.tick_params(labelsize=20)


def plot_convergence(keys: list, dist: dict, names: dict, ylabel: str = "Distance to Pareto front",
                     gens: dict = None):
    """Distance to the Pareto front against the generation (``dist[k]``: ``[inits, seeds, T + 1]``, or at the
    generations ``gens[k]``): mean over the initial points; over the seeds, in log scale (the axis): line = geometric
    mean, band = one standard deviation of log10(distance). Nash-MTL-sized fonts."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for k in keys:
        log_d = np.log10(np.maximum(dist[k].mean(axis=0), 1e-12))  # [seeds, T + 1]
        mean, std = log_d.mean(axis=0), log_d.std(axis=0, ddof=1 if len(log_d) > 1 else 0)  # one seed: no band
        g = np.arange(len(mean)) if gens is None else gens[k]  # per method: their numbers of iterations may differ
        method = next((k[:-len(x)] for x in SUFFIXES if k.endswith(x)), k)  # without its protocol suffix (_adam)
        p = next((fam.index(method) for fam in FAMILIES.values() if method in fam), 0)  # position: color, marker
        ax.plot(g, 10 ** mean, color=COLORS[p], marker=MARKERS[p], markevery=max(1, len(g) // 8),
                markersize=9, linewidth=2.5, label=names[k])
        ax.fill_between(g, 10 ** (mean - std), 10 ** (mean + std), color=COLORS[p], alpha=0.2, linewidth=0)
    ax.set_yscale("log")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Generation", size=20)
    ax.set_ylabel(ylabel, size=20)
    ax.tick_params(labelsize=16)
    ax.xaxis.get_offset_text().set_fontsize(16)
    ax.legend(fontsize=12, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))  # right of the axes
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


def traj_panel(ax, L: np.ndarray, front: np.ndarray, xlim, ylim, star: np.ndarray) -> None:
    mean = L.mean(axis=1)  # [inits, T + 1, 2]: seed mean
    ax.plot(front[:, 0], front[:, 1], "-", color=FRONT_COLOR, linewidth=8, label="Pareto front", zorder=1)
    ax.scatter(*star, marker="*", s=300, color="k", zorder=11)
    for i in range(mean.shape[0]):
        gradient_line(ax, mean[i])
        ax.scatter(*mean[i, 0], color="k", s=150, zorder=10)
        ax.scatter(*mean[i, -1], color=matplotlib.colormaps[TRAJ_CMAP](TRAJ_RANGE[1]), s=30, zorder=10)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    loss_plane_axes(ax)
    match_ticks(ax)


def plot_2d_pareto(trajectories: dict, problem: ValleyBowl, lims: dict = None, ax=None):
    """``plot_2d_pareto`` of Nash-MTL's ``experiments/toy/utils.py`` (itself adapted from CAGrad), unchanged except:
    ``sns.despine()`` is written out (no seaborn here) and the per-tick ``tick.label.set_fontsize`` (removed from
    matplotlib) is ``ax.tick_params``, and the losses come from :class:`problems.ValleyBowl` (the same function as
    its ``Toy``). ``trajectories``: ``{i: {"traj": [T, 2] x-space trajectory}}``. ``ax`` (added): draw into this
    axis (a subplot of :func:`method_grid`), without the layout and legend of a single figure (``legend`` is None)."""
    single = ax is None
    fig, ax = plt.subplots(figsize=(6, 5)) if single else (ax.figure, ax)

    F = problem

    losses = []
    for res in trajectories.values():
        losses.append(F(torch.from_numpy(res["traj"])))

    yy = problem.PARETO_SET_X2  # -8.3552
    x = np.linspace(-7, 7, 1000)

    inpt = np.stack((x, [yy] * len(x))).T
    Xs = torch.from_numpy(inpt).float()

    Ys = F(Xs)
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

    # (added) minimum of the average loss
    ax.scatter(*problem.losses(problem.global_optimum()), marker="*", s=300, color="k", zorder=11)

    ax.spines[["top", "right"]].set_visible(False)  # sns.despine()
    ax.set_xlabel(r"$\mathcal{L}_1$", size=30)
    ax.set_ylabel(r"$\mathcal{L}_2$", size=30)
    ax.xaxis.set_label_coords(1.12, -0.03)  # (moved out from (1.015, -0.03): clear of the end tick, extend_to_tick)
    ax.yaxis.set_label_coords(-0.01, 1.09)  # (moved out from (-0.01, 1.01))

    ax.tick_params(labelsize=20)
    match_ticks(ax)  # (added)
    extend_to_tick(ax)  # (added)
    if lims is not None:  # (added) fixed limits and ticks, VALUE_PLOT_LIMS of constants.py
        set_lims(ax, lims)

    if not single:  # (added) a subplot: the grid sets the layout and one legend
        return ax, fig, None

    plt.tight_layout()

    legend = ax.legend(
        loc=2, bbox_to_anchor=(-0.15, 1.3), frameon=False, fontsize=20, ncol=2
    )

    return ax, fig, legend


def plot_3d(trajectories: np.ndarray, problem: ValleyBowl, xl: float = 10.0, n: int = 300, ax=None):
    """Trajectories on the surface of the average loss over ``(theta_1, theta_2)``, as in Fig. 1 (bottom row) of
    Aligned-MTL (Senushkin et al., CVPR 2023): surface ``viridis`` (dark = low, yellow = high), Pareto set (grey line), minimum
    of the average loss (star), runs from yellow (start, black dot) to dark red. ``trajectories``: ``[inits, T, 2]``
    in x-space. ``ax``: draw into this 3-D axis (``computed_zorder=False``) instead of a new figure."""
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    if ax is None:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(projection="3d", computed_zorder=False)  # draw in zorder: runs on top of the surface
    else:
        fig = ax.figure

    g = np.linspace(-xl, xl, n)
    T1, T2 = np.meshgrid(g, g)
    Z = problem.losses(np.stack([T1, T2], axis=-1)).mean(axis=-1)
    ax.plot_surface(T1, T2, Z, cmap="viridis", rstride=1, cstride=1, linewidth=0, antialiased=False,
                    rasterized=True, zorder=1)

    ps = problem.pareto_set(500)
    ax.plot(ps[:, 0], ps[:, 1], problem.losses(ps).mean(axis=-1), color="0.5", linewidth=4, zorder=2)
    opt = problem.global_optimum()
    ax.scatter(*opt, problem.losses(opt).mean(), marker="*", s=200, color="k", depthshade=False, zorder=6)

    for traj in trajectories:
        z = problem.losses(traj).mean(axis=-1)
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


def contour_panel(problem, trajectories: np.ndarray, lims: dict, n: int = 200, levels="linear", ax=None):
    """Contour map of the average loss over ``(theta_1, theta_2)`` with the runs, for any 2-D problem of
    :mod:`problems` (the 2-D counterpart of :func:`plot_3d`) over the limits and ticks ``lims`` (an entry of
    ``SEARCH_PLOT_LIMS`` of :mod:`constants`): contour lines on white from ``viridis``, dark purple = low, yellow =
    high; the Pareto set (grey; a grey dot when it is one point); the minimum of the average loss (black star); runs
    from yellow (start, black dot) to dark red (end, small dot). ``trajectories``: ``[inits, T, 2]`` in x-space.
    ``levels``: ``"linear"`` (20 equally spaced) or ``"quantile"`` (at 20 quantiles of the map, for losses that span
    orders of magnitude). ``ax``: draw into this axis (a subplot) instead of a new figure."""
    fig, ax = plt.subplots(figsize=(5, 5)) if ax is None else (ax.figure, ax)
    T1, T2 = np.meshgrid(np.linspace(*lims["xlim"], n), np.linspace(*lims["ylim"], n))
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

    set_lims(ax, lims)
    ax.set_aspect("equal")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel(r"$\theta_1$", size=26, labelpad=0)
    ax.set_ylabel(r"$\theta_2$", size=26, labelpad=0)
    ax.tick_params(labelsize=16)
    return ax, fig


def loss_panel(problem, losses: np.ndarray, lims: dict = None, ax=None):
    """Loss-plane trajectories of a two-objective problem of :mod:`problems` in the style of :func:`plot_2d_pareto`
    (``losses``: ``[inits, T, 2]``, one seed): Pareto front (grey; a grey dot when it is one point), initial points
    (black), every iterate colored from orange (start) to purple (end), end = small dot, losses at the minimum of the
    average loss (black star). A line, unlike the scatter points of Nash-MTL's ``plot_2d_pareto`` (Valley-Bowl).
    Limits and ticks: ``lims`` (``VALUE_PLOT_LIMS`` of :mod:`constants`); fitted to the runs and the front without.
    ``ax``: draw into this axis (a subplot) instead of a new figure."""
    fig, ax = plt.subplots(figsize=(6, 5)) if ax is None else (ax.figure, ax)
    front = problem.pareto_front(1000)
    if np.ptp(front, axis=0).max() < 1e-9:
        ax.scatter(*front[0], color=FRONT_COLOR, s=250, zorder=1)
    else:
        ax.plot(*front.T, "-", linewidth=8, color=FRONT_COLOR, zorder=1)
    ax.scatter(*problem.losses(problem.global_optimum()), marker="*", s=300, color="k", zorder=11)
    for L in losses:
        gradient_line(ax, L)
        ax.scatter(*L[0], color="k", s=150, zorder=10)
        ax.scatter(*L[-1], color=matplotlib.colormaps[TRAJ_CMAP](TRAJ_RANGE[1]), s=30, zorder=10)
    if lims is not None:
        set_lims(ax, lims)
    else:
        ax.autoscale_view()
    loss_plane_axes(ax)
    ax.xaxis.set_label_coords(1.12, -0.03)  # clear of the end tick (as plot_2d_pareto)
    ax.yaxis.set_label_coords(-0.01, 1.09)
    return ax, fig



def method_grid(keys_by_family: dict, panel, width: float, height: float, subplot_kw: dict = None,
                spacing: dict = None, max_cols: int = 8):
    """One figure with a subplot per method: a row per family (``keys_by_family``: family -> present methods), a
    column per method; ``panel(key, ax)`` draws a method into its subplot (``subplot_kw``: e.g. a 3-D projection).
    Unused cells are hidden. ``spacing``: ``wspace`` / ``hspace`` between the panels, applied after the tight layout
    (e.g. ``LOSS_PLANE_SPACING``). A family of more than ``max_cols`` methods wraps onto several rows."""
    rows = [(fam, ks[i:i + max_cols]) for fam, ks in keys_by_family.items() for i in range(0, len(ks), max_cols)]
    n_col = max(len(ks) for _, ks in rows)
    fig, axes = plt.subplots(len(rows), n_col, figsize=(width * n_col, height * len(rows)), squeeze=False,
                             subplot_kw=subplot_kw)
    for r, (fam, ks) in enumerate(rows):
        for c in range(n_col):
            if c < len(ks):
                panel(ks[c], axes[r, c])
            else:
                axes[r, c].set_visible(False)
    fig.tight_layout()
    if spacing:
        fig.subplots_adjust(**spacing)
    return fig


def problem_figures(name: str, logs: Path, keys: list, fig_dir: Path, seed: int = 1, suffix: str = "") -> None:
    """Figures of a quadratic problem of :mod:`problems` comparing the methods ``keys`` of a results folder
    (``results/<problem>/<evo|grad>/<method>/``), saved in ``fig_dir``: ``search_plot.pdf``, the runs
    (``seed``) on the contour map of the average loss (2-D problems), and ``value_plot.pdf``, the same runs in the loss
    plane (two objectives), each one figure with a subplot per method (a row per family); per family, the distance of
    the search mean to the Pareto set against the iteration (``ps_dist_convergence_plot_<family>.pdf``)."""
    problem = PROBLEMS[name]
    summaries = {k: json.loads((logs / k / "summary.json").read_text()) for k in keys}
    names = {k: summaries[k]["name"] for k in keys}
    data = {k: np.load(logs / k / "trajectories.npz") for k in keys}
    by_family = {fam: [k + suffix for k in ks if k + suffix in data] for fam, ks in FAMILIES.items()}
    fig_dir.mkdir(parents=True, exist_ok=True)

    def run_of(k):  # [inits, T + 1, ...] of the run of ``seed``
        return list(data[k]["seeds"]).index(seed)

    if problem.n_params == 2:
        def search(k, ax):
            contour_panel(problem, data[k]["x"][:, run_of(k)], SEARCH_PLOT_LIMS[name], levels="quantile", ax=ax)
            ax.set_title(names[k], size=18)

        fig = method_grid(by_family, search, 5.0, 5.5)
        fig.savefig(fig_dir / "search_plot.pdf", bbox_inches="tight", dpi=300)  # vector: the runs are thinned
        plt.close(fig)
    if problem.n_obj == 2:
        def value(k, ax):
            loss_panel(problem, data[k]["losses"][:, run_of(k)], VALUE_PLOT_LIMS[name], ax=ax)
            ax.set_title(names[k], size=18, pad=28)

        fig = method_grid(by_family, value, 6.0, 5.5, spacing=LOSS_PLANE_SPACING)
        rasterize_dense(fig)
        fig.savefig(fig_dir / "value_plot.pdf", bbox_inches="tight", dpi=300)
        plt.close(fig)
    # at most 2001 iterations per curve, each method over its own number of iterations
    gens = {k: np.unique(np.linspace(0, data[k]["x"].shape[2] - 1, 2001).astype(int)) for k in keys}
    dist = {k: problem.set_distance(data[k]["x"][:, :, gens[k]]) for k in keys}  # [inits, seeds, len(gens[k])]
    for fam_name, fam in FAMILIES.items():
        shown = [k + suffix for k in fam if k + suffix in dist]
        if shown:
            ax, fig = plot_convergence(shown, dist, names, ylabel="Distance to Pareto set", gens=gens)
            ax.set_xlabel("Iteration", size=20)
            rasterize_dense(fig)
            fig.savefig(fig_dir / f"ps_dist_convergence_plot_{fam_name.lower()}.pdf", bbox_inches="tight", dpi=300)
            plt.close(fig)
    print(f"{name}: figures -> {fig_dir}")


def run_figures(run_dir: Path, seed: int = None) -> Path:
    """Figures of one run of ``main.py`` (``run_dir``: its ``trajectories.npz``, ``summary.json``), saved in
    ``<run_dir>/figures/``; the runs of ``seed`` (default: the first) for the trajectory figures, every seed for the
    convergence curve. Called by ``main.py`` after each run. Limits and ticks: ``SEARCH_PLOT_LIMS`` /
    ``VALUE_PLOT_LIMS`` of :mod:`constants`.

    * every problem: ``ps_dist_convergence_plot.pdf``, the distance to the Pareto set against the iteration;
    * 2-D problems: ``search_plot.pdf``, the runs on the contour map of the average loss (:func:`contour_panel`);
    * two objectives: ``value_plot.pdf``, the runs in the loss plane (:func:`loss_panel`; Valley-Bowl: Nash-MTL's
      :func:`plot_2d_pareto`);
    * Valley-Bowl, also: ``search_plot_3d.pdf`` (:func:`plot_3d`) and ``value_plot_seed_mean.pdf`` (seed-mean
      losses, :func:`traj_panel`)."""
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text())
    data = np.load(run_dir / "trajectories.npz")
    name, key, inits = summary["problem"], run_dir.name, summary["inits"]
    problem = PROBLEMS[name]
    names = {key: summary["name"]}
    seeds = list(data["seeds"])
    s = seeds.index(seed) if seed is not None else 0
    X, L = data["x"].astype(np.float32), data["losses"]  # [inits, seeds, T + 1, D], [inits, seeds, T + 1, K]
    out = run_dir / "figures"
    out.mkdir(exist_ok=True)

    def save(fig, name, rasterize=True, **kwargs):
        if rasterize:
            rasterize_dense(fig)
        fig.savefig(out / name, bbox_inches="tight", dpi=300, **kwargs)
        plt.close(fig)

    gens = np.unique(np.linspace(0, X.shape[2] - 1, 2001).astype(int))  # at most 2001 iterations on the curve
    if isinstance(problem, ValleyBowl):
        trajectories = {i: {"traj": X[i, s, :-1]} for i in range(len(inits))}  # Nash-MTL: x before each step
        ax, fig, legend = plot_2d_pareto(trajectories, problem, VALUE_PLOT_LIMS[name])
        save(fig, "value_plot.pdf", bbox_extra_artists=(legend,), facecolor="white")
        ax, fig = plot_3d(X[:, s], problem)
        save(fig, "search_plot_3d.pdf")
        ax, fig = contour_panel(problem, X[:, s], SEARCH_PLOT_LIMS[name])
        save(fig, "search_plot.pdf", rasterize=False)  # vector: contour_panel thins the runs
        fig, ax = plt.subplots(figsize=(6, 5))
        lims = VALUE_PLOT_LIMS[name]
        traj_panel(ax, L, problem.pareto_front(), lims["xlim"], lims["ylim"], problem.losses(problem.global_optimum()))
        set_lims(ax, lims)
        save(fig, "value_plot_seed_mean.pdf")
    else:
        if problem.n_params == 2:
            ax, fig = contour_panel(problem, X[:, s], SEARCH_PLOT_LIMS[name], levels="quantile")
            save(fig, "search_plot.pdf", rasterize=False)
        if problem.n_obj == 2:
            ax, fig = loss_panel(problem, L[:, s], VALUE_PLOT_LIMS[name])
            save(fig, "value_plot.pdf")
    dist = problem.set_distance(X[:, :, gens])  # [inits, seeds, len(gens)]
    ax, fig = plot_convergence([key], {key: dist}, names, ylabel="Distance to Pareto set", gens={key: gens})
    ax.set_xlabel("Iteration", size=20)
    save(fig, "ps_dist_convergence_plot.pdf")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", dest="logs", type=Path, required=True,
                        help="results folder of one problem and kind, e.g. examples/toy/results/cqf/grad")
    parser.add_argument("--fig-dir", type=Path, default=None, help="default: <results>/figures<suffix>")
    parser.add_argument("--suffix", default="", choices=("",) + SUFFIXES,
                        help="plot the runs of this protocol variant (e.g. _adam) instead of the reference ones")
    parser.add_argument("--seed", type=int, default=1, help="seed of the Nash-MTL-style and 3-D figures")
    args = parser.parse_args()
    if args.fig_dir is None:  # next to the runs (param_plots / value_plots of TorchJD)
        args.fig_dir = args.logs / f"figures{args.suffix}"

    keys = [k + args.suffix for fam in FAMILIES.values() for k in fam  # the methods with a run in the folder
            if (args.logs / (k + args.suffix) / "trajectories.npz").exists()]
    if not keys:
        raise SystemExit(f"No runs in {args.logs}")
    names_of_problem = {yaml.safe_load((args.logs / k / "config.yaml").read_text())["problem"]["name"] for k in keys}
    if len(names_of_problem) != 1:
        raise SystemExit(f"Runs of different problems in {args.logs}: {sorted(names_of_problem)}")
    problem_name = names_of_problem.pop()
    problem = PROBLEMS[problem_name]
    if not isinstance(problem, ValleyBowl):  # quadratic problems
        problem_figures(problem_name, args.logs, keys, args.fig_dir, args.seed, args.suffix)
        return
    data = {k: np.load(args.logs / k / "trajectories.npz") for k in keys}
    losses = {k: data[k]["losses"] for k in keys}
    summaries = {k: json.loads((args.logs / k / "summary.json").read_text()) for k in keys}
    names = {k: summaries[k]["name"] for k in keys}
    front = problem.pareto_front()
    star = problem.losses(problem.global_optimum())
    inits = summaries[keys[0]]["inits"]
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    by_family = {fam: [k + args.suffix for k in ks if k + args.suffix in data] for fam, ks in FAMILIES.items()}

    def x_of(k):  # [inits, T + 1, 2] of the run of --seed
        return data[k]["x"][:, list(data[k]["seeds"]).index(args.seed)].astype(np.float32)

    def value(k, ax):  # Nash-MTL's figure; Nash-MTL records x before each of the n_iter steps: drop the last point
        plot_2d_pareto({i: {"traj": t[:-1]} for i, t in enumerate(x_of(k))}, problem, VALUE_PLOT_LIMS[problem_name],
                       ax=ax)
        ax.set_title(names[k], size=18, pad=28)

    fig = method_grid(by_family, value, 6.0, 5.5, spacing=LOSS_PLANE_SPACING)
    first = fig.axes[0]
    legend = fig.legend(*first.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 1.0),
                        frameon=False, fontsize=20, ncol=2)
    rasterize_dense(fig)
    fig.savefig(args.fig_dir / "value_plot.pdf", bbox_extra_artists=(legend,), bbox_inches="tight", dpi=300,
                facecolor="white")
    plt.close(fig)

    def search_3d(k, ax):
        plot_3d(x_of(k), problem, ax=ax)
        ax.set_title(names[k], size=16)

    fig = method_grid(by_family, search_3d, 6.0, 5.0, subplot_kw={"projection": "3d", "computed_zorder": False})
    rasterize_dense(fig)
    fig.savefig(args.fig_dir / "search_plot_3d.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)

    def search(k, ax):
        contour_panel(problem, x_of(k), SEARCH_PLOT_LIMS[problem_name], ax=ax)
        ax.set_title(names[k], size=18)

    fig = method_grid(by_family, search, 5.0, 5.5)
    fig.savefig(args.fig_dir / "search_plot.pdf", bbox_inches="tight", dpi=300)  # vector: the runs are thinned
    plt.close(fig)

    lims = VALUE_PLOT_LIMS[problem_name]

    def seed_mean(k, ax):
        traj_panel(ax, losses[k], front, lims["xlim"], lims["ylim"], star)
        set_lims(ax, lims)
        ax.set_title(names[k], size=18, pad=28)

    fig = method_grid(by_family, seed_mean, 6.0, 5.5, spacing=LOSS_PLANE_SPACING)
    rasterize_dense(fig)
    fig.savefig(args.fig_dir / "value_plot_seed_mean.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)

    dist = {k: problem.front_distance(losses[k]) for k in keys}  # [inits, seeds, T + 1]
    for fam_name, fam in FAMILIES.items():
        shown = [k + args.suffix for k in fam if k + args.suffix in dist]
        if not shown:
            continue
        ax, fig = plot_convergence(shown, dist, names)
        rasterize_dense(fig)
        fig.savefig(args.fig_dir / f"pf_dist_convergence_plot_{fam_name.lower()}.pdf", bbox_inches="tight", dpi=300)
        plt.close(fig)

    # final distance per initial point: mean +- std over seeds, best (lowest mean) per column in bold
    final = {k: dist[k][:, :, -1] for k in keys}
    n_init = len(inits)
    cols = [f"$({x:g}, {y:g})$" for x, y in inits]
    lines = [r"\begin{tabular}{l" + "c" * (n_init + 1) + "}", r"\toprule",
             "Method & " + " & ".join(cols) + r" & Mean \\", r"\midrule"]
    for fam in FAMILIES.values():
        shown = [k + args.suffix for k in fam if k + args.suffix in final]
        if not shown:
            continue
        cells = {k: [final[k][i] for i in range(n_init)] + [final[k].mean(axis=0)] for k in shown}
        best = [min(shown, key=lambda k: cells[k][j].mean()) for j in range(n_init + 1)]
        for k in shown:
            row = []
            for j, v in enumerate(cells[k]):
                s = f"{v.mean():.3f} $\\pm$ {v.std():.3f}"
                row.append(f"\\textbf{{{s}}}" if best[j] == k else s)
            lines.append(names[k] + " & " + " & ".join(row) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    (args.logs / f"summary{args.suffix}.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
