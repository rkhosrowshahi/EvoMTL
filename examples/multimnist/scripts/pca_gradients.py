"""PCA trajectories of true vs pseudo-gradients from a MO-OpenES ``gradients.pth.tar``.

The run saves, every ``evo_es_grad_save_freq`` generations from the initialization, the backprop
gradient and the ES pseudo-gradient of each objective, plus the aggregated ES direction and the
same aggregation applied to the true gradients. True and pseudo-gradients live on different scales
(loss units vs centered ranks), so every vector is scaled to unit length first: the PCA compares
directions.

Default ``--mode trajectory``: PCA of the search mean's positions gives the 2-D path of the model;
at every snapshot arrows show the descent direction (-g) of the true gradient and of the
pseudo-gradient, projected onto that plane (one panel per objective plus the aggregated direction).
``--mode directions``: PCA of the gradient directions themselves, fit on all snapshots of both
vectors (``--fit both``) or on the true gradients only (``--fit true``).

Usage (from examples/multimnist)::

    python scripts/pca_gradients.py logs/mnist/<run>/gradients.pth.tar
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, LogNorm, Normalize


def fmt_lr(lr):
    """0.001 -> '1e-3'."""
    mant, exp = f"{lr:.0e}".split("e")
    return f"{mant}e{int(exp)}"


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-30)


def pca_2d(fit_rows, *project):
    """Fit PCA on ``fit_rows`` [M, D]; return projections of each array onto PC1-PC2 and explained variance."""
    mu = fit_rows.mean(axis=0, keepdims=True)
    X = fit_rows - mu
    # M << D: eigen-decompose the M x M Gram matrix instead of the D x D covariance
    evals, evecs = np.linalg.eigh(X @ X.T)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    comps = (X.T @ evecs[:, :2]) / np.sqrt(np.maximum(evals[:2], 1e-30))  # [D, 2] unit principal axes
    explained = evals[:2] / max(evals.sum(), 1e-30)
    return [(a - mu) @ comps for a in project], explained


def path(ax, pts, gens, cmap, norm, marker, linestyle, label):
    colors = cmap(norm(gens))
    ax.add_collection(LineCollection(np.stack([pts[:-1], pts[1:]], axis=1), colors=colors[1:],
                                     linewidths=1.4, linestyles=linestyle))
    ax.scatter(pts[:, 0], pts[:, 1], c=colors, marker=marker, s=26, edgecolors="none", zorder=3)
    ax.plot(*pts[0], marker, color=colors[0], markersize=11, markeredgecolor="k", markeredgewidth=0.8, zorder=4)
    ax.plot([], [], marker=marker, linestyle=linestyle, color="gray", label=label)


def path_levels(a, b, L, path2, n_path=10, n_above=4):
    """Contour levels concentrated where the trajectory is.

    ``n_path`` evenly spaced levels across the loss range the path covers on this plane, plus
    ``n_above`` log-spaced levels from there up to the grid maximum for the surrounding landscape.
    """
    from scipy.interpolate import RegularGridInterpolator

    on_path = RegularGridInterpolator((b, a), L)(path2[:, ::-1])
    lo, hi = float(on_path.min()), float(on_path.max())
    pad = 0.05 * (hi - lo)
    inner = np.linspace(max(lo - pad, float(L.min())), hi + pad, n_path)
    outer = np.geomspace(inner[-1], float(L.max()), n_above + 1)[1:] if L.max() > inner[-1] else []
    return np.unique(np.concatenate([inner, outer]))


def path_pca(Z):
    """PCA of the search-mean positions Z [S, D] (z = theta - theta_0).

    Returns the centre mu [1, D], unit principal axes comps [D, 2], the 2-D path [S, 2] and the
    explained-variance ratios of PC1/PC2. Shared with loss_landscape.py so the grid uses the same axes.
    """
    mu = Z.mean(axis=0, keepdims=True)
    X = Z - mu
    evals, evecs = np.linalg.eigh(X @ X.T)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    comps = (X.T @ evecs[:, :2]) / np.sqrt(np.maximum(evals[:2], 1e-30))
    return mu, comps, X @ comps, evals[:2] / max(evals.sum(), 1e-30)


def plot_trajectory(d, gens, panels, cmap, norm, args):
    """Path of the search mean (PCA of its positions) with descent arrows from both gradient estimates."""
    if "mean" not in d:
        raise SystemExit("this gradients.pth.tar has no 'mean' (saved by an older run); use --mode directions")
    mu, comps, path2, explained = path_pca(d["mean"].double().numpy())
    span = float(np.ptp(path2, axis=0).max())
    land_path = os.path.join(os.path.dirname(args.path), "landscape.pt" if args.from_gen == 10 else f"landscape_from{args.from_gen}.pt")
    land = torch.load(land_path, weights_only=False) if os.path.exists(land_path) else None

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6), gridspec_kw={"wspace": 0.28})
    print(f"{d['method']}  N={d['pop_size']}  LR={d['lr']:g}  D={d['dims']}  snapshots={len(gens)}  "
          f"path PC1+PC2 = {100 * explained.sum():.1f}% of the movement")
    for k, (ax, (title, T, P)) in enumerate(zip(axes, panels)):
        if land is not None:  # training loss on the same plane: objective 1, objective 2, total
            L = land["loss"][..., k] if k < 2 else land["loss"].sum(axis=-1)
            levels = path_levels(land["a"], land["b"], L, path2)
            ax.contourf(land["a"], land["b"], L, levels=levels, cmap="Greys", norm=LogNorm(), alpha=0.45, zorder=0)
            cs = ax.contour(land["a"], land["b"], L, levels=levels, colors="0.35", linewidths=0.6, zorder=1)
            ax.clabel(cs, cs.levels[::2], fmt="%.3g", fontsize=7)
        # descent directions (-g), unit length in D dims, projected onto the path's plane
        t2, p2 = -unit(T) @ comps, -unit(P) @ comps
        tn, pn = np.linalg.norm(t2, axis=1), np.linalg.norm(p2, axis=1)  # share of each direction in the plane
        # arrows show only the in-plane direction (fixed display length); the share goes in the legend
        t_arrow, p_arrow = unit(t2) * 0.12 * span, unit(p2) * 0.12 * span
        colors = cmap(norm(gens))
        ax.add_collection(LineCollection(np.stack([path2[:-1], path2[1:]], axis=1), colors=colors[1:], linewidths=2))
        ax.scatter(path2[:, 0], path2[:, 1], c=colors, s=5, zorder=3)
        ax.plot(*path2[0], "o", color=colors[0], markersize=6, markeredgecolor="k", zorder=4,
                label="Initialization" if gens[0] == 0 else "Start")
        kw = dict(angles="xy", scale_units="xy", scale=1, width=0.004, zorder=5, alpha=0.45)
        ax.quiver(path2[:, 0], path2[:, 1], t_arrow[:, 0], t_arrow[:, 1], color="tab:green",
                  label="True gradient", **kw)
        if land is None:
            ax.autoscale()
            ax.margins(0.12)
        else:
            ax.set_xlim(land["a"][0], land["a"][-1])
            ax.set_ylim(land["b"][0], land["b"][-1])
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_box_aspect(1)  # square panels
        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        print(f"  {title:38s} share of unit direction in path plane: true {tn.mean():.3f}, pseudo {pn.mean():.3f}")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.45, 0.93))
    name = {"upgrad": "MO-OpenES-UPGrad", "mean": "MO-OpenES-WS", "ws": "OpenES"}[d["aggregator"]]
    fig.suptitle(rf"Search trajectory of {name} (N={d['pop_size']}, $\sigma$={d['sigma']:g}, Adam, LR={fmt_lr(d['lr'])})")
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes.tolist(), label="Generation", shrink=0.75)
    stem = args.out or os.path.join(os.path.dirname(args.path),
                                    "gradient_trajectory" + ("" if args.from_gen == 10 else f"_from{args.from_gen}"))
    for ext in (".png", ".pdf"):
        fig.savefig(stem + ext, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("wrote", stem + ".png", "and .pdf")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", help="gradients.pth.tar written by the run")
    p.add_argument("--mode", choices=("trajectory", "directions"), default="trajectory",
                   help="trajectory: path of the search mean with gradient arrows; "
                        "directions: PCA of the gradient directions themselves")
    p.add_argument("--fit", choices=("both", "true"), default="both",
                   help="directions mode: fit the PCA on true and pseudo vectors together, or on true only")
    p.add_argument("--out", default=None, help="output file stem (default: next to the .pth.tar)")
    p.add_argument("--from_gen", type=int, default=10,
                   help="use only snapshots from this generation on (default 10: drops the initialization)")
    args = p.parse_args()

    d = torch.load(args.path, map_location="cpu", weights_only=False)
    keep = d["generation"] >= args.from_gen
    for k in ("generation", "mean", "true", "pseudo", "direction", "true_direction"):
        if k in d:
            d[k] = d[k][keep]
    gens = d["generation"].numpy()
    true, pseudo = d["true"].double().numpy(), d["pseudo"].double().numpy()      # [S, K, D]
    tdir, pdir = d["true_direction"].double().numpy(), d["direction"].double().numpy()  # [S, D]
    tasks = d["tasks"]
    agg_name = {"upgrad": "UPGrad", "mean": "mean", "ws": "sum of losses"}[d["aggregator"]]
    panels = [(f"Objective 1: {tasks[0]}", true[:, 0], pseudo[:, 0]),
              (f"Objective 2: {tasks[1]}", true[:, 1], pseudo[:, 1]),
              (f"Aggregated ({agg_name}), total loss", tdir, pdir)]

    cmap = ListedColormap(plt.cm.inferno(np.linspace(0.0, 0.85, 256)))
    norm = Normalize(0, gens.max())
    if args.mode == "trajectory":
        return plot_trajectory(d, gens, panels, cmap, norm, args)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), gridspec_kw={"wspace": 0.32})
    print(f"{d['method']}  N={d['pop_size']}  LR={d['lr']:g}  D={d['dims']}  snapshots={len(gens)} "
          f"(generations {gens.min()}..{gens.max()})")
    for ax, (title, T, P) in zip(axes, panels):
        T, P = unit(T), unit(P)
        fit = np.vstack([T, P]) if args.fit == "both" else T
        (t2, p2), ev = pca_2d(fit, T, P)
        path(ax, t2, gens, cmap, norm, "o", "solid", "true gradient")
        path(ax, p2, gens, cmap, norm, "^", "dashed", "pseudo-gradient")
        ax.autoscale()
        ax.set_title(title)
        ax.set_xlabel(f"PC1 ({100 * ev[0]:.1f}% of variance)")
        ax.set_ylabel(f"PC2 ({100 * ev[1]:.1f}% of variance)")
        ax.grid(True)
        cos = np.sum(T * P, axis=1)
        print(f"  {title:38s} PC1+PC2 explain {100 * ev.sum():5.1f}% | cos(pseudo, true) mean {cos.mean():.4f} "
              f"| cos(true_s, true_0) last {float(T[-1] @ T[0]):.3f}")
    axes[0].legend(loc="best")
    fig.suptitle(f"PCA of gradient directions ({d['method']}, N={d['pop_size']}, fit on "
                 f"{'true and pseudo' if args.fit == 'both' else 'true'} gradients; big marker = initialization)")
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes.tolist(), label="Generation")
    stem = args.out or os.path.join(os.path.dirname(args.path), f"gradient_pca_fit_{args.fit}")
    for ext in (".png", ".pdf"):
        fig.savefig(stem + ext, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("wrote", stem + ".png", "and .pdf")


if __name__ == "__main__":
    main()
