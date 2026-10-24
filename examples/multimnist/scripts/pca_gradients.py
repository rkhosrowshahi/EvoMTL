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
``--mode common`` (PCD-OpenES MGDA runs): at a few snapshots, the per-objective descent directions -g_k, their
mean and MGDA's common direction -G^T alpha in the plane they span (uncentred PCA of the four vectors; exact, since
both combinations lie in span(g_1, g_2)), for the pseudo-gradients (top) and the true gradients (middle), plus the
angle between MGDA and the mean and MGDA's weight on objective 1 over all snapshots (bottom). Arrows keep their
lengths (relative to the longest in the panel), so MGDA's shorter step is visible.

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
    """PCA of the search-mean positions Z [S, D] (the weights theta; theta - theta_0 in older runs, same PCA).

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
    if "Guided" in d.get("method", ""):  # Guided ES runs save their own name (GuidedES, MO-GuidedES-UPGRAD, ...)
        name = d["method"]
    fig.suptitle(rf"Search trajectory of {name} (N={d['pop_size']}, $\sigma$={d['sigma']:g}, Adam, LR={fmt_lr(d['lr'])})")
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes.tolist(), label="Generation", shrink=0.75)
    stem = args.out or os.path.join(os.path.dirname(args.path),
                                    "gradient_trajectory" + ("" if args.from_gen == 10 else f"_from{args.from_gen}"))
    for ext in (".png", ".pdf"):
        fig.savefig(stem + ext, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("wrote", stem + ".png", "and .pdf")


# reference categorical palette, slots 1-5 in fixed order; line style + direct label as secondary encoding
COMMON_STYLE = {"g1": ("#2a78d6", "-"), "g2": ("#eb6834", "-"), "mgda": ("#1baf7a", "-"), "mean": ("#eda100", "--"),
                "single": ("#e87ba4", ":")}


def _min_norm_alpha(G):
    """Two-objective MGDA weights for the rows of G [S, 2, D] (closed form, as in PCD-OpenES)."""
    M = np.einsum("skd,sjd->skj", G, G)
    denom = M[:, 0, 0] + M[:, 1, 1] - 2 * M[:, 0, 1]
    a = np.where(denom > 0, np.clip((M[:, 1, 1] - M[:, 0, 1]) / np.where(denom > 0, denom, 1), 0, 1), 0.5)
    return np.stack([a, 1 - a], axis=1)


def _angle(a, b):
    return np.degrees(np.arccos(np.clip(np.sum(a * b, -1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
                                                             ).clip(min=1e-30), -1.0, 1.0)))


def plot_common(d, gens, args):
    """MGDA's common direction vs the mean of the per-objective gradients (see module docstring)."""
    tasks = d["tasks"]
    pseudo, true = d["pseudo"].double().numpy(), d["true"].double().numpy()
    applied = "alpha" in d  # PCD-OpenES stepped along MGDA; for MO-OpenES runs alpha is what MGDA would have used
    rows = {"pseudo": (pseudo, d["alpha"].double().numpy() if applied else _min_norm_alpha(pseudo)),
            "true": (true, d["alpha_true"].double().numpy() if applied else _min_norm_alpha(true))}
    # OpenES on the sum of losses (one fitness value per member): the direction the run stepped along (ascent)
    single = {"pseudo": d["direction"].double().numpy(), "true": d["true_direction"].double().numpy()} \
        if d.get("aggregator") == "ws" else None
    # PCD-OpenES variants: MGDA, MGDA-Loss (g_k / L_k), MGDA-GN (g_k / |g_k|); the saved gradients are the normalized ones
    algo = d["method"].split("PCD-OpenES-")[-1] if applied else "MGDA"
    gk = {"loss": "$-g_{k}/L_{k}$", "l2": "$-g_{k}/\\|g_{k}\\|$", "loss+": "$-g_{k}/(L_{k}\\|g_{k}\\|)$"}.get(
        d.get("grad_norm", "none"), "$-g_{k}$")
    n_snap = min(args.snapshots, len(gens))
    pick = np.unique(np.linspace(0, len(gens) - 1, n_snap).round().astype(int))
    fig = plt.figure(figsize=(3.1 * len(pick), 10.2))
    gs = fig.add_gridspec(3, 2 * len(pick), height_ratios=[1, 1, 0.8], hspace=0.42, wspace=0.6)
    for r, (kind, (Gs, As)) in enumerate(rows.items()):
        for c, s in enumerate(pick):
            ax = fig.add_subplot(gs[r, 2 * c:2 * c + 2])
            G, a = Gs[s], As[s]
            vecs = {"g1": -G[0], "g2": -G[1], "mean": -G.mean(axis=0), "mgda": -(a @ G)}
            if single is not None:
                vecs["single"] = -single[kind][s]
            V = np.stack(list(vecs.values()))
            # uncentred PCA: the plane span(g1, g2) (the single-value direction lies in it with raw fitness)
            comps = np.linalg.svd(V, full_matrices=False)[2][:2]
            if comps[0] @ vecs["mgda"] < 0:  # orient PC1 along MGDA's direction
                comps[0] *= -1
            if comps[1] @ vecs["g1"] < 0:
                comps[1] *= -1
            scale = np.linalg.norm(V, axis=1).max()
            for name, v in vecs.items():
                xy = comps @ v / scale
                color, ls = COMMON_STYLE[name]
                ax.annotate("", xy=xy, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=color, lw=2,
                                                                      linestyle=ls, shrinkA=0, shrinkB=0))
                if name in ("g1", "g2"):  # mean and MGDA often overlap: the figure legend names them
                    ax.annotate(gk.replace("k", name[1]), xy=xy, xytext=(4, 4), textcoords="offset points",
                                fontsize=8, color="0.2")
            ax.plot(0, 0, "o", color="0.3", markersize=3)
            ax.set_xlim(-1.15, 1.15)
            ax.set_ylim(-1.15, 1.15)
            ax.set_aspect("equal")
            ax.axhline(0, color="0.88", lw=0.6, zorder=0)
            ax.axvline(0, color="0.88", lw=0.6, zorder=0)
            ax.tick_params(labelsize=7)
            ang = _angle(vecs["mgda"], vecs["mean"])
            cos12 = float(G[0] @ G[1] / (np.linalg.norm(G[0]) * np.linalg.norm(G[1])))
            ax.set_title(f"gen {gens[s]} | $\\alpha_1$={a[0]:.2f}\n"
                         f"cos($g_1,g_2$)={cos12:+.2f} | $\\angle$({algo}, mean)={ang:.0f}°", fontsize=8)
            if c == 0:
                ax.set_ylabel(("Pseudo-gradients (ES)" if kind == "pseudo" else "True gradients (backprop)")
                              + "\nPC2", fontsize=9)
    names = {"g1": gk.replace("k", "1") + f" ({tasks[0]})", "g2": gk.replace("k", "2") + f" ({tasks[1]})",
             "mgda": algo + r" $-G^T\alpha$" + ("" if applied else " (not applied)"), "mean": r"mean $-(g_1+g_2)/2$"}
    if single is not None:
        names["single"] = "OpenES single value (applied)"
    handles = [plt.Line2D([], [], color=COMMON_STYLE[k][0], linestyle=COMMON_STYLE[k][1], lw=2) for k in names]
    fig.legend(handles, list(names.values()), loc="upper center", ncol=len(names), frameon=False,
               bbox_to_anchor=(0.5, 0.965))
    # bottom: over all snapshots
    bottom = gs[2, :].subgridspec(1, 2, wspace=0.25)
    ax_ang = fig.add_subplot(bottom[0])
    ax_alpha = fig.add_subplot(bottom[1])
    for kind, (Gs, As), ls in (("pseudo", rows["pseudo"], "-"), ("true", rows["true"], "--")):
        mgda = np.einsum("sk,skd->sd", As, Gs)
        ax_ang.plot(gens, _angle(mgda, Gs.mean(axis=1)), ls, color="#1baf7a", lw=2, label=kind)
        ax_alpha.plot(gens, As[:, 0], ls, color="#2a78d6", lw=2, label=kind)
    ax_ang.set_ylabel(f"angle({algo}, mean) [°]")
    ax_alpha.set_ylabel(f"{algo} weight $\\alpha_1$ ({tasks[0]})")
    ax_alpha.axhline(0.5, color="0.6", lw=0.8, zorder=0)
    for ax in (ax_ang, ax_alpha):
        ax.set_xlabel("Generation")
        ax.grid(True, color="0.92")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"{algo} common direction vs mean ({d['method']} run, N={d['pop_size']}, $\\sigma$={d['sigma']:g}); "
                 f"objective 1 = {tasks[0]}, objective 2 = {tasks[1]}", y=0.995)
    stem = args.out or os.path.join(os.path.dirname(args.path), "gradient_common")
    for ext in (".png", ".pdf"):
        fig.savefig(stem + ext, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("wrote", stem + ".png", "and .pdf")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", help="gradients.pth.tar written by the run")
    p.add_argument("--mode", choices=("trajectory", "directions", "common"), default="trajectory",
                   help="trajectory: path of the search mean with gradient arrows; "
                        "directions: PCA of the gradient directions themselves; "
                        "common: MGDA's direction vs the mean in the plane of the gradients (PCD-OpenES)")
    p.add_argument("--snapshots", type=int, default=6, help="common mode: number of snapshots drawn")
    p.add_argument("--fit", choices=("both", "true"), default="both",
                   help="directions mode: fit the PCA on true and pseudo vectors together, or on true only")
    p.add_argument("--out", default=None, help="output file stem (default: next to the .pth.tar)")
    p.add_argument("--from_gen", type=int, default=10,
                   help="use only snapshots from this generation on (default 10: drops the initialization)")
    args = p.parse_args()

    d = torch.load(args.path, map_location="cpu", weights_only=False)
    keep = d["generation"] >= args.from_gen
    for k in ("generation", "mean", "true", "pseudo", "direction", "true_direction", "alpha", "alpha_true"):
        if k in d:
            d[k] = d[k][keep]
    gens = d["generation"].numpy()
    if args.mode == "common":
        return plot_common(d, gens, args)
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
