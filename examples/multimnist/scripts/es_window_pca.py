"""PCA trajectory of the ES mean with true and pseudo-gradient arrows, from es_window_cosine.py's vectors.npz.

* left: the mean's path projected on PC1-PC2 of all generations' positions (colour = generation), with arrows of
  the true descent direction -g and the ES estimate -p every ``--arrow_every`` generations, projected on the
  same plane. Unit vectors, one common scale for both kinds, so their lengths compare how much of each lies
  in the plane.
* middle: that in-plane fraction ||proj(v)|| for -g, -p and the applied step over training, with the
  sqrt(2 / d) a random direction would have.
* right: explained variance of PC1-PC5 next to a random walk's (6 / (pi k)^2 ... for large T; computed
  here by simulating a walk of the same length), since PCA of a random walk also gives smooth curves.

Usage (from examples/multimnist)::

    uv run python scripts/es_window_pca.py logs/mnist/window_cos/guided_es-ws/pop_128/seed_1/vectors.npz
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


def unit_rows(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-30)


def pca(X, k=5):
    """Top-k principal axes [D, k] and explained-variance ratios of the rows of X (Gram-matrix trick, n << D)."""
    Xc = X - X.mean(0, keepdims=True)
    evals, evecs = np.linalg.eigh(Xc @ Xc.T)
    order = np.argsort(evals)[::-1]
    evals, evecs = np.maximum(evals[order], 0), evecs[:, order]
    axes = (Xc.T @ evecs[:, :k]) / np.sqrt(np.maximum(evals[:k], 1e-30))
    return axes, Xc @ axes, evals[:k] / evals.sum()


def random_walk_variance(n, k=5, dims=2000, seed=0):
    rng = np.random.default_rng(seed)
    walk = np.cumsum(rng.standard_normal((n, dims)).astype(np.float32), axis=0)
    return pca(walk, k)[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("vectors", help="vectors.npz written by es_window_cosine.py --save_vectors")
    ap.add_argument("--arrow_every", type=int, default=100, help="one arrow pair every this many generations")
    ap.add_argument("--out", default=None, help="output stem (default: pca_trajectory next to the .npz)")
    args = ap.parse_args()

    z = np.load(args.vectors, allow_pickle=True)
    p, g, m, meta = z["p"], z["g"], z["m"].astype(np.float32), z["meta"].item()
    n, d = p.shape
    axes, path, ratio = pca(m)
    P2 = axes[:, :2]
    pu, gu = unit_rows(-p.astype(np.float32)), unit_rows(-g.astype(np.float32))
    proj = {"-g": gu @ P2, "-p": pu @ P2,
            "step": unit_rows(m[1:] - m[:-1]) @ P2}  # applied step m_{t+1} - m_t (already a descent move)
    rw = random_walk_variance(len(m))

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(17, 5.6), layout="constrained",
                                     gridspec_kw={"width_ratios": [1.25, 1, 0.8]})

    # path coloured by generation
    gens = np.arange(len(m))
    segs = np.stack([path[:-1, :2], path[1:, :2]], axis=1)
    lc = LineCollection(segs, cmap="viridis", linewidths=1.6)
    lc.set_array(gens[1:])
    a0.add_collection(lc)
    a0.plot(*path[0, :2], "s", color="#55544e", markersize=7, label="start")
    a0.plot(*path[-1, :2], "*", color="#55544e", markersize=11, label="end")
    idx = np.arange(0, n, args.arrow_every)
    extent = np.ptp(path[:, :2], axis=0).max()
    biggest = max(np.linalg.norm(proj["-g"][idx], axis=1).max(), np.linalg.norm(proj["-p"][idx], axis=1).max())
    scale = 0.12 * extent / biggest
    for name, color in (("-g", "#eb6834"), ("-p", "#2a78d6")):
        v = proj[name][idx] * scale
        a0.quiver(path[idx, 0], path[idx, 1], v[:, 0], v[:, 1], color=color, angles="xy", scale_units="xy",
                  scale=1, width=0.004, label={"-g": "true descent direction −g", "-p": "ES estimate −p"}[name])
    a0.set_aspect("equal", adjustable="datalim")
    a0.autoscale_view()
    a0.set_xlabel(f"PC1 ({ratio[0]:.0%} of variance)")
    a0.set_ylabel(f"PC2 ({ratio[1]:.0%} of variance)")
    a0.set_title(f"Path of the ES mean, arrows every {args.arrow_every} generations\n"
                 "(unit vectors projected on the plane, one common scale)", loc="left", fontsize=11)
    fig.colorbar(lc, ax=a0, label="generation", shrink=0.8)
    a0.legend(frameon=False, fontsize=8, loc="best")

    # in-plane fraction over training
    w = 25
    smooth = lambda x: np.convolve(x, np.ones(w) / w, mode="valid")
    t = np.arange(n)[w - 1:]
    for name, color, label in (("step", "#1baf7a", "applied step"), ("-p", "#2a78d6", "ES estimate −p"),
                               ("-g", "#eb6834", "true descent −g")):
        a1.plot(t, smooth(np.linalg.norm(proj[name], axis=1)), color=color, linewidth=2, label=label)
    a1.axhline(np.sqrt(2 / d), color="#6b6a64", linestyle="--", linewidth=1.2,
               label=f"random direction √(2/d) = {np.sqrt(2 / d):.3f}")
    a1.set_yscale("log")
    a1.set_xlabel("Generation")
    a1.set_ylabel("‖projection on PC1–PC2‖ (unit vector)")
    a1.set_title(f"How much of each direction lies in the plane\n({w}-generation moving average)",
                 loc="left", fontsize=11)
    a1.grid(color="#e4e3dd", linewidth=0.8)
    a1.legend(frameon=False, fontsize=8)

    # explained variance vs random walk
    k = np.arange(1, 6)
    a2.bar(k - 0.18, ratio, width=0.34, color="#2a78d6", label="ES mean path")
    a2.bar(k + 0.18, rw, width=0.34, color="#c3c2b7", label="random walk, same length")
    for x, v in zip(k, ratio):
        a2.annotate(f"{v:.0%}", (x - 0.18, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    a2.set_xticks(k, [f"PC{i}" for i in k])
    a2.set_ylabel("explained variance")
    a2.set_title("Is the path more than a random walk?", loc="left", fontsize=11)
    a2.legend(frameon=False, fontsize=8)

    fig.suptitle(f"PCA of the ES trajectory: aggregator={meta.get('aggregator')}, N={meta.get('pop_size')}, "
                 f"d={d:,}, {n} generations", x=0.01, ha="left", fontsize=12)
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.vectors)), "pca_trajectory")
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=200)
    print(f"explained variance PC1-5: {np.round(ratio, 3)} | random walk: {np.round(rw, 3)}")
    for name in ("-g", "-p", "step"):
        q = np.linalg.norm(proj[name], axis=1)
        print(f"in-plane fraction {name:>4}: mean {q.mean():.4f}, first 10% {q[:n // 10].mean():.4f}, "
              f"last 50% {q[n // 2:].mean():.4f}")
    print(f"wrote {out}.png / .pdf")


if __name__ == "__main__":
    main()
