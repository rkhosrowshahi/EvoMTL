"""Range of PSL fronts over generations: does the front keep its range as the knee improves?

For every PSL run (``evo_moea=psl_es``; reads ``front.csv``), at each test evaluation:

* ``knee``: test loss of the balanced network (lam_1 = 0.5), one line per task;
* ``interior worst``: the worst test loss of each task over the interior preferences (lam_1 in [0.1, 0.9]);
* ``end points``: the weaker task of each end network, the top-left loss at lam_1 = 0 and the bottom-right
  loss at lam_1 = 1.

A front that keeps its range has the interior worst losses and the end points falling with the knee; a
widening front has end points rising toward or past chance level (ln 10).

Writes ``front_range.png`` (and ``.pdf``) to ``--out``; comparisons go to ``figures/comparison/<dataset>/<name>/``.

Usage (from examples/multimnist)::

    python scripts/plot_front_range.py --psl logs/mnist/psl_guided_es-tch/pop_128/adam-lr_1e-3/seed_1-* \\
        --out figures/comparison/mnist/psl_front_range-pop_128
"""

import argparse
import glob
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_mo_openes import SKIP_TEST, save  # noqa: E402
from plot_psl_front import load_run, read_front  # noqa: E402

CHANCE = math.log(10)


def range_series(fronts):
    """Per test evaluation: generation, knee (2), interior worst (2) and end-point weaker task (2) losses."""
    out = []
    for g, F in fronts.items():
        if g <= SKIP_TEST:
            continue
        lam, l1, l2 = F[:, 0], F[:, 1], F[:, 2]
        knee = np.argmin(np.abs(lam - 0.5))
        inner = (lam > 0.05) & (lam < 0.95)
        out.append([g, l1[knee], l2[knee], l1[inner].max(), l2[inner].max(),
                    l1[np.argmin(lam)], l2[np.argmax(lam)]])
    return np.array(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--psl", nargs="+", required=True, help="PSL run folders or glob patterns (with front.csv)")
    p.add_argument("--out", required=True, help="output folder, e.g. figures/comparison/<dataset>/<name>")
    args = p.parse_args()

    runs = [load_run(d) for pat in args.psl for d in sorted(glob.glob(pat)) if os.path.exists(os.path.join(d, "front.csv"))]
    if not runs:
        raise SystemExit("no PSL runs with front.csv found")
    t1, t2 = runs[0]["tasks"]
    colors = plt.cm.tab10(np.arange(len(runs)) % 10)
    titles = ["Knee (balanced network)", "Worst interior point ($\\lambda_1 \\in [0.1, 0.9]$)",
              "End points, weaker task"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), layout="constrained")
    for run, c in zip(runs, colors):
        S = range_series(read_front(run["dir"]))
        g = S[:, 0]
        axes[0].plot(g, S[:, 1], "-", color=c, lw=1.6)
        axes[0].plot(g, S[:, 2], "--", color=c, lw=1.6)
        axes[1].plot(g, S[:, 3], "-", color=c, lw=1.6)
        axes[1].plot(g, S[:, 4], "--", color=c, lw=1.6)
        axes[2].plot(g, S[:, 5], "-", color=c, lw=1.6)
        axes[2].plot(g, S[:, 6], "--", color=c, lw=1.6)
        axes[0].plot([], [], "-", color=c, lw=2.5, label=run["name"])
    for ax, title in zip(axes, titles):
        ax.axhline(CHANCE, color="0.5", ls=":", lw=1.2)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Generation")
        ax.grid(True, alpha=0.4)
    axes[0].set_ylabel("Test loss")
    axes[2].text(0.02, 0.98, f"solid: {t1} loss at $\\lambda_1 = 0$\ndashed: {t2} loss at $\\lambda_1 = 1$",
                 transform=axes[2].transAxes, va="top", fontsize=8)
    for ax in axes[:2]:
        ax.text(0.98, 0.98, f"solid: {t1}\ndashed: {t2}", transform=ax.transAxes, ha="right", va="top", fontsize=8)
    axes[2].text(0.98, 0.02, "dotted: chance level (ln 10)", transform=axes[2].transAxes, ha="right", va="bottom",
                 fontsize=8, color="0.4")
    fig.legend(loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=min(len(runs), 2), fontsize=9)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "front_range.png")
    save(fig, path, bbox_inches="tight")
    print("wrote", path)


if __name__ == "__main__":
    main()
