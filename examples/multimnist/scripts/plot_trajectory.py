"""Training trajectory in objective space: task-1 test loss vs task-2 test loss.

ES runs are read from ``metrics.csv`` (test loss every ``evo_eval_freq`` generations); gradient-descent
runs from the ``Epoch: ... TEST: loss acc | loss acc`` lines of their ``stdout.log`` (one point per epoch).
Thin lines are single seeds, the bold line is their mean. Markers sit at each full epoch of data
(60,000 / 256 generations for ES, end of each epoch for gradient descent), numbered at epochs 1, 5, 10,
so methods can be compared at the same data budget.

Usage (from examples/multimnist)::

    uv run python scripts/plot_trajectory.py
    uv run python scripts/plot_trajectory.py --gd upgrad mgda ew
"""

import argparse
import csv
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXAMPLE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(EXAMPLE_DIR, "logs/mnist/10epochs")
ES = [("guided_es-ws-full", "Guided-ES (WS)"), ("mo_guided_es-upgrad-full", "MO-Guided-ES (UPGrad)")]
GD_NAMES = {"upgrad": "UPGrad (backprop)", "mgda": "MGDA (backprop)", "ew": "EW (backprop)"}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # categorical slots, fixed order
GENS_PER_EPOCH = 60000 / 256
EPOCH_RE = re.compile(r"^Epoch: (\d+) \|.*TEST: ([\d.]+) [\d.]+ \| ([\d.]+) [\d.]+", re.M)


def es_path(run_dir):
    rows = [r for r in csv.DictReader(open(os.path.join(run_dir, "metrics.csv"))) if r["test_loss_top-left"]]
    g = np.array([int(r["generation"]) for r in rows], float)
    xy = np.array([[float(r["test_loss_top-left"]), float(r["test_loss_bottom-right"])] for r in rows])
    epochs = np.arange(1, int(g[-1] / GENS_PER_EPOCH + 1e-6) + 1) * GENS_PER_EPOCH
    marks = np.stack([np.interp(epochs, g, xy[:, 0]), np.interp(epochs, g, xy[:, 1])], 1)
    return xy, marks


def gd_path(run_dir):
    pts = EPOCH_RE.findall(open(os.path.join(run_dir, "stdout.log"), encoding="utf-8", errors="replace").read())
    xy = np.array([[float(a), float(b)] for _, a, b in pts])
    return xy, xy


def draw(ax, paths, name, color, label_epochs):
    n = min(len(p) for p, _ in paths)
    for p, _ in paths:
        ax.plot(p[:n, 0], p[:n, 1], color=color, linewidth=0.8, alpha=0.35)
    mean = np.mean([p[:n] for p, _ in paths], 0)
    m = min(len(k) for _, k in paths)
    marks = np.mean([k[:m] for _, k in paths], 0)
    ax.plot(mean[:, 0], mean[:, 1], color=color, linewidth=2, label=f"{name} ({len(paths)} seeds)")
    ax.plot(marks[:, 0], marks[:, 1], "o", color=color, markersize=6, markeredgecolor="white", markeredgewidth=1.2)
    for e in (1, 5, 10) if label_epochs else ():
        if e <= m:
            ax.annotate(str(e), marks[e - 1], xytext=(6, -10), textcoords="offset points", fontsize=8, color="#55544e")
    return mean


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gd", nargs="*", default=["upgrad", "mgda"], help="gradient-descent methods to overlay")
    p.add_argument("--out", default=os.path.join(EXAMPLE_DIR, "figures/10epochs/trajectory"))
    args = p.parse_args()

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(6.8, 6), layout="constrained")
    start = None
    i = 0
    for method, name in ES:
        runs = sorted(glob.glob(os.path.join(LOGS, method, "pop_*", "*", "seed_*", "metrics.csv")))
        if not runs:
            continue
        mean = draw(ax, [es_path(os.path.dirname(r)) for r in runs], name, COLORS[i], start is None)
        start = mean[0] if start is None else start
        i += 1
    labelled = False
    for method in args.gd:
        runs = sorted(glob.glob(os.path.join(LOGS, method, "*", "seed_*", "stdout.log")))
        if runs:
            draw(ax, [gd_path(os.path.dirname(r)) for r in runs], GD_NAMES.get(method, method), COLORS[i], not labelled)
            labelled = True
            i += 1
    if start is not None:
        ax.plot(*start, marker="s", color="#55544e", markersize=7, linestyle="none", label="ES start (generation 1)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ticks = [0.1, 0.2, 0.3, 0.5, 1, 2]
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
        axis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        axis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.grid(color="#e4e3dd", linewidth=0.8)
    ax.set_xlabel("Task 1 (top-left) test loss")
    ax.set_ylabel("Task 2 (bottom-right) test loss")
    ax.set_title("Trajectory in objective space, 10 epochs of data\n"
                 "dots = end of each epoch (numbered at 1, 5, 10); thin lines = single seeds",
                 loc="left", fontsize=11)
    ax.legend(frameon=False, loc="upper left")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=200)
    print(f"wrote {args.out}.png / .pdf")


if __name__ == "__main__":
    main()
