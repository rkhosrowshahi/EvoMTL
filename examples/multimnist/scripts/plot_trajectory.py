"""Training trajectory in objective space: task-1 test loss vs task-2 test loss.

Local runs: ES runs are read from ``metrics.csv`` (test loss every ``evo_eval_freq`` generations); gradient-descent
runs from the ``Epoch: ... TEST: loss acc | loss acc`` lines of their ``stdout.log`` (one point per epoch).
With ``--wandb ENTITY/PROJECT`` the runs come from W&B instead: the finished runs of the groups in ``WANDB_ES``
(the OpenES-sampler aggregators of the 10-epoch protocol) and ``WANDB_GD``.
Thin lines are single seeds, the bold line is their mean. Markers sit at each full epoch of data
(60,000 / 256 generations for ES, end of each epoch for gradient descent), numbered at epochs 1, 5, 10,
so methods can be compared at the same data budget. ``--zoom E`` adds a panel of the ES trajectories from
epoch E on, where the methods separate.

Usage (from examples/multimnist)::

    uv run python scripts/plot_trajectory.py
    uv run python scripts/plot_trajectory.py --gd upgrad mgda ew
    uv run python scripts/plot_trajectory.py --wandb rasa_research/evomtl-multimnist --zoom 5 \\
        --out figures/comparison/mnist/trajectory-openes_aggregators-pop_128/trajectory
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
ES = [("guidedes", "GuidedES"), ("mo_upgrad_guidedes", "MO-UPGrad-GuidedES"),
      ("guided_es-ws", "GuidedES"), ("mo_guided_es-upgrad", "MO-UPGrad-GuidedES")]  # last two: folders before the rename
GD_NAMES = {"upgrad": "UPGrad (backprop)", "mgda": "MGDA (backprop)", "ew": "EW (backprop)"}
# W&B group prefixes (group = f"{prefix}-10epochs-pop_128-adam-lr_0.001"); openes-ws and pcd_openes-mgda_loss are
# the groups' names before the renames to OpenES and MO-MGDA-Loss-OpenES
WANDB_ES = [("mo_os_nmf_pga_openes", "MO-OS-NMF-PGA-OpenES"), ("openes-ws", "OpenES"),
            ("mo_mean_openes", "MO-Mean-OpenES"), ("mo_upgrad_openes", "MO-UPGrad-OpenES"),
            ("mo_mgda_openes_libmtl", "MO-MGDA-OpenES"), ("pcd_openes-mgda_loss", "MO-MGDA-Loss-OpenES")]
WANDB_GD = [("ew-10epochs", "EW (backprop)")]
WANDB_KEYS = ["test/epoch", "test/top-left/loss", "test/bottom-right/loss"]
# categorical slots, fixed order
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
GENS_PER_EPOCH = 60000 / 256
EPOCH_RE = re.compile(r"^Epoch: (\d+) \|.*TEST: ([\d.]+) [\d.]+ \| ([\d.]+) [\d.]+", re.M)


def es_marks(g, xy):
    """The ES path ``xy`` (test losses at generations ``g``) and its points at each full epoch of data."""
    epochs = np.arange(1, int(g[-1] / GENS_PER_EPOCH + 1e-6) + 1) * GENS_PER_EPOCH
    return xy, np.stack([np.interp(epochs, g, xy[:, 0]), np.interp(epochs, g, xy[:, 1])], 1)


def es_path(run_dir):
    rows = [r for r in csv.DictReader(open(os.path.join(run_dir, "metrics.csv"))) if r["test_loss_top-left"]]
    g = np.array([int(r["generation"]) for r in rows], float)
    return es_marks(g, np.array([[float(r["test_loss_top-left"]), float(r["test_loss_bottom-right"])] for r in rows]))


def gd_path(run_dir):
    pts = EPOCH_RE.findall(open(os.path.join(run_dir, "stdout.log"), encoding="utf-8", errors="replace").read())
    xy = np.array([[float(a), float(b)] for _, a, b in pts])
    return xy, xy


def wandb_paths(api, project, group, es):
    """Paths of the finished runs of a W&B group: ``test/epoch`` is the generation for ES runs and the epoch for
    gradient-descent runs."""
    paths = []
    for r in sorted(api.runs(project, filters={"group": group, "state": "finished"}), key=lambda r: r.name):
        h = [x for x in r.scan_history(keys=WANDB_KEYS) if x["test/top-left/loss"] is not None]
        xy = np.array([[x["test/top-left/loss"], x["test/bottom-right/loss"]] for x in h])
        paths.append(es_marks(np.array([x["test/epoch"] for x in h], float), xy) if es else (xy, xy))
    return paths


def draw(ax, paths, name, color, label_epochs):
    n = min(len(p) for p, _ in paths)
    for p, _ in paths:
        ax.plot(p[:n, 0], p[:n, 1], color=color, linewidth=0.8, alpha=0.35)
    mean = np.mean([p[:n] for p, _ in paths], 0)
    m = min(len(k) for _, k in paths)
    marks = np.mean([k[:m] for _, k in paths], 0)
    ax.plot(mean[:, 0], mean[:, 1], color=color, linewidth=2, label=f"{name} ({len(paths)} seed{'s' * (len(paths) > 1)})")
    ax.plot(marks[:, 0], marks[:, 1], "o", color=color, markersize=6, markeredgecolor="white", markeredgewidth=1.2)
    for e in label_epochs:
        if e <= m:
            ax.annotate(str(e), marks[e - 1], xytext=(6, -10), textcoords="offset points", fontsize=8, color="#55544e")
    return mean


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gd", nargs="*", default=["upgrad", "mgda"], help="gradient-descent methods to overlay")
    p.add_argument("--wandb", metavar="ENTITY/PROJECT", help="read the WANDB_ES / WANDB_GD runs from this W&B project")
    p.add_argument("--zoom", type=int, default=0, help="add a panel of the ES trajectories from this epoch on (0: none)")
    p.add_argument("--out", default=os.path.join(EXAMPLE_DIR, "figures/10epochs/trajectory"))
    args = p.parse_args()

    if args.wandb:
        import wandb
        api = wandb.Api(timeout=60)
        methods = [(name, wandb_paths(api, args.wandb, f"{key}-10epochs-pop_128-adam-lr_0.001", True), True)
                   for key, name in WANDB_ES]
        methods += [(name, wandb_paths(api, args.wandb, group, False), False) for group, name in WANDB_GD]
    else:
        methods = [(name, [es_path(os.path.dirname(r)) for r in sorted(glob.glob(
            os.path.join(LOGS, m, "pop_*", "*", "seed_*", "metrics.csv")))], True) for m, name in ES]
        methods += [(GD_NAMES.get(m, m), [gd_path(os.path.dirname(r)) for r in sorted(glob.glob(
            os.path.join(LOGS, m, "*", "seed_*", "stdout.log")))], False) for m in args.gd]
    methods = [m for m in methods if m[1]]

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2 if args.zoom else 1, figsize=(13 if args.zoom else 6.8, 6), layout="constrained",
                             squeeze=False)
    ax = axes[0, 0]
    start, labelled = None, set()
    for i, (name, paths, es) in enumerate(methods):
        mean = draw(ax, paths, name, COLORS[i], (1, 5, 10) if es not in labelled else ())
        labelled.add(es)
        if es and start is None:
            start = mean[0]
        if es and args.zoom:
            draw(axes[0, 1], paths, name, COLORS[i], range(args.zoom, 11) if i == 0 else ())
    if start is not None:
        ax.plot(*start, marker="s", color="#55544e", markersize=7, linestyle="none", label="ES start (generation 1)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ticks = [0.1, 0.2, 0.3, 0.5, 1, 2]
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
        axis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        axis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_title("Trajectory in objective space, 10 epochs of data\n"
                 "dots = end of each epoch (numbered at 1, 5, 10); thin lines = single seeds",
                 loc="left", fontsize=11)
    ax.legend(frameon=False, loc="upper left")
    if args.zoom:  # limits from every seed's epoch points from the zoom epoch on
        pts = np.concatenate([k[args.zoom - 1:] for _, paths, es in methods if es for _, k in paths])
        lo, hi = pts.min(0), pts.max(0)
        lo, hi = lo - 0.08 * (hi - lo), hi + 0.08 * (hi - lo)
        zax = axes[0, 1]
        zax.set_xlim(lo[0], hi[0])
        zax.set_ylim(lo[1], hi[1])
        ax.add_patch(matplotlib.patches.Rectangle(lo, *(hi - lo), fill=False, edgecolor="#55544e", linewidth=0.8))
        zax.set_title(f"Zoom on the boxed region: ES methods, epochs {args.zoom}-10\n"
                      f"dots numbered for {methods[0][0]}", loc="left", fontsize=11)
    for a in axes[0]:
        a.grid(color="#e4e3dd", linewidth=0.8)
        a.set_xlabel("Task 1 (top-left) test loss")
        a.set_ylabel("Task 2 (bottom-right) test loss")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=200)
    print(f"wrote {args.out}.png / .pdf")


if __name__ == "__main__":
    main()
