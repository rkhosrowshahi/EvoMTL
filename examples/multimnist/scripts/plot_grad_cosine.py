"""Cosine(ES pseudo-gradient, backprop gradient) over generations, one curve per method.

Reads ``metrics.csv`` from each run folder (``true_cos_top-left``, ``true_cos_bottom-right``,
``true_cos_direction``, logged when ``evo_es_grad_cos_freq > 0``). Seeds of the same method are
averaged after a moving average; the band is the min-max over seeds. Dashed lines are the cosine
of an isotropic Gaussian estimator, sqrt(n / (n + d)), with n = N / 2 antithetic pairs and n = N.

Usage (from examples/multimnist)::

    uv run python scripts/plot_grad_cosine.py
    uv run python scripts/plot_grad_cosine.py --runs "logs/mnist/10epochs/*guidedes/pop_*/*/seed_*" --smooth 50
"""

import argparse
import ast
import csv
import glob
import os
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXAMPLE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# GuidedES and MO-UPGrad-GuidedES runs, and the run folders from before the rename (guided_es-ws, mo_guided_es-upgrad)
DEFAULT_RUNS = [os.path.join(EXAMPLE_DIR, f"logs/mnist/10epochs/{m}/pop_*/*/seed_*")
                for m in ("guidedes", "mo_upgrad_guidedes", "*guided_es*")]
PANELS = [("true_cos_top-left", "Task 1 (top-left)"),
          ("true_cos_bottom-right", "Task 2 (bottom-right)"),
          ("true_cos_direction", "Update direction")]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots, fixed order
NAMES = {"guidedes": "GuidedES", "mo_upgrad_guidedes": "MO-UPGrad-GuidedES",
         "guided_es-ws": "GuidedES", "mo_guided_es-upgrad": "MO-UPGrad-GuidedES"}


def run_info(run_dir):
    """(method, pop size, num params) from the folder name and the run's stdout.log."""
    method = os.path.normpath(run_dir).split(os.sep)[-4]
    log = open(os.path.join(run_dir, "stdout.log"), encoding="utf-8", errors="replace").read()
    pop = ast.literal_eval(re.search(r"evo_kwargs: (\{.*\})", log).group(1))["pop_size"]
    dims = int(re.search(r"num_dimensions=(\d+)", log).group(1))
    return method, pop, dims


def read_cos(run_dir):
    rows = [r for r in csv.DictReader(open(os.path.join(run_dir, "metrics.csv"))) if r["true_cos_direction"]]
    gen = np.array([int(r["generation"]) for r in rows])
    return gen, {k: np.array([float(r[k]) for r in rows]) for k, _ in PANELS}


def moving_average(x, w):
    if w <= 1:
        return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = (c[w:] - c[:-w]) / w
    return np.concatenate([np.cumsum(x[:w - 1]) / np.arange(1, w), out])  # expanding mean at the start


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", default=DEFAULT_RUNS, help="run folders or glob patterns")
    p.add_argument("--out", default=os.path.join(EXAMPLE_DIR, "figures/10epochs/grad_cosine"))
    p.add_argument("--smooth", type=int, default=50, help="moving-average window (generations)")
    args = p.parse_args()

    runs = sorted({d for pat in args.runs for d in glob.glob(pat)
                   if os.path.exists(os.path.join(d, "metrics.csv"))})
    if not runs:
        raise SystemExit(f"no runs with metrics.csv match {args.runs}")
    groups = defaultdict(list)
    for d in runs:
        method, pop, dims = run_info(d)
        groups[(method, pop)].append((d, dims))
        print(f"{method:26s} pop={pop:<5d} {d}")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True, layout="constrained")
    pops = sorted({pop for _, pop in groups})
    dims = groups[next(iter(groups))][0][1]
    ymax = 0.0
    for i, (key, label) in enumerate(PANELS):
        ax = axes[i]
        for j, ((method, pop), members) in enumerate(sorted(groups.items())):
            curves = [read_cos(d) for d, _ in members]
            n = min(len(g) for g, _ in curves)
            gen = curves[0][0][:n]
            sm = np.stack([moving_average(c[key][:n], args.smooth) for _, c in curves])
            name = NAMES.get(method, method) + (f", N={pop}" if len(pops) > 1 else "")
            color = COLORS[j % len(COLORS)]
            ax.fill_between(gen, sm.min(0), sm.max(0), color=color, alpha=0.18, linewidth=0)
            ax.plot(gen, sm.mean(0), color=color, linewidth=2,
                    label=f"{name} (mean of {len(curves)} seeds, min-max band)")
            ymax = max(ymax, sm.max())
        for pop in pops:
            for n_dirs, style, what in [(pop // 2, "--", f"{pop // 2} antithetic pairs"), (pop, ":", f"{pop} samples")]:
                ref = np.sqrt(n_dirs / (n_dirs + dims))
                ax.axhline(ref, color="#6b6a64", linestyle=style, linewidth=1.2,
                           label=f"isotropic ES, {what}: {ref:.3f}")
                ymax = max(ymax, ref)
        ax.set_title(label, loc="left", fontsize=11)
        ax.set_xlabel("Generation")
        ax.grid(axis="y", color="#e4e3dd", linewidth=0.8)
        ax.set_xlim(0, None)
    axes[0].set_ylabel("cos(pseudo-gradient, backprop gradient)")
    axes[0].set_ylim(0, ymax * 1.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle(f"ES gradient quality during training (d = {dims:,} parameters, "
                 f"{args.smooth}-generation moving average)", x=0.01, ha="left", fontsize=12)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=200)
    print(f"wrote {args.out}.png / .pdf")


if __name__ == "__main__":
    main()
