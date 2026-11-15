"""DUST vs A-DUST over the jitter std sigma: cosine with the backprop gradient and test accuracy.

Reads ``metrics.csv`` of every run matching ``--runs`` (folders ``<method>/pop_<N>/<optim>-sigma_<s>/seed_*``, as made
by configs/tune_dust_sigma), one line per method and population size N (forward evaluations per generation). Left:
mean cos(update direction, backprop gradient) over all generations; middle and right: test accuracy of each task at
the last evaluation. Prints the same numbers as a table.

Usage (from examples/multimnist)::

    uv run python scripts/plot_sigma_sweep.py
"""

import argparse
import csv
import glob
import os
import re
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXAMPLE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAMES = {"dust": "DUST", "a_dust": "A-DUST"}
MARKERS = {"dust": "o", "a_dust": "s"}
TASKS = ("top-left", "bottom-right")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default=os.path.join(EXAMPLE_DIR, "logs/multimnist/tune_dust_sigma/*/pop_*/*-sigma_*/seed_*"))
    p.add_argument("--out", default=os.path.join(EXAMPLE_DIR, "figures/comparison/dust_vs_a_dust-sigma_sweep"))
    args = p.parse_args()

    results = {}  # (method, N) -> sigma -> (generations, mean cosine, test accuracy per task)
    for run in sorted(glob.glob(args.runs)):
        path = os.path.join(run, "metrics.csv")
        rows = list(csv.DictReader(open(path))) if os.path.exists(path) else []
        if not rows:
            continue
        method, pop = os.path.normpath(run).split(os.sep)[-4:-2]
        sigma = float(re.search(r"sigma_([\d.]+)", run).group(1))
        last = [r for r in rows if r["test_acc_top-left"]][-1]
        results.setdefault((method, int(pop.split("_")[1])), {})[sigma] = (
            len(rows), st.mean(float(r["true_cos_direction"]) for r in rows if r["true_cos_direction"]),
            *(100 * float(last[f"test_acc_{t}"]) for t in TASKS))
    results = dict(sorted(results.items(), key=lambda kv: (kv[0][0] != "dust", kv[0][1])))

    print(f"{'method':8s} {'N':>4s} {'sigma':>6s} {'gens':>5s} {'cos':>6s} " + " ".join(f"{t:>12s}" for t in TASKS))
    for (method, pop), runs in results.items():
        for sigma, (gens, cos, *acc) in sorted(runs.items()):
            print(f"{NAMES.get(method, method):8s} {pop:4d} {sigma:6g} {gens:5d} {cos:6.3f} "
                  + " ".join(f"{a:12.2f}" for a in acc))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for (method, pop), runs in results.items():
        sigmas = sorted(runs)
        for i, ax in enumerate(axes):
            ax.plot(sigmas, [runs[s][1 + i] for s in sigmas], marker=MARKERS.get(method, "^"),
                    label=f"{NAMES.get(method, method)} (N={pop})")
    for ax, title in zip(axes, ("update direction", *(f"{t} task" for t in TASKS))):
        ax.set_xscale("log", base=2)
        sigmas = sorted({s for runs in results.values() for s in runs})
        ax.set_xticks(sigmas, [f"{s:g}" for s in sigmas])
        ax.minorticks_off()
        ax.set_xlabel("Jitter Std σ")
        ax.set_title(title)
        ax.grid(True)
    axes[0].set_ylabel("Cosine with Backprop Gradient")
    axes[1].set_ylabel("Test Accuracy (%)")
    for ax in axes[1:]:
        ax.set_ylim(top=100)
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    os.makedirs(args.out, exist_ok=True)
    fig.savefig(os.path.join(args.out, "sigma_sweep.png"), dpi=150)
    print(f"saved {os.path.join(args.out, 'sigma_sweep.png')}")


if __name__ == "__main__":
    main()
