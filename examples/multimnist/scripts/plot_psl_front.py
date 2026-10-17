"""Pareto front of PSL-OpenES runs vs the single networks of the other ES methods.

A PSL run (``evo_moea=psl_es``) writes ``front.csv``: the networks ``theta(lam)`` on a grid of
preferences at every test evaluation. The baselines (MO-OpenES, OpenES, Guided ES variants)
each end with one network, the ES mean; its final test losses come from ``metrics.csv``.

Writes to ``--out`` (figures as PNG and vector PDF). With one ``--psl`` run the default is that
run's folder; comparisons of several PSL runs need ``--out``, by convention
``figures/comparison/<name>/``:

* ``front_test.png``      - test-loss objective space at the last generation: each PSL front (one
                            point per preference, coloured by lam_1), its balanced network
                            (lam = 0.5, 0.5; large marker) and the final ES mean of every baseline
* ``front_evolution.png`` - each PSL front at every test evaluation, coloured by generation
* with ``--metric accuracy``: the same two figures in test-accuracy space (``*_accuracy.png``)
* ``front_summary.csv``   - per method: test losses and accuracies (balanced network for PSL),
                            hypervolume w.r.t. ``--ref`` (whole front for PSL, one point otherwise)
                            and which baselines a PSL front dominates

Usage (from examples/multimnist)::

    python scripts/plot_psl_front.py \\
        --psl logs/mnist/psl_openes-tch-full/pop_128/adam-lr_1e-3/seed_1 \\
        --baselines logs/mnist/mo_openes-*-full/pop_128/adam-lr_1e-3/seed_1

    python scripts/plot_psl_front.py --psl <PSL run 1> <PSL run 2> --out figures/comparison/psl_fronts-pop_128
"""

import argparse
import csv
import glob
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize

from plot_mo_openes import MARKERS, SKIP_TEST, label, parse_log, save

CHANCE = math.log(10)  # cross-entropy of a uniform guess over 10 digits


def read_front(run_dir, split="test"):
    """``{generation: array [G, 5]}`` with columns lam_1, loss_1, loss_2, acc_1, acc_2 (rows by lam_1)."""
    with open(os.path.join(run_dir, "front.csv"), newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split]
    tasks = [c[len("loss_"):] for c in rows[0] if c.startswith("loss_")]
    fronts = {}
    for r in rows:
        fronts.setdefault(int(r["generation"]), []).append(
            [float(r[f"lambda_{tasks[0]}"])] + [float(r[f"loss_{t}"]) for t in tasks]
            + [float(r[f"acc_{t}"]) for t in tasks])
    return {g: np.array(sorted(v)) for g, v in sorted(fronts.items())}


def hypervolume_2d(P, ref):
    """Area dominated by the points ``P`` (minimization) and bounded by ``ref``."""
    P = P[np.all(P < ref, axis=1)]
    hv, top = 0.0, ref[1]
    for a, b in P[np.argsort(P[:, 0])]:
        if b < top:
            hv += (ref[0] - a) * (top - b)
            top = b
    return hv


def dominates(p, q):
    return bool(np.all(p <= q) and np.any(p < q))


def load_run(run_dir):
    run = parse_log(os.path.join(run_dir, "stdout.log"))
    if run is None:
        raise SystemExit(f"no ES run found in {run_dir}")
    run["dir"], run["name"] = run_dir, label(run)
    return run


def square_limits(ax, pts):
    """Same range on both axes with equal scaling (a square box in objective space)."""
    lo, hi = float(pts.min()), float(pts.max())
    pad = 0.05 * (hi - lo)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--psl", nargs="+", required=True, help="PSL run folders (with front.csv)")
    p.add_argument("--baselines", nargs="*", default=[], help="baseline run folders or glob patterns")
    p.add_argument("--ref", nargs=2, type=float, default=[CHANCE, CHANCE],
                   help="hypervolume reference point (default: chance-level loss ln 10 per task)")
    p.add_argument("--legend", default="upper right",
                   help="legend position in front_test.png: a matplotlib location, or 'below' (under the axes)")
    p.add_argument("--metric", default="loss", choices=("loss", "accuracy"),
                   help="objective space of the figures: test loss (front_test.png, front_evolution.png) or test "
                        "accuracy in percent (front_test_accuracy.png, front_evolution_accuracy.png)")
    p.add_argument("--out", default=None,
                   help="output folder (default with one --psl run: its folder; several runs: required, "
                        "e.g. figures/comparison/<name>)")
    args = p.parse_args()
    if args.out is None:
        if len(args.psl) > 1:
            raise SystemExit("several --psl runs: pass --out, e.g. figures/comparison/<name>")
        args.out = args.psl[0]

    psl = [load_run(d) for d in args.psl]
    base = [load_run(d) for pat in args.baselines for d in sorted(glob.glob(pat))]
    base = [r for r in base if not r["aggregator"].startswith("psl_")]  # a glob may also catch PSL runs
    for r in psl:
        r["fronts"] = read_front(r["dir"])
    ref = np.array(args.ref)
    t1, t2 = psl[0]["tasks"]
    os.makedirs(args.out, exist_ok=True)

    # ---- summary table -------------------------------------------------------------------------
    summary = []
    for r in psl:
        g = max(r["fronts"])
        F = r["fronts"][g]
        bal = F[np.argmin(np.abs(F[:, 0] - 0.5))]
        r["final"], r["final_gen"] = F, g
        summary.append({"method": r["name"], "solution": "front (balanced network)", "generation": g,
                        f"test_loss_{t1}": bal[1], f"test_loss_{t2}": bal[2], "test_loss_total": bal[1] + bal[2],
                        f"test_acc_{t1}": bal[3], f"test_acc_{t2}": bal[4],
                        "hypervolume": hypervolume_2d(F[:, 1:3], ref)})
    for r in base:
        g, l1, a1, l2, a2 = r["test"][-1]
        r["final_pt"] = np.array([l1, l2])
        r["final_acc"] = 100.0 * np.array([a1, a2])
        summary.append({"method": r["name"], "solution": "ES mean", "generation": int(g),
                        f"test_loss_{t1}": l1, f"test_loss_{t2}": l2, "test_loss_total": l1 + l2,
                        f"test_acc_{t1}": a1, f"test_acc_{t2}": a2,
                        "hypervolume": hypervolume_2d(r["final_pt"][None], ref)})
    for row in summary:
        if row["solution"] == "ES mean":
            pt = np.array([row[f"test_loss_{t1}"], row[f"test_loss_{t2}"]])
            row["dominated_by"] = " ".join(r["name"] for r in psl
                                           if any(dominates(q, pt) for q in r["final"][:, 1:3]))
    cols = list(summary[0].keys()) + ["dominated_by"]
    path = os.path.join(args.out, "front_summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(summary)
    print(f"{'method':20s} {'solution':26s} {'gen':>4s} {t1:>9s} {t2:>12s} {'total':>7s} "
          f"{'acc1':>6s} {'acc2':>6s} {'HV':>7s}  dominated by")
    for row in summary:
        print(f"{row['method']:20s} {row['solution']:26s} {row['generation']:4d} {row[f'test_loss_{t1}']:9.4f} "
              f"{row[f'test_loss_{t2}']:12.4f} {row['test_loss_total']:7.4f} {100 * row[f'test_acc_{t1}']:6.2f} "
              f"{100 * row[f'test_acc_{t2}']:6.2f} {row['hypervolume']:7.4f}  {row.get('dominated_by', '')}")
    print("wrote", path)

    # ---- plotted objective space: test loss, or test accuracy in percent ------------------------
    acc = args.metric == "accuracy"
    word, suffix = ("accuracy", "_accuracy") if acc else ("loss", "")

    def xy(F):
        """Front points ``[G, 2]`` in the plotted space (columns of read_front: loss 1, 2 or accuracy 3, 4)."""
        return F[:, [3, 4]] * 100.0 if acc else F[:, [1, 2]]

    def point(r):
        return r["final_acc"] if acc else r["final_pt"]

    def axis_label(k, t):
        return f"Objective {k}: {t} test {word}" + (" (%)" if acc else "")

    # ---- final fronts vs baselines ---------------------------------------------------------------
    lam_cmap, lam_norm = plt.cm.viridis, Normalize(0, 1)
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    pts = []
    # front lines: colours outside viridis (the point colours), one per PSL run
    for r, ls, lc in zip(psl, ["solid", "dashed", "dotted", "dashdot"],
                         ["tab:red", "tab:brown", "tab:pink", "tab:purple"]):
        F = r["final"]
        m = MARKERS.get(r["name"], "*")
        Pf = xy(F)
        ax.plot(Pf[:, 0], Pf[:, 1], color=lc, linestyle=ls, linewidth=1.4, zorder=2)
        ax.scatter(Pf[:, 0], Pf[:, 1], c=lam_cmap(lam_norm(F[:, 0])), marker=m, s=70, edgecolors="k",
                   linewidths=0.4, zorder=3)
        bal = F[np.argmin(np.abs(F[:, 0] - 0.5))]
        ax.scatter(*xy(bal[None])[0], c=[lam_cmap(lam_norm(bal[0]))], marker=m, s=140, edgecolors="k", linewidths=0.8, alpha=0.6,
                   zorder=4)
        ax.plot([], [], marker=m, linestyle=ls, color=lc, markersize=9, label=f"{r['name']} front")
        pts.append(Pf)
    for r in base:
        m = MARKERS.get(r["name"], "o")
        ax.plot(*point(r), m, color="k", markerfacecolor="none", markersize=9, markeredgewidth=1.3,
                zorder=5, label=r["name"])
        pts.append(point(r)[None])
    square_limits(ax, np.concatenate(pts))
    ax.set_xlabel(axis_label(1, t1))
    ax.set_ylabel(axis_label(2, t2))
    ax.grid(True)
    if args.legend == "below":  # several fronts: keep the legend off the data
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=8)
    else:
        ax.legend(loc=args.legend, fontsize=8)
    r0 = psl[0]
    ax.set_title(f"Test-{word} front at generation {r0['final_gen']}\n(N={r0['pop']}, LR={r0['lr']:.0e})".replace("e-0", "e-"))
    fig.colorbar(ScalarMappable(norm=lam_norm, cmap=lam_cmap), ax=ax, label=r"Preference $\lambda_1$")
    path = os.path.join(args.out, f"front_test{suffix}.png")
    save(fig, path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)

    # ---- front at every test evaluation ----------------------------------------------------------
    gen_cmap = ListedColormap(plt.cm.inferno(np.linspace(0.0, 0.85, 256)))  # as in plot_mo_openes.py
    fig, axes = plt.subplots(1, len(psl), figsize=(4.8 * len(psl), 4.4), squeeze=False)
    gen_norm = Normalize(0, max(r["n_iter"] for r in psl))
    for ax, r in zip(axes[0], psl):
        m = MARKERS.get(r["name"], "*")
        pts = []
        for g, F in r["fronts"].items():
            if g <= SKIP_TEST:
                continue
            c = gen_cmap(gen_norm(g))
            Pf = xy(F)
            ax.plot(Pf[:, 0], Pf[:, 1], "-", color=c, linewidth=1.2)
            ax.scatter(Pf[:, 0], Pf[:, 1], color=c, marker=m, s=22, zorder=3)
            pts.append(Pf)
        for b in base:
            ax.plot(*point(b), MARKERS.get(b["name"], "o"), color="k", markerfacecolor="none", markersize=7,
                    label=b["name"])
            pts.append(point(b)[None])
        square_limits(ax, np.concatenate(pts))
        ax.set_title(r["name"].replace(" (", "\n("), fontsize=10)  # variant on its own line
        ax.set_xlabel(axis_label(1, t1))
        ax.grid(True)
    axes[0, 0].set_ylabel(axis_label(2, t2))
    if base:
        axes[0, 0].legend(loc="lower left" if acc else "lower right", fontsize=7, title="final ES mean",
                          title_fontsize=7)
    fig.suptitle(f"PSL front over generations (test {word})")
    fig.tight_layout()
    fig.colorbar(ScalarMappable(norm=gen_norm, cmap=gen_cmap), ax=axes[0].tolist(), label="Generation")
    path = os.path.join(args.out, f"front_evolution{suffix}.png")
    save(fig, path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()
