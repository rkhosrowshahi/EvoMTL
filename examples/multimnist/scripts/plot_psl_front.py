"""Pareto front of PSL-OpenES runs vs the single networks of the other ES methods.

A PSL run (``evo_moea=psl_es``, ``tpsl_es`` or ``dcpsl_es``) writes ``front.csv``: the networks ``theta(lam)`` on a grid of
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
        --psl logs/mnist/psl_openes-tch/pop_128/adam-lr_1e-3/seed_1 \\
        --baselines logs/mnist/mo_openes-*/pop_128/adam-lr_1e-3/seed_1

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
from matplotlib.ticker import MaxNLocator

from plot_mo_openes import EPOCH_RE, MARKERS, SKIP_TEST, fmt_lr, label, parse_log, save

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


def read_gd_end(run_dir):
    """Test losses and accuracies (%) where gradient descent left off (``evo_schedule=gd_then_evolution``).

    The last ``Epoch:`` line of ``stdout.log`` before the switch to evolution; ``None`` for ES-only runs.
    """
    last = None
    with open(os.path.join(run_dir, "stdout.log"), encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("[EvoMTL] Switching from gradient descent"):
                if last is None:
                    return None
                v = [float(x) for x in last.groups()[1:]]
                return {"epochs": int(last.group(1)) + 1, "loss": np.array([v[4], v[6]]),
                        "acc": 100.0 * np.array([v[5], v[7]])}
            if m := EPOCH_RE.match(line):
                last = m
    return None


def disambiguate(runs):
    """Append the settings that differ (LR, sigma) to the names of runs of the same method.

    ``method`` keeps the plain name (marker lookup); ``name`` becomes e.g. ``MO-PSL-TCH-GuidedES (LR=1e-4, sigma=1e-3)``.
    """
    for r in runs:
        r["method"] = r["name"]
    for name in {r["method"] for r in runs}:
        same = [r for r in runs if r["method"] == name]
        if len(same) < 2:
            continue
        fields = [(k, txt) for k, txt in (("lr", "LR"), ("sigma", "sigma")) if len({r[k] for r in same}) > 1]
        for r in same:
            r["name"] = f"{name} (" + ", ".join(f"{txt}={fmt_lr(r[k])}" for k, txt in fields) + ")"


def square_limits(ax, pts):
    """Same range on both axes with equal scaling (a square box in objective space)."""
    lo, hi = float(pts.min()), float(pts.max())
    pad = 0.05 * (hi - lo)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal")


def zoom_limits(ax, pts):
    """Each axis fitted to its own range of the points (``--zoom``): small fronts fill the plot."""
    for i, set_lim in enumerate((ax.set_xlim, ax.set_ylim)):
        lo, hi = float(pts[:, i].min()), float(pts[:, i].max())
        pad = 0.08 * (hi - lo) or 1e-3
        set_lim(lo - pad, hi + pad)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(6))


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
    p.add_argument("--zoom", action="store_true",
                   help="fit each axis to the plotted points instead of a square box with equal scaling")
    p.add_argument("--gens", nargs="+", type=int, default=None,
                   help="front_evolution: plot only these generations (each needs a test evaluation in front.csv), "
                        "with a legend instead of the generation colour bar")
    p.add_argument("--gen_labels", nargs="+", default=None,
                   help="legend labels for --gens, e.g. '8 epochs + 1 step' '9 epochs' '10 epochs'")
    args = p.parse_args()
    if args.gen_labels is not None and len(args.gen_labels) != len(args.gens or []):
        raise SystemExit("--gen_labels needs one label per --gens generation")
    limits = zoom_limits if args.zoom else square_limits
    if args.out is None:
        if len(args.psl) > 1:
            raise SystemExit("several --psl runs: pass --out, e.g. figures/comparison/<name>")
        args.out = args.psl[0]

    psl = [load_run(d) for d in args.psl]
    disambiguate(psl)
    base = [load_run(d) for pat in args.baselines for d in sorted(glob.glob(pat))]
    base = [r for r in base if not r["aggregator"].startswith(("psl_", "tpsl_", "dcpsl_", "emapsl_"))]  # a glob may also catch PSL runs
    for r in psl:
        r["fronts"] = read_front(r["dir"])
        r["gd_end"] = read_gd_end(r["dir"])
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
        if r["gd_end"] is not None:
            (l1, l2), (a1, a2) = r["gd_end"]["loss"], r["gd_end"]["acc"] / 100.0
            summary.append({"method": r["name"], "solution": f"GD end (epoch {r['gd_end']['epochs']})",
                            "generation": 0, f"test_loss_{t1}": l1, f"test_loss_{t2}": l2,
                            "test_loss_total": l1 + l2, f"test_acc_{t1}": a1, f"test_acc_{t2}": a2,
                            "hypervolume": hypervolume_2d(r["gd_end"]["loss"][None], ref)})
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

    def gd_point(r):
        return r["gd_end"]["acc"] if acc else r["gd_end"]["loss"]

    gd_runs = [r for r in psl if r["gd_end"] is not None]
    gd_label = f"GD end (epoch {gd_runs[0]['gd_end']['epochs']}, start of ES)" if gd_runs else None

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
        m = MARKERS.get(r["method"], "*")
        Pf = xy(F)
        ax.plot(Pf[:, 0], Pf[:, 1], color=lc, linestyle=ls, linewidth=1.4, zorder=2)
        ax.scatter(Pf[:, 0], Pf[:, 1], c=lam_cmap(lam_norm(F[:, 0])), marker=m, s=70, edgecolors="k",
                   linewidths=0.4, zorder=3)
        bal = F[np.argmin(np.abs(F[:, 0] - 0.5))]
        ax.scatter(*xy(bal[None])[0], c=[lam_cmap(lam_norm(bal[0]))], marker=m, s=140, edgecolors="k", linewidths=0.8, alpha=0.6,
                   zorder=4)
        ax.plot([], [], marker=m, linestyle=ls, color=lc, markersize=9, label=f"{r['name']} front")
        pts.append(Pf)
        if r["gd_end"] is not None:  # where gradient descent left off, in the colour of the run's front
            ax.plot(*gd_point(r), "P", color=lc, markeredgecolor="k", markersize=11, markeredgewidth=0.8, zorder=6)
            pts.append(gd_point(r)[None])
    if gd_runs:
        ax.plot([], [], "P", color="w", markeredgecolor="k", markersize=11, linestyle="none", label=gd_label)
    for r in base:
        m = MARKERS.get(r["name"], "o")
        ax.plot(*point(r), m, color="k", markerfacecolor="none", markersize=9, markeredgewidth=1.3,
                zorder=5, label=r["name"])
        pts.append(point(r)[None])
    limits(ax, np.concatenate(pts))
    ax.set_xlabel(axis_label(1, t1))
    ax.set_ylabel(axis_label(2, t2))
    ax.grid(True)
    if args.legend == "below":  # several fronts: keep the legend off the data
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=8)
    else:
        ax.legend(loc=args.legend, fontsize=8)
    r0 = psl[0]
    shared = f"N={r0['pop']}" + (f", LR={fmt_lr(r0['lr'])}" if len({r['lr'] for r in psl}) == 1 else "")
    ax.set_title(f"Test-{word} front at generation {r0['final_gen']}\n({shared})")
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
        m = MARKERS.get(r["method"], "*")
        pts = []
        if args.gens:  # selected generations only, each labelled in the legend
            missing = [g for g in args.gens if g not in r["fronts"]]
            if missing:
                raise SystemExit(f"{r['dir']}: no test front at generation(s) {missing}; "
                                 f"evaluated: {sorted(r['fronts'])}")
            labels = args.gen_labels or [f"generation {g}" for g in args.gens]
            shown = [(g, r["fronts"][g], lab) for g, lab in zip(args.gens, labels)]
        else:
            shown = [(g, F, None) for g, F in r["fronts"].items() if g > SKIP_TEST]
        for g, F, lab in shown:
            c = gen_cmap(gen_norm(g))
            Pf = xy(F)
            ax.plot(Pf[:, 0], Pf[:, 1], "-", color=c, linewidth=1.2)
            ax.scatter(Pf[:, 0], Pf[:, 1], color=c, marker=m, s=22 if lab is None else 40, zorder=3)
            if lab is not None:
                ax.plot([], [], "-", color=c, marker=m, label=f"{lab} (gen {g})")
            pts.append(Pf)
        for b in base:
            ax.plot(*point(b), MARKERS.get(b["name"], "o"), color="k", markerfacecolor="none", markersize=7,
                    label=b["name"])
            pts.append(point(b)[None])
        if r["gd_end"] is not None:
            ax.plot(*gd_point(r), "P", color="tab:cyan", markeredgecolor="k", markersize=10, zorder=5,
                    linestyle="none", label=gd_label)
            pts.append(gd_point(r)[None])
        limits(ax, np.concatenate(pts))
        ax.set_title(r["name"].replace(" (", "\n("), fontsize=10)  # variant on its own line
        ax.set_xlabel(axis_label(1, t1))
        ax.grid(True)
    axes[0, 0].set_ylabel(axis_label(2, t2))
    if (gd_runs or args.gens) and not base:
        axes[0, 0].legend(loc="best", fontsize=7)
    if base:
        axes[0, 0].legend(loc="lower left" if acc else "lower right", fontsize=7, title="final ES mean",
                          title_fontsize=7)
    fig.suptitle(f"PSL front over generations (test {word})")
    fig.tight_layout()
    if not args.gens:
        fig.colorbar(ScalarMappable(norm=gen_norm, cmap=gen_cmap), ax=axes[0].tolist(), label="Generation")
    path = os.path.join(args.out, f"front_evolution{suffix}.png")
    save(fig, path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()
