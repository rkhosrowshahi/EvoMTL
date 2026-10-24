"""Convergence plots for MO-OpenES runs, parsed from their stdout logs.

Each log is identified by the ``evo_kwargs: {...}`` line that ``main.py`` prints, so the
aggregator (upgrad / ws), population size and LR come from the run itself. When a
``metrics.csv`` (written per generation by newer runs) sits next to the log, the curves are read
from it; older runs are parsed from the log text. Runs that are
still in progress are plotted up to their latest generation.

Writes to ``--out`` (each figure as PNG and as vector PDF):

* ``test_loss.png``       - test loss: objective 1, objective 2, total (sum)
* ``test_accuracy.png``   - test accuracy: task 1, task 2, average
* ``train_loss.png``      - train loss of the ES mean on the current mini-batch (moving average)
* ``objective_space.png`` - path of the ES mean in (objective 1, objective 2) test-loss space
* ``grad_cosine.png``     - cosine(pseudo-gradient, backprop gradient) per objective and for the
                            aggregated direction (runs with ``evo_es_grad_cos_freq > 0``)
With ``--split_by_pop`` each population size gets its own ``<out>/pop_<N>/`` folder.

* ``grad_cosine_vs_pop.png`` - mean of those cosines per run vs population size, with the
                            Gaussian-estimator reference sqrt(N / (N + d))

One column per (pop, LR) setting; UPGrad and WS are overlaid in each panel.

Usage (default paths are relative to examples/multimnist; ``--logs`` / ``--out`` given on the
command line are relative to the current directory)::

    python scripts/plot_mo_openes.py
    python scripts/plot_mo_openes.py --logs "logs/mnist/mo_openes-*/pop_*/lr_*/seed_1/stdout.log" --split_by_pop
"""

import argparse
import ast
import csv
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXAMPLE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # examples/multimnist
DEFAULT_LOGS = [
    os.path.join(EXAMPLE_DIR, "logs/mo_openes_test/pop_100_iter_500/*.log"),
    os.path.join(EXAMPLE_DIR, "logs/mnist/mo_openes-*/pop_*/lr_*/seed_*/stdout.log"),
]
LABELS = {"upgrad": "MO-UPGrad-OpenES", "mean": "MO-Mean-OpenES", "ws": "OpenES",
          "psl_tch": "MO-PSL-TCH-OpenES", "psl_ws": "MO-PSL-WS-OpenES",
          "psl_hv": "MO-PSL-HV-OpenES",  # psl_*: evo_moea=psl_es runs
          "tpsl_tch": "MO-TPSL-TCH-OpenES", "tpsl_ws": "MO-TPSL-WS-OpenES",
          "tpsl_hv": "MO-TPSL-HV-OpenES",  # tpsl_*: evo_moea=tpsl_es runs (Tangent PSL-ES)
          "dcpsl_upgrad": "MO-DCPSL-UPGrad-OpenES",
          "emapsl_tch": "MO-EMAPSL-TCH-OpenES", "emapsl_ws": "MO-EMAPSL-WS-OpenES",
          "emapsl_hv": "MO-EMAPSL-HV-OpenES"}  # evo_moea=emapsl_es runs (EMA-PSL-ES)  # evo_moea=dcpsl_es runs (Dual-cone PSL-ES)

# Objective-space marker and line style per method (also used by plot_psl_front.py)
MARKERS = {"MO-UPGrad-OpenES": "o", "MO-Mean-OpenES": "s", "OpenES": "^",
           "MO-UPGrad-GuidedES": "D", "MO-Mean-GuidedES": "P", "GuidedES": "v",
           "MO-PSL-TCH-OpenES": "*", "MO-PSL-TCH-GuidedES": "X", "MO-PSL-WS-OpenES": "h", "MO-PSL-WS-GuidedES": "p",
           "MO-PSL-TCH-GuidedES (centred)": "d", "MO-PSL-TCH-GuidedES (centred + ideal)": "8",
           "MO-PSL-TCH-GuidedES (centred + HV)": ">", "MO-PSL-HV-GuidedES (centred)": "H",
           "MO-PSL-TCH-GuidedES (centred + ideal + task guide)": "P", "MO-PSL-TCH-GuidedES (task guide)": "h",
           "MO-TPSL-TCH-OpenES": "<", "MO-TPSL-TCH-GuidedES": "^",
           "MO-DCPSL-UPGrad-OpenES": "d", "MO-DCPSL-UPGrad-GuidedES": "D"}
LINESTYLES = {"MO-UPGrad-OpenES": "solid", "MO-Mean-OpenES": "dotted", "OpenES": "dashed",
              "MO-UPGrad-GuidedES": "solid", "MO-Mean-GuidedES": "dotted", "GuidedES": "dashed",
              "MO-PSL-TCH-OpenES": "dashdot", "MO-PSL-TCH-GuidedES": "dashdot", "MO-PSL-WS-OpenES": "dashdot",
              "MO-PSL-WS-GuidedES": "dashdot", "MO-PSL-TCH-GuidedES (centred)": "dashdot",
              "MO-PSL-TCH-GuidedES (centred + ideal)": "dashdot",
              "MO-TPSL-TCH-OpenES": (0, (5, 1, 1, 1)), "MO-TPSL-TCH-GuidedES": (0, (5, 1, 1, 1)),
              "MO-DCPSL-UPGrad-OpenES": (0, (1, 1)), "MO-DCPSL-UPGrad-GuidedES": (0, (1, 1))}

# The first generations have loss ~15 (raw 0-255 inputs). Test-loss and objective-space plots
# drop evaluations at generation <= SKIP_TEST; train-loss y-axes are capped using the values
# after SKIP_TRAIN generations so the rest of the curve stays readable.
SKIP_TEST = 1
SKIP_TRAIN = 50

EPOCH_RE = re.compile(
    r"^Epoch: (\d+) \| TRAIN: ([\d.]+) ([\d.]+) \| ([\d.]+) ([\d.]+) \| Time: [\d.]+ \| "
    r"TEST: ([\d.]+) ([\d.]+) \| ([\d.]+) ([\d.]+) \|"
)
COS_RE = re.compile(
    r"^\[EvoMTL\] \S+ (\d+)/\d+ .*true_cos=\[([^\]]*)\] dir_cos=([+-]?[\d.]+)"
)
DIMS_RE = re.compile(r"dims=(\d+) \| pop=")
MEAN_RE = re.compile(r"^\[EvoMTL\] Mean (?:z|theta) TRAIN \(curr batch\) iteration (\d+) \| (.+)$")


def parse_log(path):
    """Return a dict with the run's settings, test rows (gen, l1, a1, l2, a2) and train rows (gen, l1, l2)."""
    cfg, tasks, test, train, cos, dims = None, None, [], {}, [], None
    with open(path, errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if cfg is None and "evo_kwargs:" in line:
                cfg = ast.literal_eval(line.split("evo_kwargs:", 1)[1].strip())
            elif m := COS_RE.match(line):
                c1, c2 = (float(x) for x in m.group(2).split(","))
                cos.append((int(m.group(1)), c1, c2, float(m.group(3))))
            elif dims is None and (m := DIMS_RE.search(line)):
                dims = int(m.group(1))
            elif m := MEAN_RE.match(line):
                parts = [p.rsplit(":", 1) for p in m.group(2).split(" | ")]
                tasks = tasks or [p[0].strip() for p in parts]
                train[int(m.group(1))] = (float(parts[0][1]), float(parts[1][1]))
            elif m := EPOCH_RE.match(line):
                gen = int(m.group(1))
                v = [float(x) for x in m.groups()[1:]]
                test.append((gen, v[4], v[5], v[6], v[7]))
                train[gen] = (v[0], v[2])
    if cfg is None or not ("aggregator" in cfg or "scalarization" in cfg or "center_fraction" in cfg):
        return None
    d, name = os.path.split(path)  # metrics.csv for stdout.log, <name>_metrics.csv for other logs
    csv_path = os.path.join(d, "metrics.csv" if name == "stdout.log" else os.path.splitext(name)[0] + "_metrics.csv")
    if os.path.exists(csv_path):  # newer runs: per-generation metrics written by the run itself
        test, train, cos, tasks = read_metrics_csv(csv_path)
    n_iter = int(cfg["num_iterations"])
    # The line labelled n_iter + 1 re-evaluates the final weights after the ES loop; drop it.
    test = np.array([r for r in test if r[0] <= n_iter]).reshape(-1, 5)
    train = np.array([(g, *train[g]) for g in sorted(train) if g <= n_iter]).reshape(-1, 3)
    last = int(max(test[-1, 0] if len(test) else 0, train[-1, 0] if len(train) else 0))
    return {
        # PSL runs log their scalarization instead of an aggregator; Tangent PSL-ES runs also log c0, Dual-cone
        # PSL-ES runs log center_fraction (and neither)
        "aggregator": "dcpsl_upgrad" if "center_fraction" in cfg else cfg["aggregator"] if "aggregator" in cfg
        else ("tpsl_" if "c0" in cfg else "emapsl_" if "b_ema" in cfg else "psl_") + cfg["scalarization"],
        "variant": psl_variant(cfg),
        "pop": int(cfg["pop_size"]), "lr": float(cfg["lr"]),
        "sigma": float(cfg["sigma"]) if "sigma" in cfg else None,
        "guided": int(cfg.get("guided_k", 0)) > 0,
        "n_iter": n_iter, "last": last, "tasks": tasks or ["objective 1", "objective 2"],
        "test": test, "train": train, "path": path,
        "cos": np.array(cos).reshape(-1, 4), "dims": dims,
    }


def read_metrics_csv(csv_path):
    """Test rows (gen, l1, a1, l2, a2), train rows {gen: (l1, l2)}, cosine rows and task names from metrics.csv."""
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    tasks = [c[len("train_loss_"):] for c in (rows[0].keys() if rows else []) if c.startswith("train_loss_")]
    num = lambda r, c: float(r[c]) if r.get(c) not in (None, "") else None
    test, train, cos = [], {}, []
    for r in rows:
        g = int(r["generation"])
        train[g] = tuple(num(r, f"train_loss_{t}") for t in tasks)
        if num(r, f"test_loss_{tasks[0]}") is not None:
            test.append((g, num(r, f"test_loss_{tasks[0]}"), num(r, f"test_acc_{tasks[0]}"),
                         num(r, f"test_loss_{tasks[1]}"), num(r, f"test_acc_{tasks[1]}")))
        if num(r, "true_cos_direction") is not None:
            cos.append((g, num(r, f"true_cos_{tasks[0]}"), num(r, f"true_cos_{tasks[1]}"), num(r, "true_cos_direction")))
    return test, train, cos, tasks or None


def psl_variant(cfg):
    """Name suffix of a PSL run with the split update (centred z weights, other B scores); '' otherwise."""
    parts = []
    if float(cfg.get("center_a", 1.0)) != 1.0:
        parts.append("centred")
    # spread_score (same / ideal / hv); runs from before the rename logged spread_ideal (zero / population)
    spread = cfg.get("spread_score", {"population": "ideal"}.get(cfg.get("spread_ideal"), "same"))
    if spread != "same":
        parts.append({"ideal": "ideal", "hv": "HV"}[spread])
    if cfg.get("psl_guide", "update") == "tasks":
        parts.append("task guide")
    if "b_ema" in cfg:  # EMA-PSL-ES: the length cap of B
        parts.append(f"B scale {cfg['b_scale']:g}")
    if "c0" in cfg and not cfg.get("brake", False):  # Tangent PSL-ES without the brake on c (runs before it existed too)
        parts.append("no brake")
    if "center_fraction" in cfg and not cfg.get("normalize", True):  # Dual-cone PSL-ES with raw UPGrad
        parts.append("raw")
    return f" ({' + '.join(parts)})" if parts else ""


def label(run):
    """Display name of a run, e.g. OpenES, or GuidedES when Guided ES sampling was on."""
    name = LABELS[run["aggregator"]]
    name = name.replace("OpenES", "GuidedES") if run.get("guided") else name
    return name + run.get("variant", "")


def save(fig, path, **kwargs):
    """Save ``fig`` as ``path`` (PNG) and as a vector PDF with the same name."""
    fig.savefig(path, dpi=120, **kwargs)
    fig.savefig(os.path.splitext(path)[0] + ".pdf", **kwargs)


def moving_average(y, w):
    """Trailing moving average; the first ``w - 1`` points average what is available so far."""
    c = np.cumsum(y)
    out = c.copy()
    out[w:] = c[w:] - c[:-w]
    return out / np.minimum(np.arange(1, len(y) + 1), w)


def column_title(runs):
    r = runs[0]
    title = f"N={r['pop']}, LR={fmt_lr(r['lr'])}"
    unfinished = [x for x in runs if x["last"] < x["n_iter"]]
    if unfinished:  # name each incomplete run, so a finished one does not make the column look done
        title += "\n(" + ", ".join(f"{label(x)} {x['last']}/{x['n_iter']}" for x in unfinished) + ")"
    return title


def cosine_yaxis(ax):
    """Cosine plots always span 0..1, padded so points at the edges are not clipped."""
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))


def fmt_lr(lr):
    """0.001 -> '1e-3', 0.0005 -> '5e-4'."""
    mant, exp = f"{lr:.0e}".split("e")
    return f"{mant}e{int(exp)}"


def cap_ylim(ax, curves, skip):
    """Set the y-limits from the points after generation ``skip``."""
    vals = np.concatenate([y[x > skip] for x, y in curves] or [np.empty(0)])
    if vals.size:
        pad = 0.05 * (vals.max() - vals.min() + 1e-3)
        ax.set_ylim(vals.min() - pad, vals.max() + pad)


def plot_grid(groups, row_labels, get_xy, title, path, skip=None):
    """One column per setting, one row per quantity; ``get_xy(run, row) -> (x, y)``."""
    fig, axes = plt.subplots(len(row_labels), len(groups), figsize=(4 * len(groups), 3 * len(row_labels)),
                             squeeze=False, sharey="row")
    curves = [[] for _ in row_labels]  # per row, across all columns (rows share a y-axis)
    for c, runs in enumerate(groups):
        for r, row_label in enumerate(row_labels):
            ax = axes[r, c]
            for run in runs:
                x, y = get_xy(run, r)
                ax.plot(x, y, label=label(run))
                curves[r].append((x, y))
            ax.grid(True)
            if c == 0:
                ax.set_ylabel(row_label)
            if r == 0:
                ax.set_title(column_title(runs))
            if r == len(row_labels) - 1:
                ax.set_xlabel("Generation")
    if skip is not None:
        for r in range(len(row_labels)):
            cap_ylim(axes[r, 0], curves[r], skip)
    axes[0, 0].legend()
    fig.suptitle(title)
    fig.tight_layout()
    save(fig, path)
    plt.close(fig)
    print("wrote", path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--logs", nargs="+", default=DEFAULT_LOGS, help="log files or glob patterns")
    p.add_argument("--out", default=os.path.join(EXAMPLE_DIR, "figures/comparison/mo_openes"))
    p.add_argument("--smooth", type=int, default=25, help="moving-average window for train curves")
    p.add_argument("--split_by_pop", action="store_true",
                   help="write each population size to its own <out>/pop_<N>/ folder (UPGrad vs WS at equal N)")
    args = p.parse_args()

    paths = sorted({f for pattern in args.logs for f in glob.glob(pattern)})
    runs = [r for r in map(parse_log, paths) if r is not None]
    if not runs:
        raise SystemExit(f"No MO-OpenES logs found for {args.logs}")
    for r in runs:
        print(f"{label(r):18s} pop={r['pop']:<5d} lr={r['lr']:<7g} "
              f"gen {r['last']}/{r['n_iter']}  {r['path']}")

    if args.split_by_pop:
        # One folder per population size: UPGrad vs WS (one column per LR) at the same N.
        for pop in sorted({r["pop"] for r in runs}):
            make_plots([r for r in runs if r["pop"] == pop], os.path.join(args.out, f"pop_{pop}"), args)
    else:
        make_plots(runs, args.out, args)


def make_plots(runs, out, args):
    """All figures for ``runs`` into ``out``."""
    # Group runs by (pop, LR): one plot column each, UPGrad before WS.
    order = list(LABELS)
    settings = sorted({(r["pop"], r["lr"]) for r in runs}, key=lambda s: (-s[1], s[0]))
    groups = [sorted([r for r in runs if (r["pop"], r["lr"]) == s], key=lambda r: order.index(r["aggregator"]))
              for s in settings]
    t1, t2 = runs[0]["tasks"]
    os.makedirs(out, exist_ok=True)

    # LR = 0 runs are pseudo-gradient sweeps (the mean never moves): only the cosine-vs-pop
    # figure is meaningful for them; training curves would be flat.
    if all(r["lr"] == 0 for r in runs):
        plot_grad_cosine(groups, t1, t2, args.smooth, out, only_vs_pop=True)
        return

    def test_loss(run, row):
        t = run["test"][run["test"][:, 0] > SKIP_TEST]
        return t[:, 0], [t[:, 1], t[:, 3], t[:, 1] + t[:, 3]][row]

    def test_acc(run, row):
        t = run["test"]
        return t[:, 0], 100 * np.array([t[:, 2], t[:, 4], (t[:, 2] + t[:, 4]) / 2])[row]

    def train_loss(run, row):
        t = run["train"]
        return t[:, 0], moving_average([t[:, 1], t[:, 2], t[:, 1] + t[:, 2]][row], args.smooth)

    loss_rows = [f"Objective 1: {t1} loss", f"Objective 2: {t2} loss", "Total loss (sum)"]
    plot_grid(groups, loss_rows, test_loss, "Test loss of the ES mean", os.path.join(out, "test_loss.png"))
    plot_grid(groups, [f"{t1} accuracy (%)", f"{t2} accuracy (%)", "Average accuracy (%)"], test_acc,
              "Test accuracy of the ES mean", os.path.join(out, "test_accuracy.png"))
    plot_grid(groups, loss_rows, train_loss,
              f"Train loss of the ES mean\n({args.smooth}-generation moving average)",
              os.path.join(out, "train_loss.png"), skip=SKIP_TRAIN)

    # Objective space: path of the mean in (objective 1, objective 2) test loss, coloured by
    # generation (light = early, dark = late).
    from matplotlib.collections import LineCollection

    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    from matplotlib.colors import ListedColormap

    norm = Normalize(0, max(r["n_iter"] for r in runs))
    # inferno without its last 15% (near-white, invisible on a white background)
    cmap = ListedColormap(plt.cm.inferno(np.linspace(0.0, 0.85, 256)))
    markers, linestyles = MARKERS, LINESTYLES
    fig, axes = plt.subplots(1, len(groups), figsize=(4.5 * len(groups), 4.2), squeeze=False)
    for ax, runs in zip(axes[0], groups):
        all_pts = []
        for run in runs:
            t = run["test"][run["test"][:, 0] > SKIP_TEST]
            if not len(t):
                continue
            agg = label(run)
            colors = cmap(norm(t[:, 0]))
            pts = t[:, [1, 3]]
            all_pts.append(pts)
            m, ls = markers.get(agg, "o"), linestyles.get(agg, "dashdot")
            ax.add_collection(LineCollection(np.stack([pts[:-1], pts[1:]], axis=1), colors=colors[1:],
                                             linewidths=1.5, linestyles=ls))
            ax.scatter(pts[:, 0], pts[:, 1], c=colors, marker=m, s=28,
                       edgecolors="none", zorder=3)
            ax.plot(*pts[-1], m, color=colors[-1], markersize=10, zorder=4)
            ax.plot([], [], marker=m, linestyle=ls, color="gray", label=label(run))
        if all_pts:  # same range on both axes, equal scaling: a square box in objective space
            v = np.concatenate(all_pts)
            lo, hi = float(v.min()), float(v.max())
            pad = 0.05 * (hi - lo)
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_aspect("equal")
        ax.set_title(column_title(runs))
        ax.set_xlabel(f"Objective 1: {t1} test loss")
        ax.grid(True)
    axes[0, 0].set_ylabel(f"Objective 2: {t2} test loss")
    axes[0, 0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Objective space trajectory")
    fig.tight_layout()
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes[0].tolist(), label="Generation")
    path = os.path.join(out, "objective_space.png")
    save(fig, path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)

    plot_grad_cosine(groups, t1, t2, args.smooth, out)


def plot_grad_cosine(groups, t1, t2, smooth, out, only_vs_pop=False):
    """Cosine(pseudo-gradient, true gradient) over generations, plus mean cosine vs population size."""
    groups = [[r for r in runs if len(r["cos"])] for runs in groups]
    groups = [runs for runs in groups if runs]
    if not groups:
        print("no runs with grad-cosine logging (evo_es_grad_cos_freq > 0); skipping grad_cosine plots")
        return
    rows = [f"cos(pseudo-grad, true grad)\nObjective 1: {t1}", f"cos(pseudo-grad, true grad)\nObjective 2: {t2}",
            "cos(ES direction,\ntrue-gradient direction)"]

    def get_xy(run, row):
        c = run["cos"]
        return c[:, 0], moving_average(c[:, 1 + row], smooth)

    if only_vs_pop:
        return plot_cosine_vs_pop(groups, t1, t2, rows, out)
    path = os.path.join(out, "grad_cosine.png")
    fig, axes = plt.subplots(len(rows), len(groups), figsize=(max(4 * len(groups), 7), 3 * len(rows)),
                             squeeze=False, sharey="row")
    for c, runs in enumerate(groups):
        ref = np.sqrt(runs[0]["pop"] / (runs[0]["pop"] + runs[0]["dims"]))
        for r, row_label in enumerate(rows):
            ax = axes[r, c]
            for run in runs:
                cos = run["cos"]
                raw, = ax.plot(cos[:, 0], cos[:, 1 + r], alpha=0.25, linewidth=0.8)
                ax.plot(*get_xy(run, r), color=raw.get_color(),
                        label=f"{label(run)} (N={run['pop']}, LR={fmt_lr(run['lr'])})")
            ax.axhline(0, color="gray", linewidth=0.8)
            ax.axhline(ref, color="k", linestyle="--", linewidth=1, label=r"$\sqrt{\frac{N}{N+D}}$")
            ax.grid(True)
            cosine_yaxis(ax)
            if c == 0:
                ax.set_ylabel(row_label)
            if r == 0 and any(x["last"] < x["n_iter"] for x in runs):
                ax.set_title(f"gen {max(x['last'] for x in runs)}/{runs[0]['n_iter']}")
            if r == len(rows) - 1:
                ax.set_xlabel("Generation")
        axes[0, c].legend(fontsize=10)
    fig.suptitle("Pseudo-gradient vs backprop gradient at the ES mean")
    fig.tight_layout()
    save(fig, path)
    plt.close(fig)
    print("wrote", path)
    plot_cosine_vs_pop(groups, t1, t2, rows, out)


def plot_cosine_vs_pop(groups, t1, t2, rows, out):
    """Mean cosine per run vs population size, with the sqrt reference curves."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    runs = [r for g in groups for r in g]
    d = runs[0]["dims"]
    all_ticks = [64, 100, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288,
                 1048576]
    all_labels = ["64", "100", "128", "256", "512", "1024", "2K", "4K", "8K", "16K", "32K", "64K", "128K",
                  "256K", "512K", "1M"]
    # start the axis at the smallest population actually measured
    keep = [i for i, t in enumerate(all_ticks) if t >= min(r["pop"] for r in runs)]
    ticks, tick_labels = [all_ticks[i] for i in keep], [all_labels[i] for i in keep]
    n = np.logspace(np.log10(ticks[0]), np.log10(ticks[-1]), 200)
    titles = [f"Objective 1: {t1}", f"Objective 2: {t2}", "Aggregated direction"]
    markers = {"upgrad": "o", "mean": "s", "ws": "x"}
    for i, ax in enumerate(axes):
        for agg in LABELS:
            sel = sorted((r for r in runs if r["aggregator"] == agg), key=lambda r: r["pop"])
            if sel:
                x = [r["pop"] for r in sel]
                y = [r["cos"][:, 1 + i].mean() for r in sel]
                bars = ax.errorbar(x, y, yerr=[r["cos"][:, 1 + i].std() for r in sel], fmt=markers[agg],
                                   markersize=8, capsize=3)
                # marker-only legend entry (no error-bar glyph)
                ax.plot([], [], markers[agg], color=bars[0].get_color(), markersize=8, label=LABELS[agg])
        ax.plot(n, np.sqrt(n / (n + d)), "k--", linewidth=1, label=r"$\sqrt{\frac{N}{N+D}}$")
        # Effective-sample curve sqrt(rho^2 N / (rho^2 N + D)); rho^2 from inverting it at each point.
        pts = [(r["pop"], r["cos"][:, 1 + i].mean()) for r in runs]
        rho2 = np.mean([c**2 * d / (N * (1 - c**2)) for N, c in pts if 0 < c < 1])
        ax.plot(n, np.sqrt(rho2 * n / (rho2 * n + d)), "k:", linewidth=1.5,
                label=rf"$\sqrt{{\frac{{\rho^2 N}}{{\rho^2 N+D}}}}$, $\rho$={np.sqrt(rho2):.2f}")
        ax.set_xscale("log")
        ax.set_xticks(ticks, tick_labels, fontsize=8, rotation=45)
        ax.minorticks_off()
        cosine_yaxis(ax)
        ax.set_xlim(ticks[0] / 1.15, ticks[-1] * 1.15)
        ax.set_xlabel("population size N")
        ax.set_title(titles[i])
        ax.grid(True, axis="y")
    axes[0].set_ylabel("cos(pseudo-gradient, true gradient)")
    axes[0].legend(loc="upper left", fontsize=10)
    fig.suptitle(f"Pseudo-gradient quality vs population size (D={d})")
    fig.tight_layout()
    path2 = os.path.join(out, "grad_cosine_vs_pop.png")
    save(fig, path2)
    plt.close(fig)
    print("wrote", path2)


if __name__ == "__main__":
    main()
