"""Backfill ``metrics.csv`` for MO-OpenES runs that predate the CSV writer, from their stdout logs.

Writes the same columns as ``run_mo_openes`` (see ``LibMTL/evomtl/moea/mo_openes.py``). Values the
old logs never printed stay blank: ``pop_loss_mean_*`` (the log only has Pareto-front stats) and
``train_acc_*`` except on test generations.

Output location: ``<dir>/metrics.csv`` for a ``stdout.log``; ``<dir>/<name>_metrics.csv`` for any
other ``<name>.log`` (several runs can share one folder). Existing CSVs are left alone unless
``--overwrite`` is given.

Usage (from examples/multimnist)::

    python scripts/logs_to_csv.py                       # every log under logs/
    python scripts/logs_to_csv.py --logs "logs/mnist/**/stdout.log"
"""

import argparse
import ast
import csv
import glob
import os
import re

EXAMPLE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROG_RE = re.compile(r"^\[EvoMTL\] (?:MO-OpenES-\w+|OpenES) (\d+)/(\d+) \|(.*)$")
EPOCH_RE = re.compile(
    r"^Epoch: (\d+) \| TRAIN: ([\d.]+) ([\d.]+) \| ([\d.]+) ([\d.]+) \| Time: [\d.]+ \| "
    r"TEST: ([\d.]+) ([\d.]+) \| ([\d.]+) ([\d.]+) \|"
)
MEAN_RE = re.compile(r"^\[EvoMTL\] Mean z TRAIN \(curr batch\) iteration (\d+) \| (.+)$")
TASKS_RE = re.compile(r"LOG FORMAT \| (\S+)_LOSS Acc \| (\S+)_LOSS Acc")


def csv_path_for(log_path):
    d, name = os.path.split(log_path)
    if name == "stdout.log":
        return os.path.join(d, "metrics.csv")
    return os.path.join(d, os.path.splitext(name)[0] + "_metrics.csv")


def convert(log_path):
    """Return (rows, tasks) for one log, or None if it is not an MO-OpenES run."""
    cfg, tasks, rows = None, None, {}
    with open(log_path, errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if cfg is None and "evo_kwargs:" in line:
                cfg = ast.literal_eval(line.split("evo_kwargs:", 1)[1].strip())
                continue
            if tasks is None and (m := TASKS_RE.search(line)):
                tasks = [m.group(1), m.group(2)]
                continue
            if tasks is None:
                continue
            if m := PROG_RE.match(line):
                g, rest = int(m.group(1)), m.group(3)
                r = rows.setdefault(g, {"generation": g})
                if x := re.search(r"sigma=([\d.e+-]+)", rest):
                    r["sigma"] = float(x.group(1))
                if x := re.search(r"Time: ([\d.]+)", rest):
                    r["time_s"] = float(x.group(1))
                if x := re.search(r"cos\(g\)=([+-]?[\d.]+)", rest):
                    r["pseudo_grad_cos"] = float(x.group(1))
                if x := re.search(r" w=\[([^\]]*)\]", rest):
                    for t, v in zip(tasks, x.group(1).split(",")):
                        r[f"weight_{t}"] = float(v)
                if x := re.search(r"true_cos=\[([^\]]*)\] dir_cos=([+-]?[\d.]+)", rest):
                    for t, v in zip(tasks, x.group(1).split(",")):
                        r[f"true_cos_{t}"] = float(v)
                    r["true_cos_direction"] = float(x.group(2))
            elif m := MEAN_RE.match(line):
                r = rows.setdefault(int(m.group(1)), {"generation": int(m.group(1))})
                for part in m.group(2).split(" | "):
                    t, v = part.rsplit(":", 1)
                    r[f"train_loss_{t.strip()}"] = float(v)
            elif m := EPOCH_RE.match(line):
                g = int(m.group(1))
                if cfg is not None and g > int(cfg["num_iterations"]):
                    continue  # post-ES re-evaluation of the final weights
                v = [float(x) for x in m.groups()[1:]]
                r = rows.setdefault(g, {"generation": g})
                r.update({
                    f"train_loss_{tasks[0]}": v[0], f"train_acc_{tasks[0]}": v[1],
                    f"train_loss_{tasks[1]}": v[2], f"train_acc_{tasks[1]}": v[3],
                    f"test_loss_{tasks[0]}": v[4], f"test_acc_{tasks[0]}": v[5],
                    f"test_loss_{tasks[1]}": v[6], f"test_acc_{tasks[1]}": v[7],
                })
    if cfg is None or "aggregator" not in cfg or tasks is None:
        return None
    return [rows[g] for g in sorted(rows)], tasks


def columns(tasks):
    return (
        ["generation", "time_s", "sigma"]
        + [f"pop_loss_mean_{t}" for t in tasks]
        + [f"train_loss_{t}" for t in tasks] + [f"train_acc_{t}" for t in tasks]
        + [f"test_loss_{t}" for t in tasks] + [f"test_acc_{t}" for t in tasks]
        + [f"weight_{t}" for t in tasks] + ["pseudo_grad_cos"]
        + [f"true_cos_{t}" for t in tasks] + ["true_cos_direction"]
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--logs", nargs="+", default=[os.path.join(EXAMPLE_DIR, "logs/**/*.log")])
    p.add_argument("--overwrite", action="store_true", help="replace existing CSVs")
    args = p.parse_args()

    paths = sorted({f for pat in args.logs for f in glob.glob(pat, recursive=True)})
    for path in paths:
        out = csv_path_for(path)
        if os.path.exists(out) and not args.overwrite:
            print(f"skip (exists)  {out}")
            continue
        res = convert(path)
        if res is None:
            continue
        rows, tasks = res
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=columns(tasks))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {len(rows):5d} rows  {out}")


if __name__ == "__main__":
    main()
