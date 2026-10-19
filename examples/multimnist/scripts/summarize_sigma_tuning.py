"""One row per sigma-adaptation tuning run (configs/tune_sigma_adapt): loss, accuracy, gradient alignment, sigma.

Reads ``metrics.csv`` and ``stdout.log`` of every run under ``--logs``. Columns: test loss and accuracy (mean of
the two tasks) at the last evaluation, mean cos(ES direction, true-gradient direction) over the second half of
the run, final sigma (mean / min / max over all parameters) and the mean sigma per layer type.

Usage (from examples/multimnist)::

    uv run python scripts/summarize_sigma_tuning.py
"""

import argparse
import csv
import glob
import os
import re
import statistics as st

EXAMPLE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAYERS = (("conv1.weight", "conv1"), ("conv2.weight", "conv2"), ("encoder.fc.weight", "fc"),
          ("fc_task.weight", "heads"), ("bias", "biases"))


def layer_sigmas(log):
    """Mean sigma per layer group from the '[EvoMTL] sigma <param> mean=...' lines (a plain mean over the
    parameter tensors in the group; the two heads and all biases are grouped)."""
    groups = {}
    for name, mean in re.findall(r"\[EvoMTL\] sigma (\S+)\s+mean=([\d.e+-]+)", log):
        for key, label in LAYERS:
            if key in name:
                groups.setdefault(label, []).append(float(mean))
                break
    return {k: st.mean(v) for k, v in groups.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--logs", default=os.path.join(EXAMPLE_DIR, "logs/mnist/tune_sigma_adapt"))
    args = p.parse_args()

    rows = []
    for f in sorted(glob.glob(os.path.join(args.logs, "*", "pop_*", "*", "metrics.csv"))):
        run = os.path.dirname(f)
        method, pop, variant = os.path.normpath(run).split(os.sep)[-3:]
        data = list(csv.DictReader(open(f)))
        if not data:
            continue
        tests = [r for r in data if r["test_loss_top-left"]]
        last = tests[-1]
        half = [float(r["true_cos_direction"]) for r in data[len(data) // 2:] if r["true_cos_direction"]]
        log = open(os.path.join(run, "stdout.log"), encoding="utf-8", errors="replace").read() \
            if os.path.exists(os.path.join(run, "stdout.log")) else ""
        rows.append({
            "method": method, "variant": variant, "gen": int(data[-1]["generation"]),
            "test_loss": (float(last["test_loss_top-left"]) + float(last["test_loss_bottom-right"])) / 2,
            "test_acc": (float(last["test_acc_top-left"]) + float(last["test_acc_bottom-right"])) / 2,
            "cos": st.mean(half) if half else float("nan"),
            "sigma": float(data[-1]["sigma"]),
            "sigma_min": float(data[-1].get("sigma_min") or data[-1]["sigma"]),
            "sigma_max": float(data[-1].get("sigma_max") or data[-1]["sigma"]),
            "layers": layer_sigmas(log),
        })

    labels = [label for _, label in LAYERS]
    print(f"{'method':26s} {'variant':12s} {'gen':>4} {'test loss':>9} {'test acc':>8} {'cos 2nd half':>12} "
          f"{'sigma mean':>10} {'min':>8} {'max':>8}  " + " ".join(f"{l:>7}" for l in labels))
    for r in rows:
        print(f"{r['method']:26s} {r['variant']:12s} {r['gen']:>4} {r['test_loss']:9.4f} {r['test_acc']:8.4f} "
              f"{r['cos']:12.4f} {r['sigma']:10.4g} {r['sigma_min']:8.3g} {r['sigma_max']:8.3g}  "
              + " ".join(f"{r['layers'][l]:7.4f}" if l in r["layers"] else f"{'-':>7}" for l in labels))


if __name__ == "__main__":
    main()
