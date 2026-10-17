"""Run the gradient-descent configs (gradient aggregators, e.g. EW, MGDA, CAGrad) under a folder
(recursively), several at a time, on Mac, Linux or Windows. ES configs (``evo_training: true``) are skipped.

* ``--jobs`` runs in parallel; each run gets ``--threads`` CPU threads and, with ``--gpus``, a GPU in turn.
* Runs whose ``<save_path>/stdout.log`` already ends with the final summary are skipped, so after an
  interruption the same command continues where it stopped (``--rerun`` runs them again).
* Keeps the machine from sleeping while it runs (``caffeinate`` on macOS, the power API on Windows).
* Progress and exit codes go to the terminal and to ``logs/run_grad_aggs.log``; each run writes its own
  ``stdout.log`` (and W&B, when its config enables it).

Usage (from examples/multimnist)::

    python run_grad_aggs.py configs/10epochs/mnist                      # all gradient-descent configs, 3 at a time
    python run_grad_aggs.py configs/10epochs/mnist --jobs 4             # 4 at a time
    python run_grad_aggs.py configs/10epochs/mnist --match seed1        # paths containing "seed1"
    python run_grad_aggs.py configs/10epochs/mnist --gpus 0 1           # alternate between two GPUs
    python run_grad_aggs.py configs/10epochs/mnist --dry_run            # list what would run
"""

import argparse
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
DONE_MARK = "Best Result"  # last block printed by every finished run (ES and gradient descent)


def keep_awake():
    """Prevent system sleep while this process lives; returns a cleanup function."""
    if sys.platform == "darwin":
        proc = subprocess.Popen(["caffeinate", "-is", "-w", str(os.getpid())])
        return proc.terminate
    if platform.system() == "Windows":
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        return lambda: ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    return lambda: None


def is_done(save_path):
    log = HERE / save_path / "stdout.log"
    if not log.exists():
        return False
    with open(log, "rb") as f:
        f.seek(max(log.stat().st_size - 4000, 0))
        return DONE_MARK in f.read().decode(errors="ignore")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dir", help="folder with YAML configs (searched recursively)")
    p.add_argument("--jobs", type=int, default=3, help="runs at the same time")
    p.add_argument("--threads", type=int, default=3, help="CPU threads per run (OMP_NUM_THREADS)")
    p.add_argument("--gpus", nargs="*", default=None, help="GPU ids to alternate between, e.g. 0 1")
    p.add_argument("--match", default=None, help="only configs whose path contains this text")
    p.add_argument("--rerun", action="store_true", help="also run configs that already finished")
    p.add_argument("--dry_run", action="store_true", help="list the runs and exit")
    args = p.parse_args()

    root = Path(args.dir)
    root = root if root.is_absolute() else HERE / root
    runs = []
    for cfg in sorted(root.rglob("*.yaml")):
        c = yaml.safe_load(cfg.read_text()) or {}
        if c.get("evo_training"):
            continue
        if args.match and args.match not in str(cfg):
            continue
        save = str(c.get("save_path", ""))
        if not args.rerun and save and is_done(save):
            continue
        runs.append((cfg, save))
    print(f"{len(runs)} gradient-descent runs to do under {root} ({args.jobs} at a time)")
    if args.dry_run:
        for cfg, save in runs:
            print(f"  {cfg.relative_to(HERE)}  ->  {save}")
        return

    log_path = HERE / "logs" / "run_grad_aggs.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def note(msg):
        line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    def run(i, cfg):
        env = dict(os.environ, OMP_NUM_THREADS=str(args.threads))
        cmd = [sys.executable, "main.py", "--config", str(cfg)]
        if args.gpus:
            cmd += ["--gpu_id", args.gpus[i % len(args.gpus)]]
        t0 = time.time()
        rc = subprocess.run(cmd, cwd=HERE, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
        return rc, (time.time() - t0) / 60

    release = keep_awake()
    failed = []
    try:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(run, i, cfg): (cfg, save) for i, (cfg, save) in enumerate(runs)}
            note(f"started {len(runs)} runs")
            for n, fut in enumerate(as_completed(futures), 1):
                cfg, save = futures[fut]
                rc, minutes = fut.result()
                status = "ok" if rc == 0 else f"FAILED (exit {rc}; see {save}stdout.log)"
                if rc != 0:
                    failed.append(cfg)
                note(f"[{n}/{len(runs)}] {status} {minutes:.1f} min  {cfg.relative_to(HERE)}")
    finally:
        release()
    note(f"done: {len(runs) - len(failed)} ok, {len(failed)} failed")
    for cfg in failed:
        note(f"  failed: {cfg.relative_to(HERE)}")


if __name__ == "__main__":
    main()
