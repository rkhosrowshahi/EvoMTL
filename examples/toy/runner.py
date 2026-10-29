"""OpenES-family ES on the synthetic problems of :mod:`problems`.

* ``nashmtl``: the 2-D toy problem of Nash-MTL (Navon et al. 2022; a modification of CAGrad's toy), with the protocol of
  ``experiments/toy/trainer.py`` of https://github.com/AvivNavon/nash-mtl: the five initial points, Adam with lr 1e-3
  on ``x``, 35,000 iterations and ``scale = 0.1`` on the first loss (configs in ``configs/`` and ``configs/scale_1/``).
* ``ewq``, ``cqf``, ``cqf2``, ``hqf``, ``mn2``, ``mn20``: the quadratic problems of the UPGrad trajectory plots
  (https://github.com/SimplexLab/trajectories), with their protocol: their initial points, SGD with their learning rate
  per problem times their multiplier per aggregator, and their number of iterations (configs in ``configs/<problem>/``).

Here the gradient is replaced by the ES direction of each algorithm, run through the unchanged runners of
:mod:`LibMTL.evomtl.moea` (see ``trainer.py``); the gradient methods' plots of the papers are the reference.

Every run is described by a YAML file (``moea``: the runner, ``problem``: ``name`` (default ``nashmtl``) and its
settings, ``seeds``, ``es``: keyword arguments of the runner, ``save_path``; optional ``inits`` replaces the problem's
initial points). Each config runs from every initial point with every seed and writes ``<save_path>/trajectories.npz``
(search-mean trajectories in x-space and their losses), ``summary.json`` (final points and losses, distances to the
Pareto front in loss space and to the Pareto set in parameter space) and a copy of the config.

``problem.loss_offset`` (21 in the ``nashmtl`` configs: both losses positive everywhere; f2 >= -20, f1 >= -2 with
scale 0.1) is a constant added to both losses. It changes no method except MO-MGDA(LN), which divides each pseudo-gradient by its loss
and needs positive losses (the Pareto front of this problem is entirely negative). Trajectories and reported losses
are in the original, unshifted units.

Usage (from the repository root)::

    python examples/toy/runner.py                                   # every config in examples/toy/configs/
    python examples/toy/runner.py examples/toy/configs/cqf/*.yaml       # one problem
    python examples/toy/runner.py examples/toy/configs/mo_mgda_openes.yaml --n-iter 5000
    python examples/toy/utils.py                                    # figures from examples/toy/logs/
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import itertools
import json
import os
import shutil
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
# LibMTL imports torch_geometric (QM9), which calls the deprecated torch.jit.script at import time
warnings.filterwarnings("ignore", message=r"`torch\.jit\.script` is deprecated", category=FutureWarning)

from LibMTL.evomtl.moea import (  # noqa: E402
    run_guidedes,
    run_mo_cagrad_guidedes,
    run_mo_cagrad_openes,
    run_mo_mean_guidedes,
    run_mo_mean_openes,
    run_mo_mgda2_guidedes,
    run_mo_mgda2_openes,
    run_mo_mgda_guidedes,
    run_mo_mgda_openes,
    run_mo_upgrad_guidedes,
    run_mo_upgrad_openes,
    run_openes,
)
from problems import NASHMTL_INITS, NASHMTL_PARETO_X2, NashMTLToy, make_problem  # noqa: E402
from trainer import ToyTrainer  # noqa: E402

# the Nash-MTL toy (utils.py and the reports): initial points of experiments/toy/trainer.py and its Pareto set
INITS = NASHMTL_INITS
PARETO_X2 = NASHMTL_PARETO_X2  # Pareto set: x2 = PARETO_X2, x1 in [-7, 7] (experiments/toy/utils.py)

RUNNERS = {
    "openes": run_openes,
    "guidedes": run_guidedes,
    "mo_mean_openes": run_mo_mean_openes,
    "mo_mean_guidedes": run_mo_mean_guidedes,
    "mo_upgrad_openes": run_mo_upgrad_openes,
    "mo_upgrad_guidedes": run_mo_upgrad_guidedes,
    "mo_mgda_openes": run_mo_mgda_openes,
    "mo_mgda_guidedes": run_mo_mgda_guidedes,
    "mo_cagrad_openes": run_mo_cagrad_openes,
    "mo_cagrad_guidedes": run_mo_cagrad_guidedes,
    "mo_mgda2_openes": run_mo_mgda2_openes,
    "mo_mgda2_guidedes": run_mo_mgda2_guidedes,
}


def pareto_front(scale: float, n: int = 2000) -> np.ndarray:
    """``[n, 2]`` losses along the Pareto set of the Nash-MTL toy."""
    return NashMTLToy(scale=scale).pareto_front(n)


def problem_of(cfg: dict):
    """The problem of a config (``problem`` without ``loss_offset``) and its initial points (``inits`` or the
    problem's)."""
    problem = make_problem(**{k: v for k, v in cfg["problem"].items() if k != "loss_offset"})
    return problem, cfg.get("inits") or problem.inits


def load_config(path: Path, n_iter: int = None) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    cfg["config_path"] = str(path)
    if cfg["moea"] not in RUNNERS:
        raise ValueError(f"{path}: unknown moea {cfg['moea']!r}; options: {list(RUNNERS)}")
    if n_iter is not None:
        cfg["es"]["num_iterations"] = int(n_iter)
    return cfg


def run_one(cfg: dict, init_idx: int, seed: int) -> dict:
    trainer = ToyTrainer(problem_of(cfg)[1][init_idx], **cfg["problem"])
    kwargs = dict(cfg["es"], seed=seed, grad_cos_freq=0, evo_eval_freq=0)
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):  # the runners log every generation
        RUNNERS[cfg["moea"]](trainer, None, **kwargs)
    return {"key": cfg["save_path"], "name": cfg["name"], "init": init_idx, "seed": seed,
            "traj": np.stack(trainer.trajectory),
            "time": time.perf_counter() - t0}


def _run_one(job):
    return run_one(*job)


def save_logs(cfg: dict, runs: dict) -> None:
    """``trajectories.npz``, ``summary.json`` and the config in ``cfg["save_path"]``."""
    out = ROOT / cfg["save_path"]
    out.mkdir(parents=True, exist_ok=True)
    seeds = list(cfg["seeds"])
    problem, inits = problem_of(cfg)
    # [inits, seeds, T + 1, D] in x-space, losses [inits, seeds, T + 1, K] in the original units
    X = np.stack([np.stack([runs[(i, s)]["traj"] for s in seeds]) for i in range(len(inits))]).astype(np.float64)
    L = problem.losses(X)
    np.savez_compressed(out / "trajectories.npz", x=X, losses=L, inits=np.array(inits), seeds=np.array(seeds))
    final = L[:, :, -1]  # [inits, seeds, K]
    dist = problem.front_distance(final)  # loss space (NaN without a sampled front: mn20)
    set_dist = problem.set_distance(X[:, :, -1])  # parameter space
    summary = {
        "name": cfg["name"],
        "problem": problem.name,
        "inits": inits,
        "seeds": seeds,
        "final_x": X[:, :, -1].tolist(),
        "final_losses": final.tolist(),
        "front_distance": dist.tolist(),
        "set_distance": set_dist.tolist(),
        "time_s": [[runs[(i, s)]["time"] for s in seeds] for i in range(len(inits))],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    with open(out / "config.yaml", "w") as f:
        yaml.safe_dump({k: v for k, v in cfg.items() if k != "config_path"}, f, sort_keys=False)
    print(f"[toy] {cfg['name']:22s} -> {cfg['save_path']} | distance to front / set per init (mean over seeds): "
          + " ".join(f"{d:.3g}/{e:.3g}" for d, e in zip(dist.mean(axis=1), set_dist.mean(axis=1))), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("configs", nargs="*", type=Path, help="YAML configs (default: every file in configs/)")
    parser.add_argument("--n-iter", type=int, default=None, help="override es.num_iterations (quick tests)")
    parser.add_argument("--save-root", type=Path, default=None,
                        help="write to <save-root>/<config name> instead of the configs' save_path")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()

    paths = args.configs or sorted((HERE / "configs").glob("*.yaml"))
    cfgs = [load_config(p, args.n_iter) for p in paths]
    if args.save_root is not None:
        for p, cfg in zip(paths, cfgs):
            cfg["save_path"] = str(args.save_root / Path(p).stem)

    n_inits = {c["save_path"]: len(problem_of(c)[1]) for c in cfgs}
    jobs = [(cfg, i, s) for cfg in cfgs for i, s in itertools.product(range(n_inits[cfg["save_path"]]), cfg["seeds"])]
    print(f"[toy] {len(jobs)} runs ({len(cfgs)} configs x inits x seeds), {args.workers} workers", flush=True)
    runs = {c["save_path"]: {} for c in cfgs}  # configs of different problems share method names
    # one torch thread per worker: the problems are tiny, and the default (one thread per core in every worker)
    # oversubscribes the CPU and slows each run down by orders of magnitude
    with ProcessPoolExecutor(max_workers=args.workers, initializer=torch.set_num_threads, initargs=(1,)) as pool:
        for r in pool.map(_run_one, jobs):
            runs[r["key"]][(r["init"], r["seed"])] = r
            print(f"[toy] {r['name']:22s} init {r['init']} seed {r['seed']}: {r['time']:.1f}s ({r['key']})", flush=True)
            cfg = next(c for c in cfgs if c["save_path"] == r["key"])
            if len(runs[r["key"]]) == n_inits[cfg["save_path"]] * len(cfg["seeds"]):
                save_logs(copy.deepcopy(cfg), runs[r["key"]])


if __name__ == "__main__":
    main()
