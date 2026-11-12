"""ES and gradient descent on the synthetic problems of :mod:`problems` (one config per call).

* ``vb01`` / ``vb``: the Valley-Bowl problem (2-D, two tasks; introduced with CAGrad), with the protocol of Nash-MTL's
  ``experiments/toy/trainer.py`` of https://github.com/AvivNavon/nash-mtl: the five initial points, Adam with lr 1e-3
  on ``x``, 35,000 iterations and ``scale = 0.1`` on the first loss for ``vb01`` (``vb``: scale 1).
* ``ewq``, ``cqf``, ``cqf2``, ``hqf``, ``mn2``, ``mn20``: the quadratic problems of the UPGrad trajectory plots
  (https://github.com/SimplexLab/trajectories), with their protocol: their initial points, SGD with their learning rate
  per problem times their multiplier per aggregator, and their number of iterations.

Here the gradient is replaced by the ES direction of each algorithm, run through the unchanged runners of
:mod:`LibMTL.evomtl.moea` (see ``trainer.py``); the gradient methods' plots of the papers are the reference.

Every run is described by a YAML file in ``configs/<problem>/evo/`` or ``configs/<problem>/grad/`` (``*_adam.yaml``:
the Adam protocol of the Nash-MTL configs instead of the reference one):

* evo: ``moea``, a single-network ES runner (a key of ``ES_RUNNERS`` of :mod:`LibMTL.evomtl.evo_trainer`, as
  ``evo_moea`` in the configs of the other examples), and ``es``, its keyword arguments;
* grad: ``weighting``, a weighting of :mod:`LibMTL.weighting` (``EW``, ``UPGrad``, ``MGDA``, ``SVD_PGA``,
  ``Nash_MTL``, ..., as ``weighting`` in the configs of the other examples), its ``weight_args`` (LibMTL's), and
  ``grad``: ``num_iterations`` and the optimizer (``optim``, ``lr``, ...);

and in both ``problem``: ``name``, a key of ``PROBLEMS`` in ``constants.py``, and ``loss_offset``; ``seeds``;
``save_path``; optional ``inits`` replaces the problem's initial points. Each config runs
from every initial point with every seed and writes to ``<save_path>/``: ``trajectories.npz`` (search-mean
trajectories in x-space and their losses), ``summary.json`` (final points and losses, distances to the Pareto front
in loss space and to the Pareto set in parameter space), a copy of the config, ``stdout.log`` and ``figures/`` (the
figures of the run, :func:`utils.run_figures`; ``utils.py`` compares the methods of a results folder).

``problem.loss_offset`` (21 in the Valley-Bowl configs: both losses positive everywhere; f2 >= -20, f1 >= -2 with
scale 0.1) is a constant added to both losses. It changes no method except MO-MGDA(LN), which divides each
pseudo-gradient by its loss and needs positive losses (the Pareto front of this problem is entirely negative).
Trajectories and reported losses are in the original, unshifted units.

Usage (from the repository root; flags override the YAML)::

    python examples/toy/main.py --config examples/toy/configs/vb01/evo/openes.yaml
    python examples/toy/main.py --config examples/toy/configs/cqf/evo/mo_upgrad_openes.yaml --num_iterations 50 --seeds 1
    python examples/toy/main.py --config examples/toy/configs/cqf/grad/upgrad.yaml
    for c in examples/toy/configs/cqf/*/*.yaml; do python examples/toy/main.py --config "$c"; done   # one problem
    python examples/toy/utils.py --results examples/toy/results/cqf/grad   # comparisons: <results>/figures/
"""

from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import json
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))  # first: the toy modules (constants, problems, trainer, main)
# LibMTL imports torch_geometric (QM9), which calls the deprecated torch.jit.script at import time
warnings.filterwarnings("ignore", message=r"`torch\.jit\.script` is deprecated", category=FutureWarning)

from LibMTL.evomtl.evo_trainer import ES_RUNNERS  # noqa: E402
from LibMTL.evomtl.moea.es_common import make_es_optimizer  # noqa: E402
from LibMTL.utils import set_random_seed  # noqa: E402
import LibMTL.weighting as weighting_method  # noqa: E402
from constants import DEFAULT_WORKERS, INITIAL_POINTS, PROBLEM_SETTINGS, PROBLEMS  # noqa: E402
from trainer import ToyTrainer  # noqa: E402


def weighted_model(weighting: str, init, task_num: int, epochs: int) -> torch.nn.Module:
    """The model of LibMTL's ``Trainer._prepare_model`` for the toy problems: the weighting class
    ``LibMTL.weighting.<weighting>`` mixed into a model whose encoder (the shared part of LibMTL's architectures) holds
    the point ``x`` and that has no decoders
    (an iteration is an epoch of one batch: ``epoch``, ``epochs`` and ``train_loss_buffer`` as the trainer sets them).
    Parameter gradients only (``rep_grad = False``): there is no representation, so GradDrop does not apply."""
    weighting_class = weighting_method.__dict__[weighting]

    class ToyMTLModel(weighting_class):
        def __init__(self):
            super().__init__()
            self.task_num, self.task_name = task_num, [f"f{k + 1}" for k in range(task_num)]
            self.rep_grad, self.multi_input, self.device = False, False, torch.device("cpu")
            self.encoder, self.decoders = torch.nn.Module(), torch.nn.ModuleDict()
            self.encoder.x = torch.nn.Parameter(torch.as_tensor(init, dtype=torch.float32).clone())
            self.epoch, self.epochs = 0, epochs
            self.train_loss_buffer = np.zeros([task_num, epochs])
            self.init_param()

        @property
        def x(self):
            return self.encoder.x

        def get_share_params(self):
            return self.encoder.parameters()

        def zero_grad_share_params(self):  # as LibMTL's architectures
            self.encoder.zero_grad(set_to_none=False)

    return ToyMTLModel()


def run_weighting(problem, init, weighting: str, num_iterations: int, loss_offset: float = 0.0,
                  weight_args: dict = None, **optim) -> np.ndarray:
    """Gradient descent from ``init`` with the LibMTL weighting ``weighting`` (a class of :mod:`LibMTL.weighting`),
    as ``Trainer.train_singlelevel`` of the other examples, an iteration being an epoch of one batch:
    ``optimizer.zero_grad(set_to_none=False)``, ``model.backward(losses, **weight_args)`` (three copies of the
    losses for MoDo and SDMGrad, which sample three batches), ``optimizer.step()``, FAMO's ``update_w`` on the new
    losses, and the epoch's losses into ``train_loss_buffer`` (DWA, GradNorm). ``weight_args`` override LibMTL's
    defaults (``prepare_args`` of :mod:`LibMTL.config`, as the other examples get them from their flags). The
    optimizer (that of the ES runners) is on all the model's parameters (the point and the weighting's own, e.g.
    IMTL's and UW's loss scales). Returns the ``[T + 1, D]`` trajectory (float32)."""
    from LibMTL.config import LibMTL_args, prepare_args

    if weighting == "GradDrop":
        raise ValueError("GradDrop needs representation gradients (rep_grad=True); the toy model has none")
    with contextlib.redirect_stdout(io.StringIO()):  # prepare_args prints the configuration
        kwargs = prepare_args(LibMTL_args.parse_args(["--weighting", weighting]))
    weight_args = {**(kwargs[0] if isinstance(kwargs, tuple) else kwargs)["weight_args"], **(weight_args or {})}
    T = int(num_iterations)
    model = weighted_model(weighting, init, problem.n_obj, T)
    optimizer = make_es_optimizer(list(model.parameters()), **optim)
    trajectory = [model.x.detach().numpy().copy()]
    for it in range(T):
        model.epoch = it
        losses = problem(model.x) + loss_offset  # [K]
        if weighting in ("MoDo", "SDMGrad"):
            losses = torch.stack([losses] * 3)  # three (identical) batches
        optimizer.zero_grad(set_to_none=False)
        moved = bool(torch.func.jacrev(problem)(model.x.detach()).any())
        if moved:
            model.backward(losses, **weight_args)
        # else every task gradient is zero (the Valley-Bowl plateau, e.g. (0, 0)): the update is zero whatever the
        # weights, but IMTL (unit gradients) and Nash-MTL (normalized Gramian) would divide by zero
        optimizer.step()
        with torch.no_grad():
            new_losses = problem(model.x) + loss_offset
        if weighting == "FAMO" and moved:
            model.update_w(new_losses)
        model.train_loss_buffer[:, it] = (losses[0] if losses.dim() > 1 else losses).detach().numpy()
        trajectory.append(model.x.detach().numpy().copy())
    return np.stack(trajectory)


def load_config(path: Path, n_iter: int = None) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    cfg["config_path"] = str(path)
    if sum(k in cfg for k in ("moea", "weighting")) != 1:
        raise ValueError(f"{path}: needs one of moea (evo configs) or weighting (grad configs)")
    if "moea" in cfg:
        if cfg["moea"] not in ES_RUNNERS:
            raise ValueError(f"{path}: unknown moea {cfg['moea']!r}; options: {sorted(ES_RUNNERS)}")
    elif cfg["weighting"] not in weighting_method.__all__:
        raise ValueError(f"{path}: unknown weighting {cfg['weighting']!r}; options: {weighting_method.__all__}")
    problem = cfg.get("problem", {})
    if problem.get("name") not in PROBLEMS or set(problem) - set(PROBLEM_SETTINGS):
        raise ValueError(f"{path}: problem must be {{name: <one of {list(PROBLEMS)}>, loss_offset: <float>}}, "
                         f"got {problem}")
    cfg["inits"] = cfg.get("inits") or INITIAL_POINTS[problem["name"]]  # the problem's unless the config sets them
    if n_iter is not None:
        cfg["es" if "moea" in cfg else "grad"]["num_iterations"] = int(n_iter)
    return cfg


def run_one(cfg: dict, init_idx: int, seed: int) -> dict:
    problem, init = PROBLEMS[cfg["problem"]["name"]], cfg["inits"][init_idx]
    offset = cfg["problem"].get("loss_offset", 0.0)
    t0 = time.perf_counter()
    if "weighting" in cfg:  # LibMTL weighting (seeded: PCGrad's task order, RLW's weights, ...)
        set_random_seed(seed)
        traj = run_weighting(problem, init, cfg["weighting"], loss_offset=offset, weight_args=cfg.get("weight_args"),
                             **cfg["grad"])
    else:
        trainer = ToyTrainer(init, problem, loss_offset=offset)
        kwargs = dict(cfg["es"], seed=seed, grad_cos_freq=0, evo_eval_freq=0)
        with contextlib.redirect_stdout(io.StringIO()):  # the runners log every generation
            ES_RUNNERS[cfg["moea"]](trainer, None, **kwargs)
        traj = np.stack(trainer.trajectory)
    return {"init": init_idx, "seed": seed, "traj": traj, "time": time.perf_counter() - t0}


def save_logs(cfg: dict, runs: dict) -> None:
    """``trajectories.npz``, ``summary.json`` and the config in ``cfg["save_path"]``."""
    out = ROOT / cfg["save_path"]
    out.mkdir(parents=True, exist_ok=True)
    seeds = list(cfg["seeds"])
    problem, inits = PROBLEMS[cfg["problem"]["name"]], cfg["inits"]
    # [inits, seeds, T + 1, D] in x-space, losses [inits, seeds, T + 1, K] in the original units
    X = np.stack([np.stack([runs[(i, s)]["traj"] for s in seeds]) for i in range(len(inits))]).astype(np.float32)
    L = problem.losses(X)
    np.savez_compressed(out / "trajectories.npz", x=X, losses=L, inits=np.array(inits), seeds=np.array(seeds))
    final = L[:, :, -1]  # [inits, seeds, K]
    dist = problem.front_distance(final)  # loss space (NaN without a sampled front: mn20)
    set_dist = problem.set_distance(X[:, :, -1])  # parameter space
    summary = {
        "name": cfg["name"],
        "problem": cfg["problem"]["name"],
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
    from utils import run_figures  # matplotlib only once the run is saved

    print(f"[toy] {cfg['name']:22s} figures -> {run_figures(out)}", flush=True)


def run_config(cfg: dict, workers: int = DEFAULT_WORKERS) -> None:
    """Every (initial point, seed) run of ``cfg`` on a pool of ``workers`` processes, then its logs."""
    inits, seeds = zip(*itertools.product(range(len(cfg["inits"])), cfg["seeds"]))
    print(f"[toy] {len(inits)} runs (inits x seeds), {workers} workers", flush=True)
    runs = {}
    # one torch thread per worker: the problems are tiny, and the default (one thread per core in every worker)
    # oversubscribes the CPU and slows each run down by orders of magnitude
    with ProcessPoolExecutor(max_workers=workers, initializer=torch.set_num_threads, initargs=(1,)) as pool:
        for r in pool.map(run_one, itertools.repeat(cfg), inits, seeds):
            runs[(r["init"], r["seed"])] = r
            print(f"[toy] {cfg['name']:22s} init {r['init']} seed {r['seed']}: {r['time']:.1f}s", flush=True)
    save_logs(cfg, runs)


class _Tee:
    """Write to the terminal stream and a log file at once (line-buffered)."""

    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, s):
        self.stream.write(s)
        self.log.write(s)
        return len(s)

    def flush(self):
        self.stream.flush()
        self.log.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path, help="path to a YAML config")
    parser.add_argument("--num_iterations", type=int, default=None, help="override es/grad.num_iterations")
    parser.add_argument("--sigma", type=float, default=None, help="override es.sigma (evo configs)")
    parser.add_argument("--lr", type=float, default=None, help="override es/grad.lr")
    parser.add_argument("--pop_size", type=int, default=None, help="override es.pop_size (evo configs)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="override seeds")
    parser.add_argument("--save_path", type=str, default=None,
                        help="override save_path (relative to the repository root, or absolute)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="parallel runs (inits x seeds)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.num_iterations)
    if args.seeds is not None:
        cfg["seeds"] = args.seeds
    for flag in ("sigma", "pop_size"):  # ES settings: evo configs only
        if getattr(args, flag) is not None:
            if "es" not in cfg:
                raise SystemExit(f"--{flag}: {args.config} is not an evo config")
            cfg["es"][flag] = getattr(args, flag)
    if args.lr is not None:
        cfg["es" if "moea" in cfg else "grad"]["lr"] = args.lr
    if args.save_path is not None:
        cfg["save_path"] = args.save_path
    # stdout and stderr also into <save_path>/stdout.log, as main.py of the other examples
    (ROOT / cfg["save_path"]).mkdir(parents=True, exist_ok=True)
    log = open(ROOT / cfg["save_path"] / "stdout.log", "w", buffering=1, encoding="utf-8")
    sys.stdout, sys.stderr = _Tee(sys.stdout, log), _Tee(sys.stderr, log)
    run_config(cfg, args.workers)


if __name__ == "__main__":
    main()
