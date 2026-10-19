"""Generate SNES / PGPE configs (per-parameter sigma) from the OpenES and MO-OpenES-UPGrad 10-epoch configs.

Each config is a copy of ``10epochs/mnist/evomtl/{openes-ws-full,mo_openes-upgrad-full}/pop_<N>/adam-lr_0.001/seed<s>.yaml``
with ``evo_es_sigma_adapt`` (snes or pgpe), ``evo_es_sigma_lr`` and, for PGPE, ``evo_es_sigma_max_change`` added,
and the method name in paths and W&B names changed (openes -> snes / pgpe). Everything else (sigma_0, Adam on the
mean, population, budget) stays as in the base config.

* ``--mode tune``: seed 1, one config per sigma learning rate, ``--gens`` generations, W&B off, logs under
  ``logs/mnist/tune_sigma_adapt/``; written to ``tune_sigma_adapt/mnist/<method>/pop_<N>/``.
* ``--mode long``: the full 2344-generation budget for ``--seeds``, one chosen sigma learning rate per algorithm,
  W&B as in the base configs; written to ``10epochs/mnist/evomtl/<method>/pop_<N>/adam-lr_0.001-slr_<x>/``.

Usage (from this folder)::

    python generate_sigma_adapt_configs.py --mode tune --snes_lrs 0.0174 0.2 1 4 --pgpe_lrs 0.1 0.5 2
    python generate_sigma_adapt_configs.py --mode long --snes_lrs 1 --pgpe_lrs 0.5 --seeds 1 2 3
"""

import argparse
import re
from pathlib import Path

BASES = {"openes-ws-full": "ws", "mo_openes-upgrad-full": "upgrad"}
NAMES = {"snes": "SNES", "pgpe": "PGPE"}


def sub_once(pattern, repl, text, name):
    new, n = re.subn(pattern, repl, text, count=1, flags=re.M)
    if n != 1:
        raise ValueError(f"{name}: pattern {pattern!r} not found")
    return new


def make(text, algo, slr, max_change, src_method, name, nl):
    """Insert the sigma-adaptation keys and rename openes -> algo in paths and W&B names."""
    dst_method = src_method.replace("openes", algo)
    keys = [f"evo_es_sigma_adapt: {algo}        # per-parameter sigma ({NAMES[algo]})",
            f"evo_es_sigma_lr: {slr:g}           # learning rate of the sigma update"]
    if algo == "pgpe":
        keys.append(f"evo_es_sigma_max_change: {max_change:g}  # bound on the relative change of each sigma_i per generation")
    text = sub_once(r"^(evo_es_sigma_min: .*)$", lambda m: m.group(1) + nl + nl.join(keys), text, name)
    for key in ("save_path", "wandb_name", "wandb_group"):
        text = re.sub(rf"^({key}: )(\S+)", lambda m: m.group(1) + m.group(2).replace(src_method, dst_method),
                      text, flags=re.M)
    head = (f"# {NAMES[algo]}: OpenES sampling with one sigma per parameter (evo_es_sigma = initial value), adapted "
            f"every generation;{nl}# copy of {src_method} with the sigma-adaptation keys added.{nl}")
    return head + text, dst_method


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("tune", "long"), required=True)
    p.add_argument("--pop", type=int, default=1024)
    p.add_argument("--snes_lrs", type=float, nargs="*", default=[])
    p.add_argument("--pgpe_lrs", type=float, nargs="*", default=[])
    p.add_argument("--max_change", type=float, default=0.2, help="PGPE evo_es_sigma_max_change")
    p.add_argument("--gens", type=int, default=100, help="tune mode: generations per run")
    p.add_argument("--eval_freq", type=int, default=25, help="tune mode: test evaluation every N generations")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3], help="long mode: seeds")
    args = p.parse_args()

    root = Path(__file__).resolve().parent
    n = 0
    for src_method in BASES:
        src_dir = root / "10epochs" / "mnist" / "evomtl" / src_method / f"pop_{args.pop}" / "adam-lr_0.001"
        for algo, lrs in (("snes", args.snes_lrs), ("pgpe", args.pgpe_lrs)):
            for slr in lrs:
                seeds = [1] if args.mode == "tune" else args.seeds
                for seed in seeds:
                    raw = (src_dir / f"seed{seed}.yaml").read_bytes().decode()
                    crlf = "\r\n" in raw  # edit with \n, write back with the base config's line endings
                    text, dst_method = make(raw.replace("\r\n", "\n"), algo, slr, args.max_change, src_method,
                                            f"{src_method}/seed{seed}", "\n")
                    if args.mode == "tune":
                        tag = f"slr_{slr:g}"
                        text = sub_once(r"^evo_iterations: .*$", f"evo_iterations: {args.gens}        # tuning run", text, tag)
                        text = sub_once(r"^evo_eval_freq: .*$", f"evo_eval_freq: {args.eval_freq}", text, tag)
                        text = sub_once(r"^save_path: .*$",
                                        f"save_path: logs/mnist/tune_sigma_adapt/{dst_method}/pop_{args.pop}/{tag}/", text, tag)
                        text = sub_once(r"^wandb_enable: .*$", "wandb_enable: false", text, tag)
                        dest = root / "tune_sigma_adapt" / "mnist" / dst_method / f"pop_{args.pop}" / f"{tag}.yaml"
                    else:
                        tag = f"adam-lr_0.001-slr_{slr:g}"
                        for key in ("save_path", "wandb_name", "wandb_group"):
                            text = re.sub(rf"^({key}: \S*?)adam-lr_(1e-3|0\.001)", rf"\1adam-lr_\2-slr_{slr:g}",
                                          text, count=1, flags=re.M)
                        dest = root / "10epochs" / "mnist" / "evomtl" / dst_method / f"pop_{args.pop}" / tag / f"seed{seed}.yaml"
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes((text.replace("\n", "\r\n") if crlf else text).encode())
                    n += 1
    print(f"wrote {n} {args.mode} configs")


if __name__ == "__main__":
    main()
