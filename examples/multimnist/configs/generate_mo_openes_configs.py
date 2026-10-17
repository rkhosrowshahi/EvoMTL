"""Generate per-seed YAML configs for MO-OpenES (``evo_moea: mo_openes``).

Output layout::

    configs/no_scheduler/{dataset}/evomtl/{folder}/pop_{pop_size}/{optim}-lr_{lr}/seed{seed}.yaml

Pure ES training: ``evo_schedule: moea_then_gd`` with ``epochs: 0`` (no GD phase), searching
directly in weight space (``evo_adapter: full``). Example population / LR sweep::

    python generate_mo_openes_configs.py --pop_sizes 128 256 --optims adam rmsprop
    cd .. && python runner.py --dir configs/no_scheduler/mnist/evomtl/mo_openes-upgrad-full/pop_128/lr_0.001/
"""

import argparse
import itertools
from pathlib import Path

SEEDS = [1, 12, 123, 1234, 12345, 123456]

DATASET_META = {
    "mnist": "mnist",
    "fashion": "fashion",
    "mnist_and_fashion": "fashion_and_mnist",
}

METHOD_NAME = {"upgrad": "MO-OpenES-UPGrad", "mean": "MO-OpenES-WS", "ws": "OpenES"}
# config / log folder per aggregator (the single-objective baseline is not under mo_openes)
FOLDER = {"upgrad": "mo_openes-upgrad-full", "mean": "mo_openes-mean-full", "ws": "openes-ws-full"}
AGG_DESC = {
    "upgrad": "one OpenES pseudo-gradient per objective, combined with UPGrad (mean of dual-cone projections)",
    "mean": "one OpenES pseudo-gradient per objective, combined by an equal-weight average",
    "ws": "single-objective OpenES on the equal-weight sum of the two losses (baseline)",
}

TEMPLATE = """\
# {method}: {desc}.
# ES-only training of all weights (evo_adapter: full); {optim_name} on the search mean.

# General
mode: train
seed: {seed}
gpu_id: "0"

# Model (shared LeNet encoder, one head per task)
weighting: EW                  # gradient-descent loss weighting; unused here (epochs: 0)
arch: HPS
rep_grad: false
multi_input: false

# Dataset
dataset: {dataset_value}
dataset_path: .
num_tasks: 2
train_bs: 256
test_bs: 10000
epochs: 0                      # no gradient-descent phase: training is ES only

# Evolution strategy ({method}); evo_moea selects the ES code, evo_es_aggregator the combination rule
evo_training: true
evo_schedule: moea_then_gd     # ES first, then `epochs` of gradient descent (none here)
evo_moea: mo_openes
evo_adapter: full
evo_adapter_alpha: 1.0
evo_iterations: {iterations}
evo_pop_size: {pop_size}
evo_num_batches: 1
evo_eval_freq: {eval_freq}
evo_seed: {seed}
evo_es_aggregator: {aggregator}
{upgrad_block}evo_es_lr: {lr}
{optim_block}evo_es_sigma: {sigma}
evo_es_sigma_decay: 1.0
evo_es_sigma_min: 0.0
evo_es_antithetic: true
evo_es_grad_cos_freq: 1       # log cosine(pseudo-gradient, backprop gradient); 0 = off

# Paths
save_path: logs/{dataset}/{folder}/pop_{pop_size}/{optim}-lr_{lr_tag}/seed_{seed}/
load_path: null

# Weights & Biases
wandb_entity: rasa_research
wandb_project: mtl
wandb_name: {folder}-pop_{pop_size}-{optim}-lr_{lr}-seed{seed}
wandb_group: {folder}-pop_{pop_size}-{optim}-lr_{lr}
"""


OPTIM_BLOCKS = {
    "sgd": "evo_es_optim: sgd             # optimizer on the search mean (sgd, adam or rmsprop); evo_es_lr is its learning rate\n",
    "adam": ("evo_es_optim: adam            # optimizer on the search mean (sgd, adam or rmsprop); evo_es_lr is its learning rate\n"
             "evo_es_adam_beta1: 0.9\nevo_es_adam_beta2: 0.999\nevo_es_adam_eps: 1.0e-8\n"),
    "rmsprop": ("evo_es_optim: rmsprop         # optimizer on the search mean (sgd, adam or rmsprop); evo_es_lr is its learning rate\n"
                "evo_es_rmsprop_alpha: 0.99    # smoothing constant of the squared-gradient average\nevo_es_rmsprop_eps: 1.0e-8\n"),
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=["mnist"], choices=sorted(DATASET_META))
    p.add_argument("--aggregators", nargs="+", default=["upgrad", "ws"], choices=sorted(AGG_DESC))
    p.add_argument("--lrs", nargs="+", type=float, default=[0.001])
    p.add_argument("--optims", nargs="+", default=["sgd", "adam", "rmsprop"], choices=sorted(OPTIM_BLOCKS))
    p.add_argument("--seeds", nargs="+", type=int, default=[1],
                   help=f"seeds to write (default: 1 while testing; final runs: {' '.join(map(str, SEEDS))})")
    p.add_argument("--iterations", type=int, default=5000)
    p.add_argument("--pop_sizes", nargs="+", type=int, default=[128, 256, 512, 1024, 2048, 4096])
    p.add_argument("--sigma", type=float, default=0.01)
    p.add_argument("--eval_freq", type=int, default=100)
    args = p.parse_args()

    root = Path(__file__).resolve().parent / "no_scheduler"
    n = 0
    for dataset in args.datasets:
        for agg in args.aggregators:
            for pop_size, optim, lr in itertools.product(args.pop_sizes, args.optims, args.lrs):
                if pop_size % 2:
                    raise ValueError(f"pop_size must be even (antithetic sampling), got {pop_size}")
                out_dir = (
                    root / dataset / "evomtl" / FOLDER[agg]
                    / f"pop_{pop_size}" / f"{optim}-lr_{lr:g}"
                )
                out_dir.mkdir(parents=True, exist_ok=True)
                for seed in args.seeds:
                    text = TEMPLATE.format(
                        aggregator=agg,
                        desc=AGG_DESC[agg],
                        seed=seed,
                        dataset=dataset,
                        dataset_value=DATASET_META[dataset],
                        iterations=args.iterations,
                        pop_size=pop_size,
                        eval_freq=args.eval_freq,
                        lr=f"{lr:g}",
                        sigma=f"{args.sigma:g}",
                        optim=optim,
                        optim_block=OPTIM_BLOCKS[optim],
                        optim_name={"sgd": "SGD", "adam": "Adam", "rmsprop": "RMSprop"}[optim],
                        method=METHOD_NAME[agg],
                        folder=FOLDER[agg],
                        lr_tag=f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e+"),  # 0.001 -> 1e-3
                        upgrad_block=("upgrad_norm_eps: 0.0001        # UPGrad: Gramian normalization epsilon\n"
                                      "upgrad_reg_eps: 0.0001         # UPGrad: Gramian regularization\n") if agg == "upgrad" else "",
                    )
                    (out_dir / f"seed{seed}.yaml").write_text(text)
                    n += 1
    print(f"Wrote {n} configs under {root}")


if __name__ == "__main__":
    main()
