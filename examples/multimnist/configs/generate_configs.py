"""Generate per-seed YAML configs for every MOO algorithm × dataset combination.

Output layout::

    configs/{dataset}/{algorithm}/seed{seed}.yaml

``algorithm`` is the weighting name in **lower case** (e.g. ``imtl``, ``nash_mtl``, ``evomtl_ew_nsga2``).

``save_path`` in each file follows::

    logs/{dataset}/{moo}/{optim}/seed_{seed}/

- ``dataset``: ``mnist`` | ``fashion`` | ``mnist_and_fashion`` (path segment)
- ``moo``: lower-case slug (e.g. ``mgda``), or ``evomtl-ew_nsga2`` / ``evomtl-ew_comocma`` for EvoMTL; W&B uses the same words with ``+`` for EvoMTL (e.g. ``evomtl-ew+nsga2``). YAML ``weighting:`` stays the LibMTL name (e.g. ``MGDA``).
- ``optim``: e.g. ``adam``

EvoMTL configs (``evomtl_ew_nsga2`` / ``evomtl_ew_comocma`` only) set ``evo_training: true``; checkpoints use
``save_path`` (``best.pt``, ``evo_result.pt``, ``model_evo.pt``).  Other MOO methods
have gradient-only configs.

Run from any directory; paths are resolved relative to this script's location.
"""

import shutil
from pathlib import Path

# ── seeds (mirrors the existing parallel runner scripts) ──────────────────────
SEEDS = [1, 12, 123, 1234, 12345, 123456]

# ── datasets: folder name → value for ``dataset:`` in YAML ───────────────────
# Folder name → ``dataset:`` for ``MultiMNISTDataset(..., dataset=...)`` (see mnist.py)
DATASET_META = {
    "mnist": "mnist",
    "fashion": "fashion",
    "mnist_and_fashion": "fashion_and_mnist",
}

OPTIM = "adam"


def _evomtl_segment(algo: str, moea_tag: str) -> str:
    """EvoMTL W&B group / run-name prefix, e.g. ``evomtl-ew+nsga2``."""
    return f"evomtl-{algo.lower()}+{moea_tag}"


def _evomtl_path_segment(algo: str, moea_tag: str) -> str:
    """EvoMTL ``save_path`` directory segment (no ``+``)."""
    return f"evomtl-{algo.lower()}_{moea_tag}"


# If True, also writes ``evomtl_ew_nsga2`` and ``evomtl_ew_comocma`` trees (EvoMTL; +36 configs).
GENERATE_EVO_VARIANTS = True

# Which weighting methods get EvoMTL variant folders (nsga2 / comocma); EW-only.
EVOMTL_WEIGHTINGS = frozenset({"EW"})

# ── algorithms with their algorithm-specific YAML block ───────────────────────
# Each entry: (weighting_name, extra_lines)
ALGORITHMS = {
    "EW": (
        "EW – Equal Weighting (baseline)",
        "Kendall et al., CVPR 2018",
        "Assigns equal weight to all task losses; serves as the standard MTL baseline.",
        "# EW — no LibMTL weight_args (equal task weights)",
    ),
    "MGDA": (
        "MGDA – Multiple Gradient Descent Algorithm",
        "Sener & Koltun, NeurIPS 2018",
        "Solves the MOO problem exactly by projecting onto the minimum-norm point of the gradient convex hull.",
        "mgda_gn: none          # gradient normalisation: none | l2 | loss | loss+",
    ),
    "PCGrad": (
        "PCGrad – Projecting Conflicting Gradients",
        "Yu et al., NeurIPS 2020",
        "Projects each task gradient onto the normal plane of conflicting task gradients.",
        "# PCGrad — no LibMTL weight_args",
    ),
    "GradVac": (
        "GradVac – Gradient Vaccine",
        "Wang et al., ICLR 2021",
        "Improves multi-task optimisation by encouraging gradient alignment between tasks.",
        "GradVac_beta: 0.5          # EMA coefficient for cosine similarity tracking\n"
        "GradVac_group_type: 0      # parameter granularity: 0=whole_model | 1=all_layer | 2=all_matrix",
    ),
    "CAGrad": (
        "CAGrad – Conflict-Averse Gradient",
        "Liu et al., NeurIPS 2021",
        "Minimises the worst-case loss improvement by finding a gradient that lies in a trust-region ball.",
        "calpha: 0.5    # conflict-aversion radius (c in the paper)\n"
        "rescale: 1     # rescale final gradient: 0=no | 1=by mean norm | 2=by max norm",
    ),
    "Nash_MTL": (
        "Nash_MTL – Nash Bargaining for Multi-Task Learning",
        "Navon et al., ICML 2022",
        "Frames MTL as a cooperative bargaining game and finds the Nash bargaining solution.",
        "update_weights_every: 1    # frequency (in steps) at which Nash weights are recomputed\n"
        "optim_niter: 20            # inner Frank-Wolfe iterations per weight update\n"
        "max_norm: 1.0              # gradient clipping norm for the Frank-Wolfe step",
    ),
    "IMTL": (
        "IMTL – Impartial Multi-Task Learning",
        "Liu et al., ICLR 2021",
        "Achieves impartiality by equalising the projected gradient magnitudes across tasks.",
        "# IMTL — no LibMTL weight_args",
    ),
    "MoCo": (
        "MoCo – Multi-Objective Continual Learning via Online Optimisation",
        "Fernando et al., NeurIPS 2023",
        "Tracks a Pareto-stationary solution via EMA-based gradient correction.",
        "MoCo_beta: 0.5            # EMA coefficient for gradient mean estimation\n"
        "MoCo_beta_sigma: 0.5      # EMA coefficient for gradient variance estimation\n"
        "MoCo_gamma: 0.1           # step size for the dual-variable update\n"
        "MoCo_gamma_sigma: 0.5     # EMA coefficient for variance of dual variables\n"
        "MoCo_rho: 0.0             # regularisation strength (0 = no regularisation)",
    ),
    "Aligned_MTL": (
        "Aligned_MTL – Aligned Multi-Task Learning",
        "Senushkin et al., CVPR 2023",
        "Aligns task gradients in a common subspace to prevent destructive interference.",
        "# Aligned_MTL — no LibMTL weight_args",
    ),
    "ExcessMTL": (
        "ExcessMTL – Robust Multi-Task Learning with Excess Risks",
        "He et al., ICML 2024",
        "Task weights from accumulated squared gradients; scaled by robust_step_size.",
        "robust_step_size: 1.0e-2   # step size for the robust weight update (LibMTL default)",
    ),
    "FairGrad": (
        "FairGrad – Fair Gradient Descent for Multi-Task Learning",
        "Mahapatra & Rajan, ICML 2022",
        "Ensures each task receives a fair share of gradient updates via alpha-fairness.",
        "FairGrad_alpha: 1.0    # fairness exponent α (1.0 = proportional fairness; ∞ = max-min fairness)",
    ),
    "FAMO": (
        "FAMO – Fast Adaptive Multitask Optimisation",
        "Liu et al., NeurIPS 2023",
        "Maintains per-task loss weights that converge to a balanced Pareto solution in O(1) memory.",
        "FAMO_w_lr: 0.025       # learning rate for the per-task weight update\n"
        "FAMO_w_gamma: 1.0e-3   # L2 regularisation on task weights (prevents weight collapse)",
    ),
    "MoDo": (
        "MoDo – Multi-Objective Descent with a Single-Loop Algorithm",
        "Qiu et al., ICML 2024",
        "Finds a Pareto-stationary point via a single-loop primal-dual update.",
        "MoDo_gamma: 1.0e-3   # dual-variable step size\n"
        "MoDo_rho: 0.1        # proximal regularisation coefficient",
    ),
    "DB_MTL": (
        "DB_MTL – Dual-buffer gradient aggregation",
        "LibMTL (see LibMTL.weighting.DB_MTL)",
        "Maintains per-task gradient buffers with step-weighted mixing; rep_grad must be false.",
        "DB_beta: 0.9            # mixing strength toward the new batch gradient (LibMTL default)\n"
        "DB_beta_sigma: 0      # exponent on 1/step in the mixing coefficient (LibMTL default)",
    ),
    "SDMGrad": (
        "SDMGrad – Stochastic Direction-oriented Multitask Gradient",
        "Chen et al., ICML 2023",
        "Uses random direction-oriented gradient aggregation to approximate the Pareto front.",
        "SDMGrad_lamda: 0.3     # mixing coefficient between the two sampled gradient directions\n"
        "SDMGrad_niter: 20      # inner iterations to solve the direction-finding QP",
    ),
    "UPGrad": (
        "UPGrad – Unconflicting Projection of Gradients",
        "Rey & Quinton; TorchJD (see LibMTL.weighting.UPGrad)",
        "Aggregates task gradients via a QP that enforces non-negative weights and reduces conflict.",
        "UPGrad_norm_eps: 0.0001   # epsilon when normalising gradients\n"
        "UPGrad_reg_eps: 0.0001    # diagonal regulariser on the Gram matrix",
    ),
    "STCH": (
        "STCH – Smooth Tchebycheff Scalarization for Multi-Objective Optimization",
        "Xi et al., ICML 2024",
        "Smooth Tchebycheff scalarization for multi-task learning (warmup then STCH objective).",
        "STCH_mu: 1.0              # μ smoothing parameter (LibMTL default)\n"
        "STCH_warmup_epoch: 4      # epochs of log-loss sum before switching to STCH",
    ),
    "MOML": (
        "MOML – Multi-Objective Meta Learning",
        "Ye et al.; NeurIPS 2021 / AIJ 2024",
        "Bilevel optimisation of task weights; this LibMTL build supports inner_step=1 only.",
        "outer_lr: 1.0e-3        # outer-loop LR for task weights (LibMTL default)\n"
        "inner_lr: 0.1           # inner LL step (LibMTL default)\n"
        "inner_step: 1           # required (fast MOML implementation in LibMTL)",
    ),
    "FORUM": (
        "FORUM – First-order multi-gradient for bi-level MTL",
        "ECAI 2024",
        "First-order multi-gradient steps for bi-level multi-task learning.",
        "outer_lr: 1.0e-3        # outer LR (LibMTL default)\n"
        "inner_lr: 0.1           # inner SGD on shared parameters (LibMTL default)\n"
        "inner_step: 5           # inner refinement steps (LibMTL default)\n"
        "FORUM_phi: 0.1          # mixing coefficient φ (LibMTL default)",
    ),
    "AutoLambda": (
        "AutoLambda – Disentangling dynamic task relationships",
        "Liu et al., TMLR 2022",
        "Meta-learns task weights via approximate hypergradient; inner_step must be 1 in LibMTL.",
        "outer_lr: 1.0e-3        # meta LR for task weights (LibMTL default)\n"
        "inner_step: 1           # required for this LibMTL implementation",
    ),
}

def _evo_block_nsga2(seed: int) -> str:
    return f"""
# EvoMTL (MOEA after supervised training; NSGA-II)
evo_training: true
evo_moea: nsga2
evo_ps: spherical_lora
evo_ps_alpha: 1.0
evo_ps_r: 4
evo_ps_seed: 42
evo_iterations: 30
evo_pop_size: 24
evo_z_lower: -3.0
evo_z_upper: 3.0
evo_n_eval_batches: 1
evo_seed: {seed}
evo_hv_center_aggregation: linear
evo_hv_softmax_temperature: 1.0
"""


def _evo_block_comocma(seed: int) -> str:
    return f"""
# EvoMTL (MOEA after supervised training; COMO-CMA-ES; two tasks only)
evo_training: true
evo_moea: comocma
evo_ps: spectral_all_svd
evo_ps_alpha: 1.0
evo_ps_r: 4
evo_ps_seed: 42
evo_iterations: 30
evo_pop_size: 24
evo_z_lower: -1.0
evo_z_upper: 1.0
evo_num_kernels: 10
evo_sigma0: 0.3
evo_n_eval_batches: 1
evo_seed: {seed}
evo_cma_tolx: 0.0
evo_cma_tolfun: 0.0
evo_cma_tolfunrel: 0.0
evo_cma_tolfunhist: 0.0
evo_cma_tolstagnation: 0
evo_cma_tolflatfitness: 1e9
# COMO-CMA reference points: [] = defaults (Sofomore: 2*probe_loss+1; pymoo HV: nadir+1e-6).
evo_sofomore_reference_point: []
evo_hv_ref_point: []
evo_hv_center_aggregation: linear
evo_hv_softmax_temperature: 1.0
"""


def _template(
    *,
    title: str,
    ref: str,
    desc: str,
    seed: int,
    weighting: str,
    dataset_arg: str,
    dataset_key: str,
    moo_segment: str,
    optim: str,
    wandb_name: str,
    wandb_group: str,
    extra: str,
    evo_block: str = "",
) -> str:
    save_path = f"logs/{dataset_key}/{moo_segment}/{optim}/seed_{seed}/"
    extra_block = extra.rstrip() + "\n"
    return f"""\
# {title}
# {ref}
# {desc}

# General
mode: train
seed: {seed}
gpu_id: "0"

# Weights & Biases
wandb_entity: rasa_research
wandb_project: mtl
wandb_name: {wandb_name}
wandb_group: {wandb_group}

# Architecture
weighting: {weighting}
{extra_block}arch: HPS
rep_grad: false
multi_input: false

# Dataset
dataset: {dataset_arg}
dataset_path: .               # root of data/ folder (relative to examples/multimnist/)
num_tasks: 2
train_bs: 256
test_bs: 10000
epochs: 100

# Optimizer (Adam)
optim: {optim}
lr: 1.0e-4
weight_decay: 5.0e-4
adam_beta1: 0.9
adam_beta2: 0.999
adam_eps: 1.0e-8
amsgrad: false
{evo_block}
# Paths
save_path: {save_path}
load_path: null
"""


def main():
    base = Path(__file__).parent

    # Remove previous generated trees under each dataset folder
    for ds in DATASET_META:
        ds_dir = base / ds
        if not ds_dir.is_dir():
            continue
        for child in list(ds_dir.iterdir()):
            if child.is_dir():
                shutil.rmtree(child)
            elif child.suffix == ".yaml":
                child.unlink()

    count = 0
    for dataset_key, dataset_arg in DATASET_META.items():
        for algo, (title, ref, desc, extra) in ALGORITHMS.items():
            algo_dir = base / dataset_key / algo.lower()
            algo_dir.mkdir(parents=True, exist_ok=True)
            for seed in SEEDS:
                content = _template(
                    title=title,
                    ref=ref,
                    desc=desc,
                    seed=seed,
                    weighting=algo,
                    dataset_arg=dataset_arg,
                    dataset_key=dataset_key,
                    moo_segment=algo.lower(),
                    optim=OPTIM,
                    wandb_name=f"{algo.lower()}-seed{seed}",
                    wandb_group=algo.lower(),
                    extra=extra,
                )
                out = algo_dir / f"seed{seed}.yaml"
                out.write_text(content, encoding="utf-8")
                count += 1

            if GENERATE_EVO_VARIANTS and algo in EVOMTL_WEIGHTINGS:
                for suffix, block_fn, moea_tag in (
                    ("nsga2", _evo_block_nsga2, "nsga2"),
                    ("comocma", _evo_block_comocma, "comocma"),
                ):
                    moo = f"evomtl_{algo.lower()}_{suffix}"
                    evo_dir = base / dataset_key / moo
                    evo_dir.mkdir(parents=True, exist_ok=True)
                    evomtl_seg = _evomtl_segment(algo, moea_tag)
                    path_seg = _evomtl_path_segment(algo, moea_tag)
                    for seed in SEEDS:
                        content = _template(
                            title=title,
                            ref=ref,
                            desc=desc,
                            seed=seed,
                            weighting=algo,
                            dataset_arg=dataset_arg,
                            dataset_key=dataset_key,
                            moo_segment=path_seg,
                            optim=OPTIM,
                            wandb_name=f"{evomtl_seg}-seed{seed}",
                            wandb_group=evomtl_seg,
                            extra=extra,
                            evo_block=block_fn(seed),
                        )
                        out = evo_dir / f"seed{seed}.yaml"
                        out.write_text(content, encoding="utf-8")
                        count += 1

    print(f"Generated {count} config files under {base}")


if __name__ == "__main__":
    main()
