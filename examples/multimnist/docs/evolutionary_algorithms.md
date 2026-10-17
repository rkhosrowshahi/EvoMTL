# Evolutionary algorithms (EvoMTL) on MultiMNIST

EvoMTL trains a multi-task network with evolutionary algorithms instead of (or alongside) gradient descent. The network weights are written as

```text
theta = theta_0 + adapter(z)
```

and an evolutionary algorithm searches the code `z`. With the `full` adapter, `z` is the whole weight change (search directly in weight space); the other adapters map a low-dimensional `z` onto the weights.

Two families of algorithms are available:

- **Evolution strategies (ES)**: OpenES-style methods that estimate a search gradient from a population and step with an optimizer (SGD, Adam or RMSprop). Single-objective, multi-objective (Jacobian descent) and Pareto-set-learning variants.
- **Multi-objective EAs (MOEAs)**: NSGA-II, MOPSO-CD and COMO-CMA-ES, which keep a population on the Pareto front and return a hypervolume-weighted "Pareto-center" solution.

Contents: [Setup](#setup) · [Running](#running) · [Methods](#methods) · [Options](#options) · [Outputs](#outputs) · [Configs](#configs) · [Plots and analysis](#plots-and-analysis) · [Adapter dimensions](#adapter-dimensions)

## Setup

Use the virtual environment at the repository root. LibMTL is installed into it as an editable package, so code changes take effect without reinstalling and no `PYTHONPATH` is needed.

```bash
cd /path/to/EvoMTL
source .venv/bin/activate
```

To recreate it from scratch (macOS, CPU):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision numpy scipy matplotlib pillow tqdm pyyaml wandb \
            cvxpy ecos qpsolvers torch_geometric pymoo cma comocma
pip install -e . --no-deps
```

`requirements.txt` pins CUDA builds (`+cu121`) for the GPU machines and does not install on macOS.

## Running

Runs are launched from the example folder with a YAML config. Every key in the YAML is a command-line option; flags given on the command line override the YAML.

```bash
cd examples/multimnist
python main.py --config configs/no_scheduler/mnist/evomtl/psl_guided_es-tch-full/pop_128/adam-lr_0.001/seed1_s0.05_a0.9_ema0.9_gen1000.yaml
```

- **Log file**: the run mirrors everything it prints (stdout and stderr) into `<save_path>/stdout.log`. Do not redirect output to that file yourself.
- **Weights & Biases**: off by default. Pass `--wandb_enable` to log (the config's `wandb_project` must also be set).
- **Threads**: leave `OMP_NUM_THREADS` unset on Apple silicon. PyTorch's default (one thread per performance core) was faster than 8 threads (0.84 vs 1.13 s per generation).
- **Several configs**: `python runner.py --dir <config folder>` runs every YAML in the folder in alphabetical order; extra flags are forwarded to every run.

Typical cost on a laptop CPU (MultiMNIST, population 128, full adapter): about 0.85 s per generation, so 1000 generations take roughly 15 minutes plus test evaluations.

### Schedules

`evo_training: true` switches `main.py` to the EvoMTL trainer. `evo_schedule` sets the phase order:

| `evo_schedule` | What happens |
| --- | --- |
| `gd_then_moea` (default) | `epochs` of gradient descent, then the evolutionary search starting from those weights. |
| `moea_then_gd` | Evolutionary search from the initial weights, then `epochs` of gradient descent. With `epochs: 0` the run is pure evolution; all ES experiments use this. |

## Methods

`evo_moea` selects the algorithm. The ES methods all use `evo_moea: mo_openes` or `psl_es`; the remaining options choose the variant.

| Method | `evo_moea` | Distinguishing options | Config / log folder |
| --- | --- | --- | --- |
| **OpenES** | `mo_openes` | `evo_es_aggregator: ws` | `openes-ws-full` |
| **GuidedES** | `mo_openes` | `evo_es_aggregator: ws`, `evo_es_guided_k: 1` | `guided_es-ws-full` |
| **MO-UPGrad-OpenES** | `mo_openes` | `evo_es_aggregator: upgrad` | `mo_openes-upgrad-full` |
| **MO-UPGrad-GuidedES** | `mo_openes` | `evo_es_aggregator: upgrad`, `evo_es_guided_k: 1` | `mo_guided_es-upgrad-full` |
| **MO-Mean-OpenES** | `mo_openes` | `evo_es_aggregator: mean` | `mo_openes-mean-full` |
| **MO-PSL-TCH-OpenES** | `psl_es` | `evo_psl_scalarization: tch` | `psl_openes-tch-full` |
| **MO-PSL-TCH-GuidedES** | `psl_es` | `evo_psl_scalarization: tch`, `evo_es_guided_k: 1` | `psl_guided_es-tch-full` |
| NSGA-II | `nsga2` | `evo_crossover_*`, `evo_mutation_*` | |
| MOPSO-CD | `mopso` | `evo_mopso_*` | |
| COMO-CMA-ES (two tasks) | `comocma` | `evo_cma_*` | |

How the ES variants differ:

- **OpenES**: one fitness per member, the sum of the task losses. Members are ranked, the centered ranks weight the noise vectors, and the result is the search gradient.
- **MO-…-OpenES (Jacobian descent)**: each task loss is ranked separately, giving one search gradient per task (a Jacobian). `upgrad` projects each row onto the dual cone of all rows and averages the projections, so the step does not increase any task loss to first order; `mean` averages the rows without projection (ablation).
- **…-GuidedES**: part of the sampling noise is drawn along a guiding direction, a moving average of past ES steps, instead of isotropically (Guided ES, Maheswaranathan et al. 2019).
- **MO-PSL-…**: Pareto set learning. One run learns a line of networks `theta(lam) = theta_0 + z + (lam_1 - lam_2) B`, one per preference `lam`. Each antithetic pair draws its own preference and is scored with the Chebyshev scalarization `max_k lam_k l_k` (`tch`), the weighted sum (`ws`) or the hypervolume score (`hv`: how far the losses sit inside the box of the hypervolume reference point, measured along the preference direction; averaged over uniformly spread directions this is the hypervolume itself). The returned solution is the balanced network (`lam = 0.5, 0.5`); the whole front is evaluated on a grid of preferences. Optionally the update is split (`evo_psl_center_a`, `evo_psl_spread_score`): `z` follows the near-balanced preferences and `B` follows its own scores (Chebyshev from an ideal point near the front, or the hypervolume score), using the same evaluations.

## Options

All options are `--evo_*` flags (or keys in the YAML). The full list with help text is in `python main.py -h`.

### Common

| Option | Default | Meaning |
| --- | --- | --- |
| `evo_iterations` | 30 | Generations. |
| `evo_pop_size` | 24 | Population size. Must be even for antithetic ES sampling; use powers of 2. |
| `evo_adapter` | `spherical_lora` | How `z` maps to weights. `full` = search all weights directly (used for all ES runs). |
| `evo_adapter_alpha` | 1.0 | Scale on the adapter output. |
| `evo_num_batches` | 1 | Training mini-batches per fitness evaluation (all members share them). |
| `evo_eval_freq` | 1 | Test-set evaluation every N generations (and at the last one); 0 disables. |
| `evo_seed` | 0 | Seed of the evolutionary search. |
| `evo_hv_ref_point` | auto | Hypervolume reference point. ES runs use `[2.302585, 2.302585]`, the chance-level loss ln 10 per task. |

### Evolution strategies (`mo_openes`, `psl_es`)

| Option | Default | Meaning |
| --- | --- | --- |
| `evo_es_optim` | `adam` | Optimizer on the search mean: `sgd`, `adam` or `rmsprop`. |
| `evo_es_lr` | 1e-3 | Its learning rate. |
| `evo_es_adam_beta1/beta2/eps`, `evo_es_rmsprop_alpha/eps` | standard | Optimizer settings. |
| `evo_es_sigma` | 0.01 | Noise standard deviation. |
| `evo_es_sigma_decay`, `evo_es_sigma_min` | 1.0, 0.0 | Multiply sigma by the decay after each generation, down to the minimum. |
| `evo_es_antithetic` | on | Mirrored sampling (`eps`, `-eps`). Required for PSL. |
| `evo_es_aggregator` | `upgrad` | `mo_openes` only: `ws`, `mean` or `upgrad` (see [Methods](#methods)). |
| `upgrad_norm_eps`, `upgrad_reg_eps` | 1e-4 | UPGrad numerical settings. |
| `evo_es_grad_cos_freq` | 1 | Diagnostic: cosine between the ES direction and the backprop gradient every N generations; 0 disables. |
| `evo_es_grad_save_freq` | 0 | Diagnostic: save true and ES gradients to `gradients.pth.tar` every N generations; 0 disables. |

Guided sampling:

| Option | Default | Meaning |
| --- | --- | --- |
| `evo_es_guided_k` | 0 | Number of guiding directions; 0 = plain OpenES. |
| `evo_es_guided_alpha` | 0.5 | Fraction of the noise kept isotropic (1 = plain OpenES). |
| `evo_es_guided_sigma` | 0 | Noise size along the guiding directions; 0 = `sigma * sqrt((1 - alpha) D / k)`. |
| `evo_es_guided_ema` | 0 | If > 0, the guide is a moving average of past ES steps with this decay; otherwise the last `k` steps. |

The tuned guided setting used so far is `k = 1, alpha = 0.9, sigma = 0.05, ema = 0.9` (file suffix `s0.05_a0.9_ema0.9`).

Pareto set learning (`psl_es`):

| Option | Default | Meaning |
| --- | --- | --- |
| `evo_psl_scalarization` | `tch` | Score of a member under its preference: `tch` (Chebyshev), `ws` (weighted sum) or `hv` (hypervolume score from `evo_hv_ref_point`; two tasks; use with `evo_psl_rank_groups` > 1). |
| `evo_psl_params` | `all` | Parameters that depend on the preference: `all`, or comma-separated name prefixes such as `encoder.fc,decoders`. |
| `evo_psl_eval_points` | 11 | Preferences on which the front is evaluated (11: `lam_1 = 0, 0.1, …, 1`). |
| `evo_psl_center_a` | 1.0 | Update of the shared delta `z` (the balanced network): weight each pair by the Beta(a, a) density of its `lam_1`, so the balanced network follows the near-balanced preferences. 1 = equal weights (original algorithm); 4 puts 80% of the weight on `lam_1` in [0.28, 0.72]. |
| `evo_psl_spread_score` | `same` | Scores ranked for the update of the steering matrix `B`: `same` (the ranks of `z`), `ideal` (Chebyshev from each task's lowest loss in the population on the batch, minus the margin, so off-centre preferences keep a trade-off with their weaker task) or `hv` (hypervolume score). |
| `evo_psl_ideal_margin` | 0.1 | Margin below the population's lowest losses for `evo_psl_spread_score: ideal`. |
| `evo_psl_guide` | `update` | Guiding subspace of Guided ES sampling for PSL: `update` (a moving average of past updates) or `tasks` (per-task directions estimated in three regions of the preference simplex, smoothed with `evo_es_guided_ema`, each placed in `z` and in `B`: 12 directions for two tasks). |
| `evo_psl_rank_groups` | 1 | Rank the pair differences within this many groups of neighbouring preferences instead of across all pairs; use 8 with the `hv` score, whose scale grows near the ends. |

### Multi-objective EAs

| Option | Used by | Meaning |
| --- | --- | --- |
| `evo_lb`, `evo_ub` | all | Box bounds on `z` (default ±3). |
| `evo_crossover_prob/eta`, `evo_mutation_prob/eta` | NSGA-II | SBX crossover and polynomial mutation. |
| `evo_mopso_w/c1/c2`, `evo_mopso_max_velocity_rate`, `evo_mopso_archive_size` | MOPSO-CD | Swarm settings. |
| `evo_cma_num_kernels`, `evo_cma_sigma0`, `evo_cma_tol*` | COMO-CMA-ES | Number of Pareto points, initial step size, stopping tolerances. |
| `evo_sofomore_reference_point` | COMO-CMA-ES | Reference point; auto = 2 × initial loss + 1 per task. |
| `evo_hv_center_aggregation`, `evo_hv_softmax_temperature` | all MOEAs | How the Pareto-center solution weights front points by hypervolume contribution (`linear` or `softmax`). |

## Outputs

Everything is written to `save_path`:

| File | Contents |
| --- | --- |
| `stdout.log` | Full console output, including the run settings (`evo_kwargs: {...}` line). |
| `metrics.csv` | One row per generation: time, sigma, population mean loss, train and test loss/accuracy per task, test hypervolume. ES runs only; for PSL the row describes the balanced network. |
| `front.csv` | PSL only: loss and accuracy of every grid preference at every test evaluation (train batch and test set). |
| `psl_result.pt` | PSL only: final search state and front. |
| `evo_result.pt` | Final search result (solution, Pareto front, settings). |
| `model_evo.pt` | Model weights after the evolutionary phase. |
| `gradients.pth.tar` | Only with `evo_es_grad_save_freq > 0`. |

## Configs

ES configs for MultiMNIST live under

```text
examples/multimnist/configs/no_scheduler/<dataset>/evomtl/<method folder>/pop_<N>/<optim>-lr_<lr>/seed<seed>[_<variant>].yaml
```

and write logs to the matching path under `examples/multimnist/logs/`. Variants (for example `_gen1000` or the guided suffix `_s0.05_a0.9_ema0.9`) also get their own log folder, so runs never overwrite each other.

`configs/generate_mo_openes_configs.py` writes OpenES and MO-OpenES configs for grids of population sizes, optimizers and learning rates (`--help` for options). Guided and PSL configs are written by hand from an existing one.

Conventions:

- Population sizes are powers of 2 (128, 256, …, 4096).
- Use seed 1 while exploring; run the full seed set (1, 12, 123, 1234, 12345, 123456) only for final settings.

## Plots and analysis

Scripts in `examples/multimnist/scripts/`, run from `examples/multimnist`:

| Script | Purpose |
| --- | --- |
| `plot_mo_openes.py` | Convergence curves (test loss, accuracy, hypervolume, cosine diagnostics) for one or more ES runs. |
| `plot_psl_front.py` | PSL fronts in test-loss space, front evolution over generations, and comparison with the other ES methods' final networks; writes `front_summary.csv`. |
| `pca_gradients.py` | PCA of the search trajectory with true vs ES gradient directions (needs `gradients.pth.tar`). |
| `loss_landscape.py` | Task losses on the PCA plane of the trajectory, drawn under `pca_gradients.py`'s plot. |
| `logs_to_csv.py` | Rebuild `metrics.csv` from `stdout.log` for old runs that predate the CSV writer. |
| `plot_samples.py` | Sample MultiMNIST images with their labels. |

Figures for a single run go into its log folder; comparisons of several runs go into `examples/multimnist/figures/comparison/<name>/`.

## Adapter dimensions

Search-space size for the MultiMNIST LeNet (42,350 parameters) with the `flatten_lora` adapter:

| Rank | `z` dimension | conv | linear | other |
| --- | --- | --- | --- | --- |
| 16 | 19,220 | 4,880 | 14,310 | 30 |
| 8 | 9,660 | 2,440 | 7,190 | 30 |
| 1 | 1,295 | 305 | 960 | 30 |

With `full`, `z` has all 42,350 dimensions.
