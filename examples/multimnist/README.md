## MultiMNIST

MultiMNIST overlays MNIST digits [[1]](#1) in one image [[2]](#2). In its multi-task form [[3]](#3), one digit sits in the top-left and the other in the bottom-right of a 36×36 canvas, so the two overlap, and each task classifies one digit. This is a single-input problem, so `multi_input` must stay **off** (the default).

This example supports three versions:

- **Two tasks, default** (`--dataset mnist`, `fashion` or `fashion_and_mnist`): Sener & Koltun's construction [[3]](#3). Each MNIST (or FashionMNIST) image is the top-left item, paired with one random partner placed 6 pixels further right and down on a 36×36 canvas; the brighter pixel is kept where they overlap, and the canvas is resized to 28×28 by nearest-neighbour sampling. The 60,000 training and 10,000 test images are generated on first use with a fixed seed and stored in `dataset_path/data/MultiMNISTDataset/<dataset>/`. As in their loader, inputs are converted with `ToTensor` and normalized with mean 0.1307 and standard deviation 0.3081. In `fashion_and_mnist` the MNIST digit is top-left and the Fashion item bottom-right.
- **Two tasks, 36×36** (the same names with `-36p`, e.g. `--dataset mnist-36p`): the pickles used by Pareto MTL [[4]](#4) and later Pareto-front work such as Pareto HyperNetworks [[5]](#5): 120,000 training and 20,000 test images, each digit shifted randomly by up to 4 pixels. Pixels reach the network as raw 0–255 values, as in the Pareto MTL code.
- **Three tasks:** MultiMNIST-3 of Pareto Manifold Learning [[6]](#6): three digits on a 42×42 canvas (top-left, top-right, bottom-centre), keeping the brighter pixel where they overlap, resized to 28×28. It is generated from MNIST on first use: 60,000 training and 10,000 test images with pixels in [0, 1]. Generation uses NumPy's global random state, so a regenerated set differs from an earlier one.

The training code uses a shared LeNet-style encoder (`MultiLeNetEncoder`) and one head per task (`MultiLeNetDecoder`); with `--shared_decoder`, a single decoder serves all tasks, so every task loss is computed on the same output. The encoder's linear layer takes 320 inputs for 28×28 images and 720 for the 36×36 pickles. Evaluation is classification accuracy (higher is better) on each task.

### Dataset options

1. **Two tasks (`--num_tasks 2`)**  
   - Task 1: top-left digit  
   - Task 2: bottom-right digit  
   - Choose the source with `--dataset`: `mnist` (default), `fashion`, or `fashion_and_mnist` (you may pass `mnist+fashion`; it is normalized to `fashion_and_mnist`), each 28×28; add `-36p` for the 36×36 pickles.

2. **Three tasks (`--num_tasks 3`)**  
   - Task 1: top-left digit  
   - Task 2: bottom-center digit  
   - Task 3: top-right digit  
   - Uses MNIST only. On the first run, MNIST is downloaded to `dataset_path/data/MNIST/` and the images are generated into `dataset_path/data/MultiMNISTDataset3Digits/` (`MultiMNISTDataset3Digits` with `download=True`). If the download fails with a certificate error, run with `SSL_CERT_FILE=$(python -m certifi)`.

### Data files (two tasks)

The default 28×28 versions need no files: MNIST and FashionMNIST are downloaded to `dataset_path/data/MNIST/` and `dataset_path/data/FashionMNIST/` on first use. The `-36p` versions read pickle files under **`dataset_path/data/`**:

| `--dataset`               | File                             |
|---------------------------|----------------------------------|
| `mnist-36p`               | `multi_mnist.pickle`             |
| `fashion-36p`             | `multi_fashion.pickle`           |
| `fashion_and_mnist-36p`   | `multi_fashion_and_mnist.pickle` |

A copy of the three pickles is available on [Google Drive](https://drive.google.com/open?id=1VnmCmBAVh8f_BKJg1KYx-E137gBLXbGG); place the files so the paths above resolve. They are read with the same shapes as in the Pareto MTL code: 120,000×1×36×36 for training and 20,000×1×36×36 for testing.

### Run a model

`main.py` trains or evaluates an MTL model. Run it from this directory inside the repository's virtual environment (`source ../../.venv/bin/activate`), where LibMTL is installed as an editable package.

Important arguments:

- `config`: Optional YAML file; keys set defaults that CLI flags can override.
- `dataset`: `mnist`, `fashion`, or `fashion_and_mnist` (two tasks only). Default: `mnist`.
- `num_tasks`: `2` or `3`. Default: `2`.
- `shared_decoder`: one decoder for all tasks, so every task loss is computed on the same output. Default: off (one head per task).
- `weighting`: Task weighting / gradient method. See [Supported algorithms](../../README.md#supported-algorithms).
- `arch`: MTL architecture. See [Supported algorithms](../../README.md#supported-algorithms).
- `gpu_id`: GPU id. Default: `0`.
- `seed`: Random seed. Default: `0`.
- `scheduler`: Optional LR scheduler (e.g. `step`). This example sets the default to **no** scheduler (fixed LR) in `main.py`.
- `optim`: Optimizer type; `adam` is typical here (defaults are set in `main.py`).
- `dataset_path`: Dataset root, the directory that contains `data/`: all datasets live there (the `-36p` pickles, MNIST / FashionMNIST downloads, and the generated `MultiMNISTDataset/` and `MultiMNISTDataset3Digits/`).
- `train_bs`: Training batch size. Default: `256`.
- `test_bs`: Test batch size. Default: `10000`.
- `epochs`: Training epochs. Default: `100`.
- `mode`: `train` or `test`.

**EvoMTL:** pass `--evo_training` (and optional `--evo_*` flags) to use `EvoMTLTrainer` instead of the standard `Trainer`. Every algorithm searches the network weights `theta` directly. `evo_schedule` selects one of two training directions:

| `evo_schedule` | Training |
| --- | --- |
| `evolution` | Multi-objective evolutionary training from the initial weights, no gradient descent (`epochs: 0`). Either a Pareto set / front (NSGA-II, MOPSO-CD and COMO-CMA through `evo_moea`, or Pareto set learning with ES, `evo_moea: psl_es`) or a single network from an evolution strategy: `openes` / `guidedes` (OpenES / Guided ES on the sum of losses), `snes` / `pgpe` (per-parameter step sizes), `mo_upgrad_openes` / `mo_upgrad_guidedes` (per-objective pseudo-gradients combined by UPGrad), `mo_mean_openes` (their mean), `mo_mgda_openes` / `mo_mgda_guidedes` / `mo_mgda_openes_libmtl` (MGDA common descent direction, `evo_mgda_*` flags). One file per algorithm in `LibMTL/evomtl/moea/`. |
| `gd_then_evolution` (default) | `epochs` of multi-objective gradient descent with `weighting`, then the same evolutionary training as fine-tuning that shapes the Pareto front. |

```shell
# evolution: MO-UPGrad-OpenES, 10-epoch budget (2344 generations)
python main.py --config configs/10epochs/mnist/evomtl/mo_upgrad_openes/pop_128/adam-lr_0.001/seed1.yaml

# gd_then_evolution: 30 epochs of gradient descent (EW), then NSGA-II on the weights in the box theta_0 +- 0.01
python main.py --weighting EW --arch HPS --epochs 30 --evo_training --evo_schedule gd_then_evolution \
    --evo_moea nsga2 --evo_lb -0.01 --evo_ub 0.01 --evo_iterations 100 --evo_pop_size 100
```

The ES methods (`openes`, `mo_upgrad_openes`, `mo_mgda_openes`, `psl_es`, ...) use raw fitness by default (`evo_es_fitness_shaping: raw`): the losses minus the population mean, or the unranked pair differences for PSL. Set `evo_es_fitness_shaping: centered_ranks` for the rank-based OpenAI-ES estimator; PSL's `evo_psl_rank_groups` > 1 needs it.

See [docs/evolutionary_algorithms.md](docs/evolutionary_algorithms.md) for the methods, options, configs, outputs and plotting scripts.

Full help:

```shell
python main.py -h
```

Train (two tasks, default 28×28 MultiMNIST; use `--dataset mnist-36p` for the 36×36 pickles):

```shell
python main.py --dataset mnist --num_tasks 2 --weighting WEIGHTING --arch ARCH --dataset_path PATH --gpu_id GPU_ID --mode train --save_path PATH
```

Train (three tasks; generates data under `dataset_path` on first run):

```shell
python main.py --num_tasks 3 --weighting WEIGHTING --arch ARCH --dataset_path PATH --gpu_id GPU_ID --mode train --save_path PATH
```

Test:

```shell
python main.py --dataset mnist --num_tasks 2 --weighting WEIGHTING --arch ARCH --dataset_path PATH --gpu_id GPU_ID --mode test --load_path PATH
```

```shell
python main.py --num_tasks 3 --weighting WEIGHTING --arch ARCH --dataset_path PATH --gpu_id GPU_ID --mode test --load_path PATH
```

### Batch runs from YAML

`runner.py` runs every `*.yaml` in a folder in sorted order (see its `--help`). Example:

```shell
python runner.py --dir configs/mnist/cagrad/
```

`run_grad_aggs.py` runs the gradient-descent configs under a folder (recursively; ES configs are skipped) several at a time on macOS, Linux or Windows. It skips runs that already finished, keeps the machine awake, and logs progress and exit codes to `logs/run_grad_aggs.log` (see its `--help`):

```shell
python run_grad_aggs.py configs/10epochs/mnist --jobs 3            # all gradient-descent configs
python run_grad_aggs.py configs/10epochs/mnist --dry_run           # list what would run
```

### References

<a id="1">[1]</a> Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. In *Proceedings of the IEEE*, 1998.

<a id="2">[2]</a> Sara Sabour, Nicholas Frosst, and Geoffrey E. Hinton. Dynamic routing between capsules. In *Advances in Neural Information Processing Systems*, 2017.

<a id="3">[3]</a> Ozan Sener and Vladlen Koltun. Multi-task learning as multi-objective optimization. In *Advances in Neural Information Processing Systems*, 2018.

<a id="4">[4]</a> Xi Lin, Hui-Ling Zhen, Zhenhua Li, Qingfu Zhang, and Sam Kwong. Pareto multi-task learning. In *Advances in Neural Information Processing Systems*, 2019.

<a id="5">[5]</a> Aviv Navon, Aviv Shamsian, Gal Chechik, and Ethan Fetaya. Learning the Pareto front with hypernetworks. In *International Conference on Learning Representations*, 2021.

<a id="6">[6]</a> Nikolaos Dimitriadis, Pascal Frossard, and François Fleuret. Pareto manifold learning: Tackling multiple tasks via ensembles of single-task models. In *International Conference on Machine Learning*, 2023.

## Multi-Center Pareto-OpenES

Train a persistent set of two-head LeNet networks with vector antithetic ES,
matched batches, and constrained objective-space spreading:

```sh
# From the repository root (uv sync first)
PYTHONPATH=. uv run python examples/multimnist/main.py \
  --config examples/multimnist/configs/pareto_centers/mnist.yaml \
  --dataset_path examples/multimnist
```

See [algorithm, settings, outputs and tests](../../LibMTL/evomtl/moea/MULTI_CENTER_PARETO_OPENES.md).
The checkpoint preserves all centers; the ordinary single-model checkpoint uses
center 0. No archive or weighted task-loss scalarization is used.

For the existing 10-epoch sampled-data protocol (2,344 iterations, batch size
256), use this config instead:

```sh
PYTHONPATH=. uv run python examples/multimnist/main.py \
  --config examples/multimnist/configs/10epochs/mnist/evomtl/multi_center_pareto_openes/pop_32/sgd-lr_0.001/seed1.yaml \
  --dataset_path examples/multimnist
```

This keeps five centers, 32 probes per center and the short example's update
settings. The sampled-data budget matches the other 10-epoch configs; total
probe computation does not. Full test metrics are saved for all centers every
50 iterations, plus the baseline and final state.

To enable shared probes with center-specific sample weights, use
`configs/pareto_centers/mnist_shared_pool.yaml` with the same entry point. Every
center combines perturbation offsets from the full current probe pool; its own
vector ES Jacobian still constrains the update. The original default is kept
for controls. A corresponding 10-epoch config is under
`configs/10epochs/mnist/evomtl/multi_center_pareto_openes_shared/`.
See the algorithm guide for diagnostic checkpoint fields and the initial
matched comparison, which did not show improved spread.

## Full-network-only tangent PSL test

The fixed-width tangent variant optimizes only the full network vector theta.
It computes the steering tangent from per-task ES estimates, with no learned B,
width, or preference map. The tested setting fixes the endpoint radius at 0.1
using `evo_tpsl_c_lr: 0`, with the existing Guided ES/Adam/Chebyshev protocol:

```sh
PYTHONPATH=. uv run python examples/multimnist/main.py \
  --config examples/multimnist/configs/10epochs/mnist/evomtl/tpsl_fixed_width_guided_es-tch/pop_128/adam-lr_0.001/seed1.yaml \
  --dataset_path examples/multimnist
```

The run uses 2,344 iterations, batch size 256, 128 probes and 11 preference
readouts. All 10,000 test images are evaluated in batches of 1,000. Check
`tpsl_result.pt` for `width_learned: false`, `optimized_num_parameters`, the
computed tangent T, and held-out front values. The stored B is derived as
`c*T.T` for compatibility, not an independently learned parameter. Gradient
averages and tangent/guide vectors still consume optimizer-state memory.

```sh
uv run python -m unittest discover -s tests -p 'test_tpsl_fixed_width.py'
```

The tests verify the optimizer's parameter count, constant radius, derived
steering matrix, endpoint geometry, finite front losses, actual network updates,
and the separate scalar optimizer in the learned-width control.

The seed-1 fixed-width test completed all 2,344 iterations. Only the 22,350 full
network coordinates were optimized; radius c stayed exactly 0.1 throughout.
The balanced network achieved 81.34% / 77.46% accuracy and 0.59914 / 0.70398
cross-entropy on all 10,000 test images. Eight of eleven preference readouts
were nondominated in test losses (lambda_1 from 0.3 through 1.0). Within that
nondominated subset, loss spans were 0.01101 / 0.00323 and accuracy spans about
0.62 / 0.37 percentage points. Thus the method learned a useful model and a
local trade-off curve without learning a steering matrix or width, but it did
not demonstrate broad coverage. The computed front varied across checkpoints.
These are one-seed results; no matched learned-B baseline or low-rank decoder
variant was run in this test. Three targeted fixed-width tests passed.
