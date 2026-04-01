## MultiMNIST

The MultiMNIST dataset, originally proposed by Dimitriadis et al. [[2]](#2), is a multi-task learning benchmark created by overlaying pairs of MNIST digits [[1]](#1). Each image contains two (or three) overlapping digits, and the task is to classify each digit. This is a single-input problem, so `multi_input` must stay **off** (the default).

In this example, two-task data loaded from pickles uses \(1 \times 36 \times 36\) images; the three-task synthetic pipeline uses \(1 \times 28 \times 28\) images after generation (see `mnist.py`).

The training code uses a shared LeNet-style encoder (`MultiLeNetEncoder`) and task-specific heads (`MultiLeNetDecoder`). Evaluation is classification accuracy (higher is better) on each task.

### Dataset options

1. **Two tasks (`--num_tasks 2`)**  
   - Task 1: top-left digit  
   - Task 2: bottom-right digit  
   - Choose the source with `--dataset`: `mnist` (default), `fashion`, or `fashion_and_mnist` (you may pass `mnist+fashion`; it is normalized to `fashion_and_mnist`).

2. **Three tasks (`--num_tasks 3`)**  
   - Task 1: top-left digit  
   - Task 2: bottom-center digit  
   - Task 3: top-right digit  
   - Uses MNIST only; data is generated under `dataset_path` on first run (`MultiMNISTDataset3Digits` with `download=True`).

### Data files (two tasks)

With `download=False`, `MultiMNISTDataset` expects legacy pickle files under **`dataset_path/data/`**:

| `--dataset`           | File                         |
|-----------------------|------------------------------|
| `mnist`               | `multi_mnist.pickle`         |
| `fashion`             | `multi_fashion.pickle`       |
| `fashion_and_mnist`   | `multi_fashion_and_mnist.pickle` |

You can obtain compatible archives from the paper’s release, e.g. [Google Drive](https://drive.google.com/open?id=1VnmCmBAVh8f_BKJg1KYx-E137gBLXbGG), and place the contents so the paths above resolve.

### Run a model

`main.py` trains or evaluates an MTL model. Run from this directory with the project root on `PYTHONPATH` (for example after `pip install -e .` from the repo root), or set `PYTHONPATH` to the repository root.

Important arguments:

- `config`: Optional YAML file; keys set defaults that CLI flags can override.
- `dataset`: `mnist`, `fashion`, or `fashion_and_mnist` (two tasks only). Default: `mnist`.
- `num_tasks`: `2` or `3`. Default: `2`.
- `weighting`: Task weighting / gradient method. See [Supported algorithms](../../README.md#supported-algorithms).
- `arch`: MTL architecture. See [Supported algorithms](../../README.md#supported-algorithms).
- `gpu_id`: GPU id. Default: `0`.
- `seed`: Random seed. Default: `0`.
- `scheduler`: Optional LR scheduler (e.g. `step`). This example sets the default to **no** scheduler (fixed LR) in `main.py`.
- `optim`: Optimizer type; `adam` is typical here (defaults are set in `main.py`).
- `dataset_path`: Dataset root: for two tasks with pickles, the directory that contains the `data/` folder; for three tasks, where generated `MultiMNISTDataset3Digits` caches live.
- `train_bs`: Training batch size. Default: `256`.
- `test_bs`: Test batch size. Default: `10000`.
- `epochs`: Training epochs. Default: `100`.
- `mode`: `train` or `test`.

**EvoMTL:** pass `--evo_training` (and optional `--evo_*` flags) to use `EvoMTLTrainer` instead of the standard `Trainer`.

Full help:

```shell
python main.py -h
```

Train (two tasks, MNIST pickles):

```shell
python main.py --dataset mnist --num_tasks 2 --weighting WEIGHTING --arch ARCH --dataset_path PATH_TO_ROOT_WITH_data --gpu_id GPU_ID --mode train --save_path PATH
```

Train (three tasks; generates data under `dataset_path` on first run):

```shell
python main.py --num_tasks 3 --weighting WEIGHTING --arch ARCH --dataset_path PATH --gpu_id GPU_ID --mode train --save_path PATH
```

Test:

```shell
python main.py --dataset mnist --num_tasks 2 --weighting WEIGHTING --arch ARCH --dataset_path PATH_TO_ROOT_WITH_data --gpu_id GPU_ID --mode test --load_path PATH
```

```shell
python main.py --num_tasks 3 --weighting WEIGHTING --arch ARCH --dataset_path PATH --gpu_id GPU_ID --mode test --load_path PATH
```

### Batch runs from YAML

`runner.py` runs every `*.yaml` in a folder in sorted order (see its `--help`). Example:

```shell
python runner.py --dir configs/mnist/cagrad/
```

### References

<a id="1">[1]</a> Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. In *Proceedings of the IEEE*, 1998.

<a id="2">[2]</a> Nikolaos Dimitriadis, Pascal Frossard, and François Fleuret. Pareto manifold learning: Tackling multiple tasks via ensembles of single-task models. In *International Conference on Machine Learning*, 2023.
