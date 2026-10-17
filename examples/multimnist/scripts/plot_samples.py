"""Plot sample MultiMNIST images with their labels (two digits: top-left, bottom-right; three digits:
top-left, bottom-center, top-right).

Usage (from examples/multimnist)::

    python scripts/plot_samples.py                   # first 8 training images, MNIST pickle
    python scripts/plot_samples.py --dataset fashion --start 100
    python scripts/plot_samples.py --sample_seed 1    # 8 random images -> samples_1.png
    python scripts/plot_samples.py --num_tasks 3      # three-digit images -> figures/dataset_samples/mnist3/
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

EXAMPLE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, EXAMPLE_DIR)
from mnist import MultiMNISTDataset, MultiMNISTDataset3Digits  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="mnist",
                   help="mnist, fashion or fashion_and_mnist (28x28, Sener & Koltun), or with -36p (36x36 pickles)")
    p.add_argument("--num_tasks", type=int, default=2, choices=(2, 3), help="2: two items; 3: three-digit MNIST")
    p.add_argument("--start", type=int, default=0, help="index of the first image (without --sample_seed)")
    p.add_argument("--sample_seed", type=int, default=None,
                   help="draw random training images with this seed instead of consecutive ones from --start")
    p.add_argument("--rows", type=int, default=2)
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--scale", type=float, default=1.0, help="image size as a fraction of its panel (<1 adds margins)")
    p.add_argument("--out", default=None, help="output file stem (default: figures/dataset_samples/<dataset>/samples[_<sample_seed>])")
    args = p.parse_args()

    if args.num_tasks == 3 and args.dataset != "mnist":
        raise SystemExit("three-digit images exist for --dataset mnist only")
    folder = args.dataset if args.num_tasks == 2 else "mnist3"
    if args.out is None:
        stem = "samples" if args.sample_seed is None else f"samples_{args.sample_seed}"
        args.out = os.path.join(EXAMPLE_DIR, "figures", "dataset_samples", folder, stem)
    if args.num_tasks == 2:
        ds = MultiMNISTDataset(root=EXAMPLE_DIR, dataset=args.dataset, train=True)
    else:  # generated on first use (needs the MNIST download)
        ds = MultiMNISTDataset3Digits(root=EXAMPLE_DIR, dataset="mnist", train=True, download=True)
    fig, axes = plt.subplots(args.rows, args.cols, figsize=(2.1 * args.cols, (1.9 + 0.43 * args.num_tasks) * args.rows),
                             layout="constrained")
    n = args.rows * args.cols
    if args.sample_seed is None:
        indices = range(args.start, args.start + n)
    else:
        indices = np.random.default_rng(args.sample_seed).choice(len(ds), size=n, replace=False)
    for ax, i in zip(axes.flat, indices):
        img = ds.data[i].squeeze().numpy()
        labels = [int(v) for v in ds.targets[i]]
        # draw the image at --scale of the panel, centred, with a thin frame
        h, w = img.shape
        pad = (1 / args.scale - 1) / 2
        ax.imshow(img, cmap="gray", extent=(0, w, h, 0))
        ax.add_patch(Rectangle((0, 0), w, h, fill=False, edgecolor="0.4", linewidth=0.8))
        ax.set_xlim(-pad * w, w * (1 + pad))
        ax.set_ylim(h * (1 + pad), -pad * h)
        ax.axis("off")
        ax.set_title("\n".join(f"{t}: {v}" for t, v in zip(ds.task_names, labels)), fontsize=14, pad=3)
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.02)
    base = args.dataset.removesuffix("-36p")
    name = {"mnist": "MultiMNIST", "fashion": "MultiFashion", "fashion_and_mnist": "MultiMNIST+Fashion"}[base]
    name += " (three digits)" if args.num_tasks == 3 else ""
    fig.suptitle(f"{name} images, {img.shape[0]}x{img.shape[1]} pixels", fontsize=17)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for ext in (".png", ".pdf"):
        fig.savefig(args.out + ext, dpi=120, bbox_inches="tight")
    print("wrote", args.out + ".png", "and .pdf")


if __name__ == "__main__":
    main()
