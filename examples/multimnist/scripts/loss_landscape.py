"""Loss landscape on the PCA plane of a MO-OpenES search trajectory.

Reads ``gradients.pth.tar`` (search-mean snapshots) and ``evo_result.pt`` (``theta_0``) from a run
folder, takes the same PC1-PC2 plane as ``pca_gradients.py`` (via ``path_pca``), and evaluates both
task losses at every grid point ``theta = theta_0 + mu + a * PC1 + b * PC2`` on a fixed set of
training images. Writes ``landscape.pt`` next to the inputs; ``pca_gradients.py`` then draws the
contours under the trajectory.

The network is the MultiMNIST LeNet HPS model of ``main.py`` (shared encoder, one linear head per
task), built through LibMTL's ``HPS`` so the parameter order matches the training-time vector.

Usage (from examples/multimnist)::

    python scripts/loss_landscape.py logs/mnist/<run folder>
    python scripts/pca_gradients.py logs/mnist/<run folder>/gradients.pth.tar
"""

import argparse
import functools
import os
import sys
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EXAMPLE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.dirname(os.path.dirname(EXAMPLE_DIR)), EXAMPLE_DIR, os.path.dirname(os.path.abspath(__file__))]
for _m in ("cvxpy", "torch_geometric"):  # imported by LibMTL modules unrelated to this script
    sys.modules.setdefault(_m, types.ModuleType(_m))

from LibMTL.architecture.HPS import HPS  # noqa: E402
from mnist import MultiMNISTDataset  # noqa: E402
from pca_gradients import path_pca  # noqa: E402


class MultiLeNetEncoder(nn.Module):
    """Same as ``main.py``: two convolutions and a linear layer to 50 (720 inputs at 36x36, 320 at 28x28)."""

    def __init__(self, in_channels=1, fc_in=720):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 10, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, stride=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(fc_in, 50)

    def forward(self, x):
        x = self.maxpool1(F.relu(self.conv1(x)))
        x = self.maxpool2(F.relu(self.conv2(x)))
        return F.relu(self.fc(self.flatten(x)))


class MultiLeNetDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_task = nn.Linear(50, 10)

    def forward(self, x):
        return self.fc_task(x)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", help="folder with gradients.pth.tar and evo_result.pt")
    p.add_argument("--grid", type=int, default=31, help="grid points per axis")
    p.add_argument("--pad", type=float, default=0.3, help="extend the grid beyond the path by this fraction")
    p.add_argument("--n_images", type=int, default=2048, help="first N training images (fixed set)")
    p.add_argument("--from_gen", type=int, default=10,
                   help="use only snapshots from this generation on (default 10; must match pca_gradients.py --from_gen)")
    p.add_argument("--dataset", default="mnist", help="mnist, fashion or fashion_and_mnist; -36p is added when the run used the 36x36 pickles")
    p.add_argument("--dataset_path", default=EXAMPLE_DIR)
    args = p.parse_args()

    g = torch.load(os.path.join(args.run_dir, "gradients.pth.tar"), map_location="cpu", weights_only=False)
    theta0 = torch.load(os.path.join(args.run_dir, "evo_result.pt"), map_location="cpu", weights_only=False)["theta_0"].float()
    tasks = g["tasks"]
    keep = g["generation"] >= args.from_gen
    g["mean"], g["generation"] = g["mean"][keep], g["generation"][keep]
    mu, comps, path2, _ = path_pca(g["mean"].double().numpy())

    # input size of the run: the 36x36 pickles give a 720-input linear layer, the 28x28 images a 320-input one
    for fc_in, suffix in ((720, "-36p"), (320, "")):
        model = HPS(tasks, functools.partial(MultiLeNetEncoder, fc_in=fc_in),
                    nn.ModuleDict({t: MultiLeNetDecoder() for t in tasks}),
                    rep_grad=False, multi_input=False, device="cpu").eval()
        if sum(p.numel() for p in model.parameters()) == theta0.numel():
            break
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == theta0.numel() == g["dims"], (n_params, theta0.numel(), g["dims"])

    ds = MultiMNISTDataset(root=args.dataset_path, dataset=args.dataset.removesuffix("-36p") + suffix, train=True)
    x = torch.stack([ds[i][0] for i in range(args.n_images)])  # through __getitem__, as in training
    y = {t: ds.targets[: args.n_images, k] for k, t in enumerate(tasks)}

    @torch.no_grad()
    def losses_at(z):
        theta = theta0 + torch.as_tensor(z, dtype=torch.float32)
        nn.utils.vector_to_parameters(theta, model.parameters())
        out = model(x)
        return [float(F.cross_entropy(out[t], y[t])) for t in tasks]

    # square grid centred on the path (the plot uses square panels with equal axis scales)
    lo, hi = path2.min(axis=0), path2.max(axis=0)
    centre, half = (lo + hi) / 2, (hi - lo).max() / 2 * (1 + 2 * args.pad)
    a = np.linspace(centre[0] - half, centre[0] + half, args.grid)
    b = np.linspace(centre[1] - half, centre[1] + half, args.grid)
    loss = np.zeros((args.grid, args.grid, len(tasks)))  # [b, a, K] for contour(a, b, loss[..., k])
    for j, bj in enumerate(b):
        for i, ai in enumerate(a):
            loss[j, i] = losses_at(mu[0] + ai * comps[:, 0] + bj * comps[:, 1])
        print(f"\rgrid row {j + 1}/{args.grid}", end="", flush=True)
    print()
    path_loss = np.array([losses_at(z) for z in g["mean"].double().numpy()])
    out = {"a": a, "b": b, "loss": loss, "path_loss": path_loss, "tasks": tasks,
           "n_images": args.n_images, "generation": g["generation"].numpy()}
    out_name = "landscape.pt" if args.from_gen == 10 else f"landscape_from{args.from_gen}.pt"
    torch.save(out, os.path.join(args.run_dir, out_name))
    print("path losses (first -> last snapshot):",
          " | ".join(f"{t}: {path_loss[0, k]:.3f} -> {path_loss[-1, k]:.3f}" for k, t in enumerate(tasks)))
    print("wrote", os.path.join(args.run_dir, out_name))


if __name__ == "__main__":
    main()
