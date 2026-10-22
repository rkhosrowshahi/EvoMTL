"""How PSL-ES and DCPSL-ES runs sample one generation, and the Pareto front each one learns.

Rebuilds one generation of each run's sampling at its final state (``psl_result.pt`` for PSL-ES, any scalarization;
``dcpsl_result.pt`` for DCPSL-ES), evaluates every member on the same training images, and plots one row per run:

* weight space: each member's offset from the balanced network ``theta``, along the unit front direction
  ``B / |B|`` (x) and along a fixed random direction orthogonal to it (y, on its own scale: the noise ``sigma`` is
  tiny next to ``|B|``). PSL-ES draws one preference per antithetic pair, so its pairs spread over the whole line,
  and its noise also perturbs ``B``, by ``sigma phi``: the spread is ``sigma sqrt(1 + phi^2)``, widest at the ends.
  DCPSL-ES puts its pairs at three anchors, the balanced network and the two corners, each with spread ``sigma``;
* objective space: the members' task losses, coloured by the preference they serve, with the front ``theta(lam)``
  (11 preferences, large markers) and the balanced network (star) on the same images; all rows share the axes.

Guided ES sampling is left out because the guide is not saved: the noise is the isotropic ``eps ~ N(0, I)``.
Two tasks only.

Usage (from examples/multimnist)::

    python scripts/plot_sampling.py --runs \\
        logs/mnist/gen200/psl_guided_es-tch/pop_128/adam-lr_1e-3/seed_1-s0.05_a0.9_ema0.9 \\
        logs/mnist/gen200/psl_guided_es-ws/pop_128/adam-lr_1e-3/seed_1-s0.05_a0.9_ema0.9 \\
        --out figures/comparison/psl_tch_vs_ws_sampling-gen200-seed1
"""

import argparse
import ast
import functools
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from torchvision import transforms  # noqa: E402

from loss_landscape import EXAMPLE_DIR, HPS, MultiLeNetDecoder, MultiLeNetEncoder, MultiMNISTDataset  # noqa: E402
from plot_mo_openes import MARKERS, label, parse_log, save  # noqa: E402

from LibMTL.evomtl.moea.psl_es import preference_features, preference_grid, sample_preferences  # noqa: E402

RESULTS = (("dcpsl", "dcpsl_result.pt"), ("psl", "psl_result.pt"))  # sampling scheme by result file


def evo_kwargs(run_dir):
    """Settings of the run: the ``evo_kwargs: {...}`` line of its ``stdout.log``."""
    with open(os.path.join(run_dir, "stdout.log"), errors="replace") as f:
        for line in f:
            if "evo_kwargs:" in line:
                return ast.literal_eval(line.split("evo_kwargs:", 1)[1].strip())
    raise SystemExit(f"no evo_kwargs line in {run_dir}/stdout.log")


def build_model(tasks, n_params):
    """The MultiMNIST LeNet HPS model of ``main.py`` whose size matches the saved weights (28x28 or 36x36 inputs)."""
    for fc_in in (320, 720):
        model = HPS(tasks, functools.partial(MultiLeNetEncoder, fc_in=fc_in),
                    nn.ModuleDict({t: MultiLeNetDecoder() for t in tasks}),
                    rep_grad=False, multi_input=False, device="cpu").eval()
        if sum(p.numel() for p in model.parameters()) == n_params:
            return model, fc_in
    raise SystemExit(f"no LeNet size matches {n_params} weights")


def sample_generation(kind, res, cfg, gen):
    """Member weights ``[N, D]``, the preference ``lam_1`` each member serves, and the front direction (unit, ``[D]``)
    for one generation of ``kind``'s sampling at the saved state, with isotropic noise."""
    theta, B, idx, Q = res["theta"].float(), res["B"].float(), res["index"], res["Q"].float()
    D, (P, C) = theta.numel(), B.shape
    K, pairs, sigma = C + 1, int(cfg["pop_size"]) // 2, float(cfg["sigma"])

    def theta_at(t, b, phi):
        w = t.clone()
        w.index_add_(0, idx, b @ phi)
        return w

    members, lam1 = [], []
    if kind == "psl":  # one preference per pair; the noise covers theta and B
        lam = sample_preferences(pairs, K, gen)
        phi = preference_features(lam, Q)
        eps_t = torch.randn(pairs, D, generator=gen)
        eps_b = torch.randn(pairs, P, C, generator=gen)
        for i in range(pairs):
            for sign in (1.0, -1.0):
                members.append(theta_at(theta + sign * sigma * eps_t[i], B + sign * sigma * eps_b[i], phi[i]))
                lam1.append(float(lam[i, 0]))
    else:  # anchors: the balanced network and the K corners; the noise covers the weights at each anchor
        n_corner = (pairs - int(round(float(cfg["center_fraction"]) * pairs))) // K
        n_center = pairs - K * n_corner
        corner_phi = preference_features(torch.eye(K), Q)
        anchors = [(theta, 1.0 / K)] + [(theta_at(theta, B, corner_phi[m]), 1.0 if m == 0 else 0.0) for m in range(K)]
        which = [0] * n_center + [m + 1 for m in range(K) for _ in range(n_corner)]
        eps = torch.randn(pairs, D, generator=gen)
        for i, a in enumerate(which):
            for sign in (1.0, -1.0):
                members.append(anchors[a][0] + sign * sigma * eps[i])
                lam1.append(anchors[a][1])
    direction = torch.zeros(D)
    direction[idx] = B[:, 0]
    return torch.stack(members), np.array(lam1), direction / direction.norm().clamp_min(1e-30)


@torch.no_grad()
def task_losses(model, W, x, y, tasks):
    """Cross-entropy of each task for every row of ``W`` (flat weights), eval mode as in the ES fitness."""
    out = []
    for w in W:
        nn.utils.vector_to_parameters(w, model.parameters())
        logits = model(x)
        out.append([float(F.cross_entropy(logits[t], y[:, k])) for k, t in enumerate(tasks)])
    return np.array(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True,
                   help="PSL-ES / DCPSL-ES run folders (psl_result.pt or dcpsl_result.pt, stdout.log), one row each")
    p.add_argument("--out", required=True, help="output folder, e.g. figures/comparison/<name>")
    p.add_argument("--n_images", type=int, default=256, help="first N training images (one training batch: 256)")
    p.add_argument("--seed", type=int, default=0, help="seed of the redrawn noise and preferences")
    p.add_argument("--dataset_path", default=EXAMPLE_DIR)
    args = p.parse_args()

    data, rows = None, []
    for run_dir in args.runs:
        kind, result_name = next(((k, f) for k, f in RESULTS if os.path.exists(os.path.join(run_dir, f))), (None, None))
        if kind is None:
            raise SystemExit(f"{run_dir}: no psl_result.pt or dcpsl_result.pt")
        res = torch.load(os.path.join(run_dir, result_name), map_location="cpu", weights_only=False)
        if res["B"].shape[1] != 1:
            raise SystemExit("plot_sampling.py draws two tasks only")
        cfg = evo_kwargs(run_dir)
        name = label(parse_log(os.path.join(run_dir, "stdout.log")))
        tasks = list(res["tasks"])
        model, fc_in = build_model(tasks, res["theta"].numel())
        if data is None:  # the same training images for every run, with main.py's input transform
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            ds = MultiMNISTDataset(root=args.dataset_path, dataset="mnist" if fc_in == 320 else "mnist-36p",
                                   train=True, transform=transform)  # the 36x36 pickles ignore the transform
            data = (torch.stack([ds[i][0] for i in range(args.n_images)]), ds.targets[: args.n_images])
        W, lam1, b_hat = sample_generation(kind, res, cfg, torch.Generator().manual_seed(args.seed))
        offsets = W - res["theta"].float()
        r = torch.randn(offsets.shape[1], generator=torch.Generator().manual_seed(args.seed + 1))
        r_hat = r - (r @ b_hat) * b_hat
        r_hat = r_hat / r_hat.norm()
        grid = preference_grid(2, 11)
        phi = preference_features(grid, res["Q"].float())
        front_W = torch.stack([res["theta"].float().index_add(0, res["index"], res["B"].float() @ f) for f in phi])
        row = dict(label=name, marker=MARKERS.get(name, "o"), tasks=tasks, cfg=cfg, lam1=lam1,
                   xy=np.stack([(offsets @ b_hat).numpy(), (offsets @ r_hat).numpy()], axis=1),
                   F=task_losses(model, W, *data, tasks), front=task_losses(model, front_W, *data, tasks),
                   grid=grid[:, 0].numpy(), b_norm=float(res["B"].norm()))
        rows.append(row)
        print(f"{name}: {len(W)} members, |B|={row['b_norm']:.3f}, sigma={float(cfg['sigma'])} | front ends "
              f"(lam_1=1) {row['front'][-1, 0]:.3f}/{row['front'][-1, 1]:.3f}, (lam_1=0) {row['front'][0, 0]:.3f}/"
              f"{row['front'][0, 1]:.3f}, balanced {row['front'][5, 0]:.3f}/{row['front'][5, 1]:.3f}")

    cmap, norm = plt.cm.viridis, Normalize(0, 1)
    fig, axes = plt.subplots(len(rows), 2, figsize=(12, 4.6 * len(rows)), squeeze=False,
                             gridspec_kw={"width_ratios": [1.15, 1]})
    allF = np.concatenate([np.concatenate([r["F"], r["front"]]) for r in rows])
    lo, hi = allF.min(axis=0), allF.max(axis=0)
    pad = 0.06 * (hi - lo).max()
    t1, t2 = rows[0]["tasks"]
    for ax_row, r in zip(axes, rows):
        ax = ax_row[0]  # weight space
        ax.axhline(0.0, color="0.75", lw=0.8, zorder=0)
        ax.plot([-r["b_norm"], r["b_norm"]], [0, 0], color="0.35", lw=2.2, zorder=1, label=r"the family $\theta + \phi B$")
        ax.scatter(r["xy"][:, 0], r["xy"][:, 1], c=cmap(norm(r["lam1"])), marker=r["marker"], s=22, edgecolors="k",
                   linewidths=0.25, zorder=3)
        ax.plot(0, 0, "*", color="w", markeredgecolor="k", markersize=16, zorder=4, label=r"balanced network $\theta$")
        ax.set_xlabel(r"offset along $B/|B|$ (front direction)")
        ax.set_ylabel("offset along a random orthogonal direction")
        ax.set_title(f"{r['label']}: samples in weight space", fontsize=10)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.35)

        ax = ax_row[1]  # objective space
        ax.scatter(r["F"][:, 0], r["F"][:, 1], c=cmap(norm(r["lam1"])), marker=r["marker"], s=14, alpha=0.7,
                   edgecolors="none", zorder=2)
        ax.plot(r["front"][:, 0], r["front"][:, 1], "-", color="0.25", lw=1.2, zorder=3)
        ax.scatter(r["front"][:, 0], r["front"][:, 1], c=cmap(norm(r["grid"])), marker=r["marker"], s=90,
                   edgecolors="k", linewidths=0.6, zorder=4)
        ax.plot(*r["front"][5], "*", color="w", markeredgecolor="k", markersize=16, zorder=5)
        ax.set_xlim(lo[0] - pad, hi[0] + pad)
        ax.set_ylim(lo[1] - pad, hi[1] + pad)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"Objective 1: {t1} train loss")
        ax.set_ylabel(f"Objective 2: {t2} train loss")
        ax.set_title(f"{r['label']}: samples (small) and front (large)", fontsize=10)
        ax.grid(alpha=0.35)
    cfg = rows[0]["cfg"]
    fig.suptitle(f"One generation of sampling at generation {int(cfg['num_iterations'])} "
                 f"(N={int(cfg['pop_size'])}, sigma={float(cfg['sigma'])}, isotropic noise) and the learned front, "
                 f"on {args.n_images} training images", fontsize=11)
    fig.tight_layout(rect=(0, 0, 0.92, 0.96))
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes.ravel().tolist(), fraction=0.025, pad=0.02,
                 label=r"Preference $\lambda_1$ the sample serves")
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "sampling.png")
    save(fig, path, bbox_inches="tight")
    print("wrote", path)


if __name__ == "__main__":
    main()
