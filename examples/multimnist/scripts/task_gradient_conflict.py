"""Do the two MultiMNIST tasks' backprop gradients conflict? Per shared layer, at saved weights along training.

For every checkpoint (``label=path/to/model_evo.pt``; ``init`` = the seed's initial weights) and each of
``--batches`` training mini-batches, the backprop gradients ``g_1``, ``g_2`` of the two task losses are compared per
shared encoder layer and over the whole encoder: their cosine, the fraction of mini-batches where it is negative (the
only case where UPGrad or MGDA differ from a positive weighting of the rows), and the norm ratio ``|g_1| / |g_2|``.
The cosine of the gradients averaged over all the mini-batches is reported too. Writes ``conflict.csv`` and
``conflict.png`` under ``figures/comparison/<name>/``; the training itself stays forward-only.
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'examples/multimnist'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

import main as benchmark
from LibMTL.config import LibMTL_args
from LibMTL.evomtl.evo_trainer import EvoMTLTrainer
from LibMTL.evomtl.moea.es_common import true_jacobian
from LibMTL.utils import set_device, set_random_seed

CONFIG = ROOT / 'examples/multimnist/configs/10epochs/multimnist/evomtl/dust/custom-128_128_16_16_16/adam-lr_0.001/seed1.yaml'


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('checkpoints', nargs='+', help='label=path/to/model_evo.pt, or init')
    cli.add_argument('--batches', type=int, default=50)
    cli.add_argument('--name', default='task_gradient_conflict')
    opts = cli.parse_args()
    out = ROOT / 'examples/multimnist/figures/comparison' / opts.name
    out.mkdir(parents=True, exist_ok=True)

    def measure(self, train_loader, test_loader, epochs):
        names = [n for n, _ in self.model.named_parameters()]
        sizes = [p.numel() for p in self.model.parameters()]
        starts = torch.tensor([0] + sizes).cumsum(0)
        cols = {}
        for n, a, b in zip(names, starts[:-1], starts[1:]):
            if n.startswith('encoder.'):
                layer = n.rsplit('.', 1)[0].split('.', 1)[1]
                cols.setdefault(layer, []).append(torch.arange(a, b))
        cols = {k: torch.cat(v) for k, v in cols.items()}
        cols['encoder'] = torch.cat(list(cols.values()))
        theta_0 = torch.nn.utils.parameters_to_vector(self.model.parameters()).detach().clone()  # seed's init
        batches = self._sample_batches(train_loader, opts.batches)
        rows = []
        for spec in opts.checkpoints:
            label, _, path = spec.partition('=')
            if label == 'init':
                theta = theta_0
            else:
                self.model.load_state_dict(torch.load(path, map_location=self.device))
                theta = torch.nn.utils.parameters_to_vector(self.model.parameters()).detach().clone()
            J = torch.stack([true_jacobian(self, theta, [b]) for b in batches])  # [batches, 2, D]
            for layer, c in cols.items():
                g = J[:, :, c].double()
                cos = torch.nn.functional.cosine_similarity(g[:, 0], g[:, 1], dim=1)
                mean = g.mean(0)
                rows.append({'checkpoint': label, 'layer': layer, 'cos_mean': float(cos.mean()),
                             'cos_min': float(cos.min()), 'cos_max': float(cos.max()),
                             'negative_fraction': float((cos < 0).double().mean()),
                             'cos_of_mean_gradient': float(torch.nn.functional.cosine_similarity(mean[0], mean[1], 0)),
                             'norm_ratio': float((g[:, 0].norm(dim=1) / g[:, 1].norm(dim=1)).mean())})
        with (out / 'conflict.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"{'checkpoint':<14}{'layer':<10}{'cos mean':>10}{'min':>8}{'max':>8}{'neg frac':>10}"
              f"{'cos(mean g)':>13}{'|g1|/|g2|':>11}")
        for r in rows:
            print(f"{r['checkpoint']:<14}{r['layer']:<10}{r['cos_mean']:>10.3f}{r['cos_min']:>8.3f}"
                  f"{r['cos_max']:>8.3f}{r['negative_fraction']:>10.2f}{r['cos_of_mean_gradient']:>13.3f}"
                  f"{r['norm_ratio']:>11.2f}")
        labels = list(dict.fromkeys(r['checkpoint'] for r in rows))
        fig, ax = plt.subplots(figsize=(6, 4))
        for layer in cols:
            ax.plot(labels, [r['cos_mean'] for r in rows if r['layer'] == layer], marker='o', label=layer)
        ax.axhline(0, color='k', lw=0.8)
        ax.set_xlabel('Checkpoint')
        ax.set_ylabel('Cosine of task gradients (mean over mini-batches)')
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / 'conflict.png', dpi=150)
        print(f"wrote {out / 'conflict.csv'} and {out / 'conflict.png'}")

    EvoMTLTrainer.train = measure
    sys.argv = ['main.py', '--config', str(CONFIG)]
    params = benchmark.parse_args(LibMTL_args)
    params.wandb_enable, params.save_path = False, None
    set_device(params.gpu_id)
    set_random_seed(params.seed)
    benchmark.main(params)


if __name__ == '__main__':
    main()
