"""Replay RDUST with observational per-block diagnostics and reversible trial steps.

Run from examples/multimnist with ../../.venv/bin/python scripts/investigate_rdust.py.
The training algorithm is unchanged. Trial steps restore parameters and never update
optimizer state. Gradient diagnostics do not draw random numbers or update meters.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'examples/multimnist'))
import torch
import main as benchmark
from LibMTL.config import LibMTL_args
from LibMTL.utils import set_device, set_random_seed
from LibMTL.evomtl.moea.rdust import RDUST
import LibMTL.evomtl.moea.es_common as reporting


def flat(tensors):
    return torch.cat([t.detach().reshape(-1) for t in tensors])


def cosine(a, b):
    return float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-30))


def install_diagnostics(folder, every, start=0):
    original = RDUST.step
    stream = (folder / 'blocks.csv').open('w', buffering=1)
    writer = None
    generation = start

    def step(self, batches, optimizer, diagnose=False):
        nonlocal writer, generation
        generation += 1
        enabled = generation <= 3 or generation % every == 0
        old_step = optimizer.step
        staged = []

        def objective():
            return sum(self.losses(x, y).mean(1) for x, y in batches) / len(batches)

        def step(*args, **kwargs):
            nonlocal writer
            active = [p for p in self.model.parameters() if p.grad is not None]
            name = next(n for n, ps in self.groups.items() if {id(p) for p in ps} == {id(p) for p in active})
            if not enabled:
                return old_step(*args, **kwargs)
            before = [p.detach().clone() for p in active]
            estimates = [p.grad.detach().clone() for p in active]
            with torch.enable_grad():
                pre_tasks = objective()
                exact = torch.autograd.grad(pre_tasks.sum(), active)
            g, truth = flat(estimates), flat(exact)
            result = old_step(*args, **kwargs)
            after = [p.detach().clone() for p in active]
            delta = flat([a - b for a, b in zip(after, before)])
            staged.append((g, truth, delta))
            post_tasks = objective()
            pre, post = float(pre_tasks.detach().sum()), float(post_tasks.sum())
            row = dict(generation=generation, block=name, pre_loss=pre, post_loss=post,
                       loss_change=post-pre, estimated_norm=float(g.norm()), true_norm=float(truth.norm()),
                       estimate_cos=cosine(g, truth), update_norm=float(delta.norm()),
                       descent_cos=cosine(-delta, truth), predicted_change=float(delta @ truth),
                       parameter_norm=float(flat(before).norm()))
            for i, task in enumerate(self.trainer.task_name):
                row['pre_'+task] = float(pre_tasks[i].detach())
                row['post_'+task] = float(post_tasks[i])
            # Same starting point: compare shorter Adam step, no first-moment
            # averaging with the same second moment, and exact steepest descent.
            group = optimizer.param_groups[0]
            trial_deltas = {'half_adam': [(a-b)*0.5 for a,b in zip(after,before)]}
            if isinstance(optimizer, torch.optim.Adam):
                ds = []
                for p, est in zip(active, estimates):
                    state = optimizer.state[p]
                    t = int(state['step'])
                    denom = (state['exp_avg_sq'] / (1-group['betas'][1]**t)).sqrt() + group['eps']
                    ds.append(-group['lr'] * est / denom)
                trial_deltas['no_momentum_same_v'] = ds
            scale = float(delta.norm() / truth.norm().clamp_min(1e-30))
            trial_deltas['exact_same_norm'] = [-scale*t for t in exact]
            try:
                for label, ds in trial_deltas.items():
                    for p,b,d in zip(active,before,ds):
                        p.copy_(b+d)
                    row[label+'_loss_change'] = float(objective().sum())-pre
                    row[label+'_norm'] = float(flat(ds).norm())
            finally:
                for p,a in zip(active,after):
                    p.copy_(a)
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            return result

        optimizer.step = step
        try:
            out = original(self, batches, optimizer, diagnose=diagnose)
        finally:
            optimizer.step = old_step
        if staged:
            gs, ts, ds = [torch.cat([r[k] for r in staged]) for k in range(3)]
            aggregate = dict(generation=generation, staged_estimate_cos=cosine(gs,ts),
                             staged_descent_cos=cosine(-ds,ts), estimated_norm=float(gs.norm()),
                             true_norm=float(ts.norm()), update_norm=float(ds.norm()))
            path = folder/'aggregate.csv'
            exists = path.exists()
            with path.open('a') as f:
                w = csv.DictWriter(f, fieldnames=list(aggregate))
                if not exists:w.writeheader()
                w.writerow(aggregate)
        if generation in (100, 200, 250, 300, 400, 450):
            torch.save({'model': self.model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'noise_rng': self.gen.get_state(), 'torch_rng': torch.get_rng_state(),
                        'generation': generation}, folder / f'checkpoint_{generation}.pt')
        return out

    RDUST.step = step
    return stream


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=450)
    parser.add_argument('--every', type=int, default=5)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--beta1', type=float, default=.9)
    parser.add_argument('--output', default='logs/multimnist/rdust_investigation/adam_baseline')
    parser.add_argument('--no-diagnostics', action='store_true')
    options = parser.parse_args()
    folder = Path(options.output).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    (folder/'investigation.json').write_text(json.dumps(vars(options), indent=2))
    torch.set_num_threads(options.threads)
    # Reproduce the archived run's evaluation schedule: 1,51,101,... .
    # Test-loader iterator creation consumes RNG and therefore affects later training batches.
    reporting.should_run_full_test = lambda i,n,e: i % int(e) == 0 or i == n-1
    stream = None if options.no_diagnostics else install_diagnostics(folder, options.every)
    sys.argv = ['main.py', '--config', str(ROOT/'examples/multimnist/configs/10epochs/multimnist/evomtl/rdust/custom-128_128_16_16_16/adam-lr_0.001/seed1.yaml')]
    params = benchmark.parse_args(LibMTL_args)
    params.save_path = str(folder)
    params.wandb_enable = False
    params.evo_iterations = options.iterations
    params.evo_es_adam_beta1 = options.beta1
    params.evo_es_grad_cos_freq = 0
    benchmark._log_to_save_path(params)
    set_device(params.gpu_id)
    set_random_seed(params.seed)
    try:
        benchmark.main(params)
    finally:
        if stream:
            stream.close()


if __name__ == '__main__':
    main()
