"""DUST and MO-*-DUST: exact local maps, convergence of the forward-only estimate, DUST = MO-Sum-DUST, the UPGrad / MGDA
aggregators, fallback ES, and the training loops."""
import contextlib
import csv
import io
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from LibMTL.evomtl.evo_trainer import EvoMTLTrainer
from LibMTL.evomtl.moea.dust import DUST, run_dust
from LibMTL.evomtl.moea.mo_mgda_dust import mgda_aggregate, run_mo_mgda_dust
from LibMTL.evomtl.moea.mo_sum_dust import run_mo_sum_dust
from LibMTL.evomtl.moea.mo_upgrad_dust import run_mo_upgrad_dust
from LibMTL.evomtl.moea.es_common import upgrad_aggregate
from LibMTL.loss import CELoss
from LibMTL.metrics import AccMetric

TASKS = ('a', 'b')


class Data(Dataset):
    def __len__(self): return 16
    def __getitem__(self, i):
        g = torch.Generator().manual_seed(i)
        return torch.randn(1, 6, 6, generator=g), {'a': i % 3, 'b': (i // 3) % 3}


def make_trainer(directory=None, shared=True, norm=False):
    torch.manual_seed(3)
    tasks = {t: dict(metrics=['Acc'], metrics_fn=AccMetric(), loss_fn=CELoss(), weight=[1]) for t in TASKS}
    encoder = lambda: nn.Sequential(nn.Conv2d(1, 3, 3), nn.ReLU(), nn.Flatten(), nn.Linear(48, 8),
                                    *([nn.LayerNorm(8)] if norm else []), nn.Tanh())
    head = nn.Linear(8, 3)  # shared: one decoder module called once per task
    decoders = nn.ModuleDict({t: head if shared else nn.Linear(8, 3) for t in TASKS})
    trainer = EvoMTLTrainer(task_dict=tasks, weighting='EW', architecture='HPS', encoder_class=encoder,
                            decoders=decoders, rep_grad=False, multi_input=False,
                            optim_param={'optim': 'sgd', 'lr': .01}, scheduler_param=None, save_path=directory,
                            weight_args={}, arch_args={})
    trainer.prepare_evolution()
    return trainer


def batch():
    x, y = next(iter(DataLoader(Data(), batch_size=16)))
    return x, y


def true_jacobian(trainer, x, y):
    model = trainer.model
    model.eval()
    params = list(model.parameters())
    preds = model(x)
    rows = []
    for t in TASKS:
        loss = nn.functional.cross_entropy(preds[t], y[t])
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        rows.append(torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                               for g, p in zip(grads, params)]))
    return torch.stack(rows)


def cos(a, b):
    return float(a @ b / (a.norm() * b.norm()))


class DUSTTests(unittest.TestCase):
    def test_exact_errors_reproduce_backprop(self):
        """The local linear maps (convolution, Linear, a decoder called once per task) turn the true output errors
        into the backprop gradient."""
        with contextlib.redirect_stdout(io.StringIO()):
            trainer = make_trainer()
            x, y = batch()
            dust = DUST(trainer, 12, .05, 4, 'equal', 0, .03, torch.Generator().manual_seed(0))
            dust.estimate(x, y)  # caches every site's inputs
        J_true = true_jacobian(trainer, x, y)
        outputs = {}
        hooks = [m.register_forward_hook(lambda m, a, o, n=n: outputs.setdefault(n, []).append(o.retain_grad() or o))
                 for n, m in dust.sites.items()]
        preds = trainer.model(x)
        for h in hooks + dust.hooks:
            h.remove()
        (nn.functional.cross_entropy(preds['a'], y['a'], reduction='sum')).backward()
        grads = {}
        for n, m in dust.sites.items():
            errors = [(o.grad if o.grad is not None else torch.zeros_like(o)).unsqueeze(0) for o in outputs[n]]
            dust._reassemble(m, errors, dust.inputs[n], grads)
        J = torch.cat([grads[p].reshape(-1) for p in trainer.model.parameters()]) / len(x)
        torch.testing.assert_close(J, J_true[0], rtol=1e-4, atol=1e-6)

    def test_estimate_converges_to_backprop(self):
        with contextlib.redirect_stdout(io.StringIO()):
            trainer = make_trainer(shared=False)
            x, y = batch()
            J_sum, _ = DUST(trainer, 4096, .01, 16, 'width', 0, .03, torch.Generator().manual_seed(0)).estimate(x, y)
            J, losses = DUST(trainer, 4096, .01, 16, 'width', 0, .03, torch.Generator().manual_seed(0),
                             per_task=True).estimate(x, y)
        J_true = true_jacobian(trainer, x, y)
        self.assertEqual(tuple(J_sum.shape), (1, J_true.shape[1]))
        self.assertGreater(cos(J_sum[0], J_true.sum(0)), .95)
        for k in range(len(TASKS)):
            self.assertGreater(cos(J[k], J_true[k]), .95)
        self.assertEqual(tuple(losses.shape), (len(TASKS),))

    def test_summed_loss_equals_summed_task_estimates(self):
        """Same noise: DUST on the summed loss and the sum of the MO-Sum-DUST task estimates agree up to rounding
        (also through the weight-space ES of the LayerNorm)."""
        with contextlib.redirect_stdout(io.StringIO()):
            trainer = make_trainer(norm=True)
            x, y = batch()
            J_sum, L_sum = DUST(trainer, 64, .05, 8, 'equal', 8, .03, torch.Generator().manual_seed(5)).estimate(x, y)
            J, L = DUST(trainer, 64, .05, 8, 'equal', 8, .03, torch.Generator().manual_seed(5),
                        per_task=True).estimate(x, y)
        self.assertLess(float((J_sum[0] - J.sum(0)).norm() / J.sum(0).norm()), 1e-4)  # float32 rounding only
        torch.testing.assert_close(L_sum, L)

    def test_parameters_outside_the_sites_get_weight_space_es(self):
        with contextlib.redirect_stdout(io.StringIO()):
            trainer = make_trainer(norm=True)
            x, y = batch()
            dust = DUST(trainer, 32, .05, 8, 'equal', 256, .01, torch.Generator().manual_seed(0))
            J, _ = dust.estimate(x, y)
        norm = trainer.model.encoder[4]
        self.assertEqual([id(p) for p in dust.free], [id(norm.weight), id(norm.bias)])
        J_true = true_jacobian(trainer, x, y)
        start = sum(p.numel() for p in list(trainer.model.parameters())[:4])  # conv, linear: weight and bias
        cols = slice(start, start + 16)  # the LayerNorm weight and bias
        self.assertGreater(cos(J.sum(0)[cols], J_true.sum(0)[cols]), .8)

    def test_aggregators_of_conflicting_task_gradients(self):
        torch.manual_seed(0)
        g = torch.randn(40)
        J = torch.stack([g + torch.randn(40), -0.5 * g + torch.randn(40)])  # two conflicting tasks
        losses = torch.tensor([2.0, 0.5])
        d, w = upgrad_aggregate(J)
        self.assertTrue(bool((J @ d >= -1e-6).all()))  # UPGrad: conflicts with no task
        for grad_norm, G in (('none', J), ('loss', J / losses[:, None])):
            d, alpha = mgda_aggregate(J, losses, grad_norm)
            self.assertAlmostEqual(float(alpha.sum()), 1.0, places=5)
            self.assertTrue(bool(((alpha > 1e-3) & (alpha < 1 - 1e-3)).all()))  # interior min-norm point ...
            torch.testing.assert_close(G @ d, (d @ d).repeat(2))  # ... equal descent g~_k . d = ||d||^2

    def test_runners_train_and_report(self):
        runners = ((run_dust, {}, 'DUST'), (run_mo_sum_dust, {}, 'MO-Sum-DUST'),
                   (run_mo_upgrad_dust, {}, 'MO-UPGrad-DUST'), (run_mo_mgda_dust, {}, 'MO-MGDA-DUST'),
                   (run_mo_mgda_dust, {'grad_norm': 'loss'}, 'MO-MGDA-Loss-DUST'))
        for runner, extra, method in runners:
            per_task = method != 'DUST'
            with self.subTest(method), tempfile.TemporaryDirectory() as directory, \
                    contextlib.redirect_stdout(io.StringIO()):
                trainer = make_trainer(directory)
                loader = DataLoader(Data(), batch_size=8, shuffle=True)
                out = runner(trainer, loader, num_iterations=3, pop_size=32, sigma=.05, chunk=8, lr=.01,
                             test_dataloaders=loader, evo_eval_freq=0, grad_cos_freq=1, **extra)
                with (Path(directory) / 'metrics.csv').open() as stream:
                    rows = list(csv.DictReader(stream))
            self.assertEqual(out['method'], method)
            self.assertEqual(len(rows), 3)
            self.assertTrue(rows[-1]['true_cos_direction'])
            self.assertEqual(all(rows[-1][f'true_cos_{t}'] for t in TASKS), per_task)  # per-task rows only
            self.assertEqual(all(rows[-1][f'weight_{t}'] for t in TASKS), per_task)
            theta = torch.nn.utils.parameters_to_vector(trainer.model.parameters()).detach()
            torch.testing.assert_close(theta, torch.as_tensor(out['pareto_center_X']))
            self.assertFalse(torch.equal(theta, trainer.theta_0))
            self.assertFalse(any(m._forward_hooks for m in trainer.model.modules()))  # hooks removed


if __name__ == '__main__':
    unittest.main()
