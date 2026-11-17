"""DUST variants: exact local maps, convergence, antithetic and adaptive noise, aggregators, and training loops."""
import contextlib
import math
import csv
import io
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from LibMTL.config import LibMTL_args, _make_evo_args
from LibMTL.evomtl.evo_trainer import EvoMTLTrainer
from LibMTL.evomtl.moea.a_dust import run_a_dust
from LibMTL.evomtl.moea.cadust import CADUST, _Curvature
from LibMTL.evomtl.moea.dust import DUST, run_dust
from LibMTL.evomtl.moea.mo_mgda_dust import mgda_aggregate, run_mo_mgda_dust
from LibMTL.evomtl.moea.mo_sum_dust import run_mo_sum_dust
from LibMTL.evomtl.moea.mo_upgrad_dust import run_mo_upgrad_dust
from LibMTL.evomtl.moea.es_common import upgrad_aggregate
from LibMTL.loss import CELoss, MSELoss
from LibMTL.metrics import AccMetric, L1Metric

TASKS = ('a', 'b')


class Data(Dataset):
    def __len__(self): return 16
    def __getitem__(self, i):
        g = torch.Generator().manual_seed(i)
        return torch.randn(1, 6, 6, generator=g), {'a': i % 3, 'b': (i // 3) % 3}


def make_trainer(directory=None, shared=True, norm=False, regression=False):
    torch.manual_seed(3)
    tasks = {t: dict(metrics=['L1'], metrics_fn=L1Metric(), loss_fn=MSELoss(), weight=[0]) if regression else
             dict(metrics=['Acc'], metrics_fn=AccMetric(), loss_fn=CELoss(), weight=[1]) for t in TASKS}
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

    def test_step_updates_once_using_the_mean_estimate(self):
        x, y = batch()
        batches = [(x[:6], {t: v[:6] for t, v in y.items()}), (x[6:], {t: v[6:] for t, v in y.items()})]
        with contextlib.redirect_stdout(io.StringIO()):
            trainer = make_trainer()
            theta = trainer.theta_0.detach().clone()
            initial = theta.clone()
            args = (trainer, 24, .05, 4, 'equal', 0, .03)
            baseline = DUST(*args, torch.Generator().manual_seed(7))
            estimates, batch_losses = zip(*(baseline.estimate(*b) for b in batches))
            baseline.close()
            expected_true = torch.stack([true_jacobian(trainer, *b) for b in batches]).mean(0)

            dust = DUST(*args, torch.Generator().manual_seed(7))
            optimizer = torch.optim.SGD([theta], lr=.1)
            J, losses, direction, weights, J_true = dust.step(batches, theta, optimizer, diagnose=True)
            dust.close()

        expected = torch.stack(estimates).mean(0)
        torch.testing.assert_close(J, expected)
        torch.testing.assert_close(losses, torch.stack(batch_losses).mean(0))
        torch.testing.assert_close(direction, expected[0])
        self.assertIsNone(weights)
        torch.testing.assert_close(J_true, expected_true)
        torch.testing.assert_close(theta, initial - .1 * expected[0])
        torch.testing.assert_close(nn.utils.parameters_to_vector(trainer.model.parameters()), theta)

    def test_antithetic_pairs_negate_one_noise_draw(self):
        """A-DUST: each forward pass of a site (the batch repeated chunk = 4 times) holds 2 pairs, +sigma a on the first
        two copies and -sigma a of the same draw on the last two, at every call of the shared head."""
        with contextlib.redirect_stdout(io.StringIO()):
            trainer = make_trainer()  # shared head: two calls per forward
            x, y = batch()
            dust = DUST(trainer, 24, .05, 4, 'equal', 0, .03, torch.Generator().manual_seed(0), antithetic=True)
            seen = {}
            hooks = [m.register_forward_hook(lambda m, a, o, n=n: seen.setdefault(n, []).append(
                dust.jitter[n][dust.calls[n]]) if n in dust.jitter else None) for n, m in dust.sites.items()]
            dust.estimate(x, y)
            for h in hooks:
                h.remove()
            with self.assertRaises(ValueError):
                DUST(trainer, 24, .05, 3, 'equal', 0, .03, torch.Generator(), antithetic=True)  # odd chunk
        self.assertEqual(set(seen), set(dust.draws))
        for n, jitters in seen.items():
            self.assertEqual(len(jitters), len(dust.shapes[n]) * dust.draws[n] // 4)  # every call of every pass
            for j in jitters:
                j = j.view(4, len(x), *j.shape[1:])
                self.assertTrue(torch.equal(j[2:], -j[:2]))
                self.assertFalse(torch.equal(j[0], j[1]))  # the two pairs are independent draws

    def test_antithetic_estimate_is_exact_for_a_quadratic_loss(self):
        """With a squared error the pair difference is exactly 2 sigma g.a (the even-order terms cancel), so the A-DUST
        error at a head is (1 / K) sum_k (g.a_k) a_k over its K pairs, g[b] = dL[b] / dy[b] (sign, the average over
        pairs and the per-sample credit), and its weight gradient the local linear map of that error."""
        sigma = .05
        with contextlib.redirect_stdout(io.StringIO()):
            trainer = make_trainer(shared=False, regression=True)
            x, _ = batch()
            B = len(x)
            y = {t: torch.randn(B, 3, generator=torch.Generator().manual_seed(1)) for t in TASKS}
            dust = DUST(trainer, 32, sigma, 4, 'equal', 0, .03, torch.Generator().manual_seed(0), antithetic=True)
            head = trainer.model.decoders['a']
            seen = []
            hook = head.register_forward_hook(
                lambda m, a, o: seen.append(dust.jitter['decoders.a'][0]) if 'decoders.a' in dust.jitter else None)
            J, _ = dust.estimate(x, y)
            hook.remove()
            g = 2 * (trainer.model(x)['a'] - y['a']).detach() / 3  # gradient of the per-sample mean_j (y_j - t_j)^2
        a = torch.cat([j.view(4, B, 3)[:2] for j in seen]) / sigma  # the + member of every pair: [K, B, 3]
        self.assertEqual(len(a), dust.draws['decoders.a'] // 2)
        e = torch.einsum('kb,kbo->bo', torch.einsum('bo,kbo->kb', g, a), a) / len(a)
        params = list(trainer.model.parameters())
        start = sum(p.numel() for p in params[:next(i for i, p in enumerate(params) if p is head.weight)])
        expected = torch.cat([(e.T @ dust.inputs['decoders.a'][0]).reshape(-1), e.sum(0)]) / B  # weight, bias
        torch.testing.assert_close(J[0, start:start + len(expected)], expected, rtol=1e-4, atol=1e-6)

    def test_antithetic_estimate_converges_to_backprop(self):
        with contextlib.redirect_stdout(io.StringIO()):
            trainer = make_trainer(shared=False)
            x, y = batch()
            J, _ = DUST(trainer, 4096, .01, 16, 'width', 0, .03, torch.Generator().manual_seed(0),
                        antithetic=True).estimate(x, y)
        self.assertGreater(cos(J[0], true_jacobian(trainer, x, y).sum(0)), .95)

    def test_cadust_learns_and_samples_a_low_rank_covariance(self):
        # Two known curvature directions with second moments 4 and 1; the weaker direction is truncated.
        strong = torch.tensor([.6, .8, 0.])
        weak = torch.tensor([0., 0., 1.])
        state = _Curvature(width=3, rank=1, decay=0.5, fraction=0.5)
        gen = torch.Generator().manual_seed(7)
        state.update(torch.stack([2 * strong, -2 * strong, weak, -weak]))
        torch.testing.assert_close(state.basis @ state.basis.T, torch.outer(strong, strong))

        expected = 0.5 * torch.eye(3) + 1.5 * torch.outer(strong, strong)
        draws = state.sample(30_000, gen)
        torch.testing.assert_close(draws.T @ draws / len(draws), expected, atol=.025, rtol=0)
        torch.testing.assert_close(state.inverse(expected), torch.eye(3), atol=1e-6, rtol=0)

        # This batch is weaker than the prior before discount, but stronger after discount.
        state.update(torch.stack([1.2 * weak, -1.2 * weak]))
        torch.testing.assert_close(state.basis @ state.basis.T, torch.outer(weak, weak))

    def test_cadust_warmup_and_shared_site_match_dust_contract(self):
        with contextlib.redirect_stdout(io.StringIO()):
            trainer = make_trainer(shared=True)
            x, y = batch()
            args = (trainer, 48, .05, 4, 'equal', 0, .03)
            baseline = DUST(*args, torch.Generator().manual_seed(3))
            expected, _ = baseline.estimate(x, y)
            expected_next, _ = baseline.estimate(x, y)
            baseline.close()
            for rank in (0, 2):
                cadust = CADUST(*args, torch.Generator().manual_seed(3), rank=rank)
                first, _ = cadust.estimate(x, y)
                next_estimate, _ = cadust.estimate(x, y)
                cadust.close()
                torch.testing.assert_close(first, expected, rtol=0, atol=0)
                if not rank:
                    torch.testing.assert_close(next_estimate, expected_next, rtol=0, atol=0)
                else:
                    self.assertEqual(cadust.curvature['decoders.a'][1].basis.shape, (6, 2))
                    self.assertTrue(torch.isfinite(next_estimate).all())
                    self.assertFalse(torch.equal(next_estimate, expected_next))

    def test_cadust_cli_trains_with_no_backprop_diagnostic(self):
        params = LibMTL_args.parse_args(['--evo_training', '--evo_moea', 'CADUST', '--evo_iterations', '3',
                                         '--evo_pop_size', '32', '--evo_es_grad_cos_freq', '0',
                                         '--evo_eval_freq', '0', '--evo_cadust_rank', '2',
                                         '--evo_cadust_decay', '.8', '--evo_cadust_subspace_fraction', '.3'])
        evo = _make_evo_args(params)
        self.assertEqual([evo['evo_kwargs'][k] for k in ('rank', 'decay', 'subspace_fraction')], [2, .8, .3])
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            trainer = make_trainer(directory)
            loader = DataLoader(Data(), batch_size=8, shuffle=True)
            out = trainer._run_moea(evo['moea'].lower(), loader, None, evo['evo_kwargs'])
            with (Path(directory) / 'metrics.csv').open() as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual(out['method'], 'CADUST')
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(not row['true_cos_direction'] for row in rows))
        theta = torch.nn.utils.parameters_to_vector(trainer.model.parameters()).detach()
        self.assertFalse(torch.equal(theta, trainer.theta_0))
        self.assertFalse(any(m._forward_hooks for m in trainer.model.modules()))

    def test_log_allocation_uses_the_cma_es_rule_per_layer(self):
        with contextlib.redirect_stdout(io.StringIO()):
            trainer = make_trainer()  # outputs per sample: conv 3x4x4 = 48, linear 8, shared head 3 + 3 (two calls)
            x, y = batch()
            dust = DUST(trainer, 0, .05, 4, 'log', 0, .03, torch.Generator().manual_seed(0), draw_scale=2.0)
            dust.estimate(x, y)
        expected = {d: 4 * -(-int(2 * (4 + int(3 * math.log(d)))) // 4) for d in (48, 8, 6)}  # ceil to chunks of 4
        self.assertEqual(sorted(dust.draws.values()), sorted(expected.values()))

    def test_custom_allocation_follows_the_forward_order(self):
        with contextlib.redirect_stdout(io.StringIO()):
            trainer = make_trainer(shared=False)  # forward order: conv, linear, head a, head b
            x, y = batch()
            dust = DUST(trainer, 0, .05, 4, 'custom', 0, .03, torch.Generator().manual_seed(0), draws=[12, 8, 4, 4])
            dust.estimate(x, y)
        self.assertEqual(dust.draws, {'encoder.0': 12, 'encoder.3': 8, 'decoders.a': 4, 'decoders.b': 4})
        with self.assertRaises(ValueError):
            DUST(trainer, 0, .05, 4, 'custom', 0, .03, torch.Generator(), draws=[6, 8, 4, 4])  # not a multiple of 4

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
        runners = ((run_dust, {}, 'DUST'), (run_a_dust, {}, 'A-DUST'), (run_mo_sum_dust, {}, 'MO-Sum-DUST'),
                   (run_mo_upgrad_dust, {}, 'MO-UPGrad-DUST'), (run_mo_mgda_dust, {}, 'MO-MGDA-DUST'),
                   (run_mo_mgda_dust, {'grad_norm': 'loss'}, 'MO-MGDA-Loss-DUST'))
        for runner, extra, method in runners:
            per_task = method not in ('DUST', 'A-DUST')
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

    def test_multi_objective_runners_use_the_custom_allocation(self):
        """MO-UPGrad-DUST and MO-MGDA-Loss-DUST split their draws by the per-layer counts of ``draws`` (pop_size
        unused) and still log per-task weights."""
        runners = ((run_mo_upgrad_dust, {}, 'MO-UPGrad-DUST'),
                   (run_mo_mgda_dust, {'grad_norm': 'loss'}, 'MO-MGDA-Loss-DUST'))
        for runner, extra, method in runners:
            log = io.StringIO()
            with self.subTest(method), tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(log):
                trainer = make_trainer(directory, shared=False)  # forward order: conv, linear, head a, head b
                loader = DataLoader(Data(), batch_size=8, shuffle=True)
                out = runner(trainer, loader, num_iterations=2, pop_size=0, sigma=.05, chunk=4, lr=.01,
                             allocation='custom', draws=[16, 8, 4, 4], evo_eval_freq=0, grad_cos_freq=1, **extra)
                with (Path(directory) / 'metrics.csv').open() as stream:
                    rows = list(csv.DictReader(stream))
            self.assertEqual(out['method'], method)
            line = next(l for l in log.getvalue().splitlines() if 'direct draws per generation' in l)
            self.assertIn('32 direct draws per generation (custom): ', line)
            for site in ('encoder.0=16', 'encoder.3=8', 'decoders.a=4', 'decoders.b=4'):
                self.assertIn(site, line)
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(rows[-1][f'weight_{t}'] and rows[-1][f'true_cos_{t}'] for t in TASKS))

    def test_unshuffled_training_loader_is_rejected(self):
        """Every generation restarts the loader: without shuffling, DUST and A-DUST would see one batch forever."""
        for runner, method in ((run_dust, 'DUST'), (run_a_dust, 'A-DUST')):
            with self.subTest(method), contextlib.redirect_stdout(io.StringIO()):
                trainer = make_trainer()
                with self.assertRaisesRegex(ValueError, f'^{method} .*shuffle=True'):
                    runner(trainer, DataLoader(Data(), batch_size=8), num_iterations=1, pop_size=32, chunk=8)
                self.assertTrue(torch.equal(torch.nn.utils.parameters_to_vector(trainer.model.parameters()),
                                            trainer.theta_0))


if __name__ == '__main__':
    unittest.main()
