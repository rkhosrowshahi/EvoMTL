"""Reverse DUST: sequential credit assignment and per-layer optimizer state."""

import contextlib
import copy
import csv
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader

from LibMTL.config import LibMTL_args, _make_evo_args
from LibMTL.evomtl.moea.dust import DUST
from LibMTL.evomtl.moea.es_common import make_es_optimizer
from LibMTL.evomtl.moea.rdust import RDUST, run_rdust
from test_dust import Data, batch, make_trainer


class ScalarNetwork(nn.Module):
    """Registration order deliberately differs from execution order."""

    def __init__(self, single=False):
        super().__init__()
        self.output = nn.Linear(1, 1, bias=False)
        self.output.weight.data.fill_(1.2)
        if not single:
            self.input = nn.Linear(1, 1, bias=False)
            self.input.weight.data.fill_(.7)
        self.audit = None

    def forward(self, x):
        if self.audit is not None:
            self.audit()
        return {'a': self.output(self.input(x) if hasattr(self, 'input') else x)}


class ScalarTrainer:
    def __init__(self, single=False):
        self.model = ScalarNetwork(single)
        self.device = torch.device('cpu')
        self.multi_input = False
        self.task_name = ['a']
        self.task_num = 1
        self.meter = SimpleNamespace(losses={'a': SimpleNamespace(loss_fn=nn.MSELoss())})

    def process_preds(self, preds):
        return preds

    def forward4loss(self, model, x, y):
        return torch.stack([nn.functional.mse_loss(model(x)['a'], y['a'])])

    def set_theta(self, theta):
        nn.utils.vector_to_parameters(torch.as_tensor(theta).reshape(-1), self.model.parameters())


class TokenNetwork(ScalarNetwork):
    def __init__(self):
        super().__init__(single=True)
        self.input = nn.Embedding(3, 1, padding_idx=2)
        self.input.weight.data.copy_(torch.tensor([[.7], [.3], [0.]]))

    def forward(self, x):
        return {'a': self.output(self.input(x).mean(1))}


def scalar_batches():
    return [(torch.tensor([[1.], [2.]]), {'a': torch.tensor([[.2], [-.3]])})]


def estimator(trainer, cls=RDUST, seed=5):
    return cls(trainer, 8, .03, 2, 'equal', 8, .03, torch.Generator().manual_seed(seed))


def paired_noise(dust):
    """With scalar quadratic losses, these centered probes give the exact derivative."""
    def sample(name, samples):
        return [torch.cat([torch.ones(samples // 2, *shape[1:]),
                           -torch.ones(samples // 2, *shape[1:])]) for shape in dust.shapes[name]]
    dust._sample_noise = sample


def exact_iteration(trainer, batches, optimizer):
    params = list(trainer.model.parameters())
    gradients = {p: torch.zeros_like(p) for p in params}
    for name in ('output', 'input'):
        if not hasattr(trainer.model, name):
            continue
        parameter = getattr(trainer.model, name).weight
        loss = sum(trainer.forward4loss(trainer.model, x, y).sum() for x, y in batches) / len(batches)
        gradient, = torch.autograd.grad(loss, parameter)
        gradients[parameter] = gradient.detach().clone()
        optimizer.zero_grad(set_to_none=True)
        parameter.grad = gradient
        optimizer.step()
    return torch.cat([gradients[p].reshape(-1) for p in params]).unsqueeze(0)


class RDUSTTests(unittest.TestCase):
    def test_rdust_is_independent_of_dust_inheritance(self):
        self.assertFalse(issubclass(RDUST, DUST))

    def test_reverse_execution_order_and_fresh_downstream_feedback(self):
        trainer = ScalarTrainer()
        reference = copy.deepcopy(trainer)
        batches = scalar_batches()
        reference_optimizer = torch.optim.SGD(reference.model.parameters(), lr=.05)
        expected = exact_iteration(reference, batches, reference_optimizer)
        original = {name: p.detach().clone() for name, p in trainer.model.named_parameters()}
        with contextlib.redirect_stdout(io.StringIO()):
            dust = estimator(trainer)
            paired_noise(dust)
            forwards = []
            trainer.model.audit = lambda: forwards.append((
                tuple(dust.jitter), {name: p.detach().clone() for name, p in trainer.model.named_parameters()}))
            try:
                J, losses, J_true = dust.step(batches, torch.optim.SGD(trainer.model.parameters(), lr=.05),
                                               diagnose=True)
            finally:
                trainer.model.audit = None
                dust.close()
        self.assertEqual(list(dust.groups), ['input', 'output'])
        torch.testing.assert_close(J, expected, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(J_true, expected, rtol=2e-5, atol=2e-6)
        for actual, target in zip(trainer.model.parameters(), reference.model.parameters()):
            torch.testing.assert_close(actual, target, rtol=2e-5, atol=2e-6)
        initial_trainer = ScalarTrainer()
        initial = initial_trainer.forward4loss(initial_trainer.model, *batches[0]).detach()
        torch.testing.assert_close(losses, initial)
        upstream_draws = [weights for jitter, weights in forwards if jitter == ('input',)]
        self.assertTrue(upstream_draws)
        for weights in upstream_draws:
            torch.testing.assert_close(weights['input.weight'], original['input.weight'])
            torch.testing.assert_close(weights['output.weight'], reference.model.output.weight)
            self.assertFalse(torch.equal(weights['output.weight'], original['output.weight']))
        # Every upstream perturbation has a clean forward at the same updated state.
        for i, (jitter, weights) in enumerate(forwards):
            if jitter == ('input',):
                self.assertTrue(any(not old_jitter and all(torch.equal(old_weights[k], v)
                                                          for k, v in weights.items())
                                    for old_jitter, old_weights in forwards[:i]))

    def test_optimizer_state_advances_only_for_active_layer(self):
        for name in ('sgd', 'adam', 'rmsprop'):
            with self.subTest(optimizer=name), contextlib.redirect_stdout(io.StringIO()):
                trainer, reference = ScalarTrainer(), ScalarTrainer()
                options = dict(optim=name, lr=.01, sgd_momentum=.8)
                optimizer = make_es_optimizer(list(trainer.model.parameters()), **options)
                reference_optimizer = make_es_optimizer(list(reference.model.parameters()), **options)
                params = list(trainer.model.parameters())
                steps = []
                original_step = optimizer.step

                def step(*args, **kwargs):
                    active = [i for i, p in enumerate(params) if p.grad is not None]
                    self.assertEqual(len(active), 1)
                    before = [p.detach().clone() for p in params]
                    result = original_step(*args, **kwargs)
                    for i, p in enumerate(params):
                        if i not in active:
                            torch.testing.assert_close(p, before[i], rtol=0, atol=0)
                    steps.append(active[0])
                    return result

                optimizer.step = step
                dust = estimator(trainer)
                paired_noise(dust)
                try:
                    for _ in range(3):
                        expected = exact_iteration(reference, scalar_batches(), reference_optimizer)
                        J, _, _ = dust.step(scalar_batches(), optimizer)
                        torch.testing.assert_close(J, expected, rtol=3e-5, atol=3e-6)
                        for actual, target in zip(params, reference.model.parameters()):
                            torch.testing.assert_close(actual, target, rtol=3e-5, atol=3e-6)
                finally:
                    dust.close()
                self.assertEqual(steps, [0, 1] * 3)
                if name != 'sgd':
                    self.assertEqual([int(optimizer.state[p]['step']) for p in params], [3, 3])

    def test_one_layer_uses_unchanged_dust_estimator(self):
        trainer = ScalarTrainer(single=True)
        with contextlib.redirect_stdout(io.StringIO()):
            baseline = estimator(trainer, DUST)
            try:
                expected, expected_losses = baseline.estimate(*scalar_batches()[0])
            finally:
                baseline.close()
            before = nn.utils.parameters_to_vector(trainer.model.parameters()).detach().clone()
            dust = estimator(trainer)
            try:
                actual, losses, exact = dust.step(scalar_batches(),
                                                  torch.optim.SGD(trainer.model.parameters(), lr=.01))
            finally:
                dust.close()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(losses, expected_losses, rtol=0, atol=0)
        torch.testing.assert_close(nn.utils.parameters_to_vector(trainer.model.parameters()),
                                   before - .01 * expected[0])
        self.assertIsNone(exact)

    def test_multiple_batches_are_reused_for_every_layer(self):
        batches = scalar_batches() + [(torch.tensor([[-.5]]), {'a': torch.tensor([[.4]])})]
        trainer, reference = ScalarTrainer(), ScalarTrainer()
        initial = sum(trainer.forward4loss(trainer.model, x, y).detach() for x, y in batches) / len(batches)
        expected = exact_iteration(reference, batches, torch.optim.SGD(reference.model.parameters(), lr=.02))
        with contextlib.redirect_stdout(io.StringIO()):
            dust = estimator(trainer)
            paired_noise(dust)
            try:
                J, losses, _ = dust.step(batches, torch.optim.SGD(trainer.model.parameters(), lr=.02))
            finally:
                dust.close()
        torch.testing.assert_close(J, expected, rtol=3e-5, atol=3e-6)
        torch.testing.assert_close(losses, initial)

    def test_shared_layer_and_free_normalization_form_separate_groups(self):
        with contextlib.redirect_stdout(io.StringIO()):
            trainer = make_trainer(norm=True)
            dust = RDUST(trainer, 24, .05, 4, 'equal', 4, .03, torch.Generator().manual_seed(4))
            optimizer = torch.optim.Adam(trainer.model.parameters(), lr=.001)
            steps = []
            original_step = optimizer.step

            def step(*args, **kwargs):
                steps.append({id(p) for p in trainer.model.parameters() if p.grad is not None})
                return original_step(*args, **kwargs)

            optimizer.step = step
            try:
                J, _, _ = dust.step([batch()], optimizer)
            finally:
                dust.close()
        self.assertEqual(list(dust.groups), ['encoder.0', 'encoder.3', 'encoder.4', 'decoders.a'])
        self.assertEqual(steps, [{id(p) for p in group} for group in reversed(list(dust.groups.values()))])
        head = trainer.model.decoders['a']
        self.assertEqual(sum(id(head.weight) in group for group in steps), 1)
        self.assertTrue(all(int(optimizer.state[p]['step']) == 1 for p in trainer.model.parameters()))
        self.assertTrue(torch.isfinite(J).all())
        self.assertFalse(any(module._forward_hooks for module in trainer.model.modules()))

    def test_cli_runner_reports_without_backprop(self):
        params = LibMTL_args.parse_args(['--evo_training', '--evo_moea', 'RDUST', '--evo_iterations', '3',
                                         '--evo_pop_size', '32', '--evo_es_grad_cos_freq', '0',
                                         '--evo_eval_freq', '0', '--evo_es_optim', 'adam'])
        evo = _make_evo_args(params)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            trainer = make_trainer(directory)
            loader = DataLoader(Data(), batch_size=8, shuffle=True)
            with patch('torch.autograd.grad', side_effect=AssertionError('RDUST update used backprop')):
                result = trainer._run_moea(evo['moea'].lower(), loader, None, evo['evo_kwargs'])
            with (Path(directory) / 'metrics.csv').open() as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual(result['method'], 'RDUST')
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(not row['true_cos_direction'] for row in rows))
        theta = nn.utils.parameters_to_vector(trainer.model.parameters()).detach()
        self.assertFalse(torch.equal(theta, trainer.theta_0))
        torch.testing.assert_close(theta, torch.as_tensor(result['pareto_center_X']))
        self.assertFalse(any(module._forward_hooks for module in trainer.model.modules()))

    def test_distinct_layers_with_tied_weights_fail_before_updating(self):
        trainer = ScalarTrainer()
        trainer.model.output.weight = trainer.model.input.weight
        initial = trainer.model.input.weight.detach().clone()
        with contextlib.redirect_stdout(io.StringIO()):
            dust = estimator(trainer)
            try:
                for _ in range(2):  # A failed discovery must not leave a usable, incomplete group map.
                    with self.assertRaisesRegex(ValueError, 'tied parameters'):
                        dust.step(scalar_batches(), torch.optim.SGD(trainer.model.parameters(), lr=.1))
            finally:
                dust.close()
        torch.testing.assert_close(trainer.model.input.weight, initial, rtol=0, atol=0)
        self.assertFalse(any(module._forward_hooks for module in trainer.model.modules()))

    def test_embedding_scatter_and_padding_match_sequential_backprop(self):
        trainer = ScalarTrainer()
        trainer.model = TokenNetwork()
        reference = copy.deepcopy(trainer)
        batches = [(torch.tensor([[0], [1], [1], [2]]),
                    {'a': torch.tensor([[.2], [-.3], [.1], [.4]])})]
        expected = exact_iteration(reference, batches, torch.optim.SGD(reference.model.parameters(), lr=.02))
        with contextlib.redirect_stdout(io.StringIO()):
            dust = estimator(trainer)
            paired_noise(dust)
            try:
                J, _, _ = dust.step(batches, torch.optim.SGD(trainer.model.parameters(), lr=.02))
            finally:
                dust.close()
        torch.testing.assert_close(J, expected, rtol=3e-5, atol=3e-6)
        for actual, target in zip(trainer.model.parameters(), reference.model.parameters()):
            torch.testing.assert_close(actual, target, rtol=3e-5, atol=3e-6)
        self.assertEqual(float(trainer.model.input.weight[2, 0].detach()), 0.)

    def test_layer_parameter_tied_to_non_layer_module_is_rejected(self):
        class Scale(nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.weight = weight

            def forward(self, x):
                return x * self.weight

        class MixedTiedNetwork(ScalarNetwork):
            def __init__(self):
                super().__init__()
                self.scale = Scale(self.output.weight)

            def forward(self, x):
                return {'a': self.output(self.scale(self.input(x)))}

        trainer = ScalarTrainer()
        trainer.model = MixedTiedNetwork()
        initial = [p.detach().clone() for p in trainer.model.parameters()]
        with contextlib.redirect_stdout(io.StringIO()):
            dust = estimator(trainer)
            try:
                with self.assertRaisesRegex(ValueError, 'tied parameters'):
                    dust.step(scalar_batches(), torch.optim.SGD(trainer.model.parameters(), lr=.1))
            finally:
                dust.close()
        for p, before in zip(trainer.model.parameters(), initial):
            torch.testing.assert_close(p, before, rtol=0, atol=0)

    def test_failed_weight_space_probe_restores_all_free_parameters(self):
        with contextlib.redirect_stdout(io.StringIO()):
            trainer = make_trainer(norm=True)
            dust = RDUST(trainer, 24, .05, 4, 'equal', 4, .03, torch.Generator().manual_seed(4))
            inputs, gts = batch()
            try:
                with torch.no_grad():
                    dust._capture(inputs, gts)
                    initial = [p.detach().clone() for p in dust.free]
                    with patch.object(dust, 'losses', side_effect=RuntimeError('probe failed')):
                        with self.assertRaisesRegex(RuntimeError, 'probe failed'):
                            dust._weight_space(inputs, gts, {}, 1, params=dust.free)
                for p, before in zip(dust.free, initial):
                    torch.testing.assert_close(p, before, rtol=0, atol=0)
            finally:
                dust.close()

    def test_runner_logs_diagnostics_and_cleans_up_failed_iterations(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            trainer = make_trainer(directory)
            loader = DataLoader(Data(), batch_size=8, shuffle=True)
            result = run_rdust(trainer, loader, num_iterations=1, pop_size=32, chunk=8,
                               evo_eval_freq=0, grad_cos_freq=1)
            with (Path(directory) / 'metrics.csv').open() as stream:
                row, = csv.DictReader(stream)
            self.assertEqual(result['method'], 'RDUST')
            self.assertTrue(row['true_cos_direction'])
            with patch.object(RDUST, '_estimate_layer', side_effect=RuntimeError('probe failed')):
                with self.assertRaisesRegex(RuntimeError, 'probe failed'):
                    run_rdust(trainer, loader, num_iterations=1, pop_size=32, chunk=8,
                               evo_eval_freq=0, grad_cos_freq=0)
        self.assertFalse(any(module._forward_hooks for module in trainer.model.modules()))


if __name__ == '__main__':
    unittest.main()
