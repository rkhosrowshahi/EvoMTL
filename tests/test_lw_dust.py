"""LW-DUST and RLW-DUST: forward / reverse-order groups, T_l inner steps per group, per-group optimizer state, and the training loop."""

import contextlib
import copy
import csv
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader

from LibMTL.config import LibMTL_args, _make_evo_args
from LibMTL.evomtl.moea.es_common import make_es_optimizer
from LibMTL.evomtl.moea.lw_dust import LWDUST, run_lw_dust
from LibMTL.evomtl.moea.rdust import run_rdust
from LibMTL.evomtl.moea.rlw_dust import RLWDUST, run_rlw_dust
from test_dust import Data, batch, make_trainer
from test_rdust import ScalarTrainer, paired_noise, scalar_batches


def estimator(trainer, inner_steps, seed=5, cls=LWDUST):
    return cls(trainer, inner_steps, 8, .03, 2, 'equal', 8, .03, torch.Generator().manual_seed(seed))


def exact_layerwise(trainer, batches, optimizer, steps, order=('input', 'output')):
    """Backprop reference: ``steps[name]`` exact steps on each layer, in ``order``."""
    for name in order:
        parameter = getattr(trainer.model, name).weight
        for _ in range(steps[name]):
            loss = sum(trainer.forward4loss(trainer.model, x, y).sum() for x, y in batches) / len(batches)
            gradient, = torch.autograd.grad(loss, parameter)
            optimizer.zero_grad(set_to_none=True)
            parameter.grad = gradient
            optimizer.step()


class LWDUSTTests(unittest.TestCase):
    def test_forward_order_and_inner_steps_match_exact_block_descent(self):
        """Registration order is output, input; LW-DUST trains input for T=2 steps, then output for T=3, and with
        exact probes reproduces backprop block-coordinate descent step by step."""
        trainer, reference = ScalarTrainer(), ScalarTrainer()
        batches = scalar_batches()
        exact_layerwise(reference, batches, torch.optim.SGD(reference.model.parameters(), lr=.05),
                        {'input': 2, 'output': 3})
        with contextlib.redirect_stdout(io.StringIO()):
            dust = estimator(trainer, [2, 3])
            paired_noise(dust)
            optimizer = torch.optim.SGD(trainer.model.parameters(), lr=.05)
            try:
                dust._capture(*batches[0])
                self.assertEqual(list(dust.groups), ['input', 'output'])
                self.assertEqual(dust.steps, {'input': 2, 'output': 3})
                for name in dust.groups:
                    for _ in range(dust.steps[name]):
                        dust.inner_step(name, batches, optimizer)
            finally:
                dust.close()
        for actual, target in zip(trainer.model.parameters(), reference.model.parameters()):
            torch.testing.assert_close(actual, target, rtol=3e-5, atol=3e-6)

    def test_optimizer_state_advances_only_for_the_active_group(self):
        for name in ('sgd', 'adam', 'rmsprop'):
            with self.subTest(optimizer=name), contextlib.redirect_stdout(io.StringIO()):
                trainer = ScalarTrainer()
                params = list(trainer.model.parameters())  # [output, input]
                optimizer = make_es_optimizer(params, optim=name, lr=.01, sgd_momentum=.8)
                dust = estimator(trainer, [3, 1])
                paired_noise(dust)
                try:
                    dust._capture(*scalar_batches()[0])
                    before = params[0].detach().clone()
                    for _ in range(3):
                        dust.inner_step('input', scalar_batches(), optimizer)
                    torch.testing.assert_close(params[0], before, rtol=0, atol=0)  # output untouched
                    if name != 'sgd':
                        self.assertNotIn(params[0], optimizer.state)
                        self.assertEqual(int(optimizer.state[params[1]]['step']), 3)
                    dust.inner_step('output', scalar_batches(), optimizer)
                    if name != 'sgd':
                        self.assertEqual([int(optimizer.state[p]['step']) for p in params], [1, 3])
                finally:
                    dust.close()

    def test_groups_include_free_normalization_and_validate_counts(self):
        with contextlib.redirect_stdout(io.StringIO()):
            trainer = make_trainer(norm=True)  # shared head: one site called once per task
            dust = LWDUST(trainer, [1, 2, 0, 1], 24, .05, 4, 'equal', 4, .03, torch.Generator().manual_seed(4))
            optimizer = torch.optim.Adam(trainer.model.parameters(), lr=.001)
            try:
                dust._capture(*batch())
                self.assertEqual(list(dust.groups), ['encoder.0', 'encoder.3', 'encoder.4', 'decoders.a'])
                for name in dust.groups:
                    for _ in range(dust.steps[name]):
                        dust.inner_step(name, [batch()], optimizer)
            finally:
                dust.close()
            norm = trainer.model.encoder[4]
            self.assertNotIn(norm.weight, optimizer.state)  # T = 0 freezes the group
            self.assertEqual(int(optimizer.state[trainer.model.encoder[3].weight]['step']), 2)
            bad = LWDUST(make_trainer(), [1, 1], 24, .05, 4, 'equal', 0, .03, torch.Generator())
            with self.assertRaisesRegex(ValueError, 'inner step counts'):
                bad._capture(*batch())
            bad.close()

    def test_cli_runner_counts_iterations_and_uses_no_backprop(self):
        params = LibMTL_args.parse_args(['--evo_training', '--evo_moea', 'lw_dust', '--evo_iterations', '2',
                                         '--evo_pop_size', '32', '--evo_es_grad_cos_freq', '0',
                                         '--evo_eval_freq', '0', '--evo_dust_inner_steps', '2', '1', '1'])
        evo = _make_evo_args(params)
        self.assertEqual(evo['evo_kwargs']['inner_steps'], [2, 1, 1])
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            trainer = make_trainer(directory)
            loader = DataLoader(Data(), batch_size=8, shuffle=True)
            with patch('torch.autograd.grad', side_effect=AssertionError('LW-DUST update used backprop')):
                result = trainer._run_moea(evo['moea'].lower(), loader, None, evo['evo_kwargs'])
            with (Path(directory) / 'metrics.csv').open() as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual(result['method'], 'LW-DUST')
        self.assertEqual([int(row['inner_steps']) for row in rows], [4, 4])
        self.assertEqual([int(row['total_inner_steps']) for row in rows], [4, 8])
        self.assertEqual(result['wandb_steps'], 8)
        theta = nn.utils.parameters_to_vector(trainer.model.parameters()).detach()
        self.assertFalse(torch.equal(theta, trainer.theta_0))
        torch.testing.assert_close(theta, torch.as_tensor(result['pareto_center_X']))
        self.assertFalse(any(module._forward_hooks for module in trainer.model.modules()))

    def test_runner_diagnostics_log_per_group_cosines(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()) as out:
            trainer = make_trainer(directory, shared=False)
            loader = DataLoader(Data(), batch_size=16, shuffle=True)
            run_lw_dust(trainer, loader, num_iterations=1, pop_size=4096, chunk=16, sigma=.01, allocation='width',
                        evo_eval_freq=0, grad_cos_freq=1)
            with (Path(directory) / 'metrics.csv').open() as stream:
                row, = csv.DictReader(stream)
        self.assertGreater(float(row['true_cos_direction']), .9)
        self.assertIn('decoders.b', out.getvalue())


class RLWDUSTTests(unittest.TestCase):
    def test_reverse_order_and_inner_steps_match_exact_block_descent(self):
        """Counts in forward order (input 2, output 3); RLW-DUST trains output for 3 steps, then input for 2."""
        trainer, reference = ScalarTrainer(), ScalarTrainer()
        batches = scalar_batches()
        exact_layerwise(reference, batches, torch.optim.SGD(reference.model.parameters(), lr=.05),
                        {'input': 2, 'output': 3}, order=('output', 'input'))
        with contextlib.redirect_stdout(io.StringIO()):
            dust = estimator(trainer, [2, 3], cls=RLWDUST)
            paired_noise(dust)
            optimizer = torch.optim.SGD(trainer.model.parameters(), lr=.05)
            try:
                dust._capture(*batches[0])
                self.assertEqual(list(dust.groups), ['output', 'input'])
                self.assertEqual(dust.steps, {'output': 3, 'input': 2})
                for name in dust.groups:
                    for _ in range(dust.steps[name]):
                        dust.inner_step(name, batches, optimizer)
            finally:
                dust.close()
        for actual, target in zip(trainer.model.parameters(), reference.model.parameters()):
            torch.testing.assert_close(actual, target, rtol=3e-5, atol=3e-6)

    def test_cli_runner_visits_groups_last_to_first(self):
        params = LibMTL_args.parse_args(['--evo_training', '--evo_moea', 'rlw_dust', '--evo_iterations', '2',
                                         '--evo_pop_size', '32', '--evo_es_grad_cos_freq', '0',
                                         '--evo_eval_freq', '0', '--evo_dust_inner_steps', '2', '1', '1'])
        evo = _make_evo_args(params)
        self.assertEqual(evo['evo_kwargs']['inner_steps'], [2, 1, 1])
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            trainer = make_trainer(directory)
            loader = DataLoader(Data(), batch_size=8, shuffle=True)
            visited = []
            inner_step = RLWDUST.inner_step

            def record(self, name, *args):
                visited.append(name)
                return inner_step(self, name, *args)

            with patch.object(RLWDUST, 'inner_step', record), \
                    patch('torch.autograd.grad', side_effect=AssertionError('RLW-DUST update used backprop')):
                result = trainer._run_moea(evo['moea'].lower(), loader, None, evo['evo_kwargs'])
            with (Path(directory) / 'metrics.csv').open() as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual(result['method'], 'RLW-DUST')
        self.assertEqual(visited, ['decoders.a', 'encoder.3', 'encoder.0', 'encoder.0'] * 2)
        self.assertEqual([int(row['total_inner_steps']) for row in rows], [4, 8])
        self.assertFalse(any(module._forward_hooks for module in trainer.model.modules()))


class ShuffleGuardTests(unittest.TestCase):
    def test_unshuffled_training_loader_is_rejected(self):
        for runner, method in ((run_lw_dust, 'LW-DUST'), (run_rlw_dust, 'RLW-DUST'), (run_rdust, 'RDUST')):
            with self.subTest(method), contextlib.redirect_stdout(io.StringIO()):
                trainer = make_trainer()
                with self.assertRaisesRegex(ValueError, f'^{method} .*shuffle=True'):
                    runner(trainer, DataLoader(Data(), batch_size=8), num_iterations=1, pop_size=32, chunk=8)
                self.assertTrue(torch.equal(nn.utils.parameters_to_vector(trainer.model.parameters()),
                                            trainer.theta_0))


if __name__ == '__main__':
    unittest.main()
