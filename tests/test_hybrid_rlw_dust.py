"""Hybrid-RLW-DUST: DUST for a share of the mini-batch budget, then RLW-DUST, with one optimizer throughout."""

import contextlib
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
from LibMTL.evomtl.moea.dust import run_dust
from LibMTL.evomtl.moea.hybrid_rlw_dust import run_hybrid_rlw_dust
from LibMTL.evomtl.moea.rlw_dust import run_rlw_dust
from test_dust import Data, make_trainer

ARGS = dict(pop_size=24, chunk=4, sigma=.05, lr=.01, seed=3, evo_eval_freq=0, grad_cos_freq=0)


def final_weights(runner, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        trainer = make_trainer()
        runner(trainer, DataLoader(Data(), batch_size=8, shuffle=True), **ARGS, **kwargs)
    return nn.utils.parameters_to_vector(trainer.model.parameters()).detach()


class HybridRLWDUSTTests(unittest.TestCase):
    def test_all_dust_budget_reproduces_dust(self):
        torch.testing.assert_close(final_weights(run_hybrid_rlw_dust, num_iterations=6, dust_fraction=1.0),
                                   final_weights(run_dust, num_iterations=6), rtol=1e-5, atol=1e-6)

    def test_no_dust_budget_reproduces_rlw_dust(self):
        # 3 groups (conv, linear, shared head) x T = 2: 6 mini-batches per iteration, a budget of 12 = 2 iterations
        torch.testing.assert_close(
            final_weights(run_hybrid_rlw_dust, num_iterations=12, dust_fraction=0.0, inner_steps=[2]),
            final_weights(run_rlw_dust, num_iterations=2, inner_steps=[2]), rtol=1e-5, atol=1e-6)

    def test_budget_split_logging_and_cli(self):
        params = LibMTL_args.parse_args(['--evo_training', '--evo_moea', 'hybrid_rlw_dust', '--evo_iterations', '20',
                                         '--evo_pop_size', '24', '--evo_dust_chunk', '4', '--evo_dust_fraction',
                                         '.5', '--evo_dust_inner_steps', '1', '--evo_es_grad_cos_freq', '0',
                                         '--evo_eval_freq', '0'])
        evo = _make_evo_args(params)
        self.assertEqual((evo['evo_kwargs']['dust_fraction'], evo['evo_kwargs']['inner_steps']), (.5, [1]))
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            trainer = make_trainer(directory)
            loader = DataLoader(Data(), batch_size=8, shuffle=True)
            with patch('torch.autograd.grad', side_effect=AssertionError('Hybrid-RLW-DUST update used backprop')):
                result = trainer._run_moea(evo['moea'].lower(), loader, None, evo['evo_kwargs'])
            with (Path(directory) / 'metrics.csv').open() as stream:
                rows = list(csv.DictReader(stream))
        # 10 DUST iterations, then (20 - 10) // 3 = 3 RLW-DUST iterations of 3 mini-batches; 1 mini-batch unused
        self.assertEqual(result['method'], 'Hybrid-RLW-DUST')
        self.assertEqual([row['phase'] for row in rows], ['DUST'] * 10 + ['RLW-DUST'] * 3)
        self.assertEqual([int(row['total_inner_steps']) for row in rows], list(range(1, 11)) + [13, 16, 19])
        self.assertEqual(result['wandb_steps'], 19)
        self.assertFalse(any(module._forward_hooks for module in trainer.model.modules()))

    def test_diagnostics_in_both_phases(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            trainer = make_trainer(directory, shared=False)
            run_hybrid_rlw_dust(trainer, DataLoader(Data(), batch_size=16, shuffle=True), num_iterations=8,
                                pop_size=4096, chunk=16, sigma=.01, allocation='width', dust_fraction=.5,
                                inner_steps=[1], evo_eval_freq=0, grad_cos_freq=1)
            with (Path(directory) / 'metrics.csv').open() as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual([row['phase'] for row in rows], ['DUST'] * 4 + ['RLW-DUST'])  # 4 groups x T = 1
        self.assertTrue(all(float(row['true_cos_direction']) > .9 for row in rows))


if __name__ == '__main__':
    unittest.main()
