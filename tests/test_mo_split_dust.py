import contextlib
import csv
import io
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from LibMTL.evomtl.moea.dust import DUST
from LibMTL.evomtl.moea.mo_split_dust import SplitDUST, run_mo_split_dust
from test_dust import TASKS, Data, batch, make_trainer, true_jacobian

SPLIT = dict(solver='upgrad', gram_decay=0.0, norm_eps=1e-4, reg_eps=1e-4)


def split_dust(trainer, pop, seed, **kwargs):
    return SplitDUST(trainer, pop, .05, 4, 'equal', 0, .03, torch.Generator().manual_seed(seed), True,
                     **{**SPLIT, **kwargs})


class MOSplitDUSTTests(unittest.TestCase):
    def test_estimate_matches_per_task_dust(self):
        """Recording the chunk shares leaves the per-task estimate of DUST unchanged."""
        with contextlib.redirect_stdout(io.StringIO()):
            trainer = make_trainer()
            x, y = batch()
            J_dust, _ = DUST(trainer, 32, .05, 4, 'equal', 0, .03, torch.Generator().manual_seed(2),
                             per_task=True).estimate(x, y)
            J_split, _ = split_dust(trainer, 32, 2).estimate(x, y)
        torch.testing.assert_close(J_split, J_dust)

    def test_cross_chunk_gramian_removes_the_noise_inflation(self):
        """Averaged over seeds, the diagonal of J_l J_l^T exceeds that of the expected estimate by the noise
        variance; the cross-chunk Gramian does not."""
        with contextlib.redirect_stdout(io.StringIO()):
            trainer = make_trainer(shared=False)
            x, y = batch()
            naive, cross = 0, 0
            seeds = 40
            for seed in range(seeds):
                dust = split_dust(trainer, 64, seed)
                dust.self_products, dust.chunks = {}, {}
                J, _ = dust.estimate(x, y)
                cols = dust.columns['encoder.3']  # the shared Linear layer
                Jl = J[:, cols].double()
                M = dust.chunks['encoder.3']
                naive = naive + Jl @ Jl.T / seeds
                cross = cross + M / (M - 1) * (Jl @ Jl.T - dust.self_products['encoder.3']) / seeds
                dust.close()
            # centering over a chunk of n draws shrinks the expected estimate by (n - 1) / n (n = 4 here)
            Jt = true_jacobian(trainer, x, y)[:, cols].double() * 3 / 4
            G_true = Jt @ Jt.T
        naive_err = (naive.diagonal() - G_true.diagonal()).abs()
        cross_err = (cross.diagonal() - G_true.diagonal()).abs()
        self.assertTrue(bool((naive.diagonal() > G_true.diagonal()).all()))  # inflated by the noise
        self.assertTrue(bool((cross_err < naive_err / 2).all()))

    def test_antithetic_cross_chunk_gramian_matches_backprop_scale(self):
        """Antithetic pairs need no centering: the shares are unbiased, so the cross-chunk Gramian estimates the
        backprop Gramian itself, without the (n - 1) / n shrinkage of the centered draws."""
        with contextlib.redirect_stdout(io.StringIO()):
            trainer = make_trainer(shared=False)
            x, y = batch()
            cross, seeds = 0, 40
            for seed in range(seeds):
                dust = split_dust(trainer, 64, seed, antithetic=True)
                dust.self_products, dust.chunks = {}, {}
                J, _ = dust.estimate(x, y)
                cols = dust.columns['encoder.3']
                Jl, M = J[:, cols].double(), dust.chunks['encoder.3']
                cross = cross + M / (M - 1) * (Jl @ Jl.T - dust.self_products['encoder.3']) / seeds
                dust.close()
            Jt = true_jacobian(trainer, x, y)[:, cols].double()
        G_true = Jt @ Jt.T
        torch.testing.assert_close(cross.diagonal(), G_true.diagonal(), rtol=.2, atol=0)

    def test_exact_rule_avoids_conflicts_per_layer_and_scales_single_task_layers(self):
        with contextlib.redirect_stdout(io.StringIO()):
            trainer = make_trainer(shared=False)
            x, y = batch()
            for solver in ('upgrad', 'mgda'):
                dust = split_dust(trainer, 32, 0, solver=solver)
                dust.estimate(x, y)  # sets up the layers
                dust.self_products = {}  # no pending estimate: aggregate uses the exact Gramians
                J = true_jacobian(trainer, x, y)
                d, w = dust.aggregate(J, torch.ones(len(TASKS)))
                for name, cols in dust.columns.items():
                    Jl, dl = J[:, cols], d[cols]
                    present = Jl.norm(dim=1) > 0
                    with self.subTest(solver=solver, layer=name):
                        if bool(present.all()):
                            self.assertTrue(bool((Jl @ dl >= -1e-6 * dl.norm() * Jl.norm(dim=1)).all()))
                        else:  # one task's head: g_k / K
                            torch.testing.assert_close(dl, Jl[present][0] / len(TASKS))
                if solver == 'mgda':
                    self.assertAlmostEqual(float(w.sum()), 1.0, places=5)
                dust.close()

    def test_fewer_than_two_chunks_per_layer_is_rejected(self):
        with contextlib.redirect_stdout(io.StringIO()):
            trainer = make_trainer()
            x, y = batch()
            with self.assertRaisesRegex(ValueError, 'at least 2 chunks'):
                split_dust(trainer, 16, 0).estimate(x, y)  # 4 layers, one chunk of 4 each

    def test_runner_trains_and_reports(self):
        for solver, antithetic, method in (('upgrad', False, 'MO-UPGrad-Split-DUST'),
                                           ('mgda', False, 'MO-MGDA-Split-DUST'),
                                           ('upgrad', True, 'MO-UPGrad-Split-ADUST')):
            with self.subTest(method), tempfile.TemporaryDirectory() as directory, \
                    contextlib.redirect_stdout(io.StringIO()):
                trainer = make_trainer(directory)
                loader = DataLoader(Data(), batch_size=8, shuffle=True)
                out = run_mo_split_dust(trainer, loader, solver=solver, num_iterations=3, pop_size=64, sigma=.05,
                                        chunk=8, lr=.01, test_dataloaders=loader, evo_eval_freq=0, grad_cos_freq=1,
                                        num_batches=2, antithetic=antithetic)
                with (Path(directory) / 'metrics.csv').open() as stream:
                    rows = list(csv.DictReader(stream))
            self.assertEqual(out['method'], method)
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(rows[-1][f'weight_{t}'] and rows[-1][f'true_cos_{t}'] for t in TASKS))
            self.assertTrue(rows[-1]['true_cos_direction'])
            theta = torch.nn.utils.parameters_to_vector(trainer.model.parameters()).detach()
            self.assertFalse(torch.equal(theta, trainer.theta_0))
            self.assertFalse(any(m._forward_hooks for m in trainer.model.modules()))

    def test_invalid_options_are_rejected(self):
        with contextlib.redirect_stdout(io.StringIO()):
            trainer = make_trainer()
            loader = DataLoader(Data(), batch_size=8, shuffle=True)
            with self.assertRaisesRegex(ValueError, 'solver'):
                run_mo_split_dust(trainer, loader, solver='cagrad', num_iterations=1, pop_size=32, chunk=8)
            with self.assertRaisesRegex(ValueError, 'gram_decay'):
                run_mo_split_dust(trainer, loader, gram_decay=1.0, num_iterations=1, pop_size=32, chunk=8)


if __name__ == '__main__':
    unittest.main()
