"""Dataset-free checks: python -m unittest discover -s tests -p 'test_multi_center_pareto_openes.py'."""
import random
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from LibMTL.config import LibMTL_args, _make_evo_args
from LibMTL.evomtl.evo_trainer import EvoMTLTrainer
from LibMTL.evomtl.moea.multi_center_pareto_openes import (
    CenterStatistics, antithetic_jacobian, coordinated_direction, pooled_sample_proposals,
    run_multi_center_pareto_openes,
)
from LibMTL.loss import CELoss
from LibMTL.metrics import AccMetric


class QuadraticTrainer:
    def __init__(self):
        self.theta_0 = torch.tensor([0.2, -0.4])
        self.model = nn.Linear(2, 1, bias=False)
        self.model.register_buffer('counter', torch.zeros(()))
        self.task_num = 2
        self.theta = self.theta_0.clone()
        self.batch_calls = 0
        self.evaluations = []

    def set_theta(self, theta):
        self.theta = torch.as_tensor(theta).clone()

    def _sample_batches(self, loaders, n):
        self.batch_calls += 1
        return [object()]

    def evaluate_multiobjective_losses(self, loaders, prefetched_batches, record_losses=True):
        assert not record_losses
        self.evaluations.append((id(prefetched_batches), self.theta.clone()))
        # Check that stochastic draws and mutable buffers are matched as well.
        self.model.counter.add_(1)
        shift = random.random() + np.random.random() + torch.rand(()).item()
        return np.array([((self.theta - 1)**2).sum().item(),
                         ((self.theta + 1)**2).sum().item()]) + shift


class TinyData(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        return torch.tensor([index / 4., 1.]), {'a': index % 2, 'b': (index+1) % 2}


class ParetoOpenESTests(unittest.TestCase):
    def test_raw_vector_gradient_sign_and_scale(self):
        noise = torch.eye(2, dtype=torch.float64) * np.sqrt(2)
        center = torch.tensor([0.25, -0.5], dtype=torch.float64)
        def f(x):
            return torch.stack(((x - 1).square().sum(), (x + 1).square().sum()))
        sigma = 0.02
        J = antithetic_jacobian(torch.stack([f(center+sigma*e) for e in noise]),
                                torch.stack([f(center-sigma*e) for e in noise]), noise, sigma)
        torch.testing.assert_close(J, torch.stack((2*(center-1), 2*(center+1))))

    def test_descent_and_spreading_constraints(self):
        rng = np.random.default_rng(123)
        for tasks in (2, 3, 5):
            for _ in range(8):
                J = rng.normal(size=(tasks, 9))
                d, s = coordinated_direction(J, rng.normal(size=tasks), 0.2, 0.03, 0.6)
                self.assertLessEqual(np.max(J @ d), 1e-7)
                self.assertLessEqual(np.linalg.norm(s), 0.2 + 1e-7)
                self.assertLessEqual(np.max(J @ (d+s) - 0.6*(J@d)), 0.03 + 1e-7)

    def test_stationary_conflicting_front_can_spread(self):
        J = np.array([[1., 0.], [-1., 0.]])
        d, s = coordinated_direction(J, np.array([1., -1.]), 0.2, 0.05, 0.5)
        np.testing.assert_allclose(d, 0, atol=1e-12)
        self.assertGreater(np.linalg.norm(s), 0)
        self.assertLessEqual(np.max(J @ s), 0.05+1e-7)
        _, strict = coordinated_direction(J, np.array([1., -1.]), 0.2, 0, 0.5)
        np.testing.assert_allclose(strict, 0, atol=1e-7)

    def test_zero_jacobian(self):
        d, s = coordinated_direction(np.zeros((2, 3)), [1, -1], 1, 0.1, 0.5)
        np.testing.assert_array_equal(d+s, np.zeros(3))

    def test_center_ema(self):
        stats = CenterStatistics(0.5)
        stats.update(torch.tensor([[1., 2.]]), torch.ones(1, 2, 3))
        stats.update(torch.tensor([[3., 4.]]), torch.ones(1, 2, 3)*3)
        torch.testing.assert_close(stats.loss_mean, torch.tensor([[2., 3.]]))
        torch.testing.assert_close(stats.loss_var, torch.ones(1, 2))
        self.assertEqual(stats.count, 2)

    def test_shared_batches_persistent_centers_and_restoration(self):
        trainer = QuadraticTrainer()
        out = run_multi_center_pareto_openes(trainer, None, 2, 4, num_centers=3, seed=2)
        self.assertEqual(trainer.batch_calls, 3)  # two iterations + final reevaluation
        self.assertEqual(out['center_statistics']['count'], 2)
        self.assertEqual(out['centers_X'].shape, (3, 2))
        self.assertEqual(out['centers_F'].shape, (3, 2))
        self.assertIsNone(out['hv_weights'])
        self.assertEqual(len(trainer.evaluations), 2*3*5 + 3)
        for offset in (0, 15):
            self.assertEqual(len({b for b, _ in trainer.evaluations[offset:offset+15]}), 1)
        torch.testing.assert_close(trainer.theta, torch.tensor(out['centers_X'][0]))
        self.assertTrue(trainer.model.training)
        self.assertEqual(trainer.model.counter.item(), 0)
        # A deterministic replay with stochastic evaluation must reproduce centers.
        random.seed(10); np.random.seed(10); torch.manual_seed(10)
        a = run_multi_center_pareto_openes(QuadraticTrainer(), None, 2, 4, seed=2)
        random.seed(10); np.random.seed(10); torch.manual_seed(10)
        b = run_multi_center_pareto_openes(QuadraticTrainer(), None, 2, 4, seed=2)
        np.testing.assert_array_equal(a['centers_X'], b['centers_X'])

    def test_actual_updates_respect_trust_bound(self):
        out = run_multi_center_pareto_openes(QuadraticTrainer(), None, 1, 4,
                    num_centers=3, init_std=0, lr=100, max_step_norm=0.005)
        self.assertTrue((np.linalg.norm(out['centers_X'] - [0.2, -0.4], axis=1) <= 0.005001).all())

    def test_quadratic_centers_approach_front_and_remain_distinct(self):
        # For ||theta-1||² and ||theta+1||², the front in parameter
        # space is theta=(t,t), -1<=t<=1. Initially theta=(.2,-.4).
        with contextlib.redirect_stdout(io.StringIO()):
            out = run_multi_center_pareto_openes(QuadraticTrainer(), None, 100, 32,
                        num_centers=3, lr=0.1, stats_beta=0, tradeoff=0.05,
                        spread_radius=0.2, seed=7)
        X = out['centers_X']
        self.assertLess(float(np.max(np.abs(X[:, 0]-X[:, 1]))), 0.15)
        self.assertGreater(float(np.ptp(X.mean(axis=1))), 0.01)

    def test_pool_uses_all_sources_with_center_specific_weights(self):
        centers = np.array([[-0.1], [0.1]])
        samples = np.array([[-0.5], [0.3], [-0.3], [0.5]])
        f = lambda x: np.column_stack(((x[:, 0]-1)**2, (x[:, 0]+1)**2))
        args = (centers, samples, f(samples), f(centers), np.ones(2), np.array([[1., -1.], [-1., 1.]]))
        proposals, weights, targets = pooled_sample_proposals(*args, 0.5, 0.5)
        np.testing.assert_allclose(weights.sum(axis=1), 1)
        self.assertTrue((weights > 0).all())
        self.assertFalse(np.allclose(weights[0], weights[1]))
        self.assertLess(proposals[0, 0], 0)
        self.assertGreater(proposals[1, 0], 0)
        # Changing only a foreign source's losses affects receiving center 0.
        changed = f(samples)
        changed[2:] += 4
        new_proposals, _, _ = pooled_sample_proposals(
            centers, samples, changed, f(centers), np.ones(2), args[-1], 0.5, 0.5)
        self.assertFalse(np.allclose(proposals[0], new_proposals[0]))
        # Per-objective unit changes are absorbed by the shared scale.
        units = np.array([3., 0.5])
        scaled, scaled_weights, _ = pooled_sample_proposals(
            centers, samples, f(samples)*units, f(centers)*units, units, args[-1], 0.5, 0.5)
        np.testing.assert_allclose(scaled, proposals)
        np.testing.assert_allclose(scaled_weights, weights)

    def test_uniform_pool_weights_cancel_antithetic_offsets_without_consensus(self):
        centers = np.array([[-3., 1.], [2., -4.]])
        eps = np.array([[0.2, 0.3], [-0.2, -0.3]])
        samples = np.concatenate([c + eps for c in centers])
        proposals, weights, _ = pooled_sample_proposals(
            centers, samples, np.ones((4, 2)), np.ones((2, 2)), np.ones(2), np.zeros((2, 2)))
        np.testing.assert_allclose(weights, 0.25)
        np.testing.assert_allclose(proposals, 0, atol=1e-15)

    def test_external_pool_direction_keeps_nullspace_transfer_and_bounds(self):
        J = np.array([[1., 0., 0.]])
        d, s = coordinated_direction(J, [0.], 0.2, 0, 0.5,
                                      desired_direction=np.array([0., 1., 0.]))
        self.assertGreater(s[1], 0.19)
        np.testing.assert_allclose(J @ s, 0, atol=1e-9)
        rng = np.random.default_rng(72)
        for _ in range(20):
            J = rng.normal(size=(3, 9))
            d, s = coordinated_direction(J, np.zeros(3), 0.2, 0.01, 0.7,
                                          desired_direction=rng.normal(size=9))
            self.assertLessEqual(np.linalg.norm(s), 0.2+1e-7)
            self.assertLessEqual(np.max(J@(d+s)-0.7*(J@d)), 0.01+1e-7)

    def test_pool_does_not_relabel_biased_recombination_as_gradient(self):
        def run(mode):
            random.seed(23); np.random.seed(23); torch.manual_seed(23)
            trainer = QuadraticTrainer()
            out = run_multi_center_pareto_openes(trainer, None, 1, 8, num_centers=3,
                                                sample_pool=mode, seed=4)
            self.assertEqual(len(trainer.evaluations), 3*9+3)
            return out
        local, pooled = run('off'), run('kernel')
        torch.testing.assert_close(local['center_statistics']['grad_mean'],
                                   pooled['center_statistics']['grad_mean'])
        torch.testing.assert_close(local['center_statistics']['loss_mean'],
                                   pooled['center_statistics']['loss_mean'])
        self.assertFalse(np.array_equal(local['centers_X'], pooled['centers_X']))
        self.assertEqual(pooled['last_pool']['weights'].shape, (3, 24))
        self.assertTrue((pooled['pool_diagnostics'][0]['foreign_mass'] > 0).all())
        self.assertTrue((pooled['pool_diagnostics'][0]['effective_samples'] >= 1).all())
        # Pooling stores no perturbation parameter archive.
        self.assertNotIn('samples_X', pooled['last_pool'])

    def test_pool_kernel_stability_and_invalid_inputs(self):
        proposals, weights, _ = pooled_sample_proposals(
            [[0.]], [[-1.], [1.]], [[1e6], [1e6+1]], [[0.]], [1.], [[0.]], 0.1, 1e-4)
        self.assertTrue(np.isfinite(proposals).all())
        self.assertTrue(np.isfinite(weights).all())
        np.testing.assert_allclose(weights.sum(axis=1), 1)
        for kw in ({'sample_pool': 'unknown'}, {'pool_temperature': 0},
                   {'pool_target_distance': float('nan')}):
            with self.assertRaises(ValueError):
                run_multi_center_pareto_openes(QuadraticTrainer(), None, 1, 4, **kw)

    def test_zero_iterations(self):
        out = run_multi_center_pareto_openes(QuadraticTrainer(), None, 0, 2, num_centers=1)
        self.assertEqual(out['center_statistics']['count'], 0)
        self.assertEqual(out['centers_F'].shape, (1, 2))

    def test_invalid_configuration_and_failure_restore(self):
        for kw in ({'pop_size': 3}, {'sigma': 0}, {'stats_beta': 1}, {'tradeoff': -1}, {'lr': float('nan')}):
            args = dict(num_iterations=1, pop_size=4)
            args.update(kw)
            with self.assertRaises(ValueError):
                run_multi_center_pareto_openes(QuadraticTrainer(), None, **args)
        trainer = QuadraticTrainer()
        def fail(*args, **kwargs):
            raise RuntimeError('evaluation failed')
        trainer.evaluate_multiobjective_losses = fail
        with self.assertRaises(RuntimeError):
            run_multi_center_pareto_openes(trainer, None, 1, 4)
        torch.testing.assert_close(trainer.theta, trainer.theta_0)

    def test_cli_configuration(self):
        params = LibMTL_args.parse_args(['--evo_training', '--evo_moea', 'multi-center-pareto-openes',
                                         '--evo_mc_centers', '3', '--evo_mc_sample_pool', 'kernel'])
        evo = _make_evo_args(params)
        self.assertEqual(evo['evo_kwargs']['num_centers'], 3)
        self.assertEqual(evo['evo_kwargs']['sample_pool'], 'kernel')
        self.assertNotIn('fitness_shaping', evo['evo_kwargs'])

    def test_real_trainer_checkpoint_preserves_all_centers(self):
        tasks = {t: dict(metrics=['Acc'], metrics_fn=AccMetric(), loss_fn=CELoss(), weight=[1])
                 for t in ('a', 'b')}
        with tempfile.TemporaryDirectory() as directory:
            trainer = EvoMTLTrainer(task_dict=tasks, weighting='EW', architecture='HPS',
                        encoder_class=lambda: nn.Linear(2, 3),
                        decoders=nn.ModuleDict({t: nn.Linear(3, 2) for t in tasks}),
                        rep_grad=False, multi_input=False,
                        optim_param={'optim': 'sgd', 'lr': 0.01},
                        scheduler_param=None, save_path=directory,
                        weight_args={}, arch_args={})
            loader = DataLoader(TinyData(), batch_size=2)
            trainer.prepare_evolution()
            run_multi_center_pareto_openes(trainer, loader, 1, 2, num_centers=2)
            self.assertTrue(all(not loss.record for loss in trainer.meter.losses.values()))
            out = trainer.evolve(loader, loader, moea='multi_center_pareto_openes',
                    evo_kwargs=dict(num_iterations=1, pop_size=4, num_centers=2, evo_eval_freq=1, sample_pool="kernel"))
            saved = torch.load(Path(directory)/'evo_result.pt', weights_only=False)
            np.testing.assert_array_equal(saved['centers_X'], out['centers_X'])
            self.assertEqual(saved['center_statistics']['count'], 1)
            self.assertEqual(saved['sample_pool'], 'kernel')
            self.assertEqual(saved['last_pool']['weights'].shape, (2, 8))
            self.assertEqual([e['iteration'] for e in saved['center_evaluations']], [0, 1])
            self.assertEqual(len(saved['center_evaluations'][-1]['metrics']), 2)
            self.assertEqual(len(saved['center_evaluations'][-1]['parameter_distances']), 1)
            self.assertEqual(len(saved['pareto_X']), 2)
            self.assertTrue((Path(directory)/'model_evo.pt').exists())


if __name__ == '__main__':
    unittest.main()
