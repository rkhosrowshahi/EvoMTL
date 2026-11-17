"""Joint-DUST: groups of one layer reproduce DUST / A-DUST, joint jitter converges to backprop, per-draw inputs."""
import contextlib
import csv
import io
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from LibMTL.config import LibMTL_args, _make_evo_args
from LibMTL.evomtl.moea.dust import DUST
from LibMTL.evomtl.moea.joint_dust import JointDUST
from test_dust import Data, batch, cos, make_trainer, true_jacobian


class JointDUSTTests(unittest.TestCase):
    def test_groups_of_one_layer_reproduce_dust_and_a_dust(self):
        """One layer per group: its draws see the clean input, the chunk centering is DUST's and the clean baseline
        cancels in an antithetic pair, so the estimate is DUST's (A-DUST's) for the same noise, layers in the same
        order."""
        for antithetic in (False, True):
            with self.subTest(antithetic=antithetic), contextlib.redirect_stdout(io.StringIO()):
                trainer = make_trainer(shared=False)  # forward order = module order: conv, linear, head a, head b
                x, y = batch()
                args = (trainer, 48, .05, 4, 'equal', 0, .03)
                dust = DUST(*args, torch.Generator().manual_seed(2), antithetic=antithetic)
                dust._capture(x, y)  # sets the draws up without touching the noise stream
                dust.draws = {n: dust.draws[n] for n in dust.inputs}  # estimate the layers in forward order
                J_dust, _ = dust.estimate(x, y)
                dust.close()
                J, _ = JointDUST(*args, torch.Generator().manual_seed(2), antithetic=antithetic,
                                 groups=[1, 1, 1, 1], balance=True).estimate(x, y)  # nothing to balance alone
            torch.testing.assert_close(J, J_dust, rtol=1e-4, atol=1e-6)  # float32 rounding

    def test_joint_estimate_converges_to_backprop(self):
        for antithetic in (False, True):
            with self.subTest(antithetic=antithetic), contextlib.redirect_stdout(io.StringIO()):
                trainer = make_trainer(shared=False)
                x, y = batch()
                dust = JointDUST(trainer, 8192, .01, 16, 'equal', 0, .03, torch.Generator().manual_seed(0),
                                 antithetic=antithetic)  # one group: every layer jittered in every pass
                J, _ = dust.estimate(x, y)
            self.assertEqual(list(dust.groups.values()), [['encoder.0', 'encoder.3', 'decoders.a', 'decoders.b']])
            self.assertGreater(cos(J[0], true_jacobian(trainer, x, y).sum(0)), .95)

    def test_later_layers_use_each_draws_jittered_input(self):
        """In a group, the linear layer after the conv sees the conv's jitter: the recorded inputs of a draw differ
        from the clean input, the conv's do not."""
        with contextlib.redirect_stdout(io.StringIO()):
            trainer = make_trainer(shared=False)
            x, y = batch()
            dust = JointDUST(trainer, 16, .05, 8, 'equal', 0, .03, torch.Generator().manual_seed(0), groups=[2, 2])
            dust.estimate(x, y)
        clean = {n: dust.inputs[n][0].repeat(8, *(1,) * (dust.inputs[n][0].dim() - 1)) for n in dust.inputs}
        # last draw: the group of the two heads; the first layer of a group uses the cached clean input instead
        self.assertEqual(set(dust.drawn), {'decoders.b'})
        torch.testing.assert_close(dust.drawn['decoders.b'][0], clean['decoders.b'])  # parallel head: clean input
        dust.recording, dust.drawn = {'encoder.0', 'encoder.3'}, {}
        dust.jitter = {n: [dust.sigma * a for a in dust._sample_noise(n, 8 * len(x))] for n in dust.recording}
        dust.losses(x.repeat(8, 1, 1, 1), {t: v.repeat(8) for t, v in y.items()})
        dust.jitter, dust.recording = {}, set()
        torch.testing.assert_close(dust.drawn['encoder.0'][0], clean['encoder.0'])
        self.assertFalse(torch.allclose(dust.drawn['encoder.3'][0], clean['encoder.3']))

    def test_split_half_energy_is_unbiased_and_balances_the_scales(self):
        """The product of the two half estimates recovers each layer's output-gradient energy sum_b ||dL[b]/dy[b]||^2,
        and the scales then equalize sigma_l^2 E_l, the least energetic layer keeping sigma."""
        with contextlib.redirect_stdout(io.StringIO()):
            trainer = make_trainer(shared=False)
            x, y = batch()
            model = trainer.model
            outs = {}
            names = ['encoder.0', 'encoder.3', 'decoders.a', 'decoders.b']
            modules = dict(model.named_modules())
            hooks = [modules[n].register_forward_hook(lambda m, a, o, n=n: outs.__setitem__(n, o.retain_grad() or o))
                     for n in names]
            preds = model(x)
            sum(torch.nn.functional.cross_entropy(preds[t], y[t], reduction='sum') for t in y).backward()
            for h in hooks:
                h.remove()
            model.zero_grad()
            true = {n: float((outs[n].grad ** 2).sum()) for n in names}
            dust = JointDUST(trainer, 16384, .01, 16, 'equal', 0, .03, torch.Generator().manual_seed(0),
                             balance=True, balance_decay=0.)
            dust.estimate(x, y)
        for n in names:
            self.assertLess(abs(dust.energy[n] / true[n] - 1), .25, n)
        low = min(dust.energy, key=dust.energy.get)
        self.assertEqual(dust.scale[low], 1.)
        for n in names:
            self.assertAlmostEqual(dust.scale[n] ** 2 * max(dust.energy[n], .05 ** 2 * max(dust.energy.values())),
                                   dust.energy[low], places=6)

    def test_group_sizes_must_cover_the_layers(self):
        with contextlib.redirect_stdout(io.StringIO()):
            trainer = make_trainer(shared=False)
            with self.assertRaisesRegex(ValueError, 'add up'):
                JointDUST(trainer, 32, .05, 8, 'equal', 0, .03, torch.Generator(), groups=[2, 1]).estimate(*batch())

    def test_cli_trains_and_reports(self):
        params = LibMTL_args.parse_args(['--evo_training', '--evo_moea', 'joint_dust', '--evo_iterations', '3',
                                         '--evo_pop_size', '32', '--evo_dust_chunk', '8', '--evo_dust_groups', '2',
                                         '2', '--evo_dust_antithetic', '--evo_dust_balance', '--evo_eval_freq', '0'])
        evo = _make_evo_args(params)
        self.assertEqual([evo['evo_kwargs'][k] for k in ('groups', 'antithetic', 'balance', 'balance_decay')],
                         [[2, 2], True, True, .9])
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            trainer = make_trainer(directory, shared=False)
            loader = DataLoader(Data(), batch_size=8, shuffle=True)
            out = trainer._run_moea(evo['moea'].lower(), loader, None, evo['evo_kwargs'])
            with (Path(directory) / 'metrics.csv').open() as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual(out['method'], 'Balanced-Joint-A-DUST')
        self.assertEqual(len(rows), 3)
        self.assertTrue(rows[-1]['true_cos_direction'])
        theta = torch.nn.utils.parameters_to_vector(trainer.model.parameters()).detach()
        self.assertFalse(torch.equal(theta, trainer.theta_0))
        self.assertFalse(any(m._forward_hooks for m in trainer.model.modules()))


if __name__ == '__main__':
    unittest.main()
