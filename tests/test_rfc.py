"""RFC: the readout frame is exact for the decoders, the encoder draws converge, and the runner trains."""
import contextlib
import io
import unittest

import torch
from torch.utils.data import DataLoader

from LibMTL.evomtl.moea.rfc import RFC, run_rfc
from tests.test_dust import TASKS, Data, batch, cos, make_trainer, true_jacobian


def decoder_mask(trainer):
    return torch.cat([torch.full((p.numel(),), n.startswith('decoders.')) for n, p in trainer.model.named_parameters()])


class RFCTests(unittest.TestCase):
    def test_decoder_gradients_are_exact(self):
        """The coordinate differences give the decoders' per-task gradients up to O(sigma^2), shared or not."""
        for shared in (True, False):
            with contextlib.redirect_stdout(io.StringIO()):
                trainer = make_trainer(shared=shared)
                x, y = batch()
                J, _ = RFC(trainer, 8, 1e-3, 4, 'equal', 0, .03, torch.Generator().manual_seed(0),
                           per_task=True, frame_chunk=5).estimate(x, y)
            J_true = true_jacobian(trainer, x, y)
            mask = decoder_mask(trainer)
            torch.testing.assert_close(J[:, mask], J_true[:, mask], rtol=1e-3, atol=1e-5)

    def test_estimate_converges_to_backprop(self):
        with contextlib.redirect_stdout(io.StringIO()):
            trainer = make_trainer(shared=False)
            x, y = batch()
            J_sum, _ = RFC(trainer, 4096, .01, 16, 'width', 0, .03, torch.Generator().manual_seed(0)).estimate(x, y)
            J, _ = RFC(trainer, 4096, .01, 16, 'width', 0, .03, torch.Generator().manual_seed(0), per_task=True).estimate(x, y)
        J_true = true_jacobian(trainer, x, y)
        self.assertGreater(cos(J_sum[0], J_true.sum(0)), .97)
        for k in range(len(TASKS)):
            self.assertGreater(cos(J[k], J_true[k]), .97)

    def test_free_parameters_use_weight_space_es(self):
        with contextlib.redirect_stdout(io.StringIO()):
            trainer = make_trainer(norm=True)
            x, y = batch()
            rfc = RFC(trainer, 8, .05, 4, 'equal', 4, .03, torch.Generator().manual_seed(0))
            J, _ = rfc.estimate(x, y)
        self.assertTrue(rfc.free)
        self.assertTrue(torch.isfinite(J).all())

    def test_runner_lowers_the_loss(self):
        loader = DataLoader(Data(), batch_size=16, shuffle=True)
        with contextlib.redirect_stdout(io.StringIO()):
            trainer = make_trainer()
            x, y = batch()
            losses0 = RFC(trainer, 8, .05, 4, 'equal', 0, .03, torch.Generator().manual_seed(0)).estimate(x, y)[1].sum()
            run_rfc(trainer, loader, num_iterations=40, pop_size=32, lr=.05, sigma=.05, chunk=4, optim='sgd',
                    grad_cos_freq=0, es_draws=0)
        losses1 = RFC(trainer, 8, .05, 4, 'equal', 0, .03, torch.Generator().manual_seed(0)).estimate(x, y)[1].sum()
        self.assertLess(float(losses1), float(losses0))


if __name__ == '__main__':
    unittest.main()
