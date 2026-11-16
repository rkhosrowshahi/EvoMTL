"""Full-test cadence follows completed generation numbers, with explicit first/final evaluations."""

import contextlib
import csv
import io
from pathlib import Path
import tempfile
import unittest
import types
from unittest.mock import patch

from torch.utils.data import DataLoader

from LibMTL.evomtl.moea.common import epoch_fraction_eval_steps, should_run_full_test
from LibMTL.evomtl.moea.dust import run_dust
from LibMTL.evomtl.moea.rdust import run_rdust
from test_dust import Data, make_trainer


def evaluated_generations(n_steps, every):
    return [step + 1 for step in range(n_steps) if should_run_full_test(step, n_steps, every)]


class EvaluationCadenceTests(unittest.TestCase):
    def test_interval_uses_completed_generation_numbers(self):
        self.assertEqual(evaluated_generations(102, 50), [1, 50, 100, 102])
        self.assertNotIn(51, evaluated_generations(102, 50))

    def test_ten_epoch_budget_keeps_first_interval_and_final_evaluations(self):
        self.assertEqual(evaluated_generations(2344, 50), [1, *range(50, 2344, 50), 2344])

    def test_every_generation_and_short_runs(self):
        self.assertEqual(evaluated_generations(5, 1), [1, 2, 3, 4, 5])
        self.assertEqual(evaluated_generations(1, 50), [1])
        self.assertEqual(evaluated_generations(2, 50), [1, 2])
        self.assertFalse(should_run_full_test(0, 0, 50))

    def test_nonpositive_frequency_disables_in_loop_tests(self):
        for every in (0, -1, -50):
            with self.subTest(every=every):
                self.assertEqual(evaluated_generations(5, every), [])

    def test_explicit_zero_based_indices_remain_exact(self):
        for indices in ({4, 51}, frozenset({4, 51})):
            with self.subTest(type=type(indices).__name__):
                self.assertEqual(evaluated_generations(102, indices), [5, 52, 102])

    def test_epoch_fraction_schedule_is_unchanged(self):
        indices = epoch_fraction_eval_steps(n_steps=102, iters_per_epoch=50, per_epoch=2)
        self.assertEqual(evaluated_generations(102, indices), [1, 25, 50, 75, 100, 102])

    def test_rdust_evaluates_and_records_the_expected_generations(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            trainer = make_trainer(directory)
            loader = DataLoader(Data(), batch_size=8, shuffle=True)
            with patch.object(trainer, "test", wraps=trainer.test) as evaluate:
                run_rdust(trainer, loader, num_iterations=5, pop_size=24, chunk=4, lr=.001,
                           optim="sgd", test_dataloaders=loader, evo_eval_freq=2, grad_cos_freq=0,
                           wandb_step_offset=1)
            with (Path(directory) / "metrics.csv").open() as stream:
                rows = list(csv.DictReader(stream))

        self.assertEqual([call.kwargs["wandb_log_step"] for call in evaluate.call_args_list], [1, 2, 4, 5])
        for task in ("a", "b"):
            recorded = [int(row["generation"]) for row in rows if row[f"test_loss_{task}"]]
            self.assertEqual(recorded, [1, 2, 4, 5])

    def test_wandb_test_epoch_equals_the_generation_step(self):
        """W&B test logs: initialization at step 0, then generations 1, 50, 100, 150, with test/epoch equal to the
        step (no +1 as for a 0-based epoch index)."""
        logged = []
        wandb = types.SimpleNamespace(log=lambda d, step=None: logged.append((step, d.get("test/epoch"))),
                                      Image=lambda figure: None)
        with patch.dict("sys.modules", {"wandb": wandb}), contextlib.redirect_stdout(io.StringIO()):
            trainer = make_trainer()
            trainer._wandb_enabled = True
            loader = DataLoader(Data(), batch_size=8, shuffle=True)
            trainer.test(loader, epoch=None, mode="test", wandb_log_step=0, display_tag="init")
            run_dust(trainer, loader, num_iterations=150, pop_size=24, chunk=8, test_dataloaders=loader,
                     evo_eval_freq=50, grad_cos_freq=0, wandb_step_offset=1)
        self.assertEqual([entry for entry in logged if entry[1] is not None],
                         [(0, 0.0), (1, 1.0), (50, 50.0), (100, 100.0), (150, 150.0)])


if __name__ == "__main__":
    unittest.main()
