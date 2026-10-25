"""Fixed-width TPSL really optimizes only the full network vector."""
import contextlib
import csv
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from LibMTL.evomtl.evo_trainer import EvoMTLTrainer
from LibMTL.evomtl.moea.tpsl_es import run_tpsl_es, tangent_rows
from LibMTL.evomtl.moea.psl_es import simplex_basis
from LibMTL.loss import CELoss
from LibMTL.metrics import AccMetric


class Data(Dataset):
    def __len__(self): return 4
    def __getitem__(self, i):
        return torch.tensor([i/4., 1.]), {'a': i%2, 'b': (i+1)%2}


class FixedWidthTPSLTests(unittest.TestCase):
    def run_case(self, directory, c_lr):
        torch.manual_seed(17)
        tasks={t: dict(metrics=['Acc'],metrics_fn=AccMetric(),loss_fn=CELoss(),weight=[1]) for t in ('a','b')}
        trainer=EvoMTLTrainer(task_dict=tasks,weighting='EW',architecture='HPS',
            encoder_class=lambda: nn.Linear(2,3),
            decoders=nn.ModuleDict({t: nn.Linear(3,2) for t in tasks}),
            rep_grad=False,multi_input=False,optim_param={'optim':'sgd','lr':.01},
            scheduler_param=None,save_path=directory,weight_args={},arch_args={})
        trainer.prepare_evolution()
        loader=DataLoader(Data(),batch_size=2)
        calls=[]
        adam=torch.optim.Adam
        def record(params,*args,**kwargs):
            params=list(params)
            calls.append(sum(p.numel() for p in params))
            return adam(params,*args,**kwargs)
        with patch('torch.optim.Adam',side_effect=record):
            out=run_tpsl_es(trainer,loader,num_iterations=3,pop_size=8,c0=.25,c_lr=c_lr,
                            eval_points=3,optim='adam',test_dataloaders=loader,evo_eval_freq=0,
                            guided_k=1,guided_sigma=.05,guided_alpha=.9,guided_ema=.9)
        return trainer,out,calls,torch.load(Path(directory)/'tpsl_result.pt',weights_only=False)

    def test_only_network_vector_is_optimized_and_radius_fixed(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            trainer,out,calls,saved=self.run_case(directory,0)
            D=trainer.theta_0.numel()
            self.assertEqual(calls,[D])
            self.assertEqual(saved['optimized_num_parameters'],D)
            self.assertFalse(saved['width_learned'])
            self.assertEqual(saved['c'],.25)
            with (Path(directory)/'metrics.csv').open() as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual([float(row['c']) for row in rows],[.25]*3)
            torch.testing.assert_close(saved['B'],saved['c']*saved['T'].T)
            self.assertFalse(saved['T'].requires_grad)
            theta=out['pareto_center_X']
            X=out['pareto_X']
            np.testing.assert_allclose(X[1],theta,atol=1e-6)
            np.testing.assert_allclose(np.linalg.norm(X[[0,2]]-theta,axis=1),.25,atol=1e-6)
            self.assertTrue(np.isfinite(saved['test_F']).all())
            self.assertFalse(torch.equal(saved['theta'],trainer.theta_0.cpu()))

    def test_learned_width_control_has_separate_scalar_optimizer(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            trainer,out,calls,saved=self.run_case(directory,.05)
            self.assertEqual(calls,[trainer.theta_0.numel(),1])
            self.assertTrue(saved['width_learned'])
            self.assertEqual(saved['optimized_num_parameters'],trainer.theta_0.numel()+1)

    def test_computed_tangent_orientation_and_zero_gradient(self):
        G=torch.tensor([[1.,0.,0.],[0.,1.,0.]])
        idx=torch.arange(3)
        tangent=tangent_rows(G,simplex_basis(2),idx)
        expected=torch.tensor([[-1.,1.,0.]])/np.sqrt(2)
        torch.testing.assert_close(tangent,expected)
        torch.testing.assert_close(tangent_rows(torch.zeros_like(G),simplex_basis(2),idx),torch.zeros((1,3)))


if __name__=='__main__': unittest.main()
