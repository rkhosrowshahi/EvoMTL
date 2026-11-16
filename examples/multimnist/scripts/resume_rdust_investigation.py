"""Controlled continuation from a diagnostic checkpoint, preserving data/noise RNGs."""
import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'examples/multimnist'))
import numpy as np
import torch
import main as benchmark
from investigate_rdust import install_diagnostics
from LibMTL.config import LibMTL_args
from LibMTL.utils import set_device,set_random_seed
from LibMTL.evomtl.evo_trainer import EvoMTLTrainer
from LibMTL.evomtl.moea.rdust import RDUST
from LibMTL.evomtl.moea.es_common import make_es_optimizer, MetricsCSV, metrics_columns, report_search_mean
import LibMTL.evomtl.moea.es_common as reporting


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('checkpoint',type=Path)
    parser.add_argument('--end',type=int,default=310)
    parser.add_argument('--lr',type=float,default=.0005)
    parser.add_argument('--backtrack',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    opts=parser.parse_args();folder=opts.output.resolve();folder.mkdir(parents=True,exist_ok=False)
    (folder/'investigation.json').write_text(json.dumps({k:str(v) if isinstance(v,Path) else v for k,v in vars(opts).items()},indent=2))
    checkpoint=torch.load(opts.checkpoint,map_location='cpu',weights_only=False)
    start=checkpoint['generation'];stream=install_diagnostics(folder,5,start=start)
    torch.set_num_threads(1)
    reporting.should_run_full_test=lambda i,n,e:i%int(e)==0 or i==n-1
    def train(self,train_loader,test_loader,epochs):
        self.model.load_state_dict(checkpoint['model']);self.prepare_evolution()
        theta=torch.nn.utils.parameters_to_vector(self.model.parameters()).detach().cpu().clone()
        self.set_theta(theta)
        optimizer=make_es_optimizer(list(self.model.parameters()),'adam',.001)
        optimizer.load_state_dict(checkpoint['optimizer'])
        for group in optimizer.param_groups:group['lr']=opts.lr
        gen=torch.Generator(self.device);gen.set_state(checkpoint['noise_rng'])
        dust=RDUST(self,24,.05,8,'custom',8,.03,gen,draws=[128,128,16,16,16])
        acceptance_stream = (folder/'acceptance.csv').open('w', buffering=1) if opts.backtrack else None
        acceptance_writer = csv.DictWriter(acceptance_stream, fieldnames=['generation','block','pre_loss','proposal_loss','accepted_loss','scale','evaluations']) if opts.backtrack else None
        if opts.backtrack:
            acceptance_writer.writeheader()
            original_step = optimizer.step
            def guarded_step(*args, **kwargs):
                active = [p for p in self.model.parameters() if p.grad is not None]
                name = next(n for n, ps in dust.groups.items() if {id(p) for p in ps} == {id(p) for p in active})
                before = [p.detach().clone() for p in active]
                states = [copy.deepcopy(optimizer.state[p]) for p in active]
                def loss():
                    return float(sum(dust.losses(x,y).mean(1).sum() for x,y in batches)/len(batches))
                pre = loss()
                result = original_step(*args, **kwargs)
                proposal = [p.detach().clone() for p in active]
                proposed_loss = loss()
                accepted_loss, scale, evaluations = proposed_loss, 1.0, 1
                # Clean training loss alone guides acceptance, without exact gradients or test data.
                while (not np.isfinite(accepted_loss) or accepted_loss > pre) and scale > 1/32:
                    scale /= 2
                    for p,b,a in zip(active,before,proposal):p.copy_(b+scale*(a-b))
                    accepted_loss = loss(); evaluations += 1
                if not np.isfinite(accepted_loss) or accepted_loss > pre:
                    scale, accepted_loss = 0.0, pre
                    for p,b,state in zip(active,before,states):
                        p.copy_(b); optimizer.state[p] = state
                acceptance_writer.writerow(dict(generation=i+1,block=name,pre_loss=pre,proposal_loss=proposed_loss,
                    accepted_loss=accepted_loss,scale=scale,evaluations=evaluations))
                return result
            optimizer.step = guarded_step
        torch.set_rng_state(checkpoint['torch_rng'])
        tasks=list(self.task_name);writer=MetricsCSV(self,metrics_columns(tasks,['sigma'],[]))
        try:
            for i in range(start,opts.end):
                t=time.perf_counter();batches=self._sample_batches(train_loader,1);self.set_theta(theta)
                J,losses,_=dust.step(batches,optimizer)
                theta=torch.nn.utils.parameters_to_vector(self.model.parameters()).detach().cpu().clone()
                F=losses.numpy()[None].astype(np.float32);elapsed=time.perf_counter()-t
                train_metrics,test_metrics=report_search_mean(self,theta,i,opts.end,F,F,batches,test_loader,50,i)
                writer.write(i,elapsed,tasks,F,train_metrics,test_metrics,sigma=.05)
                print(f'Continuation {i+1}/{opts.end}: norm={float(J.norm()):.5g}',flush=True)
            torch.save(self.model.state_dict(),folder/'model_evo.pt')
        finally:
            writer.close();dust.close();stream.close()
            if acceptance_stream:acceptance_stream.close()
    EvoMTLTrainer.train=train
    sys.argv=['main.py','--config',str(ROOT/'examples/multimnist/configs/10epochs/multimnist/evomtl/rdust/custom-128_128_16_16_16/adam-lr_0.001/seed1.yaml')]
    params=benchmark.parse_args(LibMTL_args);params.wandb_enable=False;params.save_path=str(folder)
    benchmark._log_to_save_path(params);set_device(params.gpu_id);set_random_seed(params.seed)
    benchmark.main(params)

if __name__=='__main__':main()
