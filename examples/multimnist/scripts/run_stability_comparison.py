"""Full matched DUST/RDUST stability experiments, with W&B and restart checkpoints.

Backtracking accepts non-increasing summed training-batch loss; on complete rejection
it restores both parameters and Adam state. Exact gradients are diagnostic only.
"""
import argparse
import copy
import csv
import hashlib
import json
import random
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'examples/multimnist'))
import numpy as np
import torch
import wandb
import main as benchmark
from LibMTL.config import LibMTL_args
from LibMTL.trainer import Trainer
from LibMTL.utils import set_device, set_random_seed
from LibMTL.evomtl.evo_trainer import EvoMTLTrainer
from LibMTL.evomtl.moea.rdust import RDUST
from LibMTL.evomtl.moea.dust import DUST
from LibMTL.evomtl.moea.es_common import make_es_optimizer, MetricsCSV, metrics_columns, report_search_mean
from investigate_rdust import flat, cosine


class Rows:
    def __init__(self, path):
        self.file = path.open('w', buffering=1)
        self.writer = None
    def write(self, row):
        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=list(row)); self.writer.writeheader()
        self.writer.writerow(row)
    def close(self): self.file.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['dust','rdust_half','rdust_backtrack'], required=True)
    parser.add_argument('--iterations', type=int, default=2344)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--diagnostic-every', type=int, default=25)
    parser.add_argument('--group', default='rdust-stability-full-seed1-20261116')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    folder = args.output.resolve(); folder.mkdir(parents=True, exist_ok=False)
    snapshots=folder/'source'; snapshots.mkdir()
    for source in [Path(__file__), ROOT/'LibMTL/evomtl/moea/rdust.py', ROOT/'LibMTL/evomtl/moea/dust.py', ROOT/'LibMTL/evomtl/moea/common.py']:
        shutil.copy2(source,snapshots/source.name)
    torch.set_num_threads(1)
    if args.offline:
        import os
        os.environ['WANDB_MODE'] = 'offline'
    lr = .0005 if args.method == 'rdust_half' else .001
    meta = {**vars(args), 'output':str(folder), 'lr':lr, 'sigma':.05, 'draws':[128,128,16,16,16],
            'batch_size':256, 'antithetic':False, 'status':'starting'}
    def status(**updates):
        meta.update(updates)
        temporary = folder/'status.tmp'; temporary.write_text(json.dumps(meta, indent=2)); temporary.replace(folder/'status.json')
    status()
    # W&B initialization must not change the matched batch sampling stream.
    setup = Trainer._setup_wandb
    def setup_preserving_rng(self, wi):
        state = torch.get_rng_state(), np.random.get_state(), random.getstate()
        try: setup(self,wi)
        finally:
            torch.set_rng_state(state[0]); np.random.set_state(state[1]); random.setstate(state[2])
    Trainer._setup_wandb = setup_preserving_rng

    def train(self, train_loader, test_loader, epochs):
        wandb.config.update({'experiment_method':args.method,'actual_es_lr':lr,'diagnostic_every':args.diagnostic_every,
            'backtracking_min_scale':1/32, 'backtracking_loss':'sum of training batch task losses',
            'adam_state_policy':'retain on accepted scaled steps; restore on rejection'}, allow_val_change=True)
        status(status='running',wandb_url=wandb.run.url,wandb_id=wandb.run.id,completed_iterations=0)
        self.prepare_evolution()
        initial_hash = hashlib.sha256(self.theta_0.cpu().numpy().tobytes()).hexdigest()
        wandb.config.update({'initial_weights_sha256':initial_hash}, allow_val_change=True)
        status(initial_weights_sha256=initial_hash)
        theta = self.theta_0.detach().cpu().clone()
        self.set_theta(theta)
        is_dust = args.method == 'dust'
        if is_dust: theta.requires_grad_(True)
        optimizer = make_es_optimizer([theta] if is_dust else list(self.model.parameters()), 'adam', lr)
        gen = torch.Generator(self.device).manual_seed(args.seed)
        estimator = (DUST if is_dust else RDUST)(self,24,.05,8,'custom',8,.03,gen,draws=[128,128,16,16,16])
        blocks = Rows(folder/'blocks.csv'); acceptance = Rows(folder/'acceptance.csv'); compute = Rows(folder/'compute.csv')
        metrics = MetricsCSV(self, metrics_columns(list(self.task_name), ['sigma'], ['gradient_cos','descent_cos']))
        phase = 'reporting'; counts = {k:0 for k in ['optimization','diagnostics','reporting']}; diagnostic_s = 0.
        def count_forward(module, inputs): counts[phase] += len(inputs[0])
        hook = self.model.register_forward_pre_hook(count_forward)
        @contextmanager
        def diagnostic():
            nonlocal phase, diagnostic_s
            previous=phase; phase='diagnostics'; t=time.perf_counter()
            try: yield
            finally: diagnostic_s += time.perf_counter()-t; phase=previous
        def objective():
            return sum(estimator.losses(x,y).mean(1).sum() for x,y in batches)/len(batches)
        offsets={}; offset=0
        for p in self.model.parameters(): offsets[p]=slice(offset,offset+p.numel()); offset+=p.numel()
        original_step=optimizer.step
        @torch.no_grad()
        def observed_step(*a, **kw):
            active=[p for p in self.model.parameters() if p.grad is not None]
            name=next(n for n,ps in estimator.groups.items() if {id(p) for p in ps}=={id(p) for p in active})
            guarded=args.method=='rdust_backtrack'
            if not guarded and not diagnose: return original_step(*a,**kw)
            before=[p.detach().clone() for p in active]
            estimate=flat([p.grad for p in active])
            if guarded: pre=float(objective())
            if diagnose:
                with diagnostic(), torch.enable_grad():
                    pre_tensor=objective(); exact=flat(torch.autograd.grad(pre_tensor,active)); pre=float(pre_tensor.detach())
            states=[copy.deepcopy(optimizer.state.get(p)) for p in active] if guarded else None
            result=original_step(*a,**kw)
            proposal=[p.detach().clone() for p in active]
            if guarded:
                proposed=float(objective()); post=proposed; scale=1.; evaluations=1
                while (not np.isfinite(post) or post>pre) and scale>1/32:
                    scale/=2
                    for p,b,v in zip(active,before,proposal): p.copy_(b+scale*(v-b))
                    post=float(objective()); evaluations+=1
                if not np.isfinite(post) or post>pre:
                    scale=0.; post=pre
                    for p,b,s in zip(active,before,states):
                        p.copy_(b)
                        if s is None: optimizer.state.pop(p,None)
                        else: optimizer.state[p]=s
                acceptance.write(dict(generation=i+1,block=name,pre_loss=pre,proposal_loss=proposed,accepted_loss=post,scale=scale,evaluations=evaluations))
                logs.update({f'backtrack/{name}/scale':scale,f'backtrack/{name}/proposal_loss_change':proposed-pre,
                             f'backtrack/{name}/accepted_loss_change':post-pre})
                scales.append(scale)
            else:
                with diagnostic(): post=float(objective())
            if diagnose:
                delta=flat([p-b for p,b in zip(active,before)])
                row=dict(generation=i+1,block=name,estimate_cos=cosine(estimate,exact),descent_cos=cosine(-delta,exact),
                         estimated_norm=float(estimate.norm()),true_norm=float(exact.norm()),update_norm=float(delta.norm()),loss_change=post-pre)
                blocks.write(row); staged.append((estimate,exact,delta))
                logs.update({f'diagnostics/{name}/{k}':v for k,v in row.items() if k not in ['generation','block']})
            return result
        if not is_dust: optimizer.step=observed_step
        wall_start=time.perf_counter(); optimization_s=0.
        try:
            for i in range(args.iterations):
                phase='optimization'; before_diag=diagnostic_s; started=time.perf_counter()
                batches=self._sample_batches(train_loader,1); logs={}; staged=[]; scales=[]
                diagnose=i==0 or (i+1)%args.diagnostic_every==0
                self.set_theta(theta.detach())
                if is_dust:
                    before=theta.detach().clone()
                    # Avoid mixing diagnostic forwards into the optimization count.
                    J,losses,direction,_,_=estimator.step(batches,theta,optimizer,diagnose=False)
                    if diagnose:
                        after=theta.detach().clone()
                        with diagnostic(), torch.enable_grad():
                            self.set_theta(before)
                            exact=flat(torch.autograd.grad(objective(),list(self.model.parameters())))
                        self.set_theta(theta.detach())
                        delta=after-before; staged=[(direction.detach(),exact,delta)]
                        for name,module in estimator.layers.items():
                            ps=list(module.parameters(recurse=False))
                            g=torch.cat([direction[offsets[p]] for p in ps]); t=torch.cat([exact[offsets[p]] for p in ps])
                            d=torch.cat([delta[offsets[p]] for p in ps])
                            row=dict(generation=i+1,block=name,estimate_cos=cosine(g,t),descent_cos=cosine(-d,t),estimated_norm=float(g.norm()),true_norm=float(t.norm()),update_norm=float(d.norm()),loss_change='')
                            blocks.write(row); logs.update({f'diagnostics/{name}/{k}':v for k,v in row.items() if k not in ['generation','block','loss_change']})
                else:
                    J,losses,_=estimator.step(batches,optimizer)
                    theta=torch.nn.utils.parameters_to_vector(self.model.parameters()).detach().cpu().clone()
                elapsed=time.perf_counter()-started; optimization_s+=elapsed-(diagnostic_s-before_diag)
                row={}
                if staged:
                    g,t,d=[torch.cat([r[k] for r in staged]) for k in range(3)]
                    row={'gradient_cos':cosine(g,t),'descent_cos':cosine(-d,t)}
                    logs.update({f'diagnostics/{k}':v for k,v in row.items()})
                if scales: logs.update({'backtrack/shrunk_blocks':sum(0<s<1 for s in scales),'backtrack/rejected_blocks':scales.count(0.),'backtrack/mean_scale':sum(scales)/len(scales)})
                phase='reporting'; F=losses.numpy()[None].astype(np.float32)
                tm,te=report_search_mean(self,theta,i,args.iterations,F,F,batches,test_loader,50,i)
                metrics.write(i,elapsed,list(self.task_name),F,tm,te,sigma=.05,**row)
                cost=dict(generation=i+1,optimization_s=optimization_s,diagnostic_s=diagnostic_s,wall_s=time.perf_counter()-wall_start,
                          **{k+'_forward_examples':v for k,v in counts.items()})
                compute.write(cost); logs.update({f'compute/{k}':v for k,v in cost.items() if k!='generation'})
                logs['comparison/generation']=i+1; self._wandb_log(logs,step=i)
                status(completed_iterations=i+1,**cost)
                if (i+1)%50==0 or i+1==args.iterations:
                    checkpoint=dict(model=self.model.state_dict(),theta=theta.detach(),optimizer=optimizer.state_dict(),noise_rng=gen.get_state(),torch_rng=torch.get_rng_state(),generation=i+1,method=args.method)
                    torch.save(checkpoint,folder/'checkpoint.tmp'); (folder/'checkpoint.tmp').replace(folder/'checkpoint_latest.pt')
                if (i+1)%50==0: print(f'Completed {i+1}/{args.iterations}; optimization seconds={optimization_s:.1f}',flush=True)
            torch.save(self.model.state_dict(),folder/'model_evo.pt')
            status(status='completed',finite_checkpoint=all(bool(torch.isfinite(p).all()) for p in self.model.parameters()))
            wandb.save(str(folder/'*.csv'),base_path=str(folder),policy='now')
        except BaseException as error:
            status(status='failed',error=repr(error)); raise
        finally:
            hook.remove(); estimator.close()
            for f in [blocks,acceptance,compute,metrics]: f.close()
            self._wandb_finish_if(True)
    EvoMTLTrainer.train=train
    sys.argv=['main.py','--config',str(ROOT/'examples/multimnist/configs/10epochs/multimnist/evomtl/rdust/custom-128_128_16_16_16/adam-lr_0.001/seed1.yaml')]
    params=benchmark.parse_args(LibMTL_args)
    params.seed=args.seed; params.evo_seed=args.seed; params.evo_iterations=args.iterations
    params.evo_moea='dust' if args.method=='dust' else 'rdust'; params.evo_es_lr=lr
    params.wandb_enable=True; params.wandb_entity='rasa_research'; params.wandb_project='evomtl-multimnist'
    params.wandb_group=args.group; params.wandb_name=f'{args.method}-full-seed{args.seed}'
    params.save_path=str(folder); benchmark._log_to_save_path(params)
    set_device(params.gpu_id); set_random_seed(params.seed); benchmark.main(params)


if __name__=='__main__':main()
