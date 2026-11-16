"""Measure clean activation scales on a fixed training batch at saved checkpoints."""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'examples/multimnist'))
import torch
import main as benchmark
from LibMTL.config import LibMTL_args
from LibMTL.evomtl.evo_trainer import EvoMTLTrainer
from LibMTL.utils import set_device,set_random_seed


def main():
    cli=argparse.ArgumentParser();cli.add_argument('folder',type=Path);opts=cli.parse_args()
    folder=opts.folder.resolve();torch.set_num_threads(1)
    def inspect(self,train_loader,test_loader,epochs):
        x,y=next(iter(train_loader));x=x.to(self.device)
        results=[]
        for path in sorted(folder.glob('checkpoint_*.pt'),key=lambda p:int(p.stem.split('_')[-1])):
            state=torch.load(path,map_location=self.device,weights_only=False)
            self.model.load_state_dict(state['model']);self.model.eval()
            sites={n:m for n,m in self.model.named_modules() if isinstance(m,(torch.nn.Conv2d,torch.nn.Linear))}
            row={'generation':state['generation'],'layers':{}}
            def hook(name):
                def collect(m,args,output):
                    row['layers'][name]={'input_rms':float(args[0].square().mean().sqrt()),
                        'output_rms':float(output.square().mean().sqrt()),
                        'weight_norm':float(m.weight.norm()),'bias_norm':float(m.bias.norm()),
                        'output_nonpositive_fraction':float((output<=0).float().mean())}
                return collect
            handles=[m.register_forward_hook(hook(n)) for n,m in sites.items()]
            with torch.no_grad():self.model(x)
            for h in handles:h.remove()
            results.append(row)
        (folder/'activation_scales.json').write_text(json.dumps(results,indent=2))
        print(json.dumps(results,indent=2))
    EvoMTLTrainer.train=inspect
    sys.argv=['main.py','--config',str(ROOT/'examples/multimnist/configs/10epochs/multimnist/evomtl/rdust/custom-128_128_16_16_16/adam-lr_0.001/seed1.yaml')]
    params=benchmark.parse_args(LibMTL_args);params.wandb_enable=False;params.save_path=None
    set_device(params.gpu_id);set_random_seed(params.seed);benchmark.main(params)

if __name__=='__main__':main()
