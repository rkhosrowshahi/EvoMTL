"""Summarize per-block diagnostics without changing or loading training state."""
import argparse
import csv
import json
import statistics
from pathlib import Path


def summarize(folder):
    metrics = list(csv.DictReader((folder/'metrics.csv').open()))
    blocks = list(csv.DictReader((folder/'blocks.csv').open()))
    summary = {'iterations': len(metrics), 'test': [
        {k: float(v) for k,v in r.items() if k=='generation' or k.startswith('test_')}
        for r in metrics if r.get('test_loss_top-left')], 'windows': {}}
    for low, high in ((1,50),(51,100),(101,150),(151,200),(201,250),(251,300),(301,350),(351,400),(401,450)):
        window = {}
        for name in sorted({r['block'] for r in blocks}):
            rs = [r for r in blocks if r['block']==name and low<=int(r['generation'])<=high]
            if not rs:
                continue
            def values(key): return [float(r[key]) for r in rs]
            harmful = [r for r in rs if float(r['loss_change'])>1e-6]
            window[name] = {
                'n':len(rs), 'harmful':len(harmful),
                'negative_estimate_cos':sum(v<0 for v in values('estimate_cos')),
                'uphill_adam_direction':sum(v<0 for v in values('descent_cos')),
                'median_estimate_cos':statistics.median(values('estimate_cos')),
                'median_descent_cos':statistics.median(values('descent_cos')),
                'median_estimated_norm':statistics.median(values('estimated_norm')),
                'median_true_norm':statistics.median(values('true_norm')),
                'median_update_norm':statistics.median(values('update_norm')),
                'mean_loss_change':statistics.mean(values('loss_change')),
                'max_loss_increase':max(values('loss_change')),
                'harmful_fixed_by_half_step':sum(float(r['half_adam_loss_change'])<0 for r in harmful),
                'harmful_fixed_by_no_momentum':sum(float(r['no_momentum_same_v_loss_change'])<0 for r in harmful),
                'harmful_fixed_by_exact_same_norm':sum(float(r['exact_same_norm_loss_change'])<0 for r in harmful),
            }
        if window:
            summary['windows'][f'{low}-{high}'] = window
    summary['largest_increases'] = sorted(blocks,key=lambda r:float(r['loss_change']),reverse=True)[:10]
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    args=parser.parse_args()
    print(json.dumps(summarize(args.folder),indent=2))
