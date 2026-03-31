import random, torch, os
import numpy as np
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple

# Added to ``min_baseline`` for pymoo HV reference (minimization objectives).
HV_MIN_BASELINE_EPS = 1e-6


def wandb_metrics_hypervolume(
    meter: Any,
    task_dict: Dict[str, Any],
    task_names: list,
    min_baseline: Optional[Dict[Tuple[str, int], float]] = None,
) -> float:
    """Hypervolume (minimization) from task metrics via :class:`pymoo.indicators.hv.Hypervolume`.

    ``meter`` should expose ``results[task][i]`` like :class:`LibMTL._record._PerformanceMeter`.

    Uses ``weight`` per metric: higher-is-better maps to ``f = 1 - value`` (clip
    to [0, 1]) with ``ref_i = 1``; lower-is-better maps to ``f = value`` with
    ``ref_i = min_baseline[(task, i)] + eps``; if no baseline was recorded, uses
    a heuristic ref. If ``pymoo`` is missing, falls back to ``prod(ref_point - f)``.
    """
    fs = []
    refs = []
    for task in task_names:
        weights = task_dict[task]["weight"]
        for i, _m in enumerate(task_dict[task]["metrics"]):
            val = float(meter.results[task][i])
            w = float(weights[i])
            if w >= 0.5:
                v = float(np.clip(val, 0.0, 1.0))
                fs.append(1.0 - v)
                refs.append(1.0)
            else:
                f = max(val, 0.0)
                fs.append(f)
                base = None
                if min_baseline is not None:
                    base = min_baseline.get((task, i))
                if base is not None:
                    b = float(base)
                    refs.append(b + HV_MIN_BASELINE_EPS)
                else:
                    refs.append(max(f + 1e-6, 1.0) + 1.0)
    F = np.atleast_2d(np.asarray(fs, dtype=np.float64))
    ref_point = np.asarray(refs, dtype=np.float64)
    if F.size == 0:
        return 0.0
    if np.any(F.ravel() > ref_point):
        return 0.0
    try:
        from pymoo.indicators.hv import Hypervolume

        return float(Hypervolume(ref_point=ref_point).do(F))
    except ImportError:  # pragma: no cover
        return float(np.prod(ref_point - F.ravel()))

def get_root_dir():
    r"""Return the root path of project."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def set_random_seed(seed):
    r"""Set the random seed for reproducibility.

    Args:
        seed (int, default=0): The random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
def set_device(gpu_id):
    r"""Set the device where model and data will be allocated. 

    Args:
        gpu_id (str, default='0'): The id of gpu.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

def count_parameters(model):
    r'''Calculate the number of parameters for a model.

    Args:
        model (torch.nn.Module): A neural network module.
    '''
    trainable_params = 0
    non_trainable_params = 0
    for p in model.parameters():
        if p.requires_grad:
            trainable_params += p.numel()
        else:
            non_trainable_params += p.numel()
    print('='*40)
    print('Total Params:', trainable_params + non_trainable_params)
    print('Trainable Params:', trainable_params)
    print('Non-trainable Params:', non_trainable_params)
        
def count_improvement(base_result, new_result, weight):
    r"""Calculate the improvement between two results as

    .. math::
        \Delta_{\mathrm{p}}=100\%\times \frac{1}{T}\sum_{t=1}^T 
        \frac{1}{M_t}\sum_{m=1}^{M_t}\frac{(-1)^{w_{t,m}}(B_{t,m}-N_{t,m})}{B_{t,m}}.

    Args:
        base_result (dict): A dictionary of scores of all metrics of all tasks.
        new_result (dict): The same structure with ``base_result``.
        weight (dict): The same structure with ``base_result`` while each element is binary integer representing whether higher or lower score is better.

    Returns:
        float: The improvement between ``new_result`` and ``base_result``.

    Examples::

        base_result = {'A': [96, 98], 'B': [0.2]}
        new_result = {'A': [93, 99], 'B': [0.5]}
        weight = {'A': [1, 0], 'B': [1]}

        print(count_improvement(base_result, new_result, weight))
    """
    improvement = 0
    count = 0
    for task in list(base_result.keys()):
        improvement += (((-1)**np.array(weight[task]))*\
                        (np.array(base_result[task])-np.array(new_result[task]))/\
                         np.array(base_result[task])).mean()
        count += 1
    return improvement/count

def set_param(curr_mod, name, param=None, mode='update'):
    if '.' in name:
        n = name.split('.')
        module_name = n[0]
        rest = '.'.join(n[1:])
        for name, mod in curr_mod.named_children():
            if module_name == name:
                return set_param(mod, rest, param, mode=mode)
    else:
        if mode == 'update':
            delattr(curr_mod, name)
            setattr(curr_mod, name, param)
        elif mode == 'get':
            if hasattr(curr_mod, name):
                p = getattr(curr_mod, name)
                return p