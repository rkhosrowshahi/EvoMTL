import argparse
from typing import Optional

import numpy as np
import torch

_parser = argparse.ArgumentParser(description='Configuration for LibMTL')
# general
_parser.add_argument('--mode', type=str, default='train', help='train, test')
_parser.add_argument('--seed', type=int, default=0, help='random seed')
_parser.add_argument('--gpu_id', default='0', type=str, help='gpu_id') 
_parser.add_argument('--weighting', type=str, default='EW',
    help='loss weighing strategies, option: EW, UW, GradNorm, GLS, RLW, \
        MGDA, PCGrad, GradVac, CAGrad, GradDrop, DWA, IMTL')
_parser.add_argument('--arch', type=str, default='HPS',
                    help='architecture for MTL, option: HPS, MTAN')
_parser.add_argument('--rep_grad', action='store_true', default=False, 
                    help='computing gradient for representation or sharing parameters')
_parser.add_argument('--multi_input', action='store_true', default=False, 
                    help='whether each task has its own input data')
_parser.add_argument('--save_path', type=str, default=None,
                    help='save path (checkpoints, best.pt; EvoMTL also writes evo_result.pt / model_evo.pt here)')
_parser.add_argument('--load_path', type=str, default=None, 
                    help='load ckpt path')
## Weights & Biases (passed to :class:`LibMTL.trainer.Trainer` as ``wandb_init``)
_parser.add_argument('--wandb_project', type=str, default=None,
                    help='wandb project name; logging is disabled when unset')
_parser.add_argument('--wandb_name', type=str, default=None,
                    help='wandb run name (optional)')
_parser.add_argument('--wandb_entity', type=str, default=None,
                    help='wandb entity (team or username; optional)')
_parser.add_argument('--wandb_group', type=str, default=None,
                    help='wandb group name for grouping runs (optional)')
## optim
_parser.add_argument('--optim', type=str, default='adam',
                    help='optimizer for training, option: adam, sgd, adagrad, rmsprop')
_parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for all types of optim')
_parser.add_argument('--momentum', type=float, default=0.9, help='momentum for sgd')
_parser.add_argument('--adam_beta1', type=float, default=0.9, help='Adam: beta1 (first moment decay)')
_parser.add_argument('--adam_beta2', type=float, default=0.999, help='Adam: beta2 (second moment decay)')
_parser.add_argument('--adam_eps', type=float, default=1e-8, help='Adam: epsilon for numerical stability')
_parser.add_argument('--amsgrad', action='store_true', default=False,
                    help='Adam: use AMSGrad variant')
_parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay for all types of optim')
## scheduler
_parser.add_argument('--scheduler', type=str, #default='step',
                    help='learning rate scheduler for training, option: step, cos, exp')
_parser.add_argument('--step_size', type=int, default=100, help='step size for StepLR')
_parser.add_argument('--gamma', type=float, default=0.5, help='gamma for StepLR')

# args for weighting
## DWA
_parser.add_argument('--T', type=float, default=2.0, help='T for DWA')
## MGDA
_parser.add_argument('--mgda_gn', default='none', type=str, 
                    help='type of gradient normalization for MGDA, option: l2, none, loss, loss+')
## GradVac
_parser.add_argument('--GradVac_beta', type=float, default=0.5, help='beta for GradVac')
_parser.add_argument('--GradVac_group_type', type=int, default=0, 
                    help='parameter granularity for GradVac (0: whole_model; 1: all_layer; 2: all_matrix)')
## GradNorm
_parser.add_argument('--alpha', type=float, default=1.5, help='alpha for GradNorm')
## GradDrop
_parser.add_argument('--leak', type=float, default=0.0, help='leak for GradDrop')
## CAGrad
_parser.add_argument('--calpha', type=float, default=0.5, help='calpha for CAGrad')
_parser.add_argument('--rescale', type=int, default=1, help='rescale for CAGrad')
## Nash_MTL
_parser.add_argument('--update_weights_every', type=int, default=1, help='update_weights_every for Nash_MTL')
_parser.add_argument('--optim_niter', type=int, default=20, help='optim_niter for Nash_MTL')
_parser.add_argument('--max_norm', type=float, default=1.0, help='max_norm for Nash_MTL')
## MoCo
_parser.add_argument('--MoCo_beta', type=float, default=0.5, help='MoCo_beta for MoCo')
_parser.add_argument('--MoCo_beta_sigma', type=float, default=0.5, help='MoCo_beta_sigma for MoCo')
_parser.add_argument('--MoCo_gamma', type=float, default=0.1, help='gamma for MoCo')
_parser.add_argument('--MoCo_gamma_sigma', type=float, default=0.5, help='MoCo_gamma_sigma for MoCo')
_parser.add_argument('--MoCo_rho', type=float, default=0, help='MoCo_rho for MoCo')
## DB_MTL
_parser.add_argument('--DB_beta', type=float, default=0.9, help=' ')
_parser.add_argument('--DB_beta_sigma', type=float, default=0, help=' ')
## STCH
_parser.add_argument('--STCH_mu', type=float, default=1.0, help=' ')
_parser.add_argument('--STCH_warmup_epoch', type=int, default=4, help=' ')
## ExcessMTL
_parser.add_argument('--robust_step_size', default=1e-2, type=float, help='step size')
## FairGrad
_parser.add_argument('--FairGrad_alpha', type=float, default=1.0, help=' ')
## FAMO
_parser.add_argument('--FAMO_w_lr', type=float, default=0.025, help=' ')
_parser.add_argument('--FAMO_w_gamma', type=float, default=1e-3, help=' ')
## MoDo
_parser.add_argument('--MoDo_gamma', type=float, default=1e-3, help=' ')
_parser.add_argument('--MoDo_rho', type=float, default=0.1, help=' ')
## SDMGrad
_parser.add_argument('--SDMGrad_lamda', type=float, default=0.3, help=' ')
_parser.add_argument('--SDMGrad_niter', type=int, default=20, help=' ')
## UPGrad
_parser.add_argument('--UPGrad_norm_eps', type=float, default=0.0001,
                     help='A small value to avoid division by zero when normalizing.')
_parser.add_argument('--UPGrad_reg_eps', type=float, default=0.0001,
                     help='A small value to add to the diagonal of the gramian to make it positive definite.')

#### bilevel methods
_parser.add_argument('--outer_lr', type=float, default=1e-3, help='outer lr')
_parser.add_argument('--inner_lr', type=float, default=0.1, help='inner lr')
_parser.add_argument('--inner_step', type=int, default=5, help=' ')
## FORUM
_parser.add_argument('--FORUM_phi', type=float, default=0.1, help=' ') # FORUM

# args for architecture
## CGC
_parser.add_argument('--img_size', nargs='+', help='image size for CGC')
_parser.add_argument('--num_experts', nargs='+', help='the number of experts for sharing and task-specific')
## DSelect_k
_parser.add_argument('--num_nonzeros', type=int, default=2, help='num_nonzeros for DSelect-k')
_parser.add_argument('--kgamma', type=float, default=1.0, help='gamma for DSelect-k')

# EvoMTL (MOEA on parameter sharing; use :class:`LibMTL.evomtl.evo_trainer.EvoMTLTrainer`)
_parser.add_argument('--evo_training', action='store_true', default=False,
                    help='after normal GD (epochs), run MOEA on parameter-sharing latent z')
_parser.add_argument('--evo_ps', type=str, default='spherical_lora',
                    help='random_proj, layerwise_random_proj, layerwise_scaled_random_proj, flatten_lora, spherical_lora, dict_lora, linear_lora, modulation_lora, spectral_lora, spectral_all_svd')
_parser.add_argument('--evo_moea', type=str, default='nsga2',
                    help='nsga2 or comocma (two tasks only for comocma)')
_parser.add_argument('--evo_ps_scale', type=float, default=1.0,
                    help='scale for theta = theta_0 + ps_scale * ps(z)')
_parser.add_argument('--evo_ps_r', type=int, default=4,
                    help='rank r for LoRA-style parameter sharing (ignored for random_proj and layerwise_random_proj)')
_parser.add_argument('--evo_ps_k', type=int, default=64,
                    help='random_proj (k+1 dims with scale), layerwise_random_proj (k per projected tensor), and layerwise_scaled_random_proj (k+1 per projected tensor)')
_parser.add_argument('--evo_ps_seed', type=int, default=42, help='seed for parameter sharing')
_parser.add_argument('--evo_iterations', type=int, default=30,
                    help='MOEA outer iterations: NSGA-II generations or COMO-CMA-ES iterations')
_parser.add_argument('--evo_pop_size', type=int, default=24, help='MOEA population size (NSGA-II; COMO-CMA popsize)')
_parser.add_argument('--evo_z_lower', type=float, default=-3.0, help='box lower for latent z (NSGA-II; COMO-CMA bounds)')
_parser.add_argument('--evo_z_upper', type=float, default=3.0, help='box upper for latent z (NSGA-II; COMO-CMA bounds)')
_parser.add_argument('--evo_n_eval_batches', type=int, default=1,
                    help='train batches per MOEA fitness evaluation')
_parser.add_argument('--evo_seed', type=int, default=0, help='MOEA random seed (NSGA-II; CMA seed for comocma)')
_parser.add_argument('--evo_num_kernels', type=int, default=10,
                    help='number of comocma kernels / Pareto points')
_parser.add_argument('--evo_sigma0', type=float, default=0.3, help='initial CMA step size')
_parser.add_argument('--evo_cma_tolx', type=float, default=None,
                    help='COMO-CMA CMA tolx (omit for comocma kernel default)')
_parser.add_argument('--evo_cma_tolfun', type=float, default=None,
                    help='COMO-CMA CMA tolfun')
_parser.add_argument('--evo_cma_tolfunrel', type=float, default=None,
                    help='COMO-CMA CMA tolfunrel')
_parser.add_argument('--evo_cma_tolfunhist', type=float, default=None,
                    help='COMO-CMA CMA tolfunhist')
_parser.add_argument('--evo_cma_tolstagnation', type=float, default=None,
                    help='COMO-CMA CMA tolstagnation (0 disables)')
_parser.add_argument('--evo_cma_tolflatfitness', type=float, default=None,
                    help='COMO-CMA CMA tolflatfitness')
_parser.add_argument(
    '--evo_sofomore_reference_point',
    nargs='*',
    type=float,
    default=None,
    help='Sofomore COMO-CMA reference point [f0, f1] (minimization). '
         'Omit or use empty list for auto: 2*probe_loss+1 per task.',
)
_parser.add_argument(
    '--evo_hv_ref_point',
    nargs='*',
    type=float,
    default=None,
    help='pymoo HV reference for marginal HV weights (two tasks). '
         'Omit or use empty list for auto: nadir of Pareto front + 1e-6.',
)
_parser.add_argument(
    '--evo_hv_center_aggregation',
    type=str,
    default='linear',
    choices=('linear', 'softmax'),
    help='Pareto center: linear = normalize marginal HV contributions; '
         'softmax = temperature-scaled softmax of contributions.',
)
_parser.add_argument(
    '--evo_hv_softmax_temperature',
    type=float,
    default=1.0,
    help='Temperature for evo_hv_center_aggregation=softmax (larger = softer weights).',
)

LibMTL_args = _parser


def prepare_args(params):
    r"""Return the configuration of hyperparameters, optimizier, and learning rate scheduler.

    The returned ``kwargs`` includes ``evo_args`` (from ``--evo_*`` flags) for
    :class:`LibMTL.evomtl.evo_trainer.EvoMTLTrainer` when ``--evo_training`` is set,
    and ``wandb_init`` (from ``--wandb_project`` / ``--wandb_name`` / ``--wandb_entity``) for
    :class:`LibMTL.trainer.Trainer` when ``--wandb_project`` is set.  ``wandb_init`` includes a
    ``config`` dict (LibMTL argparse fields plus ``weight_args``, ``arch_args``, ``optim_param``,
    ``scheduler_param``, ``evo_args``) passed to :func:`wandb.init`.

    Args:
        params (argparse.Namespace): The command-line arguments.
    """
    kwargs = {'weight_args': {}, 'arch_args': {}}
    if params.weighting in ['EW', 'UW', 'GradNorm', 'GLS', 'RLW', 'MGDA', 'IMTL',
                            'PCGrad', 'GradVac', 'CAGrad', 'GradDrop', 'DWA', 
                            'Nash_MTL', 'MoCo', 'Aligned_MTL', 'DB_MTL', 'STCH', 
                            'ExcessMTL', 'FairGrad', 'FAMO', 'MoDo', 'SDMGrad', 'UPGrad']:
        if params.weighting in ['DWA']:
            if params.T is not None:
                kwargs['weight_args']['T'] = params.T
            else:
                raise ValueError('DWA needs keywaord T')
        elif params.weighting in ['GradNorm']:
            if params.alpha is not None:
                kwargs['weight_args']['alpha'] = params.alpha
            else:
                raise ValueError('GradNorm needs keywaord alpha')
        elif params.weighting in ['MGDA']:
            if params.mgda_gn is not None:
                if params.mgda_gn in ['none', 'l2', 'loss', 'loss+']:
                    kwargs['weight_args']['mgda_gn'] = params.mgda_gn
                else:
                    raise ValueError('No support mgda_gn {} for MGDA'.format(params.mgda_gn)) 
            else:
                raise ValueError('MGDA needs keywaord mgda_gn')
        elif params.weighting in ['GradVac']:
            if params.GradVac_beta is not None:
                kwargs['weight_args']['GradVac_beta'] = params.GradVac_beta
                kwargs['weight_args']['GradVac_group_type'] = params.GradVac_group_type
            else:
                raise ValueError('GradVac needs keywaord beta')
        elif params.weighting in ['GradDrop']:
            if params.leak is not None:
                kwargs['weight_args']['leak'] = params.leak
            else:
                raise ValueError('GradDrop needs keywaord leak')
        elif params.weighting in ['CAGrad']:
            if params.calpha is not None and params.rescale is not None:
                kwargs['weight_args']['calpha'] = params.calpha
                kwargs['weight_args']['rescale'] = params.rescale
            else:
                raise ValueError('CAGrad needs keywaord calpha and rescale')
        elif params.weighting in ['Nash_MTL']:
            if params.update_weights_every is not None and params.optim_niter is not None and params.max_norm is not None:
                kwargs['weight_args']['update_weights_every'] = params.update_weights_every
                kwargs['weight_args']['optim_niter'] = params.optim_niter
                kwargs['weight_args']['max_norm'] = params.max_norm
            else:
                raise ValueError('Nash_MTL needs update_weights_every, optim_niter, and max_norm')
        elif params.weighting in ['MoCo']:
            kwargs['weight_args']['MoCo_beta'] = params.MoCo_beta
            kwargs['weight_args']['MoCo_beta_sigma'] = params.MoCo_beta_sigma
            kwargs['weight_args']['MoCo_gamma'] = params.MoCo_gamma
            kwargs['weight_args']['MoCo_gamma_sigma'] = params.MoCo_gamma_sigma
            kwargs['weight_args']['MoCo_rho'] = params.MoCo_rho
        elif params.weighting in ['DB_MTL']:
            kwargs['weight_args']['DB_beta'] = params.DB_beta
            kwargs['weight_args']['DB_beta_sigma'] = params.DB_beta_sigma
        elif params.weighting in ['STCH']:
            kwargs['weight_args']['STCH_mu'] = params.STCH_mu
            kwargs['weight_args']['STCH_warmup_epoch'] = params.STCH_warmup_epoch
        elif params.weighting in ['ExcessMTL']:
            kwargs['weight_args']['robust_step_size'] = params.robust_step_size
        elif params.weighting in ['FairGrad']:
            kwargs['weight_args']['FairGrad_alpha'] = params.FairGrad_alpha
        elif params.weighting in ['FAMO']:
            kwargs['weight_args']['FAMO_w_lr'] = params.FAMO_w_lr
            kwargs['weight_args']['FAMO_w_gamma'] = params.FAMO_w_gamma
        elif params.weighting in ['MoDo']:
            kwargs['weight_args']['MoDo_gamma'] = params.MoDo_gamma
            kwargs['weight_args']['MoDo_rho'] = params.MoDo_rho
        elif params.weighting in ['SDMGrad']:
            kwargs['weight_args']['SDMGrad_lamda'] = params.SDMGrad_lamda
            kwargs['weight_args']['SDMGrad_niter'] = params.SDMGrad_niter
        elif params.weighting in ['UPGrad']:
            kwargs['weight_args']['UPGrad_norm_eps'] = params.UPGrad_norm_eps
            kwargs['weight_args']['UPGrad_reg_eps'] = params.UPGrad_reg_eps
    elif params.weighting in ['MOML', 'FORUM', 'AutoLambda']:
        kwargs['weight_args']['outer_lr'] = params.outer_lr
        kwargs['weight_args']['inner_step'] = params.inner_step
        if params.weighting in ['FORUM']:
            kwargs['weight_args']['FORUM_phi'] = params.FORUM_phi
            kwargs['weight_args']['inner_lr'] = params.inner_lr
        elif params.weighting in ['MOML']:
            kwargs['weight_args']['inner_lr'] = params.inner_lr
    else:
        raise ValueError('No support weighting method {}'.format(params.weighting)) 
        
    if params.arch in ['HPS', 'Cross_stitch', 'MTAN', 'CGC', 'PLE', 'MMoE', 'DSelect_k', 'DIY', 'LTB']:
        if params.arch in ['CGC', 'PLE', 'MMoE', 'DSelect_k']:
            kwargs['arch_args']['img_size'] = tuple(params.img_size)#np.array(params.img_size, dtype=int).prod()
            kwargs['arch_args']['num_experts'] = [int(num) for num in params.num_experts]
        if params.arch in ['DSelect_k']:
            kwargs['arch_args']['kgamma'] = params.kgamma
            kwargs['arch_args']['num_nonzeros'] = params.num_nonzeros
    else:
        raise ValueError('No support architecture method {}'.format(params.arch)) 
        
    if params.optim in ['adam', 'sgd', 'adagrad', 'rmsprop']:
        if params.optim == 'adam':
            optim_param = {
                'optim': 'adam',
                'lr': params.lr,
                'weight_decay': params.weight_decay,
                'betas': (params.adam_beta1, params.adam_beta2),
                'eps': params.adam_eps,
                'amsgrad': params.amsgrad,
            }
        elif params.optim == 'sgd':
            optim_param = {'optim': 'sgd', 'lr': params.lr, 
                           'weight_decay': params.weight_decay, 'momentum': params.momentum}
    else:
        raise ValueError('No support optim method {}'.format(params.optim))
        
    if params.scheduler is not None:
        if params.scheduler in ['step', 'cos', 'exp']:
            if params.scheduler == 'step':
                scheduler_param = {'scheduler': 'step', 'step_size': params.step_size, 'gamma': params.gamma}
        else:
            raise ValueError('No support scheduler method {}'.format(params.scheduler))
    else:
        scheduler_param = None
    
    kwargs['evo_args'] = _make_evo_args(params)
    kwargs['wandb_init'] = _make_wandb_init(params, kwargs, optim_param, scheduler_param)

    _display(params, kwargs, optim_param, scheduler_param)
    
    return kwargs, optim_param, scheduler_param


def _wandb_sanitize(x):
    """Convert values to forms safe for ``wandb.init(config=...)`` (JSON-serializable)."""
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, (tuple, list)):
        return [_wandb_sanitize(i) for i in x]
    if isinstance(x, dict):
        return {str(k): _wandb_sanitize(v) for k, v in x.items()}
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    try:
        return float(x)
    except (TypeError, ValueError):
        return str(x)


def _build_wandb_config(params, kwargs, optim_param, scheduler_param):
    """Hyperparameters snapshot for Weights & Biases (matches :func:`prepare_args` outputs)."""
    libmtl_args = {k: _wandb_sanitize(v) for k, v in sorted(vars(params).items())}
    return {
        'libmtl_args': libmtl_args,
        'weight_args': _wandb_sanitize(kwargs.get('weight_args', {})),
        'arch_args': _wandb_sanitize(kwargs.get('arch_args', {})),
        'optim_param': _wandb_sanitize(optim_param),
        'scheduler_param': _wandb_sanitize(
            scheduler_param if scheduler_param is not None else {}
        ),
        'evo_args': _wandb_sanitize(kwargs.get('evo_args', {})),
    }


def _make_wandb_init(params, kwargs, optim_param, scheduler_param):
    """Build ``wandb_init`` for :meth:`LibMTL.trainer.Trainer` from CLI args."""
    project = getattr(params, 'wandb_project', None)
    if not project:
        return None
    out = {
        'project': project,
        'config': _build_wandb_config(params, kwargs, optim_param, scheduler_param),
    }
    name = getattr(params, 'wandb_name', None)
    if name:
        out['name'] = name
    entity = getattr(params, 'wandb_entity', None)
    if entity:
        out['entity'] = entity
    group = getattr(params, 'wandb_group', None)
    if group:
        out['group'] = group
    return out


def _make_comocma_cma_opts(params):
    """Options passed to ``comocma.get_cmas(..., inopts=...)`` (merged with comocma defaults)."""
    o = {
        'popsize': int(params.evo_pop_size),
        'seed': int(params.evo_seed),
    }
    if getattr(params, 'evo_cma_tolx', None) is not None:
        o['tolx'] = float(params.evo_cma_tolx)
    if getattr(params, 'evo_cma_tolfun', None) is not None:
        o['tolfun'] = float(params.evo_cma_tolfun)
    if getattr(params, 'evo_cma_tolfunrel', None) is not None:
        o['tolfunrel'] = float(params.evo_cma_tolfunrel)
    if getattr(params, 'evo_cma_tolfunhist', None) is not None:
        o['tolfunhist'] = float(params.evo_cma_tolfunhist)
    if getattr(params, 'evo_cma_tolstagnation', None) is not None:
        o['tolstagnation'] = float(params.evo_cma_tolstagnation)
    if getattr(params, 'evo_cma_tolflatfitness', None) is not None:
        o['tolflatfitness'] = float(params.evo_cma_tolflatfitness)
    return o


def _evo_two_floats_or_none(seq) -> Optional[np.ndarray]:
    """Parse ``--evo_*`` with ``nargs='*'``: None/empty -> None; two floats -> array."""
    if seq is None:
        return None
    if len(seq) == 0:
        return None
    if len(seq) != 2:
        raise ValueError(
            'Expected exactly two objectives for reference point, got {!r}'.format(seq)
        )
    return np.asarray([float(seq[0]), float(seq[1])], dtype=np.float64)


def _make_evo_args(params):
    """Build ``evo_args`` for :meth:`LibMTL.trainer.Trainer.train` (no ``**kwargs``)."""
    if not getattr(params, 'evo_training', False):
        return {'evo_training': False}

    moea = params.evo_moea.strip().lower()
    ps_key = params.evo_ps.strip().lower().replace('-', '_')

    ps_kwargs = {'seed': params.evo_ps_seed}
    if ps_key in ('random_proj', 'layerwise_random_proj'):
        ps_kwargs['k'] = params.evo_ps_k
    else:
        ps_kwargs['r'] = params.evo_ps_r

    hv_agg = getattr(params, 'evo_hv_center_aggregation', 'linear')
    hv_temp = float(getattr(params, 'evo_hv_softmax_temperature', 1.0))
    hv_common = {
        'hv_center_aggregation': str(hv_agg).strip().lower(),
        'hv_softmax_temperature': hv_temp,
    }

    if moea == 'nsga2':
        evo_kwargs = {
            'n_gen': params.evo_iterations,
            'pop_size': params.evo_pop_size,
            'z_lower': params.evo_z_lower,
            'z_upper': params.evo_z_upper,
            'n_eval_batches': params.evo_n_eval_batches,
            'seed': params.evo_seed,
            **hv_common,
        }
        hv = _evo_two_floats_or_none(getattr(params, 'evo_hv_ref_point', None))
        if hv is not None:
            evo_kwargs['hv_ref_point'] = hv
    elif moea in ('comocma', 'mocma', 'mo-cma-es'):
        evo_kwargs = {
            'n_iterations': params.evo_iterations,
            'num_kernels': params.evo_num_kernels,
            'sigma0': params.evo_sigma0,
            'n_eval_batches': params.evo_n_eval_batches,
            'z_lower': params.evo_z_lower,
            'z_upper': params.evo_z_upper,
            'seed': params.evo_seed,
            'cma_opts': _make_comocma_cma_opts(params),
            **hv_common,
        }
        rp = _evo_two_floats_or_none(getattr(params, 'evo_sofomore_reference_point', None))
        if rp is not None:
            evo_kwargs['reference_point'] = [float(rp[0]), float(rp[1])]
        hv = _evo_two_floats_or_none(getattr(params, 'evo_hv_ref_point', None))
        if hv is not None:
            evo_kwargs['hv_ref_point'] = hv
    else:
        raise ValueError('Unsupported evo_moea {}'.format(params.evo_moea))

    return {
        'evo_training': True,
        'evo_ps': params.evo_ps,
        'moea': params.evo_moea,
        'ps_scale': params.evo_ps_scale,
        'ps_kwargs': ps_kwargs,
        'evo_kwargs': evo_kwargs,
    }


def _display(params, kwargs, optim_param, scheduler_param):
    print('='*40)
    print('General Configuration:')
    print('\tMode:', params.mode)
    print('\tWighting:', params.weighting)
    print('\tArchitecture:', params.arch)
    print('\tRep_Grad:', params.rep_grad)
    print('\tMulti_Input:', params.multi_input)
    print('\tSeed:', params.seed)
    print('\tSave Path:', params.save_path)
    print('\tLoad Path:', params.load_path)
    print('\tDevice: {}'.format('cuda:'+params.gpu_id if torch.cuda.is_available() else 'cpu'))
    wi = kwargs.get('wandb_init')
    if wi is not None:
        print('\tW&B project:', wi.get('project'))
        if wi.get('entity'):
            print('\tW&B entity:', wi['entity'])
        if wi.get('name'):
            print('\tW&B run name:', wi['name'])
        if wi.get('group'):
            print('\tW&B group:', wi['group'])
    for wa, p in zip(['weight_args', 'arch_args'], [params.weighting, params.arch]):
        if kwargs[wa] != {}:
            print('{} Configuration:'.format(p))
            for k, v in kwargs[wa].items():
                print('\t'+k+':', v)
    print('Optimizer Configuration:')
    for k, v in optim_param.items():
        print('\t'+k+':', v)
    if scheduler_param is not None:
        print('Scheduler Configuration:')
        for k, v in scheduler_param.items():
            print('\t'+k+':', v)
    evo = kwargs.get('evo_args') or {}
    if evo.get('evo_training'):
        print('EvoMTL Configuration:')
        for k in ('evo_ps', 'moea', 'ps_scale'):
            if k in evo:
                print('\t'+k+':', evo[k])
        print('\t(save_path for GD + EvoMTL):', params.save_path)
        if evo.get('ps_kwargs'):
            print('\tps_kwargs:', evo['ps_kwargs'])
        if evo.get('evo_kwargs'):
            print('\tevo_kwargs:', evo['evo_kwargs'])