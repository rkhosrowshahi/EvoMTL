import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import random

from LibMTL import Trainer
from LibMTL.evomtl.evo_trainer import EvoMTLTrainer
from LibMTL.loss import CELoss
from LibMTL.metrics import AccMetric
from LibMTL.model import resnet
from LibMTL.utils import set_random_seed, set_device
from LibMTL.config import LibMTL_args, prepare_args
from mnist import MultiMNISTDataset, MultiMNISTDataset3Digits


def _load_yaml_defaults(parser):
    """If ``--config path/to/config.yaml`` is present, load the YAML and apply
    its key-value pairs as argparse defaults.  Any explicit CLI argument still
    takes precedence over the YAML value."""
    import yaml

    # First pass: grab --config only (ignore everything else).
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', default=None)
    known, _ = pre.parse_known_args()

    if known.config is not None:
        with open(known.config, 'r') as f:
            cfg = yaml.safe_load(f)
        if cfg:
            # Strip comment-only keys and coerce types argparse expects.
            clean = {k: v for k, v in cfg.items() if v is not None}
            parser.set_defaults(**clean)


def parse_args(parser):
    parser.add_argument('--config', default=None, type=str,
                        help='path to a YAML config file; CLI flags override YAML values')
    parser.add_argument('--dataset', default='mnist', type=str,
                        help='two tasks: mnist, fashion or fashion_and_mnist (28x28, Sener & Koltun, generated on '
                             'first use), or the same with -36p (36x36 Pareto MTL pickles in data/)')
    parser.add_argument('--num_tasks', default=2, type=int, help='number of tasks')
    parser.add_argument('--train_bs', default=256, type=int, help='batch size for training')
    parser.add_argument('--test_bs', default=10000, type=int, help='batch size for test')
    parser.add_argument('--epochs', default=100, type=int, help='training epochs')
    parser.add_argument('--dataset_path', default='.', type=str, help='dataset root (parent of data/ folder)')
    parser.add_argument('--shared_decoder', action='store_true', default=False,
                        help='one decoder for all tasks: every task loss is computed on the same output')
    # Override LibMTL_args defaults to match this benchmark's typical settings.
    parser.set_defaults(
        optim='adam', lr=1e-3, weight_decay=5e-4,
        adam_beta1=0.9, adam_beta2=0.999, adam_eps=1e-8, amsgrad=False,
        scheduler=None,  # multimnist: fixed LR, no StepLR
    )

    _load_yaml_defaults(parser)
    return parser.parse_args()

def main(params):
    # Set random seed and device
    kwargs, optim_param, scheduler_param = prepare_args(params)
    # optim_param = {'optim': 'sgd', 'lr': 0.001, 
    #                        'weight_decay': 0.0, 'momentum': 0.0}
    # scheduler_param = None
    
    # Input transform of Sener & Koltun's loader; the -36p pickles ignore it (raw 0-255, as in Pareto MTL)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Create MultiMNIST datasets (``dataset``: mnist, fashion, fashion_and_mnist, each optionally with -36p)
    ds = params.dataset.replace('mnist+fashion', 'fashion_and_mnist')
    if params.num_tasks == 2:
        train_multimnist = MultiMNISTDataset(root=params.dataset_path, dataset=ds, train=True, transform=transform)
        test_multimnist = MultiMNISTDataset(root=params.dataset_path, dataset=ds, train=False, transform=transform)
        task_dict = {'top-left': {'metrics': ['Acc'],
                        'metrics_fn': AccMetric(),
                       'loss_fn': CELoss(),
                       'weight': [1]},
                    'bottom-right': {'metrics': ['Acc'],
                       'metrics_fn': AccMetric(),
                       'loss_fn': CELoss(),
                       'weight': [1]}}
    elif params.num_tasks == 3:
        train_multimnist = MultiMNISTDataset3Digits(root=params.dataset_path, dataset='mnist', train=True, transform=transform, download=True)
        test_multimnist = MultiMNISTDataset3Digits(root=params.dataset_path, dataset='mnist', train=False, transform=transform, download=True)
        task_dict = {'top-left': {'metrics': ['Acc'],
                        'metrics_fn': AccMetric(),
                       'loss_fn': CELoss(),
                       'weight': [1]},
                    'bottom-center': {'metrics': ['Acc'],
                        'metrics_fn': AccMetric(),
                        'loss_fn': CELoss(),
                        'weight': [1]},
                    'top-right': {'metrics': ['Acc'],
                        'metrics_fn': AccMetric(),
                        'loss_fn': CELoss(),
                        'weight': [1]}}
    else:
        raise ValueError('No support dataset {}'.format(params.dataset))
        
    # Create data loaders
    multimnist_train_loader = DataLoader(train_multimnist, batch_size=params.train_bs, shuffle=True)
    multimnist_test_loader = DataLoader(test_multimnist, batch_size=params.test_bs, shuffle=False)
    
    img_size = 36 if params.num_tasks == 2 and ds.endswith('-36p') else 28  # 36x36 only for the -36p pickles
    fc_in = 20 * (((img_size - 4) // 2 - 4) // 2) ** 2  # after conv5-pool2-conv5-pool2: 720 (36x36) or 320 (28x28)

    class MultiLeNetEncoder(nn.Module):
        """The encoder part of the LeNet network adapted to MultiTask Learning. The model consists of two convolutions
        followed by a fully connected layers, resulting in a 50-dimensional embedding."""

        def __init__(self, in_channels=1):
            super().__init__()
            self.in_channels = in_channels
            self.conv1 = nn.Conv2d(in_channels, 10, kernel_size=5, stride=1)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5, stride=1)
            self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(fc_in, 50)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = self.maxpool1(x)
            x = F.relu(self.conv2(x))
            x = self.maxpool2(x)
            x = self.flatten(x)
            x = F.relu(self.fc(x))
            return x

        def get_last_layer(self):
            return self.fc


    class MultiLeNetDecoder(nn.Module):
        """The decoder part of the LeNet network adapted to MultiTask Learning. The output has 10 dimensions, since this
        model is used for datasets such as MultiMNIST."""

        def __init__(self):
            super().__init__()
            self.fc_task = nn.Linear(50, 10)

        def forward(self, x):
            x = self.fc_task(x)
            return x
    
    # Define the decoders (one module shared by all tasks with --shared_decoder: no task-specific parameters)
    if params.shared_decoder:
        shared = MultiLeNetDecoder()
        decoders = nn.ModuleDict({task: shared for task in list(task_dict.keys())})
    else:
        decoders = nn.ModuleDict({task: MultiLeNetDecoder() for task in list(task_dict.keys())})

    evo_args = kwargs.get('evo_args') or {}
    _TrainerBase = EvoMTLTrainer if evo_args.get('evo_training') else Trainer

    class MultiMNISTtrainer(_TrainerBase):
        def __init__(self, task_dict, weighting, architecture, encoder_class, 
                     decoders, rep_grad, multi_input, optim_param, 
                     scheduler_param, **kwargs):
            super(MultiMNISTtrainer, self).__init__(task_dict=task_dict, 
                                            weighting=weighting, 
                                            architecture=architecture, 
                                            encoder_class=encoder_class, 
                                            decoders=decoders,
                                            rep_grad=rep_grad,
                                            multi_input=multi_input,
                                            optim_param=optim_param,
                                            scheduler_param=scheduler_param,
                                            **kwargs)
    MultiMNISTmodel = MultiMNISTtrainer(task_dict=task_dict, 
                          weighting=params.weighting, 
                          architecture=params.arch, 
                          encoder_class=MultiLeNetEncoder, 
                          decoders=decoders,
                          rep_grad=params.rep_grad,
                          multi_input=params.multi_input,
                          optim_param=optim_param,
                          scheduler_param=scheduler_param,
                          save_path=params.save_path,
                          load_path=params.load_path,
                          **kwargs)
    # Train the model
    if params.mode == 'train':
        MultiMNISTmodel.train(multimnist_train_loader, multimnist_test_loader, params.epochs)
    elif params.mode == 'test':
        MultiMNISTmodel.test(multimnist_test_loader)
    else:
        raise ValueError

class _Tee:
    """Write to the terminal stream and a log file at once (line-buffered)."""

    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, s):
        self.stream.write(s)
        self.log.write(s)
        return len(s)

    def flush(self):
        self.stream.flush()
        self.log.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def _log_to_save_path(params):
    """Mirror stdout and stderr into <save_path>/stdout.log (read by the plot scripts)."""
    if not params.save_path:
        return
    os.makedirs(params.save_path, exist_ok=True)
    log = open(os.path.join(params.save_path, "stdout.log"), "w", buffering=1)
    sys.stdout, sys.stderr = _Tee(sys.stdout, log), _Tee(sys.stderr, log)


if __name__ == "__main__":
    params = parse_args(LibMTL_args)
    _log_to_save_path(params)
    # set device
    set_device(params.gpu_id)
    # set random seed
    set_random_seed(params.seed)
    main(params)